"""Deterministic MobileNetV2 preprocessing using only Pillow and NumPy.

This reproduces the TorchVision / Arm-course transform exactly:
    Resize(256) -> CenterCrop(224) -> ToTensor() -> Normalize(ImageNet)
so the Pi runtime does not need TorchVision. ``scripts/check_parity.py``
verifies the equivalence numerically against TorchVision.

Steps: open -> EXIF orientation -> RGB -> resize short side (bilinear, PIL
antialias) -> center crop -> float32 / 255 -> normalise -> NCHW batch of 1.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

SUPPORTED_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})


class ImageLoadError(ValueError):
    """Raised when an image cannot be used. Never replaced by a fake prediction."""


@dataclass(frozen=True)
class PreprocessSpec:
    resize: int = 256
    center_crop: int = 224
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    input_shape: tuple[int, int, int, int] = (1, 3, 224, 224)

    @classmethod
    def from_contract(cls, contract: dict) -> "PreprocessSpec":
        """Build from config/model.json (same keys as the Arm course model_spec.json)."""
        inp = contract["input"]
        prep = inp["preprocessing"]
        if inp.get("dtype") != "float32" or inp.get("layout", "nchw").lower() != "nchw":
            raise ValueError("Model contract must be float32 NCHW")
        spec = cls(
            resize=int(prep["resize"]),
            center_crop=int(prep["center_crop"]),
            mean=tuple(float(x) for x in prep["mean"]),
            std=tuple(float(x) for x in prep["std"]),
            input_shape=tuple(int(x) for x in inp["shape"]),
        )
        if spec.input_shape != (1, 3, spec.center_crop, spec.center_crop):
            raise ValueError(f"Input shape {spec.input_shape} inconsistent with crop {spec.center_crop}")
        if len(spec.mean) != 3 or len(spec.std) != 3 or min(spec.std) <= 0:
            raise ValueError("mean/std must have 3 entries and std must be positive")
        return spec

    @classmethod
    def from_file(cls, path: str | Path) -> "PreprocessSpec":
        return cls.from_contract(json.loads(Path(path).read_text(encoding="utf-8")))


def load_rgb(path: str | Path) -> Image.Image:
    """Open an image file, apply EXIF orientation and convert to RGB."""
    p = Path(path)
    if not p.is_file():
        raise ImageLoadError(f"Image not found: {p}")
    if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ImageLoadError(f"Unsupported image format '{p.suffix}' for {p.name}")
    try:
        with Image.open(p) as im:
            im.load()
            im = ImageOps.exif_transpose(im)
            if im.mode == "P" and "transparency" in im.info:
                im = im.convert("RGBA")
            # Matches the Arm course: RGBA/LA alpha is discarded, grayscale is replicated.
            rgb = im.convert("RGB")
    except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombError) as exc:
        raise ImageLoadError(f"Corrupted or unreadable image {p.name}: {exc}") from exc
    if rgb.width < 1 or rgb.height < 1:
        raise ImageLoadError(f"Image has invalid dimensions {rgb.size}: {p.name}")
    return rgb


def resize_short_side(img: Image.Image, size: int) -> Image.Image:
    """TorchVision Resize(int) semantics for PIL images (bilinear, antialiased)."""
    w, h = img.size
    short, long = (w, h) if w <= h else (h, w)
    new_short, new_long = size, int(size * long / short)
    new_w, new_h = (new_short, new_long) if w <= h else (new_long, new_short)
    if (new_w, new_h) == (w, h):
        return img
    return img.resize((new_w, new_h), Image.BILINEAR)


def center_crop(img: Image.Image, size: int) -> Image.Image:
    """TorchVision CenterCrop semantics (rounding identical)."""
    w, h = img.size
    if w < size or h < size:
        raise ImageLoadError(f"Image {img.size} smaller than crop {size} after resize")
    top = int(round((h - size) / 2.0))
    left = int(round((w - size) / 2.0))
    return img.crop((left, top, left + size, top + size))


def image_to_array(img: Image.Image, spec: PreprocessSpec = PreprocessSpec()) -> np.ndarray:
    """Resize, crop, normalise -> contiguous float32 array of shape [1, 3, H, W]."""
    cropped = center_crop(resize_short_side(img, spec.resize), spec.center_crop)
    hwc = np.asarray(cropped, dtype=np.uint8)
    chw = hwc.transpose(2, 0, 1).astype(np.float32) / np.float32(255.0)
    mean = np.asarray(spec.mean, dtype=np.float32).reshape(3, 1, 1)
    std = np.asarray(spec.std, dtype=np.float32).reshape(3, 1, 1)
    chw -= mean
    chw /= std
    arr = np.ascontiguousarray(chw[np.newaxis, ...])
    validate_input_array(arr, spec.input_shape)
    return arr


def preprocess_path(path: str | Path, spec: PreprocessSpec = PreprocessSpec()) -> np.ndarray:
    return image_to_array(load_rgb(path), spec)


def validate_input_array(arr: np.ndarray, shape: tuple[int, ...]) -> None:
    if not isinstance(arr, np.ndarray):
        raise ValueError(f"Model input must be a numpy array, got {type(arr).__name__}")
    if arr.dtype != np.float32:
        raise ValueError(f"Model input must be float32, got {arr.dtype}")
    if tuple(arr.shape) != tuple(shape):
        raise ValueError(f"Model input shape {tuple(arr.shape)} != expected {tuple(shape)}")
    if not np.isfinite(arr).all():
        raise ValueError("Model input contains NaN or Inf")
