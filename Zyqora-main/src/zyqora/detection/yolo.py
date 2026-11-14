"""YOLO detector: letterbox -> runtime -> decode -> NMS -> boxes in original-image pixels.

Supports Ultralytics-style YOLOv8 / YOLO11 detection heads exported WITHOUT built-in
NMS, whose single output is ``[1, 4 + num_classes, N]``: rows 0-3 are box centre-x,
centre-y, width and height in letterboxed input pixels, rows 4.. are per-class
scores that already went through a sigmoid. Two interchangeable runtimes:

* ``executorch``  - a ``.pte`` lowered to XNNPACK (same runtime as the classifier)
* ``onnxruntime`` - a ``.onnx`` file on the ONNX Runtime CPU execution provider

Ultralytics (AGPL-3.0) is only used offline to *export* the weights
(scripts/export_detector.py); it is not imported at runtime.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Optional, Protocol

import numpy as np
from PIL import Image

from ..config import ConfigError, load_yaml
from ..paths import CONFIG_DIR, resolve

log = logging.getLogger("zyqora.detection")
DEFAULT_DETECTOR_CONFIG = CONFIG_DIR / "detector.yaml"
RUNTIMES = ("executorch", "onnxruntime")
PAD_VALUE = 114  # Ultralytics letterbox grey


class DetectorError(RuntimeError):
    """Detector missing, incompatible or failed. Never replaced by fake boxes."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


# --------------------------------------------------------------------------- config
@dataclass(frozen=True)
class DetectorProfile:
    name: str
    file: Path
    runtime: str
    imgsz: int
    classes: tuple[str, ...]
    classes_file: Path
    sha256: Optional[str] = None
    description: str = ""
    weights: str = ""
    license: str = ""
    trained_on: str = ""


@dataclass(frozen=True)
class DetectorSettings:
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    agnostic_nms: bool = True
    max_detections: int = 20
    max_objects_classified: int = 12
    crop_padding: float = 0.08
    min_box_area_fraction: float = 0.002
    whole_image_fallback: bool = True

    def validate(self) -> "DetectorSettings":
        for name in ("conf_threshold", "iou_threshold"):
            v = getattr(self, name)
            if not isinstance(v, (int, float)) or isinstance(v, bool) or not 0.0 < float(v) < 1.0:
                raise ConfigError(f"detector {name} must be in (0, 1), got {v!r}")
        for name in ("max_detections", "max_objects_classified"):
            v = getattr(self, name)
            if not isinstance(v, int) or isinstance(v, bool) or not 1 <= v <= 300:
                raise ConfigError(f"detector {name} must be an integer in [1, 300], got {v!r}")
        if not 0.0 <= float(self.crop_padding) <= 0.5:
            raise ConfigError(f"detector crop_padding must be in [0, 0.5], got {self.crop_padding!r}")
        if not 0.0 <= float(self.min_box_area_fraction) < 0.5:
            raise ConfigError(f"detector min_box_area_fraction must be in [0, 0.5), got {self.min_box_area_fraction!r}")
        return self

    @classmethod
    def from_dict(cls, d: dict) -> "DetectorSettings":
        known = {k: d[k] for k in cls.__dataclass_fields__ if k in d}
        unknown = sorted(set(d) - set(cls.__dataclass_fields__))
        if unknown:
            raise ConfigError(f"unknown detector settings: {unknown}")
        return cls(**known).validate()

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


@dataclass(frozen=True)
class DetectorConfig:
    active: str
    profiles: dict[str, DetectorProfile]
    settings: DetectorSettings
    materials: Path
    source: Path

    def get(self, name: Optional[str] = None) -> DetectorProfile:
        name = name or self.active
        if name not in self.profiles:
            raise ConfigError(f"unknown detector profile '{name}' (available: {sorted(self.profiles)})")
        return self.profiles[name]


def _load_classes(path: Path) -> tuple[str, ...]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"cannot read detector classes {path}: {exc}") from exc
    names = data.get("classes") if isinstance(data, dict) else data
    if not isinstance(names, list) or not names or not all(isinstance(n, str) and n for n in names):
        raise ConfigError(f"{path}: 'classes' must be a non-empty list of names in output-index order")
    if len(set(names)) != len(names):
        raise ConfigError(f"{path}: duplicate class names")
    return tuple(names)


def load_detector_config(path: str | Path = DEFAULT_DETECTOR_CONFIG) -> DetectorConfig:
    path = Path(path)
    data = load_yaml(path)
    raw = data.get("profiles")
    if not isinstance(raw, dict) or not raw:
        raise ConfigError(f"{path}: 'profiles' must be a non-empty mapping")
    profiles: dict[str, DetectorProfile] = {}
    for name, p in raw.items():
        if not isinstance(p, dict):
            raise ConfigError(f"{path}: detector profile '{name}' must be a mapping")
        for key in ("file", "runtime", "imgsz", "classes"):
            if p.get(key) in (None, ""):
                raise ConfigError(f"{path}: detector profile '{name}' needs '{key}'")
        if p["runtime"] not in RUNTIMES:
            raise ConfigError(f"{path}: detector profile '{name}': runtime must be one of {RUNTIMES}")
        imgsz = p["imgsz"]
        if not isinstance(imgsz, int) or imgsz < 32 or imgsz % 32:
            raise ConfigError(f"{path}: detector profile '{name}': imgsz must be a positive multiple of 32")
        sha = p.get("sha256")
        if sha is not None and (not isinstance(sha, str) or len(sha) != 64):
            raise ConfigError(f"{path}: detector profile '{name}': sha256 must be 64 hex characters or null")
        cpath = resolve(p["classes"])
        profiles[name] = DetectorProfile(
            name=name, file=resolve(p["file"]), runtime=p["runtime"], imgsz=imgsz,
            classes=_load_classes(cpath), classes_file=cpath, sha256=sha.lower() if sha else None,
            description=p.get("description", ""), weights=p.get("weights", ""),
            license=p.get("license", ""), trained_on=p.get("trained_on", ""),
        )
    active = data.get("active")
    if active not in profiles:
        raise ConfigError(f"{path}: active detector {active!r} is not defined")
    settings = DetectorSettings.from_dict(data.get("settings") or {})
    materials = resolve(data.get("materials") or "config/detector_materials.yaml")
    return DetectorConfig(active, profiles, settings, materials, path)


# --------------------------------------------------------------------------- geometry
@dataclass(frozen=True)
class Letterbox:
    scale: float
    pad_x: int
    pad_y: int
    size: int


def letterbox(img: Image.Image, size: int) -> tuple[np.ndarray, Letterbox]:
    """Aspect-preserving resize into a size x size grey canvas -> float32 NCHW in [0, 1]."""
    w, h = img.size
    if w < 1 or h < 1:
        raise ValueError("image has no pixels")
    r = min(size / w, size / h)
    nw, nh = max(1, int(round(w * r))), max(1, int(round(h * r)))
    resized = img.resize((nw, nh), Image.Resampling.BILINEAR) if (nw, nh) != (w, h) else img
    canvas = Image.new("RGB", (size, size), (PAD_VALUE,) * 3)
    px, py = (size - nw) // 2, (size - nh) // 2
    canvas.paste(resized, (px, py))
    arr = np.asarray(canvas, dtype=np.float32) * (1.0 / 255.0)
    return np.ascontiguousarray(arr.transpose(2, 0, 1)[None]), Letterbox(r, px, py, size)


def box_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """IoU of one xyxy box against an (N, 4) array."""
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    a = (box[2] - box[0]) * (box[3] - box[1])
    b = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / np.maximum(a + b - inter, 1e-9)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float, max_keep: int = 300) -> list[int]:
    """Greedy non-maximum suppression; returns kept indices in descending score order."""
    order = np.argsort(-scores, kind="stable")
    keep: list[int] = []
    while order.size and len(keep) < max_keep:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        ious = box_iou(boxes[i], boxes[order[1:]])
        order = order[1:][ious <= iou_threshold]
    return keep


# --------------------------------------------------------------------------- runtimes
class Runner(Protocol):
    backend: str
    def run(self, x: np.ndarray) -> np.ndarray: ...
    def describe(self) -> dict: ...


class ExecuTorchRunner:
    backend = "ExecuTorch/XNNPACK"

    def __init__(self, path: Path, imgsz: int, portable_lib: Optional[ModuleType] = None):
        from ..model_runtime import ModelLoadError, RuntimeUnavailableError, check_pte_file, import_portable_lib

        try:
            check_pte_file(path)
            pl = portable_lib or import_portable_lib()
            t0 = time.perf_counter()
            self._module = pl._load_for_executorch(str(path))
            self.load_ms = (time.perf_counter() - t0) * 1e3
        except ModelLoadError as exc:
            raise DetectorError("DETECTOR_NOT_FOUND" if not path.exists() else "DETECTOR_LOAD_FAILED", str(exc)) from exc
        except RuntimeUnavailableError as exc:
            raise DetectorError("RUNTIME_UNAVAILABLE", str(exc)) from exc
        except Exception as exc:  # pybind errors
            raise DetectorError("DETECTOR_LOAD_FAILED", f"ExecuTorch could not load {path.name}: {exc}") from exc
        self.output_shape: Optional[tuple[int, ...]] = None
        if hasattr(self._module, "method_meta"):
            meta = self._module.method_meta("forward")
            sizes = tuple(int(s) for s in meta.input_tensor_meta(0).sizes())
            if sizes != (1, 3, imgsz, imgsz):
                raise DetectorError("DETECTOR_CONTRACT_MISMATCH",
                                    f"{path.name} expects input {sizes}, profile says (1, 3, {imgsz}, {imgsz})")
            if hasattr(meta, "output_tensor_meta"):
                self.output_shape = tuple(int(s) for s in meta.output_tensor_meta(0).sizes())

    def run(self, x: np.ndarray) -> np.ndarray:
        import torch

        out = self._module.forward((torch.from_numpy(x),))
        first = out[0]
        return first.numpy() if hasattr(first, "numpy") else np.asarray(first)

    def describe(self) -> dict:
        return {"backend": self.backend, "load_ms": self.load_ms}


class OnnxRunner:
    backend = "ONNX Runtime (CPU)"

    def __init__(self, path: Path, imgsz: int, num_threads: Optional[int] = None):
        if not path.is_file():
            raise DetectorError("DETECTOR_NOT_FOUND", f"Detector file not found: {path}. "
                                "Export it with scripts/export_detector.py.")
        try:
            import onnxruntime as ort  # type: ignore
        except ImportError as exc:
            raise DetectorError("RUNTIME_UNAVAILABLE", "onnxruntime is not installed (pip install onnxruntime)") from exc
        so = ort.SessionOptions()
        if num_threads:
            so.intra_op_num_threads = int(num_threads)
            so.inter_op_num_threads = 1
        t0 = time.perf_counter()
        try:
            self._sess = ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])
        except Exception as exc:
            raise DetectorError("DETECTOR_LOAD_FAILED", f"ONNX Runtime could not load {path.name}: {exc}") from exc
        self.load_ms = (time.perf_counter() - t0) * 1e3
        inp = self._sess.get_inputs()[0]
        self._input = inp.name
        shape = tuple(inp.shape)
        if shape != (1, 3, imgsz, imgsz):
            raise DetectorError("DETECTOR_CONTRACT_MISMATCH", f"{path.name} expects input {shape}, profile says imgsz {imgsz}")
        self.output_shape = tuple(self._sess.get_outputs()[0].shape)
        self.version = ort.__version__

    def run(self, x: np.ndarray) -> np.ndarray:
        return self._sess.run(None, {self._input: x})[0]

    def describe(self) -> dict:
        return {"backend": f"{self.backend} {self.version}", "load_ms": self.load_ms}


# --------------------------------------------------------------------------- detector
@dataclass
class Detection:
    box: tuple[float, float, float, float]   # x1, y1, x2, y2 in original-image pixels
    class_id: int
    label: str
    confidence: float

    def as_dict(self) -> dict:
        return {"box": [round(v, 1) for v in self.box], "class_id": self.class_id,
                "label": self.label, "confidence": float(self.confidence)}


@dataclass
class DetectorOutput:
    detections: list[Detection]
    image_size: tuple[int, int]
    preprocess_ms: float
    inference_ms: float
    postprocess_ms: float
    raw_candidates: int = 0
    suppressed_small: int = 0

    @property
    def total_ms(self) -> float:
        return self.preprocess_ms + self.inference_ms + self.postprocess_ms


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class YoloDetector:
    profile: DetectorProfile
    settings: DetectorSettings = field(default_factory=DetectorSettings)
    num_threads: Optional[int] = None
    runner: Any = None                       # injectable (tests)
    portable_lib: Optional[ModuleType] = None
    verify_hash: bool = True

    def __post_init__(self) -> None:
        p = self.profile
        if self.runner is None:
            if self.verify_hash and p.sha256 and p.file.is_file():
                actual = file_sha256(p.file)
                if actual != p.sha256:
                    raise DetectorError("DETECTOR_HASH_MISMATCH", f"{p.file.name}: SHA-256 {actual[:12]}... does not "
                                        f"match the pinned {p.sha256[:12]}... in config/detector.yaml")
            if p.runtime == "executorch":
                self.runner = ExecuTorchRunner(p.file, p.imgsz, self.portable_lib)
            else:
                self.runner = OnnxRunner(p.file, p.imgsz, self.num_threads)
        n = 4 + len(p.classes)
        shape = getattr(self.runner, "output_shape", None)
        if shape and (len(shape) != 3 or shape[0] != 1 or shape[1] != n):
            raise DetectorError("DETECTOR_CONTRACT_MISMATCH",
                                f"{p.file.name} output {shape} does not match {len(p.classes)} classes "
                                f"(expected [1, {n}, N]); wrong class file or a model exported with NMS")
        self.load_ms = float(getattr(self.runner, "load_ms", 0.0))
        log.info("detector %s loaded (%s, imgsz %d, %d classes)", p.name, self.runner.backend, p.imgsz, len(p.classes))

    def detect(self, img: Image.Image, settings: Optional[DetectorSettings] = None) -> DetectorOutput:
        s = settings or self.settings
        t0 = time.perf_counter()
        x, lb = letterbox(img, self.profile.imgsz)
        t1 = time.perf_counter()
        try:
            raw = np.asarray(self.runner.run(x), dtype=np.float32)
        except Exception as exc:
            raise DetectorError("DETECTOR_INFERENCE_FAILED", f"detector forward() failed: {exc}") from exc
        t2 = time.perf_counter()
        dets, n_raw, n_small = self.decode(raw, lb, img.size, s)
        t3 = time.perf_counter()
        return DetectorOutput(dets, img.size, (t1 - t0) * 1e3, (t2 - t1) * 1e3, (t3 - t2) * 1e3, n_raw, n_small)

    def decode(self, raw: np.ndarray, lb: Letterbox, image_size: tuple[int, int],
               s: DetectorSettings) -> tuple[list[Detection], int, int]:
        nc = len(self.profile.classes)
        if raw.ndim != 3 or raw.shape[0] != 1 or raw.shape[1] != 4 + nc:
            raise DetectorError("DETECTOR_CONTRACT_MISMATCH", f"detector output {raw.shape}, expected [1, {4 + nc}, N]")
        if not np.isfinite(raw).all():
            raise DetectorError("DETECTOR_INFERENCE_FAILED", "detector output contains NaN or Inf")
        pred = raw[0].T                                    # (N, 4 + nc)
        cls_scores = pred[:, 4:]
        cls_id = cls_scores.argmax(1)
        conf = cls_scores[np.arange(len(cls_id)), cls_id]
        mask = conf >= s.conf_threshold
        n_raw = int(mask.sum())
        if not n_raw:
            return [], 0, 0
        b, conf, cls_id = pred[mask, :4], conf[mask], cls_id[mask]
        xyxy = np.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2, b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2], 1)
        # undo letterbox, clip to the image
        w, h = image_size
        xyxy[:, [0, 2]] = np.clip((xyxy[:, [0, 2]] - lb.pad_x) / lb.scale, 0, w)
        xyxy[:, [1, 3]] = np.clip((xyxy[:, [1, 3]] - lb.pad_y) / lb.scale, 0, h)
        area = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
        big = area >= s.min_box_area_fraction * w * h
        n_small = int((~big).sum())
        xyxy, conf, cls_id = xyxy[big], conf[big], cls_id[big]
        if not len(conf):
            return [], n_raw, n_small
        if s.agnostic_nms:
            keep = nms(xyxy, conf, s.iou_threshold, s.max_detections)
        else:   # per-class NMS via coordinate offset (standard trick)
            off = xyxy + (cls_id[:, None] * (max(w, h) + 1.0))
            keep = nms(off, conf, s.iou_threshold, s.max_detections)
        names = self.profile.classes
        dets = [Detection(tuple(float(v) for v in xyxy[i]), int(cls_id[i]), names[int(cls_id[i])], float(conf[i]))
                for i in keep]
        return dets, n_raw, n_small

    def describe(self) -> dict:
        p = self.profile
        size = p.file.stat().st_size if p.file.is_file() else None
        return {"profile": p.name, "description": p.description, "file": p.file.name, "file_size_bytes": size,
                "runtime": p.runtime, "imgsz": p.imgsz, "num_classes": len(p.classes), "weights": p.weights,
                "license": p.license, "trained_on": p.trained_on, "sha256_pinned": p.sha256,
                "load_ms": self.load_ms, **self.runner.describe()}
