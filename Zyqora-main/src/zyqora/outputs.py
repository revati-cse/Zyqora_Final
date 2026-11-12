"""Output helpers: JSON writing, annotated sample images and sample selection."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Iterable, Optional

from PIL import Image, ImageDraw, ImageFont, ImageOps

from .schemas import ACCEPT, ImageResult

FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
)
ACCEPT_COLOUR = (22, 128, 61)
REVIEW_COLOUR = (196, 104, 0)


def write_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def _font(size: int) -> ImageFont.ImageFont:
    for candidate in FONT_CANDIDATES:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size)
    try:
        return ImageFont.load_default(size=size)  # Pillow >= 10.1
    except TypeError:
        return ImageFont.load_default()


def annotation_lines(r: ImageResult) -> list[str]:
    top1 = r.raw_top1
    score = f"{r.predicted_category_score:.1%}" if r.predicted_category_score is not None else "n/a"
    lines = [
        f"Category: {r.predicted_category or 'none'}  (score {score})",
        f"EcoGuard: {r.ecoguard_status}" + (f"  [{r.review_reason}]" if r.review_reason else ""),
        f"Action: {r.disposal_action}",
    ]
    if top1:
        lines.append(f"Model top-1: {top1.label} ({top1.probability:.1%})")
    return lines


def annotate(image_path: Path, r: ImageResult, out_path: Path, width: int = 640) -> Path:
    with Image.open(image_path) as im:
        img = ImageOps.exif_transpose(im).convert("RGB")
    scale = width / img.width
    img = img.resize((width, max(1, round(img.height * scale))), Image.BILINEAR)
    font = _font(max(14, width // 34))
    line_h = int(font.size * 1.35) if hasattr(font, "size") else 18
    wrapped: list[str] = []
    max_chars = max(20, int(width / (getattr(font, "size", 14) * 0.55)))
    for line in annotation_lines(r):
        wrapped.extend(textwrap.wrap(line, max_chars) or [""])
    banner_h = line_h * len(wrapped) + 16
    canvas = Image.new("RGB", (img.width, img.height + banner_h), (255, 255, 255))
    canvas.paste(img, (0, 0))
    draw = ImageDraw.Draw(canvas)
    colour = ACCEPT_COLOUR if r.ecoguard_status == ACCEPT else REVIEW_COLOUR
    draw.rectangle([0, img.height, img.width, img.height + banner_h], fill=colour)
    y = img.height + 8
    for line in wrapped:
        draw.text((10, y), line, fill=(255, 255, 255), font=font)
        y += line_h
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=90)
    return out_path


def pick_samples(results: Iterable[tuple[ImageResult, Path]], count: int) -> list[tuple[ImageResult, Path]]:
    """Prefer one genuine ACCEPT and one genuine REVIEW; never fabricate either."""
    usable = [(r, p) for r, p in results if r.error is None]
    if count <= 0 or not usable:
        return []
    picks: list[tuple[ImageResult, Path]] = []
    first_accept: Optional[tuple[ImageResult, Path]] = next((x for x in usable if x[0].ecoguard_status == ACCEPT), None)
    first_review = next((x for x in usable if x[0].ecoguard_status != ACCEPT), None)
    for x in (first_accept, first_review):
        if x is not None and len(picks) < count:
            picks.append(x)
    for x in usable:
        if len(picks) >= count:
            break
        if x not in picks:
            picks.append(x)
    return picks
