"""ImageNet-1K label list in the exact order of MobileNet_V2_Weights.IMAGENET1K_V1.

A copy is bundled with the package so the Pi never needs TorchVision or a
network connection to interpret model outputs. ``scripts/prepare_model.py``
verifies that the bundled list equals the TorchVision weights metadata.
"""

from __future__ import annotations

import json
from pathlib import Path

from .paths import BUNDLED_LABELS

NUM_IMAGENET_CLASSES = 1000


class LabelsError(ValueError):
    pass


def load_labels(path: str | Path | None = None) -> list[str]:
    p = Path(path) if path else BUNDLED_LABELS
    if not p.is_file():
        raise LabelsError(f"ImageNet label file not found: {p}")
    try:
        labels = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise LabelsError(f"ImageNet label file is not valid JSON: {p}: {exc}") from exc
    if not isinstance(labels, list) or not all(isinstance(x, str) for x in labels):
        raise LabelsError(f"ImageNet label file must be a JSON list of strings: {p}")
    if len(labels) != NUM_IMAGENET_CLASSES:
        raise LabelsError(f"Expected {NUM_IMAGENET_CLASSES} labels in {p}, found {len(labels)}")
    return labels
