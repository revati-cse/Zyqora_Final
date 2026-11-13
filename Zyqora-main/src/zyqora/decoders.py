"""Output decoders: turn a model's logits into (probabilities, raw top-k, category result).

Each decoder is bound to exactly one output size. ``zyqora.profiles`` checks the
.pte's output shape against the decoder at load time, so the 2-class waste
decoder can never be applied to the 1000-class ImageNet model or vice versa.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from .classset import ClassSet
from .materialwise import CategoryResult, WasteMapping, map_prediction
from .model_runtime import softmax, topk
from .schemas import MODEL_CLASS, RawPrediction


class Decoder(Protocol):
    kind: str
    num_classes: int

    def decode(self, logits: np.ndarray) -> tuple[np.ndarray, list[RawPrediction], CategoryResult]: ...


class DecoderShapeError(ValueError):
    pass


def _check(logits: np.ndarray, n: int, kind: str) -> np.ndarray:
    arr = np.asarray(logits, dtype=np.float32).reshape(-1)
    if arr.shape != (n,):
        raise DecoderShapeError(f"{kind} decoder expects {n} logits, got shape {np.asarray(logits).shape}")
    return arr


class ImageNetMappingDecoder:
    """1000 ImageNet logits -> MaterialWise aggregated organiser categories."""

    kind = "imagenet_mapping"
    num_classes = 1000

    def __init__(self, labels: list[str], mapping: WasteMapping, top_k: int = 5):
        if len(labels) != self.num_classes:
            raise ValueError(f"ImageNet decoder needs {self.num_classes} labels, got {len(labels)}")
        self.labels = labels
        self.mapping = mapping
        self.top_k = top_k

    @property
    def category_set(self) -> WasteMapping:
        return self.mapping

    def decode(self, logits):
        probs = softmax(_check(logits, self.num_classes, self.kind))
        return probs, topk(probs, self.labels, self.top_k), map_prediction(probs, self.mapping)


class DirectClassDecoder:
    """N logits whose indices ARE the categories (fine-tuned waste model: 0 Organic, 1 Recyclable)."""

    kind = "direct_classes"

    def __init__(self, class_set: ClassSet):
        self.class_set = class_set
        self.num_classes = len(class_set.class_names)

    @property
    def category_set(self) -> ClassSet:
        return self.class_set

    def decode(self, logits):
        probs = softmax(_check(logits, self.num_classes, self.kind))
        names = self.class_set.class_names
        raw = topk(probs, list(names), self.num_classes)
        best, second = raw[0], raw[1]
        result = CategoryResult(
            predicted_category=best.label,
            predicted_category_score=best.probability,
            category_scores={n: float(probs[i]) for i, n in enumerate(names)},
            category_margin=best.probability - second.probability,
            mapped_mass=1.0,
            mapping_status=MODEL_CLASS,
            top1_rule=None,
            category_action=self.class_set.categories[best.label],
        )
        return probs, raw, result
