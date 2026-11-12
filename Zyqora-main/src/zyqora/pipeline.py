"""Single-image Zyqora pipeline: preprocess -> ExecuTorch -> MaterialWise -> EcoGuard.

The model is loaded once by the caller and reused for every image (batch 1).
Preprocessing time and model-only inference time are measured separately.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .ecoguard import EcoGuardConfig, decide, disposal_action
from .materialwise import WasteMapping, map_prediction
from .model_runtime import InferenceError, softmax, topk
from .preprocessing import ImageLoadError, PreprocessSpec, preprocess_path
from .schemas import IMAGE_ERROR, INFERENCE_ERROR, R_IMAGE_ERROR, R_INVALID, ImageResult


@dataclass
class ZyqoraPipeline:
    classifier: object          # ExecuTorchClassifier (or a test double with .infer())
    labels: list[str]
    mapping: WasteMapping
    guard: EcoGuardConfig
    preprocess_spec: PreprocessSpec = PreprocessSpec()
    top_k: int = 5

    def process(self, path: str | Path, display_name: str | None = None) -> ImageResult:
        name = display_name or Path(path).name
        t0 = time.perf_counter_ns()
        try:
            array = preprocess_path(path, self.preprocess_spec)
        except ImageLoadError as exc:
            return ImageResult(image_filename=name, mapping_status=IMAGE_ERROR,
                               review_reason=R_IMAGE_ERROR, error=str(exc))
        t1 = time.perf_counter_ns()
        try:
            out = self.classifier.infer(array)
        except (InferenceError, ValueError) as exc:
            return ImageResult(image_filename=name, mapping_status=INFERENCE_ERROR, review_reason=R_INVALID,
                               preprocess_time_ms=(t1 - t0) / 1e6, error=str(exc))
        probs = softmax(out.logits)
        result = self._interpret(name, probs)
        result.preprocess_time_ms = (t1 - t0) / 1e6
        result.inference_time_ms = out.latency_ns / 1e6
        result.end_to_end_time_ms = (time.perf_counter_ns() - t0) / 1e6
        return result

    def _interpret(self, name: str, probs: np.ndarray) -> ImageResult:
        cat = map_prediction(probs, self.mapping)
        decision = decide(cat.predicted_category_score, cat.mapping_status, cat.category_margin, self.guard)
        return ImageResult(
            image_filename=name,
            raw_topk=topk(probs, self.labels, self.top_k),
            predicted_category=cat.predicted_category,
            predicted_category_score=cat.predicted_category_score,
            category_scores=cat.category_scores,
            category_margin=cat.category_margin,
            mapped_mass=cat.mapped_mass,
            mapping_status=cat.mapping_status,
            top1_rule=cat.top1_rule.rule if cat.top1_rule else None,
            top1_material=cat.top1_rule.material if cat.top1_rule else None,
            ecoguard_status=decision.status,
            review_reason=decision.reason,
            disposal_action=disposal_action(decision, cat.category_action),
            category_action=cat.category_action,
        )
