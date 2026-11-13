"""Single-image Zyqora pipeline: preprocess -> ExecuTorch -> decoder -> EcoGuard.

The decoder is chosen by the model profile (zyqora.profiles):
  * ImageNetMappingDecoder - 1000 ImageNet logits -> MaterialWise organiser categories
  * DirectClassDecoder     - fine-tuned model logits are the categories (0 Organic, 1 Recyclable)

The model is loaded once by the caller and reused for every image (batch 1).
Preprocessing time and model-only inference time are measured separately.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from .decoders import DecoderShapeError
from .ecoguard import EcoGuardConfig, decide, disposal_action
from .model_runtime import InferenceError
from .preprocessing import ImageLoadError, PreprocessSpec, image_to_array, load_rgb
from .schemas import IMAGE_ERROR, INFERENCE_ERROR, R_IMAGE_ERROR, R_INVALID, ImageResult


@dataclass
class ZyqoraPipeline:
    classifier: object          # ExecuTorchClassifier (or a test double with .infer())
    decoder: object             # ImageNetMappingDecoder | DirectClassDecoder
    guard: EcoGuardConfig
    preprocess_spec: PreprocessSpec = PreprocessSpec()

    def process(self, path: str | Path, display_name: str | None = None) -> ImageResult:
        """Classify an image file."""
        name = display_name or Path(path).name
        t0 = time.perf_counter_ns()
        try:
            img = load_rgb(path)
        except ImageLoadError as exc:
            return ImageResult(image_filename=name, mapping_status=IMAGE_ERROR, review_reason=R_IMAGE_ERROR, error=str(exc))
        return self._run(img, name, t0)

    def process_image(self, img: Image.Image, name: str) -> ImageResult:
        """Classify an in-memory image (e.g. a camera frame) without writing it to disk."""
        return self._run(img, name, time.perf_counter_ns())

    def _run(self, img: Image.Image, name: str, t0: int) -> ImageResult:
        try:
            array = image_to_array(img, self.preprocess_spec)
        except (ImageLoadError, ValueError, OSError) as exc:
            return ImageResult(image_filename=name, mapping_status=IMAGE_ERROR, review_reason=R_IMAGE_ERROR, error=str(exc))
        t1 = time.perf_counter_ns()
        try:
            out = self.classifier.infer(array)
            _, raw_topk, cat = self.decoder.decode(out.logits)
        except (InferenceError, DecoderShapeError, ValueError) as exc:
            return ImageResult(image_filename=name, mapping_status=INFERENCE_ERROR, review_reason=R_INVALID,
                               preprocess_time_ms=(t1 - t0) / 1e6, error=str(exc))
        decision = decide(cat.predicted_category_score, cat.mapping_status, cat.category_margin, self.guard)
        return ImageResult(
            image_filename=name,
            raw_topk=raw_topk,
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
            preprocess_time_ms=(t1 - t0) / 1e6,
            inference_time_ms=out.latency_ns / 1e6,
            end_to_end_time_ms=(time.perf_counter_ns() - t0) / 1e6,
        )
