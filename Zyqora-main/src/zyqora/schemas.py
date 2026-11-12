"""Shared constants and result records.

The raw ImageNet prediction, the organiser-category prediction and the
disposal action are deliberately kept in separate fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# EcoGuard statuses
ACCEPT = "ACCEPT"
REVIEW = "REVIEW"

# Disposal action used whenever EcoGuard does not accept a recommendation.
MANUAL_REVIEW = "MANUAL_REVIEW"

# Mapping statuses (per image)
TOP1_MAPPED = "TOP1_MAPPED"            # raw top-1 ImageNet class has a mapping rule
TOP1_UNMAPPED = "TOP1_UNMAPPED"        # category predicted from other mapped classes
UNSUPPORTED = "UNSUPPORTED"            # no mapped probability mass at all
IMAGE_ERROR = "IMAGE_ERROR"            # image could not be loaded / preprocessed
INFERENCE_ERROR = "INFERENCE_ERROR"    # model returned an invalid result

# EcoGuard review reasons
R_LOW_CONFIDENCE = "LOW_CONFIDENCE"
R_AMBIGUOUS = "AMBIGUOUS_CATEGORY_SCORES"
R_TOP1_UNMAPPED = "TOP1_CLASS_UNMAPPED"
R_NO_CATEGORY = "NO_VALID_CATEGORY_MAPPING"
R_INVALID = "INVALID_INFERENCE_RESULT"
R_IMAGE_ERROR = "IMAGE_LOAD_FAILED"


@dataclass(frozen=True)
class RawPrediction:
    index: int
    label: str
    probability: float


@dataclass
class ImageResult:
    """Everything Zyqora knows about one processed image."""

    image_filename: str
    raw_topk: list[RawPrediction] = field(default_factory=list)
    predicted_category: Optional[str] = None
    predicted_category_score: Optional[float] = None
    category_scores: dict[str, float] = field(default_factory=dict)
    category_margin: Optional[float] = None
    mapped_mass: Optional[float] = None
    mapping_status: str = UNSUPPORTED
    top1_rule: Optional[str] = None
    top1_material: Optional[str] = None
    ecoguard_status: str = REVIEW
    review_reason: str = ""
    disposal_action: str = MANUAL_REVIEW
    category_action: Optional[str] = None  # action the category would imply if accepted
    preprocess_time_ms: Optional[float] = None
    inference_time_ms: Optional[float] = None
    end_to_end_time_ms: Optional[float] = None
    error: Optional[str] = None

    @property
    def raw_top1(self) -> Optional[RawPrediction]:
        return self.raw_topk[0] if self.raw_topk else None
