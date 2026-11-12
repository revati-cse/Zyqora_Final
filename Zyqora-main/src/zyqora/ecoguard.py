"""EcoGuard: confidence-aware ACCEPT / REVIEW decision layer (no neural network).

EcoGuard decides whether an automatic disposal recommendation is sufficiently
supported. It never changes the predicted category: a REVIEW item keeps its
predicted category in the results table and is still counted in accuracy.

Rules (all that apply are reported):
    INVALID_INFERENCE_RESULT   score missing / NaN / outside [0, 1]
    NO_VALID_CATEGORY_MAPPING  no mapped probability mass (unsupported object)
    TOP1_CLASS_UNMAPPED        raw top-1 class has no rule (if require_top1_mapped)
    LOW_CONFIDENCE             category score < threshold
    AMBIGUOUS_CATEGORY_SCORES  best - second-best category score < min_margin

The threshold is a tunable operating point, not a reliability guarantee.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from .schemas import (
    ACCEPT,
    MANUAL_REVIEW,
    R_AMBIGUOUS,
    R_INVALID,
    R_LOW_CONFIDENCE,
    R_NO_CATEGORY,
    R_TOP1_UNMAPPED,
    REVIEW,
    TOP1_UNMAPPED,
    UNSUPPORTED,
)


class EcoGuardConfigError(ValueError):
    pass


@dataclass(frozen=True)
class EcoGuardConfig:
    threshold: float = 0.40
    min_margin: float = 0.0
    require_top1_mapped: bool = True
    calibrated: bool = False
    frozen: bool = False

    def __post_init__(self) -> None:
        for name in ("threshold", "min_margin"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0.0 <= v <= 1.0:
                raise EcoGuardConfigError(f"ecoguard.{name} must be a number in [0, 1], got {v!r}")
        for name in ("require_top1_mapped", "calibrated", "frozen"):
            if not isinstance(getattr(self, name), bool):
                raise EcoGuardConfigError(f"ecoguard.{name} must be true or false")

    @classmethod
    def from_dict(cls, d: dict) -> "EcoGuardConfig":
        allowed = {"threshold", "min_margin", "require_top1_mapped", "calibrated", "frozen", "note"}
        unknown = set(d) - allowed
        if unknown:
            raise EcoGuardConfigError(f"unknown ecoguard keys: {sorted(unknown)}")
        if "threshold" not in d:
            raise EcoGuardConfigError("ecoguard.threshold is required")
        return cls(**{k: v for k, v in d.items() if k != "note"})


@dataclass(frozen=True)
class Decision:
    status: str
    reasons: tuple[str, ...]

    @property
    def reason(self) -> str:
        return ";".join(self.reasons)


def decide(
    category_score: Optional[float],
    mapping_status: str,
    category_margin: Optional[float],
    config: EcoGuardConfig,
) -> Decision:
    reasons: list[str] = []
    if mapping_status == UNSUPPORTED:
        return Decision(REVIEW, (R_NO_CATEGORY,))
    if (
        category_score is None
        or isinstance(category_score, bool)
        or not math.isfinite(category_score)
        or not 0.0 <= category_score <= 1.0 + 1e-9
    ):
        return Decision(REVIEW, (R_INVALID,))
    if config.require_top1_mapped and mapping_status == TOP1_UNMAPPED:
        reasons.append(R_TOP1_UNMAPPED)
    if category_score < config.threshold:
        reasons.append(R_LOW_CONFIDENCE)
    if config.min_margin > 0.0:
        if category_margin is None or not math.isfinite(category_margin):
            reasons.append(R_INVALID)
        elif category_margin < config.min_margin:
            reasons.append(R_AMBIGUOUS)
    return Decision(REVIEW, tuple(reasons)) if reasons else Decision(ACCEPT, ())


def disposal_action(decision: Decision, category_action: Optional[str]) -> str:
    """Only an ACCEPTed decision with a real category yields a bin; everything else is manual review."""
    if decision.status == ACCEPT and category_action:
        return category_action
    return MANUAL_REVIEW
