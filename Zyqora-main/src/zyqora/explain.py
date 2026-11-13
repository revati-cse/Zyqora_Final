"""Human-readable explanation card for one classification (terminal UI)."""

from __future__ import annotations

from .schemas import ACCEPT, ImageResult

REASON_TEXT = {
    "LOW_CONFIDENCE": "the model's confidence is below the EcoGuard threshold",
    "AMBIGUOUS_CATEGORY_SCORES": "two categories scored too closely",
    "TOP1_CLASS_UNMAPPED": "the recognised object has no waste rule",
    "NO_VALID_CATEGORY_MAPPING": "the object does not match any supported waste category",
    "INVALID_INFERENCE_RESULT": "the model output was invalid",
    "IMAGE_LOAD_FAILED": "the image could not be read",
}


def bar(p: float, width: int = 20) -> str:
    n = max(0, min(width, round(p * width)))
    return "#" * n + "." * (width - n)


def explanation_card(r: ImageResult, threshold: float, profile: str, decoder: str) -> str:
    if r.error:
        return f"  ERROR: {r.error}\n  Action: {r.disposal_action} (nothing sorted automatically)"
    lines = [f"  Model ({profile}) category scores:"]
    for name, p in sorted(r.category_scores.items(), key=lambda kv: -kv[1]):
        lines.append(f"    {name:<12} {bar(p)} {p:6.1%}")
    if decoder == "imagenet_mapping" and r.raw_topk:
        lines.append("  Recognised ImageNet objects: " + ", ".join(f"{p.label} {p.probability:.0%}" for p in r.raw_topk[:3]))
    verdict = f"{r.predicted_category} at {r.predicted_category_score:.1%} (EcoGuard threshold {threshold:.0%})"
    if r.ecoguard_status == ACCEPT:
        lines.append(f"  EcoGuard: ACCEPT - {verdict}")
        lines.append(f"  Action:   {r.disposal_action}")
    else:
        why = "; ".join(REASON_TEXT.get(x, x) for x in r.review_reason.split(";") if x)
        lines.append(f"  EcoGuard: REVIEW - {verdict}; {why}")
        lines.append("  Action:   MANUAL_REVIEW - please ask a supervisor (no bin opened)")
    lines.append(f"  Timing:   preprocess {r.preprocess_time_ms:.1f} ms, model {r.inference_time_ms:.1f} ms")
    return "\n".join(lines)
