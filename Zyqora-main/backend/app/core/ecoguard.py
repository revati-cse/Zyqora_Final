def decide(top_probs, retry_probs=None, thresholds=None, mapped=True):
    """Return trust, retried, confidence, margin, reason."""
    t = thresholds or {}
    accept = float(t.get("accept_confidence", 0.80))
    review = float(t.get("review_below", 0.40))
    retry_accept = float(t.get("retry_accept_confidence", 0.70))
    retry_margin = float(t.get("retry_min_margin", 0.25))
    p1 = float(top_probs[0]) if top_probs else 0.0
    p2 = float(top_probs[1]) if len(top_probs) > 1 else 0.0
    margin = p1 - p2
    if not mapped:
        return "REVIEW", False, p1, margin, "CATEGORY_UNMAPPED"
    if p1 >= accept:
        return "ACCEPT", False, p1, margin, "HIGH_CONFIDENCE"
    if p1 < review:
        return "REVIEW", False, p1, margin, "LOW_CONFIDENCE"
    if retry_probs is None:
        return "REVIEW", False, p1, margin, "RETRY_REQUIRED_BUT_NOT_RUN"
    rp1 = float(retry_probs[0]) if retry_probs else 0.0
    rp2 = float(retry_probs[1]) if len(retry_probs) > 1 else 0.0
    rmargin = rp1 - rp2
    if rp1 >= retry_accept and rmargin >= retry_margin:
        return "ACCEPT", True, rp1, rmargin, "RETRY_PASSED"
    return "REVIEW", True, rp1, rmargin, "RETRY_FAILED"
