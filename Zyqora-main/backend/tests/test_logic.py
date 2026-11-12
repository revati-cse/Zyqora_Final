from app.core.ecoguard import decide
from app.core.label_bridge import map_category
from app.core.sortsense import sort_action
from app.core.wastepulse import summarize

TH = {"accept_confidence": .8, "review_below": .4,
      "retry_accept_confidence": .7, "retry_min_margin": .25}

def test_high_confidence_accepts():
    assert decide([.91, .05, .04], thresholds=TH)[0] == "ACCEPT"

def test_low_confidence_reviews():
    assert decide([.2, .1, .05], thresholds=TH)[0] == "REVIEW"

def test_unmapped_reviews():
    assert decide([.99, .005], thresholds=TH, mapped=False)[0] == "REVIEW"

def test_intermediate_without_retry_reviews():
    assert decide([.6, .2], thresholds=TH)[0] == "REVIEW"

def test_retry_passes():
    result = decide([.6, .2], [.78, .2], TH)
    assert result[0] == "ACCEPT" and result[1] is True

def test_unconfigured_label_map():
    assert map_category("banana", {"status": "UNCONFIGURED", "mappings": {}}) == (None, "UNCONFIGURED")

def test_sort_action():
    assert sort_action("Organic") == "COMPOST"
    assert sort_action(None) == "CHECK"

def test_wastepulse():
    rows = [{"trust":"ACCEPT","action":"RECYCLE"},
            {"trust":"ACCEPT","action":"COMPOST"},
            {"trust":"REVIEW","action":"CHECK"}]
    result = summarize(rows)
    assert result["diversion_rate"] == 1.0
    assert abs(result["review_rate"] - 1/3) < 1e-9
