import time
import torch
from .preprocess import load_image, horizontal_flip
from .label_bridge import map_category
from .ecoguard import decide
from .sortsense import sort_action
from ..settings import load_label_map, load_thresholds
from ..schemas import ItemResult, ModeEnum

def classify_image(path: str, engine, mode=ModeEnum.LIVE):
    start = time.perf_counter()
    tensor = load_image(path)
    logits = engine.infer(tensor)
    probs = torch.softmax(logits, dim=1)[0]
    values, indices = torch.topk(probs, k=3)
    top_probs, top_indices = values.tolist(), indices.tolist()
    class_name = engine.class_name(top_indices[0])
    category, map_status = map_category(class_name, load_label_map())
    thresholds = load_thresholds()
    retry_probs = None
    if category is not None and thresholds["review_below"] <= top_probs[0] < thresholds["accept_confidence"]:
        flip_logits = engine.infer(horizontal_flip(tensor))
        flip_probs = torch.softmax(flip_logits, dim=1)[0]
        averaged = (probs + flip_probs) / 2
        retry_values, _ = torch.topk(averaged, k=3)
        retry_probs = retry_values.tolist()
    trust, retried, confidence, margin, reason = decide(
        top_probs, retry_probs, thresholds, mapped=(category is not None))
    action = sort_action(category) if trust == "ACCEPT" else "CHECK"
    return ItemResult(
        filename=str(path), mode=mode, model_class=class_name,
        confidence=confidence, category=category, trust=trust, action=action,
        retried=retried, latency_ms=(time.perf_counter()-start)*1000,
        note=f"{reason}; label_map={map_status}")
