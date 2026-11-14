"""Mixed-waste pipeline (Zyqora AI 4.0).

    image -> YOLO boxes -> per-box material hint -> crop -> MobileNetV2 (same contract
    and EcoGuard as single-image mode) -> conflict check -> organiser label
    -> disposal recommendation -> Green Score -> image summary

Rules that keep the output honest:
  * the detector's COCO label is only a material hint, never a waste category;
  * COCO classes that are not waste (person, chair, ...) are drawn but not classified;
  * an item is *reliable* only if EcoGuard ACCEPTs it AND the detector hint does not
    contradict the classifier; otherwise it is shown as Unknown and not scored;
  * when no waste object is detected, the whole image is classified as one item and
    marked as such (whole-image fallback), so nothing is invented.
The MobileNetV2 .pte has a fixed batch of 1, so crops are classified one after another.
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass
from typing import Optional

from PIL import Image, ImageDraw, ImageFont

from .detection.yolo import Detection, DetectorSettings, YoloDetector
from .greenscore_engine import GreenScoreEngine
from .wasterules import LabelProfiles, MaterialHint, MaterialTable, RecommendationEngine

SOURCE_DETECTOR = "detector"
SOURCE_WHOLE = "whole_image"

D_ACCEPT, D_REVIEW, D_IGNORED, D_SKIPPED = "ACCEPT", "REVIEW", "IGNORED", "SKIPPED"


def padded_crop_box(box: tuple[float, float, float, float], size: tuple[int, int], pad: float) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    W, H = size
    cx1 = max(0, int(round(x1 - pad * w)))
    cy1 = max(0, int(round(y1 - pad * h)))
    cx2 = min(W, int(round(x2 + pad * w)))
    cy2 = min(H, int(round(y2 + pad * h)))
    if cx2 - cx1 < 2:
        cx2 = min(W, cx1 + 2)
    if cy2 - cy1 < 2:
        cy2 = min(H, cy1 + 2)
    return cx1, cy1, cx2, cy2


@dataclass
class MixedWastePipeline:
    detector: YoloDetector
    classifier: object                 # zyqora.pipeline.ZyqoraPipeline (shared with single-image mode)
    materials: MaterialTable
    labels: LabelProfiles
    recommender: RecommendationEngine
    greenscore: GreenScoreEngine
    model_classes: tuple[str, ...]

    def analyze(self, img: Image.Image, name: str, *, settings: Optional[DetectorSettings] = None,
                label_profile: Optional[str] = None) -> dict:
        s = settings or self.detector.settings
        profile = self.labels.get(label_profile)
        t0 = time.perf_counter()
        det_out = self.detector.detect(img, s)
        detections = det_out.detections
        waste = [(i, d) for i, d in enumerate(detections) if self.materials.hint(d.label).waste_item]
        notes: list[str] = []

        objects: list[dict] = []
        cls_ms = 0.0
        classified = 0
        if not waste and s.whole_image_fallback:
            W, H = img.size
            notes.append("No waste object was detected; the whole image was classified as one item.")
            item, ms = self._classify_item(img, name, None, (0.0, 0.0, float(W), float(H)), (0, 0, W, H),
                                           SOURCE_WHOLE, profile.name)
            objects.append(item)
            cls_ms += ms
            classified = 1
        for i, d in enumerate(detections):
            hint = self.materials.hint(d.label)
            if not hint.waste_item:
                objects.append(self._ignored(d, hint))
                continue
            if classified >= s.max_objects_classified:
                objects.append(self._ignored(d, hint, skipped=True))
                continue
            crop_box = padded_crop_box(d.box, img.size, s.crop_padding)
            item, ms = self._classify_item(img.crop(crop_box), f"{name}#obj{i}", d, d.box, crop_box,
                                           SOURCE_DETECTOR, profile.name)
            objects.append(item)
            cls_ms += ms
            classified += 1
        skipped = sum(1 for o in objects if o["decision"] == D_SKIPPED)
        if skipped:
            notes.append(f"{skipped} waste object(s) were not classified (limit max_objects_classified="
                         f"{s.max_objects_classified}).")
        for n, o in enumerate(objects):
            o["index"] = n

        scored_inputs = [o for o in objects if o["decision"] in (D_ACCEPT, D_REVIEW)]
        image_gs = self.greenscore.aggregate([self._gs_obj(o) for o in scored_inputs], waste_items=len(scored_inputs))
        total_ms = (time.perf_counter() - t0) * 1e3
        return {
            "filename": name,
            "image_size": list(img.size),
            "label_profile": profile.as_dict(),
            "objects": objects,
            "summary": self._summary(objects, profile),
            "image_green_score": image_gs,
            "timing": {
                "detector_preprocess_ms": det_out.preprocess_ms, "detector_inference_ms": det_out.inference_ms,
                "detector_postprocess_ms": det_out.postprocess_ms, "classification_ms": cls_ms,
                "classified_crops": classified, "pipeline_ms": total_ms,
            },
            "detector": {"profile": self.detector.profile.name, "imgsz": self.detector.profile.imgsz,
                         "runtime": self.detector.profile.runtime, "trained_on": self.detector.profile.trained_on,
                         "candidates_above_conf": det_out.raw_candidates, "suppressed_small": det_out.suppressed_small,
                         "settings": s.as_dict()},
            "notes": notes,
            "versions": {"materials": self.materials.version, "disposal_rules": self.recommender.version,
                         "greenscore_rules": self.greenscore.version},
        }

    # ------------------------------------------------------------------ helpers
    def _gs_obj(self, o: dict):
        from .greenscore_engine import GreenScoreResult
        g = o["green_score"]
        return GreenScoreResult(bool(g["scored"]), g["score"])

    def _ignored(self, d: Detection, hint: MaterialHint, skipped: bool = False) -> dict:
        return {
            "source": SOURCE_DETECTOR, "box": [round(v, 1) for v in d.box], "crop_box": None,
            "detection_label": d.label, "detection_confidence": d.confidence, "waste_item": hint.waste_item,
            "decision": D_SKIPPED if skipped else D_IGNORED,
            "classification": None,
            "material": {"category": hint.material if hint.certainty == "likely" else "Unknown",
                         "certainty": hint.certainty, "food": hint.food, "note": hint.note,
                         "hint_material": hint.material},
            "conflict": None, "organiser_category": None, "organiser_labels": {},
            "recommendation": None,
            "green_score": {"scored": False, "score": None, "band": None,
                            "not_scored_reason": "NOT_CLASSIFIED_LIMIT" if skipped else "NOT_A_WASTE_ITEM",
                            "explanation": "Not classified (object limit reached)." if skipped else "Not a waste item; not scored.",
                            "components": [], "basis": None},
        }

    def _classify_item(self, crop: Image.Image, name: str, d: Optional[Detection], box, crop_box,
                       source: str, profile_name: str) -> tuple[dict, float]:
        hint = self.materials.hint(d.label if d else None)
        r = self.classifier.process_image(crop, name)
        ms = float(r.end_to_end_time_ms or 0.0)
        model_label = r.predicted_category
        accepted = r.ecoguard_status == "ACCEPT" and r.error is None
        conflict = bool(accepted and hint.expected_model_class and model_label and hint.expected_model_class != model_label)
        conflict_reason = (f"detector saw '{hint.detector_label}' (normally {hint.expected_model_class}), classifier said "
                           f"{model_label}") if conflict else None
        reliable = accepted and not conflict
        rec = self.recommender.recommend(hint, model_label, reliable, conflict)
        gs = self.greenscore.score_item(material=hint.material, material_certainty=hint.certainty, model_class=model_label,
                                        decision="ACCEPT" if accepted else "REVIEW", waste_item=True, conflict=conflict)
        reasons = [r.review_reason] if r.review_reason else []
        if conflict:
            reasons.append("DETECTOR_CLASSIFIER_CONFLICT")
        item = {
            "source": source, "box": [round(v, 1) for v in box], "crop_box": list(crop_box),
            "detection_label": d.label if d else None, "detection_confidence": d.confidence if d else None,
            "waste_item": True,
            "decision": D_ACCEPT if reliable else D_REVIEW,
            "review_reasons": reasons,
            "classification": {
                "model_label": model_label, "confidence": r.predicted_category_score,
                "class_scores": r.category_scores or {}, "ecoguard_status": r.ecoguard_status,
                "review_reason": r.review_reason, "preprocess_ms": r.preprocess_time_ms,
                "inference_ms": r.inference_time_ms, "error": r.error,
                "score_note": "softmax score, not a calibrated probability",
            },
            "material": {"category": hint.material if hint.certainty == "likely" else "Unknown",
                         "certainty": hint.certainty, "food": hint.food, "note": hint.note, "hint_material": hint.material},
            "conflict": conflict_reason,
            "organiser_category": self.labels.get(profile_name).label(model_label, reliable),
            "organiser_labels": {n: p.label(model_label, reliable) for n, p in self.labels.profiles.items()},
            "recommendation": rec.as_dict(),
            "green_score": gs.as_dict(),
        }
        return item, ms

    def _summary(self, objects: list[dict], profile) -> dict:
        waste = [o for o in objects if o["decision"] in (D_ACCEPT, D_REVIEW)]
        by_profile = {n: dict(Counter(o["organiser_labels"][n] for o in waste)) for n in self.labels.profiles}
        return {
            "detected_objects": sum(1 for o in objects if o["source"] == SOURCE_DETECTOR),
            "waste_items": len(waste),
            "accepted": sum(1 for o in waste if o["decision"] == D_ACCEPT),
            "review": sum(1 for o in waste if o["decision"] == D_REVIEW),
            "ignored_non_waste": sum(1 for o in objects if o["decision"] == D_IGNORED),
            "skipped": sum(1 for o in objects if o["decision"] == D_SKIPPED),
            "whole_image_fallback": any(o["source"] == SOURCE_WHOLE for o in objects),
            "by_organiser_category": by_profile[profile.name],
            "by_profile": by_profile,
            "by_model_label": dict(Counter(o["classification"]["model_label"] for o in waste if o["classification"])),
            "by_material": dict(Counter(o["material"]["category"] for o in waste)),
            "by_stream": dict(Counter(o["recommendation"]["stream"] for o in waste if o["recommendation"])),
            "food_items": sum(1 for o in waste if o["material"]["food"] is True),
        }


# ---------------------------------------------------------------------- annotation
_COLORS = {D_ACCEPT: (22, 163, 74), D_REVIEW: (217, 119, 6), D_IGNORED: (120, 120, 120), D_SKIPPED: (120, 120, 120)}


def annotate(img: Image.Image, result: dict) -> Image.Image:
    """Draw boxes + labels (server-side annotated image for download / history)."""
    out = img.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    lw = max(2, round(max(out.size) / 300))
    try:
        font = ImageFont.load_default(size=max(12, round(max(out.size) / 45)))
    except TypeError:  # Pillow < 10.1
        font = ImageFont.load_default()
    for o in result["objects"]:
        if o["source"] == SOURCE_WHOLE:
            continue
        x1, y1, x2, y2 = o["box"]
        color = _COLORS.get(o["decision"], (120, 120, 120))
        draw.rectangle((x1, y1, x2, y2), outline=color, width=lw)
        if o["decision"] in (D_ACCEPT, D_REVIEW):
            gs = o["green_score"]["score"]
            text = f"#{o['index']} {o['organiser_category']}" + (f" GS {gs:.0f}" if gs is not None else "")
        else:
            text = f"#{o['index']} {o['detection_label']} (not waste)" if o["decision"] == D_IGNORED else f"#{o['index']} skipped"
        tb = draw.textbbox((x1, y1), text, font=font)
        th = tb[3] - tb[1] + 4
        ty = y1 - th if y1 - th >= 0 else y1
        draw.rectangle((x1, ty, x1 + (tb[2] - tb[0]) + 6, ty + th), fill=color)
        draw.text((x1 + 3, ty + 1), text, fill=(255, 255, 255), font=font)
    return out
