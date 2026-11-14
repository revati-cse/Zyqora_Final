"""Object-detection evaluation: precision, recall, mAP@0.5 and mAP@0.5:0.95.

Matching follows the usual COCO/Ultralytics protocol: predictions are sorted by
confidence, each is matched greedily to the unmatched ground-truth box of the
same class with the highest IoU, and AP is the pycocotools definition: the mean of
the interpolated precision at 101 recall thresholds (0, 0.01, ..., 1). Ground-truth boxes flagged
``difficult`` (PASCAL VOC) are ignored: a prediction matched to one counts as
neither true nor false positive, and they are not in the recall denominator.

Precision / recall are also reported at one operating confidence threshold
(the dashboard's setting), because mAP alone hides that choice.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np

IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)


@dataclass
class GroundTruth:
    box: tuple[float, float, float, float]
    label: str
    difficult: bool = False


@dataclass
class Prediction:
    box: tuple[float, float, float, float]
    label: str
    confidence: float


@dataclass
class ImageRecord:
    image_id: str
    ground_truth: list[GroundTruth]
    predictions: list[Prediction]


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa[:, None] + bb[None, :] - inter, 1e-9)


def match_class(preds: list[Prediction], gts: list[GroundTruth]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For one image and one class -> (tp[n_pred, 10], ignore[n_pred], conf[n_pred])."""
    order = sorted(range(len(preds)), key=lambda i: -preds[i].confidence)
    p = [preds[i] for i in order]
    pb = np.array([x.box for x in p], dtype=np.float64).reshape(-1, 4)
    gb = np.array([g.box for g in gts], dtype=np.float64).reshape(-1, 4)
    difficult = np.array([g.difficult for g in gts], dtype=bool)
    ious = _iou_matrix(pb, gb)
    tp = np.zeros((len(p), len(IOU_THRESHOLDS)), dtype=bool)
    ignore = np.zeros((len(p), len(IOU_THRESHOLDS)), dtype=bool)
    for t, thr in enumerate(IOU_THRESHOLDS):
        used = np.zeros(len(gts), dtype=bool)
        for i in range(len(p)):
            if not len(gts):
                break
            cand = np.where(~used & (ious[i] >= thr))[0]
            if not len(cand):
                continue
            # prefer non-difficult matches, then the highest IoU
            nd = cand[~difficult[cand]]
            j = int(nd[np.argmax(ious[i, nd])]) if len(nd) else int(cand[np.argmax(ious[i, cand])])
            used[j] = True
            if difficult[j]:
                ignore[i, t] = True
            else:
                tp[i, t] = True
    return tp, ignore, np.array([x.confidence for x in p])


def average_precision(recall: np.ndarray, precision: np.ndarray) -> float:
    """COCO (pycocotools) 101-point AP: mean over recall thresholds r = 0, 0.01, ..., 1 of the
    maximum precision at any recall >= r (0 where recall r is never reached)."""
    if not len(recall):
        return 0.0
    env = np.maximum.accumulate(np.asarray(precision, dtype=np.float64)[::-1])[::-1]
    idx = np.searchsorted(np.asarray(recall, dtype=np.float64), np.linspace(0, 1, 101), side="left")
    q = np.where(idx < len(env), env[np.minimum(idx, len(env) - 1)], 0.0)
    return float(q.mean())


@dataclass
class ClassResult:
    label: str
    n_gt: int
    n_pred: int
    ap50: Optional[float]
    ap50_95: Optional[float]
    precision_at_conf: Optional[float]
    recall_at_conf: Optional[float]
    tp_at_conf: int = 0
    fp_at_conf: int = 0


@dataclass
class DetectionReport:
    classes: list[ClassResult]
    images: int
    operating_conf: float
    map50: Optional[float] = None
    map50_95: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "images": self.images, "operating_conf": self.operating_conf,
            "map50": self.map50, "map50_95": self.map50_95,
            "precision_at_conf": self.precision, "recall_at_conf": self.recall,
            "per_class": [c.__dict__ for c in self.classes], "notes": self.notes,
        }


def evaluate(records: Iterable[ImageRecord], classes: list[str], operating_conf: float = 0.25) -> DetectionReport:
    per = {c: {"tp": [], "ign": [], "conf": [], "n_gt": 0} for c in classes}
    n_images = 0
    for rec in records:
        n_images += 1
        for c in classes:
            gts = [g for g in rec.ground_truth if g.label == c]
            preds = [p for p in rec.predictions if p.label == c]
            per[c]["n_gt"] += sum(1 for g in gts if not g.difficult)
            if preds:
                tp, ign, conf = match_class(preds, gts)
                per[c]["tp"].append(tp)
                per[c]["ign"].append(ign)
                per[c]["conf"].append(conf)
    results: list[ClassResult] = []
    tot_tp = tot_fp = tot_gt = 0
    for c in classes:
        d = per[c]
        n_gt = d["n_gt"]
        if d["tp"]:
            tp = np.concatenate(d["tp"])
            ign = np.concatenate(d["ign"])
            conf = np.concatenate(d["conf"])
        else:
            tp = np.zeros((0, 10), bool)
            ign = np.zeros((0, 10), bool)
            conf = np.zeros(0)
        order = np.argsort(-conf, kind="stable")
        tp, ign, conf = tp[order], ign[order], conf[order]
        aps = []
        for t in range(len(IOU_THRESHOLDS)):
            keep = ~ign[:, t]
            ttp = tp[keep, t].astype(np.float64)
            if n_gt == 0:
                break
            ctp = np.cumsum(ttp)
            cfp = np.cumsum(1 - ttp)
            rec_ = ctp / n_gt
            prec = ctp / np.maximum(ctp + cfp, 1e-9)
            aps.append(average_precision(rec_, prec) if len(ttp) else 0.0)
        at = conf >= operating_conf
        keep50 = ~ign[:, 0]
        tpc = int((tp[:, 0] & at & keep50).sum())
        fpc = int((~tp[:, 0] & at & keep50).sum())
        tot_tp, tot_fp, tot_gt = tot_tp + tpc, tot_fp + fpc, tot_gt + n_gt
        results.append(ClassResult(
            label=c, n_gt=n_gt, n_pred=int(len(conf)),
            ap50=aps[0] if aps else None, ap50_95=float(np.mean(aps)) if aps else None,
            precision_at_conf=tpc / (tpc + fpc) if tpc + fpc else None,
            recall_at_conf=tpc / n_gt if n_gt else None, tp_at_conf=tpc, fp_at_conf=fpc,
        ))
    with_gt = [r for r in results if r.n_gt > 0]
    rep = DetectionReport(results, n_images, operating_conf)
    if with_gt:
        rep.map50 = float(np.mean([r.ap50 for r in with_gt]))
        rep.map50_95 = float(np.mean([r.ap50_95 for r in with_gt]))
        rep.precision = tot_tp / (tot_tp + tot_fp) if tot_tp + tot_fp else None
        rep.recall = tot_tp / tot_gt if tot_gt else None
    missing = [r.label for r in results if r.n_gt == 0]
    if missing:
        rep.notes.append(f"classes without ground truth (excluded from mAP): {missing}")
    return rep
