"""Mandatory evaluation: per-image results table + accuracy summary.

Metric definitions (every image is in the denominator; nothing is excluded):
    overall_accuracy        = correct category predictions / total evaluated images
    review_rate             = REVIEW images / total evaluated images
    accepted_only_accuracy  = correct ACCEPTed predictions / ACCEPTed predictions
                              (None / "undefined" when nothing was accepted)
    category_coverage       = images with a predicted category / total images
A REVIEW decision is never counted as correct by itself; an image that fails to
load or has no category prediction counts as incorrect.
"""

from __future__ import annotations

import csv
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from .preprocessing import SUPPORTED_EXTENSIONS
from .schemas import ACCEPT, REVIEW, ImageResult

FILENAME_COLUMNS = ("image_filename", "filename", "image")
LABEL_COLUMNS = ("ground_truth_category", "label", "category")


class LabelsError(ValueError):
    pass


def discover_images(folder: str | Path) -> list[tuple[str, Path]]:
    """Return sorted (relative posix name, path) for every supported image under folder."""
    root = Path(folder)
    if not root.is_dir():
        raise FileNotFoundError(f"Image folder not found: {root}")
    found = [
        (p.relative_to(root).as_posix(), p)
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS and not p.name.startswith(".")
    ]
    return sorted(found)


def read_labels_csv(path: str | Path, mapping) -> dict[str, str]:
    """Read organiser labels -> {image_filename: canonical organiser category}."""
    p = Path(path)
    if not p.is_file():
        raise LabelsError(f"Labels file not found: {p}")
    with p.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        header = [h.strip() for h in (reader.fieldnames or [])]
        fcol = next((c for c in FILENAME_COLUMNS if c in header), None)
        lcol = next((c for c in LABEL_COLUMNS if c in header), None)
        if not fcol or not lcol:
            raise LabelsError(
                f"{p} must have columns {FILENAME_COLUMNS[0]} and {LABEL_COLUMNS[0]} (found {header})"
            )
        labels: dict[str, str] = {}
        unknown: dict[str, list[str]] = {}
        dupes: list[str] = []
        for row in reader:
            row = {(k or "").strip(): (v or "").strip() for k, v in row.items()}
            name, raw = row.get(fcol, ""), row.get(lcol, "")
            if not name:
                continue
            if name in labels:
                dupes.append(name)
            canon = mapping.ground_truth_category(raw)
            if canon is None:
                unknown.setdefault(raw, []).append(name)
                continue
            labels[name] = canon
    problems = []
    if dupes:
        problems.append(f"duplicate filenames in labels: {sorted(set(dupes))}")
    if unknown:
        problems.append(
            "unknown category names (not in config/mapping.json): "
            + ", ".join(f"'{k}' ({len(v)} images)" for k, v in unknown.items())
            + f". Known: {mapping.category_names}"
            + (f" (+ unsupported ground truth {sorted(mapping.unsupported_ground_truth)})"
               if getattr(mapping, "unsupported_ground_truth", None) else "")
        )
    if problems:
        raise LabelsError("; ".join(problems))
    if not labels:
        raise LabelsError(f"No labelled rows in {p}")
    return labels


def labels_from_folders(images: list[tuple[str, Path]], mapping) -> dict[str, str]:
    """Ground truth from the first folder level, e.g. TEST/O/x.jpg -> 'O' -> alias -> Organic."""
    labels: dict[str, str] = {}
    unknown: dict[str, int] = {}
    for rel, _ in images:
        parts = rel.split("/")
        if len(parts) < 2:
            raise LabelsError(f"--labels-from-folders needs <class folder>/<image>; '{rel}' is not in a class folder")
        canon = mapping.ground_truth_category(parts[0])
        if canon is None:
            unknown[parts[0]] = unknown.get(parts[0], 0) + 1
        else:
            labels[rel] = canon
    if unknown:
        raise LabelsError(f"folder names not recognised as categories or aliases: {unknown}. "
                          f"Known: {mapping.category_names}")
    return labels


def reconcile(images: list[tuple[str, Path]], labels: dict[str, str]) -> None:
    """Every image needs a label and every label needs an image (no silent exclusions)."""
    names = {n for n, _ in images}
    missing_labels = sorted(names - set(labels))
    missing_images = sorted(set(labels) - names)
    problems = []
    if missing_labels:
        problems.append(f"{len(missing_labels)} image(s) have no ground-truth label: {missing_labels[:10]}")
    if missing_images:
        problems.append(f"{len(missing_images)} labelled file(s) not found in image folder: {missing_images[:10]}")
    if problems:
        raise LabelsError("; ".join(problems))


@dataclass
class EvaluationRow:
    result: ImageResult
    ground_truth: Optional[str]

    @property
    def correct(self) -> Optional[bool]:
        if self.ground_truth is None:
            return None
        return self.result.predicted_category is not None and self.result.predicted_category == self.ground_truth


RESULT_COLUMNS = [
    "image_filename", "ground_truth_category", "raw_top1_class", "raw_top1_probability",
    "predicted_category", "predicted_category_score", "mapping_status", "ecoguard_status",
    "review_reason", "disposal_action", "inference_time_ms", "correct",
    # supporting columns (debugging / threshold sweep)
    "raw_top1_index", "raw_top5", "category_scores", "category_margin", "mapped_mass",
    "top1_rule", "top1_material", "preprocess_time_ms", "end_to_end_time_ms", "error",
]


def _fmt(v: Optional[float], nd: int = 6) -> str:
    return "" if v is None else f"{v:.{nd}f}"


def row_to_dict(row: EvaluationRow) -> dict[str, str]:
    r = row.result
    top1 = r.raw_top1
    correct = row.correct
    return {
        "image_filename": r.image_filename,
        "ground_truth_category": row.ground_truth or "",
        "raw_top1_class": top1.label if top1 else "",
        "raw_top1_probability": _fmt(top1.probability if top1 else None),
        "predicted_category": r.predicted_category or "",
        "predicted_category_score": _fmt(r.predicted_category_score),
        "mapping_status": r.mapping_status,
        "ecoguard_status": r.ecoguard_status,
        "review_reason": r.review_reason,
        "disposal_action": r.disposal_action,
        "inference_time_ms": _fmt(r.inference_time_ms, 3),
        "correct": "" if correct is None else str(correct),
        "raw_top1_index": "" if top1 is None else str(top1.index),
        "raw_top5": "|".join(f"{p.label}:{p.probability:.4f}" for p in r.raw_topk),
        "category_scores": "|".join(f"{k}:{v:.6f}" for k, v in r.category_scores.items()),
        "category_margin": _fmt(r.category_margin),
        "mapped_mass": _fmt(r.mapped_mass),
        "top1_rule": r.top1_rule or "",
        "top1_material": r.top1_material or "",
        "preprocess_time_ms": _fmt(r.preprocess_time_ms, 3),
        "end_to_end_time_ms": _fmt(r.end_to_end_time_ms, 3),
        "error": r.error or "",
    }


def write_results_csv(rows: Iterable[EvaluationRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        w.writeheader()
        for row in rows:
            w.writerow(row_to_dict(row))


def percentile(values: list[float], q: float) -> Optional[float]:
    """Linear-interpolation percentile (same as numpy.percentile default)."""
    if not values:
        return None
    xs = sorted(values)
    pos = (len(xs) - 1) * q / 100.0
    lo, hi = int(pos), min(int(pos) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def latency_stats(values: list[float]) -> dict[str, Optional[float]]:
    return {
        "mean_ms": statistics.fmean(values) if values else None,
        "median_ms": statistics.median(values) if values else None,
        "p95_ms": percentile(values, 95),
        "min_ms": min(values) if values else None,
        "max_ms": max(values) if values else None,
        "count": len(values),
    }


def summarize(rows: list[EvaluationRow], *, labelled: bool, predictable: Optional[list[str]] = None,
              unsupported: Optional[Iterable[str]] = None) -> dict:
    total = len(rows)
    accepted = [r for r in rows if r.result.ecoguard_status == ACCEPT]
    reviewed = [r for r in rows if r.result.ecoguard_status == REVIEW]
    with_category = [r for r in rows if r.result.predicted_category is not None]
    failed = [r for r in rows if r.result.error]
    infer = [r.result.inference_time_ms for r in rows if r.result.inference_time_ms is not None]
    e2e = [r.result.end_to_end_time_ms for r in rows if r.result.end_to_end_time_ms is not None]
    pre = [r.result.preprocess_time_ms for r in rows if r.result.preprocess_time_ms is not None]
    s: dict = {
        "total_images": total,
        "images_with_category_prediction": len(with_category),
        "images_without_category_prediction": total - len(with_category),
        "category_coverage": (len(with_category) / total) if total else None,
        "failed_images": len(failed),
        "accepted_count": len(accepted),
        "review_count": len(reviewed),
        "review_rate": (len(reviewed) / total) if total else None,
    }
    if labelled:
        correct = sum(1 for r in rows if r.correct)
        acc_correct = sum(1 for r in accepted if r.correct)
        s.update({
            "total_correct": correct,
            "total_incorrect": total - correct,
            "overall_accuracy": (correct / total) if total else None,
            "accepted_correct": acc_correct,
            "accepted_incorrect": len(accepted) - acc_correct,
            "accepted_only_accuracy": (acc_correct / len(accepted)) if accepted else None,
            "accepted_only_accuracy_note": None if accepted else "undefined: zero accepted predictions",
            "reviewed_but_category_correct": sum(1 for r in reviewed if r.correct),
            "per_category": _per_category(rows),
        })
        unsupported = set(unsupported or [])
        unsup_rows = [r for r in rows if r.ground_truth in unsupported]
        s["unsupported_ground_truth_images"] = len(unsup_rows)
        s["unsupported_ground_truth_note"] = (
            f"{len(unsup_rows)} image(s) belong to categories the model cannot predict "
            f"({sorted(unsupported)}); they are counted as incorrect in overall_accuracy." if unsup_rows else None)
        accepted_unsup = [r for r in unsup_rows if r.result.ecoguard_status == ACCEPT]
        s["unsupported_ground_truth_accepted_into_a_bin"] = len(accepted_unsup)
        if predictable:
            s["metrics"] = classification_metrics(rows, predictable)
            s["accepted_only_metrics"] = classification_metrics(accepted, predictable)
    else:
        s.update({k: None for k in ("total_correct", "total_incorrect", "overall_accuracy", "accepted_only_accuracy")})
        s["accuracy_note"] = "No ground-truth labels supplied; accuracy not computed."
    s["inference_latency_mean_ms"] = latency_stats(infer)["mean_ms"]
    s["inference_latency_median_ms"] = latency_stats(infer)["median_ms"]
    s["inference_latency_p95_ms"] = latency_stats(infer)["p95_ms"]
    s["preprocess_latency"] = latency_stats(pre)
    s["end_to_end_latency"] = latency_stats(e2e)
    return s


NO_PREDICTION = "<none>"


def classification_metrics(rows: list[EvaluationRow], predictable: list[str]) -> dict:
    """Confusion matrix + per-class precision / recall / F1 over ALL rows (REVIEW included).

    Rows are ground-truth categories; columns are predicted categories plus '<none>'
    (image failed or no category). Ground-truth categories the model cannot predict
    (e.g. Residual for a 2-class model) appear as rows with recall 0 by construction.
    """
    truths = sorted({r.ground_truth for r in rows if r.ground_truth is not None})
    cols = list(predictable) + [NO_PREDICTION]
    matrix = {t: {c: 0 for c in cols} for t in sorted(set(truths) | set(predictable))}
    for r in rows:
        if r.ground_truth is None:
            continue
        pred = r.result.predicted_category or NO_PREDICTION
        matrix[r.ground_truth][pred if pred in matrix[r.ground_truth] else NO_PREDICTION] += 1
    per_class = {}
    for c in matrix:
        tp = matrix[c].get(c, 0)
        support = sum(matrix[c].values())
        predicted = sum(matrix[t].get(c, 0) for t in matrix)
        precision = tp / predicted if predicted else None
        recall = tp / support if support else None
        f1 = (2 * precision * recall / (precision + recall)) if precision and recall else (0.0 if support and predicted else None)
        per_class[c] = {"support": support, "predicted": predicted, "true_positive": tp, "precision": precision,
                        "recall": recall, "f1": f1, "predictable": c in predictable}
    scored = [v for v in per_class.values() if v["support"] and v["f1"] is not None]
    macro = {k: (sum(v[k] or 0.0 for v in scored) / len(scored) if scored else None) for k in ("precision", "recall", "f1")}
    return {"confusion_matrix": {"rows": "ground truth", "columns": "predicted", "matrix": matrix},
            "per_class": per_class, "macro_avg_over_classes_with_support": macro}


def _per_category(rows: list[EvaluationRow]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for r in rows:
        gt = r.ground_truth or "?"
        d = out.setdefault(gt, {"images": 0, "correct": 0, "accepted": 0})
        d["images"] += 1
        d["correct"] += int(bool(r.correct))
        d["accepted"] += int(r.result.ecoguard_status == ACCEPT)
    for d in out.values():
        d["accuracy"] = d["correct"] / d["images"]
    return out
