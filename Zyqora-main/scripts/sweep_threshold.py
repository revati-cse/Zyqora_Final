#!/usr/bin/env python3
"""Sweep EcoGuard thresholds on DEVELOPMENT results (never the final organiser test set).

1. Run Zyqora on labelled development images (separate from the organiser set):
       python run_zyqora.py --images data/dev --labels data/dev/labels.csv --out outputs/dev --allow-example-mapping
2. Sweep:
       python scripts/sweep_threshold.py --results outputs/dev/results.csv
3. Pick a threshold, write it to config/app.yaml, set calibrated: true and frozen: true.

Reports, per threshold: accepted, reviewed, coverage, accepted-only accuracy,
incorrect accepted decisions and review rate. The predicted category does not
depend on the threshold, so overall accuracy is constant across rows.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from zyqora.config import load_app_config
from zyqora.profiles import load_models_config
from zyqora.ecoguard import EcoGuardConfig, decide
from zyqora.schemas import ACCEPT


def load_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"No rows in {path}")
    if not rows[0].get("ground_truth_category"):
        raise SystemExit("results.csv has no ground_truth_category; the sweep needs labelled dev results")
    return rows


def sweep(rows: list[dict], thresholds: list[float], min_margin: float, require_top1_mapped: bool) -> list[dict]:
    table = []
    n = len(rows)
    for t in thresholds:
        cfg = EcoGuardConfig(threshold=t, min_margin=min_margin, require_top1_mapped=require_top1_mapped)
        acc = acc_ok = 0
        for r in rows:
            score = float(r["predicted_category_score"]) if r["predicted_category_score"] else None
            margin = float(r["category_margin"]) if r["category_margin"] else None
            d = decide(score, r["mapping_status"], margin, cfg)
            if d.status == ACCEPT:
                acc += 1
                acc_ok += r["correct"] == "True"
        table.append({
            "threshold": round(t, 4),
            "accepted": acc,
            "reviewed": n - acc,
            "coverage": acc / n,
            "accepted_only_accuracy": (acc_ok / acc) if acc else None,
            "incorrect_accepted": acc - acc_ok,
            "review_rate": (n - acc) / n,
        })
    return table


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, required=True, help="results.csv from a labelled DEV run")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--stop", type=float, default=0.95)
    ap.add_argument("--step", type=float, default=0.05)
    ap.add_argument("--out", type=Path, default=None, help="default: <results dir>/threshold_sweep.csv")
    ap.add_argument("--profile", default=None, help="model profile (default: the one recorded in summary.json)")
    ap.add_argument("--extra", type=float, nargs="*", default=[0.8, 0.9, 0.95, 0.98],
                    help="extra thresholds to include (default: 0.8 0.9 0.95 0.98)")
    args = ap.parse_args()

    summary_path = args.results.parent / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
    if summary_path.is_file():
        scope = summary.get("evaluation_scope")
        if scope in ("FINAL_ORGANISER_TEST", "INDEPENDENT_TEST"):
            print(f"ERROR: these results are from a test set ({scope}). "
                  "Tuning the threshold on test labels is not allowed.", file=sys.stderr)
            return 2

    app = load_app_config()
    models = load_models_config(app)
    profile = models.get(args.profile or summary.get("model_profile") or models.active)
    guard = profile.ecoguard
    print(f"profile: {profile.name} (current threshold {guard.threshold})")
    rows = load_rows(args.results)
    n_steps = int(round((args.stop - args.start) / args.step))
    thresholds = [args.start + i * args.step for i in range(n_steps + 1)]
    if args.extra:
        thresholds = sorted(set(round(t, 4) for t in thresholds + args.extra))
    table = sweep(rows, thresholds, guard.min_margin, guard.require_top1_mapped)

    out = args.out or args.results.parent / "threshold_sweep.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)

    overall = sum(r["correct"] == "True" for r in rows) / len(rows)
    print(f"{len(rows)} dev images, overall category accuracy {overall:.1%} (threshold-independent)")
    print(f"min_margin={guard.min_margin} require_top1_mapped={guard.require_top1_mapped}")
    print(f"{'thr':>5} {'acc':>4} {'rev':>4} {'coverage':>8} {'acc-only':>8} {'wrong-acc':>9} {'review%':>7}")
    for r in table:
        ao = "undef" if r["accepted_only_accuracy"] is None else f"{r['accepted_only_accuracy']:.1%}"
        print(f"{r['threshold']:5.2f} {r['accepted']:4d} {r['reviewed']:4d} {r['coverage']:8.1%} {ao:>8} "
              f"{r['incorrect_accepted']:9d} {r['review_rate']:7.1%}")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
