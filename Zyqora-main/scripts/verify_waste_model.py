#!/usr/bin/env python3
"""Integration gate for the fine-tuned waste model (run ON THE RASPBERRY PI with the real .pte).

Checks, in order (any failure -> exit 1, nothing is activated):
  1. models/mobilenetv2_waste_fp32_xnnpack.pte exists; SHA-256 and size recorded
     (and compared with --expected-sha256 when given, e.g. the hash from the MacBook)
  2. it loads through ExecuTorch with XnnpackBackend registered
  3. input [1,3,224,224] and output [1,2] (the waste profile's decoder contract)
  4. finite logits on a deterministic input
  5. on LABELLED images (folder layout O/..., R/... or Organic/..., Recyclable/...):
     accuracy >= --min-accuracy. This catches a reversed class index (0 Organic, 1 Recyclable)
     or wrong preprocessing, which would collapse accuracy. Use validation images, not TEST.

    python scripts/verify_waste_model.py --images /path/to/VAL_SAMPLE
    python scripts/verify_waste_model.py --images /path/to/VAL_SAMPLE --activate

--activate (only after all checks pass) sets `active: waste_mobilenetv2` and pins the
verified sha256 in config/models.yaml. Writes models/mobilenetv2_waste_fp32_xnnpack.json.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401

from zyqora.benchmark import platform_facts
from zyqora.config import load_app_config
from zyqora.evaluator import EvaluationRow, discover_images, labels_from_folders, summarize
from zyqora.model_runtime import ModelLoadError, RuntimeUnavailableError
from zyqora.pipeline import ZyqoraPipeline
from zyqora.profiles import DEFAULT_MODELS_CONFIG, file_sha256, load_models_config, load_profile

PROFILE = "waste_mobilenetv2"


def check(ok: bool, label: str, detail: str = "") -> bool:
    print(f"{'PASS' if ok else 'FAIL'}  {label}{(': ' + detail) if detail else ''}")
    return ok


def activate(models_yaml: Path, sha: str) -> None:
    text = models_yaml.read_text(encoding="utf-8")
    text, n1 = re.subn(r"(?m)^active:\s*\S+", f"active: {PROFILE}", text, count=1)
    block = re.search(rf"(?ms)^  {PROFILE}:\n(.*?)(?=^  \S|\Z)", text)
    n2 = 0
    if block:
        new_block, n2 = re.subn(r"(?m)^(    sha256:)\s*\S+", rf"\g<1> {sha}", block.group(0), count=1)
        text = text[: block.start()] + new_block + text[block.end():]
    if n1 != 1 or n2 != 1:
        raise SystemExit(f"Could not update {models_yaml} automatically; set active: {PROFILE} and sha256 by hand.")
    models_yaml.write_text(text, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", type=Path, required=True, help="LABELLED class-folder images (validation sample)")
    ap.add_argument("--max-per-class", type=int, default=50)
    ap.add_argument("--min-accuracy", type=float, default=0.85)
    ap.add_argument("--expected-sha256", default=None, help="hash computed on the export machine")
    ap.add_argument("--activate", action="store_true", help="make waste_mobilenetv2 the active profile on success")
    ap.add_argument("--models-config", type=Path, default=DEFAULT_MODELS_CONFIG)
    args = ap.parse_args()

    app = load_app_config()
    models = load_models_config(app, args.models_config)
    profile = models.get(PROFILE)
    ok = True
    pte = profile.pte
    if not check(pte.is_file(), "model file present", str(pte)):
        return 1
    sha = file_sha256(pte)
    size = pte.stat().st_size
    check(True, "sha256", f"{sha} ({size:,} bytes, {size / 2**20:.2f} MiB)")
    if args.expected_sha256:
        ok &= check(sha == args.expected_sha256.lower(), "sha256 matches export machine")
    try:
        loaded = load_profile(profile, require_backend=app.require_backend)
    except (ModelLoadError, RuntimeUnavailableError, ValueError) as exc:
        check(False, "load through ExecuTorch with the waste profile", str(exc))
        return 1
    clf = loaded.classifier
    ok &= check("XnnpackBackend" in clf.registered_backends, "XNNPACK backend registered", str(clf.registered_backends))
    ok &= check(clf.output_shape == (1, 2), "output shape [1, 2]", str(clf.output_shape))
    logits = clf.infer(np.zeros(loaded.spec.input_shape, dtype=np.float32)).logits
    ok &= check(bool(np.isfinite(logits).all()), "finite logits on a zero tensor (value itself is meaningless)")

    images = discover_images(args.images)
    per_class: dict[str, list] = {}
    for rel, p in images:
        per_class.setdefault(rel.split("/")[0], []).append((rel, p))
    sample = [x for items in per_class.values() for x in items[: args.max_per_class]]
    try:
        truth = labels_from_folders(sample, loaded.category_set)
    except ValueError as exc:
        check(False, "labelled images", str(exc))
        return 1
    pipe = ZyqoraPipeline(clf, loaded.decoder, loaded.guard, loaded.spec)
    rows = [EvaluationRow(pipe.process(p, rel), truth[rel]) for rel, p in sample]
    s = summarize(rows, labelled=True, predictable=loaded.category_set.category_names)
    acc = s["overall_accuracy"] or 0.0
    for c, m in s["metrics"]["per_class"].items():
        if m["support"]:
            print(f"      {c:<11} support {m['support']:>4}  recall {m['recall'] or 0:.3f}")
    ok &= check(acc >= args.min_accuracy, f"accuracy on {len(rows)} labelled images >= {args.min_accuracy:.0%}",
                f"{acc:.1%}" + ("  <- if far below 50%, the class order may be reversed" if acc < 0.5 else ""))
    ok &= check(s["failed_images"] == 0, "no image failures", str(s["failed_images"]))

    meta = {"profile": PROFILE, "pte": pte.name, "sha256": sha, "size_bytes": size,
            "output_shape": list(clf.output_shape or []), "classes": list(loaded.category_set.class_names),
            "verification": {"passed": bool(ok), "labelled_images": len(rows), "accuracy": acc,
                             "images_folder": str(args.images), "min_accuracy": args.min_accuracy,
                             "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                             "platform": platform_facts(), "runtime": clf.describe()}}
    meta_path = pte.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {meta_path}")
    if not ok:
        print("RESULT: FAILED - production configuration NOT changed.")
        return 1
    if args.activate:
        activate(args.models_config, sha)
        print(f"RESULT: PASSED - {args.models_config} now uses active: {PROFILE} (sha256 pinned).")
    else:
        print("RESULT: PASSED - re-run with --activate to make the waste model the production model.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
