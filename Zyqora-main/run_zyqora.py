#!/usr/bin/env python3
"""Zyqora AI - offline, confidence-aware waste sorting on Raspberry Pi 5 (ExecuTorch + XNNPACK).

Classifies every image in a folder with a pretrained MobileNetV2 .pte, maps the
prediction to organiser categories (MaterialWise), decides ACCEPT / REVIEW
(EcoGuard), optionally speaks the instruction (VoiceSort) and records pending
GreenScore events for supervisor verification.

Examples
  Final organiser evaluation (requires mapping status ORGANISER):
    python run_zyqora.py --images data/organiser --labels data/organiser/labels.csv --out outputs/final
  Development run with the EXAMPLE mapping:
    python run_zyqora.py --images data/dev --labels data/dev/labels.csv --allow-example-mapping --out outputs/dev
  Tamil voice, record events for user U01:
    python run_zyqora.py --images data/demo --voice ta --greenscore-user U01 --allow-example-mapping

Outputs (in --out): results.csv, summary.json, annotated/*.jpg, run.log
Exit codes: 0 ok, 2 configuration/usage error, 3 model/runtime error, 4 data error.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from zyqora import __version__  # noqa: E402
from zyqora.benchmark import cpu_time_ns, platform_facts, read_proc_status  # noqa: E402
from zyqora.config import ConfigError, load_app_config  # noqa: E402
from zyqora.evaluator import (  # noqa: E402
    EvaluationRow, LabelsError, discover_images, read_labels_csv, reconcile, summarize, write_results_csv,
)
from zyqora.imagenet_labels import LabelsError as ImageNetLabelsError, load_labels  # noqa: E402
from zyqora.materialwise import MappingError, load_mapping  # noqa: E402
from zyqora.model_runtime import ExecuTorchClassifier, ModelLoadError, RuntimeUnavailableError  # noqa: E402
from zyqora.outputs import annotate, pick_samples, write_json  # noqa: E402
from zyqora.pipeline import ZyqoraPipeline  # noqa: E402
from zyqora.preprocessing import PreprocessSpec  # noqa: E402

EXIT_CONFIG, EXIT_MODEL, EXIT_DATA = 2, 3, 4


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", type=Path, required=True, help="folder of images (searched recursively)")
    ap.add_argument("--labels", type=Path, default=None,
                    help="CSV with image_filename,ground_truth_category (omit for inference-only)")
    ap.add_argument("--model", type=Path, default=None, help=".pte artifact (default: config/app.yaml model.pte)")
    ap.add_argument("--config", type=Path, default=ROOT / "config" / "app.yaml")
    ap.add_argument("--mapping", type=Path, default=None, help="override config mapping file")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "latest", help="output directory")
    ap.add_argument("--voice", choices=["off", "en", "ta"], default="off", help="VoiceSort language")
    ap.add_argument("--threads", type=int, default=None, help="ExecuTorch threadpool size (default: config)")
    ap.add_argument("--annotate", type=int, default=None, help="number of annotated samples (default: config)")
    ap.add_argument("--greenscore-user", default=None, help="record a PENDING GreenScore event per image for this user")
    ap.add_argument("--allow-example-mapping", action="store_true",
                    help="permit a run with the EXAMPLE mapping (results marked DEVELOPMENT, not final)")
    ap.add_argument("--quiet", action="store_true", help="only print the summary")
    return ap.parse_args(argv)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fmt_pct(x) -> str:
    return "undefined" if x is None else f"{x:.1%}"


def main(argv=None) -> int:
    args = parse_args(argv)
    started = datetime.now(timezone.utc)

    # ---- configuration -------------------------------------------------
    try:
        app = load_app_config(args.config)
        labels = load_labels(app.labels)
        mapping = load_mapping(args.mapping or app.mapping, labels)
        spec = PreprocessSpec.from_file(app.contract)
    except (ConfigError, MappingError, ImageNetLabelsError, OSError, ValueError) as exc:
        print(f"CONFIG ERROR: {exc}", file=sys.stderr)
        return EXIT_CONFIG
    if not mapping.is_organiser and not args.allow_example_mapping:
        print("CONFIG ERROR: config/mapping.json has status EXAMPLE (categories are not the organiser's).\n"
              "Configure the organiser categories and set status to ORGANISER for the final evaluation,\n"
              "or pass --allow-example-mapping for a development run.", file=sys.stderr)
        return EXIT_CONFIG
    if args.threads is not None and args.threads < 1:
        print("CONFIG ERROR: --threads must be >= 1", file=sys.stderr)
        return EXIT_CONFIG
    try:
        args.out.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(f"CONFIG ERROR: cannot create output directory {args.out}: {exc}", file=sys.stderr)
        return EXIT_CONFIG

    # ---- data ----------------------------------------------------------
    try:
        images = discover_images(args.images)
        if not images:
            raise LabelsError(f"no supported images (.jpg/.jpeg/.png/.bmp/.webp) in {args.images}")
        gt = read_labels_csv(args.labels, mapping) if args.labels else {}
        if args.labels:
            reconcile(images, gt)
    except (FileNotFoundError, LabelsError) as exc:
        print(f"DATA ERROR: {exc}", file=sys.stderr)
        return EXIT_DATA

    # ---- model (loaded once) --------------------------------------------
    pte = args.model or app.pte
    try:
        clf = ExecuTorchClassifier(pte, input_shape=spec.input_shape, num_threads=args.threads or app.num_threads,
                                   require_backend=app.require_backend)
    except (ModelLoadError, RuntimeUnavailableError, ValueError) as exc:
        print(f"MODEL ERROR: {exc}", file=sys.stderr)
        return EXIT_MODEL
    pipeline = ZyqoraPipeline(clf, labels, mapping, app.ecoguard, spec)
    # One untimed warm-up forward pass so the first image's latency is not a cold-start outlier.
    import numpy as np

    clf.infer(np.zeros(spec.input_shape, dtype=np.float32))

    voice = None
    if args.voice != "off":
        from zyqora.voicesort import VoiceSort

        voice = VoiceSort(args.voice)
    greenscore = None
    run_id = started.strftime("%Y%m%dT%H%M%SZ")
    if args.greenscore_user:
        from zyqora.greenscore import GreenScore, GreenScoreError

        try:
            app.greenscore_db.parent.mkdir(parents=True, exist_ok=True)
            greenscore = GreenScore(app.greenscore_db, app.points_correct, app.points_incorrect, mapping.actions)
            greenscore.ensure_user(args.greenscore_user)
        except (GreenScoreError, OSError) as exc:
            print(f"CONFIG ERROR: GreenScore: {exc}", file=sys.stderr)
            return EXIT_CONFIG

    scope = "FINAL_ORGANISER_TEST" if (mapping.is_organiser and args.labels) else (
        "DEVELOPMENT" if args.labels else "INFERENCE_ONLY")
    if not args.quiet:
        print(f"Zyqora AI {__version__} | ExecuTorch {clf.describe()['executorch_version']} | "
              f"{pte.name} ({pte.stat().st_size / 1e6:.2f} MB) | backends {clf.registered_backends}")
        print(f"threads: {clf.thread_count} ({clf.thread_control}) | mapping: {mapping.status} | "
              f"EcoGuard threshold {app.ecoguard.threshold} (calibrated={app.ecoguard.calibrated}, "
              f"frozen={app.ecoguard.frozen}) | scope: {scope}")
        if scope == "FINAL_ORGANISER_TEST" and not app.ecoguard.frozen:
            print("WARNING: EcoGuard threshold is not frozen for a final run.")
        print()

    # ---- run -------------------------------------------------------------
    rows: list[EvaluationRow] = []
    processed: list[tuple] = []
    events: list[int] = []
    cpu0, wall0 = cpu_time_ns(), time.perf_counter_ns()
    for rel, path in images:
        result = pipeline.process(path, display_name=rel)
        row = EvaluationRow(result, gt.get(rel) if args.labels else None)
        rows.append(row)
        processed.append((result, path))
        if not args.quiet:
            top1 = result.raw_top1
            mark = "" if row.correct is None else ("  correct" if row.correct else f"  WRONG (truth: {row.ground_truth})")
            if result.error:
                print(f"{rel}: FAILED - {result.error}{mark}")
            else:
                print(f"{rel}: {top1.label} {top1.probability:.1%} -> {result.predicted_category} "
                      f"{result.predicted_category_score:.1%} [{result.mapping_status}] -> {result.ecoguard_status}"
                      f"{' (' + result.review_reason + ')' if result.review_reason else ''} -> {result.disposal_action}"
                      f" | {result.inference_time_ms:.1f} ms{mark}")
        if voice is not None:
            voice.announce(result.disposal_action)
        if greenscore is not None and not result.error:
            events.append(greenscore.record_event(args.greenscore_user, rel, result.predicted_category,
                                                  result.disposal_action, result.ecoguard_status, run_id))
    wall_ns, cpu_ns = time.perf_counter_ns() - wall0, cpu_time_ns() - cpu0
    mem = read_proc_status()

    # ---- outputs ---------------------------------------------------------
    write_results_csv(rows, args.out / "results.csv")
    n_annot = app.annotate_count if args.annotate is None else max(0, args.annotate)
    annotated = []
    for r, p in pick_samples(processed, n_annot):
        out = args.out / "annotated" / (Path(r.image_filename).with_suffix("").as_posix().replace("/", "_") + "_annotated.jpg")
        annotate(p, r, out)
        annotated.append({"file": str(out.relative_to(args.out)), "image": r.image_filename,
                          "ecoguard_status": r.ecoguard_status, "predicted_category": r.predicted_category})

    summary = summarize(rows, labelled=bool(args.labels))
    meta = clf.describe()
    summary.update({
        "evaluation_scope": scope,
        "model_name": "MobileNetV2 (TorchVision IMAGENET1K_V1)",
        "model_precision": "int8" if "int8" in pte.name.lower() else "fp32",
        "model_file": pte.name,
        "model_size_mb": meta["pte_size_mb"],
        "model_sha256": sha256(pte),
        "runtime": meta,
        "warmup_runs": 1,
        "memory_usage_mb": mem["vm_rss_mb"],
        "memory": {**mem, "note": "VmRSS = current resident memory, VmHWM = peak resident memory of the whole "
                                   "Python process (includes interpreter, torch and numpy), not model-only memory"},
        "cpu_seconds_total": cpu_ns / 1e9,
        "cpu_seconds_per_image": cpu_ns / 1e9 / len(rows),
        "wall_seconds_total": wall_ns / 1e9,
        "mapping": {"file": mapping.source, "status": mapping.status, "aggregation": mapping.aggregation,
                    "categories": mapping.categories, "class_rules": len(mapping.rules)},
        "ecoguard": {"threshold": app.ecoguard.threshold, "min_margin": app.ecoguard.min_margin,
                     "require_top1_mapped": app.ecoguard.require_top1_mapped,
                     "calibrated": app.ecoguard.calibrated, "frozen": app.ecoguard.frozen},
        "annotated_samples": annotated,
        "greenscore_pending_events": events,
        "voice": args.voice,
        "platform": platform_facts(),
        "zyqora_version": __version__,
        "started_utc": started.isoformat(timespec="seconds"),
        "images_folder": str(args.images),
        "labels_file": str(args.labels) if args.labels else None,
    })
    write_json(summary, args.out / "summary.json")

    # ---- report ----------------------------------------------------------
    print("\n========== Zyqora AI summary ==========")
    print(f"Scope:              {scope}  (mapping {mapping.status})")
    print(f"Images:             {summary['total_images']}  (failed to load: {summary['failed_images']}, "
          f"with category prediction: {summary['images_with_category_prediction']})")
    if args.labels:
        print(f"Correct:            {summary['total_correct']} / {summary['total_images']}  "
              f"-> overall accuracy {fmt_pct(summary['overall_accuracy'])}")
        print(f"Accepted-only acc.: {fmt_pct(summary['accepted_only_accuracy'])}  "
              f"({summary['accepted_correct']} / {summary['accepted_count']} accepted)")
    print(f"EcoGuard:           ACCEPT {summary['accepted_count']}  REVIEW {summary['review_count']}  "
          f"(review rate {fmt_pct(summary['review_rate'])})")
    if summary["inference_latency_median_ms"] is not None:
        print(f"Inference (model):  median {summary['inference_latency_median_ms']:.2f} ms, "
              f"mean {summary['inference_latency_mean_ms']:.2f} ms, p95 {summary['inference_latency_p95_ms']:.2f} ms")
    print(f"Model:              {pte.name}  {summary['model_size_mb']:.2f} MB  ({summary['model_precision']})")
    if mem["vm_rss_mb"] is not None:
        print(f"Process memory:     RSS {mem['vm_rss_mb']:.0f} MB, peak {mem['vm_hwm_mb']:.0f} MB")
    print(f"Device:             {summary['platform']['device_model'] or summary['platform']['machine']}")
    print(f"Outputs:            {args.out / 'results.csv'}, {args.out / 'summary.json'}, "
          f"{len(annotated)} annotated image(s)")
    if events:
        print(f"GreenScore:         {len(events)} PENDING event(s) {events[0]}..{events[-1]} for "
              f"{args.greenscore_user}; verify with: python -m zyqora.greenscore verify --event-id {events[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
