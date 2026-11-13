#!/usr/bin/env python3
"""Zyqora AI - offline, confidence-aware waste sorting on Raspberry Pi 5 (ExecuTorch + XNNPACK).

Classifies every image in a folder with the selected model profile
(config/models.yaml), decides ACCEPT / REVIEW (EcoGuard), optionally speaks the
instruction (VoiceSort) and records pending GreenScore events.

Model profiles
  waste_mobilenetv2     fine-tuned MobileNetV2, 2 classes (Organic, Recyclable), exact 224x224 resize
  imagenet_mobilenetv2  ImageNet MobileNetV2 (1000 classes) + MaterialWise mapping (baseline / fallback)

Examples
  Independent TEST evaluation of the waste model (labels from TEST/O, TEST/R folders):
    python run_zyqora.py --profile waste_mobilenetv2 --images /data/TEST --labels-from-folders \\
        --dataset-role test --quiet --out outputs/test_waste
  Final organiser evaluation:
    python run_zyqora.py --images data/organiser --labels data/organiser/labels.csv --dataset-role organiser --out outputs/final
  Development run with the ImageNet EXAMPLE mapping:
    python run_zyqora.py --profile imagenet_mobilenetv2 --images data/dev --labels data/dev/labels.csv --allow-example-mapping

Outputs (in --out): results.csv, summary.json, annotated/*.jpg
Exit codes: 0 ok, 2 configuration/usage error, 3 model/runtime error, 4 data error.
"""

from __future__ import annotations

import argparse
import dataclasses
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
    EvaluationRow, LabelsError, discover_images, labels_from_folders, read_labels_csv, reconcile, summarize,
    write_results_csv,
)
from zyqora.logs import setup_logging  # noqa: E402
from zyqora.model_runtime import ModelLoadError, RuntimeUnavailableError  # noqa: E402
from zyqora.outputs import annotate, pick_samples, write_json  # noqa: E402
from zyqora.pipeline import ZyqoraPipeline  # noqa: E402
from zyqora.profiles import DEFAULT_MODELS_CONFIG, file_sha256, load_models_config, load_with_fallback  # noqa: E402

EXIT_CONFIG, EXIT_MODEL, EXIT_DATA = 2, 3, 4
SCOPES = {"dev": "DEVELOPMENT", "test": "INDEPENDENT_TEST", "organiser": "FINAL_ORGANISER_TEST"}


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", type=Path, required=True, help="folder of images (searched recursively)")
    lab = ap.add_mutually_exclusive_group()
    lab.add_argument("--labels", type=Path, default=None, help="CSV with image_filename,ground_truth_category")
    lab.add_argument("--labels-from-folders", action="store_true",
                     help="ground truth = first sub-folder name (e.g. TEST/O, TEST/R), matched via category aliases")
    ap.add_argument("--dataset-role", choices=sorted(SCOPES), default=None,
                    help="dev | test | organiser (recorded in summary.json; threshold sweeps refuse test/organiser)")
    ap.add_argument("--exclude-list", type=Path, default=None,
                    help="file of relative image paths to exclude (e.g. leakage duplicates); recorded in summary")
    ap.add_argument("--profile", default=None, help="model profile from config/models.yaml (default: its 'active')")
    ap.add_argument("--allow-fallback", action="store_true",
                    help="if the profile cannot be loaded, use the fallback profile (reported in summary.json)")
    ap.add_argument("--model", type=Path, default=None, help="override the profile's .pte path")
    ap.add_argument("--config", type=Path, default=ROOT / "config" / "app.yaml")
    ap.add_argument("--models-config", type=Path, default=DEFAULT_MODELS_CONFIG)
    ap.add_argument("--mapping", type=Path, default=None, help="override mapping file (imagenet profile only)")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "latest", help="output directory")
    ap.add_argument("--voice", choices=["off", "en", "ta"], default="off", help="VoiceSort language")
    ap.add_argument("--threads", type=int, default=None, help="ExecuTorch threadpool size (default: config)")
    ap.add_argument("--annotate", type=int, default=None, help="number of annotated samples (default: config)")
    ap.add_argument("--greenscore-user", default=None, help="record a PENDING GreenScore event per image for this user")
    ap.add_argument("--allow-example-mapping", action="store_true",
                    help="permit the imagenet profile with the EXAMPLE mapping (results marked DEVELOPMENT)")
    ap.add_argument("--quiet", action="store_true", help="only print progress and the summary")
    return ap.parse_args(argv)


def fmt_pct(x) -> str:
    return "undefined" if x is None else f"{x:.1%}"


def main(argv=None) -> int:
    args = parse_args(argv)
    started = datetime.now(timezone.utc)
    labelled = bool(args.labels or args.labels_from_folders)
    setup_logging()

    # ---- configuration + model (loaded once) ----------------------------
    if args.threads is not None and args.threads < 1:
        print("CONFIG ERROR: --threads must be >= 1", file=sys.stderr)
        return EXIT_CONFIG
    try:
        app = load_app_config(args.config)
        models = load_models_config(app, args.models_config)
        name = args.profile or models.active
        profile = models.get(name)
        if args.model:
            profile = dataclasses.replace(profile, pte=args.model.resolve(), sha256=None)
        if args.mapping:
            if profile.decoder != "imagenet_mapping":
                raise ConfigError("--mapping only applies to the imagenet_mapping profile")
            profile = dataclasses.replace(profile, mapping=args.mapping)
        models = dataclasses.replace(models, profiles={**models.profiles, name: profile})
    except (ConfigError, OSError, ValueError) as exc:
        print(f"CONFIG ERROR: {exc}", file=sys.stderr)
        return EXIT_CONFIG
    try:
        loaded = load_with_fallback(models, name, allow_fallback=args.allow_fallback,
                                    num_threads=args.threads or app.num_threads, require_backend=app.require_backend)
    except (ModelLoadError, RuntimeUnavailableError) as exc:
        print(f"MODEL ERROR: {exc}", file=sys.stderr)
        return EXIT_MODEL
    except (ConfigError, OSError, ValueError) as exc:
        print(f"CONFIG ERROR: {exc}", file=sys.stderr)
        return EXIT_CONFIG
    cats = loaded.category_set
    guard = loaded.guard
    if loaded.profile.decoder == "imagenet_mapping" and not cats.is_organiser and not args.allow_example_mapping:
        print("CONFIG ERROR: the imagenet profile's mapping has status EXAMPLE (categories are not the organiser's).\n"
              "Configure the organiser categories (status ORGANISER), use --profile waste_mobilenetv2,\n"
              "or pass --allow-example-mapping for a development run.", file=sys.stderr)
        return EXIT_CONFIG
    try:
        args.out.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(f"CONFIG ERROR: cannot create output directory {args.out}: {exc}", file=sys.stderr)
        return EXIT_CONFIG

    # ---- data ----------------------------------------------------------
    excluded: list[str] = []
    try:
        images = discover_images(args.images)
        if args.exclude_list:
            drop = {ln.strip() for ln in args.exclude_list.read_text(encoding="utf-8").splitlines() if ln.strip()}
            excluded = sorted(rel for rel, _ in images if rel in drop)
            images = [(rel, p) for rel, p in images if rel not in drop]
        if not images:
            raise LabelsError(f"no supported images (.jpg/.jpeg/.png/.bmp/.webp) in {args.images}")
        if args.labels:
            gt = read_labels_csv(args.labels, cats)
            gt = {k: v for k, v in gt.items() if k not in set(excluded)}
            reconcile(images, gt)
        elif args.labels_from_folders:
            gt = labels_from_folders(images, cats)
        else:
            gt = {}
    except (FileNotFoundError, LabelsError) as exc:
        print(f"DATA ERROR: {exc}", file=sys.stderr)
        return EXIT_DATA

    role = args.dataset_role or ("organiser" if getattr(cats, "is_organiser", False) else "dev")
    scope = SCOPES[role] if labelled else "INFERENCE_ONLY"
    clf = loaded.classifier
    pipeline = ZyqoraPipeline(clf, loaded.decoder, guard, loaded.spec)
    import numpy as np

    clf.infer(np.zeros(loaded.spec.input_shape, dtype=np.float32))  # untimed warm-up

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
            greenscore = GreenScore(app.greenscore_db, app.points_correct, app.points_incorrect, cats.actions)
            greenscore.ensure_user(args.greenscore_user)
        except (GreenScoreError, OSError) as exc:
            print(f"CONFIG ERROR: GreenScore: {exc}", file=sys.stderr)
            return EXIT_CONFIG

    pte = loaded.profile.pte
    print(f"Zyqora AI {__version__} | profile {loaded.profile.name} ({loaded.profile.decoder}) | "
          f"{pte.name} ({pte.stat().st_size / 1e6:.2f} MB) | output {clf.output_shape} | ExecuTorch "
          f"{clf.describe()['executorch_version']} | backends {clf.registered_backends}")
    print(f"threads: {clf.thread_count} ({clf.thread_control}) | categories: {cats.category_names} ({cats.status}) | "
          f"EcoGuard threshold {guard.threshold} (calibrated={guard.calibrated}, frozen={guard.frozen}) | scope: {scope}")
    if loaded.fallback_used:
        print(f"WARNING: FALLBACK MODEL IN USE. {loaded.failures[0]}")
    if scope in ("INDEPENDENT_TEST", "FINAL_ORGANISER_TEST") and not guard.frozen:
        print("WARNING: EcoGuard threshold is not frozen for this evaluation.")
    if excluded:
        print(f"Excluded {len(excluded)} image(s) listed in {args.exclude_list}")
    print()

    # ---- run -------------------------------------------------------------
    rows: list[EvaluationRow] = []
    processed: list[tuple] = []
    events: list[int] = []
    cpu0, wall0 = cpu_time_ns(), time.perf_counter_ns()
    for n, (rel, path) in enumerate(images, 1):
        result = pipeline.process(path, display_name=rel)
        row = EvaluationRow(result, gt.get(rel) if labelled else None)
        rows.append(row)
        processed.append((result, path))
        if args.quiet:
            if n % 250 == 0 or n == len(images):
                print(f"  {n}/{len(images)} images", flush=True)
        else:
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

    summary = summarize(rows, labelled=labelled, predictable=cats.category_names,
                        unsupported=getattr(cats, "unsupported_ground_truth", None))
    meta = loaded.describe()
    summary.update({
        "evaluation_scope": scope,
        "dataset_role": role if labelled else None,
        "model_profile": loaded.profile.name,
        "model_decoder": loaded.profile.decoder,
        "model_name": loaded.profile.description,
        "model_precision": "int8" if "int8" in pte.name.lower() else "fp32",
        "model_file": pte.name,
        "model_size_mb": meta["pte_size_mb"],
        "model_sha256": file_sha256(pte),
        "fallback_used": loaded.fallback_used,
        "runtime": meta,
        "warmup_runs": 1,
        "throughput_images_per_second": len(rows) / (wall_ns / 1e9) if wall_ns else None,
        "cpu_utilisation_percent_of_one_core": 100.0 * cpu_ns / wall_ns if wall_ns else None,
        "memory_usage_mb": mem["vm_rss_mb"],
        "memory": {**mem, "note": "VmRSS = current resident memory, VmHWM = peak resident memory of the whole "
                                   "Python process (includes interpreter, torch and numpy), not model-only memory"},
        "cpu_seconds_total": cpu_ns / 1e9,
        "cpu_seconds_per_image": cpu_ns / 1e9 / len(rows),
        "wall_seconds_total": wall_ns / 1e9,
        "categories": {"file": cats.source, "status": cats.status, "aggregation": cats.aggregation,
                       "categories": cats.categories,
                       "unsupported_ground_truth": sorted(getattr(cats, "unsupported_ground_truth", []) or [])},
        "ecoguard": {"threshold": guard.threshold, "min_margin": guard.min_margin,
                     "require_top1_mapped": guard.require_top1_mapped,
                     "calibrated": guard.calibrated, "frozen": guard.frozen},
        "excluded_images": {"count": len(excluded), "list_file": str(args.exclude_list) if args.exclude_list else None},
        "annotated_samples": annotated,
        "greenscore_pending_events": events,
        "voice": args.voice,
        "platform": platform_facts(),
        "zyqora_version": __version__,
        "started_utc": started.isoformat(timespec="seconds"),
        "images_folder": str(args.images),
        "labels_source": str(args.labels) if args.labels else ("folder names" if args.labels_from_folders else None),
    })
    write_json(summary, args.out / "summary.json")

    # ---- report ----------------------------------------------------------
    print("\n========== Zyqora AI summary ==========")
    print(f"Scope:              {scope}  | profile {loaded.profile.name}{' (FALLBACK)' if loaded.fallback_used else ''}")
    print(f"Images:             {summary['total_images']}  (failed to load: {summary['failed_images']}, "
          f"with category prediction: {summary['images_with_category_prediction']}"
          f"{', excluded: ' + str(len(excluded)) if excluded else ''})")
    if labelled:
        print(f"Correct:            {summary['total_correct']} / {summary['total_images']}  "
              f"-> overall accuracy {fmt_pct(summary['overall_accuracy'])}")
        print(f"Accepted-only acc.: {fmt_pct(summary['accepted_only_accuracy'])}  "
              f"({summary['accepted_correct']} / {summary['accepted_count']} accepted; "
              f"{summary['accepted_incorrect']} incorrect accepted)")
        if summary.get("unsupported_ground_truth_images"):
            print(f"Unsupported truth:  {summary['unsupported_ground_truth_images']} image(s) of categories the model "
                  f"cannot predict ({summary['unsupported_ground_truth_accepted_into_a_bin']} accepted into a bin)")
        for c, m in summary.get("metrics", {}).get("per_class", {}).items():
            if m["support"]:
                f = lambda v: "n/a" if v is None else f"{v:.3f}"  # noqa: E731
                print(f"  {c:<12} support {m['support']:>5}  precision {f(m['precision'])}  recall {f(m['recall'])}"
                      f"  F1 {f(m['f1'])}")
    print(f"EcoGuard:           ACCEPT {summary['accepted_count']}  REVIEW {summary['review_count']}  "
          f"(review rate {fmt_pct(summary['review_rate'])})")
    if summary["inference_latency_median_ms"] is not None:
        print(f"Inference (model):  median {summary['inference_latency_median_ms']:.2f} ms, "
              f"mean {summary['inference_latency_mean_ms']:.2f} ms, p95 {summary['inference_latency_p95_ms']:.2f} ms")
    print(f"Throughput:         {summary['throughput_images_per_second']:.1f} images/s (incl. decode + preprocessing), "
          f"CPU {summary['cpu_seconds_per_image'] * 1000:.1f} ms/image")
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
