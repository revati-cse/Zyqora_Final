#!/usr/bin/env python3
"""Benchmark the YOLO detector and the full YOLO + MobileNetV2 pipeline on THIS device.

    python scripts/benchmark_detector.py                                 # all available detector profiles
    python scripts/benchmark_detector.py --profiles yolo11n_640 yolo11n_320 --threads 2 4
    python scripts/benchmark_detector.py --images data/dev --pipeline    # + end-to-end mixed-waste latency

Per (profile, threads) a fresh subprocess measures:
  * load_ms           model load + first (warm-up) inference
  * model-only        runtime forward() on a letterboxed tensor (median / p95 / mean)
  * detect()          letterbox + forward + decode + NMS on a real image
  * cpu_ms            process CPU time (user+sys, all threads) per forward()
  * memory            VmRSS / VmHWM of that process (Linux /proc; None elsewhere)
  * throughput        forward() calls per second, single stream
  * pipeline (opt.)   MixedWastePipeline.analyze per image: detector + every crop's classification
Temperature is recorded before/after when the device exposes it. Power is NOT measured here;
no wattage is estimated from CPU load. Results: reports/benchmarks/<device>_detector_<profile>.json
(use --out-dir outputs/benchmark for scratch runs).
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401

_bootstrap.quiet_toolchain_logs()

from zyqora.benchmark import cpu_temperature_c, cpu_time_ns, platform_facts, read_proc_status  # noqa: E402
from zyqora.detection.yolo import YoloDetector, letterbox, load_detector_config  # noqa: E402
from zyqora.evaluator import latency_stats  # noqa: E402
from zyqora.paths import PROJECT_ROOT  # noqa: E402

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _images(folder: Path | None, limit: int) -> list[Path]:
    if not folder:
        return []
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXT)[:limit]


def worker(args) -> dict:
    from PIL import Image, ImageOps

    cfg = load_detector_config(args.config) if args.config else load_detector_config()
    prof = cfg.get(args.profile)
    if prof.runtime == "executorch" and args.threads:
        from zyqora.model_runtime import import_portable_lib
        import_portable_lib()._unsafe_reset_threadpool(args.threads)
    mem0 = read_proc_status()
    temp0 = cpu_temperature_c()
    t0 = time.perf_counter()
    det = YoloDetector(prof, cfg.settings, num_threads=args.threads)
    imgs = _images(Path(args.images) if args.images else None, args.max_images)
    if imgs:
        with Image.open(imgs[0]) as im:
            sample = ImageOps.exif_transpose(im).convert("RGB")
    else:
        sample = Image.fromarray((np.random.default_rng(0).random((480, 640, 3)) * 255).astype(np.uint8))
    det.detect(sample)
    load_ms = (time.perf_counter() - t0) * 1e3
    x, _ = letterbox(sample, prof.imgsz)
    for _ in range(args.warmup):
        det.runner.run(x)
    fwd, c0 = [], cpu_time_ns()
    w0 = time.perf_counter()
    for _ in range(args.runs):
        a = time.perf_counter_ns()
        det.runner.run(x)
        fwd.append((time.perf_counter_ns() - a) / 1e6)
    wall = time.perf_counter() - w0
    cpu_ms = (cpu_time_ns() - c0) / 1e6 / args.runs
    full = []
    for i in range(min(args.runs, 30)):
        img = sample
        if imgs:
            with Image.open(imgs[i % len(imgs)]) as im:
                img = ImageOps.exif_transpose(im).convert("RGB")
        out = det.detect(img)
        full.append(out.total_ms)
    res = {
        "kind": "detector_benchmark", "profile": prof.name, "file": prof.file.name,
        "file_size_bytes": prof.file.stat().st_size, "runtime": prof.runtime, "backend": det.runner.backend,
        "imgsz": prof.imgsz, "threads_requested": args.threads, "load_ms": load_ms,
        "model_only": latency_stats(fwd), "detect_end_to_end": latency_stats(full),
        "cpu_ms_per_inference": cpu_ms, "cpu_utilisation_percent_of_one_core": 100.0 * cpu_ms * args.runs / 1e3 / wall,
        "throughput_inferences_per_second": args.runs / wall,
        "memory_mb": {"start": mem0, "end": read_proc_status()},
        "temperature_c": {"before": temp0, "after": cpu_temperature_c()},
        "runs": args.runs, "warmup": args.warmup, "input": str(imgs[0].name) if imgs else "random 640x480 image",
    }
    if args.pipeline and imgs:
        res["pipeline"] = pipeline_bench(det, imgs, args)
    return res


def pipeline_bench(det, imgs, args) -> dict:
    from PIL import Image, ImageOps

    from zyqora.config import load_app_config
    from zyqora.greenscore_engine import GreenScoreEngine
    from zyqora.mixed import MixedWastePipeline
    from zyqora.pipeline import ZyqoraPipeline
    from zyqora.profiles import load_models_config, load_with_fallback
    from zyqora.wasterules import RecommendationEngine, load_label_profiles, load_material_table

    app = load_app_config()
    lm = load_with_fallback(load_models_config(app), allow_fallback=False, num_threads=args.threads or app.num_threads)
    classes = list(lm.category_set.category_names)
    cfg = load_detector_config(args.config) if args.config else load_detector_config()
    mp = MixedWastePipeline(det, ZyqoraPipeline(lm.classifier, lm.decoder, lm.guard, lm.spec),
                            load_material_table(cfg.materials, classes, det.profile.classes),
                            load_label_profiles(model_classes=classes), RecommendationEngine.load(),
                            GreenScoreEngine.load(), tuple(classes))
    rows, c0 = [], cpu_time_ns()
    w0 = time.perf_counter()
    for p in imgs:
        with Image.open(p) as im:
            img = ImageOps.exif_transpose(im).convert("RGB")
        r = mp.analyze(img, p.name)
        t = r["timing"]
        rows.append({"pipeline_ms": t["pipeline_ms"], "detector_ms": t["detector_inference_ms"],
                     "classification_ms": t["classification_ms"], "crops": t["classified_crops"],
                     "objects": r["summary"]["detected_objects"], "waste_items": r["summary"]["waste_items"]})
    wall = time.perf_counter() - w0
    return {"kind": "pipeline_benchmark", "images": len(rows), "classifier_profile": lm.profile.name,
            "pipeline_latency": latency_stats([r["pipeline_ms"] for r in rows]),
            "classification_per_crop_ms": (sum(r["classification_ms"] for r in rows) / max(1, sum(r["crops"] for r in rows))),
            "mean_crops_per_image": sum(r["crops"] for r in rows) / len(rows),
            "mean_objects_per_image": sum(r["objects"] for r in rows) / len(rows),
            "images_per_second": len(rows) / wall, "cpu_ms_per_image": (cpu_time_ns() - c0) / 1e6 / len(rows),
            "memory_mb_end": read_proc_status(), "per_image": rows}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profiles", nargs="*", default=None)
    ap.add_argument("--threads", nargs="*", type=int, default=[2])
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--images", type=Path, default=None, help="real images for detect()/pipeline timing")
    ap.add_argument("--max-images", type=int, default=50)
    ap.add_argument("--pipeline", action="store_true", help="also time the full YOLO + MobileNetV2 pipeline")
    ap.add_argument("--config", default=None)
    ap.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "reports" / "benchmarks")
    ap.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--profile", help=argparse.SUPPRESS)
    ap.add_argument("--threads-one", type=int, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args._worker:
        args.threads = args.threads_one
        print(json.dumps(worker(args)))
        return 0

    cfg = load_detector_config(args.config) if args.config else load_detector_config()
    names = args.profiles or [n for n, p in cfg.profiles.items() if p.file.is_file()]
    import importlib.util
    have_ort = importlib.util.find_spec("onnxruntime") is not None
    skipped = [n for n in names if cfg.get(n).runtime == "onnxruntime" and not have_ort]
    if skipped:
        print(f"skipped {', '.join(skipped)}: onnxruntime is not installed (optional; pip install onnxruntime to compare)")
        names = [n for n in names if n not in skipped]
    facts = platform_facts()
    device = "pi5" if facts["is_raspberry_pi_5"] else platform.machine().lower()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        results = []
        for th in args.threads:
            cmd = [sys.executable, __file__, "--_worker", "--profile", name, "--threads-one", str(th),
                   "--runs", str(args.runs), "--warmup", str(args.warmup), "--max-images", str(args.max_images)]
            if args.images:
                cmd += ["--images", str(args.images)]
            if args.pipeline:
                cmd.append("--pipeline")
            if args.config:
                cmd += ["--config", str(args.config)]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            line = next((ln for ln in reversed(proc.stdout.splitlines()) if ln.startswith("{")), None)
            if proc.returncode != 0 or not line:
                last = [ln for ln in proc.stderr.strip().splitlines() if ln.strip()][-1:] or ["(no error output)"]
                print(f"{name} threads {th}: FAILED: {last[0]}", file=sys.stderr)
                continue
            r = json.loads(line)
            results.append(r)
            mo = r["model_only"]
            pl = r.get("pipeline", {}).get("pipeline_latency", {})
            print(f"{name:18s} threads {th}: forward median {mo['median_ms']:7.2f} ms  p95 {mo['p95_ms']:7.2f}  "
                  f"cpu {r['cpu_ms_per_inference']:7.1f} ms  peak RSS {r['memory_mb']['end']['vm_hwm_mb'] or float('nan'):6.0f} MB"
                  + (f"  pipeline median {pl['median_ms']:.1f} ms" if pl else ""))
        if not results:
            continue
        out = args.out_dir / f"{device}_detector_{name}.json"
        out.write_text(json.dumps({"kind": "detector_benchmark", "profile": name, "platform": facts,
                                   "measured_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                   "warning": None if facts["is_raspberry_pi_5"] else
                                   "Measured on a development machine, NOT on the Raspberry Pi 5.",
                                   "results": results}, indent=2))
        print(f"wrote {out.relative_to(PROJECT_ROOT) if out.is_relative_to(PROJECT_ROOT) else out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
