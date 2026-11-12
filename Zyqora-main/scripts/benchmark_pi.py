#!/usr/bin/env python3
"""Benchmark a Zyqora .pte on the current device (designed for Raspberry Pi 5).

Measures, separately:
  * model-only latency: forward() on one preprocessed tensor, after warm-up
  * end-to-end latency: preprocess + forward + softmax/top-k + MaterialWise + EcoGuard per image
  * CPU time per inference (process user+sys, all threads)
  * VmRSS (current) and VmHWM (peak) resident memory of the process
  * model file size, optional CPU temperature

Thread sweep: each thread count runs in a fresh subprocess (clean VmHWM).

    python scripts/benchmark_pi.py --model models/mobilenetv2_fp32_xnnpack.pte
    python scripts/benchmark_pi.py --model models/mobilenetv2_int8_xnnpack.pte --threads 1 2 4 --images data/dev

Timer: time.perf_counter_ns(). Results are only Pi results when run on the Pi;
the JSON records the device model so host runs cannot be mistaken for Pi runs.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401

from zyqora.benchmark import cpu_temperature_c, cpu_time_ns, platform_facts, read_proc_status
from zyqora.config import load_app_config
from zyqora.evaluator import discover_images, latency_stats
from zyqora.imagenet_labels import load_labels
from zyqora.materialwise import load_mapping
from zyqora.model_runtime import ExecuTorchClassifier, import_portable_lib, softmax, topk
from zyqora.paths import OUTPUTS_DIR
from zyqora.pipeline import ZyqoraPipeline
from zyqora.preprocessing import PreprocessSpec, preprocess_path


def run_single(args) -> dict:
    app = load_app_config()
    spec = PreprocessSpec.from_file(app.contract)
    labels = load_labels(app.labels)
    mem_start = read_proc_status()
    import_portable_lib()  # imports executorch + torch so the next delta is the model only
    import torch  # noqa: F401
    mem_before = read_proc_status()
    t0 = time.perf_counter_ns()
    clf = ExecuTorchClassifier(args.model, input_shape=spec.input_shape, num_threads=args.num_threads,
                               require_backend=app.require_backend)
    load_ms = (time.perf_counter_ns() - t0) / 1e6
    mem_loaded = read_proc_status()

    images = discover_images(args.images)[: args.max_images] if args.images else []
    if images:
        x = preprocess_path(images[0][1], spec)
        bench_input = images[0][0]
    else:
        x = np.random.default_rng(0).standard_normal(spec.input_shape).astype(np.float32)
        bench_input = "random normal tensor (seed 0)"

    for _ in range(args.warmups):
        clf.infer(x)
    lat, cpu0 = [], cpu_time_ns()
    for _ in range(args.runs):
        lat.append(clf.infer(x).latency_ns / 1e6)
    cpu_ns = cpu_time_ns() - cpu0
    top = topk(softmax(clf.infer(x).logits), labels, 3)

    e2e = None
    if images:
        mapping = load_mapping(app.mapping, labels)
        pipe = ZyqoraPipeline(clf, labels, mapping, app.ecoguard, spec)
        e2e_ms, pre_ms, cpu1 = [], [], cpu_time_ns()
        for rel, path in images:
            r = pipe.process(path, rel)
            if r.end_to_end_time_ms is not None:
                e2e_ms.append(r.end_to_end_time_ms)
                pre_ms.append(r.preprocess_time_ms)
        e2e = {"images": len(e2e_ms), "end_to_end": latency_stats(e2e_ms), "preprocess": latency_stats(pre_ms),
               "cpu_ms_per_image": (cpu_time_ns() - cpu1) / 1e6 / max(1, len(e2e_ms))}
    mem_end = read_proc_status()
    return {
        "model": str(args.model),
        "model_size_mb": Path(args.model).stat().st_size / 1e6,
        "num_threads_requested": args.num_threads,
        "thread_count": clf.thread_count,
        "thread_control": clf.thread_control,
        "load_time_ms": load_ms,
        "warmups": args.warmups,
        "runs": args.runs,
        "benchmark_input": bench_input,
        "model_only": latency_stats(lat),
        "cpu_ms_per_inference": cpu_ns / 1e6 / args.runs,
        "end_to_end": e2e,
        "memory_mb": {"process_start": mem_start, "after_runtime_import": mem_before, "after_model_load": mem_loaded,
                      "end": mem_end,
                      "model_and_inference_peak_delta_mb": (mem_end["vm_hwm_mb"] - mem_before["vm_rss_mb"])
                      if mem_end["vm_hwm_mb"] is not None and mem_before["vm_rss_mb"] is not None else None,
                      "runtime_import_rss_delta_mb": (mem_before["vm_rss_mb"] - mem_start["vm_rss_mb"])
                      if mem_before["vm_rss_mb"] is not None and mem_start["vm_rss_mb"] is not None else None,
                      "model_load_rss_delta_mb": (mem_loaded["vm_rss_mb"] - mem_before["vm_rss_mb"])
                      if mem_loaded["vm_rss_mb"] is not None and mem_before["vm_rss_mb"] is not None else None},
        "top3_on_benchmark_input": [f"{p.label}:{p.probability:.4f}" for p in top],
        "cpu_temperature_c_end": cpu_temperature_c(),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, default=None, help=".pte (default: config/app.yaml)")
    ap.add_argument("--images", type=Path, default=None, help="optional image folder for end-to-end timing")
    ap.add_argument("--max-images", type=int, default=20)
    ap.add_argument("--warmups", type=int, default=10)
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--threads", type=int, nargs="*", default=None, help="thread counts to sweep, e.g. 1 2 4")
    ap.add_argument("--num-threads", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--single", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    args.model = args.model or load_app_config().pte

    if args.single:
        print(json.dumps(run_single(args)))
        return 0

    configs = args.threads if args.threads else [None]
    results = []
    for n in configs:
        cmd = [sys.executable, str(Path(__file__).resolve()), "--single", "--model", str(args.model),
               "--warmups", str(args.warmups), "--runs", str(args.runs), "--max-images", str(args.max_images)]
        if args.images:
            cmd += ["--images", str(args.images)]
        if n is not None:
            cmd += ["--num-threads", str(n)]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            print(proc.stderr, file=sys.stderr)
            print(f"Benchmark failed for threads={n}", file=sys.stderr)
            return 3
        results.append(json.loads(proc.stdout.strip().splitlines()[-1]))

    report = {"platform": platform_facts(), "timer": "time.perf_counter_ns", "results": results,
              "note": "Process memory includes the Python interpreter, torch and numpy; it is not model-only memory."}
    if not report["platform"]["is_raspberry_pi_5"]:
        report["warning"] = "NOT measured on a Raspberry Pi 5; do not report as Pi performance."
    out = args.out or OUTPUTS_DIR / "benchmark" / f"benchmark_{Path(args.model).stem}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"Device: {report['platform']['device_model'] or report['platform']['machine']}  "
          f"| model {Path(args.model).name} ({results[0]['model_size_mb']:.2f} MB)")
    if "warning" in report:
        print("WARNING:", report["warning"])
    print(f"{'threads':>8} {'median':>8} {'p95':>8} {'mean':>8} {'cpu/inf':>8} {'e2e med':>8} {'RSS':>6} {'peak':>6}")
    for r in results:
        m, e = r["model_only"], r["end_to_end"]
        print(f"{str(r['thread_count']):>8} {m['median_ms']:8.2f} {m['p95_ms']:8.2f} {m['mean_ms']:8.2f} "
              f"{r['cpu_ms_per_inference']:8.2f} {(e['end_to_end']['median_ms'] if e else float('nan')):8.2f} "
              f"{r['memory_mb']['end']['vm_rss_mb'] or float('nan'):6.0f} {r['memory_mb']['end']['vm_hwm_mb'] or float('nan'):6.0f}")
    print("(ms; cpu/inf = process CPU time per model-only inference; RSS/peak in MB)")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
