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
import dataclasses
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
from zyqora.model_runtime import import_portable_lib
from zyqora.paths import OUTPUTS_DIR
from zyqora.pipeline import ZyqoraPipeline
from zyqora.preprocessing import preprocess_path
from zyqora.profiles import load_models_config, load_profile


def run_single(args) -> dict:
    app = load_app_config()
    models = load_models_config(app)
    profile = models.get(args.profile or models.active)
    if args.model:
        profile = dataclasses.replace(profile, pte=Path(args.model).resolve(), sha256=None)
    mem_start = read_proc_status()
    import_portable_lib()  # imports executorch + torch so the next delta is the model only
    import torch  # noqa: F401
    mem_before = read_proc_status()
    t0 = time.perf_counter_ns()
    loaded = load_profile(profile, num_threads=args.num_threads, require_backend=app.require_backend)
    load_ms = (time.perf_counter_ns() - t0) / 1e6
    mem_loaded = read_proc_status()
    clf, spec = loaded.classifier, loaded.spec

    images = discover_images(args.images)[: args.max_images] if args.images else []
    if images:
        x = preprocess_path(images[0][1], spec)
        bench_input = images[0][0]
    else:
        x = np.random.default_rng(0).standard_normal(spec.input_shape).astype(np.float32)
        bench_input = "random normal tensor (seed 0)"

    for _ in range(args.warmups):
        clf.infer(x)
    lat, cpu0, wall0 = [], cpu_time_ns(), time.perf_counter_ns()
    for _ in range(args.runs):
        lat.append(clf.infer(x).latency_ns / 1e6)
    cpu_ns, wall_ns = cpu_time_ns() - cpu0, time.perf_counter_ns() - wall0
    _, top, _ = loaded.decoder.decode(clf.infer(x).logits)

    e2e = None
    if images:
        pipe = ZyqoraPipeline(clf, loaded.decoder, loaded.guard, spec)
        e2e_ms, pre_ms, cpu1, wall1 = [], [], cpu_time_ns(), time.perf_counter_ns()
        for rel, path in images:
            r = pipe.process(path, rel)
            if r.end_to_end_time_ms is not None:
                e2e_ms.append(r.end_to_end_time_ms)
                pre_ms.append(r.preprocess_time_ms)
        e2e_wall = time.perf_counter_ns() - wall1
        e2e_cpu = cpu_time_ns() - cpu1
        e2e = {"images": len(e2e_ms), "end_to_end": latency_stats(e2e_ms), "preprocess": latency_stats(pre_ms),
               "cpu_ms_per_image": e2e_cpu / 1e6 / max(1, len(e2e_ms)),
               "throughput_images_per_second": len(e2e_ms) / (e2e_wall / 1e9) if e2e_wall else None,
               "cpu_utilisation_percent_of_one_core": 100.0 * e2e_cpu / e2e_wall if e2e_wall else None}
    mem_end = read_proc_status()
    return {
        "profile": profile.name,
        "decoder": profile.decoder,
        "model": str(profile.pte),
        "model_size_mb": Path(profile.pte).stat().st_size / 1e6,
        "output_shape": list(clf.output_shape) if clf.output_shape else None,
        "num_threads_requested": args.num_threads,
        "thread_count": clf.thread_count,
        "thread_control": clf.thread_control,
        "load_time_ms": load_ms,
        "warmups": args.warmups,
        "runs": args.runs,
        "benchmark_input": bench_input,
        "model_only": latency_stats(lat),
        "model_only_throughput_inferences_per_second": args.runs / (wall_ns / 1e9) if wall_ns else None,
        "cpu_ms_per_inference": cpu_ns / 1e6 / args.runs,
        "cpu_utilisation_percent_of_one_core": 100.0 * cpu_ns / wall_ns if wall_ns else None,
        "end_to_end": e2e,
        "memory_mb": {"process_start": mem_start, "after_runtime_import": mem_before, "after_model_load": mem_loaded,
                      "end": mem_end,
                      "model_and_inference_peak_delta_mb": (mem_end["vm_hwm_mb"] - mem_before["vm_rss_mb"])
                      if mem_end["vm_hwm_mb"] is not None and mem_before["vm_rss_mb"] is not None else None,
                      "runtime_import_rss_delta_mb": (mem_before["vm_rss_mb"] - mem_start["vm_rss_mb"])
                      if mem_before["vm_rss_mb"] is not None and mem_start["vm_rss_mb"] is not None else None,
                      "model_load_rss_delta_mb": (mem_loaded["vm_rss_mb"] - mem_before["vm_rss_mb"])
                      if mem_loaded["vm_rss_mb"] is not None and mem_before["vm_rss_mb"] is not None else None},
        "top_on_benchmark_input": [f"{p.label}:{p.probability:.4f}" for p in top[:3]],
        "cpu_temperature_c_end": cpu_temperature_c(),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default=None, help="model profile in config/models.yaml (default: active)")
    ap.add_argument("--model", type=Path, default=None, help="override the profile's .pte")
    ap.add_argument("--images", type=Path, default=None, help="optional image folder for end-to-end timing")
    ap.add_argument("--max-images", type=int, default=20)
    ap.add_argument("--warmups", type=int, default=10)
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--threads", type=int, nargs="*", default=None, help="thread counts to sweep, e.g. 1 2 4")
    ap.add_argument("--num-threads", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--single", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    app = load_app_config()
    models = load_models_config(app)
    args.profile = args.profile or models.active
    pte = Path(args.model) if args.model else models.get(args.profile).pte

    if args.single:
        print(json.dumps(run_single(args)))
        return 0

    configs = args.threads if args.threads else [None]
    results = []
    for n in configs:
        cmd = [sys.executable, str(Path(__file__).resolve()), "--single", "--profile", args.profile,
               "--warmups", str(args.warmups), "--runs", str(args.runs), "--max-images", str(args.max_images)]
        if args.model:
            cmd += ["--model", str(args.model)]
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
    out = args.out or OUTPUTS_DIR / "benchmark" / f"benchmark_{pte.stem}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"Device: {report['platform']['device_model'] or report['platform']['machine']}  "
          f"| profile {args.profile} | model {pte.name} ({results[0]['model_size_mb']:.2f} MB, "
          f"output {results[0]['output_shape']})")
    if "warning" in report:
        print("WARNING:", report["warning"])
    nan = float("nan")
    print(f"{'threads':>7} {'median':>7} {'p95':>7} {'inf/s':>7} {'cpu/inf':>7} {'cpu%':>6} {'e2e med':>7} "
          f"{'e2e p95':>7} {'img/s':>6} {'RSS':>5} {'peak':>5}")
    for r in results:
        m, e = r["model_only"], r["end_to_end"]
        print(f"{str(r['thread_count']):>7} {m['median_ms']:7.2f} {m['p95_ms']:7.2f} "
              f"{r['model_only_throughput_inferences_per_second']:7.1f} {r['cpu_ms_per_inference']:7.2f} "
              f"{r['cpu_utilisation_percent_of_one_core']:6.0f} "
              f"{(e['end_to_end']['median_ms'] if e else nan):7.2f} {(e['end_to_end']['p95_ms'] if e else nan):7.2f} "
              f"{(e['throughput_images_per_second'] if e else nan):6.1f} "
              f"{r['memory_mb']['end']['vm_rss_mb'] or nan:5.0f} {r['memory_mb']['end']['vm_hwm_mb'] or nan:5.0f}")
    print("(ms; inf/s = model-only inferences per second; cpu/inf = process CPU ms per model-only inference;\n"
          " cpu% = CPU time / wall time, 100% = one core busy; e2e = decode+preprocess+inference+decision; MB)")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
