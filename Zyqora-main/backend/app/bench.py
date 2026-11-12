import argparse, json, os, statistics, time
from pathlib import Path
import psutil
from .core.engine import get_engine
from .core.pipeline import classify_image

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", default="results/mac_benchmark.json")
    args = parser.parse_args()
    paths = sorted(p for p in Path(args.images).rglob("*") if p.suffix.lower() in EXTS)
    if not paths:
        raise SystemExit("No images found")
    engine = get_engine()
    for i in range(args.warmups):
        classify_image(str(paths[i % len(paths)]), engine)
    proc = psutil.Process(os.getpid())
    cpu0 = proc.cpu_times()
    wall0 = time.perf_counter()
    latencies = []
    for _ in range(args.repeats):
        for path in paths:
            latencies.append(classify_image(str(path), engine).latency_ms)
    wall = time.perf_counter() - wall0
    cpu1 = proc.cpu_times()
    cpu_s = (cpu1.user + cpu1.system) - (cpu0.user + cpu0.system)
    ordered = sorted(latencies)
    n = len(ordered)
    record = {
        "mode": "LIVE", "device_scope": "DEVELOPMENT_HOST_NOT_PI",
        "device": os.uname().machine, "images_per_repeat": len(paths),
        "repeats": args.repeats, "warmups": args.warmups, "total_inferences": n,
        "wall_seconds": wall, "cpu_seconds": cpu_s,
        "cpu_seconds_per_image": cpu_s/n if n else None,
        "mean_latency_ms": statistics.mean(ordered),
        "p50_latency_ms": statistics.median(ordered),
        "p95_latency_ms": ordered[min(n-1, int(0.95*n))],
        "images_per_second": n/wall if wall else None,
        "process_peak_rss_mb": proc.memory_info().rss/(1024*1024),
        "note": "Development-host measurement only; not Raspberry Pi performance."
    }
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))

if __name__ == "__main__":
    run()
