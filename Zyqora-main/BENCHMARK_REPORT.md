# CPU / memory benchmark report

## Status
**No Raspberry Pi 5 figures exist in this repository yet.** Everything below was measured on the x86_64 development host (shared 4-vCPU Xeon VM) to validate the tooling. Host timings varied by up to about 2× between runs. Do not quote them as Pi performance.

## How to measure on the Pi (reproducible)
```bash
python scripts/benchmark_pi.py --profile waste_mobilenetv2 --threads 1 2 3 4 --images <50+ representative images> --runs 200 --warmups 20
```
```bash
python scripts/benchmark_pi.py --profile imagenet_mobilenetv2 --threads 1 2 3 4 --images <same images> --runs 200 --warmups 20
```
```bash
python zyqora_station.py --profile waste_mobilenetv2 --camera auto --auto 30 --voice off --out outputs/station/latency_run
```
Before measuring: Raspberry Pi OS Lite, nothing else running, active cooling, and `vcgencmd get_throttled` = `0x0` before and after the run. Each thread count runs in a fresh subprocess, so peak memory is not shared between configurations. The JSON records `device_model`, so host runs can't be mistaken for Pi runs.

What each column means:
| Metric | Definition |
|---|---|
| median / p95 | model-only `forward()` latency, `time.perf_counter_ns`, after warm-up |
| inf/s | model-only inferences per second (runs ÷ wall time) |
| cpu/inf | process CPU time (user + sys, all threads) per inference: the **efficiency** figure |
| cpu% | CPU time ÷ wall time; 100% = one core fully busy (the Pi 5 has 4 cores, so 400% maximum) |
| e2e | decode + preprocess + inference + decision per image file |
| img/s | end-to-end images per second |
| RSS / peak | `VmRSS` / `VmHWM` of the whole Python process, including torch + ExecuTorch runtime (not model-only) |
| camera-to-result | station: capture start → EcoGuard decision ready (excludes voice and actuation) |

Choosing the thread count: lower latency and lower CPU per image pull in opposite directions (see host data). For a one-item-at-a-time station, pick the **lowest CPU time per image whose p95 latency is still far below human reaction time**. Set it in `config/app.yaml → model.num_threads`.

## Development-host results (NOT Pi)
Stand-in 2-class model (same architecture as the waste model, 8.88 MB), 100 runs, 16 images:

| threads | median ms | p95 ms | inf/s | CPU ms/inf | cpu% | e2e median ms | img/s | RSS MB | peak MB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 5.15 | 5.67 | 190.3 | 5.23 | 99 | 18.31 | 32.5 | 700 | 712 |
| 2 | 3.69 | 4.96 | 255.3 | 7.15 | 183 | 18.81 | 30.3 | 700 | 712 |
| 4 | 2.00 | 2.78 | 453.9 | 7.54 | 342 | 15.92 | 32.4 | 700 | 712 |

Station simulation (folder camera, stand-in model): camera-to-result about 12–16 ms per item after the first one. The first item took about 200 ms because of a cold file read.

Observations that should carry over to the Pi but must be re-measured there:
- More threads cut latency but raise CPU time per inference (1 thread was the most CPU-efficient on the host).
- End-to-end time is dominated by JPEG decoding and resizing, not by the model. Capturing at 640×480 (the `config/hardware.yaml` default) instead of full resolution keeps that cost low.
- About 640 MB of RSS comes from importing torch + ExecuTorch on this host (CUDA-enabled PyPI torch wheel). The Pi's CPU-only aarch64 wheel will differ.

## Pi results
| Profile | threads | median ms | p95 ms | CPU ms/inf | e2e median ms | peak MB | camera-to-result median ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| waste_mobilenetv2 | *pending* | | | | | | |
| imagenet_mobilenetv2 | *pending* | | | | | | |
