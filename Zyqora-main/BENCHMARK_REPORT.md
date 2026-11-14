# CPU / memory benchmark report

## Status
**Raspberry Pi 5 figures for the waste model were measured on 2026-11-14** (section "Pi results"). Everything in the development-host sections was measured on an x86_64 VM to validate the tooling, and must not be quoted as Pi performance.

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

## Dashboard API latency (development host, NOT Pi)
`serve_zyqora.py` with the stand-in 2-class `.pte` on the x86_64 VM, 198 KB JPEG, 50 sequential `POST /api/inference` from a local client after 5 warm-ups:

| Stage | Median | p95 |
|---|---:|---:|
| Model inference (ExecuTorch `forward`) | 3.70 ms | 5.88 ms |
| On-device total (decode + resize + inference + EcoGuard) | 8.6 ms | 11.7 ms |
| HTTP round trip with thumbnail + SQLite history | 31.0 ms | 41.5 ms |
| HTTP round trip with `--no-thumbnails` | 24.9 ms | 27.8 ms |
| `GET /api/health` round trip (baseline) | 1.4 ms | n/a |

Server process RSS was about 720 MB (dominated by importing torch + ExecuTorch; the CUDA-enabled PyPI torch wheel on x86). Warm start-up to MODEL READY took about 6 s. Re-measure on the Pi:
```bash
python serve_zyqora.py --profile waste_mobilenetv2
```
and in a second terminal, the same loop (any HTTP client), plus `GET /api/device/metrics` for RSS.

## Pi results (measured on the Raspberry Pi 5)
Device: **Raspberry Pi 5 Model B Rev 1.1**, Python 3.12 venv, ExecuTorch 1.3.1+cpu with XNNPACK. Model: `mobilenetv2_waste_fp32_xnnpack.pte` (8.88 MB, SHA-256 `b019fc68…094f`).

**Thread sweep** (`scripts/benchmark_pi.py --profile waste_mobilenetv2 --threads 1 2 3 4 --images data/validation --max-images 50`, 10 warm-ups, 100 runs per setting, each in a fresh process):

| Threads | Model median | Model p95 | Inferences/s | CPU ms / inference | CPU % (100 = 1 core) | End-to-end median | End-to-end p95 | Images/s | RSS MB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 21.89 ms | 21.94 ms | 45.2 | **22.10** | 100 | 25.51 ms | 25.76 ms | 39.0 | 380 |
| **2 (configured)** | **12.73 ms** | **12.80 ms** | 77.3 | 25.32 | 196 | 16.13 ms | 16.37 ms | 61.5 | 380 |
| 3 | 9.95 ms | 10.01 ms | 98.6 | 29.18 | 288 | 13.39 ms | 13.61 ms | 74.1 | 380 |
| 4 | 9.78 ms | 9.86 ms | 100.2 | 37.91 | 380 | 13.33 ms | 13.55 ms | 74.3 | 380 |

Decision: **2 threads** (`config/app.yaml → model.num_threads: 2`). It nearly halves latency compared with 1 thread for about 15% more CPU per inference, and leaves two cores free for the web server, camera and dashboard. Going from 3 to 4 threads gains 0.17 ms for 30% more CPU, so 4 is not used. Latency is very stable (p95 within 0.1 ms of the median).

**Full validation run on the Pi** (`run_zyqora.py --profile waste_mobilenetv2 --images data/validation --labels data/validation/labels.csv --quiet`, 2,000 images, 2 threads):

| Metric | Value |
|---|---:|
| Model inference median / mean / p95 | 12.87 / 12.88 / 12.94 ms |
| Throughput (decode + preprocess + inference + decision) | 58.6 images/s |
| CPU per image (whole pipeline) | 28.8 ms |
| Process memory (RSS / peak) | 384 / 384 MB |

Still to measure on the Pi: dashboard HTTP latency and camera-to-result latency (`zyqora_station.py --camera auto --auto 30`). CPU temperature under load is optional.

## Zyqora AI 4.0: detector and mixed-waste pipeline

### Raspberry Pi 5 results (measured by the team, 2026-11-14)
`python scripts/benchmark_detector.py --threads 2 4 --images data/validation --max-images 50 --pipeline` on the Pi (Raspberry Pi 5 Model B Rev 1.1, ExecuTorch 1.3.1, XNNPACK); every setting ran in a fresh process. Pipeline = YOLO + MobileNetV2 on 50 validation photos. These are single-item photos, so about one crop each; a photo with N waste objects adds roughly one MobileNetV2 run per extra object (about 16 ms on the Pi with 2 threads: 12.7 ms inference plus preprocessing, from the single-item benchmark above).

| Profile | Threads | YOLO forward median | p95 | CPU ms / forward | Peak RSS (detector process) | Full pipeline median |
|---|---:|---:|---:|---:|---:|---:|
| **yolo11n_640** (default) | **2** (configured) | **196.7 ms** | 197.2 ms | 374.1 | 447 MB | **224.0 ms** |
| yolo11n_640 | 4 | 158.5 ms | 159.1 ms | 566.7 | 447 MB | 183.7 ms |
| yolo11n_320 | 2 | 43.1 ms | 43.2 ms | 81.6 | 392 MB | 61.8 ms |
| yolo11n_320 | 4 | 32.8 ms | 33.0 ms | 113.7 | 392 MB | 48.9 ms |

ONNX Runtime profiles were not measured on the Pi (`onnxruntime` is optional and not installed there).

What it means:
- **640 px on the Pi costs about 0.2 s per photo.** That is fine for the AI Detection page (one photo per click). For continuous Live Camera it means at most about 4 frames per second, so choose `yolo11n_320` in Settings for live demos. 320 px is 4.6× faster but less accurate on small objects (VOC bottle AP50 0.772 → 0.622).
- **4 threads instead of 2** saves 20% of YOLO latency at 640 px, but uses 52% more CPU time per image. The configured 2 threads (`config/app.yaml`, shared by both models) leaves two cores for the web server and camera.
- The x86 host is 3× faster than the Pi at 640 px (63.7 vs 196.7 ms, 2 threads), so host numbers must never be quoted as Pi performance.
- Load + warm-up time, detect() latency, throughput and temperature are in the JSON files (`reports/benchmarks/pi5_detector_*.json`) once they are committed from the Pi.

### How to measure on the Pi
```bash
python scripts/benchmark_detector.py --threads 2 4 --images data/validation --max-images 50 --pipeline
```
Each (profile, threads) runs in a fresh process. The script records load + warm-up, model-only forward() latency (median / p95), `detect()` latency (letterbox + forward + decode + NMS), CPU ms per forward, throughput, process RSS / peak, and temperature. With `--pipeline` it also records the full YOLO + MobileNetV2 analysis per image. Results go to `reports/benchmarks/pi5_detector_<profile>.json` and appear on the **Model Comparison** page, labelled "Raspberry Pi 5". **No wattage is estimated.** On the Pi, System Monitoring shows the PMIC rail readings (`vcgencmd pmic_read_adc`); these exclude USB devices and conversion losses, so they are not wall power. Measured power needs an inline USB-C meter.

### Development-host results (x86_64 VM, 4 vCPU Xeon 2.1 GHz, 2026-11-14), **NOT Pi**
From `reports/benchmarks/x86_64_detector_*.json`. Pipeline = 40 Open Images validation photos (mean 1.1 classified crops per image), with the waste classifier on ExecuTorch.

| Profile | Runtime | Thr | Load+warm-up | Forward median | p95 | detect() median | CPU ms/fwd | fwd/s | Peak RSS (detector only) | Pipeline median | MobileNetV2 per crop | Images/s | Peak RSS (pipeline) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| yolo11n_640 | ExecuTorch/XNNPACK | 2 | 130 ms | 63.7 ms | 78.0 | 67.8 | 121.2 | 15.7 | 779 MB | 78.3 ms | 8.6 ms | 10.1 | 802 MB |
| yolo11n_640 | ExecuTorch/XNNPACK | 4 | 114 ms | 39.7 ms | 44.1 | 49.3 | 112.5 | 25.1 | 779 MB | 54.9 ms | 6.5 ms | 15.6 | 802 MB |
| yolo11n_640_onnx | ONNX Runtime | 2 | 153 ms | 39.9 ms | 46.7 | 53.0 | 80.1 | 24.8 | 201 MB | 63.3 ms | 9.3 ms | 13.2 | 867 MB |
| yolo11n_640_onnx | ONNX Runtime | 4 | 168 ms | 29.3 ms | 38.1 | 45.6 | 120.5 | 32.9 | 201 MB | 46.8 ms | 9.7 ms | 17.7 | 867 MB |
| yolo11n_320 | ExecuTorch/XNNPACK | 2 | 52 ms | 14.5 ms | 19.9 | 18.4 | 37.3 | 64.0 | 728 MB | 29.1 ms | 8.6 ms | 25.9 | 754 MB |
| yolo11n_320 | ExecuTorch/XNNPACK | 4 | 50 ms | 12.3 ms | 16.4 | 14.6 | 40.7 | 78.0 | 728 MB | 21.8 ms | 6.5 ms | 32.4 | 754 MB |
| yolo11n_320_onnx | ONNX Runtime | 2 | 137 ms | 10.4 ms | 14.8 | 14.3 | 21.9 | 88.7 | 131 MB | 23.1 ms | 7.7 ms | 31.6 | 801 MB |
| yolo11n_320_onnx | ONNX Runtime | 4 | 103 ms | 7.1 ms | 7.6 | 11.5 | 28.1 | 141.8 | 131 MB | 21.8 ms | 8.5 ms | 32.0 | 802 MB |

What this does and does not show:
- **On this x86 host, ONNX Runtime is faster than ExecuTorch/XNNPACK for YOLO.** XNNPACK is tuned for Arm, so the ranking on the Pi may differ. Measure on the Pi before switching the default (`detector_profile` in Settings).
- The ExecuTorch detector process is about 730–780 MB because the ExecuTorch Python bindings import torch. The ONNX detector alone is 131–201 MB, but the full pipeline still loads torch for the classifier.
- Per-image pipeline latency grows by roughly one MobileNetV2 run (about 7–10 ms here) per waste object.
