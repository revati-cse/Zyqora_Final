# ZYQORA AI — CPU-Efficient Edge Waste Classification

**Smart. Sort. Sustain.**

Mac-first starter implementation for the NAMMA AI Hackathon. Scope: **single-image classification**. Raspberry Pi 4 deployment and all edge performance claims must be validated on the actual Pi.

## Current implementation
- MobileNetV2 ImageNet FP32 development baseline (torchvision).
- Preprocessing, label bridge, EcoGuard, SortSense and WastePulse modules.
- Label bridge intentionally ships `UNCONFIGURED`; ImageNet classes are not waste categories.
- CLI, FastAPI endpoint, SQLite history, benchmark harness and logic tests.
- ExecuTorch export is not yet implemented or verified. Do not assume a Mac artifact runs on Pi.

## Mac setup
Recommended Python 3.11.

```bash
cd zyqora
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-mac.txt
```

Torch/torchvision wheels vary by macOS and Python version. If installation fails, install a matching PyTorch + torchvision pair from the official PyTorch install selector, then install remaining requirements.

## Tests
```bash
PYTHONPATH=backend pytest -q backend/tests
```

## Classify images
```bash
PYTHONPATH=backend python -m app.cli --images ./data/sample
```
First run may download pretrained MobileNetV2 weights. With the label map unconfigured, results correctly return `REVIEW`.

## Start API
```bash
uvicorn app.api:app --app-dir backend --host 0.0.0.0 --port 8000
```
Then open `http://127.0.0.1:8000/docs`.

## Mac development benchmark
```bash
PYTHONPATH=backend python -m app.bench --images ./data/sample --repeats 5 --warmups 3
```
This is a development-host measurement, never a Pi result.

## Tomorrow: Raspberry Pi 4 fact collection
Run `bash scripts/pi_collect_facts.sh` on the Pi before installing an ML runtime. Confirm 64-bit OS (`aarch64`) and model string. Then test runtime compatibility. The current code uses PyTorch/torchvision as a development baseline; ExecuTorch/XNNPACK compatibility on the Pi is not yet verified.

## Measurement rules
- Same device and workload for comparisons.
- Record CPU-seconds/image alongside CPU %, latency, throughput, RAM and temperature.
- Do not infer power savings from CPU percentage.
- Do not tune thresholds on the final test set.
- Count REVIEW as incorrect for accuracy.
- Keep LIVE, REPLAY and DEMO distinct.
