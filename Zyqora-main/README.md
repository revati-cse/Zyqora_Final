# Zyqora AI

**Offline, confidence-aware waste sorting assistant with bilingual guidance and verified sorting rewards.**
NAMMA AI hackathon, PS3 (AI for Sustainability: Edge-Based Waste Sorting). Target: Raspberry Pi 5, ExecuTorch, XNNPACK, fully offline.

> **Status in short** (details in [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md)):
> - The team's fine-tuned **Organic / Recyclable MobileNetV2** is integrated as the `waste_mobilenetv2` profile. It was tested end-to-end with a stand-in model of the same architecture.
> - The production switch to the waste model happens on the Pi via `scripts/verify_waste_model.py --activate`.
> - Pi benchmarks, TEST-set evaluation, parity with the real checkpoint, and all physical hardware tests are **pending**. No such numbers are claimed here.
> - The waste model **cannot recognise Residual waste** (see [MODEL_EVALUATION.md](MODEL_EVALUATION.md)).

## What it does

```
camera / image file
  └─► preprocess (Pillow + NumPy, bit-identical to TorchVision)
        └─► MobileNetV2 .pte on ExecuTorch + XNNPACK (loaded once, offline)
              └─► decoder for THIS model (2-class waste  |  1000-class ImageNet + MaterialWise)
                    └─► EcoGuard: ACCEPT (bin) or REVIEW (supervisor, no bin opened)
                          ├─► explanation card + VoiceSort (Tamil / English, offline clips)
                          ├─► sorter (simulation by default; GPIO only when explicitly enabled)
                          └─► GreenScore PENDING event ─► supervisor verification ─► points
```

| Component | Implemented as |
|---|---|
| **Model profiles** (`config/models.yaml`, `src/zyqora/profiles.py`) | Each `.pte` is bound to its own preprocessing contract and decoder. A model whose output shape doesn't match its profile is refused at load, so the 2-class decoder can never run on the ImageNet model (or vice versa). There is an optional SHA-256 pin, and fallback to the ImageNet baseline is explicit and logged. |
| **Waste decoder** (`src/zyqora/decoders.py`) | logits `[1, 2]` → softmax once → index 0 Organic → COMPOST, index 1 Recyclable → RECYCLE |
| **MaterialWise** (`src/zyqora/materialwise.py`) | ImageNet baseline only: editable rules from ImageNet classes to categories |
| **EcoGuard** (`src/zyqora/ecoguard.py`) | ACCEPT only if the category probability ≥ threshold (waste model: 0.95 provisional); otherwise REVIEW with a reason. REVIEW items keep their predicted category for evaluation but are **never sorted**. |
| **Explainability** (`src/zyqora/explain.py`) | Per-item card: class probability bars, the threshold, the EcoGuard decision in plain words, timings |
| **VoiceSort** (`src/zyqora/voicesort.py`) | Pre-generated Tamil and English WAV clips via `aplay`. If playback fails, the text is printed and the station keeps going. |
| **GreenScore** (`src/zyqora/greenscore.py`) | SQLite. Points are awarded only after a supervisor verifies the real bin; each event once (atomic); parameterised SQL. |
| **Camera** (`src/zyqora/camera.py`) | Picamera2 / `rpicam-still` / folder simulation; single shot per key press |
| **Sorter** (`src/zyqora/sorter.py`) | Simulation (default) or gpiozero pulses, which need config **and** `--enable-actuator` |
| **Evaluation** (`src/zyqora/evaluator.py`) | Accuracy, per-class precision/recall/F1, confusion matrix, accepted-only metrics, review rate, unsupported ground truth |

The contribution is the integration: offline edge inference + confidence-aware review + bilingual guidance + verify-to-earn rewards on a Pi. None of the individual techniques is claimed to be new.

## Models
| Profile | Model | Contract | Status |
|---|---|---|---|
| `waste_mobilenetv2` | MobileNetV2 fine-tuned on Organic vs Recyclable, FP32 `.pte` (about 8.47 MiB) | exact 224×224 resize, ImageNet mean/std → `[1, 2]` | Integrated. Validation accuracy 96.65% (team-reported). TEST pending. |
| `imagenet_mobilenetv2` | TorchVision MobileNetV2 IMAGENET1K_V1, FP32 `.pte` (13.99 MB) | resize 256 → crop 224 → `[1, 1000]` | Baseline / fallback (current `active` until the waste gate passes) |

Model details and the evaluation plan: [MODEL_EVALUATION.md](MODEL_EVALUATION.md). INT8 experiments with the ImageNet model: [MODEL_COMPARISON.md](MODEL_COMPARISON.md).

## Quick start
Tests run anywhere (Python ≥ 3.10; torch not required):
```bash
pip install -r requirements-dev.txt
```
```bash
python -m unittest discover -s tests -t . -v
```
Raspberry Pi deployment: [README_PI.md](README_PI.md). Configuration: [CONFIGURATION.md](CONFIGURATION.md). Camera, actuator and audio: [HARDWARE.md](HARDWARE.md). Demo: [DEMO.md](DEMO.md).

## Commands
| Task | Command |
|---|---|
| Verify + activate the waste model (Pi) | `python scripts/verify_waste_model.py --images <VAL sample with O/ R/> --expected-sha256 <hash> --activate` |
| Station (simulation) | `python zyqora_station.py --camera folder:data/demo --voice ta --user DEMO01` |
| Station (Pi camera) | `python zyqora_station.py --camera auto --voice en --user DEMO01` |
| TEST evaluation | `python run_zyqora.py --profile waste_mobilenetv2 --images <TEST dir> --labels-from-folders --dataset-role test --quiet --out outputs/test_waste` |
| Organiser evaluation | `python run_zyqora.py --images data/organiser --labels data/organiser/labels.csv --dataset-role organiser --out outputs/final` |
| Leakage check | `python scripts/check_leakage.py --split train=<dir> --split val=<dir> --split test=<dir>` |
| Parity (waste model) | `python scripts/check_parity.py --images <dir> --contract config/model_waste.json --checkpoint <.pth> --pte models/mobilenetv2_waste_fp32_xnnpack.pte` |
| Re-export the waste model | `python scripts/export_fp32.py --checkpoint <.pth> --contract config/model_waste.json --out models/mobilenetv2_waste_fp32_xnnpack.pte` |
| Benchmark | `python scripts/benchmark_pi.py --profile waste_mobilenetv2 --threads 1 2 3 4 --images <dir>` |
| Threshold sweep (dev/val only) | `python scripts/sweep_threshold.py --results outputs/<dev run>/results.csv` |
| Config / offline checks | `python scripts/check_config.py` · `python scripts/check_offline.py` |
| Voice | `PYTHONPATH=src python -m zyqora.voicesort --lang ta --say RECYCLE` |
| GreenScore | `PYTHONPATH=src python -m zyqora.greenscore verify --event-id 3` · `... points --user-id DEMO01` |

`run_zyqora.py` exit codes: 0 ok, 2 config, 3 model/runtime, 4 data. `zyqora_station.py` also uses 5 for a camera error and 6 for an actuator error.

## Outputs
- `outputs/<run>/results.csv`: one row per image. Columns: `image_filename, ground_truth_category, raw_top1_class, raw_top1_probability, predicted_category, predicted_category_score, mapping_status, ecoguard_status, review_reason, disposal_action, inference_time_ms, correct`, plus supporting columns.
- `outputs/<run>/summary.json`:
  - accuracy, review rate, accepted-only accuracy, incorrect accepted
  - `metrics` (confusion matrix, per-class P/R/F1) and `accepted_only_metrics`
  - unsupported ground truth, latency, throughput, CPU per image, memory
  - model profile, SHA-256 and fallback flag, EcoGuard settings, platform, `evaluation_scope`
- `outputs/station/<run>/`: `events.jsonl`, `summary.json` (camera-to-result latency), `station.log`

## Metric definitions
- **Overall accuracy** = correct predictions / **all** evaluated images. REVIEW, failed, and unsupported-truth (e.g. Residual) images all stay in the denominator.
- **Accepted-only accuracy** = correct ACCEPTed / ACCEPTed (reported as "undefined" if none were accepted). **Review rate** = REVIEW / all.
- Precision, recall and F1 are per ground-truth class over all images. The confusion matrix has a `<none>` column for failed images.
- Latency: model-only `forward()` time is reported separately from preprocessing and end-to-end time, after an untimed warm-up.

## Known limitations
- **Residual / unknown items** are forced into Organic or Recyclable, often with high confidence. EcoGuard cannot catch this. MODEL_EVALUATION.md lists the fixes.
- 0.95 is a validation-set operating point, not a guarantee. TEST results are pending.
- No Pi measurements and no physical camera, actuator or speaker tests yet.
- One item per image (no object detection).
- The ExecuTorch wheel depends on torch, so the Pi environment is several hundred MB.
