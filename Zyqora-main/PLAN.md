# Zyqora AI: technical plan

> **This is a reconstruction.** The original `PLAN.md` was not in the repository when implementation started
> (2026-11-12). This file was rebuilt from the facts restated in the implementation brief and from what was
> verified while building the project. It contains no new claims. If the original `PLAN.md` exists, replace this file
> with it and re-check every item marked *brief* against the code.
>
> Source tags: **verified** = checked in this repository (see the linked evidence); *brief* = taken from the
> implementation brief and not re-verified here; *pending* = needs the Pi or organiser data.

## 1. Goal
An offline, confidence-aware waste sorting assistant with bilingual guidance and verified sorting rewards, for NAMMA AI
PS3 (AI for Sustainability: Edge-Based Waste Sorting and Food Categorisation). It runs a verified pretrained model
through ExecuTorch with XNNPACK on a Raspberry Pi 5, fully offline, and supports a demonstration of at most two minutes.

Mandatory PS3 deliverables: 10–20 organiser-labelled images, at least two organiser categories, a prediction for every
image, a table of actual vs predicted labels, total correct and accuracy, at least two sample outputs, the model used,
approximate inference time per image, model size or memory, a realistic waste-management scenario, and a demo of at most two minutes.

## 2. Reference workflow
| Item | Value | Source |
|---|---|---|
| Course | Arm "ExecuTorch on Arm", Module 2 (CNN inference with quantization) | *brief*; repository read: **verified** |
| Course repo | `ArmDeveloperEcosystem/arm-courses-code-executorch` @ `1045d51` | **verified** (cloned, commit checked out) |
| Course scripts used | `prepare_mobilenetv2.py`, `lower_mobilenetv2_xnnpack.py`, `quantize_mobilenetv2_xnnpack.py`, `create_calibration_images.py`, `pi/benchmark_mobilenetv2.py`, `pi/serve_image_classifier.py` | **verified** (read) |
| Pi environment | Raspberry Pi OS **Trixie** 64-bit; `pyenv shell 3.12.14`; venv `~/executorch_on_arm/.cpu_lab_venv` | *brief* + course error messages; on Pi: *pending* |

## 3. Software versions (course Module 2 pins)
| Package | Version | Status |
|---|---|---|
| Python | 3.12.14 | course pin; dev host used 3.12.3, **verified working** |
| PyTorch | 2.12.1 | **verified** (dev host) |
| TorchVision | 0.27.1 | **verified** (dev host) |
| ExecuTorch | 1.3.1 | **verified** (dev host); aarch64 cp312 wheel exists on PyPI: **verified** |
| TorchAO | 0.17.0 | **verified** (dev host) |

These are the course's documented versions, not proof of what is installed on the Pi.

## 4. Model contract (MobileNetV2 FP32 baseline)
- Weights: `torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1` (`mobilenet_v2-b0353104.pth`, hash-checked). **verified**
- Input `[1, 3, 224, 224]`, NCHW, float32. Resize 256 → CenterCrop 224 → ToTensor → Normalize (ImageNet mean/std). **verified** (`config/model.json`)
- Output: 1000 ImageNet logits. Softmax is applied once. ImageNet label order is taken from the weights metadata. **verified**
- ImageNet accuracy is **not** waste-classification accuracy.

## 5. Export and runtime
- Export: `torch.export` → `to_edge_transform_and_lower(..., partitioner=[XnnpackPartitioner()])` → `.to_executorch()` → `.pte`. **verified** (`scripts/export_fp32.py`)
- Runtime: `executorch.extension.pybindings.portable_lib._load_for_executorch(pte).forward((tensor,))[0]`. **verified** on ExecuTorch 1.3.1 (`src/zyqora/model_runtime.py`)
- There is no silent PyTorch fallback on the Pi. **verified** (it exits with an error)

## 6. Model candidates
| Model | Role | Key figures | Status |
|---|---|---|---|
| MobileNetV2 FP32 | Baseline / default | 13.99 MB `.pte`, whole graph in one XNNPACK delegate | **verified** (host); Pi *pending* |
| MobileNetV2 INT8 (PT2E, XNNPACKQuantizer) | Optimised candidate | 7.59 MB (head FP32) / 3.76 MB (full); top-1 agreement with FP32 16/20 and 15/20 | **verified** (host); not selected yet |
| ShuffleNetV2 x1.0 INT8 (Arm AI Model Portal, Pi 5) | Backup | ~2.4 MB, p50 3.63 ms, peak memory 9.22 MB, ImageNet top-1 68.76%, built with ExecuTorch **1.1.0** | *brief* (published figures; not downloaded; 1.1.0 vs 1.3.1 compatibility not assumed) |
| CLIP ViT-B/32 INT8 | Conditional zero-shot contingency | ~327.4 MB; needs offline tokenizer/text embeddings; Pi latency unknown | *brief*; not implemented by design |
| YOLOv9-S or other detector | Only if images contain multiple objects | COCO labels are not waste categories | not implemented (no evidence of multi-object images) |

Selection rule: organiser-category accuracy first, then category coverage, offline compatibility, Pi latency and CPU time
per image, memory, size, and reliability. The system must keep working with MobileNetV2 FP32 even if every optional
experiment fails. Current choice and data: [MODEL_COMPARISON.md](MODEL_COMPARISON.md).

## 7. Implementation gates
| Gate | Content | Exit criterion | Status |
|---|---|---|---|
| 0 | Inspect repository, identify missing inputs | Checklist | done |
| 1 | Prepare + export FP32 `.pte` with metadata | `.pte` loads and runs | done (host) |
| 2 | ExecuTorch runtime wrapper, XNNPACK check | Output shape valid; backend registered | done (host); Pi *pending* |
| 3 | PyTorch vs ExecuTorch parity | **FP32 top-1 agrees on every parity image** | passed: 100/100 (host) |
| 4 | MaterialWise mapping + config validation | `check_config.py` valid | done; organiser categories *pending* |
| 5 | EcoGuard + threshold sweep, freeze before final run | Threshold frozen | provisional **0.40** (*brief*: initial candidate only); calibration *pending* |
| 6 | Mandatory evaluation, annotated samples, benchmark | results.csv / summary.json from organiser set | pipeline done; organiser run *pending* |
| 7 | VoiceSort Tamil/English offline | Clip plays offline | clips done; playback + Tamil review *pending* |
| 8 | GreenScore verified rewards | Points only after verification; no duplicates | done |
| 9 | INT8 quantization + comparison | Accuracy/latency/size compared | done (host); Pi *pending* |
| 10 | Docs, Pi setup, demo | Docs match commands | done |

Status details: [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md).

## 8. Corrections and rules carried into the implementation (*brief*)
- `Recycle` / `Compost` / `Residual` are **disposal actions**, not necessarily the organiser's category labels. Keep the raw ImageNet class, the organiser category and the disposal action in separate fields.
- Every test image must receive an explicit category prediction. REVIEW items stay in the accuracy denominator, and accepted-only accuracy is "undefined" (not 100%) when nothing was accepted.
- Do not tune the threshold or mapping on the final organiser test set. Calibration images must be separate from the test images.
- CPU utilisation is not CPU efficiency. Report CPU time per image and latency, and measure 1/2/4 threads rather than assuming.
- Do not claim INT8 is 4× faster. Do not make energy-per-image claims without a power measurement. Temperature is diagnostic only.
- Published portal figures are not our measurements.
- Out of scope: blockchain, cloud analytics, mobile apps, dashboards, face recognition, continuous surveillance, robotic sorting, WastePulse analytics, SHA-256-as-novelty, multi-model cascades, and test-time augmentation without demonstrated benefit.
- Positioning: the contribution is the *integration* (verify-to-earn, uncertainty-aware, offline, bilingual on a Pi), not new individual techniques.

## 9. Two-minute demo outline (*brief*; full script in [DEMO.md](DEMO.md))
| Time | Content |
|---|---|
| 0:00–0:15 | Pi 5, internet disabled; the waste-segregation problem |
| 0:15–0:50 | Run Zyqora on the supplied images; MobileNetV2 + ExecuTorch + XNNPACK |
| 0:50–1:15 | Real results: images, correct, accuracy, reviews, median inference time, model size, memory |
| 1:15–1:35 | Two genuine annotated outputs; EcoGuard and manual review |
| 1:35–1:50 | Tamil or English instruction played offline |
| 1:50–2:00 | Supervisor verification and GreenScore reward |

## 10. Open items
Hardware facts not yet confirmed: Pi RAM, cooling, speaker/audio output, storage, camera (`scripts/pi_collect_facts.sh`).
Still needed: organiser images, labels and category names; independent labelled dev images; Pi measurements; native-speaker Tamil review.
