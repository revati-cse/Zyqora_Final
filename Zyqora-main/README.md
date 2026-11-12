# Zyqora AI

**Offline, confidence-aware waste sorting assistant with bilingual guidance and verified sorting rewards.**
NAMMA AI hackathon, PS3 (AI for Sustainability: Edge-Based Waste Sorting). Target: Raspberry Pi 5, ExecuTorch, XNNPACK, fully offline.

> **Status, in short:** the software pipeline is implemented and tested on an x86_64 Linux development host.
> **It has not yet run on a Raspberry Pi 5, and the organiser images and labels have not arrived.** The shipped
> `config/mapping.json` is an **EXAMPLE**, not the organiser's categories. See [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md).

## What it does

```
image ─► preprocess (Pillow+NumPy, bit-identical to TorchVision) ─► MobileNetV2 .pte on ExecuTorch/XNNPACK
      ─► softmax ─► MaterialWise (ImageNet classes → organiser category → bin)
      ─► EcoGuard (ACCEPT / REVIEW + reason) ─► VoiceSort (Tamil/English clip)
      ─► GreenScore (PENDING event → supervisor verification → points)
      ─► results.csv · summary.json · annotated images
```

| Component | What it is | What it is not |
|---|---|---|
| **MaterialWise** (`src/zyqora/materialwise.py`) | Editable JSON rules mapping ImageNet class indices to organiser categories and bin actions. It sums the softmax probabilities of each category's classes, so the best *category* can differ from the best *single class*. | It does not train or run a second model. |
| **EcoGuard** (`src/zyqora/ecoguard.py`) | It accepts a recommendation only when the category score is at least the threshold and the model's top-1 class is a known waste item. Otherwise it returns REVIEW with explicit reasons. | It does not guarantee reliability. The threshold of 0.40 is provisional until a sweep on dev data. |
| **VoiceSort** (`src/zyqora/voicesort.py`) | Pre-generated Tamil and English WAV clips played with `aplay`. If playback fails it prints the text and keeps going. | It does not use online TTS. The Tamil wording has not yet been reviewed by a native speaker. |
| **GreenScore** (`src/zyqora/greenscore.py`) | SQLite records. Points are awarded only after a supervisor confirms the item went in the correct bin. Each event can be verified once (atomic), and all queries are parameterised. | It does not reward AI predictions, and has no leaderboard, login or cloud sync. |

The contribution is the *integration* (verify-to-earn, uncertainty-aware, offline, bilingual) on a Pi. None of the individual techniques is claimed to be new.

### Three separate outputs per image
Every image always gets:
1. a **raw ImageNet prediction** (`raw_top1_class`),
2. an **explicit organiser-category prediction** (`predicted_category`), and
3. a **disposal action** (`disposal_action`), which is the bin only if EcoGuard accepts and `MANUAL_REVIEW` otherwise.

REVIEW items keep their predicted category and **stay in the accuracy denominator**. Images that fail to load are listed as failed and counted as incorrect.

## Model

| | |
|---|---|
| Model | TorchVision `mobilenet_v2`, weights `MobileNet_V2_Weights.IMAGENET1K_V1` (hash-checked `mobilenet_v2-b0353104.pth`) |
| Contract | input `[1,3,224,224]` float32 NCHW; Resize 256, CenterCrop 224, ImageNet mean/std; output 1000 logits |
| Export | `torch.export` → `to_edge_transform_and_lower(XnnpackPartitioner)` → `to_executorch()` → `.pte` (Arm course Module 2 workflow, commit `1045d51`) |
| Runtime | `executorch.extension.pybindings.portable_lib._load_for_executorch(pte).forward((x,))[0]`, ExecuTorch 1.3.1 |
| Default artifact | `models/mobilenetv2_fp32_xnnpack.pte`, 13.99 MB, whole graph in one XNNPACK delegate |

INT8 variants were built and compared. FP32 is the default for now; see [MODEL_COMPARISON.md](MODEL_COMPARISON.md).

## Repository layout

```
run_zyqora.py              main CLI (classification + evaluation + voice + GreenScore events)
config/app.yaml            model path, EcoGuard threshold, GreenScore points
config/mapping.json        MaterialWise rules (status EXAMPLE until organiser categories are entered)
config/model.json          model input contract (from the Arm course model_spec.json)
config/voice.yaml          English/Tamil instruction text
src/zyqora/                preprocessing, model_runtime, materialwise, ecoguard, pipeline, evaluator,
                           outputs, benchmark, voicesort, greenscore, config, imagenet_labels (+ bundled labels)
scripts/                   prepare_model, export_fp32, export_int8, check_parity, check_config,
                           sweep_threshold, benchmark_pi, generate_audio, check_offline, pi_collect_facts.sh
tests/                     68 unit tests (unittest)
audio/{en,ta}/*.wav        placeholder clips generated offline with espeak-ng
data/{dev,organiser}/      put images + labels.csv here (not committed)
models/                    generated .pte artifacts (not committed; see models/README.md)
outputs/                   generated results (not committed)
```

## Setup

**Development machine (model export).** Use Linux, macOS, or WSL2 on Windows, with Python 3.12:
```bash
python3.12 -m venv .venv-export && source .venv-export/bin/activate
pip install -r requirements-export.txt
python scripts/prepare_model.py        # downloads + hash-checks official weights (needs internet once)
python scripts/export_fp32.py          # writes models/mobilenetv2_fp32_xnnpack.pte (+ .json metadata)
python scripts/check_parity.py --images data/dev
```

**Raspberry Pi 5.** See [README_PI.md](README_PI.md).

**Unit tests** (any Python ≥ 3.10; torch is not required):
```bash
pip install -r requirements-dev.txt
python -m unittest discover -s tests -t . -v
```

## Commands

| Task | Command |
|---|---|
| Final organiser evaluation | `python run_zyqora.py --images data/organiser --labels data/organiser/labels.csv --out outputs/final` |
| Development run (example mapping) | `python run_zyqora.py --images data/dev --labels data/dev/labels.csv --allow-example-mapping --out outputs/dev` |
| Tamil voice | `python run_zyqora.py --images data/demo --voice ta --allow-example-mapping --out outputs/tamil_demo` |
| No audio (default) | `python run_zyqora.py --images data/organiser --labels data/organiser/labels.csv --voice off` |
| Record GreenScore events | add `--greenscore-user U01` to any run |
| INT8 model | add `--model models/mobilenetv2_int8_xnnpack.pte` |
| Thread count | add `--threads 2` (or set `model.num_threads` in `config/app.yaml`) |
| Benchmark | `python scripts/benchmark_pi.py --model models/mobilenetv2_fp32_xnnpack.pte --threads 1 2 4 --images data/dev` |
| Check configuration | `python scripts/check_config.py` (or `--mapping config/mapping.json`) |
| FP32 parity gate | `python scripts/check_parity.py --images data/dev` |
| INT8 vs FP32 report | `python scripts/check_parity.py --images data/dev --pte models/mobilenetv2_int8_xnnpack.pte --compare-pte models/mobilenetv2_fp32_xnnpack.pte` |
| INT8 export | `python scripts/export_int8.py --calibration <calibration folder> --exclude data/organiser data/dev` |
| Threshold sweep (dev only) | `python scripts/sweep_threshold.py --results outputs/dev/results.csv` |
| Play an instruction | `PYTHONPATH=src python -m zyqora.voicesort --lang ta --say RECYCLE` |
| Generate clips (offline) | `python scripts/generate_audio.py` (needs `espeak-ng`) |
| GreenScore verify | `PYTHONPATH=src python -m zyqora.greenscore verify --event-id 3` (interactive) or `... --observed RECYCLE [--correct RECYCLE]` |
| GreenScore points | `PYTHONPATH=src python -m zyqora.greenscore points --user-id U01` |
| Offline readiness | `python scripts/check_offline.py` |

After `pip install -e . --no-deps`, the `PYTHONPATH=src` prefix is not needed. `run_zyqora.py` exit codes: 0 ok, 2 config, 3 model/runtime, 4 data.

## Output files

`outputs/<run>/results.csv` has one row per image. The mandatory columns are `image_filename, ground_truth_category, raw_top1_class, raw_top1_probability, predicted_category, predicted_category_score, mapping_status, ecoguard_status, review_reason, disposal_action, inference_time_ms, correct`. Supporting columns follow: top-5, per-category scores, margin, mapped mass, rule id, material, preprocess and end-to-end time, and error.

`outputs/<run>/summary.json` holds `total_images, total_correct, total_incorrect, overall_accuracy, accepted_count, review_count, review_rate, accepted_only_accuracy, model_name, model_precision, model_size_mb, inference_latency_{mean,median,p95}_ms, memory_usage_mb`. It also holds coverage, failed images, per-category accuracy, CPU seconds per image, runtime/backends/threads, the mapping and EcoGuard settings (including `calibrated`/`frozen`), the device model, and `evaluation_scope` (`FINAL_ORGANISER_TEST` / `DEVELOPMENT` / `INFERENCE_ONLY`).

`outputs/<run>/annotated/*.jpg` contains at least two genuine samples. The run prefers one ACCEPT and one REVIEW, but never stages a missing case.

## Metric definitions
- **Overall accuracy** = correct organiser-category predictions / **all** evaluated images. REVIEW and failed images are included.
- **Review rate** = REVIEW images / all images.
- **Accepted-only accuracy** = correct ACCEPTed predictions / ACCEPTed predictions. It is reported as `null` ("undefined") when nothing was accepted.
- A REVIEW decision is never counted as correct by itself. An image with no category prediction counts as incorrect.
- Latency: model-only `forward()` time (`perf_counter_ns`) is reported separately from preprocessing and end-to-end time. One untimed warm-up runs first.
- Memory: Linux `VmRSS` (current) and `VmHWM` (peak) of the whole Python process. This includes the interpreter and torch, so it is **not** model-only memory.

## Known limitations
- MobileNetV2 knows **ImageNet** objects, not waste categories. Coverage depends on how well the organiser's images show ImageNet-recognisable objects (bottles, fruit, packets). Crumpled, dirty, or mixed waste will often be REVIEW or wrong.
- The organiser categories, the mapping review, the threshold calibration, Pi latency and memory, offline verification on the Pi, audio playback, and the Tamil text review are all **pending** (see IMPLEMENTATION_STATUS.md).
- INT8 reduced size and host latency but changed top-1 on 20–25% of 20 held-out general images, so it is not the default.
- The ExecuTorch wheel depends on torch, so the Pi environment is not small (about 640 MB RSS for torch + runtime on the x86 host; the Pi figure has not been measured).
- No object detection: one image is classified as one item.
