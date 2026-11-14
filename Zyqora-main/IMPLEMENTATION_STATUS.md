# Implementation status

## Phase 3: Zyqora AI 3.0 dashboard (branch `feature/zyqora-ai-3-dashboard`, 2026-11-13)

**Where this work ran:** the same x86_64 cloud container as Phase 2. The Raspberry Pi, the MacBook, the real `.pte`, the checkpoints and the TEST dataset were **not reachable**; no Remote Control folders were shared. GitHub still had **no committed waste `.pte`** (`models/` only contains `.gitkeep` and `README.md`).

### Built and tested here
| Area | Evidence |
|---|---|
| FastAPI backend (`src/zyqora/api/`, `serve_zyqora.py`) reusing the existing pipeline; model loaded once; inference lock + bounded queue; typed schemas; structured error codes; upload validation; CORS only for listed origins; SQLite history (WAL); thumbnails; telemetry from /proc; GreenScore, voice, camera and batch endpoints; serves the built SPA | 16 API tests (fake ExecuTorch binding via dependency injection) + live HTTP smoke tests with **real ExecuTorch** on the stand-in `.pte` |
| React dashboard (`frontend/`): Overview, Live Classification, Batch, Analytics, GreenScore, Model Performance, Device, History, Settings; light/dark; English/Tamil; reduced motion; lazy 3D bin with SVG fallback | `npm run lint` clean, `npm run typecheck` clean, 11 Vitest UI tests, `npm run build` OK |
| Browser end-to-end on the **real stack** (built SPA → FastAPI → ExecuTorch stand-in → EcoGuard) | Playwright/Chromium: upload → Classify Waste → `Recyclable · ACCEPT`; a low-confidence image → `REVIEW REQUIRED`, no bin; Tamil rendering; 0 console errors; **0 px horizontal overflow on all 9 pages at 375 px** and at desktop width; iPhone 13 emulation |
| API latency on host | BENCHMARK_REPORT.md: model 3.7 ms, on-device 8.6 ms, HTTP round trip 31 ms median (stand-in, x86) |
| Python suite | **115 tests** pass with torch + ExecuTorch + FastAPI. Without torch, 113 pass and 2 skip. |

The UI tests and API tests mock the network or the ExecuTorch binding. They prove the UI and HTTP logic, **not** that the trained model predicts correctly. The only real-model evidence is still the team's own Pi run (`.pte` loads, output `[1, 2]`).

### Verified on the Raspberry Pi 5 (2026-11-14, run by the team, outputs reviewed)
- Branch checked out on the Pi; **114/115 tests passed** (the 1 failure was a test that depended on the default `num_threads` value; fixed in `1ba195f`), including the real-ExecuTorch integration test on ARM.
- `check_config.py`: VALID.
- **Validation set reproduced on the Pi: 96.60% (1,932/2,000)** vs 96.65% reported; review rate 10.8%; accepted-only accuracy 98.7% (MODEL_EVALUATION.md).
- `verify_waste_model.py --activate`: all checks PASS; `active: waste_mobilenetv2`, SHA-256 `b019fc68…094f` pinned.
- Thread sweep: **2 threads → 12.73 ms median**, 25.3 ms CPU per inference, 380 MB RSS (BENCHMARK_REPORT.md).
- Model + verification record + config committed **from the Pi** (`33a5480`, over SSH). The committed `.pte` is byte-identical to the verified one.

### Not done (needs the Pi, the Mac or the datasets)
1. Dashboard on the Pi (README_PI §16: `npm run build`, `start_pi_demo.sh`), opened from a laptop; dashboard hardware checklist in HARDWARE.md.
2. (done: the model is committed)
3. Parity of the real checkpoint vs the `.pte` on the Mac, and confirmation that the training transform was `Resize((224,224))`.
4. Leakage check and independent TEST evaluation. The Model Performance page shows "Not measured yet" until `outputs/test_waste/summary.json` exists.
5. Pi API latency and camera-to-result latency (the model benchmark is done).
6. Tamil text review by a fluent speaker.

### Requirement gaps (honest list)
- **Residual:** not a model class. REVIEW is a decision, not a third class; residual items can be confidently mis-sorted. Proposed fixes are in MODEL_EVALUATION.md.
- **Browser camera** needs a secure context (https or localhost). Over plain http on the LAN, use the Pi camera button or file upload. No TLS setup is included.
- **Continuous video classification** is not implemented (single-shot by design).
- **Lenis smooth scrolling** was not added: a dashboard of short pages doesn't benefit from it. Motion is used only for the result card entrance. Three.js is lazy-loaded only on Overview.
- shadcn/ui components were written into `src/components/ui` (shadcn style: Radix + CVA + tailwind-merge) instead of fetched with the shadcn CLI.
- ESLint 9 is used for linting; npm reports it as out of support (dev-only tool).

---

## Phase 2: waste-model integration (branch `feature/mobilenetv2-waste-integration`, 2026-11-13)

**Where this work ran:** an x86_64 Linux cloud container, not the Raspberry Pi and not the MacBook. `/home/pi/...` and `/Users/darius/...` were not reachable, so the team's `.pte`, checkpoints, TEST dataset and the Pi's uncommitted changes were **not available**. Everything that needs those files is listed under "Still to run" with exact commands. No metric below comes from those files.

### Done and tested here
| Area | Evidence |
|---|---|
| Model profiles (`config/models.yaml`, `profiles.py`) | unit tests: wrong `.pte` for a profile refused at load (both directions); SHA-256 pin; invalid configs rejected |
| Waste decoder: 0 Organic → COMPOST, 1 Recyclable → RECYCLE, softmax once | unit tests incl. exact probabilities and "never reversed" |
| Exact 224×224 preprocessing contract | bit-identical to `torchvision.transforms.Resize((224,224))` (max diff 0.0) |
| EcoGuard 0.95 (provisional) | boundary tests (0.95 accepts, 0.9499 reviews); REVIEW never sorted |
| Explicit fallback to the ImageNet profile with its own decoder | unit tests + CLI run (`--allow-fallback`; station default, `--no-fallback`) |
| Evaluation: P/R/F1, confusion matrix, accepted-only metrics, folder labels (O/R), unsupported Residual ground truth, `--dataset-role test`, `--exclude-list` | unit tests + stand-in run |
| Leakage check (SHA-256 exact + dHash near-duplicates) | unit test with planted exact and resized duplicates |
| Camera (Picamera2 / rpicam-still / folder), sorter (simulation / GPIO), station UI, logging | unit tests with fakes; end-to-end station simulation with all network connections blocked |
| Integration gate `verify_waste_model.py` | stand-in: passes and activates; swapped class folders fail at 20% and leave config unchanged |
| Export / parity / benchmark tooling for the 2-class model | stand-in `.pte` (8.47 MiB): 1 XNNPACK delegate, parity 15/15, max abs logit diff 1.8e-5 |
| Tests | **99 tests pass** with torch + ExecuTorch (export venv); with system Python 3.13 without torch, 97 pass and 2 skip |

The stand-in is an untrained 2-class head on the ImageNet backbone. It is used **only** to exercise the code and was never committed. None of its accuracy figures describe your model.

### Still to run, needs the Pi, the Mac or the datasets
1. On the Pi: `scripts/verify_waste_model.py --images <VAL sample> --expected-sha256 <Mac hash>`, then `--activate` (README_PI §0.4).
2. On the Mac: parity of the real checkpoint vs `.pte` (MODEL_EVALUATION.md). Also confirm the training transform was `Resize((224,224))`.
3. `scripts/check_leakage.py` over TRAIN/VAL/TEST.
4. TEST evaluation (full set, and without duplicates if any are found) on the Pi.
5. `scripts/benchmark_pi.py --threads 1 2 3 4` and a station `--auto 30` latency run on the Pi; fill in BENCHMARK_REPORT.md.
6. Commit `models/mobilenetv2_waste_fp32_xnnpack.pte` + `.json` from the Pi (README_PI §0.5).

### Physical hardware tests still needed (none done)
| Test | How |
|---|---|
| Pi camera capture (Picamera2 and rpicam-still) | HARDWARE.md, camera section; station `--camera auto` |
| Camera unplugged / failing mid-run | unplug and capture: expect `CAMERA ERROR`, stop after 3 failures |
| Speaker playback, Tamil and English | `python -m zyqora.voicesort --lang ta --say RECYCLE` |
| GPIO actuator per bin, REVIEW keeps bins closed | HARDWARE.md checklist, then `--enable-actuator` |
| Offline run with Wi-Fi off | README_PI §14 checklist |
| Thermal stability over a long run | station `--auto 200`, `vcgencmd get_throttled` |

### Known gaps
- **Residual is not recognised** by the waste model; unknown items can be confidently forced into a class. Proposed fixes are in MODEL_EVALUATION.md.
- The Pi's local `config/app.yaml` changes and untracked data were not visible from here. This branch does not modify `config/app.yaml`.
- `.pte` artifacts can only be committed from a machine that has them (the Pi).

---

## Phase 1 (2026-11-12)
Last updated: 2026-11-12. Development host: x86_64 Linux cloud container (4 vCPU Xeon, 16 GB RAM), Python 3.12.3 venv with torch 2.12.1, torchvision 0.27.1, executorch 1.3.1, torchao 0.17.0. **No Raspberry Pi was available.**

## Inputs that were missing
| Input | Effect |
|---|---|
| **The original `PLAN.md` was not in the repository.** | The verified facts restated in the implementation brief were used instead (versions, model contract, course commit `1045d51`, ShuffleNet/CLIP figures, 0.40 threshold candidate, demo outline). The Arm course source at `1045d51` was read directly to confirm the workflow and pins. [PLAN.md](PLAN.md) is a **reconstruction** from those facts, tagged by what was verified. If the original exists, replace it and re-check anything that differs. |
| **Organiser images, labels and category names** | `config/mapping.json` is an EXAMPLE (`Recyclable`/`Organic`/`Residual`). The final evaluation is blocked by design until `status` is `ORGANISER`. |
| **Independent labelled dev images of waste** | The threshold of 0.40 is provisional (`calibrated: false`, `frozen: false`). `sweep_threshold.py` is implemented and tested but not yet run on real dev data. |
| **Raspberry Pi 5 hardware facts** (RAM, cooling, storage, audio, camera) | Not confirmed. Run `scripts/pi_collect_facts.sh`. |
| **Fluent Tamil reviewer** | The Tamil text in `config/voice.yaml` is a draft. The clips are robotic espeak-ng placeholders. |

## Gate status
| Gate | Status | Evidence |
|---|---|---|
| 0 Inspect | Done | Legacy code reviewed. See "Changes to the existing project" below. |
| 1 Model export | **Done (host)** | `prepare_model.py` hash-checked the official weights. `export_fp32.py` produced a 13.99 MB `.pte`, entire graph in 1 XNNPACK delegate, which loads and runs via the ExecuTorch runtime. |
| 2 ExecuTorch runtime | **Done (host)**, Pi pending | `model_runtime.py` uses the verified API `_load_for_executorch(...).forward((x,))[0]`. It checks for `XnnpackBackend`, validates input shape via `method_meta` and output shape, configures threads, and has no PyTorch fallback. |
| 3 Parity | **Passed (host)** | FP32 vs PyTorch: 100/100 top-1 agreement, max abs logit diff 4.4e-5. NumPy preprocessing vs TorchVision: max diff 0.0. |
| 4 MaterialWise | Done, **organiser categories pending** | `materialwise.py`, `check_config.py`, unit tests. |
| 5 EcoGuard | Done, **calibration pending** | `ecoguard.py`, `sweep_threshold.py` (refuses results marked as the final test set). |
| 6 Mandatory evaluation | Pipeline done and verified on dev images, **organiser run pending** | `run_zyqora.py` writes results.csv, summary.json and annotated images; benchmark_pi.py. Smoke run on 7 images: a corrupted file was counted, REVIEW items were kept in the denominator, and genuine ACCEPT and REVIEW annotations were produced. |
| 7 VoiceSort | Done, **playback not verified** | 12 WAV clips (en/ta). The host has no audio device; the fallback to printed text was verified. |
| 8 GreenScore | Done | Verify/duplicate/concurrency/injection paths tested (unit tests + CLI run). |
| 9 INT8 | Done (host) | INT8 7.59 MB and full-INT8 3.76 MB. Top-1 agreement with FP32 on held-out images was 80% / 75%, so FP32 stays the default (MODEL_COMPARISON.md). |
| 10 Docs and demo | Done | README, README_PI, DEMO, MODEL_COMPARISON, this file. |

## Tested
- `python -m unittest discover -s tests -t .`: **68 tests pass** under the export venv (Python 3.12 + torch + ExecuTorch, 0 skipped) and under system Python 3.13 without torch (2 skipped: TorchVision equivalence and the real-`.pte` integration test).
- `scripts/check_config.py` reports VALID, with warnings for the EXAMPLE mapping and the provisional threshold.
- `run_zyqora.py` error paths: missing image folder gives exit 4; missing `.pte` gives exit 3; ExecuTorch not installed gives exit 3 with no PyTorch fallback; EXAMPLE mapping without `--allow-example-mapping` gives exit 2.
- `check_parity.py` (FP32 gate and INT8 report), `export_int8.py` (refuses calibration images that overlap the test folder, verified), `benchmark_pi.py` (1/2/4 threads, subprocess per config), `generate_audio.py`, `check_offline.py`, and the GreenScore CLI.

## Hardware-dependent items still to do (on the Pi)
1. `bash scripts/pi_collect_facts.sh`, then fill in RAM, cooling, storage, audio and camera here.
2. Install per README_PI.md and confirm `XnnpackBackend` is registered on aarch64.
3. `python -m unittest discover -s tests -t .` (the integration test should run, not skip).
4. `python scripts/check_parity.py --images data/dev` on the Pi.
5. `python scripts/benchmark_pi.py --threads 1 2 4` for FP32 and INT8. Update MODEL_COMPARISON.md with Pi figures and set `model.num_threads`.
6. Play a clip through the actual speaker, in both languages.
7. Run the whole demo with Wi-Fi disabled and tick the offline checklist in README_PI.md section 14.

## Dataset-dependent items still to do
1. Enter the organiser categories in `config/mapping.json`, review the class rules, and set `status: ORGANISER`.
2. Collect separate labelled dev images, run `sweep_threshold.py`, then set `threshold`, `calibrated: true` and `frozen: true`.
3. Run the final evaluation and keep `outputs/final/` (results.csv, summary.json, annotated/).
4. Compare FP32 vs INT8 **category** accuracy on the organiser set before the final model choice.

## Known issues / limitations
- ImageNet does not contain many waste concepts (crushed cans, food scraps, mixed packaging). Expect REVIEWs and errors on such images. The fix is a better mapping (or, as a last resort, the CLIP contingency); a confidence trick will not solve it.
- `memory_usage_mb` in summary.json is whole-process RSS, dominated by importing torch and ExecuTorch (about 640 MB on the x86 host with the CUDA-enabled torch wheel).
- The offline probe in `check_offline.py` uses a direct socket test; it is not a full proof of isolation. Confirm with `nmcli`/`ip` as well.
- `.pte` files are not committed (gitignored). Regenerate them or copy them to the Pi.

## Changes to the existing project
The repository previously contained a Mac/Pi 4 starter in `backend/app/` (PyTorch-eager engine, FastAPI endpoint, flip-TTA "retry", WastePulse statistics, `label_map.yaml`/`thresholds.yaml`, `requirements-mac.txt`). It was **replaced** because it did not use ExecuTorch, mixed REVIEW handling with accuracy differently, and included features the brief excludes (TTA without demonstrated benefit, WastePulse analytics, a web API). Its useful parts were ported:
- The preprocessing contract was ported, and is now verified bit-exact against TorchVision.
- The idea of EcoGuard ACCEPT/REVIEW (now category-score based, with reasons).
- The UNCONFIGURED label-map guard became the EXAMPLE/ORGANISER mapping status.
- The SQLite history became GreenScore.
- The "REVIEW counts in accuracy" and "do not tune on the test set" rules.

The old code is still available in git history (commit `5719901`).
