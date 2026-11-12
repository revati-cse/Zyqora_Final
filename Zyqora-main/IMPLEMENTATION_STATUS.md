# Implementation status

Last updated: 2026-11-12. Development host: x86_64 Linux cloud container (4 vCPU Xeon, 16 GB RAM), Python 3.12.3 venv with torch 2.12.1, torchvision 0.27.1, executorch 1.3.1, torchao 0.17.0. **No Raspberry Pi was available.**

## Inputs that were missing
| Input | Effect |
|---|---|
| **`PLAN.md` was not in the repository.** | The verified facts restated in the implementation brief were used instead (versions, model contract, course commit `1045d51`, ShuffleNet/CLIP figures, 0.40 threshold candidate, demo outline). The Arm course source at `1045d51` was read directly to confirm the workflow and pins. Add `PLAN.md` to the repo and re-check anything it says that differs. |
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
