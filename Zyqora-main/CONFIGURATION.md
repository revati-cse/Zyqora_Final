# Configuration guide

All configuration is local. `python scripts/check_config.py` validates every file below and must end with `RESULT: VALID`.

| File | Purpose |
|---|---|
| `config/models.yaml` | **Which model runs.** Profiles (one `.pte` + contract + decoder + EcoGuard each), `active`, `fallback` |
| `config/model_waste.json` | Waste model input/output contract: exact 224×224 resize, ImageNet mean/std, output `[1, 2]` |
| `config/classes_waste.json` | Waste model classes in output order (0 Organic → COMPOST, 1 Recyclable → RECYCLE), aliases, unsupported ground truth (Residual) |
| `config/app.yaml` | ImageNet baseline profile (pte, contract, mapping, EcoGuard 0.40), thread count, GreenScore points, voice file |
| `config/model.json` | ImageNet model contract: resize 256, center-crop 224, output `[1, 1000]` |
| `config/mapping.json` | MaterialWise rules for the ImageNet model only (status EXAMPLE) |
| `config/voice.yaml` | English/Tamil instruction text per action |
| `config/hardware.yaml` | Camera backend and size; actuator backend and GPIO pins |

## Switching the production model
`config/models.yaml → active` stays on `imagenet_mobilenetv2` until the waste model passes the on-Pi gate:
```bash
python scripts/verify_waste_model.py --images <labelled VALIDATION sample with O/ and R/ folders> --expected-sha256 <hash from the Mac>
```
```bash
python scripts/verify_waste_model.py --images <same> --expected-sha256 <hash from the Mac> --activate
```
`--activate` sets `active: waste_mobilenetv2` and pins `sha256`. From then on, a different or corrupted file is refused at load. To roll back, set `active: imagenet_mobilenetv2`.

One-off runs can choose a profile without editing anything: `--profile waste_mobilenetv2`.

## Profile fields (`config/models.yaml`)
| Key | Meaning |
|---|---|
| `decoder` | `direct_classes` (model outputs are the categories) or `imagenet_mapping` (1000 ImageNet logits → MaterialWise) |
| `pte`, `contract` | model file and its contract. The `.pte` must report the contract's input and output shapes, or loading fails. |
| `classes` / `mapping` | class file (direct) or mapping file (imagenet) |
| `sha256` | optional pin; a mismatch is refused |
| `ecoguard.threshold` | minimum category probability to ACCEPT (0–1, boundary inclusive) |
| `ecoguard.min_margin` | optional: best − second-best probability must be at least this (0 disables) |
| `ecoguard.require_top1_mapped` | imagenet only: REVIEW if the top ImageNet class has no rule |
| `ecoguard.calibrated` / `frozen` | bookkeeping recorded in `summary.json`; set them only when true |
| `from_app_config: true` | take everything from `config/app.yaml` (used by the ImageNet profile) |

`fallback` is used by the **station** when the active profile cannot be loaded (logged, printed and announced). `run_zyqora.py` never falls back unless `--allow-fallback` is given, so evaluation results always come from the model you asked for.

## Thresholds
- Waste model: 0.95 is provisional, from the team's validation table (see MODEL_EVALUATION.md). To recalibrate, run on **validation/dev** images, then:
  ```bash
  python scripts/sweep_threshold.py --results outputs/<dev run>/results.csv
  ```
  The sweep refuses results marked `--dataset-role test` or `organiser`.
- A threshold controls how often items are sent to review. It does not detect items outside the model's classes.

## Threads
`config/app.yaml → model.num_threads` (null = ExecuTorch default). Choose it from `scripts/benchmark_pi.py --threads 1 2 3 4` on the Pi.

## Local changes on the Pi
If `config/app.yaml` has uncommitted edits on the Pi, keep them: `git stash`, then `git pull`, then `git stash pop`. This feature branch does **not** modify `config/app.yaml`; all new model settings live in `config/models.yaml`.
