# Waste model: card and evaluation report

## Status at a glance
| Item | Status |
|---|---|
| Training and validation results | **Reported by the team** (MacBook Air M3, PyTorch MPS). This repository did not reproduce them; the training code and data are on the MacBook. |
| `.pte` export, Pi load, output `[1, 2]`, SHA-256 match Mac ↔ Pi | **Reported by the team.** Re-checked automatically by `scripts/verify_waste_model.py` on the Pi. |
| Integration in the Zyqora pipeline | **Implemented and tested** with a stand-in 2-class `.pte` of the same architecture (see below). Waiting for `verify_waste_model.py` with the real file. |
| PyTorch ↔ ExecuTorch parity for the real model | **Pending** (needs the checkpoint and the `.pte` on one machine; command below) |
| Leakage check train/val ↔ TEST | **Pending** (needs the datasets; command below) |
| Independent TEST evaluation (1,401 Organic, 1,112 Recyclable) | **Pending.** Not run, so no numbers are claimed. |

## Model card
| | |
|---|---|
| Architecture | TorchVision `mobilenet_v2`; `classifier.1` replaced by `Linear(1280, 2)` |
| Initialisation | ImageNet (`MobileNet_V2_Weights.IMAGENET1K_V1`), then fine-tuned |
| Classes (output index) | **0 = Organic**, **1 = Recyclable** (`config/classes_waste.json`) |
| Disposal actions | Organic → `COMPOST`, Recyclable → `RECYCLE` |
| Input contract | RGB, **Resize to exactly 224×224** (no crop, aspect not preserved), scale to [0, 1], ImageNet mean/std, NCHW float32 `[1, 3, 224, 224]` (`config/model_waste.json`) |
| Output | `[1, 2]` logits. Softmax is applied once, in `zyqora.decoders.DirectClassDecoder`. |
| Artifact | `models/mobilenetv2_waste_fp32_xnnpack.pte`, FP32, ExecuTorch 1.3.1 + XNNPACK, about 8.47 MiB |
| Data (team-reported) | 20,556 training / 2,000 validation images; best checkpoint at fine-tuning epoch 4 |
| **Not supported** | **Residual / any third category.** The model always outputs Organic or Recyclable. |

The NumPy preprocessing used on the Pi is bit-identical to `torchvision.transforms.Resize((224, 224))` + `ToTensor` + `Normalize` (max difference 0.0, unit test `test_torchvision_equivalence_if_available`). **Still to confirm:** that your training/validation transform was exactly `Resize((224, 224))` and not `Resize(256)` + `CenterCrop(224)`. The checkpoint parity run below checks the export, not the training transform.

EXIF orientation: Zyqora applies `ImageOps.exif_transpose` (correct for real photos). TorchVision's default `ImageFolder` loader does not, so a training image with an EXIF rotation tag would have been seen un-rotated. This only matters for images that carry a rotation tag.

## Validation results (team-reported, NOT independent test results)
Accuracy 96.65% (1,933 / 2,000).

| Actual \ Predicted | Organic | Recyclable |
|---|---:|---:|
| Organic | 960 | 40 |
| Recyclable | 27 | 973 |

Derived from that matrix (arithmetic only):

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| Organic | 0.9726 | 0.9600 | 0.9663 |
| Recyclable | 0.9605 | 0.9730 | 0.9667 |

EcoGuard threshold table (team-reported, validation set; the counts are derived):

| Threshold | Review rate | Reviewed | Accepted | Accepted accuracy | ≈ wrong but accepted |
|---|---:|---:|---:|---:|---:|
| 0.80 | 4.10% | 82 | 1,918 | 97.86% | 41 |
| 0.90 | 7.55% | 151 | 1,849 | 98.32% | 31 |
| **0.95 (provisional)** | 10.85% | 217 | 1,783 | 98.71% | 23 |
| 0.98 | 15.05% | 301 | 1,699 | 99.12% | 15 |

0.95 is configured as a **provisional** candidate (`config/models.yaml`, `calibrated: false`). It was chosen on validation data. It is not a guarantee, and **it cannot detect unknown items** (next section).

## The Residual / unknown-item gap
The model has no Residual class and no "none of these" output. Softmax over two classes always sums to 1, so an item that is neither organic nor recyclable can still receive a very high score for one of them.

In the stand-in integration test (below), all 5 television images (labelled `Residual`) were **accepted as Recyclable at about 100%**. That stand-in is not your model, but your model shares the same closed-set output, so you should expect the same failure mode. Zyqora therefore:
- never predicts Residual and never claims three-class accuracy
- accepts `Residual` ground-truth labels in evaluation, counts them in the denominator as incorrect, and reports `unsupported_ground_truth_images` and `unsupported_ground_truth_accepted_into_a_bin` separately
- holds every low-confidence item for a supervisor (no bin is opened)

**Proposed, testable fixes** (not implemented, because each needs data or retraining you would need to approve):
1. Retrain with a third class, `Residual`, using labelled residual images. This is the only fix that actually classifies Residual. The pipeline already supports it: add the class to `classes_waste.json` and change the contract output to `[1, 3]`.
2. Open-set rejection without retraining: compare the penultimate-layer (1280-d) features to Organic/Recyclable class centroids from the training set, and send items far from both to REVIEW. Validate it on a held-out "unknown" set before use.
3. Process: in the demo and deployment, state that the station handles Organic/Recyclable only, and that residual items go to the supervisor.

## Stand-in integration test (development host, not your model)
To test the 2-class path without your files, a **stand-in** checkpoint was built. It uses the ImageNet backbone plus a 2-row head averaged from the ImageNet head rows of food classes and bottle/paper classes, with **no training**. It was exported through the same `scripts/export_fp32.py`. It was never committed. It exported to 8,880,392 bytes (8.47 MiB, matching your reported size, as expected for the same architecture).

| Check (x86_64 host, ExecuTorch 1.3.1) | Result |
|---|---|
| Export → `.pte`, XNNPACK delegation | 1 delegate (whole graph), 1 non-delegated `getitem` |
| ExecuTorch load, output shape | `[1, 2]` |
| PyTorch vs ExecuTorch on identical tensors (15 images) | top-1 15/15, max abs logit diff 1.8e-5, max abs prob diff 2.8e-6 |
| Waste decoder on the ImageNet `.pte` (and vice versa) | refused at load (`produces output (1, 1000), but this model profile expects (1, 2)`) |
| `verify_waste_model.py` with swapped O/R folders | FAILED at 20% accuracy with a "class order may be reversed" warning; config left unchanged |

## Commands still to run (with your files)
On the machine that has both the checkpoint and the `.pte` (the MacBook export venv):
```bash
python scripts/check_parity.py --images <a few hundred VALIDATION images> --contract config/model_waste.json --checkpoint /Users/darius/zyqora-training/exports/mobilenetv2_waste_fp32.pth --pte models/mobilenetv2_waste_fp32_xnnpack.pte --out outputs/parity_waste
```
Leakage check (wherever the dataset folders are):
```bash
python scripts/check_leakage.py --split train=<TRAIN dir> --split val=<VAL dir> --split test=<TEST dir> --out outputs/leakage
```
Independent TEST evaluation (on the Pi for Pi timings), frozen model and threshold. Run it on the full TEST set and, if leakage was found, again without the overlapping files:
```bash
python run_zyqora.py --profile waste_mobilenetv2 --images <TEST dir with O/ and R/> --labels-from-folders --dataset-role test --quiet --out outputs/test_waste
```
```bash
python run_zyqora.py --profile waste_mobilenetv2 --images <TEST dir> --labels-from-folders --dataset-role test --quiet --exclude-list outputs/leakage/test_overlap_files.txt --out outputs/test_waste_dedup
```
`outputs/test_waste/summary.json` then holds overall accuracy, per-class precision/recall/F1, the confusion matrix, accepted-only accuracy and metrics, review rate, and incorrect accepted predictions. `results.csv` has one row per image. Do not change the threshold after looking at these results (`sweep_threshold.py` refuses `INDEPENDENT_TEST` results).
