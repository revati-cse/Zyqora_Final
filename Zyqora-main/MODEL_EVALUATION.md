# Waste model: card and evaluation report

## Status at a glance
| Item | Status |
|---|---|
| Training and validation results | **Reported by the team** (MacBook Air M3, PyTorch MPS) |
| `.pte` on the Pi: loads with XNNPACK, output `[1, 2]`, SHA-256 `b019fc68…094f` pinned | **Verified on the Pi** (`verify_waste_model.py --activate`, 2026-11-14); committed in `33a5480` |
| Integration in the Zyqora pipeline on the Pi | **Verified on the Pi.** The full validation set reproduces the team's result (next section). |
| Thread count / latency / memory on the Pi | **Measured** (BENCHMARK_REPORT.md): 12.7 ms median with 2 threads, 380 MB RSS |
| PyTorch ↔ ExecuTorch parity for the real checkpoint | **Pending** (needs the checkpoint and the `.pte` on one machine) |
| Leakage check train/val ↔ TEST | **Pending** |
| Independent TEST evaluation (1,401 Organic, 1,112 Recyclable) | **Pending.** Not run, so no numbers are claimed. |

## Validation reproduced on the Raspberry Pi 5 (2026-11-14)
`run_zyqora.py --profile waste_mobilenetv2 --images data/validation --labels data/validation/labels.csv --dataset-role dev`, on Raspberry Pi 5 Model B Rev 1.1, ExecuTorch 1.3.1+cpu, XNNPACK, 2 threads:

| | MacBook (team-reported) | Raspberry Pi 5 through Zyqora |
|---|---:|---:|
| Accuracy | 96.65% (1,933 / 2,000) | **96.60% (1,932 / 2,000)** |
| Organic precision / recall | 0.9726 / 0.9600 | 0.973 / 0.959 |
| Recyclable precision / recall | 0.9605 / 0.9730 | 0.960 / 0.973 |
| EcoGuard 0.95: review rate | 10.85% | **10.8%** (216) |
| EcoGuard 0.95: accepted-only accuracy | 98.71% | **98.7%** (1,761 / 1,784; 23 wrong but accepted) |

The near-identical result (one Organic image differs) shows that the Pi deployment matches training. That covers the exact 224×224 resize, ImageNet normalisation, class order (0 Organic, 1 Recyclable) and the exported weights; a wrong resize policy or swapped classes would change many predictions, not one. The remaining difference is most likely a boundary case: MPS vs XNNPACK numerics, or Zyqora's EXIF orientation handling.

**Caveat:** this is the validation set that was used to pick the checkpoint and the threshold, so it is **not** an independent accuracy estimate. The TEST evaluation is still required.

## Real-model behaviour on non-dataset photos (x86 host, 2026-11-14)
The committed `.pte` was run through the dashboard API on 15 general photos (Open Images, not from the waste dataset):
- 5 bottles: 4 Recyclable ACCEPT (0.97–0.998); 1 Organic at 0.66 → REVIEW
- 5 houseplants: 4 Organic ACCEPT (~1.00); 1 Organic at 0.55 → REVIEW
- 5 televisions (Residual): **3 ACCEPTED as Recyclable at 0.98–1.00**, 2 REVIEW

A small sample, but real evidence for the Residual gap described below.

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

With the real model, 3 of 5 television images (labelled `Residual`) were **accepted as Recyclable at 0.98–1.00** (section above). The untrained stand-in did the same with 5 of 5. Zyqora therefore:
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

## Zyqora AI 4.0: evaluations of the detector and the combined pipeline

| Evaluation | Status |
|---|---|
| YOLO decoder parity vs Ultralytics (32 images, x86) | **Done**: 117/122 boxes matched at 640 px (mean IoU 0.985); `.pte` = `.onnx` boxes |
| Detector held-out accuracy (generic objects), PASCAL VOC2007 test, 4,952 images | **Done** (x86; accuracy is device-independent). YOLO11n-640 mAP@0.5 **0.832**, mAP@0.5:0.95 **0.628**, P/R @0.25 0.680/0.859; bottle AP50 0.772, tv 0.831. Full table: DETECTION.md, `reports/detector_eval/` |
| Detector accuracy on **waste** (box-annotated waste test set) | **BLOCKED**: TACO images not reachable from the build environment; no other annotated waste set available |
| MobileNetV2 accuracy on **crops** | **Not measured**. The 0.95 EcoGuard threshold comes from whole-image validation; crops are a different distribution, so the threshold is not validated for crops |
| Combined pipeline: per-object correctness, missed-detection rate, mixed-image success rate, unknown rate | **Not measured**: needs mixed-waste photos with per-object boxes and labels. `scripts/evaluate_detector.py --images/--labels` covers the detection part once such a set exists |
| End-to-end latency | x86 host only (BENCHMARK_REPORT.md); Pi pending |

Observed behaviour with the real models on the x86 host (examples, not statistics):
- **Bottle + wine glass photo:** two separate items, both Recyclable (0.96 / 0.98); a dining table in the frame is ignored.
- **Two TV photos:** detected as `tv` and routed to e-waste, while the classifier alone says Recyclable (up to 1.000). This is the Residual gap, now mitigated for objects the detector knows.
- **An Organic validation photo:** nothing detected, so the whole-image fallback gives the same answer as single-item mode.
