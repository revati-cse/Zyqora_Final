# Object detection, labels and datasets (Zyqora AI 4.0)

## What is shipped

| Item | Value |
|---|---|
| Detector | Ultralytics **YOLO11n**, pretrained on **COCO 2017** (80 generic classes). **Not trained on waste.** |
| Source weights | `yolo11n.pt`, Ultralytics assets release v8.3.0, SHA-256 `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1` |
| Exported files | `models/yolo11n_640_xnnpack.pte` (default), `yolo11n_320_xnnpack.pte`, `yolo11n_640.onnx`, `yolo11n_320.onnx`, all pinned by SHA-256 in `config/detector.yaml` |
| Export | `scripts/export_detector.py` (ExecuTorch 1.3.1 + XNNPACK partitioner; ONNX opset 17), raw head without NMS |
| Runtime | ExecuTorch (same runtime as the classifier) or ONNX Runtime CPU; Zyqora does letterbox, decode and NMS in NumPy, so **Ultralytics is not needed at runtime** |
| Licence | Ultralytics weights are **AGPL-3.0**. This repository is public, which is compatible; a closed-source product would need an Ultralytics licence or other weights. |

### Decoder parity

`scripts/check_detector_parity.py` compares Zyqora's boxes with Ultralytics' own predictions. Development host, 32 images (2 Ultralytics samples + 30 Open Images): at 640 px, **117 of 122** Ultralytics boxes have a same-class Zyqora box with IoU > 0.8 (mean IoU 0.985). At 320 px the figure is 75 of 88. The `.pte` and `.onnx` give identical boxes. The residual difference is the resize (Pillow bilinear vs OpenCV), which moves scores near the threshold.

## Model choice: measured, not assumed

The candidates were YOLO11n and YOLOv8n, each at 640 and 320 px. Accuracy was measured on **PASCAL VOC2007 test** (4,952 images; held out; never used to train COCO models). Development host, `scripts/evaluate_detector.py`, pycocotools 101-point AP, VOC "difficult" boxes ignored. Full JSON is in `reports/detector_eval/`.

| profile | runtime | px | mAP@0.5 | mAP@0.5:0.95 | bottle AP50 | tv AP50 | P / R @0.25 |
|---|---|---|---|---|---|---|---|
| **yolo11n_640** (shipped `.pte`) | executorch | 640 | **0.832** | **0.628** | 0.772 | 0.831 | 0.680 / 0.859 |
| yolo11n_640_onnx | onnxruntime | 640 | 0.832 | 0.628 | 0.772 | 0.831 | 0.680 / 0.860 |
| yolov8n_640 | onnxruntime | 640 | 0.813 | 0.600 | 0.774 | 0.836 | 0.676 / 0.843 |
| yolo11n_320 | onnxruntime | 320 | 0.762 | 0.563 | 0.622 | 0.748 | 0.719 / 0.761 |
| yolov8n_320 | onnxruntime | 320 | 0.757 | 0.558 | 0.647 | 0.745 | 0.725 / 0.757 |

YOLO11n beats YOLOv8n overall at both sizes and is also faster on the development host (x86: 38 vs 45 ms at 640, 2 threads). It is smaller too (10.2 vs 12.2 MiB as `.pte`). 640 px is the default because small objects such as bottles lose 15 AP points at 320 px. The 320-px profile is offered for faster live-camera use. **Measured on the Raspberry Pi 5 (2 threads):** YOLO forward is 196.7 ms at 640 px and 43.1 ms at 320 px. The full pipeline median is 224.0 / 61.8 ms (BENCHMARK_REPORT.md). Use 640 for photo analysis and 320 for continuous Live Camera.

**These are generic-object numbers.** No waste-detection mAP exists, because no box-annotated waste test set was available in the build environment (see TACO below). The end-to-end accuracy on mixed-waste photos is also **not measured**: it needs labelled mixed-waste images.

## Material hints (`config/detector_materials.yaml`)

The COCO label only gives a material where the class is normally one material:
- `wine glass` → Glass (likely)
- `book` → Paper
- foods → Organic
- `cell phone` / `laptop` / `tv` / … → E-Waste (routed to the e-waste stream)

Ambiguous classes such as `bottle`, `cup`, `bowl` and cutlery stay **Unknown**. COCO classes that are not listed (person, chair, car, …) are **not waste items**: they are drawn but neither classified nor scored. `expected_model_class` lets a contradiction send the item to REVIEW. Example: a `banana` that the classifier calls Recyclable.

**Common waste that COCO cannot detect.** Cans, plastic bags, cardboard boxes, crumpled paper and wrappers have no COCO class and are often missed. When no waste object is found, the whole image is classified as one item and marked as such (`whole_image_fallback`).

## Organiser-defined labels (`config/label_profiles.yaml`)

The trained MobileNetV2 has **two** outputs, `0 Organic, 1 Recyclable`. These are the labels of the organiser dataset (`O_…` / `R_…` images, verified on the Pi's validation set: 1,000 + 1,000). Profiles rename them one-to-one; they cannot add classes:

| profile | Organic → | Recyclable → |
|---|---|---|
| `organiser` (default) | Organic | Recyclable |
| `recyclable` | Non-Recyclable (= not for the dry stream; may be compostable) | Recyclable |
| `food` | Food (= food / organic; the class also contains leaves and peels) | Waste |

The model label, the organiser label, the material and the disposal stream are stored separately. Unreliable items are shown as `Unknown` in every profile. The active profile is chosen in **Settings**.

## Datasets

| Dataset | Use | Status |
|---|---|---|
| Organiser waste images (O/R) | MobileNetV2 training (team) and validation | validation reproduced on the Pi (MODEL_EVALUATION.md); TEST pending |
| COCO 2017 | pretraining of the shipped YOLO11n (by Ultralytics) | not used by us |
| PASCAL VOC2007 test | held-out detector **selection** (generic objects) | evaluated, `reports/detector_eval/` |
| TACO (tacodataset.org, MIT licence, 1,500 photos, 4,784 boxes, 28 supercategories) | waste-detector fine-tuning | **BLOCKED** in the build environment: the image host is not reachable; training also needs a GPU and team approval |

### Fine-tuning pipeline (implemented, not run)

```bash
python scripts/prepare_taco.py --taco /path/TACO/data --out data/taco_yolo     # split by image, seeded
```
```bash
python scripts/train_detector.py --data data/taco_yolo/data.yaml --base yolo11n.pt --epochs 100 --confirm
```
```bash
python scripts/export_detector.py --weights runs/detect/zyqora_taco/weights/best.pt --name taco_yolo11n --imgsz 640
```
```bash
python scripts/evaluate_detector.py --profile taco_yolo11n_640 --images data/taco_yolo/test/images --labels data/taco_yolo/test/labels --dataset-classes data/taco_yolo/classes.txt
```
Then use `config/detector_materials_taco.yaml`, whose TACO classes do carry materials (Can → Metal, Carton → Cardboard).

`prepare_taco.py` was verified on the real TACO `annotations.json` (labels only):
- 1,500 images, split 1,200 / 151 / 149
- 3,813 / 531 / 440 boxes
- 8 supercategories have fewer than 20 training boxes, so expect weak AP for those.

`train_detector.py` refuses to start without `--confirm`.
