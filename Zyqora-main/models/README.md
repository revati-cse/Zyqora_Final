# models/

Generated artifacts (not committed; see .gitignore). Re-create them with:

```bash
python scripts/prepare_model.py   # mobilenet_v2.pth, imagenet_labels.json, model_spec.json
python scripts/export_fp32.py     # mobilenetv2_fp32_xnnpack.pte + .json (size, sha256, delegation, integrity check)
python scripts/export_int8.py --calibration <folder> --exclude data/organiser   # mobilenetv2_int8_xnnpack.pte
```

Reference values from the development-host export (ExecuTorch 1.3.1, torch 2.12.1):

| file | bytes | sha256 |
|---|---|---|
| mobilenet_v2.pth (state dict of IMAGENET1K_V1) | 14,261,323 | bb900d7ca3143987112eef231e05b167922470113df1ee7ab4aefa821c0ba8ea |
| mobilenetv2_fp32_xnnpack.pte | 13,994,144 | 9e73ba3a7df22db18a7b1f1d8f9640899ff3c43d88069fc4bbb35ede5fe5363d |
| mobilenetv2_int8_xnnpack.pte (calibrated on 80 Arm course images) | 7,594,272 | ee367557930ae41ce939b92b7f6688ad5d23efa012e81525809d5acfb4fc6fcd |

FP32 exports are expected to be reproducible; INT8 depends on the calibration images.
The runtime never downloads anything: only the `.pte` (plus the bundled label list) is needed on the Pi.

## Zyqora AI 4.0 detector (committed)

Pretrained Ultralytics YOLO11n (COCO-80), exported without NMS by `scripts/export_detector.py`; pinned in `config/detector.yaml`. **AGPL-3.0 weights.** Not trained on waste (DETECTION.md).

| file | bytes | sha256 |
|---|---|---|
| yolo11n_640_xnnpack.pte (default) | 10,683,072 | 5e9e694e561b1d5a138545f59f5c22060aa710ae62ae68c8daab76902566dde2 |
| yolo11n_320_xnnpack.pte | 10,607,552 | d6e3d9399854a3bfc8def4abdee465f7497674f749e4721db1bff68698b2ffc0 |
| yolo11n_640.onnx | 10,741,397 | 080b81054ff536853b34f32bc4aab2d0635c5b3071ba3286134c427aefb67190 |
| yolo11n_320.onnx | 10,615,289 | 35fdbf95b1d8c82f447a455742bc79db013973b73b36b0f3d92912e895a6eadf |

Source: `yolo11n.pt`, Ultralytics assets release v8.3.0, sha256 `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1`.
