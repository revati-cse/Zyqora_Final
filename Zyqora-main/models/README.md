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
