#!/usr/bin/env python3
"""Check that Zyqora's detector decoding matches Ultralytics' own predictions (development machine).

    python scripts/check_detector_parity.py --weights yolo11n.pt --profile yolo11n_640 --images <folder>

For every Ultralytics box (conf 0.25, per-class NMS IoU 0.7) it looks for a Zyqora box of the
same class with IoU > 0.8. Small differences are expected: Zyqora resizes with Pillow
(bilinear) while Ultralytics uses OpenCV, which changes scores near the threshold.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401

_bootstrap.quiet_toolchain_logs()

from zyqora.detection.yolo import DetectorSettings, YoloDetector, box_iou, load_detector_config  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--profile", default=None)
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=50)
    args = ap.parse_args()
    from PIL import Image, ImageOps
    from ultralytics import YOLO

    cfg = load_detector_config()
    prof = cfg.get(args.profile)
    s = DetectorSettings(conf_threshold=0.25, iou_threshold=0.7, agnostic_nms=False, max_detections=300,
                         min_box_area_fraction=0.0)
    det = YoloDetector(prof, s)
    ref = YOLO(args.weights)
    files = sorted(p for p in args.images.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})[: args.limit]
    total = matched = 0
    ious = []
    for p in files:
        with Image.open(p) as im:
            img = ImageOps.exif_transpose(im).convert("RGB")
        r = ref.predict(np.asarray(img)[:, :, ::-1], imgsz=prof.imgsz, conf=0.25, iou=0.7, verbose=False)[0]
        rb, rc = r.boxes.xyxy.numpy(), r.boxes.cls.numpy().astype(int)
        mine = det.detect(img).detections
        mb = np.array([d.box for d in mine]).reshape(-1, 4)
        mc = np.array([d.class_id for d in mine])
        total += len(rb)
        for b, c in zip(rb, rc):
            sel = mb[mc == c] if len(mb) else mb
            if len(sel) and box_iou(b, sel).max() > 0.8:
                matched += 1
                ious.append(float(box_iou(b, sel).max()))
    res = {"profile": prof.name, "images": len(files), "ultralytics_boxes": total, "matched_same_class_iou_gt_0_8": matched,
           "match_rate": matched / total if total else None, "mean_iou_of_matches": float(np.mean(ious)) if ious else None}
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
