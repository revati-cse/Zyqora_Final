#!/usr/bin/env python3
"""Evaluate a Zyqora detector profile on a held-out, box-annotated dataset.

Two dataset formats:

  PASCAL VOC (XML) - used to choose the pretrained COCO detector (model + input size)
      python scripts/evaluate_detector.py --voc /path/VOCdevkit/VOC2007 --split test --profile yolo11n_640_onnx

  YOLO txt labels  - e.g. a TACO conversion or your own annotated waste photos
      python scripts/evaluate_detector.py --images data/det_test/images --labels data/det_test/labels \
          --dataset-classes data/det_test/classes.txt --class-map config/detector_eval_map_example.yaml

--class-map maps each dataset class to one or more detector class names (YAML mapping).
Without it, dataset class names must equal detector class names. Detector classes
that no dataset class maps to are dropped from the predictions (they cannot be scored).

mAP uses conf >= 0.001 and up to 300 boxes per image (standard protocol);
precision / recall are additionally reported at --operating-conf (the app setting).
Results go to outputs/detector_eval/<profile>_<dataset>.json and record the device.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401

_bootstrap.quiet_toolchain_logs()

import yaml  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402

from zyqora.benchmark import platform_facts  # noqa: E402
from zyqora.detection.evaluate import GroundTruth, ImageRecord, Prediction, evaluate  # noqa: E402
from zyqora.detection.yolo import DetectorSettings, YoloDetector, load_detector_config  # noqa: E402
from zyqora.paths import OUTPUTS_DIR  # noqa: E402

# PASCAL VOC class -> COCO class name (standard correspondence; all 20 VOC classes exist in COCO).
VOC_TO_COCO = {
    "aeroplane": "airplane", "bicycle": "bicycle", "bird": "bird", "boat": "boat", "bottle": "bottle",
    "bus": "bus", "car": "car", "cat": "cat", "chair": "chair", "cow": "cow", "diningtable": "dining table",
    "dog": "dog", "horse": "horse", "motorbike": "motorcycle", "person": "person", "pottedplant": "potted plant",
    "sheep": "sheep", "sofa": "couch", "train": "train", "tvmonitor": "tv",
}


def load_voc(root: Path, split: str, limit: int | None):
    ids = (root / "ImageSets" / "Main" / f"{split}.txt").read_text().split()
    if limit:
        ids = ids[:limit]
    for i in ids:
        tree = ET.parse(root / "Annotations" / f"{i}.xml")
        gts = []
        for obj in tree.findall("object"):
            name = obj.findtext("name").strip()
            bb = obj.find("bndbox")
            # VOC is 1-based inclusive pixel coordinates
            box = tuple(float(bb.findtext(k)) - (1 if k in ("xmin", "ymin") else 0) for k in ("xmin", "ymin", "xmax", "ymax"))
            gts.append(GroundTruth(box, name, obj.findtext("difficult", "0").strip() == "1"))
        yield i, root / "JPEGImages" / f"{i}.jpg", gts


def load_yolo(images: Path, labels: Path, names: list[str], limit: int | None):
    files = sorted(p for p in images.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"})
    if limit:
        files = files[:limit]
    for f in files:
        lab = labels / f.relative_to(images).with_suffix(".txt")
        with Image.open(f) as im:
            w, h = ImageOps.exif_transpose(im).size
        gts = []
        if lab.is_file():
            for line in lab.read_text().splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                c, cx, cy, bw, bh = int(parts[0]), *map(float, parts[1:5])
                gts.append(GroundTruth(((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h), names[c]))
        yield str(f.relative_to(images)), f, gts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", help="detector profile from config/detector.yaml (default: active)")
    ap.add_argument("--config", default=None, help="detector config (default config/detector.yaml)")
    ap.add_argument("--voc", type=Path, help="VOC year folder containing Annotations/ JPEGImages/ ImageSets/")
    ap.add_argument("--split", default="test")
    ap.add_argument("--images", type=Path)
    ap.add_argument("--labels", type=Path)
    ap.add_argument("--dataset-classes", type=Path, help="classes.txt (one name per line, index order)")
    ap.add_argument("--class-map", type=Path, help="YAML: dataset class -> detector class name or list")
    ap.add_argument("--operating-conf", type=float, default=None, help="default: conf_threshold in detector.yaml")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None, help="evaluate only the first N images (smoke test)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    cfg = load_detector_config(args.config) if args.config else load_detector_config()
    profile = cfg.get(args.profile)
    if args.threads and profile.runtime == "executorch":
        from zyqora.model_runtime import import_portable_lib
        import_portable_lib()._unsafe_reset_threadpool(args.threads)
    eval_settings = DetectorSettings(conf_threshold=0.001, iou_threshold=0.7, agnostic_nms=False,
                                     max_detections=300, max_objects_classified=300, min_box_area_fraction=0.0)
    det = YoloDetector(profile, eval_settings, num_threads=args.threads)
    op_conf = args.operating_conf if args.operating_conf is not None else cfg.settings.conf_threshold

    if args.voc:
        ds_name = f"{args.voc.name}_{args.split}"
        class_map = {k: [v] for k, v in VOC_TO_COCO.items()}
        records_src = load_voc(args.voc, args.split, args.limit)
    elif args.images and args.labels and args.dataset_classes:
        ds_name = args.images.parent.name or "yolo"
        names = [ln.strip() for ln in args.dataset_classes.read_text().splitlines() if ln.strip()]
        if args.class_map:
            raw = yaml.safe_load(args.class_map.read_text()) or {}
            class_map = {k: ([v] if isinstance(v, str) else list(v)) for k, v in raw.items()}
        else:
            class_map = {n: [n] for n in names}
        records_src = load_yolo(args.images, args.labels, names, args.limit)
    else:
        ap.error("give --voc, or --images + --labels + --dataset-classes")

    unknown = sorted({c for v in class_map.values() for c in v} - set(profile.classes))
    if unknown:
        ap.error(f"class map names not in detector classes: {unknown}")
    inv = {det_c: ds_c for ds_c, dets in class_map.items() for det_c in dets}
    eval_classes = sorted(class_map)

    records, lat = [], []
    t0 = time.perf_counter()
    for n, (image_id, path, gts) in enumerate(records_src, 1):
        with Image.open(path) as im:
            img = ImageOps.exif_transpose(im).convert("RGB")
        out = det.detect(img)
        lat.append(out.inference_ms)
        preds = [Prediction(d.box, inv[d.label], d.confidence) for d in out.detections if d.label in inv]
        records.append(ImageRecord(image_id, [g for g in gts if g.label in class_map], preds))
        if n % 500 == 0:
            print(f"  {n} images ...", file=sys.stderr)
    rep = evaluate(records, eval_classes, operating_conf=op_conf)
    result = {
        "scope": "held-out detection evaluation",
        "dataset": ds_name, "dataset_format": "voc" if args.voc else "yolo",
        "profile": profile.name, "file": profile.file.name, "runtime": profile.runtime, "imgsz": profile.imgsz,
        "weights": profile.weights, "trained_on": profile.trained_on,
        "class_map": class_map, "protocol": "pycocotools 101-point interpolated AP; conf>=0.001, max 300 boxes, per-class NMS IoU 0.7; "
                                            "VOC 'difficult' boxes ignored",
        "inference_ms_median": float(np.median(lat)) if lat else None,
        "wall_seconds": time.perf_counter() - t0,
        "platform": platform_facts(),
        **rep.as_dict(),
    }
    out = args.out or OUTPUTS_DIR / "detector_eval" / f"{profile.name}_{ds_name}{'_limit' + str(args.limit) if args.limit else ''}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"{profile.name} on {ds_name} ({rep.images} images): mAP@0.5 {rep.map50:.4f}  mAP@0.5:0.95 {rep.map50_95:.4f}  "
          f"P {rep.precision:.3f} R {rep.recall:.3f} @conf {op_conf}")
    for c in rep.classes:
        if c.label in ("bottle", "tvmonitor") or not args.voc:
            f = lambda v: "n/a" if v is None else f"{v:.4f}"
            print(f"  {c.label:12s} n_gt {c.n_gt:5d}  AP50 {f(c.ap50)}  AP50-95 {f(c.ap50_95)}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
