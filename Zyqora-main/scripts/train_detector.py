#!/usr/bin/env python3
"""Fine-tune a YOLO detector on a box-annotated waste dataset (e.g. TACO from prepare_taco.py).

NOT run for the submitted build: it needs the dataset images, a GPU for practical training
time, and the team's explicit approval (project rule: no retraining without approval).
--confirm is required so it never starts by accident.

    pip install ultralytics                       # AGPL-3.0, training machine only
    python scripts/prepare_taco.py --taco /path/TACO/data --out data/taco_yolo
    python scripts/train_detector.py --data data/taco_yolo/data.yaml --base yolo11n.pt --epochs 100 --confirm
    # held-out evaluation on the TEST split (never used for training or model selection):
    python scripts/export_detector.py --weights runs/detect/zyqora_taco/weights/best.pt --name taco_yolo11n --imgsz 640
    python scripts/evaluate_detector.py --profile taco_yolo11n_640 --images data/taco_yolo/test/images \
        --labels data/taco_yolo/test/labels --dataset-classes data/taco_yolo/classes.txt

Then add the profile to config/detector.yaml and a material table for the new classes
(config/detector_materials_taco.yaml is a starting point for TACO supercategories).
Ultralytics' own validation uses the VAL split for checkpoint selection; report TEST-split
numbers from evaluate_detector.py only.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True, help="data.yaml written by prepare_taco.py")
    ap.add_argument("--base", default="yolo11n.pt", help="pretrained starting weights (COCO)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None, help="0 for the first GPU, cpu for CPU (very slow)")
    ap.add_argument("--name", default="zyqora_taco")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--confirm", action="store_true", help="required: confirms training was approved")
    args = ap.parse_args()
    if not args.confirm:
        print("Refusing to train without --confirm (training must be approved by the team).")
        return 2
    if not args.data.is_file():
        raise SystemExit(f"{args.data} not found; run scripts/prepare_taco.py first")
    from ultralytics import YOLO

    model = YOLO(args.base)
    model.train(data=str(args.data), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, device=args.device,
                name=args.name, seed=args.seed, deterministic=True, patience=30, plots=True)
    print(f"best checkpoint: runs/detect/{args.name}/weights/best.pt  (select on VAL; report TEST separately)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
