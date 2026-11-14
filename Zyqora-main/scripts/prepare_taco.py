#!/usr/bin/env python3
"""Convert TACO (Trash Annotations in Context, http://tacodataset.org, MIT licence) to YOLO format.

TACO has 1500 photos of litter with 4784 box/polygon annotations in 60 categories grouped into
28 supercategories (Bottle, Can, Carton, Cup, Paper, Plastic bag & wrapper, Food waste, ...).

    # 1. get TACO (on a machine that can reach tacodataset.org):
    #    git clone https://github.com/pedropro/TACO && cd TACO && python download.py
    # 2. convert, split by image (seeded, no image in two splits):
    python scripts/prepare_taco.py --taco /path/TACO/data --out data/taco_yolo
    # 3. check labels without images (e.g. before downloading):
    python scripts/prepare_taco.py --annotations annotations.json --out /tmp/taco_check --labels-only

Writes <out>/{train,val,test}/{images,labels}/, <out>/classes.txt and <out>/data.yaml
(for scripts/train_detector.py and scripts/evaluate_detector.py --images/--labels).
Images are copied, not modified. Crowd annotations are skipped. Boxes are clipped to the image.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path


def split_of(file_name: str, seed: str, val: float, test: float) -> str:
    h = int(hashlib.sha256(f"{seed}:{file_name}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "test" if h < test else "val" if h < test + val else "train"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--taco", type=Path, help="TACO data/ folder (annotations.json + batch_*/ images)")
    ap.add_argument("--annotations", type=Path, help="annotations.json (default <taco>/annotations.json)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--level", choices=["supercategory", "category"], default="supercategory",
                    help="supercategory = 28 classes (more samples each); category = 60 fine classes")
    ap.add_argument("--val", type=float, default=0.1)
    ap.add_argument("--test", type=float, default=0.1)
    ap.add_argument("--seed", default="zyqora")
    ap.add_argument("--labels-only", action="store_true", help="write labels/data.yaml only (no images needed)")
    args = ap.parse_args()
    ann_path = args.annotations or (args.taco / "annotations.json" if args.taco else None)
    if not ann_path or not ann_path.is_file():
        ap.error("give --taco <TACO/data> or --annotations <annotations.json>")
    data = json.loads(ann_path.read_text(encoding="utf-8"))

    cats = {c["id"]: c[args.level] for c in data["categories"]}
    names = sorted(set(cats.values()))
    index = {n: i for i, n in enumerate(names)}
    images = {im["id"]: im for im in data["images"]}
    per_image: dict[int, list[str]] = {i: [] for i in images}
    skipped = Counter()
    for a in data["annotations"]:
        if a.get("iscrowd"):
            skipped["crowd"] += 1
            continue
        im = images[a["image_id"]]
        W, H = im["width"], im["height"]
        x, y, w, h = a["bbox"]
        x1, y1, x2, y2 = max(0.0, x), max(0.0, y), min(W, x + w), min(H, y + h)
        if x2 - x1 < 1 or y2 - y1 < 1:
            skipped["degenerate"] += 1
            continue
        c = index[cats[a["category_id"]]]
        per_image[a["image_id"]].append(
            f"{c} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")

    counts = {s: Counter() for s in ("train", "val", "test")}
    n_images = Counter()
    missing = 0
    for iid, lines in per_image.items():
        im = images[iid]
        split = split_of(im["file_name"], args.seed, args.val, args.test)
        stem = im["file_name"].replace("/", "_").rsplit(".", 1)[0]
        lab_dir = args.out / split / "labels"
        lab_dir.mkdir(parents=True, exist_ok=True)
        (lab_dir / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        n_images[split] += 1
        for ln in lines:
            counts[split][names[int(ln.split()[0])]] += 1
        if not args.labels_only:
            src = args.taco / im["file_name"]
            if not src.is_file():
                missing += 1
                continue
            img_dir = args.out / split / "images"
            img_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, img_dir / f"{stem}{src.suffix.lower()}")

    (args.out / "classes.txt").write_text("\n".join(names) + "\n")
    root = args.out.resolve()
    (args.out / "data.yaml").write_text(
        f"# TACO ({args.level}) converted by scripts/prepare_taco.py; split seed '{args.seed}'\n"
        f"path: {root.as_posix()}\ntrain: train/images\nval: val/images\ntest: test/images\n"
        f"names:\n" + "".join(f"  {i}: {json.dumps(n)}\n" for i, n in enumerate(names)))
    report = {"source": str(ann_path), "level": args.level, "classes": len(names), "images": dict(n_images),
              "boxes": {s: sum(c.values()) for s, c in counts.items()}, "skipped": dict(skipped),
              "missing_images": missing if not args.labels_only else "not checked (--labels-only)",
              "per_class_train": dict(counts["train"].most_common())}
    (args.out / "prepare_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "per_class_train"}, indent=2))
    rare = [n for n in names if counts["train"][n] < 20]
    if rare:
        print(f"WARNING: {len(rare)} classes have < 20 training boxes: {rare}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
