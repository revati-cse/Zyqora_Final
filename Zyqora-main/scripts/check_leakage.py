#!/usr/bin/env python3
"""Check for data leakage between dataset splits BEFORE evaluating on TEST.

* exact duplicates: identical file bytes (SHA-256)
* near-duplicates:  64-bit difference hash (dHash, 9x8 grayscale) with Hamming distance
                    <= --near-threshold (default 4). Catches re-encoded/resized copies;
                    it is a screening tool, so review the reported pairs by eye.

    python scripts/check_leakage.py --split train=/data/TRAIN --split val=/data/VAL --split test=/data/TEST

Writes <out>/leakage_report.json, <out>/leakage_pairs.csv and
<out>/test_overlap_files.txt (TEST images duplicated in another split; pass it to
run_zyqora.py --exclude-list for a second, leakage-free evaluation).
Exit code 1 if any cross-split exact or near duplicate is found.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

import _bootstrap  # noqa: F401

from zyqora.evaluator import discover_images
from zyqora.paths import OUTPUTS_DIR

POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def dhash(path: Path) -> int:
    with Image.open(path) as im:
        g = np.asarray(im.convert("L").resize((9, 8), Image.BILINEAR), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int(np.packbits(bits).view(">u8")[0])


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hamming(a: np.ndarray, b: int) -> np.ndarray:
    x = np.bitwise_xor(a, np.uint64(b)).view(np.uint8).reshape(-1, 8)
    return POPCOUNT[x].sum(axis=1)


def index_split(name: str, folder: Path) -> dict:
    files, shas, hashes, unreadable = [], [], [], []
    for rel, p in discover_images(folder):
        try:
            hashes.append(dhash(p))
        except (UnidentifiedImageError, OSError) as exc:
            unreadable.append(f"{rel}: {exc}")
            continue
        files.append(rel)
        shas.append(sha256(p))
    print(f"{name}: {len(files)} images indexed ({len(unreadable)} unreadable) from {folder}")
    return {"files": files, "sha": shas, "dhash": np.array(hashes, dtype=np.uint64), "unreadable": unreadable}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", action="append", required=True, metavar="NAME=DIR",
                    help="dataset split, e.g. train=DIR (repeat; one should be named test)")
    ap.add_argument("--near-threshold", type=int, default=4, help="max dHash Hamming distance for near-duplicates")
    ap.add_argument("--out", type=Path, default=OUTPUTS_DIR / "leakage")
    args = ap.parse_args()

    splits = {}
    for item in args.split:
        name, _, folder = item.partition("=")
        if not name or not folder:
            ap.error(f"--split must be NAME=DIR, got {item!r}")
        splits[name] = index_split(name, Path(folder))
    names = list(splits)

    pairs, summary = [], {}
    for i, a in enumerate(names):
        # within-split exact duplicates (informational)
        seen: dict[str, str] = {}
        dup_within = 0
        for f, h in zip(splits[a]["files"], splits[a]["sha"]):
            if h in seen:
                dup_within += 1
            seen.setdefault(h, f)
        summary[f"{a}_within_split_exact_duplicates"] = dup_within
        for b in names[i + 1:]:
            sha_b = {}
            for f, h in zip(splits[b]["files"], splits[b]["sha"]):
                sha_b.setdefault(h, f)
            exact = near = 0
            hb = splits[b]["dhash"]
            for f, h, dh in zip(splits[a]["files"], splits[a]["sha"], splits[a]["dhash"]):
                if h in sha_b:
                    exact += 1
                    pairs.append({"split_a": a, "file_a": f, "split_b": b, "file_b": sha_b[h], "kind": "exact", "distance": 0})
                    continue
                if len(hb):
                    d = hamming(hb, int(dh))
                    j = int(np.argmin(d))
                    if d[j] <= args.near_threshold:
                        near += 1
                        pairs.append({"split_a": a, "file_a": f, "split_b": b, "file_b": splits[b]["files"][j],
                                      "kind": "near", "distance": int(d[j])})
            summary[f"{a}_vs_{b}"] = {"exact_duplicates": exact, "near_duplicates": near}
            print(f"{a} vs {b}: {exact} exact, {near} near-duplicate (dHash <= {args.near_threshold})")

    test_name = next((n for n in names if n.lower() == "test"), None)
    test_overlap = sorted({p["file_a"] if p["split_a"] == test_name else p["file_b"]
                           for p in pairs if test_name in (p["split_a"], p["split_b"])})
    args.out.mkdir(parents=True, exist_ok=True)
    report = {"splits": {n: {"folder": item.partition("=")[2], "images": len(splits[n]["files"]),
                             "unreadable": splits[n]["unreadable"]} for n, item in zip(names, args.split)},
              "near_threshold": args.near_threshold, "summary": summary,
              "test_images_with_any_cross_split_duplicate": len(test_overlap) if test_name else None}
    (args.out / "leakage_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (args.out / "leakage_pairs.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["split_a", "file_a", "split_b", "file_b", "kind", "distance"])
        w.writeheader()
        w.writerows(pairs)
    if test_name:
        (args.out / "test_overlap_files.txt").write_text("".join(f"{x}\n" for x in test_overlap), encoding="utf-8")
        print(f"TEST images duplicated in another split: {len(test_overlap)} -> {args.out / 'test_overlap_files.txt'}")
    print(f"Wrote {args.out / 'leakage_report.json'} and leakage_pairs.csv")
    return 1 if pairs else 0


if __name__ == "__main__":
    sys.exit(main())
