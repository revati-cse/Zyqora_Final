#!/usr/bin/env python3
"""Export a YOLO detector to the files Zyqora runs (development machine, NOT the Pi).

    pip install -r requirements-export.txt ultralytics onnx onnxslim
    python scripts/export_detector.py --weights yolo11n.pt --imgsz 640 320
    python scripts/export_detector.py --weights runs/detect/taco/weights/best.pt --imgsz 640 --name taco_yolo11n

Writes, per input size, without NMS (Zyqora decodes and runs NMS itself):
  models/<name>_<imgsz>_xnnpack.pte   ExecuTorch, lowered to XNNPACK  (runtime: executorch)
  models/<name>_<imgsz>.onnx          ONNX opset 17                  (runtime: onnxruntime)
  config/detector_<name>.json         class names in output order (from the weights)
and prints the SHA-256 lines to paste into config/detector.yaml.

Ultralytics is AGPL-3.0; it is only needed here, never at runtime. The pretrained
yolo11n.pt is downloaded by Ultralytics from its GitHub release on first use.
Verify each export against Ultralytics with scripts/check_detector_parity.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import _bootstrap  # noqa: F401

_bootstrap.quiet_toolchain_logs()

from zyqora.paths import CONFIG_DIR, MODELS_DIR  # noqa: E402


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def export_pte(weights: str, imgsz: int, out: Path) -> None:
    import torch
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    from executorch.exir import to_edge_transform_and_lower
    from ultralytics import YOLO

    net = YOLO(weights).model.eval().float()
    for m in net.modules():           # head returns the raw [1, 4+nc, N] tensor, no NMS
        if hasattr(m, "export"):
            m.export = True
        if hasattr(m, "format"):
            m.format = "executorch"
    net = net.fuse()
    x = torch.zeros(1, 3, imgsz, imgsz)
    with torch.no_grad():
        ep = torch.export.export(net, (x,))
    prog = to_edge_transform_and_lower(ep, partitioner=[XnnpackPartitioner()]).to_executorch()
    out.write_bytes(prog.buffer)


def export_onnx(weights: str, imgsz: int, out: Path) -> None:
    from ultralytics import YOLO

    p = YOLO(weights).export(format="onnx", imgsz=imgsz, opset=17, simplify=True, dynamic=False, nms=False)
    shutil.move(p, out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default="yolo11n.pt")
    ap.add_argument("--imgsz", nargs="+", type=int, default=[640, 320])
    ap.add_argument("--name", default=None, help="output name prefix (default: weights stem)")
    ap.add_argument("--formats", nargs="+", default=["pte", "onnx"], choices=["pte", "onnx"])
    args = ap.parse_args()
    from ultralytics import YOLO

    name = args.name or Path(args.weights).stem
    names = YOLO(args.weights).names
    classes = [names[i] for i in range(len(names))]
    cfile = CONFIG_DIR / f"detector_{name}.json"
    cfile.write_text(json.dumps({"schema_version": 1, "note": f"class names of {Path(args.weights).name} in output order",
                                 "classes": classes}, indent=1))
    print(f"classes ({len(classes)}): {cfile}")
    MODELS_DIR.mkdir(exist_ok=True)
    for sz in args.imgsz:
        if sz % 32:
            raise SystemExit(f"imgsz must be a multiple of 32, got {sz}")
        if "pte" in args.formats:
            out = MODELS_DIR / f"{name}_{sz}_xnnpack.pte"
            export_pte(args.weights, sz, out)
            print(f"{out.name}: {out.stat().st_size} bytes  sha256: {sha(out)}")
        if "onnx" in args.formats:
            out = MODELS_DIR / f"{name}_{sz}.onnx"
            export_onnx(args.weights, sz, out)
            print(f"{out.name}: {out.stat().st_size} bytes  sha256: {sha(out)}")
    print("Add the profiles to config/detector.yaml (file, runtime, imgsz, classes, sha256).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
