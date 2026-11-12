#!/usr/bin/env python3
"""Gate 1a: prepare TorchVision MobileNetV2 IMAGENET1K_V1 weights + labels.

Run on the development machine (or the Pi, inside the Arm course venv) with
internet access. Afterwards everything needed is local in models/.

    python scripts/prepare_model.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import _bootstrap

from export_common import load_contract, sha256_file, toolchain_versions
from zyqora.imagenet_labels import load_labels
from zyqora.paths import MODELS_DIR


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=MODELS_DIR, help="output directory (default: models/)")
    args = ap.parse_args()
    _bootstrap.quiet_toolchain_logs()

    import torch
    from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

    contract = load_contract()
    weights = MobileNet_V2_Weights.IMAGENET1K_V1
    print(f"Loading {weights} from {weights.url} (cached by torch.hub; hash-checked)...")
    # check_hash=True makes torch.hub verify the SHA-256 prefix embedded in the file name.
    state = torch.hub.load_state_dict_from_url(weights.url, progress=False, check_hash=True, map_location="cpu")
    model = mobilenet_v2(weights=None)
    model.load_state_dict(state)
    model.eval()

    labels = weights.meta["categories"]
    bundled = load_labels()
    if labels != bundled:
        print("ERROR: bundled src/zyqora/data/imagenet_labels.json differs from TorchVision metadata", file=sys.stderr)
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    weights_path = args.out / "mobilenet_v2.pth"
    torch.save(model.state_dict(), weights_path)
    labels_path = args.out / "imagenet_labels.json"
    labels_path.write_text(json.dumps(labels, indent=1) + "\n", encoding="utf-8")

    spec = dict(contract)
    spec.update(
        {
            "weights": weights_path.name,
            "labels": labels_path.name,
            "weights_sha256": sha256_file(weights_path),
            "weights_url": weights.url,
            "published_imagenet_acc1": weights.meta.get("_metrics", {}).get("ImageNet-1K", {}).get("acc@1"),
            "num_params": weights.meta.get("num_params"),
            "toolchain": toolchain_versions(),
        }
    )
    (args.out / "model_spec.json").write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {weights_path} ({weights_path.stat().st_size / 1e6:.2f} MB)")
    print(f"Wrote {labels_path} (identical to bundled labels: yes)")
    print(f"Wrote {args.out / 'model_spec.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
