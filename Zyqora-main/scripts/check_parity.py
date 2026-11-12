#!/usr/bin/env python3
"""Gate 3: model parity checks on identical inputs.

Mode 1 (FP32 gate):  PyTorch reference vs ExecuTorch FP32 .pte
    python scripts/check_parity.py --images data/dev
    Gate: top-1 must agree on EVERY image. Also checks that Zyqora's NumPy
    preprocessing equals the TorchVision transform used by the Arm course.

Mode 2 (INT8 report): ExecuTorch FP32 .pte vs ExecuTorch INT8 .pte
    python scripts/check_parity.py --images data/dev \
        --pte models/mobilenetv2_int8_xnnpack.pte --compare-pte models/mobilenetv2_fp32_xnnpack.pte
    Reports top-1 disagreements; exact equivalence is NOT expected.

Mode 1 needs torch + torchvision (export venv, or the Pi course venv).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

import _bootstrap

from zyqora.evaluator import discover_images
from zyqora.imagenet_labels import load_labels
from zyqora.model_runtime import ExecuTorchClassifier, softmax
from zyqora.paths import DEFAULT_MODEL_CONTRACT, MODELS_DIR, OUTPUTS_DIR
from zyqora.preprocessing import ImageLoadError, PreprocessSpec, load_rgb, image_to_array

FP32_LOGIT_TOLERANCE = 1e-3  # documented tolerance for FP32 XNNPACK vs PyTorch eager


def torchvision_transform(spec: PreprocessSpec):
    import torchvision.transforms as T

    return T.Compose([T.Resize(spec.resize), T.CenterCrop(spec.center_crop), T.ToTensor(),
                      T.Normalize(mean=spec.mean, std=spec.std)])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--pte", type=Path, default=MODELS_DIR / "mobilenetv2_fp32_xnnpack.pte")
    ap.add_argument("--compare-pte", type=Path, default=None, help="second .pte (INT8 report mode)")
    ap.add_argument("--weights", type=Path, default=MODELS_DIR / "mobilenet_v2.pth")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, default=OUTPUTS_DIR / "parity")
    args = ap.parse_args()
    _bootstrap.quiet_toolchain_logs()

    labels = load_labels()
    spec = PreprocessSpec.from_file(DEFAULT_MODEL_CONTRACT)
    images = discover_images(args.images)[: args.limit]
    if not images:
        print(f"No supported images in {args.images}", file=sys.stderr)
        return 2

    candidate = ExecuTorchClassifier(args.pte, input_shape=spec.input_shape)
    int8_mode = args.compare_pte is not None
    if int8_mode:
        reference = ExecuTorchClassifier(args.compare_pte, input_shape=spec.input_shape)
        ref_name = f"ExecuTorch:{args.compare_pte.name}"
        tv = None
    else:
        import torch

        from export_common import build_reference_model

        torch_model = build_reference_model(args.weights)
        tv = torchvision_transform(spec)
        ref_name = "PyTorch eager (TorchVision MobileNetV2 IMAGENET1K_V1)"

    rows, preprocess_max_diff = [], 0.0
    for rel, path in images:
        try:
            img = load_rgb(path)
        except ImageLoadError as exc:
            print(f"SKIP (unreadable) {rel}: {exc}")
            continue
        x = image_to_array(img, spec)
        if int8_mode:
            ref_logits = reference.infer(x).logits
        else:
            x_tv = tv(img).unsqueeze(0).numpy()
            preprocess_max_diff = max(preprocess_max_diff, float(np.max(np.abs(x_tv - x))))
            with torch.inference_mode():
                ref_logits = torch_model(torch.from_numpy(x))[0].numpy()
        cand_logits = candidate.infer(x).logits
        r1, c1 = int(np.argmax(ref_logits)), int(np.argmax(cand_logits))
        rp, cp = softmax(ref_logits), softmax(cand_logits)
        rows.append({
            "image": rel,
            "reference_top1": labels[r1], "candidate_top1": labels[c1],
            "reference_top1_prob": round(float(rp[r1]), 6), "candidate_top1_prob": round(float(cp[c1]), 6),
            "top1_agree": r1 == c1,
            "max_abs_logit_diff": float(np.max(np.abs(ref_logits - cand_logits))),
            "max_abs_prob_diff": float(np.max(np.abs(rp - cp))),
        })

    if not rows:
        print("No readable images; parity not evaluated.", file=sys.stderr)
        return 2
    n_agree = sum(r["top1_agree"] for r in rows)
    max_logit = max(r["max_abs_logit_diff"] for r in rows)
    summary = {
        "mode": "INT8_vs_FP32_report" if int8_mode else "FP32_gate",
        "reference": ref_name,
        "candidate": f"ExecuTorch:{args.pte.name}",
        "images": len(rows),
        "top1_agreement": n_agree,
        "top1_agreement_rate": n_agree / len(rows),
        "max_abs_logit_diff": max_logit,
        "mean_max_abs_logit_diff": float(np.mean([r["max_abs_logit_diff"] for r in rows])),
        "fp32_logit_tolerance": None if int8_mode else FP32_LOGIT_TOLERANCE,
        "numpy_vs_torchvision_preprocess_max_abs_diff": None if int8_mode else preprocess_max_diff,
        "disagreements": [r for r in rows if not r["top1_agree"]],
    }
    if int8_mode:
        summary["gate_passed"] = None  # report only
    else:
        summary["gate_passed"] = n_agree == len(rows) and max_logit <= FP32_LOGIT_TOLERANCE and preprocess_max_diff <= 1e-6

    args.out.mkdir(parents=True, exist_ok=True)
    stem = "int8_vs_fp32" if int8_mode else "fp32_vs_pytorch"
    with (args.out / f"{stem}.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (args.out / f"{stem}.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({k: v for k, v in summary.items() if k != "disagreements"}, indent=2))
    for r in summary["disagreements"]:
        print(f"  DISAGREE {r['image']}: {r['reference_top1']} vs {r['candidate_top1']}")
    print(f"Wrote {args.out / (stem + '.csv')} and .json")
    if int8_mode:
        return 0
    return 0 if summary["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
