#!/usr/bin/env python3
"""Gate 1b: export MobileNetV2 FP32 to an ExecuTorch .pte with XNNPACK delegation.

Sequence (Arm course Module 2, lower_mobilenetv2_xnnpack.py):
    torch.export.export
      -> to_edge_transform_and_lower(..., partitioner=[XnnpackPartitioner()])
      -> .to_executorch()
      -> .pte

Writes models/mobilenetv2_fp32_xnnpack.pte and a .json metadata file, then
loads the .pte through the ExecuTorch runtime as an integrity check.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap

from export_common import (
    build_reference_model,
    inspect_delegation,
    load_contract,
    sha256_file,
    smoke_test_pte,
    toolchain_versions,
    write_pte,
)
from zyqora.paths import MODELS_DIR


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", type=Path, default=MODELS_DIR / "mobilenet_v2.pth")
    ap.add_argument("--out", type=Path, default=MODELS_DIR / "mobilenetv2_fp32_xnnpack.pte")
    args = ap.parse_args()
    _bootstrap.quiet_toolchain_logs()

    import torch
    from torch.export import export
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    from executorch.exir import to_edge_transform_and_lower

    contract = load_contract()
    model = build_reference_model(args.weights)
    example = (torch.randn(*contract["input"]["shape"]),)

    print("torch.export ...")
    with torch.inference_mode():
        exported = export(model, example)
    print("to_edge_transform_and_lower(XnnpackPartitioner) ...")
    edge = to_edge_transform_and_lower(exported, partitioner=[XnnpackPartitioner()])
    delegation = inspect_delegation(edge)
    print("to_executorch ...")
    program = edge.to_executorch()
    write_pte(program, args.out)

    print("Integrity check: loading .pte through ExecuTorch runtime ...")
    smoke = smoke_test_pte(args.out, model, contract)

    meta = {
        "model": contract["name"],
        "weights_variant": contract["model_source"]["weights_variant"],
        "backend": "xnnpack",
        "precision": "fp32",
        "pte": args.out.name,
        "pte_size_bytes": args.out.stat().st_size,
        "pte_size_mb": args.out.stat().st_size / 1e6,
        "pte_sha256": sha256_file(args.out),
        "input_shape": contract["input"]["shape"],
        "delegation": delegation,
        "integrity_check": smoke,
        "exported_with": toolchain_versions(),
    }
    meta_path = args.out.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {args.out} ({meta['pte_size_mb']:.2f} MB)")
    print(f"Wrote {meta_path}")
    print(f"XNNPACK delegate call sites: {delegation['delegate_call_count']}")
    print(f"Non-delegated call_function ops: {delegation['fallback_call_function_count']} {delegation['fallback_ops']}")
    print(f"Integrity: {smoke}")
    ok = smoke["output_shape_ok"] and smoke["random_input_top1_agree"]
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
