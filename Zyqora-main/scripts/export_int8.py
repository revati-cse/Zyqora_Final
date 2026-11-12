#!/usr/bin/env python3
"""Gate 9: post-training INT8 quantization (PT2E + XNNPACKQuantizer) -> ExecuTorch .pte.

Follows the Arm course Module 2 quantize_mobilenetv2_xnnpack.py:
    export -> prepare_pt2e(XNNPACKQuantizer, symmetric per-channel) -> calibrate
    -> convert_pt2e -> export -> to_edge_transform_and_lower(XnnpackPartitioner) -> .pte
The classifier head (classifier.1) stays FP32 by default, as in the course.

Calibration images must be SEPARATE from the final test images. Pass the test
folder(s) with --exclude to have the script refuse any overlap (by SHA-256).

    python scripts/export_int8.py --calibration path/to/calibration_images --exclude data/organiser
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import _bootstrap

from export_common import (
    build_reference_model,
    inspect_delegation,
    load_contract,
    sha256_file,
    toolchain_versions,
    write_pte,
)
from zyqora.evaluator import discover_images
from zyqora.paths import MODELS_DIR
from zyqora.preprocessing import ImageLoadError, PreprocessSpec, preprocess_path


def skip_classifier_head(node) -> bool:
    """Course filter: return False (do not quantize) for nodes inside classifier.1."""
    for fqn, _typ in node.meta.get("nn_module_stack", {}).values():
        if "classifier.1" in str(fqn):
            return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calibration", type=Path, required=True, help="folder of calibration images (recursive)")
    ap.add_argument("--exclude", type=Path, nargs="*", default=[], help="test folders that must not overlap")
    ap.add_argument("--num-calib", type=int, default=100)
    ap.add_argument("--quantize-classifier-head", action="store_true")
    ap.add_argument("--weights", type=Path, default=MODELS_DIR / "mobilenet_v2.pth")
    ap.add_argument("--out", type=Path, default=MODELS_DIR / "mobilenetv2_int8_xnnpack.pte")
    args = ap.parse_args()
    _bootstrap.quiet_toolchain_logs()

    calib = discover_images(args.calibration)[: args.num_calib]
    if not calib:
        print(f"No calibration images in {args.calibration}", file=sys.stderr)
        return 2
    excluded = {sha256_file(p) for folder in args.exclude for _, p in discover_images(folder)}
    overlap = [rel for rel, p in calib if sha256_file(p) in excluded]
    if overlap:
        print(f"ERROR: {len(overlap)} calibration image(s) also appear in the test set: {overlap[:5]}", file=sys.stderr)
        return 2

    import torch
    from torch.export import export
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
        XNNPACKQuantizer,
        get_symmetric_quantization_config,
    )
    from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
    from torchao.quantization.pt2e import move_exported_model_to_eval
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

    contract = load_contract()
    spec = PreprocessSpec.from_contract(contract)
    model = build_reference_model(args.weights)
    example = (torch.randn(*contract["input"]["shape"]),)

    quantizer = XNNPACKQuantizer()
    quantizer.set_global(get_symmetric_quantization_config(is_per_channel=True))
    if not args.quantize_classifier_head:
        quantizer.set_filter_function(skip_classifier_head)

    exported = export(model, example)
    prepared = prepare_pt2e(exported.module(), quantizer)
    move_exported_model_to_eval(prepared)

    used = 0
    print(f"Calibrating on {len(calib)} images from {args.calibration} (same preprocessing as deployment)...")
    with torch.inference_mode():
        for rel, path in calib:
            try:
                x = preprocess_path(path, spec)
            except ImageLoadError as exc:
                print(f"  skip {rel}: {exc}")
                continue
            prepared(torch.from_numpy(x))
            used += 1

    quantized = convert_pt2e(prepared)
    exported_q = export(quantized, example)
    edge = to_edge_transform_and_lower(
        exported_q,
        partitioner=[XnnpackPartitioner()],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    )
    delegation = inspect_delegation(edge)
    write_pte(edge.to_executorch(), args.out)

    from zyqora.model_runtime import ExecuTorchClassifier

    clf = ExecuTorchClassifier(args.out)  # integrity check through the runtime
    logits = clf.infer(example[0].numpy()).logits

    fp32 = MODELS_DIR / "mobilenetv2_fp32_xnnpack.pte"
    meta = {
        "model": contract["name"],
        "backend": "xnnpack",
        "precision": "int8",
        "quantization": "PT2E static, XNNPACKQuantizer symmetric per-channel"
                        + ("" if args.quantize_classifier_head else "; classifier.1 kept FP32"),
        "pte": args.out.name,
        "pte_size_bytes": args.out.stat().st_size,
        "pte_size_mb": args.out.stat().st_size / 1e6,
        "fp32_pte_size_mb": fp32.stat().st_size / 1e6 if fp32.exists() else None,
        "pte_sha256": sha256_file(args.out),
        "calibration_folder": str(args.calibration),
        "calibration_images_used": used,
        "excluded_test_folders": [str(p) for p in args.exclude],
        "delegation": delegation,
        "integrity_check": {"runtime_load_ok": True, "output_shape_ok": logits.shape == (1000,)},
        "exported_with": toolchain_versions(),
    }
    args.out.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.out} ({meta['pte_size_mb']:.2f} MB; FP32 {meta['fp32_pte_size_mb']} MB)")
    print(f"XNNPACK delegate call sites: {delegation['delegate_call_count']}, "
          f"non-delegated ops: {delegation['fallback_call_function_count']} {delegation['fallback_ops'][:10]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
