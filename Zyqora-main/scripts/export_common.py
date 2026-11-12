"""Helpers shared by the export scripts (development machine, export venv only).

Mirrors the Arm course Module 2 wrappers (lower_mobilenetv2_xnnpack.py,
quantize_mobilenetv2_xnnpack.py) at commit 1045d51.
"""

from __future__ import annotations

import hashlib
import json
import platform
from collections import Counter
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401

from zyqora.paths import DEFAULT_MODEL_CONTRACT, MODELS_DIR


def load_contract(path: Path = DEFAULT_MODEL_CONTRACT) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_reference_model(weights_path: Path):
    """Rebuild TorchVision MobileNetV2 from the prepared state dict (no download)."""
    import torch
    from torchvision import models

    if not weights_path.is_file():
        raise SystemExit(f"Weights not found: {weights_path}. Run scripts/prepare_model.py first.")
    model = models.mobilenet_v2(weights=None)
    model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    return model.eval()


def inspect_delegation(edge: Any) -> dict[str, Any]:
    """Count XNNPACK delegate call sites and operators left on the portable path."""
    import torch

    graph = edge.exported_program().graph_module.graph
    delegate_op = torch.ops.higher_order.executorch_call_delegate
    delegate_calls = 0
    fallback: list[str] = []
    for node in graph.nodes:
        if node.op != "call_function":
            continue
        if node.target == delegate_op:
            delegate_calls += 1
        else:
            fallback.append(str(node.target))
    counts = Counter(fallback)
    return {
        "delegate_call_count": delegate_calls,
        "fallback_call_function_count": len(fallback),
        "fallback_ops": [f"{name} ({n})" for name, n in sorted(counts.items())],
    }


def write_pte(program: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        if hasattr(program, "write_to_file"):
            program.write_to_file(f)
        else:
            f.write(getattr(program, "buffer", None) or getattr(program, "_buffer"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pkg_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not installed"


def toolchain_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": pkg_version("torch"),
        "torchvision": pkg_version("torchvision"),
        "torchao": pkg_version("torchao"),
        "executorch": pkg_version("executorch"),
    }


def smoke_test_pte(pte: Path, reference_model, contract: dict, *, seed: int = 0) -> dict[str, Any]:
    """Load the written .pte through the ExecuTorch runtime and compare with PyTorch."""
    import numpy as np
    import torch

    from zyqora.model_runtime import ExecuTorchClassifier

    torch.manual_seed(seed)
    x = torch.randn(*contract["input"]["shape"])
    clf = ExecuTorchClassifier(pte)
    et_logits = clf.infer(x.numpy()).logits
    with torch.inference_mode():
        ref = reference_model(x)[0].numpy()
    return {
        "runtime_load_ok": True,
        "output_shape_ok": et_logits.shape == (1000,),
        "random_input_top1_agree": int(np.argmax(et_logits)) == int(np.argmax(ref)),
        "random_input_max_abs_diff": float(np.max(np.abs(et_logits - ref))),
        "registered_backends": clf.registered_backends,
    }
