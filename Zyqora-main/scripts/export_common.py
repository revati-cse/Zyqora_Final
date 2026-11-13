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


def contract_num_classes(contract: dict) -> int:
    return int(contract.get("output", {}).get("shape", [1, 1000])[-1])


def build_reference_model(weights_path: Path, num_classes: int = 1000, *, trusted_pickle: bool = False):
    """Rebuild TorchVision MobileNetV2 (num_classes outputs) from a local checkpoint (no download).

    Accepted checkpoint layouts: a plain state_dict; {"model_state_dict"|"state_dict"|"model": state_dict};
    keys prefixed with "module." (DataParallel); or, only with trusted_pickle=True, a whole pickled
    nn.Module (torch.load(weights_only=False) executes pickle code: use it only for your own files).
    """
    import torch
    from torchvision import models

    if not weights_path.is_file():
        raise SystemExit(f"Checkpoint not found: {weights_path}")
    try:
        obj = torch.load(weights_path, map_location="cpu", weights_only=True)
    except Exception as exc:  # weights_only refuses pickled modules / custom classes
        if not trusted_pickle:
            raise SystemExit(f"{weights_path.name} is not a plain state_dict checkpoint ({exc}).\n"
                             "If it is your own full-model pickle, re-run with --trusted-pickle.") from exc
        obj = torch.load(weights_path, map_location="cpu", weights_only=False)
    if isinstance(obj, torch.nn.Module):
        state = obj.state_dict()
    elif isinstance(obj, dict):
        state = next((obj[k] for k in ("model_state_dict", "state_dict", "model") if isinstance(obj.get(k), dict)), obj)
    else:
        raise SystemExit(f"{weights_path.name}: unsupported checkpoint type {type(obj).__name__}")
    state = {k[len("module."):] if k.startswith("module.") else k: v for k, v in state.items()}
    head = state.get("classifier.1.weight")
    if head is None:
        raise SystemExit(f"{weights_path.name}: no 'classifier.1.weight' key; keys look like {list(state)[:5]}")
    if tuple(head.shape) != (num_classes, 1280):
        raise SystemExit(f"{weights_path.name}: classifier head is {tuple(head.shape)}, expected ({num_classes}, 1280)")
    model = models.mobilenet_v2(weights=None, num_classes=num_classes)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise SystemExit(f"{weights_path.name}: state_dict mismatch. missing={missing[:5]} unexpected={unexpected[:5]}")
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
    n = int(contract.get("output", {}).get("shape", [1, 1000])[-1])
    clf = ExecuTorchClassifier(pte, num_classes=n)
    et_logits = clf.infer(x.numpy()).logits
    with torch.inference_mode():
        ref = reference_model(x)[0].numpy()
    return {
        "note": "random input: checks the export is numerically faithful; the predicted class is meaningless",
        "runtime_load_ok": True,
        "output_shape_ok": et_logits.shape == (n,),
        "random_input_top1_agree": int(np.argmax(et_logits)) == int(np.argmax(ref)),
        "random_input_max_abs_diff": float(np.max(np.abs(et_logits - ref))),
        "registered_backends": clf.registered_backends,
    }
