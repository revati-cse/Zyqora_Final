"""ExecuTorch runtime wrapper.

Uses the runtime API from the Arm course (Module 2, pi/core/mobilenetv2_core.py):

    executorch.extension.pybindings.portable_lib._load_for_executorch(pte).forward((tensor,))[0]

verified against ExecuTorch 1.3.1. The model is loaded once and reused.
There is intentionally NO PyTorch fallback: if ExecuTorch cannot run the
model, the caller gets a clear error.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from types import ModuleType
from typing import Any, Optional

import numpy as np

from .preprocessing import validate_input_array
from .schemas import RawPrediction

XNNPACK_BACKEND = "XnnpackBackend"


class RuntimeUnavailableError(RuntimeError):
    """ExecuTorch (or a required backend) is not usable in this environment."""


class ModelLoadError(RuntimeError):
    """The .pte artifact is missing, invalid or incompatible with the runtime."""


class InferenceError(RuntimeError):
    """The model ran but produced an unusable output."""


def _pkg_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not installed"


def import_portable_lib() -> ModuleType:
    try:
        from executorch.extension.pybindings import portable_lib  # type: ignore
    except ImportError as exc:
        import sys

        raise RuntimeUnavailableError(
            "Could not import ExecuTorch pybindings (executorch.extension.pybindings.portable_lib).\n"
            "Install the pinned runtime: pip install -r requirements-pi.txt (Python 3.12).\n"
            f"Python {sys.version.split()[0]}, torch {_pkg_version('torch')}, "
            f"executorch {_pkg_version('executorch')}. Original error: {exc}"
        ) from exc
    return portable_lib


def check_pte_file(path: Path) -> None:
    """Fail early with a clear message instead of an opaque runtime crash."""
    if not path.exists():
        raise ModelLoadError(
            f"Model file not found: {path}\n"
            "Export it with scripts/prepare_model.py + scripts/export_fp32.py, or copy it to models/."
        )
    if not path.is_file():
        raise ModelLoadError(f"Model path is not a file: {path}")
    size = path.stat().st_size
    if size < 64:
        raise ModelLoadError(f"Model file is too small to be an ExecuTorch program ({size} bytes): {path}")
    with path.open("rb") as f:
        header = f.read(8)
    # ExecuTorch programs are flatbuffers with file identifier "ETxx" at byte offset 4.
    if header[4:6] != b"ET":
        raise ModelLoadError(f"Not an ExecuTorch .pte file (missing 'ET' flatbuffer identifier): {path}")


def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over a 1-D logit vector (applied exactly once)."""
    x = np.asarray(logits, dtype=np.float64)
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()


def topk(probs: np.ndarray, labels: list[str], k: int = 5) -> list[RawPrediction]:
    idx = np.argsort(-probs, kind="stable")[:k]
    return [RawPrediction(int(i), labels[int(i)], float(probs[int(i)])) for i in idx]


@dataclass
class InferenceOutput:
    logits: np.ndarray  # shape (num_classes,), raw logits kept for debugging
    latency_ns: int     # model-only forward() time

    @property
    def latency_ms(self) -> float:
        return self.latency_ns / 1e6


@dataclass
class ExecuTorchClassifier:
    pte_path: Path
    input_shape: tuple[int, ...] = (1, 3, 224, 224)
    num_classes: int = 1000
    num_threads: Optional[int] = None
    require_backend: Optional[str] = XNNPACK_BACKEND
    portable_lib: Optional[ModuleType] = None  # injectable for unit tests

    registered_backends: list[str] = field(init=False, default_factory=list)
    thread_count: Optional[int] = field(init=False, default=None)
    thread_control: str = field(init=False, default="not requested")
    load_time_ms: float = field(init=False, default=0.0)
    _module: Any = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        self.pte_path = Path(self.pte_path)
        check_pte_file(self.pte_path)
        pl = self.portable_lib or import_portable_lib()
        self.portable_lib = pl

        if hasattr(pl, "_get_registered_backend_names"):
            self.registered_backends = [str(b) for b in pl._get_registered_backend_names()]
            if self.require_backend and self.require_backend not in self.registered_backends:
                raise RuntimeUnavailableError(
                    f"Required ExecuTorch backend '{self.require_backend}' is not registered "
                    f"(available: {self.registered_backends})."
                )
        elif self.require_backend:
            self.registered_backends = ["<unknown: runtime has no _get_registered_backend_names>"]

        self._configure_threads(pl)

        start = time.perf_counter_ns()
        try:
            self._module = pl._load_for_executorch(str(self.pte_path))
        except Exception as exc:  # pybind raises RuntimeError / ValueError subclasses
            raise ModelLoadError(
                f"ExecuTorch failed to load {self.pte_path.name}: {exc}\n"
                f"Runtime executorch=={_pkg_version('executorch')}. The .pte may have been "
                "exported with an incompatible ExecuTorch version; re-export with the pinned toolchain."
            ) from exc
        self.load_time_ms = (time.perf_counter_ns() - start) / 1e6
        self._validate_method_meta()

    # -- setup -------------------------------------------------------------
    def _configure_threads(self, pl: ModuleType) -> None:
        if self.num_threads is None:
            if hasattr(pl, "_threadpool_get_thread_count"):
                self.thread_count = int(pl._threadpool_get_thread_count())
                self.thread_control = "runtime default"
            return
        if self.num_threads < 1:
            raise ValueError(f"num_threads must be >= 1, got {self.num_threads}")
        if not hasattr(pl, "_unsafe_reset_threadpool"):
            self.thread_control = "unsupported by this ExecuTorch build; runtime default used"
            return
        # Must be called before any inference is in flight (hence 'unsafe').
        pl._unsafe_reset_threadpool(int(self.num_threads))
        self.thread_count = int(pl._threadpool_get_thread_count()) if hasattr(pl, "_threadpool_get_thread_count") else None
        self.thread_control = "set via _unsafe_reset_threadpool"

    def _validate_method_meta(self) -> None:
        if not hasattr(self._module, "method_meta"):
            return
        try:
            meta = self._module.method_meta("forward")
            sizes = tuple(int(s) for s in meta.input_tensor_meta(0).sizes())
        except Exception as exc:
            raise ModelLoadError(f"{self.pte_path.name} has no usable 'forward' method: {exc}") from exc
        if sizes != tuple(self.input_shape):
            raise ModelLoadError(
                f"{self.pte_path.name} expects input {sizes}, but the model contract says {tuple(self.input_shape)}"
            )

    # -- inference ---------------------------------------------------------
    def infer(self, array: np.ndarray) -> InferenceOutput:
        """Run one forward pass on a preprocessed [1,3,224,224] float32 array."""
        validate_input_array(array, self.input_shape)
        import torch  # the ExecuTorch Python bindings exchange torch tensors

        tensor = torch.from_numpy(array)  # zero-copy
        start = time.perf_counter_ns()
        try:
            out = self._module.forward((tensor,))
        except Exception as exc:
            raise InferenceError(f"ExecuTorch forward() failed: {exc}") from exc
        latency_ns = time.perf_counter_ns() - start

        if not out:
            raise InferenceError("ExecuTorch forward() returned no outputs")
        first = out[0]
        logits = first.numpy() if hasattr(first, "numpy") else np.asarray(first)
        if tuple(logits.shape) != (1, self.num_classes):
            raise InferenceError(f"Unexpected output shape {tuple(logits.shape)}; expected (1, {self.num_classes})")
        logits = np.array(logits[0], dtype=np.float32, copy=True)
        if not np.isfinite(logits).all():
            raise InferenceError("Model output contains NaN or Inf")
        return InferenceOutput(logits=logits, latency_ns=latency_ns)

    def describe(self) -> dict[str, Any]:
        return {
            "runtime": "ExecuTorch",
            "executorch_version": _pkg_version("executorch"),
            "pte": str(self.pte_path),
            "pte_size_mb": self.pte_path.stat().st_size / 1e6,
            "registered_backends": self.registered_backends,
            "num_threads_requested": self.num_threads,
            "thread_count": self.thread_count,
            "thread_control": self.thread_control,
            "load_time_ms": self.load_time_ms,
        }
