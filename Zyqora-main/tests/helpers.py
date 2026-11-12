"""Shared synthetic fixtures (no real organiser data, no network)."""

from __future__ import annotations

import sys
import types

import numpy as np

from zyqora.imagenet_labels import load_labels
from zyqora.materialwise import mapping_from_dict

LABELS = load_labels()

# Small mapping over real ImageNet indices: 898 water bottle, 737 pop bottle,
# 954 banana, 950 orange, 529 diaper.
MAPPING_DICT = {
    "schema_version": 1,
    "status": "EXAMPLE",
    "aggregation": "sum_softmax",
    "categories": {
        "Recyclable": {"action": "RECYCLE", "aliases": ["dry"]},
        "Organic": {"action": "COMPOST", "aliases": ["wet"]},
        "Residual": {"action": "RESIDUAL"},
    },
    "classes": [
        {"index": 898, "label": "water bottle", "category": "Recyclable", "rule": "REC-01", "material": "PET"},
        {"index": 737, "label": "pop bottle", "category": "Recyclable", "rule": "REC-02"},
        {"index": 954, "label": "banana", "category": "Organic", "rule": "ORG-01"},
        {"index": 950, "label": "orange", "category": "Organic", "rule": "ORG-02"},
        {"index": 529, "label": "diaper", "category": "Residual", "rule": "RES-01"},
    ],
}


def mapping():
    return mapping_from_dict(MAPPING_DICT, source="test")


def probs_with(**by_index: float) -> np.ndarray:
    """Probability vector with given mass on indices; the rest spread over UNMAPPED classes."""
    p = np.zeros(1000, dtype=np.float64)
    for k, v in by_index.items():
        p[int(k.lstrip("i"))] = v
    rest = 1.0 - p.sum()
    assert rest >= -1e-12
    free = p == 0
    free[[c["index"] for c in MAPPING_DICT["classes"]]] = False
    p[free] = rest / free.sum()
    return p


def logits_for(probs: np.ndarray) -> np.ndarray:
    return np.log(np.clip(probs, 1e-30, None)).astype(np.float32)


def ensure_torch_like() -> None:
    """Runtime code calls torch.from_numpy; provide a stand-in when torch is absent."""
    try:
        import torch  # noqa: F401
    except ImportError:
        fake = types.ModuleType("torch")
        fake.from_numpy = lambda a: a
        sys.modules["torch"] = fake


class FakeOutput:
    def __init__(self, arr):
        self._arr = arr

    def numpy(self):
        return self._arr


class FakeMeta:
    def __init__(self, shape):
        self._shape = shape

    def input_tensor_meta(self, i):
        shape = self._shape

        class T:
            def sizes(self):
                return shape

        return T()


class FakeModule:
    def __init__(self, logits, input_shape=(1, 3, 224, 224), raise_on_forward=False):
        self.logits = logits
        self.input_shape = input_shape
        self.raise_on_forward = raise_on_forward
        self.calls = 0

    def method_meta(self, name):
        return FakeMeta(self.input_shape)

    def forward(self, inputs):
        self.calls += 1
        if self.raise_on_forward:
            raise RuntimeError("boom")
        return [FakeOutput(np.asarray(self.logits, dtype=np.float32))]


def fake_portable_lib(module, backends=("XnnpackBackend",), load_error=None):
    lib = types.SimpleNamespace()
    lib.threads = 4
    lib._get_registered_backend_names = lambda: list(backends)
    lib._threadpool_get_thread_count = lambda: lib.threads

    def reset(n):
        lib.threads = n

    lib._unsafe_reset_threadpool = reset

    def load(path):
        if load_error:
            raise RuntimeError(load_error)
        lib.loads = getattr(lib, "loads", 0) + 1
        return module

    lib._load_for_executorch = load
    return lib


def write_fake_pte(path) -> None:
    """Bytes with the ExecuTorch flatbuffer identifier at offset 4 (content irrelevant for fakes)."""
    path.write_bytes(b"\x00\x00\x00\x00ET12" + b"\x00" * 128)
