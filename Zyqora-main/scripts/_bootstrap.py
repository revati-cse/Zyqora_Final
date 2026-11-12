"""Make ``src/`` importable when scripts are run directly (no install needed)."""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def ensure_interpreter_bin_on_path() -> None:
    """Expose venv tools such as ``flatc`` when venv/bin/python is called directly (Arm course fix)."""
    bin_path = Path(sys.executable).parent
    flatc = bin_path / "flatc"
    if flatc.exists():
        os.environ.setdefault("FLATC_EXECUTABLE", str(flatc))
    parts = [p for p in os.environ.get("PATH", "").split(os.pathsep) if p and p != str(bin_path)]
    os.environ["PATH"] = os.pathsep.join([str(bin_path), *parts])


def quiet_toolchain_logs() -> None:
    """Filter known noisy dependency warnings (as the Arm course does); errors stay visible."""
    ensure_interpreter_bin_on_path()
    os.environ.setdefault("GLOG_minloglevel", "2")
    os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "ERROR")
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=SyntaxWarning)
    warnings.filterwarnings("ignore", category=UserWarning, message="The given buffer is not writable.*")
    import logging

    for name in ("torch", "torch.utils._pytree", "torchvision", "torchao", "executorch"):
        logging.getLogger(name).setLevel(logging.ERROR)
