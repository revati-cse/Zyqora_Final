"""Logging setup: a local log file (no telemetry) plus warnings on stderr."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def setup_logging(log_file: Optional[Path] = None, level: int = logging.INFO, console_level: int = logging.WARNING) -> None:
    root = logging.getLogger("zyqora")
    root.setLevel(level)
    for h in list(root.handlers):
        root.removeHandler(h)
        h.close()
    console = logging.StreamHandler()
    console.setLevel(console_level)
    console.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    root.addHandler(console)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(logging.Formatter(FORMAT))
        root.addHandler(fh)
    root.propagate = False
