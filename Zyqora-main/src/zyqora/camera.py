"""Single-shot image capture for the Zyqora station (no continuous video processing).

Backends
  picamera2   Raspberry Pi camera via Picamera2 (camera stays open, frame returned in memory:
              lowest latency). Needs the python3-picamera2 system package visible in the venv
              (create the venv with --system-site-packages).
  rpicam      `rpicam-still` subprocess (no Python dependency; camera starts per capture, slower).
  folder      SIMULATION: returns images from a folder in order (no camera hardware).
  auto        picamera2, else rpicam, else error.

Every failure raises CameraError; the station reports it and never invents a frame.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PIL import Image

from .evaluator import discover_images
from .preprocessing import ImageLoadError, load_rgb

log = logging.getLogger("zyqora.camera")


class CameraError(RuntimeError):
    pass


@dataclass
class Frame:
    image: Image.Image          # RGB
    name: str                   # identifier for logs / GreenScore (no personal data)
    capture_ms: float           # time spent acquiring the frame
    source: str                 # backend name
    simulated: bool


class FolderCamera:
    """Simulation backend: cycles through the images in a folder."""

    name = "folder"
    simulated = True

    def __init__(self, folder: Path, loop: bool = True):
        self.items = discover_images(folder)
        if not self.items:
            raise CameraError(f"simulation folder has no images: {folder}")
        self.loop, self.i = loop, 0

    def capture(self) -> Frame:
        if self.i >= len(self.items):
            if not self.loop:
                raise CameraError("simulation folder exhausted")
            self.i = 0
        rel, path = self.items[self.i]
        self.i += 1
        t0 = time.perf_counter_ns()
        try:
            img = load_rgb(path)
        except ImageLoadError as exc:
            raise CameraError(f"simulated capture failed: {exc}") from exc
        return Frame(img, rel, (time.perf_counter_ns() - t0) / 1e6, self.name, True)

    def close(self) -> None:
        pass


class RpicamStillCamera:
    """`rpicam-still` per capture (libcamera CLI shipped with Raspberry Pi OS)."""

    name = "rpicam"
    simulated = False

    def __init__(self, width: int = 640, height: int = 480, timeout_ms: int = 300, exe: Optional[str] = None):
        self.exe = exe or shutil.which("rpicam-still") or shutil.which("libcamera-still")
        if not self.exe:
            raise CameraError("rpicam-still not found (install rpicam-apps, or use --camera folder:<dir>)")
        self.width, self.height, self.timeout_ms = width, height, timeout_ms
        self.tmp = Path(tempfile.mkdtemp(prefix="zyqora_cam_"))
        self.n = 0

    def capture(self) -> Frame:
        self.n += 1
        out = self.tmp / f"capture_{self.n:05d}.jpg"
        cmd = [self.exe, "-n", "-t", str(self.timeout_ms), "--width", str(self.width), "--height", str(self.height),
               "-o", str(out)]
        t0 = time.perf_counter_ns()
        try:
            proc = subprocess.run(cmd, capture_output=True, timeout=15 + self.timeout_ms / 1000, check=False)
        except (subprocess.TimeoutExpired, OSError) as exc:
            raise CameraError(f"rpicam-still failed: {exc}") from exc
        if proc.returncode != 0 or not out.is_file():
            err = proc.stderr.decode(errors="replace").strip().splitlines()
            raise CameraError(f"rpicam-still exit {proc.returncode}: {err[-1] if err else 'no camera detected?'}")
        try:
            img = load_rgb(out)
        except ImageLoadError as exc:
            raise CameraError(f"captured file unreadable: {exc}") from exc
        finally:
            out.unlink(missing_ok=True)
        return Frame(img, f"capture_{self.n:05d}", (time.perf_counter_ns() - t0) / 1e6, self.name, False)

    def close(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)


class Picamera2Camera:
    """Picamera2 still capture with the camera kept running between shots."""

    name = "picamera2"
    simulated = False

    def __init__(self, width: int = 640, height: int = 480, picamera2_cls=None):
        if picamera2_cls is None:
            try:
                from picamera2 import Picamera2 as picamera2_cls  # type: ignore
            except ImportError as exc:
                raise CameraError("picamera2 not importable (apt install python3-picamera2 and create the venv "
                                  "with --system-site-packages)") from exc
        try:
            self.cam = picamera2_cls()
            self.cam.configure(self.cam.create_still_configuration(main={"size": (width, height), "format": "RGB888"}))
            self.cam.start()
        except Exception as exc:  # libcamera raises various errors when no camera is attached
            raise CameraError(f"Picamera2 could not start the camera: {exc}") from exc
        self.n = 0

    def capture(self) -> Frame:
        self.n += 1
        t0 = time.perf_counter_ns()
        try:
            img = self.cam.capture_image("main").convert("RGB")
        except Exception as exc:
            raise CameraError(f"Picamera2 capture failed: {exc}") from exc
        return Frame(img, f"capture_{self.n:05d}", (time.perf_counter_ns() - t0) / 1e6, self.name, False)

    def close(self) -> None:
        try:
            self.cam.stop()
            self.cam.close()
        except Exception as exc:  # closing must not hide the original problem
            log.warning("camera close failed: %s", exc)


def open_camera(spec: str, width: int = 640, height: int = 480):
    """spec: 'auto' | 'picamera2' | 'rpicam' | 'folder:<dir>'."""
    if spec.startswith("folder:"):
        return FolderCamera(Path(spec.split(":", 1)[1]))
    if spec == "picamera2":
        return Picamera2Camera(width, height)
    if spec == "rpicam":
        return RpicamStillCamera(width, height)
    if spec == "auto":
        errors = []
        for factory in (lambda: Picamera2Camera(width, height), lambda: RpicamStillCamera(width, height)):
            try:
                return factory()
            except CameraError as exc:
                errors.append(str(exc))
        raise CameraError("no camera available: " + " | ".join(errors))
    raise CameraError(f"unknown camera backend {spec!r} (auto, picamera2, rpicam, folder:<dir>)")
