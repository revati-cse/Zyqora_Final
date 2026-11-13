"""Camera, sorter and end-to-end station simulation (no camera, no GPIO, no network, no real ExecuTorch)."""

import json
import os
import socket
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from tests.helpers import FakeModule, ensure_torch_like, fake_portable_lib, two_logits, write_fake_pte
from zyqora.camera import CameraError, FolderCamera, Picamera2Camera, RpicamStillCamera, open_camera
from zyqora.schemas import ACCEPT, MANUAL_REVIEW, REVIEW
from zyqora.sorter import ActuatorError, GpioSorter, SimulatedSorter, decide_sort, make_sorter

ROOT = Path(__file__).resolve().parents[1]


class FakeDevice:
    log = []

    def __init__(self, pin, active_high=True, initial_value=False):
        self.pin = pin

    def on(self):
        FakeDevice.log.append(("on", self.pin))

    def off(self):
        FakeDevice.log.append(("off", self.pin))

    def close(self):
        FakeDevice.log.append(("close", self.pin))


class CameraTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_folder_camera_cycles_and_reports_simulation(self):
        for i in range(2):
            Image.new("RGB", (64, 48), (i * 100, 0, 0)).save(self.dir / f"{i}.png")
        cam = open_camera(f"folder:{self.dir}")
        names = [cam.capture().name for _ in range(3)]
        self.assertEqual(names, ["0.png", "1.png", "0.png"])
        self.assertTrue(cam.simulated)

    def test_folder_camera_errors(self):
        with self.assertRaises(CameraError):
            FolderCamera(self.dir)  # empty
        (self.dir / "bad.jpg").write_bytes(b"x")
        with self.assertRaises(CameraError):
            FolderCamera(self.dir).capture()

    def test_rpicam_missing_or_failing(self):
        with mock.patch("shutil.which", return_value=None):
            with self.assertRaises(CameraError):
                RpicamStillCamera()
        fake = self.dir / "rpicam-still"
        fake.write_text("#!/bin/sh\necho 'ERROR: no cameras available' >&2\nexit 1\n")
        fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
        cam = RpicamStillCamera(exe=str(fake))
        with self.assertRaises(CameraError) as ctx:
            cam.capture()
        self.assertIn("no cameras available", str(ctx.exception))
        cam.close()

    def test_picamera2_failures_become_camera_errors(self):
        class NoCamera:
            def __init__(self):
                raise RuntimeError("Camera __init__ sequence did not complete")

        with self.assertRaises(CameraError):
            Picamera2Camera(picamera2_cls=NoCamera)

        class Flaky:
            def create_still_configuration(self, **kw):
                return kw

            def configure(self, c):
                pass

            def start(self):
                pass

            def capture_image(self, name):
                raise RuntimeError("frame timeout")

            def stop(self):
                pass

            def close(self):
                pass

        cam = Picamera2Camera(picamera2_cls=Flaky)
        with self.assertRaises(CameraError):
            cam.capture()

    def test_unknown_backend(self):
        with self.assertRaises(CameraError):
            open_camera("webcam9")


class SorterTests(unittest.TestCase):
    def test_review_is_never_sorted(self):
        d = decide_sort(REVIEW, MANUAL_REVIEW)
        self.assertFalse(d.actuate)
        self.assertEqual(d.action, MANUAL_REVIEW)
        self.assertFalse(decide_sort(ACCEPT, MANUAL_REVIEW).actuate)
        self.assertTrue(decide_sort(ACCEPT, "RECYCLE").actuate)

    def test_simulation_is_default_and_gpio_needs_flag(self):
        self.assertIsInstance(make_sorter({}, enable_actuator=True), SimulatedSorter)
        cfg = {"backend": "gpio", "pins": {"RECYCLE": 27}}
        self.assertIsInstance(make_sorter(cfg, enable_actuator=False, device_cls=FakeDevice), SimulatedSorter)
        self.assertIsInstance(make_sorter(cfg, enable_actuator=True, device_cls=FakeDevice), GpioSorter)

    def test_gpio_pulse_and_hold(self):
        FakeDevice.log = []
        s = GpioSorter({"RECYCLE": 27, "COMPOST": 17}, pulse_s=0.01, device_cls=FakeDevice)
        s.execute(decide_sort(ACCEPT, "RECYCLE"))
        self.assertEqual(FakeDevice.log, [("on", 27), ("off", 27)])
        FakeDevice.log = []
        s.execute(decide_sort(REVIEW, MANUAL_REVIEW))
        self.assertEqual(FakeDevice.log, [])  # nothing moves for a review item
        with self.assertRaises(ActuatorError):
            s.execute(decide_sort(ACCEPT, "RESIDUAL"))  # no pin configured
        s.close()

    def test_gpio_config_validation(self):
        with self.assertRaises(ActuatorError):
            GpioSorter({}, device_cls=FakeDevice)
        with self.assertRaises(ActuatorError):
            GpioSorter({"A": 5, "B": 5}, device_cls=FakeDevice)
        with self.assertRaises(ActuatorError):
            GpioSorter({"A": 5}, pulse_s=30, device_cls=FakeDevice)


class StationSimulationTest(unittest.TestCase):
    """Full station loop: folder camera -> real pipeline/decoder/EcoGuard -> simulated sorter -> GreenScore,
    with a fake ExecuTorch runtime and every network connection blocked."""

    def test_end_to_end_offline(self):
        ensure_torch_like()
        sys.path.insert(0, str(ROOT))
        import zyqora_station

        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            imgs = d / "imgs"
            imgs.mkdir()
            for i in range(3):
                Image.new("RGB", (320, 240), (40 * i, 120, 60)).save(imgs / f"item{i}.jpg")
            pte = d / "waste.pte"
            write_fake_pte(pte)
            models_yaml = d / "models.yaml"
            models_yaml.write_text(f"""
active: waste_mobilenetv2
fallback: null
profiles:
  waste_mobilenetv2:
    decoder: direct_classes
    pte: {pte}
    contract: config/model_waste.json
    classes: config/classes_waste.json
    ecoguard: {{threshold: 0.95, require_top1_mapped: false}}
""")
            outputs = iter([two_logits(0.99), two_logits(0.70), two_logits(0.02), two_logits(0.5)])

            class Module(FakeModule):
                def forward(self, inputs):
                    return [type("O", (), {"numpy": lambda s, a=next(outputs): a.reshape(1, 2)})()]

            lib = fake_portable_lib(Module(np.zeros((1, 2), np.float32)))

            def no_network(*a, **k):
                raise AssertionError("network access attempted during offline station run")

            with mock.patch("zyqora.model_runtime.import_portable_lib", return_value=lib), \
                    mock.patch.object(socket.socket, "connect", no_network), \
                    mock.patch.object(socket, "create_connection", no_network):
                rc = zyqora_station.main(["--models-config", str(models_yaml), "--camera", f"folder:{imgs}",
                                          "--auto", "3", "--user", "T01", "--greenscore-db", str(d / "gs.sqlite3"),
                                          "--hardware", str(d / "none.yaml"), "--out", str(d / "run")])
            self.assertEqual(rc, 0)
            events = [json.loads(x) for x in (d / "run" / "events.jsonl").read_text().splitlines()]
            # outputs[0] was consumed by the warm-up forward pass
            self.assertEqual([(e["predicted_category"], e["ecoguard"], e["action"]) for e in events],
                             [("Organic", "REVIEW", MANUAL_REVIEW), ("Recyclable", "ACCEPT", "RECYCLE"),
                              ("Organic", "REVIEW", MANUAL_REVIEW)])
            self.assertTrue(all(e["simulated_camera"] for e in events))
            self.assertIn("[SIMULATION] would open the RECYCLE bin", events[1]["sorter"])
            self.assertIn("held for review", events[0]["sorter"])
            self.assertTrue(all(e["camera_to_result_ms"] > 0 for e in events))
            summary = json.loads((d / "run" / "summary.json").read_text())
            self.assertEqual((summary["accepted"], summary["review"]), (1, 2))
            self.assertFalse(summary["sorter_hardware"])
            from zyqora.greenscore import GreenScore

            self.assertEqual(GreenScore(d / "gs.sqlite3").points("T01"), 0)  # nothing awarded without verification

    def test_missing_model_without_fallback_exits_cleanly(self):
        sys.path.insert(0, str(ROOT))
        import zyqora_station

        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            yaml = d / "models.yaml"
            yaml.write_text(f"""
active: w
profiles:
  w: {{decoder: direct_classes, pte: {d / 'missing.pte'}, contract: config/model_waste.json,
       classes: config/classes_waste.json, ecoguard: {{threshold: 0.95}}}}
""")
            rc = zyqora_station.main(["--models-config", str(yaml), "--camera", "folder:/nonexistent",
                                      "--auto", "1", "--out", str(d / "run")])
            self.assertEqual(rc, 3)


if __name__ == "__main__":
    unittest.main()
