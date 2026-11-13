"""Script-level tests (run as subprocesses, like a user would)."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def run(*args):
    return subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True, timeout=120)


class LeakageScriptTest(unittest.TestCase):
    def test_exact_and_near_duplicates_detected(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            rng = np.random.default_rng(0)
            for split in ("train/O", "train/R", "test/O", "test/R"):
                (d / split).mkdir(parents=True)
            base = [Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)) for _ in range(4)]
            for i, im in enumerate(base):
                im.save(d / "train" / "O" / f"t{i}.png")
            base[0].save(d / "test" / "O" / "copy.png")                          # exact duplicate
            base[1].resize((128, 128)).save(d / "test" / "R" / "resized.png")    # near duplicate
            Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)).save(d / "test" / "R" / "new.png")
            r = run("scripts/check_leakage.py", "--split", f"train={d / 'train'}", "--split", f"test={d / 'test'}",
                    "--out", str(d / "out"))
            self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
            rep = json.loads((d / "out" / "leakage_report.json").read_text())
            self.assertEqual(rep["summary"]["train_vs_test"], {"exact_duplicates": 1, "near_duplicates": 1})
            overlap = (d / "out" / "test_overlap_files.txt").read_text().split()
            self.assertEqual(sorted(overlap), ["O/copy.png", "R/resized.png"])


class ConfigScriptTest(unittest.TestCase):
    def test_check_config_valid(self):
        r = run("scripts/check_config.py")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("profile waste_mobilenetv2: direct_classes, output 2", r.stdout)
        self.assertIn("RESULT: VALID", r.stdout)


if __name__ == "__main__":
    unittest.main()
