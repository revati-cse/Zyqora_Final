import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from tests import helpers  # noqa: F401  (sets sys.path)
from zyqora.paths import DEFAULT_MODEL_CONTRACT
from zyqora.preprocessing import (
    ImageLoadError,
    PreprocessSpec,
    center_crop,
    load_rgb,
    preprocess_path,
    resize_short_side,
    validate_input_array,
)


class PreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.spec = PreprocessSpec.from_file(DEFAULT_MODEL_CONTRACT)

    def tearDown(self):
        self.tmp.cleanup()

    def _save(self, img, name):
        p = self.dir / name
        img.save(p)
        return p

    def test_contract_matches_course_spec(self):
        self.assertEqual(self.spec.input_shape, (1, 3, 224, 224))
        self.assertEqual((self.spec.resize, self.spec.center_crop), (256, 224))
        self.assertEqual(self.spec.mean, (0.485, 0.456, 0.406))
        self.assertEqual(self.spec.std, (0.229, 0.224, 0.225))

    def test_rgb_shape_dtype(self):
        p = self._save(Image.new("RGB", (640, 480), (10, 200, 30)), "rgb.jpg")
        x = preprocess_path(p, self.spec)
        self.assertEqual(x.shape, (1, 3, 224, 224))
        self.assertEqual(x.dtype, np.float32)
        self.assertTrue(x.flags["C_CONTIGUOUS"])

    def test_normalization_values(self):
        p = self._save(Image.new("RGB", (300, 300), (255, 0, 128)), "solid.png")
        x = preprocess_path(p, self.spec)
        expected = [(1.0 - 0.485) / 0.229, (0.0 - 0.456) / 0.224, (128 / 255 - 0.406) / 0.225]
        for c in range(3):
            self.assertAlmostEqual(float(x[0, c].mean()), expected[c], places=5)

    def test_grayscale_is_replicated(self):
        p = self._save(Image.new("L", (256, 256), 128), "gray.png")
        x = preprocess_path(p, self.spec)
        raw = x[0] * np.array(self.spec.std, np.float32).reshape(3, 1, 1) + np.array(self.spec.mean, np.float32).reshape(3, 1, 1)
        self.assertTrue(np.allclose(raw[0], raw[1], atol=1e-6) and np.allclose(raw[1], raw[2], atol=1e-6))

    def test_rgba_drops_alpha(self):
        p = self._save(Image.new("RGBA", (256, 256), (50, 60, 70, 0)), "rgba.png")
        self.assertEqual(preprocess_path(p, self.spec).shape, (1, 3, 224, 224))

    def test_invalid_and_missing_images(self):
        bad = self.dir / "broken.jpg"
        bad.write_bytes(b"not an image")
        with self.assertRaises(ImageLoadError):
            load_rgb(bad)
        with self.assertRaises(ImageLoadError):
            load_rgb(self.dir / "missing.jpg")
        txt = self.dir / "notes.txt"
        txt.write_text("x")
        with self.assertRaises(ImageLoadError):
            load_rgb(txt)

    def test_truncated_jpeg_rejected(self):
        p = self._save(Image.new("RGB", (400, 400), (1, 2, 3)), "full.jpg")
        trunc = self.dir / "trunc.jpg"
        trunc.write_bytes(p.read_bytes()[:200])
        with self.assertRaises(ImageLoadError):
            load_rgb(trunc)

    def test_resize_and_crop_follow_torchvision_rules(self):
        img = Image.new("RGB", (640, 480))
        r = resize_short_side(img, 256)
        self.assertEqual(r.size, (int(256 * 640 / 480), 256))  # (341, 256)
        tall = resize_short_side(Image.new("RGB", (300, 1000)), 256)
        self.assertEqual(tall.size, (256, int(256 * 1000 / 300)))
        self.assertEqual(center_crop(r, 224).size, (224, 224))

    def test_tiny_image_is_upsampled(self):
        p = self._save(Image.new("RGB", (3, 2), (9, 9, 9)), "tiny.png")
        self.assertEqual(preprocess_path(p, self.spec).shape, (1, 3, 224, 224))

    def test_validate_input_array(self):
        validate_input_array(np.zeros((1, 3, 224, 224), np.float32), (1, 3, 224, 224))
        with self.assertRaises(ValueError):
            validate_input_array(np.zeros((1, 3, 224, 224), np.float64), (1, 3, 224, 224))
        with self.assertRaises(ValueError):
            validate_input_array(np.zeros((3, 224, 224), np.float32), (1, 3, 224, 224))
        nan = np.zeros((1, 3, 224, 224), np.float32)
        nan[0, 0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            validate_input_array(nan, (1, 3, 224, 224))

    def test_torchvision_equivalence_if_available(self):
        try:
            import torchvision.transforms as T
        except ImportError:
            self.skipTest("torchvision not installed (checked by scripts/check_parity.py in the export venv)")
        rng = np.random.default_rng(1)
        img = Image.fromarray(rng.integers(0, 255, (333, 517, 3), dtype=np.uint8))
        p = self._save(img, "noise.png")
        tv = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize(self.spec.mean, self.spec.std)])
        ref = tv(Image.open(p).convert("RGB")).unsqueeze(0).numpy()
        self.assertEqual(float(np.max(np.abs(ref - preprocess_path(p, self.spec)))), 0.0)


if __name__ == "__main__":
    unittest.main()
