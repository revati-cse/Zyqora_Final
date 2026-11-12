"""Runtime wrapper tests with a fake ExecuTorch binding (unit level).

A real-artifact integration test runs only when ExecuTorch and the exported
.pte are present (export venv or Raspberry Pi); otherwise it is skipped.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from tests.helpers import FakeModule, ensure_torch_like, fake_portable_lib, write_fake_pte
from zyqora.model_runtime import (
    ExecuTorchClassifier,
    InferenceError,
    ModelLoadError,
    RuntimeUnavailableError,
    check_pte_file,
    softmax,
)
from zyqora.paths import MODELS_DIR

X = np.zeros((1, 3, 224, 224), dtype=np.float32)


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        ensure_torch_like()
        self.tmp = tempfile.TemporaryDirectory()
        self.pte = Path(self.tmp.name) / "model.pte"
        write_fake_pte(self.pte)

    def tearDown(self):
        self.tmp.cleanup()

    def make(self, module=None, **kw):
        module = module or FakeModule(np.arange(1000, dtype=np.float32).reshape(1, 1000))
        lib = kw.pop("lib", None) or fake_portable_lib(module)
        return ExecuTorchClassifier(self.pte, portable_lib=lib, **kw), module, lib

    def test_loads_once_and_runs(self):
        clf, module, lib = self.make()
        for _ in range(3):
            out = clf.infer(X)
        self.assertEqual(lib.loads, 1)
        self.assertEqual(module.calls, 3)
        self.assertEqual(out.logits.shape, (1000,))
        self.assertEqual(int(np.argmax(out.logits)), 999)
        self.assertGreaterEqual(out.latency_ns, 0)

    def test_input_validation(self):
        clf, module, _ = self.make()
        for bad in (np.zeros((1, 3, 224, 224), np.float64), np.zeros((1, 3, 256, 256), np.float32),
                    np.zeros((3, 224, 224), np.float32)):
            with self.assertRaises(ValueError):
                clf.infer(bad)
        self.assertEqual(module.calls, 0)

    def test_output_shape_validation(self):
        clf, _, _ = self.make(FakeModule(np.zeros((1, 10), np.float32)))
        with self.assertRaises(InferenceError):
            clf.infer(X)
        nan = np.zeros((1, 1000), np.float32)
        nan[0, 3] = np.nan
        clf, _, _ = self.make(FakeModule(nan))
        with self.assertRaises(InferenceError):
            clf.infer(X)
        clf, _, _ = self.make(FakeModule(np.zeros((1, 1000)), raise_on_forward=True))
        with self.assertRaises(InferenceError):
            clf.infer(X)

    def test_missing_and_invalid_artifacts(self):
        with self.assertRaises(ModelLoadError):
            check_pte_file(Path(self.tmp.name) / "missing.pte")
        junk = Path(self.tmp.name) / "junk.pte"
        junk.write_bytes(b"PK\x03\x04" + b"\x00" * 200)
        with self.assertRaises(ModelLoadError):
            check_pte_file(junk)
        tiny = Path(self.tmp.name) / "tiny.pte"
        tiny.write_bytes(b"\x00\x00\x00\x00ET12")
        with self.assertRaises(ModelLoadError):
            check_pte_file(tiny)

    def test_incompatible_runtime_load_error(self):
        lib = fake_portable_lib(FakeModule(np.zeros((1, 1000))), load_error="version mismatch")
        with self.assertRaises(ModelLoadError):
            self.make(lib=lib)

    def test_missing_xnnpack_backend(self):
        lib = fake_portable_lib(FakeModule(np.zeros((1, 1000))), backends=("QnnBackend",))
        with self.assertRaises(RuntimeUnavailableError):
            self.make(lib=lib)

    def test_model_input_shape_mismatch_detected_at_load(self):
        with self.assertRaises(ModelLoadError):
            self.make(FakeModule(np.zeros((1, 1000)), input_shape=(1, 3, 299, 299)))

    def test_thread_configuration(self):
        clf, _, lib = self.make(num_threads=2)
        self.assertEqual((clf.thread_count, lib.threads), (2, 2))
        with self.assertRaises(ValueError):
            self.make(num_threads=0)

    def test_softmax_is_stable_and_normalised(self):
        p = softmax(np.array([1000.0, 1000.0, -1000.0]))
        self.assertTrue(np.isfinite(p).all())
        self.assertAlmostEqual(float(p.sum()), 1.0)
        self.assertAlmostEqual(float(p[0]), 0.5)


@unittest.skipUnless((MODELS_DIR / "mobilenetv2_fp32_xnnpack.pte").is_file(), "exported .pte not present")
class RealArtifactIntegrationTest(unittest.TestCase):
    def test_real_pte_runs_through_executorch(self):
        try:
            import executorch  # noqa: F401
        except ImportError:
            self.skipTest("executorch not installed in this interpreter")
        clf = ExecuTorchClassifier(MODELS_DIR / "mobilenetv2_fp32_xnnpack.pte")
        out = clf.infer(np.random.default_rng(0).standard_normal((1, 3, 224, 224)).astype(np.float32))
        self.assertEqual(out.logits.shape, (1000,))
        self.assertIn("XnnpackBackend", clf.registered_backends)


if __name__ == "__main__":
    unittest.main()
