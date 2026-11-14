"""Fine-tuned 2-class waste model: class mapping, decoding, EcoGuard at 0.95, profiles, fallback, metrics."""

import copy
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from tests.helpers import (
    LABELS, WASTE_CLASSES_DICT, FakeModule, ensure_torch_like, fake_portable_lib, mapping, two_logits, waste_classes,
    write_fake_pte,
)
from zyqora.classset import load_class_set, validate_class_set_dict
from zyqora.config import ConfigError, load_app_config
from zyqora.decoders import DecoderShapeError, DirectClassDecoder, ImageNetMappingDecoder
from zyqora.ecoguard import EcoGuardConfig
from zyqora.evaluator import EvaluationRow, LabelsError, discover_images, labels_from_folders, summarize
from zyqora.materialwise import MappingError
from zyqora.model_runtime import InferenceOutput, ModelLoadError
from zyqora.paths import CONFIG_DIR, DEFAULT_MODEL_CONTRACT
from zyqora.pipeline import ZyqoraPipeline
from zyqora.preprocessing import PreprocessSpec
from zyqora.profiles import load_models_config, load_profile, load_with_fallback
from zyqora.schemas import ACCEPT, MANUAL_REVIEW, MODEL_CLASS, R_LOW_CONFIDENCE, REVIEW

GUARD_095 = EcoGuardConfig(threshold=0.95, require_top1_mapped=False)


class FixedClassifier:
    def __init__(self, logits_list):
        self.logits_list = list(logits_list)

    def infer(self, array):
        assert array.shape == (1, 3, 224, 224) and array.dtype == np.float32
        return InferenceOutput(self.logits_list.pop(0), 2_000_000)


class ClassSetTests(unittest.TestCase):
    def test_shipped_class_file(self):
        cs = load_class_set(CONFIG_DIR / "classes_waste.json", num_classes=2)
        self.assertEqual(cs.class_names, ("Organic", "Recyclable"))  # index 0, index 1 - never reversed
        self.assertEqual(cs.categories, {"Organic": "COMPOST", "Recyclable": "RECYCLE"})
        self.assertEqual(cs.ground_truth_category("O"), "Organic")
        self.assertEqual(cs.ground_truth_category("R"), "Recyclable")
        self.assertEqual(cs.ground_truth_category("Residual"), "Residual")    # accepted as truth...
        self.assertIsNone(cs.canonical_category("Residual"))                  # ...but never predictable
        self.assertNotIn("Residual", cs.category_names)

    def test_class_file_validation(self):
        self.assertEqual(validate_class_set_dict(WASTE_CLASSES_DICT, 2), [])
        bad = copy.deepcopy(WASTE_CLASSES_DICT)
        bad["classes"][0]["index"] = 1
        bad["classes"][1]["action"] = "MANUAL_REVIEW"
        bad["unsupported_ground_truth"] = {"Organic": {}}
        joined = " ".join(validate_class_set_dict(bad, 3))
        for needle in ("output order", "reserved", "both a model class and unsupported", "3 outputs"):
            self.assertIn(needle, joined)
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaises(MappingError):
                load_class_set(Path(t) / "missing.json")


class DecoderTests(unittest.TestCase):
    def test_index_0_is_organic_index_1_is_recyclable(self):
        dec = DirectClassDecoder(waste_classes())
        probs, raw, cat = dec.decode(np.array([3.0, -1.0], np.float32))
        self.assertEqual(cat.predicted_category, "Organic")
        self.assertEqual(cat.category_action, "COMPOST")
        _, _, cat = dec.decode(np.array([-2.0, 4.0], np.float32))
        self.assertEqual((cat.predicted_category, cat.category_action), ("Recyclable", "RECYCLE"))
        self.assertEqual(cat.mapping_status, MODEL_CLASS)

    def test_softmax_applied_once(self):
        logits = np.array([2.0, 0.0], np.float32)
        probs, raw, cat = DirectClassDecoder(waste_classes()).decode(logits)
        expected = np.exp(2.0) / (np.exp(2.0) + 1.0)
        self.assertAlmostEqual(cat.predicted_category_score, expected, places=6)
        self.assertAlmostEqual(float(probs.sum()), 1.0, places=9)
        self.assertAlmostEqual(cat.category_margin, expected - (1 - expected), places=6)
        self.assertEqual([p.label for p in raw], ["Organic", "Recyclable"])

    def test_decoders_reject_the_other_models_output(self):
        with self.assertRaises(DecoderShapeError):
            DirectClassDecoder(waste_classes()).decode(np.zeros(1000, np.float32))
        with self.assertRaises(DecoderShapeError):
            ImageNetMappingDecoder(LABELS, mapping()).decode(np.zeros(2, np.float32))


class WastePipelineTests(unittest.TestCase):
    def setUp(self):
        ensure_torch_like()
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.spec = PreprocessSpec.from_file(CONFIG_DIR / "model_waste.json")

    def tearDown(self):
        self.tmp.cleanup()

    def _img(self, rel):
        p = self.dir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (300, 200), (90, 140, 60)).save(p)
        return p

    def test_contract_is_exact_resize_without_crop(self):
        self.assertEqual(self.spec.resize, (224, 224))
        self.assertIsNone(self.spec.center_crop)
        self.assertEqual(self.spec.input_shape, (1, 3, 224, 224))

    def test_ecoguard_095_boundary_and_review_never_sorted(self):
        p = self._img("x.jpg")
        cases = [(0.99, ACCEPT, "COMPOST"), (0.95, ACCEPT, "COMPOST"), (0.9499, REVIEW, MANUAL_REVIEW),
                 (0.50, REVIEW, MANUAL_REVIEW), (0.03, ACCEPT, "RECYCLE"), (0.06, REVIEW, MANUAL_REVIEW)]
        clf = FixedClassifier([two_logits(po) for po, _, _ in cases])
        pipe = ZyqoraPipeline(clf, DirectClassDecoder(waste_classes()), GUARD_095, self.spec)
        for po, status, action in cases:
            r = pipe.process(p)
            self.assertEqual((r.ecoguard_status, r.disposal_action), (status, action), po)
            if status == REVIEW:
                self.assertEqual(r.review_reason, R_LOW_CONFIDENCE)
                self.assertIsNotNone(r.predicted_category)  # still recorded for the accuracy table

    def test_folder_labels_unsupported_truth_and_metrics(self):
        files = ["O/a.jpg", "O/b.jpg", "R/c.jpg", "R/d.jpg", "Residual/e.jpg"]
        for f in files:
            self._img(f)
        images = discover_images(self.dir)
        cs = waste_classes()
        truth = labels_from_folders(images, cs)
        self.assertEqual(truth["O/a.jpg"], "Organic")
        self.assertEqual(truth["Residual/e.jpg"], "Residual")
        # predictions: O/a Organic .99, O/b Recyclable .97, R/c Recyclable .99, R/d Recyclable .60, Residual/e Recyclable .99
        clf = FixedClassifier([two_logits(x) for x in (0.99, 0.03, 0.01, 0.40, 0.01)])
        pipe = ZyqoraPipeline(clf, DirectClassDecoder(cs), GUARD_095, self.spec)
        rows = [EvaluationRow(pipe.process(p, rel), truth[rel]) for rel, p in images]
        s = summarize(rows, labelled=True, predictable=cs.category_names, unsupported=cs.unsupported_ground_truth)
        self.assertEqual(s["total_images"], 5)
        self.assertEqual(s["total_correct"], 3)               # O/a, R/c, R/d (REVIEW but category right)
        self.assertEqual(s["accepted_count"], 4)
        self.assertEqual(s["accepted_incorrect"], 2)           # O/b and the Residual item
        self.assertEqual(s["unsupported_ground_truth_images"], 1)
        self.assertEqual(s["unsupported_ground_truth_accepted_into_a_bin"], 1)
        m = s["metrics"]
        cm = m["confusion_matrix"]["matrix"]
        self.assertEqual(cm["Organic"], {"Organic": 1, "Recyclable": 1, "<none>": 0})
        self.assertEqual(cm["Residual"]["Recyclable"], 1)
        rec = m["per_class"]["Recyclable"]
        self.assertAlmostEqual(rec["precision"], 2 / 4)
        self.assertAlmostEqual(rec["recall"], 1.0)
        self.assertAlmostEqual(rec["f1"], 2 * 0.5 * 1.0 / 1.5)
        self.assertEqual(m["per_class"]["Residual"]["recall"], 0.0)
        self.assertFalse(m["per_class"]["Residual"]["predictable"])

    def test_folder_label_errors(self):
        self._img("plastic/z.jpg")
        with self.assertRaises(LabelsError):
            labels_from_folders(discover_images(self.dir), waste_classes())
        with self.assertRaises(LabelsError):
            labels_from_folders([("loose.jpg", self.dir / "loose.jpg")], waste_classes())


class ProfileTests(unittest.TestCase):
    """Profiles bind .pte + contract + decoder; the output-shape guard and fallback are enforced."""

    def setUp(self):
        ensure_torch_like()
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        self.waste_pte, self.imnet_pte = d / "waste.pte", d / "imagenet.pte"
        write_fake_pte(self.waste_pte)
        write_fake_pte(self.imnet_pte)
        self.yaml = d / "models.yaml"
        self.yaml.write_text(f"""
active: waste_mobilenetv2
fallback: imagenet_mobilenetv2
profiles:
  waste_mobilenetv2:
    decoder: direct_classes
    pte: {self.waste_pte}
    contract: config/model_waste.json
    classes: config/classes_waste.json
    sha256: null
    ecoguard: {{threshold: 0.95, min_margin: 0.0, require_top1_mapped: false, calibrated: false, frozen: false}}
  imagenet_mobilenetv2:
    decoder: imagenet_mapping
    pte: {self.imnet_pte}
    contract: config/model.json
    mapping: config/mapping.json
    ecoguard: {{threshold: 0.4}}
""")
        self.models = load_models_config(load_app_config(), self.yaml)

    def tearDown(self):
        self.tmp.cleanup()

    def lib_for(self, out_n_by_path):
        """Fake runtime that returns a module with `n` outputs depending on which file is loaded."""
        lib = fake_portable_lib(None)

        def load(path):
            n = out_n_by_path.get(Path(path))
            if n is None:
                raise RuntimeError("corrupt program")
            return FakeModule(np.zeros((1, n), np.float32))

        lib._load_for_executorch = load
        return lib

    def test_active_waste_profile_loads_with_2_outputs(self):
        lib = self.lib_for({self.waste_pte: 2, self.imnet_pte: 1000})
        loaded = load_with_fallback(self.models, allow_fallback=False, portable_lib=lib)
        self.assertEqual(loaded.profile.name, "waste_mobilenetv2")
        self.assertIsInstance(loaded.decoder, DirectClassDecoder)
        self.assertEqual(loaded.classifier.output_shape, (1, 2))
        self.assertEqual(loaded.guard.threshold, 0.95)
        self.assertFalse(loaded.fallback_used)

    def test_wrong_model_for_profile_is_refused(self):
        swapped = self.lib_for({self.waste_pte: 1000, self.imnet_pte: 2})
        with self.assertRaises(ModelLoadError):
            load_profile(self.models.get("waste_mobilenetv2"), portable_lib=swapped)
        with self.assertRaises(ModelLoadError):
            load_profile(self.models.get("imagenet_mobilenetv2"), portable_lib=swapped)

    def test_fallback_only_when_allowed_and_reported(self):
        lib = self.lib_for({self.imnet_pte: 1000})  # waste .pte "corrupt"
        with self.assertRaises(ModelLoadError):
            load_with_fallback(self.models, allow_fallback=False, portable_lib=lib)
        loaded = load_with_fallback(self.models, allow_fallback=True, portable_lib=lib)
        self.assertTrue(loaded.fallback_used)
        self.assertEqual(loaded.profile.name, "imagenet_mobilenetv2")
        self.assertIsInstance(loaded.decoder, ImageNetMappingDecoder)  # its own decoder, never the waste one
        self.assertIn("waste_mobilenetv2", loaded.failures[0])

    def test_missing_model_file_falls_back(self):
        self.waste_pte.unlink()
        lib = self.lib_for({self.imnet_pte: 1000})
        self.assertTrue(load_with_fallback(self.models, allow_fallback=True, portable_lib=lib).fallback_used)

    def test_sha256_pin_mismatch_rejected(self):
        import dataclasses

        prof = dataclasses.replace(self.models.get("waste_mobilenetv2"), sha256="0" * 64)
        with self.assertRaises(ModelLoadError):
            load_profile(prof, portable_lib=self.lib_for({self.waste_pte: 2}))

    def test_invalid_models_config(self):
        bad = Path(self.tmp.name) / "bad.yaml"
        for text in ("active: nope\nprofiles: {a: {decoder: direct_classes}}",
                     "active: a\nprofiles: {a: {decoder: magic}}",
                     "active: a\nprofiles: {a: {decoder: direct_classes, pte: x.pte, contract: c.json, ecoguard: {threshold: 2}}}"):
            bad.write_text(text)
            with self.assertRaises(ConfigError, msg=text):
                load_models_config(load_app_config(), bad)

    def test_shipped_models_config_only_activates_a_verified_waste_model(self):
        """The waste profile may be active only after verify_waste_model.py --activate pinned its SHA-256,
        and a present model file must match that pin (a swapped or corrupted .pte is caught here too)."""
        from zyqora.profiles import file_sha256

        models = load_models_config(load_app_config())
        waste = models.get("waste_mobilenetv2")
        self.assertIn(models.active, ("imagenet_mobilenetv2", "waste_mobilenetv2"))
        if models.active == "waste_mobilenetv2":
            self.assertIsNotNone(waste.sha256, "waste profile active without a pinned sha256 (run verify_waste_model.py --activate)")
            if waste.pte.is_file():
                self.assertEqual(file_sha256(waste.pte), waste.sha256)
        self.assertEqual(waste.ecoguard.threshold, 0.95)
        self.assertFalse(waste.ecoguard.calibrated)
        self.assertEqual(models.get("imagenet_mobilenetv2").contract, DEFAULT_MODEL_CONTRACT)


if __name__ == "__main__":
    unittest.main()
