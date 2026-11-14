"""Zyqora AI 4.0 unit tests: Green Score, label profiles, materials, recommendations, YOLO decoding,
detection metrics, the mixed-waste pipeline and its SQLite store.

The detector and classifier here are FAKES with scripted outputs: these tests prove the logic
(geometry, NMS, mapping, scoring, persistence), NOT that the real models are accurate. Real-model
evidence is in reports/ (scripts/evaluate_detector.py, scripts/benchmark_detector.py).
"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from zyqora.config import ConfigError
from zyqora.detection.evaluate import GroundTruth, ImageRecord, Prediction, evaluate
from zyqora.detection.yolo import (
    DetectorError, DetectorProfile, DetectorSettings, YoloDetector, letterbox, load_detector_config, nms,
)
from zyqora.greenscore_engine import NS_CONFLICT, NS_NO_RULE, NS_NOT_WASTE, NS_REVIEW, GreenScoreEngine
from zyqora.mixed import MixedWastePipeline, padded_crop_box
from zyqora.paths import CONFIG_DIR
from zyqora.schemas import ImageResult
from zyqora.wasterules import RecommendationEngine, load_label_profiles, load_material_table

COCO = tuple(json.loads((CONFIG_DIR / "detector_coco80.json").read_text())["classes"])
CLASSES = ["Organic", "Recyclable"]


# ---------------------------------------------------------------- fakes
def raw_output(dets, nc=80, n=50):
    """Synthetic YOLO head output [1, 4+nc, n]; dets = (cx, cy, w, h, class_id, score) in input pixels."""
    out = np.zeros((1, 4 + nc, n), np.float32)
    for i, (cx, cy, w, h, c, s) in enumerate(dets):
        out[0, :4, i] = (cx, cy, w, h)
        out[0, 4 + c, i] = s
    return out


class FakeRunner:
    backend = "fake"

    def __init__(self, raw, imgsz=640, nc=80):
        self.raw = raw
        self.output_shape = (1, 4 + nc, raw.shape[2])
        self.load_ms = 0.0
        self.calls = 0

    def run(self, x):
        self.calls += 1
        assert x.shape[0] == 1 and x.shape[1] == 3
        return self.raw

    def describe(self):
        return {"backend": self.backend, "load_ms": 0.0}


def profile(imgsz=640, classes=COCO):
    return DetectorProfile("fake", Path("fake.pte"), "executorch", imgsz, classes, Path("x"))


class FakeClassifier:
    """Stands in for ZyqoraPipeline.process_image: returns scripted (label, score, status)."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def process_image(self, img, name):
        self.calls.append((name, img.size))
        label, score, status = self.script.pop(0) if self.script else ("Recyclable", 0.99, "ACCEPT")
        other = "Organic" if label == "Recyclable" else "Recyclable"
        return ImageResult(image_filename=name, predicted_category=label, predicted_category_score=score,
                           category_scores={label: score, other: 1 - score}, ecoguard_status=status,
                           review_reason="" if status == "ACCEPT" else "LOW_CONFIDENCE",
                           preprocess_time_ms=1.0, inference_time_ms=2.0, end_to_end_time_ms=3.0)


def mixed_pipeline(dets, script, settings=None):
    det = YoloDetector(profile(), settings or DetectorSettings(min_box_area_fraction=0.0), runner=FakeRunner(raw_output(dets)))
    dc = load_detector_config()
    return MixedWastePipeline(det, FakeClassifier(script), load_material_table(dc.materials, CLASSES, COCO),
                              load_label_profiles(model_classes=CLASSES), RecommendationEngine.load(),
                              GreenScoreEngine.load(), tuple(CLASSES))


# ---------------------------------------------------------------- Green Score
class GreenScoreTests(unittest.TestCase):
    def setUp(self):
        self.eng = GreenScoreEngine.load()

    def test_required_formula_example(self):
        score = (90 * 0.40 + 80 * 0.35 + 100 * 0.25)
        assert score == 89
        r = self.eng.score_components(90, 80, 100)
        self.assertTrue(r.scored)
        self.assertEqual(r.score, 89)
        self.assertEqual(r.band, "High sustainability potential")
        self.assertEqual([c.weight for c in r.components], [0.40, 0.35, 0.25])
        self.assertIn("= 89/100", r.explanation)

    def test_bands_are_inclusive_lower_bounds(self):
        for value, band in [(100, "High"), (80, "High"), (79.99, "Moderate-to-high"), (60, "Moderate-to-high"),
                            (40, "Moderate potential"), (20, "Low"), (19.9, "Very low"), (0, "Very low")]:
            self.assertTrue(self.eng.band(value).startswith(band), (value, self.eng.band(value)))

    def test_material_rule_and_explanations(self):
        r = self.eng.score_item(material="Metal", material_certainty="likely", model_class="Recyclable", decision="ACCEPT")
        self.assertEqual(r.basis, "material:Metal")
        self.assertAlmostEqual(r.score, 95 * 0.4 + 80 * 0.35 + 85 * 0.25)
        self.assertTrue(all(c.why for c in r.components))      # every component is explained

    def test_unknown_material_falls_back_to_model_class(self):
        r = self.eng.score_item(material="Unknown", material_certainty="unknown", model_class="Recyclable", decision="ACCEPT")
        self.assertEqual(r.basis, "model_class:Recyclable")

    def test_unreliable_or_unsupported_items_are_not_scored(self):
        cases = [
            (dict(decision="REVIEW"), NS_REVIEW),
            (dict(decision="ACCEPT", conflict=True), NS_CONFLICT),
            (dict(decision="ACCEPT", waste_item=False), NS_NOT_WASTE),
        ]
        for kw, reason in cases:
            r = self.eng.score_item(material="Metal", material_certainty="likely", model_class="Recyclable", **kw)
            self.assertFalse(r.scored)
            self.assertIsNone(r.score)
            self.assertEqual(r.not_scored_reason, reason)
        r = self.eng.score_item(material="Unknown", material_certainty="unknown", model_class="Residual", decision="ACCEPT")
        self.assertEqual(r.not_scored_reason, NS_NO_RULE)

    def test_confidence_is_not_an_input(self):
        import inspect
        params = inspect.signature(self.eng.score_item).parameters
        self.assertFalse(any("confidence" in p or "score" in p for p in params))

    def test_missing_component_and_range_checks(self):
        r = self.eng.score_components(90, None, 100)
        self.assertFalse(r.scored)
        with self.assertRaises(ValueError):
            self.eng.score_components(101, 50, 50)

    def test_aggregate(self):
        a = self.eng.score_components(90, 80, 100)
        b = self.eng.score_components(50, 50, 50)
        ns = self.eng.score_item(material=None, material_certainty=None, model_class="Organic", decision="REVIEW")
        agg = self.eng.aggregate([a, b, ns])
        self.assertTrue(agg["valid"])
        self.assertEqual(agg["score"], 69.5)
        self.assertEqual((agg["scored_items"], agg["waste_items"]), (2, 3))
        self.assertTrue(agg["partial"])
        none = self.eng.aggregate([ns])
        self.assertFalse(none["valid"])
        self.assertIsNone(none["score"])

    def test_invalid_rules_rejected(self):
        base = {"weights": {"recycling_potential": 0.5, "environmental_impact": 0.35, "disposal_suitability": 0.25},
                "bands": [{"min": 0, "label": "x"}]}
        with self.assertRaises(ConfigError):
            GreenScoreEngine(base)
        base["weights"]["recycling_potential"] = 0.40
        GreenScoreEngine(base)
        with self.assertRaises(ConfigError):
            GreenScoreEngine({**base, "materials": {"Metal": {"recycling_potential": 120, "environmental_impact": 1,
                                                              "disposal_suitability": 1}}})


# ---------------------------------------------------------------- labels / materials / rules
class RulesTests(unittest.TestCase):
    def test_label_profiles(self):
        lp = load_label_profiles(model_classes=CLASSES)
        self.assertEqual(lp.active, "organiser")
        self.assertEqual(lp.get("organiser").label("Organic", True), "Organic")
        self.assertEqual(lp.get("recyclable").label("Organic", True), "Non-Recyclable")
        self.assertEqual(lp.get("food").label("Organic", True), "Food")
        self.assertEqual(lp.get("food").label("Recyclable", True), "Waste")
        for p in lp.profiles.values():
            self.assertEqual(p.label("Recyclable", False), "Unknown")   # unreliable -> Unknown everywhere

    def test_label_profiles_must_match_model_classes(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "lp.yaml"
            p.write_text("active: a\nprofiles:\n  a: {labels: {Organic: X, Plastic: Y}}\n")
            with self.assertRaises(ConfigError):
                load_label_profiles(p, CLASSES)
            p.write_text("active: a\nprofiles:\n  a: {labels: {Organic: X, Recyclable: X}}\n")
            with self.assertRaises(ConfigError):
                load_label_profiles(p, CLASSES)

    def test_material_table(self):
        t = load_material_table(CONFIG_DIR / "detector_materials.yaml", CLASSES, COCO)
        self.assertEqual(t.hint("bottle").material, "Unknown")       # PET or glass: not determinable
        self.assertEqual(t.hint("tv").special_stream, "E_WASTE")
        self.assertTrue(t.hint("banana").food)
        self.assertFalse(t.hint("person").waste_item)
        self.assertTrue(t.hint(None).waste_item)                      # whole-image fallback

    def test_material_table_rejects_unknown_detector_class(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "m.yaml"
            p.write_text("classes:\n  soda can: {material: Metal, certainty: likely}\n")
            with self.assertRaises(ConfigError):
                load_material_table(p, CLASSES, COCO)

    def test_recommendations(self):
        t = load_material_table(CONFIG_DIR / "detector_materials.yaml", CLASSES, COCO)
        rec = RecommendationEngine.load()
        self.assertEqual(rec.recommend(t.hint("tv"), "Recyclable", reliable=True).stream, "E_WASTE")
        r = rec.recommend(t.hint("tv"), "Recyclable", reliable=False)
        self.assertEqual(r.stream, "E_WASTE")                         # electronics never go to a household bin
        self.assertTrue(any("dry-recycling" in w for w in r.warnings))
        self.assertEqual(rec.recommend(t.hint("banana"), "Organic", reliable=True).stream, "COMPOST")
        self.assertEqual(rec.recommend(t.hint("bottle"), "Recyclable", reliable=True).basis, "model_class:Recyclable")
        self.assertEqual(rec.recommend(t.hint("bottle"), "Recyclable", reliable=False).stream, "MANUAL_REVIEW")
        self.assertEqual(rec.recommend(t.hint("banana"), "Recyclable", reliable=True, conflict=True).stream, "MANUAL_REVIEW")
        self.assertIn("local", rec.recommend(t.hint("book"), "Recyclable", reliable=True).guidance["en"].lower() + "local")


# ---------------------------------------------------------------- detector
class DetectorTests(unittest.TestCase):
    def test_letterbox_geometry(self):
        x, lb = letterbox(Image.new("RGB", (1280, 640), (255, 0, 0)), 640)
        self.assertEqual(x.shape, (1, 3, 640, 640))
        self.assertEqual(x.dtype, np.float32)
        self.assertAlmostEqual(lb.scale, 0.5)
        self.assertEqual((lb.pad_x, lb.pad_y), (0, 160))
        self.assertAlmostEqual(float(x[0, 1, 0, 0]), 114 / 255, places=5)     # grey padding
        self.assertAlmostEqual(float(x[0, 0, 320, 320]), 1.0, places=5)       # red image content

    def test_nms(self):
        boxes = np.array([[0, 0, 10, 10], [1, 1, 10, 10], [20, 20, 30, 30]], float)
        self.assertEqual(nms(boxes, np.array([0.9, 0.8, 0.7]), 0.5), [0, 2])

    def test_decode_maps_boxes_back_to_the_original_image(self):
        # 1280x640 image -> scale 0.5, pad_y 160. An object at original (100,100)-(300,200)
        # is at input (50,210)-(150,260) -> centre (100,235), size (100,50).
        det = YoloDetector(profile(), DetectorSettings(min_box_area_fraction=0.0),
                           runner=FakeRunner(raw_output([(100, 235, 100, 50, 39, 0.9), (400, 400, 60, 60, 46, 0.1)])))
        out = det.detect(Image.new("RGB", (1280, 640)))
        self.assertEqual(len(out.detections), 1)                     # 0.1 is below conf 0.25
        d = out.detections[0]
        self.assertEqual(d.label, "bottle")
        np.testing.assert_allclose(d.box, (100, 100, 300, 200), atol=1e-3)

    def test_agnostic_vs_per_class_nms(self):
        dets = [(320, 320, 100, 100, 39, 0.9), (322, 322, 100, 100, 41, 0.8)]   # bottle + cup on one object
        agn = YoloDetector(profile(), DetectorSettings(agnostic_nms=True, min_box_area_fraction=0.0),
                           runner=FakeRunner(raw_output(dets)))
        per = YoloDetector(profile(), DetectorSettings(agnostic_nms=False, min_box_area_fraction=0.0),
                           runner=FakeRunner(raw_output(dets)))
        img = Image.new("RGB", (640, 640))
        self.assertEqual([d.label for d in agn.detect(img).detections], ["bottle"])
        self.assertEqual(len(per.detect(img).detections), 2)

    def test_small_boxes_dropped_and_counted(self):
        det = YoloDetector(profile(), DetectorSettings(min_box_area_fraction=0.01),
                           runner=FakeRunner(raw_output([(100, 100, 10, 10, 39, 0.9), (300, 300, 200, 200, 39, 0.9)])))
        out = det.detect(Image.new("RGB", (640, 640)))
        self.assertEqual((len(out.detections), out.suppressed_small), (1, 1))

    def test_empty_output(self):
        det = YoloDetector(profile(), runner=FakeRunner(raw_output([])))
        self.assertEqual(det.detect(Image.new("RGB", (64, 48))).detections, [])

    def test_contract_and_output_errors(self):
        with self.assertRaises(DetectorError) as cm:
            YoloDetector(profile(), runner=FakeRunner(raw_output([], nc=3), nc=3))
        self.assertEqual(cm.exception.code, "DETECTOR_CONTRACT_MISMATCH")
        bad = raw_output([(1, 1, 1, 1, 0, 0.9)])
        bad[0, 0, 1] = np.nan
        det = YoloDetector(profile(), runner=FakeRunner(bad))
        with self.assertRaises(DetectorError):
            det.detect(Image.new("RGB", (64, 64)))

    def test_missing_file_and_hash(self):
        p = DetectorProfile("x", Path("/nonexistent/x.onnx"), "onnxruntime", 640, COCO, Path("x"))
        with self.assertRaises(DetectorError) as cm:
            YoloDetector(p)
        self.assertEqual(cm.exception.code, "DETECTOR_NOT_FOUND")
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "m.onnx"
            f.write_bytes(b"not a model")
            p = DetectorProfile("x", f, "onnxruntime", 640, COCO, Path("x"), sha256="0" * 64)
            with self.assertRaises(DetectorError) as cm:
                YoloDetector(p)
            self.assertEqual(cm.exception.code, "DETECTOR_HASH_MISMATCH")

    def test_shipped_config_and_settings_validation(self):
        cfg = load_detector_config()
        self.assertIn(cfg.active, cfg.profiles)
        for p in cfg.profiles.values():
            self.assertEqual(len(p.classes), 80)
            self.assertIsNotNone(p.sha256)
            self.assertIn("not waste", p.trained_on)
        with self.assertRaises(ConfigError):
            DetectorSettings.from_dict({"conf_threshold": 1.5})
        with self.assertRaises(ConfigError):
            DetectorSettings.from_dict({"bogus": 1})


class ShippedArtifactTests(unittest.TestCase):
    def test_shipped_detector_files_match_their_pins(self):
        from zyqora.detection.yolo import file_sha256

        for p in load_detector_config().profiles.values():
            if not p.file.is_file():
                self.skipTest(f"{p.file.name} not present (models are fetched separately)")
            self.assertEqual(file_sha256(p.file), p.sha256, p.name)

    def test_taco_material_table_is_valid(self):
        t = load_material_table(CONFIG_DIR / "detector_materials_taco.yaml", CLASSES)
        self.assertEqual(t.hint("Can").material, "Metal")
        self.assertEqual(t.hint("Battery").special_stream, "E_WASTE")
        self.assertEqual(len(t.classes), 28)


class DetectionMetricTests(unittest.TestCase):
    def test_perfect_and_imperfect(self):
        gt = [GroundTruth((0, 0, 10, 10), "bottle"), GroundTruth((20, 20, 40, 40), "bottle")]
        perfect = evaluate([ImageRecord("a", gt, [Prediction(g.box, "bottle", 0.9) for g in gt])], ["bottle"])
        self.assertAlmostEqual(perfect.map50, 1.0, places=6)
        self.assertAlmostEqual(perfect.map50_95, 1.0, places=6)
        half = evaluate([ImageRecord("a", gt, [Prediction((0, 0, 10, 10), "bottle", 0.9),
                                               Prediction((100, 100, 120, 120), "bottle", 0.8)])], ["bottle"])
        self.assertAlmostEqual(half.recall, 0.5)
        self.assertAlmostEqual(half.precision, 0.5)
        self.assertLess(half.map50, 0.6)

    def test_difficult_ground_truth_is_ignored(self):
        gt = [GroundTruth((0, 0, 10, 10), "bottle"), GroundTruth((20, 20, 40, 40), "bottle", difficult=True)]
        rep = evaluate([ImageRecord("a", gt, [Prediction((0, 0, 10, 10), "bottle", 0.9),
                                              Prediction((20, 20, 40, 40), "bottle", 0.8)])], ["bottle"])
        self.assertEqual(rep.classes[0].n_gt, 1)
        self.assertEqual((rep.classes[0].tp_at_conf, rep.classes[0].fp_at_conf), (1, 0))
        self.assertAlmostEqual(rep.map50, 1.0, places=6)


# ---------------------------------------------------------------- mixed pipeline
class MixedPipelineTests(unittest.TestCase):
    def test_mixed_image(self):
        dets = [
            (100, 100, 80, 120, 39, 0.90),   # bottle   -> Recyclable, material unknown
            (300, 100, 80, 80, 46, 0.85),    # banana   -> classifier says Recyclable: CONFLICT
            (500, 300, 120, 100, 62, 0.80),  # tv       -> e-waste stream
            (300, 450, 100, 300, 0, 0.95),   # person   -> not waste, not classified
            (500, 500, 60, 60, 46, 0.70),    # banana   -> low confidence: REVIEW
        ]
        mp = mixed_pipeline(dets, [("Recyclable", 0.99, "ACCEPT"), ("Recyclable", 0.98, "ACCEPT"),
                                   ("Recyclable", 0.97, "ACCEPT"), ("Organic", 0.70, "REVIEW")])
        r = mp.analyze(Image.new("RGB", (640, 640)), "mix.jpg")
        objs = {o["index"]: o for o in r["objects"]}
        self.assertEqual(len(objs), 5)
        self.assertEqual(len(mp.classifier.calls), 4)                    # person not classified
        by_label = [(o["detection_label"], o["decision"]) for o in r["objects"]]
        self.assertIn(("person", "IGNORED"), by_label)
        bottle = next(o for o in r["objects"] if o["detection_label"] == "bottle")
        self.assertEqual((bottle["decision"], bottle["organiser_category"], bottle["material"]["category"]),
                         ("ACCEPT", "Recyclable", "Unknown"))
        self.assertEqual(bottle["recommendation"]["stream"], "RECYCLE")
        self.assertTrue(bottle["green_score"]["scored"])
        conflict = [o for o in r["objects"] if o["detection_label"] == "banana" and o["conflict"]]
        self.assertEqual(len(conflict), 1)
        self.assertEqual(conflict[0]["decision"], "REVIEW")
        self.assertEqual(conflict[0]["organiser_category"], "Unknown")
        self.assertEqual(conflict[0]["classification"]["model_label"], "Recyclable")   # model label kept
        self.assertFalse(conflict[0]["green_score"]["scored"])
        tv = next(o for o in r["objects"] if o["detection_label"] == "tv")
        self.assertEqual(tv["recommendation"]["stream"], "E_WASTE")
        s = r["summary"]
        self.assertEqual((s["detected_objects"], s["waste_items"], s["accepted"], s["review"], s["ignored_non_waste"]),
                         (5, 4, 2, 2, 1))
        self.assertEqual(s["by_profile"]["food"], {"Waste": 2, "Unknown": 2})
        self.assertEqual(r["image_green_score"]["scored_items"], 2)
        self.assertTrue(r["image_green_score"]["partial"])

    def test_whole_image_fallback_when_nothing_detected(self):
        mp = mixed_pipeline([], [("Organic", 0.99, "ACCEPT")])
        r = mp.analyze(Image.new("RGB", (320, 240)), "peel.jpg")
        self.assertEqual(len(r["objects"]), 1)
        o = r["objects"][0]
        self.assertEqual((o["source"], o["detection_label"], o["box"]), ("whole_image", None, [0.0, 0.0, 320.0, 240.0]))
        self.assertTrue(r["summary"]["whole_image_fallback"])
        self.assertTrue(r["notes"])
        mp2 = mixed_pipeline([], [], DetectorSettings(whole_image_fallback=False))
        r2 = mp2.analyze(Image.new("RGB", (320, 240)), "x.jpg")
        self.assertEqual(r2["objects"], [])
        self.assertFalse(r2["image_green_score"]["valid"])

    def test_object_limit(self):
        dets = [(50 + 100 * i, 300, 60, 60, 39, 0.9 - 0.01 * i) for i in range(5)]
        mp = mixed_pipeline(dets, [], DetectorSettings(max_objects_classified=2, min_box_area_fraction=0.0))
        r = mp.analyze(Image.new("RGB", (640, 640)), "many.jpg")
        self.assertEqual(sum(1 for o in r["objects"] if o["decision"] == "SKIPPED"), 3)
        self.assertEqual(len(mp.classifier.calls), 2)
        self.assertTrue(any("not classified" in n for n in r["notes"]))

    def test_padded_crop_box_is_clipped(self):
        self.assertEqual(padded_crop_box((0, 0, 100, 50), (120, 60), 0.1), (0, 0, 110, 55))
        self.assertEqual(padded_crop_box((10, 10, 10.5, 10.5), (100, 100), 0.0)[2:], (12, 12))


class DetectionStoreTests(unittest.TestCase):
    def setUp(self):
        from zyqora.api.detection_store import DetectionStore

        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        self.store = DetectionStore(d / "db.sqlite3", d / "imgs")
        self.mp = mixed_pipeline([(100, 100, 80, 120, 39, 0.9), (300, 300, 100, 100, 0, 0.9)],
                                 [("Recyclable", 0.99, "ACCEPT")])

    def tearDown(self):
        self.tmp.cleanup()

    def _add(self):
        r = self.mp.analyze(Image.new("RGB", (640, 640)), "a.jpg")
        img_dir = self.store.images_dir

        def saver(i):
            img_dir.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (64, 64)).save(img_dir / f"{i}.jpg")
            return f"{i}.jpg"
        self.mp.classifier.script = [("Recyclable", 0.99, "ACCEPT")]
        return self.store.add(r, source="upload", user_id=None, classifier_profile="waste", model_version="v",
                              image_saver=saver)

    def test_add_get_list_delete_cascade(self):
        i = self._add()
        self.assertEqual(self.store.schema_version, 1)
        got = self.store.get(i)
        self.assertEqual(got["id"], i)
        self.assertEqual(got["image_url"], f"/api/detections/{i}/image")
        items, total = self.store.list()
        self.assertEqual(total, 1)
        self.assertEqual([o["decision"] for o in items[0]["objects"]], ["ACCEPT", "IGNORED"])
        self.assertEqual(self.store.list(q="bottle")[1], 1)
        self.assertEqual(self.store.list(q="zzz")[1], 0)
        import sqlite3
        conn = sqlite3.connect(self.store.db_path)
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM green_scores").fetchone()[0], 1)
        self.assertTrue(self.store.delete(i))
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM detected_objects").fetchone()[0], 0)
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM green_scores").fetchone()[0], 0)
        conn.close()
        self.assertIsNone(self.store.image_file(i))
        self.assertFalse(self.store.delete(i))

    def test_migration_is_idempotent_and_settings(self):
        self.assertEqual(self.store.migrate(), 1)
        self.store.put_settings({"label_profile": "food", "conf_threshold": 0.3})
        self.assertEqual(self.store.get_settings(), {"label_profile": "food", "conf_threshold": 0.3})
        self.store.put_settings({"label_profile": None})
        self.assertEqual(self.store.get_settings(), {"conf_threshold": 0.3})

    def test_summary_csv_and_purge(self):
        self._add()
        self.store.add_metrics({"cpu_percent": 12.0, "memory": {"used_percent": 30.0}, "temperature_c": None}, "poll", 5.0)
        s = self.store.summary(load_label_profiles(model_classes=CLASSES))
        self.assertEqual((s["images"], s["detected_objects"], s["waste_items"]), (1, 2, 1))
        self.assertEqual(s["by_profile"]["recyclable"]["counts"], {"Recyclable": 1})
        self.assertEqual(len(s["system_metrics"]), 1)
        self.assertIsNone(s["system_metrics"][0]["measured_power_watts"])
        self.assertIn("bottle", self.store.export_csv())
        self.assertEqual(self.store.purge_older_than(1)["detections_deleted"], 0)
        res = self.store.purge_older_than(-1)          # cutoff in the future -> everything goes
        self.assertEqual(res["detections_deleted"], 1)


class PmicParseTests(unittest.TestCase):
    def test_parse(self):
        from zyqora.api.mixed_service import parse_pmic

        text = ("   VDD_CORE_A current(7)=2.00000000A\n   VDD_CORE_V volt(15)=0.80000000V\n"
                "  3V3_SYS_A current(1)=0.10000000A\n  3V3_SYS_V volt(9)=3.30000000V\n  EXT5V_V volt(24)=5.1V\n")
        r = parse_pmic(text)
        self.assertAlmostEqual(r["rails_watts"], 1.6 + 0.33, places=3)
        self.assertEqual(r["rails"], 2)
        self.assertIsNone(parse_pmic("garbage"))


if __name__ == "__main__":
    unittest.main()
