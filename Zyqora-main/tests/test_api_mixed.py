"""API tests for the Zyqora AI 4.0 endpoints with a FAKE detector runner and a FAKE ExecuTorch classifier.
They test HTTP behaviour, validation, persistence and error codes, not model accuracy."""

import io
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from tests.helpers import ensure_torch_like, fake_portable_lib, two_logits, write_fake_pte
from tests.test_api import ScriptedModule
from tests.test_mixed import FakeRunner, raw_output

try:
    from fastapi.testclient import TestClient

    from zyqora.api.app import create_app
    from zyqora.api.service import ServiceSettings, ZyqoraService
    HAVE_FASTAPI = True
except ImportError:  # pragma: no cover
    HAVE_FASTAPI = False


def jpeg(size=(640, 480)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, (120, 120, 120)).save(buf, format="JPEG")
    return buf.getvalue()


@unittest.skipUnless(HAVE_FASTAPI, "fastapi/httpx not installed")
class MixedApiTests(unittest.TestCase):
    def setUp(self):
        ensure_torch_like()
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)
        pte = self.d / "waste.pte"
        write_fake_pte(pte)
        self.models = self.d / "models.yaml"
        self.models.write_text(f"""
active: waste_mobilenetv2
fallback: null
profiles:
  waste_mobilenetv2:
    description: test profile
    decoder: direct_classes
    pte: {pte}
    contract: config/model_waste.json
    classes: config/classes_waste.json
    ecoguard: {{threshold: 0.95, require_top1_mapped: false}}
""")
        # two boxes in a 640x480 image (letterbox: scale 1, pad_y 80): bottle + person
        self.raw = raw_output([(100, 180, 80, 120, 39, 0.9), (400, 300, 100, 200, 0, 0.9)])
        self.module = ScriptedModule([two_logits(0.5), two_logits(0.01)])   # warm-up, then bottle crop
        self.client = self.make_client()

    def make_client(self, factory=None, **kw):
        base = dict(models_config=self.models, db_path=self.d / "web.sqlite3", thumbs_dir=self.d / "thumbs",
                    greenscore_db=self.d / "gs.sqlite3", hardware_config=self.d / "none.yaml",
                    test_summary=self.d / "none.json", benchmark_dir=self.d, portable_lib=fake_portable_lib(self.module),
                    detector_runner_factory=factory or (lambda p: FakeRunner(self.raw)))
        base.update(kw)
        c = TestClient(create_app(ZyqoraService(ServiceSettings(**base))))
        c.__enter__()
        return c

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.tmp.cleanup()

    def test_models_status(self):
        r = self.client.get("/api/models/status").json()
        self.assertTrue(r["classifier"]["ready"])
        self.assertTrue(r["detector"]["ready"])
        self.assertFalse(r["detector"]["class_names_are_waste_categories"])
        self.assertTrue(r["pipeline"]["ready"])
        self.assertEqual(r["label_profiles"]["active"], "organiser")

    def test_detect_persist_list_delete(self):
        r = self.client.post("/api/detect", files={"file": ("mix.jpg", jpeg(), "image/jpeg")})
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        decisions = sorted(o["decision"] for o in body["objects"])
        self.assertEqual(decisions, ["ACCEPT", "IGNORED"])
        bottle = next(o for o in body["objects"] if o["detection_label"] == "bottle")
        self.assertEqual(bottle["classification"]["model_label"], "Recyclable")
        self.assertAlmostEqual(bottle["box"][1], 40, delta=0.5)         # input y1 120 minus pad 80
        self.assertTrue(bottle["green_score"]["scored"])
        self.assertIsNotNone(body["id"])
        self.assertIn("never actuates", body["mode_note"])
        did = body["id"]
        self.assertEqual(self.client.get(f"/api/detections/{did}").json()["id"], did)
        self.assertEqual(self.client.get(f"/api/detections/{did}/image").status_code, 200)
        ann = self.client.get(f"/api/detections/{did}/annotated.jpg")
        self.assertEqual((ann.status_code, ann.headers["content-type"]), (200, "image/jpeg"))
        lst = self.client.get("/api/detections").json()
        self.assertEqual(lst["total"], 1)
        self.assertIn("bottle", self.client.get("/api/detections/export.csv").text)
        an = self.client.get("/api/analytics").json()["detections"]
        self.assertEqual((an["images"], an["waste_items"]), (1, 1))
        self.assertEqual(self.client.delete(f"/api/detections/{did}").status_code, 200)
        self.assertEqual(self.client.get(f"/api/detections/{did}").json()["error"]["code"], "NOT_FOUND")

    def test_invalid_upload(self):
        r = self.client.post("/api/detect", files={"file": ("x.jpg", b"nope", "image/jpeg")})
        self.assertEqual((r.status_code, r.json()["error"]["code"]), (422, "INVALID_IMAGE"))
        r = self.client.post("/api/detect", files={"file": ("x.gif", b"GIF89a", "image/gif")})
        self.assertEqual(r.status_code, 415)

    def test_greenscore_endpoint(self):
        r = self.client.post("/api/greenscore", json={"components": {"recycling_potential": 90, "environmental_impact": 80,
                                                                     "disposal_suitability": 100}}).json()
        self.assertEqual(r["score"], 89)
        r = self.client.post("/api/greenscore", json={"model_class": "Organic", "decision": "REVIEW"}).json()
        self.assertFalse(r["scored"])
        self.assertEqual(self.client.post("/api/greenscore", json={}).status_code, 422)
        self.assertEqual(self.client.post("/api/greenscore", json={"components": {"recycling_potential": 900,
                         "environmental_impact": 1, "disposal_suitability": 1}}).status_code, 422)
        rules = self.client.get("/api/greenscore/rules").json()
        self.assertEqual(rules["weights"]["recycling_potential"], 0.4)

    def test_settings(self):
        s = self.client.get("/api/settings").json()
        self.assertEqual(s["effective"]["conf_threshold"], 0.25)
        r = self.client.put("/api/settings", json={"label_profile": "recyclable", "conf_threshold": 0.4})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["effective"]["label_profile"], "recyclable")
        for bad in ({"conf_threshold": 2}, {"label_profile": "nope"}, {"detector_profile": "nope"}, {"x": 1},
                    {"retention_days": 0}):
            self.assertEqual(self.client.put("/api/settings", json=bad).status_code, 422, bad)

    def test_classification_history_delete(self):
        r = self.client.post("/api/classify", files={"file": ("a.jpg", jpeg(), "image/jpeg")})
        self.assertEqual(r.status_code, 200, r.text)
        cid = r.json()["id"]
        self.assertEqual(self.client.delete(f"/api/history/{cid}").status_code, 200)
        self.assertEqual(self.client.delete(f"/api/history/{cid}").status_code, 404)

    def test_system_metrics_and_benchmarks(self):
        m = self.client.get("/api/system/metrics").json()
        self.assertIn("cpu_percent", m)
        self.assertIsNone(m["measured_power_watts"])                    # never estimated
        self.assertTrue(m["detector_ready"])
        b = self.client.get("/api/benchmarks").json()
        self.assertEqual(set(b), {"classifier_benchmarks", "detector_benchmarks", "pipeline_benchmarks",
                                  "detector_evaluations"})

    def test_missing_detector_is_reported_and_single_mode_still_works(self):
        self.client.__exit__(None, None, None)
        cfg = self.d / "detector.yaml"
        cfg.write_text("""
active: missing
profiles:
  missing: {file: models/does_not_exist.onnx, runtime: onnxruntime, imgsz: 640, classes: config/detector_coco80.json}
""")
        self.module.queue = [two_logits(0.5), two_logits(0.01)]
        self.client = self.make_client(factory=None, detector_runner_factory=None, detector_config=cfg)
        st = self.client.get("/api/models/status").json()
        self.assertFalse(st["detector"]["ready"])
        self.assertEqual(st["detector"]["code"], "DETECTOR_NOT_FOUND")
        r = self.client.post("/api/detect", files={"file": ("mix.jpg", jpeg(), "image/jpeg")})
        self.assertEqual((r.status_code, r.json()["error"]["code"]), (503, "DETECTOR_NOT_FOUND"))
        self.assertEqual(self.client.post("/api/classify", files={"file": ("a.jpg", jpeg(), "image/jpeg")}).status_code, 200)


if __name__ == "__main__":
    unittest.main()
