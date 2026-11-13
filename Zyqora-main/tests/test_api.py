"""API tests with a FAKE ExecuTorch binding (dependency injection). They test the HTTP layer,
validation, decoding, EcoGuard, persistence and error handling; they are NOT evidence that the real
model predicts correctly (that is scripts/verify_waste_model.py on the Pi)."""

import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from tests.helpers import FakeModule, ensure_torch_like, fake_portable_lib, two_logits, write_fake_pte

try:
    from fastapi.testclient import TestClient

    from zyqora.api.app import create_app
    from zyqora.api.service import ServiceSettings, ZyqoraService
    HAVE_FASTAPI = True
except ImportError:  # pragma: no cover
    HAVE_FASTAPI = False


class ScriptedModule(FakeModule):
    """Returns queued 2-class logits; the first call is the server's warm-up."""

    def __init__(self, queue):
        super().__init__(np.zeros((1, 2), np.float32))
        self.queue = list(queue)

    def forward(self, inputs):
        self.calls += 1
        arr = (self.queue.pop(0) if self.queue else two_logits(0.5)).reshape(1, 2)
        return [type("Out", (), {"numpy": lambda s, a=arr: a})()]


def jpeg_bytes(color=(90, 140, 60), size=(320, 240)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="JPEG")
    return buf.getvalue()


@unittest.skipUnless(HAVE_FASTAPI, "fastapi/httpx not installed (pip install -r requirements-dev.txt)")
class ApiTests(unittest.TestCase):
    def setUp(self):
        ensure_torch_like()
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)
        self.pte = self.d / "waste.pte"
        write_fake_pte(self.pte)
        self.models = self.d / "models.yaml"
        self.models.write_text(f"""
active: waste_mobilenetv2
fallback: null
profiles:
  waste_mobilenetv2:
    description: test profile
    decoder: direct_classes
    pte: {self.pte}
    contract: config/model_waste.json
    classes: config/classes_waste.json
    ecoguard: {{threshold: 0.95, require_top1_mapped: false}}
""")
        self.module = ScriptedModule([two_logits(0.5)])  # warm-up output
        self.client = self.make_client(self.module)

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.tmp.cleanup()

    def settings(self, module, **kw):
        lib = fake_portable_lib(module)
        base = dict(models_config=self.models, db_path=self.d / "web.sqlite3", thumbs_dir=self.d / "thumbs",
                    greenscore_db=self.d / "gs.sqlite3", hardware_config=self.d / "none.yaml",
                    test_summary=self.d / "none.json", benchmark_dir=self.d, portable_lib=lib)
        base.update(kw)
        return ServiceSettings(**base)

    def make_client(self, module, static_dir=None, cors=None, **kw):
        self.service = ZyqoraService(self.settings(module, **kw))
        c = TestClient(create_app(self.service, static_dir=static_dir, cors_origins=cors))
        c.__enter__()
        return c

    def classify(self, p_organic, **form):
        self.module.queue.append(two_logits(p_organic))
        return self.client.post("/api/inference", files={"file": ("item.jpg", jpeg_bytes(), "image/jpeg")}, data=form)

    # ------------------------------------------------------------------ health / model
    def test_health_and_model_status(self):
        h = self.client.get("/api/health").json()
        self.assertTrue(h["model_ready"])
        self.assertEqual(h["profile"], "waste_mobilenetv2")
        m = self.client.get("/api/model/status").json()
        self.assertEqual(m["classes"], ["Organic", "Recyclable"])
        self.assertEqual(m["output_shape"], [1, 2])
        self.assertEqual(m["preprocessing"]["resize"], [224, 224])
        self.assertIsNone(m["preprocessing"]["center_crop"])
        self.assertEqual(m["ecoguard"]["threshold"], 0.95)
        self.assertTrue(m["xnnpack"])
        self.assertEqual(len(m["sha256"]), 64)

    def test_missing_model_reports_model_not_found(self):
        self.client.__exit__(None, None, None)
        self.pte.unlink()
        self.client = self.make_client(ScriptedModule([]))
        self.assertEqual(self.client.get("/api/health").json()["model_code"], "MODEL_NOT_FOUND")
        r = self.client.post("/api/inference", files={"file": ("a.jpg", jpeg_bytes(), "image/jpeg")})
        self.assertEqual(r.status_code, 503)
        self.assertEqual(r.json()["error"]["code"], "MODEL_NOT_FOUND")
        self.assertIn("expected_pte", r.json()["error"]["details"])

    def test_wrong_model_is_refused_not_substituted(self):
        self.client.__exit__(None, None, None)
        self.client = self.make_client(FakeModule(np.zeros((1, 1000), np.float32)))
        st = self.client.get("/api/model/status").json()
        self.assertFalse(st["ready"])
        self.assertEqual(st["code"], "MODEL_CONTRACT_MISMATCH")

    # ------------------------------------------------------------------ inference
    def test_accept_response_schema_and_mapping(self):
        r = self.classify(0.99)
        self.assertEqual(r.status_code, 200, r.text)
        d = r.json()
        self.assertEqual(d["predicted_category"], "Organic")       # index 0
        self.assertAlmostEqual(d["confidence"], 0.99, places=5)
        self.assertAlmostEqual(sum(d["class_scores"].values()), 1.0, places=6)
        self.assertEqual(d["ecoguard"]["status"], "ACCEPT")
        self.assertEqual(d["disposal_action"], "COMPOST")
        self.assertTrue(d["sort"]["actuate"])
        self.assertFalse(d["sort"]["hardware"])                     # simulation by default
        self.assertTrue(d["guidance"]["ta"])
        self.assertEqual(d["model"]["runtime"], "ExecuTorch")
        self.assertGreaterEqual(d["timing"]["inference_ms"], 0)
        self.assertIn("not a calibrated probability", d["score_note"])
        r2 = self.classify(0.01)
        self.assertEqual((r2.json()["predicted_category"], r2.json()["disposal_action"]), ("Recyclable", "RECYCLE"))

    def test_threshold_boundary_and_review_never_sorted(self):
        self.assertEqual(self.classify(0.95).json()["ecoguard"]["status"], "ACCEPT")
        d = self.classify(0.9499).json()
        self.assertEqual(d["ecoguard"]["status"], "REVIEW")
        self.assertEqual(d["ecoguard"]["reasons"], ["LOW_CONFIDENCE"])
        self.assertEqual(d["disposal_action"], "MANUAL_REVIEW")
        self.assertFalse(d["sort"]["actuate"])
        self.assertEqual(d["predicted_category"], "Organic")         # prediction kept, not a third class
        self.assertTrue(d["ecoguard"]["explanation"]["ta"])

    def test_upload_validation(self):
        bad = self.client.post("/api/inference", files={"file": ("x.jpg", b"not an image", "image/jpeg")})
        self.assertEqual((bad.status_code, bad.json()["error"]["code"]), (422, "INVALID_IMAGE"))
        txt = self.client.post("/api/inference", files={"file": ("x.txt", b"hello", "text/plain")})
        self.assertEqual((txt.status_code, txt.json()["error"]["code"]), (415, "UNSUPPORTED_MEDIA_TYPE"))
        empty = self.client.post("/api/inference", files={"file": ("x.jpg", b"", "image/jpeg")})
        self.assertEqual(empty.json()["error"]["code"], "EMPTY_FILE")
        gif = io.BytesIO()
        Image.new("RGB", (10, 10)).save(gif, format="GIF")
        g = self.client.post("/api/inference", files={"file": ("x.jpg", gif.getvalue(), "image/jpeg")})
        self.assertEqual(g.json()["error"]["code"], "INVALID_IMAGE")
        self.client.__exit__(None, None, None)
        self.module = ScriptedModule([two_logits(0.5)])
        self.client = self.make_client(self.module, max_upload_bytes=1000)
        big = self.client.post("/api/inference", files={"file": ("big.jpg", jpeg_bytes(size=(800, 800)), "image/jpeg")})
        self.assertEqual((big.status_code, big.json()["error"]["code"]), (413, "FILE_TOO_LARGE"))
        self.assertEqual(self.client.get("/api/history").json()["total"], 0)  # nothing persisted for rejects

    def test_png_webp_and_exif_orientation(self):
        for fmt in ("PNG", "WEBP"):
            buf = io.BytesIO()
            Image.new("RGBA", (100, 60), (1, 2, 3, 0)).save(buf, format=fmt)
            self.module.queue.append(two_logits(0.99))
            r = self.client.post("/api/inference", files={"file": (f"x.{fmt.lower()}", buf.getvalue(), f"image/{fmt.lower()}")})
            self.assertEqual(r.status_code, 200, r.text)

    def test_server_busy_is_bounded(self):
        for _ in range(self.service.settings.max_pending + 1):
            self.service._slots.acquire()
        r = self.classify(0.99)
        self.assertEqual((r.status_code, r.json()["error"]["code"]), (503, "SERVER_BUSY"))

    # ------------------------------------------------------------------ batch / history / analytics
    def test_batch_with_ground_truth_metrics(self):
        self.module.queue += [two_logits(0.99), two_logits(0.02), two_logits(0.6)]
        files = [("files", (f"i{i}.jpg", jpeg_bytes(), "image/jpeg")) for i in range(3)]
        files.append(("files", ("bad.jpg", b"xx", "image/jpeg")))
        truths = json.dumps({"i0.jpg": "Organic", "i1.jpg": "Organic", "i2.jpg": "Residual"})
        r = self.client.post("/api/inference/batch", files=files, data={"ground_truths": truths})
        self.assertEqual(r.status_code, 200, r.text)
        d = r.json()
        self.assertEqual(len(d["results"]), 3)
        self.assertEqual(d["errors"][0]["error"]["code"], "INVALID_IMAGE")
        acc = d["summary"]["labelled_accuracy"]
        self.assertEqual((acc["labelled_images"], acc["correct"]), (3, 1))
        self.assertEqual(acc["metrics"]["per_class"]["Residual"]["recall"], 0.0)  # unsupported truth counted
        self.assertEqual(self.client.get(f"/api/batches/{d['batch_id']}").json()["summary"]["total"], 3)

    def test_history_persists_across_restart_and_analytics(self):
        self.classify(0.99)
        self.classify(0.6)
        self.client.__exit__(None, None, None)
        self.module = ScriptedModule([two_logits(0.5)])
        self.client = self.make_client(self.module)
        h = self.client.get("/api/history").json()
        self.assertEqual(h["total"], 2)
        self.assertEqual(h["items"][0]["ecoguard_status"], "REVIEW")      # newest first
        img = self.client.get(h["items"][0]["image_url"])
        self.assertEqual((img.status_code, img.headers["content-type"]), (200, "image/jpeg"))
        a = self.client.get("/api/analytics/summary").json()
        self.assertEqual((a["total"], a["accepted"], a["review"]), (2, 1, 1))
        self.assertEqual(a["by_category"], {"Organic": 2})
        self.assertIsNone(a["labelled_accuracy"])                          # no labels -> no accuracy
        self.assertEqual(self.client.get("/api/history?status=ACCEPT").json()["total"], 1)
        csv_text = self.client.get("/api/history/export.csv").text
        self.assertEqual(len(csv_text.strip().splitlines()), 3)

    # ------------------------------------------------------------------ greenscore
    def test_greenscore_points_only_after_verification(self):
        ev = self.classify(0.99, user_id="U01").json()["greenscore_event_id"]
        self.assertEqual(self.client.get("/api/greenscore/U01").json()["total_points"], 0)
        v = self.client.post("/api/greenscore/verify", json={"event_id": ev, "observed_action": "COMPOST"}).json()
        self.assertEqual((v["status"], v["awarded_points"]), ("VERIFIED_CORRECT", 10))
        dup = self.client.post("/api/greenscore/verify", json={"event_id": ev, "observed_action": "COMPOST"})
        self.assertEqual((dup.status_code, dup.json()["error"]["code"]), (409, "ALREADY_VERIFIED"))
        rev = self.classify(0.6, user_id="U01").json()["greenscore_event_id"]
        need = self.client.post("/api/greenscore/verify", json={"event_id": rev, "observed_action": "COMPOST"})
        self.assertEqual(need.json()["error"]["code"], "GREENSCORE_ERROR")  # REVIEW needs the correct bin
        u = self.client.get("/api/greenscore/U01").json()
        self.assertEqual((u["total_points"], u["contribution_by_bin"]), (10, {"COMPOST": 10}))
        self.assertEqual(self.classify(0.99, user_id="bad id!").json()["error"]["code"], "INVALID_USER_ID")

    # ------------------------------------------------------------------ device / voice / misc
    def test_device_metrics_never_fabricated(self):
        d = self.client.get("/api/device/metrics").json()
        for key in ("cpu_percent", "temperature_c", "memory", "uptime_s", "process_rss_mb", "is_raspberry_pi"):
            self.assertIn(key, d)
        if not Path("/sys/class/thermal/thermal_zone0/temp").exists():
            self.assertIsNone(d["temperature_c"])
        self.assertFalse(d["sorter"]["hardware"])

    def test_voice_clip_and_messages(self):
        self.assertEqual(self.client.get("/api/voice/ta/RECYCLE.wav").headers["content-type"], "audio/wav")
        self.assertEqual(self.client.get("/api/voice/fr/RECYCLE.wav").status_code, 404)
        self.assertIn("ta", self.client.get("/api/voice/messages").json()["MANUAL_REVIEW"])

    def test_camera_unavailable_and_unknown_route(self):
        r = self.client.post("/api/camera/capture")
        self.assertEqual((r.status_code, r.json()["error"]["code"]), (503, "CAMERA_UNAVAILABLE"))
        self.assertEqual(self.client.get("/api/does-not-exist").json()["error"]["code"], "NOT_FOUND")

    def test_simulated_camera_capture(self):
        cam_dir = self.d / "cam"
        cam_dir.mkdir()
        Image.new("RGB", (64, 48)).save(cam_dir / "f.jpg")
        self.client.__exit__(None, None, None)
        self.module = ScriptedModule([two_logits(0.5), two_logits(0.99)])
        self.client = self.make_client(self.module, camera=f"folder:{cam_dir}")
        d = self.client.post("/api/camera/capture").json()
        self.assertEqual(d["source"], "camera-simulated")
        self.assertIn("capture_ms", d)

    def test_static_spa_and_cors(self):
        self.client.__exit__(None, None, None)
        dist = self.d / "dist"
        (dist / "assets").mkdir(parents=True)
        (dist / "index.html").write_text("<html>zyqora</html>")
        (dist / "assets" / "a.js").write_text("x")
        self.client = self.make_client(ScriptedModule([two_logits(0.5)]), static_dir=dist,
                                       cors=["http://localhost:5173"])
        self.assertIn("zyqora", self.client.get("/live").text)              # SPA fallback
        self.assertEqual(self.client.get("/assets/a.js").text, "x")
        self.assertEqual(self.client.get("/../../etc/passwd").text, "<html>zyqora</html>")
        ok = self.client.get("/api/health", headers={"Origin": "http://localhost:5173"})
        self.assertEqual(ok.headers.get("access-control-allow-origin"), "http://localhost:5173")
        bad = self.client.get("/api/health", headers={"Origin": "http://evil.example"})
        self.assertIsNone(bad.headers.get("access-control-allow-origin"))


if __name__ == "__main__":
    unittest.main()
