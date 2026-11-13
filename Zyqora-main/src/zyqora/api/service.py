"""ZyqoraService: one loaded model per process, shared by every API request.

* The model is loaded once (zyqora.profiles.load_with_fallback: contract + decoder + output-shape checks).
* Inference is serialised by a lock (ExecuTorch modules are not assumed thread-safe); at most
  `max_pending` requests may wait, further requests get SERVER_BUSY instead of piling up.
* A missing/invalid model is reported with an explicit error code; no other model is substituted
  unless the server was started with --allow-fallback, and then every response says so.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Optional

import numpy as np
from PIL import Image

from ..config import ConfigError, load_app_config, load_yaml
from ..ecoguard import EcoGuardConfig
from ..explain import REASON_TEXT
from ..model_runtime import ModelLoadError, RuntimeUnavailableError
from ..paths import CONFIG_DIR, DEFAULT_VOICE_CONFIG, OUTPUTS_DIR, PROJECT_ROOT, resolve
from ..pipeline import ZyqoraPipeline
from ..preprocessing import ImageLoadError, load_rgb_bytes
from ..profiles import DEFAULT_MODELS_CONFIG, LoadedModel, file_sha256, load_models_config, load_with_fallback
from ..schemas import ACCEPT, MANUAL_REVIEW
from ..sorter import ActuatorError, decide_sort, make_sorter
from .store import HistoryStore, now_iso
from .telemetry import Telemetry

log = logging.getLogger("zyqora.api")

# Tamil explanations of EcoGuard review reasons (draft; needs review by a fluent Tamil speaker).
REASON_TEXT_TA = {
    "LOW_CONFIDENCE": "மாதிரியின் நம்பக மதிப்பெண் EcoGuard வரம்பை விடக் குறைவாக உள்ளது",
    "AMBIGUOUS_CATEGORY_SCORES": "இரண்டு வகைகளின் மதிப்பெண்களும் மிக நெருக்கமாக உள்ளன",
    "TOP1_CLASS_UNMAPPED": "அடையாளம் காணப்பட்ட பொருளுக்கு கழிவு விதி இல்லை",
    "NO_VALID_CATEGORY_MAPPING": "இந்தப் பொருள் ஆதரிக்கப்படும் எந்த கழிவு வகையுடனும் பொருந்தவில்லை",
    "INVALID_INFERENCE_RESULT": "மாதிரியின் வெளியீடு செல்லாதது",
    "IMAGE_LOAD_FAILED": "படத்தைப் படிக்க முடியவில்லை",
}


class ServiceError(Exception):
    """Error with a stable machine-readable code and an HTTP status."""

    def __init__(self, code: str, message: str, status: int = 400, details: Optional[dict] = None):
        super().__init__(message)
        self.code, self.message, self.status, self.details = code, message, status, details or {}


@dataclass
class ServiceSettings:
    profile: Optional[str] = None
    models_config: Path = DEFAULT_MODELS_CONFIG
    allow_fallback: bool = False
    num_threads: Optional[int] = None
    db_path: Path = OUTPUTS_DIR / "web" / "zyqora_web.sqlite3"
    thumbs_dir: Path = OUTPUTS_DIR / "web" / "thumbs"
    greenscore_db: Optional[Path] = None
    store_thumbnails: bool = True
    max_upload_bytes: int = 10 * 1024 * 1024
    max_pending: int = 4
    max_batch_files: int = 50
    hardware_config: Path = CONFIG_DIR / "hardware.yaml"
    camera: Optional[str] = None             # overrides hardware.yaml camera.backend
    enable_actuator: bool = False
    metrics_file: Path = CONFIG_DIR / "model_metrics.json"
    test_summary: Path = OUTPUTS_DIR / "test_waste" / "summary.json"
    benchmark_dir: Path = OUTPUTS_DIR / "benchmark"
    portable_lib: Optional[ModuleType] = None  # test injection only


@dataclass
class ModelState:
    ready: bool = False
    code: Optional[str] = None
    message: Optional[str] = None
    loaded: Optional[LoadedModel] = None
    profile: Optional[str] = None
    sha256: Optional[str] = None
    loaded_at: Optional[str] = None
    load_ms: Optional[float] = None
    expected_pte: Optional[str] = None


@dataclass
class ZyqoraService:
    settings: ServiceSettings = field(default_factory=ServiceSettings)

    def __post_init__(self):
        self.app = load_app_config()
        self.models = load_models_config(self.app, self.settings.models_config)
        self.store = HistoryStore(self.settings.db_path)
        self.telemetry = Telemetry()
        self.state = ModelState()
        self.voice_cfg = load_yaml(DEFAULT_VOICE_CONFIG)
        self.hw = load_yaml(self.settings.hardware_config) if self.settings.hardware_config.is_file() else {}
        self._model_lock = threading.Lock()
        self._sort_lock = threading.Lock()
        self._slots = threading.BoundedSemaphore(self.settings.max_pending + 1)
        self._camera = None
        self._camera_lock = threading.Lock()
        self.camera_error: Optional[str] = None
        self.sorter = make_sorter(self.hw.get("actuator", {}), self.settings.enable_actuator)
        self._greenscore = None
        self.started_at = now_iso()
        self.last_inference_ms: Optional[float] = None

    # ------------------------------------------------------------------ model
    def load_model(self) -> ModelState:
        name = self.settings.profile or self.models.active
        st = ModelState(profile=name)
        try:
            profile = self.models.get(name)
        except ConfigError as exc:
            self.state = ModelState(code="CONFIG_ERROR", message=str(exc), profile=name)
            return self.state
        st.expected_pte = str(profile.pte.relative_to(PROJECT_ROOT) if profile.pte.is_relative_to(PROJECT_ROOT)
                              else profile.pte)
        if not profile.pte.is_file() and not self.settings.allow_fallback:
            st.code, st.message = "MODEL_NOT_FOUND", f"Model file not found: {st.expected_pte}"
            self.state = st
            log.error(st.message)
            return st
        t0 = time.perf_counter()
        try:
            loaded = load_with_fallback(self.models, name, allow_fallback=self.settings.allow_fallback,
                                        num_threads=self.settings.num_threads or self.app.num_threads,
                                        require_backend=self.app.require_backend,
                                        portable_lib=self.settings.portable_lib)
            loaded.classifier.infer(np.zeros(loaded.spec.input_shape, dtype=np.float32))  # warm-up
        except RuntimeUnavailableError as exc:
            st.code, st.message = "RUNTIME_UNAVAILABLE", str(exc)
        except ModelLoadError as exc:
            msg = str(exc)
            st.code = ("MODEL_NOT_FOUND" if "not found" in msg.lower() else
                       "MODEL_CONTRACT_MISMATCH" if "produces output" in msg or "expects input" in msg else
                       "MODEL_HASH_MISMATCH" if "SHA-256" in msg else "MODEL_LOAD_FAILED")
            st.message = msg
        except (ConfigError, ValueError, OSError) as exc:
            st.code, st.message = "CONFIG_ERROR", str(exc)
        else:
            st.ready, st.loaded = True, loaded
            st.profile = loaded.profile.name
            st.sha256 = file_sha256(loaded.profile.pte)
            st.load_ms = (time.perf_counter() - t0) * 1000
            st.loaded_at = now_iso()
            log.info("model ready: %s (%s)", st.profile, loaded.profile.pte.name)
        if not st.ready:
            log.error("model unavailable [%s]: %s", st.code, st.message)
        self.state = st
        return st

    def require_model(self) -> LoadedModel:
        if not self.state.ready or self.state.loaded is None:
            raise ServiceError(self.state.code or "MODEL_UNAVAILABLE",
                               self.state.message or "Model is not loaded", 503,
                               {"expected_pte": self.state.expected_pte})
        return self.state.loaded

    def model_status(self) -> dict[str, Any]:
        st = self.state
        base = {"ready": st.ready, "code": st.code, "message": st.message, "profile": st.profile,
                "expected_pte": st.expected_pte, "active_profile_in_config": self.models.active,
                "available_profiles": sorted(self.models.profiles)}
        if not st.ready or st.loaded is None:
            return base
        lm = st.loaded
        clf = lm.classifier
        rt = clf.describe()
        guard: EcoGuardConfig = lm.guard
        cats = lm.category_set
        spec = lm.spec
        return {
            **base,
            "description": lm.profile.description,
            "decoder": lm.profile.decoder,
            "pte": lm.profile.pte.name,
            "pte_size_bytes": lm.profile.pte.stat().st_size,
            "sha256": st.sha256,
            "sha256_pinned": lm.profile.sha256,
            "precision": "int8" if "int8" in lm.profile.pte.name.lower() else "fp32",
            "input_shape": list(spec.input_shape),
            "output_shape": list(clf.output_shape) if clf.output_shape else None,
            "preprocessing": {"resize": list(spec.resize) if isinstance(spec.resize, tuple) else spec.resize,
                              "center_crop": spec.center_crop, "mean": list(spec.mean), "std": list(spec.std)},
            "classes": list(cats.category_names),
            "actions": dict(cats.categories),
            "unsupported_ground_truth": sorted(getattr(cats, "unsupported_ground_truth", []) or []),
            "runtime": "ExecuTorch",
            "executorch_version": rt["executorch_version"],
            "registered_backends": rt["registered_backends"],
            "xnnpack": "XnnpackBackend" in rt["registered_backends"],
            "threads": rt["thread_count"],
            "thread_control": rt["thread_control"],
            "fallback_used": lm.fallback_used,
            "failed_profiles": lm.failures,
            "ecoguard": {"threshold": guard.threshold, "min_margin": guard.min_margin,
                         "calibrated": guard.calibrated, "frozen": guard.frozen},
            "loaded_at": st.loaded_at,
            "load_ms": st.load_ms,
        }

    # -------------------------------------------------------------- inference
    def classify_bytes(self, data: bytes, filename: str, **kw) -> dict[str, Any]:
        if len(data) > self.settings.max_upload_bytes:
            raise ServiceError("FILE_TOO_LARGE", f"Image is larger than {self.settings.max_upload_bytes // 1048576} MB", 413)
        try:
            img = load_rgb_bytes(data, filename)
        except ImageLoadError as exc:
            raise ServiceError("INVALID_IMAGE", str(exc), 422) from exc
        return self.classify_image(img, filename, **kw)

    def classify_image(self, img: Image.Image, filename: str, *, source: str = "upload", user_id: Optional[str] = None,
                       batch_id: Optional[str] = None, ground_truth: Optional[str] = None,
                       persist: bool = True) -> dict[str, Any]:
        lm = self.require_model()
        if ground_truth:
            canon = lm.category_set.ground_truth_category(ground_truth)
            if canon is None:
                raise ServiceError("INVALID_GROUND_TRUTH", f"Unknown ground-truth category '{ground_truth}'", 422)
            ground_truth = canon
        if user_id:
            self._check_user(user_id)
        if not self._slots.acquire(blocking=False):
            raise ServiceError("SERVER_BUSY", "Too many classification requests are waiting; retry shortly", 503)
        try:
            with self._model_lock:
                pipe = ZyqoraPipeline(lm.classifier, lm.decoder, lm.guard, lm.spec)
                result = pipe.process_image(img, filename)
        finally:
            self._slots.release()
        if result.error:
            raise ServiceError("INFERENCE_FAILED", result.error, 500)
        self.last_inference_ms = result.inference_time_ms

        decision = decide_sort(result.ecoguard_status, result.disposal_action)
        with self._sort_lock:
            try:
                sort_message = self.sorter.execute(decision)
            except ActuatorError as exc:
                sort_message = f"ACTUATOR ERROR: {exc} (item held for supervisor)"
                log.error("actuator failed: %s", exc)

        reasons = [r for r in result.review_reason.split(";") if r]
        rec = {
            "created_at": now_iso(), "source": source, "filename": filename[:255], "batch_id": batch_id,
            "user_id": user_id, "predicted_category": result.predicted_category,
            "confidence": result.predicted_category_score, "class_scores": result.category_scores,
            "ecoguard_status": result.ecoguard_status, "review_reason": result.review_reason,
            "disposal_action": result.disposal_action, "preprocess_ms": result.preprocess_time_ms,
            "inference_ms": result.inference_time_ms, "total_ms": result.end_to_end_time_ms,
            "model_profile": lm.profile.name, "model_sha256": self.state.sha256, "ground_truth": ground_truth,
            "sort_message": sort_message,
        }
        item_id = None
        event_id = None
        if persist:
            rec["has_thumbnail"] = self.settings.store_thumbnails
            item_id = self.store.add(rec)
            if self.settings.store_thumbnails and not self._save_thumbnail(item_id, img):
                self.store.set_thumbnail(item_id, False)  # only on the rare failure path
            if user_id:
                gs = self.greenscore()
                gs.ensure_user(user_id)
                event_id = gs.record_event(user_id, filename[:200], result.predicted_category,
                                           result.disposal_action, result.ecoguard_status, run_id=f"web-{item_id}")
                self.store.set_greenscore_event(item_id, event_id)
        log.info("classified %s -> %s %.3f %s (%.1f ms)", filename, result.predicted_category,
                 result.predicted_category_score or 0, result.ecoguard_status, result.inference_time_ms or 0)
        guard = lm.guard
        return {
            "id": item_id,
            "created_at": rec["created_at"],
            "filename": rec["filename"],
            "source": source,
            "batch_id": batch_id,
            "predicted_category": result.predicted_category,
            "confidence": result.predicted_category_score,
            "class_scores": result.category_scores,
            "score_note": "softmax score of the model output; not a calibrated probability",
            "ecoguard": {
                "status": result.ecoguard_status,
                "threshold": guard.threshold,
                "reasons": reasons,
                "explanation": {
                    "en": "; ".join(REASON_TEXT.get(r, r) for r in reasons),
                    "ta": "; ".join(REASON_TEXT_TA.get(r, r) for r in reasons),
                },
            },
            "disposal_action": result.disposal_action,
            "guidance": self.messages(result.disposal_action),
            "sort": {"action": decision.action, "actuate": decision.actuate, "message": sort_message,
                     "hardware": bool(self.sorter.hardware)},
            "timing": {"preprocess_ms": result.preprocess_time_ms, "inference_ms": result.inference_time_ms,
                       "total_ms": result.end_to_end_time_ms},
            "model": {"profile": lm.profile.name, "description": lm.profile.description, "decoder": lm.profile.decoder,
                      "pte": lm.profile.pte.name, "sha256": self.state.sha256, "runtime": "ExecuTorch",
                      "backend": "XNNPACK" if "XnnpackBackend" in lm.classifier.registered_backends else "portable",
                      "fallback_used": lm.fallback_used, "device": self.telemetry.snapshot_device()},
            "ground_truth": ground_truth,
            "correct": (result.predicted_category == ground_truth) if ground_truth else None,
            "greenscore_event_id": event_id,
            "image_url": f"/api/history/{item_id}/image" if item_id and self.settings.store_thumbnails else None,
        }

    def _save_thumbnail(self, item_id: int, img: Image.Image) -> bool:
        try:
            self.settings.thumbs_dir.mkdir(parents=True, exist_ok=True)
            thumb = img.copy()
            thumb.thumbnail((320, 320))
            thumb.save(self.settings.thumbs_dir / f"{item_id}.jpg", quality=82)
            return True
        except OSError as exc:
            log.warning("thumbnail not saved for %s: %s", item_id, exc)
            return False

    def thumbnail_path(self, item_id: int) -> Optional[Path]:
        p = self.settings.thumbs_dir / f"{int(item_id)}.jpg"
        return p if p.is_file() else None

    # ------------------------------------------------------------------ batch
    def batch_summary(self, batch_id: str) -> dict[str, Any]:
        items, _ = self.store.list(limit=10_000, batch_id=batch_id)
        items.reverse()
        cats = self.state.loaded.category_set.category_names if self.state.loaded else []
        summary = self.store.summary(cats, batch_id=batch_id)
        return {"batch_id": batch_id, "items": items, "summary": summary}

    # --------------------------------------------------------------- camera
    def camera_capture(self, **kw) -> dict[str, Any]:
        from ..camera import CameraError, open_camera

        self.require_model()
        cfg = self.hw.get("camera", {}) or {}
        spec = self.settings.camera or cfg.get("backend", "auto")
        with self._camera_lock:
            try:
                if self._camera is None:
                    self._camera = open_camera(spec, int(cfg.get("width", 640)), int(cfg.get("height", 480)))
                frame = self._camera.capture()
                self.camera_error = None
            except CameraError as exc:
                self.camera_error = str(exc)
                if self._camera is not None:
                    self._camera.close()
                    self._camera = None
                raise ServiceError("CAMERA_UNAVAILABLE", str(exc), 503) from exc
        source = "camera-simulated" if frame.simulated else f"pi-camera:{frame.source}"
        res = self.classify_image(frame.image, frame.name, source=source, **kw)
        res["capture_ms"] = frame.capture_ms
        return res

    def camera_status(self) -> dict[str, Any]:
        cfg = self.hw.get("camera", {}) or {}
        return {"configured_backend": self.settings.camera or cfg.get("backend", "auto"),
                "open": self._camera is not None,
                "simulated": bool(self._camera and self._camera.simulated),
                "last_error": self.camera_error}

    # ------------------------------------------------------------ greenscore
    def greenscore(self):
        from ..greenscore import GreenScore

        if self._greenscore is None:
            db = self.settings.greenscore_db or self.app.greenscore_db
            db.parent.mkdir(parents=True, exist_ok=True)
            actions = self.state.loaded.category_set.actions if self.state.loaded else None
            self._greenscore = GreenScore(db, self.app.points_correct, self.app.points_incorrect, actions)
        return self._greenscore

    @staticmethod
    def _check_user(user_id: str) -> None:
        from ..greenscore import USER_ID_RE

        if not USER_ID_RE.match(user_id):
            raise ServiceError("INVALID_USER_ID", "user id must be 1-32 letters, digits, '_' or '-'", 422)

    # ------------------------------------------------------------------ voice
    def messages(self, key: str) -> dict[str, str]:
        entry = (self.voice_cfg.get("messages") or {}).get(key) or {}
        return {"en": entry.get("en", ""), "ta": entry.get("ta", "")}

    def all_messages(self) -> dict[str, dict[str, str]]:
        return {k: self.messages(k) for k in (self.voice_cfg.get("messages") or {})}

    def clip(self, lang: str, key: str) -> Optional[Path]:
        from ..paths import AUDIO_DIR
        from ..voicesort import LANGUAGES, clip_path

        key = key.upper()
        if lang not in LANGUAGES or key not in (self.voice_cfg.get("messages") or {}):
            return None
        p = clip_path(AUDIO_DIR, lang, key)
        return p if p.is_file() else None

    def play_on_device(self, lang: str, key: str) -> dict[str, Any]:
        from ..voicesort import LANGUAGES, VoiceSort

        key = key.upper()
        if lang not in LANGUAGES or key not in (self.voice_cfg.get("messages") or {}):
            raise ServiceError("INVALID_VOICE_KEY", f"unknown language/message {lang}/{key}", 422)
        vs = VoiceSort(lang, echo=False)
        a = vs.announce(key)  # VoiceSort never raises on playback failure
        return {"played": a.played, "detail": a.detail, "text": a.text, "player": vs.player}

    # ----------------------------------------------------------------- metrics
    def model_metrics(self) -> dict[str, Any]:
        out: dict[str, Any] = {"validation": None, "test": None, "benchmark": None, "live": None}
        if self.settings.metrics_file.is_file():
            out["validation"] = json.loads(self.settings.metrics_file.read_text(encoding="utf-8"))
        if self.settings.test_summary.is_file():
            s = json.loads(self.settings.test_summary.read_text(encoding="utf-8"))
            out["test"] = {
                "scope": s.get("evaluation_scope"), "source": f"{self.settings.test_summary.name} produced by run_zyqora.py",
                "model_profile": s.get("model_profile"), "model_sha256": s.get("model_sha256"),
                "measured_on": (s.get("platform") or {}).get("device_model") or (s.get("platform") or {}).get("machine"),
                "started_utc": s.get("started_utc"), "total_images": s.get("total_images"),
                "overall_accuracy": s.get("overall_accuracy"), "review_rate": s.get("review_rate"),
                "accepted_only_accuracy": s.get("accepted_only_accuracy"),
                "accepted_incorrect": s.get("accepted_incorrect"), "metrics": s.get("metrics"),
                "inference_latency_median_ms": s.get("inference_latency_median_ms"),
                "inference_latency_p95_ms": s.get("inference_latency_p95_ms"),
                "ecoguard": s.get("ecoguard"),
            }
        if self.state.loaded is not None:
            bench = self.settings.benchmark_dir / f"benchmark_{self.state.loaded.profile.pte.stem}.json"
            if bench.is_file():
                b = json.loads(bench.read_text(encoding="utf-8"))
                out["benchmark"] = {
                    "source": bench.name, "platform": b.get("platform"), "warning": b.get("warning"),
                    "results": [{"threads": r.get("thread_count"), "median_ms": r["model_only"]["median_ms"],
                                 "p95_ms": r["model_only"]["p95_ms"], "mean_ms": r["model_only"]["mean_ms"],
                                 "cpu_ms_per_inference": r.get("cpu_ms_per_inference"),
                                 "cpu_percent": r.get("cpu_utilisation_percent_of_one_core"),
                                 "throughput": r.get("model_only_throughput_inferences_per_second"),
                                 "peak_rss_mb": (r.get("memory_mb") or {}).get("end", {}).get("vm_hwm_mb")}
                                for r in b.get("results", [])],
                }
        cats = self.state.loaded.category_set.category_names if self.state.loaded else []
        live = self.store.summary(cats)
        out["live"] = {"scope": "LIVE_DASHBOARD_SESSION", "source": "classifications served by this API",
                       "measured_on": self.telemetry.snapshot_device()["device_model"] or
                       self.telemetry.snapshot_device()["machine"],
                       "total": live["total"], "review_rate": live["review_rate"],
                       "inference_latency": live["inference_latency"], "total_latency": live["total_latency"],
                       "labelled_accuracy": live["labelled_accuracy"]}
        return out

    # --------------------------------------------------------------- shutdown
    def close(self) -> None:
        with self._camera_lock:
            if self._camera is not None:
                self._camera.close()
                self._camera = None
        self.sorter.close()


def resolve_path(p: Optional[str]) -> Optional[Path]:
    return resolve(p) if p else None
