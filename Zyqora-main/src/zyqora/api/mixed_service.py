"""Mixed-waste (YOLO + MobileNetV2) part of the API service (Zyqora AI 4.0).

Owned by ZyqoraService as ``service.mixed``. Shares the classifier, the inference lock
and the request-slot semaphore with single-image classification, so the two modes never
run ExecuTorch concurrently. A missing detector is reported with an explicit code; the
single-image endpoints keep working without it.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from PIL import Image

from ..config import ConfigError
from ..detection.yolo import DEFAULT_DETECTOR_CONFIG, DetectorError, DetectorSettings, YoloDetector, load_detector_config
from ..greenscore_engine import GreenScoreEngine
from ..mixed import MixedWastePipeline, annotate
from ..paths import CONFIG_DIR, OUTPUTS_DIR, PROJECT_ROOT
from ..pipeline import ZyqoraPipeline
from ..preprocessing import ImageLoadError, load_rgb_bytes
from ..wasterules import RecommendationEngine, load_label_profiles, load_material_table
from .detection_store import DetectionStore

if TYPE_CHECKING:
    from .service import ZyqoraService

log = logging.getLogger("zyqora.api.mixed")
STORED_IMAGE_MAX_SIDE = 1600
EDITABLE = {
    "detector_profile": "detector model profile (fixed input size)",
    "label_profile": "organiser label profile",
    "conf_threshold": "detector confidence threshold",
    "iou_threshold": "NMS IoU threshold",
    "agnostic_nms": "class-agnostic NMS",
    "max_detections": "max boxes per image",
    "max_objects_classified": "max crops classified per image",
    "crop_padding": "crop padding fraction",
    "min_box_area_fraction": "minimum box area (fraction of image)",
    "whole_image_fallback": "classify the whole image when nothing is detected",
    "retention_days": "delete analyses and metrics older than N days (null = keep)",
    "metrics_sample_seconds": "minimum interval between stored system-metric samples",
}


@dataclass
class MixedSettings:
    detector_config: Path = DEFAULT_DETECTOR_CONFIG
    label_profiles: Path = CONFIG_DIR / "label_profiles.yaml"
    disposal_rules: Path = CONFIG_DIR / "disposal_rules.yaml"
    greenscore_rules: Path = CONFIG_DIR / "greenscore_rules.yaml"
    images_dir: Path = OUTPUTS_DIR / "web" / "detections"
    reports_dir: Path = PROJECT_ROOT / "reports"
    detector_eval_dir: Path = OUTPUTS_DIR / "detector_eval"
    store_images: bool = True


class MixedService:
    def __init__(self, svc: "ZyqoraService", settings: Optional[MixedSettings] = None, runner_factory=None):
        self.svc = svc
        self.cfg = settings or MixedSettings()
        self.store = DetectionStore(svc.settings.db_path, self.cfg.images_dir)
        self.detector: Optional[YoloDetector] = None
        self.code: Optional[str] = None
        self.message: Optional[str] = None
        self.loaded_at: Optional[str] = None
        self._runner_factory = runner_factory      # tests: profile -> runner
        self._reload_lock = threading.Lock()
        self._last_metrics_sample = 0.0
        self._pmic_cache: tuple[float, Optional[dict]] = (0.0, None)
        try:
            self.det_cfg = load_detector_config(self.cfg.detector_config)
            self.greenscore = GreenScoreEngine.load(self.cfg.greenscore_rules)
            self.recommender = RecommendationEngine.load(self.cfg.disposal_rules)
            self.config_error = None
        except (ConfigError, OSError) as exc:
            self.det_cfg = None
            self.config_error = str(exc)
            log.error("mixed-waste config error: %s", exc)

    # ---------------------------------------------------------------- settings
    def overrides(self) -> dict:
        return self.store.get_settings()

    def effective(self) -> dict:
        if self.det_cfg is None:
            return {}
        o = self.overrides()
        base = self.det_cfg.settings.as_dict()
        det = {k: o.get(k, v) for k, v in base.items()}
        return {
            "detector_profile": o.get("detector_profile", self.det_cfg.active),
            "label_profile": o.get("label_profile"),
            **det,
            "retention_days": o.get("retention_days"),
            "metrics_sample_seconds": o.get("metrics_sample_seconds", 30),
        }

    def detector_settings(self) -> DetectorSettings:
        e = self.effective()
        return DetectorSettings.from_dict({k: e[k] for k in DetectorSettings.__dataclass_fields__})

    def settings_payload(self) -> dict:
        e = self.effective()
        labels = self._labels()
        if labels and not e.get("label_profile"):
            e["label_profile"] = labels.active
        return {
            "effective": e,
            "overrides": self.overrides(),
            "defaults": {**(self.det_cfg.settings.as_dict() if self.det_cfg else {}),
                         "detector_profile": self.det_cfg.active if self.det_cfg else None,
                         "label_profile": labels.active if labels else None, "retention_days": None,
                         "metrics_sample_seconds": 30},
            "descriptions": EDITABLE,
            "options": {
                "detector_profiles": [{"name": p.name, "description": p.description, "imgsz": p.imgsz,
                                       "runtime": p.runtime, "available": p.file.is_file()}
                                      for p in (self.det_cfg.profiles.values() if self.det_cfg else [])],
                "label_profiles": [p.as_dict() for p in labels.profiles.values()] if labels else [],
            },
        }

    def update_settings(self, values: dict) -> dict:
        from .service import ServiceError

        if self.det_cfg is None:
            raise ServiceError("CONFIG_ERROR", self.config_error or "detector config not loaded", 503)
        unknown = sorted(set(values) - set(EDITABLE))
        if unknown:
            raise ServiceError("VALIDATION_ERROR", f"unknown settings: {unknown}", 422)
        merged = {**self.overrides(), **values}
        try:
            DetectorSettings.from_dict({k: merged.get(k, v) for k, v in self.det_cfg.settings.as_dict().items()})
        except ConfigError as exc:
            raise ServiceError("VALIDATION_ERROR", str(exc), 422) from exc
        if values.get("detector_profile") is not None and values["detector_profile"] not in self.det_cfg.profiles:
            raise ServiceError("VALIDATION_ERROR", f"unknown detector profile {values['detector_profile']!r}", 422)
        labels = self._labels()
        if values.get("label_profile") is not None and (not labels or values["label_profile"] not in labels.profiles):
            raise ServiceError("VALIDATION_ERROR", f"unknown label profile {values['label_profile']!r}", 422)
        rd = values.get("retention_days")
        if rd is not None and (isinstance(rd, bool) or not isinstance(rd, int) or not 1 <= rd <= 3650):
            raise ServiceError("VALIDATION_ERROR", "retention_days must be an integer 1-3650 or null", 422)
        ms = values.get("metrics_sample_seconds")
        if ms is not None and (isinstance(ms, bool) or not isinstance(ms, int) or not 5 <= ms <= 3600):
            raise ServiceError("VALIDATION_ERROR", "metrics_sample_seconds must be an integer 5-3600", 422)
        old_profile = self.effective().get("detector_profile")
        self.store.put_settings(values)
        if self.effective().get("detector_profile") != old_profile:
            self.load()
        if values.get("retention_days"):
            self.store.purge_older_than(values["retention_days"])
        return self.settings_payload()

    # ---------------------------------------------------------------- loading
    def _labels(self):
        lm = self.svc.state.loaded
        classes = list(lm.category_set.category_names) if lm else None
        try:
            return load_label_profiles(self.cfg.label_profiles, classes if lm and lm.profile.decoder == "direct_classes" else None)
        except ConfigError as exc:
            log.error("label profiles: %s", exc)
            return None

    def load(self) -> None:
        with self._reload_lock:
            self.detector, self.code, self.message = None, None, None
            if self.det_cfg is None:
                self.code, self.message = "CONFIG_ERROR", self.config_error
                return
            name = self.effective().get("detector_profile") or self.det_cfg.active
            try:
                profile = self.det_cfg.get(name)
                t0 = time.perf_counter()
                runner = self._runner_factory(profile) if self._runner_factory else None
                threads = self.svc.settings.num_threads or self.svc.app.num_threads
                det = YoloDetector(profile, self.detector_settings(), num_threads=threads, runner=runner,
                                   portable_lib=self.svc.settings.portable_lib)
                det.detect(Image.new("RGB", (profile.imgsz, profile.imgsz)))  # warm-up (ExecuTorch maps lazily)
                det.load_ms = (time.perf_counter() - t0) * 1e3                # load + first inference
            except DetectorError as exc:
                self.code, self.message = exc.code, str(exc)
            except ConfigError as exc:
                self.code, self.message = "CONFIG_ERROR", str(exc)
            else:
                self.detector = det
                from .store import now_iso
                self.loaded_at = now_iso()
                log.info("detector ready: %s", profile.name)
            if self.code:
                log.error("detector unavailable [%s]: %s", self.code, self.message)

    def status(self) -> dict:
        d = self.detector
        base = {"ready": d is not None, "code": self.code, "message": self.message,
                "active_profile_in_config": self.det_cfg.active if self.det_cfg else None,
                "available_profiles": sorted(self.det_cfg.profiles) if self.det_cfg else []}
        if d is None:
            return base
        return {**base, **d.describe(), "loaded_at": self.loaded_at, "settings": self.detector_settings().as_dict(),
                "class_names_are_waste_categories": False,
                "note": "Pretrained COCO-80 detector used as an object-proposal stage; its labels are material hints only."}

    def pipeline(self) -> MixedWastePipeline:
        from .service import ServiceError

        lm = self.svc.require_model()
        if self.detector is None:
            raise ServiceError(self.code or "DETECTOR_UNAVAILABLE", self.message or "detector not loaded", 503)
        if lm.profile.decoder != "direct_classes":
            raise ServiceError("UNSUPPORTED_CLASSIFIER", "mixed-waste analysis needs the waste classifier "
                               f"(direct_classes); loaded profile is {lm.profile.name}", 503)
        labels = self._labels()
        if labels is None:
            raise ServiceError("CONFIG_ERROR", "label profiles could not be loaded (see server log)", 503)
        classes = tuple(lm.category_set.category_names)
        materials = load_material_table(self.det_cfg.materials, list(classes), self.detector.profile.classes)
        return MixedWastePipeline(self.detector, ZyqoraPipeline(lm.classifier, lm.decoder, lm.guard, lm.spec),
                                  materials, labels, self.recommender, self.greenscore, classes)

    # ---------------------------------------------------------------- analysis
    def analyze_bytes(self, data: bytes, filename: str, **kw) -> dict:
        from .service import ServiceError

        if len(data) > self.svc.settings.max_upload_bytes:
            raise ServiceError("FILE_TOO_LARGE", f"Image is larger than {self.svc.settings.max_upload_bytes // 1048576} MB", 413)
        try:
            img = load_rgb_bytes(data, filename)
        except ImageLoadError as exc:
            raise ServiceError("INVALID_IMAGE", str(exc), 422) from exc
        return self.analyze_image(img, filename, **kw)

    def analyze_image(self, img: Image.Image, filename: str, *, source: str = "upload", user_id: Optional[str] = None,
                      persist: bool = True) -> dict:
        from .service import ServiceError

        if user_id:
            self.svc._check_user(user_id)
        pipe = self.pipeline()
        eff = self.effective()
        if not self.svc._slots.acquire(blocking=False):
            raise ServiceError("SERVER_BUSY", "Too many analysis requests are waiting; retry shortly", 503)
        try:
            with self.svc._model_lock:
                result = pipe.analyze(img, filename[:255], settings=self.detector_settings(),
                                      label_profile=eff.get("label_profile"))
        except DetectorError as exc:
            raise ServiceError(exc.code, str(exc), 500) from exc
        finally:
            self.svc._slots.release()
        errs = [o["classification"]["error"] for o in result["objects"] if o.get("classification") and o["classification"]["error"]]
        if errs:
            raise ServiceError("INFERENCE_FAILED", errs[0], 500)
        lm = self.svc.state.loaded
        result["model"] = {
            "classifier_profile": lm.profile.name, "classifier_pte": lm.profile.pte.name,
            "classifier_sha256": self.svc.state.sha256, "classifier_runtime": "ExecuTorch",
            "detector_profile": self.detector.profile.name, "detector_file": self.detector.profile.file.name,
            "detector_runtime": self.detector.runner.backend, "device": self.svc.telemetry.snapshot_device(),
        }
        result["source"] = source
        result["mode_note"] = ("Mixed-waste mode never actuates a bin: one image can hold items for several bins. "
                               "Use single-item mode for the sorting station.")
        self.svc.last_inference_ms = result["timing"]["detector_inference_ms"] + result["timing"]["classification_ms"]
        if persist:
            version = f"{lm.profile.name}@{(self.svc.state.sha256 or '')[:12]}+{self.detector.profile.name}"
            saver = (lambda det_id: self._save_image(det_id, img, result)) if self.cfg.store_images else None
            self.store.add(result, source=source, user_id=user_id, classifier_profile=lm.profile.name,
                           model_version=version, image_saver=saver)
            self.sample_metrics("detection", inference_ms=result["timing"]["pipeline_ms"], force=True)
            rd = eff.get("retention_days")
            if rd and result["id"] % 50 == 0:
                self.store.purge_older_than(rd)
        else:
            result.update(id=None, image_url=None, annotated_url=None)
        log.info("analysed %s: %d objects, %d waste items, image GS %s (%.0f ms)", filename,
                 result["summary"]["detected_objects"], result["summary"]["waste_items"],
                 result["image_green_score"]["score"], result["timing"]["pipeline_ms"])
        return result

    def _save_image(self, det_id: int, img: Image.Image, result: dict) -> Optional[str]:
        try:
            self.cfg.images_dir.mkdir(parents=True, exist_ok=True)
            out = img.copy()
            out.thumbnail((STORED_IMAGE_MAX_SIDE, STORED_IMAGE_MAX_SIDE))
            name = f"{int(det_id)}.jpg"
            out.save(self.cfg.images_dir / name, quality=88)
            result["stored_image_size"] = list(out.size)
            return name
        except OSError as exc:
            log.warning("detection image not saved for %s: %s", det_id, exc)
            return None

    def annotated(self, det_id: int) -> Optional[bytes]:
        import io

        res = self.store.get(det_id)
        p = self.store.image_file(det_id)
        if not res or not p:
            return None
        with Image.open(p) as im:
            img = im.convert("RGB")
        sx = img.size[0] / res["image_size"][0]
        sy = img.size[1] / res["image_size"][1]
        scaled = {**res, "objects": [{**o, "box": [o["box"][0] * sx, o["box"][1] * sy, o["box"][2] * sx, o["box"][3] * sy]}
                                     for o in res["objects"]]}
        buf = io.BytesIO()
        annotate(img, scaled).save(buf, "JPEG", quality=88)
        return buf.getvalue()

    # ---------------------------------------------------------------- metrics
    def pmic_power(self) -> Optional[dict]:
        """Raspberry Pi 5 PMIC rail readings (vcgencmd pmic_read_adc): sum of V x I over rails.

        This is the PMIC's own ADC measurement of its output rails; it excludes USB devices,
        the fan and conversion losses, so it is NOT wall-socket power. None when unavailable.
        """
        now = time.monotonic()
        if now - self._pmic_cache[0] < 5:
            return self._pmic_cache[1]
        out = None
        exe = shutil.which("vcgencmd")
        if exe:
            try:
                text = subprocess.run([exe, "pmic_read_adc"], capture_output=True, text=True, timeout=2).stdout
                out = parse_pmic(text)
            except (OSError, subprocess.SubprocessError):
                out = None
        self._pmic_cache = (now, out)
        return out

    def sample_metrics(self, source: str, inference_ms: Optional[float] = None, force: bool = False) -> dict:
        snap = self.svc.telemetry.snapshot()
        snap["pmic"] = self.pmic_power()
        snap["measured_power_watts"] = None   # only from an external meter; never estimated from CPU load
        interval = self.effective().get("metrics_sample_seconds", 30) or 30
        now = time.monotonic()
        if force or now - self._last_metrics_sample >= interval:
            self._last_metrics_sample = now
            try:
                self.store.add_metrics(snap, source, inference_ms)
            except Exception as exc:  # metrics must never break a request
                log.warning("system metric not stored: %s", exc)
        return snap

    # ---------------------------------------------------------------- benchmarks / evaluations
    def benchmarks(self) -> dict:
        """Every benchmark / evaluation JSON on disk, with where it was measured."""
        groups = {"classifier_benchmarks": [], "detector_benchmarks": [], "pipeline_benchmarks": [],
                  "detector_evaluations": []}
        seen: set[str] = set()
        dirs = [self.svc.settings.benchmark_dir, self.cfg.detector_eval_dir,
                self.cfg.reports_dir / "benchmarks", self.cfg.reports_dir / "detector_eval"]
        for d in dirs:
            if not d.is_dir():
                continue
            for p in sorted(d.glob("*.json")):
                try:
                    data = json.loads(p.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                key = f"{p.parent.name}/{p.name}"
                if key in seen:
                    continue
                seen.add(key)
                committed = self.cfg.reports_dir in p.parents
                entry = {"file": key, "committed_evidence": committed, "platform": data.get("platform"), "data": data}
                kind = data.get("kind")
                if "map50" in data:
                    groups["detector_evaluations"].append(entry)
                elif kind == "detector_benchmark":
                    groups["detector_benchmarks"].append(entry)
                elif kind == "pipeline_benchmark":
                    groups["pipeline_benchmarks"].append(entry)
                elif "results" in data:
                    groups["classifier_benchmarks"].append(entry)
        return groups


_PMIC_RE = re.compile(r"^\s*(\S+?)_(A|V)\s+(?:current|volt)\(\d+\)=([0-9.]+)([AV])\s*$")


def parse_pmic(text: str) -> Optional[dict]:
    currents, volts = {}, {}
    for line in text.splitlines():
        m = _PMIC_RE.match(line)
        if not m:
            continue
        name, kind, val, _unit = m.groups()
        (currents if kind == "A" else volts)[name] = float(val)
    rails = {n: volts[n] * currents[n] for n in currents if n in volts}
    if not rails:
        return None
    return {"rails_watts": round(sum(rails.values()), 3), "rails": len(rails),
            "source": "vcgencmd pmic_read_adc (PMIC rails only; excludes USB, fan and conversion losses)"}
