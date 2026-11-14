"""SQLite persistence for mixed-waste analyses (Zyqora AI 4.0).

Tables (same database file as the single-image `classifications` history):
  detections        one row per analysed image
  detected_objects  one row per box / item, FK -> detections (ON DELETE CASCADE)
  green_scores      one row per scored item, FK -> detected_objects (ON DELETE CASCADE)
  system_metrics    device samples taken at each analysis and by the monitoring page
  settings          key -> JSON value overrides edited from the dashboard

Migrations are numbered and tracked with PRAGMA user_version; each runs once, in order.
All SQL is parameterised. Retention: purge_older_than(days) removes old rows and images.
"""

from __future__ import annotations

import csv
import io
import json
import sqlite3
import threading
from collections import Counter, defaultdict
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from ..evaluator import latency_stats
from .store import now_iso

MIGRATIONS: list[str] = [
    # 1: Zyqora AI 4.0 mixed-waste tables
    """
    CREATE TABLE IF NOT EXISTS detections (
        id                   INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at           TEXT NOT NULL,
        source               TEXT NOT NULL,
        filename             TEXT NOT NULL,
        user_id              TEXT,
        image_path           TEXT,
        image_width          INTEGER,
        image_height         INTEGER,
        total_objects        INTEGER NOT NULL,
        waste_items          INTEGER NOT NULL,
        accepted_items       INTEGER NOT NULL,
        review_items         INTEGER NOT NULL,
        ignored_objects      INTEGER NOT NULL,
        whole_image_fallback INTEGER NOT NULL DEFAULT 0,
        image_green_score    REAL,
        green_score_coverage REAL,
        detector_inference_ms REAL,
        classification_ms    REAL,
        inference_time_ms    REAL,
        pipeline_ms          REAL,
        detector_profile     TEXT NOT NULL,
        classifier_profile   TEXT NOT NULL,
        model_version        TEXT NOT NULL,
        label_profile        TEXT NOT NULL,
        result_json          TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_det_created ON detections(created_at);

    CREATE TABLE IF NOT EXISTS detected_objects (
        id                        INTEGER PRIMARY KEY AUTOINCREMENT,
        detection_id              INTEGER NOT NULL REFERENCES detections(id) ON DELETE CASCADE,
        obj_index                 INTEGER NOT NULL,
        source                    TEXT NOT NULL,
        bounding_box              TEXT NOT NULL,
        detection_label           TEXT,
        detection_confidence      REAL,
        classification_label      TEXT,
        classification_confidence REAL,
        ecoguard_status           TEXT,
        decision                  TEXT NOT NULL,
        material_category         TEXT,
        material_certainty        TEXT,
        food                      INTEGER,
        organiser_category        TEXT,
        recommended_disposal      TEXT,
        conflict                  TEXT,
        review_reasons            TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_obj_detection ON detected_objects(detection_id);
    CREATE INDEX IF NOT EXISTS idx_obj_label ON detected_objects(classification_label);

    CREATE TABLE IF NOT EXISTS green_scores (
        id                    INTEGER PRIMARY KEY AUTOINCREMENT,
        object_id             INTEGER NOT NULL REFERENCES detected_objects(id) ON DELETE CASCADE,
        recycling_potential   REAL NOT NULL,
        environmental_impact  REAL NOT NULL,
        disposal_suitability  REAL NOT NULL,
        final_score           REAL NOT NULL,
        band                  TEXT,
        basis                 TEXT,
        scoring_version       TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_gs_object ON green_scores(object_id);

    CREATE TABLE IF NOT EXISTS system_metrics (
        id                   INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp            TEXT NOT NULL,
        source               TEXT NOT NULL,
        cpu_percent          REAL,
        memory_percent       REAL,
        process_rss_mb       REAL,
        inference_time_ms    REAL,
        temperature_c        REAL,
        measured_power_watts REAL
    );
    CREATE INDEX IF NOT EXISTS idx_sys_ts ON system_metrics(timestamp);

    CREATE TABLE IF NOT EXISTS settings (
        key        TEXT PRIMARY KEY,
        value      TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );
    """,
]


def _iso_days_ago(days: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days)).isoformat(timespec="milliseconds")


class DetectionStore:
    def __init__(self, db_path: Path, images_dir: Path):
        self.db_path = Path(db_path)
        self.images_dir = Path(images_dir)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.migrate()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def migrate(self) -> int:
        with self._lock, closing(self._connect()) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            ver = int(conn.execute("PRAGMA user_version").fetchone()[0])
            for n, sql in enumerate(MIGRATIONS, start=1):
                if ver < n:
                    with conn:
                        conn.executescript(sql)
                        conn.execute(f"PRAGMA user_version = {n}")
            return int(conn.execute("PRAGMA user_version").fetchone()[0])

    @property
    def schema_version(self) -> int:
        with closing(self._connect()) as conn:
            return int(conn.execute("PRAGMA user_version").fetchone()[0])

    # ------------------------------------------------------------------ writes
    def add(self, result: dict, *, source: str, user_id: Optional[str], classifier_profile: str,
            model_version: str, image_saver: Optional[Callable[[int], Optional[str]]] = None) -> int:
        s = result["summary"]
        t = result["timing"]
        gs = result["image_green_score"]
        created = now_iso()
        with self._lock, closing(self._connect()) as conn, conn:
            cur = conn.execute(
                """INSERT INTO detections (created_at, source, filename, user_id, image_width, image_height,
                   total_objects, waste_items, accepted_items, review_items, ignored_objects, whole_image_fallback,
                   image_green_score, green_score_coverage, detector_inference_ms, classification_ms, inference_time_ms,
                   pipeline_ms, detector_profile, classifier_profile, model_version, label_profile, result_json)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (created, source, result["filename"][:255], user_id, result["image_size"][0], result["image_size"][1],
                 s["detected_objects"], s["waste_items"], s["accepted"], s["review"], s["ignored_non_waste"],
                 int(s["whole_image_fallback"]), gs["score"], gs.get("coverage"), t["detector_inference_ms"],
                 t["classification_ms"], t["detector_inference_ms"] + t["classification_ms"], t["pipeline_ms"],
                 result["detector"]["profile"], classifier_profile, model_version, result["label_profile"]["name"],
                 "{}"))
            det_id = int(cur.lastrowid)
            for o in result["objects"]:
                c = o.get("classification") or {}
                rec = o.get("recommendation") or {}
                oc = conn.execute(
                    """INSERT INTO detected_objects (detection_id, obj_index, source, bounding_box, detection_label,
                       detection_confidence, classification_label, classification_confidence, ecoguard_status, decision,
                       material_category, material_certainty, food, organiser_category, recommended_disposal, conflict,
                       review_reasons) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (det_id, o["index"], o["source"], json.dumps(o["box"]), o["detection_label"], o["detection_confidence"],
                     c.get("model_label"), c.get("confidence"), c.get("ecoguard_status"), o["decision"],
                     o["material"]["category"], o["material"]["certainty"],
                     None if o["material"]["food"] is None else int(o["material"]["food"]),
                     o["organiser_category"], rec.get("stream"), o.get("conflict"),
                     json.dumps(o.get("review_reasons") or [])))
                o["object_id"] = int(oc.lastrowid)
                g = o["green_score"]
                if g.get("scored"):
                    comp = {c_["key"]: c_["value"] for c_ in g["components"]}
                    conn.execute(
                        """INSERT INTO green_scores (object_id, recycling_potential, environmental_impact,
                           disposal_suitability, final_score, band, basis, scoring_version) VALUES (?,?,?,?,?,?,?,?)""",
                        (o["object_id"], comp["recycling_potential"], comp["environmental_impact"],
                         comp["disposal_suitability"], g["score"], g["band"], g["basis"], g["scoring_version"]))
            result["id"] = det_id
            result["created_at"] = created
            image_path = image_saver(det_id) if image_saver else None
            result["image_url"] = f"/api/detections/{det_id}/image" if image_path else None
            result["annotated_url"] = f"/api/detections/{det_id}/annotated.jpg" if image_path else None
            conn.execute("UPDATE detections SET image_path = ?, result_json = ? WHERE id = ?",
                         (image_path, json.dumps(result), det_id))
        return det_id

    def add_metrics(self, sample: dict, source: str, inference_ms: Optional[float] = None) -> None:
        with self._lock, closing(self._connect()) as conn, conn:
            conn.execute(
                """INSERT INTO system_metrics (timestamp, source, cpu_percent, memory_percent, process_rss_mb,
                   inference_time_ms, temperature_c, measured_power_watts) VALUES (?,?,?,?,?,?,?,?)""",
                (now_iso(), source, sample.get("cpu_percent"), (sample.get("memory") or {}).get("used_percent"),
                 sample.get("process_rss_mb"), inference_ms, sample.get("temperature_c"),
                 sample.get("measured_power_watts")))

    def delete(self, det_id: int) -> bool:
        with self._lock, closing(self._connect()) as conn, conn:
            row = conn.execute("SELECT image_path FROM detections WHERE id = ?", (det_id,)).fetchone()
            if not row:
                return False
            conn.execute("DELETE FROM detections WHERE id = ?", (det_id,))
        self._remove_image(row["image_path"])
        return True

    def _remove_image(self, rel: Optional[str]) -> None:
        if not rel:
            return
        p = (self.images_dir / Path(rel).name)
        try:
            p.unlink(missing_ok=True)
        except OSError:
            pass

    def purge_older_than(self, days: float) -> dict:
        cutoff = _iso_days_ago(days)
        with self._lock, closing(self._connect()) as conn, conn:
            paths = [r[0] for r in conn.execute("SELECT image_path FROM detections WHERE created_at < ?", (cutoff,))]
            n_det = conn.execute("DELETE FROM detections WHERE created_at < ?", (cutoff,)).rowcount
            n_sys = conn.execute("DELETE FROM system_metrics WHERE timestamp < ?", (cutoff,)).rowcount
        for p in paths:
            self._remove_image(p)
        return {"detections_deleted": n_det, "system_metrics_deleted": n_sys, "cutoff": cutoff}

    # ------------------------------------------------------------------ settings
    def get_settings(self) -> dict[str, Any]:
        with closing(self._connect()) as conn:
            return {r["key"]: json.loads(r["value"]) for r in conn.execute("SELECT key, value FROM settings")}

    def put_settings(self, values: dict[str, Any]) -> None:
        with self._lock, closing(self._connect()) as conn, conn:
            for k, v in values.items():
                if v is None:
                    conn.execute("DELETE FROM settings WHERE key = ?", (k,))
                else:
                    conn.execute("INSERT INTO settings (key, value, updated_at) VALUES (?,?,?) "
                                 "ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
                                 (k, json.dumps(v), now_iso()))

    # ------------------------------------------------------------------ reads
    def get(self, det_id: int) -> Optional[dict]:
        with closing(self._connect()) as conn:
            r = conn.execute("SELECT result_json FROM detections WHERE id = ?", (det_id,)).fetchone()
        return json.loads(r[0]) if r else None

    def image_file(self, det_id: int) -> Optional[Path]:
        with closing(self._connect()) as conn:
            r = conn.execute("SELECT image_path FROM detections WHERE id = ?", (det_id,)).fetchone()
        if not r or not r[0]:
            return None
        p = self.images_dir / Path(r[0]).name
        return p if p.is_file() else None

    def _where(self, f: dict) -> tuple[str, list]:
        clauses, params = [], []
        if f.get("date_from"):
            clauses.append("d.created_at >= ?")
            params.append(f["date_from"])
        if f.get("date_to"):
            clauses.append("d.created_at <= ?")
            params.append(f["date_to"])
        if f.get("q"):
            clauses.append("(d.filename LIKE ? OR EXISTS (SELECT 1 FROM detected_objects o WHERE o.detection_id = d.id "
                           "AND (o.detection_label LIKE ? OR o.classification_label LIKE ? OR o.material_category LIKE ?)))")
            q = f"%{f['q']}%"
            params += [q, q, q, q]
        if f.get("material"):
            clauses.append("EXISTS (SELECT 1 FROM detected_objects o WHERE o.detection_id = d.id AND o.material_category = ?)")
            params.append(f["material"])
        if f.get("model_label"):
            clauses.append("EXISTS (SELECT 1 FROM detected_objects o WHERE o.detection_id = d.id "
                           "AND o.classification_label = ? AND o.decision = 'ACCEPT')")
            params.append(f["model_label"])
        if f.get("has_review"):
            clauses.append("d.review_items > 0")
        if f.get("min_score") is not None:
            clauses.append("d.image_green_score >= ?")
            params.append(float(f["min_score"]))
        return (" WHERE " + " AND ".join(clauses)) if clauses else "", params

    def list(self, *, limit: int = 50, offset: int = 0, **filters) -> tuple[list[dict], int]:
        where, params = self._where(filters)
        with closing(self._connect()) as conn:
            total = conn.execute(f"SELECT COUNT(*) FROM detections d{where}", params).fetchone()[0]
            rows = conn.execute(
                f"""SELECT d.id, d.created_at, d.source, d.filename, d.user_id, d.image_path, d.total_objects, d.waste_items,
                    d.accepted_items, d.review_items, d.ignored_objects, d.whole_image_fallback, d.image_green_score,
                    d.green_score_coverage, d.inference_time_ms, d.pipeline_ms, d.detector_profile, d.label_profile
                    FROM detections d{where} ORDER BY d.id DESC LIMIT ? OFFSET ?""", [*params, limit, offset]).fetchall()
            ids = [r["id"] for r in rows]
            objs: dict[int, list] = defaultdict(list)
            if ids:
                q = ",".join("?" * len(ids))
                for o in conn.execute(f"""SELECT detection_id, obj_index, detection_label, classification_label, decision,
                                         material_category, organiser_category FROM detected_objects
                                         WHERE detection_id IN ({q}) ORDER BY obj_index""", ids):
                    objs[o["detection_id"]].append(dict(o))
        items = []
        for r in rows:
            d = dict(r)
            d["whole_image_fallback"] = bool(d["whole_image_fallback"])
            d["image_url"] = f"/api/detections/{d['id']}/image" if d.pop("image_path") else None
            d["objects"] = [{k: v for k, v in o.items() if k != "detection_id"} for o in objs[d["id"]]]
            items.append(d)
        return items, int(total)

    def summary(self, label_profiles, *, days: Optional[int] = None) -> dict:
        """Aggregates over stored analyses. Organiser-profile counts are derived from the stored model label
        (only for reliable items) through the current profile mappings."""
        params: list = []
        where = ""
        if days:
            where = " WHERE d.created_at >= ?"
            params.append(_iso_days_ago(days))
        with closing(self._connect()) as conn:
            dets = [dict(r) for r in conn.execute(
                f"SELECT d.id, d.created_at, d.total_objects, d.waste_items, d.accepted_items, d.review_items, "
                f"d.image_green_score, d.pipeline_ms, d.inference_time_ms, d.detector_inference_ms, d.classification_ms "
                f"FROM detections d{where} ORDER BY d.id", params)]
            objs = [dict(r) for r in conn.execute(
                f"SELECT o.classification_label, o.decision, o.material_category, o.recommended_disposal, o.food, "
                f"d.created_at FROM detected_objects o JOIN detections d ON d.id = o.detection_id{where}", params)]
            gscores = [dict(r) for r in conn.execute(
                f"SELECT g.final_score, g.band, d.created_at FROM green_scores g JOIN detected_objects o ON o.id = g.object_id "
                f"JOIN detections d ON d.id = o.detection_id{where}", params)]
            sysm = [dict(r) for r in conn.execute(
                "SELECT timestamp, source, cpu_percent, memory_percent, process_rss_mb, inference_time_ms, temperature_c, "
                "measured_power_watts FROM system_metrics" + (" WHERE timestamp >= ?" if days else "") +
                " ORDER BY id DESC LIMIT 500", params)]
        waste = [o for o in objs if o["decision"] in ("ACCEPT", "REVIEW")]
        by_profile = {}
        for name, p in label_profiles.profiles.items():
            by_profile[name] = {"title": p.title, "counts": dict(Counter(
                p.label(o["classification_label"], o["decision"] == "ACCEPT") for o in waste))}
        days_map: dict[str, dict] = {}
        for d in dets:
            day = d["created_at"][:10]
            e = days_map.setdefault(day, {"date": day, "images": 0, "objects": 0, "waste_items": 0, "scores": []})
            e["images"] += 1
            e["objects"] += d["total_objects"]
            e["waste_items"] += d["waste_items"]
            if d["image_green_score"] is not None:
                e["scores"].append(d["image_green_score"])
        by_day = []
        for e in sorted(days_map.values(), key=lambda x: x["date"]):
            sc = e.pop("scores")
            e["mean_green_score"] = sum(sc) / len(sc) if sc else None
            by_day.append(e)
        scores = [g["final_score"] for g in gscores]
        img_scores = [d["image_green_score"] for d in dets if d["image_green_score"] is not None]
        return {
            "images": len(dets),
            "detected_objects": sum(d["total_objects"] for d in dets),
            "waste_items": len(waste),
            "accepted_items": sum(1 for o in waste if o["decision"] == "ACCEPT"),
            "review_items": sum(1 for o in waste if o["decision"] == "REVIEW"),
            "by_profile": by_profile,
            "by_model_label": dict(Counter(o["classification_label"] for o in waste if o["decision"] == "ACCEPT")),
            "by_material": dict(Counter(o["material_category"] for o in waste)),
            "by_stream": dict(Counter(o["recommended_disposal"] for o in waste if o["recommended_disposal"])),
            "food_items": sum(1 for o in waste if o["food"] == 1),
            "green_score": {
                "scored_items": len(scores), "mean_item_score": sum(scores) / len(scores) if scores else None,
                "mean_image_score": sum(img_scores) / len(img_scores) if img_scores else None,
                "by_band": dict(Counter(g["band"] for g in gscores)),
                "series": [{"id": d["id"], "created_at": d["created_at"], "score": d["image_green_score"]}
                           for d in dets[-200:] if d["image_green_score"] is not None],
            },
            "by_day": by_day,
            "latency": {"pipeline": latency_stats([d["pipeline_ms"] for d in dets if d["pipeline_ms"] is not None]),
                        "detector": latency_stats([d["detector_inference_ms"] for d in dets if d["detector_inference_ms"] is not None]),
                        "series": [{"id": d["id"], "created_at": d["created_at"], "pipeline_ms": d["pipeline_ms"],
                                    "detector_ms": d["detector_inference_ms"], "classification_ms": d["classification_ms"]}
                                   for d in dets[-200:]]},
            "system_metrics": list(reversed(sysm)),
            "latest": dets[-1]["id"] if dets else None,
        }

    def export_csv(self, **filters) -> str:
        where, params = self._where(filters)
        cols = ["detection_id", "created_at", "filename", "obj_index", "source", "bounding_box", "detection_label",
                "detection_confidence", "classification_label", "classification_confidence", "decision",
                "material_category", "organiser_category", "recommended_disposal", "green_score", "conflict"]
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=cols)
        w.writeheader()
        with closing(self._connect()) as conn:
            for r in conn.execute(
                    f"""SELECT o.detection_id, d.created_at, d.filename, o.obj_index, o.source, o.bounding_box,
                        o.detection_label, o.detection_confidence, o.classification_label, o.classification_confidence,
                        o.decision, o.material_category, o.organiser_category, o.recommended_disposal,
                        g.final_score AS green_score, o.conflict
                        FROM detected_objects o JOIN detections d ON d.id = o.detection_id
                        LEFT JOIN green_scores g ON g.object_id = o.id{where}
                        ORDER BY o.detection_id, o.obj_index""", params):
                w.writerow(dict(r))
        return buf.getvalue()
