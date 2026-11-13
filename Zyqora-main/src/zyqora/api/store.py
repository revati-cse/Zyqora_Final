"""SQLite history of classifications served by the API (local file, parameterised SQL)."""

from __future__ import annotations

import csv
import io
import json
import sqlite3
import threading
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from ..evaluator import EvaluationRow, classification_metrics, latency_stats
from ..schemas import ACCEPT, ImageResult

SCHEMA = """
CREATE TABLE IF NOT EXISTS classifications (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at          TEXT NOT NULL,
    source              TEXT NOT NULL,
    filename            TEXT NOT NULL,
    batch_id            TEXT,
    user_id             TEXT,
    predicted_category  TEXT,
    confidence          REAL,
    class_scores        TEXT,
    ecoguard_status     TEXT NOT NULL,
    review_reason       TEXT,
    disposal_action     TEXT NOT NULL,
    preprocess_ms       REAL,
    inference_ms        REAL,
    total_ms            REAL,
    model_profile       TEXT NOT NULL,
    model_sha256        TEXT,
    ground_truth        TEXT,
    greenscore_event_id INTEGER,
    sort_message        TEXT,
    has_thumbnail       INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_cls_created ON classifications(created_at);
CREATE INDEX IF NOT EXISTS idx_cls_batch ON classifications(batch_id);
"""

COLUMNS = ["id", "created_at", "source", "filename", "batch_id", "user_id", "predicted_category", "confidence",
           "ecoguard_status", "review_reason", "disposal_action", "preprocess_ms", "inference_ms", "total_ms",
           "model_profile", "model_sha256", "ground_truth", "correct", "greenscore_event_id", "sort_message"]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class HistoryStore:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with closing(self._connect()) as conn, conn:
            conn.execute("PRAGMA journal_mode=WAL")  # persistent; lets readers run during writes
            conn.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA synchronous=NORMAL")  # WAL + NORMAL: durable across app crashes, far fewer fsyncs
        return conn

    # -- writes --------------------------------------------------------------
    def add(self, rec: dict[str, Any]) -> int:
        cols = ["created_at", "source", "filename", "batch_id", "user_id", "predicted_category", "confidence",
                "class_scores", "ecoguard_status", "review_reason", "disposal_action", "preprocess_ms",
                "inference_ms", "total_ms", "model_profile", "model_sha256", "ground_truth", "greenscore_event_id",
                "sort_message", "has_thumbnail"]
        values = [rec.get(c) for c in cols]
        values[cols.index("class_scores")] = json.dumps(rec.get("class_scores") or {})
        values[cols.index("has_thumbnail")] = int(bool(rec.get("has_thumbnail")))
        with self._lock, closing(self._connect()) as conn, conn:
            cur = conn.execute(f"INSERT INTO classifications ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                               values)
            return int(cur.lastrowid)

    def set_thumbnail(self, item_id: int, present: bool) -> None:
        with self._lock, closing(self._connect()) as conn, conn:
            conn.execute("UPDATE classifications SET has_thumbnail = ? WHERE id = ?", (int(present), item_id))

    def set_greenscore_event(self, item_id: int, event_id: int) -> None:
        with self._lock, closing(self._connect()) as conn, conn:
            conn.execute("UPDATE classifications SET greenscore_event_id = ? WHERE id = ?", (event_id, item_id))

    # -- reads ---------------------------------------------------------------
    @staticmethod
    def _row(r: sqlite3.Row) -> dict[str, Any]:
        d = dict(r)
        d["class_scores"] = json.loads(d.get("class_scores") or "{}")
        d["has_thumbnail"] = bool(d.get("has_thumbnail"))
        gt = d.get("ground_truth")
        d["correct"] = None if not gt else (d.get("predicted_category") == gt)
        return d

    def _where(self, date_from: Optional[str], date_to: Optional[str], category: Optional[str],
               status: Optional[str], batch_id: Optional[str] = None, source: Optional[str] = None):
        clauses, params = [], []
        if date_from:
            clauses.append("created_at >= ?")
            params.append(date_from)
        if date_to:
            clauses.append("created_at <= ?")
            params.append(date_to)
        if category:
            clauses.append("predicted_category = ?")
            params.append(category)
        if status:
            clauses.append("ecoguard_status = ?")
            params.append(status)
        if batch_id:
            clauses.append("batch_id = ?")
            params.append(batch_id)
        if source:
            clauses.append("source = ?")
            params.append(source)
        return (" WHERE " + " AND ".join(clauses)) if clauses else "", params

    def get(self, item_id: int) -> Optional[dict[str, Any]]:
        with closing(self._connect()) as conn:
            r = conn.execute("SELECT * FROM classifications WHERE id = ?", (item_id,)).fetchone()
        return self._row(r) if r else None

    def list(self, *, limit: int = 50, offset: int = 0, date_from=None, date_to=None, category=None, status=None,
             batch_id=None, source=None) -> tuple[list[dict[str, Any]], int]:
        where, params = self._where(date_from, date_to, category, status, batch_id, source)
        with closing(self._connect()) as conn:
            total = conn.execute(f"SELECT COUNT(*) FROM classifications{where}", params).fetchone()[0]
            rows = conn.execute(f"SELECT * FROM classifications{where} ORDER BY id DESC LIMIT ? OFFSET ?",
                                [*params, limit, offset]).fetchall()
        return [self._row(r) for r in rows], int(total)

    def _all(self, **filters) -> list[dict[str, Any]]:
        where, params = self._where(filters.get("date_from"), filters.get("date_to"), filters.get("category"),
                                    filters.get("status"), filters.get("batch_id"), filters.get("source"))
        with closing(self._connect()) as conn:
            rows = conn.execute(f"SELECT * FROM classifications{where} ORDER BY id", params).fetchall()
        return [self._row(r) for r in rows]

    # -- analytics -------------------------------------------------------------
    def summary(self, predictable: list[str], **filters) -> dict[str, Any]:
        rows = self._all(**filters)
        total = len(rows)
        accepted = [r for r in rows if r["ecoguard_status"] == ACCEPT]
        by_cat: dict[str, int] = {}
        by_action: dict[str, int] = {}
        days: dict[str, dict[str, int]] = {}
        for r in rows:
            cat = r["predicted_category"] or "none"
            by_cat[cat] = by_cat.get(cat, 0) + 1
            by_action[r["disposal_action"]] = by_action.get(r["disposal_action"], 0) + 1
            day = r["created_at"][:10]
            d = days.setdefault(day, {"total": 0, "accepted": 0, "review": 0})
            d["total"] += 1
            d["accepted" if r["ecoguard_status"] == ACCEPT else "review"] += 1
            d[cat] = d.get(cat, 0) + 1
        inf = [r["inference_ms"] for r in rows if r["inference_ms"] is not None]
        tot = [r["total_ms"] for r in rows if r["total_ms"] is not None]
        labelled = [r for r in rows if r["ground_truth"]]
        accuracy = None
        if labelled:
            ev = [EvaluationRow(ImageResult(image_filename=r["filename"], predicted_category=r["predicted_category"],
                                            ecoguard_status=r["ecoguard_status"]), r["ground_truth"]) for r in labelled]
            acc_rows = [e for e in ev if e.result.ecoguard_status == ACCEPT]
            correct = sum(1 for e in ev if e.correct)
            accuracy = {
                "scope": "LIVE_LABELLED_UPLOADS",
                "note": "computed only over uploads that were given a ground-truth label in this dashboard",
                "labelled_images": len(ev), "correct": correct, "accuracy": correct / len(ev),
                "accepted": len(acc_rows),
                "accepted_only_accuracy": (sum(1 for e in acc_rows if e.correct) / len(acc_rows)) if acc_rows else None,
                "metrics": classification_metrics(ev, predictable),
            }
        batches: dict[str, list[dict]] = {}
        for r in rows:
            if r["batch_id"]:
                batches.setdefault(r["batch_id"], []).append(r)
        throughput = []
        for bid, items in batches.items():
            if len(items) < 2:
                continue
            t0 = datetime.fromisoformat(items[0]["created_at"])
            t1 = datetime.fromisoformat(items[-1]["created_at"])
            secs = (t1 - t0).total_seconds()
            if secs > 0:
                throughput.append({"batch_id": bid, "images": len(items), "seconds": secs,
                                   "images_per_second": (len(items) - 1) / secs})
        return {
            "total": total,
            "accepted": len(accepted),
            "review": total - len(accepted),
            "review_rate": ((total - len(accepted)) / total) if total else None,
            "by_category": by_cat,
            "by_action": by_action,
            "by_day": [{"date": k, **v} for k, v in sorted(days.items())],
            "latest": self._row_brief(rows[-1]) if rows else None,
            "inference_latency": latency_stats(inf),
            "total_latency": latency_stats(tot),
            "latency_series": [{"id": r["id"], "created_at": r["created_at"], "inference_ms": r["inference_ms"],
                                "total_ms": r["total_ms"]} for r in rows[-200:] if r["inference_ms"] is not None],
            "batch_throughput": throughput[-20:],
            "labelled_accuracy": accuracy,
        }

    @staticmethod
    def _row_brief(r: dict) -> dict:
        return {k: r[k] for k in ("id", "created_at", "predicted_category", "confidence", "ecoguard_status",
                                  "disposal_action", "inference_ms")}

    def export_csv(self, **filters) -> str:
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=COLUMNS + ["class_scores"], extrasaction="ignore")
        w.writeheader()
        for r in self._all(**filters):
            r = dict(r)
            r["class_scores"] = json.dumps(r["class_scores"])
            w.writerow(r)
        return buf.getvalue()
