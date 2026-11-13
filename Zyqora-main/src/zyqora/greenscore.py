"""GreenScore: local rewards for SUPERVISOR-VERIFIED sorting actions (not AI predictions).

Workflow
    1. Zyqora classifies an item and records a PENDING sorting event.
    2. The user places the item in a bin.
    3. A supervisor checks the bin the item was ACTUALLY placed in and states the
       bin it SHOULD have gone in (the AI recommendation for ACCEPT items, the
       supervisor's own judgement for REVIEW items).
    4. Points are awarded only when the observed bin equals the correct bin.
    5. An event can be verified exactly once; the reward and the status change
       happen in one transaction, so a reward can never be granted twice.

All queries are parameterised. Only a short user id and optional display name
are stored (no other personal data).

CLI (python -m zyqora.greenscore ...):
    add-user --user-id U01 [--name "Team A"]
    record   --user-id U01 --image bottle.jpg --predicted-category Recyclable --action RECYCLE --ecoguard ACCEPT
    events   [--status PENDING]
    verify   --event-id 1 [--observed RECYCLE] [--correct RECYCLE] [--supervisor S1]
    points   --user-id U01
    history  --user-id U01
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import sys
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .schemas import ACCEPT, MANUAL_REVIEW, REVIEW

USER_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,32}$")
ACTION_RE = re.compile(r"^[A-Z][A-Z0-9_]{1,31}$")
PENDING, VERIFIED_CORRECT, VERIFIED_INCORRECT = "PENDING", "VERIFIED_CORRECT", "VERIFIED_INCORRECT"

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    user_id       TEXT PRIMARY KEY CHECK (length(user_id) BETWEEN 1 AND 32),
    display_name  TEXT,
    total_points  INTEGER NOT NULL DEFAULT 0 CHECK (total_points >= 0),
    created_at    TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sorting_events (
    event_id            INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id             TEXT NOT NULL REFERENCES users(user_id),
    image_id            TEXT NOT NULL,
    run_id              TEXT NOT NULL DEFAULT '',
    predicted_category  TEXT,
    recommended_action  TEXT NOT NULL,
    ecoguard_status     TEXT NOT NULL CHECK (ecoguard_status IN ('ACCEPT', 'REVIEW')),
    verification_status TEXT NOT NULL DEFAULT 'PENDING'
                        CHECK (verification_status IN ('PENDING', 'VERIFIED_CORRECT', 'VERIFIED_INCORRECT')),
    observed_action     TEXT,
    verified_action     TEXT,
    supervisor_id       TEXT,
    awarded_points      INTEGER NOT NULL DEFAULT 0 CHECK (awarded_points >= 0),
    created_at          TEXT NOT NULL,
    verified_at         TEXT,
    UNIQUE (user_id, image_id, run_id)
);
"""


class GreenScoreError(ValueError):
    pass


class AlreadyVerifiedError(GreenScoreError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _check_user_id(user_id: str) -> str:
    if not isinstance(user_id, str) or not USER_ID_RE.match(user_id):
        raise GreenScoreError(f"invalid user id {user_id!r}: use 1-32 letters, digits, '_' or '-'")
    return user_id


def _check_action(action: str, field: str) -> str:
    if not isinstance(action, str) or not ACTION_RE.match(action) or action == MANUAL_REVIEW:
        raise GreenScoreError(f"invalid {field} {action!r}: must be a bin action such as RECYCLE / COMPOST / RESIDUAL")
    return action


@dataclass(frozen=True)
class VerificationResult:
    event_id: int
    status: str
    observed_action: str
    correct_action: str
    awarded_points: int
    total_points: int


class GreenScore:
    def __init__(self, db_path: str | Path, points_correct: int = 10, points_incorrect: int = 0,
                 valid_actions: Optional[list[str]] = None):
        for name, v in (("points_correct", points_correct), ("points_incorrect", points_incorrect)):
            if isinstance(v, bool) or not isinstance(v, int) or v < 0:
                raise GreenScoreError(f"{name} must be a non-negative integer")
        self.db_path = Path(db_path)
        self.points_correct = points_correct
        self.points_incorrect = points_incorrect
        self.valid_actions = set(valid_actions) if valid_actions else None
        if not self.db_path.parent.is_dir():
            raise GreenScoreError(f"database directory does not exist: {self.db_path.parent}")
        with closing(self._connect()) as conn, conn:
            conn.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _validate_bin(self, action: str, field: str) -> str:
        _check_action(action, field)
        if self.valid_actions is not None and action not in self.valid_actions:
            raise GreenScoreError(f"{field} {action!r} is not a configured bin action {sorted(self.valid_actions)}")
        return action

    # -- users -------------------------------------------------------------
    def add_user(self, user_id: str, display_name: Optional[str] = None) -> None:
        _check_user_id(user_id)
        if display_name is not None and len(display_name) > 64:
            raise GreenScoreError("display name must be at most 64 characters")
        with closing(self._connect()) as conn:
            try:
                conn.execute("INSERT INTO users (user_id, display_name, created_at) VALUES (?, ?, ?)",
                             (user_id, display_name, _now()))
            except sqlite3.IntegrityError as exc:
                raise GreenScoreError(f"user {user_id!r} already exists") from exc

    def ensure_user(self, user_id: str) -> None:
        _check_user_id(user_id)
        with closing(self._connect()) as conn:
            conn.execute("INSERT OR IGNORE INTO users (user_id, created_at) VALUES (?, ?)", (user_id, _now()))

    def list_users(self) -> list[sqlite3.Row]:
        with closing(self._connect()) as conn:
            return conn.execute("SELECT user_id, display_name, total_points, created_at FROM users "
                                "ORDER BY total_points DESC, user_id").fetchall()

    def points(self, user_id: str) -> int:
        _check_user_id(user_id)
        with closing(self._connect()) as conn:
            row = conn.execute("SELECT total_points FROM users WHERE user_id = ?", (user_id,)).fetchone()
        if row is None:
            raise GreenScoreError(f"unknown user {user_id!r}")
        return int(row["total_points"])

    # -- events ------------------------------------------------------------
    def record_event(self, user_id: str, image_id: str, predicted_category: Optional[str],
                     recommended_action: str, ecoguard_status: str, run_id: str = "") -> int:
        _check_user_id(user_id)
        if ecoguard_status not in (ACCEPT, REVIEW):
            raise GreenScoreError(f"ecoguard_status must be ACCEPT or REVIEW, got {ecoguard_status!r}")
        if ecoguard_status == ACCEPT:
            self._validate_bin(recommended_action, "recommended_action")
        elif recommended_action != MANUAL_REVIEW:
            raise GreenScoreError("REVIEW events must carry recommended_action MANUAL_REVIEW")
        if not image_id or len(image_id) > 255:
            raise GreenScoreError("image_id must be 1-255 characters")
        with closing(self._connect()) as conn:
            if conn.execute("SELECT 1 FROM users WHERE user_id = ?", (user_id,)).fetchone() is None:
                raise GreenScoreError(f"unknown user {user_id!r}; add it first with add-user")
            try:
                cur = conn.execute(
                    "INSERT INTO sorting_events (user_id, image_id, run_id, predicted_category, recommended_action,"
                    " ecoguard_status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (user_id, image_id, run_id, predicted_category, recommended_action, ecoguard_status, _now()),
                )
            except sqlite3.IntegrityError as exc:
                raise GreenScoreError(f"event for user {user_id!r}, image {image_id!r}, run {run_id!r} already exists") from exc
            return int(cur.lastrowid)

    def get_event(self, event_id: int) -> sqlite3.Row:
        with closing(self._connect()) as conn:
            row = conn.execute("SELECT * FROM sorting_events WHERE event_id = ?", (int(event_id),)).fetchone()
        if row is None:
            raise GreenScoreError(f"unknown event id {event_id}")
        return row

    def list_events(self, status: Optional[str] = None, user_id: Optional[str] = None) -> list[sqlite3.Row]:
        q, params = "SELECT * FROM sorting_events WHERE 1=1", []
        if status:
            q += " AND verification_status = ?"
            params.append(status)
        if user_id:
            q += " AND user_id = ?"
            params.append(_check_user_id(user_id))
        with closing(self._connect()) as conn:
            return conn.execute(q + " ORDER BY event_id", params).fetchall()

    def verify(self, event_id: int, observed_action: str, correct_action: Optional[str] = None,
               supervisor_id: Optional[str] = None) -> VerificationResult:
        """Record the supervisor's check. Awards points only if observed bin == correct bin."""
        self._validate_bin(observed_action, "observed bin")
        event = self.get_event(event_id)
        if event["verification_status"] != PENDING:
            raise AlreadyVerifiedError(f"event {event_id} was already verified ({event['verification_status']})")
        if correct_action is None:
            if event["ecoguard_status"] != ACCEPT:
                raise GreenScoreError(
                    f"event {event_id} was flagged REVIEW by EcoGuard: the supervisor must state the correct bin")
            correct_action = event["recommended_action"]
        self._validate_bin(correct_action, "correct bin")
        if supervisor_id is not None:
            _check_user_id(supervisor_id)

        is_correct = observed_action == correct_action
        status = VERIFIED_CORRECT if is_correct else VERIFIED_INCORRECT
        points = self.points_correct if is_correct else self.points_incorrect
        with closing(self._connect()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                cur = conn.execute(
                    "UPDATE sorting_events SET verification_status = ?, observed_action = ?, verified_action = ?,"
                    " supervisor_id = ?, awarded_points = ?, verified_at = ?"
                    " WHERE event_id = ? AND verification_status = 'PENDING'",
                    (status, observed_action, correct_action, supervisor_id, points, _now(), int(event_id)),
                )
                if cur.rowcount != 1:
                    raise AlreadyVerifiedError(f"event {event_id} was verified concurrently")
                conn.execute("UPDATE users SET total_points = total_points + ? WHERE user_id = ?",
                             (points, event["user_id"]))
                total = conn.execute("SELECT total_points FROM users WHERE user_id = ?",
                                     (event["user_id"],)).fetchone()["total_points"]
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise
        return VerificationResult(int(event_id), status, observed_action, correct_action, points, int(total))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _store_from_config(args) -> GreenScore:
    from .config import load_app_config
    from .imagenet_labels import load_labels
    from .materialwise import load_mapping

    app = load_app_config(args.config) if args.config else load_app_config()
    actions = load_mapping(app.mapping, load_labels(app.labels)).actions
    db = Path(args.db) if args.db else app.greenscore_db
    db.parent.mkdir(parents=True, exist_ok=True)
    return GreenScore(db, app.points_correct, app.points_incorrect, actions)


def _prompt_choice(prompt: str, choices: list[str]) -> str:
    while True:
        ans = input(f"{prompt} {choices}: ").strip().upper()
        if ans in choices:
            return ans
        print(f"  please type one of {choices}")


def _interactive_verify(gs: GreenScore, event_id: int) -> tuple[str, Optional[str]]:
    ev = gs.get_event(event_id)
    bins = sorted(gs.valid_actions or [])
    print(f"\nSupervisor check for event {event_id}: user {ev['user_id']}, image {ev['image_id']}")
    print(f"  Zyqora predicted category: {ev['predicted_category']}  |  EcoGuard: {ev['ecoguard_status']}"
          f"  |  recommendation: {ev['recommended_action']}")
    print("  Look at the bin the item was ACTUALLY placed in.")
    observed = _prompt_choice("  Observed bin", bins)
    if ev["ecoguard_status"] == ACCEPT:
        ok = input(f"  Was {ev['recommended_action']} the correct bin for this item? [y/n]: ").strip().lower()
        correct = ev["recommended_action"] if ok.startswith("y") else _prompt_choice("  Correct bin", bins)
    else:
        correct = _prompt_choice("  EcoGuard asked for review. Correct bin for this item", bins)
    return observed, correct


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m zyqora.greenscore", description="GreenScore verified rewards")
    ap.add_argument("--config", default=None, help="config/app.yaml (default)")
    ap.add_argument("--db", default=None, help="override database path")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("add-user", help="register a participant")
    p.add_argument("--user-id", required=True)
    p.add_argument("--name", default=None)
    p = sub.add_parser("record", help="record a PENDING sorting event manually")
    p.add_argument("--user-id", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--predicted-category", default=None)
    p.add_argument("--action", required=True, help="recommended action (MANUAL_REVIEW for REVIEW items)")
    p.add_argument("--ecoguard", choices=[ACCEPT, REVIEW], required=True)
    p = sub.add_parser("events", help="list sorting events")
    p.add_argument("--status", choices=[PENDING, VERIFIED_CORRECT, VERIFIED_INCORRECT], default=None)
    p.add_argument("--user-id", default=None)
    p = sub.add_parser("verify", help="supervisor verification of one event")
    p.add_argument("--event-id", type=int, required=True)
    p.add_argument("--observed", default=None, help="bin the item was actually placed in")
    p.add_argument("--correct", default=None, help="bin it should be in (default: AI recommendation if ACCEPT)")
    p.add_argument("--supervisor", default=None)
    p.add_argument("--voice", choices=["off", "en", "ta"], default="off")
    p = sub.add_parser("points", help="show a user's verified points")
    p.add_argument("--user-id", required=True)
    p = sub.add_parser("history", help="show a user's events")
    p.add_argument("--user-id", required=True)
    args = ap.parse_args(argv)

    try:
        gs = _store_from_config(args)
        if args.cmd == "add-user":
            gs.add_user(args.user_id, args.name)
            print(f"Added user {args.user_id}")
        elif args.cmd == "record":
            eid = gs.record_event(args.user_id, args.image, args.predicted_category, args.action, args.ecoguard)
            print(f"Recorded PENDING event {eid}")
        elif args.cmd in ("events", "history"):
            rows = gs.list_events(getattr(args, "status", None), args.user_id)
            for r in rows:
                print(f"#{r['event_id']:<4} {r['user_id']:<10} {r['image_id']:<28} {r['ecoguard_status']:<6} "
                      f"rec={r['recommended_action']:<13} {r['verification_status']:<18} "
                      f"observed={r['observed_action'] or '-':<9} correct={r['verified_action'] or '-':<9} "
                      f"+{r['awarded_points']}")
            if not rows:
                print("(no events)")
        elif args.cmd == "verify":
            observed, correct = args.observed, args.correct
            if observed is None:
                if not sys.stdin.isatty():
                    raise GreenScoreError("--observed is required when not running interactively")
                observed, correct = _interactive_verify(gs, args.event_id)
            res = gs.verify(args.event_id, observed.upper(), correct.upper() if correct else None, args.supervisor)
            print(f"Event {res.event_id}: {res.status} (observed {res.observed_action}, correct {res.correct_action})"
                  f" -> +{res.awarded_points} points, total {res.total_points}")
            if args.voice != "off":
                from .voicesort import VoiceSort

                VoiceSort(args.voice).announce(res.status)
        elif args.cmd == "points":
            print(f"{args.user_id}: {gs.points(args.user_id)} verified GreenScore points")
    except ValueError as exc:  # GreenScoreError, ConfigError, MappingError are ValueErrors
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
