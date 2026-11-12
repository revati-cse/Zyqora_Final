import sqlite3
import tempfile
import threading
import unittest
from pathlib import Path

from tests import helpers  # noqa: F401
from zyqora.greenscore import (
    VERIFIED_CORRECT,
    VERIFIED_INCORRECT,
    AlreadyVerifiedError,
    GreenScore,
    GreenScoreError,
)

ACTIONS = ["COMPOST", "RECYCLE", "RESIDUAL"]


class GreenScoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "gs.sqlite3"
        self.gs = GreenScore(self.db, 10, 0, ACTIONS)
        self.gs.add_user("U01", "Team A")

    def tearDown(self):
        self.tmp.cleanup()

    def _accept_event(self, image="a.jpg"):
        return self.gs.record_event("U01", image, "Recyclable", "RECYCLE", "ACCEPT")

    def test_no_points_before_verification(self):
        self._accept_event()
        self.assertEqual(self.gs.points("U01"), 0)

    def test_correct_verification_awards_points(self):
        eid = self._accept_event()
        res = self.gs.verify(eid, "RECYCLE", supervisor_id="S1")
        self.assertEqual((res.status, res.awarded_points, res.total_points), (VERIFIED_CORRECT, 10, 10))
        ev = self.gs.get_event(eid)
        self.assertEqual((ev["observed_action"], ev["verified_action"], ev["supervisor_id"]), ("RECYCLE", "RECYCLE", "S1"))

    def test_failed_verification_awards_zero(self):
        eid = self._accept_event()
        res = self.gs.verify(eid, "COMPOST")
        self.assertEqual((res.status, res.awarded_points), (VERIFIED_INCORRECT, 0))
        self.assertEqual(self.gs.points("U01"), 0)

    def test_supervisor_can_overrule_wrong_ai_recommendation(self):
        eid = self._accept_event()
        res = self.gs.verify(eid, "RESIDUAL", correct_action="RESIDUAL")
        self.assertEqual((res.status, res.awarded_points), (VERIFIED_CORRECT, 10))

    def test_duplicate_verification_cannot_award_twice(self):
        eid = self._accept_event()
        self.gs.verify(eid, "RECYCLE")
        with self.assertRaises(AlreadyVerifiedError):
            self.gs.verify(eid, "RECYCLE")
        self.assertEqual(self.gs.points("U01"), 10)

    def test_concurrent_verification_awards_once(self):
        eid = self._accept_event()
        results = []

        def worker():
            try:
                results.append(GreenScore(self.db, 10, 0, ACTIONS).verify(eid, "RECYCLE").awarded_points)
            except AlreadyVerifiedError:
                results.append("dup")

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(results.count(10), 1)
        self.assertEqual(self.gs.points("U01"), 10)

    def test_review_event_requires_supervisor_correct_bin(self):
        eid = self.gs.record_event("U01", "b.jpg", "Organic", "MANUAL_REVIEW", "REVIEW")
        with self.assertRaises(GreenScoreError):
            self.gs.verify(eid, "COMPOST")
        self.assertEqual(self.gs.verify(eid, "COMPOST", correct_action="COMPOST").awarded_points, 10)

    def test_duplicate_event_rejected(self):
        self._accept_event("same.jpg")
        with self.assertRaises(GreenScoreError):
            self._accept_event("same.jpg")

    def test_invalid_inputs(self):
        with self.assertRaises(GreenScoreError):
            self.gs.add_user("bad id; DROP TABLE users")
        with self.assertRaises(GreenScoreError):
            self.gs.record_event("NOPE", "x.jpg", None, "RECYCLE", "ACCEPT")
        with self.assertRaises(GreenScoreError):
            self.gs.record_event("U01", "x.jpg", None, "MANUAL_REVIEW", "ACCEPT")
        with self.assertRaises(GreenScoreError):
            self.gs.record_event("U01", "x.jpg", None, "RECYCLE", "REVIEW")
        eid = self._accept_event()
        with self.assertRaises(GreenScoreError):
            self.gs.verify(eid, "LANDFILL")  # not a configured bin
        with self.assertRaises(GreenScoreError):
            self.gs.verify(eid, "recycle'; --")
        with self.assertRaises(GreenScoreError):
            self.gs.verify(9999, "RECYCLE")
        with self.assertRaises(GreenScoreError):
            GreenScore(self.db, -5, 0)
        with self.assertRaises(GreenScoreError):
            GreenScore(Path(self.tmp.name) / "missing_dir" / "x.db")

    def test_sql_injection_is_inert(self):
        self.gs.record_event("U01", "x'); DROP TABLE users; --.jpg", None, "RECYCLE", "ACCEPT")
        self.assertEqual(self.gs.points("U01"), 0)  # users table still exists

    def test_negative_balance_impossible(self):
        with sqlite3.connect(self.db) as conn:
            with self.assertRaises(sqlite3.IntegrityError):
                conn.execute("UPDATE users SET total_points = -1 WHERE user_id = 'U01'")

    def test_history_persists_across_instances(self):
        eid = self._accept_event()
        self.gs.verify(eid, "RECYCLE")
        again = GreenScore(self.db, 10, 0, ACTIONS)
        self.assertEqual(again.points("U01"), 10)
        self.assertEqual(len(again.list_events(user_id="U01")), 1)
        self.assertEqual(len(again.list_events(status="PENDING")), 0)


if __name__ == "__main__":
    unittest.main()
