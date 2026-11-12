import math
import unittest

from tests import helpers  # noqa: F401
from zyqora.ecoguard import EcoGuardConfig, EcoGuardConfigError, decide, disposal_action
from zyqora.schemas import (
    ACCEPT,
    MANUAL_REVIEW,
    R_AMBIGUOUS,
    R_INVALID,
    R_LOW_CONFIDENCE,
    R_NO_CATEGORY,
    R_TOP1_UNMAPPED,
    REVIEW,
    TOP1_MAPPED,
    TOP1_UNMAPPED,
    UNSUPPORTED,
)

CFG = EcoGuardConfig(threshold=0.40, min_margin=0.0, require_top1_mapped=True)


class EcoGuardTests(unittest.TestCase):
    def test_high_confidence_accepts(self):
        d = decide(0.91, TOP1_MAPPED, 0.8, CFG)
        self.assertEqual(d.status, ACCEPT)
        self.assertEqual(d.reason, "")
        self.assertEqual(disposal_action(d, "RECYCLE"), "RECYCLE")

    def test_low_confidence_reviews(self):
        d = decide(0.39, TOP1_MAPPED, 0.3, CFG)
        self.assertEqual((d.status, d.reasons), (REVIEW, (R_LOW_CONFIDENCE,)))
        self.assertEqual(disposal_action(d, "RECYCLE"), MANUAL_REVIEW)

    def test_threshold_boundary_is_inclusive(self):
        self.assertEqual(decide(0.40, TOP1_MAPPED, 0.4, CFG).status, ACCEPT)
        self.assertEqual(decide(0.3999999, TOP1_MAPPED, 0.4, CFG).status, REVIEW)
        self.assertEqual(decide(0.0, TOP1_MAPPED, 0.0, EcoGuardConfig(threshold=0.0)).status, ACCEPT)
        self.assertEqual(decide(1.0, TOP1_MAPPED, 1.0, EcoGuardConfig(threshold=1.0)).status, ACCEPT)

    def test_unmapped_reviews(self):
        d = decide(0.95, TOP1_UNMAPPED, 0.9, CFG)
        self.assertEqual((d.status, d.reasons), (REVIEW, (R_TOP1_UNMAPPED,)))
        relaxed = EcoGuardConfig(threshold=0.4, require_top1_mapped=False)
        self.assertEqual(decide(0.95, TOP1_UNMAPPED, 0.9, relaxed).status, ACCEPT)

    def test_unsupported_reviews(self):
        d = decide(None, UNSUPPORTED, None, CFG)
        self.assertEqual((d.status, d.reasons), (REVIEW, (R_NO_CATEGORY,)))
        self.assertEqual(disposal_action(d, None), MANUAL_REVIEW)

    def test_invalid_scores_review(self):
        for bad in (None, math.nan, math.inf, -0.1, 1.5, True):
            self.assertEqual(decide(bad, TOP1_MAPPED, 0.1, CFG).reasons, (R_INVALID,), bad)

    def test_ambiguity_rule(self):
        cfg = EcoGuardConfig(threshold=0.4, min_margin=0.10)
        self.assertEqual(decide(0.5, TOP1_MAPPED, 0.05, cfg).reasons, (R_AMBIGUOUS,))
        self.assertEqual(decide(0.5, TOP1_MAPPED, 0.10, cfg).status, ACCEPT)

    def test_multiple_reasons_reported(self):
        d = decide(0.1, TOP1_UNMAPPED, 0.0, EcoGuardConfig(threshold=0.4, min_margin=0.1))
        self.assertEqual(set(d.reasons), {R_TOP1_UNMAPPED, R_LOW_CONFIDENCE, R_AMBIGUOUS})

    def test_accept_without_action_is_still_manual_review(self):
        self.assertEqual(disposal_action(decide(0.9, TOP1_MAPPED, 0.9, CFG), None), MANUAL_REVIEW)

    def test_invalid_config(self):
        for kwargs in ({"threshold": 1.2}, {"threshold": -0.1}, {"threshold": math.nan},
                       {"threshold": "0.4"}, {"threshold": 0.4, "min_margin": 2}, {"threshold": 0.4, "frozen": "yes"}):
            with self.assertRaises(EcoGuardConfigError, msg=kwargs):
                EcoGuardConfig(**kwargs)
        with self.assertRaises(EcoGuardConfigError):
            EcoGuardConfig.from_dict({"min_margin": 0.1})
        with self.assertRaises(EcoGuardConfigError):
            EcoGuardConfig.from_dict({"threshold": 0.4, "typo": 1})


if __name__ == "__main__":
    unittest.main()
