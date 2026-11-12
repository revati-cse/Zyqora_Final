import copy
import json
import tempfile
import unittest
from pathlib import Path

from tests.helpers import LABELS, MAPPING_DICT, mapping, probs_with
from zyqora.materialwise import MappingError, load_mapping, map_prediction, validate_mapping_dict
from zyqora.paths import DEFAULT_MAPPING
from zyqora.schemas import TOP1_MAPPED, TOP1_UNMAPPED, UNSUPPORTED


class MaterialWiseTests(unittest.TestCase):
    def test_valid_mapping(self):
        self.assertEqual(validate_mapping_dict(MAPPING_DICT, LABELS), [])
        m = mapping()
        self.assertEqual(m.categories["Organic"], "COMPOST")
        self.assertEqual(m.canonical_category("wet"), "Organic")
        self.assertEqual(m.canonical_category("RECYCLABLE"), "Recyclable")
        self.assertIsNone(m.canonical_category("Plastic"))

    def test_shipped_example_mapping_is_valid(self):
        m = load_mapping(DEFAULT_MAPPING, LABELS)
        self.assertGreaterEqual(len(m.categories), 2)
        self.assertEqual(m.status, "EXAMPLE")  # must stay EXAMPLE until organiser labels exist

    def test_conflicting_and_duplicate_mappings(self):
        d = copy.deepcopy(MAPPING_DICT)
        d["classes"].append({"index": 898, "label": "water bottle", "category": "Residual"})
        self.assertTrue(any("conflicting" in p for p in validate_mapping_dict(d, LABELS)))
        d = copy.deepcopy(MAPPING_DICT)
        d["classes"].append({"index": 898, "label": "water bottle", "category": "Recyclable"})
        self.assertTrue(any("duplicate" in p for p in validate_mapping_dict(d, LABELS)))

    def test_wrong_label_for_index_and_bad_values(self):
        d = copy.deepcopy(MAPPING_DICT)
        d["classes"][0]["label"] = "banana"
        d["classes"].append({"index": 1000, "label": "x", "category": "Organic"})
        d["classes"].append({"index": 951, "label": "lemon", "category": "Compostables"})
        d["categories"]["Organic"]["action"] = "compost bin"
        d["status"] = "FINAL"
        d["extra"] = 1
        problems = validate_mapping_dict(d, LABELS)
        joined = " ".join(problems)
        for needle in ("is 'water bottle'", "index 1000", "not a declared", "action", "status", "unknown top-level"):
            self.assertIn(needle, joined)

    def test_reserved_action_and_unused_category(self):
        d = copy.deepcopy(MAPPING_DICT)
        d["categories"]["Residual"]["action"] = "MANUAL_REVIEW"
        d["categories"]["Hazardous"] = {"action": "HAZARDOUS"}
        joined = " ".join(validate_mapping_dict(d, LABELS))
        self.assertIn("reserved", joined)
        self.assertIn("could never be predicted", joined)

    def test_load_mapping_errors(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "m.json"
            p.write_text("{not json")
            with self.assertRaises(MappingError):
                load_mapping(p, LABELS)
            with self.assertRaises(MappingError):
                load_mapping(Path(t) / "missing.json", LABELS)
            p.write_text(json.dumps({"status": "EXAMPLE"}))
            with self.assertRaises(MappingError):
                load_mapping(p, LABELS)

    def test_top1_mapped(self):
        r = map_prediction(probs_with(i898=0.8, i954=0.1), mapping())
        self.assertEqual(r.predicted_category, "Recyclable")
        self.assertEqual(r.mapping_status, TOP1_MAPPED)
        self.assertEqual(r.top1_rule.rule, "REC-01")
        self.assertEqual(r.category_action, "RECYCLE")

    def test_aggregation_beats_single_top_class(self):
        # Top single class is banana (0.30) but the two bottle classes together carry 0.50.
        r = map_prediction(probs_with(i954=0.30, i898=0.26, i737=0.24), mapping())
        self.assertEqual(r.predicted_category, "Recyclable")
        self.assertAlmostEqual(r.predicted_category_score, 0.50, places=9)
        self.assertAlmostEqual(r.category_scores["Organic"], 0.30, places=9)
        self.assertAlmostEqual(r.category_margin, 0.20, places=9)
        self.assertEqual(r.top1_rule.label, "banana")

    def test_unmapped_top1_still_gives_explicit_category(self):
        # 'tiger cat' (282) unmapped top-1; category still predicted from mapped mass.
        r = map_prediction(probs_with(i282=0.9, i950=0.05), mapping())
        self.assertEqual(r.mapping_status, TOP1_UNMAPPED)
        self.assertEqual(r.predicted_category, "Organic")
        self.assertIsNone(r.top1_rule)

    def test_unsupported_when_no_mapped_mass(self):
        import numpy as np

        p = np.zeros(1000)
        p[282] = 1.0
        r = map_prediction(p, mapping())
        self.assertEqual(r.mapping_status, UNSUPPORTED)
        self.assertIsNone(r.predicted_category)
        self.assertIsNone(r.category_action)

    def test_rejects_wrong_probability_shape(self):
        import numpy as np

        with self.assertRaises(ValueError):
            map_prediction(np.zeros(10), mapping())


if __name__ == "__main__":
    unittest.main()
