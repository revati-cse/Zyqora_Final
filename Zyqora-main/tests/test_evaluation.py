import csv
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from tests.helpers import LABELS, ensure_torch_like, logits_for, mapping, probs_with
from zyqora.ecoguard import EcoGuardConfig
from zyqora.evaluator import (
    RESULT_COLUMNS,
    EvaluationRow,
    LabelsError,
    discover_images,
    percentile,
    read_labels_csv,
    reconcile,
    summarize,
    write_results_csv,
)
from zyqora.model_runtime import InferenceOutput
from zyqora.outputs import pick_samples, write_json
from zyqora.pipeline import ZyqoraPipeline
from zyqora.schemas import ACCEPT, IMAGE_ERROR, REVIEW, ImageResult

REQUIRED = ["image_filename", "ground_truth_category", "raw_top1_class", "raw_top1_probability",
            "predicted_category", "predicted_category_score", "mapping_status", "ecoguard_status",
            "review_reason", "disposal_action", "inference_time_ms", "correct"]


def row(pred, truth, status, err=None):
    return EvaluationRow(ImageResult(image_filename=f"{pred}-{truth}-{status}.jpg", predicted_category=pred,
                                     ecoguard_status=status, inference_time_ms=5.0, error=err), truth)


class FakeClassifier:
    """Returns pre-set logits by call order (no ExecuTorch needed)."""

    def __init__(self, logits_list):
        self.logits_list = list(logits_list)

    def infer(self, array):
        return InferenceOutput(self.logits_list.pop(0), 4_000_000)


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        ensure_torch_like()
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_accuracy_counts_review_in_denominator(self):
        rows = [row("Recyclable", "Recyclable", ACCEPT), row("Organic", "Organic", REVIEW),
                row("Organic", "Recyclable", ACCEPT), row(None, "Residual", REVIEW, err="corrupt")]
        s = summarize(rows, labelled=True)
        self.assertEqual(s["total_images"], 4)
        self.assertEqual(s["total_correct"], 2)
        self.assertEqual(s["overall_accuracy"], 0.5)
        self.assertEqual(s["accepted_count"], 2)
        self.assertEqual(s["review_count"], 2)
        self.assertEqual(s["review_rate"], 0.5)
        self.assertEqual(s["accepted_only_accuracy"], 0.5)
        self.assertEqual(s["reviewed_but_category_correct"], 1)
        self.assertEqual(s["failed_images"], 1)
        self.assertEqual(s["images_without_category_prediction"], 1)

    def test_zero_accepted_is_undefined_not_100(self):
        s = summarize([row("Organic", "Organic", REVIEW), row("Organic", "Recyclable", REVIEW)], labelled=True)
        self.assertIsNone(s["accepted_only_accuracy"])
        self.assertIn("undefined", s["accepted_only_accuracy_note"])
        self.assertEqual(s["overall_accuracy"], 0.5)

    def test_missing_prediction_is_incorrect(self):
        self.assertFalse(row(None, "Organic", REVIEW).correct)
        self.assertIsNone(row("Organic", None, REVIEW).correct)

    def test_unlabelled_summary(self):
        s = summarize([row("Organic", None, ACCEPT)], labelled=False)
        self.assertIsNone(s["overall_accuracy"])

    def test_percentile_matches_numpy(self):
        vals = [5.0, 1.0, 9.0, 3.0, 7.0, 2.0]
        try:
            import numpy as np
            self.assertAlmostEqual(percentile(vals, 95), float(np.percentile(vals, 95)))
        except ImportError:
            pass
        self.assertIsNone(percentile([], 50))

    def _labels_file(self, rows, header=("image_filename", "ground_truth_category")):
        p = self.dir / "labels.csv"
        with p.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)
        return p

    def test_labels_csv_parsing_and_errors(self):
        m = mapping()
        got = read_labels_csv(self._labels_file([("a.jpg", "Recyclable"), ("b.jpg", "wet")]), m)
        self.assertEqual(got, {"a.jpg": "Recyclable", "b.jpg": "Organic"})
        with self.assertRaises(LabelsError):
            read_labels_csv(self._labels_file([("a.jpg", "Plastic")]), m)
        with self.assertRaises(LabelsError):
            read_labels_csv(self._labels_file([("a.jpg", "Organic"), ("a.jpg", "Organic")]), m)
        with self.assertRaises(LabelsError):
            read_labels_csv(self._labels_file([("a.jpg", "Organic")], header=("file", "truth")), m)
        with self.assertRaises(LabelsError):
            read_labels_csv(self.dir / "missing.csv", m)

    def test_reconcile_missing_labels_and_images(self):
        imgs = [("a.jpg", Path("a.jpg")), ("b.jpg", Path("b.jpg"))]
        reconcile(imgs, {"a.jpg": "Organic", "b.jpg": "Organic"})
        with self.assertRaises(LabelsError):
            reconcile(imgs, {"a.jpg": "Organic"})
        with self.assertRaises(LabelsError):
            reconcile(imgs, {"a.jpg": "Organic", "b.jpg": "Organic", "c.jpg": "Organic"})

    def test_pipeline_csv_and_json_outputs(self):
        for name in ("bottle.jpg", "banana.png", "cat.jpg"):
            Image.new("RGB", (320, 240), (100, 100, 100)).save(self.dir / name)
        (self.dir / "broken.jpg").write_bytes(b"garbage")
        imgs = discover_images(self.dir)
        self.assertEqual([n for n, _ in imgs], ["banana.png", "bottle.jpg", "broken.jpg", "cat.jpg"])
        clf = FakeClassifier([
            logits_for(probs_with(i954=0.9)),            # banana -> Organic ACCEPT
            logits_for(probs_with(i898=0.3, i737=0.25)),  # bottle -> Recyclable 0.55 ACCEPT
            logits_for(probs_with(i282=0.9, i950=0.02)),  # cat -> Organic, top-1 unmapped -> REVIEW
        ])
        pipe = ZyqoraPipeline(clf, LABELS, mapping(), EcoGuardConfig(threshold=0.4))
        truth = {"banana.png": "Organic", "bottle.jpg": "Recyclable", "broken.jpg": "Residual", "cat.jpg": "Residual"}
        rows = [EvaluationRow(pipe.process(p, n), truth[n]) for n, p in imgs]
        by = {r.result.image_filename: r for r in rows}
        self.assertEqual(by["banana.png"].result.disposal_action, "COMPOST")
        self.assertEqual(by["bottle.jpg"].result.predicted_category, "Recyclable")
        self.assertAlmostEqual(by["bottle.jpg"].result.predicted_category_score, 0.55, places=4)
        self.assertEqual(by["broken.jpg"].result.mapping_status, IMAGE_ERROR)
        self.assertEqual(by["cat.jpg"].result.ecoguard_status, REVIEW)
        self.assertEqual(by["cat.jpg"].result.predicted_category, "Organic")  # kept despite REVIEW
        self.assertEqual(by["cat.jpg"].result.disposal_action, "MANUAL_REVIEW")

        out = self.dir / "out"
        write_results_csv(rows, out / "results.csv")
        with (out / "results.csv").open() as f:
            reader = csv.DictReader(f)
            data = list(reader)
            self.assertEqual(reader.fieldnames[: len(REQUIRED)], REQUIRED)
        self.assertEqual(reader.fieldnames, RESULT_COLUMNS)
        self.assertEqual(len(data), 4)
        self.assertEqual(sum(d["correct"] == "True" for d in data), 2)

        s = summarize(rows, labelled=True)
        write_json(s, out / "summary.json")
        loaded = json.loads((out / "summary.json").read_text())
        for key in ("total_images", "total_correct", "total_incorrect", "overall_accuracy", "accepted_count",
                    "review_count", "review_rate", "accepted_only_accuracy", "inference_latency_mean_ms",
                    "inference_latency_median_ms", "inference_latency_p95_ms"):
            self.assertIn(key, loaded)
        self.assertEqual(loaded["total_images"], 4)
        self.assertEqual(loaded["overall_accuracy"], 0.5)

        samples = pick_samples([(r.result, p) for r, (_, p) in zip(rows, imgs)], 2)
        self.assertEqual({s[0].ecoguard_status for s in samples}, {ACCEPT, REVIEW})

    def test_pick_samples_does_not_fabricate_review(self):
        only_accept = [(ImageResult(image_filename=f"{i}.jpg", ecoguard_status=ACCEPT), Path(f"{i}.jpg")) for i in range(3)]
        picks = pick_samples(only_accept, 2)
        self.assertEqual([p[0].ecoguard_status for p in picks], [ACCEPT, ACCEPT])


if __name__ == "__main__":
    unittest.main()
