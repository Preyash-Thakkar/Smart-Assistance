"""Regression tests for evaluation metric averaging.

The first version of ``evaluate.py`` averaged macro/weighted F1 over the union
of true and predicted labels. A single prediction of a class absent from the
split added a support-0 label and dropped 11-intent validation macro-F1 from
90.31% to 45.15%, which made config selection meaningless.

Run with::

    python -m unittest discover -s tests
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.evaluate import compute_metrics  # noqa: E402


class ComputeMetricsTest(unittest.TestCase):
    def test_predicted_absent_class_does_not_dilute_macro_f1(self):
        # One prediction of "C" while the split only contains A and B.
        y_true = ["A", "A", "B", "B"]
        y_pred = ["A", "A", "B", "C"]

        accuracy, macro, weighted, report, _, labels = compute_metrics(y_true, y_pred)

        self.assertEqual(labels, ["A", "B"])
        self.assertAlmostEqual(accuracy, 0.75)
        # A scores 1.0, B scores 2/3 -> 5/6. With a support-0 "C" it was 5/9.
        self.assertAlmostEqual(macro, 5 / 6, places=6)
        self.assertAlmostEqual(weighted, 5 / 6, places=6)
        self.assertNotIn("C", report)

    def test_perfect_predictions_score_one(self):
        y_true = ["A", "B", "C"] * 3

        accuracy, macro, weighted, _, _, labels = compute_metrics(y_true, list(y_true))

        self.assertEqual(accuracy, 1.0)
        self.assertEqual(macro, 1.0)
        self.assertEqual(weighted, 1.0)
        self.assertEqual(labels, ["A", "B", "C"])

    def test_confusion_matrix_rows_sum_to_support(self):
        y_true = ["A", "A", "B", "C"]
        y_pred = ["A", "B", "B", "C"]

        _, _, _, report, matrix, labels = compute_metrics(y_true, y_pred)

        self.assertEqual(labels, ["A", "B", "C"])
        self.assertEqual([int(row.sum()) for row in matrix], [2, 1, 1])
        self.assertEqual(report["A"]["support"], 2)
        self.assertEqual(report["B"]["support"], 1)


if __name__ == "__main__":
    unittest.main()
