import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


class TestMetrics(unittest.TestCase):
    def test_accuracy_perfect(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        self.assertEqual(accuracy_score(y_true, y_pred), 1.0)

    def test_accuracy_half(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 0, 0, 0])
        self.assertEqual(accuracy_score(y_true, y_pred), 0.5)

    def test_confusion_matrix_binary(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1])
        cm = confusion_matrix(y_true, y_pred)
        expected = np.array([[1, 1], [0, 2]])
        np.testing.assert_array_equal(cm, expected)

    def test_precision_recall_known_values(self):
        # class 1: TP=2, FP=1, FN=0 -> precision=2/3, recall=1.0
        # class 0: TP=1, FP=0, FN=1 -> precision=1.0, recall=0.5
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1])
        prec = precision_score(y_true, y_pred, average=None)
        rec = recall_score(y_true, y_pred, average=None)
        np.testing.assert_allclose(prec, [1.0, 2 / 3])
        np.testing.assert_allclose(rec, [0.5, 1.0])

    def test_f1_matches_manual_harmonic_mean(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1])
        f1 = f1_score(y_true, y_pred, average=None)
        p, r = 1.0, 0.5
        expected_f1_class0 = 2 * p * r / (p + r)
        self.assertAlmostEqual(f1[0], expected_f1_class0)

    def test_macro_average_is_mean_of_per_class(self):
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 1, 1, 1, 2, 0])
        macro_f1 = f1_score(y_true, y_pred, average="macro")
        per_class_f1 = f1_score(y_true, y_pred, average=None)
        self.assertAlmostEqual(macro_f1, per_class_f1.mean())

    def test_empty_arrays(self):
        self.assertEqual(accuracy_score(np.array([]), np.array([])), 0.0)


if __name__ == "__main__":
    unittest.main()
