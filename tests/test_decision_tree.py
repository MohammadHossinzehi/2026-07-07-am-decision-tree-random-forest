import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from decision_tree import DecisionTreeClassifier, gini, entropy


class TestImpurityFunctions(unittest.TestCase):
    def test_gini_pure_node(self):
        y = np.array([1, 1, 1, 1])
        self.assertAlmostEqual(gini(y), 0.0)

    def test_gini_even_binary_split(self):
        y = np.array([0, 0, 1, 1])
        self.assertAlmostEqual(gini(y), 0.5)

    def test_entropy_pure_node(self):
        y = np.array([0, 0, 0])
        self.assertAlmostEqual(entropy(y), 0.0)

    def test_entropy_even_binary_split(self):
        y = np.array([0, 1])
        self.assertAlmostEqual(entropy(y), 1.0)

    def test_gini_empty(self):
        self.assertEqual(gini(np.array([], dtype=int)), 0.0)


class TestDecisionTreeClassifier(unittest.TestCase):
    def test_perfectly_separable_1d(self):
        X = np.array([[0.0], [1.0], [2.0], [10.0], [11.0], [12.0]])
        y = np.array([0, 0, 0, 1, 1, 1])
        tree = DecisionTreeClassifier(random_state=0)
        tree.fit(X, y)
        preds = tree.predict(X)
        np.testing.assert_array_equal(preds, y)

    def test_perfectly_separable_2d(self):
        rng = np.random.RandomState(0)
        cluster_a = rng.normal(loc=[0, 0], scale=0.3, size=(30, 2))
        cluster_b = rng.normal(loc=[10, 10], scale=0.3, size=(30, 2))
        X = np.vstack([cluster_a, cluster_b])
        y = np.concatenate([np.zeros(30, dtype=int), np.ones(30, dtype=int)])

        tree = DecisionTreeClassifier(random_state=0)
        tree.fit(X, y)
        preds = tree.predict(X)
        self.assertGreaterEqual(np.mean(preds == y), 0.95)

    def test_max_depth_is_respected(self):
        rng = np.random.RandomState(1)
        X = rng.uniform(-5, 5, size=(200, 3))
        y = (X[:, 0] + X[:, 1] - X[:, 2] > 0).astype(int)
        tree = DecisionTreeClassifier(max_depth=2, random_state=0)
        tree.fit(X, y)
        self.assertLessEqual(tree.depth(), 2)

    def test_single_class_input_returns_leaf(self):
        X = np.array([[1.0], [2.0], [3.0]])
        y = np.array([1, 1, 1])
        tree = DecisionTreeClassifier()
        tree.fit(X, y)
        self.assertTrue(tree.root.is_leaf())
        preds = tree.predict(X)
        np.testing.assert_array_equal(preds, [1, 1, 1])

    def test_predict_proba_sums_to_one(self):
        rng = np.random.RandomState(2)
        X = rng.uniform(-5, 5, size=(100, 2))
        y = (X[:, 0] > 0).astype(int)
        tree = DecisionTreeClassifier(random_state=0)
        tree.fit(X, y)
        proba = tree.predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), np.ones(len(X)))

    def test_entropy_criterion_also_fits(self):
        X = np.array([[0.0], [1.0], [2.0], [10.0], [11.0], [12.0]])
        y = np.array([0, 0, 0, 1, 1, 1])
        tree = DecisionTreeClassifier(criterion="entropy", random_state=0)
        tree.fit(X, y)
        preds = tree.predict(X)
        np.testing.assert_array_equal(preds, y)

    def test_unbounded_tree_overfits_more_than_shallow_tree(self):
        rng = np.random.RandomState(3)
        X = rng.uniform(-5, 5, size=(300, 2))
        y = (np.sin(X[:, 0]) + rng.normal(scale=0.5, size=300) > 0).astype(int)
        split = 200

        deep = DecisionTreeClassifier(random_state=0)
        deep.fit(X[:split], y[:split])

        shallow = DecisionTreeClassifier(max_depth=2, random_state=0)
        shallow.fit(X[:split], y[:split])

        self.assertGreater(deep.depth(), shallow.depth())

    def test_invalid_criterion_raises(self):
        with self.assertRaises(ValueError):
            DecisionTreeClassifier(criterion="nonsense")

    def test_feature_importances_sum_to_one(self):
        rng = np.random.RandomState(4)
        X = rng.uniform(-5, 5, size=(150, 4))
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        tree = DecisionTreeClassifier(random_state=0)
        tree.fit(X, y)
        self.assertAlmostEqual(tree.feature_importances_.sum(), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
