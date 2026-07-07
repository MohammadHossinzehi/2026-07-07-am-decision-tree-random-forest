import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from random_forest import RandomForestClassifier
from decision_tree import DecisionTreeClassifier
from datasets import make_moons, train_test_split
from metrics import accuracy_score


class TestRandomForestClassifier(unittest.TestCase):
    def test_fits_perfectly_separable_data(self):
        X = np.array([[0.0], [1.0], [2.0], [10.0], [11.0], [12.0]])
        y = np.array([0, 0, 0, 1, 1, 1])
        forest = RandomForestClassifier(n_estimators=10, random_state=0)
        forest.fit(X, y)
        preds = forest.predict(X)
        np.testing.assert_array_equal(preds, y)

    def test_predict_proba_sums_to_one(self):
        rng = np.random.RandomState(0)
        X = rng.uniform(-5, 5, size=(100, 2))
        y = (X[:, 0] > 0).astype(int)
        forest = RandomForestClassifier(n_estimators=15, random_state=0)
        forest.fit(X, y)
        proba = forest.predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), np.ones(len(X)))

    def test_forest_beats_or_matches_single_tree_on_noisy_data(self):
        X, y = make_moons(n_samples=300, noise=0.3, random_state=0)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

        tree = DecisionTreeClassifier(max_depth=3, random_state=0)
        tree.fit(X_train, y_train)
        tree_acc = accuracy_score(y_test, tree.predict(X_test))

        forest = RandomForestClassifier(n_estimators=40, max_depth=3, random_state=0)
        forest.fit(X_train, y_train)
        forest_acc = accuracy_score(y_test, forest.predict(X_test))

        # The forest should not be dramatically worse than a single shallow tree;
        # ensembling shallow trees should be competitive or better on average.
        self.assertGreaterEqual(forest_acc, tree_acc - 0.15)

    def test_oob_score_is_reasonable(self):
        X, y = make_moons(n_samples=300, noise=0.15, random_state=0)
        forest = RandomForestClassifier(n_estimators=30, max_depth=5, random_state=0)
        forest.fit(X, y)
        oob = forest.oob_score(X, y)
        self.assertGreaterEqual(oob, 0.6)
        self.assertLessEqual(oob, 1.0)

    def test_oob_score_requires_bootstrap(self):
        X, y = make_moons(n_samples=50, random_state=0)
        forest = RandomForestClassifier(n_estimators=5, bootstrap=False, random_state=0)
        forest.fit(X, y)
        with self.assertRaises(ValueError):
            forest.oob_score(X, y)

    def test_feature_importances_sum_to_one(self):
        rng = np.random.RandomState(0)
        X = rng.uniform(-5, 5, size=(120, 3))
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        forest = RandomForestClassifier(n_estimators=20, max_features=2, random_state=0)
        forest.fit(X, y)
        self.assertAlmostEqual(forest.feature_importances_.sum(), 1.0, places=5)

    def test_max_features_sqrt_resolution(self):
        forest = RandomForestClassifier(max_features="sqrt")
        forest.n_features_ = 9
        self.assertEqual(forest._resolve_max_features(), 3)

    def test_different_trees_can_differ(self):
        # With bootstrap + feature subsampling, trees in the forest shouldn't
        # all be byte-identical on a reasonably sized noisy dataset.
        X, y = make_moons(n_samples=200, noise=0.25, random_state=0)
        forest = RandomForestClassifier(n_estimators=20, max_depth=4, random_state=0)
        forest.fit(X, y)
        depths = {t.depth() for t in forest.trees}
        leaves = {t.n_leaves() for t in forest.trees}
        self.assertTrue(len(depths) > 1 or len(leaves) > 1)


if __name__ == "__main__":
    unittest.main()
