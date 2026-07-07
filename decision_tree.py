"""
decision_tree.py

A CART (Classification And Regression Trees) style decision tree classifier
built from scratch using only numpy for array math. No scikit-learn or any
other ML library is used anywhere in this project.

The tree is grown by recursively picking the (feature, threshold) split that
most reduces impurity (Gini index or entropy), stopping when a stopping rule
is hit (max depth, minimum samples to split, or a pure node).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


def gini(y: np.ndarray) -> float:
    """Gini impurity of a label array. 0 means perfectly pure."""
    if len(y) == 0:
        return 0.0
    counts = np.bincount(y)
    probs = counts / len(y)
    return 1.0 - np.sum(probs ** 2)


def entropy(y: np.ndarray) -> float:
    """Shannon entropy (base 2) of a label array."""
    if len(y) == 0:
        return 0.0
    counts = np.bincount(y)
    probs = counts[counts > 0] / len(y)
    return -np.sum(probs * np.log2(probs))


_CRITERIA = {"gini": gini, "entropy": entropy}


@dataclass
class Node:
    """A single node in the tree. Leaf nodes have `value` set and no children."""

    feature_index: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional["Node"] = None
    right: Optional["Node"] = None
    value: Optional[int] = None
    class_counts: Optional[np.ndarray] = None
    impurity_decrease: float = 0.0
    n_samples: int = 0

    def is_leaf(self) -> bool:
        return self.value is not None


class DecisionTreeClassifier:
    """A from-scratch binary-split decision tree classifier.

    Parameters
    ----------
    max_depth: maximum depth of the tree. None means grow until pure/stopping
        rule is hit.
    min_samples_split: a node must have at least this many samples to be
        considered for splitting.
    min_samples_leaf: a split is only accepted if both children end up with
        at least this many samples.
    criterion: "gini" or "entropy".
    max_features: number of features to consider at each split (used by
        RandomForestClassifier to decorrelate trees). None means use all
        features.
    random_state: seed for the feature-subsampling RNG.
    """

    def __init__(
        self,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        criterion: str = "gini",
        max_features: Optional[int] = None,
        random_state: Optional[int] = None,
    ):
        if criterion not in _CRITERIA:
            raise ValueError(f"Unknown criterion {criterion!r}, expected one of {list(_CRITERIA)}")
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.max_features = max_features
        self.random_state = random_state
        self._impurity_fn = _CRITERIA[criterion]
        self.root: Optional[Node] = None
        self.n_classes_: int = 0
        self.n_features_: int = 0
        self.feature_importances_: Optional[np.ndarray] = None
        self._rng = np.random.RandomState(random_state)

    # -- public API ---------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTreeClassifier":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        if X.ndim != 2:
            raise ValueError("X must be 2D (n_samples, n_features)")
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of rows")

        self.n_features_ = X.shape[1]
        self.n_classes_ = int(y.max()) + 1 if len(y) else 0
        self._feature_gain = np.zeros(self.n_features_)
        self.root = self._grow(X, y, depth=0)

        total_gain = self._feature_gain.sum()
        if total_gain > 0:
            self.feature_importances_ = self._feature_gain / total_gain
        else:
            self.feature_importances_ = np.zeros(self.n_features_)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return np.array([self._predict_one(row, self.root) for row in X])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        out = np.zeros((len(X), self.n_classes_))
        for i, row in enumerate(X):
            node = self.root
            while not node.is_leaf():
                node = node.left if row[node.feature_index] <= node.threshold else node.right
            out[i] = node.class_counts / node.class_counts.sum()
        return out

    def depth(self) -> int:
        def _depth(node: Optional[Node]) -> int:
            if node is None or node.is_leaf():
                return 0
            return 1 + max(_depth(node.left), _depth(node.right))

        return _depth(self.root)

    def n_leaves(self) -> int:
        def _count(node: Optional[Node]) -> int:
            if node is None:
                return 0
            if node.is_leaf():
                return 1
            return _count(node.left) + _count(node.right)

        return _count(self.root)

    # -- internals ------------------------------------------------------

    def _predict_one(self, row: np.ndarray, node: Node) -> int:
        while not node.is_leaf():
            node = node.left if row[node.feature_index] <= node.threshold else node.right
        return node.value

    def _make_leaf(self, y: np.ndarray) -> Node:
        counts = np.bincount(y, minlength=self.n_classes_)
        return Node(value=int(np.argmax(counts)), class_counts=counts, n_samples=len(y))

    def _grow(self, X: np.ndarray, y: np.ndarray, depth: int) -> Node:
        n_samples = len(y)
        if n_samples == 0:
            return self._make_leaf(np.array([0], dtype=int))

        if (
            n_samples < self.min_samples_split
            or (self.max_depth is not None and depth >= self.max_depth)
            or len(np.unique(y)) == 1
        ):
            return self._make_leaf(y)

        split = self._best_split(X, y)
        if split is None:
            return self._make_leaf(y)

        feature_index, threshold, gain, left_mask = split
        self._feature_gain[feature_index] += gain * n_samples

        left = self._grow(X[left_mask], y[left_mask], depth + 1)
        right = self._grow(X[~left_mask], y[~left_mask], depth + 1)
        counts = np.bincount(y, minlength=self.n_classes_)
        return Node(
            feature_index=feature_index,
            threshold=threshold,
            left=left,
            right=right,
            impurity_decrease=gain,
            n_samples=n_samples,
            class_counts=counts,
        )

    def _candidate_features(self) -> np.ndarray:
        if self.max_features is None or self.max_features >= self.n_features_:
            return np.arange(self.n_features_)
        return self._rng.choice(self.n_features_, size=self.max_features, replace=False)

    def _best_split(self, X: np.ndarray, y: np.ndarray):
        n_samples = len(y)
        parent_impurity = self._impurity_fn(y)
        best_gain = 0.0
        best = None

        for feature_index in self._candidate_features():
            values = X[:, feature_index]
            order = np.argsort(values)
            sorted_vals = values[order]
            sorted_y = y[order]

            # Candidate thresholds are midpoints between consecutive distinct values.
            distinct = np.where(np.diff(sorted_vals) != 0)[0]
            for i in distinct:
                left_size = i + 1
                right_size = n_samples - left_size
                if left_size < self.min_samples_leaf or right_size < self.min_samples_leaf:
                    continue
                threshold = (sorted_vals[i] + sorted_vals[i + 1]) / 2.0
                left_y = sorted_y[: i + 1]
                right_y = sorted_y[i + 1 :]

                left_impurity = self._impurity_fn(left_y)
                right_impurity = self._impurity_fn(right_y)
                weighted = (left_size * left_impurity + right_size * right_impurity) / n_samples
                gain = parent_impurity - weighted

                if gain > best_gain:
                    left_mask = X[:, feature_index] <= threshold
                    best_gain = gain
                    best = (feature_index, threshold, gain, left_mask)

        if best is None or best_gain <= 1e-12:
            return None
        return best
