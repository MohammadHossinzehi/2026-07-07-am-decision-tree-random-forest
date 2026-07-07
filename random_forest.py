"""
random_forest.py

A from-scratch Random Forest classifier built on top of decision_tree.py.

Two sources of randomness decorrelate the individual trees, which is the
whole point of a random forest (a single deep tree overfits; averaging many
decorrelated overfit trees generalizes much better):

1. Bootstrap aggregating ("bagging"): each tree is trained on a random
   sample of the training rows, drawn with replacement.
2. Random feature subsampling: at every split, each tree only considers a
   random subset of the features (max_features), not all of them.

Predictions are made by majority vote across all trees.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from decision_tree import DecisionTreeClassifier


class RandomForestClassifier:
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        criterion: str = "gini",
        max_features: Optional[str] = "sqrt",
        bootstrap: bool = True,
        random_state: Optional[int] = None,
    ):
        self.n_estimators = n_estimators
        self.max_depth = maX_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.trees: List[DecisionTreeClassifier] = []
        self.n_classes_: int = 0
        self.n_features_: int = 0
        self.feature_importances_: Optional[np.ndarray] = None
        self._rng = np.random.RandomState(random_state)

    def _resolve_max_features(self) -> Optional[int]:
        if self.max_features is None:
            return None
        if self.max_features == "sqrt":
            return max(1, int(np.sqrt(self.n_features_)))
        if self.max_features == "log2":
            return max(1, int(np.log2(self.n_features_)))
        if isinstance(self.max_features, int):
            return min(self.max_features, self.n_features_)
        raise ValueError(f"Unrecognized max_features={self.max_features!r}")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestClassifier":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        n_samples, self.n_features_ = X.shape
        self.n_classes_ = int(y.max()) + 1 if len(y) else 0
        max_features = self._resolve_max_features()

        self.trees = []
        importances = np.zeros(self.n_features_)

        for i in range(self.n_estimators):
            seed = None if self.random_state is None else self.random_state + i
            if self.bootstrap:
                idx = self._rng.randint(0, n_samples, size=n_samples)
            else:
                idx = np.arange(n_samples)

            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                criterion=self.criterion,
                max_features=max_features,
                random_state=seed,
            )
            tree.fit(X[idx], y[idx])
            self.trees.append(tree)
            if tree.feature_importances_ is not None:
                importances += tree.feature_importances_

        total = importances.sum()
        self.feature_importances_ = importances / total if total > 0 else importances
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        probs = np.zeros((len(X), self.n_classes_))
        for tree in self.trees:
            probs += tree.predict_proba(X)
        return probs / len(self.trees)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)

    def oob_score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Out-of-bag accuracy estimate. Only valid when bootstrap=True and
        the forest was just fit on this exact (X, y). Re-derives, for each
        row, which trees never saw it during training (because it wasn't in
        that tree's bootstrap sample) and averages their votes.
        """
        if not self.bootstrap:
            raise ValueError("oob_score requires bootstrap=True")
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        n_samples = len(X)

        # Recompute which rows were used per tree using the same seeding scheme.
        rng = np.random.RandomState(self.random_state)
        votes = np.zeros((n_samples, self.n_classes_))
        voted_at_all = np.zeros(n_samples, dtype=bool)

        for tree in self.trees:
            idx = rng.randint(0, n_samples, size=n_samples)
            in_bag = np.zeros(n_samples, dtype=bool)
            in_bag[idx] = True
            oob_mask = ~in_bag
            if not oob_mask.any():
                continue
            preds = tree.predict_proba(X[oob_mask])
            votes[oob_mask] += preds
            voted_at_all[oob_mask] = True

        if not voted_at_all.any():
            return float("nan")
        final_preds = np.argmax(votes[voted_at_all], axis=1)
        return float(np.mean(final_preds == y[voted_at_all]))
