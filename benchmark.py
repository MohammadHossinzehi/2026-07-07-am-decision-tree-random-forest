"""
benchmark.py

Runnable demo / benchmark script. Trains a single DecisionTreeClassifier and
a RandomForestClassifier on both an easy dataset (blobs) and a hard,
non-linearly-separable dataset (moons), then reports accuracy, precision,
recall, F1, tree depth, and out-of-bag score.

Run with: python benchmark.py
"""

from __future__ import annotations

import time

import numpy as np

from datasets import make_blobs, make_moons, train_test_split
from decision_tree import DecisionTreeClassifier
from random_forest import RandomForestClassifier
from metrics import accuracy_score, precision_score, recall_score, f1_score


def evaluate(name: str, model, X_train, X_test, y_train, y_test):
    start = time.perf_counter()
    model.fit(X_train, y_train)
    fit_time = time.perf_counter() - start

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    extra = ""
    if isinstance(model, DecisionTreeClassifier):
        extra = f" | depth={model.depth()} leaves={model.n_leaves()}"
    elif isinstance(model, RandomForestClassifier):
        oob = model.oob_score(X_train, y_train)
        extra = f" | oob_acc={oob:.3f}"

    print(
        f"{name:<28} acc={acc:.3f} precision={prec:.3f} recall={rec:.3f} "
        f"f1={f1:.3f} fit_time={fit_time*1000:.1f}ms{extra}"
    )
    return acc


def run_dataset(dataset_name: str, X, y):
    print(f"\n=== {dataset_name} (n={len(X)}, features={X.shape[1]}, classes={len(np.unique(y))}) ===")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    evaluate(
        "DecisionTree (unbounded)",
        DecisionTreeClassifier(random_state=0),
        X_train, X_test, y_train, y_test,
    )
    evaluate(
        "DecisionTree (max_depth=3)",
        DecisionTreeClassifier(max_depth=3, random_state=0),
        X_train, X_test, y_train, y_test,
    )
    evaluate(
        "RandomForest (50 trees)",
        RandomForestClassifier(n_estimators=50, max_depth=6, random_state=0),
        X_train, X_test, y_train, y_test,
    )


def main():
    X_blobs, y_blobs = make_blobs(n_samples=400, centers=3, cluster_std=1.5, random_state=0)
    run_dataset("blobs (easy, roughly linearly separable)", X_blobs, y_blobs)

    X_moons, y_moons = make_moons(n_samples=400, noise=0.2, random_state=0)
    run_dataset("moons (hard, non-linear boundary)", X_moons, y_moons)

    print("\n=== Feature importances on moons dataset (RandomForest, 50 trees) ===")
    X_train, X_test, y_train, y_test = train_test_split(X_moons, y_moons, test_size=0.3, random_state=1)
    forest = RandomForestClassifier(n_estimators=50, max_depth=6, random_state=0)
    forest.fit(X_train, y_train)
    for i, importance in enumerate(forest.feature_importances_):
        print(f"  feature {i}: {importance:.3f}")


if __name__ == "__main__":
    main()
