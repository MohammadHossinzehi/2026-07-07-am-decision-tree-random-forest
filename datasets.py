"""
datasets.py

Small synthetic dataset generators (no external downloads, fully
reproducible via random_state) used to exercise and benchmark the
decision tree and random forest models.
"""

from __future__ import annotations

import numpy as np


def make_blobs(n_samples: int = 300, centers: int = 3, n_features: int = 2,
                cluster_std: float = 1.0, random_state: int = 0):
    """Gaussian blobs around `centers` random cluster centers. Linearly
    separable-ish; easy mode for sanity-checking a classifier.
    """
    rng = np.random.RandomState(random_state)
    center_points = rng.uniform(-10, 10, size=(centers, n_features))
    samples_per_center = n_samples // centers
    X_parts, y_parts = [], []
    for c in range(centers):
        pts = center_points[c] + rng.normal(scale=cluster_std, size=(samples_per_center, n_features))
        X_parts.append(pts)
        y_parts.append(np.full(samples_per_center, c))
    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    order = rng.permutation(len(X))
    return X[order], y[order]


def make_moons(n_samples: int = 300, noise: float = 0.15, random_state: int = 0):
    """Two interleaving crescent moons. Not linearly separable -- a good
    stress test that shows why a single shallow tree struggles but an
    ensemble (or a deep enough tree) can still carve out the boundary.
    """
    rng = np.random.RandomState(random_state)
    n_per = n_samples // 2

    theta1 = rng.uniform(0, np.pi, n_per)
    x1 = np.cos(theta1)
    y1 = np.sin(theta1)

    theta2 = rng.uniform(0, np.pi, n_per)
    x2 = 1 - np.cos(theta2)
    y2 = 1 - np.sin(theta2) - 0.5

    X = np.vstack([
        np.column_stack([x1, y1]),
        np.column_stack([x2, y2]),
    ])
    y = np.concatenate([np.zeros(n_per, dtype=int), np.ones(n_per, dtype=int)])

    X += rng.normal(scale=noise, size=X.shape)

    order = rng.permutation(len(X))
    return X[order], y[order]


def train_test_split(X: np.ndarray, y: np.ndarray, test_size: float = 0.25, random_state: int = 0):
    rng = np.random.RandomState(random_state)
    n = len(X)
    idx = rng.permutation(n)
    n_test = int(n * test_size)
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
