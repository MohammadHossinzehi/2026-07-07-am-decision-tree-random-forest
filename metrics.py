"""
metrics.py

Small, dependency-free (beyond numpy) classification metrics used to
evaluate the decision tree and random forest models. Supports both binary
and multiclass labels via macro-averaging.
"""

from __future__ import annotations

import numpy as np


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = None) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if n_classes is None:
        n_classes = int(max(y_true.max(), y_pred.max())) + 1
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        matrix[t, p] += 1
    return matrix


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) == 0:
        return 0.0
    return float(np.mean(y_true == y_pred))


def _per_class_precision_recall_f1(cm: np.ndarray):
    n_classes = cm.shape[0]
    precisions = np.zeros(n_classes)
    recalls = np.zeros(n_classes)
    f1s = np.zeros(n_classes)
    for c in range(n_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        precisions[c] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recalls[c] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1s[c] = (
            2 * precisions[c] * recalls[c] / (precisions[c] + recalls[c])
            if (precisions[c] + recalls[c]) > 0
            else 0.0
        )
    return precisions, recalls, f1s


def precision_score(y_true: np.ndarray, y_pred: np.ndarray, average: str = "macro") -> float:
    cm = confusion_matrix(y_true, y_pred)
    precisions, _, _ = _per_class_precision_recall_f1(cm)
    return float(np.mean(precisions)) if average == "macro" else precisions


def recall_score(y_true: np.ndarray, y_pred: np.ndarray, average: str = "macro") -> float:
    cm = confusion_matrix(y_true, y_pred)
    _, recalls, _ = _per_class_precision_recall_f1(cm)
    return float(np.mean(recalls)) if average == "macro" else recalls


def f1_score(y_true: np.ndarray, y_pred: np.ndarray, average: str = "macro") -> float:
    cm = confusion_matrix(y_true, y_pred)
    _, _, f1s = _per_class_precision_recall_f1(cm)
    return float(np.mean(f1s)) if average == "macro" else f1s


def classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    cm = confusion_matrix(y_true, y_pred)
    precisions, recalls, f1s = _per_class_precision_recall_f1(cm)
    lines = [f"{'class':>8} {'precision':>10} {'recall':>10} {'f1':>10}"]
    for c in range(cm.shape[0]):
        lines.append(f"{c:>8} {precisions[c]:>10.3f} {recalls[c]:>10.3f} {f1s[c]:>10.3f}")
    lines.append(f"{'macro':>8} {precisions.mean():>10.3f} {recalls.mean():>10.3f} {f1s.mean():>10.3f}")
    lines.append(f"accuracy: {accuracy_score(y_true, y_pred):.3f}")
    return "\n".join(lines)
