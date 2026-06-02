"""Pure metric functions for evaluating match prediction quality."""

from __future__ import annotations

import math
import numpy as np

_EPS = 1e-10


def log_loss(y_true: list[int], probs: list[list[float]]) -> float:
    """Multi-class log loss. y_true in {0,1,2} (win/draw/loss)."""
    total = sum(-math.log(max(probs[i][label], _EPS)) for i, label in enumerate(y_true))
    return total / len(y_true)


def brier(y_true: list[int], probs: list[list[float]]) -> float:
    """Multi-class Brier score (mean squared prob error over all classes)."""
    n_classes = len(probs[0])
    total = 0.0
    for label, p in zip(y_true, probs):
        for c in range(n_classes):
            target = 1.0 if c == label else 0.0
            total += (p[c] - target) ** 2
    return total / len(y_true)


def accuracy(y_true: list[int], y_pred: list[int]) -> float:
    correct = sum(a == b for a, b in zip(y_true, y_pred))
    return correct / len(y_true)


def macro_f1(y_true: list[int], y_pred: list[int], n_classes: int = 3) -> float:
    f1s = []
    for c in range(n_classes):
        tp = sum(1 for a, b in zip(y_true, y_pred) if a == c and b == c)
        fp = sum(1 for a, b in zip(y_true, y_pred) if a != c and b == c)
        fn = sum(1 for a, b in zip(y_true, y_pred) if a == c and b != c)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0)
    return float(np.mean(f1s))


def goal_mae(true_a: list[int], pred_a: list[float], true_b: list[int], pred_b: list[float]) -> float:
    mae_a = np.mean(np.abs(np.array(true_a, float) - np.array(pred_a, float)))
    mae_b = np.mean(np.abs(np.array(true_b, float) - np.array(pred_b, float)))
    return float((mae_a + mae_b) / 2)


def goal_rmse(true_a: list[int], pred_a: list[float], true_b: list[int], pred_b: list[float]) -> float:
    rmse_a = np.sqrt(np.mean((np.array(true_a, float) - np.array(pred_a, float)) ** 2))
    rmse_b = np.sqrt(np.mean((np.array(true_b, float) - np.array(pred_b, float)) ** 2))
    return float((rmse_a + rmse_b) / 2)


def exact_score_logscore(
    true_a: list[int],
    true_b: list[int],
    score_matrices: list[list[list[float]]],
) -> float:
    """Mean log score for the exact scoreline."""
    total = 0.0
    for ga, gb, matrix in zip(true_a, true_b, score_matrices):
        from wcpredictor.config import MAX_GOALS
        if ga <= MAX_GOALS and gb <= MAX_GOALS:
            p = max(matrix[ga][gb], _EPS)
        else:
            p = _EPS
        total += math.log(p)
    return total / len(true_a)


def topn_hit_rate(true_a: list[int], true_b: list[int], top_scorelines: list[list[dict]], n: int = 5) -> float:
    """Fraction of matches where the actual score is in the top-N predictions."""
    hits = 0
    for ga, gb, top in zip(true_a, true_b, top_scorelines):
        if any(t["goals_a"] == ga and t["goals_b"] == gb for t in top[:n]):
            hits += 1
    return hits / len(true_a)
