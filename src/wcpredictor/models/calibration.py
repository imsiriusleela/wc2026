"""Probability calibration utilities.

v1: Temperature scaling — fits a single scalar T that minimizes log loss on a
time-aware validation set, then applies it to all probability outputs.

A T < 1 sharpens probabilities; T > 1 softens them.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import minimize_scalar

_EPS = 1e-10


def _softmax_temp(probs: np.ndarray, T: float) -> np.ndarray:
    log_p = np.log(np.clip(probs, _EPS, 1.0)) / T
    log_p -= log_p.max(axis=-1, keepdims=True)
    exp_p = np.exp(log_p)
    return exp_p / exp_p.sum(axis=-1, keepdims=True)


def _log_loss(labels: list[int], arr: np.ndarray) -> float:
    return sum(-math.log(max(float(arr[i, lbl]), _EPS)) for i, lbl in enumerate(labels)) / len(labels)


def fit_temperature(labels: list[int], probs: list[list[float]]) -> float:
    """Fit temperature T minimizing log loss on validation data.

    Parameters
    ----------
    labels : true class indices (0=win, 1=draw, 2=loss)
    probs  : predicted probability vectors, shape (n, 3)

    Returns
    -------
    T > 0 (scalar float)
    """
    arr = np.array(probs, dtype=float)

    def objective(log_T: float) -> float:
        return _log_loss(labels, _softmax_temp(arr, math.exp(log_T)))

    result = minimize_scalar(objective, bounds=(-2.0, 2.0), method="bounded")
    return float(math.exp(result.x))


def apply(probs: list[list[float]], T: float) -> list[list[float]]:
    """Apply temperature scaling and return calibrated probability vectors."""
    arr = np.array(probs, dtype=float)
    return _softmax_temp(arr, T).tolist()


def expected_calibration_error(
    labels: list[int],
    probs: list[list[float]],
    n_bins: int = 10,
) -> float:
    """Multi-class ECE using max-confidence binning."""
    arr = np.array(probs, dtype=float)
    max_conf = arr.max(axis=1)
    preds = arr.argmax(axis=1)
    correct = np.array([int(p == t) for p, t in zip(preds, labels)], dtype=float)

    ece = 0.0
    n = len(labels)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (max_conf >= lo) & (max_conf < hi)
        if not mask.any():
            continue
        ece += (mask.sum() / n) * abs(correct[mask].mean() - max_conf[mask].mean())
    return float(ece)
