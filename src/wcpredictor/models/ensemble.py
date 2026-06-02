"""Ensemble combiner for W/D/L probability pools.

Supported pool types:
  "log"    — log-opinion pool (weighted geometric mean, renormalized).
             Reduces to the best member at extremes; naturally calibratable.
  "linear" — linear pool (weighted arithmetic mean).

Weights are fit by minimizing validation log loss with L2 regularization
toward equal weights (mitigates noise on thin validation slices).

Score matrices are always blended linearly regardless of pool type,
because the logistic member produces no score matrix.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import minimize

_EPS = 1e-10


def _softmax(w_raw: np.ndarray) -> np.ndarray:
    e = np.exp(w_raw - w_raw.max())
    return e / e.sum()


def _blend(arrs: list[np.ndarray], log_arrs: list[np.ndarray], w: np.ndarray, pool: str) -> np.ndarray:
    if pool == "log":
        log_c = sum(w[m] * log_arrs[m] for m in range(len(arrs)))
        log_c -= log_c.max(axis=1, keepdims=True)
        c = np.exp(log_c)
    else:
        c = sum(w[m] * arrs[m] for m in range(len(arrs)))
    c /= c.sum(axis=1, keepdims=True)
    return c


def fit_weights(
    member_probs_val: list[list[list[float]]],
    labels: list[int],
    pool: str = "log",
    reg_lambda: float = 1.0,
) -> np.ndarray:
    """Fit blending weights by minimizing validation log loss.

    Parameters
    ----------
    member_probs_val : list of n_members probability lists, each (n_matches, 3)
    labels           : true class indices (0=win, 1=draw, 2=loss)
    pool             : "log" (log-opinion) or "linear"
    reg_lambda       : L2 regularization toward equal weights

    Returns
    -------
    weights : np.ndarray shape (n_members,), non-negative, sum=1
    """
    n_members = len(member_probs_val)
    n = len(labels)
    arrs = [np.clip(np.array(p, dtype=float), _EPS, 1.0) for p in member_probs_val]
    log_arrs = [np.log(a) for a in arrs]
    equal_w = np.ones(n_members) / n_members

    def objective(w_raw: np.ndarray) -> float:
        w = _softmax(w_raw)
        c = _blend(arrs, log_arrs, w, pool)
        ll = sum(-math.log(max(float(c[i, labels[i]]), _EPS)) for i in range(n)) / n
        reg = reg_lambda * float(np.sum((w - equal_w) ** 2))
        return ll + reg

    result = minimize(
        objective,
        x0=np.zeros(n_members),
        method="Nelder-Mead",
        options={"maxiter": 10_000, "xatol": 1e-5, "fatol": 1e-6},
    )
    return _softmax(result.x)


def combine_probs(
    member_probs: list[list[list[float]]],
    weights: np.ndarray,
    pool: str = "log",
) -> list[list[float]]:
    """Blend probability vectors; returns list of shape (n_matches, 3)."""
    arrs = [np.clip(np.array(p, dtype=float), _EPS, 1.0) for p in member_probs]
    log_arrs = [np.log(a) for a in arrs]
    return _blend(arrs, log_arrs, weights, pool).tolist()


def combine_matrices(
    matrices: list[list[list[list[float]]]],
    weights: np.ndarray,
) -> list[list[list[float]]]:
    """Linear blend of score matrices and renormalize.

    matrices : [n_members][n_matches][rows][cols]
    weights  : shape (n_members,); caller is responsible for renormalizing
               to cover only the members that have matrices (Poisson + DC).
    """
    n_members = len(matrices)
    n_matches = len(matrices[0])
    result = []
    for i in range(n_matches):
        m = sum(float(weights[k]) * np.array(matrices[k][i], dtype=float) for k in range(n_members))
        m /= m.sum()
        result.append(m.tolist())
    return result


def matrix_to_lambdas(matrix: list[list[float]]) -> tuple[float, float]:
    """Compute expected goals (lambda_a, lambda_b) from a score probability matrix."""
    arr = np.array(matrix, dtype=float)
    rows, cols = arr.shape
    lambda_a = float((arr.sum(axis=1) * np.arange(rows, dtype=float)).sum())
    lambda_b = float((arr.sum(axis=0) * np.arange(cols, dtype=float)).sum())
    return lambda_a, lambda_b


def matrix_to_top_scorelines(matrix: list[list[float]], n: int = 5) -> list[dict]:
    """Extract top-n scorelines from a score probability matrix."""
    arr = np.array(matrix, dtype=float)
    rows, cols = arr.shape
    flat = [(float(arr[i, j]), i, j) for i in range(rows) for j in range(cols)]
    flat.sort(reverse=True)
    return [{"goals_a": i, "goals_b": j, "prob": round(p, 6)} for p, i, j in flat[:n]]
