"""Independent Poisson goal model.

Model:
    λ_a = base * exp(+beta * d / 2)
    λ_b = base * exp(-beta * d / 2)
where d = elo_diff_adj (positive means team_a is favoured).

Fitting minimises W/D/L log loss over (base, beta) using scipy.optimize.minimize.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson

from wcpredictor.config import MAX_GOALS

_EPS = 1e-10


def _lambdas(elo_diff_adj: float, base: float, beta: float) -> tuple[float, float]:
    half = beta * elo_diff_adj / 2.0
    return base * np.exp(half), base * np.exp(-half)


def _wdl_probs(la: float, lb: float) -> tuple[float, float, float]:
    """Win/draw/loss probabilities from Poisson parameters."""
    mg = MAX_GOALS
    probs = np.outer(
        [poisson.pmf(i, la) for i in range(mg + 1)],
        [poisson.pmf(j, lb) for j in range(mg + 1)],
    )
    probs /= probs.sum()
    p_win = float(np.tril(probs, -1).sum())
    p_draw = float(np.trace(probs))
    p_loss = float(np.triu(probs, 1).sum())
    return p_win, p_draw, p_loss


def _wdl_probs_vectorized(la: float, lb: float) -> tuple[float, float, float]:
    """Vectorised W/D/L for a single (la, lb) pair using numpy."""
    mg = MAX_GOALS
    pa = np.array([poisson.pmf(i, la) for i in range(mg + 1)])
    pb = np.array([poisson.pmf(j, lb) for j in range(mg + 1)])
    matrix = np.outer(pa, pb)
    matrix /= matrix.sum()
    p_win = float(np.tril(matrix, -1).sum())
    p_draw = float(np.trace(matrix))
    p_loss = 1.0 - p_win - p_draw
    return p_win, p_draw, p_loss


def fit(features: "pd.DataFrame", n_bins: int = 200) -> tuple[float, float]:
    """Fit (base, beta) to minimise W/D/L log loss on the supplied feature rows.

    features must have columns: elo_diff_adj, goals_a, goals_b.
    Returns (base, beta).

    Elo diff values are binned into n_bins to keep optimizer calls fast.
    """
    diffs = features["elo_diff_adj"].to_numpy(float)
    ga = features["goals_a"].to_numpy(int)
    gb = features["goals_b"].to_numpy(int)

    labels = np.where(ga > gb, 0, np.where(ga == gb, 1, 2))  # win=0, draw=1, loss=2

    # Bin elo_diff_adj for speed: aggregate counts per (bin, label)
    d_min, d_max = diffs.min(), diffs.max()
    bins = np.linspace(d_min - 1, d_max + 1, n_bins + 1)
    bin_idx = np.digitize(diffs, bins) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    # For each bin: count win/draw/loss outcomes
    win_counts = np.zeros(n_bins)
    draw_counts = np.zeros(n_bins)
    loss_counts = np.zeros(n_bins)
    for i, lbl in zip(bin_idx, labels):
        if lbl == 0:
            win_counts[i] += 1
        elif lbl == 1:
            draw_counts[i] += 1
        else:
            loss_counts[i] += 1

    total_counts = win_counts + draw_counts + loss_counts
    active = total_counts > 0
    active_centers = bin_centers[active]
    active_win = win_counts[active]
    active_draw = draw_counts[active]
    active_loss = loss_counts[active]
    active_total = total_counts[active]
    n = len(diffs)

    def neg_log_loss(params: np.ndarray) -> float:
        base, beta = float(params[0]), float(params[1])
        if base <= 0:
            return 1e9
        total_loss = 0.0
        for d, wc, dc, lc, tc in zip(
            active_centers, active_win, active_draw, active_loss, active_total
        ):
            la, lb = _lambdas(d, base, beta)
            pw, pd_, pl = _wdl_probs_vectorized(la, lb)
            total_loss -= wc * np.log(max(pw, _EPS))
            total_loss -= dc * np.log(max(pd_, _EPS))
            total_loss -= lc * np.log(max(pl, _EPS))
        return total_loss / n

    result = minimize(
        neg_log_loss,
        x0=[1.3, 0.003],
        method="Nelder-Mead",
        options={"maxiter": 5000, "xatol": 1e-5, "fatol": 1e-5},
    )
    base, beta = result.x
    return float(base), float(beta)


def predict_one(elo_diff_adj: float, base: float, beta: float) -> dict:
    """Return W/D/L probs, λ values, score matrix, and top-5 scorelines.

    Returns a dict with keys:
        p_win, p_draw, p_loss,
        lambda_a, lambda_b,
        score_matrix (list[list[float]], shape (MAX_GOALS+1)^2, rows=goals_a, cols=goals_b),
        top_scorelines (list of {goals_a, goals_b, prob})
    """
    mg = MAX_GOALS
    la, lb = _lambdas(elo_diff_adj, base, beta)

    matrix = np.outer(
        [poisson.pmf(i, la) for i in range(mg + 1)],
        [poisson.pmf(j, lb) for j in range(mg + 1)],
    )
    matrix /= matrix.sum()

    p_win = float(np.tril(matrix, -1).sum())
    p_draw = float(np.trace(matrix))
    p_loss = float(np.triu(matrix, 1).sum())

    # Top-5 scorelines
    flat = [(float(matrix[i, j]), i, j) for i in range(mg + 1) for j in range(mg + 1)]
    flat.sort(reverse=True)
    top = [{"goals_a": i, "goals_b": j, "prob": round(p, 6)} for p, i, j in flat[:5]]

    return {
        "p_win": round(p_win, 6),
        "p_draw": round(p_draw, 6),
        "p_loss": round(p_loss, 6),
        "lambda_a": round(la, 4),
        "lambda_b": round(lb, 4),
        "score_matrix": matrix.tolist(),
        "top_scorelines": top,
    }
