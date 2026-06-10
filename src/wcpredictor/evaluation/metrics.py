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


# ─── Asian-handicap / totals metrics ─────────────────────────────────────────

def _settle_outcome(true_diff: int, threshold: float) -> float:
    """Settle actual goal difference at a threshold for the positive side.

    Returns fractional stake returned:
      1.0  = full win (key > threshold, ignoring quarter-line payouts)
      0.5  = push or half-win / half-loss
      0.0  = full loss

    This is a coarse settlement for the Brier / calibration scorer (we want the
    probability that the model-estimated cover probability matches realized cover).
    Full settlement including quarter-line payouts is in markets/asian.py.
    """
    diff = true_diff - threshold
    if abs(diff) < 1e-9:
        return 0.5  # push
    return 1.0 if diff > 0 else 0.0


def ah_cover_brier(
    true_a: list[int],
    true_b: list[int],
    model_p_cover: list[float],
    ah_thresholds: list[float],
) -> float:
    """Brier score for the model's AH home-cover probability.

    Parameters
    ----------
    true_a, true_b   : actual goals scored.
    model_p_cover    : model's estimated P(home covers) at the given threshold.
    ah_thresholds    : settlement threshold for each match (threshold = -AH_line).

    Returns mean squared error between model_p_cover and realized cover (0/0.5/1).
    """
    total = 0.0
    for ga, gb, p, thr in zip(true_a, true_b, model_p_cover, ah_thresholds):
        y = _settle_outcome(ga - gb, thr)
        total += (p - y) ** 2
    return total / len(true_a)


def ah_cover_calibration(
    true_a: list[int],
    true_b: list[int],
    model_p_cover: list[float],
    ah_thresholds: list[float],
    n_bins: int = 5,
) -> float:
    """Expected calibration error (ECE) for the AH home-cover probability.

    Bins matches by model_p_cover, compares average predicted probability to
    average realized cover rate within each bin.
    """
    from collections import defaultdict
    bins: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for ga, gb, p, thr in zip(true_a, true_b, model_p_cover, ah_thresholds):
        b = min(int(p * n_bins), n_bins - 1)
        y = _settle_outcome(ga - gb, thr)
        bins[b].append((p, y))

    n = len(true_a)
    ece = 0.0
    for items in bins.values():
        probs = [x[0] for x in items]
        outcomes = [x[1] for x in items]
        ece += (len(items) / n) * abs(
            sum(probs) / len(probs) - sum(outcomes) / len(outcomes)
        )
    return ece


def closing_line_value(
    model_ah_lines: list[float],
    market_ah_lines: list[float],
) -> float:
    """Mean closing-line value (CLV) for the model's AH fair line.

    CLV = average signed difference between the model's fair main AH line and
    the market closing line (both in standard AH notation for the home team).
    Positive CLV means the model is systematically sharper than market consensus.
    """
    diffs = [m - c for m, c in zip(model_ah_lines, market_ah_lines)]
    return float(np.mean(diffs))


def ah_roi(
    true_a: list[int],
    true_b: list[int],
    market_ah_lines: list[float],
    market_ah_home_odds: list[float],
    model_p_cover: list[float],
    edge_threshold: float = 0.0,
) -> float:
    """Flat-stake ROI from betting model edge into the market AH (home side).

    Only bets when model_p_cover > fair_p_cover + edge_threshold, where
    fair_p_cover = 1 / market_ah_home_odds (implied probability).

    Returns ROI as a fraction of total bets placed (positive = profitable).
    Returns NaN if no bets qualify.
    """
    total_stakes = 0.0
    total_returns = 0.0

    for ga, gb, ah_line, home_odds, p_model in zip(
        true_a, true_b, market_ah_lines, market_ah_home_odds, model_p_cover
    ):
        if home_odds <= 1.0:
            continue
        fair_p = 1.0 / home_odds
        if p_model < fair_p + edge_threshold:
            continue  # no edge

        # Bet 1 unit at home_odds
        threshold = -ah_line  # convert AH notation to positive-side threshold
        result = _settle_outcome(ga - gb, threshold)
        total_stakes += 1.0
        total_returns += result * home_odds if result == 1.0 else result  # win: odds; push: 0.5

    if total_stakes == 0:
        return float("nan")
    return (total_returns - total_stakes) / total_stakes
