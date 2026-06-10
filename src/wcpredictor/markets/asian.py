"""Asian handicap and Asian totals markets derived from the score probability matrix.

All derived quantities follow from the 9×9 goal-count joint probability matrix,
M[i][j] = P(home=i goals, away=j goals).  No additional data is required — this
module is a pure post-processing layer over the existing score matrix output.

Line conventions
----------------
Asian handicap (AH):
  Expressed in standard AH notation relative to the home team.
    Negative → home gives goals (favourite).  e.g., -1.5: home must win by ≥ 2.
    Positive → home receives goals (underdog). e.g., +1.5: home can lose by 1.
  Quarter lines (-0.75, -0.25, +0.25, +0.75, …) produce half-win / half-loss
  outcomes via equal-stake decomposition to the two adjacent lines.

Asian totals (O/U):
  Expressed as the total-goals threshold.
    e.g., 2.5 → over means T ≥ 3, under means T ≤ 2.
  Quarter total lines (e.g., 2.25) are settled identically to quarter AH lines.

Settlement payouts (per 1 unit staked)
---------------------------------------
  Full win      → return = odds
  Half win      → return = 0.5 × odds + 0.5   (one half wins, one half pushed)
  Push          → return = 1                   (stake refunded)
  Half loss     → return = 0.5                 (one half pushed, one half lost)
  Full loss     → return = 0

Fair decimal odds
-----------------
  The price at which E[return per unit stake] = 1 (zero bookmaker margin).
  Derived from the settlement probabilities:

      fair_odds = (1 − 0.5·p_half_win − p_push − 0.5·p_half_loss)
                  / (p_win + 0.5·p_half_win)
"""
from __future__ import annotations

import math

from wcpredictor.models.ensemble import matrix_to_lambdas

_EPS = 1e-12


# ─── Distributions ─────────────────────────────────────────────────────────────

def goal_diff_distribution(matrix: list[list[float]]) -> dict[int, float]:
    """P(D=d) where D = home_goals − away_goals.

    Positive d means home is ahead; negative d means away is ahead.
    """
    dist: dict[int, float] = {}
    for i, row in enumerate(matrix):
        for j, prob in enumerate(row):
            d = i - j
            dist[d] = dist.get(d, 0.0) + prob
    return dist


def total_goals_distribution(matrix: list[list[float]]) -> dict[int, float]:
    """P(T=t) where T = home_goals + away_goals."""
    dist: dict[int, float] = {}
    for i, row in enumerate(matrix):
        for j, prob in enumerate(row):
            t = i + j
            dist[t] = dist.get(t, 0.0) + prob
    return dist


# ─── Settlement engine ─────────────────────────────────────────────────────────

def _fair_odds_from_settlement(
    p_win: float,
    p_half_win: float,
    p_push: float,
    p_half_loss: float,
    p_loss: float,  # noqa: ARG001
) -> float:
    """Decimal fair odds for the positive side at zero EV.

    Solves E[return] = 1 for *odds*:
        odds·(p_win + 0.5·p_half_win) + 0.5·p_half_win + p_push + 0.5·p_half_loss = 1
    """
    denom = p_win + 0.5 * p_half_win
    if denom < _EPS:
        return float("inf")
    numer = 1.0 - 0.5 * p_half_win - p_push - 0.5 * p_half_loss
    if numer <= 0.0:
        return float("inf")
    return round(numer / denom, 4)


def settle_line(
    dist: dict[int, float],
    line: float,
) -> dict[str, float]:
    """Settle a two-way market for the *positive* side (win when key > *line*).

    Works with any integer-keyed probability distribution (goal difference,
    total goals, or any discrete distribution).

    Handles whole lines (push possible), half lines (no push), and quarter lines
    (half-win / half-loss outcomes via equal-stake decomposition).

    Parameters
    ----------
    dist : integer-keyed probability distribution summing to ≈1.
    line : settlement threshold for the positive side; win iff key > line (strict).

    Returns
    -------
    dict with keys: p_win, p_half_win, p_push, p_half_loss, p_loss, fair_odds
    """
    line_x4_int = round(line * 4)
    if abs(line_x4_int - line * 4) > 1e-6:
        raise ValueError(f"line must be a multiple of 0.25; got {line!r}")

    is_quarter = (line_x4_int % 2) != 0

    if not is_quarter:
        p_win = p_push = p_loss = 0.0
        for key, prob in dist.items():
            diff = key - line
            if diff > 1e-9:
                p_win += prob
            elif abs(diff) < 1e-9:
                p_push += prob
            else:
                p_loss += prob
        fo = _fair_odds_from_settlement(p_win, 0.0, p_push, 0.0, p_loss)
        return {
            "p_win": round(p_win, 6),
            "p_half_win": 0.0,
            "p_push": round(p_push, 6),
            "p_half_loss": 0.0,
            "p_loss": round(p_loss, 6),
            "fair_odds": fo,
        }

    # Quarter line: split into equal half-stakes on the two adjacent 0.5-multiple lines
    lo = math.floor(line * 2) / 2.0
    hi = lo + 0.5

    full_win = half_win = push = half_loss = full_loss = 0.0

    for key, prob in dist.items():
        diff_lo = key - lo
        if diff_lo > 1e-9:
            r_lo = "win"
        elif abs(diff_lo) < 1e-9:
            r_lo = "push"
        else:
            r_lo = "loss"

        diff_hi = key - hi
        if diff_hi > 1e-9:
            r_hi = "win"
        elif abs(diff_hi) < 1e-9:
            r_hi = "push"
        else:
            r_hi = "loss"

        if r_lo == "win" and r_hi == "win":
            full_win += prob
        elif (r_lo == "win" and r_hi == "push") or (r_lo == "push" and r_hi == "win"):
            half_win += prob
        elif (r_lo == "win" and r_hi == "loss") or (r_lo == "loss" and r_hi == "win"):
            push += prob  # one half wins, one half loses → net stake refunded
        elif r_lo == "push" and r_hi == "push":
            push += prob
        elif (r_lo == "push" and r_hi == "loss") or (r_lo == "loss" and r_hi == "push"):
            half_loss += prob
        else:
            full_loss += prob

    fo = _fair_odds_from_settlement(full_win, half_win, push, half_loss, full_loss)
    return {
        "p_win": round(full_win, 6),
        "p_half_win": round(half_win, 6),
        "p_push": round(push, 6),
        "p_half_loss": round(half_loss, 6),
        "p_loss": round(full_loss, 6),
        "fair_odds": fo,
    }


# ─── Market wrappers ───────────────────────────────────────────────────────────

def asian_handicap(
    matrix: list[list[float]],
    side: str = "home",
    line: float = 0.0,
) -> dict[str, float]:
    """Asian handicap settlement for *side* at *line* (standard AH notation).

    *line* is in standard AH notation relative to the home team:
      Negative → home gives goals (e.g., -1.5: home must win by ≥ 2).
      Positive → home receives goals (e.g., +1.5: home can lose by 1).

    For side='home': win iff D > −line  where D = home_goals − away_goals.
    For side='away': win iff D < −line  (i.e., away scores enough to cover).
    """
    diff_dist = goal_diff_distribution(matrix)
    threshold = -line  # convert AH notation to settle_line threshold

    if side == "home":
        return settle_line(diff_dist, threshold)

    # Away side: win when home goal-diff < threshold → flip the distribution
    neg_dist = {-k: v for k, v in diff_dist.items()}
    return settle_line(neg_dist, -threshold)


def asian_total(
    matrix: list[list[float]],
    side: str = "over",
    line: float = 2.5,
) -> dict[str, float]:
    """Asian totals settlement for 'over' or 'under' at *line*.

    For side='over':  win iff T > line  (T = home_goals + away_goals).
    For side='under': win iff T < line.
    """
    total_dist = total_goals_distribution(matrix)

    if side == "over":
        return settle_line(total_dist, line)

    # Under: win when T < line → flip distribution (key → -key), threshold → -line
    neg_dist = {-k: v for k, v in total_dist.items()}
    return settle_line(neg_dist, -line)


# ─── Fair lines ────────────────────────────────────────────────────────────────

def _quarter_round(x: float) -> float:
    """Round to the nearest 0.25."""
    return round(x * 4) / 4


def fair_handicap(matrix: list[list[float]]) -> float:
    """Fair AH line for the home team in standard AH notation, quarter-rounded.

    Returns −(λ_a − λ_b).  Negative when home is the stronger team (gives goals).
    This is the handicap closest to a 50 % home-cover probability.
    """
    la, lb = matrix_to_lambdas(matrix)
    return _quarter_round(-(la - lb))


def fair_total(matrix: list[list[float]]) -> float:
    """Fair total-goals line, quarter-rounded.

    Returns λ_a + λ_b (expected total goals), quarter-rounded.
    """
    la, lb = matrix_to_lambdas(matrix)
    return _quarter_round(la + lb)


# ─── Ladder ────────────────────────────────────────────────────────────────────

def ladder(matrix: list[list[float]]) -> dict:
    """Full Asian-markets report for a single match.

    Returns fair odds and cover probabilities across the configured AH and
    totals line sets, plus main-line summaries at the two fair lines.

    Keys in the returned dict
    -------------------------
    ah_fair_line       : home AH fair line in standard notation (negative = gives goals).
    total_fair_line    : fair total-goals line.
    ah_main            : full settle_line result for home at the fair AH line.
    total_main_over    : full settle_line result for over at the fair total line.
    total_main_under   : full settle_line result for under at the fair total line.
    ah_ladder          : list of {line, p_win, p_half_win, p_push, p_half_loss,
                                  p_loss, fair_odds} for each configured AH line.
    total_ladder       : list of {line, over: {…}, under: {…}} for each configured
                                  totals line.
    """
    from wcpredictor.config import ASIAN_HANDICAP_LINES, ASIAN_TOTAL_LINES

    ah_line = fair_handicap(matrix)
    tot_line = fair_total(matrix)

    ah_main = asian_handicap(matrix, side="home", line=ah_line)
    tot_over = asian_total(matrix, side="over", line=tot_line)
    tot_under = asian_total(matrix, side="under", line=tot_line)

    ah_rungs: list[dict] = []
    for ln in ASIAN_HANDICAP_LINES:
        s = asian_handicap(matrix, side="home", line=ln)
        ah_rungs.append({"line": ln, **s})

    tot_rungs: list[dict] = []
    for ln in ASIAN_TOTAL_LINES:
        over = asian_total(matrix, side="over", line=ln)
        under = asian_total(matrix, side="under", line=ln)
        tot_rungs.append({"line": ln, "over": over, "under": under})

    return {
        "ah_fair_line": ah_line,
        "total_fair_line": tot_line,
        "ah_main": ah_main,
        "total_main_over": tot_over,
        "total_main_under": tot_under,
        "ah_ladder": ah_rungs,
        "total_ladder": tot_rungs,
    }
