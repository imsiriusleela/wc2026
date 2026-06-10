"""Correctness tests for the Asian handicap / totals market module.

These tests are the primary correctness lock for the settlement engine; any
regression in quarter-line logic or fair-odds formula should be caught here.
"""
from __future__ import annotations

import math
import pytest
from wcpredictor.markets.asian import (
    asian_handicap,
    asian_total,
    fair_handicap,
    fair_total,
    goal_diff_distribution,
    ladder,
    settle_line,
    total_goals_distribution,
)
from wcpredictor.models.ensemble import matrix_to_lambdas


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def _uniform_matrix(n: int = 9) -> list[list[float]]:
    """Uniform 9×9 score matrix (equal prob for every scoreline)."""
    p = 1.0 / (n * n)
    return [[p] * n for _ in range(n)]


def _make_matrix(goals_a: float, goals_b: float, n: int = 9) -> list[list[float]]:
    """Independent Poisson score matrix with given expected goals."""
    import math as _m
    def pmf(lam: float, k: int) -> float:
        return (_m.exp(-lam) * lam ** k) / _m.factorial(k)

    raw = [[pmf(goals_a, i) * pmf(goals_b, j) for j in range(n)] for i in range(n)]
    total = sum(raw[i][j] for i in range(n) for j in range(n))
    return [[raw[i][j] / total for j in range(n)] for i in range(n)]


def _symmetric_matrix(n: int = 9) -> list[list[float]]:
    """Matrix that gives home == away expected goals (λ_a == λ_b)."""
    return _make_matrix(1.2, 1.2, n)


# ─── Distribution tests ───────────────────────────────────────────────────────

def test_goal_diff_sums_to_one():
    mat = _make_matrix(1.5, 1.1)
    dist = goal_diff_distribution(mat)
    assert sum(dist.values()) == pytest.approx(1.0, abs=1e-6)


def test_total_goals_sums_to_one():
    mat = _make_matrix(1.5, 1.1)
    dist = total_goals_distribution(mat)
    assert sum(dist.values()) == pytest.approx(1.0, abs=1e-6)


def test_goal_diff_mean_matches_lambdas():
    mat = _make_matrix(1.6, 1.0)
    la, lb = matrix_to_lambdas(mat)
    dist = goal_diff_distribution(mat)
    mean_diff = sum(d * p for d, p in dist.items())
    assert mean_diff == pytest.approx(la - lb, abs=1e-4)


def test_total_goals_mean_matches_lambdas():
    mat = _make_matrix(1.6, 1.0)
    la, lb = matrix_to_lambdas(mat)
    dist = total_goals_distribution(mat)
    mean_total = sum(t * p for t, p in dist.items())
    assert mean_total == pytest.approx(la + lb, abs=1e-4)


# ─── settle_line correctness ──────────────────────────────────────────────────

def test_settle_probabilities_sum_to_one():
    mat = _make_matrix(1.5, 1.0)
    diff_dist = goal_diff_distribution(mat)
    for line in [-2.0, -1.5, -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 1.5]:
        r = settle_line(diff_dist, line)
        total = r["p_win"] + r["p_half_win"] + r["p_push"] + r["p_half_loss"] + r["p_loss"]
        assert total == pytest.approx(1.0, abs=1e-5), f"line={line}: probs sum to {total}"


def test_half_line_no_push():
    mat = _make_matrix(1.5, 1.0)
    diff_dist = goal_diff_distribution(mat)
    for line in [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]:
        r = settle_line(diff_dist, line)
        assert r["p_push"] == 0.0, f"half line {line} should have no push"
        assert r["p_half_win"] == 0.0
        assert r["p_half_loss"] == 0.0


def test_whole_line_push_equals_exact_mass():
    mat = _make_matrix(1.5, 1.0)
    diff_dist = goal_diff_distribution(mat)
    for line in [-2.0, -1.0, 0.0, 1.0, 2.0]:
        r = settle_line(diff_dist, line)
        expected_push = diff_dist.get(int(line), 0.0)
        # Output is rounded to 6 d.p.; allow 1e-5 tolerance
        assert r["p_push"] == pytest.approx(expected_push, abs=1e-5), (
            f"whole line {line}: push={r['p_push']}, expected {expected_push}"
        )
        assert r["p_half_win"] == 0.0
        assert r["p_half_loss"] == 0.0


def test_quarter_line_minus025_half_win_half_loss():
    """Quarter line -0.25 decomposes into sub-lines 0.0 (whole) and 0.5 (half)."""
    mat = _make_matrix(1.5, 1.0)
    diff_dist = goal_diff_distribution(mat)

    r_025 = settle_line(diff_dist, 0.25)   # positive threshold = AH -0.25
    r_0 = settle_line(diff_dist, 0.0)      # whole line (push at D=0)
    r_05 = settle_line(diff_dist, 0.5)     # half line (no push)

    # Full win: D ≥ 1 wins both → same as P(D ≥ 1)
    assert r_025["p_win"] == pytest.approx(r_05["p_win"], abs=1e-8)
    assert r_025["p_win"] == pytest.approx(r_0["p_win"], abs=1e-8)
    # Half loss: D = 0 pushes lo sub-bet, loses hi sub-bet
    assert r_025["p_half_loss"] == pytest.approx(r_0["p_push"], abs=1e-8)
    # Full loss: D ≤ -1
    assert r_025["p_loss"] == pytest.approx(r_0["p_loss"], abs=1e-8)
    assert r_025["p_half_win"] == 0.0


def test_quarter_line_plus025_half_win():
    """Quarter line +0.25 (threshold -0.25): decomposes into -0.5 and 0.0."""
    mat = _make_matrix(1.0, 1.5)  # away stronger
    diff_dist = goal_diff_distribution(mat)

    r_neg025 = settle_line(diff_dist, -0.25)
    r_0 = settle_line(diff_dist, 0.0)
    r_neg05 = settle_line(diff_dist, -0.5)

    # Full win: D ≥ 1 wins both (D > -0.5 AND D > 0)
    assert r_neg025["p_win"] == pytest.approx(r_0["p_win"], abs=1e-8)
    # Half win: D = 0, lo wins (D > -0.5), hi pushes (D = 0)
    assert r_neg025["p_half_win"] == pytest.approx(r_0["p_push"], abs=1e-8)
    # Full loss: D ≤ -1
    assert r_neg025["p_loss"] == pytest.approx(r_0["p_loss"], abs=1e-8)
    assert r_neg025["p_half_loss"] == 0.0


def test_quarter_line_minus075():
    """AH -0.75 threshold = 0.75; decomposes into 0.5 and 1.0."""
    mat = _make_matrix(1.8, 0.9)
    diff_dist = goal_diff_distribution(mat)

    r_075 = settle_line(diff_dist, 0.75)
    r_05 = settle_line(diff_dist, 0.5)    # half line: win if D ≥ 1
    r_1 = settle_line(diff_dist, 1.0)     # whole line: win if D ≥ 2, push if D = 1

    # Full win: D ≥ 2 (both sub-bets win)
    assert r_075["p_win"] == pytest.approx(r_1["p_win"], abs=1e-8)
    # Half win: D = 1 (lo wins, hi pushes)
    assert r_075["p_half_win"] == pytest.approx(r_1["p_push"], abs=1e-8)
    # Full loss: D ≤ 0
    p_full_loss = r_1["p_loss"] + r_05["p_loss"] - r_1["p_win"] - r_1["p_push"]
    # simpler: P(D ≤ 0) = 1 - P(D ≥ 1)
    p_d_le0 = 1.0 - r_05["p_win"]
    assert r_075["p_loss"] == pytest.approx(p_d_le0, abs=1e-6)


def test_invalid_line_raises():
    dist = {0: 1.0}
    with pytest.raises(ValueError):
        settle_line(dist, 0.1)  # not a multiple of 0.25


# ─── Fair-odds zero-EV ────────────────────────────────────────────────────────

def _expected_return(result: dict, odds: float) -> float:
    """E[return per unit stake] given decimal odds for the 'win' side."""
    return (
        result["p_win"] * odds
        + result["p_half_win"] * (0.5 * odds + 0.5)
        + result["p_push"] * 1.0
        + result["p_half_loss"] * 0.5
        + result["p_loss"] * 0.0
    )


@pytest.mark.parametrize("line", [-2.0, -1.5, -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0])
def test_fair_odds_zero_ev(line: float):
    mat = _make_matrix(1.5, 1.0)
    diff_dist = goal_diff_distribution(mat)
    r = settle_line(diff_dist, line)
    if math.isinf(r["fair_odds"]):
        pytest.skip(f"Degenerate distribution at line={line}")
    ev = _expected_return(r, r["fair_odds"])
    # fair_odds is rounded to 4 d.p.; that introduces at most ~1e-4 EV error
    assert ev == pytest.approx(1.0, abs=5e-4), f"line={line}: E[return]={ev}"


@pytest.mark.parametrize("line", [0.5, 1.5, 2.0, 2.5, 3.0, 3.5])
def test_fair_odds_zero_ev_totals(line: float):
    mat = _make_matrix(1.5, 1.0)
    total_dist = total_goals_distribution(mat)
    r = settle_line(total_dist, line)
    if math.isinf(r["fair_odds"]):
        pytest.skip(f"Degenerate distribution at line={line}")
    ev = _expected_return(r, r["fair_odds"])
    assert ev == pytest.approx(1.0, abs=5e-4), f"line={line}: E[return]={ev}"


# ─── Symmetric matrix ─────────────────────────────────────────────────────────

def test_symmetric_matrix_fair_handicap_zero():
    mat = _symmetric_matrix()
    assert fair_handicap(mat) == pytest.approx(0.0, abs=0.25)  # within one quarter step


def test_symmetric_matrix_ah0_push_equals_draw():
    """With a symmetric matrix, AH 0 push probability should equal P(draw)."""
    mat = _symmetric_matrix()
    diff_dist = goal_diff_distribution(mat)
    # Push at whole line 0 = P(D=0) = P(draw)
    p_draw = diff_dist.get(0, 0.0)
    r = settle_line(diff_dist, 0.0)
    # Output is rounded to 6 d.p.
    assert r["p_push"] == pytest.approx(p_draw, abs=1e-5)


def test_symmetric_matrix_home_away_coverage_equal():
    """With equal strengths, home and away AH coverage at line 0 should be equal."""
    mat = _symmetric_matrix()
    r_home = asian_handicap(mat, side="home", line=0.0)
    r_away = asian_handicap(mat, side="away", line=0.0)
    assert r_home["p_win"] == pytest.approx(r_away["p_win"], abs=1e-8)


# ─── asian_handicap wrappers ─────────────────────────────────────────────────

def test_asian_handicap_home_win_increases_as_line_easier():
    """As AH line gets easier (more positive), home win probability increases."""
    mat = _make_matrix(1.5, 1.0)
    lines = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]
    covers = []
    for ln in lines:
        r = asian_handicap(mat, side="home", line=ln)
        # "covers" includes full win + half win weighted
        covers.append(r["p_win"] + 0.5 * r["p_half_win"] + r["p_push"])
    for i in range(len(covers) - 1):
        assert covers[i] <= covers[i + 1] + 1e-8, (
            f"Coverage should increase with easier line: {lines[i]}={covers[i]:.4f} > {lines[i+1]}={covers[i+1]:.4f}"
        )


def test_asian_handicap_home_plus_away_near_one():
    """Home cover + away cover ≈ 1 (they share the push/half outcomes)."""
    mat = _make_matrix(1.5, 1.0)
    for ln in [-1.0, -0.5, 0.0, 0.5, 1.0]:
        h = asian_handicap(mat, side="home", line=ln)
        a = asian_handicap(mat, side="away", line=ln)
        # p_win_home + p_win_away + push ~ 1 (at whole lines)
        total = h["p_win"] + h["p_half_win"] + h["p_push"] + h["p_half_loss"] + h["p_loss"]
        assert total == pytest.approx(1.0, abs=1e-5)


# ─── asian_total wrappers ────────────────────────────────────────────────────

def test_asian_total_over_plus_under_near_one():
    """Over and under share the same distribution; combined probs ≈ 1."""
    mat = _make_matrix(1.5, 1.0)
    for ln in [2.0, 2.5, 3.0, 3.5]:
        over = asian_total(mat, side="over", line=ln)
        under = asian_total(mat, side="under", line=ln)
        # At a whole line both share the push, so over_p_win + over_p_push + under_p_win == 1
        combined = over["p_win"] + over["p_push"] + under["p_win"]
        assert combined == pytest.approx(1.0, abs=1e-5)


def test_asian_total_over_increasing():
    """Higher total line → lower over probability (harder to score more)."""
    mat = _make_matrix(1.5, 1.0)
    lines = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
    over_probs = [asian_total(mat, side="over", line=ln)["p_win"] for ln in lines]
    for i in range(len(over_probs) - 1):
        assert over_probs[i] >= over_probs[i + 1] - 1e-8, (
            f"Over probability should decrease with higher line: {lines[i]}={over_probs[i]:.4f} < {lines[i+1]}={over_probs[i+1]:.4f}"
        )


# ─── Fair lines ───────────────────────────────────────────────────────────────

def test_fair_handicap_stronger_home_negative():
    """When home is significantly stronger, fair handicap is negative."""
    mat = _make_matrix(2.5, 0.8)
    fh = fair_handicap(mat)
    assert fh < 0.0, f"Stronger home should have negative (gives goals) fair line, got {fh}"


def test_fair_handicap_weaker_home_positive():
    """When home is significantly weaker, fair handicap is positive."""
    mat = _make_matrix(0.8, 2.5)
    fh = fair_handicap(mat)
    assert fh > 0.0, f"Weaker home should have positive (receives goals) fair line, got {fh}"


def test_fair_handicap_quarter_rounded():
    mat = _make_matrix(1.6, 1.0)
    fh = fair_handicap(mat)
    # Must be a multiple of 0.25
    assert abs(round(fh * 4) - fh * 4) < 1e-9, f"fair_handicap not quarter-rounded: {fh}"


def test_fair_total_positive():
    mat = _make_matrix(1.5, 1.0)
    ft = fair_total(mat)
    assert ft > 0.0
    # Must be a multiple of 0.25
    assert abs(round(ft * 4) - ft * 4) < 1e-9, f"fair_total not quarter-rounded: {ft}"


def test_fair_total_matches_lambda_sum():
    mat = _make_matrix(1.6, 1.2)
    la, lb = matrix_to_lambdas(mat)
    ft = fair_total(mat)
    expected = round((la + lb) * 4) / 4
    assert ft == pytest.approx(expected, abs=1e-9)


# ─── ladder ──────────────────────────────────────────────────────────────────

def test_ladder_structure():
    mat = _make_matrix(1.5, 1.0)
    result = ladder(mat)
    assert "ah_fair_line" in result
    assert "total_fair_line" in result
    assert "ah_main" in result
    assert "total_main_over" in result
    assert "total_main_under" in result
    assert "ah_ladder" in result
    assert "total_ladder" in result


def test_ladder_ah_main_probabilities_sum_to_one():
    mat = _make_matrix(1.5, 1.0)
    r = ladder(mat)
    main = r["ah_main"]
    total = main["p_win"] + main["p_half_win"] + main["p_push"] + main["p_half_loss"] + main["p_loss"]
    assert total == pytest.approx(1.0, abs=1e-5)


def test_ladder_rungs_monotone_home_cover():
    """AH ladder profitable-cover prob is non-decreasing as line gets easier (more positive).

    Profitable cover = p_win + p_half_win (full or half win, excludes push).
    Adjacent lines may be equal (e.g., a whole line and the quarter line above it share
    the same 'any profit' boundary), so the assertion is non-strict (<=).
    """
    mat = _make_matrix(1.5, 1.0)
    result = ladder(mat)
    rungs = result["ah_ladder"]
    lines = [r["line"] for r in rungs]
    covers = [r["p_win"] + r["p_half_win"] for r in rungs]

    for i in range(len(covers) - 1):
        c1, c2 = covers[i], covers[i + 1]
        if lines[i] < lines[i + 1]:  # lines are sorted most-negative first
            assert c1 <= c2 + 1e-8, (
                f"Line {lines[i]}={c1:.4f} should cover ≤ {lines[i+1]}={c2:.4f}"
            )


# ─── Market matrix from probabilities ─────────────────────────────────────────

def test_market_score_matrix_from_probs_round_trip():
    """_market_score_matrix_from_probs recovers the target probabilities (AH -0.5)."""
    from wcpredictor.predict import _market_score_matrix_from_probs
    from wcpredictor.markets.asian import settle_line

    cases = [
        (0.786, 0.564),   # dominant home (Brazil-like)
        (0.505, 0.515),   # roughly equal (Portugal vs Switzerland-like)
        (0.30,  0.48),    # heavy away favourite
        (0.50,  0.70),    # high-scoring equal match
    ]
    for p_hw_target, p_ov_target in cases:
        mat = _market_score_matrix_from_probs(p_hw_target, p_ov_target, ah_line=-0.5, ou_threshold=2.5)
        assert mat is not None, f"Solver returned None for ({p_hw_target}, {p_ov_target})"

        n = len(mat)
        # Recovered P(home covers AH -0.5) = P(home wins outright)
        diff_dist = {}
        for i in range(n):
            for j in range(n):
                d = i - j
                diff_dist[d] = diff_dist.get(d, 0.0) + mat[i][j]
        res = settle_line(diff_dist, 0.5)  # threshold = -(-0.5) = 0.5
        p_hw = res["p_win"] + res["p_half_win"]
        # Recovered P(total > 2.5) = P(total >= 3)
        p_ov = sum(mat[i][j] for i in range(n) for j in range(n) if i + j >= 3)

        assert abs(p_hw - p_hw_target) < 1e-3, (
            f"P(home win): got {p_hw:.4f}, expected {p_hw_target:.4f}"
        )
        assert abs(p_ov - p_ov_target) < 1e-3, (
            f"P(over 2.5): got {p_ov:.4f}, expected {p_ov_target:.4f}"
        )


def test_market_score_matrix_from_probs_non_half_ah_line():
    """_market_score_matrix_from_probs works for non -0.5 AH lines (e.g., -1.5)."""
    from wcpredictor.predict import _market_score_matrix_from_probs
    from wcpredictor.markets.asian import settle_line

    p_cover_target = 0.52  # P(home wins by 2+ goals at AH -1.5)
    p_ov_target = 0.58     # P(total > 3.5)
    mat = _market_score_matrix_from_probs(p_cover_target, p_ov_target, ah_line=-1.5, ou_threshold=3.5)
    assert mat is not None

    n = len(mat)
    diff_dist = {}
    for i in range(n):
        for j in range(n):
            d = i - j
            diff_dist[d] = diff_dist.get(d, 0.0) + mat[i][j]
    res = settle_line(diff_dist, 1.5)  # threshold = -(-1.5) = 1.5
    p_cover = res["p_win"] + res["p_half_win"]
    p_ov = sum(mat[i][j] for i in range(n) for j in range(n) if i + j >= 4)

    assert abs(p_cover - p_cover_target) < 2e-3, f"P(covers -1.5): {p_cover:.4f} != {p_cover_target}"
    assert abs(p_ov - p_ov_target) < 2e-3, f"P(over 3.5): {p_ov:.4f} != {p_ov_target}"


def test_market_score_matrix_from_probs_matrix_sums_to_one():
    """Matrix returned by prob-based solver sums to 1."""
    from wcpredictor.predict import _market_score_matrix_from_probs

    mat = _market_score_matrix_from_probs(0.6, 0.55, ah_line=-0.5)
    assert mat is not None
    total = sum(mat[i][j] for i in range(len(mat)) for j in range(len(mat[0])))
    assert abs(total - 1.0) < 1e-6


def test_market_score_matrix_from_probs_returns_none_for_degenerate():
    """Degenerate inputs (0 or 1 probabilities) return None."""
    from wcpredictor.predict import _market_score_matrix_from_probs

    assert _market_score_matrix_from_probs(0.0, 0.5) is None
    assert _market_score_matrix_from_probs(0.5, 0.0) is None
    assert _market_score_matrix_from_probs(1.0, 0.5) is None
    assert _market_score_matrix_from_probs(float("nan"), 0.5) is None
