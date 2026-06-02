"""Tests for ensemble combiner."""

from __future__ import annotations

import math

import numpy as np
import pytest

from wcpredictor.models.ensemble import (
    combine_matrices,
    combine_probs,
    fit_weights,
    matrix_to_lambdas,
    matrix_to_top_scorelines,
)


def _log_loss(labels: list[int], probs: list[list[float]]) -> float:
    return sum(-math.log(max(probs[i][l], 1e-10)) for i, l in enumerate(labels)) / len(labels)


def _make_probs(n: int, seed: int, alpha: list[float] | None = None) -> list[list[float]]:
    rng = np.random.default_rng(seed)
    a = alpha or [2.0, 1.0, 1.0]
    return rng.dirichlet(a, n).tolist()


def _make_labels(n: int, seed: int) -> list[int]:
    return np.random.default_rng(seed).integers(0, 3, n).tolist()


# ── weights shape ─────────────────────────────────────────────────────────────

def test_weights_nonneg_and_sum_to_one():
    labels = _make_labels(100, 0)
    p1 = _make_probs(100, 1)
    p2 = _make_probs(100, 2)
    w = fit_weights([p1, p2], labels)
    assert w.shape == (2,)
    assert all(wi >= -1e-9 for wi in w)
    assert abs(w.sum() - 1.0) < 1e-6


def test_weights_three_members():
    labels = _make_labels(80, 7)
    members = [_make_probs(80, i) for i in range(3)]
    w = fit_weights(members, labels)
    assert w.shape == (3,)
    assert abs(w.sum() - 1.0) < 1e-6


def test_identical_members_near_equal_weights():
    """Identical members → regularization pushes weights toward equal."""
    labels = _make_labels(60, 42)
    p = _make_probs(60, 42)
    w = fit_weights([p, p], labels, reg_lambda=10.0)
    assert abs(w[0] - 0.5) < 0.05
    assert abs(w[1] - 0.5) < 0.05


# ── log loss property ─────────────────────────────────────────────────────────

def test_ensemble_log_loss_le_worst_member_no_reg():
    """Without regularization, the ensemble can put all weight on the best member."""
    rng = np.random.default_rng(99)
    labels = rng.integers(0, 3, 120).tolist()
    # One member is much better (more informed)
    p_good = rng.dirichlet([3.0, 1.0, 1.0], 120).tolist()
    p_bad = [[1 / 3, 1 / 3, 1 / 3]] * 120

    w = fit_weights([p_good, p_bad], labels, reg_lambda=0.0)
    ens = combine_probs([p_good, p_bad], w, pool="log")
    ll_ens = _log_loss(labels, ens)
    ll_worst = max(_log_loss(labels, p_good), _log_loss(labels, p_bad))
    assert ll_ens <= ll_worst + 1e-4


# ── combine_probs ─────────────────────────────────────────────────────────────

def test_combine_probs_sum_to_one_log():
    labels = _make_labels(50, 5)
    p1, p2 = _make_probs(50, 6), _make_probs(50, 7)
    w = fit_weights([p1, p2], labels, pool="log")
    combined = combine_probs([p1, p2], w, pool="log")
    for row in combined:
        assert abs(sum(row) - 1.0) < 1e-6
        assert all(v >= 0 for v in row)


def test_combine_probs_sum_to_one_linear():
    labels = _make_labels(50, 5)
    p1, p2 = _make_probs(50, 6), _make_probs(50, 7)
    w = fit_weights([p1, p2], labels, pool="linear")
    combined = combine_probs([p1, p2], w, pool="linear")
    for row in combined:
        assert abs(sum(row) - 1.0) < 1e-6


def test_single_member_pool_is_identity():
    """Combining one member at weight 1.0 returns that member."""
    rng = np.random.default_rng(0)
    p = rng.dirichlet([2.0, 1.0, 1.0], 30).tolist()
    w = np.array([1.0])
    combined = combine_probs([p], w, pool="log")
    for orig, c in zip(p, combined):
        for a, b in zip(orig, c):
            assert abs(a - b) < 1e-5


# ── determinism ───────────────────────────────────────────────────────────────

def test_fit_weights_deterministic():
    labels = _make_labels(60, 11)
    p1, p2 = _make_probs(60, 12), _make_probs(60, 13)
    w1 = fit_weights([p1, p2], labels)
    w2 = fit_weights([p1, p2], labels)
    np.testing.assert_allclose(w1, w2, atol=1e-6)


# ── combine_matrices ──────────────────────────────────────────────────────────

def test_combine_matrices_sums_to_one():
    rng = np.random.default_rng(20)
    n = 15
    m1 = [rng.dirichlet(np.ones(81)).reshape(9, 9).tolist() for _ in range(n)]
    m2 = [rng.dirichlet(np.ones(81)).reshape(9, 9).tolist() for _ in range(n)]
    w = np.array([0.6, 0.4])
    blended = combine_matrices([m1, m2], w)
    assert len(blended) == n
    for mat in blended:
        total = sum(mat[i][j] for i in range(9) for j in range(9))
        assert abs(total - 1.0) < 1e-6


def test_combine_matrices_nonneg():
    rng = np.random.default_rng(21)
    n = 5
    m1 = [rng.dirichlet(np.ones(81)).reshape(9, 9).tolist() for _ in range(n)]
    m2 = [rng.dirichlet(np.ones(81)).reshape(9, 9).tolist() for _ in range(n)]
    w = np.array([0.5, 0.5])
    for mat in combine_matrices([m1, m2], w):
        assert all(mat[i][j] >= 0 for i in range(9) for j in range(9))


# ── matrix helpers ────────────────────────────────────────────────────────────

def test_matrix_to_lambdas_uniform():
    n = 9
    mat = [[1 / (n * n)] * n for _ in range(n)]
    la, lb = matrix_to_lambdas(mat)
    assert abs(la - 4.0) < 1e-5
    assert abs(lb - 4.0) < 1e-5


def test_matrix_to_top_scorelines_length_and_sum():
    rng = np.random.default_rng(30)
    mat = rng.dirichlet(np.ones(81)).reshape(9, 9).tolist()
    top = matrix_to_top_scorelines(mat, n=5)
    assert len(top) == 5
    probs = [s["prob"] for s in top]
    assert probs == sorted(probs, reverse=True)
