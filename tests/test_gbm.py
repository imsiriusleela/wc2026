"""Tests for the gradient-boosted tree W/D/L member."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wcpredictor.models.gbm import _build_X, fit, predict_proba


def _make_features(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "elo_diff_adj": rng.normal(0, 100, n),
        "neutral": rng.integers(0, 2, n),
        "form_diff": rng.normal(0, 0.3, n),
        "momentum_diff": rng.normal(0, 0.3, n),
        "rest_diff": rng.normal(0, 2, n),
        "elo_a_pre": rng.normal(1500, 100, n),
        "elo_b_pre": rng.normal(1500, 100, n),
    })


def _make_labels(n: int, seed: int = 0) -> list[int]:
    return np.random.default_rng(seed).integers(0, 3, n).tolist()


# ── output shape and validity ─────────────────────────────────────────────────

def test_predict_proba_shape():
    df = _make_features(80)
    labels = _make_labels(80)
    model = fit(df, labels)
    preds = predict_proba(model, df[:20])
    assert len(preds) == 20
    for row in preds:
        assert len(row) == 3


def test_predict_proba_sums_to_one():
    df = _make_features(80)
    labels = _make_labels(80)
    model = fit(df, labels)
    preds = predict_proba(model, df)
    for row in preds:
        assert abs(sum(row) - 1.0) < 1e-6
        assert all(v >= 0 for v in row)


def test_predict_proba_no_nan():
    df = _make_features(80)
    labels = _make_labels(80)
    model = fit(df, labels)
    preds = predict_proba(model, df)
    for row in preds:
        assert all(not (v != v) for v in row)  # NaN check


# ── determinism ───────────────────────────────────────────────────────────────

def test_determinism():
    df = _make_features(100, seed=7)
    labels = _make_labels(100, seed=7)
    m1 = fit(df, labels)
    m2 = fit(df, labels)
    p1 = predict_proba(m1, df)
    p2 = predict_proba(m2, df)
    for r1, r2 in zip(p1, p2):
        for a, b in zip(r1, r2):
            assert abs(a - b) < 1e-10


# ── feature builder ───────────────────────────────────────────────────────────

def test_build_X_shape_with_all_cols():
    df = _make_features(30)
    X = _build_X(df)
    # 2 elo cols + 3 form cols + 2 raw elo cols = 7
    assert X.shape == (30, 7)


def test_build_X_missing_form_cols_zero_filled():
    df = pd.DataFrame({
        "elo_diff_adj": np.zeros(10),
        "neutral": np.zeros(10),
        "elo_a_pre": np.ones(10) * 1500,
        "elo_b_pre": np.ones(10) * 1500,
    })
    X = _build_X(df)
    assert X.shape == (10, 7)
    # form_diff, momentum_diff, rest_diff columns (indices 2,3,4) must be 0
    np.testing.assert_array_equal(X[:, 2], np.zeros(10))
    np.testing.assert_array_equal(X[:, 3], np.zeros(10))
    np.testing.assert_array_equal(X[:, 4], np.zeros(10))


def test_build_X_missing_raw_elo_cols_zero_filled():
    df = pd.DataFrame({
        "elo_diff_adj": np.zeros(5),
        "neutral": np.zeros(5),
    })
    X = _build_X(df)
    assert X.shape == (5, 7)
    np.testing.assert_array_equal(X[:, 5], np.zeros(5))
    np.testing.assert_array_equal(X[:, 6], np.zeros(5))


# ── missing class handling ────────────────────────────────────────────────────

def test_missing_class_renormalize():
    """If training labels contain only 0 and 2 (no draws), output still sums to 1."""
    rng = np.random.default_rng(99)
    df = pd.DataFrame({
        "elo_diff_adj": rng.normal(0, 100, 60),
        "neutral": np.zeros(60),
        "elo_a_pre": np.ones(60) * 1500,
        "elo_b_pre": np.ones(60) * 1500,
    })
    labels = [0 if i < 30 else 2 for i in range(60)]
    model = fit(df, labels)
    preds = predict_proba(model, df[:5])
    for row in preds:
        assert abs(sum(row) - 1.0) < 1e-6


# ── directional sanity ────────────────────────────────────────────────────────

def test_strong_team_wins_more_often():
    """Team A much stronger → p_win should exceed p_loss on average."""
    rng = np.random.default_rng(42)
    n = 200
    df_train = pd.DataFrame({
        "elo_diff_adj": rng.normal(200, 30, n),
        "neutral": np.zeros(n),
        "form_diff": np.zeros(n),
        "momentum_diff": np.zeros(n),
        "rest_diff": np.zeros(n),
        "elo_a_pre": np.ones(n) * 1700,
        "elo_b_pre": np.ones(n) * 1500,
    })
    # Biased labels: 70% wins for team A
    labels = [0 if rng.random() < 0.70 else (1 if rng.random() < 0.5 else 2) for _ in range(n)]

    model = fit(df_train, labels)
    test_row = pd.DataFrame({
        "elo_diff_adj": [250.0],
        "neutral": [0],
        "form_diff": [0.0],
        "momentum_diff": [0.0],
        "rest_diff": [0.0],
        "elo_a_pre": [1750.0],
        "elo_b_pre": [1500.0],
    })
    preds = predict_proba(model, test_row)
    assert preds[0][0] > preds[0][2], "stronger team should have higher p_win than p_loss"
