"""Tests for multinomial logistic W/D/L member."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wcpredictor.models.logistic import fit, predict_proba


def _make_df(
    elo_diffs: list[float],
    neutral: list[bool] | None = None,
    with_form: bool = True,
) -> pd.DataFrame:
    if neutral is None:
        neutral = [True] * len(elo_diffs)
    n = len(elo_diffs)
    data: dict = {"elo_diff_adj": elo_diffs, "neutral": neutral}
    if with_form:
        rng = np.random.default_rng(0)
        data["form_diff"] = rng.uniform(-1.5, 1.5, n).tolist()
        data["momentum_diff"] = rng.uniform(-2.0, 2.0, n).tolist()
        data["rest_diff"] = rng.uniform(-20.0, 20.0, n).tolist()
    return pd.DataFrame(data)


def _make_labels(elo_diffs: list[float], seed: int = 0) -> list[int]:
    """Generate labels that are weakly correlated with elo_diff_adj."""
    rng = np.random.default_rng(seed)
    labels = []
    for d in elo_diffs:
        p_win = 0.35 + 0.0003 * max(min(d, 300), -300)
        p_draw = 0.25
        p_loss = max(0.0, 1.0 - p_win - p_draw)
        labels.append(int(rng.choice(3, p=[p_win, p_draw, p_loss])))
    return labels


# ── output shape and validity ─────────────────────────────────────────────────

def test_probs_sum_to_one():
    n = 200
    diffs = np.linspace(-300, 300, n).tolist()
    df_train = _make_df(diffs)
    labels = _make_labels(diffs)
    scaler, model = fit(df_train, labels)
    probs = predict_proba(scaler, model, df_train)
    assert len(probs) == n
    for row in probs:
        assert len(row) == 3
        assert abs(sum(row) - 1.0) < 1e-6
        assert all(v >= 0 for v in row)


def test_predict_on_subset():
    diffs = np.linspace(-300, 300, 150).tolist()
    df_train = _make_df(diffs)
    labels = _make_labels(diffs)
    scaler, model = fit(df_train, labels)
    df_test = _make_df([0.0, 100.0, -200.0])
    probs = predict_proba(scaler, model, df_test)
    assert len(probs) == 3
    for row in probs:
        assert abs(sum(row) - 1.0) < 1e-6


# ── monotonicity in elo_diff_adj ──────────────────────────────────────────────

def test_win_prob_monotone_in_elo_diff():
    """Higher elo_diff_adj → higher p_win (team_a favored), form held at 0."""
    diffs_train = np.linspace(-400, 400, 300).tolist()
    df_train = _make_df(diffs_train)
    labels = _make_labels(diffs_train)
    scaler, model = fit(df_train, labels)

    test_diffs = [-200.0, -100.0, 0.0, 100.0, 200.0]
    # Hold form features at 0 so only elo_diff_adj varies
    probs = predict_proba(scaler, model, _make_df(test_diffs, with_form=False))
    p_wins = [row[0] for row in probs]
    p_losses = [row[2] for row in probs]

    # All pairwise comparisons
    for i in range(len(p_wins) - 1):
        assert p_wins[i] < p_wins[i + 1], f"p_win not increasing at i={i}"
        assert p_losses[i] > p_losses[i + 1], f"p_loss not decreasing at i={i}"


# ── time-aware leakage check ──────────────────────────────────────────────────

def test_disjoint_time_slices_no_leakage():
    """Fit on early slice, predict on later slice — no data from test used in fit."""
    rng = np.random.default_rng(77)
    n_train, n_test = 200, 50
    diffs_train = rng.normal(0, 150, n_train).tolist()
    diffs_test = rng.normal(0, 150, n_test).tolist()

    df_train = _make_df(diffs_train)
    df_test = _make_df(diffs_test)
    labels_train = _make_labels(diffs_train, seed=1)

    scaler, model = fit(df_train, labels_train)
    probs = predict_proba(scaler, model, df_test)

    assert len(probs) == n_test
    for row in probs:
        assert abs(sum(row) - 1.0) < 1e-6


# ── neutral flag ──────────────────────────────────────────────────────────────

def test_neutral_flag_accepted():
    diffs = [0.0, 100.0, -100.0]
    df_n = _make_df(diffs, neutral=[True, True, False])
    df_nn = _make_df(diffs, neutral=[False, False, True])
    labels = [0, 1, 2]
    scaler, model = fit(pd.concat([df_n, df_nn], ignore_index=True), labels * 2)
    probs_n = predict_proba(scaler, model, df_n)
    probs_nn = predict_proba(scaler, model, df_nn)
    # Outputs should differ (neutral flag has an effect) — just check validity
    for row in probs_n + probs_nn:
        assert abs(sum(row) - 1.0) < 1e-6


# ── absent form columns default to 0.0 ───────────────────────────────────────

def test_absent_form_columns_default_zero():
    """_build_X works when form columns are missing; defaults to 0.0."""
    diffs_train = np.linspace(-300, 300, 100).tolist()
    df_full = _make_df(diffs_train, with_form=True)
    df_no_form = _make_df(diffs_train, with_form=False)
    labels = _make_labels(diffs_train)

    scaler_full, model_full = fit(df_full, labels)
    scaler_bare, model_bare = fit(df_no_form, labels)

    # Predict with the bare-trained model on bare data — should work without KeyError
    probs = predict_proba(scaler_bare, model_bare, df_no_form)
    assert len(probs) == 100
    for row in probs:
        assert abs(sum(row) - 1.0) < 1e-6

    # Predict with the full-trained model on bare data (form cols absent → 0.0)
    probs_zero = predict_proba(scaler_full, model_full, df_no_form)
    assert len(probs_zero) == 100
    for row in probs_zero:
        assert abs(sum(row) - 1.0) < 1e-6


# ── monotonicity with form held constant ─────────────────────────────────────

def test_win_prob_monotone_with_form_held_constant():
    """With form_diff/momentum_diff/rest_diff fixed at 0, win prob still monotone in elo_diff."""
    diffs_train = np.linspace(-400, 400, 300).tolist()
    df_train = _make_df(diffs_train, with_form=True)
    labels = _make_labels(diffs_train)
    scaler, model = fit(df_train, labels)

    test_diffs = [-200.0, -100.0, 0.0, 100.0, 200.0]
    df_test = pd.DataFrame({
        "elo_diff_adj": test_diffs,
        "neutral": [True] * 5,
        "form_diff": [0.0] * 5,
        "momentum_diff": [0.0] * 5,
        "rest_diff": [0.0] * 5,
    })
    probs = predict_proba(scaler, model, df_test)
    p_wins = [row[0] for row in probs]
    p_losses = [row[2] for row in probs]

    for i in range(len(p_wins) - 1):
        assert p_wins[i] < p_wins[i + 1], f"p_win not increasing at i={i}"
        assert p_losses[i] > p_losses[i + 1], f"p_loss not decreasing at i={i}"
