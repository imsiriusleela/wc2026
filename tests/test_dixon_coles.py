"""Tests for Dixon-Coles model."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import poisson

from wcpredictor.config import MAX_GOALS
from wcpredictor.models.dixon_coles import DCParams, fit, predict_one


# ─── helpers ────────────────────────────────────────────────────────────────

def _params(alpha_a=0.0, beta_a=0.0, alpha_b=0.0, beta_b=0.0,
            home_adv=0.3, rho=-0.13) -> DCParams:
    return DCParams(
        attack={"A": alpha_a, "B": alpha_b},
        defense={"A": beta_a, "B": beta_b},
        home_adv=home_adv,
        rho=rho,
        mean_attack=0.0,
        mean_defense=0.0,
    )


def _make_matches(n: int = 300, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    teams = [f"Team{i}" for i in range(10)]
    dates = pd.date_range("2015-01-01", periods=n, freq="3D")
    rows = []
    for date in dates:
        a, b = rng.choice(teams, 2, replace=False)
        la = 1.8 if a == "Team0" else 1.0
        mu = 1.8 if b == "Team0" else 1.0
        ga = int(rng.poisson(la))
        gb = int(rng.poisson(mu))
        rows.append({"team_a": a, "team_b": b, "goals_a": ga, "goals_b": gb,
                     "neutral": True, "date": date})
    return pd.DataFrame(rows)


# ─── predict_one (no fit needed) ────────────────────────────────────────────

def test_probs_nonneg_sum_to_one():
    p = predict_one(_params(), "A", "B")
    assert p["p_win"] >= 0
    assert p["p_draw"] >= 0
    assert p["p_loss"] >= 0
    assert abs(p["p_win"] + p["p_draw"] + p["p_loss"] - 1.0) < 1e-6


def test_matrix_shape_sums_to_one():
    p = predict_one(_params(), "A", "B")
    m = np.array(p["score_matrix"])
    assert m.shape == (MAX_GOALS + 1, MAX_GOALS + 1)
    assert abs(m.sum() - 1.0) < 1e-6


def test_tau_applied_only_to_low_scores():
    """Cells (i,j) in the DC matrix must equal the τ-corrected independent product."""
    par = _params(rho=-0.13)
    p = predict_one(par, "A", "B")
    la, mu, rho = p["lambda_a"], p["lambda_b"], par.rho

    pa = np.array([poisson.pmf(i, la) for i in range(MAX_GOALS + 1)])
    pb = np.array([poisson.pmf(j, mu) for j in range(MAX_GOALS + 1)])
    indep = np.outer(pa, pb)

    indep[0, 0] *= max(1.0 - la * mu * rho, 1e-10)
    indep[0, 1] *= max(1.0 + la * rho, 1e-10)
    indep[1, 0] *= max(1.0 + mu * rho, 1e-10)
    indep[1, 1] *= max(1.0 - rho, 1e-10)
    indep /= indep.sum()

    np.testing.assert_allclose(np.array(p["score_matrix"]), indep, atol=1e-6)


def test_stronger_attack_higher_lambda_and_win_prob():
    p_strong = predict_one(_params(alpha_a=0.5), "A", "B")
    p_weak = predict_one(_params(alpha_a=-0.5), "A", "B")
    assert p_strong["lambda_a"] > p_weak["lambda_a"]
    assert p_strong["p_win"] > p_weak["p_win"]


def test_neutral_drops_home_advantage():
    par = _params(home_adv=0.5)
    p_home = predict_one(par, "A", "B", neutral=False)
    p_neutral = predict_one(par, "A", "B", neutral=True)
    # Home flag affects team_a's λ only
    assert p_home["lambda_a"] > p_neutral["lambda_a"]
    # team_b's μ = exp(alpha_b + beta_a) — unaffected by home flag
    assert abs(p_home["lambda_b"] - p_neutral["lambda_b"]) < 1e-6


def test_unseen_team_fallback_no_crash():
    par = _params()
    p = predict_one(par, "Unknown", "B")
    assert abs(p["p_win"] + p["p_draw"] + p["p_loss"] - 1.0) < 1e-6


def test_predict_one_deterministic():
    par = _params()
    assert predict_one(par, "A", "B") == predict_one(par, "A", "B")


def test_top_scorelines_has_five():
    p = predict_one(_params(), "A", "B")
    assert len(p["top_scorelines"]) == 5


# ─── fit ────────────────────────────────────────────────────────────────────

def test_fit_strong_team_positive_attack():
    matches = _make_matches()
    ref = matches["date"].max() + pd.Timedelta(days=1)
    params = fit(matches, ref_date=ref)
    assert "Team0" in params.attack
    # Team0 generated goals at 1.8x rate → above-average attack
    assert params.attack["Team0"] > 0.0


def test_fit_mean_alpha_near_zero():
    matches = _make_matches()
    ref = matches["date"].max() + pd.Timedelta(days=1)
    params = fit(matches, ref_date=ref)
    mean_alpha = float(np.mean(list(params.attack.values())))
    assert abs(mean_alpha) < 0.15


def test_fit_unseen_team_predict_no_crash():
    matches = _make_matches()
    ref = matches["date"].max() + pd.Timedelta(days=1)
    params = fit(matches, ref_date=ref)
    p = predict_one(params, "Atlantis", "Team0")
    assert abs(p["p_win"] + p["p_draw"] + p["p_loss"] - 1.0) < 1e-6


def test_fit_deterministic():
    matches = _make_matches()
    ref = matches["date"].max() + pd.Timedelta(days=1)
    p1 = fit(matches, ref_date=ref)
    p2 = fit(matches, ref_date=ref)
    assert abs(p1.home_adv - p2.home_adv) < 1e-5
    assert abs(p1.rho - p2.rho) < 1e-5
    for team in p1.attack:
        assert abs(p1.attack[team] - p2.attack.get(team, 0.0)) < 1e-5
