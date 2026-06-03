"""Regression test: ensemble_mkt market-blend activation.

Guards the NameError that would fire when a year==2026 odds row is present, and
verifies that the blend actually shifts W/D/L toward the injected market odds.
"""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest


# Synthetic 2026 odds strongly favouring Brazil (team_a)
_MARKET_WIN = 0.75
_MARKET_DRAW = 0.15
_MARKET_LOSS = 0.10
_BLEND_ALPHA = 0.50


def _fake_odds_df() -> pd.DataFrame:
    """One 2026 row for Brazil vs France plus a dummy 2022 row (hist data)."""
    return pd.DataFrame([
        {
            "year": 2026,
            "date": pd.Timestamp("2026-06-15"),
            "team_a": "Brazil",
            "team_b": "France",
            "p_win": _MARKET_WIN,
            "p_draw": _MARKET_DRAW,
            "p_loss": _MARKET_LOSS,
        },
        {
            "year": 2022,
            "date": pd.Timestamp("2022-12-14"),
            "team_a": "Brazil",
            "team_b": "Croatia",
            "p_win": 0.60,
            "p_draw": 0.20,
            "p_loss": 0.20,
        },
    ])


@pytest.fixture(scope="module")
def blend_results():
    """Run both models once; share the heavy ensemble fit across assertions."""
    from wcpredictor.predict import predict_match

    with (
        patch("wcpredictor.features.odds.load_wc_odds", return_value=_fake_odds_df()),
        patch("wcpredictor.predict._resolve_odds_alpha", return_value=_BLEND_ALPHA),
    ):
        res_ens = predict_match("Brazil", "France", "2026-06-15", neutral=True, model="ensemble")
        res_mkt = predict_match("Brazil", "France", "2026-06-15", neutral=True, model="ensemble_mkt")

    return res_ens, res_mkt


def test_no_exception(blend_results):
    """Neither model raises — specifically guards the NameError on ODDS_ALPHA_PRIOR."""
    res_ens, res_mkt = blend_results
    assert "p_win" in res_ens
    assert "p_win" in res_mkt


def test_ensemble_probs_sum_to_one(blend_results):
    res_ens, _ = blend_results
    total = res_ens["p_win"] + res_ens["p_draw"] + res_ens["p_loss"]
    assert abs(total - 1.0) < 1e-4


def test_ensemble_mkt_probs_sum_to_one(blend_results):
    _, res_mkt = blend_results
    total = res_mkt["p_win"] + res_mkt["p_draw"] + res_mkt["p_loss"]
    assert abs(total - 1.0) < 1e-4


def test_blend_shifts_toward_market_odds(blend_results):
    """ensemble_mkt p_win must be between ensemble p_win and the injected market p_win."""
    res_ens, res_mkt = blend_results
    pw_ens = res_ens["p_win"]
    pw_mkt = res_mkt["p_win"]
    # Market odds strongly favour Brazil (0.75); ensemble alone is typically < 0.75.
    # With alpha=0.5 the blend must land strictly between the two.
    expected = (1 - _BLEND_ALPHA) * pw_ens + _BLEND_ALPHA * _MARKET_WIN
    assert abs(pw_mkt - expected) < 1e-4, (
        f"expected blend {expected:.6f}, got {pw_mkt:.6f} (ens={pw_ens:.6f})"
    )
