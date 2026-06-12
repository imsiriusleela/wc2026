"""Tests for evaluation/ev_backtest.py — settlement correctness, threshold grid,
haircut sensitivity, and bootstrap reproducibility."""
from __future__ import annotations

import csv
import json
import math
import tempfile
from pathlib import Path

import numpy as np
import pytest


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _make_probs_csv(tmp_path: Path, rows: list[dict]) -> Path:
    """Write a minimal backtest_permatch_probs.csv for testing."""
    p = tmp_path / "backtest_permatch_probs.csv"
    fields = [
        "fold", "team_a", "team_b", "goals_a", "goals_b", "label",
        "ens_mkt_p_win", "ens_mkt_p_draw", "ens_mkt_p_loss",
        "model_p_win", "model_p_draw", "model_p_loss",
        "h_avg", "d_avg", "a_avg",
        "ah_line", "ah_home_odds", "ah_away_odds",
        "ou_line", "over_odds", "under_odds",
        "ah_fair_odds_home", "ah_fair_odds_away",
        "ou_fair_odds_over", "ou_fair_odds_under",
    ]
    with open(p, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            full = {k: row.get(k, "") for k in fields}
            writer.writerow(full)
    return p


def _base_row(goals_a=1, goals_b=0, fold=2022) -> dict:
    """A row where home wins (1-0), model gives 60% home.
    All market prices are above model fair → positive EV for at least one side per market.
    """
    return {
        "fold": fold,
        "team_a": "Argentina",
        "team_b": "France",
        "goals_a": goals_a,
        "goals_b": goals_b,
        "label": 0 if goals_a > goals_b else (1 if goals_a == goals_b else 2),
        "ens_mkt_p_win": 0.60, "ens_mkt_p_draw": 0.20, "ens_mkt_p_loss": 0.20,
        "model_p_win": 0.60, "model_p_draw": 0.20, "model_p_loss": 0.20,
        # 1X2: h_avg=1.70 > fair=1/0.60≈1.667 → +EV home
        "h_avg": 1.70, "d_avg": 3.60, "a_avg": 4.50,
        # AH: ah_home_odds=1.95 > ah_fair_home=1.80 → +EV home
        "ah_line": -0.5, "ah_home_odds": 1.95, "ah_away_odds": 1.90,
        # Total: over_odds=1.85 > ou_fair_over=1.70 → +EV over
        "ou_line": 2.5, "over_odds": 1.85, "under_odds": 1.95,
        "ah_fair_odds_home": 1.80, "ah_fair_odds_away": 2.10,
        "ou_fair_odds_over": 1.70, "ou_fair_odds_under": 1.80,
    }


# ─── Settlement correctness ───────────────────────────────────────────────────

def test_1x2_settle_home_win():
    from wcpredictor.evaluation.ev_backtest import _settle_1x2
    assert _settle_1x2(label=0, side="home") == 1.0
    assert _settle_1x2(label=0, side="draw") == 0.0
    assert _settle_1x2(label=0, side="away") == 0.0


def test_1x2_settle_draw():
    from wcpredictor.evaluation.ev_backtest import _settle_1x2
    assert _settle_1x2(label=1, side="draw") == 1.0
    assert _settle_1x2(label=1, side="home") == 0.0
    assert _settle_1x2(label=1, side="away") == 0.0


def test_ah_settle_full_win():
    """1-0 result, AH home -0.5 → home wins by 1 > 0.5 → full win."""
    from wcpredictor.evaluation.ev_backtest import _settle_ah_raw
    s = _settle_ah_raw(1, 0, -0.5, "home")
    assert s["p_win"] > 0.9


def test_ah_settle_push():
    """1-0 result, AH home -1.0 → diff=1 exactly covers 1.0 → push."""
    from wcpredictor.evaluation.ev_backtest import _settle_ah_raw
    s = _settle_ah_raw(1, 0, -1.0, "home")
    assert s["p_push"] > 0.9


def test_ah_settle_half_win():
    """1-0 result, AH home -0.75 quarter line → half win."""
    from wcpredictor.evaluation.ev_backtest import _settle_ah_raw
    s = _settle_ah_raw(1, 0, -0.75, "home")
    # -0.75 decomposes to -0.5 (win) and -1.0 (push) → half_win
    assert s["p_half_win"] > 0.9


def test_total_settle_over():
    """1-2 total=3, line=2.5 → over wins."""
    from wcpredictor.evaluation.ev_backtest import _settle_total_raw
    s = _settle_total_raw(1, 2, 2.5, "over")
    assert s["p_win"] > 0.9


def test_total_settle_under():
    """0-1 total=1, line=2.5 → under wins."""
    from wcpredictor.evaluation.ev_backtest import _settle_total_raw
    s = _settle_total_raw(0, 1, 2.5, "under")
    assert s["p_win"] > 0.9


# ─── EV formulas ─────────────────────────────────────────────────────────────

def test_ev_1x2_zero_at_fair():
    from wcpredictor.evaluation.ev_backtest import _ev_1x2
    p = 0.6
    fair = 1.0 / p
    assert abs(_ev_1x2(p, fair)) < 1e-6


def test_ev_1x2_haircut_reduces_value():
    from wcpredictor.evaluation.ev_backtest import _ev_1x2
    p = 0.6
    price = 1.80
    ev_0 = _ev_1x2(p, price, haircut=0.0)
    ev_5 = _ev_1x2(p, price, haircut=0.05)
    assert ev_5 < ev_0


def test_ev_asian_zero_at_fair():
    from wcpredictor.evaluation.ev_backtest import _ev_asian
    fair = 1.90
    ev = _ev_asian(fair, fair)
    assert abs(ev) < 1e-6


# ─── Threshold grid ───────────────────────────────────────────────────────────

def test_grid_returns_expected_keys(tmp_path):
    from wcpredictor.evaluation.ev_backtest import run_ev_backtest
    csv_path = _make_probs_csv(tmp_path, [_base_row()])
    report = run_ev_backtest(probs_csv=csv_path, n_bootstrap=10, seed=0)
    assert "grid" in report
    assert "verdict" in report
    assert "sizing_tier" in report
    assert "folds" in report


def test_grid_markets_present(tmp_path):
    from wcpredictor.evaluation.ev_backtest import run_ev_backtest
    csv_path = _make_probs_csv(tmp_path, [_base_row()] * 5)
    report = run_ev_backtest(probs_csv=csv_path, n_bootstrap=10, seed=0)
    markets = {r["market"] for r in report["grid"]}
    assert markets == {"1x2", "ah", "total"}


def test_grid_threshold_zero_has_bets(tmp_path):
    """At τ=0 and haircut=0 there should be bets in 1x2/ah/total."""
    from wcpredictor.evaluation.ev_backtest import run_ev_backtest
    rows = [_base_row()] * 10
    csv_path = _make_probs_csv(tmp_path, rows)
    report = run_ev_backtest(probs_csv=csv_path, n_bootstrap=10, seed=0)
    tau0_rows = [r for r in report["grid"] if r["threshold"] == 0.0 and r["haircut"] == 0.0]
    assert all(r["n_bets"] > 0 for r in tau0_rows)


def test_higher_threshold_fewer_bets(tmp_path):
    """EV ≥ τ filter: more bets at τ=0 than τ=0.10."""
    from wcpredictor.evaluation.ev_backtest import run_ev_backtest
    rows = [_base_row()] * 20
    csv_path = _make_probs_csv(tmp_path, rows)
    report = run_ev_backtest(probs_csv=csv_path, n_bootstrap=10, seed=0)
    for market in ("1x2", "ah", "total"):
        n_tau0 = next(r["n_bets"] for r in report["grid"]
                      if r["market"] == market and r["threshold"] == 0.0 and r["haircut"] == 0.0)
        n_tau10 = next(r["n_bets"] for r in report["grid"]
                       if r["market"] == market and r["threshold"] == 0.10 and r["haircut"] == 0.0)
        assert n_tau0 >= n_tau10


def test_haircut_reduces_bets_or_roi(tmp_path):
    """Shaving odds reduces EV → fewer bets or lower ROI at any given τ."""
    from wcpredictor.evaluation.ev_backtest import run_ev_backtest
    rows = [_base_row()] * 20
    csv_path = _make_probs_csv(tmp_path, rows)
    report = run_ev_backtest(probs_csv=csv_path, n_bootstrap=10, seed=0)
    for market in ("1x2",):
        r0 = next(r for r in report["grid"]
                  if r["market"] == market and r["threshold"] == 0.0 and r["haircut"] == 0.0)
        r5 = next(r for r in report["grid"]
                  if r["market"] == market and r["threshold"] == 0.0 and r["haircut"] == 0.05)
        # Either fewer bets or lower ROI after haircut
        assert r5["n_bets"] <= r0["n_bets"] or (
            isinstance(r5["roi"], float) and isinstance(r0["roi"], float) and r5["roi"] <= r0["roi"] + 1e-9
        )


# ─── Bootstrap reproducibility ────────────────────────────────────────────────

def test_bootstrap_seed_determinism(tmp_path):
    from wcpredictor.evaluation.ev_backtest import run_ev_backtest
    rows = [_base_row(ga, gb) for ga, gb in [(1, 0), (0, 0), (0, 1), (2, 1), (1, 2)]]
    csv_path = _make_probs_csv(tmp_path, rows)
    r1 = run_ev_backtest(probs_csv=csv_path, n_bootstrap=50, seed=42)
    r2 = run_ev_backtest(probs_csv=csv_path, n_bootstrap=50, seed=42)
    # Same seed → same CI bounds (NaN == NaN handled)
    for row1, row2 in zip(r1["grid"], r2["grid"]):
        if isinstance(row1["ci_lo"], float) and math.isnan(row1["ci_lo"]):
            assert isinstance(row2["ci_lo"], float) and math.isnan(row2["ci_lo"])
        else:
            assert row1["ci_lo"] == row2["ci_lo"]
            assert row1["ci_hi"] == row2["ci_hi"]


def test_missing_probs_csv_raises(tmp_path):
    from wcpredictor.evaluation.ev_backtest import run_ev_backtest
    with pytest.raises(FileNotFoundError):
        run_ev_backtest(probs_csv=tmp_path / "nonexistent.csv")


# ─── No-data rows gracefully handled ─────────────────────────────────────────

def test_missing_ah_odds_skipped(tmp_path):
    """Rows without AH odds are skipped without error."""
    from wcpredictor.evaluation.ev_backtest import run_ev_backtest
    row = _base_row()
    row["ah_line"] = ""
    row["ah_home_odds"] = ""
    row["ah_away_odds"] = ""
    row["ah_fair_odds_home"] = ""
    row["ah_fair_odds_away"] = ""
    csv_path = _make_probs_csv(tmp_path, [row] * 5)
    report = run_ev_backtest(probs_csv=csv_path, n_bootstrap=10, seed=0)
    ah_rows = [r for r in report["grid"] if r["market"] == "ah" and r["threshold"] == 0.0 and r["haircut"] == 0.0]
    assert ah_rows[0]["n_bets"] == 0
