"""Tests for form feature computation."""

from __future__ import annotations

import pandas as pd
import pytest

from wcpredictor.config import REST_DAYS_CAP
from wcpredictor.features.form import compute_form, form_row


def _matches(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df["neutral"] = df.get("neutral", False)
    if "neutral" not in df.columns:
        df["neutral"] = False
    return df.sort_values("date").reset_index(drop=True)


def _base_matches():
    return _matches([
        {"match_id": 1, "date": "2020-01-01", "team_a": "A", "team_b": "B",
         "goals_a": 2, "goals_b": 0, "neutral": False},
        {"match_id": 2, "date": "2020-01-08", "team_a": "B", "team_b": "C",
         "goals_a": 1, "goals_b": 1, "neutral": True},
        {"match_id": 3, "date": "2020-01-15", "team_a": "A", "team_b": "C",
         "goals_a": 0, "goals_b": 1, "neutral": False},
    ])


# ── output shape and no NaNs ──────────────────────────────────────────────────

def test_no_nans():
    df, _ = compute_form(_base_matches())
    assert not df.isnull().any().any()


def test_columns_present():
    df, _ = compute_form(_base_matches())
    assert set(df.columns) == {"match_id", "form_diff", "momentum_diff", "rest_diff"}


def test_row_count_matches_input():
    m = _base_matches()
    df, _ = compute_form(m)
    assert len(df) == len(m)


# ── cold-start / first-appearance defaults ────────────────────────────────────

def test_cold_start_neutral_defaults():
    """First match of every team → neutral (0, 0) form/momentum; rest = REST_DAYS_CAP each."""
    df, _ = compute_form(_base_matches())
    first_row = df[df["match_id"] == 1].iloc[0]
    assert first_row["form_diff"] == 0.0
    assert first_row["momentum_diff"] == 0.0
    # Both teams cold-start → rest_diff = cap - cap = 0
    assert first_row["rest_diff"] == 0.0


def test_first_appearance_uses_only_prior_matches():
    """Team C first appears in match_id=2; its pre-match form must be from cold-start."""
    df, _ = compute_form(_base_matches())
    row2 = df[df["match_id"] == 2].iloc[0]
    # B has one prior result (match 1: loss → 0 pts, gd=-2) vs C cold-start
    # form_diff = ppg(B) - ppg(C) = 0 - 0 = 0  (B got 0 pts in match 1)
    assert row2["form_diff"] == pytest.approx(0.0)
    # momentum_diff = mean_gd(B) - mean_gd(C) = (-2) - 0 = -2
    assert row2["momentum_diff"] == pytest.approx(-2.0)


# ── leak-free: pre-match emission ─────────────────────────────────────────────

def test_pre_match_emission():
    """form_diff in match 3 must NOT reflect match 3 outcome."""
    matches = _base_matches()
    df, _ = compute_form(matches)
    row3 = df[df["match_id"] == 3].iloc[0]
    # A had match 1 (win, gd=+2). C had match 2 (draw, gd=0).
    # form_diff = ppg(A) - ppg(C) = 3 - 1 = 2
    assert row3["form_diff"] == pytest.approx(2.0)
    # momentum_diff = 2 - 0 = 2
    assert row3["momentum_diff"] == pytest.approx(2.0)


# ── differenced symmetry ─────────────────────────────────────────────────────

def test_swap_teams_negates_diffs():
    """Swapping team_a / team_b in a match should negate all three diffs."""
    m1 = _matches([
        {"match_id": 1, "date": "2020-01-01", "team_a": "X", "team_b": "Y",
         "goals_a": 3, "goals_b": 1, "neutral": False},
        {"match_id": 2, "date": "2020-01-10", "team_a": "X", "team_b": "Y",
         "goals_a": 1, "goals_b": 0, "neutral": True},
    ])
    m2 = _matches([
        {"match_id": 1, "date": "2020-01-01", "team_a": "X", "team_b": "Y",
         "goals_a": 3, "goals_b": 1, "neutral": False},
        {"match_id": 2, "date": "2020-01-10", "team_a": "Y", "team_b": "X",
         "goals_a": 0, "goals_b": 1, "neutral": True},
    ])
    df1, _ = compute_form(m1)
    df2, _ = compute_form(m2)

    row1 = df1[df1["match_id"] == 2].iloc[0]
    row2 = df2[df2["match_id"] == 2].iloc[0]

    assert row1["form_diff"] == pytest.approx(-row2["form_diff"])
    assert row1["momentum_diff"] == pytest.approx(-row2["momentum_diff"])
    # rest_diff sign: same absolute rest values but swapped teams
    assert row1["rest_diff"] == pytest.approx(-row2["rest_diff"])


# ── rest_diff sign and cap ────────────────────────────────────────────────────

def test_rest_diff_sign():
    """Team A plays just before match 2; team B hasn't played recently → A has lower rest."""
    matches = _matches([
        {"match_id": 1, "date": "2020-01-01", "team_a": "A", "team_b": "X",
         "goals_a": 1, "goals_b": 0, "neutral": True},
        {"match_id": 2, "date": "2020-01-05", "team_a": "A", "team_b": "B",
         "goals_a": 0, "goals_b": 0, "neutral": True},
    ])
    df, _ = compute_form(matches)
    row2 = df[df["match_id"] == 2].iloc[0]
    # A last played Jan 1 → 4 days rest; B cold → REST_DAYS_CAP days rest
    # rest_diff = 4 - 30 = -26
    assert row2["rest_diff"] == pytest.approx(4.0 - REST_DAYS_CAP)


def test_rest_cap_applied():
    """Very long gap is capped at REST_DAYS_CAP."""
    matches = _matches([
        {"match_id": 1, "date": "2010-01-01", "team_a": "A", "team_b": "B",
         "goals_a": 1, "goals_b": 0, "neutral": True},
        {"match_id": 2, "date": "2020-06-01", "team_a": "A", "team_b": "B",
         "goals_a": 0, "goals_b": 0, "neutral": True},
    ])
    df, _ = compute_form(matches)
    row2 = df[df["match_id"] == 2].iloc[0]
    # Both teams had match 1; 10+ years > cap → rest = cap for both → diff = 0
    assert row2["rest_diff"] == pytest.approx(0.0)


# ── determinism ───────────────────────────────────────────────────────────────

def test_deterministic():
    m = _base_matches()
    df1, _ = compute_form(m)
    df2, _ = compute_form(m)
    pd.testing.assert_frame_equal(df1, df2)


# ── form_row helper ───────────────────────────────────────────────────────────

def test_form_row_cold_start():
    """form_row on empty state returns neutral defaults."""
    fr = form_row({}, "NewTeam", "OtherTeam", "2026-06-01")
    assert fr["form_diff"] == 0.0
    assert fr["momentum_diff"] == 0.0
    assert fr["rest_diff"] == 0.0


def test_form_row_matches_compute_form():
    """form_row using the returned state should match the next-match emission."""
    matches = _base_matches()
    _, state = compute_form(matches)

    # Match 3 was the last; simulate a hypothetical match 4 for same teams
    fr = form_row(state, "A", "C", "2020-01-20")
    # A: won match 1 (3pts, gd=+2), lost match 3 (0pts, gd=-1)
    # C: drew match 2 (1pt, gd=0), won match 3 (3pts, gd=+1)
    # ppg(A)=1.5, ppg(C)=2.0  → form_diff = -0.5
    assert fr["form_diff"] == pytest.approx(-0.5)
    # mean_gd(A) = (2 + -1)/2 = 0.5, mean_gd(C) = (0 + 1)/2 = 0.5 → diff = 0
    assert fr["momentum_diff"] == pytest.approx(0.0)
