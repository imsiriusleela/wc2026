"""Elo leakage and correctness tests."""

import pandas as pd
import pytest

from wcpredictor.features.elo import compute_elo
from wcpredictor.config import INITIAL_RATING, HOME_ADVANTAGE


def _make_matches(rows):
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df["match_id"] = [f"m{i}" for i in range(len(df))]
    return df


def test_pre_match_elo_independent_of_result():
    """The pre-match Elo must not depend on that match's score."""
    matches = _make_matches([
        {"date": "2020-01-01", "team_a": "A", "team_b": "B",
         "goals_a": 5, "goals_b": 0, "neutral": True, "competition": "friendly"},
    ])
    elo1, _ = compute_elo(matches)

    matches2 = _make_matches([
        {"date": "2020-01-01", "team_a": "A", "team_b": "B",
         "goals_a": 0, "goals_b": 5, "neutral": True, "competition": "friendly"},
    ])
    elo2, _ = compute_elo(matches2)

    # Pre-match Elo should be identical regardless of the result
    assert elo1.iloc[0]["elo_a_pre"] == elo2.iloc[0]["elo_a_pre"]
    assert elo1.iloc[0]["elo_b_pre"] == elo2.iloc[0]["elo_b_pre"]


def test_neutral_zeroes_home_advantage():
    matches = _make_matches([
        {"date": "2020-01-01", "team_a": "A", "team_b": "B",
         "goals_a": 1, "goals_b": 1, "neutral": True, "competition": "friendly"},
    ])
    elo, _ = compute_elo(matches)
    row = elo.iloc[0]
    # Both start at INITIAL_RATING; neutral → diff = 0
    assert row["elo_diff_adj"] == pytest.approx(0.0)


def test_home_advantage_applied_on_non_neutral():
    matches = _make_matches([
        {"date": "2020-01-01", "team_a": "A", "team_b": "B",
         "goals_a": 1, "goals_b": 1, "neutral": False, "competition": "friendly"},
    ])
    elo, _ = compute_elo(matches)
    row = elo.iloc[0]
    # elo_diff_adj = r_a + HOME_ADVANTAGE - r_b = 0 + 50 = 50 (both seed at 1500)
    assert row["elo_diff_adj"] == pytest.approx(HOME_ADVANTAGE)


def test_deterministic():
    matches = _make_matches([
        {"date": "2020-01-01", "team_a": "A", "team_b": "B",
         "goals_a": 2, "goals_b": 1, "neutral": True, "competition": "world_cup"},
        {"date": "2020-02-01", "team_a": "B", "team_b": "C",
         "goals_a": 0, "goals_b": 0, "neutral": False, "competition": "qualifier"},
    ])
    elo1, r1 = compute_elo(matches)
    elo2, r2 = compute_elo(matches)
    pd.testing.assert_frame_equal(elo1, elo2)
    assert r1 == r2


def test_ratings_change_after_match():
    matches = _make_matches([
        {"date": "2020-01-01", "team_a": "A", "team_b": "B",
         "goals_a": 3, "goals_b": 0, "neutral": True, "competition": "world_cup"},
        {"date": "2020-02-01", "team_a": "A", "team_b": "B",
         "goals_a": 1, "goals_b": 1, "neutral": True, "competition": "world_cup"},
    ])
    elo, final = compute_elo(matches)
    # Second match pre-match rating should differ from first (result updated it)
    assert elo.iloc[1]["elo_a_pre"] != INITIAL_RATING
    assert elo.iloc[1]["elo_b_pre"] != INITIAL_RATING
    # Final ratings should differ from pre-match of second match
    assert final["A"] != elo.iloc[1]["elo_a_pre"]


def test_final_ratings_reflect_post_match_update():
    matches = _make_matches([
        {"date": "2020-01-01", "team_a": "A", "team_b": "B",
         "goals_a": 1, "goals_b": 0, "neutral": True, "competition": "world_cup"},
    ])
    elo, final = compute_elo(matches)
    # A won → A's final rating > initial; B's < initial
    assert final["A"] > INITIAL_RATING
    assert final["B"] < INITIAL_RATING
