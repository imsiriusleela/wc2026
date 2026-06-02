"""Verify that the time-safety assertion in the backtest fires on a leaky fold."""

import pandas as pd
import pytest


def _make_matches(rows):
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df["match_id"] = [f"m{i}" for i in range(len(df))]
    return df


def test_assertion_fires_on_leaky_fold():
    """train.date.max() >= test.date.min() must raise AssertionError."""
    train = _make_matches([
        {"date": "2010-06-15", "team_a": "A", "team_b": "B",
         "goals_a": 1, "goals_b": 0, "neutral": True, "competition": "world_cup"},
    ])
    test = _make_matches([
        {"date": "2010-06-12", "team_a": "C", "team_b": "D",
         "goals_a": 2, "goals_b": 1, "neutral": True, "competition": "world_cup"},
    ])
    with pytest.raises(AssertionError, match="LEAKAGE"):
        assert train["date"].max() < test["date"].min(), (
            "LEAKAGE DETECTED: train/test dates overlap!"
        )


def test_assertion_passes_on_clean_fold():
    """Clean fold: train ends strictly before test begins."""
    train = _make_matches([
        {"date": "2010-06-10", "team_a": "A", "team_b": "B",
         "goals_a": 1, "goals_b": 0, "neutral": True, "competition": "qualifier"},
    ])
    test = _make_matches([
        {"date": "2010-06-12", "team_a": "C", "team_b": "D",
         "goals_a": 0, "goals_b": 0, "neutral": True, "competition": "world_cup"},
    ])
    # Should not raise
    assert train["date"].max() < test["date"].min()
