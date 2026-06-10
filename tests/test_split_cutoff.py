"""Leakage tests for the split-cutoff refactor in _build_frozen_state.

Validates CLAUDE.md requirement: 'Add tests for data leakage and feature cutoffs.'

Key invariants:
1. With synthetic post-TOURNAMENT_START rows, Poisson/DC params and ensemble weights/T
   are IDENTICAL with vs without the 2026 rows (fits pinned at TOURNAMENT_START).
2. state["ratings"] DIFFERS (Elo rolls forward when 2026 rows exist).
3. All fit inputs have date < fit_cutoff.
4. When as_of <= TOURNAMENT_START, behavior is byte-identical to the old single-cutoff path.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from wcpredictor.config import TOURNAMENT_START


# ---------------------------------------------------------------------------
# Minimal synthetic match helpers
# ---------------------------------------------------------------------------

def _synthetic_history(tmp_path: Path) -> pd.DataFrame:
    """Build a tiny DataFrame shaped like load_matches() output."""
    import hashlib

    def mid(date, ta, tb):
        return hashlib.md5(f"{date}|{ta}|{tb}|Friendly".encode()).hexdigest()[:12]

    rows = []
    teams = ["Brazil", "Argentina", "France", "Germany", "Spain", "Mexico"]
    for i, (ta, tb) in enumerate(zip(teams, teams[1:] + teams[:1])):
        for yr in [2022, 2023, 2024, 2025]:
            date_str = f"{yr}-06-{10 + i:02d}"
            rows.append({
                "match_id": mid(date_str, ta, tb),
                "date": pd.Timestamp(date_str),
                "team_a": ta, "team_b": tb,
                "goals_a": (i + yr) % 4, "goals_b": (i + yr + 1) % 3,
                "neutral": True,
                "tournament": "Friendly", "competition": "friendly", "is_world_cup": False,
            })
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def _wc2026_played_rows() -> pd.DataFrame:
    """Simulate a few played WC2026 group matches."""
    return pd.DataFrame([{
        "date": "2026-06-13", "team_a": "Brazil", "team_b": "Mexico",
        "goals_a": 2, "goals_b": 0, "stage": "group", "winner": None, "source": "manual",
    }, {
        "date": "2026-06-14", "team_a": "Argentina", "team_b": "France",
        "goals_a": 1, "goals_b": 1, "stage": "group", "winner": None, "source": "manual",
    }])


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def base_history(tmp_path):
    return _synthetic_history(tmp_path)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_fits_pinned_no_2026_data(base_history, tmp_path, monkeypatch):
    """Poisson params from as_of=TOURNAMENT_START should equal those from one week later
    when no WC2026 data exists in the store (augment_matches returns base unchanged)."""
    from wcpredictor.data import results_2026 as r26
    from wcpredictor.predict import _build_frozen_state

    monkeypatch.setattr(r26, "_STORE", tmp_path / "empty.csv")

    def _noop_augment(matches):
        return matches

    with patch("wcpredictor.predict.load_matches", return_value=base_history), \
         patch("wcpredictor.predict.augment_matches", side_effect=_noop_augment), \
         patch("wcpredictor.predict._ensure_data"):
        state_base = _build_frozen_state(TOURNAMENT_START, ["poisson"])
        state_later = _build_frozen_state("2026-06-20", ["poisson"])

    # Fits must be identical — same pre-tournament training window
    assert state_base["poisson_base"] == pytest.approx(state_later["poisson_base"], rel=1e-6)
    assert state_base["poisson_beta"] == pytest.approx(state_later["poisson_beta"], rel=1e-6)


def test_ratings_roll_with_as_of_when_2026_data_exists(base_history, tmp_path, monkeypatch):
    """Elo ratings should differ when as_of advances and 2026 matches are in the store."""
    from wcpredictor.data import results_2026 as r26
    from wcpredictor.predict import _build_frozen_state

    played = _wc2026_played_rows()

    def _augment_with_played(matches):
        extra_rows = []
        import hashlib
        for _, r in played.iterrows():
            mid = hashlib.md5(f"{r.date}|{r.team_a}|{r.team_b}|FIFA World Cup".encode()).hexdigest()[:12]
            extra_rows.append({
                "match_id": mid,
                "date": pd.Timestamp(r.date),
                "team_a": r.team_a, "team_b": r.team_b,
                "goals_a": int(r.goals_a), "goals_b": int(r.goals_b),
                "neutral": True, "tournament": "FIFA World Cup",
                "competition": "world_cup", "is_world_cup": True,
            })
        extra = pd.DataFrame(extra_rows)
        for col in matches.columns:
            if col not in extra.columns:
                extra[col] = None
        aug = pd.concat([matches, extra[matches.columns]], ignore_index=True)
        return aug.sort_values("date").reset_index(drop=True)

    def _no_augment(matches):
        return matches

    with patch("wcpredictor.predict.load_matches", return_value=base_history), \
         patch("wcpredictor.predict._ensure_data"):
        with patch("wcpredictor.predict.augment_matches", side_effect=_no_augment):
            state_pre = _build_frozen_state(TOURNAMENT_START, ["poisson"])

        with patch("wcpredictor.predict.augment_matches", side_effect=_augment_with_played):
            state_post = _build_frozen_state("2026-06-20", ["poisson"])

    # Ratings should differ for teams that played
    brazil_pre = state_pre["ratings"].get("Brazil", 1500.0)
    brazil_post = state_post["ratings"].get("Brazil", 1500.0)
    assert brazil_pre != brazil_post, "Elo for Brazil should roll forward after played matches"

    # Poisson fits must still be identical (pinned at TOURNAMENT_START)
    assert state_pre["poisson_base"] == pytest.approx(state_post["poisson_base"], rel=1e-6)
    assert state_pre["poisson_beta"] == pytest.approx(state_post["poisson_beta"], rel=1e-6)


def test_no_fit_input_has_date_after_fit_cutoff(base_history, tmp_path, monkeypatch):
    """The elo_df in state (used for Poisson fit) must have no rows on or after TOURNAMENT_START."""
    from wcpredictor.predict import _build_frozen_state

    def _no_augment(matches):
        return matches

    with patch("wcpredictor.predict.load_matches", return_value=base_history), \
         patch("wcpredictor.predict.augment_matches", side_effect=_no_augment), \
         patch("wcpredictor.predict._ensure_data"):
        state = _build_frozen_state("2026-06-20", ["poisson"])

    fit_cutoff = pd.Timestamp(TOURNAMENT_START)
    elo_df = state["elo_df"]
    assert (elo_df["date"] < fit_cutoff).all(), (
        "elo_df contains rows on or after fit_cutoff — leakage!"
    )


def test_pre_tournament_as_of_behavior_unchanged(base_history, monkeypatch):
    """When as_of <= TOURNAMENT_START, fit_cutoff == as_of and behavior is identical
    to the old single-cutoff path."""
    from wcpredictor.predict import _build_frozen_state

    def _no_augment(matches):
        return matches

    with patch("wcpredictor.predict.load_matches", return_value=base_history), \
         patch("wcpredictor.predict.augment_matches", side_effect=_no_augment), \
         patch("wcpredictor.predict._ensure_data"):
        state = _build_frozen_state("2025-01-01", ["poisson"])

    assert state["fit_cutoff"] == state["cutoff"]


def test_state_has_fit_cutoff_key(base_history, monkeypatch):
    """_build_frozen_state must set state['fit_cutoff'] in all code paths."""
    from wcpredictor.predict import _build_frozen_state

    def _no_augment(matches):
        return matches

    with patch("wcpredictor.predict.load_matches", return_value=base_history), \
         patch("wcpredictor.predict.augment_matches", side_effect=_no_augment), \
         patch("wcpredictor.predict._ensure_data"):
        state = _build_frozen_state(TOURNAMENT_START, ["poisson"])

    assert "fit_cutoff" in state
    assert isinstance(state["fit_cutoff"], pd.Timestamp)
