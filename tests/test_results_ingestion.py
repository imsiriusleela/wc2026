"""Tests for results_2026.py: filtering, canonicalization, dedup, mark_fixtures, augment_matches."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest

from wcpredictor.config import KO_START, TOURNAMENT_START


# ---------------------------------------------------------------------------
# Helpers to build minimal DataFrames
# ---------------------------------------------------------------------------

def _make_matches_df(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


def _martj42_row(date: str, home: str, away: str, hs: int, as_: int,
                 tournament: str = "FIFA World Cup") -> dict:
    return {
        "date": date, "home_team": home, "away_team": away,
        "home_score": hs, "away_score": as_, "tournament": tournament,
        "neutral": True, "country": "USA", "city": "New York",
    }


# ---------------------------------------------------------------------------
# fetch_master_results — covered via update_wc2026_results(source_csv=...)
# ---------------------------------------------------------------------------

def test_update_filters_to_wc2026_rows_only(tmp_path: Path, monkeypatch):
    """Only FIFA World Cup rows on or after TOURNAMENT_START with scores are ingested."""
    from wcpredictor.data import results_2026 as r26

    monkeypatch.setattr(r26, "_STORE", tmp_path / "wc2026_results.csv")
    monkeypatch.setattr(r26, "_MASTER", tmp_path / "results_master.csv")

    csv = tmp_path / "input.csv"
    rows = [
        _martj42_row("2026-06-12", "Brazil", "Mexico", 2, 1),          # WC, in-window → keep
        _martj42_row("2026-06-10", "Brazil", "Mexico", 1, 0),          # WC, before start → drop
        _martj42_row("2026-06-12", "Spain", "France", 0, 1, "Friendly"),  # not WC → drop
        _martj42_row("2026-06-15", "Germany", "Argentina", None, None), # null score → drop
    ]
    pd.DataFrame(rows).to_csv(csv, index=False)

    stats = r26.update_wc2026_results(source_csv=csv)
    assert stats["n_total"] == 1
    assert stats["n_group"] == 1

    stored = r26.load_wc2026_results()
    assert len(stored) == 1
    assert stored.iloc[0]["team_a"] == "Brazil"
    assert stored.iloc[0]["team_b"] == "Mexico"


def test_update_stage_classification(tmp_path: Path, monkeypatch):
    """Rows before KO_START → group; on or after → knockout."""
    from wcpredictor.data import results_2026 as r26

    monkeypatch.setattr(r26, "_STORE", tmp_path / "wc2026_results.csv")
    monkeypatch.setattr(r26, "_MASTER", tmp_path / "results_master.csv")

    csv = tmp_path / "input.csv"
    rows = [
        _martj42_row("2026-06-15", "Brazil", "Mexico", 2, 1),          # before KO_START → group
        _martj42_row(KO_START, "France", "Germany", 1, 0),             # on KO_START → knockout
    ]
    pd.DataFrame(rows).to_csv(csv, index=False)

    r26.update_wc2026_results(source_csv=csv)
    stored = r26.load_wc2026_results()
    assert set(stored["stage"].tolist()) == {"group", "knockout"}
    grp = stored[stored["team_a"] == "Brazil"]
    ko = stored[stored["team_a"] == "France"]
    assert grp.iloc[0]["stage"] == "group"
    assert ko.iloc[0]["stage"] == "knockout"


def test_manual_row_wins_over_martj42(tmp_path: Path, monkeypatch):
    """If the store already has a manual row for a match, martj42 must not overwrite it."""
    from wcpredictor.data import results_2026 as r26

    monkeypatch.setattr(r26, "_STORE", tmp_path / "wc2026_results.csv")
    monkeypatch.setattr(r26, "_MASTER", tmp_path / "results_master.csv")

    # Seed the store with a manual row (goals 3-0)
    manual = _make_matches_df([{
        "date": "2026-06-15", "team_a": "Brazil", "team_b": "Mexico",
        "goals_a": 3, "goals_b": 0, "stage": "group", "winner": None, "source": "manual",
    }])
    manual.to_csv(tmp_path / "wc2026_results.csv", index=False)

    # Simulate a martj42 fetch that returns a different score (1-1)
    m42_df = pd.DataFrame([_martj42_row("2026-06-15", "Brazil", "Mexico", 1, 1)])
    monkeypatch.setattr(r26, "fetch_master_results", lambda: m42_df)
    r26.update_wc2026_results()  # no source_csv → martj42 path

    stored = r26.load_wc2026_results()
    assert len(stored) == 1
    # Manual row preserved (goals_a == 3)
    assert int(stored.iloc[0]["goals_a"]) == 3
    assert stored.iloc[0]["source"] == "manual"


def test_dedupe_same_match_no_duplicate(tmp_path: Path, monkeypatch):
    """Updating twice with the same martj42 row must not create duplicates."""
    from wcpredictor.data import results_2026 as r26

    monkeypatch.setattr(r26, "_STORE", tmp_path / "wc2026_results.csv")
    monkeypatch.setattr(r26, "_MASTER", tmp_path / "results_master.csv")

    m42_df = pd.DataFrame([_martj42_row("2026-06-15", "Brazil", "Mexico", 2, 1)])
    monkeypatch.setattr(r26, "fetch_master_results", lambda: m42_df)

    r26.update_wc2026_results()  # first martj42 pull
    r26.update_wc2026_results()  # second martj42 pull — same data

    stored = r26.load_wc2026_results()
    assert len(stored) == 1


def test_canonical_team_names(tmp_path: Path, monkeypatch):
    """Team names are canonicalized on ingestion."""
    from wcpredictor.data import results_2026 as r26
    from wcpredictor.data.normalize_teams import canonical

    monkeypatch.setattr(r26, "_STORE", tmp_path / "wc2026_results.csv")
    monkeypatch.setattr(r26, "_MASTER", tmp_path / "results_master.csv")

    csv = tmp_path / "input.csv"
    # "usa" should canonicalize to "United States"
    pd.DataFrame([_martj42_row("2026-06-15", "USA", "Canada", 2, 0)]).to_csv(csv, index=False)
    r26.update_wc2026_results(source_csv=csv)

    stored = r26.load_wc2026_results()
    assert stored.iloc[0]["team_a"] == canonical("USA")
    assert stored.iloc[0]["team_b"] == canonical("Canada")


def test_mark_fixtures_played(tmp_path: Path, monkeypatch):
    """mark_fixtures_played fills goals in fixtures CSV from the store."""
    from wcpredictor.data import results_2026 as r26

    monkeypatch.setattr(r26, "_STORE", tmp_path / "wc2026_results.csv")

    # Seed the store
    store_df = _make_matches_df([{
        "date": "2026-06-15", "team_a": "Brazil", "team_b": "Mexico",
        "goals_a": 2, "goals_b": 1, "stage": "group", "winner": None, "source": "manual",
    }])
    store_df.to_csv(tmp_path / "wc2026_results.csv", index=False)

    fixtures = tmp_path / "fixtures.csv"
    pd.DataFrame([
        {"date": "2026-06-15", "team_a": "Brazil", "team_b": "Mexico", "neutral": True, "goals_a": "", "goals_b": ""},
        {"date": "2026-06-16", "team_a": "France", "team_b": "Germany", "neutral": True, "goals_a": "", "goals_b": ""},
    ]).to_csv(fixtures, index=False)

    n = r26.mark_fixtures_played(fixtures_path=fixtures)
    assert n == 1

    updated = pd.read_csv(fixtures)
    played = updated[updated["team_a"] == "Brazil"].iloc[0]
    assert int(played["goals_a"]) == 2
    assert int(played["goals_b"]) == 1

    unplayed = updated[updated["team_a"] == "France"].iloc[0]
    assert str(unplayed.get("goals_a", "")) in ("", "nan")


def test_augment_matches_no_duplicates(tmp_path: Path, monkeypatch):
    """augment_matches must not duplicate rows already present in the base matches DataFrame."""
    from wcpredictor.data import results_2026 as r26

    monkeypatch.setattr(r26, "_STORE", tmp_path / "wc2026_results.csv")

    existing_mid = hashlib.md5("2026-06-15|Brazil|Mexico|FIFA World Cup".encode()).hexdigest()[:12]
    base_matches = pd.DataFrame([{
        "match_id": existing_mid,
        "date": pd.Timestamp("2026-06-15"),
        "team_a": "Brazil", "team_b": "Mexico",
        "goals_a": 2, "goals_b": 1,
        "neutral": True, "tournament": "FIFA World Cup",
        "competition": "world_cup", "is_world_cup": True,
    }])

    store_df = _make_matches_df([{
        "date": "2026-06-15", "team_a": "Brazil", "team_b": "Mexico",
        "goals_a": 2, "goals_b": 1, "stage": "group", "winner": None, "source": "manual",
    }])
    store_df.to_csv(tmp_path / "wc2026_results.csv", index=False)

    augmented = r26.augment_matches(base_matches)
    assert len(augmented) == 1  # no duplicate


def test_augment_matches_adds_new(tmp_path: Path, monkeypatch):
    """augment_matches appends WC2026 rows not already in base matches."""
    from wcpredictor.data import results_2026 as r26

    monkeypatch.setattr(r26, "_STORE", tmp_path / "wc2026_results.csv")

    base_matches = pd.DataFrame([{
        "match_id": "aaa000000001",
        "date": pd.Timestamp("2025-09-01"),
        "team_a": "Brazil", "team_b": "Argentina",
        "goals_a": 1, "goals_b": 0,
        "neutral": False, "tournament": "Friendly",
        "competition": "friendly", "is_world_cup": False,
    }])

    store_df = _make_matches_df([{
        "date": "2026-06-15", "team_a": "Brazil", "team_b": "Mexico",
        "goals_a": 2, "goals_b": 1, "stage": "group", "winner": None, "source": "manual",
    }])
    store_df.to_csv(tmp_path / "wc2026_results.csv", index=False)

    augmented = r26.augment_matches(base_matches)
    assert len(augmented) == 2
    # WC2026 row has correct fields
    wc_row = augmented[augmented["tournament"] == "FIFA World Cup"].iloc[0]
    assert bool(wc_row["is_world_cup"]) is True
    assert bool(wc_row["neutral"]) is True


def test_load_wc2026_results_absent_returns_empty(tmp_path: Path, monkeypatch):
    """load_wc2026_results returns a typed empty DataFrame when the store file is absent."""
    from wcpredictor.data import results_2026 as r26

    monkeypatch.setattr(r26, "_STORE", tmp_path / "nonexistent.csv")
    df = r26.load_wc2026_results()
    assert df.empty
    assert "date" in df.columns
    assert "stage" in df.columns
