"""Tests for features/odds.py and data/download_odds.py."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest

from wcpredictor.config import DATA_RAW, FDCO_ODDS_SHA256
from wcpredictor.features.odds import _implied_probs, align_odds_to_test, load_wc_odds


# ── helpers ────────────────────────────────────────────────────────────────


def _xlsx_present() -> bool:
    return (DATA_RAW / "WorldCup_fdco.xlsx").exists()


# ── implied_probs unit tests ────────────────────────────────────────────────


def test_implied_probs_sum_to_one() -> None:
    p_win, p_draw, p_loss = _implied_probs(2.5, 3.2, 3.0)
    assert abs(p_win + p_draw + p_loss - 1.0) < 1e-9


def test_implied_probs_favourite_highest() -> None:
    # Odds of 1.5 means strong favourite; implied prob should be highest
    p_win, p_draw, p_loss = _implied_probs(1.5, 4.0, 6.0)
    assert p_win > p_draw
    assert p_win > p_loss


def test_implied_probs_symmetric_draw() -> None:
    # Equal-odds match: p_win == p_loss
    p_win, p_draw, p_loss = _implied_probs(2.0, 3.5, 2.0)
    assert abs(p_win - p_loss) < 1e-9


def test_implied_probs_no_margin() -> None:
    # Input odds that already sum to 1 in raw prob → normalised unchanged
    # 50/50 match, pure odds: 2.0, draw = 3.33, loss = 2.0 (margins removed by caller)
    p_win, _, p_loss = _implied_probs(2.0, 1e10, 2.0)
    assert abs(p_win - 0.5) < 1e-3
    assert abs(p_loss - 0.5) < 1e-3


# ── sha256 pin ─────────────────────────────────────────────────────────────


@pytest.mark.skipif(not _xlsx_present(), reason="WorldCup_fdco.xlsx not downloaded")
def test_xlsx_sha256_matches_pin() -> None:
    path = DATA_RAW / "WorldCup_fdco.xlsx"
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    assert actual == FDCO_ODDS_SHA256, (
        f"SHA-256 mismatch: expected {FDCO_ODDS_SHA256}, got {actual}. "
        "Update FDCO_ODDS_SHA256 in config.py if the file was intentionally refreshed."
    )


# ── load_wc_odds ────────────────────────────────────────────────────────────


@pytest.mark.skipif(not _xlsx_present(), reason="WorldCup_fdco.xlsx not downloaded")
def test_load_wc_odds_shape() -> None:
    df = load_wc_odds()
    assert set(df["year"].unique()) == {2014, 2018, 2022}
    # 64 matches per WC
    for year in (2014, 2018, 2022):
        assert len(df[df["year"] == year]) == 64, f"Expected 64 rows for WC{year}"


@pytest.mark.skipif(not _xlsx_present(), reason="WorldCup_fdco.xlsx not downloaded")
def test_load_wc_odds_probs_sum_to_one() -> None:
    df = load_wc_odds()
    totals = df["p_win"] + df["p_draw"] + df["p_loss"]
    assert (totals.abs() - 1.0).abs().max() < 1e-6


@pytest.mark.skipif(not _xlsx_present(), reason="WorldCup_fdco.xlsx not downloaded")
def test_load_wc_odds_team_names_canonical() -> None:
    df = load_wc_odds()
    # "USA" → "United States", "Bosnia & Herzegovina" → "Bosnia and Herzegovina"
    assert "USA" not in df["team_a"].values
    assert "USA" not in df["team_b"].values
    assert "Bosnia & Herzegovina" not in df["team_a"].values
    assert "Bosnia & Herzegovina" not in df["team_b"].values


@pytest.mark.skipif(not _xlsx_present(), reason="WorldCup_fdco.xlsx not downloaded")
def test_load_wc_odds_2014_final() -> None:
    df = load_wc_odds()
    row = df[(df["team_a"] == "Germany") & (df["team_b"] == "Argentina") & (df["year"] == 2014)]
    assert len(row) == 1
    r = row.iloc[0]
    # Germany were slight favourites (p_win > p_loss)
    assert r["p_win"] > r["p_loss"]


# ── align_odds_to_test ──────────────────────────────────────────────────────


@pytest.mark.skipif(not _xlsx_present(), reason="WorldCup_fdco.xlsx not downloaded")
def test_align_odds_full_coverage() -> None:
    from wcpredictor.data.load_matches import load_matches
    from wcpredictor.features.elo import compute_elo
    from wcpredictor.features.form import compute_form

    matches = load_matches()
    elo_all, _ = compute_elo(matches)
    form_all, _ = compute_form(matches)
    elo_all = elo_all.merge(form_all, on="match_id", how="left")

    odds_df = load_wc_odds()
    wc_start = pd.Timestamp("2014-06-12")
    wc_matches = matches[
        (matches["date"] >= wc_start) & matches["is_world_cup"] & (matches["date"].dt.year == 2014)
    ]
    test_elo = elo_all[
        (elo_all["date"] >= wc_start) & (elo_all["date"].dt.year == 2014)
    ]
    test_elo = test_elo[test_elo["match_id"].isin(wc_matches["match_id"])]

    aligned = align_odds_to_test(odds_df, 2014, test_elo)
    assert aligned is not None
    assert len(aligned) == len(test_elo)
    # Probabilities should sum to ~1
    for row in aligned:
        assert abs(sum(row) - 1.0) < 1e-6


@pytest.mark.skipif(not _xlsx_present(), reason="WorldCup_fdco.xlsx not downloaded")
def test_align_odds_returns_none_for_2010() -> None:
    """2010 has no odds in the source file → None."""
    odds_df = load_wc_odds()
    # Create a dummy test_elo for 2010 (actual content doesn't matter)
    dummy = pd.DataFrame({"team_a": ["Brazil"], "team_b": ["South Africa"]})
    result = align_odds_to_test(odds_df, 2010, dummy)
    assert result is None
