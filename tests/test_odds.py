"""Tests for features/odds.py, data/download_odds.py, and data/download_wc2010_odds.py."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest

import math

from wcpredictor.config import DATA_RAW, FDCO_ODDS_SHA256, WC2010_ODDS_CSV_SHA256
from wcpredictor.features.odds import (
    _implied_probs,
    align_odds_to_test,
    load_wc_odds,
    merge_odds_features,
)


# ── helpers ────────────────────────────────────────────────────────────────


def _xlsx_present() -> bool:
    return (DATA_RAW / "WorldCup_fdco.xlsx").exists()


def _wc2010_csv_present() -> bool:
    return (DATA_RAW / "wc2010_odds.csv").exists()


# ── merge_odds_features unit tests ─────────────────────────────────────────


def _make_odds_df() -> pd.DataFrame:
    return pd.DataFrame({
        "year": [2014, 2014, 2018],
        "date": pd.to_datetime(["2014-06-12", "2014-06-13", "2018-06-15"]),
        "team_a": ["Germany", "Brazil", "France"],
        "team_b": ["Portugal", "Croatia", "Australia"],
        "p_win": [0.60, 0.70, 0.65],
        "p_draw": [0.25, 0.18, 0.22],
        "p_loss": [0.15, 0.12, 0.13],
    })


def _make_features_df() -> pd.DataFrame:
    return pd.DataFrame({
        "date": pd.to_datetime(["2014-06-12", "2014-06-13", "2018-06-15", "2015-08-01"]),
        "team_a": ["Germany", "Brazil", "France", "Spain"],
        "team_b": ["Portugal", "Croatia", "Australia", "England"],
        "elo_diff_adj": [50.0, 80.0, 60.0, 20.0],
        "neutral": [True, True, True, False],
    })


def test_merge_odds_features_matched_rows() -> None:
    feats = _make_features_df()
    odds = _make_odds_df()
    result = merge_odds_features(feats, odds)
    # First three rows have odds
    for i in range(3):
        assert result["has_odds"].iloc[i] == 1.0
        assert not math.isnan(result["odds_p_win"].iloc[i])
        assert abs(result["odds_p_win"].iloc[i] + result["odds_p_draw"].iloc[i] + result["odds_p_loss"].iloc[i] - 1.0) < 1e-9


def test_merge_odds_features_unmatched_row_nan() -> None:
    feats = _make_features_df()
    odds = _make_odds_df()
    result = merge_odds_features(feats, odds)
    # Last row (Spain vs England, 2015) has no odds
    assert result["has_odds"].iloc[3] == 0.0
    assert math.isnan(result["odds_p_win"].iloc[3])
    assert math.isnan(result["odds_p_draw"].iloc[3])
    assert math.isnan(result["odds_p_loss"].iloc[3])


def test_merge_odds_features_symmetric_wl_swap() -> None:
    """(Portugal, Germany, 2014) should get p_win = original p_loss and vice versa."""
    feats = pd.DataFrame({
        "date": pd.to_datetime(["2014-06-12"]),
        "team_a": ["Portugal"],
        "team_b": ["Germany"],
        "elo_diff_adj": [-50.0],
        "neutral": [True],
    })
    odds = _make_odds_df()
    result = merge_odds_features(feats, odds)
    assert result["has_odds"].iloc[0] == 1.0
    # Germany (team_a in odds) had p_win=0.60 → Portugal (reversed) gets p_win=0.15
    assert abs(result["odds_p_win"].iloc[0] - 0.15) < 1e-9
    assert abs(result["odds_p_loss"].iloc[0] - 0.60) < 1e-9
    assert abs(result["odds_p_draw"].iloc[0] - 0.25) < 1e-9


def test_merge_odds_features_does_not_mutate_input() -> None:
    feats = _make_features_df()
    original_cols = list(feats.columns)
    merge_odds_features(feats, _make_odds_df())
    assert list(feats.columns) == original_cols


def test_merge_odds_features_empty_odds_all_nan() -> None:
    feats = _make_features_df()
    empty_odds = pd.DataFrame(columns=["year", "date", "team_a", "team_b", "p_win", "p_draw", "p_loss"])
    result = merge_odds_features(feats, empty_odds)
    assert all(result["has_odds"] == 0.0)
    assert all(math.isnan(v) for v in result["odds_p_win"])


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
    expected_years = {2014, 2018, 2022}
    if _wc2010_csv_present():
        expected_years.add(2010)
    assert expected_years.issubset(set(df["year"].unique()))
    # 64 matches per WC
    for year in (2014, 2018, 2022):
        assert len(df[df["year"] == year]) == 64, f"Expected 64 rows for WC{year}"
    if _wc2010_csv_present():
        assert len(df[df["year"] == 2010]) == 64, "Expected 64 rows for WC2010"


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


@pytest.mark.skipif(
    _xlsx_present() and _wc2010_csv_present(),
    reason="WC2010 odds are now present — use test_align_odds_2010_present instead",
)
@pytest.mark.skipif(not _xlsx_present(), reason="WorldCup_fdco.xlsx not downloaded")
def test_align_odds_returns_none_for_2010_when_csv_absent() -> None:
    """When wc2010_odds.csv is absent, 2010 fold returns None."""
    odds_df = load_wc_odds()
    dummy = pd.DataFrame({"team_a": ["Brazil"], "team_b": ["South Africa"]})
    result = align_odds_to_test(odds_df, 2010, dummy)
    assert result is None


# ── WC2010 betexplorer CSV tests ────────────────────────────────────────────


@pytest.mark.skipif(not _wc2010_csv_present(), reason="wc2010_odds.csv not generated")
def test_wc2010_csv_row_count() -> None:
    df = pd.read_csv(DATA_RAW / "wc2010_odds.csv")
    assert len(df) == 64, f"Expected 64 WC2010 matches, got {len(df)}"


@pytest.mark.skipif(not _wc2010_csv_present(), reason="wc2010_odds.csv not generated")
def test_wc2010_csv_sha256_matches_pin() -> None:
    path = DATA_RAW / "wc2010_odds.csv"
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    assert actual == WC2010_ODDS_CSV_SHA256, (
        f"SHA-256 mismatch: expected {WC2010_ODDS_CSV_SHA256}, got {actual}. "
        "Update WC2010_ODDS_CSV_SHA256 in config.py if snapshots were refreshed."
    )


@pytest.mark.skipif(not _wc2010_csv_present(), reason="wc2010_odds.csv not generated")
def test_wc2010_csv_prob_sums_to_one() -> None:
    df = pd.read_csv(DATA_RAW / "wc2010_odds.csv")
    for _, row in df.iterrows():
        pw, pd_, pl = _implied_probs(row["odds_h"], row["odds_d"], row["odds_a"])
        assert abs(pw + pd_ + pl - 1.0) < 1e-9


@pytest.mark.skipif(not _wc2010_csv_present(), reason="wc2010_odds.csv not generated")
def test_wc2010_csv_canonical_names() -> None:
    df = pd.read_csv(DATA_RAW / "wc2010_odds.csv")
    # "USA" should be mapped to "United States" by canonical()
    assert "USA" not in df["home"].values
    assert "USA" not in df["away"].values
    # "United States" should be present
    assert "United States" in df["home"].values or "United States" in df["away"].values


@pytest.mark.skipif(not _wc2010_csv_present(), reason="wc2010_odds.csv not generated")
def test_wc2010_load_wc_odds_includes_2010(tmp_path) -> None:
    if not _xlsx_present():
        pytest.skip("WorldCup_fdco.xlsx not downloaded")
    df = load_wc_odds()
    assert 2010 in df["year"].values
    assert len(df[df["year"] == 2010]) == 64


@pytest.mark.skipif(not _wc2010_csv_present(), reason="wc2010_odds.csv not generated")
def test_wc2010_symmetric_alignment() -> None:
    """align_odds_to_test registers both (a,b) and (b,a); W/L swap on reversal."""
    if not _xlsx_present():
        pytest.skip("WorldCup_fdco.xlsx not downloaded")
    odds_df = load_wc_odds()
    yr = odds_df[odds_df["year"] == 2010].iloc[0]
    ta, tb = str(yr.team_a), str(yr.team_b)

    # Forward: (a, b) → [p_win, p_draw, p_loss]
    fwd = pd.DataFrame({"team_a": [ta], "team_b": [tb]})
    res_fwd = align_odds_to_test(odds_df, 2010, fwd)
    assert res_fwd is not None
    assert abs(sum(res_fwd[0]) - 1.0) < 1e-6

    # Reversed: (b, a) → [p_loss, p_draw, p_win]  (W/L swap)
    rev = pd.DataFrame({"team_a": [tb], "team_b": [ta]})
    res_rev = align_odds_to_test(odds_df, 2010, rev)
    assert res_rev is not None
    assert abs(res_rev[0][0] - res_fwd[0][2]) < 1e-9  # p_win reversed = p_loss forward
    assert abs(res_rev[0][1] - res_fwd[0][1]) < 1e-9  # p_draw unchanged
    assert abs(res_rev[0][2] - res_fwd[0][0]) < 1e-9  # p_loss reversed = p_win forward


@pytest.mark.skipif(not _xlsx_present(), reason="WorldCup_fdco.xlsx not downloaded")
def test_align_odds_2010_graceful_skip_when_csv_absent(tmp_path) -> None:
    """When wc2010_odds.csv is absent, align_odds_to_test returns None for year=2010."""
    # Build odds_df without 2010 rows (simulate absent CSV)
    full_df = load_wc_odds()
    df_no_2010 = full_df[full_df["year"] != 2010].copy()
    dummy = pd.DataFrame({"team_a": ["Brazil"], "team_b": ["South Africa"]})
    result = align_odds_to_test(df_no_2010, 2010, dummy)
    assert result is None
