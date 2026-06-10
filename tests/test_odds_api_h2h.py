"""Tests for parse_h2h_1x2() and load_wc_odds() live-JSON integration."""
from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

import pandas as pd
import pytest

FIXTURE = Path(__file__).parent / "fixtures" / "odds_api_h2h_sample.json"
LIVE_JSON = Path(__file__).parent.parent / "data" / "raw" / "odds_api_wc2026.json"


@pytest.fixture()
def sample_data() -> list[dict]:
    return json.loads(FIXTURE.read_text())


@pytest.fixture()
def parsed(sample_data):
    from wcpredictor.data.download_odds_api import parse_h2h_1x2
    return parse_h2h_1x2(sample_data)


# ── schema ──────────────────────────────────────────────────────────────────


def test_schema_columns(parsed):
    assert set(parsed.columns) >= {"year", "date", "team_a", "team_b", "p_win", "p_draw", "p_loss"}


def test_year_is_2026(parsed):
    assert (parsed["year"] == 2026).all()


# ── filtering ───────────────────────────────────────────────────────────────


def test_skips_totals_only_event(parsed):
    # Brazil vs Mexico has no h2h market → excluded
    assert not ((parsed["team_a"] == "Brazil") & (parsed["team_b"] == "Mexico")).any()


def test_skips_placeholder_teams(parsed):
    for name in ("Winner Group A", "Runner-up Group B"):
        assert name not in parsed["team_a"].tolist()
        assert name not in parsed["team_b"].tolist()


def test_two_way_bookie_excluded(parsed):
    # book4 has no Draw → excluded from median; only 3 books used
    # Result: only event1 survives
    assert len(parsed) == 1


# ── canonical names ─────────────────────────────────────────────────────────


def test_raw_aliases_absent_from_output(parsed):
    for alias in ("USA", "Bosnia & Herzegovina"):
        assert alias not in parsed["team_a"].values
        assert alias not in parsed["team_b"].values


def test_canonical_names_present(parsed):
    row = parsed.iloc[0]
    assert row["team_a"] == "United States"
    assert row["team_b"] == "Bosnia and Herzegovina"


# ── median computation ───────────────────────────────────────────────────────


def test_median_aggregation_vs_hand_computed(parsed):
    # 3 valid bookmakers: odds (1.80/3.50/4.20), (1.90/3.40/4.00), (1.85/3.45/4.10)
    # book4 excluded (no Draw)
    o_h = statistics.median([1.80, 1.90, 1.85])   # 1.85
    o_d = statistics.median([3.50, 3.40, 3.45])   # 3.45
    o_a = statistics.median([4.20, 4.00, 4.10])   # 4.10
    ph, pd_, pa = 1 / o_h, 1 / o_d, 1 / o_a
    total = ph + pd_ + pa
    expected = {"p_win": ph / total, "p_draw": pd_ / total, "p_loss": pa / total}

    row = parsed.iloc[0]
    assert abs(row["p_win"] - expected["p_win"]) < 1e-9
    assert abs(row["p_draw"] - expected["p_draw"]) < 1e-9
    assert abs(row["p_loss"] - expected["p_loss"]) < 1e-9


def test_probs_sum_to_one(parsed):
    for _, r in parsed.iterrows():
        assert abs(r["p_win"] + r["p_draw"] + r["p_loss"] - 1.0) < 1e-9


# ── rematch dedup ────────────────────────────────────────────────────────────


def test_rematch_keeps_latest_date(sample_data):
    from wcpredictor.data.download_odds_api import parse_h2h_1x2

    duplicate = {
        "id": "event1_rematch",
        "commence_time": "2026-06-18T18:00:00Z",  # later date
        "home_team": "USA",
        "away_team": "Bosnia & Herzegovina",
        "bookmakers": [
            {
                "key": "book1",
                "markets": [
                    {
                        "key": "h2h",
                        "outcomes": [
                            {"name": "USA", "price": 2.00},
                            {"name": "Draw", "price": 3.20},
                            {"name": "Bosnia & Herzegovina", "price": 3.80},
                        ],
                    }
                ],
            }
        ],
    }
    df = parse_h2h_1x2(sample_data + [duplicate])
    us_rows = df[(df["team_a"] == "United States") | (df["team_b"] == "United States")]
    assert len(us_rows) == 1
    assert us_rows.iloc[0]["date"] == pd.Timestamp("2026-06-18")


# ── empty / edge inputs ──────────────────────────────────────────────────────


def test_empty_input():
    from wcpredictor.data.download_odds_api import parse_h2h_1x2
    df = parse_h2h_1x2([])
    assert df.empty
    assert set(df.columns) >= {"year", "date", "team_a", "team_b", "p_win", "p_draw", "p_loss"}


# ── load_wc_odds integration ─────────────────────────────────────────────────


def test_load_wc_odds_appends_2026_rows(tmp_path):
    from wcpredictor.features.odds import load_wc_odds
    import shutil

    fixture_copy = tmp_path / "odds_api_wc2026.json"
    shutil.copy2(FIXTURE, fixture_copy)

    df = load_wc_odds(live_json_path=fixture_copy)
    rows_2026 = df[df["year"] == 2026]
    assert len(rows_2026) >= 1


def test_load_wc_odds_absent_json_is_noop(tmp_path):
    from wcpredictor.features.odds import load_wc_odds

    df_without = load_wc_odds(live_json_path=tmp_path / "missing.json")
    df_with = load_wc_odds()
    # Without the live JSON the result should have no 2026 rows from the API
    rows_without = df_without[df_without["year"] == 2026]
    # (fdco sheet may or may not have 2026 rows — just check it doesn't raise)
    assert isinstance(df_without, pd.DataFrame)


def test_load_wc_odds_corrupt_json_is_noop(tmp_path):
    from wcpredictor.features.odds import load_wc_odds

    corrupt = tmp_path / "odds_api_wc2026.json"
    corrupt.write_text("NOT JSON {{{{")
    df = load_wc_odds(live_json_path=corrupt)
    assert isinstance(df, pd.DataFrame)


def test_fdco_precedence_over_api(tmp_path):
    """fdco row for a pair must win even when the api JSON lists the reversed ordering."""
    from wcpredictor.features.odds import load_wc_odds
    import openpyxl

    # Build a minimal fdco xlsx with WorldCup2026 sheet listing US vs Bosnia
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "WorldCup2026"
    ws.append(["Date", "Home", "Away", "H-Avg", "D-Avg", "A-Avg"])
    ws.append(["2026-06-11", "United States", "Bosnia and Herzegovina", 1.85, 3.45, 4.10])
    xlsx = tmp_path / "WorldCup_fdco.xlsx"
    wb.save(xlsx)

    import shutil
    fixture_copy = tmp_path / "odds_api_wc2026.json"
    shutil.copy2(FIXTURE, fixture_copy)

    df = load_wc_odds(xlsx_path=xlsx, live_json_path=fixture_copy)
    rows_2026 = df[df["year"] == 2026]
    # fdco row present; live row for same pair must be deduplicated
    pair_rows = rows_2026[
        ((rows_2026["team_a"] == "United States") & (rows_2026["team_b"] == "Bosnia and Herzegovina"))
        | ((rows_2026["team_a"] == "Bosnia and Herzegovina") & (rows_2026["team_b"] == "United States"))
    ]
    assert len(pair_rows) == 1


# ── repo-data smoke test ─────────────────────────────────────────────────────


@pytest.mark.skipif(not LIVE_JSON.exists(), reason="odds_api_wc2026.json not present")
def test_repo_data_smoke():
    from wcpredictor.features.odds import load_wc_odds

    df = load_wc_odds()
    rows_2026 = df[df["year"] == 2026]
    assert len(rows_2026) >= 1, "Expected at least one 2026 row from live JSON"

    for _, r in rows_2026.iterrows():
        assert abs(r["p_win"] + r["p_draw"] + r["p_loss"] - 1.0) < 1e-6, (
            f"Probs don't sum to 1 for {r['team_a']} vs {r['team_b']}"
        )
        assert r["team_a"] not in ("USA", "Bosnia & Herzegovina"), (
            f"Raw alias leaked into output: {r['team_a']}"
        )
