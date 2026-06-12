"""Tests for SG Pools CSV ingestion and fetcher (M3)."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest


# ─── add_offer / load_sgpools_offers ─────────────────────────────────────────

def test_add_and_load_offer(tmp_path):
    from wcpredictor.data.sgpools import add_offer, load_sgpools_offers
    csv_path = tmp_path / "sgpools_offers.csv"
    add_offer("Brazil", "Argentina", "2026-06-20", "1x2", 0.0, "home", 1.85, csv_path=csv_path)
    df = load_sgpools_offers(csv_path)
    assert len(df) == 1
    assert df.iloc[0]["market"] == "1x2"
    assert df.iloc[0]["side"] == "home"
    assert abs(df.iloc[0]["price"] - 1.85) < 1e-9


def test_add_offer_ah(tmp_path):
    from wcpredictor.data.sgpools import add_offer, load_sgpools_offers
    csv_path = tmp_path / "sgpools.csv"
    add_offer("Brazil", "Germany", "2026-06-21", "ah", -0.75, "home", 1.90, csv_path=csv_path)
    df = load_sgpools_offers(csv_path)
    assert len(df) == 1
    assert abs(df.iloc[0]["line"] - (-0.75)) < 1e-9


def test_add_offer_total(tmp_path):
    from wcpredictor.data.sgpools import add_offer, load_sgpools_offers
    csv_path = tmp_path / "sgpools.csv"
    add_offer("France", "Spain", "2026-06-22", "total", 2.5, "over", 1.85, csv_path=csv_path)
    df = load_sgpools_offers(csv_path)
    assert df.iloc[0]["market"] == "total"
    assert df.iloc[0]["side"] == "over"


def test_canonical_names_applied(tmp_path):
    from wcpredictor.data.sgpools import add_offer, load_sgpools_offers
    from wcpredictor.data.normalize_teams import canonical
    csv_path = tmp_path / "sgpools.csv"
    # "USA" should be canonicalised to "United States"
    add_offer("USA", "Mexico", "2026-06-12", "1x2", 0.0, "home", 2.0, csv_path=csv_path)
    df = load_sgpools_offers(csv_path)
    assert df.iloc[0]["team_a"] == canonical("USA")


def test_invalid_market_raises(tmp_path):
    from wcpredictor.data.sgpools import add_offer
    csv_path = tmp_path / "sgpools.csv"
    with pytest.raises(ValueError, match="market"):
        add_offer("Brazil", "Germany", "2026-06-21", "outright", 0.0, "home", 2.0, csv_path=csv_path)


def test_invalid_side_raises(tmp_path):
    from wcpredictor.data.sgpools import add_offer
    csv_path = tmp_path / "sgpools.csv"
    with pytest.raises(ValueError, match="side"):
        add_offer("Brazil", "Germany", "2026-06-21", "ah", -0.5, "push", 1.90, csv_path=csv_path)


def test_price_below_one_raises(tmp_path):
    from wcpredictor.data.sgpools import add_offer
    csv_path = tmp_path / "sgpools.csv"
    with pytest.raises(ValueError, match="price"):
        add_offer("Brazil", "Germany", "2026-06-21", "1x2", 0.0, "home", 0.90, csv_path=csv_path)


def test_non_quarter_line_raises(tmp_path):
    from wcpredictor.data.sgpools import add_offer
    csv_path = tmp_path / "sgpools.csv"
    with pytest.raises(ValueError, match="0.25"):
        add_offer("Brazil", "Germany", "2026-06-21", "ah", -0.3, "home", 1.90, csv_path=csv_path)


def test_manual_beats_fetched_on_dedup(tmp_path):
    """Manual source must win over fetched for same market/line/side."""
    from wcpredictor.data.sgpools import load_sgpools_offers
    csv_path = tmp_path / "sgpools.csv"
    rows = [
        {"entered_at": "2026-06-12T10:00:00Z", "date": "2026-06-20",
         "team_a": "Brazil", "team_b": "Germany",
         "market": "1x2", "line": 0.0, "side": "home", "price": 1.85, "source": "fetched"},
        {"entered_at": "2026-06-12T11:00:00Z", "date": "2026-06-20",
         "team_a": "Brazil", "team_b": "Germany",
         "market": "1x2", "line": 0.0, "side": "home", "price": 1.90, "source": "manual"},
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    df = load_sgpools_offers(csv_path)
    assert len(df) == 1
    assert abs(df.iloc[0]["price"] - 1.90) < 1e-9
    assert df.iloc[0]["source"] == "manual"


def test_invalid_price_rows_dropped(tmp_path):
    """Rows with price <= 1.0 must be dropped silently."""
    from wcpredictor.data.sgpools import load_sgpools_offers
    csv_path = tmp_path / "sgpools.csv"
    rows = [
        {"entered_at": "2026-06-12T10:00:00Z", "date": "2026-06-20",
         "team_a": "Brazil", "team_b": "Germany",
         "market": "1x2", "line": 0.0, "side": "home", "price": 0.95, "source": "manual"},
        {"entered_at": "2026-06-12T10:00:00Z", "date": "2026-06-20",
         "team_a": "Brazil", "team_b": "Germany",
         "market": "1x2", "line": 0.0, "side": "draw", "price": 3.50, "source": "manual"},
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    df = load_sgpools_offers(csv_path)
    assert len(df) == 1
    assert df.iloc[0]["side"] == "draw"


def test_load_empty_file_returns_empty_df(tmp_path):
    from wcpredictor.data.sgpools import load_sgpools_offers
    csv_path = tmp_path / "empty.csv"
    csv_path.write_text("entered_at,date,team_a,team_b,market,line,side,price,source\n")
    df = load_sgpools_offers(csv_path)
    assert df.empty


def test_load_nonexistent_returns_empty_df(tmp_path):
    from wcpredictor.data.sgpools import load_sgpools_offers
    df = load_sgpools_offers(tmp_path / "nonexistent.csv")
    assert df.empty


# ─── Fetcher parse (no network) ───────────────────────────────────────────────

def test_parse_sgpools_response_empty():
    from wcpredictor.data.sgpools import _parse_sgpools_response
    assert _parse_sgpools_response([]) == []
    assert _parse_sgpools_response({}) == []


def test_parse_sgpools_response_1x2():
    from wcpredictor.data.sgpools import _parse_sgpools_response
    event = {
        "homeTeam": "Brazil",
        "awayTeam": "Germany",
        "matchDate": "2026-06-20",
        "homeOdds": 1.85,
        "drawOdds": 3.40,
        "awayOdds": 4.20,
    }
    rows = _parse_sgpools_response([event])
    sides = {r["side"] for r in rows if r["market"] == "1x2"}
    assert sides == {"home", "draw", "away"}
    home_row = next(r for r in rows if r["side"] == "home")
    assert abs(home_row["price"] - 1.85) < 1e-9
    assert home_row["source"] == "fetched"


def test_parse_sgpools_response_ah():
    from wcpredictor.data.sgpools import _parse_sgpools_response
    event = {
        "homeTeam": "France",
        "awayTeam": "Argentina",
        "matchDate": "2026-06-22",
        "ahLine": -0.5,
        "ahHomeOdds": 1.92,
        "ahAwayOdds": 1.88,
    }
    rows = _parse_sgpools_response([event])
    ah = [r for r in rows if r["market"] == "ah"]
    assert len(ah) == 2
    assert {r["side"] for r in ah} == {"home", "away"}


def test_parse_sgpools_response_invalid_team_skipped():
    from wcpredictor.data.sgpools import _parse_sgpools_response
    event = {"homeTeam": "", "awayTeam": "", "matchDate": "2026-06-20",
             "homeOdds": 1.80, "drawOdds": 3.50, "awayOdds": 4.00}
    rows = _parse_sgpools_response([event])
    assert rows == []
