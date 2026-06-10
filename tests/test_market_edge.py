"""Tests for parse_market_offers (Step 1) and edge.py evaluate_offers (Step 2)."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "odds_api_h2h_sample.json"
LIVE_JSON = Path(__file__).parent.parent / "data" / "raw" / "odds_api_wc2026.json"


# ─── Parser tests ────────────────────────────────────────────────────────────

@pytest.fixture
def fixture_data() -> list[dict]:
    return json.loads(FIXTURE_PATH.read_text())


def test_parse_market_offers_columns(fixture_data):
    from wcpredictor.data.download_odds_api import parse_market_offers
    df = parse_market_offers(fixture_data)
    assert set(["year", "date", "team_a", "team_b", "market", "line", "side", "price", "bookmaker"]).issubset(df.columns)


def test_parse_market_offers_row_count(fixture_data):
    from wcpredictor.data.download_odds_api import parse_market_offers
    df = parse_market_offers(fixture_data)
    # event1/book1: spreads (home+away) + totals (over+under) = 4
    # event1/book2: spreads (home+away) = 2
    # event2 (Brazil/Mexico): totals (over+under) = 2
    # event3 is placeholder → skipped
    assert len(df) == 8


def test_parse_market_offers_both_sides_share_home_line(fixture_data):
    from wcpredictor.data.download_odds_api import parse_market_offers
    df = parse_market_offers(fixture_data)
    ah_rows = df[df["market"] == "ah"]
    # Each (bookmaker, line) pair has both home and away rows with the SAME line value
    for (bk, line), grp in ah_rows.groupby(["bookmaker", "line"]):
        assert set(grp["side"]) == {"home", "away"}, f"bk={bk} line={line} missing a side"
        home_row = grp[grp["side"] == "home"].iloc[0]
        away_row = grp[grp["side"] == "away"].iloc[0]
        assert home_row["line"] == away_row["line"]


def test_parse_market_offers_prices_match_fixture(fixture_data):
    from wcpredictor.data.download_odds_api import parse_market_offers
    df = parse_market_offers(fixture_data)
    # book1 spreads: USA (home) -0.75 @ 1.95, away @ 1.90
    usa_home = df[(df["bookmaker"] == "book1") & (df["market"] == "ah") & (df["side"] == "home")]
    assert len(usa_home) == 1
    assert abs(usa_home.iloc[0]["price"] - 1.95) < 1e-9
    assert abs(usa_home.iloc[0]["line"] - (-0.75)) < 1e-9

    usa_away = df[(df["bookmaker"] == "book1") & (df["market"] == "ah") & (df["side"] == "away")]
    assert abs(usa_away.iloc[0]["price"] - 1.90) < 1e-9


def test_parse_market_offers_placeholder_skipped(fixture_data):
    from wcpredictor.data.download_odds_api import parse_market_offers
    df = parse_market_offers(fixture_data)
    # "Winner Group A" / "Runner-up Group B" must not appear
    for col in ("team_a", "team_b"):
        assert not df[col].str.contains("Winner|Runner", case=False, na=False).any()


def test_parse_market_offers_totals_only_event(fixture_data):
    from wcpredictor.data.download_odds_api import parse_market_offers
    from wcpredictor.data.normalize_teams import canonical
    df = parse_market_offers(fixture_data)
    total_rows = df[df["market"] == "total"]
    # event2 (Brazil/Mexico totals only) + event1/book1 totals = 2+2=4 total rows
    assert len(total_rows) == 4
    # AH rows only from event1 (canonical names may differ from raw)
    usa_canon = canonical("USA")
    brazil_canon = canonical("Brazil")
    assert (total_rows["team_a"].isin([brazil_canon, usa_canon])).all()


def test_parse_market_offers_non_quarter_line_skipped():
    from wcpredictor.data.download_odds_api import parse_market_offers
    data = [{
        "id": "e1",
        "commence_time": "2026-06-15T18:00:00Z",
        "home_team": "Argentina",
        "away_team": "France",
        "bookmakers": [{
            "key": "bk",
            "markets": [{
                "key": "spreads",
                "outcomes": [
                    {"name": "Argentina", "point": -1.3, "price": 1.90},
                    {"name": "France", "point": 1.3, "price": 1.90},
                ]
            }]
        }]
    }]
    df = parse_market_offers(data)
    assert df.empty


def test_parse_market_offers_empty_input():
    from wcpredictor.data.download_odds_api import parse_market_offers
    df = parse_market_offers([])
    assert df.empty


# ─── Settlement orientation ───────────────────────────────────────────────────

def _tiny_matrix():
    """3×3 matrix: P(0-0)=0.1, P(0-1)=0.2, P(1-0)=0.4, P(1-1)=0.2, P(2-0)=0.1.
    Goal diff distribution: {-1: 0.2, 0: 0.3, 1: 0.5}
    """
    m = [[0.0] * 3 for _ in range(3)]
    m[0][0] = 0.1  # 0-0
    m[0][1] = 0.2  # 0-1  diff=-1
    m[1][0] = 0.4  # 1-0  diff=+1
    m[1][1] = 0.2  # 1-1  diff=0
    m[2][0] = 0.1  # 2-0  diff=+2
    return m


def test_ev_per_unit_formula():
    """EV = 0 at fair odds."""
    from wcpredictor.markets.edge import ev_per_unit
    from wcpredictor.markets.asian import asian_handicap
    mat = _tiny_matrix()
    # AH -0.5: home wins when diff > 0.5, i.e., diff >= 1 → p_win = 0.5
    s = asian_handicap(mat, side="home", line=-0.5)
    assert abs(ev_per_unit(s, s["fair_odds"]) - 0.0) < 1e-6


def test_evaluate_offers_ev_property():
    """ev = (price − fair_odds)·(p_win + 0.5·p_half_win) for all offers with finite fair_odds."""
    import math
    from wcpredictor.markets.edge import evaluate_offers
    mat = _tiny_matrix()
    offers = [
        {"market": "ah", "line": -0.5, "side": "home", "price": 2.10, "bookmaker": "bk"},
        {"market": "ah", "line": -0.5, "side": "away", "price": 1.80, "bookmaker": "bk"},
        {"market": "total", "line": 1.5, "side": "over", "price": 1.90, "bookmaker": "bk"},
    ]
    results = evaluate_offers(mat, offers)
    for r in results:
        if not math.isfinite(r["fair_odds"]):
            continue  # (price - inf)*0 is nan; formula undefined, EV still correct via direct calc
        a = r["p_win"] + 0.5 * r["p_half_win"]
        expected_ev = (r["price"] - r["fair_odds"]) * a
        assert abs(r["ev"] - round(expected_ev, 4)) < 1e-4, f"EV mismatch for {r}"


def test_evaluate_offers_sorted_desc():
    from wcpredictor.markets.edge import evaluate_offers
    mat = _tiny_matrix()
    offers = [
        {"market": "ah", "line": -0.5, "side": "home", "price": 1.60, "bookmaker": "bk"},
        {"market": "ah", "line": -0.5, "side": "home", "price": 2.20, "bookmaker": "bk2"},
        {"market": "total", "line": 2.5, "side": "over", "price": 1.90, "bookmaker": "bk"},
    ]
    results = evaluate_offers(mat, offers)
    evs = [r["ev"] for r in results]
    assert evs == sorted(evs, reverse=True)


def test_evaluate_offers_best_offer_is_first():
    from wcpredictor.markets.edge import evaluate_offers
    mat = _tiny_matrix()
    offers = [
        {"market": "ah", "line": -0.5, "side": "home", "price": 2.20, "bookmaker": "bk"},
        {"market": "ah", "line": -0.5, "side": "home", "price": 1.60, "bookmaker": "bk2"},
    ]
    results = evaluate_offers(mat, offers)
    assert results[0]["ev"] >= results[-1]["ev"]


def test_evaluate_offers_flip_symmetry():
    """Reversed lookup entry (line negated, side swapped) on the transposed matrix gives same EV.

    When a user queries (B, A) the prediction matrix is the transpose of the (A, B) matrix
    (home/away labels flip).  The reversed offer {line=-L → +L, side=home → away} evaluated
    against the transposed matrix must match the forward offer evaluated against the original.
    """
    from wcpredictor.markets.edge import evaluate_offers
    mat_fwd = _tiny_matrix()
    n = len(mat_fwd)
    # Transpose: swap home and away (new home is old away)
    mat_rev = [[mat_fwd[j][i] for j in range(n)] for i in range(n)]

    # Forward: home team at -0.75 evaluated against mat_fwd
    forward = [{"market": "ah", "line": -0.75, "side": "home", "price": 1.95, "bookmaker": "bk"}]
    # Reversed (from _load_offers_lookup): line negated → +0.75, side flipped → "away"
    # evaluated against mat_rev (where original away is now home)
    reversed_ = [{"market": "ah", "line": 0.75, "side": "away", "price": 1.95, "bookmaker": "bk"}]

    fwd_ev = evaluate_offers(mat_fwd, forward)[0]["ev"]
    rev_ev = evaluate_offers(mat_rev, reversed_)[0]["ev"]
    assert abs(fwd_ev - rev_ev) < 1e-6, f"fwd_ev={fwd_ev} rev_ev={rev_ev}"


def test_evaluate_offers_p_cover_field():
    from wcpredictor.markets.edge import evaluate_offers
    mat = _tiny_matrix()
    offers = [{"market": "ah", "line": -0.5, "side": "home", "price": 2.0, "bookmaker": "bk"}]
    r = evaluate_offers(mat, offers)[0]
    assert abs(r["p_cover"] - (r["p_win"] + r["p_half_win"])) < 1e-9


# ─── Integration: frozen-state predict ───────────────────────────────────────

@pytest.mark.skipif(not LIVE_JSON.exists(), reason="live odds JSON not present")
def test_frozen_predict_has_offers_when_live_json_present():
    """predict_fixtures result contains markets.offers for 2026 matches when JSON exists."""
    import pandas as pd
    from wcpredictor.predict import _build_frozen_state, _predict_one_frozen

    state = _build_frozen_state("2026-06-11", ["ensemble_mkt"])
    offers_lookup = state.get("offers_lookup", {})
    if not offers_lookup:
        pytest.skip("offers_lookup empty — no spread/totals in cached JSON")

    pair = next(iter(offers_lookup))
    team_a, team_b = pair
    result = _predict_one_frozen(state, "ensemble_mkt", team_a, team_b, neutral=True)
    assert "offers" in result["markets"], "markets.offers missing"
    offers = result["markets"]["offers"]
    assert isinstance(offers, list)
    assert len(offers) > 0
    for o in offers:
        assert math.isfinite(o["ev"]), f"non-finite ev: {o}"
        assert math.isfinite(o["fair_odds"]) or o["fair_odds"] == float("inf")
