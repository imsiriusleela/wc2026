"""Tests for parse_market_offers (Step 1), edge.py evaluate_offers (Step 2),
and 1X2 evaluation plumbing (M1)."""
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
    # event1/book1: h2h(3) + spreads(2) + totals(2) = 7
    # event1/book2: h2h(3) + spreads(2) = 5
    # event1/book3: h2h(3) = 3
    # book4_two_way (no Draw → skipped)
    # event2/book1: totals(2) = 2
    # event3 placeholder → skipped
    # Total spreads+totals+1x2: 7+5+3+2 = 17
    # After dedup: same team+market+line+side+bookmaker; no exact dups here
    assert len(df) == 17


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


# ─── 1X2 evaluation (M1) ────────────────────────────────────────────────────

def test_parse_market_offers_1x2_columns(fixture_data):
    from wcpredictor.data.download_odds_api import parse_market_offers
    df = parse_market_offers(fixture_data)
    x2_rows = df[df["market"] == "1x2"]
    assert not x2_rows.empty
    assert set(x2_rows["side"].unique()).issuperset({"home", "draw", "away"})


def test_parse_market_offers_1x2_prices_match_fixture(fixture_data):
    """event1/book1 h2h: USA@1.80, Draw@3.50, BiH@4.20."""
    from wcpredictor.data.download_odds_api import parse_market_offers
    from wcpredictor.data.normalize_teams import canonical
    df = parse_market_offers(fixture_data)
    usa = canonical("USA")
    x2 = df[(df["market"] == "1x2") & (df["bookmaker"] == "book1") & (df["team_a"] == usa)]
    home_row = x2[x2["side"] == "home"]
    draw_row = x2[x2["side"] == "draw"]
    away_row = x2[x2["side"] == "away"]
    assert len(home_row) == 1 and abs(home_row.iloc[0]["price"] - 1.80) < 1e-9
    assert len(draw_row) == 1 and abs(draw_row.iloc[0]["price"] - 3.50) < 1e-9
    assert len(away_row) == 1 and abs(away_row.iloc[0]["price"] - 4.20) < 1e-9


def test_parse_market_offers_two_way_h2h_skipped(fixture_data):
    """book4_two_way has no Draw → should be skipped in 1x2 parsing."""
    from wcpredictor.data.download_odds_api import parse_market_offers
    df = parse_market_offers(fixture_data)
    book4 = df[df["bookmaker"] == "book4_two_way"]
    assert book4.empty, "Two-way h2h (no Draw) must be skipped"


def test_outcome_prob_sums_to_one():
    from wcpredictor.markets.edge import outcome_prob
    mat = _tiny_matrix()
    ph = outcome_prob(mat, "home")
    pd_ = outcome_prob(mat, "draw")
    pa = outcome_prob(mat, "away")
    assert abs(ph + pd_ + pa - 1.0) < 1e-9


def test_outcome_prob_values():
    """Tiny matrix: home P(1-0,2-0)=0.5; draw P(0-0,1-1)=0.3; away P(0-1)=0.2."""
    from wcpredictor.markets.edge import outcome_prob
    mat = _tiny_matrix()
    assert abs(outcome_prob(mat, "home") - 0.5) < 1e-9
    assert abs(outcome_prob(mat, "draw") - 0.3) < 1e-9
    assert abs(outcome_prob(mat, "away") - 0.2) < 1e-9


def test_evaluate_offers_1x2_fair_odds():
    """1X2 EV = 0 at fair odds (= 1/p_outcome)."""
    from wcpredictor.markets.edge import evaluate_offers, outcome_prob
    mat = _tiny_matrix()
    p_home = outcome_prob(mat, "home")  # 0.5
    fair = round(1.0 / p_home, 4)
    offers = [{"market": "1x2", "line": 0.0, "side": "home", "price": fair, "bookmaker": "bk"}]
    r = evaluate_offers(mat, offers)[0]
    assert abs(r["ev"]) < 1e-4
    assert abs(r["fair_odds"] - fair) < 1e-3


def test_evaluate_offers_1x2_no_push_fields():
    """1X2 must have p_push=0, p_half_win=0, p_half_loss=0."""
    from wcpredictor.markets.edge import evaluate_offers
    mat = _tiny_matrix()
    offers = [{"market": "1x2", "line": 0.0, "side": "draw", "price": 3.0, "bookmaker": "bk"}]
    r = evaluate_offers(mat, offers)[0]
    assert r["p_push"] == 0.0
    assert r["p_half_win"] == 0.0
    assert r["p_half_loss"] == 0.0


def test_1x2_flip_symmetry():
    """Reversed lookup for 1x2 (home↔away swap) gives same EV from the reversed match perspective."""
    from wcpredictor.markets.edge import evaluate_offers
    mat_fwd = _tiny_matrix()
    n = len(mat_fwd)
    mat_rev = [[mat_fwd[j][i] for j in range(n)] for i in range(n)]

    fwd = [{"market": "1x2", "line": 0.0, "side": "home", "price": 2.0, "bookmaker": "bk"}]
    # After _load_offers_lookup reversal: home→away
    rev = [{"market": "1x2", "line": 0.0, "side": "away", "price": 2.0, "bookmaker": "bk"}]

    fwd_ev = evaluate_offers(mat_fwd, fwd)[0]["ev"]
    rev_ev = evaluate_offers(mat_rev, rev)[0]["ev"]
    assert abs(fwd_ev - rev_ev) < 1e-6


def test_consensus_fair_1x2():
    """consensus_fair for a 3-way market: median de-margined prob."""
    from wcpredictor.markets.edge import consensus_fair
    offers = [
        {"market": "1x2", "side": "home", "price": 1.80, "bookmaker": "bk1"},
        {"market": "1x2", "side": "draw", "price": 3.50, "bookmaker": "bk1"},
        {"market": "1x2", "side": "away", "price": 4.20, "bookmaker": "bk1"},
        {"market": "1x2", "side": "home", "price": 1.90, "bookmaker": "bk2"},
        {"market": "1x2", "side": "draw", "price": 3.40, "bookmaker": "bk2"},
        {"market": "1x2", "side": "away", "price": 4.00, "bookmaker": "bk2"},
    ]
    fair = consensus_fair(offers, "1x2", None, "home")
    assert fair is not None
    # fair_home should be > 1.80 (raw price) and < 2.5 (rough ballpark)
    assert 1.5 < fair < 2.5


def test_consensus_fair_ah():
    """consensus_fair for a two-way AH market."""
    from wcpredictor.markets.edge import consensus_fair
    offers = [
        {"market": "ah", "side": "home", "price": 1.95, "line": -0.75, "bookmaker": "bk1"},
        {"market": "ah", "side": "away", "price": 1.90, "line": -0.75, "bookmaker": "bk1"},
        {"market": "ah", "side": "home", "price": 1.92, "line": -0.75, "bookmaker": "bk2"},
        {"market": "ah", "side": "away", "price": 1.93, "line": -0.75, "bookmaker": "bk2"},
    ]
    fair = consensus_fair(offers, "ah", -0.75, "home")
    assert fair is not None
    assert 1.85 < fair < 2.05  # ballpark near 1.92


def test_consensus_fair_returns_none_on_empty():
    from wcpredictor.markets.edge import consensus_fair
    assert consensus_fair([], "1x2", None, "home") is None


def test_consensus_fair_ignores_wrong_line():
    """consensus_fair for AH must ignore offers at a different line."""
    from wcpredictor.markets.edge import consensus_fair
    offers = [
        {"market": "ah", "side": "home", "price": 1.95, "line": -0.5, "bookmaker": "bk"},
        {"market": "ah", "side": "away", "price": 1.90, "line": -0.5, "bookmaker": "bk"},
    ]
    # Request for -1.0 line → no matching offers → None
    assert consensus_fair(offers, "ah", -1.0, "home") is None


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
