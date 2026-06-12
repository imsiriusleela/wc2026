"""Fetch live WC2026 Asian-handicap, O/U, and 1X2 odds from the-odds-api.com.

Requires an API key in the environment variable ODDS_API_KEY.
Gracefully returns an empty DataFrame when the key is absent or the
API returns no data — the rest of the pipeline auto-degrades to
model-only predictions in that case.

Usage:
    export ODDS_API_KEY=your_key_here
    uv run python -m wcpredictor.data.download_odds_api

Fetches:
    soccer_fifa_world_cup, markets=h2h,spreads,totals
    Saves raw JSON to data/raw/odds_api_wc2026.json (overwritten each run;
    prior copy archived to data/raw/odds_api_snapshots/ before overwrite).
    Returns a tidy DataFrame with schema matching wc_ah_odds.csv.
    1X2 probabilities are available via parse_h2h_1x2().
"""
from __future__ import annotations

import datetime
import json
import os
import shutil
import statistics
import warnings
from pathlib import Path

import pandas as pd

from wcpredictor.config import DATA_RAW
from wcpredictor.data.normalize_teams import canonical

_ODDS_API_BASE = "https://api.the-odds-api.com/v4/sports"
_SPORT = "soccer_fifa_world_cup"
_REGIONS = "eu"  # EU decimal odds
_MARKETS = "h2h,spreads,totals"
_JSON_DEST = DATA_RAW / "odds_api_wc2026.json"


def fetch_odds_api(force: bool = False) -> pd.DataFrame:
    """Fetch live WC2026 AH + O/U odds.  Returns empty DataFrame if no key.

    Parameters
    ----------
    force : if False (default), return cached JSON if it exists and is recent.
            When True, archives the existing JSON before overwriting.

    Returns
    -------
    DataFrame with columns: year, date, home, away, ah_line, ah_home_odds,
                            ah_away_odds, ou_line, over_odds, under_odds.
    """
    api_key = os.environ.get("ODDS_API_KEY", "").strip()
    if not api_key:
        return pd.DataFrame(columns=[
            "year", "date", "home", "away",
            "ah_line", "ah_home_odds", "ah_away_odds",
            "ou_line", "over_odds", "under_odds",
        ])

    import urllib.request

    url = (
        f"{_ODDS_API_BASE}/{_SPORT}/odds/"
        f"?apiKey={api_key}&regions={_REGIONS}&markets={_MARKETS}&oddsFormat=decimal"
    )

    DATA_RAW.mkdir(parents=True, exist_ok=True)

    if not force and _JSON_DEST.exists():
        raw = json.loads(_JSON_DEST.read_text())
    else:
        # Archive existing snapshot before overwriting (feed shrinks as matches are played)
        if _JSON_DEST.exists():
            snapshots_dir = DATA_RAW / "odds_api_snapshots"
            snapshots_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
            shutil.copy2(_JSON_DEST, snapshots_dir / f"odds_api_wc2026_{ts}.json")

        try:
            with urllib.request.urlopen(url, timeout=30) as resp:  # noqa: S310
                raw = json.loads(resp.read().decode())
            _JSON_DEST.write_text(json.dumps(raw, indent=2))
        except Exception as exc:
            if _JSON_DEST.exists():
                warnings.warn(
                    f"odds-api fetch failed ({exc}); returning cached {_JSON_DEST}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                raw = json.loads(_JSON_DEST.read_text())
            else:
                raise

    return _parse_odds_api_json(raw)


def _is_tbd(name: str) -> bool:
    """Return True for knockout-stage placeholder names like "Winner Group A"."""
    lower = name.lower()
    return lower.startswith(("winner ", "loser ", "runner-up ", "runner "))


def parse_h2h_1x2(data: list[dict]) -> pd.DataFrame:
    """Parse the-odds-api JSON → margin-stripped 1X2 probabilities for WC 2026.

    Uses median decimal odds across accepted bookmakers, then 3-way normalization.
    Bookmakers are accepted only if they quote home, away, AND Draw, all > 1.0.
    Events with placeholder team names (e.g., "Winner Group A") are skipped.

    Returns DataFrame with columns:
        year, date, team_a, team_b, p_win, p_draw, p_loss
    where team_a is the home team (canonical name).
    """
    _EMPTY_COLS = ["year", "date", "team_a", "team_b", "p_win", "p_draw", "p_loss"]
    rows: list[dict] = []

    for event in data:
        home_raw = event.get("home_team", "")
        away_raw = event.get("away_team", "")
        ta = canonical(home_raw)
        tb = canonical(away_raw)
        # Skip empty or TBD/placeholder team names (knockout-stage "Winner Group X")
        if not ta or not tb or _is_tbd(ta) or _is_tbd(tb):
            continue

        home_odds_list: list[float] = []
        draw_odds_list: list[float] = []
        away_odds_list: list[float] = []

        for bookie in event.get("bookmakers", []):
            for market in bookie.get("markets", []):
                if market.get("key") != "h2h":
                    continue
                price_map = {o["name"]: o["price"] for o in market.get("outcomes", [])}
                ho = price_map.get(home_raw)
                do_ = price_map.get("Draw")
                ao = price_map.get(away_raw)
                if ho is None or do_ is None or ao is None:
                    continue
                try:
                    ho, do_, ao = float(ho), float(do_), float(ao)
                except (TypeError, ValueError):
                    continue
                if ho <= 1.0 or do_ <= 1.0 or ao <= 1.0:
                    continue
                home_odds_list.append(ho)
                draw_odds_list.append(do_)
                away_odds_list.append(ao)

        if not home_odds_list:
            continue

        o_h = statistics.median(home_odds_list)
        o_d = statistics.median(draw_odds_list)
        o_a = statistics.median(away_odds_list)

        ph, pd_, pa = 1.0 / o_h, 1.0 / o_d, 1.0 / o_a
        total = ph + pd_ + pa
        rows.append({
            "year": 2026,
            "date": pd.Timestamp(str(event.get("commence_time", ""))[:10]),
            "team_a": ta,
            "team_b": tb,
            "p_win": ph / total,
            "p_draw": pd_ / total,
            "p_loss": pa / total,
        })

    if not rows:
        return pd.DataFrame(columns=_EMPTY_COLS)

    df = pd.DataFrame(rows)
    return df.sort_values("date").drop_duplicates(["team_a", "team_b"], keep="last").reset_index(drop=True)


def parse_market_offers(data: list[dict]) -> pd.DataFrame:
    """Parse the-odds-api JSON → individual bookmaker spread, totals, and 1X2 offers.

    Returns DataFrame with columns:
        year, date, team_a, team_b, market, line, side, price, bookmaker

    market='1x2'   : 1X2 match result; line=0 (unused); side ∈ {home, draw, away}.
                     h2h_lay and two-way (no Draw) markets are skipped.
    market='ah'    : Asian handicap; line is always the home AH line in standard notation
                     (negative = home gives goals).  Both home and away sides are emitted
                     as separate rows so each is a distinct bettable offer.
    market='total' : Asian totals over/under; line is the total-goals threshold.

    Validation: prices > 1.0, line on the quarter grid (multiples of 0.25).
    TBD/placeholder events (knockout-stage "Winner Group X" etc.) are skipped.
    Exact duplicate rows (same team pair, market, line, side, bookmaker) are dropped.
    """
    _COLS = ["year", "date", "team_a", "team_b", "market", "line", "side", "price", "bookmaker"]
    rows: list[dict] = []

    for event in data:
        home_raw = event.get("home_team", "")
        away_raw = event.get("away_team", "")
        ta = canonical(home_raw)
        tb = canonical(away_raw)
        if not ta or not tb or _is_tbd(ta) or _is_tbd(tb):
            continue

        date = pd.Timestamp(str(event.get("commence_time", ""))[:10])

        for bookie in event.get("bookmakers", []):
            bk = bookie.get("key", "")
            for market in bookie.get("markets", []):
                mkey = market.get("key", "")
                outcomes = market.get("outcomes", [])

                if mkey == "h2h":
                    # 3-way 1X2 only — skip if Draw is missing (two-way market)
                    price_map = {o["name"]: o["price"] for o in outcomes}
                    ho = price_map.get(home_raw) or price_map.get(ta)
                    do_ = price_map.get("Draw")
                    ao = price_map.get(away_raw) or price_map.get(tb)
                    if ho is None or do_ is None or ao is None:
                        continue
                    try:
                        ho, do_, ao = float(ho), float(do_), float(ao)
                    except (TypeError, ValueError):
                        continue
                    if ho <= 1.0 or do_ <= 1.0 or ao <= 1.0:
                        continue
                    base = {"year": 2026, "date": date, "team_a": ta, "team_b": tb,
                            "market": "1x2", "line": 0.0, "bookmaker": bk}
                    rows.append({**base, "side": "home", "price": ho})
                    rows.append({**base, "side": "draw", "price": do_})
                    rows.append({**base, "side": "away", "price": ao})

                elif mkey == "spreads":
                    home_oc = next(
                        (o for o in outcomes if o.get("name") in (home_raw, ta)), None
                    )
                    away_oc = next(
                        (o for o in outcomes if o.get("name") in (away_raw, tb)), None
                    )
                    if home_oc is None or away_oc is None:
                        continue
                    try:
                        home_line = float(home_oc.get("point", 0))
                        home_price = float(home_oc["price"])
                        away_price = float(away_oc["price"])
                    except (TypeError, ValueError, KeyError):
                        continue
                    if home_price <= 1.0 or away_price <= 1.0:
                        continue
                    if abs(round(home_line * 4) - home_line * 4) > 1e-6:
                        continue
                    base = {"year": 2026, "date": date, "team_a": ta, "team_b": tb,
                            "market": "ah", "line": home_line, "bookmaker": bk}
                    rows.append({**base, "side": "home", "price": home_price})
                    rows.append({**base, "side": "away", "price": away_price})

                elif mkey == "totals":
                    over_oc = next((o for o in outcomes if o.get("name") == "Over"), None)
                    under_oc = next((o for o in outcomes if o.get("name") == "Under"), None)
                    if over_oc is None or under_oc is None:
                        continue
                    try:
                        tot_line = float(over_oc.get("point", 2.5))
                        over_price = float(over_oc["price"])
                        under_price = float(under_oc["price"])
                    except (TypeError, ValueError, KeyError):
                        continue
                    if over_price <= 1.0 or under_price <= 1.0:
                        continue
                    if abs(round(tot_line * 4) - tot_line * 4) > 1e-6:
                        continue
                    base = {"year": 2026, "date": date, "team_a": ta, "team_b": tb,
                            "market": "total", "line": tot_line, "bookmaker": bk}
                    rows.append({**base, "side": "over", "price": over_price})
                    rows.append({**base, "side": "under", "price": under_price})

    if not rows:
        return pd.DataFrame(columns=_COLS)

    df = pd.DataFrame(rows)
    return df.drop_duplicates(
        ["team_a", "team_b", "market", "line", "side", "bookmaker"], keep="last"
    ).reset_index(drop=True)


def _parse_odds_api_json(data: list[dict]) -> pd.DataFrame:
    """Parse the-odds-api JSON response → tidy DataFrame."""
    rows: list[dict] = []

    for event in data:
        home_raw = event.get("home_team", "")
        away_raw = event.get("away_team", "")
        home = canonical(home_raw) or home_raw
        away = canonical(away_raw) or away_raw
        date = str(event.get("commence_time", ""))[:10]  # YYYY-MM-DD

        ah_lines: list[tuple[float, float, float]] = []  # (line, home_odds, away_odds)
        ou_lines: list[tuple[float, float, float]] = []  # (line, over_odds, under_odds)

        for bookie in event.get("bookmakers", []):
            for market in bookie.get("markets", []):
                key = market.get("key", "")
                outcomes = {o["name"]: o["price"] for o in market.get("outcomes", [])}

                if key == "spreads":
                    for outcome in market.get("outcomes", []):
                        if outcome["name"] == home_raw or outcome["name"] == home:
                            try:
                                ah_lines.append((
                                    float(outcome.get("point", 0)),
                                    float(outcome["price"]),
                                    float([o["price"] for o in market["outcomes"]
                                           if o["name"] != outcome["name"]][0]),
                                ))
                            except (KeyError, IndexError, ValueError):
                                pass
                            break

                elif key == "totals":
                    for outcome in market.get("outcomes", []):
                        if outcome.get("name") == "Over":
                            try:
                                under_p = next(
                                    o["price"] for o in market["outcomes"]
                                    if o.get("name") == "Under"
                                )
                                ou_lines.append((
                                    float(outcome.get("point", 2.5)),
                                    float(outcome["price"]),
                                    float(under_p),
                                ))
                            except (StopIteration, KeyError, ValueError):
                                pass
                            break

        if ah_lines:
            ah_line, ah_h, ah_a = ah_lines[0]  # first bookmaker's main line
        else:
            ah_line = ah_h = ah_a = None

        if ou_lines:
            ou_line, ov, un = ou_lines[0]
        else:
            ou_line = ov = un = None

        rows.append({
            "year": 2026,
            "date": date,
            "home": home,
            "away": away,
            "ah_line": ah_line,
            "ah_home_odds": ah_h,
            "ah_away_odds": ah_a,
            "ou_line": ou_line,
            "over_odds": ov,
            "under_odds": un,
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = fetch_odds_api(force=True)
    if df.empty:
        print("No data returned — check ODDS_API_KEY environment variable.")
    else:
        print(df.head())
        print(f"\n{len(df)} matches fetched.")
