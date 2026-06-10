"""Fetch live WC2026 Asian-handicap and O/U odds from the-odds-api.com.

Requires an API key in the environment variable ODDS_API_KEY.
Gracefully returns an empty DataFrame when the key is absent or the
API returns no data — the rest of the pipeline auto-degrades to
model-only predictions in that case.

Usage:
    export ODDS_API_KEY=your_key_here
    uv run python -m wcpredictor.data.download_odds_api

Fetches:
    soccer_fifa_world_cup, markets=h2h,spreads,totals
    Saves raw JSON to data/raw/odds_api_wc2026.json (overwritten each run).
    Returns a tidy DataFrame with schema matching wc_ah_odds.csv.
"""
from __future__ import annotations

import json
import os
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
        with urllib.request.urlopen(url, timeout=30) as resp:  # noqa: S310
            raw = json.loads(resp.read().decode())
        _JSON_DEST.write_text(json.dumps(raw, indent=2))

    return _parse_odds_api_json(raw)


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
