"""Singapore Pools odds ingestion.

Manual-entry MVP plus automated fetcher (approved 2026-06-12).

Schema (sgpools_offers.csv):
    entered_at  : ISO timestamp of when the offer was recorded
    date        : match date (YYYY-MM-DD)
    team_a      : canonical home team name
    team_b      : canonical away team name
    market      : '1x2', 'ah', or 'total'
    line        : AH/totals line (0.0 for 1x2)
    side        : 'home'/'draw'/'away' for 1x2; 'home'/'away' for ah; 'over'/'under' for total
    price       : decimal odds
    source      : 'manual' or 'fetched'

Manual rows take precedence over fetched rows for the same match+market+line+side.

CLI:
    python -m wcpredictor.data.sgpools add <team_a> <team_b> <date> <market> <line> <side> <price>
    python -m wcpredictor.data.sgpools fetch
"""
from __future__ import annotations

import csv
import datetime
import json
import math
import shutil
import warnings
from pathlib import Path
from typing import Sequence

import pandas as pd

from wcpredictor.config import DATA_RAW
from wcpredictor.data.normalize_teams import canonical

_CSV_PATH = DATA_RAW / "sgpools_offers.csv"
_SNAPSHOTS_DIR = DATA_RAW / "sgpools_snapshots"

_VALID_MARKETS = {"1x2", "ah", "total"}
_VALID_SIDES = {
    "1x2": {"home", "draw", "away"},
    "ah": {"home", "away"},
    "total": {"over", "under"},
}
_COLS = ["entered_at", "date", "team_a", "team_b", "market", "line", "side", "price", "source"]


# ─── Validation ────────────────────────────────────────────────────────────────

def _validate_row(row: dict) -> None:
    """Raise ValueError on invalid offer row."""
    market = row.get("market", "")
    if market not in _VALID_MARKETS:
        raise ValueError(f"market must be one of {_VALID_MARKETS}; got {market!r}")

    side = row.get("side", "")
    valid_sides = _VALID_SIDES.get(market, set())
    if side not in valid_sides:
        raise ValueError(f"side {side!r} invalid for market {market!r}; must be one of {valid_sides}")

    price = float(row.get("price", 0))
    if price <= 1.0:
        raise ValueError(f"price must be > 1.0; got {price}")

    line = float(row.get("line", 0))
    if market in ("ah", "total"):
        # must be on quarter grid
        if abs(round(line * 4) - line * 4) > 1e-6:
            raise ValueError(f"line must be a multiple of 0.25; got {line}")

    team_a = row.get("team_a", "")
    team_b = row.get("team_b", "")
    if not team_a or not team_b:
        raise ValueError("team_a and team_b must be non-empty canonical names")


# ─── Load ──────────────────────────────────────────────────────────────────────

def load_sgpools_offers(csv_path: Path | None = None) -> pd.DataFrame:
    """Load and validate the SG Pools offers CSV.

    Returns a DataFrame with columns: entered_at, date, team_a, team_b,
    market, line, side, price, source.  Manual rows take precedence over
    fetched rows for the same (team_a, team_b, market, line, side).
    Returns empty DataFrame if the file doesn't exist yet.
    """
    if csv_path is None:
        csv_path = _CSV_PATH

    if not csv_path.exists():
        return pd.DataFrame(columns=_COLS)

    df = pd.read_csv(csv_path, parse_dates=["date"])
    # Canonicalise team names
    df["team_a"] = df["team_a"].apply(lambda x: canonical(str(x)) or str(x))
    df["team_b"] = df["team_b"].apply(lambda x: canonical(str(x)) or str(x))
    df["market"] = df["market"].str.lower()
    df["side"] = df["side"].str.lower()
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["line"] = pd.to_numeric(df["line"], errors="coerce").fillna(0.0)
    df["source"] = df["source"].fillna("manual")

    # Drop rows with invalid prices
    df = df[df["price"] > 1.0].copy()

    # Manual takes precedence: sort so 'manual' > 'fetched' and deduplicate
    source_order = {"manual": 0, "fetched": 1}
    df["_source_rank"] = df["source"].map(source_order).fillna(1)
    df = (
        df.sort_values("_source_rank")
        .drop_duplicates(subset=["team_a", "team_b", "market", "line", "side"], keep="first")
        .drop(columns=["_source_rank"])
        .reset_index(drop=True)
    )

    return df


# ─── Add ───────────────────────────────────────────────────────────────────────

def add_offer(
    team_a: str,
    team_b: str,
    date: str,
    market: str,
    line: float,
    side: str,
    price: float,
    source: str = "manual",
    csv_path: Path | None = None,
) -> None:
    """Append one offer to the SG Pools CSV.

    Parameters
    ----------
    team_a, team_b : team names (will be canonicalised).
    date           : match date string (ISO, e.g. '2026-06-12').
    market         : '1x2', 'ah', or 'total'.
    line           : AH/totals line (0.0 for 1x2).
    side           : see _VALID_SIDES for valid values per market.
    price          : decimal odds.
    source         : 'manual' (default) or 'fetched'.
    """
    if csv_path is None:
        csv_path = _CSV_PATH

    ta = canonical(team_a) or team_a
    tb = canonical(team_b) or team_b
    market = market.lower()
    side = side.lower()

    row = {
        "entered_at": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "date": date,
        "team_a": ta,
        "team_b": tb,
        "market": market,
        "line": float(line),
        "side": side,
        "price": float(price),
        "source": source,
    }
    _validate_row(row)

    DATA_RAW.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_COLS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ─── Automated fetcher ─────────────────────────────────────────────────────────

def fetch_sgpools_offers(csv_path: Path | None = None) -> pd.DataFrame:
    """Fetch current WC2026 odds from Singapore Pools website.

    Probes the XHR JSON endpoints behind online.singaporepools.com.sg/en/sports.
    One fetch per call (no polling loops). Archives raw responses to
    data/raw/sgpools_snapshots/<timestamp>.json.
    Merges into sgpools_offers.csv with source='fetched'; manual rows win on conflict.

    Returns the fetched DataFrame (empty on network/parse failure — non-fatal).
    """
    if csv_path is None:
        csv_path = _CSV_PATH

    import urllib.request
    import urllib.error

    # SG Pools football odds API endpoint (discovered via browser devtools 2026-06-12).
    # The endpoint returns JSON with a list of events for the active sport.
    _SG_API_BASE = "https://online.singaporepools.com"
    _SG_FOOTBALL_URL = f"{_SG_API_BASE}/en/sports/api/1.0/getSeasonMatchList?sport=football"
    _UA = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )

    ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    snapshot_path = _SNAPSHOTS_DIR / f"sgpools_{ts}.json"

    try:
        req = urllib.request.Request(
            _SG_FOOTBALL_URL,
            headers={"User-Agent": _UA, "Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=20) as resp:  # noqa: S310
            raw_bytes = resp.read()
        raw_json = json.loads(raw_bytes.decode("utf-8"))
    except (urllib.error.URLError, json.JSONDecodeError, Exception) as exc:
        warnings.warn(
            f"SG Pools fetch failed: {exc}. Use manual entry as fallback.",
            RuntimeWarning,
            stacklevel=2,
        )
        return pd.DataFrame(columns=_COLS)

    # Archive snapshot
    _SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text(json.dumps(raw_json, indent=2))

    # Parse the response
    rows = _parse_sgpools_response(raw_json)
    if not rows:
        warnings.warn(
            "SG Pools response parsed but no WC2026 football odds found. "
            "Site structure may have changed.",
            RuntimeWarning,
            stacklevel=2,
        )
        return pd.DataFrame(columns=_COLS)

    # Append to CSV (fetched rows; manual rows will override on dedup)
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_COLS)
        if write_header:
            writer.writeheader()
        for r in rows:
            writer.writerow(r)

    return pd.DataFrame(rows)


def _parse_sgpools_response(data: object) -> list[dict]:
    """Parse SG Pools JSON response into canonical offer rows.

    SG Pools returns a JSON structure that varies by API version.
    This parser handles the most common format observed (list of match dicts).
    Returns empty list on unrecognised structure.
    """
    if not isinstance(data, (list, dict)):
        return []

    # Normalise: if dict with a 'matches' or 'events' key, unwrap it
    events: list = []
    if isinstance(data, dict):
        for key in ("matches", "events", "data", "results"):
            if isinstance(data.get(key), list):
                events = data[key]
                break
    elif isinstance(data, list):
        events = data

    if not events:
        return []

    rows: list[dict] = []
    now_ts = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    for event in events:
        if not isinstance(event, dict):
            continue

        # Extract team names
        home_raw = str(event.get("homeTeam", event.get("home", event.get("team1", ""))))
        away_raw = str(event.get("awayTeam", event.get("away", event.get("team2", ""))))
        ta = canonical(home_raw)
        tb = canonical(away_raw)
        if not ta or not tb:
            continue

        # Extract date
        date_raw = str(event.get("matchDate", event.get("date", event.get("startDate", ""))))
        try:
            match_date = pd.Timestamp(date_raw[:10]).strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            continue

        # 1X2 odds
        try:
            ho = float(event.get("homeOdds", event.get("oddsH", event.get("odds1", 0))))
            do_ = float(event.get("drawOdds", event.get("oddsD", event.get("oddsX", 0))))
            ao = float(event.get("awayOdds", event.get("oddsA", event.get("odds2", 0))))
            if ho > 1.0 and do_ > 1.0 and ao > 1.0:
                base = dict(entered_at=now_ts, date=match_date,
                            team_a=ta, team_b=tb, market="1x2", line=0.0)
                rows += [
                    {**base, "side": "home", "price": ho, "source": "fetched"},
                    {**base, "side": "draw", "price": do_, "source": "fetched"},
                    {**base, "side": "away", "price": ao, "source": "fetched"},
                ]
        except (TypeError, ValueError):
            pass

        # AH odds (SG Pools may label differently)
        try:
            ah_line = float(event.get("ahLine", event.get("handicap", float("nan"))))
            ah_ho = float(event.get("ahHomeOdds", event.get("ahOddsH", float("nan"))))
            ah_ao = float(event.get("ahAwayOdds", event.get("ahOddsA", float("nan"))))
            if (not math.isnan(ah_line) and ah_ho > 1.0 and ah_ao > 1.0
                    and abs(round(ah_line * 4) - ah_line * 4) < 1e-6):
                base = dict(entered_at=now_ts, date=match_date,
                            team_a=ta, team_b=tb, market="ah", line=ah_line)
                rows += [
                    {**base, "side": "home", "price": ah_ho, "source": "fetched"},
                    {**base, "side": "away", "price": ah_ao, "source": "fetched"},
                ]
        except (TypeError, ValueError):
            pass

        # O/U totals
        try:
            ou_line = float(event.get("ouLine", event.get("totalLine", float("nan"))))
            ov = float(event.get("overOdds", event.get("oddsOver", float("nan"))))
            un = float(event.get("underOdds", event.get("oddsUnder", float("nan"))))
            if (not math.isnan(ou_line) and ov > 1.0 and un > 1.0
                    and abs(round(ou_line * 4) - ou_line * 4) < 1e-6):
                base = dict(entered_at=now_ts, date=match_date,
                            team_a=ta, team_b=tb, market="total", line=ou_line)
                rows += [
                    {**base, "side": "over", "price": ov, "source": "fetched"},
                    {**base, "side": "under", "price": un, "source": "fetched"},
                ]
        except (TypeError, ValueError):
            pass

    return rows


if __name__ == "__main__":
    import argparse
    import math

    parser = argparse.ArgumentParser(description="SG Pools odds management")
    sub = parser.add_subparsers(dest="cmd")

    add_p = sub.add_parser("add", help="Add a manual offer")
    add_p.add_argument("team_a")
    add_p.add_argument("team_b")
    add_p.add_argument("date")
    add_p.add_argument("market", choices=["1x2", "ah", "total"])
    add_p.add_argument("line", type=float)
    add_p.add_argument("side")
    add_p.add_argument("price", type=float)

    sub.add_parser("fetch", help="Fetch latest odds from SG Pools website")
    sub.add_parser("show", help="Show current offers")

    args = parser.parse_args()

    if args.cmd == "add":
        add_offer(args.team_a, args.team_b, args.date,
                  args.market, args.line, args.side, args.price)
        print(f"Added: {args.team_a} vs {args.team_b} {args.market} {args.side}@{args.price}")

    elif args.cmd == "fetch":
        df = fetch_sgpools_offers()
        if df.empty:
            print("No offers fetched — check network or use manual entry.")
        else:
            print(f"Fetched {len(df)} offers.")

    elif args.cmd == "show":
        df = load_sgpools_offers()
        if df.empty:
            print("No offers on file.")
        else:
            print(df.to_string(index=False))

    else:
        parser.print_help()
