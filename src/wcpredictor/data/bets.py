"""Bet ledger: record, settle, and score bets against Singapore Pools.

Schema (data/bets.csv):
    placed_at           : ISO timestamp when bet was logged
    date                : match date (YYYY-MM-DD)
    match               : 'TeamA vs TeamB' string
    market              : '1x2', 'ah', or 'total'
    line                : AH/totals line (0.0 for 1x2)
    side                : side bet is on
    price_taken         : decimal odds accepted at SG Pools
    stake               : amount staked (SGD)
    status              : open | won | half_won | push | half_lost | lost
    pnl                 : profit/loss in SGD (None until settled)
    consensus_fair_at_placement : consensus fair price when bet was placed
    closing_consensus_fair      : consensus fair price at last /refresh-odds near kickoff
    clv_pct             : price_taken / closing_consensus_fair - 1 (real CLV, filled post-kickoff)

Settlement uses markets/asian.py semantics (correct quarter-line payout).
"""
from __future__ import annotations

import csv
import datetime
import math
from pathlib import Path
from typing import Any

import pandas as pd

from wcpredictor.config import DATA_RAW, STOP_CLV_FLOOR, STOP_DRAWDOWN_UNITS, STOP_TRAILING_BETS
from wcpredictor.data.normalize_teams import canonical

_BETS_CSV = DATA_RAW.parent / "bets.csv"

_STATUS_VALUES = {"open", "won", "half_won", "push", "half_lost", "lost"}
_COLS = [
    "placed_at", "date", "match", "market", "line", "side",
    "price_taken", "stake", "status", "pnl",
    "consensus_fair_at_placement", "closing_consensus_fair", "clv_pct",
]


# ─── Load ──────────────────────────────────────────────────────────────────────

def load_ledger(path: Path | None = None) -> pd.DataFrame:
    """Load the bet ledger. Returns empty DataFrame if file doesn't exist."""
    if path is None:
        path = _BETS_CSV
    if not path.exists():
        return pd.DataFrame(columns=_COLS)
    df = pd.read_csv(path, parse_dates=["date"])
    for col in ("pnl", "consensus_fair_at_placement", "closing_consensus_fair", "clv_pct"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ─── Record ────────────────────────────────────────────────────────────────────

def record_bet(
    team_a: str,
    team_b: str,
    date: str,
    market: str,
    line: float,
    side: str,
    price_taken: float,
    stake: float,
    consensus_fair_at_placement: float | None = None,
    path: Path | None = None,
) -> None:
    """Append a new bet to the ledger.

    Parameters
    ----------
    team_a, team_b : canonical team names.
    date           : match date (ISO string).
    market         : '1x2', 'ah', or 'total'.
    line           : AH/totals line (0.0 for 1x2).
    side           : bet side.
    price_taken    : decimal odds obtained.
    stake          : SGD amount staked.
    consensus_fair_at_placement: consensus de-margined fair price at time of entry.
    """
    if path is None:
        path = _BETS_CSV

    ta = canonical(team_a) or team_a
    tb = canonical(team_b) or team_b

    if price_taken <= 1.0:
        raise ValueError(f"price_taken must be > 1.0; got {price_taken}")
    if stake <= 0.0:
        raise ValueError(f"stake must be > 0; got {stake}")

    row: dict[str, Any] = {
        "placed_at": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "date": date,
        "match": f"{ta} vs {tb}",
        "market": market.lower(),
        "line": float(line),
        "side": side.lower(),
        "price_taken": float(price_taken),
        "stake": float(stake),
        "status": "open",
        "pnl": "",
        "consensus_fair_at_placement": float(consensus_fair_at_placement) if consensus_fair_at_placement else "",
        "closing_consensus_fair": "",
        "clv_pct": "",
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_COLS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ─── Settle ────────────────────────────────────────────────────────────────────

def _settle_1x2(goals_a: int, goals_b: int, side: str, price: float, stake: float) -> tuple[str, float]:
    """Return (status, pnl) for a 1X2 bet."""
    outcome = 0 if goals_a > goals_b else (1 if goals_a == goals_b else 2)
    side_map = {"home": 0, "draw": 1, "away": 2}
    if outcome == side_map.get(side, -1):
        return "won", round((price - 1.0) * stake, 2)
    return "lost", round(-stake, 2)


def _settle_asian(
    goals_a: int,
    goals_b: int,
    market: str,
    line: float,
    side: str,
    price: float,
    stake: float,
) -> tuple[str, float]:
    """Return (status, pnl) for an AH or total bet using markets/asian.py semantics."""
    from wcpredictor.markets.asian import asian_handicap, asian_total

    max_g = max(goals_a, goals_b, int(abs(line)) + 2, 10)
    mat = [[0.0] * (max_g + 1) for _ in range(max_g + 1)]
    mat[goals_a][goals_b] = 1.0

    if market == "ah":
        s = asian_handicap(mat, side=side, line=line)
    else:
        s = asian_total(mat, side=side, line=float(abs(line)))

    pw = s["p_win"]
    phw = s["p_half_win"]
    pp = s["p_push"]
    phl = s["p_half_loss"]
    pl = s["p_loss"]

    if pw > 0.5:
        status = "won"
        pnl = round((price - 1.0) * stake, 2)
    elif phw > 0.5:
        status = "half_won"
        pnl = round((price * 0.5 + 0.5 - 1.0) * stake, 2)
    elif pp > 0.5:
        status = "push"
        pnl = 0.0
    elif phl > 0.5:
        status = "half_lost"
        pnl = round(-0.5 * stake, 2)
    else:
        status = "lost"
        pnl = round(-stake, 2)

    return status, pnl


def settle_open_bets(results_lookup: dict[str, tuple[int, int]], path: Path | None = None) -> int:
    """Settle open bets from the ledger using a results lookup.

    Parameters
    ----------
    results_lookup : dict mapping 'TeamA vs TeamB' → (goals_a, goals_b).
                     Use the Phase 12 load_wc2026_results store.
    path           : path to bets.csv.

    Returns number of bets settled.
    """
    if path is None:
        path = _BETS_CSV

    df = load_ledger(path)
    if df.empty:
        return 0

    n_settled = 0
    updated_rows: list[dict] = []

    for _, row in df.iterrows():
        if str(row.get("status", "open")) != "open":
            updated_rows.append(row.to_dict())
            continue

        match_key = str(row.get("match", ""))
        if match_key not in results_lookup:
            updated_rows.append(row.to_dict())
            continue

        goals_a, goals_b = results_lookup[match_key]
        market = str(row.market)
        side = str(row.side)
        price = float(row.price_taken)
        stake = float(row.stake)
        line = float(row.line)

        if market == "1x2":
            status, pnl = _settle_1x2(goals_a, goals_b, side, price, stake)
        elif market in ("ah", "total"):
            status, pnl = _settle_asian(goals_a, goals_b, market, line, side, price, stake)
        else:
            updated_rows.append(row.to_dict())
            continue

        row_dict = row.to_dict()
        row_dict["status"] = status
        row_dict["pnl"] = pnl
        updated_rows.append(row_dict)
        n_settled += 1

    if n_settled > 0:
        _write_ledger(updated_rows, path)

    return n_settled


def update_closing_consensus(
    match_key: str,
    market: str,
    line: float,
    side: str,
    closing_price: float,
    path: Path | None = None,
) -> None:
    """Fill closing_consensus_fair and clv_pct for open bets on a specific outcome.

    Called from POST /refresh-odds near kickoff to record true CLV.
    """
    if path is None:
        path = _BETS_CSV

    df = load_ledger(path)
    if df.empty:
        return

    updated = False
    rows: list[dict] = []
    for _, row in df.iterrows():
        row_dict = row.to_dict()
        if (str(row.get("match")) == match_key
                and str(row.get("market")) == market
                and abs(float(row.get("line", 0)) - line) < 1e-6
                and str(row.get("side")) == side
                and str(row.get("status")) == "open"):
            row_dict["closing_consensus_fair"] = round(closing_price, 4)
            price_taken = float(row.get("price_taken", 0))
            if closing_price > 1.0 and price_taken > 1.0:
                row_dict["clv_pct"] = round(price_taken / closing_price - 1.0, 4)
            updated = True
        rows.append(row_dict)

    if updated:
        _write_ledger(rows, path)


def _write_ledger(rows: list[dict], path: Path) -> None:
    """Rewrite the ledger CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_COLS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# ─── P&L and stop-rule scorecard ───────────────────────────────────────────────

def ledger_summary(path: Path | None = None) -> dict:
    """Compute P&L summary and stop-rule flags from the ledger.

    Returns dict with keys:
        total_bets, settled_bets, open_bets,
        total_staked, total_pnl, roi,
        trailing_clv_mean (last STOP_TRAILING_BETS settled bets),
        drawdown_units (max drawdown in units from peak),
        stop_clv, stop_drawdown (bool flags).
    """
    df = load_ledger(path)
    total_bets = len(df)
    settled = df[df["status"] != "open"]
    open_bets = len(df[df["status"] == "open"])
    settled_bets = len(settled)

    total_staked = float(df["stake"].fillna(0).sum())
    total_pnl = float(settled["pnl"].fillna(0).sum())
    roi = round(total_pnl / total_staked, 4) if total_staked > 0 else float("nan")

    # Trailing CLV (last N settled bets with clv_pct recorded)
    clv_series = settled["clv_pct"].dropna()
    trailing_clv = float(clv_series.tail(STOP_TRAILING_BETS).mean()) if len(clv_series) > 0 else float("nan")

    # Drawdown in units (using stake-normalised P&L)
    drawdown_units = 0.0
    if settled_bets > 0:
        unit_pnl = (settled["pnl"].fillna(0) / settled["stake"].replace(0, float("nan"))).dropna()
        cumsum = unit_pnl.cumsum()
        peak = cumsum.cummax()
        dd = (peak - cumsum).max()
        drawdown_units = round(float(dd), 2) if not math.isnan(dd) else 0.0

    stop_clv = (
        not math.isnan(trailing_clv)
        and len(clv_series) >= STOP_TRAILING_BETS
        and trailing_clv < STOP_CLV_FLOOR
    )
    stop_drawdown = drawdown_units >= STOP_DRAWDOWN_UNITS

    return {
        "total_bets": total_bets,
        "settled_bets": settled_bets,
        "open_bets": open_bets,
        "total_staked": round(total_staked, 2),
        "total_pnl": round(total_pnl, 2),
        "roi": roi,
        "trailing_clv_mean": round(trailing_clv, 4) if not math.isnan(trailing_clv) else None,
        "drawdown_units": drawdown_units,
        "stop_clv": stop_clv,
        "stop_drawdown": stop_drawdown,
        "stop_rule_triggered": stop_clv or stop_drawdown,
    }
