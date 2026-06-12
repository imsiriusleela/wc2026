"""EV evaluation of bookmaker offers against the score-probability matrix.

Pure post-processing functions — no I/O, no model fitting.
"""
from __future__ import annotations

import statistics

from wcpredictor.markets.asian import asian_handicap, asian_total

_EPS = 1e-12


def outcome_prob(matrix: list[list[float]], side: str) -> float:
    """Derive 1X2 outcome probability from the score-probability matrix.

    Parameters
    ----------
    matrix : MAX_GOALS × MAX_GOALS joint probability matrix.
    side   : 'home' (team_a wins), 'draw', or 'away' (team_b wins).

    Returns
    -------
    Float probability in [0, 1].
    """
    p = 0.0
    n = len(matrix)
    if side == "home":
        for i in range(n):
            for j in range(i):
                p += matrix[i][j]
    elif side == "draw":
        for i in range(n):
            p += matrix[i][i]
    elif side == "away":
        for i in range(n):
            for j in range(i + 1, n):  # j > i → away goals > home goals
                p += matrix[i][j]
    else:
        raise ValueError(f"side must be 'home', 'draw', or 'away'; got {side!r}")
    return p


def ev_per_unit(settlement: dict, price: float) -> float:
    """Expected value per unit staked at quoted decimal price.

    EV = price·(p_win + 0.5·p_half_win) + 0.5·p_half_win + p_push + 0.5·p_half_loss − 1
    Equivalently: (price − fair_odds)·(p_win + 0.5·p_half_win).
    """
    a = settlement["p_win"] + 0.5 * settlement["p_half_win"]
    b = 0.5 * settlement["p_half_win"] + settlement["p_push"] + 0.5 * settlement["p_half_loss"]
    return price * a + b - 1.0


def consensus_fair(
    offers: list[dict],
    market: str,
    line: float | None,
    side: str,
) -> float | None:
    """De-margined consensus fair price for a specific outcome across bookmakers.

    For 1x2: normalise 1/odds within each bookmaker's three outcomes, then take
    the median of each bookmaker's normalised implied prob; return 1/median_prob.
    For ah/total: normalise the two-way market per bookmaker, median across books.

    Parameters
    ----------
    offers : list of offer dicts (same schema as parse_market_offers output).
             Each dict must have: market, side, price, bookmaker.
             For ah/total offers must also have: line.
    market : '1x2', 'ah', or 'total'.
    line   : AH or totals line (ignored for '1x2').
    side   : outcome to price ('home'/'draw'/'away' for 1x2; 'home'/'away' for ah;
             'over'/'under' for total).

    Returns
    -------
    Fair decimal price (float) or None if not enough data to compute.
    """
    if market == "1x2":
        # For each bookmaker that quoted all three outcomes, normalise and extract side prob.
        by_bookie: dict[str, dict[str, float]] = {}
        for o in offers:
            if o.get("market") != "1x2":
                continue
            bk = o.get("bookmaker", "")
            s = o.get("side", "")
            if s in ("home", "draw", "away"):
                by_bookie.setdefault(bk, {})[s] = float(o["price"])

        fair_probs: list[float] = []
        for bk_prices in by_bookie.values():
            if not all(s in bk_prices and bk_prices[s] > 1.0 for s in ("home", "draw", "away")):
                continue
            ph = 1.0 / bk_prices["home"]
            pd_ = 1.0 / bk_prices["draw"]
            pa = 1.0 / bk_prices["away"]
            total = ph + pd_ + pa
            fair_p = {"home": ph / total, "draw": pd_ / total, "away": pa / total}
            if side in fair_p:
                fair_probs.append(fair_p[side])

        if not fair_probs:
            return None
        p = statistics.median(fair_probs)
        return round(1.0 / p, 4) if p > _EPS else None

    elif market in ("ah", "total"):
        opposite = {"home": "away", "away": "home", "over": "under", "under": "over"}
        opp_side = opposite.get(side)
        if opp_side is None:
            return None

        by_bookie: dict[str, dict[str, float]] = {}
        for o in offers:
            if o.get("market") != market:
                continue
            if line is not None and abs(float(o.get("line", float("nan"))) - line) > 1e-6:
                continue
            bk = o.get("bookmaker", "")
            s = o.get("side", "")
            if s in (side, opp_side):
                by_bookie.setdefault(bk, {})[s] = float(o["price"])

        fair_probs: list[float] = []
        for bk_prices in by_bookie.values():
            if side not in bk_prices or opp_side not in bk_prices:
                continue
            p1, p2 = 1.0 / bk_prices[side], 1.0 / bk_prices[opp_side]
            total = p1 + p2
            fair_probs.append(p1 / total)

        if not fair_probs:
            return None
        p = statistics.median(fair_probs)
        return round(1.0 / p, 4) if p > _EPS else None

    return None


def evaluate_offers(
    matrix: list[list[float]],
    offers: list[dict],
) -> list[dict]:
    """Evaluate bookmaker offers against the score-probability matrix.

    Each offer dict must contain: market, line, side, price, bookmaker.
    Supported markets: 'ah', 'total', '1x2'.
    Returns a new list (sorted descending by EV) with additional fields:
        p_win, p_half_win, p_push, p_half_loss, p_loss, fair_odds, ev, p_cover.
    """
    results: list[dict] = []
    for offer in offers:
        market = offer["market"]
        side = offer["side"]
        price = float(offer["price"])

        if market == "ah":
            line = float(offer["line"])
            s = asian_handicap(matrix, side=side, line=line)
        elif market == "total":
            line = float(offer["line"])
            s = asian_total(matrix, side=side, line=line)
        elif market == "1x2":
            p = outcome_prob(matrix, side)
            # 1X2 has no push or half outcomes
            s = {
                "p_win": p,
                "p_half_win": 0.0,
                "p_push": 0.0,
                "p_half_loss": 0.0,
                "p_loss": 1.0 - p,
                "fair_odds": round(1.0 / p, 4) if p > _EPS else float("inf"),
            }
        else:
            continue

        ev = ev_per_unit(s, price)
        results.append({
            **offer,
            "p_win": s["p_win"],
            "p_half_win": s["p_half_win"],
            "p_push": s["p_push"],
            "p_half_loss": s["p_half_loss"],
            "p_loss": s["p_loss"],
            "fair_odds": s["fair_odds"],
            "ev": round(ev, 4),
            "p_cover": round(s["p_win"] + s["p_half_win"], 6),
        })

    results.sort(key=lambda r: r["ev"], reverse=True)
    return results
