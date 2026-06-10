"""EV evaluation of bookmaker offers against the score-probability matrix.

Pure post-processing functions — no I/O, no model fitting.
"""
from __future__ import annotations

from wcpredictor.markets.asian import asian_handicap, asian_total


def ev_per_unit(settlement: dict, price: float) -> float:
    """Expected value per unit staked at quoted decimal price.

    EV = price·(p_win + 0.5·p_half_win) + 0.5·p_half_win + p_push + 0.5·p_half_loss − 1
    Equivalently: (price − fair_odds)·(p_win + 0.5·p_half_win).
    """
    a = settlement["p_win"] + 0.5 * settlement["p_half_win"]
    b = 0.5 * settlement["p_half_win"] + settlement["p_push"] + 0.5 * settlement["p_half_loss"]
    return price * a + b - 1.0


def evaluate_offers(
    matrix: list[list[float]],
    offers: list[dict],
) -> list[dict]:
    """Evaluate bookmaker offers against the score-probability matrix.

    Each offer dict must contain: market, line, side, price, bookmaker.
    Returns a new list (sorted descending by EV) with additional fields:
        p_win, p_half_win, p_push, p_half_loss, p_loss, fair_odds, ev, p_cover.
    """
    results: list[dict] = []
    for offer in offers:
        market = offer["market"]
        line = float(offer["line"])
        side = offer["side"]
        price = float(offer["price"])

        if market == "ah":
            s = asian_handicap(matrix, side=side, line=line)
        elif market == "total":
            s = asian_total(matrix, side=side, line=line)
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
