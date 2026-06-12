"""Value scan: find +EV bets vs Singapore Pools prices.

For each unplayed fixture with SG Pools offers loaded, computes:
  - EV_model: EV using the model's fair odds from ensemble_mkt matrix
  - EV_consensus: EV using consensus fair price from the-odds-api books
  - Recommends when EV_model ≥ τ AND price ≥ consensus_fair (double-confirmation)

The consensus benchmark requires a fresh odds-api snapshot (POST /refresh-odds first).

Note: ensemble_mkt already blends 25-book consensus at α=0.3, so EV_model is
conservative. Double confirmation means the model AND the consensus both agree
the price is value.

CLI:
    python -m wcpredictor.markets.value_scan --min-ev 0.0
"""
from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from typing import Any

import pandas as pd

from wcpredictor.config import DATA_PROCESSED, DATA_RAW, EV_THRESHOLD, TOURNAMENT_START
from wcpredictor.data.normalize_teams import canonical
from wcpredictor.data.sgpools import load_sgpools_offers
from wcpredictor.markets.edge import consensus_fair, evaluate_offers


def _load_consensus_offers() -> dict[tuple[str, str], list[dict]]:
    """Load consensus (25-book) offers from the odds-api JSON snapshot."""
    live_json = DATA_RAW / "odds_api_wc2026.json"
    if not live_json.exists():
        return {}
    try:
        from wcpredictor.data.download_odds_api import parse_market_offers
        raw = json.loads(live_json.read_text())
        df = parse_market_offers(raw)
        if df.empty:
            return {}
        lookup: dict[tuple[str, str], list[dict]] = {}
        for (ta, tb), grp in df.groupby(["team_a", "team_b"], sort=False):
            lookup[(str(ta), str(tb))] = grp.to_dict("records")
        return lookup
    except Exception:
        return {}


def _load_frozen_matrices(as_of: str) -> dict[tuple[str, str], list[list[float]]]:
    """Load precomputed score matrices from the fixtures CSV (frozen state)."""
    path = DATA_PROCESSED / "wc2026_predictions_ensemble_mkt.csv"
    if not path.exists():
        # Try any predictions CSV
        import glob
        files = sorted(glob.glob(str(DATA_PROCESSED / "wc2026_predictions_*.csv")))
        if not files:
            return {}
        path = Path(files[-1])

    try:
        df = pd.read_csv(path)
        import ast
        lookup: dict[tuple[str, str], list[list[float]]] = {}
        for _, row in df.iterrows():
            ta = canonical(str(row.get("team_a", "")))
            tb = canonical(str(row.get("team_b", "")))
            date_str = str(row.get("date", ""))
            if not ta or not tb:
                continue
            # Skip already-kicked-off fixtures
            if date_str and date_str < as_of:
                continue
            raw_mat = row.get("score_matrix")
            if raw_mat is None or (isinstance(raw_mat, float) and math.isnan(raw_mat)):
                continue
            try:
                mat = ast.literal_eval(str(raw_mat)) if isinstance(raw_mat, str) else raw_mat
                if isinstance(mat, list) and mat and isinstance(mat[0], list):
                    lookup[(ta, tb)] = mat
            except Exception:
                continue
        return lookup
    except Exception:
        return {}


def _is_played(date_str: str, as_of: str) -> bool:
    """Return True if the match date is strictly before as_of."""
    try:
        return str(date_str)[:10] < str(as_of)[:10]
    except Exception:
        return False


def scan(
    as_of: str | None = None,
    min_ev: float = EV_THRESHOLD,
    require_consensus: bool = True,
) -> list[dict[str, Any]]:
    """Scan SG Pools offers for value bets against the model and consensus.

    Parameters
    ----------
    as_of           : data cutoff date (ISO); defaults to today (but not before tournament start).
    min_ev          : EV_model threshold for inclusion in output.
    require_consensus: if True (default), also require price ≥ consensus_fair.
                       If False, model-only EV suffices.

    Returns
    -------
    List of dicts (sorted by ev_model descending) with keys:
        date, team_a, team_b, market, line, side, sgpools_price,
        fair_model, ev_model, fair_consensus, ev_consensus, recommended,
        recommended_stake, confidence_flags.
    """
    if as_of is None:
        as_of = max(TOURNAMENT_START, pd.Timestamp.today().strftime("%Y-%m-%d"))

    sgpools = load_sgpools_offers()
    if sgpools.empty:
        return []

    matrices = _load_frozen_matrices(as_of)
    consensus_lookup = _load_consensus_offers()

    from wcpredictor.config import BANKROLL, KELLY_FRACTION, MAX_STAKE_PCT, MIN_TIER_STAKE

    # Load EV backtest verdict to determine sizing tier
    sizing_tier = "min_stake"
    ev_report_path = DATA_PROCESSED / "ev_backtest_report.json"
    if ev_report_path.exists():
        try:
            ev_report = json.loads(ev_report_path.read_text())
            sizing_tier = ev_report.get("sizing_tier", "min_stake")
        except Exception:
            pass

    results: list[dict] = []

    # Group SG Pools offers by fixture to enforce one-bet-per-match later
    sgp_by_fixture: dict[tuple[str, str], list[dict]] = {}
    for _, row in sgpools.iterrows():
        date_str = str(row["date"])[:10]
        if _is_played(date_str, as_of):
            continue
        ta = str(row["team_a"])
        tb = str(row["team_b"])
        sgp_by_fixture.setdefault((ta, tb), []).append(row.to_dict())

    for (ta, tb), sgp_offers in sgp_by_fixture.items():
        mat = matrices.get((ta, tb))
        if mat is None:
            warnings.warn(
                f"No frozen score matrix for {ta} vs {tb}; skipping value scan.",
                RuntimeWarning,
                stacklevel=2,
            )
            continue

        # Get consensus offers for this fixture (both orientations)
        con_offers = consensus_lookup.get((ta, tb), []) + consensus_lookup.get((tb, ta), [])

        # Build offer list for evaluate_offers
        model_offers = []
        for sgp in sgp_offers:
            model_offers.append({
                "market": str(sgp["market"]),
                "line": float(sgp.get("line", 0.0)),
                "side": str(sgp["side"]),
                "price": float(sgp["price"]),
                "bookmaker": "sgpools",
            })

        evaluated = evaluate_offers(mat, model_offers)

        for ev_row in evaluated:
            ev_model = float(ev_row["ev"])
            fair_model = float(ev_row["fair_odds"])
            sgp_price = float(ev_row["price"])
            market = str(ev_row["market"])
            line = float(ev_row.get("line", 0.0))
            side = str(ev_row["side"])

            if ev_model < min_ev:
                continue

            # Consensus fair price
            fair_con = consensus_fair(con_offers, market, line if market != "1x2" else None, side)
            ev_consensus = None
            if fair_con is not None and math.isfinite(fair_con) and fair_con > 1.0:
                ev_consensus = round(sgp_price / fair_con - 1.0, 4)

            # Double-confirmation logic
            beats_consensus = (
                fair_con is not None
                and math.isfinite(fair_con)
                and sgp_price >= fair_con
            )
            recommended = ev_model >= min_ev and (beats_consensus if require_consensus else True)

            # Confidence flags
            flags: list[str] = []
            if fair_con is None:
                flags.append("no_consensus_data")
            elif not beats_consensus:
                flags.append("below_consensus_fair")
            if not math.isfinite(fair_model):
                flags.append("infinite_fair_model")

            # Stake calculation (quarter-Kelly capped, or minimum tier)
            stake: float = 0.0
            if recommended:
                if sizing_tier == "quarter_kelly" and math.isfinite(fair_model) and fair_model > 1.0:
                    p = 1.0 / fair_model
                    b = sgp_price - 1.0
                    if b > 0 and p > 0:
                        kelly_f = (b * p - (1 - p)) / b
                        stake = round(KELLY_FRACTION * kelly_f * BANKROLL, 2)
                        stake = max(0.0, min(stake, BANKROLL * MAX_STAKE_PCT))
                elif sizing_tier == "min_stake":
                    stake = MIN_TIER_STAKE

            # Find date from sgp_offers
            match_date = str(sgp_offers[0].get("date", ""))[:10]

            results.append({
                "date": match_date,
                "team_a": ta,
                "team_b": tb,
                "market": market,
                "line": line,
                "side": side,
                "sgpools_price": round(sgp_price, 4),
                "fair_model": round(fair_model, 4) if math.isfinite(fair_model) else None,
                "ev_model": round(ev_model, 4),
                "fair_consensus": round(fair_con, 4) if fair_con is not None and math.isfinite(fair_con) else None,
                "ev_consensus": ev_consensus,
                "recommended": recommended,
                "recommended_stake": round(stake, 2),
                "sizing_tier": sizing_tier,
                "confidence_flags": flags,
            })

    results.sort(key=lambda r: r["ev_model"], reverse=True)
    return results


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Value scan vs Singapore Pools")
    ap.add_argument("--min-ev", type=float, default=0.0, dest="min_ev",
                    help="Minimum EV threshold (default 0.0 = show all)")
    ap.add_argument("--no-consensus", action="store_true",
                    help="Skip consensus double-confirmation (model EV only)")
    args = ap.parse_args()

    today = max(TOURNAMENT_START, pd.Timestamp.today().strftime("%Y-%m-%d"))
    bets = scan(as_of=today, min_ev=args.min_ev, require_consensus=not args.no_consensus)

    if not bets:
        print("No value bets found with current SG Pools offers and settings.")
    else:
        print(f"\n{'Date':<12} {'Match':<30} {'Market':<8} {'Line':>6} {'Side':<6} "
              f"{'Price':>6} {'FairM':>6} {'EV_M':>6} {'FairC':>6} {'Rec':>4} {'Stake':>6}")
        for b in bets:
            match_str = f"{b['team_a']} vs {b['team_b']}"[:28]
            fc = f"{b['fair_consensus']:.3f}" if b['fair_consensus'] else "  -  "
            rec_str = "YES" if b["recommended"] else "no"
            print(
                f"{b['date']:<12} {match_str:<30} {b['market']:<8} {b['line']:>6} "
                f"{b['side']:<6} {b['sgpools_price']:>6.3f} "
                f"{b['fair_model'] or 0:>6.3f} {b['ev_model']:>6.3f} "
                f"{fc:>6} {rec_str:>4} {b['recommended_stake']:>6.1f}"
            )
