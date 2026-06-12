"""Honest EV backtest over historical WC odds.

Reads backtest_permatch_probs.csv (produced by backtest.py --dump-probs) and
runs a threshold grid for 1x2, ah, and total markets. Settles using markets/asian.py
semantics. Reports ROI, bootstrap 95% CI, haircut sensitivity, and home-bias diagnosis.

Usage:
    uv run python -m wcpredictor.evaluation.ev_backtest
    uv run python -m wcpredictor.evaluation.ev_backtest --min-ev 0.05

Output: data/processed/ev_backtest_report.json
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from wcpredictor.config import DATA_PROCESSED


_EPS = 1e-12
_THRESHOLDS = [0.0, 0.02, 0.05, 0.10]
_HAIRCUTS = [0.0, 0.02, 0.05]  # fraction shaved from raw odds (soft-book proxy)
_N_BOOTSTRAP = 2000
_BOOTSTRAP_SEED = 42


# ─── Settlement ────────────────────────────────────────────────────────────────

def _settle_1x2(label: int, side: str) -> float:
    """Return 1 on win, 0 on loss (no push in 1X2)."""
    side_label = {"home": 0, "draw": 1, "away": 2}
    return 1.0 if label == side_label.get(side, -1) else 0.0


def _settle_ah_raw(goals_a: int, goals_b: int, ah_line: float, side: str) -> float:
    """Settle Asian handicap bet using markets/asian.py settlement semantics.

    Returns fractional return per unit staked:
      full win = 1.0, half win = 0.5, push = 0.0 (stake returned as 0-EV),
      half loss = -0.5, full loss = -1.0.

    (Positive = won; negative = lost; caller adds 1 + return to get payout.)
    """
    from wcpredictor.markets.asian import asian_handicap

    mat_1 = [[0.0] * (goals_b + 2) for _ in range(goals_a + 2)]
    mat_1[goals_a][goals_b] = 1.0
    s = asian_handicap(mat_1, side=side, line=ah_line)
    ret = (s["p_win"] + 0.5 * s["p_half_win"]) - (s["p_loss"] + 0.5 * s["p_half_loss"])
    # payout per unit staked: win→ odds, half_win→ 0.5*odds+0.5, push→1, half_loss→0.5, loss→0
    # return for EV purposes: price*(p_win+0.5*p_half_win) + 0.5*p_half_win+p_push+0.5*p_half_loss - 1
    # simplified settlement outcome (for flat-stake ROI):
    #   won if ret > 0, push if ret == 0, lost if ret < 0
    # Return as a settlement dict with p_ fields:
    return s


def _settle_total_raw(goals_a: int, goals_b: int, ou_line: float, side: str) -> dict:
    """Settle Asian total bet using markets/asian.py settlement semantics."""
    from wcpredictor.markets.asian import asian_total

    total = goals_a + goals_b
    mat_1 = [[0.0] * (total + 2) for _ in range(1)]
    mat_1[0][total] = 1.0
    return asian_total(mat_1, side=side, line=ou_line)


# ─── EV and bet outcome ────────────────────────────────────────────────────────

def _ev_1x2(model_p: float, price: float, haircut: float = 0.0) -> float:
    """EV per unit staked at price shaved by haircut fraction."""
    adjusted = price * (1.0 - haircut)
    return adjusted * model_p - 1.0


def _ev_asian(model_fair_odds: float, price: float, haircut: float = 0.0) -> float:
    """EV for AH/total side: (shaved_price - fair_odds) * p_effective."""
    adjusted = price * (1.0 - haircut)
    if not math.isfinite(model_fair_odds) or model_fair_odds <= 1.0:
        return float("nan")
    # p_effective ≈ 1/fair_odds for whole/half lines (simplified for cross-line EV)
    p_eff = 1.0 / model_fair_odds
    return adjusted * p_eff - 1.0


def _pnl_1x2(label: int, side: str, price: float, haircut: float = 0.0) -> float:
    """P&L per unit staked for a 1X2 bet that was placed."""
    adjusted = price * (1.0 - haircut)
    win = _settle_1x2(label, side)
    return adjusted * win - 1.0


def _pnl_asian_settle(settle: dict, price: float, haircut: float = 0.0) -> float:
    """P&L per unit staked using full settlement dict from asian.py."""
    from wcpredictor.markets.edge import ev_per_unit
    adjusted = price * (1.0 - haircut)
    return ev_per_unit(settle, adjusted)


# ─── Bootstrap CI ──────────────────────────────────────────────────────────────

def _bootstrap_roi_ci(
    pnl_series: list[float],
    n_bootstrap: int = _N_BOOTSTRAP,
    seed: int = _BOOTSTRAP_SEED,
    level: float = 0.95,
) -> tuple[float, float]:
    """Paired-bootstrap 95% CI on ROI.  Returns (lower_bound, upper_bound)."""
    if len(pnl_series) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    arr = np.array(pnl_series, dtype=float)
    boot_means = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(arr), size=len(arr))
        boot_means.append(float(np.mean(arr[idx])))
    boot_means.sort()
    alpha = (1.0 - level) / 2.0
    lo = boot_means[int(alpha * n_bootstrap)]
    hi = boot_means[int((1.0 - alpha) * n_bootstrap)]
    return (round(lo, 4), round(hi, 4))


# ─── Main EV grid ──────────────────────────────────────────────────────────────

def run_ev_backtest(
    probs_csv: Path | None = None,
    n_bootstrap: int = _N_BOOTSTRAP,
    seed: int = _BOOTSTRAP_SEED,
) -> dict:
    """Run EV backtest grid over historical WC odds.

    Parameters
    ----------
    probs_csv  : path to backtest_permatch_probs.csv; defaults to DATA_PROCESSED path.
    n_bootstrap: number of bootstrap resamples.
    seed       : random seed for reproducibility.

    Returns
    -------
    dict with keys:
        grid       : list of rows (market, threshold, haircut, n_bets, roi, ci_lo, ci_hi, ...)
        bias       : home vs. away / favorite vs. dog decomposition
        verdict    : string summary of findings
        folds      : per-fold ROI breakdown
    """
    if probs_csv is None:
        probs_csv = DATA_PROCESSED / "backtest_permatch_probs.csv"

    if not probs_csv.exists():
        raise FileNotFoundError(
            f"Per-match probs CSV not found: {probs_csv}\n"
            "Run: uv run python -m wcpredictor.evaluation.backtest --dump-probs"
        )

    df = pd.read_csv(probs_csv)
    required = ["fold", "team_a", "team_b", "goals_a", "goals_b", "label",
                "ens_mkt_p_win", "ens_mkt_p_draw", "ens_mkt_p_loss",
                "model_p_win", "model_p_draw", "model_p_loss"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in probs CSV: {missing}")

    grid_rows: list[dict] = []
    fold_rows: list[dict] = []
    bias_rows: list[dict] = []

    for haircut in _HAIRCUTS:
        for tau in _THRESHOLDS:
            for market in ("1x2", "ah", "total"):
                pnl_all: list[float] = []
                pnl_home: list[float] = []
                pnl_away: list[float] = []
                pnl_fav: list[float] = []
                pnl_dog: list[float] = []
                fold_pnl: dict[int, list[float]] = {}

                for _, row in df.iterrows():
                    goals_a = int(row.goals_a)
                    goals_b = int(row.goals_b)
                    label = int(row.label)
                    fold = int(row.fold)

                    if market == "1x2":
                        # Use ens_mkt probs if available, else model probs
                        p_win = row.ens_mkt_p_win if not pd.isna(row.ens_mkt_p_win) else row.model_p_win
                        p_draw = row.ens_mkt_p_draw if not pd.isna(row.ens_mkt_p_draw) else row.model_p_draw
                        p_loss = row.ens_mkt_p_loss if not pd.isna(row.ens_mkt_p_loss) else row.model_p_loss

                        for side, mp, price_col in (
                            ("home", p_win, "h_avg"),
                            ("draw", p_draw, "d_avg"),
                            ("away", p_loss, "a_avg"),
                        ):
                            raw_price = row.get(price_col, float("nan"))
                            if pd.isna(raw_price) or float(raw_price) <= 1.0:
                                continue
                            ev = _ev_1x2(float(mp), float(raw_price), haircut)
                            if pd.isna(ev) or ev < tau:
                                continue
                            pnl = _pnl_1x2(label, side, float(raw_price), haircut)
                            pnl_all.append(pnl)
                            fold_pnl.setdefault(fold, []).append(pnl)
                            if side == "home":
                                pnl_home.append(pnl)
                                if float(raw_price) < 2.0:
                                    pnl_fav.append(pnl)
                                else:
                                    pnl_dog.append(pnl)
                            elif side == "away":
                                pnl_away.append(pnl)

                    elif market == "ah":
                        ah_line = row.get("ah_line", float("nan"))
                        ah_ho = row.get("ah_home_odds", float("nan"))
                        ah_ao = row.get("ah_away_odds", float("nan"))
                        ah_fair_h = row.get("ah_fair_odds_home", float("nan"))
                        ah_fair_a = row.get("ah_fair_odds_away", float("nan"))
                        if any(pd.isna(x) for x in (ah_line, ah_ho, ah_ao)):
                            continue
                        ah_line_f = float(ah_line)
                        for side, raw_price, fair_odds in (
                            ("home", float(ah_ho), float(ah_fair_h)),
                            ("away", float(ah_ao), float(ah_fair_a)),
                        ):
                            if raw_price <= 1.0 or pd.isna(fair_odds):
                                continue
                            ev = _ev_asian(fair_odds, raw_price, haircut)
                            if pd.isna(ev) or ev < tau:
                                continue
                            settle = _settle_ah_raw(goals_a, goals_b, ah_line_f, side)
                            pnl = _pnl_asian_settle(settle, raw_price, haircut)
                            pnl_all.append(pnl)
                            fold_pnl.setdefault(fold, []).append(pnl)
                            if side == "home":
                                pnl_home.append(pnl)
                                if raw_price < 2.0:
                                    pnl_fav.append(pnl)
                                else:
                                    pnl_dog.append(pnl)
                            else:
                                pnl_away.append(pnl)

                    elif market == "total":
                        ou_line = row.get("ou_line", float("nan"))
                        ov_p = row.get("over_odds", float("nan"))
                        un_p = row.get("under_odds", float("nan"))
                        fair_ov = row.get("ou_fair_odds_over", float("nan"))
                        fair_un = row.get("ou_fair_odds_under", float("nan"))
                        if any(pd.isna(x) for x in (ou_line, ov_p, un_p)):
                            continue
                        ou_line_f = float(ou_line)
                        for side, raw_price, fair_odds in (
                            ("over", float(ov_p), float(fair_ov)),
                            ("under", float(un_p), float(fair_un)),
                        ):
                            if raw_price <= 1.0 or pd.isna(fair_odds):
                                continue
                            ev = _ev_asian(fair_odds, raw_price, haircut)
                            if pd.isna(ev) or ev < tau:
                                continue
                            settle = _settle_total_raw(goals_a, goals_b, ou_line_f, side)
                            pnl = _pnl_asian_settle(settle, raw_price, haircut)
                            pnl_all.append(pnl)
                            fold_pnl.setdefault(fold, []).append(pnl)

                n = len(pnl_all)
                roi = round(float(np.mean(pnl_all)), 4) if n > 0 else float("nan")
                ci_lo, ci_hi = _bootstrap_roi_ci(pnl_all, n_bootstrap, seed)

                grid_rows.append({
                    "market": market,
                    "threshold": tau,
                    "haircut": haircut,
                    "n_bets": n,
                    "roi": roi,
                    "ci_lo": ci_lo,
                    "ci_hi": ci_hi,
                    "n_home": len(pnl_home) if market in ("1x2", "ah") else None,
                    "n_away": len(pnl_away) if market in ("1x2", "ah") else None,
                    "roi_home": round(float(np.mean(pnl_home)), 4) if pnl_home else float("nan"),
                    "roi_away": round(float(np.mean(pnl_away)), 4) if pnl_away else float("nan"),
                    "n_fav": len(pnl_fav) if market in ("1x2", "ah") else None,
                    "n_dog": len(pnl_dog) if market in ("1x2", "ah") else None,
                    "roi_fav": round(float(np.mean(pnl_fav)), 4) if pnl_fav else float("nan"),
                    "roi_dog": round(float(np.mean(pnl_dog)), 4) if pnl_dog else float("nan"),
                })

                for fold_yr, pnls in sorted(fold_pnl.items()):
                    fold_rows.append({
                        "market": market,
                        "threshold": tau,
                        "haircut": haircut,
                        "fold": fold_yr,
                        "n_bets": len(pnls),
                        "roi": round(float(np.mean(pnls)), 4) if pnls else float("nan"),
                    })

    # Verdict
    # Look for any τ=0, haircut=0 row with ci_lo > 0 (credible edge)
    credible_markets = []
    for row in grid_rows:
        if row["threshold"] == 0.0 and row["haircut"] == 0.0:
            if isinstance(row["ci_lo"], float) and row["ci_lo"] > 0 and row["n_bets"] >= 10:
                credible_markets.append(row["market"])

    # Check haircut survival for credible markets
    haircut_survivor = []
    for mkt in credible_markets:
        for hc in (0.02, 0.05):
            hc_row = next(
                (r for r in grid_rows if r["market"] == mkt and r["threshold"] == 0.0 and r["haircut"] == hc),
                None,
            )
            if hc_row and isinstance(hc_row["ci_lo"], float) and hc_row["ci_lo"] > 0:
                haircut_survivor.append(f"{mkt}@{int(hc*100)}%haircut")

    if haircut_survivor:
        verdict = (
            f"CREDIBLE EDGE: {', '.join(credible_markets)} survive haircut — "
            f"use quarter-Kelly staking tier. Survivors: {', '.join(haircut_survivor)}"
        )
        sizing_tier = "quarter_kelly"
    elif credible_markets:
        verdict = (
            f"INCONCLUSIVE: {', '.join(credible_markets)} show positive CI at 0% haircut "
            f"but not after SG-Pools-realistic haircut — use minimum-stake experiment tier."
        )
        sizing_tier = "min_stake"
    else:
        verdict = (
            "NO CREDIBLE EDGE at 0% haircut across any market — "
            "use minimum-stake experiment tier (~$10/bet) per pre-committed plan."
        )
        sizing_tier = "min_stake"

    return {
        "grid": grid_rows,
        "folds": fold_rows,
        "verdict": verdict,
        "sizing_tier": sizing_tier,
        "credible_markets": credible_markets,
        "haircut_survivors": haircut_survivor,
    }


if __name__ == "__main__":
    import argparse as _argparse

    _ap = _argparse.ArgumentParser(description="EV backtest over historical WC odds")
    _ap.add_argument("--min-ev", type=float, default=0.0,
                     help="Minimum EV threshold to print (default 0.0)")
    _args = _ap.parse_args()

    report = run_ev_backtest()

    print("\n=== EV Backtest — Threshold Grid ===")
    print(f"{'Market':<8} {'τ':>5} {'HC':>5} {'n':>5} {'ROI':>8} {'CI_lo':>8} {'CI_hi':>8}")
    for row in report["grid"]:
        if row["roi"] < _args.min_ev - 0.01 and not math.isfinite(row["ci_hi"]):
            continue
        print(
            f"{row['market']:<8} {row['threshold']:>5.2f} {row['haircut']:>5.2f} "
            f"{row['n_bets']:>5} {row['roi']:>8.4f} {row['ci_lo']:>8.4f} {row['ci_hi']:>8.4f}"
        )

    print(f"\nVerdict: {report['verdict']}")
    print(f"Sizing tier: {report['sizing_tier']}")

    out_path = DATA_PROCESSED / "ev_backtest_report.json"
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nReport written to {out_path}")
