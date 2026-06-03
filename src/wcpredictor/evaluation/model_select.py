"""Paired bootstrap model comparison for WC holdout per-match data.

Reads data/processed/backtest_permatch.csv (written by backtest.py).
Prints a comparison table and a documented default-model recommendation.

Usage:
    uv run python -m wcpredictor.evaluation.model_select
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from wcpredictor.config import DATA_PROCESSED

_N_BOOT = 10_000
_SEED = 42
# CI upper bound (ens_mkt - ens_cal) must be strictly negative to call "significant"
_SIG_THRESHOLD = 0.0


def _boot_mean_ci(
    arr: np.ndarray,
    n_boot: int = _N_BOOT,
    seed: int = _SEED,
) -> tuple[float, float, float]:
    """Return (mean, ci_lo_2.5, ci_hi_97.5) via percentile bootstrap."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    boot_means = arr[idx].mean(axis=1)
    return (
        float(arr.mean()),
        float(np.percentile(boot_means, 2.5)),
        float(np.percentile(boot_means, 97.5)),
    )


def _boot_win_frac(
    diffs: np.ndarray,
    n_boot: int = _N_BOOT,
    seed: int = _SEED,
) -> float:
    """Fraction of bootstrap resamples where mean(diffs) < 0 (A beats B)."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diffs), size=(n_boot, len(diffs)))
    boot_means = diffs[idx].mean(axis=1)
    return float((boot_means < 0).mean())


def run_model_selection(csv_path: Path | None = None) -> dict:
    """Load per-match data, run paired bootstrap, return results dict.

    Returns
    -------
    dict with keys:
      "pooled_means"  — mean log_loss per model across all folds
      "pairwise"      — pairwise Δlog_loss comparisons with CIs
      "fold_alpha"    — per-fold alpha_odds from backtest_report.json
      "decision"      — chosen default model name
      "rationale"     — human-readable explanation
    """
    if csv_path is None:
        csv_path = DATA_PROCESSED / "backtest_permatch.csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Per-match data not found: {csv_path}\n"
            "Run: uv run python -m wcpredictor.evaluation.backtest"
        )

    df = pd.read_csv(csv_path)
    models = sorted(df["model"].unique().tolist())

    # Pivot to wide form: one row per (fold, match_idx), one column per model
    pivot = df.pivot_table(
        index=["fold", "match_idx", "has_odds"],
        columns="model",
        values="log_loss_i",
        aggfunc="first",
    ).reset_index()
    pivot.columns.name = None

    out: dict = {}

    # ── 1. Per-model pooled mean log-loss with bootstrap CIs ─────────────
    print("\n=== Pooled mean log-loss across WC 2010–2022 ===")
    print(f"{'Model':<15} {'Mean':>8} {'CI95 lo':>10} {'CI95 hi':>10}  n")
    print("─" * 52)
    pooled: dict = {}
    for m in models:
        if m not in pivot.columns:
            continue
        arr = pivot[m].dropna().values
        mean, lo, hi = _boot_mean_ci(arr)
        pooled[m] = {"mean": mean, "ci_lo": lo, "ci_hi": hi, "n": len(arr)}
        print(f"  {m:<13} {mean:>8.4f} [{lo:>8.4f}, {hi:>8.4f}]  {len(arr)}")
    out["pooled_means"] = pooled

    # ── 2. Pairwise Δlog-loss: ens_mkt − competitor ──────────────────────
    print("\n=== Pairwise Δlog-loss = ens_mkt − competitor  (negative ⇒ ens_mkt better) ===")
    print(f"{'Competitor':<12} {'Subset':<16} {'n':>5} {'Δ mean':>8} "
          f"{'CI lo':>9} {'CI hi':>9} {'P(mkt wins)':>13}")
    print("─" * 80)

    pairwise: dict = {}
    subsets = {
        "all folds":    pivot,
        "odds-present": pivot[pivot["has_odds"] == 1],
        "odds-absent":  pivot[pivot["has_odds"] == 0],
    }

    for competitor in ["poisson", "dc_cal", "ens_cal"]:
        if competitor not in pivot.columns or "ens_mkt" not in pivot.columns:
            continue
        pairwise[competitor] = {}
        for subset_name, sub in subsets.items():
            paired = sub[["ens_mkt", competitor]].dropna()
            n = len(paired)
            if n < 10:
                continue
            diffs = paired["ens_mkt"].values - paired[competitor].values
            mean, lo, hi = _boot_mean_ci(diffs)
            wf = _boot_win_frac(diffs)
            sig = "**" if hi < _SIG_THRESHOLD else "  "
            pairwise[competitor][subset_name] = {
                "n": n, "mean": mean, "ci_lo": lo, "ci_hi": hi,
                "p_ens_mkt_better": wf,
            }
            print(f"  {competitor:<10} {subset_name:<16} {n:>5} {mean:>8.4f} "
                  f"[{lo:>7.4f}, {hi:>7.4f}]  {wf:>8.1%}  {sig}")
    out["pairwise"] = pairwise

    # ── 3. Odds-subset decomposition & fold-level alpha summary ──────────
    print("\n=== Fold-level summary (ens_mkt vs ens_cal) ===")
    report_path = DATA_PROCESSED / "backtest_report.json"
    fold_alpha: dict = {}
    if report_path.exists():
        report = json.loads(report_path.read_text())
        print(f"{'Fold':<6} {'alpha_odds':>12} {'ens_cal LL':>12} {'ens_mkt LL':>12} {'Δ':>8}")
        print("─" * 56)
        for yr in ["2010", "2014", "2018", "2022"]:
            if yr not in report:
                continue
            mec = report[yr].get("model_ensemble_cal", {})
            mem = report[yr].get("model_ensemble_market", {})
            alpha = mem.get("alpha_odds", float("nan"))
            ec_ll = mec.get("log_loss", float("nan"))
            em_ll = mem.get("log_loss", float("nan"))
            delta = em_ll - ec_ll if not (pd.isna(ec_ll) or pd.isna(em_ll)) else float("nan")
            fold_alpha[yr] = alpha
            print(f"  {yr:<4} {alpha:>12.4f} {ec_ll:>12.4f} {em_ll:>12.4f} {delta:>8.4f}")
    out["fold_alpha"] = fold_alpha

    # ── 4. Decision ───────────────────────────────────────────────────────
    key = "ens_cal"
    comp_all = pairwise.get(key, {}).get("all folds", {})
    delta_all = comp_all.get("mean", 0.0)
    ci_hi_all = comp_all.get("ci_hi", 1.0)
    p_mkt_better = comp_all.get("p_ens_mkt_better", 0.5)

    if ci_hi_all < _SIG_THRESHOLD:
        decision = "ensemble_mkt"
        rationale = (
            f"ens_mkt vs ens_cal: Δ={delta_all:.4f}, 95% CI upper bound "
            f"{ci_hi_all:.4f} < 0. Statistically significant advantage. "
            "Ship ensemble_mkt; it auto-degrades to ens_cal when no odds."
        )
    else:
        decision = "ensemble"
        rationale = (
            f"ens_mkt vs ens_cal: Δ={delta_all:.4f}, 95% CI [{comp_all.get('ci_lo',0):.4f}, "
            f"{ci_hi_all:.4f}] includes 0 (P(mkt wins)={p_mkt_better:.1%}). "
            "Result within bootstrap noise. Ship robust ensemble (=ens_cal); "
            "market blending auto-activates via ensemble_mkt once odds arrive (alpha≈0.25 in 2022)."
        )

    out["decision"] = decision
    out["rationale"] = rationale

    print(f"\n=== DECISION: default model = '{decision}' ===")
    print(f"Rationale: {rationale}\n")

    return out


if __name__ == "__main__":
    run_model_selection()
