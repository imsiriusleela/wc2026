"""WC2026 live scoring and rolling recalibration.

Entry-point usage:
    uv run python -m wcpredictor.evaluation.live                    # score + recalibrate
    uv run python -m wcpredictor.evaluation.live --results=path.csv # with custom results

Flow
----
1. Re-pull latest results from martj42 master (or accept user-provided CSV).
2. Load saved pre-match predictions from data/processed/wc2026_predictions_*.csv.
3. Match predictions to completed WC2026 results (join on date + team pair).
4. Compute log_loss, Brier, accuracy on completed matches.
5. Recalibrate temperature on the completed WC2026 window.
6. Write data/processed/wc2026_scorecard.json.

Leakage safety: only matches where a pre-match prediction CSV exists AND the
match date is strictly before as_of_date are scored.  Temperature is never
fit on the match being predicted.
"""

from __future__ import annotations

import json
import ssl
import sys
import urllib.request
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from wcpredictor.config import DATA_PROCESSED, DATA_RAW, RESULTS_URL_FALLBACK
from wcpredictor.evaluation.metrics import accuracy, brier, log_loss
from wcpredictor.models.calibration import (
    apply as cal_apply,
    expected_calibration_error,
    fit_temperature,
)

_FIXTURES_CSV = DATA_RAW / "wc2026_fixtures.csv"


# ── helpers ──────────────────────────────────────────────────────────────────


def _label(ga: int, gb: int) -> int:
    if ga > gb:
        return 0
    if ga == gb:
        return 1
    return 2


def _load_latest_results(results_csv: Path | None = None) -> pd.DataFrame:
    """Return all known results up to now.

    If results_csv is provided, use it (user-supplied or existing download).
    Otherwise try to download martj42 master (RESULTS_URL_FALLBACK).
    Falls back to the existing local results.csv if the network request fails.
    """
    local = DATA_RAW / "results.csv"

    if results_csv is not None:
        df = pd.read_csv(results_csv, parse_dates=["date"])
        return df

    try:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        with urllib.request.urlopen(RESULTS_URL_FALLBACK, context=ctx, timeout=30) as resp:
            raw = resp.read()
        local.write_bytes(raw)
    except Exception as exc:
        warnings.warn(f"Could not refresh results from martj42 master: {exc}")

    if not local.exists():
        raise FileNotFoundError(
            f"results.csv not found at {local}. "
            "Run: uv run python -m wcpredictor.data.download"
        )
    return pd.read_csv(local, parse_dates=["date"])


def _load_predictions() -> pd.DataFrame:
    """Concatenate all wc2026_predictions_*.csv files in data/processed/."""
    files = sorted(DATA_PROCESSED.glob("wc2026_predictions_*.csv"))
    if not files:
        return pd.DataFrame()
    frames = [pd.read_csv(f, parse_dates=["date"]) for f in files]
    combined = pd.concat(frames, ignore_index=True)
    # Keep the earliest prediction per (team_a, team_b, date)
    combined = combined.sort_values("date").drop_duplicates(
        subset=["team_a", "team_b", "date"], keep="first"
    )
    return combined


def _match_key(ta: str, tb: str, date: pd.Timestamp) -> str:
    return f"{date.date()}|{ta}|{tb}"


# ── main pipeline ─────────────────────────────────────────────────────────────


def score_completed_matches(
    predictions: pd.DataFrame,
    results: pd.DataFrame,
    as_of_date: str,
) -> list[dict]:
    """Join predictions to completed WC2026 results.

    Only includes rows where:
      - A pre-match prediction exists.
      - The match date is strictly before as_of_date (confirmed completed).

    Returns list of match-level dicts with actual + predicted outcomes.
    """
    from wcpredictor.data.normalize_teams import canonical

    as_of = pd.Timestamp(as_of_date)
    if predictions.empty:
        return []

    # Build a date+team lookup from results to avoid pandas StringDtype issues
    results_lookup: dict[tuple, tuple[int, int]] = {}
    for _, r in results.iterrows():
        try:
            d = pd.Timestamp(r["date"]).date()
            h = canonical(str(r["home_team"]))
            a = canonical(str(r["away_team"]))
            gh, ga = int(r["home_score"]), int(r["away_score"])
        except (ValueError, TypeError, KeyError):
            continue
        # Register both orderings; value = (goals_for_team_a, goals_for_team_b)
        results_lookup[(d, h, a)] = (gh, ga)
        results_lookup[(d, a, h)] = (ga, gh)

    matched: list[dict] = []
    for _, pred in predictions.iterrows():
        pred_date = pd.Timestamp(pred["date"])
        if pred_date >= as_of:
            continue  # not yet played

        ta = str(pred["team_a"])
        tb = str(pred["team_b"])
        key = (pred_date.date(), ta, tb)

        if key not in results_lookup:
            continue

        ga, gb = results_lookup[key]
        label = _label(ga, gb)

        def _safe_prob(val: object, default: float = 1 / 3) -> float:
            try:
                v = float(val)  # type: ignore[arg-type]
                return v if v == v else default  # NaN guard
            except (TypeError, ValueError):
                return default

        matched.append({
            "date": pred_date.strftime("%Y-%m-%d"),
            "team_a": ta,
            "team_b": tb,
            "goals_a": ga,
            "goals_b": gb,
            "label": label,
            "p_win": _safe_prob(pred.get("p_win")),
            "p_draw": _safe_prob(pred.get("p_draw")),
            "p_loss": _safe_prob(pred.get("p_loss")),
        })

    return matched


def run_refresh(
    as_of_date: str,
    results_csv: Path | None = None,
) -> dict:
    """Full live pipeline: pull results, score, recalibrate, write scorecard.

    Parameters
    ----------
    as_of_date  : ISO date; only matches before this date are scored.
    results_csv : optional path to a user-supplied results CSV.

    Returns
    -------
    Scorecard dict (also written to data/processed/wc2026_scorecard.json).
    """
    results = _load_latest_results(results_csv)
    predictions = _load_predictions()

    completed = score_completed_matches(predictions, results, as_of_date)

    scorecard: dict = {
        "as_of_date": as_of_date,
        "n_completed": len(completed),
        "log_loss": None,
        "brier": None,
        "accuracy": None,
        "ece_uncal": None,
        "ece_cal": None,
        "temperature": 1.0,
        "matches": completed,
    }

    if completed:
        labels = [m["label"] for m in completed]
        probs = [[m["p_win"], m["p_draw"], m["p_loss"]] for m in completed]
        y_pred = [int(np.argmax(p)) for p in probs]

        T = fit_temperature(labels, probs)
        probs_cal = cal_apply(probs, T)

        scorecard.update({
            "log_loss": round(log_loss(labels, probs), 4),
            "brier": round(brier(labels, probs), 4),
            "accuracy": round(accuracy(labels, y_pred), 4),
            "ece_uncal": round(expected_calibration_error(labels, probs), 4),
            "ece_cal": round(expected_calibration_error(labels, probs_cal), 4),
            "temperature": round(float(T), 4),
        })

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    scorecard_out = DATA_PROCESSED / "wc2026_scorecard.json"
    scorecard_out.write_text(json.dumps(scorecard, indent=2))
    print(f"Scorecard written → {scorecard_out}")
    print(
        f"n_completed={scorecard['n_completed']}  "
        f"log_loss={scorecard['log_loss']}  "
        f"brier={scorecard['brier']}  "
        f"accuracy={scorecard['accuracy']}  "
        f"T={scorecard['temperature']}"
    )
    return scorecard


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="WC2026 live scoring pipeline")
    parser.add_argument("--as-of", default=str(pd.Timestamp.today().date()),
                        help="Cutoff date (ISO); default: today")
    parser.add_argument("--results", default=None,
                        help="Path to a user-supplied results CSV (optional)")
    args = parser.parse_args()

    results_path = Path(args.results) if args.results else None
    run_refresh(args.as_of, results_csv=results_path)
