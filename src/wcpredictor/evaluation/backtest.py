"""World Cup holdout backtest.

For each World Cup year in [2010, 2014, 2018, 2022]:
  - train = all matches with date < first match of that WC
  - test  = all group + knockout matches of that WC
  - Fit Poisson and Dixon-Coles on train, evaluate on test

Hard assertion: train.date.max() < test.date.min()  (no leakage)
Elo runs continuously across all history; only model fits are per-fold.

Baselines:
  - most_common : always predict the most frequent outcome in train set
  - elo_only    : predict win/draw/loss based purely on Elo diff sign + threshold

Calibration (DC only):
  - Temperature T is fit on a pre-tournament validation slice (last
    DC_CAL_VALIDATION_YEARS years of training data), never on the holdout.
  - ECE is reported before/after calibration on that validation slice.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from wcpredictor.config import DATA_PROCESSED, DC_CAL_VALIDATION_YEARS, ENSEMBLE_POOL
from wcpredictor.data.download import download_results
from wcpredictor.data.load_matches import load_matches
from wcpredictor.evaluation.metrics import (
    accuracy,
    brier,
    exact_score_logscore,
    goal_mae,
    goal_rmse,
    log_loss,
    macro_f1,
    topn_hit_rate,
)
from wcpredictor.features.elo import compute_elo
from wcpredictor.models.calibration import apply as cal_apply
from wcpredictor.models.calibration import expected_calibration_error, fit_temperature
from wcpredictor.models.dixon_coles import fit as dc_fit
from wcpredictor.models.dixon_coles import predict_one as dc_predict_one
from wcpredictor.models.ensemble import (
    combine_matrices as ens_combine_matrices,
    combine_probs as ens_combine_probs,
    fit_weights as ens_fit_weights,
    matrix_to_lambdas,
    matrix_to_top_scorelines,
)
from wcpredictor.models.logistic import fit as log_fit
from wcpredictor.models.logistic import predict_proba as log_predict
from wcpredictor.models.poisson import fit as poisson_fit
from wcpredictor.models.poisson import predict_one

# Approximate first-match dates for each World Cup
_WC_START: dict[int, str] = {
    2010: "2010-06-11",
    2014: "2014-06-12",
    2018: "2018-06-14",
    2022: "2022-11-20",
}


def _label(ga: int, gb: int) -> int:
    if ga > gb:
        return 0  # team_a win
    if ga == gb:
        return 1  # draw
    return 2  # team_b win


def _most_common_baseline(train_labels: list[int], n: int) -> list[list[float]]:
    counts = [train_labels.count(c) for c in (0, 1, 2)]
    total = sum(counts)
    probs = [c / total for c in counts]
    return [probs] * n


def _elo_baseline_probs(elo_diffs: list[float]) -> list[list[float]]:
    result = []
    for d in elo_diffs:
        p_win = 1.0 / (1.0 + 10.0 ** (-d / 400.0))
        p_draw = 0.22
        p_win = p_win * (1 - p_draw)
        p_loss = 1.0 - p_win - p_draw
        result.append([max(p_win, 0.0), p_draw, max(p_loss, 0.0)])
    return result


def _score_model(
    labels: list[int],
    probs: list[list[float]],
    true_a: list[int],
    true_b: list[int],
    pred_a: list[float],
    pred_b: list[float],
    score_matrices: list[list[list[float]]],
    top_scorelines_list: list[list[dict]],
) -> dict:
    y_pred = [int(np.argmax(p)) for p in probs]
    return {
        "log_loss": round(log_loss(labels, probs), 4),
        "brier": round(brier(labels, probs), 4),
        "accuracy": round(accuracy(labels, y_pred), 4),
        "macro_f1": round(macro_f1(labels, y_pred), 4),
        "goal_mae": round(goal_mae(true_a, pred_a, true_b, pred_b), 4),
        "goal_rmse": round(goal_rmse(true_a, pred_a, true_b, pred_b), 4),
        "exact_score_logscore": round(exact_score_logscore(true_a, true_b, score_matrices), 4),
        "top5_hit_rate": round(topn_hit_rate(true_a, true_b, top_scorelines_list), 4),
    }


def backtest_world_cups(years: list[int] | None = None) -> dict:
    if years is None:
        years = [2010, 2014, 2018, 2022]

    from wcpredictor.config import DATA_RAW
    if not (DATA_RAW / "results.csv").exists():
        download_results()

    matches = load_matches()
    elo_all, _final_ratings = compute_elo(matches)

    results = {}
    for year in years:
        wc_start = pd.Timestamp(_WC_START[year])
        cal_start = wc_start - pd.DateOffset(years=DC_CAL_VALIDATION_YEARS)

        train_matches = matches[matches["date"] < wc_start]
        test_matches = matches[
            (matches["date"] >= wc_start)
            & matches["is_world_cup"]
            & (matches["date"].dt.year == year)
        ]

        if test_matches.empty:
            warnings.warn(f"No WC matches found for {year}; skipping.")
            continue

        # Hard leakage assertion
        assert train_matches["date"].max() < test_matches["date"].min(), (
            f"LEAKAGE DETECTED for {year}: train/test dates overlap!"
        )

        train_elo = elo_all[elo_all["date"] < wc_start]

        # ── Poisson model (unchanged from Phase 1) ──────────────────────────
        base, beta_param = poisson_fit(train_elo)

        test_elo = elo_all[
            (elo_all["date"] >= wc_start) & (elo_all["date"].dt.year == year)
        ]
        test_elo = test_elo[test_elo["match_id"].isin(test_matches["match_id"])]

        if test_elo.empty:
            warnings.warn(f"No Elo features for WC {year} test set; skipping.")
            continue

        labels, probs, elo_diffs = [], [], []
        true_a, true_b, pred_a, pred_b = [], [], [], []
        score_matrices, top_scorelines_list = [], []

        for _, row in test_elo.iterrows():
            ga, gb = int(row["goals_a"]), int(row["goals_b"])
            d = float(row["elo_diff_adj"])
            p = predict_one(d, base, beta_param)

            labels.append(_label(ga, gb))
            probs.append([p["p_win"], p["p_draw"], p["p_loss"]])
            elo_diffs.append(d)
            true_a.append(ga)
            true_b.append(gb)
            pred_a.append(p["lambda_a"])
            pred_b.append(p["lambda_b"])
            score_matrices.append(p["score_matrix"])
            top_scorelines_list.append(p["top_scorelines"])

        train_labels = [_label(int(r["goals_a"]), int(r["goals_b"])) for _, r in train_elo.iterrows()]
        mc_probs = _most_common_baseline(train_labels, len(labels))
        elo_probs = _elo_baseline_probs(elo_diffs)

        # ── Dixon-Coles model ───────────────────────────────────────────────
        dc_params = dc_fit(train_matches, ref_date=wc_start)

        labels_dc, probs_dc = [], []
        true_a_dc, true_b_dc, pred_a_dc, pred_b_dc = [], [], [], []
        sc_dc, top_dc = [], []

        for _, row in test_elo.iterrows():
            ga, gb = int(row["goals_a"]), int(row["goals_b"])
            p = dc_predict_one(dc_params, row["team_a"], row["team_b"], bool(row["neutral"]))

            labels_dc.append(_label(ga, gb))
            probs_dc.append([p["p_win"], p["p_draw"], p["p_loss"]])
            true_a_dc.append(ga)
            true_b_dc.append(gb)
            pred_a_dc.append(p["lambda_a"])
            pred_b_dc.append(p["lambda_b"])
            sc_dc.append(p["score_matrix"])
            top_dc.append(p["top_scorelines"])

        # ── Calibration (on pre-tournament validation slice) ────────────────
        val_elo = elo_all[
            (elo_all["date"] >= cal_start) & (elo_all["date"] < wc_start)
        ]
        val_labels_dc, val_probs_dc = [], []
        for _, row in val_elo.iterrows():
            ga, gb = int(row["goals_a"]), int(row["goals_b"])
            p = dc_predict_one(dc_params, row["team_a"], row["team_b"], bool(row["neutral"]))
            val_labels_dc.append(_label(ga, gb))
            val_probs_dc.append([p["p_win"], p["p_draw"], p["p_loss"]])

        if val_labels_dc:
            T = fit_temperature(val_labels_dc, val_probs_dc)
            ece_before = round(expected_calibration_error(val_labels_dc, val_probs_dc), 4)
            ece_after = round(expected_calibration_error(val_labels_dc, cal_apply(val_probs_dc, T)), 4)
        else:
            T, ece_before, ece_after = 1.0, 0.0, 0.0

        probs_dc_cal = cal_apply(probs_dc, T) if probs_dc else []

        # ── Ensemble (leakage-safe stacking) ───────────────────────────────
        # Step 1: early fits on data < cal_start
        early_train_elo = elo_all[elo_all["date"] < cal_start]
        early_train_matches = matches[matches["date"] < cal_start]

        early_base, early_beta = poisson_fit(early_train_elo)
        early_dc_params = dc_fit(early_train_matches, ref_date=cal_start)
        early_labels_for_log = [_label(int(r.goals_a), int(r.goals_b))
                                 for _, r in early_train_elo.iterrows()]
        early_log_scaler, early_log_model = log_fit(early_train_elo, early_labels_for_log)

        # Step 2: out-of-time member predictions on validation slice
        val_labels_ens: list[int] = []
        val_p_poi: list[list[float]] = []
        val_p_dc2: list[list[float]] = []

        for _, row in val_elo.iterrows():
            ga2, gb2 = int(row.goals_a), int(row.goals_b)
            val_labels_ens.append(_label(ga2, gb2))
            vp = predict_one(float(row.elo_diff_adj), early_base, early_beta)
            val_p_poi.append([vp["p_win"], vp["p_draw"], vp["p_loss"]])
            vd = dc_predict_one(early_dc_params, row.team_a, row.team_b, bool(row.neutral))
            val_p_dc2.append([vd["p_win"], vd["p_draw"], vd["p_loss"]])

        val_p_log: list[list[float]] = (
            log_predict(early_log_scaler, early_log_model, val_elo)
            if len(val_elo) > 0 else []
        )

        # Step 3: fit weights + temperature on validation predictions
        if val_labels_ens:
            member_probs_val = [val_p_poi, val_p_dc2, val_p_log]
            ens_weights = ens_fit_weights(member_probs_val, val_labels_ens, pool=ENSEMBLE_POOL)
            val_ens = ens_combine_probs(member_probs_val, ens_weights, pool=ENSEMBLE_POOL)
            T_ens = fit_temperature(val_labels_ens, val_ens)
            ece_ens_before = round(expected_calibration_error(val_labels_ens, val_ens), 4)
            ece_ens_after = round(expected_calibration_error(val_labels_ens, cal_apply(val_ens, T_ens)), 4)
        else:
            ens_weights = np.array([1 / 3, 1 / 3, 1 / 3])
            T_ens, ece_ens_before, ece_ens_after = 1.0, 0.0, 0.0

        # Step 4: full-data logistic (Poisson + DC already fit above on < wc_start)
        train_labels_for_log = [_label(int(r.goals_a), int(r.goals_b))
                                 for _, r in train_elo.iterrows()]
        log_scaler, log_model = log_fit(train_elo, train_labels_for_log)
        probs_log_test: list[list[float]] = log_predict(log_scaler, log_model, test_elo)

        # Step 5: combine members on test set with frozen weights
        member_probs_test = [probs, probs_dc, probs_log_test]
        ens_probs = ens_combine_probs(member_probs_test, ens_weights, pool=ENSEMBLE_POOL)
        ens_probs_cal = cal_apply(ens_probs, T_ens)

        # Score matrices: Poisson + DC only (logistic has none); renormalize weights
        w_score = ens_weights[:2].copy()
        w_score /= w_score.sum()
        ens_matrices = ens_combine_matrices([score_matrices, sc_dc], w_score)

        ens_pred_a = [matrix_to_lambdas(m)[0] for m in ens_matrices]
        ens_pred_b = [matrix_to_lambdas(m)[1] for m in ens_matrices]
        ens_top_sc = [matrix_to_top_scorelines(m) for m in ens_matrices]

        # ── Compile fold results ────────────────────────────────────────────
        fold: dict = {
            "n_matches": len(labels),
            "model": _score_model(
                labels, probs, true_a, true_b, pred_a, pred_b, score_matrices, top_scorelines_list
            ),
            "model_dc": _score_model(
                labels_dc, probs_dc, true_a_dc, true_b_dc, pred_a_dc, pred_b_dc, sc_dc, top_dc
            ),
            "model_dc_cal": {
                **_score_model(
                    labels_dc, probs_dc_cal, true_a_dc, true_b_dc, pred_a_dc, pred_b_dc, sc_dc, top_dc
                ),
                "temperature": round(T, 4),
                "ece_before_val": ece_before,
                "ece_after_val": ece_after,
            },
            "model_ensemble": _score_model(
                labels_dc, ens_probs, true_a_dc, true_b_dc,
                ens_pred_a, ens_pred_b, ens_matrices, ens_top_sc,
            ),
            "model_ensemble_cal": {
                **_score_model(
                    labels_dc, ens_probs_cal, true_a_dc, true_b_dc,
                    ens_pred_a, ens_pred_b, ens_matrices, ens_top_sc,
                ),
                "temperature": round(T_ens, 4),
                "ece_before_val": ece_ens_before,
                "ece_after_val": ece_ens_after,
                "weights_poisson": round(float(ens_weights[0]), 4),
                "weights_dc": round(float(ens_weights[1]), 4),
                "weights_logistic": round(float(ens_weights[2]), 4),
                "pool": ENSEMBLE_POOL,
            },
            "baseline_most_common": {
                "log_loss": round(log_loss(labels, mc_probs), 4),
                "brier": round(brier(labels, mc_probs), 4),
                "accuracy": round(accuracy(labels, [int(np.argmax(p)) for p in mc_probs]), 4),
            },
            "baseline_elo_only": {
                "log_loss": round(log_loss(labels, elo_probs), 4),
                "brier": round(brier(labels, elo_probs), 4),
                "accuracy": round(accuracy(labels, [int(np.argmax(p)) for p in elo_probs]), 4),
            },
            "base": round(base, 5),
            "beta": round(beta_param, 6),
        }
        results[str(year)] = fold

    return results


def _print_report(results: dict) -> None:
    print("\n=== World Cup Holdout Backtest ===")
    for year, fold in results.items():
        n = fold["n_matches"]
        m = fold["model"]
        dc = fold["model_dc"]
        dcc = fold["model_dc_cal"]
        ens = fold.get("model_ensemble", {})
        ensc = fold.get("model_ensemble_cal", {})
        mc = fold["baseline_most_common"]
        elo = fold["baseline_elo_only"]

        print(f"\n--- {year} (n={n}) ---")
        print(f"  Poisson    log_loss={m['log_loss']:.4f}  brier={m['brier']:.4f}  "
              f"acc={m['accuracy']:.3f}  goal_mae={m['goal_mae']:.3f}  "
              f"exact_log={m['exact_score_logscore']:.3f}  top5={m['top5_hit_rate']:.3f}")
        print(f"  DC         log_loss={dc['log_loss']:.4f}  brier={dc['brier']:.4f}  "
              f"acc={dc['accuracy']:.3f}  goal_mae={dc['goal_mae']:.3f}  "
              f"exact_log={dc['exact_score_logscore']:.3f}  top5={dc['top5_hit_rate']:.3f}")
        print(f"  DC+Cal     log_loss={dcc['log_loss']:.4f}  brier={dcc['brier']:.4f}  "
              f"acc={dcc['accuracy']:.3f}  T={dcc['temperature']:.3f}  "
              f"ECE {dcc['ece_before_val']:.3f}→{dcc['ece_after_val']:.3f}")
        if ens:
            print(f"  Ensemble   log_loss={ens['log_loss']:.4f}  brier={ens['brier']:.4f}  "
                  f"acc={ens['accuracy']:.3f}  goal_mae={ens['goal_mae']:.3f}  "
                  f"exact_log={ens['exact_score_logscore']:.3f}  top5={ens['top5_hit_rate']:.3f}")
        if ensc:
            print(f"  Ens+Cal    log_loss={ensc['log_loss']:.4f}  brier={ensc['brier']:.4f}  "
                  f"acc={ensc['accuracy']:.3f}  T={ensc['temperature']:.3f}  "
                  f"ECE {ensc['ece_before_val']:.3f}→{ensc['ece_after_val']:.3f}  "
                  f"w=[poi={ensc['weights_poisson']:.2f} dc={ensc['weights_dc']:.2f} "
                  f"log={ensc['weights_logistic']:.2f}]")
        print(f"  MostComm   log_loss={mc['log_loss']:.4f}  brier={mc['brier']:.4f}  "
              f"acc={mc['accuracy']:.3f}")
        print(f"  EloOnly    log_loss={elo['log_loss']:.4f}  brier={elo['brier']:.4f}  "
              f"acc={elo['accuracy']:.3f}")


if __name__ == "__main__":
    res = backtest_world_cups()
    _print_report(res)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    out = DATA_PROCESSED / "backtest_report.json"
    out.write_text(json.dumps(res, indent=2))
    print(f"\nReport written to {out}")
