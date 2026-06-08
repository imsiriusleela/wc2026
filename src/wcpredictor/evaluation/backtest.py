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

import csv
import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.optimize import minimize_scalar

from wcpredictor.config import (
    DATA_PROCESSED,
    DC_CAL_VALIDATION_YEARS,
    ENSEMBLE_POOL,
    ODDS_ALPHA_CAP,
    ODDS_ALPHA_PRIOR,
)
from wcpredictor.data.download import download_results
from wcpredictor.data.download_odds import download_odds
from wcpredictor.data.download_wc2010_odds import parse_wc2010_odds
from wcpredictor.data.load_matches import load_matches
from wcpredictor.features.odds import align_odds_to_test, load_wc_odds, merge_odds_features
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
from wcpredictor.features.form import compute_form
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
from wcpredictor.models.gbm import fit as tree_fit
from wcpredictor.models.gbm import predict_proba as tree_predict
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


def _fit_odds_alpha(
    labels: list[int],
    ens_probs: list[list[float]],
    market_probs: list[list[float]],
) -> float:
    """Return α ∈ [0,1] minimising log_loss of (1-α)·ens + α·market on given data."""
    def _neg_ll(alpha: float) -> float:
        blended = [
            [(1 - alpha) * e[i] + alpha * m[i] for i in range(3)]
            for e, m in zip(ens_probs, market_probs)
        ]
        return log_loss(labels, blended)

    res = minimize_scalar(_neg_ll, bounds=(0.0, 1.0), method="bounded")
    return float(res.x)


def _per_match_vec(
    fold: int,
    model_name: str,
    labels: list[int],
    probs: list[list[float]],
    has_odds: bool,
) -> list[dict]:
    """Return one dict per match with per-sample log-loss, Brier, and correct flag."""
    rows = []
    for i, (lbl, p) in enumerate(zip(labels, probs)):
        ll_i = -math.log(max(p[lbl], 1e-10))
        br_i = sum((p[c] - (1.0 if c == lbl else 0.0)) ** 2 for c in range(3))
        rows.append({
            "fold": fold,
            "model": model_name,
            "match_idx": i,
            "label": lbl,
            "log_loss_i": round(ll_i, 6),
            "brier_i": round(br_i, 6),
            "correct_i": int(int(np.argmax(p)) == lbl),
            "has_odds": int(has_odds),
        })
    return rows


def build_wc_stacking_validation(
    year: int,
    matches: pd.DataFrame,
    elo_all: pd.DataFrame,
) -> tuple[list[int], list[list[list[float]]]]:
    """Walk-forward WC stacking validation for ensemble weight fitting.

    For fold year Y, fits all four ensemble members on data strictly before each
    past WC w < Y, then predicts WC w (odds-bearing rows). Leakage-safe by
    construction: each WC w is predicted out-of-time using members trained on
    data before wc_start(w).

    Returns (labels, [poi_probs, dc_probs, log_probs, tree_probs]).
    Returns empty lists when no past WC exists (e.g. 2010 fold).
    """
    past_years = sorted(w for w in _WC_START if w < year)
    if not past_years:
        return [], [[], [], [], []]

    all_labels: list[int] = []
    all_poi: list[list[float]] = []
    all_dc: list[list[float]] = []
    all_log: list[list[float]] = []
    all_tree: list[list[float]] = []

    for w in past_years:
        wc_start_w = pd.Timestamp(_WC_START[w])

        train_elo_w = elo_all[elo_all["date"] < wc_start_w]
        train_matches_w = matches[matches["date"] < wc_start_w]

        if train_elo_w.empty:
            continue

        base_w, beta_w = poisson_fit(train_elo_w)
        dc_params_w = dc_fit(train_matches_w, ref_date=wc_start_w)
        fit_labels = [
            _label(int(r.goals_a), int(r.goals_b)) for _, r in train_elo_w.iterrows()
        ]
        log_scaler_w, log_model_w = log_fit(train_elo_w, fit_labels)
        tree_model_w = tree_fit(train_elo_w, fit_labels)

        test_matches_w = matches[
            (matches["date"] >= wc_start_w)
            & matches["is_world_cup"]
            & (matches["date"].dt.year == w)
        ]
        test_elo_w = elo_all[
            (elo_all["date"] >= wc_start_w) & (elo_all["date"].dt.year == w)
        ]
        test_elo_w = test_elo_w[test_elo_w["match_id"].isin(test_matches_w["match_id"])]

        if test_elo_w.empty:
            continue

        for _, row in test_elo_w.iterrows():
            ga, gb = int(row.goals_a), int(row.goals_b)
            all_labels.append(_label(ga, gb))
            vp = predict_one(float(row.elo_diff_adj), base_w, beta_w)
            all_poi.append([vp["p_win"], vp["p_draw"], vp["p_loss"]])
            vd = dc_predict_one(dc_params_w, row.team_a, row.team_b, bool(row.neutral))
            all_dc.append([vd["p_win"], vd["p_draw"], vd["p_loss"]])

        all_log.extend(log_predict(log_scaler_w, log_model_w, test_elo_w))
        all_tree.extend(tree_predict(tree_model_w, test_elo_w))

    return all_labels, [all_poi, all_dc, all_log, all_tree]


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
    if not (DATA_RAW / "WorldCup_fdco.xlsx").exists():
        download_odds()
    if not (DATA_RAW / "wc2010_odds.csv").exists():
        parse_wc2010_odds()

    matches = load_matches()
    elo_all, _final_ratings = compute_elo(matches)
    form_all, _ = compute_form(matches)
    elo_all = elo_all.merge(form_all, on="match_id", how="left")

    # Load market odds (WC 2010 from betexplorer CSV; 2014/2018/2022 from fdco xlsx)
    odds_df = load_wc_odds()
    # Attach odds_p_win/draw/loss + has_odds to every row; NaN where no WC odds exist.
    # GBM is the only member that uses these cols; HGB handles NaN natively.
    elo_all = merge_odds_features(elo_all, odds_df)

    # Rolling collector for time-aware α fitting (no leakage)
    prev_odds_labels: list[int] = []
    prev_odds_ens_probs: list[list[float]] = []
    prev_odds_market_probs: list[list[float]] = []

    # Per-match collector for model_select.py bootstrap analysis
    permatch_rows: list[dict] = []

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
        early_tree_model = tree_fit(early_train_elo, early_labels_for_log)

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
        val_p_tree: list[list[float]] = (
            tree_predict(early_tree_model, val_elo)
            if len(val_elo) > 0 else []
        )

        # Step 3: fit weights on odds-bearing WC validation; temperature on non-WC slice
        wc_val_labels, wc_val_member_probs = build_wc_stacking_validation(
            year, matches, elo_all
        )
        member_probs_val = [val_p_poi, val_p_dc2, val_p_log, val_p_tree]
        if wc_val_labels:
            ens_weights = ens_fit_weights(wc_val_member_probs, wc_val_labels, pool=ENSEMBLE_POOL)
            if val_labels_ens:
                val_ens = ens_combine_probs(member_probs_val, ens_weights, pool=ENSEMBLE_POOL)
                T_ens = fit_temperature(val_labels_ens, val_ens)
                ece_ens_before = round(expected_calibration_error(val_labels_ens, val_ens), 4)
                ece_ens_after = round(expected_calibration_error(val_labels_ens, cal_apply(val_ens, T_ens)), 4)
            else:
                T_ens, ece_ens_before, ece_ens_after = 1.0, 0.0, 0.0
        elif val_labels_ens:
            ens_weights = ens_fit_weights(member_probs_val, val_labels_ens, pool=ENSEMBLE_POOL)
            val_ens = ens_combine_probs(member_probs_val, ens_weights, pool=ENSEMBLE_POOL)
            T_ens = fit_temperature(val_labels_ens, val_ens)
            ece_ens_before = round(expected_calibration_error(val_labels_ens, val_ens), 4)
            ece_ens_after = round(expected_calibration_error(val_labels_ens, cal_apply(val_ens, T_ens)), 4)
        else:
            ens_weights = np.array([0.25, 0.25, 0.25, 0.25])
            T_ens, ece_ens_before, ece_ens_after = 1.0, 0.0, 0.0

        # Step 4: full-data logistic + tree (Poisson + DC already fit above on < wc_start)
        train_labels_for_log = [_label(int(r.goals_a), int(r.goals_b))
                                 for _, r in train_elo.iterrows()]
        log_scaler, log_model = log_fit(train_elo, train_labels_for_log)
        probs_log_test: list[list[float]] = log_predict(log_scaler, log_model, test_elo)
        tree_model = tree_fit(train_elo, train_labels_for_log)
        probs_tree_test: list[list[float]] = tree_predict(tree_model, test_elo)

        # Step 5: combine members on test set with frozen weights
        member_probs_test = [probs, probs_dc, probs_log_test, probs_tree_test]
        ens_probs = ens_combine_probs(member_probs_test, ens_weights, pool=ENSEMBLE_POOL)
        ens_probs_cal = cal_apply(ens_probs, T_ens)

        # Score matrices: Poisson + DC only (logistic has none); renormalize weights
        w_score = ens_weights[:2].copy()
        w_score /= w_score.sum()
        ens_matrices = ens_combine_matrices([score_matrices, sc_dc], w_score)

        ens_pred_a = [matrix_to_lambdas(m)[0] for m in ens_matrices]
        ens_pred_b = [matrix_to_lambdas(m)[1] for m in ens_matrices]
        ens_top_sc = [matrix_to_top_scorelines(m) for m in ens_matrices]

        # ── Market-odds blending (all four folds: 2010 betexplorer + 2014/2018/2022 fdco) ──
        market_probs_test = align_odds_to_test(odds_df, year, test_elo)
        ens_market_info: dict = {}
        ens_probs_market: list[list[float]] = []

        if market_probs_test is not None:
            # Time-aware α: use prior for 2014 (no past fold with odds), else fit on past folds
            if not prev_odds_labels:
                alpha_odds = ODDS_ALPHA_PRIOR
            else:
                alpha_odds = _fit_odds_alpha(
                    prev_odds_labels, prev_odds_ens_probs, prev_odds_market_probs
                )

            alpha_effective = min(alpha_odds, ODDS_ALPHA_CAP)
            ens_probs_market = [
                [(1 - alpha_effective) * e[i] + alpha_effective * m[i] for i in range(3)]
                for e, m in zip(ens_probs_cal, market_probs_test)
            ]

            ens_market_info = {
                **_score_model(
                    labels_dc, ens_probs_market, true_a_dc, true_b_dc,
                    ens_pred_a, ens_pred_b, ens_matrices, ens_top_sc,
                ),
                "alpha_odds": round(alpha_odds, 4),
                "alpha_effective": round(alpha_effective, 4),
            }

            # Update rolling collector for the next fold (time-aware, no leakage)
            prev_odds_labels.extend(labels_dc)
            prev_odds_ens_probs.extend(ens_probs_cal)
            prev_odds_market_probs.extend(market_probs_test)

        # ── Per-match capture for bootstrap model selection ─────────────────
        _has_odds = year >= 2018  # effective odds split: 2018/2022 vs 2010/2014
        permatch_rows.extend(_per_match_vec(year, "poisson", labels, probs, _has_odds))
        permatch_rows.extend(_per_match_vec(year, "dc_cal", labels_dc, probs_dc_cal, _has_odds))
        permatch_rows.extend(_per_match_vec(year, "ens_cal", labels_dc, ens_probs_cal, _has_odds))
        if ens_probs_market:
            permatch_rows.extend(_per_match_vec(year, "ens_mkt", labels_dc, ens_probs_market, _has_odds))

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
                "weights_tree": round(float(ens_weights[3]), 4),
                "pool": ENSEMBLE_POOL,
            },
            **({"model_ensemble_market": ens_market_info} if ens_market_info else {}),
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

    # Pooled α across all four WCs — best transfer estimate for 2026
    if prev_odds_labels:
        odds_alpha_pooled = float(_fit_odds_alpha(
            prev_odds_labels, prev_odds_ens_probs, prev_odds_market_probs
        ))
    else:
        odds_alpha_pooled = float(ODDS_ALPHA_PRIOR)
    results["odds_alpha_pooled"] = round(odds_alpha_pooled, 4)
    results["odds_alpha_effective"] = round(min(odds_alpha_pooled, ODDS_ALPHA_CAP), 4)

    # Write per-match CSV for bootstrap model selection
    if permatch_rows:
        DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
        pm_path = DATA_PROCESSED / "backtest_permatch.csv"
        _pm_fields = ["fold", "model", "match_idx", "label",
                      "log_loss_i", "brier_i", "correct_i", "has_odds"]
        with open(pm_path, "w", newline="") as _f:
            _w = csv.DictWriter(_f, fieldnames=_pm_fields)
            _w.writeheader()
            _w.writerows(permatch_rows)

    return results


def _print_report(results: dict) -> None:
    print("\n=== World Cup Holdout Backtest ===")
    for year, fold in results.items():
        if not isinstance(fold, dict):
            continue
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
                  f"log={ensc['weights_logistic']:.2f} tree={ensc['weights_tree']:.2f}]")
        ensm = fold.get("model_ensemble_market", {})
        if ensm:
            print(f"  Ens+Mkt    log_loss={ensm['log_loss']:.4f}  brier={ensm['brier']:.4f}  "
                  f"acc={ensm['accuracy']:.3f}  α={ensm['alpha_odds']:.3f}")
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
