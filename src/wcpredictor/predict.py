"""High-level prediction facade.

predict_match(team_a, team_b, date, neutral=True, model="poisson") -> dict
  Returns the §11.1 JSON-shape output for any (team_a, team_b) pair.
  Uses the full historical dataset up to (but not including) `date` for Elo,
  and fits the chosen model on all pre-date matches.

  model="poisson"      : 2-param Elo→λ Poisson (Phase 1 baseline; default)
  model="dixon_coles"  : per-team Dixon-Coles with τ low-score correction

  neutral=True is the default because World Cup matches are played at a
  third-country venue or declared neutral by FIFA.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd

from wcpredictor.config import INITIAL_RATING
from wcpredictor.data.download import download_results
from wcpredictor.data.load_matches import load_matches
from wcpredictor.data.normalize_teams import canonical
from wcpredictor.features.elo import compute_elo, latest_elo
from wcpredictor.features.form import compute_form, form_row as _form_row
from wcpredictor.models.poisson import fit as poisson_fit
from wcpredictor.models.poisson import predict_one as poisson_predict_one


def _lbl(ga: int, gb: int) -> int:
    if ga > gb:
        return 0
    if ga == gb:
        return 1
    return 2


def _ensure_data() -> None:
    from wcpredictor.config import DATA_RAW

    if not (DATA_RAW / "results.csv").exists():
        download_results()


def predict_match(
    team_a: str,
    team_b: str,
    date: str,
    neutral: bool = True,
    model: Literal["poisson", "dixon_coles", "ensemble"] = "poisson",
) -> dict:
    """Predict a single match.

    Parameters
    ----------
    team_a : str   Home/first team name (canonical or alias).
    team_b : str   Away/second team name.
    date   : str   ISO date string; only matches strictly before this date are used.
    neutral: bool  True (default) suppresses home-advantage in Elo diff / DC home flag.
    model  : str   "poisson" (default) or "dixon_coles".

    Returns
    -------
    dict with keys: team_a, team_b, date, neutral,
                    p_win, p_draw, p_loss,
                    lambda_a, lambda_b,
                    score_matrix, top_scorelines,
                    elo_a, elo_b, model_version.
    """
    _ensure_data()

    team_a = canonical(team_a)
    team_b = canonical(team_b)
    cutoff = pd.Timestamp(date)

    matches = load_matches()
    train = matches[matches["date"] < cutoff].copy()

    elo_df, final_ratings = compute_elo(train)
    form_df, form_state = compute_form(train)
    elo_df = elo_df.merge(form_df, on="match_id", how="left")

    ratings = latest_elo(elo_df, before_date=cutoff, final_ratings=final_ratings)
    r_a = ratings.get(team_a, INITIAL_RATING)
    r_b = ratings.get(team_b, INITIAL_RATING)

    if model == "ensemble":
        from wcpredictor.config import DC_CAL_VALIDATION_YEARS, ENSEMBLE_POOL
        from wcpredictor.models.calibration import apply as cal_apply
        from wcpredictor.models.calibration import fit_temperature
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

        import numpy as _np

        cal_start = cutoff - pd.DateOffset(years=DC_CAL_VALIDATION_YEARS)

        # Early fits (< cal_start) for leakage-safe weight estimation
        early_elo_df = elo_df[elo_df["date"] < cal_start]
        early_train = train[train["date"] < cal_start]
        early_base, early_beta = poisson_fit(early_elo_df)
        early_dc = dc_fit(early_train, ref_date=cal_start)
        early_labels = [
            _lbl(int(r.goals_a), int(r.goals_b)) for _, r in early_elo_df.iterrows()
        ]
        early_log_scaler, early_log_model = log_fit(early_elo_df, early_labels)

        # Validation slice: out-of-time member predictions
        val_elo = elo_df[(elo_df["date"] >= cal_start) & (elo_df["date"] < cutoff)]
        val_labels_e: list[int] = []
        val_p_poi: list[list[float]] = []
        val_p_dc: list[list[float]] = []
        for _, row in val_elo.iterrows():
            val_labels_e.append(_lbl(int(row.goals_a), int(row.goals_b)))
            vp = poisson_predict_one(float(row.elo_diff_adj), early_base, early_beta)
            val_p_poi.append([vp["p_win"], vp["p_draw"], vp["p_loss"]])
            vd = dc_predict_one(early_dc, row.team_a, row.team_b, bool(row.neutral))
            val_p_dc.append([vd["p_win"], vd["p_draw"], vd["p_loss"]])
        val_p_log: list[list[float]] = (
            log_predict(early_log_scaler, early_log_model, val_elo)
            if len(val_elo) > 0 else []
        )

        # Fit ensemble weights on validation predictions
        if val_labels_e:
            member_val = [val_p_poi, val_p_dc, val_p_log]
            ens_weights = ens_fit_weights(member_val, val_labels_e, pool=ENSEMBLE_POOL)
            val_ens = ens_combine_probs(member_val, ens_weights, pool=ENSEMBLE_POOL)
            T_ens = fit_temperature(val_labels_e, val_ens)
        else:
            ens_weights = _np.array([1 / 3, 1 / 3, 1 / 3])
            T_ens = 1.0

        # Full-data member fits (< cutoff)
        dc_params = dc_fit(train, ref_date=cutoff)
        full_base, full_beta = poisson_fit(elo_df)
        full_labels = [_lbl(int(r.goals_a), int(r.goals_b)) for _, r in elo_df.iterrows()]
        log_scaler, log_model = log_fit(elo_df, full_labels)

        # Predict single match with all members
        home_bonus = 0.0 if neutral else 50.0
        elo_diff_adj = (r_a + home_bonus) - r_b
        p_poi = poisson_predict_one(elo_diff_adj, full_base, full_beta)
        p_dc = dc_predict_one(dc_params, team_a, team_b, neutral=neutral)

        fr = _form_row(form_state, team_a, team_b, cutoff)
        test_row = pd.DataFrame({
            "elo_diff_adj": [elo_diff_adj],
            "neutral": [neutral],
            "form_diff": [fr["form_diff"]],
            "momentum_diff": [fr["momentum_diff"]],
            "rest_diff": [fr["rest_diff"]],
        })
        p_log_proba = log_predict(log_scaler, log_model, test_row)[0]

        # Combine W/D/L
        member_single = [
            [[p_poi["p_win"], p_poi["p_draw"], p_poi["p_loss"]]],
            [[p_dc["p_win"], p_dc["p_draw"], p_dc["p_loss"]]],
            [p_log_proba],
        ]
        combined_wdl = ens_combine_probs(member_single, ens_weights, pool=ENSEMBLE_POOL)[0]
        combined_cal = cal_apply([combined_wdl], T_ens)[0]

        # Score matrix (Poisson + DC blend; logistic has none)
        w_score = ens_weights[:2].copy()
        w_score /= w_score.sum()
        mat = ens_combine_matrices([[p_poi["score_matrix"]], [p_dc["score_matrix"]]], w_score)[0]
        la_ens, lb_ens = matrix_to_lambdas(mat)
        top_sc = matrix_to_top_scorelines(mat, n=5)

        result = {
            "p_win": round(combined_cal[0], 6),
            "p_draw": round(combined_cal[1], 6),
            "p_loss": round(combined_cal[2], 6),
            "lambda_a": round(la_ens, 4),
            "lambda_b": round(lb_ens, 4),
            "score_matrix": mat,
            "top_scorelines": top_sc,
        }
        model_version = "ensemble-0.1"

    elif model == "dixon_coles":
        from wcpredictor.models.dixon_coles import fit as dc_fit
        from wcpredictor.models.dixon_coles import predict_one as dc_predict_one

        dc_params = dc_fit(train, ref_date=cutoff)
        result = dc_predict_one(dc_params, team_a, team_b, neutral=neutral)
        model_version = "dc-0.1"
    else:
        home_bonus = 0.0 if neutral else 50.0
        elo_diff_adj = (r_a + home_bonus) - r_b
        base, beta = poisson_fit(elo_df)
        result = poisson_predict_one(elo_diff_adj, base, beta)
        model_version = "mvp-0.1"

    result.update(
        {
            "team_a": team_a,
            "team_b": team_b,
            "date": date,
            "neutral": neutral,
            "elo_a": round(r_a, 1),
            "elo_b": round(r_b, 1),
            "model_version": model_version,
        }
    )
    return result
