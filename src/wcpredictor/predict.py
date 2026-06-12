"""High-level prediction facade.

predict_match(team_a, team_b, date, neutral=True, model="poisson") -> dict
  Returns the §11.1 JSON-shape output for any (team_a, team_b) pair.
  Uses the full historical dataset up to (but not including) `date` for Elo,
  and fits the chosen model on all pre-date matches.

  model="poisson"      : 2-param Elo→λ Poisson (Phase 1 baseline; default)
  model="dixon_coles"  : per-team Dixon-Coles with τ low-score correction

  neutral=True is the default because World Cup matches are played at a
  third-country venue or declared neutral by FIFA.

predict_fixtures(as_of_date, ..., models=["poisson", "ensemble"]) -> DataFrame
  Batch pre-tournament prediction: fits each model ONCE at the cutoff, then
  predicts all upcoming fixtures from those frozen fits.  Output includes a
  `model` column so live.py can score each model independently.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from wcpredictor.config import DATA_PROCESSED, DATA_RAW, INITIAL_RATING, ODDS_ALPHA_CAP, TOURNAMENT_START
from wcpredictor.data.download import download_results
from wcpredictor.data.load_matches import load_matches
from wcpredictor.data.normalize_teams import canonical
from wcpredictor.data.results_2026 import augment_matches
from wcpredictor.features.elo import compute_elo, latest_elo
from wcpredictor.features.form import compute_form, form_row as _form_row
from wcpredictor.models.poisson import fit as poisson_fit
from wcpredictor.models.poisson import predict_one as poisson_predict_one


def _resolve_odds_alpha() -> float:
    """Return the effective odds-blend weight, capped at ODDS_ALPHA_CAP.

    Reads odds_alpha_pooled from the backtest report (unconstrained optimum ~0.64),
    then caps it so the market stays a low-weight calibration overlay, not the driver.
    Falls back to ODDS_ALPHA_PRIOR (0.0) if the report is absent or unreadable.
    """
    from wcpredictor.config import ODDS_ALPHA_PRIOR

    _path = DATA_PROCESSED / "backtest_report.json"
    if _path.exists():
        try:
            report = json.loads(_path.read_text())
            raw = float(report.get("odds_alpha_pooled", ODDS_ALPHA_PRIOR))
            return min(raw, ODDS_ALPHA_CAP)
        except (KeyError, ValueError, json.JSONDecodeError):
            pass
    return float(ODDS_ALPHA_PRIOR)


def _resolve_ah_alpha() -> float:
    """Return the effective AH-blend weight, capped at AH_ALPHA_CAP.

    Reads ah_alpha_pooled from the backtest report.  Falls back to AH_ALPHA_PRIOR
    (0.0) if the report is absent or the field is missing (data not yet available).
    """
    from wcpredictor.config import AH_ALPHA_CAP, AH_ALPHA_PRIOR

    _path = DATA_PROCESSED / "backtest_report.json"
    if _path.exists():
        try:
            report = json.loads(_path.read_text())
            raw = float(report.get("ah_alpha_pooled", AH_ALPHA_PRIOR))
            return min(raw, AH_ALPHA_CAP)
        except (KeyError, ValueError, json.JSONDecodeError):
            pass
    return float(AH_ALPHA_PRIOR)


def _market_score_matrix(
    ah_line: float,
    ou_line: float,
    max_goals: int = 8,
) -> list[list[float]] | None:
    """Construct an implied Poisson score matrix from market AH and O/U lines.

    Inverts the fair-line approximation:
        supremacy  s ≈ −ah_line  (goal-diff where home covers ≈ 50 %)
        total      μ ≈ ou_line   (total goals where over ≈ 50 %)
        λ_a = (μ + s) / 2 = (ou_line − ah_line) / 2
        λ_b = (μ − s) / 2 = (ou_line + ah_line) / 2

    Returns None if the implied λ's are non-positive (degenerate market).

    NOTE: betexplorer always sets the "active" AH tab to -0.5 regardless of the true
    market main line. When using betexplorer data, prefer
    _market_score_matrix_from_probs() which uses implied win / over probabilities
    and is correct regardless of which line is shown as active.
    """
    import math as _m
    if math.isnan(ah_line) or math.isnan(ou_line):
        return None

    la = (ou_line - ah_line) / 2.0
    lb = (ou_line + ah_line) / 2.0

    if la <= 0.0 or lb <= 0.0:
        return None

    def pmf(lam: float, k: int) -> float:
        return _m.exp(-lam) * lam ** k / _m.factorial(k)

    raw = [[pmf(la, i) * pmf(lb, j) for j in range(max_goals + 1)] for i in range(max_goals + 1)]
    total = sum(raw[i][j] for i in range(max_goals + 1) for j in range(max_goals + 1))
    return [[raw[i][j] / total for j in range(max_goals + 1)] for i in range(max_goals + 1)]


def _market_score_matrix_from_probs(
    p_cover: float,
    p_over: float,
    ah_line: float = -0.5,
    ou_threshold: float = 2.5,
    max_goals: int = 8,
) -> list[list[float]] | None:
    """Construct an implied Poisson score matrix from market AH and O/U implied probs.

    More robust than _market_score_matrix() because it uses margin-stripped implied
    probabilities rather than raw line values, which is required when:
    - betexplorer always shows -0.5 as the active AH tab (historical data)
    - the-odds-api gives the actual main AH line (any value, e.g., -1.5, -2.5)

    Parameters
    ----------
    p_cover      : margin-stripped P(home covers the given AH line).
                   For ah_line = -0.5: this equals P(home wins outright).
                   For ah_line = -1.5: this equals P(home wins by 2+ goals).
    p_over       : margin-stripped P(total > ou_threshold).
    ah_line      : the AH line being settled (negative = home favoured). Used to
                   determine the settlement condition for the cover probability.
    ou_threshold : the O/U threshold (default 2.5).
    max_goals    : matrix dimension.

    Two-step bisection solve:
        1. Find μ = λ_a + λ_b from p_over (Poisson sum constraint).
        2. Find s = λ_a − λ_b from p_cover at ah_line (via settle_line).
        3. λ_a = (μ + s) / 2,  λ_b = (μ − s) / 2.

    Returns None if inputs are invalid or the numerical solve fails.
    """
    import math as _m
    from wcpredictor.markets.asian import settle_line as _settle

    if math.isnan(p_cover) or math.isnan(p_over):
        return None
    if not (0.0 < p_cover < 1.0) or not (0.0 < p_over < 1.0):
        return None

    # AH settlement threshold for home: wins when goal_diff > (-ah_line)
    ah_threshold = -ah_line  # e.g., 0.5 for ah_line=-0.5, 1.5 for ah_line=-1.5

    def pmf(lam: float, k: int) -> float:
        if lam <= 0:
            return 1.0 if k == 0 else 0.0
        return _m.exp(-lam) * lam ** k / _m.factorial(k)

    # Step 1: solve for μ = λ_a + λ_b from p_over
    threshold_int = int(math.ceil(ou_threshold))  # 3 for ou_threshold=2.5

    def ou_residual(mu: float) -> float:
        p_under_eq = sum(pmf(mu, k) for k in range(threshold_int))
        return (1.0 - p_under_eq) - p_over

    lo, hi = 0.01, 20.0
    try:
        f_lo, f_hi = ou_residual(lo), ou_residual(hi)
        if f_lo * f_hi > 0:
            return None
        for _ in range(60):
            mid = (lo + hi) / 2.0
            if ou_residual(mid) * f_lo <= 0:
                hi = mid
            else:
                lo = mid
        mu = (lo + hi) / 2.0
    except Exception:
        return None

    if mu <= 0.0:
        return None

    # Step 2: solve for s = λ_a − λ_b from p_cover at ah_line
    def ah_cover_prob(s: float) -> float:
        """P(home covers AH line) given la=(μ+s)/2, lb=(μ-s)/2."""
        la = (mu + s) / 2.0
        lb = (mu - s) / 2.0
        if la <= 0.0 or lb <= 0.0:
            return 0.0
        # Build integer goal-diff distribution
        diff_dist: dict[int, float] = {}
        for ga in range(max_goals + 1):
            pa = pmf(la, ga)
            if pa < 1e-12:
                continue
            for gb in range(max_goals + 1):
                pb = pmf(lb, gb)
                if pb < 1e-12:
                    continue
                d = ga - gb
                diff_dist[d] = diff_dist.get(d, 0.0) + pa * pb
        try:
            result = _settle(diff_dist, ah_threshold)
            return result["p_win"] + result["p_half_win"]
        except Exception:
            return 0.0

    def s_residual(s: float) -> float:
        return ah_cover_prob(s) - p_cover

    s_lo, s_hi = -(mu - 0.01), (mu - 0.01)
    try:
        f_slo, f_shi = s_residual(s_lo), s_residual(s_hi)
        if f_slo * f_shi > 0:
            return None
        for _ in range(60):
            mid = (s_lo + s_hi) / 2.0
            if s_residual(mid) * f_slo <= 0:
                s_hi = mid
            else:
                s_lo = mid
        s = (s_lo + s_hi) / 2.0
    except Exception:
        return None

    la = (mu + s) / 2.0
    lb = (mu - s) / 2.0
    if la <= 0.0 or lb <= 0.0:
        return None

    raw = [[pmf(la, i) * pmf(lb, j) for j in range(max_goals + 1)] for i in range(max_goals + 1)]
    total = sum(raw[i][j] for i in range(max_goals + 1) for j in range(max_goals + 1))
    if total <= 0.0:
        return None
    return [[raw[i][j] / total for j in range(max_goals + 1)] for i in range(max_goals + 1)]


def _blend_matrix_ah(
    model_mat: list[list[float]],
    ah_line: float,
    ou_line: float,
    alpha: float,
    max_goals: int = 8,
    *,
    ah_p_home: float | None = None,
    ou_p_over: float | None = None,
) -> list[list[float]]:
    """Blend the model score matrix with a market-implied matrix at weight *alpha*.

    If *ah_p_home* and *ou_p_over* are supplied (margin-stripped implied probabilities),
    the market matrix is built via _market_score_matrix_from_probs() which correctly
    handles the betexplorer artifact of always showing -0.5 as the active AH line.
    Falls back to _market_score_matrix() (line-based inversion) otherwise.

    If market data is unavailable or degenerate, returns the model matrix unchanged.
    M' = (1 − α) · M_model + α · M_market  (renormalized)
    """
    if alpha <= 0.0:
        return model_mat

    # Prefer prob-based inversion (robust to betexplorer -0.5 default line and
    # handles any AH line correctly, e.g., -1.5, -2.5 from the-odds-api).
    if ah_p_home is not None and ou_p_over is not None and \
            not (math.isnan(ah_p_home) or math.isnan(ou_p_over)):
        mkt_mat = _market_score_matrix_from_probs(
            ah_p_home, ou_p_over, ah_line=ah_line, ou_threshold=ou_line, max_goals=max_goals
        )
    else:
        mkt_mat = _market_score_matrix(ah_line, ou_line, max_goals)

    if mkt_mat is None:
        return model_mat

    n = max_goals + 1
    blended = [
        [(1.0 - alpha) * model_mat[i][j] + alpha * mkt_mat[i][j] for j in range(n)]
        for i in range(n)
    ]
    total = sum(blended[i][j] for i in range(n) for j in range(n))
    if total <= 0.0:
        return model_mat
    return [[blended[i][j] / total for j in range(n)] for i in range(n)]


def _build_odds_lookup(
    odds_df: "pd.DataFrame",
) -> "dict[tuple[str, str], tuple[float, float, float]]":
    """Build a (team_a, team_b) → (p_win, p_draw, p_loss) lookup for 2026 fixtures."""
    lookup: dict = {}
    if odds_df.empty or "year" not in odds_df.columns:
        return lookup
    yr2026 = odds_df[odds_df["year"] == 2026].sort_values("date")
    for _, r in yr2026.iterrows():
        ta, tb = str(r.team_a), str(r.team_b)
        lookup[(ta, tb)] = (float(r.p_win), float(r.p_draw), float(r.p_loss))
        lookup[(tb, ta)] = (float(r.p_loss), float(r.p_draw), float(r.p_win))
    return lookup


def _lbl(ga: int, gb: int) -> int:
    if ga > gb:
        return 0
    if ga == gb:
        return 1
    return 2


def _load_offers_lookup() -> "dict[tuple[str, str], list[dict]]":
    """Load bookmaker spread/totals offers from the cached odds-API JSON.

    Returns a dict keyed by (team_a, team_b) in both orientations.
    Empty dict when the JSON is absent, corrupt, or has no offer data.
    """
    try:
        from wcpredictor.data.download_odds_api import parse_market_offers as _parse_offers
        _live_json = DATA_RAW / "odds_api_wc2026.json"
        if not _live_json.exists():
            return {}
        offers_df = _parse_offers(json.loads(_live_json.read_text()))
        if offers_df.empty:
            return {}
        lookup: dict[tuple[str, str], list[dict]] = {}
        for (ta, tb), grp in offers_df.groupby(["team_a", "team_b"], sort=False):
            forward = grp.to_dict("records")
            reversed_list = []
            for o in forward:
                if o["market"] == "ah":
                    reversed_list.append({
                        **o, "team_a": tb, "team_b": ta,
                        "line": -o["line"],
                        "side": "away" if o["side"] == "home" else "home",
                    })
                elif o["market"] == "1x2":
                    # 1x2: swap home↔away; draw unchanged
                    side_map = {"home": "away", "away": "home", "draw": "draw"}
                    reversed_list.append({
                        **o, "team_a": tb, "team_b": ta,
                        "side": side_map.get(o["side"], o["side"]),
                    })
                else:
                    reversed_list.append({**o, "team_a": tb, "team_b": ta})
            lookup[(ta, tb)] = forward
            lookup[(tb, ta)] = reversed_list
        return lookup
    except Exception:
        return {}


def _attach_offers(
    markets_dict: dict,
    matrix: "list[list[float]]",
    offers: "list[dict]",
) -> None:
    """Attach EV-ranked bookmaker offers to a markets dict in-place."""
    from wcpredictor.markets.edge import evaluate_offers
    evaluated = evaluate_offers(matrix, offers)
    markets_dict["offers"] = evaluated
    markets_dict["best_offer"] = evaluated[0] if evaluated else None


def _ensure_data() -> None:
    from wcpredictor.config import DATA_RAW

    if not (DATA_RAW / "results.csv").exists():
        download_results()


def predict_match(
    team_a: str,
    team_b: str,
    date: str,
    neutral: bool = True,
    model: Literal["poisson", "dixon_coles", "ensemble", "ensemble_mkt"] = "ensemble_mkt",
) -> dict:
    """Predict a single match.

    Parameters
    ----------
    team_a : str   Home/first team name (canonical or alias).
    team_b : str   Away/second team name.
    date   : str   ISO date string; only matches strictly before this date are used.
    neutral: bool  True (default) suppresses home-advantage in Elo diff / DC home flag.
    model  : str   Model name. Default "ensemble_mkt" (market-blended ensemble, auto-degrades to ensemble_cal when no odds are available).

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

    if model in {"ensemble", "ensemble_mkt"}:
        from wcpredictor.config import DATA_RAW, DC_CAL_VALIDATION_YEARS, ENSEMBLE_POOL
        from wcpredictor.data.download_odds import download_odds
        from wcpredictor.features.odds import load_wc_odds, merge_odds_features
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
        from wcpredictor.models.gbm import fit as tree_fit
        from wcpredictor.models.gbm import predict_proba as tree_predict
        from wcpredictor.models.logistic import fit as log_fit
        from wcpredictor.models.logistic import predict_proba as log_predict

        import numpy as _np

        if not (DATA_RAW / "WorldCup_fdco.xlsx").exists():
            download_odds()
        _wc_odds_df = load_wc_odds()
        elo_df = merge_odds_features(elo_df, _wc_odds_df)

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
        early_tree_model = tree_fit(early_elo_df, early_labels)

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
        val_p_tree: list[list[float]] = (
            tree_predict(early_tree_model, val_elo)
            if len(val_elo) > 0 else []
        )

        # Fit ensemble weights on odds-bearing WC stacking validation (regime-matched)
        from wcpredictor.evaluation.backtest import (
            build_wc_stacking_validation,
            _WC_START,
        )
        # Effective fold year: one past the latest WC whose start is before cutoff
        past_wcs = [w for w, s in _WC_START.items() if pd.Timestamp(s) < cutoff]
        eff_year = max(past_wcs) + 1 if past_wcs else min(_WC_START)
        wc_val_labels, wc_val_member_probs = build_wc_stacking_validation(
            eff_year, train, elo_df
        )
        member_val = [val_p_poi, val_p_dc, val_p_log, val_p_tree]
        if wc_val_labels:
            ens_weights = ens_fit_weights(wc_val_member_probs, wc_val_labels, pool=ENSEMBLE_POOL)
            if val_labels_e:
                val_ens = ens_combine_probs(member_val, ens_weights, pool=ENSEMBLE_POOL)
                T_ens = fit_temperature(val_labels_e, val_ens)
            else:
                T_ens = 1.0
        elif val_labels_e:
            ens_weights = ens_fit_weights(member_val, val_labels_e, pool=ENSEMBLE_POOL)
            val_ens = ens_combine_probs(member_val, ens_weights, pool=ENSEMBLE_POOL)
            T_ens = fit_temperature(val_labels_e, val_ens)
        else:
            ens_weights = _np.array([0.25, 0.25, 0.25, 0.25])
            T_ens = 1.0

        # Full-data member fits (< cutoff)
        dc_params = dc_fit(train, ref_date=cutoff)
        full_base, full_beta = poisson_fit(elo_df)
        full_labels = [_lbl(int(r.goals_a), int(r.goals_b)) for _, r in elo_df.iterrows()]
        log_scaler, log_model = log_fit(elo_df, full_labels)
        tree_model = tree_fit(elo_df, full_labels)

        # Predict single match with all members
        home_bonus = 0.0 if neutral else 50.0
        elo_diff_adj = (r_a + home_bonus) - r_b
        p_poi = poisson_predict_one(elo_diff_adj, full_base, full_beta)
        p_dc = dc_predict_one(dc_params, team_a, team_b, neutral=neutral)

        fr = _form_row(form_state, team_a, team_b, cutoff)
        import math as _math
        odds_lookup = _build_odds_lookup(_wc_odds_df)
        _odds_entry = odds_lookup.get((team_a, team_b))
        if _odds_entry:
            _odds_pw, _odds_pd, _odds_pl, _has_odds = _odds_entry[0], _odds_entry[1], _odds_entry[2], 1.0
        else:
            _odds_pw, _odds_pd, _odds_pl, _has_odds = _math.nan, _math.nan, _math.nan, 0.0
        test_row = pd.DataFrame({
            "elo_diff_adj": [elo_diff_adj],
            "neutral": [neutral],
            "form_diff": [fr["form_diff"]],
            "momentum_diff": [fr["momentum_diff"]],
            "rest_diff": [fr["rest_diff"]],
            "elo_a_pre": [r_a],
            "elo_b_pre": [r_b],
            "odds_p_win": [_odds_pw],
            "odds_p_draw": [_odds_pd],
            "odds_p_loss": [_odds_pl],
            "has_odds": [_has_odds],
        })
        p_log_proba = log_predict(log_scaler, log_model, test_row)[0]
        p_tree_proba = tree_predict(tree_model, test_row)[0]

        # Combine W/D/L
        member_single = [
            [[p_poi["p_win"], p_poi["p_draw"], p_poi["p_loss"]]],
            [[p_dc["p_win"], p_dc["p_draw"], p_dc["p_loss"]]],
            [p_log_proba],
            [p_tree_proba],
        ]
        combined_wdl = ens_combine_probs(member_single, ens_weights, pool=ENSEMBLE_POOL)[0]
        combined_cal = cal_apply([combined_wdl], T_ens)[0]

        # Market-odds blending for ensemble_mkt
        if model == "ensemble_mkt":
            _odds_entry = odds_lookup.get((team_a, team_b))
            if _odds_entry:
                _alpha = _resolve_odds_alpha()
                _blended = [(1 - _alpha) * combined_cal[i] + _alpha * _odds_entry[i] for i in range(3)]
                _total = sum(_blended)
                combined_cal = [v / _total for v in _blended]
            model_version = "ensemble_mkt-0.1"
        else:
            model_version = "ensemble-0.2"

        # Score matrix (Poisson + DC blend; logistic has none)
        w_score = ens_weights[:2].copy()
        w_score /= w_score.sum()
        mat = ens_combine_matrices([[p_poi["score_matrix"]], [p_dc["score_matrix"]]], w_score)[0]
        la_ens, lb_ens = matrix_to_lambdas(mat)
        top_sc = matrix_to_top_scorelines(mat, n=5)

        from wcpredictor.markets.asian import ladder as _ah_ladder
        result = {
            "p_win": round(combined_cal[0], 6),
            "p_draw": round(combined_cal[1], 6),
            "p_loss": round(combined_cal[2], 6),
            "lambda_a": round(la_ens, 4),
            "lambda_b": round(lb_ens, 4),
            "score_matrix": mat,
            "top_scorelines": top_sc,
            "markets": _ah_ladder(mat),
        }

    elif model == "dixon_coles":
        from wcpredictor.models.dixon_coles import fit as dc_fit
        from wcpredictor.models.dixon_coles import predict_one as dc_predict_one
        from wcpredictor.markets.asian import ladder as _ah_ladder

        dc_params = dc_fit(train, ref_date=cutoff)
        result = dc_predict_one(dc_params, team_a, team_b, neutral=neutral)
        result["markets"] = _ah_ladder(result["score_matrix"])
        model_version = "dc-0.1"
    else:
        from wcpredictor.markets.asian import ladder as _ah_ladder
        home_bonus = 0.0 if neutral else 50.0
        elo_diff_adj = (r_a + home_bonus) - r_b
        base, beta = poisson_fit(elo_df)
        result = poisson_predict_one(elo_diff_adj, base, beta)
        result["markets"] = _ah_ladder(result["score_matrix"])
        model_version = "mvp-0.1"

    _pm_offers = _load_offers_lookup().get((team_a, team_b))
    if _pm_offers:
        _attach_offers(result["markets"], result["score_matrix"], _pm_offers)

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


def _build_frozen_state(
    as_of_date: str,
    models: list[str],
    fit_cutoff: str | None = None,
) -> dict:
    """Load data and fit all requested models once at the given cutoff.

    Two cutoff dates are used:
    - rating_cutoff: advances with as_of_date; Elo/form roll forward with played results.
    - fit_cutoff:    pinned at TOURNAMENT_START (or min of as_of, TOURNAMENT_START);
                     all model fits (Poisson, DC, logistic, GBM, calibration) use only
                     pre-tournament history — no rolling refits during the tournament.

    This mirrors the validated backtest configuration (backtest.py:9).
    """
    _ensure_data()
    rating_cutoff = pd.Timestamp(as_of_date)
    fc = pd.Timestamp(fit_cutoff) if fit_cutoff else pd.Timestamp(TOURNAMENT_START)
    # min() ensures pre-tournament calls (as_of <= TOURNAMENT_START) are byte-identical
    fit_cutoff_ts = min(fc, rating_cutoff)

    # Augment historical matches with any played WC2026 results (empty before tournament)
    matches = augment_matches(load_matches())
    rating_train = matches[matches["date"] < rating_cutoff].copy()
    fit_train = matches[matches["date"] < fit_cutoff_ts].copy()

    # Elo/form roll forward with rating_cutoff
    elo_df, final_ratings = compute_elo(rating_train)
    form_df, form_state = compute_form(rating_train)
    elo_df = elo_df.merge(form_df, on="match_id", how="left")
    ratings = latest_elo(elo_df, before_date=rating_cutoff, final_ratings=final_ratings)

    # fit_elo: Elo rows for the pinned fit window only
    fit_elo_df = elo_df[elo_df["date"] < fit_cutoff_ts].copy()

    state: dict = {
        "cutoff": rating_cutoff,        # used by _predict_one_frozen for form row lookup
        "fit_cutoff": fit_cutoff_ts,
        "train": fit_train,             # kept for DC fit reference
        "elo_df": fit_elo_df,           # Poisson/logistic/GBM features — pinned window
        "ratings": ratings,             # rolled forward to rating_cutoff
        "form_state": form_state,
    }

    needs_poisson = "poisson" in models or "ensemble" in models or "ensemble_mkt" in models
    needs_dc = "dixon_coles" in models or "ensemble" in models or "ensemble_mkt" in models

    if needs_poisson:
        base, beta = poisson_fit(fit_elo_df)
        state["poisson_base"] = base
        state["poisson_beta"] = beta

    if needs_dc:
        from wcpredictor.models.dixon_coles import fit as dc_fit
        state["dc_params"] = dc_fit(fit_train, ref_date=fit_cutoff_ts)

    if "ensemble" in models or "ensemble_mkt" in models:
        from wcpredictor.config import DC_CAL_VALIDATION_YEARS, ENSEMBLE_POOL
        from wcpredictor.data.download_odds import download_odds
        from wcpredictor.features.odds import load_wc_odds, merge_odds_features
        from wcpredictor.models.calibration import apply as cal_apply, fit_temperature
        from wcpredictor.models.dixon_coles import fit as dc_fit
        from wcpredictor.models.dixon_coles import predict_one as dc_predict_one
        from wcpredictor.models.ensemble import (
            combine_probs as ens_combine_probs,
            fit_weights as ens_fit_weights,
        )
        from wcpredictor.models.gbm import fit as tree_fit
        from wcpredictor.models.gbm import predict_proba as tree_predict
        from wcpredictor.models.logistic import fit as log_fit
        from wcpredictor.models.logistic import predict_proba as log_predict
        from wcpredictor.evaluation.backtest import build_wc_stacking_validation, _WC_START

        if not (DATA_RAW / "WorldCup_fdco.xlsx").exists():
            download_odds()

        odds_df = load_wc_odds()
        elo_all = merge_odds_features(fit_elo_df, odds_df)

        # Build odds lookup for 2026 fixtures (live JSON or fdco sheet, whichever is available)
        state["odds_lookup"] = _build_odds_lookup(odds_df)

        cal_start = fit_cutoff_ts - pd.DateOffset(years=DC_CAL_VALIDATION_YEARS)
        early_elo = elo_all[elo_all["date"] < cal_start]
        early_train = fit_train[fit_train["date"] < cal_start]
        early_base, early_beta = poisson_fit(early_elo)
        early_dc = dc_fit(early_train, ref_date=cal_start)
        early_labels = [_lbl(int(r.goals_a), int(r.goals_b)) for _, r in early_elo.iterrows()]
        early_log_scaler, early_log_model = log_fit(early_elo, early_labels)
        early_tree_model = tree_fit(early_elo, early_labels)

        val_elo = elo_all[(elo_all["date"] >= cal_start) & (elo_all["date"] < fit_cutoff_ts)]
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
        val_p_tree: list[list[float]] = (
            tree_predict(early_tree_model, val_elo)
            if len(val_elo) > 0 else []
        )

        past_wcs = [w for w, s in _WC_START.items() if pd.Timestamp(s) < fit_cutoff_ts]
        eff_year = max(past_wcs) + 1 if past_wcs else min(_WC_START)
        wc_val_labels, wc_val_member_probs = build_wc_stacking_validation(
            eff_year, fit_train, elo_all
        )
        member_val = [val_p_poi, val_p_dc, val_p_log, val_p_tree]
        if wc_val_labels:
            ens_weights = ens_fit_weights(wc_val_member_probs, wc_val_labels, pool=ENSEMBLE_POOL)
            if val_labels_e:
                val_ens = ens_combine_probs(member_val, ens_weights, pool=ENSEMBLE_POOL)
                T_ens = fit_temperature(val_labels_e, val_ens)
            else:
                T_ens = 1.0
        elif val_labels_e:
            ens_weights = ens_fit_weights(member_val, val_labels_e, pool=ENSEMBLE_POOL)
            val_ens = ens_combine_probs(member_val, ens_weights, pool=ENSEMBLE_POOL)
            T_ens = fit_temperature(val_labels_e, val_ens)
        else:
            ens_weights = np.array([0.25, 0.25, 0.25, 0.25])
            T_ens = 1.0

        full_labels = [_lbl(int(r.goals_a), int(r.goals_b)) for _, r in elo_all.iterrows()]
        log_scaler, log_model_fit = log_fit(elo_all, full_labels)
        tree_model = tree_fit(elo_all, full_labels)

        state["ens_log_scaler"] = log_scaler
        state["ens_log_model"] = log_model_fit
        state["ens_tree_model"] = tree_model
        state["ens_weights"] = ens_weights
        state["ens_T"] = T_ens
        state["ens_elo_all"] = elo_all  # needed for DC fit that's already in state["dc_params"]

        # Resolve odds_alpha for ensemble_mkt — read from pre-computed report when available
        state["odds_alpha"] = _resolve_odds_alpha()

    # AH matrix-blend: load AH odds lookup and alpha (all models benefit from this)
    from wcpredictor.features.ah_odds import load_wc_ah_odds as _load_ah
    ah_df = _load_ah()
    ah_lookup: dict[tuple[str, str], dict] = {}

    def _sf(v: object) -> float | None:
        """Safely convert to float; return None for NaN / missing."""
        try:
            x = float(v)  # type: ignore[arg-type]
            return x if not math.isnan(x) else None
        except (TypeError, ValueError):
            return None

    if not ah_df.empty:
        for _, r in ah_df[ah_df["year"] == 2026].iterrows():
            ta, tb = str(r.team_a), str(r.team_b)
            # Include implied probs so _blend_matrix_ah uses prob-based inversion,
            # which is robust to betexplorer always showing -0.5 as the active AH line.
            ah_ph = _sf(r.get("ah_p_home"))
            ou_po = _sf(r.get("ou_p_over"))
            entry: dict = {
                "ah_line": float(r.ah_line),
                "ou_line": float(r.ou_line),
                "ah_p_home": ah_ph,
                "ou_p_over": ou_po,
            }
            away_entry: dict = {
                "ah_line": -float(r.ah_line),
                "ou_line": float(r.ou_line),
                # Away perspective: home cover prob flips to away cover prob
                "ah_p_home": (1.0 - ah_ph) if ah_ph is not None else None,
                "ou_p_over": ou_po,
            }
            ah_lookup[(ta, tb)] = entry
            ah_lookup[(tb, ta)] = away_entry
    state["ah_lookup"] = ah_lookup
    state["ah_alpha"] = _resolve_ah_alpha()

    state["offers_lookup"] = _load_offers_lookup()

    return state


def _predict_one_frozen(
    state: dict,
    model_name: str,
    team_a: str,
    team_b: str,
    neutral: bool,
) -> dict:
    """Predict a single match using pre-fitted frozen state."""
    from wcpredictor.models.ensemble import (
        combine_matrices as ens_combine_matrices,
        combine_probs as ens_combine_probs,
        matrix_to_lambdas,
        matrix_to_top_scorelines,
    )

    cutoff = state["cutoff"]
    ratings = state["ratings"]
    r_a = ratings.get(team_a, INITIAL_RATING)
    r_b = ratings.get(team_b, INITIAL_RATING)
    home_bonus = 0.0 if neutral else 50.0
    elo_diff_adj = (r_a + home_bonus) - r_b

    from wcpredictor.markets.asian import ladder as _ah_ladder

    if model_name == "poisson":
        result = poisson_predict_one(elo_diff_adj, state["poisson_base"], state["poisson_beta"])
        result["markets"] = _ah_ladder(result["score_matrix"])
        model_version = "mvp-0.1"

    elif model_name == "dixon_coles":
        from wcpredictor.models.dixon_coles import predict_one as dc_predict_one
        result = dc_predict_one(state["dc_params"], team_a, team_b, neutral=neutral)
        result["markets"] = _ah_ladder(result["score_matrix"])
        model_version = "dc-0.1"

    elif model_name in {"ensemble", "ensemble_mkt"}:
        from wcpredictor.config import ENSEMBLE_POOL
        from wcpredictor.models.calibration import apply as cal_apply
        from wcpredictor.models.dixon_coles import predict_one as dc_predict_one
        from wcpredictor.models.gbm import predict_proba as tree_predict
        from wcpredictor.models.logistic import predict_proba as log_predict

        p_poi = poisson_predict_one(elo_diff_adj, state["poisson_base"], state["poisson_beta"])
        p_dc = dc_predict_one(state["dc_params"], team_a, team_b, neutral=neutral)

        fr = _form_row(state["form_state"], team_a, team_b, cutoff)

        # Look up 2026 odds if available
        odds_key = (team_a, team_b)
        odds_entry = state.get("odds_lookup", {}).get(odds_key)
        if odds_entry:
            odds_pw, odds_pd, odds_pl, has_odds = odds_entry[0], odds_entry[1], odds_entry[2], 1.0
        else:
            odds_pw, odds_pd, odds_pl, has_odds = math.nan, math.nan, math.nan, 0.0

        test_row = pd.DataFrame({
            "elo_diff_adj": [elo_diff_adj],
            "neutral": [neutral],
            "form_diff": [fr["form_diff"]],
            "momentum_diff": [fr["momentum_diff"]],
            "rest_diff": [fr["rest_diff"]],
            "elo_a_pre": [r_a],
            "elo_b_pre": [r_b],
            "odds_p_win": [odds_pw],
            "odds_p_draw": [odds_pd],
            "odds_p_loss": [odds_pl],
            "has_odds": [has_odds],
        })
        p_log_proba = log_predict(state["ens_log_scaler"], state["ens_log_model"], test_row)[0]
        p_tree_proba = tree_predict(state["ens_tree_model"], test_row)[0]

        member_single = [
            [[p_poi["p_win"], p_poi["p_draw"], p_poi["p_loss"]]],
            [[p_dc["p_win"], p_dc["p_draw"], p_dc["p_loss"]]],
            [p_log_proba],
            [p_tree_proba],
        ]
        combined_wdl = ens_combine_probs(member_single, state["ens_weights"], pool=ENSEMBLE_POOL)[0]
        combined_cal = cal_apply([combined_wdl], state["ens_T"])[0]

        w_score = state["ens_weights"][:2].copy()
        w_score = w_score / w_score.sum()
        mat = ens_combine_matrices([[p_poi["score_matrix"]], [p_dc["score_matrix"]]], w_score)[0]

        # AH market matrix blend (auto-degrades when no AH odds or alpha == 0)
        ah_entry = state.get("ah_lookup", {}).get((team_a, team_b))
        if ah_entry:
            from wcpredictor.config import MAX_GOALS
            mat = _blend_matrix_ah(
                mat,
                ah_entry["ah_line"],
                ah_entry["ou_line"],
                state.get("ah_alpha", 0.0),
                MAX_GOALS,
                ah_p_home=ah_entry.get("ah_p_home"),
                ou_p_over=ah_entry.get("ou_p_over"),
            )

        la_ens, lb_ens = matrix_to_lambdas(mat)
        top_sc = matrix_to_top_scorelines(mat, n=5)

        if model_name == "ensemble_mkt":
            if odds_entry:
                alpha = state.get("odds_alpha", 0.0)
                blended = [
                    (1 - alpha) * combined_cal[i] + alpha * (odds_pw, odds_pd, odds_pl)[i]
                    for i in range(3)
                ]
                total = sum(blended)
                combined_cal = [v / total for v in blended]
            model_version = "ensemble_mkt-0.1"
        else:
            model_version = "ensemble-0.2"

        result = {
            "p_win": round(combined_cal[0], 6),
            "p_draw": round(combined_cal[1], 6),
            "p_loss": round(combined_cal[2], 6),
            "lambda_a": round(la_ens, 4),
            "lambda_b": round(lb_ens, 4),
            "score_matrix": mat,
            "top_scorelines": top_sc,
            "markets": _ah_ladder(mat),
        }

    else:
        raise ValueError(f"Unknown model: {model_name!r}")

    _offers = state.get("offers_lookup", {}).get((team_a, team_b))
    if _offers:
        _attach_offers(result["markets"], result["score_matrix"], _offers)

    result.update({
        "team_a": team_a,
        "team_b": team_b,
        "neutral": neutral,
        "elo_a": round(r_a, 1),
        "elo_b": round(r_b, 1),
        "model_version": model_version,
        "model": model_name,
    })
    return result


def predict_fixtures(
    as_of_date: str,
    fixtures_path: Path | None = None,
    model: Literal["poisson", "dixon_coles", "ensemble", "ensemble_mkt"] = "ensemble",
    output_path: Path | None = None,
    models: list[str] | None = None,
) -> pd.DataFrame:
    """Generate predictions for all WC2026 fixtures not yet played.

    Batch mode: each model is fitted ONCE at `as_of_date`; all fixtures share
    the same frozen model state.  This is the correct approach for pre-tournament
    predictions (all matches use pre-tournament data) and avoids 72× refits.

    Parameters
    ----------
    as_of_date    : ISO date string; fixtures on or after this date are predicted.
    fixtures_path : path to wc2026_fixtures.csv; defaults to DATA_RAW/wc2026_fixtures.csv.
    model         : single model name (used when `models` is None).
    output_path   : where to save the predictions CSV; defaults to
                    DATA_PROCESSED/wc2026_predictions_<as_of_date>.csv.
    models        : list of model names to run; if given, overrides `model`.
                    Output will contain one row per (fixture, model) with a `model` column.

    Returns
    -------
    DataFrame with one row per (upcoming fixture × model) plus p_win/draw/loss.
    Raises FileNotFoundError if fixtures_path does not exist.
    """
    if fixtures_path is None:
        fixtures_path = DATA_RAW / "wc2026_fixtures.csv"

    if not Path(fixtures_path).exists():
        raise FileNotFoundError(
            f"WC2026 fixtures file not found: {fixtures_path}\n"
            "Run: uv run python -m wcpredictor.data.download_wc2026"
        )

    model_list = models if models is not None else [model]

    as_of = pd.Timestamp(as_of_date)
    fixtures = pd.read_csv(fixtures_path, parse_dates=["date"])
    # Skip fixtures already played (goals filled in) and past dates
    played = fixtures["goals_a"].notna() & fixtures["goals_b"].notna()
    upcoming = fixtures[~played & (fixtures["date"] >= as_of)].copy()

    # Fit all models once at the cutoff; fits pinned at TOURNAMENT_START
    state = _build_frozen_state(as_of_date, model_list)

    rows: list[dict] = []
    for _, fix in upcoming.iterrows():
        ta = canonical(str(fix["team_a"]))
        tb = canonical(str(fix["team_b"]))
        neutral = bool(fix.get("neutral", True))
        date_str = fix["date"].strftime("%Y-%m-%d")
        for m_name in model_list:
            try:
                pred = _predict_one_frozen(state, m_name, ta, tb, neutral)
                pred["date"] = date_str
            except Exception as exc:
                pred = {
                    "team_a": ta, "team_b": tb, "date": date_str, "neutral": neutral,
                    "p_win": None, "p_draw": None, "p_loss": None,
                    "model": m_name, "error": str(exc),
                }
            rows.append(pred)

    df = pd.DataFrame(rows)

    if output_path is None:
        DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
        output_path = DATA_PROCESSED / f"wc2026_predictions_{as_of_date}.csv"

    df.to_csv(output_path, index=False)
    return df
