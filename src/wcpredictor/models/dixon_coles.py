"""Dixon-Coles score model.

Model:
    λ = exp(α_a + β_b + γ·home_flag)   # team_a expected goals
    μ = exp(α_b + β_a)                  # team_b expected goals
    P(x,y) = τ(x,y; λ,μ,ρ) · Poisson(x;λ) · Poisson(y;μ)

τ correction (low-score dependence, Dixon & Coles 1997):
    τ(0,0) = 1 − λμρ
    τ(0,1) = 1 + λρ
    τ(1,0) = 1 + μρ
    τ(1,1) = 1 − ρ
    τ      = 1 otherwise

Fitting: time-weighted log-likelihood via L-BFGS-B with vectorized numpy NLL.
Identifiability: mean(α) anchored to 0 post-fit (shift α, compensate β).
Unseen teams (< DC_MIN_MATCHES) fall back to the average sentinel's α/β.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import poisson

from wcpredictor.config import (
    DC_HOME_ADV_INIT,
    DC_MIN_MATCHES,
    DC_RHO_INIT,
    DC_TIME_DECAY_XI,
    DC_TRAIN_WINDOW_YEARS,
    MAX_GOALS,
)

_EPS = 1e-10
_AVG = "__avg__"


@dataclass
class DCParams:
    attack: dict[str, float] = field(default_factory=dict)
    defense: dict[str, float] = field(default_factory=dict)
    home_adv: float = 0.0
    rho: float = 0.0
    mean_attack: float = 0.0
    mean_defense: float = 0.0


def _make_nll(
    a_idx: np.ndarray,
    b_idx: np.ndarray,
    goals_a: np.ndarray,
    goals_b: np.ndarray,
    home_flag: np.ndarray,
    weights: np.ndarray,
    n_teams: int,
):
    """Return a vectorized NLL callable for L-BFGS-B."""
    log_fact_a = gammaln(goals_a + 1)
    log_fact_b = gammaln(goals_b + 1)
    m00 = (goals_a == 0) & (goals_b == 0)
    m01 = (goals_a == 0) & (goals_b == 1)
    m10 = (goals_a == 1) & (goals_b == 0)
    m11 = (goals_a == 1) & (goals_b == 1)

    def nll(x: np.ndarray) -> float:
        alpha = x[:n_teams]
        beta = x[n_teams : 2 * n_teams]
        gamma = x[-2]
        rho = x[-1]

        la = np.exp(alpha[a_idx] + beta[b_idx] + gamma * home_flag)
        mu = np.exp(alpha[b_idx] + beta[a_idx])

        log_pa = goals_a * np.log(la + _EPS) - la - log_fact_a
        log_pb = goals_b * np.log(mu + _EPS) - mu - log_fact_b

        tau = np.ones(len(goals_a))
        tau[m00] = 1.0 - la[m00] * mu[m00] * rho
        tau[m01] = 1.0 + la[m01] * rho
        tau[m10] = 1.0 + mu[m10] * rho
        tau[m11] = 1.0 - rho
        np.maximum(tau, _EPS, out=tau)

        log_lik = weights * (np.log(tau) + log_pa + log_pb)

        # Soft identifiability penalty on mean(alpha)
        penalty = 1000.0 * np.mean(alpha) ** 2

        return float(-log_lik.sum() + penalty)

    return nll


def fit(matches: pd.DataFrame, ref_date: pd.Timestamp | None = None) -> DCParams:
    """Fit Dixon-Coles parameters to historical match data.

    Parameters
    ----------
    matches  : DataFrame with columns: team_a, team_b, goals_a, goals_b, neutral, date
    ref_date : reference date for time decay; defaults to max(date) in matches

    Returns
    -------
    DCParams with fitted attack/defense strengths, home advantage, and ρ.
    """
    if ref_date is None:
        ref_date = matches["date"].max()

    window_start = ref_date - pd.DateOffset(years=DC_TRAIN_WINDOW_YEARS)
    m = matches[matches["date"] >= window_start].copy()

    if len(m) == 0:
        return DCParams()

    delta_days = (ref_date - m["date"]).dt.days.to_numpy(float)
    weights = np.exp(-DC_TIME_DECAY_XI * delta_days)

    counts: Counter = Counter()
    for row in m[["team_a", "team_b"]].itertuples(index=False):
        counts[row.team_a] += 1
        counts[row.team_b] += 1

    qualified = {t for t, c in counts.items() if c >= DC_MIN_MATCHES}

    teams = sorted(qualified) + [_AVG]
    team_idx = {t: i for i, t in enumerate(teams)}
    n_teams = len(teams)

    def _q(t: str) -> str:
        return t if t in qualified else _AVG

    a_col = m["team_a"].map(_q)
    b_col = m["team_b"].map(_q)
    a_idx = np.array([team_idx[t] for t in a_col])
    b_idx = np.array([team_idx[t] for t in b_col])
    goals_a = m["goals_a"].to_numpy(int)
    goals_b = m["goals_b"].to_numpy(int)
    home_flag = (~m["neutral"]).to_numpy(float)

    x0 = np.zeros(n_teams * 2 + 2)
    x0[-2] = DC_HOME_ADV_INIT
    x0[-1] = DC_RHO_INIT

    nll_fn = _make_nll(a_idx, b_idx, goals_a, goals_b, home_flag, weights, n_teams)

    bounds = [(None, None)] * (2 * n_teams) + [(None, None), (-0.99, 0.99)]

    res = minimize(nll_fn, x0, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": 2000, "ftol": 1e-8})

    alpha = res.x[:n_teams].copy()
    beta = res.x[n_teams : 2 * n_teams].copy()
    gamma = float(res.x[-2])
    rho = float(res.x[-1])

    # Normalize: anchor mean(alpha over qualified teams) to 0
    q_idx = [team_idx[t] for t in qualified]
    if q_idx:
        shift = float(np.mean(alpha[q_idx]))
        alpha -= shift
        beta += shift

    attack = {t: float(alpha[i]) for t, i in team_idx.items() if t != _AVG}
    defense = {t: float(beta[i]) for t, i in team_idx.items() if t != _AVG}

    return DCParams(
        attack=attack,
        defense=defense,
        home_adv=gamma,
        rho=rho,
        mean_attack=float(alpha[team_idx[_AVG]]),
        mean_defense=float(beta[team_idx[_AVG]]),
    )


def predict_one(
    params: DCParams,
    team_a: str,
    team_b: str,
    neutral: bool = True,
) -> dict:
    """Return W/D/L probs, λ/μ, score matrix, and top-5 scorelines.

    Output shape matches poisson.predict_one exactly.
    Unseen teams fall back to params.mean_attack / mean_defense.
    """
    alpha_a = params.attack.get(team_a, params.mean_attack)
    beta_a = params.defense.get(team_a, params.mean_defense)
    alpha_b = params.attack.get(team_b, params.mean_attack)
    beta_b = params.defense.get(team_b, params.mean_defense)

    home_flag = 0.0 if neutral else 1.0
    la = math.exp(alpha_a + beta_b + params.home_adv * home_flag)
    mu = math.exp(alpha_b + beta_a)

    mg = MAX_GOALS
    pa = np.array([poisson.pmf(i, la) for i in range(mg + 1)])
    pb = np.array([poisson.pmf(j, mu) for j in range(mg + 1)])

    matrix = np.outer(pa, pb)

    rho = params.rho
    matrix[0, 0] *= max(1.0 - la * mu * rho, _EPS)
    matrix[0, 1] *= max(1.0 + la * rho, _EPS)
    matrix[1, 0] *= max(1.0 + mu * rho, _EPS)
    matrix[1, 1] *= max(1.0 - rho, _EPS)

    matrix /= matrix.sum()

    p_win = float(np.tril(matrix, -1).sum())
    p_draw = float(np.trace(matrix))
    p_loss = float(np.triu(matrix, 1).sum())

    flat = [(float(matrix[i, j]), i, j) for i in range(mg + 1) for j in range(mg + 1)]
    flat.sort(reverse=True)
    top = [{"goals_a": i, "goals_b": j, "prob": round(p, 6)} for p, i, j in flat[:5]]

    return {
        "p_win": round(p_win, 6),
        "p_draw": round(p_draw, 6),
        "p_loss": round(p_loss, 6),
        "lambda_a": round(la, 4),
        "lambda_b": round(mu, 4),
        "score_matrix": matrix.tolist(),
        "top_scorelines": top,
    }
