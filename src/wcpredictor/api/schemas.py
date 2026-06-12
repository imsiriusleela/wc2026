from __future__ import annotations

from typing import Any
from pydantic import BaseModel


class PredictResponse(BaseModel):
    team_a: str
    team_b: str
    neutral: bool
    model: str
    model_version: str
    p_win: float
    p_draw: float
    p_loss: float
    lambda_a: float
    lambda_b: float
    elo_a: float
    elo_b: float
    score_matrix: list[list[float]]
    top_scorelines: list[dict[str, Any]]
    markets: dict[str, Any] | None = None


class FixtureRow(BaseModel):
    team_a: str
    team_b: str
    date: str
    neutral: bool
    model: str
    model_version: str
    p_win: float | None
    p_draw: float | None
    p_loss: float | None
    lambda_a: float | None
    lambda_b: float | None
    elo_a: float | None
    elo_b: float | None
    score_matrix: list[list[float]] | None
    top_scorelines: list[dict[str, Any]] | None
    markets: dict[str, Any] | None = None


class TeamStanding(BaseModel):
    team: str
    group: str
    p_win_group: float
    p_runner_up: float
    p_r32: float
    p_r16: float
    p_qf: float
    p_sf: float
    p_final: float
    p_champion: float


class TournamentResponse(BaseModel):
    as_of: str
    model: str
    n_sims: int
    n_group_fixed: int = 0
    n_ko_fixed: int = 0
    standings: list[TeamStanding]
    top10_champion: list[dict[str, Any]]


class ScorecardResponse(BaseModel):
    as_of_date: str
    n_completed: int
    log_loss: float | None
    brier: float | None
    accuracy: float | None
    ece_uncal: float | None
    ece_cal: float | None
    temperature: float
    models: dict[str, Any]
    matches: list[dict[str, Any]]


class RefreshOddsResponse(BaseModel):
    status: str
    n_odds_2026: int
    n_odds_2026_fdco: int = 0
    n_odds_2026_api: int = 0
    odds_api_refreshed: bool = False
    file_sha256: str = ""
    pinned_sha256: str
    sha_changed: bool
    state_cache_cleared: bool
    note: str


class RefreshResultsResponse(BaseModel):
    status: str
    n_results_total: int
    n_new: int
    n_group: int
    n_knockout: int
    n_fixtures_updated: int
    note: str


class ResimulateResponse(BaseModel):
    status: str
    as_of: str
    model: str
    n_sims: int
    n_group_fixed: int
    n_ko_fixed: int
    output_csv: str
    note: str


class ValueBet(BaseModel):
    date: str
    team_a: str
    team_b: str
    market: str
    line: float
    side: str
    sgpools_price: float
    fair_model: float | None
    ev_model: float
    fair_consensus: float | None
    ev_consensus: float | None
    recommended: bool
    recommended_stake: float
    sizing_tier: str
    confidence_flags: list[str]


class ValueBetsResponse(BaseModel):
    as_of: str
    n_offers: int
    n_recommended: int
    sizing_tier: str
    consensus_age_note: str
    offers_age_note: str
    bets: list[ValueBet]


class RecordBetRequest(BaseModel):
    team_a: str
    team_b: str
    date: str
    market: str
    line: float = 0.0
    side: str
    price_taken: float
    stake: float
    consensus_fair_at_placement: float | None = None


class RecordBetResponse(BaseModel):
    status: str
    match: str
    market: str
    side: str
    price_taken: float
    stake: float


class LedgerResponse(BaseModel):
    total_bets: int
    settled_bets: int
    open_bets: int
    total_staked: float
    total_pnl: float
    roi: float | None
    trailing_clv_mean: float | None
    drawdown_units: float
    stop_clv: bool
    stop_drawdown: bool
    stop_rule_triggered: bool
    bets: list[dict[str, Any]]


class SgpoolsAddRequest(BaseModel):
    team_a: str
    team_b: str
    date: str
    market: str
    line: float = 0.0
    side: str
    price: float


class RefreshSgpoolsResponse(BaseModel):
    status: str
    n_fetched: int
    note: str
