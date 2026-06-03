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
