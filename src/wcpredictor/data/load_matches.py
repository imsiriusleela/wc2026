"""Load and validate the international match results dataset.

Caveats (documented per HANDOFF.md):
- Scores for knockout matches include extra time but not penalties.
  This slightly contaminates the strict 90-minute label for those fixtures.
  Accepted for MVP; can be filtered downstream if needed.
"""

from __future__ import annotations

import hashlib

import pandas as pd

from wcpredictor.config import DATA_RAW
from wcpredictor.data.normalize_teams import canonical

_RAW = DATA_RAW / "results.csv"

# Mapping from tournament string patterns to competition tier
_COMPETITION_MAP: list[tuple[str, str]] = [
    ("FIFA World Cup", "world_cup"),
    ("UEFA Euro", "continental"),
    ("Copa America", "continental"),
    ("Africa Cup of Nations", "continental"),
    ("Asian Cup", "continental"),
    ("Gold Cup", "continental"),
    ("Oceania", "continental"),
    ("qualification", "qualifier"),
    ("Qualifier", "qualifier"),
    ("Qualifying", "qualifier"),
    ("friendly", "friendly"),
    ("Friendly", "friendly"),
    ("Nations League", "qualifier"),
    ("Confederations Cup", "continental"),
]


def _competition(tournament: str) -> str:
    for pattern, tier in _COMPETITION_MAP:
        if pattern in tournament:
            return tier
    return "friendly"


def _match_id(row: pd.Series) -> str:
    key = f"{row['date']}|{row['team_a']}|{row['team_b']}|{row['tournament']}"
    return hashlib.md5(key.encode()).hexdigest()[:12]


def load_matches() -> pd.DataFrame:
    """Read raw results.csv and return a clean, validated match DataFrame.

    Output schema:
        match_id, date, team_a, team_b, goals_a, goals_b,
        neutral, tournament, competition, is_world_cup, country, city
    """
    if not _RAW.exists():
        raise FileNotFoundError(
            f"{_RAW} not found — run `python -m wcpredictor.data.download` first."
        )

    df = pd.read_csv(_RAW, parse_dates=["date"])

    required = {"date", "home_team", "away_team", "home_score", "away_score", "tournament", "neutral"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in results.csv: {missing}")

    df = df.rename(
        columns={
            "home_team": "team_a",
            "away_team": "team_b",
            "home_score": "goals_a",
            "away_score": "goals_b",
        }
    )

    df["team_a"] = df["team_a"].map(canonical)
    df["team_b"] = df["team_b"].map(canonical)

    null_scores = df["goals_a"].isna() | df["goals_b"].isna()
    if null_scores.any():
        n = null_scores.sum()
        df = df[~null_scores].copy()
        import warnings
        warnings.warn(f"Dropped {n} rows with null scores.")

    df["goals_a"] = df["goals_a"].astype(int)
    df["goals_b"] = df["goals_b"].astype(int)
    df["neutral"] = df["neutral"].astype(bool)

    df["competition"] = df["tournament"].map(_competition)
    df["is_world_cup"] = df["competition"] == "world_cup"

    df["match_id"] = df.apply(_match_id, axis=1)

    df = df.sort_values("date").reset_index(drop=True)

    cols = [
        "match_id", "date", "team_a", "team_b",
        "goals_a", "goals_b", "neutral",
        "tournament", "competition", "is_world_cup",
        "country", "city",
    ]
    existing = [c for c in cols if c in df.columns]
    return df[existing]
