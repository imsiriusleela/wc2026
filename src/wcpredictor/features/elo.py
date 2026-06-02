"""Chronological Elo ratings with pre-match feature emission.

Design:
- Walk matches in date order.
- For each match: emit pre-match ratings first (no leakage), then update.
- Home advantage is applied as an Elo bonus (zeroed for neutral venues).
- Goal-difference multiplier G follows the standard World Football Elo formula.
- Deterministic given the same input.
"""

from __future__ import annotations

import pandas as pd

from wcpredictor.config import INITIAL_RATING, K_MAP, HOME_ADVANTAGE


def _k(competition: str) -> float:
    return K_MAP.get(competition, K_MAP["friendly"])


def _goal_diff_multiplier(gd: int) -> float:
    if gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    return (11 + gd) / 8.0


def compute_elo(matches: pd.DataFrame) -> tuple["pd.DataFrame", dict[str, float]]:
    """Walk matches chronologically and return (features_df, final_ratings).

    Input: load_matches() output (must be sorted by date).

    features_df columns:
        match_id, date, team_a, team_b,
        elo_a_pre, elo_b_pre, elo_diff_adj,
        neutral, goals_a, goals_b

    final_ratings: current (post-last-match) Elo per team.
    """
    ratings: dict[str, float] = {}

    rows = []
    for _, m in matches.iterrows():
        a, b = m["team_a"], m["team_b"]
        neutral = bool(m["neutral"])
        goals_a, goals_b = int(m["goals_a"]), int(m["goals_b"])
        competition = m["competition"]

        r_a = ratings.get(a, INITIAL_RATING)
        r_b = ratings.get(b, INITIAL_RATING)

        # Home advantage applied to team_a (the home team when not neutral)
        home_bonus = 0.0 if neutral else HOME_ADVANTAGE
        r_a_adj = r_a + home_bonus

        elo_diff_adj = r_a_adj - r_b

        # Emit pre-match row before any update
        rows.append(
            {
                "match_id": m["match_id"],
                "date": m["date"],
                "team_a": a,
                "team_b": b,
                "elo_a_pre": r_a,
                "elo_b_pre": r_b,
                "elo_diff_adj": elo_diff_adj,
                "neutral": neutral,
                "goals_a": goals_a,
                "goals_b": goals_b,
            }
        )

        # Elo update
        e_a = 1.0 / (1.0 + 10.0 ** ((r_b - r_a_adj) / 400.0))
        e_b = 1.0 - e_a

        if goals_a > goals_b:
            w_a, w_b = 1.0, 0.0
        elif goals_a < goals_b:
            w_a, w_b = 0.0, 1.0
        else:
            w_a, w_b = 0.5, 0.5

        gd = abs(goals_a - goals_b)
        g = _goal_diff_multiplier(gd)
        k = _k(competition)

        ratings[a] = r_a + k * g * (w_a - e_a)
        ratings[b] = r_b + k * g * (w_b - e_b)

    return pd.DataFrame(rows), dict(ratings)


def latest_elo(
    elo_df: pd.DataFrame,
    before_date: str | pd.Timestamp,
    final_ratings: dict[str, float] | None = None,
) -> dict[str, float]:
    """Return the most recent (current) Elo for every team before the given date.

    If final_ratings is provided and before_date is after all match dates,
    returns final_ratings directly (post-last-match values).
    Otherwise falls back to the last seen pre-match Elo from elo_df.
    """
    cutoff = pd.Timestamp(before_date)

    if final_ratings is not None and elo_df["date"].max() < cutoff:
        return dict(final_ratings)

    past = elo_df[elo_df["date"] < cutoff]
    result: dict[str, float] = {}
    for _, row in past.iterrows():
        result[row["team_a"]] = row["elo_a_pre"]
        result[row["team_b"]] = row["elo_b_pre"]
    return result
