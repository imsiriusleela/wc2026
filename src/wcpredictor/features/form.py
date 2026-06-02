"""Chronological form features with pre-match emission.

Design mirrors compute_elo: walk matches in date order, emit the pre-match row
BEFORE updating per-team state (leak-free).

Features (all differenced team_a − team_b):
  form_diff      — rolling points-per-game over last FORM_WINDOW matches
  momentum_diff  — rolling mean goal difference over last FORM_WINDOW matches
  rest_diff      — days since previous match (each side capped at REST_DAYS_CAP)

Cold-start teams get neutral defaults (0, 0, cap) — no NaNs.
"""

from __future__ import annotations

from collections import deque

import pandas as pd

from wcpredictor.config import FORM_WINDOW, REST_DAYS_CAP


def compute_form(
    matches: pd.DataFrame,
    window: int = FORM_WINDOW,
) -> tuple[pd.DataFrame, dict]:
    """Walk matches chronologically and return (form_df, form_state).

    Input: load_matches() output (must be sorted by date).

    form_df columns: match_id, form_diff, momentum_diff, rest_diff

    form_state: dict mapping team -> {"pts": deque, "gd": deque, "last_date": Timestamp|None}
    """
    state: dict[str, dict] = {}

    def _get(team: str) -> dict:
        if team not in state:
            state[team] = {
                "pts": deque(maxlen=window),
                "gd": deque(maxlen=window),
                "last_date": None,
            }
        return state[team]

    def _ppg(pts_deque: deque) -> float:
        if not pts_deque:
            return 0.0
        return sum(pts_deque) / len(pts_deque)

    def _mean_gd(gd_deque: deque) -> float:
        if not gd_deque:
            return 0.0
        return sum(gd_deque) / len(gd_deque)

    def _rest(last_date, match_date) -> float:
        if last_date is None:
            return float(REST_DAYS_CAP)
        delta = (match_date - last_date).days
        return float(min(delta, REST_DAYS_CAP))

    rows = []
    for _, m in matches.iterrows():
        a, b = m["team_a"], m["team_b"]
        match_date = pd.Timestamp(m["date"])
        goals_a, goals_b = int(m["goals_a"]), int(m["goals_b"])

        sa = _get(a)
        sb = _get(b)

        # Emit pre-match row (state not yet updated with this match)
        form_a = _ppg(sa["pts"])
        form_b = _ppg(sb["pts"])
        mom_a = _mean_gd(sa["gd"])
        mom_b = _mean_gd(sb["gd"])
        rest_a = _rest(sa["last_date"], match_date)
        rest_b = _rest(sb["last_date"], match_date)

        rows.append({
            "match_id": m["match_id"],
            "form_diff": form_a - form_b,
            "momentum_diff": mom_a - mom_b,
            "rest_diff": rest_a - rest_b,
        })

        # Update state
        if goals_a > goals_b:
            pts_a, pts_b = 3, 0
        elif goals_a < goals_b:
            pts_a, pts_b = 0, 3
        else:
            pts_a, pts_b = 1, 1

        sa["pts"].append(pts_a)
        sa["gd"].append(goals_a - goals_b)
        sa["last_date"] = match_date

        sb["pts"].append(pts_b)
        sb["gd"].append(goals_b - goals_a)
        sb["last_date"] = match_date

    return pd.DataFrame(rows), state


def form_row(
    form_state: dict,
    team_a: str,
    team_b: str,
    query_date: pd.Timestamp | str,
    window: int = FORM_WINDOW,
) -> dict:
    """Build the three differenced form features for a hypothetical match.

    Uses pre-computed form_state (from compute_form).
    Cold-start teams get neutral defaults.
    """
    query_date = pd.Timestamp(query_date)

    def _ppg(pts_deque: deque) -> float:
        if not pts_deque:
            return 0.0
        return sum(pts_deque) / len(pts_deque)

    def _mean_gd(gd_deque: deque) -> float:
        if not gd_deque:
            return 0.0
        return sum(gd_deque) / len(gd_deque)

    def _rest(last_date) -> float:
        if last_date is None:
            return float(REST_DAYS_CAP)
        delta = (query_date - last_date).days
        return float(min(max(delta, 0), REST_DAYS_CAP))

    sa = form_state.get(team_a, {})
    sb = form_state.get(team_b, {})

    pts_a = sa.get("pts", deque())
    pts_b = sb.get("pts", deque())
    gd_a = sa.get("gd", deque())
    gd_b = sb.get("gd", deque())
    last_a = sa.get("last_date", None)
    last_b = sb.get("last_date", None)

    return {
        "form_diff": _ppg(pts_a) - _ppg(pts_b),
        "momentum_diff": _mean_gd(gd_a) - _mean_gd(gd_b),
        "rest_diff": _rest(last_a) - _rest(last_b),
    }
