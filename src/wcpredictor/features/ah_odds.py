"""Load and align Asian-handicap / O-U market odds from historical CSV + live API.

Historical source: data/raw/wc_ah_odds.csv (betexplorer snapshots, WC 2010–2022).
Live source     : data/raw/odds_api_wc2026.json (the-odds-api.com, WC 2026 when key set).

The two-way AH prices (home/away) and O/U prices (over/under) are margin-stripped
identically to how _implied_probs works in features/odds.py:

    p_i = (1/o_i) / sum(1/o_j for j in {home, away})

The resulting fair probabilities are used both for evaluation (closing-line value)
and for the matrix blend in Phase 9.4 (invert implied λ's back into a market matrix).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from wcpredictor.config import DATA_RAW
from wcpredictor.data.normalize_teams import canonical

_AH_CSV = DATA_RAW / "wc_ah_odds.csv"


# ─── Margin removal ────────────────────────────────────────────────────────────

def _implied_two_way(o1: float, o2: float) -> tuple[float, float]:
    """Remove bookmaker margin from a two-way market (decimal odds)."""
    p1 = 1.0 / o1
    p2 = 1.0 / o2
    total = p1 + p2
    return p1 / total, p2 / total


# ─── Load ──────────────────────────────────────────────────────────────────────

def load_wc_ah_odds(csv_path: Path | None = None) -> pd.DataFrame:
    """Return tidy DataFrame with AH and O/U market odds for all WC years available.

    Columns:
        year       : int (2010, 2014, 2018, 2022, 2026)
        date       : pd.Timestamp
        team_a     : str (canonical; home team)
        team_b     : str (canonical; away team)
        ah_line    : float (standard AH notation for home, e.g., -1.5)
        ah_p_home  : float (margin-stripped implied prob home covers)
        ah_p_away  : float (margin-stripped implied prob away covers)
        ou_line    : float (total-goals threshold, e.g., 2.5)
        ou_p_over  : float (margin-stripped implied prob over)
        ou_p_under : float (margin-stripped implied prob under)
        has_ah     : float (1.0 if AH data present, else 0.0)
        has_ou     : float (1.0 if O/U data present, else 0.0)

    Merges historical CSV (betexplorer) with live API data when available.
    Returns empty DataFrame when no data files exist (e.g., snapshots not yet rendered).
    """
    if csv_path is None:
        csv_path = _AH_CSV

    frames: list[pd.DataFrame] = []

    # Historical (betexplorer CSV)
    if csv_path.exists():
        df_hist = pd.read_csv(csv_path)
        frames.append(df_hist)

    # Live 2026 (odds-api JSON → parsed CSV)
    live_json = DATA_RAW / "odds_api_wc2026.json"
    if live_json.exists():
        from wcpredictor.data.download_odds_api import _parse_odds_api_json
        import json
        raw = json.loads(live_json.read_text())
        df_live = _parse_odds_api_json(raw)
        # Rename columns to match CSV schema
        df_live = df_live.rename(columns={"home": "home", "away": "away"})
        frames.append(df_live)

    if not frames:
        return pd.DataFrame(columns=[
            "year", "date", "team_a", "team_b",
            "ah_line", "ah_p_home", "ah_p_away",
            "ou_line", "ou_p_over", "ou_p_under",
            "has_ah", "has_ou",
        ])

    raw_df = pd.concat(frames, ignore_index=True)

    records: list[dict] = []
    for _, r in raw_df.iterrows():
        ta = canonical(str(r.get("home", "")))
        tb = canonical(str(r.get("away", "")))
        if not ta or not tb:
            continue

        # AH
        try:
            ah_line = float(r["ah_line"])
            ah_ph, ah_pa = _implied_two_way(float(r["ah_home_odds"]), float(r["ah_away_odds"]))
            has_ah = 1.0
        except (TypeError, ValueError, KeyError):
            ah_line = float("nan")
            ah_ph = ah_pa = float("nan")
            has_ah = 0.0

        # O/U
        try:
            ou_line = float(r["ou_line"])
            ou_po, ou_pu = _implied_two_way(float(r["over_odds"]), float(r["under_odds"]))
            has_ou = 1.0
        except (TypeError, ValueError, KeyError):
            ou_line = float("nan")
            ou_po = ou_pu = float("nan")
            has_ou = 0.0

        records.append({
            "year": int(r.get("year", 0)),
            "date": pd.Timestamp(r["date"]),
            "team_a": ta,
            "team_b": tb,
            "ah_line": ah_line,
            "ah_p_home": ah_ph,
            "ah_p_away": ah_pa,
            "ou_line": ou_line,
            "ou_p_over": ou_po,
            "ou_p_under": ou_pu,
            "has_ah": has_ah,
            "has_ou": has_ou,
        })

    return pd.DataFrame(records)


# ─── Alignment ─────────────────────────────────────────────────────────────────

def align_ah_to_test(
    ah_df: pd.DataFrame,
    year: int,
    test_df: pd.DataFrame,
) -> list[dict] | None:
    """Align AH/O-U odds to test_df rows by (team_a, team_b) within WC *year*.

    Returns a list of dicts with ah_line, ah_p_home, ah_p_away, ou_line,
    ou_p_over, ou_p_under for each row in test_df, or None if <50 % of rows
    can be matched (signals unusable data).  Unmatched rows get NaN values.

    Symmetric lookup: (a,b) and (b,a) are both registered (with AH side flipped).
    """
    yr_odds = ah_df[ah_df["year"] == year]
    if yr_odds.empty:
        return None

    lookup: dict[tuple[str, str], dict] = {}
    for _, r in yr_odds.iterrows():
        ta, tb = str(r.team_a), str(r.team_b)
        entry = {
            "ah_line": r.ah_line,
            "ah_p_home": r.ah_p_home,
            "ah_p_away": r.ah_p_away,
            "ou_line": r.ou_line,
            "ou_p_over": r.ou_p_over,
            "ou_p_under": r.ou_p_under,
        }
        lookup[(ta, tb)] = entry
        # Symmetric: reversed fixture flips AH home/away (but not O/U)
        lookup[(tb, ta)] = {
            "ah_line": -r.ah_line if not np.isnan(r.ah_line) else r.ah_line,
            "ah_p_home": r.ah_p_away,
            "ah_p_away": r.ah_p_home,
            "ou_line": r.ou_line,
            "ou_p_over": r.ou_p_over,
            "ou_p_under": r.ou_p_under,
        }

    _nan = float("nan")
    _empty = {
        "ah_line": _nan, "ah_p_home": _nan, "ah_p_away": _nan,
        "ou_line": _nan, "ou_p_over": _nan, "ou_p_under": _nan,
    }

    result: list[dict] = []
    matched = 0
    for _, row in test_df.iterrows():
        key = (str(row.team_a), str(row.team_b))
        if key in lookup:
            result.append(lookup[key])
            matched += 1
        else:
            result.append(_empty.copy())

    if matched < len(result) * 0.5:
        return None
    return result


def merge_ah_features(
    features_df: pd.DataFrame,
    ah_df: pd.DataFrame,
) -> pd.DataFrame:
    """Attach AH/O-U odds columns to features_df.

    Keyed by (year, team_a, team_b).  Symmetric: (a,b) and (b,a) both registered.
    Rows with no match get NaN for odds columns and 0.0 for has_ah/has_ou.
    """
    lookup: dict[tuple[int, str, str], dict] = {}
    for _, r in ah_df.iterrows():
        ta, tb = str(r.team_a), str(r.team_b)
        yr = int(r.year)
        import numpy as np
        entry = {
            "ah_line": r.ah_line,
            "ah_p_home": r.ah_p_home,
            "ah_p_away": r.ah_p_away,
            "ou_line": r.ou_line,
            "ou_p_over": r.ou_p_over,
            "ou_p_under": r.ou_p_under,
            "has_ah": r.has_ah,
            "has_ou": r.has_ou,
        }
        lookup[(yr, ta, tb)] = entry
        lookup[(yr, tb, ta)] = {
            "ah_line": -r.ah_line if not np.isnan(r.ah_line) else r.ah_line,
            "ah_p_home": r.ah_p_away,
            "ah_p_away": r.ah_p_home,
            "ou_line": r.ou_line,
            "ou_p_over": r.ou_p_over,
            "ou_p_under": r.ou_p_under,
            "has_ah": r.has_ah,
            "has_ou": r.has_ou,
        }

    years = features_df["date"].dt.year.to_numpy(int)
    ta_arr = features_df["team_a"].astype(str).to_numpy()
    tb_arr = features_df["team_b"].astype(str).to_numpy()

    cols: dict[str, list] = {
        "ah_line": [], "ah_p_home": [], "ah_p_away": [],
        "ou_line": [], "ou_p_over": [], "ou_p_under": [],
        "has_ah": [], "has_ou": [],
    }
    _nan = float("nan")
    _empty: dict = {
        "ah_line": _nan, "ah_p_home": _nan, "ah_p_away": _nan,
        "ou_line": _nan, "ou_p_over": _nan, "ou_p_under": _nan,
        "has_ah": 0.0, "has_ou": 0.0,
    }

    for yr, ta, tb in zip(years, ta_arr, tb_arr):
        e = lookup.get((int(yr), ta, tb), _empty)
        for col in cols:
            cols[col].append(e[col])

    out = features_df.copy()
    for col, vals in cols.items():
        out[col] = vals
    return out
