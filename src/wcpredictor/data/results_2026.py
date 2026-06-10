"""Canonical store for played WC2026 results.

Schema of data/raw/wc2026_results.csv:
    date, team_a, team_b, goals_a, goals_b, stage, winner, source

- stage:  "group" if date < KO_START else "knockout"
- winner: set only for penalty-decided KO matches (martj42 reports those as draws)
- source: "martj42" | "manual"   manual rows win over martj42 on dedup
"""

from __future__ import annotations

import hashlib
import ssl
import urllib.request
import warnings
from pathlib import Path

import pandas as pd

from wcpredictor.config import DATA_RAW, KO_START, RESULTS_URL_FALLBACK, TOURNAMENT_START
from wcpredictor.data.normalize_teams import canonical

_STORE = DATA_RAW / "wc2026_results.csv"
_MASTER = DATA_RAW / "results_master.csv"

_STORE_COLS = ["date", "team_a", "team_b", "goals_a", "goals_b", "stage", "winner", "source"]


def _match_id_2026(date: str, ta: str, tb: str) -> str:
    key = f"{date}|{ta}|{tb}"
    return hashlib.md5(key.encode()).hexdigest()[:12]


def _typed_empty() -> pd.DataFrame:
    return pd.DataFrame({
        "date": pd.Series(dtype="datetime64[ns]"),
        "team_a": pd.Series(dtype=str),
        "team_b": pd.Series(dtype=str),
        "goals_a": pd.Series(dtype="Int64"),
        "goals_b": pd.Series(dtype="Int64"),
        "stage": pd.Series(dtype=str),
        "winner": pd.Series(dtype=str),
        "source": pd.Series(dtype=str),
    })


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

def fetch_master_results() -> pd.DataFrame:
    """Download martj42 master branch to results_master.csv; return DataFrame.

    Does NOT touch the pinned data/raw/results.csv.
    Falls back to the existing local copy on network failure.
    """
    try:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        with urllib.request.urlopen(RESULTS_URL_FALLBACK, context=ctx, timeout=30) as resp:
            raw = resp.read()
        _MASTER.write_bytes(raw)
    except Exception as exc:
        warnings.warn(f"Could not refresh results_master from martj42: {exc}")

    if not _MASTER.exists():
        if not (DATA_RAW / "results.csv").exists():
            raise FileNotFoundError(
                "No local results file available. "
                "Run: uv run python -m wcpredictor.data.download"
            )
        return pd.read_csv(DATA_RAW / "results.csv", parse_dates=["date"])

    return pd.read_csv(_MASTER, parse_dates=["date"])


# ---------------------------------------------------------------------------
# Store read/write
# ---------------------------------------------------------------------------

def load_wc2026_results() -> pd.DataFrame:
    """Read the canonical 2026 results store; return empty typed DataFrame when absent."""
    if not _STORE.exists():
        return _typed_empty()
    df = pd.read_csv(_STORE, parse_dates=["date"])
    for col in _STORE_COLS:
        if col not in df.columns:
            df[col] = None
    return df[_STORE_COLS]


def update_wc2026_results(source_csv: Path | None = None) -> dict:
    """Pull WC2026 results from martj42 master (or a user CSV) and merge into the store.

    Manual rows (source == "manual") always win over martj42 rows on the same match.

    Returns
    -------
    dict with keys: n_total, n_new, n_group, n_knockout
    """
    t_start = pd.Timestamp(TOURNAMENT_START)
    ko_start = pd.Timestamp(KO_START)

    if source_csv is not None:
        raw_df = pd.read_csv(source_csv, parse_dates=["date"])
        source_tag = "manual"
    else:
        raw_df = fetch_master_results()
        source_tag = "martj42"

    # Ensure date column is Timestamp regardless of source
    raw_df["date"] = pd.to_datetime(raw_df["date"])

    # Detect column layout: martj42 uses home_team/away_team, canonical store uses team_a/team_b
    if "home_team" in raw_df.columns:
        raw_df = raw_df.rename(columns={
            "home_team": "team_a",
            "away_team": "team_b",
            "home_score": "goals_a",
            "away_score": "goals_b",
        })

    # Filter to FIFA World Cup rows with scores and date >= TOURNAMENT_START
    if "tournament" in raw_df.columns:
        raw_df = raw_df[raw_df["tournament"].str.contains("FIFA World Cup", na=False)]
    raw_df = raw_df[raw_df["date"] >= t_start]
    raw_df = raw_df[raw_df["goals_a"].notna() & raw_df["goals_b"].notna()].copy()

    if raw_df.empty:
        existing = load_wc2026_results()
        return {
            "n_total": len(existing),
            "n_new": 0,
            "n_group": int((existing["stage"] == "group").sum()) if not existing.empty else 0,
            "n_knockout": int((existing["stage"] == "knockout").sum()) if not existing.empty else 0,
        }

    raw_df["team_a"] = raw_df["team_a"].map(canonical)
    raw_df["team_b"] = raw_df["team_b"].map(canonical)
    raw_df["goals_a"] = raw_df["goals_a"].astype(int)
    raw_df["goals_b"] = raw_df["goals_b"].astype(int)
    raw_df["stage"] = raw_df["date"].apply(
        lambda d: "group" if d < ko_start else "knockout"
    )
    raw_df["winner"] = raw_df.get("winner", pd.NA)
    raw_df["source"] = source_tag

    new_rows = raw_df[_STORE_COLS].copy()

    existing = load_wc2026_results()

    if existing.empty:
        merged = new_rows
    else:
        # Build a dedup key: canonical sorted pair + date
        def _key(row: pd.Series) -> str:
            pair = tuple(sorted([str(row.team_a), str(row.team_b)]))
            return f"{pd.Timestamp(row.date).date()}|{pair[0]}|{pair[1]}"

        existing["_key"] = existing.apply(_key, axis=1)
        new_rows["_key"] = new_rows.apply(_key, axis=1)

        # Merge: combine all, deduplicate by key keeping manual over martj42.
        # Sort so manual rows sort last (alphabetically "manual" > "martj42"),
        # then drop_duplicates(keep="last") retains manual rows when both exist.
        combined = pd.concat([existing, new_rows], ignore_index=True)
        combined["_source_priority"] = (combined["source"] == "manual").astype(int)
        combined = (
            combined.sort_values(["_key", "_source_priority"])
            .drop_duplicates(subset=["_key"], keep="last")
            .drop(columns=["_key", "_source_priority"])
        )
        merged = combined

    merged = merged.sort_values("date").reset_index(drop=True)
    # Enforce column order and types
    merged["goals_a"] = merged["goals_a"].astype("Int64")
    merged["goals_b"] = merged["goals_b"].astype("Int64")
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    merged[_STORE_COLS].to_csv(_STORE, index=False)

    n_new = len(merged) - len(existing) if not existing.empty else len(merged)
    n_group = int((merged["stage"] == "group").sum())
    n_ko = int((merged["stage"] == "knockout").sum())
    return {"n_total": len(merged), "n_new": max(0, n_new), "n_group": n_group, "n_knockout": n_ko}


# ---------------------------------------------------------------------------
# Match augmentation
# ---------------------------------------------------------------------------

def augment_matches(matches: pd.DataFrame) -> pd.DataFrame:
    """Append WC2026 played rows not already in `matches`; re-sort by date.

    Uses the same match_id md5 scheme as load_matches (date|team_a|team_b|tournament).
    """
    results = load_wc2026_results()
    if results.empty:
        return matches

    wc_rows = []
    existing_ids = set(matches["match_id"].tolist()) if "match_id" in matches.columns else set()

    for _, r in results.iterrows():
        ta, tb = str(r.team_a), str(r.team_b)
        date_str = str(pd.Timestamp(r.date).date())
        mid = hashlib.md5(f"{date_str}|{ta}|{tb}|FIFA World Cup".encode()).hexdigest()[:12]
        if mid in existing_ids:
            continue
        row: dict = {
            "match_id": mid,
            "date": pd.Timestamp(r.date),
            "team_a": ta,
            "team_b": tb,
            "goals_a": int(r.goals_a),
            "goals_b": int(r.goals_b),
            "neutral": True,
            "tournament": "FIFA World Cup",
            "competition": "world_cup",
            "is_world_cup": True,
        }
        wc_rows.append(row)

    if not wc_rows:
        return matches

    extra = pd.DataFrame(wc_rows)
    # Align columns
    for col in matches.columns:
        if col not in extra.columns:
            extra[col] = None
    augmented = pd.concat([matches, extra[matches.columns]], ignore_index=True)
    return augmented.sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Fixtures update
# ---------------------------------------------------------------------------

def mark_fixtures_played(fixtures_path: Path | None = None) -> int:
    """Fill goals_a/goals_b in wc2026_fixtures.csv from the results store.

    Returns number of fixtures updated.
    """
    if fixtures_path is None:
        from wcpredictor.config import DATA_RAW as _DR
        fixtures_path = _DR / "wc2026_fixtures.csv"

    if not fixtures_path.exists():
        return 0

    fixtures = pd.read_csv(fixtures_path, parse_dates=["date"])
    results = load_wc2026_results()
    if results.empty:
        return 0

    results_lookup: dict[tuple, tuple[int, int]] = {}
    for _, r in results.iterrows():
        ta, tb = str(r.team_a), str(r.team_b)
        d = pd.Timestamp(r.date).date()
        ga, gb = int(r.goals_a), int(r.goals_b)
        results_lookup[(d, ta, tb)] = (ga, gb)
        results_lookup[(d, tb, ta)] = (gb, ga)

    n_updated = 0
    for idx, row in fixtures.iterrows():
        d = pd.Timestamp(row["date"]).date()
        ta = canonical(str(row["team_a"]))
        tb = canonical(str(row["team_b"]))
        key = (d, ta, tb)
        if key in results_lookup and (pd.isna(row.get("goals_a")) or str(row.get("goals_a", "")) == ""):
            ga, gb = results_lookup[key]
            fixtures.at[idx, "goals_a"] = ga
            fixtures.at[idx, "goals_b"] = gb
            n_updated += 1

    if n_updated:
        fixtures.to_csv(fixtures_path, index=False)

    return n_updated


# ---------------------------------------------------------------------------
# CLI (smoke test / manual ingestion)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Update WC2026 results store")
    parser.add_argument("--source-csv", default=None, help="Path to a manual results CSV")
    args = parser.parse_args()

    stats = update_wc2026_results(Path(args.source_csv) if args.source_csv else None)
    print(f"Results store: {stats}")
    n = mark_fixtures_played()
    print(f"Fixtures updated: {n}")
