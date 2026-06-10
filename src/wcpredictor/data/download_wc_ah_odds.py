"""Fetch WC AH/O-U odds from betexplorer.com API (one-time, user-approved).

Source   : betexplorer.com per-match API for WC 2010, 2014, 2018, 2022.
Approval : user approved 2026-06-09 ("ok do it").
Output   : data/raw/wc_ah_odds.csv  (year, date, home, away,
                                      ah_line, ah_home_odds, ah_away_odds,
                                      ou_line, over_odds, under_odds)

Stage IDs (discovered 2026-06-09):
  2010: groups=QN1QYX1j  knockout=lxOVoutC
  2014: groups=61tCiOIs  knockout=lMyc6LHg
  2018: groups=OneVXSrp  knockout=6BpzXnbj
  2022: groups=zkyDYRLU  knockout=823QwKIu

API endpoint pattern (discovered 2026-06-09):
  GET /match-odds/{match_id}/1/{bet_type}/bestOdds/?lang=en
  bet_type: ah | ou
  Returns JSON {"odds": "<html>..."} with per-bookmaker odds tables.
  Main line identified by id="oddsComparison__activeSubLi" in sub-nav.
  Per-table odds extracted via id="best-odds-{line}".

SHA-256  : WC_AH_ODDS_CSV_SHA256 in config.py; update after initial fetch.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import time
import urllib.request
from pathlib import Path

from wcpredictor.config import DATA_RAW, WC_AH_ODDS_CSV_SHA256
from wcpredictor.data.normalize_teams import canonical

_CSV_DEST = DATA_RAW / "wc_ah_odds.csv"

# WC year → [groups_stage_id, knockout_stage_id]
_STAGES: dict[int, list[str]] = {
    2010: ["QN1QYX1j", "lxOVoutC"],
    2014: ["61tCiOIs", "lMyc6LHg"],
    2018: ["OneVXSrp", "6BpzXnbj"],
    2022: ["zkyDYRLU", "823QwKIu"],
}

_WC_YEARS = [2010, 2014, 2018, 2022]

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/149.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-GB,en;q=0.9",
}

_API_HEADERS = {
    **_HEADERS,
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "X-Requested-With": "XMLHttpRequest",
}

_DELAY = 0.8  # seconds between requests to avoid rate-limiting


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _get(url: str, is_api: bool = False, timeout: int = 15) -> str:
    h = _API_HEADERS if is_api else _HEADERS
    req = urllib.request.Request(url, headers=h)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="replace")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ─── Results-page scraper ─────────────────────────────────────────────────────

def _fetch_match_ids(year: int, stage: str) -> list[dict]:
    """Return list of {match_id, home, away, date} for one stage."""
    from datetime import datetime

    url = (
        f"https://www.betexplorer.com/football/world/"
        f"world-cup-{year}/results/?stage={stage}"
    )
    html = _get(url)
    time.sleep(_DELAY)

    # Each match row has an anchor with class="in-match" containing team names in <span>s
    # href="/football/world/world-cup-YYYY/slug/MATCHID/" class="in-match"
    link_re = re.compile(
        rf'href="/football/world/world-cup-{year}/[a-z0-9-]+/([A-Za-z0-9]+)/" class="in-match">(.*?)</a>',
        re.DOTALL | re.IGNORECASE,
    )
    date_re = re.compile(r"(\d{2}\.\d{2}\.\d{4})")
    strip_tags = re.compile(r"<[^>]+>")

    rows_html = re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.DOTALL)
    matches: dict[str, dict] = {}
    for row in rows_html:
        m = link_re.search(row)
        if not m:
            continue
        match_id = m.group(1)
        inner = m.group(2)
        teams_text = strip_tags.sub("", inner).strip()
        if " - " not in teams_text:
            continue
        date_m = date_re.search(row)
        date_str = ""
        if date_m:
            try:
                date_str = datetime.strptime(date_m.group(1), "%d.%m.%Y").date().isoformat()
            except ValueError:
                date_str = date_m.group(1)

        home_raw, _, away_raw = teams_text.partition(" - ")
        home = canonical(home_raw.strip())
        away = canonical(away_raw.strip())
        if not home or not away:
            continue
        if match_id not in matches:
            matches[match_id] = {
                "match_id": match_id,
                "home": home,
                "away": away,
                "date": date_str,
            }

    return list(matches.values())


# ─── Odds API fetcher ─────────────────────────────────────────────────────────

def _fetch_odds(match_id: str, bet_type: str) -> dict | None:
    """Fetch AH or O/U best odds for *match_id*.  Returns {line, odds1, odds2} or None."""
    referer = (
        f"https://www.betexplorer.com/football/world/"
        f"world-cup-2022/{match_id}/"
    )
    url = (
        f"https://www.betexplorer.com/match-odds/{match_id}/1/"
        f"{bet_type}/bestOdds/?lang=en"
    )
    headers = {**_API_HEADERS, "Referer": referer}
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            raw = r.read().decode("utf-8", errors="replace")
    except Exception:
        return None

    try:
        data = json.loads(raw)
        html = data.get("odds", "")
    except (json.JSONDecodeError, AttributeError):
        return None

    # Identify the active (main) line
    main_m = re.search(r'id="(-?[\d.]+)"\s+class="oddsComparison__activeSubLi', html)
    if not main_m:
        # Try alternate pattern
        main_m = re.search(r'oddsComparison__activeSubLi[^"]*"\s+id="(-?[\d.]+)"', html)
    if not main_m:
        return None
    line = float(main_m.group(1))

    # Extract odds from the matching table section.
    # For O/U: betexplorer always uses id="best-odds-ou" for the active line table.
    # For AH: numeric IDs e.g. id="best-odds--0.50" or id="best-odds-0.50".
    if bet_type == "ou":
        search_ids = ["ou"]
    else:
        # Try both numeric formats: -0.50 and -0.5
        search_ids = [f"{line:.2f}", f"{line:g}"]

    for fmt in search_ids:
        table_start = html.find(f'id="best-odds-{fmt}"')
        if table_start >= 0:
            section = html[table_start : table_start + 3000]
            odds_vals = re.findall(r'data-odd="([\d.]+)"', section)
            if len(odds_vals) >= 2:
                try:
                    o1, o2 = float(odds_vals[0]), float(odds_vals[1])
                    if o1 > 1.01 and o2 > 1.01:
                        return {"line": line, "odds1": o1, "odds2": o2}
                except (ValueError, IndexError):
                    pass

    return None


# ─── O/U patch ───────────────────────────────────────────────────────────────

def patch_ou_odds(csv_path: Path | None = None) -> Path:
    """Backfill missing O/U data for rows that already have AH odds.

    Reads the existing CSV, fetches O/U for any row where ou_line is empty,
    and rewrites the CSV.  Faster than a full rerun when AH data is already present.
    """
    dest = csv_path or _CSV_DEST
    if not dest.exists():
        raise FileNotFoundError(f"CSV not found: {dest}")

    rows: list[dict] = []
    with dest.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)

    needs_ou = [r for r in rows if not r.get("ou_line")]
    print(f"Rows missing O/U: {len(needs_ou)} of {len(rows)}", flush=True)

    # Build match_id lookup from match pages (need to re-fetch IDs by team/date)
    # Strategy: only fetch match IDs for years with missing O/U, then match by home+away+date
    years_needed = sorted({int(r["year"]) for r in needs_ou})
    match_id_lookup: dict[tuple[str, str, str], str] = {}  # (home, away, date) → match_id
    for year in years_needed:
        print(f"  Re-fetching match IDs for WC {year}…", flush=True)
        for stage in _STAGES[year]:
            for m in _fetch_match_ids(year, stage):
                key = (m["home"], m["away"], m["date"])
                match_id_lookup[key] = m["match_id"]

    updated = 0
    for row in rows:
        if row.get("ou_line"):
            continue
        key = (row["home"], row["away"], row["date"])
        mid = match_id_lookup.get(key)
        if not mid:
            # Try flipped (away vs home lookup)
            key_flip = (row["away"], row["home"], row["date"])
            mid = match_id_lookup.get(key_flip)
        if not mid:
            print(f"  WARNING: no match_id for {row['home']} vs {row['away']} ({row['date']})", flush=True)
            continue
        time.sleep(_DELAY)
        ou = _fetch_odds(mid, "ou")
        if ou:
            row["ou_line"] = ou["line"]
            row["over_odds"] = ou["odds1"]
            row["under_odds"] = ou["odds2"]
            updated += 1
            print(f"  Patched O/U for {row['home']} vs {row['away']}: {ou['line']} ({ou['odds1']}/{ou['odds2']})", flush=True)
        else:
            print(f"  No O/U data for {row['home']} vs {row['away']}", flush=True)

    print(f"\nPatched {updated} rows.", flush=True)

    fieldnames = ["year", "date", "home", "away", "ah_line", "ah_home_odds",
                  "ah_away_odds", "ou_line", "over_odds", "under_odds"]
    with dest.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    actual = _sha256(dest)
    print(f"Updated CSV SHA-256: {actual}", flush=True)
    print("Pin this value as WC_AH_ODDS_CSV_SHA256 in config.py", flush=True)
    return dest


# ─── Main fetch ───────────────────────────────────────────────────────────────

def fetch_wc_ah_odds(force: bool = False) -> Path:
    """Fetch betexplorer WC AH/O-U odds → data/raw/wc_ah_odds.csv.

    Rate-limited at ~0.8 s/request.  Skips CSV if it exists and SHA matches.
    On first run, prints the SHA-256 to pin in config.py.
    """
    if _CSV_DEST.exists() and not force:
        if WC_AH_ODDS_CSV_SHA256 and _sha256(_CSV_DEST) == WC_AH_ODDS_CSV_SHA256:
            print(f"CSV already up-to-date: {_CSV_DEST}")
            return _CSV_DEST

    all_rows: list[dict] = []

    for year in _WC_YEARS:
        print(f"\n=== WC {year} ===")
        stage_ids = _STAGES[year]
        year_matches: list[dict] = []

        for stage in stage_ids:
            stage_matches = _fetch_match_ids(year, stage)
            print(f"  Stage {stage}: {len(stage_matches)} matches")
            year_matches.extend(stage_matches)

        # Deduplicate (some matches might appear in multiple stage pages)
        seen: set[str] = set()
        unique: list[dict] = []
        for m in year_matches:
            if m["match_id"] not in seen:
                seen.add(m["match_id"])
                unique.append(m)
        print(f"  Total unique matches: {len(unique)}")

        for i, match in enumerate(unique):
            mid = match["match_id"]
            print(f"  [{i+1}/{len(unique)}] {match['home']} vs {match['away']} ({match['date']})", end="  ", flush=True)

            time.sleep(_DELAY)
            ah = _fetch_odds(mid, "ah")
            time.sleep(_DELAY)
            ou = _fetch_odds(mid, "ou")

            row: dict = {
                "year": year,
                "date": match["date"],
                "home": match["home"],
                "away": match["away"],
                "ah_line": ah["line"] if ah else "",
                "ah_home_odds": ah["odds1"] if ah else "",
                "ah_away_odds": ah["odds2"] if ah else "",
                "ou_line": ou["line"] if ou else "",
                "over_odds": ou["odds1"] if ou else "",
                "under_odds": ou["odds2"] if ou else "",
            }
            all_rows.append(row)

            ah_str = f"AH {ah['line']} ({ah['odds1']}/{ah['odds2']})" if ah else "AH n/a"
            ou_str = f"O/U {ou['line']} ({ou['odds1']}/{ou['odds2']})" if ou else "O/U n/a"
            print(f"{ah_str}  {ou_str}")

    DATA_RAW.mkdir(parents=True, exist_ok=True)
    fieldnames = ["year", "date", "home", "away", "ah_line", "ah_home_odds",
                  "ah_away_odds", "ou_line", "over_odds", "under_odds"]
    with _CSV_DEST.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    actual = _sha256(_CSV_DEST)
    print(f"\nWrote {len(all_rows)} rows to {_CSV_DEST}")
    print(f"SHA-256: {actual}")
    print("Pin this value as WC_AH_ODDS_CSV_SHA256 in config.py")

    if WC_AH_ODDS_CSV_SHA256 and actual != WC_AH_ODDS_CSV_SHA256:
        _CSV_DEST.unlink()
        raise RuntimeError(
            f"SHA-256 mismatch.\n  expected: {WC_AH_ODDS_CSV_SHA256}\n  actual: {actual}\n"
            "Update WC_AH_ODDS_CSV_SHA256 in config.py if intentionally refreshed."
        )

    return _CSV_DEST


# ─── Legacy snapshot parser (kept for reproducibility) ───────────────────────

def parse_wc_ah_odds(force: bool = False) -> Path:
    """Alias for fetch_wc_ah_odds() — kept for backward compatibility."""
    return fetch_wc_ah_odds(force=force)


if __name__ == "__main__":
    fetch_wc_ah_odds(force=True)
