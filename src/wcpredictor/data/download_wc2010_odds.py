"""Parse betexplorer.com WC2010 odds from pre-saved HTML snapshots.

Source   : betexplorer.com/football/world/world-cup-2010/results/
Fetch    : one-time Playwright render (user-approved, see project CLAUDE.md).
Snapshots: data/raw/wc2010_odds_snapshot_groups.html  (stage=QN1QYX1j, 48 group matches)
           data/raw/wc2010_odds_snapshot_knockout.html (stage=lxOVoutC, 16 KO matches)
Odds type: 1X2 average decimal odds shown on the results page.
Fetch date: 2026-06-02.

The snapshots are the reproducibility anchor; this module re-parses them
deterministically on every run.  SHA-256 of the generated CSV is pinned in
config.py and verified after every parse.
"""

from __future__ import annotations

import csv
import hashlib
import re
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path

from wcpredictor.config import (
    DATA_RAW,
    WC2010_ODDS_CSV_SHA256,
)
from wcpredictor.data.normalize_teams import canonical

_SNAPSHOT_GROUPS = DATA_RAW / "wc2010_odds_snapshot_groups.html"
_SNAPSHOT_KNOCKOUT = DATA_RAW / "wc2010_odds_snapshot_knockout.html"
_CSV_DEST = DATA_RAW / "wc2010_odds.csv"


class _TableParser(HTMLParser):
    """Extract data rows from betexplorer results HTML."""

    def __init__(self) -> None:
        super().__init__()
        self._in_table = False
        self._table_depth = 0
        self._in_cell = False
        self._cell_buf = ""
        self._current_row: list[str] = []
        self.rows: list[list[str]] = []

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag == "table":
            self._table_depth += 1
            self._in_table = True
        elif tag == "tr" and self._in_table:
            self._current_row = []
        elif tag in ("td", "th") and self._in_table:
            self._cell_buf = ""
            self._in_cell = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "table":
            self._table_depth -= 1
            if self._table_depth == 0:
                self._in_table = False
        elif tag == "tr" and self._in_table:
            if self._current_row:
                self.rows.append(self._current_row[:])
        elif tag in ("td", "th") and self._in_table:
            self._current_row.append(self._cell_buf.strip())
            self._in_cell = False

    def handle_data(self, data: str) -> None:
        if self._in_cell:
            self._cell_buf += data


def _parse_snapshot(path: Path) -> list[dict]:
    """Parse one betexplorer snapshot into a list of match dicts."""
    parser = _TableParser()
    parser.feed(path.read_text(encoding="utf-8", errors="replace"))

    matches: list[dict] = []
    for row in parser.rows:
        # Expected columns: [match_name, score, odds_h, odds_d, odds_a, date]
        if len(row) < 5:
            continue
        try:
            o_h = float(row[2])
            o_d = float(row[3])
            o_a = float(row[4])
        except (ValueError, IndexError):
            continue
        if o_h <= 1.0 or o_d <= 1.0 or o_a <= 1.0:
            continue

        match_text = row[0].strip()
        if " - " not in match_text:
            continue

        date_raw = row[5].strip() if len(row) > 5 else ""
        try:
            date = datetime.strptime(date_raw, "%d.%m.%Y").date().isoformat()
        except ValueError:
            date = date_raw

        home_raw, _, away_raw = match_text.partition(" - ")
        home = canonical(home_raw.strip())
        away = canonical(away_raw.strip())
        if not home or not away:
            continue

        matches.append(
            {"date": date, "home": home, "away": away, "odds_h": o_h, "odds_d": o_d, "odds_a": o_a}
        )
    return matches


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_wc2010_odds(force: bool = False) -> Path:
    """Parse snapshots → data/raw/wc2010_odds.csv; verify pinned SHA-256.

    Returns the path to the CSV.  Skips re-parsing if CSV exists and hash
    matches, unless *force* is True.
    """
    if _CSV_DEST.exists() and not force:
        if _sha256(_CSV_DEST) == WC2010_ODDS_CSV_SHA256:
            return _CSV_DEST

    for snap in (_SNAPSHOT_GROUPS, _SNAPSHOT_KNOCKOUT):
        if not snap.exists():
            raise FileNotFoundError(
                f"Snapshot not found: {snap}\n"
                "Re-render with Playwright and save to data/raw/ before running."
            )

    rows = _parse_snapshot(_SNAPSHOT_GROUPS) + _parse_snapshot(_SNAPSHOT_KNOCKOUT)

    if len(rows) != 64:
        raise RuntimeError(
            f"Expected 64 WC2010 matches but parsed {len(rows)}. "
            "Check snapshot integrity."
        )

    DATA_RAW.mkdir(parents=True, exist_ok=True)
    with _CSV_DEST.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["date", "home", "away", "odds_h", "odds_d", "odds_a"])
        writer.writeheader()
        writer.writerows(rows)

    actual = _sha256(_CSV_DEST)
    if actual != WC2010_ODDS_CSV_SHA256:
        _CSV_DEST.unlink()
        raise RuntimeError(
            f"SHA-256 mismatch after parsing.\n"
            f"  expected : {WC2010_ODDS_CSV_SHA256}\n"
            f"  actual   : {actual}\n"
            "Update WC2010_ODDS_CSV_SHA256 in config.py if snapshots were intentionally refreshed."
        )

    print(f"WC2010 odds parsed and verified: {_CSV_DEST} ({len(rows)} matches)")
    return _CSV_DEST


if __name__ == "__main__":
    parse_wc2010_odds(force=True)
