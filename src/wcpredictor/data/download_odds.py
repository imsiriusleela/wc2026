"""Download and cache football-data.co.uk World Cup odds xlsx.

Source : football-data.co.uk (free, non-commercial use)
URL    : config.FDCO_ODDS_URL  (pinned to the file as of 2026-05-28)
SHA-256: config.FDCO_ODDS_SHA256  (verify after every refresh)

The file contains historical WC match odds sheets for 2014, 2018, and 2022
alongside 2026 qualifier data.  We consume only the historical WC sheets.
"""

from __future__ import annotations

import hashlib
import ssl
import urllib.request
from pathlib import Path

from wcpredictor.config import DATA_RAW, FDCO_ODDS_SHA256, FDCO_ODDS_URL

_DEST = DATA_RAW / "WorldCup_fdco.xlsx"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download_odds(force: bool = False, verify: bool = True) -> Path:
    """Fetch WorldCup_fdco.xlsx to data/raw/; skip if already present and hash matches.

    verify=False: download and save even if the SHA differs from the pinned value.
    Use this for on-demand refresh (the new SHA is surfaced to the caller for re-pinning).
    """
    DATA_RAW.mkdir(parents=True, exist_ok=True)

    if _DEST.exists() and not force:
        if _sha256(_DEST) == FDCO_ODDS_SHA256:
            return _DEST
        print(f"SHA-256 mismatch on cached file — re-downloading …")

    print(f"Downloading odds from {FDCO_ODDS_URL} …")
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    with urllib.request.urlopen(FDCO_ODDS_URL, context=ctx) as resp:
        data = resp.read()
    _DEST.write_bytes(data)

    actual = _sha256(_DEST)
    if actual != FDCO_ODDS_SHA256:
        if not verify:
            print(
                f"SHA-256 changed: {actual} (pinned: {FDCO_ODDS_SHA256})"
                " — file saved; re-pin FDCO_ODDS_SHA256 in config.py for reproducibility."
            )
            return _DEST
        _DEST.unlink()
        raise RuntimeError(
            f"SHA-256 mismatch after download.\n"
            f"  expected : {FDCO_ODDS_SHA256}\n"
            f"  actual   : {actual}\n"
            "Update FDCO_ODDS_SHA256 in config.py if the file was intentionally refreshed."
        )
    print(f"Saved and verified: {_DEST}")
    return _DEST


if __name__ == "__main__":
    download_odds()
