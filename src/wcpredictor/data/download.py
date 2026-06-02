"""Download pinned international match results dataset.

Source: martj42/international_results (MIT)
Pinned SHA in config.RESULTS_URL for reproducible backtests.
Falls back to master branch if pinned SHA is unreachable.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

from wcpredictor.config import DATA_RAW, RESULTS_URL, RESULTS_URL_FALLBACK

_DEST = DATA_RAW / "results.csv"


def download_results(force: bool = False) -> Path:
    """Fetch results.csv to data/raw/; skip if already present unless force=True."""
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    if _DEST.exists() and not force:
        print(f"Already downloaded: {_DEST}")
        return _DEST

    for url in (RESULTS_URL, RESULTS_URL_FALLBACK):
        try:
            print(f"Downloading from {url} …")
            urllib.request.urlretrieve(url, _DEST)
            print(f"Saved to {_DEST}")
            return _DEST
        except Exception as exc:
            print(f"  Failed ({exc}), trying fallback …")

    raise RuntimeError("Could not download results.csv from any source.")


if __name__ == "__main__":
    download_results()
