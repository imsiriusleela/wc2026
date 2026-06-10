from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]

DATA_RAW = _ROOT / "data" / "raw"
DATA_PROCESSED = _ROOT / "data" / "processed"

# Pinned commit SHA for reproducibility
_COMMIT_SHA = "dda8a418608cab5fea3f55b7fe6c6c801a38a906"
RESULTS_URL = (
    f"https://raw.githubusercontent.com/martj42/international_results/"
    f"{_COMMIT_SHA}/results.csv"
)
# Fallback to master if pinned commit is unavailable (documented trade-off)
RESULTS_URL_FALLBACK = (
    "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
)

# Elo parameters
INITIAL_RATING: float = 1500.0
K_MAP: dict[str, float] = {
    "friendly": 15.0,
    "qualifier": 25.0,
    "continental": 35.0,
    "world_cup": 45.0,
}
HOME_ADVANTAGE: float = 50.0

# Poisson score model
MAX_GOALS: int = 8

# Dixon-Coles model
DC_HALF_LIFE_DAYS: int = 730                       # ~2-year half-life
DC_TIME_DECAY_XI: float = 0.693147 / 730           # ln(2) / half_life
DC_MIN_MATCHES: int = 5                            # min participations for own α/β
DC_TRAIN_WINDOW_YEARS: int = 10                    # restrict training to last N years
DC_RHO_INIT: float = -0.13                         # initial ρ (low-score dependence)
DC_HOME_ADV_INIT: float = 0.3                      # initial γ (log-scale home advantage)
DC_CAL_VALIDATION_YEARS: int = 2                   # validation slice length for calibration

# Ensemble
ENSEMBLE_POOL: str = "log"                         # "log" (log-opinion) or "linear"

# Market-odds blending (football-data.co.uk WC xlsx)
FDCO_ODDS_URL: str = "https://www.football-data.co.uk/WorldCup2026.xlsx"
# SHA-256 of WorldCup_fdco.xlsx as of 2026-05-28; verify with download_odds.py
FDCO_ODDS_SHA256: str = "777652ce1a400aabbbb5778f306311f36197b7e382ff87d5077b5a017473ae72"
ODDS_ALPHA_PRIOR: float = 0.0  # conservative: no market weight until time-aware data supports it
# Unconstrained pooled fit is ~0.64, driven by WC2022; cap keeps market a low-weight overlay
ODDS_ALPHA_CAP: float = 0.3

# WC2010 odds from betexplorer.com (one-time Playwright render, user-approved 2026-06-02)
# Source: betexplorer.com/football/world/world-cup-2010/results/
# Snapshots: data/raw/wc2010_odds_snapshot_groups.html, wc2010_odds_snapshot_knockout.html
WC2010_ODDS_URL: str = "https://www.betexplorer.com/football/world/world-cup-2010/results/"
# SHA-256 of data/raw/wc2010_odds.csv generated from the Playwright snapshots
WC2010_ODDS_CSV_SHA256: str = "602bb2c598a91e81359e2728537a300edaa1cc5571c1dec0cc1be4c85c0a52b7"

# WC AH/totals odds from betexplorer.com (one-time Playwright render, user approval required)
# Source: betexplorer.com per-match AH/O-U pages for WC 2010, 2014, 2018, 2022
# Snapshots: data/raw/wc_ah_odds_snapshot_{year}_{ah|ou}.html
# SHA-256 of data/raw/wc_ah_odds.csv — set after parse_wc_ah_odds(force=True)
WC_AH_ODDS_CSV_SHA256: str = "95df1a65a17c68413b8eaa1cdec92978e32b5f31df25a39289e45701962132f7"

# Form features
FORM_WINDOW: int = 5                               # rolling window for recent-form features
REST_DAYS_CAP: int = 30                            # cap on days-since-last-match

# Asian handicap and Asian totals market lines (standard AH notation; negative = home gives)
ASIAN_HANDICAP_LINES: list[float] = [
    -2.5, -2.0, -1.75, -1.5, -1.25, -1.0, -0.75, -0.5, -0.25,
    0.0,
    0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5,
]
ASIAN_TOTAL_LINES: list[float] = [
    0.5, 1.0, 1.5, 2.0, 2.25, 2.5, 2.75, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5,
]

# Asian-handicap odds blend (Phase 9.4; conservative cap mirrors ODDS_ALPHA_CAP)
AH_ALPHA_CAP: float = 0.3
AH_ALPHA_PRIOR: float = 0.0
