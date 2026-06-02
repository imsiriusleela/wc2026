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

# Form features
FORM_WINDOW: int = 5                               # rolling window for recent-form features
REST_DAYS_CAP: int = 30                            # cap on days-since-last-match
