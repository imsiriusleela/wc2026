# World Cup 2026 Match Predictor

Pre-match FIFA World Cup predictor. Given two teams, it returns win/draw/loss probabilities, expected goals, an exact-score matrix, and the top-5 scorelines.

## Setup

```bash
uv sync --extra dev
```

## Models

| Model | `model=` arg | Description |
|-------|-------------|-------------|
| Ensemble+Market **(default)** | `"ensemble_mkt"` | Calibrated ensemble blended with bookmaker market odds (α≈0.27); auto-degrades to ens_cal when odds are unavailable |
| Ensemble | `"ensemble"` | Calibrated blend of Poisson + DC + multinomial-logistic + GBM; leakage-safe stacking |
| Poisson | `"poisson"` | 2-param Elo→λ model; fast, Elo-derived rates |
| Dixon-Coles | `"dixon_coles"` | Per-team attack/defense strengths + τ low-score correction + time-decay weighting |

## Verification commands

```bash
# Download data
uv run python -m wcpredictor.data.download

# Predict with the default (ensemble_mkt — auto-degrades to ens_cal until odds arrive)
uv run python -c "from wcpredictor.predict import predict_match; print(predict_match('Brazil','France','2026-06-15',neutral=True))"

# Predict with Dixon-Coles
uv run python -c "from wcpredictor.predict import predict_match; print(predict_match('Brazil','France','2026-06-15',neutral=True,model='dixon_coles'))"

# Predict with calibrated ensemble (no market blending)
uv run python -c "from wcpredictor.predict import predict_match; print(predict_match('Brazil','France','2026-06-15',neutral=True,model='ensemble'))"

# Run backtest (World Cup holdouts 2010–2022; Poisson, DC, DC+Cal, Ensemble, Ens+Cal, Ens+Mkt, baselines)
uv run python -m wcpredictor.evaluation.backtest

# Run bootstrap model selection (reads backtest_permatch.csv; prints CI table and default recommendation)
uv run python -m wcpredictor.evaluation.model_select

# Run all tests
uv run pytest -q
```

## Reading the backtest report

`data/processed/backtest_report.json` contains per-year results for four models:

- **`model`** — Poisson (Elo-derived rates, 2 global params)
- **`model_dc`** — Dixon-Coles (per-team α/β, ρ, γ, time-decay weights)
- **`model_dc_cal`** — Dixon-Coles with temperature scaling; `temperature`, `ece_before_val`, `ece_after_val` measure calibration improvement on the pre-tournament validation slice
- **`baseline_most_common`** / **`baseline_elo_only`** — naive baselines

Key metrics: `log_loss`, `brier`, `accuracy`, `goal_mae`, `goal_rmse`, `exact_score_logscore`, `top5_hit_rate`.

The ensemble adds two more blocks:
- **`model_ensemble`** — log-opinion pool over {Poisson, DC, multinomial-logistic}; leakage-safe weights fit on the out-of-time validation slice; score matrix is a Poisson+DC linear blend (logistic contributes W/D/L only).
- **`model_ensemble_cal`** — same, with temperature calibration; `weights_poisson/dc/logistic` show the fitted blend; `ece_before_val` / `ece_after_val` measure calibration on the validation slice.

**Final backtest result (2010–2022, full pipeline):**

| Fold | Poisson | DC+Cal | Ens+Cal | **Ens+Mkt** | α (mkt) |
|------|---------|--------|---------|-------------|---------|
| 2010 | 0.9828 | **0.9474** | 0.9644 | 0.9644 | 0.000 |
| 2014 | **0.9152** | 0.9502 | 0.9233 | 0.9231 | 0.045 |
| 2018 | 0.9690 | 0.9901 | 0.9563 | **0.9541** | 0.080 |
| 2022 | 1.0541 | 1.1104 | 1.0682 | **1.0349** | 0.273 |
| mean | 0.9803 | 0.9995 | 0.9780 | **0.9691** | — |

Bootstrap model selection (10 000 resamples, 256 matched pairs): Ens+Mkt vs Ens+Cal Δ=−0.0089, 95% CI [−0.0186, −0.0016] entirely negative; P(Ens+Mkt better)=99.5%. Default promoted to `"ensemble_mkt"`. Market blend uses time-aware α fit on past WC folds; auto-degrades to ens_cal when 2026 odds are unavailable (pre–2026-06-09 window).

## Kickoff-day runbook (2026)

### One-time setup
```bash
export ODDS_API_KEY=<your-key>    # from the-odds-api.com
```

### Near kickoff — refresh closing lines
```bash
# Refreshes 1X2 odds (ensemble_mkt blend) + AH/O-U (matrix blend) in one call.
# Archives the prior JSON snapshot before overwriting.
curl -s -X POST localhost:8001/refresh-odds | jq
# Expected: n_odds_2026==72, odds_api_refreshed==true, HTTP 200
```

### Regenerate Fixtures / Tournament snapshots
After refreshing odds, the `/predict` endpoint updates immediately (cache cleared).
To update the Fixtures and Tournament tabs (pre-generated snapshots), re-run:
```bash
uv run python -m wcpredictor.data.predict_fixtures --as-of 2026-06-11
uv run python -m wcpredictor.cli simulate --as-of 2026-06-11 --n-sims 20000 --seed 42
```

### Knockout stage
When group-stage results are known, the API returns knockout matches with real team names.
Run another `curl -X POST localhost:8001/refresh-odds` to pick them up; placeholder events
("Winner Group A") are skipped until names are resolved.

### If fdco publishes the WorldCup2026 sheet
The `/refresh-odds` response surfaces the new SHA in `file_sha256`.
Re-pin `FDCO_ODDS_SHA256` in `config.py` and commit; fdco rows then take per-match
precedence over the API automatically — no code change needed.

### API quota note
One `/refresh-odds` call consumes one the-odds-api request (metered free tier).
