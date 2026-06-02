# World Cup 2026 Match Predictor

Pre-match FIFA World Cup predictor. Given two teams, it returns win/draw/loss probabilities, expected goals, an exact-score matrix, and the top-5 scorelines.

## Setup

```bash
uv sync --extra dev
```

## Models

| Model | `model=` arg | Description |
|-------|-------------|-------------|
| Poisson (default) | `"poisson"` | 2-param Elo→λ model; fast, Elo-derived rates |
| Dixon-Coles | `"dixon_coles"` | Per-team attack/defense strengths + τ low-score correction + time-decay weighting |
| Ensemble | `"ensemble"` | Calibrated blend of Poisson + DC + multinomial-logistic; leakage-safe stacking |

## Verification commands

```bash
# Download data
uv run python -m wcpredictor.data.download

# Predict with the default Poisson model
uv run python -c "from wcpredictor.predict import predict_match; print(predict_match('Brazil','France','2026-06-15',neutral=True))"

# Predict with Dixon-Coles
uv run python -c "from wcpredictor.predict import predict_match; print(predict_match('Brazil','France','2026-06-15',neutral=True,model='dixon_coles'))"

# Predict with the ensemble
uv run python -c "from wcpredictor.predict import predict_match; print(predict_match('Brazil','France','2026-06-15',neutral=True,model='ensemble'))"

# Run backtest (World Cup holdouts 2010–2022; Poisson, DC, DC+Cal, Ensemble, Ens+Cal, baselines)
uv run python -m wcpredictor.evaluation.backtest

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

**Phase 3 backtest result (2010–2022):**

| Fold | Poisson | DC+Cal | Ens+Cal |
|------|---------|--------|---------|
| 2010 | 0.9828 | **0.9474** | 0.9683 |
| 2014 | **0.9152** | 0.9376 | 0.9200 |
| 2018 | 0.9690 | 0.9904 | **0.9713** |
| 2022 | 0.9541 | 1.1105 | **0.9605** |
| mean | 0.9803 | 0.9965 | **0.9800** |

The ensemble wins on aggregate and dominates DC+Cal in 3/4 folds, but falls behind DC+Cal in 2010 and fractionally behind Poisson in 2014–2018.  Per the build rules, the default stays `"poisson"` until the ensemble wins on every fold.  Weights converge to ≈1/3 each — the thin 2-year validation slice gives the optimizer insufficient signal to differentiate members, and the logistic member encodes similar Elo information to the Poisson model.  Consider revisiting with a longer validation window or richer features (xG, squad depth) before promoting ensemble as the default.
