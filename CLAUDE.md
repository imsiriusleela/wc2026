# World Cup Match Predictor

## Objective
Build a pre-match FIFA World Cup predictor that outputs:
1. Win / draw / loss probabilities for 90-minute regulation time
2. Full exact-score probability matrix
3. Top-N scorelines
4. Expected goals for each team
5. Tournament simulation probabilities later

## Core modelling philosophy
Do not build a black-box-only model first.

The system should be built in this order:
1. Data pipeline
2. Elo / rating baseline
3. Independent Poisson score model
4. Dixon-Coles score model
5. XGBoost / LightGBM outcome overlay
6. Calibration
7. Ensemble / stacking
8. Tournament simulator
9. API + frontend

## Important modelling constraints
- Default label is 90-minute regulation-time result.
- Extra time and penalties are separate future modules.
- Avoid data leakage. All features must be available before kickoff.
- Use time-aware train/test splits.
- Prefer probabilistic outputs over hard predictions.
- Evaluate with log loss, Brier score, calibration, accuracy, macro-F1, scoreline log score, MAE/RMSE goals, and exact-score hit rate.

## Data assumptions
Use open/prototype data first:
- International match results — MVP source is the `martj42/international_results` dataset, fetched via an explicit, pinned (commit-SHA) download (honors the no-silent-scraping rule).
- FIFA ranking / Elo ratings
- World Cup historical results
- Optional later: squad values, lineups, injuries, xG/event data

## Engineering rules
- Python project.
- Use uv for environment and dependency management (`uv run`, `uv sync`).
- Use src/ layout.
- Use pandas/polars for data processing.
- Use scikit-learn for baseline ML.
- Use scipy/statsmodels for Poisson where useful.
- Keep every model reproducible.
- Add tests for data leakage and feature cutoffs.
- Do not silently scrape websites without explicit approval.
- Do not hard-code 2026 teams into modelling logic.

## First MVP
The first shippable version should:
- Load historical international match data.
- Build team Elo ratings over time.
- Train a Poisson score model.
- Predict W/D/L and exact score for arbitrary Team A vs Team B.
- Backtest on past World Cup matches.