# HANDOFF — Deferred: odds refresh (~2026-06-09)

> Phase 3.9 committed: da624f6  
> Phase 4.0 committed: 7ed152e  
> Suite: 135 + 7 new (TestEnsembleMkt) tests passing.

## Status

`ensemble_mkt` is live and wired into `predict_fixtures`.  Without the
`WorldCup2026` sheet the model falls back to plain ensemble (α=0).

## Next action (~2026-06-09 when WC2026 odds publish)

1. `uv run python -m wcpredictor.data.download_odds` — re-download
   `WorldCup_fdco.xlsx`; update `config.FDCO_ODDS_SHA256` with the new hash.
2. Re-run the backtest to populate `odds_alpha_pooled` in
   `data/processed/backtest_report.json`:
   ```bash
   uv run python -m wcpredictor.evaluation.backtest
   ```
3. Regenerate predictions:
   ```bash
   uv run python -c "
   from wcpredictor.predict import predict_fixtures
   predict_fixtures('2026-06-10', models=['poisson','ensemble','ensemble_mkt'])
   "
   ```
4. Verify `ensemble_mkt` means diverge from `ensemble` (market weight applied).

## Deferred (future phases)

- **Knockout fixtures** — after group stage (2026-06-27): standings → R32 bracket → predict.
  Needs extra-time/penalty resolver.
- **Tournament simulator** — Monte Carlo over groups + knockout; per-team win-cup probs.
  Reuses `_build_frozen_state` / `_predict_one_frozen`.
- **Matchday loop** — `uv run python -m wcpredictor.evaluation.live --as-of <next-day>` after
  each matchday (first real results 2026-06-11); per-model scorecard in
  `data/processed/wc2026_scorecard.json`.
