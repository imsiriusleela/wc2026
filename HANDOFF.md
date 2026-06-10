# HANDOFF — Phase 9 (Asian handicap & Asian totals) SHIPPED

> Implemented 2026-06-09; promotion gate run and artifacts finalized **2026-06-10**.
> Full suite **270 passed, 1 skipped**. WC2026 kicks off **2026-06-11**.
> The fdco odds-day runbook at the bottom is the standing action — the WC2026 sheet
> has **still not published** as of 2026-06-10 (`load_wc_odds()` has 0 rows for 2026).

## What shipped

- **9.1 Derived markets** — `src/wcpredictor/markets/asian.py`: goal-diff / total-goals
  distributions, `settle_line` (whole/half/quarter lines, half-win/half-loss), fair odds,
  `ladder()`. Attached as a `"markets"` block to every `predict_match` / fixtures response
  for all model branches. Config: `ASIAN_HANDICAP_LINES`, `ASIAN_TOTAL_LINES`.
- **9.2 Market AH/totals data** —
  - Historical 2010–2022: `data/raw/wc_ah_odds.csv` (betexplorer one-time Playwright render,
    user-approved; 256 rows = 64/WC; AH line missing for 15), pinned via
    `WC_AH_ODDS_CSV_SHA256` in `config.py`.
  - Live 2026: `data/download_odds_api.py` → `data/raw/odds_api_wc2026.json`
    (the-odds-api.com `spreads,totals`, key in `ODDS_API_KEY`, graceful degrade).
  - Parser: `features/ah_odds.py` (`load_wc_ah_odds` merges live 2026 rows;
    `align_ah_to_test`, `merge_ah_features`).
- **9.3 AH evaluation** — `evaluation/metrics.py`: `ah_cover_brier`, `ah_cover_calibration`,
  `closing_line_value`, `ah_roi`. Backtest settles AH per fold; report carries
  `model_ensemble_ah` / `model_ensemble_ah_market` per fold.
- **9.4 Market-AH matrix blend** — market AH+O/U implied probs → implied Poisson matrix
  (prob-based bisection inversion, robust to betexplorer's always-active −0.5 tab) →
  `M' = (1−α)·M_model + α·M_market`. α fitted time-aware in backtest
  (`ah_alpha_pooled = 0.6252` → capped `ah_alpha_effective = 0.3`), resolved at serve time
  by `_resolve_ah_alpha()`; auto-degrades to α=0 with no AH odds.
- **9.5 API + frontend** — optional `markets` on `PredictResponse`/`FixtureRow`; "Asian
  Markets" panel (fair line/total headlines + AH/totals ladders, main-line highlight).

## Promotion gate (run 2026-06-10) — PROMOTE

The 9.4 blend was implemented without the required paired-bootstrap gate; the gate was
added and run this session. `backtest.py` now writes `backtest_permatch_ah.csv` (per-match
1X2 log loss + AH cover Brier for the ensemble matrix, unblended vs matrix-blended at the
fold's time-aware α and at fixed 0.3); `model_select.py` gained `run_ah_gate()`.

Result over n=241 matches with market AH odds (paired bootstrap, 10k resamples):

| Comparison (blend − model) | Δ mean | 95% CI | P(blend wins) |
|---|---|---|---|
| 1X2 log loss, time-aware α | −0.0107 | [−0.0221, −0.0018] | 99.3% |
| AH cover Brier, time-aware α | −0.0028 | [−0.0052, −0.0005] | 99.1% |
| 1X2 log loss, fixed α=0.3 | −0.0136 | [−0.0269, −0.0018] | 98.9% |
| AH cover Brier, fixed α=0.3 | −0.0037 | [−0.0075, +0.0002] | 96.9% |

Both gate legs pass on the time-aware primary (credibly better on AH Brier, and in fact
significantly *better* — not just not-worse — on 1X2 log loss). **α stays at 0.3.**
The 1X2 model-select re-run also re-confirmed `ensemble_mkt` as default
(Δ vs ens_cal = −0.0089, CI hi −0.0016 < 0).

## Artifacts (regenerated 2026-06-10, post-gate)

- `data/processed/backtest_report.json` + `backtest_permatch.csv` + `backtest_permatch_ah.csv`
- `data/processed/wc2026_predictions_2026-06-11.csv` (ensemble + ensemble_mkt, AH blend in)
- `data/processed/wc2026_tournament_sim_2026-06-11.{csv,json}` (ensemble_mkt, 20k, seed 42)

## Verification commands

```bash
uv run pytest -q                                      # 270 passed, 1 skipped
uv run python -m wcpredictor.evaluation.model_select  # 1X2 selection + AH gate (PROMOTE)
uv run python -m wcpredictor.evaluation.backtest      # report + per-match CSVs
uv run python -c "from wcpredictor.predict import predict_match; import json; \
  print(json.dumps(predict_match('Brazil','Serbia','2026-06-11')['markets'], indent=2))"
uv run uvicorn wcpredictor.api.app:app --port 8001    # Asian Markets panel; /fixtures markets
```

## Notes / residual caveats

- **90-min settlement convention:** the-odds-api docs do not explicitly state the
  settlement period for soccer `spreads`/`totals`. Asian handicap and totals are by
  near-universal bookmaker convention regular-time (90-min) markets, matching the
  project's label; treat knockout-match AH quotes with that convention in mind. Checked
  2026-06-10; not doc-verified.
- The live odds-api snapshot is from **2026-06-09 16:42** — re-pull near kickoff
  (`uv run python -m wcpredictor.data.download_odds_api`) for fresher closing lines, then
  `POST /refresh-odds` (or restart the API) so `_STATE_CACHE` rebuilds.
- `ah_roi_model` is positive in all four backtest folds (0.41/0.14/0.48/0.11 flat-stake vs
  closing) — interesting but small-n; treat as anecdote, not edge.

## What not to repeat / failed approaches

- Do **not** look for AH columns in `WorldCup_fdco.xlsx` — verified absent (H/D/A only).
- Do **not** use line-based inversion (`_market_score_matrix`) for betexplorer data — it
  always shows −0.5 as the active tab; use the prob-based inversion
  (`_market_score_matrix_from_probs`).
- The blend promotion is now bootstrap-validated (above) — do not re-litigate without new
  data; the same evidence bar applies to any future α changes.
- Do **not** retrain models to get AH/totals — they fall out of the score matrix.

---

## Standing action — fdco odds-day runbook (1X2)

When the `WorldCup2026` sheet appears in `WorldCup_fdco.xlsx` (still absent 2026-06-10;
kickoff 06-11 — check daily/hourly):
1. Pull odds: click **↻ Refresh odds** in the UI, or
   `uv run python -m wcpredictor.data.download_odds`.
2. Confirm 2026 rows:
   `uv run python -c "from wcpredictor.features.odds import load_wc_odds; df=load_wc_odds(); print((df['year']==2026).sum())"` → > 0.
3. Re-pin `FDCO_ODDS_SHA256` in `config.py`; commit.
4. Regenerate predictions + 20k sim for `as_of=2026-06-11`.
5. Confirm `ensemble` vs `ensemble_mkt` diverge (small, capped divergence expected).
6. Restart API / `POST /refresh-odds` so `_STATE_CACHE` rebuilds.
