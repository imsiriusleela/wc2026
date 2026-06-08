# HANDOFF — Model-first reframe: cap the market overlay at α≤0.3

> Planning session **2026-06-08**. **SHIPPED 2026-06-09** — alpha cap live, all
> artifacts regenerated, 212 tests passing (4 new). The odds-day runbook at the
> bottom is the next standing action once the WC2026 fdco sheet is published.

## Goal

Make the predictor **model-first**: keep `ensemble_mkt` as the default model but
**cap its market-blend weight at α ≤ 0.3**, so the bespoke model leads and the market
is a low-weight calibration overlay rather than the driver. Stay on the
football-data.co.uk consensus odds source (no live API, no scraping).

## Context and constraints

- The shipped default `ensemble_mkt` blends bookmaker odds at a fitted
  **α = 0.6388** (`odds_alpha_pooled` in `data/processed/backtest_report.json`) — the
  market drives ~64% of every prediction.
- Backtest shows that 64% weight earns almost nothing. Log loss, model-only
  (`ensemble_cal`) vs market-blended (`ensemble_mkt`):

  | WC | model-only | market-blend | Δ |
  |----|-----------|--------------|---|
  | 2010 | 0.9644 | 0.9644 | 0.0000 |
  | 2014 | 0.9233 | 0.9231 | 0.0002 |
  | 2018 | 0.9563 | 0.9541 | 0.0022 |
  | 2022 | 1.0682 | 1.0349 | 0.0333 |

  The entire benefit is WC2022; in 3/4 WCs a 64% market lean changes the answer by ≈0.
  The standalone model already roughly matches a two-thirds-market blend.
- **Decisions made this session:**
  - Objective = **model-first** (over calibration-first / value-finding).
  - Market role = **capped overlay, α ≤ 0.3** (over pure model-only / disagreement view).
  - Odds source = **stay on fdco consensus** (rejected: live The-Odds-API, Betfair,
    single-book scrape). With the market demoted to ≤0.3, fdco's consensus sheet
    (arrives days pre-kickoff) is sufficient; the "2026 sheet not yet published"
    issue is no longer a blocker.
- Constraints: do **not** reopen the `ensemble_mkt`-vs-`ensemble_cal` calibration
  decision (`memory/project_model_decision.md`) — `ensemble_mkt` stays default; we
  only retune its weight. Keep models reproducible; no silent scraping. GBM
  weight-gate dead ends (`memory/project_gbm_gate.md`) stay closed.

## Files inspected

- `src/wcpredictor/predict.py` — `_resolve_odds_alpha()` (L40–54) is the single funnel
  for both the live path (`predict_match`, L262) and batch path (`_build_frozen_state`,
  L442). Capping here covers everything.
- `src/wcpredictor/evaluation/backtest.py` — `_fit_odds_alpha()` (L110), per-year
  `model_ensemble_market` eval (~L461–477), pooled fit (~L546).
- `src/wcpredictor/config.py` — `ODDS_ALPHA_PRIOR` (L48), the place for a new cap const.
- `tests/test_ensemble_mkt_blend.py` — patches `_resolve_odds_alpha` directly (L53),
  so a serve-time cap will not break it.
- `data/processed/backtest_report.json` — holds `odds_alpha_pooled = 0.6388`.

## Current findings

The cap is a one-line behavioral change at a single consumption point, plus honest
re-reporting and a snapshot regen. Low risk, high clarity-of-intent.

## Proposed implementation plan

1. **Config** (`config.py`, near L48): add
   `ODDS_ALPHA_CAP: float = 0.3` with a comment (unconstrained optimum ~0.64 is
   WC2022-driven; cap keeps the market an overlay).
2. **Serve-time cap** (`predict.py` `_resolve_odds_alpha`):
   `return min(float(report.get("odds_alpha_pooled", ODDS_ALPHA_PRIOR)), ODDS_ALPHA_CAP)`.
3. **Honest reporting** (`backtest.py`): blend `model_ensemble_market` with
   `alpha_eff = min(alpha_odds, ODDS_ALPHA_CAP)`; store `alpha_odds` (raw) and
   `alpha_effective` per year; add top-level `odds_alpha_effective`. Keep raw
   `odds_alpha_pooled` for transparency.
4. **Frontend** (`api/static/index.html`): one-line note that the market is a capped
   calibration overlay (α≤0.3) and the headline is model-led. No structural change.
5. **Tests** (`tests/test_odds_alpha_cap.py` or extend `test_api.py`): pooled 0.64 →
   resolved 0.3; pooled 0.1 → 0.1; missing report → 0.0. Keep
   `test_ensemble_mkt_blend.py` green.
6. **Regenerate artifacts** (after code lands):
   ```bash
   uv run python -m wcpredictor.evaluation.backtest
   uv run python -c "from wcpredictor.predict import predict_fixtures; predict_fixtures('2026-06-11', models=['ensemble','ensemble_mkt'])"
   uv run python -m wcpredictor.simulate --as-of 2026-06-11 --model ensemble_mkt --n-sims 20000 --seed 42
   ```
   then refresh `wc2026_scorecard.json` via its existing generation step.

## Exact next steps
1. Fresh execution session (out of plan mode).
2. Edits in order: config → predict → backtest → frontend → tests.
3. Run the suite, then regenerate artifacts.
4. Update this HANDOFF to reflect the cap shipped.

## Verification commands
```bash
uv run pytest -q                                         # full: was 208 passed, 1 skipped
uv run python -c "from wcpredictor.predict import _resolve_odds_alpha; print(_resolve_odds_alpha())"   # -> 0.3
```
- After regenerating the backtest, confirm the `ens_mkt` vs `ens_cal` gap shrinks
  (WC2022's 0.033 LL edge compresses as α drops 0.64→0.30) — expected/acceptable.
- Confirm a default `/predict` (`ensemble_mkt`) lands closer to model-only than before.

## Risks and open questions
- **0.3 is a judgment call**, not a fitted optimum (the LL curve is convex with min
  ~0.64, so a constrained re-fit to [0,0.3] would just hit the 0.3 boundary —
  `min(fitted, cap)` is equivalent here).
- Capping reduces the one tournament (2022) where the market helped; accepted given
  n=64/WC variance and the model-skill objective.
- Reproducibility preserved: report keeps both raw (0.64) and effective (0.3) alpha.

## What not to repeat / already settled
- Do **not** re-implement `/refresh-odds` — shipped (Phase 8.1) and tested.
- Do **not** reopen `ensemble_mkt` vs `ensemble_cal` (`memory/project_model_decision.md`).
- Do **not** pursue live-odds APIs / scraping — evaluated and rejected this session;
  staying on fdco consensus because the market is now a ≤0.3 overlay.
- GBM weight-gate dead ends: `memory/project_gbm_gate.md`.

---

## Related / still valid — fdco odds-day runbook (now lower-stakes)

When the `WorldCup2026` sheet appears in `WorldCup_fdco.xlsx` (check ~06-09 onward):
1. Pull odds: click **↻ Refresh odds** in the UI, or
   `uv run python -m wcpredictor.data.download_odds`.
2. Confirm 2026 rows:
   `uv run python -c "from wcpredictor.features.odds import load_wc_odds; df=load_wc_odds(); print((df['year']==2026).sum())"` → > 0.
3. Re-pin `FDCO_ODDS_SHA256` in `config.py:47`; commit.
4. Regenerate predictions + 20k sim for `as_of=2026-06-11` (commands above).
5. Confirm `ensemble` vs `ensemble_mkt` diverge (now a smaller, capped divergence).
6. Restart API / `POST /refresh-odds` so `_STATE_CACHE` rebuilds.
