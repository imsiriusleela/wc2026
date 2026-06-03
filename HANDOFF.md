# HANDOFF — WC2026 pre-tournament operational readiness

> Planning session **2026-06-03**; executed same session on branch `ops/pre-tournament-readiness`.
> **Parts 1 and 3 complete.** Part 2 (odds-day refresh) executes ~2026-06-09.
> Part 4 (model improvements) is optional / separate session.
> Supersedes the previous handoff (the `ensemble_mkt` blend fix it described as "not yet executed" already shipped in commit `4c47135`).

## Goal

Get the predictor production-ready for kickoff. All 9 CLAUDE.md phases are done,
committed, and green (~200 tests). What remains: replace placeholder artifacts
with the real default model, execute the odds-day refresh when the bookmaker sheet
lands, stand up in-tournament monitoring, and (optionally) evaluate model
improvements before the data freeze.

## Context & constraints

- Today **2026-06-03**; WC2026 kickoff **2026-06-11**; fdco `WorldCup2026` odds
  sheet publishes **~2026-06-09** (tight window).
- Default model is `ensemble_mkt`. **Do not reopen** the `ensemble_mkt` vs
  `ens_cal` selection without new backtest data (`memory/project_model_decision.md`).
- All leakage-safety / time-aware-split guarantees must stay intact.
- No silent data fetches: re-pin SHA after any fdco refresh.
- Full test suite is slow (~23 min); iterate with `-k`.

## Files inspected (read-only)

- `src/wcpredictor/predict.py` — `predict_fixtures` (batch, ~L566); frozen path
  (`_build_frozen_state`, `_predict_one_frozen`); helpers `_resolve_odds_alpha`
  (L40), `_build_odds_lookup` (L57) — **blend fix already present**.
- `src/wcpredictor/simulate.py` — CLI; defaults `--model ensemble_mkt`,
  `--n-sims 20000`; writes `wc2026_tournament_sim_<as_of>.{csv,json}`.
- `src/wcpredictor/data/download_odds.py` — SHA-pinned fdco fetch (`force=`).
- `src/wcpredictor/data/download_wc2026.py` — parses `WorldCup2026` sheet → fixtures,
  prints fresh SHA-256.
- `src/wcpredictor/features/odds.py` — `load_wc_odds()`; `_SHEET_YEARS` maps
  `WorldCup2026`→2026.
- `src/wcpredictor/evaluation/live.py` — `run_refresh()` + `__main__`
  (`--as-of`, `--results`); writes `wc2026_scorecard.json`.
- `src/wcpredictor/api/app.py` — endpoints `/predict`, `/fixtures`, `/tournament`,
  `/teams`, `/health`; `_latest_artifact()` glob helper. **No `/scorecard`.**
- `src/wcpredictor/config.py` — `FDCO_ODDS_URL` (L45), `FDCO_ODDS_SHA256` (L47),
  `ODDS_ALPHA_PRIOR=0.0` (L48).

## Current findings

1. **Stale handoff fixed by this file.** Blend fix (`_resolve_odds_alpha` /
   `_build_odds_lookup`, `tests/test_ensemble_mkt_blend.py`) shipped in `4c47135`.
2. **Odds not published yet.** `WorldCup_fdco.xlsx` has only
   `WorldCup2026Qualifiers`; `load_wc_odds()` returns 0 rows for 2026 → default
   `ensemble_mkt` auto-degrades to `ens_cal` (by design, pre-06-09).
3. **Snapshot is a placeholder.** `wc2026_tournament_sim_2026-06-10.json` is
   `model=poisson, n_sims=500` — not the real default `ensemble_mkt @ 20000`.
   Predictions CSV dated Jun 2.
4. **Scorecard not surfaced by API.** `live.py` writes `wc2026_scorecard.json`,
   but there is no `/scorecard` endpoint or frontend panel.

## Proposed implementation plan

### Part 1 — Pre-odds readiness (do now, 2026-06-03)
1. Regenerate artifacts with the correct default (auto-degrades to ens_cal with no
   odds — the correct pre-odds baseline, and validates the 20k run + timing):
   ```bash
   uv run python -c "from wcpredictor.predict import predict_fixtures; \
     predict_fixtures('2026-06-11', models=['poisson','ensemble','ensemble_mkt'])"
   uv run python -m wcpredictor.simulate --as-of 2026-06-11 --model ensemble_mkt --n-sims 20000
   ```
   Confirm new `wc2026_tournament_sim_2026-06-11.json` shows
   `model=ensemble_mkt, n_sims=20000`, `p_champion_sum≈1.0`. Remove obsolete
   `*_2026-06-10.*` placeholders so `_latest_artifact()` serves the new ones.
2. Dry-run the live scorer (no completed matches yet → empty-window scorecard):
   ```bash
   uv run python -m wcpredictor.evaluation.live --as-of 2026-06-11
   ```
3. Commit Part 1 (branch off `main` first).

### Part 2 — Odds-day runbook (execute ~2026-06-09)
1. Refresh + re-pin fdco:
   ```bash
   uv run python -m wcpredictor.data.download_odds    # force=True if hash mismatch
   uv run python -m wcpredictor.data.download_wc2026   # parses sheet, prints fresh SHA
   ```
   Update `FDCO_ODDS_SHA256` in `config.py:47` to the printed value.
2. Verify odds landed (gate before regenerating):
   ```bash
   uv run python -c "from wcpredictor.features.odds import load_wc_odds; \
     d=load_wc_odds(); print('2026 rows:', int((d.year==2026).sum()))"   # must be > 0
   ```
3. Regenerate full-res artifacts (same two commands as Part 1 step 1) — blend now fires.
4. Confirm blend activation: `ensemble` vs `ensemble_mkt` probs must differ for an
   odds-present fixture; run `tests/test_ensemble_mkt_blend.py`; commit refreshed
   odds file + SHA bump + new snapshot.

### Part 3 — In-tournament monitoring (2026-06-11+)
1. Run the live loop as matches complete:
   `uv run python -m wcpredictor.evaluation.live --as-of <today>`
   (leakage-safe: only matches strictly before `--as-of` with a saved pre-match
   prediction; recalibrates temperature on the WC2026 window).
2. Add `GET /scorecard` in `api/app.py` (mirror `/tournament`; read
   `wc2026_scorecard.json`, 503 if absent) + schema in `api/schemas.py` + a panel
   in `static/index.html` (mirror the Tournament Simulation table).
3. Add an endpoint test (mirror `tests/test_api.py`).

### Part 4 — Model improvements (optional, separate session)
Evaluate only via the existing backtest harness (`evaluation/backtest.py` +
`model_select.py`) with documented paired-bootstrap comparisons — never fold-mean
log-loss alone. Candidates: squad value/availability feature (leakage-safe,
walk-forward gated); re-fit `odds_alpha` once real 2026 odds exist if the in-sample
pooled α (0.6388) looks miscalibrated. Do not bundle with Parts 1–3.

## Exact next steps
1. Branch off `main`.
2. Part 1: regenerate snapshot + predictions with `ensemble_mkt`/20k; remove
   `*_2026-06-10.*` placeholders; live dry-run; commit.
3. Hold Part 2 until `WorldCup2026` sheet is live (~06-09).
4. Part 3 endpoint work can proceed any time.

## Verification / test commands
```bash
# Part 1
uv run python -c "import json; d=json.load(open('data/processed/wc2026_tournament_sim_2026-06-11.json')); print(d['model'], d['n_sims'], d['p_champion_sum'])"
uv run python -c "from wcpredictor.predict import predict_match; print(predict_match('Brazil','France','2026-06-15',neutral=True))"  # no odds → must NOT error
uv run pytest -q -k "ensemble_mkt or blend or api"
# Part 2 (post-06-09)
uv run python -c "from wcpredictor.features.odds import load_wc_odds; d=load_wc_odds(); print(int((d.year==2026).sum()))"  # > 0
# Full suite (~23 min)
uv run pytest -q
```

## Risks & open questions
- 06-09 → 06-11 window is tight; Part 1 de-risks the 20k regen timing.
- SHA re-pin is mandatory after the sheet appears, or `download_odds` warns/refetches.
- Do NOT regenerate with odds before `load_wc_odds()` confirms 2026 rows > 0 —
  otherwise you reproduce another ens_cal-degraded placeholder.

## What not to repeat
- Do not reopen `ensemble_mkt` vs `ens_cal` without new backtest data
  (`memory/project_model_decision.md`).
- Do not select a default on fold-mean log-loss alone.
- GBM weight-gate dead ends — see `memory/project_gbm_gate.md`.
