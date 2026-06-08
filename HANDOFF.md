# HANDOFF — Odds-day refresh runbook

> Updated **2026-06-08** after Part A shipped. Part B is the only live next action.

## Status

**Part A — refresh-odds feature — DONE**
Committed on `ops/pre-tournament-readiness` (commit "Phase 8.1: /refresh-odds endpoint + frontend button"), merged to `main`.
Smoke test confirmed: `POST /refresh-odds` returns `200`, `n_odds_2026=0`, `sha_changed=false` (correct pre-odds state).

**Part B — odds-day refresh — WAITING on upstream**
`WorldCup2026` sheet not yet published as of 2026-06-08. Check ~06-09 onward.

---

## Goal

Run the Part B runbook the moment the `WorldCup2026` sheet appears in `WorldCup_fdco.xlsx`, so predictions are market-calibrated before kickoff (~2026-06-11).

## Context and constraints

- WC2026 predictor: all 9 build phases + Phase 8.1 complete; `ensemble_mkt` is the shipped default.
- `DEFAULT_AS_OF="2026-06-11"` in `api/app.py`.
- Until the 2026 sheet lands, `ensemble_mkt` auto-degrades to `ensemble` for 2026 fixtures — correct behavior.
- CLAUDE.md rules: no silent scraping; SHA re-pin is manual (the endpoint never auto-rewrites `config.py`); no leakage; don't hard-code 2026 teams.

## Part B — Odds-day refresh runbook (run when `WorldCup2026` sheet exists)

1. Pull odds: click **↻ Refresh odds** in the UI, or
   `uv run python -m wcpredictor.data.download_odds` (verify mode raises on SHA change — confirms the file changed).
2. Confirm 2026 rows:
   ```bash
   uv run python -c "from wcpredictor.features.odds import load_wc_odds; df=load_wc_odds(); print((df['year']==2026).sum())"
   ```
   → expect > 0.
3. Re-pin `FDCO_ODDS_SHA256` in `src/wcpredictor/config.py:47` to the new SHA; commit.
4. Regenerate snapshots for `as_of=2026-06-11`:
   ```bash
   uv run python -c "from wcpredictor.predict import predict_fixtures; predict_fixtures('2026-06-11', models=['ensemble','ensemble_mkt'])"
   uv run python -m wcpredictor.simulate --as-of 2026-06-11 --model ensemble_mkt --n-sims 20000 --seed 42
   ```
5. **Acceptance check**: `ensemble` vs `ensemble_mkt` predictions now **diverge** for some fixtures (identical pre-odds). Spot-check the predictions CSV or compare `/predict?model=ensemble` vs `?model=ensemble_mkt`.
6. Restart the API (or `POST /refresh-odds`) so `_STATE_CACHE` rebuilds; confirm UI reflects market odds.

If kickoff `as_of` changes, update `DEFAULT_AS_OF` in `api/app.py` and regenerate for the new date.

## Key files

- `src/wcpredictor/config.py:47` — `FDCO_ODDS_SHA256` pin to update on odds day
- `src/wcpredictor/predict.py` — `predict_fixtures()` for snapshot regen
- `src/wcpredictor/simulate.py` — CLI for tournament sim
- `src/wcpredictor/api/app.py` — `DEFAULT_AS_OF`, `POST /refresh-odds`

## Verification commands

```bash
uv run pytest tests/test_api.py -q                       # 12 passed
uv run uvicorn wcpredictor.api.app:app --port 8000
curl -X POST http://127.0.0.1:8000/refresh-odds          # 200; n_odds_2026==0 until sheet lands
uv run pytest -q                                         # full: 208 passed, 1 skipped
```

## Risks

- Upstream timing: the `WorldCup2026` sheet may appear only hours before kickoff or with sparse pricing.
- The 20k sim (step 4) takes minutes — intentionally CLI-only; the UI button only refreshes the live `/predict` path, not Fixtures/Tournament snapshots.
- `n_odds_2026 == 0` and identical predictions today are **correct** pre-odds state, not a bug.

## What not to repeat

- Do **not** reopen `ensemble_mkt` vs `ensemble_cal` — decided (`memory/project_model_decision.md`).
- GBM weight-gate dead ends: `memory/project_gbm_gate.md` — don't re-explore.
- Do **not** re-implement `/refresh-odds`; it exists, is tested, and is shipped.
