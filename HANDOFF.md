# HANDOFF — Fix `ensemble_mkt` market-blend activation bug, cover it, commit

> Planning session 2026-06-03. **Not yet executed.** Start a fresh session to implement.

## Goal

The model-selection work that promoted the default to `ensemble_mkt` is complete
in the working tree but **uncommitted**, and it contains a **latent `NameError`**
that will fire the moment the WC2026 odds sheet lands (~2026-06-09) — exactly when
the new default's market blend is supposed to activate. Fix the bug (by removing
the duplication that caused it), add the missing regression test, then commit the
whole effort.

## Context & constraints

- Today 2026-06-03; WC2026 kickoff 2026-06-11; fdco WC2026 odds publish ~2026-06-09.
- Default is `ensemble_mkt` (paired-bootstrap decision; do NOT reopen — see
  `memory/project_model_decision.md`).
- Suite is green today: **200 passed, 1 skipped** (full run ~23 min).
- All existing leakage-safety / time-aware-split guarantees must stay intact.

## Files inspected (read-only)

- `src/wcpredictor/predict.py` — `predict_match` (buggy path) + `_build_frozen_state`
  / `_predict_one_frozen` (correct path).
- `src/wcpredictor/api/app.py`, `simulate.py` — default wiring (correct, use frozen path).
- `src/wcpredictor/models/calibration.py` — `_MIN_CAL_SAMPLES=30` guard.
- `src/wcpredictor/evaluation/backtest.py`, `evaluation/model_select.py` (new, untracked).
- `README.md`, `data/processed/backtest_report.json` (`odds_alpha_pooled`=0.6388).

## Current findings

**The bug.** `predict_match`'s `ensemble_mkt` branch references `ODDS_ALPHA_PRIOR`
at `predict.py:240` and `:246`, but the branch's import block at `predict.py:98`
imports only `DATA_RAW, DC_CAL_VALIDATION_YEARS, ENSEMBLE_POOL`, and the
module-level import (`predict.py:30`) omits it too → guaranteed `NameError` when
reached. It is **masked today**: no 2026 odds exist, so `_odds_entry` is `None`
and the branch is skipped (also why no test caught it).

**The correct path.** `_predict_one_frozen` (`predict.py:531`), used by the API and
simulator, is fine — it uses `state["odds_alpha"]` and `state["odds_lookup"]`
populated in `_build_frozen_state` (`predict.py:365-371` and `:434-444`).

**Root cause is duplication.** `predict_match` re-implements two things the frozen
path already does correctly, and the copies drifted:
- 2026 odds-lookup build: `predict_match:232-238` vs `_build_frozen_state:365-371`.
- `backtest_report.json` α-read: `predict_match:240-248` vs `_build_frozen_state:434-444`.

**No test covers blend activation** — that gap is what let the `NameError` ship.

**Uncommitted set** (all green, from the prior model-selection session):
`predict.py`, `api/app.py`, `simulate.py`, `models/calibration.py`,
`evaluation/backtest.py`, `evaluation/model_select.py` (new), `tests/test_calibration.py`,
`README.md`, `HANDOFF.md`.

## Proposed implementation plan

1. **Fix by de-duplicating** (`predict.py`). Add two module-level helpers and call
   them from both paths:
   - `_resolve_odds_alpha() -> float` — read `DATA_PROCESSED/"backtest_report.json"`,
     return `odds_alpha_pooled` (fallback `ODDS_ALPHA_PRIOR`). Replaces
     `predict_match:240-248` and `_build_frozen_state:434-444`.
   - `_build_odds_lookup(odds_df) -> dict[(str,str),(float,float,float)]` — the
     `year==2026`, both-orientation lookup at `predict_match:232-238` and
     `_build_frozen_state:365-371`.
   This removes the `ODDS_ALPHA_PRIOR` reference from `predict_match` entirely, so
   the import gap is moot. (Minimal alt if a smaller diff is wanted: just add
   `ODDS_ALPHA_PRIOR` to the `predict.py:98` import — but prefer the refactor since
   drift caused the bug.)

2. **Regression test** (`tests/`, mirror existing ensemble test setup). Monkeypatch
   `load_wc_odds` to return a frame with a synthetic `year==2026` row for the test
   pair (with `odds_alpha_pooled` > 0 in effect). Call `predict_match` with
   `model="ensemble_mkt"` and `model="ensemble"`. Assert: (a) no exception (guards
   the `NameError`); (b) probs sum to ~1; (c) `ensemble_mkt` W/D/L is shifted toward
   the injected market odds vs `ensemble` (blend actually fires). One heavy fit;
   keep it a single focused test.

3. **Commit** the full uncommitted set once 1–2 are green.

4. **Follow-up, time-gated — NOT executable before ~2026-06-09.** When odds publish:
   refresh odds, regenerate the full-resolution snapshot
   (`uv run python -m wcpredictor.simulate --as-of <date> --model ensemble_mkt
   --n-sims 20000`) replacing the poisson n_sims=100 placeholder
   (`data/processed/wc2026_tournament_sim_2026-06-10.json`, 971 bytes), and confirm
   blend activation on real data. Window to kickoff (06-11) is tight.

## Exact next steps

1. Add `_resolve_odds_alpha` + `_build_odds_lookup`; rewire both paths in `predict.py`.
2. Add the activation regression test in `tests/`.
3. Run targeted tests, then full suite.
4. Commit all changes.

## Verification / test commands

```bash
uv run pytest tests/ -k "ensemble_mkt or blend or market" -v      # new test
uv run pytest tests/test_calibration.py tests/test_ensemble.py -v # guard + ensemble
uv run python -c "from wcpredictor.predict import predict_match; print(predict_match('Brazil','France','2026-06-15',neutral=True))"  # no odds → must NOT error
uv run pytest -q                                                  # full suite (~23 min)
```

Success = `predict_match(model="ensemble_mkt")` blend path runs without `NameError`
and shifts toward market odds when a 2026 row is present; full suite green; work committed.

## Risks & open questions

- Refactor touches the (currently-correct) frozen path — re-run `test_ensemble.py`
  and simulator-touching tests to confirm no regression.
- Full suite is slow (~23 min); iterate with the `-k` runs.
- Step 4 depends on an external publish date; do not regenerate the snapshot now
  (would just produce another ens_cal-degraded placeholder).

## What not to repeat (documented dead ends)

- Do not re-select the default on fold-mean log-loss alone; do not reopen
  `ensemble_mkt` vs `ensemble_cal` without new backtest data
  (`memory/project_model_decision.md`).
- GBM weight-gate fixes — see `memory/project_gbm_gate.md`.
