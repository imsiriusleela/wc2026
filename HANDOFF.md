# HANDOFF — Phase 12: Live Tournament Operations (COMPLETE)

> Planned 2026-06-10 (planning session). Implemented 2026-06-10. Branch: feat/phase12-live-ops.
> WC2026 kicks off 2026-06-11. All phases 1–11 are merged to main.

## Goal

Make the predictor operable DURING the live tournament:
1. Ingest played 2026 results (martj42 re-pull + manual edits) into a canonical store.
2. Roll Elo/form ratings forward with played results while keeping all model fits
   pinned at the tournament-start cutoff (mirrors the validated backtest).
3. Condition the Monte Carlo simulator on played group results and decided KO winners.
4. Expose `/refresh-results` and `/resimulate` endpoints + UI buttons.

## Relevant context and constraints

- **Rating policy (user-confirmed):** backtest.py:9 — "Elo runs continuously across all
  history; only model fits are per-fold." So rolling Elo IS the validated configuration;
  what must stay pinned at `TOURNAMENT_START = 2026-06-11` are the model fits:
  Poisson/DC/logistic/GBM params, ensemble weights, calibration T, odds/AH alphas.
  This is NOT a rolling refit — split the single cutoff into `fit_cutoff` (fixed) and
  `as_of`/rating cutoff (advances).
- Default model stays `ensemble_mkt`, α caps stay at 0.3 (gate-validated; do not re-litigate).
- No Kelly/stake sizing. No `markets` schema changes. No exact-offer-count test assertions.
- No silent scraping — the martj42 master re-pull is the already-approved source
  (live.py uses it today via RESULTS_URL_FALLBACK).
- All features must be available strictly before kickoff of the predicted match
  (leakage tests required by CLAUDE.md).
- martj42 scores include extra time but NOT penalties (load_matches.py:4-6), so a
  penalty-decided KO match looks like a draw — needs an explicit `winner` column.

## Files inspected

- `src/wcpredictor/predict.py` — `_build_frozen_state` (lines ~611-780), `predict_match`
  (~368-608), `predict_fixtures` (~927+), `_predict_one_frozen` (~798)
- `src/wcpredictor/simulate.py` — `infer_groups` (~90-153), group loop (~314-341),
  thirds (~343-348), knockout (~350-379), frozen-state call (~275)
- `src/wcpredictor/evaluation/backtest.py` — fold semantics (header lines 1-17)
- `src/wcpredictor/evaluation/live.py` — `_load_latest_results` (~55-83), `_load_predictions`
- `src/wcpredictor/api/app.py` — `DEFAULT_AS_OF` (:39), `_STATE_CACHE` (:42),
  `/refresh-odds` (~289-393), `/scorecard` (~280-286)
- `src/wcpredictor/data/download_odds_api.py`, `features/elo.py`, `features/form.py`,
  `features/odds.py`, `config.py`
- `data/raw/wc2026_fixtures.csv` — schema verified: `date,team_a,team_b,neutral,goals_a,goals_b`
  (72 group rows, goals empty, no group/stage column, no KO rows yet)

## Current findings

1. **No results ingestion**: nothing feeds played 2026 results into ratings or the simulator.
   `live.py` is manual scorecard only.
2. **Bug — live.py un-pins results.csv**: `_load_latest_results` (live.py:63-74) overwrites
   the SHA-pinned `data/raw/results.csv` with the martj42 master (`local.write_bytes(raw)`).
3. **Latent bug — infer_groups breaks on KO fixtures**: connected-components over ALL fixture
   rows (simulate.py:90-153); the first R32 row merges two groups → "Expected 12 groups"
   ValueError. Must slice fixtures to `date < KO_START` first.
4. **Single-cutoff problem**: `_build_frozen_state` uses one `cutoff` for both Elo/form AND
   all model fits (`train = matches[date < cutoff]`, predict.py:621-629). Advancing as_of
   today would refit models on 2026 data — a deviation from the backtest.
5. **Free lunch**: Elo/form rows are emitted pre-match in chronological order, so ONE
   `compute_elo` pass over the augmented history serves both paths — subsetting `elo_df`
   to `date < fit_cutoff` is identical to recomputing on pre-cutoff data only.
   (`compute_elo` cannot resume from saved state; full ~49k-row walk per cache miss is
   already the status quo and fast enough.)
6. `DEFAULT_AS_OF` is hardcoded `"2026-06-11"` (app.py:39) — used by `/predict` default and
   lifespan pre-warm; must become dynamic or ratings never roll forward in the API.
7. Already complete (verified, don't redo): 1X2 h2h parsing (`parse_h2h_1x2`), /refresh-odds
   with snapshot archiving to `data/raw/odds_api_snapshots/`, offer EV ranking, frozen-state
   serving. No TODO/FIXME/stubs anywhere in src/.

## Proposed implementation plan

### Step 0 — config.py
Add `TOURNAMENT_START = "2026-06-11"` and `KO_START = "2026-06-28"` (first R32 date).

### Step 1 — NEW `src/wcpredictor/data/results_2026.py`
Canonical store `data/raw/wc2026_results.csv`:
`date, team_a, team_b, goals_a, goals_b, stage, winner, source`
(`stage` derived from date vs KO_START; `winner` only for penalty-decided KO matches;
`source` ∈ {martj42, manual}).

- `fetch_master_results()` — extracted from `live._load_latest_results` but writes to
  `data/raw/results_master.csv`, NEVER touching pinned `results.csv`; falls back to local
  copy on network failure.
- `update_wc2026_results(source_csv=None)` — pull master (or user CSV), filter to FIFA World
  Cup rows `date >= TOURNAMENT_START` with non-null scores, canonicalize team names, merge
  into store; dedupe on date + team pair; **manual rows win** over martj42 rows.
- `load_wc2026_results()` — read store; empty typed DataFrame when absent.
- `augment_matches(matches)` — append store rows not already in `load_matches()` output
  (same `match_id` md5 scheme, `is_world_cup=True`, `neutral=True`), re-sort by date.
- `mark_fixtures_played()` — fill `goals_a`/`goals_b` in `wc2026_fixtures.csv` from store.

Refactor `live._load_latest_results` to delegate to `fetch_master_results()` + store rows
(fixes finding #2, keeps `/scorecard` consistent).

### Step 2 — split cutoffs in `predict.py:_build_frozen_state`
New param `fit_cutoff: str | None = None`; resolve
`fit_cutoff = min(Timestamp(fit_cutoff or TOURNAMENT_START), rating_cutoff)` —
the `min()` keeps every existing pre-tournament call and all current tests byte-identical.

- `matches = augment_matches(load_matches())`.
- Ratings path (rolls): `rating_train = matches[date < rating_cutoff]` → `compute_elo`,
  `compute_form`, `latest_elo(before_date=rating_cutoff)`. `state["cutoff"]` stays the
  rating cutoff (used by `_predict_one_frozen:798` for form rows); add `state["fit_cutoff"]`.
- Fit path (pinned): `fit_elo = elo_df[date < fit_cutoff]`, `fit_train = matches[date < fit_cutoff]`.
  Switch every fit: `poisson_fit` (~:643), `dc_fit(ref_date=fit_cutoff)` (~:649),
  calibration window + validation slices (~:677-703), past-WC ensemble-weight walk-forward
  (~:705-709), logistic/GBM fits (~:726-728). Alphas come from the static backtest report —
  unchanged.
- `predict_match` (~368-608): apply the same `min(as_of, TOURNAMENT_START)` fit-cutoff rule
  (full delegation to frozen-state is optional cleanup, not required).
- `predict_fixtures`: pass `fit_cutoff` through; skip fixtures with `goals_a` already filled.

### Step 3 — conditioned simulation in `simulate.py`
- Guard: `fixtures_gs = fixtures[date < KO_START]` before `infer_groups` (fixes finding #3
  unconditionally).
- `_load_conditioning(as_of)` from `load_wc2026_results()` rows `date < as_of` (leakage guard):
  `fixed_scores: dict[frozenset(pair) -> (team_a, ga, gb)]` for group rows;
  `ko_winners: dict[frozenset -> winner]` for KO rows (winner from score, else `winner`
  column; drawn KO row without winner → warn + fall back to sampling).
- `simulate_tournament(..., condition_on_results=True, played_results=None)` (injectable
  for tests). Group loop (~:314-329): if pair in `fixed_scores`, use stored score instead of
  `sample_scoreline` — downstream standings/thirds untouched, eliminated teams naturally → ~0.
  Knockout (~:354-379): `w = ko_winners.get(pair) or _ko_match(...)`.
- Rolled-forward ratings come free via Step 2's `_build_frozen_state(as_of, ...)` call (~:275).
- Summary JSON: add `n_group_fixed`, `n_ko_fixed`, `fit_cutoff`. CLI: `--as-of <today>`
  now means "condition on everything played before today"; add `--no-condition`.
- Known limitation (accept + document): pts/gd/gf + random tiebreak, not FIFA head-to-head;
  rare standings mismatch possible; KO-winner conditioning corrects it once pairings are real.

### Step 4 — API + UI
- `app.py`: replace `DEFAULT_AS_OF` constant with `_default_as_of() = max(TOURNAMENT_START, today)`
  for `/predict` default and lifespan pre-warm (`_STATE_CACHE` already keys on as_of).
- NEW `POST /refresh-results` (mirror /refresh-odds lock pattern):
  `update_wc2026_results()` (non-fatal on network error) → `mark_fixtures_played()` →
  `_STATE_CACHE.clear()` → rewrite scorecard via live.py → response
  `{n_results_total, n_new, n_group, n_knockout, note}`.
- NEW `POST /resimulate?n_sims=5000&model=ensemble_mkt` — synchronous under the lock
  (~1-2 min OK for personal tool; CLI for 20k runs); writes dated artifacts `/tournament`
  already picks up via `_latest_artifact`.
- `/tournament`: additive optional meta fields `n_group_fixed`, `n_ko_fixed`.
- `index.html`: "↻ Refresh results" button (same wiring as refresh-odds), "Re-simulate"
  button on Tournament tab, "conditioned on N played matches" caption.

### Step 5 — tests (patterns: synthetic DataFrames, tmp_path, monkeypatched loaders as in
tests/test_wc2026.py and tests/test_api.py)
1. `tests/test_results_ingestion.py` — filtering, canonical names, dedupe, manual precedence,
   stage classification, winner handling, `mark_fixtures_played`, `augment_matches` no-dup.
2. `tests/test_split_cutoff.py` (CLAUDE.md-required leakage tests) — with synthetic
   post-TOURNAMENT_START rows: poisson/dc params, ensemble weights, calibration T
   **identical** with vs without 2026 rows (fits pinned); `state["ratings"]` **differs**
   and equals explicit pre-match recompute (Elo rolls); all fit inputs `date < fit_cutoff`;
   `as_of <= TOURNAMENT_START` ⇒ behavior unchanged.
3. `tests/test_simulate_conditioned.py` — three fixed losses → `p_win_group == 0`; fixed
   score constant across sims; forced KO winner always advances; rows `>= as_of` ignored;
   seed reproducibility.
4. Extend `tests/test_api.py` — `/refresh-results`, `/resimulate` with monkeypatched
   network/sim (no real downloads, no exact-count assertions).

## Exact next steps (execution session)

1. Branch: `git checkout -b feat/phase12-live-ops`
2. Step 0 (config) → Step 1 + ingestion tests → Step 2 + leakage tests →
   Step 3 + conditioning tests → Step 4 + API tests → full suite.
3. Housekeeping: delete or commit stray `predict-offers-result.png`; update memory
   `project_state.md` (Phases 10-11 are MERGED to main — memory says pending; record the
   Phase 12 rolling-Elo/pinned-fits decision citing backtest.py:9).
4. Commit per-step or as one Phase 12 commit; PR to main.

## Verification / test commands

```bash
uv run pytest tests/test_results_ingestion.py tests/test_split_cutoff.py tests/test_simulate_conditioned.py -q
uv run pytest -q          # full suite; baseline 302 passed / 1 skipped + new tests
# Smoke (after first results exist):
uv run python -m wcpredictor.data.results_2026
uv run python -m wcpredictor.simulate --as-of <today> --model ensemble_mkt --n-sims 20000
# API: restart uvicorn (port 8001); POST /refresh-results; POST /resimulate;
# GET /tournament shows n_group_fixed/n_ko_fixed; UI buttons render and work.
```

## Risks and open questions

- martj42 master lags ~1 day behind real results → manual-edit path in `wc2026_results.csv`
  is the mitigation; manual-row precedence must be tested.
- Penalty-decided KO matches need `winner` filled manually (or later: martj42 shootouts.csv).
- KO fixtures absent from `wc2026_fixtures.csv` until fdco publishes them — KO conditioning
  works off the results store, so it doesn't depend on fixtures.
- Scorecard dedupe keeps the EARLIEST prediction per match (`live._load_predictions`) —
  correct pre-kickoff semantics; re-running predict_fixtures won't overwrite scoring inputs.
- fdco 2026 sheet still unpublished as of 2026-06-10; fdco auto-precedence logic unaffected.

## What not to repeat / failed approaches

- Do NOT reuse `live._load_latest_results` as-is for ingestion — it overwrites the pinned
  `results.csv` (the bug being fixed). New downloads go to `results_master.csv`.
- Do NOT advance the fit cutoff with as_of (no rolling refits) — only ratings roll.
- Do NOT re-litigate ensemble_mkt, α=0.3 caps, or the AH promotion gate.
- Do NOT add Kelly/stake sizing, change the `markets` schema, or assert exact offer counts.
- Do NOT pass full fixtures (incl. future KO rows) to `infer_groups` — slice to
  `date < KO_START` first.
