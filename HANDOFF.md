# HANDOFF — Phase 3.5: gradient-boosted tree member (Option A)

> Decision locked (2026-06-02): Phase 3.4 market-odds is CLOSED/deferred. Next is **Option A** —
> add a 4th, non-linear gradient-boosted member to the ensemble. Library choice delegated to the
> implementer with the instruction "best for accuracy" → pick empirically (LightGBM vs sklearn
> HistGradientBoosting) on out-of-time validation log loss.

## Goal
Add a gradient-boosted tree member to the Poisson + Dixon-Coles + Logistic ensemble, wired into the
leakage-safe stacking pipeline in both `backtest.py` and `predict.py`, then re-run the backtest to test
whether the 4-fold promotion gate now passes. Ship the member; promote the default model **only if** the
gate passes on all four folds (2010, 2014, 2018, 2022) — otherwise commit "promotion deferred" with the
updated gate table, matching the 3.3/3.4 convention.

## Relevant context and constraints
- Current ensemble members are all either count-based (Poisson, DC) or **linear** (multinomial Logistic
  on Elo/form). Hypothesis: a non-linear tree captures interactions the linear member misses → orthogonal
  signal that closes the razor-thin gate failures.
- Gate = "ensemble beats best single model on **all four** folds." Phase 3.4 failures were tiny:
  2014 by **0.0023**, 2018 by **0.0003** log loss; 2010 by 0.0186 (structural, market couldn't fix it).
- Constraints from CLAUDE.md: time-aware splits, **no leakage** (all features pre-kickoff), probabilistic
  outputs, **every model reproducible**, uv for env, src/ layout, add leakage/cutoff tests.
- The ensemble combiner (`models/ensemble.py` `fit_weights`/`combine_probs`) is **already generic over N
  members** — no change needed there. Member order must stay `[poisson, dc, logistic, tree]` so the
  score-matrix renormalization `ens_weights[:2]` (Poisson+DC are the only members with score matrices)
  remains correct.
- The ensemble stacking logic is **duplicated** in `backtest.py` and `predict.py`; both must be updated in
  lockstep.

## Files inspected
- `src/wcpredictor/evaluation/backtest.py` — leakage-safe stacking: early fits `< cal_start`, validation
  predictions on the 2-yr slice, weight+temperature fit on validation, full fits `< wc_start`, combine on
  test. Member lists at `:300` (val) and `:317` (test); score-matrix weights at `:322`; report at `:391`.
- `src/wcpredictor/models/logistic.py` — the interface to mirror: `fit(features_df, labels) -> (scaler,
  model)`, `predict_proba(...) -> [[p_win,p_draw,p_loss],...]`, missing-class→zero-col renormalize
  (`:56-66`), `_build_X` (elo_diff_adj, neutral, form_diff, momentum_diff, rest_diff).
- `src/wcpredictor/models/ensemble.py` — generic N-member combiner; no change required.
- `src/wcpredictor/predict.py` — duplicated ensemble path (`:88-191`); single-row feature frame at
  `:157-163`; `model_version="ensemble-0.1"`.
- `pyproject.toml` — deps: numpy, openpyxl, pandas, scikit-learn>=1.8, scipy. No tree lib yet.

## Current findings
- Phase 3.4 work is **uncommitted** in the working tree (config.py, backtest.py, odds.py, test_odds.py,
  new data/download_wc2010_odds.py, HANDOFF.md). Tests: 88 passed / 1 skipped.
- `.playwright-mcp/` is an untracked tooling dir — should be gitignored, not committed.
- sklearn>=1.8 is already a dependency, so `HistGradientBoostingClassifier` is available with **zero new
  deps** — the safe fallback if LightGBM doesn't win the empirical comparison.

## Proposed implementation plan
1. **Commit Phase 3.4 first** (`Phase 3.4: ... (promotion deferred)`) and add `.playwright-mcp/` to
   `.gitignore`, so 3.5 starts clean.
2. **New `src/wcpredictor/models/gbm.py`**, mirroring `logistic.py`:
   - `fit(features_df, labels)` — regularized GBM; deterministic (fixed seed, single-thread/deterministic
     flags); conservative params (shallow trees / small num_leaves, high min_child_samples, modest
     n_estimators+lr, L1/L2) to fight overfitting on small data.
   - `predict_proba(...)` — reuse logistic's missing-class→zero-col renormalize.
   - `_build_X` — same features as logistic **plus raw `elo_a_pre`, `elo_b_pre`** (cheap orthogonal signal
     a tree can use that the diff-only linear member cannot). **No new feature modules** this phase.
3. **Empirical library pick**: train LightGBM and HistGradientBoosting, compare **validation-slice log
   loss**, keep the winner. If LightGBM wins, add `lightgbm>=4.0` to `pyproject.toml` + `uv sync`; if
   HistGradientBoosting wins, no new dep.
4. **Wire into `backtest.py`** as the 4th member: `early_tree` fit alongside `early_log_*` (`:278`) →
   append to `member_probs_val` (`:300`); `tree_model` fit alongside `log_fit` (`:313`) → append to
   `member_probs_test` (`:317`). Add `weights_tree` to `model_ensemble_cal` and the print line.
5. **Wire into `predict.py`** identically (`:136` val member, `:167` single member; build tree's single-row
   frame like logistic `test_row` + elo_a_pre/elo_b_pre). Bump `model_version` → `"ensemble-0.2"`.
6. **New `tests/test_gbm.py`**: shape (n,3)/sum-to-1/no-NaN, **determinism** (same seed→identical preds),
   missing-class handling, feature-builder columns, strong-vs-weak smoke. Ensure `test_ensemble.py` covers
   a 4-member case.
7. Re-run full suite + backtest; update this HANDOFF with the gate table; commit (promote default only if
   gate passes all four folds).

## Exact next steps (for the execution session)
1. `git add -A && git commit` the Phase 3.4 changes; add `.playwright-mcp/` to `.gitignore`.
2. Write `src/wcpredictor/models/gbm.py` + `tests/test_gbm.py`; `uv run pytest -q tests/test_gbm.py`.
3. Run the LightGBM-vs-HistGradientBoosting validation comparison; lock the winner (+ `uv sync` if dep added).
4. Edit `backtest.py`, then `predict.py`, to add the 4th member.
5. `uv run pytest -q` (stay green) → `uv run python -m wcpredictor.evaluation.backtest` → inspect gate.

## Verification / test commands
```bash
uv sync                                            # if lightgbm added
uv run pytest -q                                   # currently 88 passed / 1 skipped; must stay green + new tests
uv run python -m wcpredictor.evaluation.backtest   # all four folds + gate
uv run python -c "from wcpredictor.predict import predict_match; \
  print(predict_match('Brazil','Argentina','2022-11-20', model='ensemble')['p_win'])"  # facade parity
```
Gate check: read `data/processed/backtest_report.json` — new `model_ensemble_cal` (and
`model_ensemble_market`) must beat the best single model on **all four** folds; watch the 2014 (0.0023)
and 2018 (0.0003) deficits close.

## Risks and open questions
- **Overfitting on small data** (main risk) → conservative regularization + empirical lib pick + the
  existing temperature scaling on ensemble output; select on the out-of-time validation slice, never the
  holdout.
- **Orthogonality not guaranteed**: if the tree just re-learns the linear Elo→W/D/L map, its ensemble
  weight → ~0 and the gate stays failed. That's still a clean documented result → follow-up would add
  H2H / FIFA-rank features (deferred) or revisit Option C (odds-as-feature).
- **Reproducibility** is non-negotiable: GBM fit must be deterministic or the backtest report and tests
  become unstable.
- **Two code paths** (`backtest.py` + `predict.py`) must change in lockstep; a future refactor could
  extract a shared `build_ensemble()` (out of scope).

## What not to repeat (failed approaches)
- Any `ODDS_ALPHA_PRIOR` tuning — no value beats the gate (Phase 3.4 grid proved it).
- Betexplorer/alternative 2010 market odds — 2010 underperformance is **structural**, not data quality.
- Loosening the gate (Option B) — explicitly out of bounds unless the user reopens it.

---

## Appendix — Phase 3.4 closure (for reference)
- Scraped betexplorer WC2010 1X2 average odds (user-approved one-time Playwright render); deterministic
  re-parse via `data/download_wc2010_odds.py`, SHA-256 pinned in `config.py`.
- `features/odds.py`: `load_wc_odds()` merges WC2010 CSV; `align_odds_to_test()` symmetric (W/L swap).
- `evaluation/backtest.py`: auto-parses WC2010 odds on startup. Tests: 88 pass / 1 skip (+7 WC2010).
- Backtest (ODDS_ALPHA_PRIOR=0.0): Ens+Mkt passes **only 2022** (1.0253 < Poisson 1.0541); 2010/2014/2018
  fail. Oracle α for 2010 = 0.073 still > gate. Default unchanged: `"poisson"`, no promotion.
