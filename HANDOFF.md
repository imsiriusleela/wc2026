# HANDOFF — Phase 3.1: Decorrelate the ensemble with richer derived features

## Goal

Make the calibrated ensemble {Poisson, Dixon-Coles, multinomial-logistic} good enough to
**beat every World Cup fold** so `predict_match`'s default can flip from `"poisson"` to
`"ensemble"`. We do this by attacking the **root cause of the ≈1/3-each weight collapse**:
the logistic member currently encodes the same Elo signal as Poisson, so the three members
are redundant. Give the logistic member **leak-free derived features from the existing match
data** (recent form, goal momentum, rest/fatigue) that the strength-based members
structurally cannot capture. No new external data sources.

User decisions for this phase:
- **Approach:** richer *derived* features from existing data (not a longer validation
  window, not new external data).
- **Promotion bar:** strict — win on every fold (see "Promotion decision" below).

## Relevant context and constraints

- **Phase 3 is complete and green: 57 tests pass.** `models/ensemble.py`,
  `models/logistic.py`, leakage-safe stacking in `evaluation/backtest.py`, and the
  `model="ensemble"` path in `predict.py` all exist and work. Do **not** rebuild them.
- **Why this phase exists.** From `data/processed/backtest_report.json`: the ensemble wins
  on aggregate and dominates DC+Cal in 3/4 folds, but loses to DC+Cal in 2010 and is
  fractionally behind Poisson in 2014–2018. Fitted blend weights converge to ≈1/3 each
  because (a) the logistic member uses only `elo_diff_adj`+`neutral` — redundant with
  Poisson — and (b) the thin 2-year validation slice gives the optimizer little signal.
  We fix (a); (a) is the dominant cause.
- **Binding rules (CLAUDE.md):** 90-min regulation label; no leakage; time-aware splits
  only; reproducible; no hard-coded 2026 teams; leakage tests are the gate; no silent
  scraping (this phase adds **no** new data source). Reuse `metrics.py`, `calibration.py`.
- **Leakage discipline:** new features must be emitted **pre-match** (before the per-team
  state is updated with the current result), exactly like `compute_elo`.

## Files inspected

- `src/wcpredictor/features/elo.py` — `compute_elo(matches) -> (features_df, final_ratings)`
  walks matches in date order, **emits the pre-match row before updating** Elo (the
  leak-free pattern to mirror). `latest_elo(...)` returns current ratings before a date.
- `src/wcpredictor/models/logistic.py` — `_build_X` stacks only `elo_diff_adj`, `neutral`;
  `fit`/`predict_proba` wrap `StandardScaler` + `LogisticRegression`. The extension point.
- `src/wcpredictor/evaluation/backtest.py` — `elo_all = compute_elo(matches)` at L127; all
  fold slices (`train_elo`, `early_train_elo`, `val_elo`, `test_elo`) are slices of
  `elo_all` and feed `log_fit`/`log_predict`. Leakage-safe stacking (steps 1–5, L227–286)
  fits weights/temperature on an out-of-time validation slice. Hard assertion
  `train.date.max() < test.date.min()` at L146.
- `src/wcpredictor/predict.py` — `model="ensemble"` path (L84–180) hand-builds the logistic
  feature row at **L152** `test_row = pd.DataFrame({"elo_diff_adj": [...], "neutral": [...]})`.
  This row must also carry the new features for predict/backtest parity.
- `src/wcpredictor/config.py` — Elo/DC/ensemble constants; `DC_CAL_VALIDATION_YEARS = 2`,
  `ENSEMBLE_POOL = "log"`. Add the new form constants here.

## Current findings

- The logistic member is fit on the **full train set** (thousands of matches), while only
  the ensemble *weights* are fit on the thin validation slice. So adding features to the
  logistic member does **not** worsen thin-slice overfit — it sharpens the member, which is
  exactly what lets the stacker move weights off 1/3.
- Poisson uses `elo_diff_adj` (long-run strength + home); DC uses per-team attack/defense
  over 10 years with a 2-year half-life (medium-run strength). **Neither encodes short-term
  state** — last-5-games form, scoring/conceding momentum, fixture congestion. Those are the
  orthogonal signals.
- Because every fold slice derives from `elo_all`, merging the new features into `elo_all`
  **once** propagates them through backtest and (with a parallel merge) predict — minimal
  surface area.

## Proposed implementation plan

### New: `src/wcpredictor/features/form.py` (mirror `compute_elo`)
`compute_form(matches, window=FORM_WINDOW) -> (form_df, form_state)`
- One date-ordered walk; per-team rolling state: deque of last-`window` results (pts 3/1/0),
  deque of last-`window` goal differences, `last_match_date`.
- **Emit the pre-match row before updating state** (leak-free). Features differenced
  (team_a − team_b) to stay signed like `elo_diff_adj`:
  - `form_diff` — rolling points-per-game over last `window`
  - `momentum_diff` — rolling mean goal difference over last `window`
  - `rest_diff` — days since previous match (each side capped at `REST_DAYS_CAP`)
- Columns: `match_id, form_diff, momentum_diff, rest_diff`. Cold start → neutral defaults
  (0, 0, cap); **no NaNs**.
- `form_state` maps team → current deques + last date (parallels `final_ratings`).
- Helper `form_row(form_state, team_a, team_b, query_date, window) -> dict` builds the three
  differenced features for one hypothetical match (for the predict single-match path).

### Modify: `src/wcpredictor/models/logistic.py`
Extend `_build_X` to stack `["elo_diff_adj", "neutral", "form_diff", "momentum_diff",
"rest_diff"]`, **tolerant of absent columns (default 0.0)** so old call sites/tests stay
valid. Update the module docstring's feature list. `StandardScaler` handles the new scales.

### Modify: `src/wcpredictor/evaluation/backtest.py`
After `elo_all, _ = compute_elo(matches)` (~L127): `form_all, _ = compute_form(matches)`
then `elo_all = elo_all.merge(form_all, on="match_id", how="left")`. **No other backtest
edit** — downstream slices inherit the columns; Poisson/DC paths ignore them.

### Modify: `src/wcpredictor/predict.py` (ensemble path only)
- After `compute_elo(train)`, also `compute_form(train)`; merge form columns into `elo_df`
  before the logistic fits (L79/L112/L144).
- Replace the L152 `test_row` to include `form_diff/momentum_diff/rest_diff` from
  `form_row(form_state, team_a, team_b, cutoff, FORM_WINDOW)`.
- `poisson` / `dixon_coles` paths unchanged.

### Modify: `src/wcpredictor/config.py`
Add `FORM_WINDOW: int = 5` and `REST_DAYS_CAP: int = 30`. Leave `DC_CAL_VALIDATION_YEARS = 2`.

### Tests
- **New `tests/test_form.py`:** leak-free (first appearance of every team → neutral
  defaults; a row uses only strictly-earlier matches); differenced symmetry (swap team_a/b →
  negated diffs); `rest_diff` sign + cap; deterministic; no NaNs.
- **Update `tests/test_logistic.py`:** add the three columns to fixtures; keep
  monotonic-in-`elo_diff_adj` (others held constant); add a case proving `_build_X` works
  when the new columns are absent (defaults 0.0).
- `tests/test_ensemble.py` unaffected (operates on probability vectors).

## Promotion decision (strict bar)

Re-run the backtest, then flip `predict_match`'s default to `"ensemble"` **only if**
`model_ensemble_cal` satisfies **all**:
1. W/D/L `log_loss` ≤ best single model (Poisson / DC+Cal) on **all four folds**,
2. best **aggregate** mean log loss, and
3. calibrated ECE no worse than `model_dc_cal`.

If met: change the `Literal[...] = "poisson"` default + docstring in `predict.py`; update
`README.md` result table + `model_version`. If **not** met: leave default `"poisson"`,
record new weights/metrics, report promotion still deferred — **do not loosen the bar to
force a flip**. Regenerate `data/processed/backtest_report.json` either way.

## Exact next steps

1. Start a fresh **execution** session (this was planning-only).
2. `config.py`: add `FORM_WINDOW`, `REST_DAYS_CAP`.
3. `features/form.py`: `compute_form` + `form_row` → `tests/test_form.py` green.
4. `models/logistic.py`: extend `_build_X` (+ docstring) → update `tests/test_logistic.py` green.
5. `backtest.py`: merge `form_all` into `elo_all`.
6. `predict.py`: merge form into `elo_df`; build form features in the single-match `test_row`.
7. Run full verification (below); read per-fold `model_ensemble_cal` + printed weights.
8. Apply the **Promotion decision** strictly; update `README.md` + this `HANDOFF.md` verdict;
   regenerate `backtest_report.json`.

## Verification / test commands

- `uv run pytest -q` — all green (57 existing + new `test_form`; leakage tests gate).
- `uv run python -m wcpredictor.evaluation.backtest` — inspect per-fold `model_ensemble_cal`
  vs Poisson/DC+Cal and the printed `w=[poi … dc … log …]`. **Success signals:** weights
  move meaningfully off 1/3 (logistic earns weight) and the strict bar is met on every fold.
- `uv run python -c "from wcpredictor.predict import predict_match; import json;
  print(json.dumps(predict_match('Brazil','France','2026-06-15',neutral=True,model='ensemble'),indent=2)[:400])"`
  — §11.1 shape intact, probs sum≈1, 5 scorelines, sane λ.

## Risks and open questions

- **May still not clear the strict bar.** Form features are orthogonal but Elo dominance is
  strong; the ensemble may improve yet still lose a fold. Acceptable — default stays
  `"poisson"`. Do not force the flip.
- **`FORM_WINDOW` choice.** 5 is the default; if borderline, try 10 as a one-line documented
  change — not a free hyperparameter search.
- **Cold-start teams** get neutral defaults; `StandardScaler` centres them and the bulk of
  training data is well-populated.
- **Predict/backtest parity.** `form_row` must reuse the same `FORM_WINDOW`/`REST_DAYS_CAP`
  as the walk, or query features drift from the training distribution.

## What not to repeat / failed approaches

- Don't widen the validation window or add external data (xG/odds/squads) in this phase —
  the user chose derived features; new sources also trip the no-scraping rule.
- Don't re-add DC-strength-diff (`α_a−α_b`) to the logistic member — it re-introduces
  redundancy with the DC member; the point is *orthogonal* short-term signals.
- Don't fit the stacker weights or logistic on the holdout / in-sample member preds — keep
  the out-of-time validation slice and the per-fold leakage assertion.
- Don't compute form features in a second non-chronological pass — emit pre-match within one
  date-ordered walk, like `compute_elo`.
- Don't loosen the strict promotion bar to make the flip happen.
