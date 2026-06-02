# HANDOFF — Phase 3.2: Fix the thin validation slice (escape the 1/3 weight collapse)

## Goal

Attack the **remaining** root cause of the ensemble's ≈1/3-each weight collapse so
`predict_match`'s default can flip from `"poisson"` to `"ensemble"`. Phase 3.1 added
orthogonal recent-form features to the logistic member (cause **a**), but weights still stayed
≈1/3 across all four folds — the diagnosis was cause **b: the 2-year validation slice is too
thin** for the weight optimizer to find signal above the Elo noise floor. This phase widens
the validation slice and re-measures.

This is an **experiment with a decision gate**, not a feature build. The code surface is tiny;
the value is in the measurement and the promote/defer decision.

User decisions for this phase:
- **Direction:** Phase 3.2 = validation-slice fix (not tournament simulator, not new external
  signal yet, not API/frontend).
- **Phase 3.1 disposition:** commit the completed Phase 3.1 work first as its own commit, then
  start 3.2 clean.

## Relevant context and constraints

- **Phase 3.1 is complete and green (71 tests pass) but UNCOMMITTED.** Working tree carries
  `src/wcpredictor/features/form.py`, `tests/test_form.py`, and edits to `logistic.py`,
  `backtest.py`, `predict.py`, `config.py`, `README.md`, `HANDOFF.md`. Promotion was deferred;
  default stayed `"poisson"`. Land this first (Step 0).
- **Phase 3.1 verdict (from prior HANDOFF / README):** form features were orthogonal to Elo
  but the ensemble barely moved (aggregate 0.9800 → 0.9790) and weights stayed ≈1/3. Strict
  bar not met. Per-fold Ens+Cal lost to DC+Cal in 2010 and to Poisson in 2014/2018/2022.
- **Binding rules (CLAUDE.md):** 90-min regulation label; no leakage; time-aware splits only;
  reproducible; no hard-coded 2026 teams; leakage tests are the gate; **no new external data
  source this phase**. Reuse existing `metrics.py` / `calibration.py` / `ensemble.py`.

## Files inspected

- `src/wcpredictor/config.py:39` — `DC_CAL_VALIDATION_YEARS: int = 2`. The constant to change.
- `src/wcpredictor/evaluation/backtest.py`:
  - `L135` `cal_start = wc_start - pd.DateOffset(years=DC_CAL_VALIDATION_YEARS)`.
  - `L211–228` DC calibration validation slice (`val_elo` → temperature `T`, ECE before/after).
  - `L230–269` ensemble stacking: `early_train_*` fit on `< cal_start` (L232–233); member preds
    on the `cal_start → wc_start` slice; weights + ensemble temperature fit there (L242–269).
  - `L300–324` where `model_dc_cal` ECE and `model_ensemble_cal` weights/log_loss are emitted.
  - `L149` hard leakage assertion `train.date.max() < test.date.min()`.
- `tests/test_backtest_time_safety.py` — self-contained; does **not** depend on the constant;
  unaffected by any value tried here.

## Current findings

- **`DC_CAL_VALIDATION_YEARS` is one constant doing double duty.** `cal_start` drives **both**
  the DC calibration slice **and** the ensemble's `early_train` cutoff + weight-fitting slice.
  So bumping it widens the ensemble's weight-fitting data (the intended fix) **but also**
  widens DC's calibration slice and shrinks the `early_train` window. This coupling is exactly
  the "may destabilize DC calibration" risk flagged previously.
- Larger N only moves `cal_start` *earlier* (still far after the 1870s data start), so the
  leakage assertion and `cal_start < wc_start` / `early_train < cal_start` orderings keep
  holding. Existing empty-slice guards (L221/L260/L267) fall back safely.

## Proposed implementation plan

### Step 0 — Land Phase 3.1 (commit first)
Commit the green working tree as its own Phase 3.1 commit. Suggested message:
`Phase 3.1: form features for logistic member (promotion deferred)`.

### Step 1 — Experiment A: shared bump (cheapest, try first)
Set `config.py:39` `DC_CAL_VALIDATION_YEARS = 3`, run the backtest, record per fold:
- `model_ensemble_cal` weights `[poi, dc, log]` — did they move meaningfully off 1/3?
- `model_ensemble_cal.log_loss` vs Poisson and DC+Cal (strict bar, all four folds + aggregate).
- `model_dc_cal` `ece_before_val → ece_after_val` + `temperature` — **did DC calibration
  destabilize** vs the committed N=2 numbers?

Repeat with `4`. Compare 2 / 3 / 4 side by side.

### Step 2 — Decision gate
- **Clean win** (weights off 1/3, DC ECE not worse, strict bar met every fold at N=3 or 4):
  promote → Step 4.
- **Ensemble helped but DC calibration destabilized:** decouple → Step 3, re-measure, re-gate.
- **No movement / bar still missed even decoupled:** record the negative result, leave default
  `"poisson"`, document, stop. Do **not** loosen the bar.

### Step 3 — Experiment B: decouple (only if Step 2 says DC destabilized)
- Add `ENSEMBLE_VAL_YEARS: int = 3` (or 4) to `config.py`; **leave `DC_CAL_VALIDATION_YEARS = 2`**.
- In `backtest.py`, compute a separate `ens_cal_start = wc_start - pd.DateOffset(years=
  ENSEMBLE_VAL_YEARS)` used **only** for the ensemble block: the `early_train_elo`/
  `early_train_matches` cutoff (L232–233) and a dedicated `ens_val_elo` slice feeding steps 2–3
  (L242–269). Keep the existing `cal_start`/`val_elo` DC-calibration path byte-for-byte
  unchanged. Add `ENSEMBLE_VAL_YEARS` to the config import at L30.
- Re-run; re-apply the Step 2 gate.

### Step 4 — Promote (only if the gate is cleared)
- `predict.py`: change the `model` param default `Literal[...] = "poisson"` → `"ensemble"` and
  update its docstring.
- `README.md`: mark Ensemble as default in the model table, replace the Phase 3.1 result block
  with Phase 3.2 numbers, bump `model_version`.
- Regenerate `data/processed/backtest_report.json` (the backtest `__main__` already writes it).

### Step 5 — Always: update this HANDOFF
Record the chosen `N` (and whether decoupled), per-fold table, weights, gate outcome, and the
promote/defer decision (mirror the Phase 3.1 verdict section). Fill the "Exact next steps"
below with what follows (e.g. the orthogonal-signal path if promotion is still deferred).

## Strict promotion bar (unchanged from Phase 3.1 — do NOT loosen)

Flip the default to `"ensemble"` **only if** `model_ensemble_cal` satisfies **all**:
1. W/D/L `log_loss` ≤ best single model (Poisson / DC+Cal) on **all four folds**,
2. best **aggregate** mean log loss, and
3. calibrated ECE no worse than `model_dc_cal`.

## Exact next steps

1. `git add -A && git commit` the Phase 3.1 working tree (Step 0).
2. Edit `config.py:39` → `3`; run the backtest; record metrics (Step 1).
3. Repeat at `4`; compare; apply the gate (Step 2).
4. If DC destabilized, decouple (Step 3) and re-measure.
5. Promote or document-and-defer (Step 4 / Step 5).

## Verification / test commands

- `uv run pytest -q` — all 71 tests stay green for every config value tried (time-safety +
  leakage tests gate).
- `uv run python -m wcpredictor.evaluation.backtest` — inspect per-fold
  `w=[poi … dc … log …]` (success = weights clearly off 1/3) and `model_ensemble_cal` vs
  Poisson/DC+Cal; confirm `model_dc_cal` ECE not worse than the committed N=2 baseline.
- If promoted: `uv run python -c "from wcpredictor.predict import predict_match; import json;
  print(json.dumps(predict_match('Brazil','France','2026-06-15',neutral=True),indent=2)[:400])"`
  — default now routes through the ensemble; shape (§11.1) intact, probs sum≈1, 5 scorelines.

## Risks and open questions

- **May still not clear the bar.** Widening trades optimizer signal against DC calibration
  stability and a shorter `early_train`. Acceptable outcome: stays `"poisson"`, documented.
  Do not force the flip.
- **N=4 on the 2010 fold:** `early_train` becomes `< 2006`; confirm `early_train_*` and
  `val_elo`/`ens_val_elo` are non-empty for every fold (guards at L221/L260/L267 cover it).
- **Decoupling:** keep DC's `val_elo`/`cal_start` path unchanged so `model_dc`/`model_dc_cal`
  numbers don't move for reasons unrelated to the experiment.

## What not to repeat / failed approaches

- Don't revisit `FORM_WINDOW` / form features — Phase 3.1 settled that; the problem is the
  slice, not the window.
- Don't add external data (xG/odds/squads) this phase — that's the *next* fallback if 3.2 also
  defers, and it needs explicit approval (no-scraping rule).
- Don't fit weights/temperature/logistic on the holdout or in-sample member preds; keep the
  out-of-time slice and the per-fold leakage assertion.
- Don't loosen the strict promotion bar to manufacture a flip.
