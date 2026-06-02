# HANDOFF — Phase 3.5 closed / Phase 3.6 open

> Phase 3.5 complete (2026-06-02): GBM member shipped. Gate **failed** on all four folds.
> Default model remains `"poisson"`. See gate table below.

## Phase 3.5 closure summary

**What was done**
- `src/wcpredictor/models/gbm.py` added — `HistGradientBoostingClassifier`-based 4th ensemble member.
  Features: elo_diff_adj, neutral, form_diff, momentum_diff, rest_diff, elo_a_pre, elo_b_pre.
  Deterministic (seed=42, early_stopping=False). No new deps (sklearn>=1.8 already present).
- `tests/test_gbm.py` added (9 tests: shape, sum-to-1, no-NaN, determinism, feature builder,
  missing-class renormalize, directional sanity).
- `backtest.py` + `predict.py` wired in the 4th member; `model_version` bumped to `ensemble-0.2`.
- LightGBM not installed → HistGradientBoosting selected by default (mean val log loss 0.8962).

**Gate table (Phase 3.5 — 4-member Ens+Cal vs best single model)**

| Fold | Ens+Cal | Best single | Deficit | Pass? |
|------|---------|-------------|---------|-------|
| 2010 | 0.9645  | DC+Cal 0.9474 | −0.0171 | ✗ |
| 2014 | 0.9196  | Poisson 0.9152 | −0.0044 | ✗ |
| 2018 | 0.9696  | Poisson 0.9690 | −0.0006 | ✗ |
| 2022 | 1.0657  | EloOnly 1.0539 | −0.0118 | ✗ |

All four ensemble weights locked at 0.25 — the optimizer found equal weights on the validation
slice, indicating the tree adds no net signal beyond the linear/Poisson/DC members.

**Why the tree didn't help**
- The tree re-learns the Elo→W/D/L mapping the Poisson + logistic members already capture.
  Equal weights (0.25) confirm the optimizer sees no orthogonal signal.
- 2010 structural underperformance persists (DC is the best single model there).
- 2014 deficit widened from 0.0023 (Phase 3.3) to 0.0044 — the tree added noise.

**Default unchanged**: `"poisson"`. Promotion deferred.

---

## Phase 3.6: next steps (open)

The tree hypothesis is exhausted with current features. Remaining options:

### Option A — richer features for the tree
Add H2H win-rate, FIFA ranking diff, squad value ratio, or confederation encoding.
Each feature is cheap to add; hypothesis: the tree gains orthogonal signal it can't find
in Elo alone. Risk: data quality / availability for pre-2010 matches.

### Option B — revisit ensemble architecture
Replace hard stacking with soft blending calibrated on a rolling out-of-time window
(slide rather than 2-year fixed slice). May close the 2022 deficit without new features.

### Option C — odds-as-feature (was Phase 3.4 Option C)
Feed normalised market probabilities as GBM features rather than blending at the output
layer. Hypothesis: tree can learn non-linear corrections from odds. Requires re-opening
the market-odds work.

### Option D — tournament simulator (Phase 4)
Skip further member tuning; accept current accuracy and build the tournament simulator.
Useful for the June 2026 demo regardless of whether the ensemble beats Poisson.

**Recommended**: Option D (tournament simulator) — the marginal accuracy gains from
more ensemble tuning are tiny, and the simulator unlocks the demo-able end-to-end product.

## Files to inspect for the next session

- `src/wcpredictor/models/gbm.py` — current GBM member (baseline for feature expansion)
- `src/wcpredictor/evaluation/backtest.py` — leakage-safe stacking, 4-member (lines ~270–420)
- `src/wcpredictor/predict.py` — ensemble path (lines ~88–191), `model_version="ensemble-0.2"`
- `data/processed/backtest_report.json` — latest gate numbers
- `CLAUDE.md` — phase roadmap (next planned phase: tournament simulator)

## Verification commands

```bash
uv run pytest -q                                       # 97 passed / 1 skipped
uv run python -m wcpredictor.evaluation.backtest       # see gate table above
uv run python -c "from wcpredictor.predict import predict_match; \
  print(predict_match('Brazil','Argentina','2022-11-20', model='ensemble')['p_win'])"
# → ~0.469
```
