# HANDOFF — Phase 4.0: `ensemble_mkt` model (market-odds output blend)

> Planned 2026-06-02. Phase 3.9 is implemented but **uncommitted**. Next execution session:
> commit Phase 3.9, then wire the `ensemble_mkt` member into the live prediction path.

## Goal

Add the **`ensemble_mkt`** model to the live prediction path so
`predict_fixtures(..., models=['ensemble_mkt'])` works. `ensemble_mkt` blends the calibrated
ensemble W/D/L with market-implied probabilities at the output layer:

```
p_final = normalize( (1-α)·p_ensemble_cal + α·p_market )    # W/D/L vector only
```

α is fitted on historical WC odds. The exact-score matrix / lambdas / top scorelines are
carried over unchanged from the ensemble (market odds constrain 1X2 only, not the scoreline).

## Relevant context and constraints

- Today is 2026-06-02; kickoff 2026-06-11. The **WC2026 odds sheet is not yet published**
  (`WorldCup2026` tab of the fdco xlsx, expected ~2026-06-09). So for live 2026 fixtures
  `ensemble_mkt` falls back to the plain ensemble until odds land — it is safe to build/ship
  now and regenerate predictions later.
- The blend math already exists and is validated in the backtest — this phase only wires it
  into `predict.py`'s frozen-state path. **Do not invent new blending logic.**
- Suite is currently green: `uv run pytest -q` → **135 passed / 1 skipped** (469s).
- WC2026 format in `data/raw/wc2026_fixtures.csv` = 48 teams / 12 groups of 4 / 72 group
  matches. Knockouts and tournament simulator remain deferred (separate future phases).

## Files inspected

- `src/wcpredictor/predict.py` — `_build_frozen_state` (271), `_predict_one_frozen` (406),
  `predict_fixtures` (514). Ensemble state built at 311-400; 2026 `odds_lookup` already built
  at 334-341; ensemble predict branch at 437-497.
- `src/wcpredictor/evaluation/backtest.py` — `_fit_odds_alpha` (108); rolling α collectors
  `prev_odds_*` (241-243, extended 451-453); four-fold market-blend loop (423-453);
  time-aware α with `ODDS_ALPHA_PRIOR` fallback (430-435).
- `src/wcpredictor/features/odds.py` — `align_odds_to_test` (190), symmetric (a,b)/(b,a) lookup.
- `src/wcpredictor/config.py` — `ENSEMBLE_POOL="log"` (42), `FDCO_ODDS_SHA256` (47),
  `ODDS_ALPHA_PRIOR=0.0` (48).

## Current findings

- α should be the **pooled weight fit on all four past WCs (2010/14/18/22)** — the right
  transfer estimate for 2026, since all folds precede the 2026 cutoff. The backtest's fold loop
  already accumulates the needed `(labels, ens_cal_probs, market_probs)` triples; fitting
  `_fit_odds_alpha` on the full accumulator after the loop yields the pooled α.
- `_predict_one_frozen`'s ensemble branch already produces `combined_cal` (calibrated W/D/L)
  and `mat`/lambdas/top scorelines — `ensemble_mkt` reuses this path and only post-blends the
  W/D/L when the fixture has odds.
- 2026 `odds_lookup` is already constructed in `_build_frozen_state` and populated only when
  the `WorldCup2026` sheet exists — so the no-odds fallback is automatic.

## Proposed implementation plan

1. **`backtest.py` — expose pooled α.** After the fold loop, fit one
   `_fit_odds_alpha(prev_odds_labels, prev_odds_ens_probs, prev_odds_market_probs)` (guard the
   empty case → `ODDS_ALPHA_PRIOR`) and add `odds_alpha_pooled` to the returned `results` dict.
   Persist it to `data/processed/backtest_report.json` where the report is written.

2. **`_build_frozen_state` — resolve α once.** Make `ensemble_mkt` imply the `ensemble` state
   (extend the `needs_*` / `"ensemble" in models` guards to also fire on `ensemble_mkt`). Store
   `state["odds_alpha"]`: read `odds_alpha_pooled` from `backtest_report.json` if present;
   otherwise compute via the backtest helper. Reuse the existing `odds_lookup`.

3. **`_predict_one_frozen` — add `ensemble_mkt` branch.** Refactor so the ensemble computation
   runs for `model_name in {"ensemble","ensemble_mkt"}`. Then if `model_name == "ensemble_mkt"`
   and the fixture has odds (`odds_entry`/`has_odds==1.0`): blend `p_win/p_draw/p_loss` with
   `state["odds_alpha"]` against `(odds_pw, odds_pd, odds_pl)`, renormalize, round to 6; keep
   `score_matrix`/`lambda_a/b`/`top_scorelines` from the ensemble. No odds → plain ensemble
   probs (α=0). Set `model_version="ensemble_mkt-0.1"`, `model="ensemble_mkt"`.

4. **`predict_fixtures` signature.** Add `"ensemble_mkt"` to the `model: Literal[...]`
   annotation (line 517). The `models=` list path already passes names through.

5. **Tests (`tests/test_wc2026.py`).**
   - `ensemble_mkt` probs sum to 1.0, all in [0,1].
   - No 2026 odds → `ensemble_mkt` W/D/L == `ensemble` W/D/L.
   - Synthetic 2026 odds injected into `odds_lookup` → blended probs move toward the market
     vector by exactly the stored α (use a small fixed α).
   - `predict_fixtures(models=['poisson','ensemble','ensemble_mkt'])` → 3× rows, correct
     `model` tags, 0 non-summing rows.
   - α ∈ [0,1].

## Exact next steps

1. Verify green (already confirmed) and **commit Phase 3.9** (the 4 modified files:
   `HANDOFF.md`, `src/wcpredictor/evaluation/live.py`, `src/wcpredictor/predict.py`,
   `tests/test_wc2026.py`) — clean base before new work.
2. Implement steps 1-5 above.
3. Run verification (below); commit Phase 4.0.
4. (Deferred, ~2026-06-09) When the `WorldCup2026` sheet publishes: update
   `config.FDCO_ODDS_SHA256` with the new hash, then regenerate with
   `predict_fixtures('2026-06-10', models=['poisson','ensemble','ensemble_mkt'])`.

## Verification / test commands

```bash
uv run pytest -q                                      # expect 135 + new tests pass
uv run pytest -q tests/test_wc2026.py -k "mkt or ensemble"

# pooled α is exposed and sane
uv run python -c "from wcpredictor.evaluation.backtest import backtest_world_cups as b; print('alpha_pooled=', b()['odds_alpha_pooled'])"

# live path runs end-to-end (no 2026 odds yet → ensemble_mkt == ensemble)
uv run python -c "
from wcpredictor.predict import predict_fixtures
import numpy as np
df = predict_fixtures('2026-06-10', models=['poisson','ensemble','ensemble_mkt'])
print(df['model'].value_counts())
s = df[['p_win','p_draw','p_loss']].sum(axis=1)
print('bad rows:', int((np.abs(s-1)>1e-6).sum()))
print(df.groupby('model')[['p_win','p_draw','p_loss']].mean().round(3))
"
```
Expect: 3 models × 72 rows, 0 bad rows, and (pre-odds) `ensemble_mkt` means ≈ `ensemble` means.

## Risks and open questions

- **α transfer mismatch**: pooled α is fit on the backtest's ensemble, which shares code with
  the live frozen ensemble but isn't byte-identical. Accepted — same code path, correct regime.
- **Degenerate fit**: if the pooled fit is empty/degenerate, α falls back to `ODDS_ALPHA_PRIOR`
  (0.0) → `ensemble_mkt` collapses to `ensemble`. Safe.
- **Cost**: prefer reading `odds_alpha_pooled` from `backtest_report.json` over re-running the
  multi-minute backtest at predict time; recompute only if the field is absent.
- **Odds timing**: 2026 `ensemble_mkt` is inert until the `WorldCup2026` sheet lands (~06-09).

## What not to repeat

- Don't add a 4th ensemble member or refit stacking weights — `ensemble_mkt` is a thin
  **output-layer** blend over the existing ensemble.
- Don't re-run the full backtest per fixture — α is resolved once into frozen state.
- Don't blend the score matrix with market 1X2 — odds inform W/D/L only.
- **72× model refits** — always use `predict_fixtures(models=[...])` for batch work, never the
  per-fixture `predict_match` loop.
- Don't loosen the modelling gate; don't use non-WC validation for ensemble weight fitting.

---

## Deferred (future phases, unchanged)
- **Knockout fixtures** — after group stage (2026-06-27): standings → R32 bracket → predict.
  Needs extra-time/penalty resolver.
- **Tournament simulator** — Monte Carlo over groups + knockout; per-team win-cup probs.
  Reuses `_build_frozen_state` / `_predict_one_frozen`.
- **Matchday loop** — `uv run python -m wcpredictor.evaluation.live --as-of <next-day>` after
  each matchday (first real results 2026-06-11); per-model scorecard in
  `data/processed/wc2026_scorecard.json`.
