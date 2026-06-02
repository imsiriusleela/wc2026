# HANDOFF — Phase 3.8 complete / next phase TBD

> Completed 2026-06-02. Phase 3.8 weight-fitting fix implemented and gated.
> Result: GBM weights remain ≈0.25 — odds edge is not consistently strong across past WCs.
> Gate still fails 2010/2014/2022. 2018 continues to pass. Next steps below.

## What was done (Phase 3.8)

**Goal**: Fix the weight-fitting blind spot so the GBM's odds signal could earn ensemble weight.

**Change**: Replaced the non-WC validation slice (which had zero odds-bearing rows) with a
walk-forward WC stacking validation for `ens_fit_weights`. For fold year Y, fits all four
members on data `< wc_start(w)` for each past WC `w < Y`, predicts WC `w` (odds-bearing),
and concatenates. Falls back to non-WC slice / equal weights for 2010 (no prior WC).

**Files changed (committed `b1c7e9b`)**:
- `src/wcpredictor/evaluation/backtest.py` — `build_wc_stacking_validation` + wired into fold loop
- `src/wcpredictor/predict.py` — mirrored: `predict_match("ensemble")` uses same WC stacking val
- `tests/test_backtest_time_safety.py` — 4 new tests (structural, leakage-safe monkeypatch,
  fit_weights dominance)

**Tests**: 129 passed / 1 skipped (all green).

## Gate results (Phase 3.8)

| Fold | Best single model (LL) | Ens+Cal (LL) | Ens+Mkt (LL) | Gate |
|------|------------------------|--------------|--------------|------|
| 2010 | DC+Cal  0.9474 | 0.9645 | 0.9645 (α=0.000) | FAIL |
| 2014 | Poisson 0.9152 | 0.9210 | 0.9210 (α=0.046) | FAIL |
| 2018 | Poisson 0.9690 | 0.9564 | 0.9552 (α=0.041) | PASS |
| 2022 | Poisson 1.0541 | 1.0681 | 1.0366 (α=0.254) | FAIL |

Weights for all folds (after fix): ≈[poi=0.25, dc=0.25, log=0.25, tree=0.25].

**Root cause of ≈0.25 weights**: The GBM's odds-informed edge is not consistently dominant
across past WCs (2010 betexplorer odds are low-signal; 2014 only has WC2010 as val — 64
matches with L2 prior). The equal-weight L2 prior absorbs the small per-WC signal. This is
the documented risk from the previous HANDOFF and is a clean result, not a bug.

## Key findings / current state

- **Ens+Mkt is the primary odds mechanism** (output-layer blending, not weight-fitting):
  - 2022 improved dramatically: 1.0681 → 1.0366 (α=0.254), now beats Poisson (1.0541)
  - 2018: 0.9564 → 0.9552 (α=0.041), marginal improvement
  - 2014: 0.9210 → 0.9210 (α=0.046), effectively zero
  - 2010: 0.9645 → 0.9645 (α=0.000), no odds signal (prior fold)
- **Gate still requires all-four-folds**: 2010 and 2014 are structurally hard regardless of odds.
- **Part B (live pipeline)**: fdco `WorldCup2026` sheet not yet fetched locally. `download_wc2026.py`
  exists but `wc2026_fixtures.csv` not yet written. Sheet IS published per user.

## Possible next directions (user's call)

**Option A — Loosen gate for 2010/2014 (not recommended without user approval)**
- 2010 and 2014 may be fundamentally hard epochs; ensemble not expected to beat DC+Cal/Poisson there.
- Current gate is an absolute rule; loosen only if user explicitly reopens it.

**Option B — Alternative ensemble target: Ens+Mkt instead of Ens+Cal**
- For 2022 and 2018 (WC years with good odds), the Ens+Mkt beats best single model.
- Gate could be redefined as "Ens+Mkt beats best single model on folds where odds available".
- This would be a cleaner framing: odds improve things when odds are reliable.

**Option C — Investigate why 2014 Ens+Cal regresses vs Poisson**
- Poisson (0.9152) beats Ensemble (0.9210) on 2014 — Poisson is actually the strongest member.
- The ensemble dilutes Poisson by blending it with DC (weaker on 2014), logistic, and tree.
- May need a more aggressive L1/sparsity prior or per-fold minimum-weight enforcement.

**Option D — Part B: activate live pipeline**
- Re-fetch fdco WC2026 sheet via `download_wc2026.py`
- Or receive `wc2026_fixtures.csv` from user's hermes agent
- Run `live.py:run_refresh` → `wc2026_scorecard.json`
- This produces live predictions regardless of gate status

**Option E — Feature engineering to make GBM distinct**
- GBM currently uses same Elo/form features as logistic → small signal gap
- Interaction features, momentum decay, head-to-head recency might differentiate it
- More likely to earn non-trivial weight once it's genuinely complementary to Poisson/DC

## What not to repeat

- **Output-layer odds blending (Phase 3.4)** — failed the gate at every α (before Ens+Mkt was tuned).
- **Zero-filling missing odds in GBM** — destroys the "no odds" signal.
- **Plain GBM on Elo/form only (Phase 3.5)** — 0.25 weights, no signal.
- **Non-WC validation for weight fitting** — the Phase 3.8 blind spot; now fixed.
- **Loosening the gate without explicit user approval.**

## Verification commands

```bash
uv run pytest -q                                # 129 passed / 1 skipped
uv run python -m wcpredictor.evaluation.backtest  # gate; weights per fold
uv run python -m wcpredictor.data.download_wc2026  # (Part B) fetch WC2026 sheet
```
