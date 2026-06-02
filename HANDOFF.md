# HANDOFF — Phase 3.2 CLOSED (deferred) → Phase 3.3 planning

## Phase 3.2 Result: DEFERRED — default stays `"poisson"`

### Experiment results (N=2 baseline + A at N=3, N=4)

**Per-fold Ens+Cal log_loss vs. best single model:**

| Fold | N=2 Ens+Cal | N=3 Ens+Cal | N=4 Ens+Cal | Poisson | DC+Cal | Winner |
|------|-------------|-------------|-------------|---------|--------|--------|
| 2010 | 0.9660 | 0.9666 | 0.9677 | 0.9828 | **0.9474** | DC+Cal |
| 2014 | 0.9172 | **0.9147** | **0.9145** | 0.9152 | 0.9376 | Ens+Cal (N≥3) |
| 2018 | 0.9693 | **0.9661** | **0.9689** | 0.9690 | 0.9904 | Ens+Cal (N≥3) |
| 2022 | 1.0634 | 1.0594 | 1.0603 | **1.0541** | 1.1037 | Poisson |

**Ensemble weights across all N:** always ≈ [0.33, 0.33-0.34, 0.34] — no meaningful escape from uniform.

**DC calibration ECE (before→after):**

| Fold | N=2 | N=3 | N=4 |
|------|-----|-----|-----|
| 2010 | 0.040→0.014 | 0.033→0.007 | 0.032→0.007 |
| 2014 | 0.031→0.021 | 0.024→0.014 | 0.022→0.011 |
| 2018 | 0.034→0.038 | 0.024→0.018 | 0.020→0.015 |
| 2022 | 0.047→0.020 | 0.039→0.017 | 0.022→0.011 |

DC calibration did **not** destabilize — ECE improved at N=3 and N=4. No decoupling needed.

**Aggregate mean log_loss:**
- N=2: Ens+Cal 0.9790, Poisson 0.9803, DC+Cal 0.9965 → Ens+Cal wins aggregate
- N=3: Ens+Cal 0.9767, Poisson 0.9803, DC+Cal 0.9950 → Ens+Cal wins aggregate
- N=4: Ens+Cal 0.9779, Poisson 0.9803, DC+Cal 0.9920 → Ens+Cal wins aggregate

### Gate decision

Strict bar requires Ens+Cal ≤ best single on **all four folds**. 
- 2010: Ens+Cal loses to DC+Cal by ≥0.019 at all N → **FAIL**
- 2022: Ens+Cal loses to Poisson by ≥0.005 at all N → **FAIL**

Verdict: **DEFER.** Default remains `"poisson"`. Config reverted to N=2. 71 tests green.

### Root-cause diagnosis (for Phase 3.3)

The weight optimizer cannot escape uniform because all three members are too correlated:
- Poisson and DC both model score counts from the same match data
- Logistic is Elo-based — same signal source as Poisson's Elo-diff input
- Widening the slice from 2→3→4 years gives more optimizer data but doesn't reduce member correlation
- The signal that would differentiate them (per-match xG, lineup strength, market odds) isn't in the current feature set

## Phase 3.3 planning: break member correlation

Three candidate paths (need explicit user decision before implementation):

### Option A: External signal — market odds or xG
- Add pre-match closing odds or xG-based team strength as features for the logistic member
- This would make logistic genuinely orthogonal to Elo/count-based members
- **Blocker**: no-silent-scraping rule requires explicit approval of each new data source
- Best odds source: football-data.co.uk (free, historical WC odds available)
- Best xG source: FBref / StatsBomb open data

### Option B: Structural fix — replace logistic with a model that sees different information
- Swap the logistic member for a gradient boosted model (XGBoost/LightGBM) trained on
  richer time-series features: Elo trajectory, recent scoring rates, opponent strength index
- This is "XGBoost overlay" from the CLAUDE.md model roadmap (step 5)
- Stays within existing data; no new scraping needed
- Requires building the feature engineering pipeline first (~1 day)

### Option C: Score-matrix ensemble only (no outcome blending)
- Accept that outcome (W/D/L) blending adds no signal
- Focus ensemble on score-matrix level: blend Poisson and DC matrices with learned weights
- Logistic member dropped from the ensemble entirely
- Simpler, already partially implemented; may squeeze 0.005–0.01 log_loss on the 2014/2018 folds

### What NOT to try
- Don't revisit form features (Phase 3.1 settled that)
- Don't widen the validation slice further (N=4 showed no gain; N=5+ would overlap folds)
- Don't loosen the strict promotion bar

## Exact next steps for Phase 3.3

User must choose Option A, B, or C before work begins. Then:

**If Option A (external odds/xG):**
1. User approves specific data source (e.g., football-data.co.uk)
2. Write `data/download_odds.py` with pinned URL/version
3. Build odds feature in `features/odds.py` (pre-match closing line → implied prob)
4. Add to logistic feature set; re-run backtest; apply same gate

**If Option B (XGBoost member):**
1. `src/wcpredictor/features/advanced.py` — build richer time-series features
2. `src/wcpredictor/models/gradient_boost.py` — XGBoost/LightGBM wrapper
3. Swap logistic for gradient_boost in `backtest.py` ensemble block
4. Re-run backtest; apply same gate

**If Option C (score-matrix only):**
1. Remove logistic from `ensemble.py` weight fit; set `member_probs_val = [val_p_poi, val_p_dc2]`
2. Keep score-matrix blend as today; re-run backtest on 2-member ensemble
3. Apply same gate (now easier: only 2010 fold needs DC+Cal beat)

## Verification commands
```
uv run pytest -q                                        # must stay 71 passed
uv run python -m wcpredictor.evaluation.backtest        # baseline reference above
```

## Files changed in Phase 3.2
- `config.py`: temporarily edited N=2→3→4; reverted to N=2 (no net change)
- `HANDOFF.md`: this document (Phase 3.2 results + Phase 3.3 options)
- No source files modified; no new commits (Phase 3.1 commit stands)
