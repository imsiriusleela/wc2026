# HANDOFF — Phase 3.3 CLOSED (deferred) → Phase 3.4 planning

## Phase 3.3 Result: DEFERRED — default stays `"poisson"`

### What was built

Phase 3.3 implemented the full market-odds pipeline:
- `src/wcpredictor/data/download_odds.py` — downloads `WorldCup_fdco.xlsx` from
  football-data.co.uk, verifies SHA-256, caches to `data/raw/`
- `src/wcpredictor/features/odds.py` — parses 2014/2018/2022 WC sheets, normalises
  bookmaker margin → implied probabilities, aligns to test rows by (team_a, team_b)
- `src/wcpredictor/data/normalize_teams.py` — added "Bosnia & Herzegovina" alias
- `src/wcpredictor/config.py` — added `FDCO_ODDS_URL`, `FDCO_ODDS_SHA256`,
  `ODDS_ALPHA_PRIOR=0.0`
- `src/wcpredictor/evaluation/backtest.py` — added `model_ensemble_market` column via
  time-aware α blending (`_fit_odds_alpha` using scipy bounded minimiser)
- `tests/test_odds.py` — 11 tests (SHA pin, shape, prob sums, name canonicalisation,
  alignment coverage, 2010-returns-None guard)
- 82 tests pass total (71 existing + 11 new)

### Experiment results

**Market-only log_loss (market-implied probs, no model):**
- WC2014: 0.9546 → market WORSE than our Ens+Cal (0.9172) — unusual, likely
  H-Avg represents a less-than-closing market position or 2014 WC sampling variance.
- WC2018: 0.9410 → market BETTER than Ens+Cal (0.9693) — significant edge.
- WC2022: 1.0120 → market BETTER than Ens+Cal (1.0634) — significant edge.
- WC2010: NOT AVAILABLE (absent from football-data.co.uk file).

**Time-aware blending results (α fit on past folds, α_prior=0.0):**

| Fold | Ens+Cal | Ens+Mkt | α_used | Poisson | DC+Cal | Verdict |
|------|---------|---------|--------|---------|--------|---------|
| 2010 | 0.9660 | — (no odds) | — | 0.9828 | **0.9474** | FAIL |
| 2014 | **0.9172** | 0.9172 | 0.000 | 0.9152 | 0.9376 | FAIL (0.002 gap) |
| 2018 | 0.9693 | 0.9693 | 0.000 | **0.9690** | 0.9904 | FAIL (marginal) |
| 2022 | 1.0634 | **1.0238** | 0.440 | 1.0541 | 1.1105 | PASS |

**Why α=0.000 for 2014 and 2018:**
- α_prior=0.0 (conservative: no market weight without evidence) → 2014 Ens+Mkt=Ens+Cal.
- Optimizer for 2018 fits on 2014 data, where market underperformed (0.9546 > 0.9172) → α=0.
- Optimizer for 2022 fits on 2014+2018 combined (128 matches): the strong 2018 market
  signal (0.9410 vs 0.9693) outweighs the weak 2014 market, giving α=0.440.

### Gate decision

Strict bar: Ens+Mkt ≤ best single on ALL FOUR FOLDS. Fails on:
1. **2010** — no odds available; Ens+Cal 0.9660 > DC+Cal 0.9474 (structural data gap)
2. **2014** — Ens+Mkt=0.9172 > Poisson 0.9152 by 0.002
3. **2018** — Ens+Mkt=0.9693 > Poisson 0.9690 by 0.0003 (marginal, but still a fail)

Verdict: **DEFER.** Default remains `"poisson"`. 82 tests green.

### Root-cause diagnosis (for Phase 3.4)

**The 2010 fold is the structural blocker.** No odds source was found for WC2010 in any
freely accessible static file. The football-data.co.uk WC xlsx starts at 2014.

**The 2014/2018 failures are small gaps** driven by:
- 2014: Elo-based Poisson is strong there (best across all folds); market underperforms
  (opening/average odds may be less calibrated than closing). Gap = 0.002.
- 2018: The 3-member ensemble narrowly loses to Poisson (gap = 0.0003); market α=0
  because 2014 data poisoned the prior.

**If WC2010 odds were available:**
- They would inform α_prior for 2014 using data rather than a constant.
- If market was good in 2010, α>0 might have been applied for 2014 → smaller gap.
- Better α estimates for all subsequent folds.
- The 2010 fold itself would be improvable with market blending.

## Phase 3.4 planning: WC2010 odds

### Option A (recommended): betexplorer.com for WC2010 odds
- betexplorer.com has historical 1X2 odds for WC2010 matches (all 64).
- Covers all group and knockout matches with pre-match closing odds.
- **Blocker:** requires explicit approval (no-silent-scraping rule).
- **Implementation:** pinned snapshot approach — fetch once, hash, save to data/raw/.

### Option B: Historical Pinnacle/Betfair odds from researcher datasets
- Some researchers (e.g., Dixon et al., Forrest et al.) have published WC odds datasets
  as paper supplementary data or public GitHub repos.
- Search: "World Cup 2010 betting odds dataset csv github"
- If found as a static CSV with a pinned URL/hash, no scraping needed.

### Option C: Use market odds only for 2018/2022 (drop 2010/2014 from promotion gate)
- Revise the bar to: "ensemble must beat best single on the **three folds with market data**"
  and separately benchmark 2010 as a reference.
- **BUT:** the HANDOFF explicitly says "Do not loosen the bar." Do not pursue unless
  user approves a bar revision.

### Option D: Improve the 3-member ensemble for 2010 without odds
- Investigate WHY DC+Cal is so strong in 2010 (0.9474 vs Ens+Cal 0.9660).
- Hypothesis: WC2010 had many 1-0 / 0-0 results; DC's low-score adjustment (ρ) is key.
- Fix: give DC higher weight in the ensemble by solving the member-correlation problem.
- This is the "XGBoost member" path (Option B from Phase 3.2 HANDOFF).

## Exact next steps for Phase 3.4

**If Option A (betexplorer):**
1. User approves betexplorer.com as data source.
2. Write `data/download_wc2010_odds.py`: fetch the WC2010 page, parse all 64 matches,
   save as `data/raw/wc2010_odds.csv` with SHA pin.
3. Update `features/odds.py`: extend `load_wc_odds()` to read the CSV if present.
4. Re-run backtest; apply gate.

**If Option B (researcher dataset):**
1. Search for static CSV with WC2010 odds (GitHub or paper supplementary material).
2. If found: download, verify, pin SHA. Extend `features/odds.py`.
3. Same as Option A step 4.

**If Option D (XGBoost member):**
1. `src/wcpredictor/features/advanced.py` — rolling Elo trajectory, recent scoring rate,
   team-vs-opponent strength index.
2. `src/wcpredictor/models/gradient_boost.py` — XGBoost wrapper, same interface as logistic.
3. Swap logistic for gradient_boost in the ensemble block of `backtest.py`.
4. Re-run; apply gate.

## Verification commands
```
uv run pytest -q                                        # must stay 82 passed
uv run python -m wcpredictor.evaluation.backtest        # baseline reference above
```

## Files changed in Phase 3.3
- `src/wcpredictor/config.py` — FDCO_ODDS_URL, FDCO_ODDS_SHA256, ODDS_ALPHA_PRIOR
- `src/wcpredictor/data/normalize_teams.py` — added "Bosnia & Herzegovina" alias
- `src/wcpredictor/data/download_odds.py` — NEW: download + SHA verify xlsx
- `src/wcpredictor/features/odds.py` — NEW: parse xlsx → implied probs, align to test rows
- `src/wcpredictor/evaluation/backtest.py` — time-aware α blending, Ens+Mkt column
- `tests/test_odds.py` — NEW: 11 tests
- `data/raw/WorldCup_fdco.xlsx` — NEW: cached odds file (not in git if .gitignored)
- `data/processed/backtest_report.json` — updated with Ens+Mkt results
