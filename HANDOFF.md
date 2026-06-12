# HANDOFF — Phase 13 (money-first revision): +EV betting vs Singapore Pools

> Planned 2026-06-12. Supersedes the earlier Phase 13 draft (generic EV bet finder).
> Phases 1–12 complete and merged. WC2026 is live (kicked off 2026-06-11).
> ULTIMATE AIM (user-stated): make money. Counterparty: **Singapore Pools only**.

## User decisions (locked in 2026-06-12, do not re-ask)

1. **Books:** Singapore Pools only. No line shopping. the-odds-api books are
   unreachable for betting — their role is consensus benchmark + model blend input.
2. **Staking:** quarter-Kelly, capped (~2% bankroll per bet), one bet per match.
3. **Scope:** match markets only (1X2, AH, totals). No outrights this phase.
4. **If the honest backtest shows no credible edge:** bet anyway at minimum stakes
   ("paid experiment"), judge by CLV after ~30 bets. So the backtest sets the
   SIZING TIER (credible edge → quarter-Kelly; inconclusive → minimum stakes),
   it does not gate betting entirely.
5. **SG Pools settlement is 90-minute** (user-confirmed 2026-06-12) — matches the
   model's label convention; no market exclusions needed. Still verify the AH line
   menu and min/max stakes when first entering prices.
6. **Bankroll: $5,000.** Quarter-Kelly with 2% cap → max $100/bet; 6% daily
   exposure cap → max $300/day staked. Minimum-stake tier for the no-edge case:
   configurable, default $10/bet.
7. **Automated SG Pools fetcher: APPROVED** (explicit user approval 2026-06-12 —
   satisfies the CLAUDE.md no-silent-scraping rule). Manual CSV entry remains the
   fallback when the fetcher breaks.
8. **ODDS_API_KEY: CONFIGURED + VALIDATED** (2026-06-12) — exported in ~/.zshrc,
   HTTP 200 against /v4/sports, 497/500 free-tier credits remaining,
   `soccer_fifa_world_cup` available. Uvicorn must be (re)started from a fresh
   shell to inherit it. Budget: ~2-3 consensus refreshes/day fits the free tier;
   no polling loops.

## Goal

A daily pipeline that: ingests Singapore Pools 1X2/AH/total prices, computes EV
against model fair prices (ensemble_mkt) cross-checked against the de-margined
25-book consensus, recommends stakes (quarter-Kelly capped), records every bet in
a ledger, settles from the Phase 12 results store, and tracks CLV and P&L with
pre-committed stop rules.

## Relevant context and constraints

- Default model stays `ensemble_mkt`, α caps 0.3 (gate-validated; do not re-litigate).
- **Edge logic for a single soft book:** value = SG Pools price > fair odds, where
  fair odds come from (a) the model and (b) the de-margined consensus of the 25
  the-odds-api books. Recommend requiring BOTH: `EV_model > τ` AND
  `price > consensus_fair` — the double-confirmation ("beat the sharp consensus")
  filter is the credible part; the model adds selectivity.
- **No silent scraping** (CLAUDE.md). Singapore Pools is NOT in the-odds-api.
  MVP ingestion = manual entry. An automated fetcher from
  online.singaporepools.com.sg requires explicit user approval first (open question).
- **Existing profit signals are NOT money-grade (audited this session):**
  - `metrics.ah_roi` reports +41%/+14%/+48%/+11% across the 2010–2022 folds, but it
    settles with `_settle_outcome` (coarse: push returns 0.5 units instead of 1.0;
    quarter-lines settled as full win/loss — its own docstring defers real settlement
    to `markets/asian.py`), bets the HOME side only, at average odds, threshold 0.
  - The report's `clv` is NOT closing-line value: `closing_line_value()` is the mean
    gap between model fair AH line and market line = +0.25…+0.47 GOALS every fold.
    The model is systematically more bullish on the home side than the market —
    as likely a bias (e.g. neutral-venue home designation, favorite skew) as an edge.
    M2 must diagnose this before money rides on model-disagreement bets.
  - Head-on, the market predicted AH covers better than the model in 3 of 4 WCs
    (`ah_cover_brier_market` < `_model` in 2014/2018/2022). Any real edge is selective.
- **Variance expectations (state in UI/docs):** ~90 matches remain; a true 3% ROI at
  flat stakes over ~80 bets ≈ +2.4u expected, σ ≈ 9u. One tournament is mostly
  variance; the deliverable is a disciplined process, not guaranteed profit.
- **Ops blockers:** no ODDS_API_KEY on this machine — cached odds JSON is a
  pre-kickoff snapshot. Fresh consensus is required for both the blend and the
  benchmark. Leakage rules: never scan/bet fixtures already kicked off.
- **Verify Singapore Pools rules before first bet (M0 checklist):** 90-minute
  settlement for 1X2/AH/totals incl. knockout matches (model is 90'-only), available
  AH line menu (quarter lines?), min/max stakes, decimal odds format.
- 90' label convention everywhere. Rolling Elo / pinned fits unchanged (Phase 12).

## Files inspected

- `src/wcpredictor/markets/edge.py` — `ev_per_unit`, `evaluate_offers` (:43 skips
  non-ah/total markets; 1X2 never evaluated). Settlement math here is correct.
- `src/wcpredictor/markets/asian.py` — correct quarter-line AH/total settlement from
  the score matrix; this is what the honest backtest must use.
- `src/wcpredictor/evaluation/metrics.py` — `_settle_outcome` (:85, coarse),
  `ah_roi` (:170, flawed as money metric per above), `closing_line_value` (line-gap,
  misleadingly named).
- `src/wcpredictor/evaluation/backtest.py` — fold machinery; `clv` row :645;
  per-match artifacts lack raw probabilities (probability dump needed).
- `src/wcpredictor/data/download_odds_api.py` — `parse_market_offers` (:178, spreads
  + totals only), `parse_h2h_1x2` (consensus, median across books, normalised).
- `src/wcpredictor/predict.py` — `_load_offers_lookup` (:316), `_attach_offers` (:350).
- `src/wcpredictor/features/odds.py` — `load_wc_odds`: 1X2 history 2010 (betexplorer)
  + fdco 2014/18/22 (H-Avg/D-Avg/A-Avg — margin-normalised probs; the EV backtest
  needs RAW average odds read directly).
- `src/wcpredictor/api/app.py` — existing endpoints incl. /refresh-odds,
  /refresh-results, /resimulate; lock pattern to copy for new POST endpoints.
- Data: `wc_ah_odds.csv` (256 rows 2010–22), `wc2010_odds.csv`,
  `odds_api_wc2026.json` (72 events, 25 books, h2h/spreads/totals; pre-kickoff),
  `data/processed/backtest_report.json` (per-fold AH ROI/CLV figures quoted above).

## Proposed implementation plan

### M0 — Ops prerequisites (mostly resolved 2026-06-12)
1. ~~Obtain ODDS_API_KEY~~ → DONE: exported in ~/.zshrc and validated (decision #8).
   First execution-session action: restart uvicorn from a fresh shell, then
   POST /refresh-odds to replace the stale pre-kickoff consensus snapshot.
2. ~~Verify settlement~~ → 90-minute confirmed by user. Residual check at first
   price entry: AH line menu granularity and min/max stakes.
3. ~~Bankroll~~ → $5,000 (decision #6).

### M1 — 1X2 evaluation plumbing (small, from previous draft)
- `parse_market_offers`: add `h2h` branch → market="1x2", side ∈ {home,draw,away},
  price, bookmaker; skip `h2h_lay`. (Used for consensus fair price per outcome, and
  to keep the existing offers table working.)
- `markets/edge.py`: `outcome_prob(matrix, side)` (home: Σ i>j; draw: Σ diag; away:
  Σ j<i... j>i); `evaluate_offers` handles "1x2" via `ev_per_unit` with
  p_push=p_half=0; fair_odds=1/p.
- `_load_offers_lookup` reversal: 1x2 swaps home↔away (draw unchanged).
- NEW `consensus_fair(offers, market, line, side)` — de-margined median consensus
  price for an outcome across books (normalise 1/odds within each book's market,
  then median): the benchmark SG Pools prices are compared against.

### M2 — Honest EV backtest (decides τ and sizing tier)
1. backtest.py: optional per-match dump `backtest_permatch_probs.csv` — fold, teams,
   label, goals, ens_mkt p_win/p_draw/p_loss, blended-matrix AH/total cover probs for
   the matched `wc_ah_odds.csv` line, RAW average odds (fdco H/D/A-Avg, wc2010,
   betexplorer AH/OU prices).
2. NEW `evaluation/ev_backtest.py` (pure consumer): for market ∈ {1x2, ah, total} ×
   τ ∈ {0, 0.02, 0.05, 0.10}: bet BOTH sides where EV > τ, settle with
   `markets/asian.py` semantics (and exact 1X2), flat 1u; report n_bets, ROI,
   paired-bootstrap 95% CI per market×τ; per-fold breakdown.
3. **Margin haircut sensitivity:** SG Pools prices are softer than EU averages —
   re-run the grid with prices shaved 2%/5%. Edge must survive the haircut to
   justify quarter-Kelly tier; otherwise minimum-stake tier (user decision #4).
4. **Diagnose the home-bullish bias:** decompose ROI by home/away side and by
   favorite/dog; check whether the +0.25..0.47 line gap comes from neutral-venue
   home designation or favorite skew. If one-sided, consider a bias correction or
   restrict bets to the unbiased direction. Document verdict here.
5. Replace/rename misleading metrics: `clv` → `fair_line_gap`; keep old keys out of
   the report or mark deprecated (additive change, don't break tests gratuitously).
6. Output `data/processed/ev_backtest_report.json` + verdict in HANDOFF + memory:
   chosen markets, τ, sizing tier.

### M3 — Singapore Pools price ingestion
- NEW `data/raw/sgpools_offers.csv` (manual-entry MVP):
  `entered_at, date, team_a, team_b, market, line, side, price`
  (canonical team names via `normalize_teams.canonical`; quarter-line validation;
  market ∈ {1x2, ah, total}).
- NEW `data/sgpools.py`: `load_sgpools_offers()` (validate + canonicalize),
  `add_offer(...)` helper; CLI `python -m wcpredictor.data.sgpools add ...` and/or
  POST /sgpools-offers for quick entry from the UI.
- **Automated fetcher (APPROVED, decision #7)** — `fetch_sgpools_offers()`:
  1. Execution session must first probe online.singaporepools.com.sg for the JSON
     endpoints behind the football odds pages (browser devtools / curl; the odds
     UI is JS-driven, so expect an XHR API — do NOT assume HTML parsing works).
  2. Polite fetching: one fetch per refresh action (no polling loops), realistic
     User-Agent, timeouts, and snapshot archiving to
     `data/raw/sgpools_snapshots/<timestamp>.json` (mirrors the odds-api pattern).
  3. Parse → same canonical offers schema; merge into `sgpools_offers.csv` with
     `source ∈ {fetched, manual}`; manual rows win on conflict (mirrors the
     Phase 12 results-store precedence pattern).
  4. Wire into POST /refresh-sgpools (lock pattern from /refresh-odds); non-fatal
     on network/parse failure — manual entry is the fallback, never a hard error.
  5. Tests use saved fixture responses only; no network in the suite.
  Risk: site structure unknown until probed; if the endpoint is hostile
  (auth/captcha), fall back to manual entry and note it here.

### M4 — Value scan vs Singapore Pools
- NEW `markets/value_scan.py`: for each unplayed fixture with SG Pools offers:
  model matrix (frozen state) → EV_model per offer; consensus fair price per offer
  (M1 helper; nearest-line match for AH/totals, else model-only with a flag);
  recommend when `EV_model ≥ τ` AND `price ≥ consensus_fair` (configurable).
  Output: date, match, market, line, side, sgpools price, fair_model, fair_consensus,
  ev_model, ev_consensus, recommended stake (M5), confidence flags.
- `GET /value-bets` + UI "Value Bets" tab: consensus-age + offers-age staleness
  banners; caveat notes (self-reference: ensemble_mkt already blends consensus α=0.3,
  so EV_model is conservative).
- CLI `python -m wcpredictor.markets.value_scan --min-ev <τ>`.

### M5 — Money discipline layer
- `config.py`: `BANKROLL=5000.0` (dollars), `KELLY_FRACTION=0.25`,
  `MAX_STAKE_PCT=0.02` (→ $100/bet), `MAX_DAILY_EXPOSURE_PCT=0.06` (→ $300/day),
  `MIN_TIER_STAKE=10.0` (no-edge experiment mode). Stakes in dollars throughout.
- Stake calc: quarter-Kelly on EV_model (b = price−1, f = ¼·(bp−q)/b), capped;
  ONE bet per match — highest-EV offer among correlated 1X2/AH/total on same game.
- NEW bet ledger `data/bets.csv`: `placed_at, date, match, market, line, side,
  price_taken, stake, status(open/won/half_won/push/half_lost/lost), pnl,
  consensus_fair_at_placement, closing_consensus_fair (filled later), clv_pct`.
  POST /bets to record; settlement job reads the Phase 12 results store
  (`load_wc2026_results`) and `markets/asian.py` settlement → fills status/pnl.
- **Real CLV:** on each /refresh-odds nearest kickoff, store closing consensus fair
  price per logged bet; `clv_pct = price_taken / closing_consensus_fair − 1`.
- **Stop rules (pre-committed):** pause betting if trailing-30-bet mean CLV < 0,
  or drawdown > 15u. Surface both on the scorecard/UI. P&L + CLV summary endpoint
  (`GET /ledger`) and UI panel.

### M6 — Tests (house patterns: synthetic data, tmp_path, monkeypatched loaders)
- M1: h2h parsing (3 sides, h2h_lay skipped), 1x2 EV hand-checked, flip symmetry,
  consensus de-margining math.
- M2: settlement correctness (1X2; AH full/half/push on known margins), threshold
  grid, haircut, bootstrap seed determinism, dump columns.
- M3: sgpools CSV validation, canonical names, bad-line rejection.
- M4: played-fixture exclusion (leakage), double-confirmation filter, nearest-line
  consensus matching, empty-offers graceful.
- M5: Kelly math + caps, one-bet-per-match, ledger settlement vs results store,
  CLV computation, stop-rule triggers.
- API: /value-bets, /sgpools-offers, /bets, /ledger (monkeypatched, no network).

## Daily loop during the tournament (document in README)

Morning: POST /refresh-results → /resimulate → /refresh-odds (consensus) →
enter today's SG Pools prices → GET /value-bets → review shortlist.
Pre-kickoff: re-enter SG Pools prices for shortlist (lines move), re-scan, place
bets, POST /bets with price/stake taken. Near kickoff: /refresh-odds for closing
consensus (CLV). Next morning: settlement + CLV + stop-rule check on /ledger.

## Exact next steps (execution session)

1. `git checkout -b feat/phase13-sgpools-ev`
2. M1 + tests → M2 (run backtest + ev_backtest; **record verdict + chosen τ + sizing
   tier in this file and memory**) → M3 + tests → M4 + tests → M5 + tests → full suite.
3. First actions: restart uvicorn from a fresh shell (inherits the key) and
   POST /refresh-odds for a post-kickoff consensus snapshot.
4. Early in execution: probe the SG Pools site for its odds API (M3 fetcher) so the
   schema is known before building the scan around it.
4. PR to main; restart uvicorn (port 8001).

## Verification / test commands

```bash
uv run pytest tests/test_market_edge.py tests/test_ev_backtest.py \
  tests/test_sgpools.py tests/test_value_scan.py tests/test_ledger.py -q
uv run pytest -q     # full suite; baseline 329 passed / 1 skipped + new
uv run python -m wcpredictor.evaluation.backtest      # + probs dump
uv run python -m wcpredictor.evaluation.ev_backtest   # ROI grid + report JSON
uv run python -m wcpredictor.markets.value_scan --min-ev 0.0
# API: GET /value-bets, POST /sgpools-offers, POST /bets, GET /ledger; UI tabs render.
```

## Risks and open questions

- **SG Pools margin may eat the edge** — their overround is fat; the consensus
  double-confirmation filter + haircut sensitivity are the mitigations. If nothing
  clears the bar at real SG Pools prices, the system correctly recommends ~no bets;
  minimum-stake experiment tier is the user-chosen floor.
- **Model home-bullish bias** (+0.25..0.47 goal line gap every fold) must be
  diagnosed in M2 before model-disagreement bets get real stakes.
- **SG Pools site structure unknown** until probed — fetcher may need a JS-driven
  XHR endpoint; if hostile (auth/captcha), manual entry is the working fallback.
- Settlement confirmed 90' (user); residual: AH line menu + min/max stakes at first entry.
- Tiny historical sample (~250 priced matches) → wide CIs; "inconclusive" likely;
  sizing-tier design absorbs this honestly.
- Historical backtest uses EU average odds, not SG Pools quotes — directionally
  informative only; the haircut runs bound the gap.
- KO-stage TBD events: `_is_tbd` handles; scan must tolerate zero-offer fixtures.

## What not to repeat / failed approaches

- Do NOT trust `metrics.ah_roi` (+41/+14/+48/+11%) or report `clv` as money
  evidence — coarse settlement, home-side-only, line-gap-not-CLV (audited 2026-06-12).
- Do NOT settle bets with `_settle_outcome` — use `markets/asian.py` semantics.
- Do NOT use `load_wc_odds` normalised probs as historical prices (margin removed);
  read raw average odds.
- SG Pools fetching is APPROVED (2026-06-12) but keep it polite: one fetch per
  user-triggered refresh, no polling loops, archive snapshots.
- Do NOT line-shop the-odds-api books — user cannot bet them; benchmark only.
- Do NOT re-litigate ensemble_mkt, α=0.3 caps, or the AH promotion gate.
- Do NOT stack correlated bets (1X2 + AH + total on the same match) — one per match.
- Do NOT let the scan/ledger touch fixtures already kicked off.
