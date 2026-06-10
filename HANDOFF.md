# HANDOFF — AH bet-edge evaluation: model win-prob & EV per actual bookmaker offer

> Written 2026-06-10 (planning session). WC2026 kicks off 2026-06-11.
> User intent (verbatim direction): "I want to know the odds of winning the bet by taking
> on that specific asian handicap, which asian handicap has the best edge."
> i.e. NOT the model-only fair ladder we show today — compare the model's cover
> probability against the **actual prices bookmakers are quoting**, per line, and rank by EV.

## ⚠️ Pending commit — do this FIRST

The working tree contains the **completed and fully verified** h2h/1X2 integration from the
previous execution session (uncommitted): `download_odds_api.py`, `features/odds.py`,
`predict.py`, `api/app.py`, `api/schemas.py`, `index.html`, `tests/test_odds_api_h2h.py` (new),
`tests/fixtures/odds_api_h2h_sample.json` (new), `tests/test_api.py`, `README.md`.
Full suite: **287 passed, 1 skipped** (baseline was 270+1). `predict_match("Mexico","South
Africa","2026-06-11")` → `ensemble_mkt-0.1`, p_win 0.7147 (market blend active, 72 rows).
**Commit this as its own commit before starting the work below** (suggested message:
"Phase 10: 1X2 odds from the-odds-api + resilient /refresh-odds + fixtures AH columns").

## Goal

For every **actual bookmaker offer** (AH and Asian-totals) on a 2026 match, compute:
1. The model's probability of winning that specific bet (push/half-win aware).
2. The EV per unit staked at the quoted decimal price.
3. A ranking — which offer has the best edge — surfaced in the API + Predict tab.

This is a post-processing layer: no new model fitting, no change to predictions.

## Relevant context and constraints

- **The data is already on disk.** `data/raw/odds_api_wc2026.json` carries, per event,
  per bookmaker, full `spreads` and `totals` markets with both sides priced. Verified
  2026-06-10: Mexico–South Africa has 25 bookmakers, 7 of which quote spreads across
  3 distinct lines ({−1.0: 3 books, −1.25: 2, −1.5: 2}); e.g. Pinnacle quotes
  Mexico −1.25 @ 2.04 / South Africa +1.25 @ 1.87. The API's `point` on the home
  outcome IS the home AH line in standard notation; the away outcome's point is its
  negation. **The current parser `_parse_odds_api_json` keeps only the FIRST
  bookmaker's line and discards everything else** — that's the gap.
- **The settlement engine already exists and is exactly right.**
  `markets/asian.py::settle_line` handles whole/half/quarter lines with
  p_win/p_half_win/p_push/p_half_loss/p_loss and `fair_odds`. EV at quoted price `o`:
  `EV = o·(p_win + 0.5·p_half_win) + 0.5·p_half_win + p_push + 0.5·p_half_loss − 1`
  (equivalently `EV = (o − fair_odds)·(p_win + 0.5·p_half_win)`; EV=0 at o=fair_odds —
  use this as a property test).
- **Which matrix to evaluate against — decision needed, recommendation below.**
  The final score matrix is AH-blended (α≤0.3) toward the market. Evaluating offers
  against it is mildly circular (shrinks measured edges toward zero) but it is the
  gate-validated best estimate of true probabilities — shrinkage here is a feature
  (conservative edges), not a bug. **Recommendation: evaluate against the final
  (post-blend) matrix** — the same one `ladder()` already receives. The model-only
  alternative inflates edges by exactly the amount the blend was validated to correct.
- No new API calls: same JSON, refreshed by the existing `/refresh-odds`.
- `markets` in API schemas is `dict[str, Any]` — **zero schema changes** needed.
- Engineering rules: uv, src/ layout, tests, no exact-count assertions on live feed data.
- Local API on port 8001; hard-refresh (Cmd+Shift+R) to bust cached index.html.

## Files inspected

- `src/wcpredictor/markets/asian.py` — `settle_line` (push-aware settlement, the core),
  `asian_handicap(matrix, side, line)` (side='home'/'away', AH home notation),
  `asian_total(matrix, side, line)`, `ladder()` (~line 263) builds the current
  fair-odds-only report attached as `result["markets"]`.
- `src/wcpredictor/data/download_odds_api.py` — `_parse_odds_api_json` keeps first
  bookmaker only (lines ~124–132 post-h2h-merge); `parse_h2h_1x2` + `_is_tbd` (new from
  prior session) show the per-bookmaker iteration + placeholder-filtering pattern to reuse.
- `src/wcpredictor/predict.py` — `_build_frozen_state` (~line 685 onward) builds
  `ah_lookup` from `load_wc_ah_odds()` with away-perspective flipping — the pattern for
  the new offers lookup; `_predict_one_frozen` attaches `result["markets"] = _ah_ladder(mat)`
  after the AH blend (post-blend matrix in scope); `predict_match` has a parallel
  non-frozen path that also attaches markets.
- `src/wcpredictor/api/static/index.html` — `buildMarketsPanel` (~line 311) renders the
  AH/totals ladders on the Predict tab; `fmtLine`, `fmtOdds`, `coverPct` helpers exist.
- `src/wcpredictor/config.py` — `ASIAN_HANDICAP_LINES` (line 70), `ASIAN_TOTAL_LINES` (75).
- `data/raw/odds_api_wc2026.json` — structure verified live (see context above).

## Current findings

1. Everything needed exists except (a) a parser that keeps **all** offers, (b) an EV
   evaluator over offers, (c) wiring + UI. No new math beyond one EV formula.
2. Real spread coverage is thin but usable: ~2–7 books quote spreads per match, clustered
   within ±0.5 of the fair line. The "ladder of edges" is over offers that actually exist,
   not the configured line grid.
3. The away side of each spread is independently priced (not derived) — both sides of every
   offer should be evaluated; sometimes the dog side carries the edge.
4. Orientation: offers are stored home-oriented from the API event. A reversed lookup
   (user queries B vs A) must flip: AH line → −line, side home↔away; totals unchanged.
   Mirror `ah_lookup`'s away_entry construction in `_build_frozen_state`.
5. Played matches drop out of the feed (known shrinkage) — offers vanish with them;
   snapshot archive from the prior session preserves history.

## Proposed implementation plan

### Step 1 — `parse_market_offers()` in `data/download_odds_api.py`
`parse_market_offers(data: list[dict]) -> pd.DataFrame` with columns
`[year, date, team_a, team_b, market, line, side, price, bookmaker]`:
- Reuse the `parse_h2h_1x2` skeleton: `canonical()` both teams, skip if falsy or `_is_tbd`.
- Per bookmaker: for `spreads`, find the outcome matching raw `home_team` → emit
  `(market="ah", line=point, side="home", price)` and the away outcome →
  `(market="ah", line=point_home, side="away", price_away)` (store the HOME line on both
  rows so one number defines the bet; away side wins when away covers +line).
  Validate: both outcomes present, prices > 1.0, `abs(point*4 − round(point*4)) < 1e-6`
  (quarter-line grid; skip weird lines).
- For `totals`: Over/Under at `point` → two rows `(market="total", line=point,
  side="over"/"under", price)`.
- `year=2026`, `date = pd.Timestamp(commence_time[:10])`, keep `bookmaker = bookie["key"]`.
- No dedup across books (each offer is a distinct bet); dedupe exact duplicates only
  (`drop_duplicates(["team_a","team_b","market","line","side","bookmaker"], keep="last")`).

### Step 2 — EV evaluator: new module `src/wcpredictor/markets/edge.py`
Keep `asian.py` pure (matrix-only, no market data). New module:
- `ev_per_unit(settlement: dict, price: float) -> float` — the EV formula above.
- `evaluate_offers(matrix, offers: list[dict]) -> list[dict]`:
  for each offer dict `{market, line, side, price, bookmaker}`:
  - `market=="ah"`: `s = asian_handicap(matrix, side=side, line=line)` (asian.py's away
    side already settles "away covers +line" given the home-notation line — verify with a
    hand-computed test before trusting; if not, negate for away).
  - `market=="total"`: `s = asian_total(matrix, side=side, line=line)`.
  - Emit `{market, line, side, price, bookmaker, p_win, p_half_win, p_push, p_half_loss,
    p_loss, fair_odds, ev: round(ev,4), p_cover: p_win + p_half_win}`.
  - Sort descending by `ev`. Caller takes `[0]` as best.
- Pure functions, no I/O — trivially testable.

### Step 3 — Wiring in `predict.py`
- `_build_frozen_state`: next to the existing `ah_lookup` block, build
  `state["offers_lookup"]: dict[tuple[str,str], list[dict]]` from
  `parse_market_offers(json.loads(live_json.read_text()))` (lazy import, try/except like
  `ah_odds.py`; empty dict when JSON absent/corrupt). Register both orientations:
  forward = offers as parsed; reversed key = flipped copies (ah: `line=-line`,
  `side` home↔away; total: unchanged).
- `_predict_one_frozen`: after `result["markets"] = _ah_ladder(mat)` (mat is final,
  post-AH-blend), if `state["offers_lookup"]` has `(team_a, team_b)`:
  `result["markets"]["offers"] = evaluate_offers(mat, offers)` and
  `result["markets"]["best_offer"] = offers_evaluated[0]` (or None when list empty).
- `predict_match` (non-frozen ensemble path): same attach after its ladder call — factor a
  tiny helper `_attach_offers(markets_dict, matrix, offers)` used by both paths to avoid
  divergence (the exact bug class fixed last session).
- Snapshots: `predict_fixtures` uses the frozen path → regenerated fixture snapshots pick
  offers up automatically; no extra work.

### Step 4 — Frontend: "Market Offers" table in the Predict-tab Asian Markets panel
In `buildMarketsPanel` (index.html ~line 311), after the two ladder tables, when
`m.offers && m.offers.length`:
- Table: Book | Market | Line | Side | Price | Fair | Win % | EV %.
  `Win %` = `p_cover` (full+half win); `EV %` = `(r.ev*100).toFixed(1)`.
- Rows sorted as delivered (already EV-desc). Style `ev > 0` green / `ev < 0` muted red;
  bold the first row. Reuse `fmtLine`; price/fair via `.toFixed(2)`.
- A one-line caption: "Model edge vs quoted prices · positive EV = model disagrees with
  the market in your favor" — keep it factual, no staking advice.
- Fixtures tab: untouched (headline fair lines only, per prior session's Step 8).

### Step 5 — Tests: `tests/test_market_edge.py` (+ extend the existing h2h fixture)
Extend `tests/fixtures/odds_api_h2h_sample.json` event1 with 2–3 bookmakers' spreads
(different lines, e.g. −0.75 and −1.0) and one totals market; keep existing tests green
(they ignore spreads).
- Parser: row count/columns; home AND away rows share the home line; prices match raw;
  placeholder event skipped; totals-only event yields total rows only; non-quarter
  line skipped.
- Settlement orientation: hand-build a tiny matrix (e.g. 3×3 with known diff dist),
  hand-compute EV for a quarter line on both sides; assert `evaluate_offers` matches.
- Property: for any offer, `ev ≈ (price − fair_odds)·(p_win + 0.5·p_half_win)` and
  `ev == 0` at `price = fair_odds` (within 1e-6, fair_odds finite).
- Flip symmetry: reversed orientation entry (line negated, sides swapped) gives the same
  EV for the same physical bet.
- Sorted: `evaluate_offers` output is non-increasing in `ev`; `best_offer == offers[0]`.
- Integration (skipif live JSON absent): frozen-state predict for a 2026 pair returns
  `markets["offers"]` non-empty with finite EVs; never assert offer counts.

### Sequencing
Commit pending work → Step 1 → 2 → 5 (parser+evaluator tests) → 3 → 4 → full suite →
snapshot regen + browser check.

## Exact next steps

1. `git add`-and-commit the pending h2h work (see "Pending commit" above) on
   `ops/pre-tournament-readiness`.
2. Implement Steps 1–5 in order; run `tests/test_market_edge.py` after Step 2.
3. Full suite (`uv run pytest -q`) — expect 287+new passed, 1 skipped.
4. Regenerate fixture snapshots, restart uvicorn :8001, hard-refresh; Predict tab
   (e.g. Mexico vs South Africa) shows the offers table with Pinnacle −1.25 @ 2.04 ranked.
5. Commit; PR to `main`; merge before kickoff.

## Verification/test commands

```bash
uv run pytest tests/test_market_edge.py tests/test_asian_handicap.py tests/test_odds_api_h2h.py -q
uv run pytest -q
uv run python -c "
import json
from wcpredictor.data.download_odds_api import parse_market_offers
offers = parse_market_offers(json.loads(open('data/raw/odds_api_wc2026.json').read()))
print(offers[(offers.team_a=='Mexico')])"
# Then: predict Mexico vs South Africa via API and eyeball markets.offers —
# Pinnacle home −1.25 @ 2.04 should show EV ≈ 2.04·(p_w+0.5·p_hw)+0.5·p_hw+p_push+0.5·p_hl − 1
```

## Risks and open questions

- **Open (decision made, revisitable): matrix choice.** Plan uses the post-blend matrix
  (conservative, gate-validated). If the user wants raw model edges, add a
  `pre_blend: bool` later — do NOT silently switch.
- EV numbers inherit model error; with 2–7 books per match a "best edge" always exists
  even when no offer is genuinely +EV. The UI copy must present EV as model opinion, not
  advice. Do not add stake sizing / Kelly without being asked.
- the-odds-api `spreads`/`totals` settlement period assumed 90-min (same caveat as Phase 9;
  near-universal convention, not doc-verified).
- Feed shrinkage: offers for played matches vanish on refresh — never assert counts.
- Away-side settlement in `asian_handicap(side="away", line=home_line)`: verify the sign
  convention with the hand-computed test BEFORE wiring (cheap to check, expensive to get
  wrong silently).

## What not to repeat / failed approaches

- Do **not** re-litigate `odds_alpha`/`ah_alpha`/model selection (gate-validated).
- Do **not** duplicate evaluation logic between `predict_match` and `_predict_one_frozen`
  — use one shared helper (last session fixed exactly this divergence for 1X2 odds).
- Do **not** assert exact row/offer counts against the live feed.
- Do **not** SHA-pin the live JSON.
- The fair-odds ladder (`ladder()`) is NOT what the user asked for here — it shows model
  fair prices on a configured grid; this feature compares against real quoted prices.
  Keep both; don't replace the ladder.
