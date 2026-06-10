# HANDOFF — Phase 11 complete: bookmaker offer EV ranking

> Written 2026-06-10 (execution session). Phase 11 implemented and verified.
> WC2026 kicks off 2026-06-11.

## Status: DONE — ready for final commit + PR merge to main

Phase 11 is fully implemented, all tests pass, UI verified in browser.
Commit the work below and merge `ops/pre-tournament-readiness` → `main` before kickoff.

## What was built

For every bookmaker offer (AH and totals) on a 2026 match, the system now:
1. Computes the model's probability of winning that specific bet (push/half-win aware).
2. Computes EV per unit staked at the quoted decimal price.
3. Ranks offers by EV and surfaces them in the API + Predict tab.

This is a pure post-processing layer: no new model fitting, no change to predictions.

## Files changed in this session

- `src/wcpredictor/data/download_odds_api.py` — added `parse_market_offers()`
- `src/wcpredictor/markets/edge.py` — NEW: `ev_per_unit()`, `evaluate_offers()`
- `src/wcpredictor/predict.py` — added `_load_offers_lookup()`, `_attach_offers()`;
  wired into `_build_frozen_state`, `_predict_one_frozen`, `predict_match`
- `src/wcpredictor/api/static/index.html` — "Market Offers" table in `buildMarketsPanel`
- `tests/fixtures/odds_api_h2h_sample.json` — extended with spreads + totals for event1/event2
- `tests/test_market_edge.py` — NEW: 15 tests covering parser, EV property, flip symmetry,
  sorted output, integration

## Verification results

- All 15 new tests pass.
- Full suite: 302 passed, 1 skipped (baseline 287+1 + 15 new).
- Live API `predict("Mexico","South Africa")` → 42 offers, best = betanysports total 2.5 over @ +5.9% EV.
- Browser UI renders "Market Offers" table with Book/Market/Price/Fair/Win%/EV% columns,
  EV-sorted, positive EV green / negative muted, disclaimer present.
- Pinnacle AH −1.25 home @ 2.04 shows EV +0.7% as expected from pre-planning.

## Commit command

```bash
git add src/wcpredictor/data/download_odds_api.py \
        src/wcpredictor/markets/edge.py \
        src/wcpredictor/predict.py \
        src/wcpredictor/api/static/index.html \
        tests/fixtures/odds_api_h2h_sample.json \
        tests/test_market_edge.py \
        HANDOFF.md
git commit -m "Phase 11: bookmaker offer EV ranking (parse_market_offers + edge.py + UI)"
```

Then:
```bash
git push origin ops/pre-tournament-readiness
gh pr create --title "Phase 11: bookmaker AH/totals offer EV ranking" --body "..."
```

## Design decisions (do not re-litigate)

- Post-blend matrix used for EV evaluation (conservative, gate-validated; see Phase 8/9).
- `edge.py` is pure functions, no I/O.
- Both `_predict_one_frozen` and `predict_match` use the shared `_attach_offers` helper
  to avoid divergence (lesson from Phase 10).
- Reversed lookup orientation: line negated + sides swapped + evaluated on the transposed
  matrix (home/away swap) — verified via flip symmetry test.
- No stake sizing / Kelly — do not add without being asked.

## What NOT to do next

- Do not re-litigate model selection (ensemble_mkt, α≤0.3).
- Do not add Kelly criterion or stake sizing without user request.
- Do not assert exact offer counts in tests (feed shrinks as matches play).
- Do not change `markets: dict[str, Any]` schema — it's already correct.
