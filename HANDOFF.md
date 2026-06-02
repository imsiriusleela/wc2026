# HANDOFF — Phase 5.0: Tournament simulator ✅ COMPLETE (2026-06-02)

> Planned 2026-06-02. Phases 3.9 (`da624f6`) and 4.0 `ensemble_mkt` (`7ed152e`) committed;
> working tree clean. This phase builds the tournament simulator — the main unblocked
> deliverable (item #8 in CLAUDE.md). Needs no market odds and no match results.

## Goal

Monte-Carlo the full 48-team / 12-group / knockout tournament from pre-match model
probabilities and emit **per-team round-reach probabilities**:
P(win group), P(runner-up), P(reach R32 / R16 / QF / SF / Final), P(champion).

Reuses the existing frozen-state prediction path — no model refits, no new model members.

## Decisions locked with the user
- **Build the simulator now** (vs. operational-only prep).
- **Official bracket + group labels** fidelity (vs. inferred/seeded). The R32→Final pairing
  template is the real FIFA 2026 structure (sourced below).

## Relevant context and constraints
- Today 2026-06-02; group stage 06-11→06-27, knockouts 06-28→07-19. Simulator is pre-tournament.
- Reuse, do not reinvent: `_build_frozen_state(as_of, models)` and
  `_predict_one_frozen(state, model, team_a, team_b, neutral)` in
  `src/wcpredictor/predict.py` (latter returns `p_win/p_draw/p_loss` **and** `score_matrix`).
- `data/raw/wc2026_fixtures.csv` = 72 group matches; clusters into exactly **12 groups of 4**
  (verified). No group-label column.
- Knockout draws: 30 min extra time, then penalties (per FIFA). CLAUDE.md keeps ET/penalties a
  separate module — here we use a **simple, clearly-flagged in-simulator resolver**, not a
  trained ET model.
- Comply with "no hard-coded 2026 teams": the bracket template and third-place table are keyed
  by **group position** (1A, 2B, 3X), never by team name. Group→label and group→teams are
  derived from the fixtures, not hard-coded.

## Findings — grounded tournament structure

### Group labels (A–L)
Inferred deterministically: cluster fixtures into 12 groups (connected components), assign
labels by **schedule first-appearance order**. This yields Mexico→A, which matches the official
template (match 79 `1A` is played in Mexico City). Resulting map (validate in code, allow an
optional explicit `group` column in the fixtures CSV to override):
A=[Czech Rep, Mexico, South Africa, South Korea], B=[Bosnia, Canada, Qatar, Switzerland],
C=[Australia, Paraguay, Turkey, USA], D=[Brazil, Haiti, Morocco, Scotland],
E=[Curaçao, Ecuador, Germany, Ivory Coast], F=[Japan, Netherlands, Sweden, Tunisia],
G=[Cape Verde, Saudi Arabia, Spain, Uruguay], H=[Belgium, Egypt, Iran, New Zealand],
I=[France, Iraq, Norway, Senegal], J=[Algeria, Argentina, Austria, Jordan],
K=[Colombia, DR Congo, Portugal, Uzbekistan], L=[Croatia, England, Ghana, Panama].

### Official Round of 32 template (match → pairing; `3X` = a qualifying third-placed team)
73:2A-2B  74:1E-3[A/B/C/D/F]  75:1F-2C  76:1C-2F  77:1I-3[C/D/F/G/H]  78:2E-2I
79:1A-3[C/E/F/H/I]  80:1L-3[E/H/I/J/K]  81:1D-3[B/E/F/I/J]  82:1G-3[A/E/H/I/J]
83:2K-2L  84:1H-2J  85:1B-3[E/F/G/I/J]  86:1J-2H  87:1K-3[D/E/I/J/L]  88:2D-2G

### Knockout progression
R16: 89=W73/W75, 90=W74/W77, 91=W76/W78, 92=W79/W80, 93=W83/W84, 94=W81/W82,
95=W86/W88, 96=W85/W87.
QF: 97=W89/W90, 98=W93/W94, 99=W91/W92, 100=W95/W96.
SF: 101=W97/W98, 102=W99/W100.  Final: W101/W102.

### Best-third-placed allocation
8 of 12 third-placed teams advance, ranked by points→GD→GF→(fair-play/random). They fill the
eight `3X` slots (matches 74,77,79,80,81,82,85,87). The official FIFA rule is a fixed
495-row lookup table (C(12,8)) keyed by the *set* of qualifying-third groups.
**Primary approach:** constraint-based assignment — each `3X` slot lists its 5 allowed groups
(above); solve a bipartite matching of the 8 qualifying thirds to the 8 slots respecting those
sets (and never pairing a third with a winner from its own group). This is deterministic given
a tiebreak/seed and avoids encoding 495 rows. **Optional higher-fidelity:** encode the official
495-row table as a data file if exact FIFA slotting is later required.

## Proposed implementation plan

New module `src/wcpredictor/simulate.py` + bracket data in
`src/wcpredictor/tournament_bracket.py` (or `data/raw/wc2026_bracket.json`). Reuse predict.py.

1. **`infer_groups(fixtures)`** → `{label: [teams]}` via connected-components + schedule-order
   labels; assert exactly 12×4; honor an optional `group` column override.

2. **`precompute_pairwise(state, model)`** → memoized lookup `(team_a, team_b) → (p_win, p_draw,
   p_loss, score_matrix)` via `_predict_one_frozen` (neutral=True). Lazy + cached so each
   ordered pair is computed at most once across all sims.

3. **`sample_scoreline(score_matrix, rng)`** → `(ga, gb)` sampled from the 9×9 joint matrix
   (flatten → `rng.choice`). Gives outcome + goals for standings/tiebreaks.

4. **Group stage** per sim: sample all 6 matches/group → points (3/1/0), GD, GF. Rank by
   points→GD→GF→random. Emit group winner, runner-up, and the third-placed team (with its
   record) per group.

5. **Third-place qualification**: rank the 12 thirds globally (points→GD→GF→random), take top
   8, assign to the eight `3X` slots by constraint-based bipartite matching (step in Findings).

6. **Knockout** per sim: build R32 from template; for each tie sample a scoreline from the
   pairwise matrix; on a draw resolve with the simple ET/penalty resolver
   (`resolve_draw(p_win, p_loss, rng)` → pick winner with prob `p_win/(p_win+p_loss)`; flagged
   placeholder). Advance W73…W102 per the progression map; record each team's furthest round
   and the champion.

7. **`simulate_tournament(as_of, model='ensemble', n_sims=20000, seed=...)`** → aggregate to a
   per-team probability table; write `data/processed/wc2026_tournament_sim_<as_of>.csv`
   (columns: team, group, p_win_group, p_runner_up, p_r32, p_r16, p_qf, p_sf, p_final,
   p_champion) + a small JSON summary (top-10 favorites). Add a `__main__` CLI entry.

8. **Tests (`tests/test_wc2026.py` or new `tests/test_simulate.py`)**:
   - `infer_groups` → 12 groups × 4 teams; labels stable; Mexico→A.
   - bracket template integrity: 16 R32 / 8 R16 / 4 QF / 2 SF references resolve; no team can
     meet a same-group team before QF.
   - third-place assignment: always fills 8 slots, respects allowed-group sets, no same-group
     R32 clash (fuzz over random qualifying sets).
   - standings tiebreak ordering is correct on crafted inputs.
   - knockout never propagates a draw (resolver always returns a winner).
   - per-team P(champion) ∈ [0,1] and **sums to 1.0** across all 48 teams; same for P(reach
     final)=2.0 total, etc. (sanity totals).
   - fixed seed → reproducible output.
   - monotonicity sanity: a clearly stronger group (higher Elo) yields higher P(advance).

## Exact next steps
1. Branch from clean `main`; create `simulate.py` + bracket data module.
2. Implement steps 1–7; wire CLI.
3. Add tests (step 8); `uv run pytest -q` green.
4. Run `simulate_tournament('2026-06-10')`; eyeball top favorites for face validity.
5. Commit as Phase 5.0; update HANDOFF.

## Verification / test commands
```bash
uv run pytest -q                              # existing 142 + new sim tests pass
uv run pytest -q tests/test_simulate.py
uv run python -m wcpredictor.simulate --as-of 2026-06-10 --model ensemble --n-sims 20000
uv run python -c "
import pandas as pd
df = pd.read_csv('data/processed/wc2026_tournament_sim_2026-06-10.csv')
print('champion sum:', round(df.p_champion.sum(),4))   # ~1.0
print(df.sort_values('p_champion', ascending=False).head(10)[['team','group','p_champion']])
"
```
Expect: P(champion) sums to ~1.0; favorites are credible (e.g. Brazil/Spain/France/Argentina/
England high).

## Risks and open questions
- **Group labels from a synthetic-looking draw**: USA lands in C here (not the real-2026 D), so
  these fixtures may be illustrative. The schedule-order labeling is self-consistent and aligns
  with the host-opener template; if the real FIFA draw is later loaded, add an explicit `group`
  column and the override path handles it.
- **Third-place allocation**: constraint-matching is a faithful *valid* assignment but not
  guaranteed identical to FIFA's exact 495-row slotting. Encode the official table only if exact
  slotting becomes a requirement.
- **ET/penalty resolver is a placeholder** (conditional-win-prob coin weighting), not a trained
  module — matches CLAUDE.md's "separate future module" stance. Document clearly.
- **Independence assumption**: matches sampled independently from frozen pre-tournament probs;
  no in-tournament Elo updating. Acceptable for a pre-tournament forecast; note as a limitation.
- **Cost**: 20k sims × ~63 matches with memoized pairwise lookup should run in well under a
  minute once the pairwise cache is warm; verify and tune `n_sims`.

## What not to repeat
- Don't refit models inside the sim loop — fit once via `_build_frozen_state`, predict via
  `_predict_one_frozen`, memoize pairwise results.
- Don't hard-code team names into bracket logic — key everything on group position.
- Don't build a heavyweight ET/penalty model now — simple resolver only.

---

## Deferred (separate phases)
- **Odds refresh** (~2026-06-09): re-download fdco xlsx, update `FDCO_ODDS_SHA256`, re-run
  `python -m wcpredictor.evaluation.backtest` to populate `odds_alpha_pooled`, regenerate with
  `models=['poisson','ensemble','ensemble_mkt']`. (`ensemble_mkt` is inert until then.)
- **Real knockout fixtures** after 06-27: replace simulated standings with actual ones; the
  bracket module here is reused directly.
- **Matchday loop** from 06-11: `python -m wcpredictor.evaluation.live --as-of <next-day>`.
