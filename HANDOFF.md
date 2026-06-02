# HANDOFF — Phase 6.0: API + frontend ✅ COMPLETE

> Completed 2026-06-03. Phase 6.0 committed. All pipeline stages from CLAUDE.md are done.

## What was built

New package `src/wcpredictor/api/`:
- `app.py` — FastAPI app with lazy per-model frozen-state cache and lifespan warm-up.
- `schemas.py` — Pydantic response models (PredictResponse, FixtureRow, TeamStanding, TournamentResponse).
- `static/index.html` — Single-page vanilla HTML/JS/CSS frontend (no build step).

`tests/test_api.py` — 8 tests covering all endpoints via FastAPI TestClient.

`pyproject.toml` — Added `api` extra (`fastapi>=0.111`, `uvicorn[standard]>=0.29`) and `httpx>=0.27` to `dev`.

## Endpoints
| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness check |
| GET | `/teams` | Sorted list of known teams |
| GET | `/predict` | Live single-match prediction (cached frozen state) |
| GET | `/fixtures` | Precomputed fixture predictions |
| GET | `/tournament` | Precomputed tournament simulation |
| GET | `/` | Static frontend |

## How to run
```bash
uv sync --extra api
uv run uvicorn wcpredictor.api.app:app --port 8000
# open http://localhost:8000/
```

## Test commands
```bash
uv run pytest -q                    # all suites including test_api.py
uv run pytest tests/test_api.py -v  # API-only (fast, ~12s)
```

## Deferred (date-gated)
- **Odds refresh** (~2026-06-09): re-download fdco xlsx, update SHA256, re-run backtest to populate `odds_alpha_pooled`, regenerate predictions with all 3 models, re-run sim.
- **Matchday loop** from 06-11: `python -m wcpredictor.evaluation.live --as-of <next-day>`.
- **Real knockouts** after 06-27: feed actual standings into bracket module in `simulate.py`.
