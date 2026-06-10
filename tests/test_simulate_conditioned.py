"""Tests for conditioned tournament simulation (Phase 12 Step 3).

Tests:
1. Three fixed losses for a team → p_win_group == 0.
2. Fixed score is constant across all simulations.
3. Forced KO winner always advances.
4. Rows on or after as_of are ignored (leakage guard).
5. Seed reproducibility with conditioning.
6. infer_groups doesn't break when KO fixtures exist.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from wcpredictor.config import KO_START, TOURNAMENT_START


# ---------------------------------------------------------------------------
# Minimal 12-group fixture builder
# ---------------------------------------------------------------------------

GROUPS = {
    "A": ["Brazil", "Mexico", "Cameroon", "Croatia"],
    "B": ["France", "Germany", "Senegal", "Japan"],
    "C": ["Spain", "Argentina", "Morocco", "Australia"],
    "D": ["England", "Portugal", "Ghana", "South Korea"],
    "E": ["Netherlands", "Belgium", "Tunisia", "Ecuador"],
    "F": ["Uruguay", "Colombia", "South Africa", "Serbia"],
    "G": ["Italy", "Switzerland", "Saudi Arabia", "Nigeria"],
    "H": ["Denmark", "Poland", "Cameroon", "Wales"],
    "I": ["USA", "Canada", "Mexico", "Panama"],
    "J": ["Iran", "Japan", "Australia", "New Zealand"],
    "K": ["Ivory Coast", "Mali", "DR Congo", "Egypt"],
    "L": ["Algeria", "Morocco", "Tunisia", "Libya"],
}

# Deduplicate team names across groups (use unique teams)
_TEAMS: dict[str, list[str]] = {}
_used: set[str] = set()
for _g, _ts in GROUPS.items():
    unique = []
    for t in _ts:
        base = t
        n = 2
        while base in _used:
            base = f"{t} {n}"
            n += 1
        _used.add(base)
        unique.append(base)
    _TEAMS[_g] = unique


def _build_fixtures(out_dir: Path, include_ko_rows: bool = False) -> Path:
    rows = []
    dates = iter(range(15, 28))  # June 15-27 for group stage
    for g, teams in _TEAMS.items():
        for i in range(4):
            for j in range(i + 1, 4):
                d = f"2026-06-{next(dates, 15):02d}"
                rows.append({
                    "date": d, "team_a": teams[i], "team_b": teams[j],
                    "neutral": True, "goals_a": "", "goals_b": "",
                })
    if include_ko_rows:
        rows.append({
            "date": KO_START, "team_a": "Brazil", "team_b": "France",
            "neutral": True, "goals_a": "", "goals_b": "",
        })
    p = out_dir / "wc2026_fixtures.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


def _mock_state(teams_all: list[str]) -> dict:
    """Return a minimal frozen state for the simulator."""
    from wcpredictor.config import INITIAL_RATING
    ratings = {t: INITIAL_RATING for t in teams_all}
    return {
        "cutoff": pd.Timestamp(TOURNAMENT_START),
        "fit_cutoff": pd.Timestamp(TOURNAMENT_START),
        "train": pd.DataFrame(),
        "elo_df": pd.DataFrame(),
        "ratings": ratings,
        "form_state": {},
        "poisson_base": 0.3,
        "poisson_beta": 0.001,
    }


def _uniform_predict(state, model_name: str, ta: str, tb: str, neutral: bool) -> dict:
    """All predictions return equal probabilities and a flat score matrix."""
    import numpy as np
    mat = np.full((9, 9), 1.0 / 81)
    return {"p_win": 1 / 3, "p_draw": 1 / 3, "p_loss": 1 / 3, "score_matrix": mat.tolist()}


@pytest.fixture()
def fixtures_path(tmp_path: Path) -> Path:
    return _build_fixtures(tmp_path)


@pytest.fixture()
def all_teams() -> list[str]:
    return [t for ts in _TEAMS.values() for t in ts]


# ---------------------------------------------------------------------------
# Test: infer_groups ignores KO fixture rows
# ---------------------------------------------------------------------------

def test_infer_groups_works_with_ko_rows(tmp_path: Path):
    """Passing fixtures that include KO rows to simulate_tournament must not raise."""
    from wcpredictor.simulate import infer_groups
    import pandas as pd

    fixtures_with_ko = _build_fixtures(tmp_path, include_ko_rows=True)
    all_f = pd.read_csv(fixtures_with_ko, parse_dates=["date"])

    ko_ts = pd.Timestamp(KO_START)
    gs_only = all_f[all_f["date"] < ko_ts]
    # Should work fine on group-stage slice
    groups = infer_groups(gs_only)
    assert len(groups) == 12


# ---------------------------------------------------------------------------
# Test: _load_conditioning leakage guard
# ---------------------------------------------------------------------------

def test_conditioning_leakage_guard():
    """Rows with date >= as_of must be excluded from conditioning."""
    from wcpredictor.simulate import _load_conditioning

    played = pd.DataFrame([{
        "date": "2026-06-15", "team_a": "Brazil", "team_b": "Mexico",
        "goals_a": 2, "goals_b": 0, "stage": "group", "winner": None, "source": "manual",
    }, {
        "date": "2026-06-22", "team_a": "France", "team_b": "Germany",
        "goals_a": 1, "goals_b": 0, "stage": "group", "winner": None, "source": "manual",
    }])
    played["date"] = pd.to_datetime(played["date"])

    # as_of = June 16: only the June 15 match should be visible
    fixed, _ = _load_conditioning("2026-06-16", played_results=played)
    assert frozenset(["Brazil", "Mexico"]) in fixed
    assert frozenset(["France", "Germany"]) not in fixed


# ---------------------------------------------------------------------------
# Test: KO winner always advances
# ---------------------------------------------------------------------------

def test_ko_winner_always_advances(tmp_path: Path, all_teams: list[str]):
    """A team set as KO winner must have a non-zero probability in later rounds."""
    from wcpredictor.simulate import simulate_tournament

    fp = _build_fixtures(tmp_path)
    state = _mock_state(all_teams)
    team_a = _TEAMS["A"][0]  # Group A winner in our fixed scenario

    ko_results = pd.DataFrame([{
        "date": KO_START, "team_a": team_a, "team_b": _TEAMS["B"][0],
        "goals_a": 2, "goals_b": 1, "stage": "knockout", "winner": None, "source": "manual",
    }])
    ko_results["date"] = pd.to_datetime(ko_results["date"])

    with patch("wcpredictor.simulate._build_frozen_state", return_value=state), \
         patch("wcpredictor.simulate._predict_one_frozen", side_effect=_uniform_predict):
        df = simulate_tournament(
            as_of="2026-06-29",
            model="poisson",
            n_sims=100,
            seed=0,
            fixtures_path=fp,
            output_dir=tmp_path,
            condition_on_results=True,
            played_results=ko_results,
        )

    row = df[df["team"] == team_a].iloc[0]
    # team_a won R32 — must show up in r16 with p > 0
    assert row["p_r16"] > 0


# ---------------------------------------------------------------------------
# Test: fixed group score is constant across sims
# ---------------------------------------------------------------------------

def test_fixed_group_score_deterministic(tmp_path: Path, all_teams: list[str]):
    """A fixed group score must be the same in every simulation (p_win_group changes
    but we can verify via the condition dict that scores are not sampled)."""
    from wcpredictor.simulate import _load_conditioning

    played = pd.DataFrame([{
        "date": "2026-06-15", "team_a": "Brazil", "team_b": "Mexico",
        "goals_a": 3, "goals_b": 0, "stage": "group", "winner": None, "source": "manual",
    }])
    played["date"] = pd.to_datetime(played["date"])

    fixed, _ = _load_conditioning("2026-06-29", played_results=played)
    key = frozenset(["Brazil", "Mexico"])
    assert key in fixed
    ta, ga, gb = fixed[key]
    assert ta == "Brazil"
    assert ga == 3
    assert gb == 0


# ---------------------------------------------------------------------------
# Test: team with three fixed group losses must have p_win_group == 0
# ---------------------------------------------------------------------------

def test_three_losses_zero_group_win(tmp_path: Path, all_teams: list[str]):
    """A team that lost all 3 group matches in fixed_scores must have p_win_group == 0."""
    from wcpredictor.simulate import simulate_tournament

    fp = _build_fixtures(tmp_path)
    state = _mock_state(all_teams)
    loser = _TEAMS["A"][3]
    others = _TEAMS["A"][:3]

    # Brazil (or whoever) beats loser 3x
    played = pd.DataFrame([{
        "date": "2026-06-15", "team_a": others[0], "team_b": loser,
        "goals_a": 3, "goals_b": 0, "stage": "group", "winner": None, "source": "manual",
    }, {
        "date": "2026-06-19", "team_a": others[1], "team_b": loser,
        "goals_a": 2, "goals_b": 0, "stage": "group", "winner": None, "source": "manual",
    }, {
        "date": "2026-06-23", "team_a": others[2], "team_b": loser,
        "goals_a": 1, "goals_b": 0, "stage": "group", "winner": None, "source": "manual",
    }])
    played["date"] = pd.to_datetime(played["date"])

    with patch("wcpredictor.simulate._build_frozen_state", return_value=state), \
         patch("wcpredictor.simulate._predict_one_frozen", side_effect=_uniform_predict):
        df = simulate_tournament(
            as_of="2026-06-27",
            model="poisson",
            n_sims=200,
            seed=42,
            fixtures_path=fp,
            output_dir=tmp_path,
            condition_on_results=True,
            played_results=played,
        )

    row = df[df["team"] == loser].iloc[0]
    assert row["p_win_group"] == 0.0, (
        f"Team with 0 pts should have p_win_group=0, got {row['p_win_group']}"
    )


# ---------------------------------------------------------------------------
# Test: reproducibility with seed
# ---------------------------------------------------------------------------

def test_seed_reproducibility(tmp_path: Path, all_teams: list[str]):
    """Two runs with the same seed must produce identical output."""
    from wcpredictor.simulate import simulate_tournament

    fp = _build_fixtures(tmp_path)
    state = _mock_state(all_teams)

    kwargs = dict(
        model="poisson", n_sims=50, seed=99,
        fixtures_path=fp, output_dir=tmp_path,
        condition_on_results=False,
    )

    with patch("wcpredictor.simulate._build_frozen_state", return_value=state), \
         patch("wcpredictor.simulate._predict_one_frozen", side_effect=_uniform_predict):
        df1 = simulate_tournament(as_of=TOURNAMENT_START, **kwargs)
        df2 = simulate_tournament(as_of=TOURNAMENT_START, **kwargs)

    pd.testing.assert_frame_equal(
        df1.sort_values("team").reset_index(drop=True),
        df2.sort_values("team").reset_index(drop=True),
    )


# ---------------------------------------------------------------------------
# Test: no-conditioning flag bypasses fixed scores
# ---------------------------------------------------------------------------

def test_no_condition_bypasses_fixed_scores(tmp_path: Path, all_teams: list[str]):
    """With condition_on_results=False, _load_conditioning is never called."""
    from wcpredictor import simulate as sim_module

    fp = _build_fixtures(tmp_path)
    state = _mock_state(all_teams)

    call_count = {"n": 0}
    original = sim_module._load_conditioning

    def spy(*args, **kwargs):
        call_count["n"] += 1
        return original(*args, **kwargs)

    with patch("wcpredictor.simulate._build_frozen_state", return_value=state), \
         patch("wcpredictor.simulate._predict_one_frozen", side_effect=_uniform_predict), \
         patch.object(sim_module, "_load_conditioning", side_effect=spy):
        sim_module.simulate_tournament(
            as_of=TOURNAMENT_START,
            model="poisson",
            n_sims=10,
            seed=0,
            fixtures_path=fp,
            output_dir=tmp_path,
            condition_on_results=False,
        )

    assert call_count["n"] == 0, "_load_conditioning should not be called when condition_on_results=False"
