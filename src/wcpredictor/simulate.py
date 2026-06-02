"""Tournament simulator for FIFA World Cup 2026.

Monte-Carlo simulation of the 48-team / 12-group / knockout bracket.
Uses frozen model predictions — no model refitting inside the sim loop.

Usage
-----
    uv run python -m wcpredictor.simulate --as-of 2026-06-10 --model ensemble --n-sims 20000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from wcpredictor.config import DATA_PROCESSED, DATA_RAW
from wcpredictor.data.normalize_teams import canonical
from wcpredictor.predict import _build_frozen_state, _predict_one_frozen

# ---------------------------------------------------------------------------
# Bracket template
# ---------------------------------------------------------------------------
# Position specs: "1A" = group-A winner, "2B" = runner-up, "3[X/Y/...]" = qualifying third.
_R32_SLOTS: dict[int, tuple[str, str]] = {
    73: ("2A", "2B"),
    74: ("1E", "3[A/B/C/D/F]"),
    75: ("1F", "2C"),
    76: ("1C", "2F"),
    77: ("1I", "3[C/D/F/G/H]"),
    78: ("2E", "2I"),
    79: ("1A", "3[C/E/F/H/I]"),
    80: ("1L", "3[E/H/I/J/K]"),
    81: ("1D", "3[B/E/F/I/J]"),
    82: ("1G", "3[A/E/H/I/J]"),
    83: ("2K", "2L"),
    84: ("1H", "2J"),
    85: ("1B", "3[E/F/G/I/J]"),
    86: ("1J", "2H"),
    87: ("1K", "3[D/E/I/J/L]"),
    88: ("2D", "2G"),
}

_R16_SLOTS: dict[int, tuple[int, int]] = {
    89: (73, 75),
    90: (74, 77),
    91: (76, 78),
    92: (79, 80),
    93: (83, 84),
    94: (81, 82),
    95: (86, 88),
    96: (85, 87),
}

_QF_SLOTS: dict[int, tuple[int, int]] = {
    97: (89, 90),
    98: (93, 94),
    99: (91, 92),
    100: (95, 96),
}

_SF_SLOTS: dict[int, tuple[int, int]] = {
    101: (97, 98),
    102: (99, 100),
}

_FINAL_SLOT: tuple[int, int] = (101, 102)

# Third-place slot constraints: match_id → frozenset of allowed group labels.
# Constraint sets already exclude the same-group winner in each match.
_THIRD_SLOTS: dict[int, frozenset[str]] = {
    74: frozenset("ABCDF"),
    77: frozenset("CDFGH"),
    79: frozenset("CEFHI"),
    80: frozenset("EHIJK"),
    81: frozenset("BEFIJ"),
    82: frozenset("AEHIJ"),
    85: frozenset("EFGIJ"),
    87: frozenset("DEIJL"),
}


# ---------------------------------------------------------------------------
# Group inference
# ---------------------------------------------------------------------------

def infer_groups(fixtures: pd.DataFrame) -> dict[str, list[str]]:
    """Cluster fixtures into groups via connected-components; label A-L by first appearance.

    An explicit 'group' column (all non-null) in fixtures overrides inference.
    Raises ValueError if the result does not yield exactly 12 groups of 4.
    """
    if "group" in fixtures.columns and fixtures["group"].notna().all():
        groups: dict[str, list[str]] = {}
        seen: dict[str, set[str]] = {}
        for _, row in fixtures.iterrows():
            g = str(row["group"]).upper()
            ta = canonical(str(row["team_a"]))
            tb = canonical(str(row["team_b"]))
            if g not in seen:
                seen[g] = set()
                groups[g] = []
            for t in (ta, tb):
                if t not in seen[g]:
                    seen[g].add(t)
                    groups[g].append(t)
        return groups

    # Build adjacency from fixtures (teams sharing a match are in the same group)
    adjacency: dict[str, set[str]] = {}
    for _, row in fixtures.iterrows():
        ta = canonical(str(row["team_a"]))
        tb = canonical(str(row["team_b"]))
        adjacency.setdefault(ta, set()).add(tb)
        adjacency.setdefault(tb, set()).add(ta)

    visited: set[str] = set()
    component_order: list[set[str]] = []

    for _, row in fixtures.iterrows():
        for team in (canonical(str(row["team_a"])), canonical(str(row["team_b"]))):
            if team not in visited:
                component: set[str] = set()
                queue = [team]
                while queue:
                    t = queue.pop()
                    if t in component:
                        continue
                    component.add(t)
                    queue.extend(adjacency.get(t, set()) - component)
                visited |= component
                component_order.append(component)

    if len(component_order) != 12:
        raise ValueError(f"Expected 12 groups, got {len(component_order)}")
    for c in component_order:
        if len(c) != 4:
            raise ValueError(f"Expected 4 teams per group, got {len(c)}: {sorted(c)}")

    # Order teams within each group by first appearance in the schedule
    labels = "ABCDEFGHIJKL"
    result: dict[str, list[str]] = {}
    for i, component in enumerate(component_order):
        ordered: list[str] = []
        for _, row in fixtures.iterrows():
            for t in (canonical(str(row["team_a"])), canonical(str(row["team_b"]))):
                if t in component and t not in ordered:
                    ordered.append(t)
        result[labels[i]] = ordered
    return result


# ---------------------------------------------------------------------------
# Simulation primitives
# ---------------------------------------------------------------------------

def sample_scoreline(
    score_matrix: np.ndarray,
    rng: np.random.Generator,
) -> tuple[int, int]:
    """Sample (goals_a, goals_b) from the joint probability matrix."""
    n = score_matrix.shape[0]
    flat = score_matrix.ravel().astype(float)
    total = flat.sum()
    flat = flat / total if total > 0 else np.ones(len(flat)) / len(flat)
    idx = int(rng.choice(len(flat), p=flat))
    return idx // n, idx % n


def resolve_draw(
    p_win: float,
    p_loss: float,
    rng: np.random.Generator,
) -> bool:
    """Placeholder ET/penalty resolver — returns True if team_a wins.

    Uses P(A wins | not 90-min draw) = p_win / (p_win + p_loss).
    This is NOT a trained ET/penalty model; see CLAUDE.md deferred scope.
    """
    denom = p_win + p_loss
    p_a = (p_win / denom) if denom > 1e-9 else 0.5
    return bool(rng.random() < p_a)


def _ko_match(
    team_a: str,
    team_b: str,
    get_fn: object,  # callable: (ta, tb) -> (p_win, p_draw, p_loss, score_matrix)
    rng: np.random.Generator,
) -> str:
    """Simulate a single knockout match; always returns a winner (never a draw)."""
    p_win, p_draw, p_loss, smat = get_fn(team_a, team_b)  # type: ignore[operator]
    ga, gb = sample_scoreline(smat, rng)
    if ga > gb:
        return team_a
    if gb > ga:
        return team_b
    return team_a if resolve_draw(p_win, p_loss, rng) else team_b


def _assign_thirds(
    qualifying: list[tuple[str, str, int, int, int]],
) -> dict[int, str]:
    """Assign 8 qualifying thirds to the 8 third-place R32 slots via bipartite matching.

    Args:
        qualifying: [(group_label, team, pts, gd, gf), ...] — top 8 thirds in ranked order.

    Returns:
        {match_id: team_name}

    Raises:
        ValueError: if no valid assignment exists (should not happen with FIFA structure).
    """
    assert len(qualifying) == 8
    slots = sorted(_THIRD_SLOTS.keys())  # [74, 77, 79, 80, 81, 82, 85, 87]

    cost = np.full((8, 8), 1_000_000.0)
    for i, (g_label, _team, *_) in enumerate(qualifying):
        for j, slot_id in enumerate(slots):
            if g_label in _THIRD_SLOTS[slot_id]:
                cost[i, j] = 0.0

    row_ind, col_ind = linear_sum_assignment(cost)
    if cost[row_ind, col_ind].sum() > 0:
        raise ValueError(
            "No valid third-place assignment found for groups "
            f"{[q[0] for q in qualifying]}. "
            "Check that group labels match the bracket template."
        )
    return {slots[j]: qualifying[i][1] for i, j in zip(row_ind, col_ind)}


# ---------------------------------------------------------------------------
# Main simulator
# ---------------------------------------------------------------------------

def simulate_tournament(
    as_of: str,
    model: str = "ensemble",
    n_sims: int = 20_000,
    seed: int = 42,
    fixtures_path: Path | None = None,
) -> pd.DataFrame:
    """Run a Monte-Carlo tournament simulation; return per-team probability table.

    Writes:
        data/processed/wc2026_tournament_sim_<as_of>.csv  — full probability table
        data/processed/wc2026_tournament_sim_<as_of>.json — top-10 champion summary

    Args:
        as_of:         Cutoff date string (YYYY-MM-DD); historical data up to (not including) this date.
        model:         Model name: "poisson", "dixon_coles", "ensemble", or "ensemble_mkt".
        n_sims:        Number of Monte-Carlo simulations.
        seed:          RNG seed for reproducibility.
        fixtures_path: Path to wc2026_fixtures.csv; defaults to DATA_RAW/wc2026_fixtures.csv.

    Returns:
        DataFrame with columns: team, group, p_win_group, p_runner_up,
        p_r32, p_r16, p_qf, p_sf, p_final, p_champion.
    """
    if fixtures_path is None:
        fixtures_path = DATA_RAW / "wc2026_fixtures.csv"

    fixtures = pd.read_csv(fixtures_path, parse_dates=["date"])
    groups = infer_groups(fixtures)
    all_teams = [t for teams in groups.values() for t in teams]
    group_labels = list(groups.keys())

    state = _build_frozen_state(as_of, [model])

    # Lazy pairwise prediction cache: (team_a, team_b) → (p_win, p_draw, p_loss, score_matrix)
    cache: dict[tuple[str, str], tuple[float, float, float, np.ndarray]] = {}

    def _get(ta: str, tb: str) -> tuple[float, float, float, np.ndarray]:
        if (ta, tb) not in cache:
            r = _predict_one_frozen(state, model, ta, tb, neutral=True)
            cache[(ta, tb)] = (
                float(r["p_win"]),
                float(r["p_draw"]),
                float(r["p_loss"]),
                np.asarray(r["score_matrix"], dtype=float),
            )
        return cache[(ta, tb)]

    # Pre-warm cache for all fixed group-stage pairs
    for _, row in fixtures.iterrows():
        ta = canonical(str(row["team_a"]))
        tb = canonical(str(row["team_b"]))
        _get(ta, tb)

    # Accumulators
    _ROUNDS = ("r32", "r16", "qf", "sf", "final", "champion")
    reach: dict[str, dict[str, int]] = {t: {r: 0 for r in _ROUNDS} for t in all_teams}
    grp_pos: dict[str, dict[str, int]] = {t: {"win_group": 0, "runner_up": 0} for t in all_teams}

    rng = np.random.default_rng(seed)

    for _sim in range(n_sims):
        # --- Group stage ---
        pos: dict[str, str] = {}  # "1A", "2B", ... → team
        third_records: list[tuple[str, str, int, int, int]] = []  # (group, team, pts, gd, gf)

        for g, teams in groups.items():
            pts: dict[str, int] = {t: 0 for t in teams}
            gd: dict[str, int] = {t: 0 for t in teams}
            gf: dict[str, int] = {t: 0 for t in teams}

            for i in range(4):
                for j in range(i + 1, 4):
                    ta, tb = teams[i], teams[j]
                    _, _, _, smat = _get(ta, tb)
                    ga, gb = sample_scoreline(smat, rng)
                    gf[ta] += ga
                    gf[tb] += gb
                    gd[ta] += ga - gb
                    gd[tb] += gb - ga
                    if ga > gb:
                        pts[ta] += 3
                    elif ga == gb:
                        pts[ta] += 1
                        pts[tb] += 1
                    else:
                        pts[tb] += 3

            tb_idx = {t: i for i, t in enumerate(teams)}
            tb_val = rng.random(4)
            ranked = sorted(
                teams,
                key=lambda t: (-pts[t], -gd[t], -gf[t], tb_val[tb_idx[t]]),
            )
            pos[f"1{g}"] = ranked[0]
            pos[f"2{g}"] = ranked[1]
            grp_pos[ranked[0]]["win_group"] += 1
            grp_pos[ranked[1]]["runner_up"] += 1
            third_records.append((g, ranked[2], pts[ranked[2]], gd[ranked[2]], gf[ranked[2]]))

        # --- Third-place qualification: rank 12 thirds, keep top 8 ---
        tb3 = rng.random(12)
        third_records.sort(
            key=lambda x: (-x[2], -x[3], -x[4], tb3[group_labels.index(x[0])]),
        )
        third_slot: dict[int, str] = _assign_thirds(third_records[:8])

        # --- Knockout ---
        winner: dict[int, str] = {}

        # R32
        for m_id, (spec_a, spec_b) in _R32_SLOTS.items():
            ta = pos[spec_a]
            tb = third_slot[m_id] if spec_b.startswith("3[") else pos[spec_b]
            w = _ko_match(ta, tb, _get, rng)
            winner[m_id] = w
            reach[ta]["r32"] += 1
            reach[tb]["r32"] += 1

        for slots_map, round_key in (
            (_R16_SLOTS, "r16"),
            (_QF_SLOTS, "qf"),
            (_SF_SLOTS, "sf"),
        ):
            for m_id, (ma, mb) in slots_map.items():
                ta, tb = winner[ma], winner[mb]
                w = _ko_match(ta, tb, _get, rng)
                winner[m_id] = w
                reach[ta][round_key] += 1
                reach[tb][round_key] += 1

        # Final
        ta, tb = winner[_FINAL_SLOT[0]], winner[_FINAL_SLOT[1]]
        champ = _ko_match(ta, tb, _get, rng)
        reach[ta]["final"] += 1
        reach[tb]["final"] += 1
        reach[champ]["champion"] += 1

    # --- Assemble results ---
    rows = []
    for team in all_teams:
        g = next(g for g, ts in groups.items() if team in ts)
        row: dict = {"team": team, "group": g}
        row["p_win_group"] = grp_pos[team]["win_group"] / n_sims
        row["p_runner_up"] = grp_pos[team]["runner_up"] / n_sims
        for r in _ROUNDS:
            row[f"p_{r}"] = reach[team][r] / n_sims
        rows.append(row)

    df = pd.DataFrame(rows)

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    out_csv = DATA_PROCESSED / f"wc2026_tournament_sim_{as_of}.csv"
    df.to_csv(out_csv, index=False)

    top10 = df.nlargest(10, "p_champion")[["team", "group", "p_champion"]].to_dict("records")
    summary = {
        "as_of": as_of,
        "model": model,
        "n_sims": n_sims,
        "seed": seed,
        "p_champion_sum": round(float(df["p_champion"].sum()), 6),
        "top10_champion": top10,
    }
    (DATA_PROCESSED / f"wc2026_tournament_sim_{as_of}.json").write_text(
        json.dumps(summary, indent=2)
    )

    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="WC2026 Monte-Carlo tournament simulator")
    parser.add_argument("--as-of", required=True, metavar="DATE", help="Cutoff date YYYY-MM-DD")
    parser.add_argument(
        "--model", default="ensemble",
        choices=["poisson", "dixon_coles", "ensemble", "ensemble_mkt"],
    )
    parser.add_argument("--n-sims", type=int, default=20_000, metavar="N")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    print(
        f"Simulating WC2026: as_of={args.as_of}, model={args.model}, "
        f"n_sims={args.n_sims:,}, seed={args.seed}"
    )
    df = simulate_tournament(
        as_of=args.as_of,
        model=args.model,
        n_sims=args.n_sims,
        seed=args.seed,
    )
    cols = ["team", "group", "p_win_group", "p_r32", "p_champion"]
    print("\nTop 10 by P(champion):")
    print(df.nlargest(10, "p_champion")[cols].to_string(index=False))
    print(f"\np_champion sum: {df['p_champion'].sum():.6f}")
    print(f"p_final sum:    {df['p_final'].sum():.6f}")


if __name__ == "__main__":
    _main()
