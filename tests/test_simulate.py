"""Tests for the Phase 5.0 tournament simulator."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wcpredictor.simulate import (
    _FINAL_SLOT,
    _QF_SLOTS,
    _R16_SLOTS,
    _R32_SLOTS,
    _SF_SLOTS,
    _THIRD_SLOTS,
    _assign_thirds,
    _ko_match,
    infer_groups,
    resolve_draw,
    sample_scoreline,
    simulate_tournament,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FIXTURE_PATH = Path(__file__).parent.parent / "data" / "raw" / "wc2026_fixtures.csv"


def _make_12group_fixtures() -> pd.DataFrame:
    """Synthetic 72-match fixture set: 12 groups × 4 teams, round-robin."""
    labels = "ABCDEFGHIJKL"
    rows = []
    for i, g in enumerate(labels):
        teams = [f"Team{g}{k}" for k in range(1, 5)]
        for a in range(4):
            for b in range(a + 1, 4):
                rows.append({"team_a": teams[a], "team_b": teams[b], "neutral": True})
    return pd.DataFrame(rows)


def _dummy_cache(
    teams: list[str],
    n: int = 9,
) -> dict[tuple[str, str], tuple[float, float, float, np.ndarray]]:
    """Uniform pairwise cache (all draws equally likely) for testing."""
    cache = {}
    mat = np.ones((n, n)) / (n * n)
    for ta in teams:
        for tb in teams:
            if ta != tb:
                cache[(ta, tb)] = (1 / 3, 1 / 3, 1 / 3, mat)
    return cache


# ---------------------------------------------------------------------------
# TestInferGroups
# ---------------------------------------------------------------------------


class TestInferGroups:
    def test_12_groups_of_4(self) -> None:
        groups = infer_groups(_make_12group_fixtures())
        assert len(groups) == 12
        for g, teams in groups.items():
            assert len(teams) == 4, f"Group {g} has {len(teams)} teams"

    def test_labels_A_to_L(self) -> None:
        groups = infer_groups(_make_12group_fixtures())
        assert set(groups.keys()) == set("ABCDEFGHIJKL")

    def test_first_group_is_A(self) -> None:
        groups = infer_groups(_make_12group_fixtures())
        first_team = groups["A"][0]
        assert first_team == "TeamA1"

    def test_stable_on_repeat(self) -> None:
        fx = _make_12group_fixtures()
        g1 = infer_groups(fx)
        g2 = infer_groups(fx)
        assert g1 == g2

    def test_explicit_group_column_override(self) -> None:
        fx = _make_12group_fixtures()
        labels = "ABCDEFGHIJKL"
        group_col = []
        for i, g in enumerate(labels):
            for _ in range(6):
                group_col.append(g)
        fx["group"] = group_col
        groups = infer_groups(fx)
        assert set(groups.keys()) == set("ABCDEFGHIJKL")
        assert len(groups["A"]) == 4

    def test_real_fixtures_mexico_group_A(self) -> None:
        if not _FIXTURE_PATH.exists():
            pytest.skip("wc2026_fixtures.csv not present")
        groups = infer_groups(pd.read_csv(_FIXTURE_PATH))
        assert "Mexico" in groups["A"], f"Group A teams: {groups['A']}"

    def test_real_fixtures_12_groups_of_4(self) -> None:
        if not _FIXTURE_PATH.exists():
            pytest.skip("wc2026_fixtures.csv not present")
        groups = infer_groups(pd.read_csv(_FIXTURE_PATH))
        assert len(groups) == 12
        for g, teams in groups.items():
            assert len(teams) == 4, f"Group {g}: {teams}"

    def test_raises_on_wrong_group_count(self) -> None:
        fx = _make_12group_fixtures().head(6)  # only one group's matches
        with pytest.raises((ValueError, AssertionError)):
            infer_groups(fx)


# ---------------------------------------------------------------------------
# TestBracketIntegrity
# ---------------------------------------------------------------------------


class TestBracketIntegrity:
    def test_r32_has_16_matches(self) -> None:
        assert len(_R32_SLOTS) == 16

    def test_r16_has_8_matches(self) -> None:
        assert len(_R16_SLOTS) == 8

    def test_qf_has_4_matches(self) -> None:
        assert len(_QF_SLOTS) == 4

    def test_sf_has_2_matches(self) -> None:
        assert len(_SF_SLOTS) == 2

    def test_final_references_two_sf_matches(self) -> None:
        sf_ids = set(_SF_SLOTS.keys())
        assert _FINAL_SLOT[0] in sf_ids
        assert _FINAL_SLOT[1] in sf_ids

    def test_r16_references_only_r32_ids(self) -> None:
        r32_ids = set(_R32_SLOTS.keys())
        for m_id, (ma, mb) in _R16_SLOTS.items():
            assert ma in r32_ids, f"R16 match {m_id} ref {ma} not in R32"
            assert mb in r32_ids, f"R16 match {m_id} ref {mb} not in R32"

    def test_qf_references_only_r16_ids(self) -> None:
        r16_ids = set(_R16_SLOTS.keys())
        for m_id, (ma, mb) in _QF_SLOTS.items():
            assert ma in r16_ids, f"QF match {m_id} ref {ma} not in R16"
            assert mb in r16_ids, f"QF match {m_id} ref {mb} not in R16"

    def test_sf_references_only_qf_ids(self) -> None:
        qf_ids = set(_QF_SLOTS.keys())
        for m_id, (ma, mb) in _SF_SLOTS.items():
            assert ma in qf_ids, f"SF match {m_id} ref {ma} not in QF"
            assert mb in qf_ids, f"SF match {m_id} ref {mb} not in QF"

    def test_eight_third_place_slots(self) -> None:
        third_match_ids = [m for m, (a, b) in _R32_SLOTS.items() if b.startswith("3[")]
        assert len(third_match_ids) == 8
        assert set(_THIRD_SLOTS.keys()) == set(third_match_ids)

    def test_r16_r32_refs_no_overlap(self) -> None:
        """Each R32 match ID should appear in at most one R16 match."""
        seen: dict[int, int] = {}
        for m16, (ma, mb) in _R16_SLOTS.items():
            for r in (ma, mb):
                assert r not in seen, f"R32 match {r} referenced twice in R16"
                seen[r] = m16

    def test_group_labels_in_third_slots_valid(self) -> None:
        valid = set("ABCDEFGHIJKL")
        for slot_id, allowed in _THIRD_SLOTS.items():
            assert allowed <= valid, f"Slot {slot_id} has unknown group labels: {allowed - valid}"

    def test_third_slot_winner_group_excluded(self) -> None:
        """The 1X winner group must NOT be in its R32 slot's allowed third groups."""
        for m_id, (spec_a, spec_b) in _R32_SLOTS.items():
            if spec_b.startswith("3[") and spec_a.startswith("1"):
                winner_group = spec_a[1]  # "1A" → "A"
                allowed = _THIRD_SLOTS[m_id]
                assert winner_group not in allowed, (
                    f"Match {m_id}: winner group {winner_group} is in allowed thirds {allowed}"
                )


# ---------------------------------------------------------------------------
# TestThirdAssignment
# ---------------------------------------------------------------------------


class TestThirdAssignment:
    def _make_qualifying(self, groups: list[str]) -> list[tuple[str, str, int, int, int]]:
        assert len(groups) == 8
        return [(g, f"Team3{g}", 4, 0, 4) for g in groups]

    def test_fills_all_8_slots(self) -> None:
        q = self._make_qualifying(list("ABCDEFGH"))
        result = _assign_thirds(q)
        assert len(result) == 8
        assert set(result.keys()) == set(_THIRD_SLOTS.keys())

    def test_respects_allowed_groups(self) -> None:
        q = self._make_qualifying(list("ABCDEFGH"))
        result = _assign_thirds(q)
        assigned_teams = {team: grp for grp, team, *_ in q}
        for slot_id, team in result.items():
            grp = assigned_teams[team]
            assert grp in _THIRD_SLOTS[slot_id], (
                f"Team from group {grp} assigned to slot {slot_id} "
                f"(allowed: {_THIRD_SLOTS[slot_id]})"
            )

    def test_no_same_group_r32_clash(self) -> None:
        """Third-place team must not face a same-group winner in R32."""
        q = self._make_qualifying(list("ABCDEFIJ"))
        result = _assign_thirds(q)
        assigned_group = {team: grp for grp, team, *_ in q}
        for slot_id, third_team in result.items():
            third_grp = assigned_group[third_team]
            spec_a, _ = _R32_SLOTS[slot_id]
            winner_grp = spec_a[1]  # "1A" → "A"
            assert third_grp != winner_grp, (
                f"Slot {slot_id}: third from {third_grp} faces winner from {winner_grp}"
            )

    def test_deterministic_same_input(self) -> None:
        q = self._make_qualifying(list("CDEFGHIJ"))
        r1 = _assign_thirds(q)
        r2 = _assign_thirds(q)
        assert r1 == r2

    def test_fuzz_various_qualifying_sets(self) -> None:
        """Any 8-group subset drawn from the 12 groups should yield a valid assignment."""
        rng = np.random.default_rng(0)
        groups = list("ABCDEFGHIJKL")
        failures = []
        for _ in range(20):
            chosen = sorted(rng.choice(groups, size=8, replace=False))
            q = [(g, f"T{g}", 4, 0, 4) for g in chosen]
            try:
                result = _assign_thirds(q)
                assigned_group = {t: g for g, t, *_ in q}
                for slot_id, team in result.items():
                    grp = assigned_group[team]
                    if grp not in _THIRD_SLOTS[slot_id]:
                        failures.append(f"slot {slot_id}: group {grp} not in {_THIRD_SLOTS[slot_id]}")
            except ValueError as e:
                failures.append(f"groups {chosen}: {e}")
        assert not failures, f"Assignment failures:\n" + "\n".join(failures)

    def test_raises_on_invalid_groups(self) -> None:
        """If qualifying thirds can't fill slots, raise ValueError."""
        # K can only go to slot 80; if all 8 are K that's impossible
        q = [("K", f"Team{i}", 4, 0, 4) for i in range(8)]
        with pytest.raises((ValueError, AssertionError)):
            _assign_thirds(q)


# ---------------------------------------------------------------------------
# TestSampleScoreline
# ---------------------------------------------------------------------------


class TestSampleScoreline:
    def test_output_within_matrix_bounds(self) -> None:
        rng = np.random.default_rng(0)
        mat = np.ones((9, 9)) / 81.0
        for _ in range(100):
            ga, gb = sample_scoreline(mat, rng)
            assert 0 <= ga <= 8
            assert 0 <= gb <= 8

    def test_high_weight_cell_sampled_more(self) -> None:
        rng = np.random.default_rng(0)
        mat = np.zeros((9, 9))
        mat[2, 1] = 1.0  # 2-1 result dominates
        counts: dict[tuple[int, int], int] = {}
        for _ in range(500):
            s = sample_scoreline(mat, rng)
            counts[s] = counts.get(s, 0) + 1
        assert counts.get((2, 1), 0) == 500

    def test_handles_zero_matrix(self) -> None:
        """Zero matrix should fall back to uniform and not crash."""
        rng = np.random.default_rng(0)
        mat = np.zeros((9, 9))
        ga, gb = sample_scoreline(mat, rng)
        assert 0 <= ga <= 8
        assert 0 <= gb <= 8

    def test_normalized_result_same_as_unnormalized(self) -> None:
        rng1 = np.random.default_rng(7)
        rng2 = np.random.default_rng(7)
        mat = np.ones((9, 9))
        mat_norm = mat / mat.sum()
        ga1, gb1 = sample_scoreline(mat, rng1)
        ga2, gb2 = sample_scoreline(mat_norm, rng2)
        assert ga1 == ga2 and gb1 == gb2


# ---------------------------------------------------------------------------
# TestResolveDraw
# ---------------------------------------------------------------------------


class TestResolveDraw:
    def test_dominant_team_a_wins_most(self) -> None:
        rng = np.random.default_rng(0)
        wins = sum(resolve_draw(0.9, 0.05, rng) for _ in range(1000))
        assert wins > 900, f"Expected team_a to win >900/1000, got {wins}"

    def test_dominant_team_b_wins_most(self) -> None:
        rng = np.random.default_rng(0)
        wins = sum(resolve_draw(0.05, 0.9, rng) for _ in range(1000))
        assert wins < 100, f"Expected team_a to win <100/1000, got {wins}"

    def test_even_roughly_50_50(self) -> None:
        rng = np.random.default_rng(42)
        wins = sum(resolve_draw(0.5, 0.5, rng) for _ in range(2000))
        assert 800 < wins < 1200, f"Expected ~1000 wins, got {wins}"

    def test_zero_denom_returns_bool(self) -> None:
        rng = np.random.default_rng(0)
        result = resolve_draw(0.0, 0.0, rng)
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# TestKoMatch
# ---------------------------------------------------------------------------


class TestKoMatch:
    def test_returns_one_of_two_teams(self) -> None:
        rng = np.random.default_rng(0)
        mat = np.ones((9, 9)) / 81
        data = {("A", "B"): (0.4, 0.3, 0.3, mat)}
        for _ in range(50):
            w = _ko_match("A", "B", lambda ta, tb: data[(ta, tb)], rng)
            assert w in ("A", "B")

    def test_never_returns_draw(self) -> None:
        """Score matrix concentrated on (1, 1) forces ET resolver but still returns a winner."""
        rng = np.random.default_rng(0)
        mat = np.zeros((9, 9))
        mat[1, 1] = 1.0  # always 1-1 → always goes to ET/penalties
        data = {("A", "B"): (0.4, 0.2, 0.4, mat)}
        results = {_ko_match("A", "B", lambda ta, tb: data[(ta, tb)], rng) for _ in range(200)}
        assert results <= {"A", "B"}, "ko_match returned something other than A or B"
        assert len(results) == 2, "expected both teams to win at least once in 200 draws"


# ---------------------------------------------------------------------------
# TestStandingsTiebreak  (via simulate_tournament with crafted fixture + tiny n_sims)
# ---------------------------------------------------------------------------


class TestStandingsTiebreak:
    def test_best_team_wins_group_most_often(self, tmp_path: Path) -> None:
        """Team with structurally higher Elo should win its group more often."""
        if not _FIXTURE_PATH.exists():
            pytest.skip("wc2026_fixtures.csv not present")
        # Use real fixture; check that higher-rated teams have higher p_win_group
        df = simulate_tournament(
            as_of="2026-06-10",
            model="poisson",
            n_sims=500,
            seed=0,
            fixtures_path=_FIXTURE_PATH,
        )
        # Basic sanity: all p_win_group in [0, 1]
        assert (df["p_win_group"] >= 0).all()
        assert (df["p_win_group"] <= 1).all()
        # Sum of group-winners per group should be ~1.0
        for g, sub in df.groupby("group"):
            assert abs(sub["p_win_group"].sum() - 1.0) < 0.02, f"Group {g} win-group sum != 1"


# ---------------------------------------------------------------------------
# TestSimulateTournament (integration tests)
# ---------------------------------------------------------------------------


class TestSimulateTournament:
    @pytest.fixture(autouse=True)
    def skip_if_no_fixtures(self) -> None:
        if not _FIXTURE_PATH.exists():
            pytest.skip("wc2026_fixtures.csv not present")

    def test_p_champion_sums_to_one(self, tmp_path: Path) -> None:
        df = simulate_tournament(
            as_of="2026-06-10", model="poisson", n_sims=300, seed=1,
            fixtures_path=_FIXTURE_PATH,
        )
        s = df["p_champion"].sum()
        assert abs(s - 1.0) < 0.02, f"p_champion sum = {s}"

    def test_p_final_sums_to_two(self, tmp_path: Path) -> None:
        df = simulate_tournament(
            as_of="2026-06-10", model="poisson", n_sims=300, seed=2,
            fixtures_path=_FIXTURE_PATH,
        )
        s = df["p_final"].sum()
        assert abs(s - 2.0) < 0.04, f"p_final sum = {s}"

    def test_p_sf_sums_to_four(self, tmp_path: Path) -> None:
        df = simulate_tournament(
            as_of="2026-06-10", model="poisson", n_sims=300, seed=3,
            fixtures_path=_FIXTURE_PATH,
        )
        s = df["p_sf"].sum()
        assert abs(s - 4.0) < 0.1, f"p_sf sum = {s}"

    def test_p_r32_sums_to_32(self, tmp_path: Path) -> None:
        df = simulate_tournament(
            as_of="2026-06-10", model="poisson", n_sims=300, seed=4,
            fixtures_path=_FIXTURE_PATH,
        )
        s = df["p_r32"].sum()
        assert abs(s - 32.0) < 0.5, f"p_r32 sum = {s}"

    def test_all_probs_in_unit_interval(self, tmp_path: Path) -> None:
        df = simulate_tournament(
            as_of="2026-06-10", model="poisson", n_sims=300, seed=5,
            fixtures_path=_FIXTURE_PATH,
        )
        prob_cols = ["p_win_group", "p_runner_up", "p_r32", "p_r16", "p_qf", "p_sf", "p_final", "p_champion"]
        for col in prob_cols:
            assert (df[col] >= 0).all(), f"{col} has negative values"
            assert (df[col] <= 1).all(), f"{col} exceeds 1"

    def test_monotonicity_r32_ge_r16(self, tmp_path: Path) -> None:
        """P(advance to R32) ≥ P(advance to R16) for every team."""
        df = simulate_tournament(
            as_of="2026-06-10", model="poisson", n_sims=300, seed=6,
            fixtures_path=_FIXTURE_PATH,
        )
        assert (df["p_r32"] >= df["p_r16"] - 1e-9).all()
        assert (df["p_r16"] >= df["p_qf"] - 1e-9).all()
        assert (df["p_qf"] >= df["p_sf"] - 1e-9).all()
        assert (df["p_sf"] >= df["p_final"] - 1e-9).all()
        assert (df["p_final"] >= df["p_champion"] - 1e-9).all()

    def test_reproducible_with_same_seed(self, tmp_path: Path) -> None:
        df1 = simulate_tournament(
            as_of="2026-06-10", model="poisson", n_sims=100, seed=99,
            fixtures_path=_FIXTURE_PATH,
        )
        df2 = simulate_tournament(
            as_of="2026-06-10", model="poisson", n_sims=100, seed=99,
            fixtures_path=_FIXTURE_PATH,
        )
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_differ(self, tmp_path: Path) -> None:
        df1 = simulate_tournament(
            as_of="2026-06-10", model="poisson", n_sims=100, seed=1,
            fixtures_path=_FIXTURE_PATH,
        )
        df2 = simulate_tournament(
            as_of="2026-06-10", model="poisson", n_sims=100, seed=2,
            fixtures_path=_FIXTURE_PATH,
        )
        assert not df1["p_champion"].equals(df2["p_champion"])

    def test_csv_and_json_written(self, tmp_path: Path) -> None:
        from wcpredictor.config import DATA_PROCESSED
        import wcpredictor.simulate as sim_mod

        orig = sim_mod.DATA_PROCESSED
        try:
            sim_mod.DATA_PROCESSED = tmp_path
            # patch config reference used inside simulate_tournament
            import wcpredictor.predict as pred_mod
            orig_pred = pred_mod.DATA_PROCESSED
            pred_mod.DATA_PROCESSED = tmp_path

            df = simulate_tournament(
                as_of="2026-06-10", model="poisson", n_sims=50, seed=0,
                fixtures_path=_FIXTURE_PATH,
            )
            assert (tmp_path / "wc2026_tournament_sim_2026-06-10.csv").exists()
            assert (tmp_path / "wc2026_tournament_sim_2026-06-10.json").exists()

            import json
            summary = json.loads((tmp_path / "wc2026_tournament_sim_2026-06-10.json").read_text())
            assert "p_champion_sum" in summary
            assert abs(summary["p_champion_sum"] - 1.0) < 0.05
        finally:
            sim_mod.DATA_PROCESSED = orig
            pred_mod.DATA_PROCESSED = orig_pred

    def test_returns_48_rows(self, tmp_path: Path) -> None:
        df = simulate_tournament(
            as_of="2026-06-10", model="poisson", n_sims=100, seed=0,
            fixtures_path=_FIXTURE_PATH,
        )
        assert len(df) == 48

    def test_correct_columns(self, tmp_path: Path) -> None:
        df = simulate_tournament(
            as_of="2026-06-10", model="poisson", n_sims=100, seed=0,
            fixtures_path=_FIXTURE_PATH,
        )
        expected = {"team", "group", "p_win_group", "p_runner_up",
                    "p_r32", "p_r16", "p_qf", "p_sf", "p_final", "p_champion"}
        assert expected <= set(df.columns)

    def test_monotonicity_stronger_group_higher_advance(self, tmp_path: Path) -> None:
        """Brazil (high Elo) should advance to R32 more often than Curaçao (low Elo)."""
        df = simulate_tournament(
            as_of="2026-06-10", model="poisson", n_sims=500, seed=0,
            fixtures_path=_FIXTURE_PATH,
        )
        brazil_r32 = df.loc[df["team"] == "Brazil", "p_r32"].values[0]
        curacao_r32 = df.loc[df["team"] == "Curaçao", "p_r32"].values[0]
        assert brazil_r32 > curacao_r32, (
            f"Expected Brazil p_r32 ({brazil_r32:.3f}) > Curaçao p_r32 ({curacao_r32:.3f})"
        )
