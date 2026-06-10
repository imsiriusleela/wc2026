"""Tests for Part B: WC2026 fixtures download/parse, predict_fixtures, live scorecard."""

from __future__ import annotations

import io
import json
import math
from pathlib import Path
from unittest.mock import patch

import numpy as np
import openpyxl
import pandas as pd
import pytest

from wcpredictor.config import DATA_RAW, DATA_PROCESSED
from wcpredictor.evaluation.live import _label, run_refresh, score_completed_matches
from wcpredictor.predict import predict_fixtures


# ── helpers ────────────────────────────────────────────────────────────────────


def _make_fixtures_csv(tmp_path: Path, matches: list[dict] | None = None) -> Path:
    if matches is None:
        matches = [
            {"date": "2026-06-15", "team_a": "Brazil", "team_b": "Mexico", "neutral": True, "goals_a": "", "goals_b": ""},
            {"date": "2026-06-16", "team_a": "France", "team_b": "Germany", "neutral": True, "goals_a": "", "goals_b": ""},
            {"date": "2026-06-10", "team_a": "Spain", "team_b": "Argentina", "neutral": True, "goals_a": "2", "goals_b": "1"},
        ]
    p = tmp_path / "wc2026_fixtures.csv"
    pd.DataFrame(matches).to_csv(p, index=False)
    return p


def _make_wc2026_xlsx(sheet_name: str = "WorldCup2026") -> bytes:
    """Build a minimal xlsx with a WC2026 sheet matching fdco column layout."""
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = sheet_name
    headers = ["Competition", "Home", "Away", "Date", "Time", "HGFT", "AGFT",
               "HG1st", "AG1st", "HG2nd", "AG2nd", "HGET", "AGET", "HGP", "HGP",
               "Finished", "HS", "AS", "HST", "AST", "HF", "AF", "HC", "AC",
               "HY", "AY", "HR", "AR", "bet365-H", "bet365-D", "bet365-A",
               "H-Max", "D-Max", "A-Max", "H-Avg", "D-Avg", "A-Avg"]
    ws.append(headers)
    # One played match
    ws.append(["WC", "Brazil", "Mexico", "2026-06-15", "15:00",
               2, 1, 1, 0, 1, 1, None, None, None, None,
               True, 12, 8, 5, 3, 15, 14, 6, 4, 2, 1, 0, 0,
               2.1, 3.5, 3.8, 2.15, 3.6, 3.9, 2.12, 3.55, 3.85])
    # One unplayed match
    ws.append(["WC", "France", "Germany", "2026-06-16", "18:00",
               None, None, None, None, None, None, None, None, None, None,
               None, None, None, None, None, None, None, None, None, None, None, None, None,
               1.9, 3.6, 4.2, 1.95, 3.7, 4.3, 1.92, 3.65, 4.25])
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


# ── download_wc2026 parser unit tests ─────────────────────────────────────────


class TestParseWc2026Sheet:
    def test_parse_returns_dataframe(self) -> None:
        from wcpredictor.data.download_wc2026 import _parse_wc2026_sheet
        xlsx = _make_wc2026_xlsx()
        df = _parse_wc2026_sheet(xlsx)
        assert not df.empty
        assert set(df.columns) >= {"date", "team_a", "team_b", "neutral"}

    def test_parse_row_count(self) -> None:
        from wcpredictor.data.download_wc2026 import _parse_wc2026_sheet
        df = _parse_wc2026_sheet(_make_wc2026_xlsx())
        assert len(df) == 2

    def test_parse_goals_present_for_played(self) -> None:
        from wcpredictor.data.download_wc2026 import _parse_wc2026_sheet
        df = _parse_wc2026_sheet(_make_wc2026_xlsx())
        played = df[df["goals_a"] != ""].iloc[0]
        assert int(played["goals_a"]) == 2
        assert int(played["goals_b"]) == 1

    def test_parse_goals_empty_for_unplayed(self) -> None:
        from wcpredictor.data.download_wc2026 import _parse_wc2026_sheet
        df = _parse_wc2026_sheet(_make_wc2026_xlsx())
        unplayed = df[df["goals_a"] == ""].iloc[0]
        assert unplayed["goals_b"] == ""

    def test_parse_missing_sheet_returns_empty(self) -> None:
        from wcpredictor.data.download_wc2026 import _parse_wc2026_sheet
        wb = openpyxl.Workbook()
        wb.active.title = "WorldCup2022"
        buf = io.BytesIO()
        wb.save(buf)
        df = _parse_wc2026_sheet(buf.getvalue())
        assert df.empty

    def test_parse_deterministic(self) -> None:
        from wcpredictor.data.download_wc2026 import _parse_wc2026_sheet
        xlsx = _make_wc2026_xlsx()
        df1 = _parse_wc2026_sheet(xlsx)
        df2 = _parse_wc2026_sheet(xlsx)
        pd.testing.assert_frame_equal(df1, df2)

    def test_parse_team_names_canonical(self) -> None:
        from wcpredictor.data.download_wc2026 import _parse_wc2026_sheet
        df = _parse_wc2026_sheet(_make_wc2026_xlsx())
        assert "Brazil" in df["team_a"].values or "Brazil" in df["team_b"].values

    def test_all_neutral(self) -> None:
        from wcpredictor.data.download_wc2026 import _parse_wc2026_sheet
        df = _parse_wc2026_sheet(_make_wc2026_xlsx())
        assert all(df["neutral"])


# ── predict_fixtures unit tests ───────────────────────────────────────────────


class TestPredictFixtures:
    def test_raises_if_fixtures_missing(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            predict_fixtures("2026-06-15", fixtures_path=tmp_path / "nonexistent.csv")

    def test_filters_past_matches(self, tmp_path: Path) -> None:
        """Fixtures before as_of_date are not predicted."""
        path = _make_fixtures_csv(tmp_path)
        df = predict_fixtures("2026-06-15", fixtures_path=path, output_path=tmp_path / "out.csv")
        # 2026-06-10 (Spain vs Argentina) is before as_of_date → excluded
        # 2026-06-15 and 2026-06-16 are on/after → included
        dates = pd.to_datetime(df["date"])
        assert all(dates >= pd.Timestamp("2026-06-15"))

    def test_output_has_probabilities(self, tmp_path: Path) -> None:
        path = _make_fixtures_csv(tmp_path)
        df = predict_fixtures("2026-06-15", fixtures_path=path, output_path=tmp_path / "out.csv")
        assert "p_win" in df.columns
        assert "p_draw" in df.columns
        assert "p_loss" in df.columns

    def test_probabilities_sum_to_one(self, tmp_path: Path) -> None:
        path = _make_fixtures_csv(tmp_path)
        df = predict_fixtures("2026-06-15", fixtures_path=path, output_path=tmp_path / "out.csv")
        for _, row in df.iterrows():
            if row["p_win"] is not None and not (isinstance(row["p_win"], float) and math.isnan(row["p_win"])):
                total = float(row["p_win"]) + float(row["p_draw"]) + float(row["p_loss"])
                assert abs(total - 1.0) < 1e-4

    def test_csv_written(self, tmp_path: Path) -> None:
        path = _make_fixtures_csv(tmp_path)
        out = tmp_path / "preds.csv"
        predict_fixtures("2026-06-15", fixtures_path=path, output_path=out)
        assert out.exists()
        loaded = pd.read_csv(out)
        assert len(loaded) >= 1

    def test_all_upcoming_if_far_past_date(self, tmp_path: Path) -> None:
        path = _make_fixtures_csv(tmp_path)
        df = predict_fixtures("2020-01-01", fixtures_path=path, output_path=tmp_path / "out.csv")
        assert len(df) == 3  # all fixtures included when cutoff is before all dates

    def test_model_column_present(self, tmp_path: Path) -> None:
        """predict_fixtures always includes a `model` column."""
        path = _make_fixtures_csv(tmp_path)
        df = predict_fixtures(
            "2026-06-15", fixtures_path=path, output_path=tmp_path / "out.csv",
            model="poisson",
        )
        assert "model" in df.columns
        assert (df["model"] == "poisson").all()

    def test_multi_model_rows(self, tmp_path: Path) -> None:
        """Using `models` list emits one row per (fixture, model)."""
        path = _make_fixtures_csv(tmp_path)
        df = predict_fixtures(
            "2026-06-15", fixtures_path=path, output_path=tmp_path / "out.csv",
            models=["poisson", "dixon_coles"],
        )
        # 2 upcoming fixtures × 2 models = 4 rows
        assert len(df) == 4
        assert set(df["model"].unique()) == {"poisson", "dixon_coles"}

    def test_batch_matches_single_predict(self, tmp_path: Path) -> None:
        """Batch path (predict_fixtures) must give same W/D/L as predict_match for Poisson."""
        from wcpredictor.predict import predict_match

        path = _make_fixtures_csv(tmp_path)
        df = predict_fixtures(
            "2026-06-15", fixtures_path=path,
            output_path=tmp_path / "out.csv",
            models=["poisson"],
        )
        row = df[df["team_a"] == "Brazil"].iloc[0]
        single = predict_match("Brazil", "Mexico", "2026-06-15", neutral=True, model="poisson")
        assert abs(float(row["p_win"]) - single["p_win"]) < 1e-4
        assert abs(float(row["p_draw"]) - single["p_draw"]) < 1e-4
        assert abs(float(row["p_loss"]) - single["p_loss"]) < 1e-4


# ── label helper ─────────────────────────────────────────────────────────────


def test_label_win():
    assert _label(2, 1) == 0


def test_label_draw():
    assert _label(1, 1) == 1


def test_label_loss():
    assert _label(0, 2) == 2


# ── score_completed_matches leakage guard ─────────────────────────────────────


def test_score_completed_does_not_use_future_matches() -> None:
    """Matches with date >= as_of_date must NOT appear in scored output."""
    preds = pd.DataFrame({
        "date": pd.to_datetime(["2026-06-20"]),
        "team_a": ["Brazil"],
        "team_b": ["Mexico"],
        "p_win": [0.5],
        "p_draw": [0.3],
        "p_loss": [0.2],
    })
    results = pd.DataFrame({
        "date": pd.to_datetime(["2026-06-20"]),
        "home_team": ["Brazil"],
        "away_team": ["Mexico"],
        "home_score": [2],
        "away_score": [1],
    })
    # as_of_date = same day → match is NOT completed yet → should be excluded
    matched = score_completed_matches(preds, results, as_of_date="2026-06-20")
    assert len(matched) == 0


def test_score_completed_uses_past_matches() -> None:
    """Matches strictly before as_of_date that exist in results should be scored."""
    preds = pd.DataFrame({
        "date": pd.to_datetime(["2026-06-15"]),
        "team_a": ["Brazil"],
        "team_b": ["Mexico"],
        "p_win": [0.6],
        "p_draw": [0.2],
        "p_loss": [0.2],
    })
    results = pd.DataFrame({
        "date": pd.to_datetime(["2026-06-15"]),
        "home_team": ["Brazil"],
        "away_team": ["Mexico"],
        "home_score": [2],
        "away_score": [0],
    })
    matched = score_completed_matches(preds, results, as_of_date="2026-06-20")
    assert len(matched) == 1
    assert matched[0]["label"] == 0  # team_a (Brazil) wins


# ── run_refresh with no completed matches ────────────────────────────────────


def test_run_refresh_no_predictions(tmp_path: Path, monkeypatch) -> None:
    """With no saved predictions, scorecard should show n_completed=0."""
    import wcpredictor.evaluation.live as live_mod

    monkeypatch.setattr(live_mod, "DATA_PROCESSED", tmp_path)
    monkeypatch.setattr(live_mod, "_load_predictions", lambda: pd.DataFrame())
    monkeypatch.setattr(
        live_mod,
        "_load_latest_results",
        lambda results_csv=None: pd.DataFrame(
            columns=["date", "home_team", "away_team", "home_score", "away_score"]
        ),
    )

    scorecard = run_refresh("2026-06-12")
    assert scorecard["n_completed"] == 0
    assert scorecard["log_loss"] is None
    assert scorecard["temperature"] == 1.0
    assert (tmp_path / "wc2026_scorecard.json").exists()


def test_run_refresh_per_model_scoring(tmp_path: Path, monkeypatch) -> None:
    """With two models, scorecard must contain a per-model block with correct labels."""
    import wcpredictor.evaluation.live as live_mod

    monkeypatch.setattr(live_mod, "DATA_PROCESSED", tmp_path)

    preds_df = pd.DataFrame({
        "date": pd.to_datetime(["2026-06-15", "2026-06-15"]),
        "team_a": ["Brazil", "Brazil"],
        "team_b": ["Mexico", "Mexico"],
        "p_win": [0.6, 0.55],
        "p_draw": [0.2, 0.22],
        "p_loss": [0.2, 0.23],
        "model": ["poisson", "ensemble"],
    })
    monkeypatch.setattr(live_mod, "_load_predictions", lambda: preds_df)
    monkeypatch.setattr(
        live_mod,
        "_load_latest_results",
        lambda results_csv=None: pd.DataFrame({
            "date": pd.to_datetime(["2026-06-15"]),
            "home_team": ["Brazil"],
            "away_team": ["Mexico"],
            "home_score": [2],
            "away_score": [0],
        }),
    )

    scorecard = run_refresh("2026-06-20")
    assert scorecard["n_completed"] == 2
    assert "models" in scorecard
    assert "poisson" in scorecard["models"]
    assert "ensemble" in scorecard["models"]
    assert scorecard["models"]["poisson"]["n"] == 1
    assert scorecard["models"]["ensemble"]["n"] == 1


def test_score_completed_model_passthrough() -> None:
    """model column from predictions is passed through to matched dicts."""
    preds = pd.DataFrame({
        "date": pd.to_datetime(["2026-06-15", "2026-06-15"]),
        "team_a": ["Brazil", "Brazil"],
        "team_b": ["Mexico", "Mexico"],
        "p_win": [0.6, 0.55],
        "p_draw": [0.2, 0.22],
        "p_loss": [0.2, 0.23],
        "model": ["poisson", "ensemble"],
    })
    results = pd.DataFrame({
        "date": pd.to_datetime(["2026-06-15"]),
        "home_team": ["Brazil"],
        "away_team": ["Mexico"],
        "home_score": [2],
        "away_score": [0],
    })
    matched = score_completed_matches(preds, results, as_of_date="2026-06-20")
    assert len(matched) == 2
    models_found = {m["model"] for m in matched}
    assert models_found == {"poisson", "ensemble"}


# ── ensemble_mkt tests ────────────────────────────────────────────────────────


class TestEnsembleMkt:
    def test_probs_sum_to_one(self, tmp_path: Path) -> None:
        """ensemble_mkt W/D/L must sum to 1.0 within 1e-4 (rounding/truncation tolerance)."""
        path = _make_fixtures_csv(tmp_path)
        df = predict_fixtures(
            "2026-06-15", fixtures_path=path,
            output_path=tmp_path / "out.csv",
            models=["ensemble_mkt"],
        )
        s = df[["p_win", "p_draw", "p_loss"]].astype(float).sum(axis=1)
        assert (np.abs(s - 1.0) < 1e-4).all()

    def test_probs_in_unit_interval(self, tmp_path: Path) -> None:
        """All ensemble_mkt probabilities must be in [0, 1]."""
        path = _make_fixtures_csv(tmp_path)
        df = predict_fixtures(
            "2026-06-15", fixtures_path=path,
            output_path=tmp_path / "out.csv",
            models=["ensemble_mkt"],
        )
        for col in ["p_win", "p_draw", "p_loss"]:
            vals = df[col].astype(float)
            assert (vals >= 0.0).all() and (vals <= 1.0).all()

    def test_no_odds_falls_back_to_ensemble(self, tmp_path: Path) -> None:
        """Without 2026 odds, ensemble_mkt W/D/L must equal ensemble W/D/L."""
        path = _make_fixtures_csv(tmp_path)
        df = predict_fixtures(
            "2026-06-15", fixtures_path=path,
            output_path=tmp_path / "out.csv",
            models=["ensemble", "ensemble_mkt"],
        )
        ens = df[df["model"] == "ensemble"].set_index(["team_a", "team_b"]).sort_index()
        mkt = df[df["model"] == "ensemble_mkt"].set_index(["team_a", "team_b"]).sort_index()
        for col in ["p_win", "p_draw", "p_loss"]:
            np.testing.assert_allclose(
                ens[col].astype(float).values,
                mkt[col].astype(float).values,
                atol=1e-6,
            )

    def test_synthetic_odds_blend(self, tmp_path: Path) -> None:
        """Synthetic odds injected into odds_lookup blend by exactly state['odds_alpha']."""
        from wcpredictor.predict import _build_frozen_state, _predict_one_frozen

        state = _build_frozen_state("2026-06-15", ["ensemble", "ensemble_mkt"])
        ta, tb = "Brazil", "Mexico"

        odds_pw, odds_pd, odds_pl = 0.60, 0.25, 0.15
        alpha = 0.3
        state["odds_lookup"] = {(ta, tb): (odds_pw, odds_pd, odds_pl)}
        state["odds_alpha"] = alpha

        ens_pred = _predict_one_frozen(state, "ensemble", ta, tb, True)
        mkt_pred = _predict_one_frozen(state, "ensemble_mkt", ta, tb, True)

        for ens_p, mkt_p, odds_p in [
            (ens_pred["p_win"], mkt_pred["p_win"], odds_pw),
            (ens_pred["p_draw"], mkt_pred["p_draw"], odds_pd),
            (ens_pred["p_loss"], mkt_pred["p_loss"], odds_pl),
        ]:
            expected = (1 - alpha) * ens_p + alpha * odds_p
            assert abs(mkt_p - expected) < 1e-5, f"blend mismatch: {mkt_p} != {expected}"

    def test_multi_model_rows_with_ensemble_mkt(self, tmp_path: Path) -> None:
        """predict_fixtures with 3 models → 3× rows, correct model tags, 0 bad-sum rows."""
        path = _make_fixtures_csv(tmp_path)
        df = predict_fixtures(
            "2026-06-15", fixtures_path=path,
            output_path=tmp_path / "out.csv",
            models=["poisson", "ensemble", "ensemble_mkt"],
        )
        assert len(df) == 6  # 2 upcoming fixtures × 3 models
        assert set(df["model"].unique()) == {"poisson", "ensemble", "ensemble_mkt"}
        s = df[["p_win", "p_draw", "p_loss"]].astype(float).sum(axis=1)
        assert int((np.abs(s - 1.0) > 1e-4).sum()) == 0

    def test_alpha_in_unit_interval(self, tmp_path: Path) -> None:
        """Stored odds_alpha must be in [0, 1]."""
        from wcpredictor.predict import _build_frozen_state
        state = _build_frozen_state("2026-06-15", ["ensemble_mkt"])
        assert 0.0 <= state["odds_alpha"] <= 1.0

    def test_model_tag(self, tmp_path: Path) -> None:
        """All rows from ensemble_mkt must carry model='ensemble_mkt'."""
        path = _make_fixtures_csv(tmp_path)
        df = predict_fixtures(
            "2026-06-15", fixtures_path=path,
            output_path=tmp_path / "out.csv",
            models=["ensemble_mkt"],
        )
        assert (df["model"] == "ensemble_mkt").all()


def test_load_predictions_model_dedup(tmp_path: Path) -> None:
    """Two rows for same fixture but different models must NOT be deduped."""
    import wcpredictor.evaluation.live as live_mod

    pred_file = tmp_path / "wc2026_predictions_2026-06-10.csv"
    pd.DataFrame({
        "date": ["2026-06-15", "2026-06-15"],
        "team_a": ["Brazil", "Brazil"],
        "team_b": ["Mexico", "Mexico"],
        "p_win": [0.6, 0.55],
        "p_draw": [0.2, 0.22],
        "p_loss": [0.2, 0.23],
        "model": ["poisson", "ensemble"],
    }).to_csv(pred_file, index=False)

    orig_processed = live_mod.DATA_PROCESSED
    try:
        live_mod.DATA_PROCESSED = tmp_path
        df = live_mod._load_predictions()
    finally:
        live_mod.DATA_PROCESSED = orig_processed

    # Both rows must survive deduplication
    assert len(df) == 2
    assert set(df["model"].unique()) == {"poisson", "ensemble"}
