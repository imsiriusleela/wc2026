"""Tests for markets/value_scan.py (M4)."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _make_sgpools_csv(tmp_path: Path, rows: list[dict]) -> Path:
    p = tmp_path / "sgpools_offers.csv"
    fields = ["entered_at", "date", "team_a", "team_b", "market", "line", "side", "price", "source"]
    with open(p, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return p


def _tiny_matrix():
    """3×3 matrix: home favoured."""
    m = [[0.0] * 3 for _ in range(3)]
    m[0][0] = 0.1  # 0-0
    m[0][1] = 0.1  # 0-1
    m[1][0] = 0.4  # 1-0
    m[1][1] = 0.2  # 1-1
    m[2][0] = 0.2  # 2-0
    return m


def _make_fixtures_csv(tmp_path: Path, team_a: str, team_b: str, matrix: list, date: str = "2026-07-01") -> Path:
    p = tmp_path / "wc2026_predictions_ensemble_mkt.csv"
    import pandas as pd
    df = pd.DataFrame([{
        "team_a": team_a,
        "team_b": team_b,
        "date": date,
        "neutral": True,
        "model": "ensemble_mkt",
        "model_version": "test",
        "p_win": 0.6,
        "p_draw": 0.2,
        "p_loss": 0.2,
        "lambda_a": 1.5,
        "lambda_b": 0.8,
        "elo_a": 1650.0,
        "elo_b": 1550.0,
        "score_matrix": str(matrix),
        "top_scorelines": "[]",
    }])
    df.to_csv(p, index=False)
    return p


# ─── scan() basic tests ───────────────────────────────────────────────────────

def test_scan_empty_sgpools_returns_empty(monkeypatch, tmp_path):
    """Empty SG Pools offers → empty result."""
    import wcpredictor.data.sgpools as sgm
    import wcpredictor.markets.value_scan as vsm

    monkeypatch.setattr(sgm, "_CSV_PATH", tmp_path / "empty.csv")
    monkeypatch.setattr(vsm, "load_sgpools_offers", lambda: __import__("pandas").DataFrame())
    monkeypatch.setattr(vsm, "_load_frozen_matrices", lambda *a: {})

    result = vsm.scan(as_of="2026-07-01")
    assert result == []


def test_scan_played_fixture_excluded(monkeypatch, tmp_path):
    """Fixtures with date < as_of must not appear in scan."""
    import pandas as pd
    import wcpredictor.markets.value_scan as vsm

    sgp_row = {"entered_at": "2026-06-12T10:00:00Z", "date": "2026-06-10",
               "team_a": "Brazil", "team_b": "Germany",
               "market": "1x2", "line": 0.0, "side": "home", "price": 1.85, "source": "manual"}
    sgp_df = pd.DataFrame([sgp_row])
    sgp_df["date"] = pd.to_datetime(sgp_df["date"])

    monkeypatch.setattr(vsm, "load_sgpools_offers", lambda: sgp_df)
    monkeypatch.setattr(vsm, "_load_frozen_matrices",
                        lambda *a: {("Brazil", "Germany"): _tiny_matrix()})
    monkeypatch.setattr(vsm, "_load_consensus_offers", lambda: {})

    result = vsm.scan(as_of="2026-06-12")
    assert result == []


def test_scan_returns_ev_fields(monkeypatch, tmp_path):
    """scan() output must contain all required fields per bet."""
    import pandas as pd
    import wcpredictor.markets.value_scan as vsm

    sgp_row = {"entered_at": "2026-06-12T10:00:00Z", "date": "2026-07-01",
               "team_a": "Brazil", "team_b": "Germany",
               "market": "1x2", "line": 0.0, "side": "home", "price": 1.85, "source": "manual"}
    sgp_df = pd.DataFrame([sgp_row])
    sgp_df["date"] = pd.to_datetime(sgp_df["date"])

    monkeypatch.setattr(vsm, "load_sgpools_offers", lambda: sgp_df)
    monkeypatch.setattr(vsm, "_load_frozen_matrices",
                        lambda *a: {("Brazil", "Germany"): _tiny_matrix()})
    monkeypatch.setattr(vsm, "_load_consensus_offers", lambda: {})

    result = vsm.scan(as_of="2026-06-12", min_ev=-999.0, require_consensus=False)
    assert len(result) == 1
    row = result[0]
    for key in ("date", "team_a", "team_b", "market", "line", "side",
                "sgpools_price", "fair_model", "ev_model",
                "recommended", "recommended_stake", "confidence_flags"):
        assert key in row, f"Missing key: {key}"


def test_scan_no_matrix_warns_and_skips(monkeypatch):
    """Fixture with no frozen matrix must be skipped with a warning."""
    import pandas as pd
    import warnings
    import wcpredictor.markets.value_scan as vsm

    sgp_row = {"entered_at": "2026-06-12T10:00:00Z", "date": "2026-07-01",
               "team_a": "Brazil", "team_b": "Germany",
               "market": "1x2", "line": 0.0, "side": "home", "price": 1.85, "source": "manual"}
    sgp_df = pd.DataFrame([sgp_row])
    sgp_df["date"] = pd.to_datetime(sgp_df["date"])

    monkeypatch.setattr(vsm, "load_sgpools_offers", lambda: sgp_df)
    monkeypatch.setattr(vsm, "_load_frozen_matrices", lambda *a: {})  # no matrix
    monkeypatch.setattr(vsm, "_load_consensus_offers", lambda: {})

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = vsm.scan(as_of="2026-06-12")
    assert result == []
    assert any("matrix" in str(warning.message).lower() for warning in w)


def test_scan_double_confirmation_filter(monkeypatch):
    """Bet is not recommended if sgpools_price < consensus_fair."""
    import pandas as pd
    import wcpredictor.markets.value_scan as vsm

    sgp_row = {"entered_at": "2026-06-12T10:00:00Z", "date": "2026-07-01",
               "team_a": "Brazil", "team_b": "Germany",
               "market": "1x2", "line": 0.0, "side": "home", "price": 1.85, "source": "manual"}
    sgp_df = pd.DataFrame([sgp_row])
    sgp_df["date"] = pd.to_datetime(sgp_df["date"])

    monkeypatch.setattr(vsm, "load_sgpools_offers", lambda: sgp_df)
    monkeypatch.setattr(vsm, "_load_frozen_matrices",
                        lambda *a: {("Brazil", "Germany"): _tiny_matrix()})
    monkeypatch.setattr(vsm, "_load_consensus_offers", lambda: {})
    # Patch consensus_fair in the value_scan module's namespace (not edge module)
    monkeypatch.setattr(vsm, "consensus_fair", lambda *a, **kw: 2.10)

    result = vsm.scan(as_of="2026-06-12", min_ev=-999.0, require_consensus=True)
    home_bets = [r for r in result if r["side"] == "home"]
    assert len(home_bets) == 1
    assert not home_bets[0]["recommended"]
    assert "below_consensus_fair" in home_bets[0]["confidence_flags"]


def test_scan_recommended_has_nonzero_stake(monkeypatch):
    """Recommended bets in quarter-Kelly tier must have stake > 0."""
    import pandas as pd
    import wcpredictor.markets.value_scan as vsm
    import wcpredictor.markets.edge as edge_mod

    sgp_row = {"entered_at": "2026-06-12T10:00:00Z", "date": "2026-07-01",
               "team_a": "Brazil", "team_b": "Germany",
               "market": "1x2", "line": 0.0, "side": "home", "price": 2.20, "source": "manual"}
    sgp_df = pd.DataFrame([sgp_row])
    sgp_df["date"] = pd.to_datetime(sgp_df["date"])

    monkeypatch.setattr(vsm, "load_sgpools_offers", lambda: sgp_df)
    monkeypatch.setattr(vsm, "_load_frozen_matrices",
                        lambda *a: {("Brazil", "Germany"): _tiny_matrix()})
    # Consensus fair is 1.80 < 2.20 (sgpools beats it) → recommended
    monkeypatch.setattr(vsm, "_load_consensus_offers", lambda: {})
    monkeypatch.setattr(edge_mod, "consensus_fair", lambda *a, **kw: 1.80)

    # Patch sizing tier to quarter_kelly
    import wcpredictor.config as cfg
    monkeypatch.setattr(cfg, "EV_THRESHOLD", 0.0)

    result = vsm.scan(as_of="2026-06-12", min_ev=0.0, require_consensus=True)
    recs = [r for r in result if r["recommended"]]
    if recs:
        assert recs[0]["recommended_stake"] > 0
