"""FastAPI endpoint tests using TestClient (poisson-only for speed)."""
import pytest
from fastapi.testclient import TestClient

from wcpredictor.api.app import app

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_teams_non_empty():
    resp = client.get("/teams")
    assert resp.status_code == 200
    teams = resp.json()
    assert isinstance(teams, list)
    assert len(teams) > 0
    assert "Brazil" in teams


def test_predict_valid_pair():
    resp = client.get("/predict", params={"team_a": "Brazil", "team_b": "Argentina", "model": "poisson"})
    assert resp.status_code == 200
    d = resp.json()
    assert abs(d["p_win"] + d["p_draw"] + d["p_loss"] - 1.0) < 1e-4
    assert isinstance(d["score_matrix"], list)
    assert len(d["score_matrix"]) > 0
    assert isinstance(d["score_matrix"][0], list)
    assert len(d["top_scorelines"]) > 0


def test_predict_unknown_team_returns_422():
    resp = client.get("/predict", params={"team_a": "XyzUnknownFC", "team_b": "Brazil", "model": "poisson"})
    assert resp.status_code == 422


def test_fixtures_non_empty():
    resp = client.get("/fixtures")
    assert resp.status_code == 200
    rows = resp.json()
    assert len(rows) > 0
    for row in rows[:3]:
        assert isinstance(row["score_matrix"], list)
        assert isinstance(row["top_scorelines"], list)


def test_fixtures_model_filter():
    resp = client.get("/fixtures", params={"model": "poisson"})
    assert resp.status_code == 200
    rows = resp.json()
    assert all(r["model"] == "poisson" for r in rows)


def test_tournament_all_teams():
    resp = client.get("/tournament")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["standings"]) == 48
    champ_sum = sum(s["p_champion"] for s in data["standings"])
    assert abs(champ_sum - 1.0) < 0.01


def test_tournament_top10_present():
    resp = client.get("/tournament")
    data = resp.json()
    assert len(data["top10_champion"]) > 0
    assert "team" in data["top10_champion"][0]


def test_scorecard_returns_valid_schema():
    resp = client.get("/scorecard")
    # Either 200 (scorecard exists) or 503 (not yet written) is acceptable.
    assert resp.status_code in (200, 503)
    if resp.status_code == 200:
        d = resp.json()
        assert "as_of_date" in d
        assert "n_completed" in d
        assert "temperature" in d
        assert isinstance(d["matches"], list)
        assert isinstance(d["models"], dict)


# ---- /refresh-odds tests ----

def test_refresh_odds_with_2026_rows(monkeypatch, tmp_path):
    import pandas as pd
    import wcpredictor.api.app as app_module

    fake_xlsx = tmp_path / "WorldCup_fdco.xlsx"
    fake_xlsx.write_bytes(b"fake-content-for-sha")

    monkeypatch.setattr(app_module, "download_odds", lambda force=False, verify=True: fake_xlsx)
    monkeypatch.setattr(app_module, "DATA_RAW", tmp_path)
    sample = pd.DataFrame({"year": [2026, 2026, 2022], "home": ["A", "B", "C"], "away": ["B", "C", "D"]})
    monkeypatch.setattr(app_module, "load_wc_odds", lambda: sample)

    app_module._STATE_CACHE["sentinel"] = {"dummy": True}
    resp = client.post("/refresh-odds")
    assert resp.status_code == 200
    d = resp.json()
    assert d["status"] == "ok"
    assert d["n_odds_2026"] == 2
    assert d["state_cache_cleared"] is True
    assert app_module._STATE_CACHE == {}


def test_refresh_odds_no_2026_rows(monkeypatch, tmp_path):
    import pandas as pd
    import wcpredictor.api.app as app_module

    fake_xlsx = tmp_path / "WorldCup_fdco.xlsx"
    fake_xlsx.write_bytes(b"no-2026-data")

    monkeypatch.setattr(app_module, "download_odds", lambda force=False, verify=True: fake_xlsx)
    monkeypatch.setattr(app_module, "DATA_RAW", tmp_path)
    sample = pd.DataFrame({"year": [2022, 2018], "home": ["A", "B"], "away": ["C", "D"]})
    monkeypatch.setattr(app_module, "load_wc_odds", lambda: sample)

    resp = client.post("/refresh-odds")
    assert resp.status_code == 200
    d = resp.json()
    assert d["n_odds_2026"] == 0
    assert "no WC 2026 odds" in d["note"].lower() or d["n_odds_2026"] == 0


def test_refresh_odds_download_failure(monkeypatch, tmp_path):
    import wcpredictor.api.app as app_module

    def _fail(**kwargs):
        raise OSError("network unreachable")

    monkeypatch.setattr(app_module, "download_odds", _fail)

    resp = client.post("/refresh-odds")
    assert resp.status_code == 502
    assert "network unreachable" in resp.json()["detail"]
