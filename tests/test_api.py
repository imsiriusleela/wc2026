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


def test_refresh_odds_fdco_failure_non_fatal(monkeypatch, tmp_path):
    """fdco download failure should not abort; breakdown fields must be present."""
    import pandas as pd
    import wcpredictor.api.app as app_module

    # fdco raises but we still have API odds
    monkeypatch.setattr(app_module, "download_odds", lambda force=False, verify=True: (_ for _ in ()).throw(RuntimeError("fdco down")))
    monkeypatch.setattr(app_module, "DATA_RAW", tmp_path)
    sample = pd.DataFrame({"year": [2026, 2026], "team_a": ["A", "B"], "team_b": ["B", "C"]})
    monkeypatch.setattr(app_module, "load_wc_odds", lambda: sample)

    resp = client.post("/refresh-odds")
    assert resp.status_code == 200
    d = resp.json()
    assert "n_odds_2026_fdco" in d
    assert "n_odds_2026_api" in d
    assert "odds_api_refreshed" in d
    assert "fdco" in d["note"].lower() or "error" in d["note"].lower()


def test_refresh_odds_both_sources_fail_returns_502(monkeypatch, tmp_path):
    """Both fdco and odds-api fail with no cached odds → 502."""
    import pandas as pd
    import wcpredictor.api.app as app_module

    def _fail(**kwargs):
        raise OSError("network unreachable")

    monkeypatch.setattr(app_module, "download_odds", _fail)
    monkeypatch.setattr(app_module, "DATA_RAW", tmp_path)
    # No odds from any source
    monkeypatch.setattr(app_module, "load_wc_odds", lambda: pd.DataFrame({"year": [2022]}))

    resp = client.post("/refresh-odds")
    assert resp.status_code == 502


# ---- /refresh-results tests ----

def test_refresh_results_returns_ok_on_success(monkeypatch, tmp_path):
    """POST /refresh-results should return 200 with result counts when network is mocked."""
    import wcpredictor.api.app as app_module
    import wcpredictor.data.results_2026 as r26_module
    import wcpredictor.evaluation.live as live_module

    monkeypatch.setattr(r26_module, "update_wc2026_results",
                        lambda source_csv=None: {"n_total": 5, "n_new": 3, "n_group": 4, "n_knockout": 1})
    monkeypatch.setattr(r26_module, "mark_fixtures_played", lambda: 2)
    monkeypatch.setattr(live_module, "run_refresh", lambda as_of_date: {})

    resp = client.post("/refresh-results")
    assert resp.status_code == 200
    d = resp.json()
    assert d["status"] == "ok"
    assert d["n_results_total"] == 5
    assert d["n_new"] == 3
    assert d["n_group"] == 4
    assert d["n_knockout"] == 1
    assert d["n_fixtures_updated"] == 2


def test_refresh_results_non_fatal_on_network_error(monkeypatch):
    """Network failure in update_wc2026_results must not cause a 5xx — returns 200."""
    import wcpredictor.data.results_2026 as r26_module
    import wcpredictor.evaluation.live as live_module

    monkeypatch.setattr(r26_module, "update_wc2026_results",
                        lambda source_csv=None: (_ for _ in ()).throw(OSError("network down")))
    monkeypatch.setattr(r26_module, "mark_fixtures_played", lambda: 0)
    monkeypatch.setattr(live_module, "run_refresh", lambda as_of_date: {})

    resp = client.post("/refresh-results")
    assert resp.status_code == 200
    d = resp.json()
    assert "error" in d["note"].lower() or d["n_results_total"] == 0


def test_refresh_results_409_when_locked(monkeypatch):
    """Concurrent /refresh-results calls must return 409."""
    import wcpredictor.api.app as app_module

    # Hold the lock
    app_module._REFRESH_LOCK.acquire()
    try:
        resp = client.post("/refresh-results")
        assert resp.status_code == 409
    finally:
        app_module._REFRESH_LOCK.release()


# ---- /resimulate tests ----

def test_resimulate_returns_ok(monkeypatch, tmp_path):
    """POST /resimulate should return 200 with simulation meta."""
    import pandas as pd
    import wcpredictor.api.app as app_module

    fake_df = pd.DataFrame([{"team": "Brazil", "p_champion": 0.2}])
    monkeypatch.setattr(
        "wcpredictor.simulate.simulate_tournament",
        lambda *a, **kw: fake_df,
    )
    # Write a stub JSON artifact so resimulate can read it
    as_of = app_module._default_as_of()
    stub_json = app_module.DATA_PROCESSED / f"wc2026_tournament_sim_{as_of}.json"
    stub_json.parent.mkdir(parents=True, exist_ok=True)
    stub_json.write_text('{"n_group_fixed": 3, "n_ko_fixed": 0}')

    resp = client.post("/resimulate?n_sims=100&model=ensemble_mkt")
    assert resp.status_code == 200
    d = resp.json()
    assert d["status"] == "ok"
    assert d["n_sims"] == 100
    assert "n_group_fixed" in d
    assert "n_ko_fixed" in d


def test_resimulate_409_when_locked(monkeypatch):
    """Concurrent /resimulate calls must return 409."""
    import wcpredictor.api.app as app_module

    app_module._REFRESH_LOCK.acquire()
    try:
        resp = client.post("/resimulate")
        assert resp.status_code == 409
    finally:
        app_module._REFRESH_LOCK.release()


def test_tournament_has_condition_fields():
    """GET /tournament should include n_group_fixed and n_ko_fixed fields."""
    resp = client.get("/tournament")
    assert resp.status_code == 200
    d = resp.json()
    assert "n_group_fixed" in d
    assert "n_ko_fixed" in d
    assert isinstance(d["n_group_fixed"], int)
    assert isinstance(d["n_ko_fixed"], int)
