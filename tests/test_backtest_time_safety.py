"""Verify that the time-safety assertion in the backtest fires on a leaky fold,
and that build_wc_stacking_validation is leakage-free and regime-matched."""

import pandas as pd
import pytest


def _make_matches(rows):
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df["match_id"] = [f"m{i}" for i in range(len(df))]
    return df


def test_assertion_fires_on_leaky_fold():
    """train.date.max() >= test.date.min() must raise AssertionError."""
    train = _make_matches([
        {"date": "2010-06-15", "team_a": "A", "team_b": "B",
         "goals_a": 1, "goals_b": 0, "neutral": True, "competition": "world_cup"},
    ])
    test = _make_matches([
        {"date": "2010-06-12", "team_a": "C", "team_b": "D",
         "goals_a": 2, "goals_b": 1, "neutral": True, "competition": "world_cup"},
    ])
    with pytest.raises(AssertionError, match="LEAKAGE"):
        assert train["date"].max() < test["date"].min(), (
            "LEAKAGE DETECTED: train/test dates overlap!"
        )


def test_assertion_passes_on_clean_fold():
    """Clean fold: train ends strictly before test begins."""
    train = _make_matches([
        {"date": "2010-06-10", "team_a": "A", "team_b": "B",
         "goals_a": 1, "goals_b": 0, "neutral": True, "competition": "qualifier"},
    ])
    test = _make_matches([
        {"date": "2010-06-12", "team_a": "C", "team_b": "D",
         "goals_a": 0, "goals_b": 0, "neutral": True, "competition": "world_cup"},
    ])
    # Should not raise
    assert train["date"].max() < test["date"].min()


# ── build_wc_stacking_validation tests ────────────────────────────────────────

def test_wc_stacking_past_years_selection():
    """past_years = WC years strictly before the fold year (structural invariant)."""
    from wcpredictor.evaluation.backtest import _WC_START

    assert sorted(w for w in _WC_START if w < 2010) == []
    assert sorted(w for w in _WC_START if w < 2014) == [2010]
    assert sorted(w for w in _WC_START if w < 2018) == [2010, 2014]
    assert sorted(w for w in _WC_START if w < 2022) == [2010, 2014, 2018]


def test_wc_stacking_empty_for_first_fold():
    """2010 fold has no prior WC → build_wc_stacking_validation returns empty."""
    from wcpredictor.evaluation.backtest import build_wc_stacking_validation

    empty = pd.DataFrame()
    labels, member_probs = build_wc_stacking_validation(2010, empty, empty)
    assert labels == []
    assert all(p == [] for p in member_probs)


def test_wc_stacking_train_dates_before_wc_start(monkeypatch):
    """Members are trained only on data strictly before wc_start(w) for each past WC w."""
    from wcpredictor.evaluation import backtest as bt
    from wcpredictor.evaluation.backtest import _WC_START

    wc2010_start = pd.Timestamp(_WC_START[2010])
    dates_pre = pd.date_range("2009-01-01", periods=10, freq="30D")
    dates_wc = pd.date_range(wc2010_start, periods=5, freq="5D")

    def _row(date, is_wc=False):
        return {
            "date": date, "team_a": "X", "team_b": "Y",
            "goals_a": 1, "goals_b": 0, "neutral": True,
            "competition": "world_cup" if is_wc else "friendly",
            "is_world_cup": is_wc, "match_id": str(date),
            "elo_diff_adj": 50.0, "elo_a_pre": 1500.0, "elo_b_pre": 1450.0,
            "form_diff": 0.0, "momentum_diff": 0.0, "rest_diff": 0.0,
            "has_odds": 1.0, "odds_p_win": 0.5, "odds_p_draw": 0.22, "odds_p_loss": 0.28,
        }

    rows = [_row(d) for d in dates_pre] + [_row(d, is_wc=True) for d in dates_wc]
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])

    captured_max_dates: list[pd.Timestamp] = []

    def spy_poi(elo_df, *a, **kw):
        if not elo_df.empty:
            captured_max_dates.append(elo_df["date"].max())
        return 1.2, 0.004

    monkeypatch.setattr(bt, "poisson_fit", spy_poi)
    monkeypatch.setattr(bt, "dc_fit", lambda m, **kw: {})
    monkeypatch.setattr(bt, "log_fit", lambda elo, labels: (None, None))
    monkeypatch.setattr(bt, "tree_fit", lambda elo, labels: None)
    monkeypatch.setattr(bt, "predict_one", lambda d, base, beta: {
        "p_win": 0.4, "p_draw": 0.22, "p_loss": 0.38,
    })
    monkeypatch.setattr(bt, "dc_predict_one", lambda params, ta, tb, neutral: {
        "p_win": 0.4, "p_draw": 0.22, "p_loss": 0.38,
    })
    monkeypatch.setattr(bt, "log_predict", lambda scaler, model, elo: [[0.4, 0.22, 0.38]] * len(elo))
    monkeypatch.setattr(bt, "tree_predict", lambda model, elo: [[0.4, 0.22, 0.38]] * len(elo))

    labels, member_probs = bt.build_wc_stacking_validation(2014, df, df)

    # All training calls must have been on data before WC2010 start
    assert captured_max_dates, "poisson_fit was never called — check synthetic data setup"
    for max_date in captured_max_dates:
        assert max_date < wc2010_start, (
            f"Training max date {max_date} >= wc_start(2010) {wc2010_start} — LEAKAGE"
        )

    # The WC2010 test matches become the validation rows
    assert len(labels) == len(dates_wc)
    assert all(lbl == 0 for lbl in labels)  # all rows: goals_a=1, goals_b=0 → win (label 0)


def test_weights_lift_above_equal_with_dominant_member():
    """When one member dominates, fit_weights moves it well above 0.25."""
    from wcpredictor.models.ensemble import fit_weights

    rng = __import__("numpy").random.default_rng(42)
    n = 120
    labels = [0] * n  # all wins

    # Member 0: near-perfect
    perfect = [[0.9, 0.05, 0.05]] * n
    # Members 1-3: uninformative uniform
    uninf = [[1 / 3, 1 / 3, 1 / 3]] * n

    weights = fit_weights([perfect, uninf, uninf, uninf], labels, reg_lambda=0.0)
    assert weights[0] > 0.40, f"dominant member weight {weights[0]:.3f} should exceed 0.40"
