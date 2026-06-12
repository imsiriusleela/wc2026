"""Tests for data/bets.py — ledger record/settle/CLV/stop-rules (M5)."""
from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _write_csv(tmp_path: Path, rows: list[dict]) -> Path:
    p = tmp_path / "bets.csv"
    if not rows:
        p.write_text("")
        return p
    fields = list(rows[0].keys())
    with open(p, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return p


# ─── record_bet ────────────────────────────────────────────────────────────────

def test_record_bet_appends_row(tmp_path):
    from wcpredictor.data.bets import record_bet, load_ledger
    p = tmp_path / "bets.csv"
    record_bet("Brazil", "Germany", "2026-06-20", "1x2", 0.0, "home",
               price_taken=1.85, stake=50.0, path=p)
    df = load_ledger(p)
    assert len(df) == 1
    assert df.iloc[0]["status"] == "open"
    assert df.iloc[0]["market"] == "1x2"
    assert abs(df.iloc[0]["stake"] - 50.0) < 1e-9


def test_record_bet_validates_price(tmp_path):
    from wcpredictor.data.bets import record_bet
    p = tmp_path / "bets.csv"
    with pytest.raises(ValueError, match="price_taken"):
        record_bet("Brazil", "Germany", "2026-06-20", "1x2", 0.0, "home",
                   price_taken=0.90, stake=50.0, path=p)


def test_record_bet_validates_stake(tmp_path):
    from wcpredictor.data.bets import record_bet
    p = tmp_path / "bets.csv"
    with pytest.raises(ValueError, match="stake"):
        record_bet("Brazil", "Germany", "2026-06-20", "1x2", 0.0, "home",
                   price_taken=1.85, stake=0.0, path=p)


# ─── Settlement: 1X2 ─────────────────────────────────────────────────────────

def test_settle_1x2_home_win(tmp_path):
    from wcpredictor.data.bets import record_bet, settle_open_bets, load_ledger
    p = tmp_path / "bets.csv"
    record_bet("Brazil", "Germany", "2026-06-20", "1x2", 0.0, "home",
               price_taken=1.85, stake=100.0, path=p)
    n = settle_open_bets({"Brazil vs Germany": (2, 0)}, path=p)
    assert n == 1
    df = load_ledger(p)
    assert df.iloc[0]["status"] == "won"
    assert abs(df.iloc[0]["pnl"] - (1.85 - 1.0) * 100) < 1e-6


def test_settle_1x2_away_win(tmp_path):
    from wcpredictor.data.bets import record_bet, settle_open_bets, load_ledger
    p = tmp_path / "bets.csv"
    record_bet("Brazil", "Germany", "2026-06-20", "1x2", 0.0, "home",
               price_taken=1.85, stake=100.0, path=p)
    n = settle_open_bets({"Brazil vs Germany": (0, 1)}, path=p)
    assert n == 1
    df = load_ledger(p)
    assert df.iloc[0]["status"] == "lost"
    assert abs(df.iloc[0]["pnl"] - (-100.0)) < 1e-6


def test_settle_1x2_draw_bet_wins(tmp_path):
    from wcpredictor.data.bets import record_bet, settle_open_bets, load_ledger
    p = tmp_path / "bets.csv"
    record_bet("Brazil", "Germany", "2026-06-20", "1x2", 0.0, "draw",
               price_taken=3.40, stake=30.0, path=p)
    n = settle_open_bets({"Brazil vs Germany": (1, 1)}, path=p)
    df = load_ledger(p)
    assert df.iloc[0]["status"] == "won"
    assert abs(df.iloc[0]["pnl"] - (3.40 - 1.0) * 30.0) < 1e-6


# ─── Settlement: AH ──────────────────────────────────────────────────────────

def test_settle_ah_full_win(tmp_path):
    """AH home -0.5, result 2-0 → home wins by 2 > 0.5 → full win."""
    from wcpredictor.data.bets import record_bet, settle_open_bets, load_ledger
    p = tmp_path / "bets.csv"
    record_bet("Brazil", "Germany", "2026-06-20", "ah", -0.5, "home",
               price_taken=1.90, stake=100.0, path=p)
    settle_open_bets({"Brazil vs Germany": (2, 0)}, path=p)
    df = load_ledger(p)
    assert df.iloc[0]["status"] == "won"
    assert abs(df.iloc[0]["pnl"] - (1.90 - 1.0) * 100) < 1e-6


def test_settle_ah_push(tmp_path):
    """AH home -1.0, result 1-0 → diff=1 exactly → push."""
    from wcpredictor.data.bets import record_bet, settle_open_bets, load_ledger
    p = tmp_path / "bets.csv"
    record_bet("Brazil", "Germany", "2026-06-20", "ah", -1.0, "home",
               price_taken=1.90, stake=100.0, path=p)
    settle_open_bets({"Brazil vs Germany": (1, 0)}, path=p)
    df = load_ledger(p)
    assert df.iloc[0]["status"] == "push"
    assert abs(df.iloc[0]["pnl"]) < 1e-6


def test_settle_ah_half_win(tmp_path):
    """AH home -0.75, result 1-0 → diff=1 > 0.5 (win) and = 1.0 (push) → half_won."""
    from wcpredictor.data.bets import record_bet, settle_open_bets, load_ledger
    p = tmp_path / "bets.csv"
    record_bet("Brazil", "Germany", "2026-06-20", "ah", -0.75, "home",
               price_taken=1.92, stake=100.0, path=p)
    settle_open_bets({"Brazil vs Germany": (1, 0)}, path=p)
    df = load_ledger(p)
    assert df.iloc[0]["status"] == "half_won"
    assert df.iloc[0]["pnl"] > 0  # half win = profit


def test_settle_ah_half_lost(tmp_path):
    """AH home -0.25, result 0-0 → diff=0, vs threshold 0.25 → half loss."""
    from wcpredictor.data.bets import record_bet, settle_open_bets, load_ledger
    p = tmp_path / "bets.csv"
    record_bet("Brazil", "Germany", "2026-06-20", "ah", -0.25, "home",
               price_taken=1.92, stake=100.0, path=p)
    settle_open_bets({"Brazil vs Germany": (0, 0)}, path=p)
    df = load_ledger(p)
    assert df.iloc[0]["status"] == "half_lost"
    assert abs(df.iloc[0]["pnl"] - (-50.0)) < 1e-6


# ─── Settlement: totals ───────────────────────────────────────────────────────

def test_settle_total_over_win(tmp_path):
    """Over 2.5 stake, result 2-1 (total=3) → win."""
    from wcpredictor.data.bets import record_bet, settle_open_bets, load_ledger
    p = tmp_path / "bets.csv"
    record_bet("Brazil", "Germany", "2026-06-20", "total", 2.5, "over",
               price_taken=1.88, stake=50.0, path=p)
    settle_open_bets({"Brazil vs Germany": (2, 1)}, path=p)
    df = load_ledger(p)
    assert df.iloc[0]["status"] == "won"


def test_settle_total_under_win(tmp_path):
    """Under 2.5 stake, result 0-1 (total=1) → win."""
    from wcpredictor.data.bets import record_bet, settle_open_bets, load_ledger
    p = tmp_path / "bets.csv"
    record_bet("Brazil", "Germany", "2026-06-20", "total", 2.5, "under",
               price_taken=1.92, stake=50.0, path=p)
    settle_open_bets({"Brazil vs Germany": (0, 1)}, path=p)
    df = load_ledger(p)
    assert df.iloc[0]["status"] == "won"


# ─── Already-settled bets not re-settled ──────────────────────────────────────

def test_already_settled_not_changed(tmp_path):
    from wcpredictor.data.bets import record_bet, settle_open_bets, load_ledger
    p = tmp_path / "bets.csv"
    record_bet("Brazil", "Germany", "2026-06-20", "1x2", 0.0, "home",
               price_taken=1.85, stake=100.0, path=p)
    settle_open_bets({"Brazil vs Germany": (1, 0)}, path=p)
    n = settle_open_bets({"Brazil vs Germany": (0, 1)}, path=p)  # different result
    assert n == 0  # not re-settled
    df = load_ledger(p)
    assert df.iloc[0]["status"] == "won"  # unchanged


# ─── CLV computation ─────────────────────────────────────────────────────────

def test_update_closing_consensus_fills_clv(tmp_path):
    from wcpredictor.data.bets import record_bet, update_closing_consensus, load_ledger
    p = tmp_path / "bets.csv"
    record_bet("Brazil", "Germany", "2026-06-20", "1x2", 0.0, "home",
               price_taken=1.90, stake=100.0, path=p)
    update_closing_consensus("Brazil vs Germany", "1x2", 0.0, "home",
                              closing_price=1.80, path=p)
    df = load_ledger(p)
    assert not math.isnan(float(df.iloc[0]["clv_pct"]))
    expected_clv = round(1.90 / 1.80 - 1.0, 4)
    assert abs(float(df.iloc[0]["clv_pct"]) - expected_clv) < 1e-4


# ─── P&L summary / stop rules ────────────────────────────────────────────────

def test_ledger_summary_empty(tmp_path):
    from wcpredictor.data.bets import ledger_summary
    s = ledger_summary(tmp_path / "missing.csv")
    assert s["total_bets"] == 0
    assert s["stop_rule_triggered"] is False


def test_ledger_summary_roi(tmp_path):
    from wcpredictor.data.bets import record_bet, settle_open_bets, ledger_summary
    p = tmp_path / "bets.csv"
    record_bet("Brazil", "Germany", "2026-06-20", "1x2", 0.0, "home",
               price_taken=2.00, stake=100.0, path=p)
    record_bet("France", "Spain", "2026-06-21", "1x2", 0.0, "home",
               price_taken=2.00, stake=100.0, path=p)
    settle_open_bets({"Brazil vs Germany": (1, 0), "France vs Spain": (0, 1)}, path=p)
    s = ledger_summary(p)
    # One win (+100) one loss (-100) → pnl=0, roi=0
    assert s["total_pnl"] == 0.0
    assert abs(s["roi"]) < 1e-6


def test_stop_clv_triggered(tmp_path):
    """If trailing CLV is negative and we have STOP_TRAILING_BETS records, stop."""
    from wcpredictor.data.bets import load_ledger, ledger_summary, _write_ledger
    from wcpredictor.config import STOP_TRAILING_BETS

    p = tmp_path / "bets.csv"
    rows = []
    for i in range(STOP_TRAILING_BETS + 2):
        rows.append({
            "placed_at": f"2026-06-{i+1:02d}T10:00:00Z",
            "date": f"2026-06-{i+1:02d}",
            "match": f"Team{i} vs Team{i+1}",
            "market": "1x2",
            "line": 0.0,
            "side": "home",
            "price_taken": 1.90,
            "stake": 100.0,
            "status": "lost",
            "pnl": -100.0,
            "consensus_fair_at_placement": 1.85,
            "closing_consensus_fair": 1.95,
            "clv_pct": -0.03,  # negative CLV
        })
    _write_ledger(rows, p)
    s = ledger_summary(p)
    assert s["stop_clv"] is True
    assert s["stop_rule_triggered"] is True


def test_stop_drawdown_triggered(tmp_path):
    """Drawdown > STOP_DRAWDOWN_UNITS triggers stop."""
    from wcpredictor.data.bets import _write_ledger, ledger_summary
    from wcpredictor.config import STOP_DRAWDOWN_UNITS

    p = tmp_path / "bets.csv"
    rows = []
    n_losses = int(STOP_DRAWDOWN_UNITS) + 2
    for i in range(n_losses):
        rows.append({
            "placed_at": f"2026-06-{i+1:02d}T10:00:00Z",
            "date": f"2026-06-{i+1:02d}",
            "match": f"Team{i} vs Team{i+1}",
            "market": "1x2",
            "line": 0.0,
            "side": "home",
            "price_taken": 2.00,
            "stake": 100.0,
            "status": "lost",
            "pnl": -100.0,
            "consensus_fair_at_placement": "",
            "closing_consensus_fair": "",
            "clv_pct": "",
        })
    _write_ledger(rows, p)
    s = ledger_summary(p)
    assert s["stop_drawdown"] is True
    assert s["stop_rule_triggered"] is True
