"""Tests for the ODDS_ALPHA_CAP serve-time enforcement in _resolve_odds_alpha."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch


def _write_report(path: Path, content: dict) -> None:
    path.write_text(json.dumps(content))


def test_pooled_above_cap_is_clamped():
    """Pooled alpha 0.64 (typical unconstrained fit) should resolve to 0.3."""
    from wcpredictor.config import ODDS_ALPHA_CAP

    with tempfile.TemporaryDirectory() as tmp:
        report_path = Path(tmp) / "backtest_report.json"
        _write_report(report_path, {"odds_alpha_pooled": 0.6388})
        with patch("wcpredictor.predict.DATA_PROCESSED", Path(tmp)):
            from wcpredictor.predict import _resolve_odds_alpha
            assert _resolve_odds_alpha() == ODDS_ALPHA_CAP


def test_pooled_below_cap_passes_through():
    """Pooled alpha 0.10 is already below cap — should pass through unchanged."""
    with tempfile.TemporaryDirectory() as tmp:
        report_path = Path(tmp) / "backtest_report.json"
        _write_report(report_path, {"odds_alpha_pooled": 0.10})
        with patch("wcpredictor.predict.DATA_PROCESSED", Path(tmp)):
            from wcpredictor.predict import _resolve_odds_alpha
            assert abs(_resolve_odds_alpha() - 0.10) < 1e-9


def test_missing_report_returns_prior():
    """No report on disk → fall back to ODDS_ALPHA_PRIOR (0.0)."""
    with tempfile.TemporaryDirectory() as tmp:
        with patch("wcpredictor.predict.DATA_PROCESSED", Path(tmp)):
            from wcpredictor.predict import _resolve_odds_alpha
            assert _resolve_odds_alpha() == 0.0


def test_cap_constant_value():
    """Sanity-check that ODDS_ALPHA_CAP is exactly 0.3."""
    from wcpredictor.config import ODDS_ALPHA_CAP
    assert ODDS_ALPHA_CAP == 0.3
