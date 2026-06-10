"""Tests for AH_ALPHA_CAP serve-time enforcement and _fit_ah_alpha optimiser.

Mirrors test_odds_alpha_cap.py for the AH blend pathway.
"""
from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from unittest.mock import patch


def _write_report(path: Path, content: dict) -> None:
    path.write_text(json.dumps(content))


# ─── _resolve_ah_alpha ─────────────────────────────────────────────────────────

def test_ah_pooled_above_cap_is_clamped():
    """ah_alpha_pooled 0.50 should resolve to AH_ALPHA_CAP (0.3)."""
    from wcpredictor.config import AH_ALPHA_CAP

    with tempfile.TemporaryDirectory() as tmp:
        report_path = Path(tmp) / "backtest_report.json"
        _write_report(report_path, {"ah_alpha_pooled": 0.50})
        with patch("wcpredictor.predict.DATA_PROCESSED", Path(tmp)):
            from wcpredictor.predict import _resolve_ah_alpha
            assert _resolve_ah_alpha() == AH_ALPHA_CAP


def test_ah_pooled_below_cap_passes_through():
    """ah_alpha_pooled 0.12 is already below cap — should pass through unchanged."""
    with tempfile.TemporaryDirectory() as tmp:
        report_path = Path(tmp) / "backtest_report.json"
        _write_report(report_path, {"ah_alpha_pooled": 0.12})
        with patch("wcpredictor.predict.DATA_PROCESSED", Path(tmp)):
            from wcpredictor.predict import _resolve_ah_alpha
            assert abs(_resolve_ah_alpha() - 0.12) < 1e-9


def test_ah_missing_report_returns_prior():
    """No report on disk → fall back to AH_ALPHA_PRIOR (0.0)."""
    with tempfile.TemporaryDirectory() as tmp:
        with patch("wcpredictor.predict.DATA_PROCESSED", Path(tmp)):
            from wcpredictor.predict import _resolve_ah_alpha
            assert _resolve_ah_alpha() == 0.0


def test_ah_cap_constant_value():
    """Sanity-check that AH_ALPHA_CAP is exactly 0.3."""
    from wcpredictor.config import AH_ALPHA_CAP
    assert AH_ALPHA_CAP == 0.3


def test_ah_prior_constant_value():
    """Sanity-check that AH_ALPHA_PRIOR is exactly 0.0."""
    from wcpredictor.config import AH_ALPHA_PRIOR
    assert AH_ALPHA_PRIOR == 0.0


# ─── _fit_ah_alpha ─────────────────────────────────────────────────────────────

def test_fit_ah_alpha_returns_float_in_unit_interval():
    """_fit_ah_alpha always returns a value in [0, 1]."""
    from wcpredictor.evaluation.backtest import _fit_ah_alpha

    true_a   = [2, 1, 0, 3, 1]
    true_b   = [1, 1, 2, 1, 0]
    model_p  = [0.62, 0.50, 0.35, 0.75, 0.55]
    market_p = [0.58, 0.52, 0.38, 0.72, 0.60]
    thresholds = [0.5, 0.5, 0.5, 0.5, 0.5]  # AH -0.5 for all

    alpha = _fit_ah_alpha(true_a, true_b, model_p, market_p, thresholds)
    assert 0.0 <= alpha <= 1.0


def test_fit_ah_alpha_perfect_market_converges_to_one():
    """When the market exactly recovers the true outcomes, alpha→1."""
    from wcpredictor.evaluation.backtest import _fit_ah_alpha
    from wcpredictor.evaluation.metrics import ah_cover_brier

    # Construct outcomes where market_p is a perfect probabilistic forecast
    # and model_p is noisy — optimal alpha should be near 1.
    true_a     = [2, 0, 1, 3, 0] * 10
    true_b     = [0, 2, 1, 0, 1] * 10
    thresholds = [0.5] * len(true_a)
    # market_p closely matches realised cover (outcome at 0.5 threshold)
    market_p   = [0.95, 0.05, 0.50, 0.95, 0.05] * 10
    # model_p is much less certain
    model_p    = [0.55, 0.45, 0.50, 0.60, 0.40] * 10

    alpha = _fit_ah_alpha(true_a, true_b, model_p, market_p, thresholds)

    brier_model  = ah_cover_brier(true_a, true_b, model_p, thresholds)
    brier_market = ah_cover_brier(true_a, true_b, market_p, thresholds)
    blend_p = [(1 - alpha) * m + alpha * k for m, k in zip(model_p, market_p)]
    brier_blend  = ah_cover_brier(true_a, true_b, blend_p, thresholds)

    # Blended Brier must be ≤ min(model, market) Brier (allow small optimiser tolerance)
    assert brier_blend <= min(brier_model, brier_market) + 1e-4
    # Market is better — alpha should lean toward market
    assert alpha >= 0.5


def test_fit_ah_alpha_perfect_model_converges_to_zero():
    """When the model exactly recovers the true outcomes, alpha→0."""
    from wcpredictor.evaluation.backtest import _fit_ah_alpha
    from wcpredictor.evaluation.metrics import ah_cover_brier

    true_a     = [2, 0, 1, 3, 0] * 10
    true_b     = [0, 2, 1, 0, 1] * 10
    thresholds = [0.5] * len(true_a)
    # model_p closely matches realised cover
    model_p    = [0.95, 0.05, 0.50, 0.95, 0.05] * 10
    # market_p is much less certain
    market_p   = [0.55, 0.45, 0.50, 0.60, 0.40] * 10

    alpha = _fit_ah_alpha(true_a, true_b, model_p, market_p, thresholds)

    # Model is better — alpha should lean toward model (→ 0)
    assert alpha <= 0.5


def test_fit_ah_alpha_blend_is_not_worse_than_both():
    """Blended prediction at fitted alpha is no worse than both model and market."""
    from wcpredictor.evaluation.backtest import _fit_ah_alpha
    from wcpredictor.evaluation.metrics import ah_cover_brier

    import random
    rng = random.Random(42)
    n = 40
    true_a = [rng.randint(0, 3) for _ in range(n)]
    true_b = [rng.randint(0, 3) for _ in range(n)]
    thresholds = [0.5] * n
    model_p  = [rng.uniform(0.3, 0.7) for _ in range(n)]
    market_p = [rng.uniform(0.3, 0.7) for _ in range(n)]

    alpha = _fit_ah_alpha(true_a, true_b, model_p, market_p, thresholds)
    blend = [(1 - alpha) * m + alpha * k for m, k in zip(model_p, market_p)]

    b_model  = ah_cover_brier(true_a, true_b, model_p, thresholds)
    b_market = ah_cover_brier(true_a, true_b, market_p, thresholds)
    b_blend  = ah_cover_brier(true_a, true_b, blend, thresholds)

    # Blended must be ≤ best of {model, market} (convex optimisation; allow small optimiser tolerance)
    assert b_blend <= min(b_model, b_market) + 1e-4
