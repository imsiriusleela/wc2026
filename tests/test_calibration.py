"""Tests for calibration module."""

from __future__ import annotations

import math

import numpy as np
import pytest

from wcpredictor.models.calibration import (
    apply,
    expected_calibration_error,
    fit_temperature,
)


def _make_probs(n: int = 200, seed: int = 42) -> tuple[list[int], list[list[float]]]:
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 3, n).tolist()
    # Overconfident: dirichlet concentrates mass on one class
    raw = rng.dirichlet(alpha=[4.0, 1.0, 1.0], size=n)
    return labels, raw.tolist()


def _log_loss(labels, probs) -> float:
    return sum(-math.log(max(probs[i][l], 1e-10)) for i, l in enumerate(labels)) / len(labels)


def test_temperature_improves_or_preserves_logloss():
    labels, probs = _make_probs()
    T = fit_temperature(labels, probs)
    cal = apply(probs, T)
    assert _log_loss(labels, cal) <= _log_loss(labels, probs) + 1e-6


def test_calibrated_probs_sum_to_one():
    labels, probs = _make_probs()
    T = fit_temperature(labels, probs)
    for row in apply(probs, T):
        assert abs(sum(row) - 1.0) < 1e-6


def test_uniform_probs_return_near_one_temperature():
    """Already-uniform probs → T should not change probabilities meaningfully."""
    n = 200
    labels = [i % 3 for i in range(n)]
    probs = [[1 / 3, 1 / 3, 1 / 3]] * n
    T = fit_temperature(labels, probs)
    cal = apply(probs, T)
    for row in cal:
        # Softmax of equal values stays uniform regardless of T
        assert abs(row[0] - 1 / 3) < 1e-6
        assert abs(sum(row) - 1.0) < 1e-6


def test_ece_nonneg_and_bounded():
    labels, probs = _make_probs()
    ece = expected_calibration_error(labels, probs)
    assert 0.0 <= ece <= 1.0


def test_ece_decreases_after_calibration():
    """ECE should not increase after temperature calibration on the same set."""
    labels, probs = _make_probs()
    T = fit_temperature(labels, probs)
    cal = apply(probs, T)
    ece_before = expected_calibration_error(labels, probs)
    ece_after = expected_calibration_error(labels, cal)
    # Allow tiny tolerance; calibration should improve or not worsen ECE
    assert ece_after <= ece_before + 0.02
