"""Metric function unit tests."""

import math
import pytest
from wcpredictor.evaluation.metrics import log_loss, brier, accuracy, macro_f1


def test_log_loss_perfect():
    probs = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    labels = [0, 1, 2]
    assert log_loss(labels, probs) == pytest.approx(0.0, abs=1e-6)


def test_log_loss_uniform():
    probs = [[1/3, 1/3, 1/3]] * 3
    labels = [0, 1, 2]
    expected = math.log(3)
    assert log_loss(labels, probs) == pytest.approx(expected, abs=1e-4)


def test_brier_perfect():
    probs = [[1.0, 0.0, 0.0]]
    labels = [0]
    assert brier(labels, probs) == pytest.approx(0.0, abs=1e-6)


def test_brier_worst():
    probs = [[0.0, 0.0, 1.0]]
    labels = [0]
    # (0-1)^2 + (0-0)^2 + (1-0)^2 = 2
    assert brier(labels, probs) == pytest.approx(2.0, abs=1e-6)


def test_accuracy():
    y_true = [0, 1, 2, 0]
    y_pred = [0, 1, 0, 0]
    assert accuracy(y_true, y_pred) == pytest.approx(0.75)


def test_macro_f1_perfect():
    y = [0, 1, 2, 0, 1, 2]
    assert macro_f1(y, y) == pytest.approx(1.0)
