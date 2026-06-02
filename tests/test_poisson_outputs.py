"""Poisson model output correctness tests."""

import pytest
from wcpredictor.models.poisson import predict_one
from wcpredictor.config import MAX_GOALS


def test_probabilities_non_negative_and_sum_to_one():
    result = predict_one(0.0, 1.3, 0.003)
    assert result["p_win"] >= 0
    assert result["p_draw"] >= 0
    assert result["p_loss"] >= 0
    assert result["p_win"] + result["p_draw"] + result["p_loss"] == pytest.approx(1.0, abs=1e-4)


def test_score_matrix_sums_to_one():
    result = predict_one(100.0, 1.3, 0.003)
    total = sum(result["score_matrix"][i][j]
                for i in range(MAX_GOALS + 1)
                for j in range(MAX_GOALS + 1))
    assert total == pytest.approx(1.0, abs=1e-4)


def test_score_matrix_shape():
    result = predict_one(0.0, 1.3, 0.003)
    assert len(result["score_matrix"]) == MAX_GOALS + 1
    assert all(len(row) == MAX_GOALS + 1 for row in result["score_matrix"])


def test_top_scorelines_length():
    result = predict_one(0.0, 1.3, 0.003)
    assert len(result["top_scorelines"]) == 5


def test_stronger_elo_higher_win_prob():
    # Large positive elo_diff_adj → team_a heavily favoured
    strong = predict_one(500.0, 1.3, 0.003)
    weak = predict_one(-500.0, 1.3, 0.003)
    assert strong["p_win"] > strong["p_loss"]
    assert weak["p_loss"] > weak["p_win"]
    assert strong["p_win"] > weak["p_win"]


def test_symmetric_at_zero_diff():
    result = predict_one(0.0, 1.3, 0.003)
    # With equal ratings, win ≈ loss
    assert abs(result["p_win"] - result["p_loss"]) < 0.02
