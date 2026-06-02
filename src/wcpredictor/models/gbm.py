"""Gradient-boosted tree W/D/L member.

Features: elo_diff_adj, neutral, form_diff, momentum_diff, rest_diff,
          elo_a_pre, elo_b_pre (raw ratings — orthogonal to the diff).
Output: [p_win, p_draw, p_loss] probability vectors only.
No score matrix — contributes only to the W/D/L ensemble pool.

Uses sklearn HistGradientBoostingClassifier (no extra deps).
Conservative regularisation: shallow trees, high min_samples_leaf,
modest n_iter + learning_rate, L2 regularisation.
All randomness is pinned to seed=42; max_iter is fixed so the fit
is deterministic (no early stopping by default).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier


_SEED = 42
_MAX_DEPTH = 3
_MAX_ITER = 200
_LEARNING_RATE = 0.05
_MIN_SAMPLES_LEAF = 20
_L2_REG = 1.0


def fit(
    features_df: pd.DataFrame,
    labels: list[int],
) -> HistGradientBoostingClassifier:
    """Fit a regularised GBM classifier on Elo + form features.

    Parameters
    ----------
    features_df : DataFrame — must contain at minimum elo_diff_adj, neutral.
    labels      : list of int (0=win, 1=draw, 2=loss)

    Returns
    -------
    Fitted HistGradientBoostingClassifier (single object; no scaler needed).
    """
    X = _build_X(features_df)
    model = HistGradientBoostingClassifier(
        max_depth=_MAX_DEPTH,
        max_iter=_MAX_ITER,
        learning_rate=_LEARNING_RATE,
        min_samples_leaf=_MIN_SAMPLES_LEAF,
        l2_regularization=_L2_REG,
        random_state=_SEED,
        early_stopping=False,
    )
    model.fit(X, labels)
    return model


def predict_proba(
    model: HistGradientBoostingClassifier,
    features_df: pd.DataFrame,
) -> list[list[float]]:
    """Return [[p_win, p_draw, p_loss], ...] for each row in features_df."""
    X = _build_X(features_df)
    raw = model.predict_proba(X)

    classes = list(model.classes_)
    if classes == [0, 1, 2]:
        return raw.tolist()

    # A class absent in training gets zero probability; renormalize.
    out = np.zeros((len(raw), 3))
    for col_i, cls in enumerate(classes):
        out[:, cls] = raw[:, col_i]
    row_sums = out.sum(axis=1, keepdims=True)
    out /= np.where(row_sums > 0, row_sums, 1.0)
    return out.tolist()


_FORM_COLS = ("form_diff", "momentum_diff", "rest_diff")
_ELO_COLS = ("elo_a_pre", "elo_b_pre")
# Odds cols must NOT be zero-filled when absent — NaN tells HGB "no odds here"
_ODDS_COLS = ("odds_p_win", "odds_p_draw", "odds_p_loss", "has_odds")


def _build_X(features_df: pd.DataFrame) -> np.ndarray:
    cols = [
        features_df["elo_diff_adj"].to_numpy(float),
        features_df["neutral"].astype(float).to_numpy(),
    ]
    for col in _FORM_COLS:
        if col in features_df.columns:
            cols.append(features_df[col].to_numpy(float))
        else:
            cols.append(np.zeros(len(features_df), dtype=float))
    for col in _ELO_COLS:
        if col in features_df.columns:
            cols.append(features_df[col].to_numpy(float))
        else:
            cols.append(np.zeros(len(features_df), dtype=float))
    # Odds cols: pass NaN through — HGB handles NaN natively via learned split direction.
    # has_odds absent → full NaN column (all rows treated as no-odds by the tree).
    for col in _ODDS_COLS:
        if col in features_df.columns:
            cols.append(features_df[col].to_numpy(float))
        else:
            cols.append(np.full(len(features_df), np.nan))
    return np.column_stack(cols)
