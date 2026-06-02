"""Multinomial logistic W/D/L member.

Features: elo_diff_adj, neutral (binary 0/1).
Output: [p_win, p_draw, p_loss] probability vectors only.
No score matrix — contributes only to the W/D/L ensemble pool.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def fit(
    features_df: pd.DataFrame,
    labels: list[int],
) -> tuple:
    """Fit multinomial logistic regression on Elo features.

    Parameters
    ----------
    features_df : DataFrame with columns elo_diff_adj, neutral
    labels      : list of int (0=win, 1=draw, 2=loss)

    Returns
    -------
    (scaler, model) — pass both to predict_proba
    """
    X = _build_X(features_df)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        C=1.0,
        random_state=42,
    )
    model.fit(X_scaled, labels)
    return scaler, model


def predict_proba(
    scaler: StandardScaler,
    model: LogisticRegression,
    features_df: pd.DataFrame,
) -> list[list[float]]:
    """Return [[p_win, p_draw, p_loss], ...] for each row in features_df."""
    X = _build_X(features_df)
    X_scaled = scaler.transform(X)
    raw = model.predict_proba(X_scaled)

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


def _build_X(features_df: pd.DataFrame) -> np.ndarray:
    return np.column_stack([
        features_df["elo_diff_adj"].to_numpy(float),
        features_df["neutral"].astype(float).to_numpy(),
    ])
