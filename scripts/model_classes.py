"""Shared model classes for IPO prediction pipeline.

This module is imported by both baseline_models.py and predict.py so that
joblib/pickle can resolve class references regardless of which script is
running as __main__.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


class BoardMeanModel:
    """Simple per-board mean baseline."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BoardMeanModel":
        self._global = float(y.mean())
        self._means = y.groupby(X["board"].values).mean().to_dict()
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.array([self._means.get(b, self._global) for b in X["board"]])


class _ColSelector:
    """Wraps a Pipeline so it selects only its declared columns from a wider DataFrame."""

    def __init__(self, pipe: Pipeline, cols: list[str]) -> None:
        self._pipe, self._cols = pipe, cols

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "_ColSelector":
        avail = [c for c in self._cols if c in X.columns]
        self._pipe.fit(X[avail], y)
        self._avail = avail
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_use = X.copy()
        for c in self._avail:
            if c not in X_use.columns:
                X_use[c] = np.nan
        return self._pipe.predict(X_use[self._avail])
