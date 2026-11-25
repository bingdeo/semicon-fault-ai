"""Leakage-safe, train-fitted feature transformations."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest
from sklearn.impute import SimpleImputer
from sklearn.utils.validation import check_is_fitted


class TrainOnlyColumnFilter(TransformerMixin, BaseEstimator):
    """Drop high-missing and constant columns based only on fit data."""

    def __init__(
        self, max_missing_fraction: float = 0.95, drop_constant: bool = True
    ) -> None:
        self.max_missing_fraction = max_missing_fraction
        self.drop_constant = drop_constant

    def fit(self, X: pd.DataFrame, y: Any = None) -> "TrainOnlyColumnFilter":
        frame = self._require_frame(X)
        if not 0.0 <= self.max_missing_fraction <= 1.0:
            raise ValueError("max_missing_fraction must be between 0 and 1.")

        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        missing_fraction = frame.isna().mean()
        high_missing = set(missing_fraction[missing_fraction > self.max_missing_fraction].index)
        constant = (
            set(frame.columns[frame.nunique(dropna=True) <= 1])
            if self.drop_constant
            else set()
        )
        self.dropped_high_missing_ = sorted(high_missing)
        self.dropped_constant_ = sorted(constant)
        dropped = high_missing | constant
        self.selected_columns_ = [column for column in frame.columns if column not in dropped]
        if not self.selected_columns_:
            raise ValueError("Column filtering removed every feature.")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "selected_columns_")
        frame = self._require_frame(X)
        missing = set(self.selected_columns_) - set(frame.columns)
        if missing:
            raise ValueError(f"Input is missing fitted columns: {sorted(missing)}")
        return frame.loc[:, self.selected_columns_].copy()

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        check_is_fitted(self, "selected_columns_")
        return np.asarray(self.selected_columns_, dtype=object)

    @staticmethod
    def _require_frame(X: Any) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("TrainOnlyColumnFilter requires a pandas DataFrame.")
        return X


class SensorSelector(TransformerMixin, BaseEstimator):
    """Rank raw sensors by ANOVA while preserving their original missing values."""

    def __init__(self, k: int = 40) -> None:
        self.k = k

    def fit(self, X: pd.DataFrame, y: Any) -> "SensorSelector":
        if not 1 <= self.k <= X.shape[1]:
            raise ValueError("k must be between one and the number of available sensors.")
        # Imputed values are used to rank sensors. Tree models still see NaNs.
        imputed = SimpleImputer(strategy="median", keep_empty_features=True).fit_transform(X)
        centered = imputed - imputed.mean(axis=0)
        selector = SelectKBest(k=self.k).fit(centered, y)
        self.selected_columns_ = X.columns[selector.get_support()].tolist()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "selected_columns_")
        return X.loc[:, self.selected_columns_].copy()
