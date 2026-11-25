"""Factories and probability helpers for classical baseline models."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features import TrainOnlyColumnFilter


def _column_filter(feature_config: dict[str, Any]) -> TrainOnlyColumnFilter:
    return TrainOnlyColumnFilter(
        max_missing_fraction=float(feature_config["max_missing_fraction"]),
        drop_constant=bool(feature_config["drop_constant"]),
    )


def build_dummy(feature_config: dict[str, Any]) -> Pipeline:
    return Pipeline(
        [
            ("column_filter", _column_filter(feature_config)),
            (
                "imputer",
                SimpleImputer(
                    strategy=feature_config["imputation_strategy"],
                    add_indicator=bool(feature_config["add_missing_indicators"]),
                ),
            ),
            ("model", DummyClassifier(strategy="prior")),
        ]
    )


def build_logistic(
    feature_config: dict[str, Any], model_config: dict[str, Any], seed: int
) -> Pipeline:
    return Pipeline(
        [
            ("column_filter", _column_filter(feature_config)),
            (
                "imputer",
                SimpleImputer(
                    strategy=feature_config["imputation_strategy"],
                    add_indicator=bool(feature_config["add_missing_indicators"]),
                ),
            ),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=float(model_config["C"]),
                    penalty=model_config["penalty"],
                    class_weight=model_config["class_weight"],
                    solver=model_config["solver"],
                    max_iter=int(model_config["max_iter"]),
                    random_state=seed,
                ),
            ),
        ]
    )


def build_xgboost(
    feature_config: dict[str, Any], model_config: dict[str, Any], seed: int, n_jobs: int
) -> Pipeline:
    from xgboost import XGBClassifier

    return Pipeline(
        [
            ("column_filter", _column_filter(feature_config)),
            (
                "model",
                XGBClassifier(
                    **model_config,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    tree_method="hist",
                    n_jobs=n_jobs,
                    random_state=seed,
                    verbosity=0,
                ),
            ),
        ]
    )


def positive_class_probability(estimator: Any, X: Any) -> np.ndarray:
    probabilities = np.asarray(estimator.predict_proba(X))
    classes = np.asarray(estimator.classes_)
    matches = np.flatnonzero(classes == 1)
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one positive class 1, got {classes.tolist()}.")
    return probabilities[:, int(matches[0])]


class ProbabilityBlendClassifier(ClassifierMixin, BaseEstimator):
    """Weighted probability blend of already fitted binary classifiers."""

    def __init__(self, estimators: list[Any], weights: list[float]) -> None:
        self.estimators = estimators
        self.weights = weights
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X: Any) -> np.ndarray:
        positive = np.zeros(len(X), dtype=float)
        for estimator, weight in zip(self.estimators, self.weights, strict=True):
            positive += float(weight) * positive_class_probability(estimator, X)
        return np.column_stack([1.0 - positive, positive])
