import numpy as np
import pandas as pd
import pytest

from src.improve import (
    ProbabilityBlendClassifier,
    blend_weight_grid,
    select_blend_weights,
)


class FixedProbabilityClassifier:
    classes_ = np.array([0, 1])

    def __init__(self, positive_probabilities):
        self.positive_probabilities = np.asarray(positive_probabilities)

    def predict_proba(self, X):
        positive = self.positive_probabilities[: len(X)]
        return np.column_stack([1.0 - positive, positive])


def test_blend_weight_grid_is_convex() -> None:
    weights = blend_weight_grid(0.05)

    assert len(weights) == 231
    assert all(sum(item) == pytest.approx(1.0) for item in weights)
    assert all(value >= 0 for item in weights for value in item)


def test_probability_blend_uses_declared_weights() -> None:
    first = FixedProbabilityClassifier([0.2, 0.8])
    second = FixedProbabilityClassifier([0.6, 0.4])
    blend = ProbabilityBlendClassifier([first, second], [0.25, 0.75])

    probabilities = blend.predict_proba(pd.DataFrame({"x": [1, 2]}))

    assert probabilities[:, 1].tolist() == pytest.approx([0.5, 0.5])


def test_weight_selection_can_choose_best_component() -> None:
    truth = pd.Series([0, 0, 1, 1])
    matrix = np.column_stack(
        [
            [0.1, 0.2, 0.8, 0.9],
            [0.9, 0.8, 0.2, 0.1],
            [0.5, 0.5, 0.5, 0.5],
        ]
    )

    weights, score = select_blend_weights(truth, matrix, 0.05)

    assert score == pytest.approx(1.0)
    assert weights[0] > 0
