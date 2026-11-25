from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.data import load_secom
from src.features import TrainOnlyColumnFilter
from src.models.baseline import positive_class_probability
from src.splits import create_stratified_split_frame, load_locked_splits


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE_CONFIG = PROJECT_ROOT / "configs" / "experiments" / "baseline.yaml"


def test_column_filter_never_learns_from_test_rows() -> None:
    train = pd.DataFrame(
        {
            "useful": [1.0, 2.0, 3.0],
            "constant_in_train": [7.0, 7.0, 7.0],
            "mostly_missing": [np.nan, np.nan, 1.0],
        }
    )
    test = pd.DataFrame(
        {
            "useful": [10.0, 11.0],
            "constant_in_train": [1.0, 2.0],
            "mostly_missing": [2.0, 3.0],
        }
    )
    transformer = TrainOnlyColumnFilter(max_missing_fraction=0.50).fit(train)

    transformed = transformer.transform(test)

    assert transformer.selected_columns_ == ["useful"]
    assert transformed.columns.tolist() == ["useful"]


def test_positive_probability_uses_class_label_not_fixed_column() -> None:
    class ReversedClassifier:
        classes_ = np.array([1, 0])

        def predict_proba(self, X):
            return np.array([[0.8, 0.2], [0.3, 0.7]])

    probabilities = positive_class_probability(ReversedClassifier(), [[0], [1]])

    assert probabilities.tolist() == [0.8, 0.3]


def test_split_generation_is_deterministic_and_complete() -> None:
    config = load_config(BASELINE_CONFIG)
    first = create_stratified_split_frame(config)
    second = create_stratified_split_frame(config)

    pd.testing.assert_frame_equal(first, second)
    assert first["row_id"].is_unique
    assert set(first["row_id"]) == set(range(len(first)))


def test_locked_splits_have_no_overlap_and_match_labels() -> None:
    config = load_config(BASELINE_CONFIG)
    try:
        split_frame = load_locked_splits(config)
    except FileNotFoundError:
        pytest.skip("Locked split has not been generated yet.")
    dataset = load_secom(config)

    split_sets = {
        name: set(split_frame.loc[split_frame["split"] == name, "row_id"])
        for name in ("train", "validation", "test")
    }
    assert split_sets["train"].isdisjoint(split_sets["validation"])
    assert split_sets["train"].isdisjoint(split_sets["test"])
    assert split_sets["validation"].isdisjoint(split_sets["test"])
    assert split_frame.sort_values("row_id")["is_fail"].tolist() == dataset.targets.tolist()


def test_locked_split_has_expected_class_counts() -> None:
    config = load_config(BASELINE_CONFIG)
    try:
        split_frame = load_locked_splits(config)
    except FileNotFoundError:
        pytest.skip("Locked split has not been generated yet.")

    counts = {
        name: {
            "rows": len(group),
            "fail": int(group["is_fail"].sum()),
        }
        for name, group in split_frame.groupby("split")
    }
    assert counts == {
        "test": {"rows": 314, "fail": 21},
        "train": {"rows": 940, "fail": 62},
        "validation": {"rows": 313, "fail": 21},
    }


def test_v2_development_pool_excludes_locked_test() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "experiments" / "baseline_v2.yaml")
    split_frame = load_locked_splits(config)
    development = set(
        split_frame.loc[
            split_frame["split"].isin(config["improvement"]["development_splits"]),
            "row_id",
        ]
    )
    test = set(split_frame.loc[split_frame["split"] == "test", "row_id"])

    assert len(development) == 1253
    assert len(test) == 314
    assert development.isdisjoint(test)
