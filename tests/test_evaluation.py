from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.evaluation import (
    binary_metrics,
    expected_calibration_error,
    select_max_f1_threshold,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_binary_metrics_use_fail_as_positive_class() -> None:
    truth = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.4, 0.6, 0.9])

    metrics = binary_metrics(truth, scores, threshold=0.5)

    assert metrics["average_precision"] == pytest.approx(1.0)
    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(1.0)
    assert (metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"]) == (
        2,
        0,
        0,
        2,
    )


def test_threshold_is_selected_from_supplied_validation_values() -> None:
    truth = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.4, 0.6, 0.9])

    threshold, best_f1 = select_max_f1_threshold(truth, scores)

    assert threshold == pytest.approx(0.6)
    assert best_f1 == pytest.approx(1.0)


def test_calibration_error_is_zero_for_perfect_bin_means() -> None:
    truth = np.array([0, 1])
    scores = np.array([0.0, 1.0])

    assert expected_calibration_error(truth, scores) == pytest.approx(0.0)


def test_saved_raw_predictions_reproduce_test_primary_metrics() -> None:
    metrics_path = PROJECT_ROOT / "results" / "metrics.csv"
    prediction_dir = PROJECT_ROOT / "results" / "predictions"
    if not metrics_path.is_file():
        pytest.skip("Baseline metrics have not been generated.")

    metrics = pd.read_csv(metrics_path).set_index("run_id")
    prediction_paths = sorted(prediction_dir.glob("A-E1-baseline_*.csv"))
    assert len(prediction_paths) == 3
    for prediction_path in prediction_paths:
        predictions = pd.read_csv(prediction_path)
        test_rows = predictions.loc[predictions["split"] == "test"]
        run_id = str(test_rows["run_id"].iloc[0])
        calculated = binary_metrics(
            test_rows["y_true"],
            test_rows["y_score"],
            threshold=float(test_rows["threshold"].iloc[0]),
        )
        assert calculated["average_precision"] == pytest.approx(
            metrics.loc[run_id, "primary_metric"], abs=1e-12
        )
