from pathlib import Path

import pytest

from src.config import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = PROJECT_ROOT / "configs" / "base.yaml"


def test_base_config_loads() -> None:
    config = load_config(BASE_CONFIG)

    assert config["project"]["name"] == "semicon-fault-ai"
    assert config["evaluation"]["primary_metric"] == "average_precision"
    assert config["dataset"]["raw_fail_label"] == 1


def test_split_fractions_sum_to_one() -> None:
    split = load_config(BASE_CONFIG)["split"]
    total = sum(
        split[key]
        for key in ("train_fraction", "validation_fraction", "test_fraction")
    )

    assert total == pytest.approx(1.0)


def test_configured_project_directories_exist() -> None:
    paths = load_config(BASE_CONFIG)["paths"]

    for relative_path in paths.values():
        assert (PROJECT_ROOT / relative_path).is_dir(), relative_path


def test_runtime_uses_current_machine_cpu_budget() -> None:
    runtime = load_config(BASE_CONFIG)["runtime"]

    assert runtime["n_jobs"] == 8
    assert runtime["cpu_threads"] == 8
    assert runtime["optimization_goal"] == "best_predictive_performance"


def test_model_selection_never_uses_test_data() -> None:
    selection = load_config(BASE_CONFIG)["model_selection"]

    assert selection["objective"] == "maximize_validation_average_precision"
    assert selection["test_usage"] == "final_evaluation_only"

