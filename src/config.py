"""Configuration loading and validation shared by project commands."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


REQUIRED_SECTIONS = (
    "project",
    "paths",
    "dataset",
    "split",
    "evaluation",
    "runtime",
    "model_selection",
    "benchmark",
)


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML config and fail early when required sections are missing."""

    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    config = _read_yaml(path)
    base_reference = config.pop("base_config", None)
    if base_reference:
        base_path = Path(base_reference)
        if not base_path.is_absolute():
            project_candidate = Path(__file__).resolve().parents[1] / base_path
            local_candidate = path.parent / base_path
            base_path = (
                project_candidate if project_candidate.is_file() else local_candidate
            )
        config = _deep_merge(_read_yaml(base_path), config)

    if not isinstance(config, dict):
        raise ValueError(f"Config root must be a mapping: {path}")

    missing = [section for section in REQUIRED_SECTIONS if section not in config]
    if missing:
        raise ValueError(f"Missing required config sections: {', '.join(missing)}")

    _validate_split(config["split"])
    _validate_runtime(config["runtime"])
    return config


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return config


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _validate_split(split_config: Any) -> None:
    if not isinstance(split_config, dict):
        raise ValueError("The split section must be a mapping.")

    fraction_keys = ("train_fraction", "validation_fraction", "test_fraction")
    try:
        fractions = [float(split_config[key]) for key in fraction_keys]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "Split config needs numeric train/validation/test fractions."
        ) from error

    if any(fraction <= 0 for fraction in fractions):
        raise ValueError("All split fractions must be greater than zero.")
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError("Split fractions must sum to 1.0.")


def _validate_runtime(runtime_config: Any) -> None:
    if not isinstance(runtime_config, dict):
        raise ValueError("The runtime section must be a mapping.")

    for key in ("n_jobs", "cpu_threads"):
        value = runtime_config.get(key)
        if not isinstance(value, int) or value < 1:
            raise ValueError(f"runtime.{key} must be a positive integer.")

