"""Tune classical baselines on validation data and evaluate once on locked test rows."""

from __future__ import annotations

import argparse
import itertools
import json
import hashlib
import time
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any, Callable

import joblib
import pandas as pd
import psutil
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits

from src.config import load_config
from src.data import PROJECT_ROOT, load_secom, project_path
from src.evaluation import binary_metrics, select_max_f1_threshold
from src.models.baseline import (
    build_dummy,
    build_logistic,
    build_xgboost,
    positive_class_probability,
)
from src.splits import load_locked_splits


@dataclass
class SelectedModel:
    name: str
    estimator: Any
    parameters: dict[str, Any]
    validation_score: float
    validation_probabilities: Any
    fit_seconds: float
    tuning_seconds: float
    candidate_count: int


def logistic_candidates(config: dict[str, Any]) -> list[dict[str, Any]]:
    search = config["logistic_search"]
    return [
        {
            "C": c_value,
            "penalty": penalty,
            "class_weight": class_weight,
            "solver": search["solver"],
            "max_iter": search["max_iter"],
        }
        for c_value, penalty, class_weight in itertools.product(
            search["c_values"], search["penalties"], search["class_weights"]
        )
    ]


def resolve_xgboost_candidate(
    candidate: dict[str, Any], y_train: pd.Series
) -> dict[str, Any]:
    resolved = dict(candidate)
    if resolved.get("scale_pos_weight") == "balanced":
        positives = int(y_train.sum())
        negatives = int(len(y_train) - positives)
        if positives == 0:
            raise ValueError("Cannot balance XGBoost without positive train samples.")
        resolved["scale_pos_weight"] = negatives / positives
    return resolved


def build_candidate(
    model_name: str,
    config: dict[str, Any],
    candidate: dict[str, Any],
    y_train: pd.Series,
) -> tuple[Any, dict[str, Any]]:
    seed = int(config["project"]["random_seed"])
    if model_name == "dummy":
        return build_dummy(config["features"]), {}
    if model_name == "logistic_regression":
        return build_logistic(config["features"], candidate, seed), candidate
    if model_name == "xgboost":
        resolved = resolve_xgboost_candidate(candidate, y_train)
        return (
            build_xgboost(
                config["features"],
                resolved,
                seed=seed,
                n_jobs=int(config["runtime"]["n_jobs"]),
            ),
            resolved,
        )
    raise ValueError(f"Unknown model: {model_name}")


def candidates_for_model(
    model_name: str, config: dict[str, Any]
) -> list[dict[str, Any]]:
    if model_name == "dummy":
        return [{}]
    if model_name == "logistic_regression":
        return logistic_candidates(config)
    if model_name == "xgboost":
        return list(config["xgboost_search"]["candidates"])
    raise ValueError(f"Unknown model: {model_name}")


def tune_model(
    model_name: str,
    config: dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    tuning_dir: Path,
) -> SelectedModel:
    records: list[dict[str, Any]] = []
    best: SelectedModel | None = None
    start_tuning = time.perf_counter()
    candidates = candidates_for_model(model_name, config)

    for candidate_index, candidate in enumerate(candidates):
        try:
            estimator, resolved = build_candidate(
                model_name, config, candidate, y_train
            )
            start_fit = time.perf_counter()
            with threadpool_limits(limits=int(config["runtime"]["cpu_threads"])):
                estimator.fit(X_train, y_train)
            fit_seconds = time.perf_counter() - start_fit
            validation_probabilities = positive_class_probability(
                estimator, X_validation
            )
            score = float(
                average_precision_score(y_validation, validation_probabilities)
            )
            records.append(
                {
                    "candidate_index": candidate_index,
                    "status": "ok",
                    "validation_average_precision": score,
                    "fit_seconds": fit_seconds,
                    "parameters": json.dumps(resolved, sort_keys=True),
                    "error": "",
                }
            )
            if best is None or score > best.validation_score:
                best = SelectedModel(
                    name=model_name,
                    estimator=estimator,
                    parameters=resolved,
                    validation_score=score,
                    validation_probabilities=validation_probabilities,
                    fit_seconds=fit_seconds,
                    tuning_seconds=0.0,
                    candidate_count=len(candidates),
                )
        except Exception as error:  # preserve candidate failure instead of faking a score
            records.append(
                {
                    "candidate_index": candidate_index,
                    "status": "failed",
                    "validation_average_precision": None,
                    "fit_seconds": None,
                    "parameters": json.dumps(candidate, sort_keys=True),
                    "error": f"{type(error).__name__}: {error}",
                }
            )

    tuning_seconds = time.perf_counter() - start_tuning
    tuning_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(tuning_dir / f"{model_name}_candidates.csv", index=False)
    if best is None:
        raise RuntimeError(
            f"Every {model_name} candidate failed. See {tuning_dir / f'{model_name}_candidates.csv'}"
        )
    best.tuning_seconds = tuning_seconds
    return best


def benchmark_inference(
    estimator: Any, X: pd.DataFrame, warmups: int, repeats: int, cpu_threads: int
) -> float:
    with threadpool_limits(limits=cpu_threads):
        for _ in range(warmups):
            positive_class_probability(estimator, X)
        start = time.perf_counter()
        for _ in range(repeats):
            positive_class_probability(estimator, X)
        elapsed = time.perf_counter() - start
    return elapsed * 1000.0 / (repeats * len(X))


def package_version_for(model_name: str) -> str:
    return version("xgboost") if model_name == "xgboost" else version("scikit-learn")


def data_version() -> str:
    manifest_path = PROJECT_ROOT / "data" / "data_manifest.json"
    if not manifest_path.is_file():
        return "unknown"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw_hashes = sorted(file_info["sha256"] for file_info in manifest["files"])
    stable_identity = f"{manifest['dataset_id']}|{'|'.join(raw_hashes)}"
    return hashlib.sha256(stable_identity.encode("utf-8")).hexdigest()[:16]


def update_metrics_file(metrics_path: Path, new_rows: list[dict[str, Any]]) -> None:
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    new_frame = pd.DataFrame(new_rows)
    if metrics_path.is_file():
        existing = pd.read_csv(metrics_path)
        existing = existing.loc[~existing["run_id"].isin(new_frame["run_id"])]
        new_frame = pd.concat([existing, new_frame], ignore_index=True)
    new_frame.to_csv(metrics_path, index=False)


def train_baselines(config: dict[str, Any]) -> pd.DataFrame:
    dataset = load_secom(config)
    split_frame = load_locked_splits(config)
    split_frame = split_frame.set_index("row_id")
    if split_frame.sort_index()["is_fail"].tolist() != dataset.targets.tolist():
        raise ValueError("Locked split labels do not match the current raw dataset.")

    row_ids = {
        split_name: split_frame.index[split_frame["split"] == split_name].to_numpy()
        for split_name in ("train", "validation", "test")
    }
    X = {name: dataset.features.loc[ids] for name, ids in row_ids.items()}
    y = {name: dataset.targets.loc[ids] for name, ids in row_ids.items()}

    results_dir = project_path(config["paths"]["results"])
    tuning_dir = results_dir / "tuning"
    model_dir = results_dir / "models"
    prediction_dir = project_path(config["paths"]["predictions"])
    for directory in (tuning_dir, model_dir, prediction_dir):
        directory.mkdir(parents=True, exist_ok=True)

    selected: list[SelectedModel] = []
    for model_name in config["experiment"]["models"]:
        print(f"Tuning {model_name} ...", flush=True)
        chosen = tune_model(
            model_name,
            config,
            X["train"],
            y["train"],
            X["validation"],
            y["validation"],
            tuning_dir,
        )
        selected.append(chosen)
        print(
            f"  best validation AP={chosen.validation_score:.6f} "
            f"from {chosen.candidate_count} candidate(s)",
            flush=True,
        )

    metric_rows: list[dict[str, Any]] = []
    seed = int(config["project"]["random_seed"])
    process = psutil.Process()
    fixed_precision = float(config["selection"]["recall_fixed_precision"])

    for chosen in selected:
        threshold, validation_best_f1 = select_max_f1_threshold(
            y["validation"], chosen.validation_probabilities
        )
        test_probabilities = positive_class_probability(chosen.estimator, X["test"])
        validation_metrics = binary_metrics(
            y["validation"],
            chosen.validation_probabilities,
            threshold,
            fixed_precision,
        )
        test_metrics = binary_metrics(
            y["test"], test_probabilities, threshold, fixed_precision
        )
        latency_ms = benchmark_inference(
            chosen.estimator,
            X["test"],
            warmups=int(config["benchmark"]["warmup_runs"]),
            repeats=int(config["benchmark"]["measurement_runs"]),
            cpu_threads=int(config["runtime"]["cpu_threads"]),
        )
        run_id = f"{config['experiment']['id']}_{chosen.name}_seed{seed}"

        prediction_frames: list[pd.DataFrame] = []
        for split_name, probabilities in (
            ("validation", chosen.validation_probabilities),
            ("test", test_probabilities),
        ):
            probabilities = probabilities.astype(float)
            prediction_frames.append(
                pd.DataFrame(
                    {
                        "run_id": run_id,
                        "model": chosen.name,
                        "split": split_name,
                        "row_id": row_ids[split_name],
                        "y_true": y[split_name].to_numpy(),
                        "y_score": probabilities,
                        "threshold": threshold,
                        "y_pred": (probabilities >= threshold).astype(int),
                    }
                )
            )
        pd.concat(prediction_frames, ignore_index=True).to_csv(
            prediction_dir / f"{run_id}.csv", index=False
        )
        joblib.dump(chosen.estimator, model_dir / f"{run_id}.joblib")

        column_filter = chosen.estimator.named_steps["column_filter"]
        metadata = {
            "run_id": run_id,
            "model": chosen.name,
            "model_version": package_version_for(chosen.name),
            "best_parameters": chosen.parameters,
            "validation_average_precision": chosen.validation_score,
            "validation_best_f1": validation_best_f1,
            "selected_threshold": threshold,
            "selected_feature_count": len(column_filter.selected_columns_),
            "dropped_high_missing": column_filter.dropped_high_missing_,
            "dropped_constant": column_filter.dropped_constant_,
        }
        (model_dir / f"{run_id}.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        secondary = {
            key: value
            for key, value in test_metrics.items()
            if key != "average_precision"
        }
        metric_rows.append(
            {
                "run_id": run_id,
                "dataset_version": data_version(),
                "split_id": config["experiment"]["split_id"],
                "model": chosen.name,
                "model_version": package_version_for(chosen.name),
                "input_modalities": "590 anonymized numeric process measurements",
                "train_budget": json.dumps(
                    {
                        "train_rows": len(X["train"]),
                        "positive_rows": int(y["train"].sum()),
                        "seed": seed,
                        "candidate_count": chosen.candidate_count,
                    },
                    sort_keys=True,
                ),
                "primary_metric": test_metrics["average_precision"],
                "validation_primary_metric": validation_metrics["average_precision"],
                "secondary_metrics": json.dumps(secondary, sort_keys=True),
                "latency_ms": latency_ms,
                "peak_vram_mb": None,
                "process_rss_mb": process.memory_info().rss / (1024**2),
                "robustness_drop": None,
                "fit_seconds": chosen.fit_seconds,
                "tuning_seconds": chosen.tuning_seconds,
                "selected_feature_count": len(column_filter.selected_columns_),
                **{f"test_{key}": value for key, value in test_metrics.items()},
                "notes": "Threshold selected by maximum validation F1; test not used for tuning.",
            }
        )

    metrics_path = results_dir / "metrics.csv"
    update_metrics_file(metrics_path, metric_rows)
    return pd.DataFrame(metric_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    config = load_config(parse_args().config)
    metrics = train_baselines(config)
    display_columns = [
        "model",
        "validation_primary_metric",
        "primary_metric",
        "test_roc_auc",
        "test_recall",
        "test_precision",
        "latency_ms",
    ]
    print("\nBaseline results:")
    print(metrics[display_columns].to_string(index=False))
    print(f"\nSaved: {config['paths']['results']}/metrics.csv")


if __name__ == "__main__":
    main()
