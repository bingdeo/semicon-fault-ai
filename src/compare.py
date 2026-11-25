"""Compare boosting and sensor selection with repeated nested development CV."""

from __future__ import annotations

import argparse
import json
from importlib.metadata import version
from itertools import product
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from sklearn.base import clone
from sklearn.metrics import average_precision_score
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_predict,
)
from sklearn.pipeline import Pipeline

from src.config import load_config
from src.data import load_secom, project_path
from src.evaluation import binary_metrics, select_max_f1_threshold
from src.features import SensorSelector, TrainOnlyColumnFilter
from src.models.baseline import build_logistic, build_xgboost, positive_class_probability
from src.splits import load_locked_splits


MODELS = ("logistic_regression", "xgboost", "lightgbm")
SENSOR_CONDITIONS = ("all", "selected")


def make_search(
    config: dict[str, Any], model: str, sensors: str, seed: int, n_jobs: int = 1
) -> GridSearchCV:
    settings = config["comparison"]
    candidates = settings[model]
    model_seed = config["project"]["random_seed"]
    features = config["features"]
    if model == "logistic_regression":
        estimator = build_logistic(
            features, {**candidates[0], "solver": "liblinear", "max_iter": 3000}, model_seed
        )
    elif model == "xgboost":
        estimator = build_xgboost(features, candidates[0], model_seed, n_jobs=1)
    elif model == "lightgbm":
        from lightgbm import LGBMClassifier

        estimator = Pipeline(
            [
                ("column_filter", TrainOnlyColumnFilter(
                    max_missing_fraction=features["max_missing_fraction"],
                    drop_constant=features["drop_constant"],
                )),
                ("model", LGBMClassifier(
                    **candidates[0],
                    objective="binary",
                    random_state=model_seed,
                    n_jobs=1,
                    verbosity=-1,
                    deterministic=True,
                    force_col_wise=True,
                    colsample_bytree=0.8,
                )),
            ]
        )
    else:
        raise ValueError(f"Unknown model: {model}")

    grid = [{f"model__{key}": [value] for key, value in candidate.items()}
            for candidate in candidates]
    if sensors == "selected":
        estimator.steps.insert(1, ("sensor_selector", SensorSelector()))
        for option in grid:
            option["sensor_selector__k"] = settings["sensor_counts"]

    return GridSearchCV(
        estimator,
        param_grid=grid,
        scoring="average_precision",
        cv=StratifiedKFold(n_splits=settings["inner_folds"], shuffle=True, random_state=seed),
        n_jobs=n_jobs,
        error_score="raise",
    )


def fitted_sensors(estimator: Pipeline) -> list[str]:
    selector = estimator.named_steps.get("sensor_selector", estimator.named_steps["column_filter"])
    return selector.selected_columns_


def evaluate_fold(
    config: dict[str, Any], X: pd.DataFrame, y: pd.Series, index: int,
    train: np.ndarray, validation: np.ndarray,
) -> list[dict[str, Any]]:
    settings = config["comparison"]
    rows = []
    for model, sensors in product(MODELS, SENSOR_CONDITIONS):
        search = make_search(config, model, sensors, settings["seed"] + index)
        search.fit(X.iloc[train], y.iloc[train])
        scores = positive_class_probability(search.best_estimator_, X.iloc[validation])
        rows.append({
            "repeat": index // settings["outer_folds"] + 1,
            "fold": index % settings["outer_folds"] + 1,
            "model": model,
            "sensors": sensors,
            "sensor_count": len(fitted_sensors(search.best_estimator_)),
            "validation_failures": int(y.iloc[validation].sum()),
            "outer_ap": float(average_precision_score(y.iloc[validation], scores)),
            "inner_ap": search.best_score_,
            "best_parameters": json.dumps(search.best_params_, sort_keys=True),
        })
    return rows


def paired_contrasts(folds: pd.DataFrame) -> pd.DataFrame:
    scores = folds.pivot(index=["repeat", "fold"], columns=["model", "sensors"], values="outer_ap")
    comparisons = [((model, "selected"), (model, "all")) for model in MODELS]
    comparisons.extend((("lightgbm", sensors), ("xgboost", sensors)) for sensors in SENSOR_CONDITIONS)
    rows = []
    for candidate, reference in comparisons:
        differences = scores[candidate] - scores[reference]
        row = {
            "candidate": "/".join(candidate),
            "reference": "/".join(reference),
            "mean_ap_change": differences.mean(),
            "fold_wins": int((differences > 0).sum()),
            "fold_count": len(differences),
        }
        for repeat, value in differences.groupby(level="repeat").mean().items():
            row[f"repeat_{repeat}_mean_ap_change"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def run(config: dict[str, Any]) -> None:
    settings = config["comparison"]
    data = load_secom(config)
    splits = load_locked_splits(config).set_index("row_id")
    if splits.sort_index()["is_fail"].tolist() != data.targets.tolist():
        raise ValueError("Locked split labels do not match the current raw dataset.")
    development_ids = splits.index[splits["split"].isin(["train", "validation"])]
    X, y = data.features.loc[development_ids], data.targets.loc[development_ids]

    output_dir = project_path(config["paths"]["results"]) / "tuning" / "model_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    outer = RepeatedStratifiedKFold(
        n_splits=settings["outer_folds"], n_repeats=settings["outer_repeats"],
        random_state=settings["seed"],
    )
    n_jobs = min(config["runtime"]["n_jobs"], outer.get_n_splits())
    fold_rows = []
    print(f"Development rows={len(X)}, failures={y.sum()}, outer folds={outer.get_n_splits()}", flush=True)
    with parallel_config(backend="loky", inner_max_num_threads=1):
        completed_folds = Parallel(n_jobs=n_jobs, return_as="generator")(
            delayed(evaluate_fold)(config, X, y, index, train, validation)
            for index, (train, validation) in enumerate(outer.split(X, y))
        )
        for rows in completed_folds:
            fold_rows.extend(rows)
            scores = ", ".join(f"{row['model']}/{row['sensors']}={row['outer_ap']:.3f}" for row in rows)
            print(f"Repeat {rows[0]['repeat']}, fold {rows[0]['fold']}: {scores}", flush=True)

    folds = pd.DataFrame(fold_rows)
    summary = (folds.groupby(["model", "sensors"], as_index=False)
               .agg(mean_ap=("outer_ap", "mean"), fold_sd=("outer_ap", "std"),
                    median_sensor_count=("sensor_count", "median"))
               .sort_values("mean_ap", ascending=False, kind="stable"))
    contrasts = paired_contrasts(folds)
    folds.to_csv(output_dir / "outer_folds.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    contrasts.to_csv(output_dir / "contrasts.csv", index=False)
    print(summary.to_string(index=False), flush=True)
    print(contrasts.to_string(index=False), flush=True)

    chosen = summary.iloc[0]
    print(f"Preparing test evaluation for {chosen['model']}/{chosen['sensors']} ...", flush=True)
    seed = settings["seed"] + outer.get_n_splits()
    with parallel_config(backend="loky", inner_max_num_threads=1):
        final_search = make_search(config, chosen["model"], chosen["sensors"], seed, n_jobs)
        final_search.fit(X, y)
        final_model = final_search.best_estimator_
        oof = cross_val_predict(
            clone(final_model), X, y,
            cv=StratifiedKFold(n_splits=settings["outer_folds"], shuffle=True, random_state=seed + 1),
            method="predict_proba", n_jobs=n_jobs,
        )
    positive_column = int(np.flatnonzero(final_model.classes_ == 1)[0])
    threshold, _ = select_max_f1_threshold(y, oof[:, positive_column])
    selection = {
        "selection_metric": "mean outer-fold development average precision",
        "split_id": config["experiment"]["split_id"],
        "model": chosen["model"],
        "sensors": chosen["sensors"],
        "mean_outer_ap": float(chosen["mean_ap"]),
        "best_parameters": final_search.best_params_,
        "threshold": threshold,
        "sensor_columns": fitted_sensors(final_model),
        "package_versions": {package: version(package) for package in ("scikit-learn", "xgboost", "lightgbm")},
    }
    (output_dir / "selection.json").write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")

    test_ids = splits.index[splits["split"] == "test"]
    test_y = data.targets.loc[test_ids]
    test_scores = positive_class_probability(final_model, data.features.loc[test_ids]).astype(float)
    metrics = binary_metrics(test_y, test_scores, threshold, config["selection"]["recall_fixed_precision"])
    pd.DataFrame({"row_id": test_ids, "y_true": test_y.to_numpy(), "y_score": test_scores,
                  "y_pred": (test_scores >= threshold).astype(int)}).to_csv(
        output_dir / "test_predictions.csv", index=False,
    )
    pd.DataFrame([{"model": chosen["model"], "sensors": chosen["sensors"], **metrics}]).to_csv(
        output_dir / "test_metrics.csv", index=False,
    )
    print(json.dumps(metrics, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    run(load_config(parser.parse_args().config))


if __name__ == "__main__":
    main()
