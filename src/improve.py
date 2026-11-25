"""Build stronger leakage-safe baselines with CV tuning and OOF blending."""

from __future__ import annotations

import argparse
import ast
import json
import time
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score
from sklearn.model_selection import (
    GridSearchCV,
    ParameterGrid,
    ParameterSampler,
    RandomizedSearchCV,
    StratifiedKFold,
)
from sklearn.pipeline import Pipeline

from src.config import load_config
from src.data import load_secom, project_path
from src.evaluation import binary_metrics, select_max_f1_threshold
from src.features import TrainOnlyColumnFilter
from src.models.baseline import (
    ProbabilityBlendClassifier,
    build_logistic,
    build_xgboost,
    positive_class_probability,
)
from src.splits import load_locked_splits
from src.train import (
    benchmark_inference,
    data_version,
    update_metrics_file,
)


@dataclass
class TunedModel:
    name: str
    estimator: Any
    best_parameters: dict[str, Any]
    search_score: float
    search_seconds: float | None
    candidate_count: int


def build_extra_trees(config: dict[str, Any], seed: int, n_jobs: int) -> Pipeline:
    feature_config = config["features"]
    return Pipeline(
        [
            (
                "column_filter",
                TrainOnlyColumnFilter(
                    max_missing_fraction=float(
                        feature_config["max_missing_fraction"]
                    ),
                    drop_constant=bool(feature_config["drop_constant"]),
                ),
            ),
            (
                "imputer",
                SimpleImputer(
                    strategy=feature_config["imputation_strategy"],
                    add_indicator=bool(feature_config["add_missing_indicators"]),
                ),
            ),
            (
                "model",
                ExtraTreesClassifier(
                    n_estimators=400,
                    random_state=seed,
                    n_jobs=n_jobs,
                ),
            ),
        ]
    )


def _search_cv(config: dict[str, Any]) -> StratifiedKFold:
    settings = config["improvement"]
    return StratifiedKFold(
        n_splits=int(settings["search_cv_folds"]),
        shuffle=True,
        random_state=int(settings["search_seed"]),
    )


def _save_search_results(search: Any, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
    pending = output_path.with_suffix(".csv.tmp")
    frame.to_csv(pending, index=False)
    pending.replace(output_path)


def _run_search(
    search: GridSearchCV | RandomizedSearchCV,
    X: pd.DataFrame,
    y: pd.Series,
    output_path: Path,
    reuse_search: bool,
) -> TunedModel:
    if reuse_search and output_path.exists():
        frame = pd.read_csv(output_path, float_precision="round_trip")
        parameters = frame["params"].map(ast.literal_eval)
        if isinstance(search, GridSearchCV):
            expected = ParameterGrid(search.param_grid)
        else:
            expected = ParameterSampler(
                search.param_distributions, search.n_iter, random_state=search.random_state
            )
        fold_columns = [f"split{i}_test_score" for i in range(search.cv.get_n_splits())]
        saved_candidates = {tuple(sorted(params.items())) for params in parameters}
        expected_candidates = {tuple(sorted(params.items())) for params in expected}
        best_index = frame["mean_test_score"].idxmax()
        if (
            len(parameters) != len(expected)
            or saved_candidates != expected_candidates
            or not np.isfinite(frame[fold_columns].to_numpy()).all()
            or not np.allclose(frame[fold_columns].mean(axis=1), frame["mean_test_score"])
            or frame.loc[best_index, "rank_test_score"] != 1
        ):
            raise ValueError(f"Saved results do not match the search: {output_path}")
        best_parameters = parameters.loc[best_index]
        score = float(frame.loc[best_index, "mean_test_score"])
        elapsed = None
        count = len(frame)
    else:
        start = time.perf_counter()
        search.fit(X, y)
        elapsed = time.perf_counter() - start
        _save_search_results(search, output_path)
        best_parameters = search.best_params_
        score = float(search.best_score_)
        count = len(search.cv_results_["params"])

    return TunedModel(
        output_path.stem,
        clone(search.estimator).set_params(**best_parameters),
        best_parameters,
        score,
        elapsed,
        count,
    )


def tune_logistic(
    config: dict[str, Any], X: pd.DataFrame, y: pd.Series, output_dir: Path,
    reuse_search: bool = False,
) -> TunedModel:
    search_config = config["logistic_search"]
    base_parameters = {
        "C": 1.0,
        "penalty": "l2",
        "class_weight": None,
        "solver": search_config["solver"],
        "max_iter": search_config["max_iter"],
    }
    estimator = build_logistic(
        config["features"], base_parameters, int(config["project"]["random_seed"])
    )
    grid = {
        "model__C": search_config["c_values"],
        "model__penalty": search_config["penalties"],
        "model__class_weight": search_config["class_weights"],
    }
    search = GridSearchCV(
        estimator,
        param_grid=grid,
        scoring="average_precision",
        cv=_search_cv(config),
        n_jobs=int(config["improvement"]["outer_search_jobs"]),
        refit=False,
        error_score="raise",
        return_train_score=False,
    )
    return _run_search(search, X, y, output_dir / "logistic_regression_cv.csv", reuse_search)


def tune_xgboost(
    config: dict[str, Any], X: pd.DataFrame, y: pd.Series, output_dir: Path,
    reuse_search: bool = False,
) -> TunedModel:
    seed = int(config["project"]["random_seed"])
    initial = {
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.03,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "reg_alpha": 0.1,
        "reg_lambda": 10.0,
        "scale_pos_weight": 1.0,
    }
    estimator = build_xgboost(config["features"], initial, seed=seed, n_jobs=1)
    search = RandomizedSearchCV(
        estimator,
        param_distributions=config["xgboost_search_space"],
        n_iter=int(config["improvement"]["xgboost_iterations"]),
        scoring="average_precision",
        cv=_search_cv(config),
        n_jobs=int(config["improvement"]["outer_search_jobs"]),
        pre_dispatch=int(config["improvement"]["outer_search_jobs"]),
        random_state=int(config["improvement"]["search_seed"]),
        refit=False,
        error_score="raise",
        return_train_score=False,
        verbose=1,
    )
    result = _run_search(search, X, y, output_dir / "xgboost_cv.csv", reuse_search)
    result.estimator.set_params(model__n_jobs=int(config["runtime"]["n_jobs"]))
    return result


def tune_extra_trees(
    config: dict[str, Any], X: pd.DataFrame, y: pd.Series, output_dir: Path,
    reuse_search: bool = False,
) -> TunedModel:
    estimator = build_extra_trees(
        config, int(config["project"]["random_seed"]), n_jobs=1
    )
    search = RandomizedSearchCV(
        estimator,
        param_distributions=config["extratrees_search_space"],
        n_iter=int(config["improvement"]["extratrees_iterations"]),
        scoring="average_precision",
        cv=_search_cv(config),
        n_jobs=int(config["improvement"]["outer_search_jobs"]),
        pre_dispatch=int(config["improvement"]["outer_search_jobs"]),
        random_state=int(config["improvement"]["search_seed"]),
        refit=False,
        error_score="raise",
        return_train_score=False,
        verbose=1,
    )
    result = _run_search(search, X, y, output_dir / "extratrees_cv.csv", reuse_search)
    result.estimator.set_params(model__n_jobs=int(config["runtime"]["n_jobs"]))
    return result


def out_of_fold_probabilities(
    estimator: Any,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int,
    seed: int,
) -> np.ndarray:
    probabilities = np.full(len(X), np.nan, dtype=float)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_positions, validation_positions in splitter.split(X, y):
        fold_estimator = clone(estimator)
        fold_estimator.fit(X.iloc[train_positions], y.iloc[train_positions])
        probabilities[validation_positions] = positive_class_probability(
            fold_estimator, X.iloc[validation_positions]
        )
    if np.isnan(probabilities).any():
        raise RuntimeError("OOF generation did not predict every development row.")
    return probabilities


def blend_weight_grid(step: float) -> list[tuple[float, float, float]]:
    units = int(round(1.0 / step))
    return [
        (first / units, second / units, (units - first - second) / units)
        for first in range(units + 1)
        for second in range(units - first + 1)
    ]


def select_blend_weights(
    y: pd.Series, probability_matrix: np.ndarray, step: float
) -> tuple[list[float], float]:
    best_weights: tuple[float, float, float] | None = None
    best_score = -np.inf
    for weights in blend_weight_grid(step):
        blended = probability_matrix @ np.asarray(weights)
        score = float(average_precision_score(y, blended))
        if score > best_score:
            best_score = score
            best_weights = weights
    if best_weights is None:
        raise RuntimeError("No blend weights were evaluated.")
    return list(best_weights), best_score


def _model_version(name: str) -> str:
    if name == "xgboost_cv":
        return version("xgboost")
    if name == "ensemble_cv":
        return f"scikit-learn={version('scikit-learn')};xgboost={version('xgboost')}"
    return version("scikit-learn")


def run_improvement(config: dict[str, Any], *, reuse_search: bool = False) -> pd.DataFrame:
    dataset = load_secom(config)
    split_frame = load_locked_splits(config).set_index("row_id")
    development_names = set(config["improvement"]["development_splits"])
    development_ids = split_frame.index[split_frame["split"].isin(development_names)]
    test_ids = split_frame.index[split_frame["split"] == "test"]
    if set(development_ids) & set(test_ids):
        raise ValueError("Development and test row IDs overlap.")

    X_development = dataset.features.loc[development_ids].reset_index(drop=True)
    y_development = dataset.targets.loc[development_ids].reset_index(drop=True)
    X_test = dataset.features.loc[test_ids]
    y_test = dataset.targets.loc[test_ids]

    results_dir = project_path(config["paths"]["results"])
    tuning_dir = results_dir / "tuning" / "baseline_v2"
    model_dir = results_dir / "models"
    prediction_dir = project_path(config["paths"]["predictions"])
    for directory in (tuning_dir, model_dir, prediction_dir):
        directory.mkdir(parents=True, exist_ok=True)

    tuners = (tune_logistic, tune_xgboost, tune_extra_trees)
    tuned: list[TunedModel] = []
    for tuner in tuners:
        print(f"Running {tuner.__name__} ...", flush=True)
        result = tuner(config, X_development, y_development, tuning_dir, reuse_search)
        tuned.append(result)
        timing = (
            "reused saved search" if result.search_seconds is None
            else f"seconds={result.search_seconds:.1f}"
        )
        print(
            f"  CV AP={result.search_score:.6f}, candidates={result.candidate_count}, "
            f"{timing}",
            flush=True,
        )

    oof_probabilities: dict[str, np.ndarray] = {}
    fitted_models: dict[str, Any] = {}
    oof_folds = int(config["improvement"]["oof_cv_folds"])
    oof_seed = int(config["improvement"]["oof_seed"])
    for result in tuned:
        print(f"Generating OOF predictions for {result.name} ...", flush=True)
        oof_probabilities[result.name] = out_of_fold_probabilities(
            result.estimator,
            X_development,
            y_development,
            n_splits=oof_folds,
            seed=oof_seed,
        )
        final_estimator = clone(result.estimator)
        final_estimator.fit(X_development, y_development)
        fitted_models[result.name] = final_estimator

    ordered_names = [result.name for result in tuned]
    probability_matrix = np.column_stack(
        [oof_probabilities[name] for name in ordered_names]
    )
    weights, blend_oof_score = select_blend_weights(
        y_development,
        probability_matrix,
        float(config["improvement"]["ensemble_weight_step"]),
    )
    blend = ProbabilityBlendClassifier(
        [fitted_models[name] for name in ordered_names], weights
    )
    oof_probabilities["ensemble_cv"] = probability_matrix @ np.asarray(weights)
    fitted_models["ensemble_cv"] = blend

    development_scores = {
        name: float(average_precision_score(y_development, probabilities))
        for name, probabilities in oof_probabilities.items()
    }
    pretest_selection = max(development_scores, key=development_scores.get)
    selection_record = {
        "selection_basis": "development OOF average precision only",
        "development_rows": len(X_development),
        "development_failures": int(y_development.sum()),
        "oof_scores": development_scores,
        "ensemble_order": ordered_names,
        "ensemble_weights": weights,
        "ensemble_oof_score": blend_oof_score,
        "selected_before_test": pretest_selection,
    }
    (results_dir / "baseline_v2_pretest_selection.json").write_text(
        json.dumps(selection_record, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    metric_rows: list[dict[str, Any]] = []
    tuned_by_name = {result.name: result for result in tuned}
    for name, estimator in fitted_models.items():
        oof_scores = oof_probabilities[name]
        threshold, _ = select_max_f1_threshold(y_development, oof_scores)
        test_scores = positive_class_probability(estimator, X_test).astype(float)
        test_metrics = binary_metrics(
            y_test,
            test_scores,
            threshold,
            float(config["selection"]["recall_fixed_precision"]),
        )
        latency_ms = benchmark_inference(
            estimator,
            X_test,
            warmups=int(config["benchmark"]["warmup_runs"]),
            repeats=int(config["benchmark"]["measurement_runs"]),
            cpu_threads=int(config["runtime"]["cpu_threads"]),
        )
        run_id = f"{config['experiment']['id']}_{name}_seed{config['project']['random_seed']}"
        predictions = pd.DataFrame(
            {
                "run_id": run_id,
                "model": name,
                "split": "test",
                "row_id": test_ids,
                "y_true": y_test.to_numpy(),
                "y_score": test_scores,
                "threshold": threshold,
                "y_pred": (test_scores >= threshold).astype(int),
            }
        )
        predictions.to_csv(prediction_dir / f"{run_id}.csv", index=False)
        joblib.dump(estimator, model_dir / f"{run_id}.joblib")

        if name == "ensemble_cv":
            best_parameters: dict[str, Any] = {
                "model_order": ordered_names,
                "weights": weights,
            }
            candidate_count = len(blend_weight_grid(float(config["improvement"]["ensemble_weight_step"])))
            search_seconds = 0.0
            selected_feature_count = None
        else:
            tuned_result = tuned_by_name[name]
            best_parameters = tuned_result.best_parameters
            candidate_count = tuned_result.candidate_count
            search_seconds = tuned_result.search_seconds
            selected_feature_count = len(
                estimator.named_steps["column_filter"].selected_columns_
            )
        metadata = {
            "run_id": run_id,
            "model": name,
            "model_version": _model_version(name),
            "best_parameters": best_parameters,
            "development_oof_average_precision": development_scores[name],
            "selected_threshold": threshold,
            "selected_feature_count": selected_feature_count,
            "selected_before_test": name == pretest_selection,
        }
        (model_dir / f"{run_id}.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        metric_rows.append(
            {
                "run_id": run_id,
                "dataset_version": data_version(),
                "split_id": config["experiment"]["split_id"],
                "model": name,
                "model_version": _model_version(name),
                "input_modalities": "590 anonymized numeric process measurements",
                "train_budget": json.dumps(
                    {
                        "development_rows": len(X_development),
                        "positive_rows": int(y_development.sum()),
                        "search_candidates": candidate_count,
                        "search_cv_folds": int(config["improvement"]["search_cv_folds"]),
                        "oof_cv_folds": oof_folds,
                    },
                    sort_keys=True,
                ),
                "primary_metric": test_metrics["average_precision"],
                "validation_primary_metric": development_scores[name],
                "secondary_metrics": json.dumps(
                    {key: value for key, value in test_metrics.items() if key != "average_precision"},
                    sort_keys=True,
                ),
                "latency_ms": latency_ms,
                "peak_vram_mb": None,
                "process_rss_mb": None,
                "robustness_drop": None,
                "fit_seconds": None,
                "tuning_seconds": search_seconds,
                "selected_feature_count": selected_feature_count,
                **{f"test_{key}": value for key, value in test_metrics.items()},
                "notes": (
                    "Selected before test by development OOF AP."
                    if name == pretest_selection
                    else "Predefined comparison model; test not used for tuning."
                ),
            }
        )

    update_metrics_file(results_dir / "metrics.csv", metric_rows)
    metrics = pd.DataFrame(metric_rows)
    _write_improvement_report(config, metrics, selection_record)
    return metrics


def _write_improvement_report(
    config: dict[str, Any], metrics: pd.DataFrame, selection: dict[str, Any]
) -> None:
    lines = [
        "| Model | Development OOF PR-AUC | Test PR-AUC | Precision | Recall | F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in metrics.sort_values("validation_primary_metric", ascending=False).itertuples(index=False):
        lines.append(
            f"| {row.model} | {row.validation_primary_metric:.6f} | "
            f"{row.primary_metric:.6f} | {row.test_precision:.6f} | "
            f"{row.test_recall:.6f} | {row.test_f1:.6f} |"
        )
    weights = ", ".join(
        f"{name}={weight:.2f}"
        for name, weight in zip(
            selection["ensemble_order"], selection["ensemble_weights"], strict=True
        )
    )
    report = f"""# 개선 Baseline v2 보고서

## 선택 절차

- 개발 풀: 기존 train+validation 1,253행, 불량 83개
- 잠긴 test 314행은 CV 탐색·앙상블 가중치·threshold 선택에 사용하지 않음
- Logistic Regression: 전체 grid, 5-fold CV
- XGBoost: 32개 조합, 5-fold CV
- ExtraTrees: 24개 조합, 5-fold CV
- 최종 threshold와 앙상블 가중치: 별도 seed의 5-fold OOF 예측

Test를 보기 전에 OOF PR-AUC로 선택된 모델: `{selection['selected_before_test']}`

앙상블 가중치: {weights}

## 결과

{chr(10).join(lines)}

기존 v1 XGBoost test PR-AUC는 `0.304619`이다. v2의 개선 여부는 test PR-AUC뿐 아니라
개발 OOF 성능과 이후 반복·강건성 실험의 변동성을 함께 보고 판단한다. 이미 공개된 test
결과에 맞춘 수동 파라미터 수정은 하지 않았다.
"""
    reports_dir = project_path(config["paths"]["reports"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "stage_2_5_improved_baselines.md").write_text(
        report, encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--reuse-search", action="store_true",
        help="Reuse complete CV CSVs from the same data, split, config, and package versions.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    metrics = run_improvement(config, reuse_search=args.reuse_search)
    print("\nImproved baseline results:")
    print(
        metrics[
            [
                "model",
                "validation_primary_metric",
                "primary_metric",
                "test_precision",
                "test_recall",
                "test_f1",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
