"""Recompute test metrics from raw predictions and create baseline figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, precision_recall_curve

from src.config import load_config
from src.data import PROJECT_ROOT, project_path
from src.evaluation import binary_metrics
from src.splits import load_locked_splits, split_summary


def evaluate_predictions(config: dict[str, Any], split: str) -> pd.DataFrame:
    prediction_dir = project_path(config["paths"]["predictions"])
    prediction_paths = sorted(prediction_dir.glob("*.csv"))
    if not prediction_paths:
        raise FileNotFoundError(
            f"No raw predictions found in {prediction_dir}. Run src.train first."
        )
    metrics_path = project_path(config["paths"]["results"]) / "metrics.csv"
    stored_metrics = pd.read_csv(metrics_path, float_precision="round_trip")
    fixed_precision = float(config["selection"]["recall_fixed_precision"])
    rows: list[dict[str, Any]] = []

    for prediction_path in prediction_paths:
        predictions = pd.read_csv(prediction_path, float_precision="round_trip")
        subset = predictions.loc[predictions["split"] == split].copy()
        if subset.empty:
            continue
        if subset["row_id"].duplicated().any():
            raise ValueError(f"Duplicate row IDs in {prediction_path}")
        if subset["threshold"].nunique() != 1:
            raise ValueError(f"Threshold changed within {prediction_path}")
        threshold = float(subset["threshold"].iloc[0])
        labels = (subset["y_score"] >= threshold).astype(int)
        if not labels.equals(subset["y_pred"]):
            raise ValueError(f"Predicted labels do not match scores in {prediction_path}")
        calculated = binary_metrics(
            subset["y_true"],
            subset["y_score"],
            threshold=threshold,
            fixed_precision=fixed_precision,
        )
        run_id = str(subset["run_id"].iloc[0])
        model = str(subset["model"].iloc[0])
        expected_row = stored_metrics.loc[stored_metrics["run_id"] == run_id]
        if len(expected_row) != 1:
            raise ValueError(f"Expected exactly one metrics row for {run_id}")
        if split == "test":
            for metric, value in calculated.items():
                stored_value = float(expected_row[f"test_{metric}"].iloc[0])
                if abs(stored_value - value) > 1e-12:
                    raise ValueError(f"Raw predictions do not reproduce {metric} for {run_id}")
        rows.append({"run_id": run_id, "model": model, "split": split, **calculated})

    summary = pd.DataFrame(rows).sort_values("average_precision", ascending=False)
    output_path = project_path(config["paths"]["results"]) / "evaluation_summary.csv"
    summary.to_csv(output_path, index=False)
    create_figures(config, split, prediction_paths)
    if split == "test" and config["experiment"]["id"] == "A-E1-baseline":
        run_prefix = f"{config['experiment']['id']}_"
        write_stage_report(config, summary.loc[summary["run_id"].str.startswith(run_prefix)])
    return summary


def create_figures(
    config: dict[str, Any], split: str, prediction_paths: list[Path]
) -> None:
    figure_dir = project_path(config["paths"]["figures"])
    figure_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    fig, axis = plt.subplots(figsize=(8, 6))

    for prediction_path in prediction_paths:
        predictions = pd.read_csv(prediction_path, float_precision="round_trip")
        subset = predictions.loc[predictions["split"] == split]
        if subset.empty:
            continue
        model = str(subset["model"].iloc[0])
        precision, recall, _ = precision_recall_curve(
            subset["y_true"], subset["y_score"]
        )
        ap = binary_metrics(
            subset["y_true"],
            subset["y_score"],
            float(subset["threshold"].iloc[0]),
        )["average_precision"]
        axis.plot(recall, precision, label=f"{model} (AP={ap:.3f})")

        matrix_fig, matrix_axis = plt.subplots(figsize=(4.5, 4))
        ConfusionMatrixDisplay.from_predictions(
            subset["y_true"],
            subset["y_pred"],
            labels=[0, 1],
            display_labels=["pass", "fail"],
            cmap="Blues",
            colorbar=False,
            ax=matrix_axis,
        )
        matrix_axis.set_title(f"{model} — {split}")
        matrix_fig.tight_layout()
        matrix_fig.savefig(
            figure_dir / f"confusion_matrix_{model}_{split}.png", dpi=160
        )
        plt.close(matrix_fig)

    axis.set_xlabel("Recall (fail class)")
    axis.set_ylabel("Precision (fail class)")
    axis.set_title(f"SECOM Precision–Recall Curve — {split}")
    axis.legend(loc="best")
    fig.tight_layout()
    fig.savefig(figure_dir / f"pr_curve_{split}.png", dpi=160)
    plt.close(fig)


def write_stage_report(config: dict[str, Any], summary: pd.DataFrame) -> None:
    reports_dir = project_path(config["paths"]["reports"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    split_frame = load_locked_splits(config)
    splits = split_summary(split_frame)

    result_lines = [
        "| Model | Test PR-AUC | ROC-AUC | Precision | Recall | F1 | Threshold |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        result_lines.append(
            f"| {row.model} | {row.average_precision:.6f} | {row.roc_auc:.6f} | "
            f"{row.precision:.6f} | {row.recall:.6f} | {row.f1:.6f} | "
            f"{row.threshold:.6f} |"
        )

    metrics = pd.read_csv(project_path(config["paths"]["results"]) / "metrics.csv")
    experiment_metrics = metrics.loc[
        metrics["run_id"].isin(summary["run_id"])
    ]
    selection_lines = [
        "| Model | Candidates | Validation PR-AUC | Selected features | Tuning seconds |",
        "|---|---:|---:|---:|---:|",
    ]
    parameter_sections: list[str] = []
    model_dir = project_path(config["paths"]["results"]) / "models"
    for row in experiment_metrics.sort_values("model").itertuples(index=False):
        budget = json.loads(row.train_budget)
        selection_lines.append(
            f"| {row.model} | {budget['candidate_count']} | "
            f"{row.validation_primary_metric:.6f} | {row.selected_feature_count} | "
            f"{row.tuning_seconds:.3f} |"
        )
        metadata_path = model_dir / f"{row.run_id}.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        parameters = json.dumps(
            metadata["best_parameters"], ensure_ascii=False, indent=2, sort_keys=True
        )
        parameter_sections.append(
            f"### {row.model}\n\n```json\n{parameters}\n```"
        )

    split_lines = [
        "| Split | Rows | Pass | Fail | Fail rate |",
        "|---|---:|---:|---:|---:|",
    ]
    for split_name in ("train", "validation", "test"):
        item = splits[split_name]
        split_lines.append(
            f"| {split_name} | {item['rows']} | {item['pass']} | {item['fail']} | "
            f"{item['fail_rate']:.4%} |"
        )

    report = f"""# 2단계 고정 Split 및 Classical Baseline 보고서

## 분할

{chr(10).join(split_lines)}

Split ID: `{config['experiment']['split_id']}`. 모든 모델이 동일한 행 ID를 사용하며 세
집합은 겹치지 않는다. 모델과 threshold 선택은 validation에서만 수행했고 test는 선택이
완료된 뒤 평가했다.

## 전처리 및 탐색

- 열 제거 기준은 train에서만 fit: 결측률 > {config['features']['max_missing_fraction']:.0%}, 상수 열
- Logistic Regression: train 중앙값 대치, 결측 indicator, 표준화
- XGBoost: 결측값을 모델이 직접 처리하고 train-fitted 열 필터만 적용
- Logistic 후보 수: {len(config['logistic_search']['c_values']) * len(config['logistic_search']['penalties']) * len(config['logistic_search']['class_weights'])}
- XGBoost 후보 수: {len(config['xgboost_search']['candidates'])}
- 선택 지표: validation average precision
- 운영 threshold 임시 규칙: validation F1 최대화

{chr(10).join(selection_lines)}

## 선택된 모델 설정

{chr(10).join(parameter_sections)}

## 잠긴 Test 결과

{chr(10).join(result_lines)}

PR-AUC는 threshold와 무관한 순위 성능이며, Precision/Recall/F1은 validation에서 선택한
threshold를 test에 그대로 적용한 결과다. 아직 강건성·저데이터·TabPFN 실험 전이므로 이
표만으로 최종 도입 결론을 내리지 않는다.

Dummy의 모든 확률은 동일하다. validation F1 최대화 규칙이 그 동일 확률을 threshold로
선택하여 test의 모든 행을 불량으로 분류했으므로 Recall 1.0이지만 Precision은 불량률과
같다. 이는 유용한 분류기가 아니라 threshold 규칙의 동작을 보여주는 최소 기준선이다.

Test 불량은 21개뿐이므로 1개 분류 결과가 Recall 약 4.76%p를 바꾼다. 모델 간 차이는 이후
반복·저데이터·강건성 실험의 불확실성과 함께 해석해야 한다.

## 재현 명령

```powershell
.\\.venv\\Scripts\\python.exe -m src.splits --config configs/experiments/baseline.yaml --save-splits
.\\.venv\\Scripts\\python.exe -m src.train --config configs/experiments/baseline.yaml
.\\.venv\\Scripts\\python.exe -m src.evaluate --config configs/experiments/baseline.yaml --split test
.\\.venv\\Scripts\\python.exe -m pytest tests -q
```
"""
    (reports_dir / "stage_2_baselines.md").write_text(report, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    summary = evaluate_predictions(config, args.split)
    print(summary.to_string(index=False))
    print(f"\nRaw predictions reproduced all {args.split} metrics.")


if __name__ == "__main__":
    main()
