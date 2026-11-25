"""Download, load, validate, and document the UCI SECOM dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ID = "UCI-179-SECOM"
EXPECTED_ROWS = 1567
OFFICIAL_LABEL_MAPPING = {-1: "pass", 1: "fail"}
LICENSE_NAME = "Creative Commons Attribution 4.0 International (CC BY 4.0)"
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"


class DataNotFoundError(FileNotFoundError):
    """Raised when the expected official raw files are not present."""


@dataclass(frozen=True)
class SecomDataset:
    """Aligned SECOM features, labels, and timestamps."""

    features: pd.DataFrame
    raw_labels: pd.Series
    targets: pd.Series
    timestamps: pd.Series


def project_path(configured_path: str | Path) -> Path:
    path = Path(configured_path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def required_raw_paths(config: dict[str, Any]) -> dict[str, Path]:
    raw_dir = project_path(config["paths"]["raw_data"])
    return {
        filename: raw_dir / filename
        for filename in config["dataset"]["expected_files"]
    }


def missing_data_message(config: dict[str, Any]) -> str:
    raw_dir = project_path(config["paths"]["raw_data"])
    expected = ", ".join(config["dataset"]["expected_files"])
    return (
        f"SECOM raw data is missing from: {raw_dir}\n"
        f"Expected files: {expected}\n"
        "Automatic download:\n"
        "  python -m src.data --config configs/base.yaml --download --validate\n"
        f"Manual download: {config['dataset']['download_url']}\n"
        f"Extract the files into: {raw_dir}"
    )


def download_secom(config: dict[str, Any], force: bool = False) -> list[Path]:
    """Download the official archive and safely extract expected files."""

    raw_dir = project_path(config["paths"]["raw_data"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive_path = raw_dir / "secom.zip"
    partial_path = raw_dir / "secom.zip.part"
    download_url = config["dataset"]["download_url"]

    if force or not archive_path.is_file():
        request = urllib.request.Request(
            download_url,
            headers={"User-Agent": "semicon-fault-ai/0.1 (UCI dataset download)"},
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                with partial_path.open("wb") as output_file:
                    shutil.copyfileobj(response, output_file)
            partial_path.replace(archive_path)
        except (OSError, urllib.error.URLError) as error:
            partial_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"Failed to download official SECOM archive: {download_url}"
            ) from error

    expected_names = set(config["dataset"]["expected_files"])
    extracted: list[Path] = []
    try:
        with zipfile.ZipFile(archive_path) as archive:
            members_by_name = {
                Path(member.filename).name: member
                for member in archive.infolist()
                if not member.is_dir()
            }
            missing_in_archive = expected_names - members_by_name.keys()
            if missing_in_archive:
                raise RuntimeError(
                    "Official archive is missing expected files: "
                    + ", ".join(sorted(missing_in_archive))
                )

            for filename in sorted(expected_names):
                destination = raw_dir / filename
                if force or not destination.is_file():
                    with archive.open(members_by_name[filename]) as source:
                        with destination.open("wb") as output_file:
                            shutil.copyfileobj(source, output_file)
                extracted.append(destination)
    except zipfile.BadZipFile as error:
        raise RuntimeError(f"Downloaded file is not a valid ZIP: {archive_path}") from error

    return [archive_path, *extracted]


def load_secom(config: dict[str, Any]) -> SecomDataset:
    """Load official raw files and enforce row/label/timestamp alignment."""

    paths = required_raw_paths(config)
    missing = [path for path in paths.values() if not path.is_file()]
    if missing:
        raise DataNotFoundError(missing_data_message(config))

    feature_path = paths["secom.data"]
    label_path = paths["secom_labels.data"]

    features = pd.read_csv(
        feature_path,
        sep=r"\s+",
        header=None,
        na_values=["NaN"],
        dtype=float,
    )
    features.columns = [f"feature_{index:03d}" for index in range(features.shape[1])]
    features.index = pd.RangeIndex(len(features), name="row_id")

    label_frame = pd.read_csv(
        label_path,
        sep=r"\s+",
        header=None,
        quotechar='"',
        names=["raw_label", "timestamp"],
    )
    if label_frame.shape[1] != 2:
        raise ValueError(
            f"Expected two label columns (label, timestamp), got {label_frame.shape[1]}."
        )
    if len(features) != len(label_frame):
        raise ValueError(
            "Feature/label row mismatch: "
            f"features={len(features)}, labels={len(label_frame)}"
        )

    raw_labels = pd.to_numeric(label_frame["raw_label"], errors="raise").astype(int)
    observed_labels = set(raw_labels.unique().tolist())
    expected_labels = set(OFFICIAL_LABEL_MAPPING)
    if observed_labels != expected_labels:
        raise ValueError(
            f"Unexpected label values: {sorted(observed_labels)}; "
            f"expected {sorted(expected_labels)}."
        )

    timestamps = pd.to_datetime(
        label_frame["timestamp"],
        format="%d/%m/%Y %H:%M:%S",
        errors="raise",
    )
    raw_labels.index = features.index
    timestamps.index = features.index
    targets = raw_labels.eq(1).astype("int8").rename("is_fail")

    return SecomDataset(
        features=features,
        raw_labels=raw_labels.rename("raw_label"),
        targets=targets,
        timestamps=timestamps.rename("timestamp"),
    )


def validate_secom(dataset: SecomDataset) -> dict[str, Any]:
    """Return a JSON-serializable profile and fail on structural violations."""

    row_count, feature_count = dataset.features.shape
    if row_count != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} rows, got {row_count}.")
    if not dataset.features.index.equals(dataset.targets.index):
        raise ValueError("Feature and target row IDs are not aligned.")
    if not dataset.features.index.equals(dataset.timestamps.index):
        raise ValueError("Feature and timestamp row IDs are not aligned.")

    numeric_values = dataset.features.to_numpy(dtype=float)
    if np.isinf(numeric_values).any():
        raise ValueError("Feature matrix contains positive or negative infinity.")

    missing_by_column = dataset.features.isna().sum().sort_values(ascending=False)
    label_counts = dataset.raw_labels.value_counts().sort_index()
    constant_columns = [
        column
        for column in dataset.features.columns
        if dataset.features[column].nunique(dropna=True) <= 1
    ]
    all_missing_columns = [
        column for column in dataset.features.columns if dataset.features[column].isna().all()
    ]

    return {
        "rows": row_count,
        "raw_feature_columns": feature_count,
        "label_counts": {
            OFFICIAL_LABEL_MAPPING[int(label)]: int(count)
            for label, count in label_counts.items()
        },
        "fail_count": int(dataset.targets.sum()),
        "fail_rate": float(dataset.targets.mean()),
        "missing_cells": int(dataset.features.isna().sum().sum()),
        "missing_rate": float(dataset.features.isna().to_numpy().mean()),
        "columns_with_missing": int((missing_by_column > 0).sum()),
        "all_missing_columns": all_missing_columns,
        "constant_columns": constant_columns,
        "duplicate_feature_rows": int(dataset.features.duplicated().sum()),
        "timestamp_min": dataset.timestamps.min().isoformat(),
        "timestamp_max": dataset.timestamps.max().isoformat(),
        "top_missing_columns": {
            column: int(count)
            for column, count in missing_by_column.head(10).items()
            if count > 0
        },
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_validation_artifacts(
    config: dict[str, Any], dataset: SecomDataset, profile: dict[str, Any]
) -> tuple[Path, Path, Path]:
    """Write the manifest, complete column dictionary, and human-readable report."""

    data_dir = PROJECT_ROOT / "data"
    reports_dir = project_path(config["paths"]["reports"])
    data_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    raw_dir = project_path(config["paths"]["raw_data"])
    manifest_path = data_dir / "data_manifest.json"
    dictionary_path = data_dir / "data_dictionary.md"
    report_path = reports_dir / "stage_1_data.md"

    raw_files = [raw_dir / "secom.zip", *required_raw_paths(config).values()]
    manifest = {
        "dataset_id": DATASET_ID,
        "dataset_name": config["dataset"]["name"],
        "source_page": config["dataset"]["source"],
        "download_url": config["dataset"]["download_url"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "license": {"name": LICENSE_NAME, "url": LICENSE_URL},
        "label_mapping": {"-1": "pass", "1": "fail"},
        "files": [
            {
                "path": path.relative_to(PROJECT_ROOT).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in raw_files
            if path.is_file()
        ],
        "profile": profile,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    dictionary_lines = [
        "# SECOM Data Dictionary",
        "",
        f"Source: {config['dataset']['source']}",
        "",
        "The sensor/process feature meanings are anonymized by the dataset provider. ",
        "Project names are positional and zero-based: `feature_000` is the first raw column.",
        "",
        "| Column | Role | Type | Missing | Missing rate | Unique non-null | Notes |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for column in dataset.features.columns:
        series = dataset.features[column]
        missing_count = int(series.isna().sum())
        dictionary_lines.append(
            f"| {column} | anonymized process measurement | float64 | "
            f"{missing_count} | {missing_count / len(series):.6f} | "
            f"{int(series.nunique(dropna=True))} | raw positional column |"
        )
    dictionary_lines.extend(
        [
            "| raw_label | original target | int | 0 | 0.000000 | 2 | -1=pass, 1=fail |",
            "| is_fail | normalized target | int8 | 0 | 0.000000 | 2 | 0=pass, 1=fail |",
            "| timestamp | test timestamp | datetime64 | 0 | 0.000000 | "
            f"{dataset.timestamps.nunique()} | parsed as day/month/year |",
            "",
        ]
    )
    dictionary_path.write_text("\n".join(dictionary_lines), encoding="utf-8")

    report = _render_stage_report(config, profile, manifest_path, dictionary_path)
    report_path.write_text(report, encoding="utf-8")
    return manifest_path, dictionary_path, report_path


def _render_stage_report(
    config: dict[str, Any],
    profile: dict[str, Any],
    manifest_path: Path,
    dictionary_path: Path,
) -> str:
    metadata_difference = 591 - int(profile["raw_feature_columns"])
    return f"""# 1단계 SECOM 데이터 확보 및 검증 보고서

## 출처와 이용 조건

- 공식 출처: {config['dataset']['source']}
- 데이터셋 ID: {DATASET_ID}
- 라이선스: {LICENSE_NAME}
- 라벨 매핑: `-1=pass`, `1=fail`

## 실제 파일 검증 결과

| 항목 | 결과 |
|---|---:|
| 행 수 | {profile['rows']} |
| 원본 센서 열 수 | {profile['raw_feature_columns']} |
| 정상(pass) | {profile['label_counts']['pass']} |
| 불량(fail) | {profile['label_counts']['fail']} |
| 불량률 | {profile['fail_rate']:.4%} |
| 결측 셀 | {profile['missing_cells']} |
| 전체 셀 대비 결측률 | {profile['missing_rate']:.4%} |
| 결측이 있는 열 | {profile['columns_with_missing']} |
| 전부 결측인 열 | {len(profile['all_missing_columns'])} |
| 상수 열 | {len(profile['constant_columns'])} |
| 중복 특징 행 | {profile['duplicate_feature_rows']} |
| 최초 타임스탬프 | {profile['timestamp_min']} |
| 최종 타임스탬프 | {profile['timestamp_max']} |

UCI 웹 메타데이터의 591 features와 원본 `secom.data`의 센서 열 수는
{metadata_difference}개 차이가 있다. 이 프로젝트는 예측 입력으로 실제 센서 열
{profile['raw_feature_columns']}개를 사용하고, 라벨과 타임스탬프는 별도 열로 관리한다.

## 생성 산출물

- `{manifest_path.relative_to(PROJECT_ROOT).as_posix()}`: 파일 크기와 SHA-256, 프로파일
- `{dictionary_path.relative_to(PROJECT_ROOT).as_posix()}`: 모든 익명 특징의 결측·고유값 사전
- `data/raw/`: 공식 ZIP과 압축 해제 원본(Git 제외)

## 재실행 명령

```powershell
.\\.venv\\Scripts\\python.exe -m src.data --config configs/base.yaml --validate
.\\.venv\\Scripts\\python.exe -m pytest tests -q
```

특징명이 익명화되어 있으므로 이후 중요도나 SHAP 결과는 예측 관련 특징 설명이며 실제
공정 불량 원인으로 단정하지 않는다.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to the YAML config.")
    parser.add_argument("--download", action="store_true", help="Download official data.")
    parser.add_argument(
        "--force-download", action="store_true", help="Replace existing raw files."
    )
    parser.add_argument("--validate", action="store_true", help="Validate and document data.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.download and not args.validate and not args.force_download:
        raise SystemExit("Choose at least one action: --download or --validate")

    config = load_config(args.config)
    try:
        if args.download or args.force_download:
            paths = download_secom(config, force=args.force_download)
            print("Official files ready:")
            for path in paths:
                print(f"  {path.relative_to(PROJECT_ROOT)}")

        if args.validate:
            dataset = load_secom(config)
            profile = validate_secom(dataset)
            artifacts = write_validation_artifacts(config, dataset, profile)
            print(json.dumps(profile, ensure_ascii=False, indent=2))
            print("Validation artifacts:")
            for path in artifacts:
                print(f"  {path.relative_to(PROJECT_ROOT)}")
    except (DataNotFoundError, RuntimeError, ValueError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()

