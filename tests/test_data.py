from pathlib import Path

import pandas as pd
import pytest

from src.data import (
    EXPECTED_ROWS,
    DataNotFoundError,
    load_secom,
    missing_data_message,
    required_raw_paths,
    validate_secom,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = PROJECT_ROOT / "configs" / "base.yaml"


def _test_config(raw_dir: Path) -> dict:
    return {
        "paths": {"raw_data": str(raw_dir)},
        "dataset": {
            "download_url": "https://archive.ics.uci.edu/static/public/179/secom.zip",
            "expected_files": ["secom.data", "secom_labels.data", "secom.names"],
        },
    }


def _write_small_valid_dataset(raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True)
    (raw_dir / "secom.data").write_text(
        "1.0 NaN 3.0\n2.0 4.0 6.0\n", encoding="utf-8"
    )
    (raw_dir / "secom_labels.data").write_text(
        '-1 "19/07/2008 11:55:00"\n1 "19/07/2008 12:32:00"\n',
        encoding="utf-8",
    )
    (raw_dir / "secom.names").write_text("test metadata\n", encoding="utf-8")


def test_missing_data_message_has_exact_recovery_command(tmp_path: Path) -> None:
    config = _test_config(tmp_path / "raw")

    with pytest.raises(DataNotFoundError, match="--download --validate"):
        load_secom(config)
    assert "secom.data" in missing_data_message(config)


def test_loader_aligns_rows_and_normalizes_fail_label(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    _write_small_valid_dataset(raw_dir)

    dataset = load_secom(_test_config(raw_dir))

    assert dataset.features.shape == (2, 3)
    assert pd.isna(dataset.features.loc[0, "feature_001"])
    assert dataset.raw_labels.tolist() == [-1, 1]
    assert dataset.targets.tolist() == [0, 1]
    assert dataset.timestamps.dt.year.tolist() == [2008, 2008]


def test_loader_rejects_feature_label_row_mismatch(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    _write_small_valid_dataset(raw_dir)
    (raw_dir / "secom_labels.data").write_text(
        '-1 "19/07/2008 11:55:00"\n', encoding="utf-8"
    )

    with pytest.raises(ValueError, match="row mismatch"):
        load_secom(_test_config(raw_dir))


def test_official_raw_data_passes_structural_validation() -> None:
    from src.config import load_config

    config = load_config(BASE_CONFIG)
    if any(not path.is_file() for path in required_raw_paths(config).values()):
        pytest.skip("Official SECOM raw files are not downloaded.")

    dataset = load_secom(config)
    profile = validate_secom(dataset)

    assert profile["rows"] == EXPECTED_ROWS
    assert profile["raw_feature_columns"] == 590
    assert profile["label_counts"] == {"pass": 1463, "fail": 104}
    assert profile["fail_count"] == 104
    assert profile["missing_cells"] > 0
