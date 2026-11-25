"""Create and validate locked row-level splits for SECOM."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import load_config
from src.data import PROJECT_ROOT, load_secom, project_path


VALID_SPLITS = ("train", "validation", "test")


def create_stratified_split_frame(config: dict[str, Any]) -> pd.DataFrame:
    dataset = load_secom(config)
    split_config = config["split"]
    row_ids = dataset.features.index.to_numpy()
    targets = dataset.targets.to_numpy()
    seed = int(split_config["random_seed"])

    train_ids, remainder_ids = train_test_split(
        row_ids,
        train_size=float(split_config["train_fraction"]),
        random_state=seed,
        stratify=targets,
    )
    remainder_targets = dataset.targets.loc[remainder_ids].to_numpy()
    validation_share = float(split_config["validation_fraction"]) / (
        float(split_config["validation_fraction"])
        + float(split_config["test_fraction"])
    )
    validation_ids, test_ids = train_test_split(
        remainder_ids,
        train_size=validation_share,
        random_state=seed,
        stratify=remainder_targets,
    )

    assignments = {
        **{int(row_id): "train" for row_id in train_ids},
        **{int(row_id): "validation" for row_id in validation_ids},
        **{int(row_id): "test" for row_id in test_ids},
    }
    frame = pd.DataFrame(
        {
            "row_id": dataset.features.index,
            "split": [assignments[int(row_id)] for row_id in dataset.features.index],
            "raw_label": dataset.raw_labels.to_numpy(),
            "is_fail": dataset.targets.to_numpy(),
            "timestamp": dataset.timestamps.dt.strftime("%Y-%m-%dT%H:%M:%S").to_numpy(),
        }
    )
    validate_split_frame(frame, expected_rows=len(dataset.features))
    return frame


def validate_split_frame(frame: pd.DataFrame, expected_rows: int) -> None:
    required = {"row_id", "split", "raw_label", "is_fail", "timestamp"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Split file is missing columns: {sorted(missing)}")
    if len(frame) != expected_rows:
        raise ValueError(f"Expected {expected_rows} split rows, got {len(frame)}.")
    if frame["row_id"].duplicated().any():
        raise ValueError("Split row IDs overlap or are duplicated.")
    if set(frame["row_id"]) != set(range(expected_rows)):
        raise ValueError("Split row IDs do not cover the complete dataset.")
    if set(frame["split"]) != set(VALID_SPLITS):
        raise ValueError(f"Expected split names: {VALID_SPLITS}")
    for split_name in VALID_SPLITS:
        subset = frame.loc[frame["split"] == split_name]
        if subset.empty or subset["is_fail"].nunique() != 2:
            raise ValueError(f"Split '{split_name}' must contain both classes.")


def split_summary(frame: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for split_name in VALID_SPLITS:
        subset = frame.loc[frame["split"] == split_name]
        summary[split_name] = {
            "rows": int(len(subset)),
            "pass": int((subset["is_fail"] == 0).sum()),
            "fail": int((subset["is_fail"] == 1).sum()),
            "fail_rate": float(subset["is_fail"].mean()),
            "timestamp_min": str(subset["timestamp"].min()),
            "timestamp_max": str(subset["timestamp"].max()),
        }
    return summary


def time_order_audit(frame: pd.DataFrame) -> dict[str, Any]:
    ordered = frame.sort_values("timestamp").reset_index(drop=True)
    sizes = {name: int((frame["split"] == name).sum()) for name in VALID_SPLITS}
    start = 0
    audit: dict[str, Any] = {}
    for split_name in VALID_SPLITS:
        stop = start + sizes[split_name]
        subset = ordered.iloc[start:stop]
        audit[split_name] = {
            "rows": int(len(subset)),
            "pass": int((subset["is_fail"] == 0).sum()),
            "fail": int((subset["is_fail"] == 1).sum()),
            "fail_rate": float(subset["is_fail"].mean()),
            "timestamp_min": str(subset["timestamp"].min()),
            "timestamp_max": str(subset["timestamp"].max()),
        }
        start = stop
    return audit


def save_locked_splits(
    config: dict[str, Any], output_path: Path, force: bool = False
) -> tuple[pd.DataFrame, Path]:
    new_frame = create_stratified_split_frame(config)
    rendered = new_frame.to_csv(index=False, lineterminator="\n")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.is_file() and not force:
        existing = output_path.read_text(encoding="utf-8")
        if existing != rendered:
            raise FileExistsError(
                f"Locked split differs from regenerated split: {output_path}. "
                "Use --force only after documenting the split change."
            )
    else:
        output_path.write_text(rendered, encoding="utf-8")

    split_hash = hashlib.sha256(rendered.encode("utf-8")).hexdigest()
    manifest = {
        "split_id": config["experiment"]["split_id"]
        if "experiment" in config
        else "secom_stratified_seed42_v1",
        "strategy": config["split"]["strategy"],
        "random_seed": config["split"]["random_seed"],
        "sha256": split_hash,
        "summary": split_summary(new_frame),
        "time_order_feasibility_audit": time_order_audit(new_frame),
    }
    manifest_path = output_path.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return new_frame, manifest_path


def load_locked_splits(config: dict[str, Any]) -> pd.DataFrame:
    split_path = project_path(config["experiment"]["split_file"])
    if not split_path.is_file():
        raise FileNotFoundError(
            f"Locked split not found: {split_path}\n"
            "Run: python -m src.splits --config configs/experiments/baseline.yaml "
            "--save-splits"
        )
    frame = pd.read_csv(split_path)
    validate_split_frame(frame, expected_rows=len(frame))
    return frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--save-splits", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.save_splits:
        raise SystemExit("Choose --save-splits")
    config = load_config(args.config)
    output_path = project_path(config["experiment"]["split_file"])
    frame, manifest_path = save_locked_splits(config, output_path, force=args.force)
    print(json.dumps(split_summary(frame), ensure_ascii=False, indent=2))
    print(f"Locked split: {output_path.relative_to(PROJECT_ROOT)}")
    print(f"Manifest: {manifest_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
