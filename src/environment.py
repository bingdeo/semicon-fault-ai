"""Print a reproducibility-oriented snapshot of the local execution environment."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from src.config import load_config


def collect_environment(config_path: str | Path) -> dict[str, Any]:
    """Collect stable environment facts without assuming unavailable hardware."""

    config = load_config(config_path)
    memory_total_gb = _memory_total_gb()
    return {
        "project": config["project"]["name"],
        "config": str(Path(config_path).as_posix()),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "operating_system": platform.platform(),
        "machine": platform.machine() or "unknown",
        "processor": platform.processor() or "unavailable",
        "logical_cpu_count": os.cpu_count(),
        "memory_total_gb": memory_total_gb,
        "gpu": _nvidia_gpu_summary(),
        "random_seed": config["project"]["random_seed"],
        "configured_n_jobs": config["runtime"]["n_jobs"],
        "configured_cpu_threads": config["runtime"]["cpu_threads"],
        "optimization_goal": config["runtime"]["optimization_goal"],
        "device_setting": config["runtime"]["device"],
    }


def _memory_total_gb() -> float | None:
    try:
        import psutil
    except ImportError:
        return None
    return round(psutil.virtual_memory().total / (1024**3), 2)


def _nvidia_gpu_summary() -> list[str]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return []

    command = [
        executable,
        "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to the YAML config.")
    parser.add_argument("--out", help="Optional JSON output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    environment = collect_environment(args.config)
    rendered = json.dumps(environment, ensure_ascii=False, indent=2)
    print(rendered)

    if args.out:
        output_path = Path(args.out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
        print(f"Environment snapshot saved to: {output_path}")


if __name__ == "__main__":
    main()

