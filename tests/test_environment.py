from pathlib import Path

from src.environment import collect_environment


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_environment_snapshot_has_reproducibility_fields() -> None:
    snapshot = collect_environment(PROJECT_ROOT / "configs" / "base.yaml")

    assert snapshot["project"] == "semicon-fault-ai"
    assert snapshot["python_version"]
    assert snapshot["python_executable"]
    assert snapshot["logical_cpu_count"] is None or snapshot["logical_cpu_count"] > 0
    assert isinstance(snapshot["gpu"], list)

