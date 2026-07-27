from __future__ import annotations

from pathlib import Path

from fugitive.psro_cli import main
from fugitive.psro_experiment import PSROExperimentCheckpoint


def _start_arguments(output: Path) -> list[str]:
    return [
        "--fugitive",
        "hierarchical-random",
        "--marshal",
        "support-catalogue-random",
        "--games-per-cell",
        "1",
        "--generations",
        "0",
        "--seed",
        "91",
        "--solver-iterations",
        "10",
        "--solver-check-interval",
        "1",
        "--workers",
        "1",
        "--quiet-progress",
        "--output",
        str(output),
    ]


def test_cli_writes_a_deterministic_self_contained_generation_zero(tmp_path) -> None:
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"

    assert main(_start_arguments(first_path)) == 0
    assert main(_start_arguments(second_path)) == 0

    first = PSROExperimentCheckpoint.from_json(first_path.read_text("utf-8"))
    second = PSROExperimentCheckpoint.from_json(second_path.read_text("utf-8"))
    assert first == second
    assert first.psro.generation == 0
    assert len(first.psro.payoff_matrix.entries) == 1
    assert first.run_config.payoff_config.max_decisions is None


def test_cli_resume_uses_the_checkpoint_configuration(tmp_path) -> None:
    path = tmp_path / "checkpoint.json"
    assert main(_start_arguments(path)) == 0
    before = PSROExperimentCheckpoint.from_json(path.read_text("utf-8"))

    assert main(["--resume", str(path), "--generations", "0"]) == 0
    after = PSROExperimentCheckpoint.from_json(path.read_text("utf-8"))
    assert after == before
