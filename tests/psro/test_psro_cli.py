from __future__ import annotations

from pathlib import Path

from fugitive.psro.cli import main
from fugitive.psro.checkpoint import PSROExperimentCheckpoint


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


def test_cli_start_is_deterministic_and_resume_uses_its_checkpoint(tmp_path) -> None:
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"

    assert main(_start_arguments(first_path)) == 0
    assert main(_start_arguments(second_path)) == 0

    first = PSROExperimentCheckpoint.from_json(first_path.read_text("utf-8"))
    second = PSROExperimentCheckpoint.from_json(second_path.read_text("utf-8"))
    assert first == second
    assert first.psro.generation == 0
    assert len(first.psro.payoff_matrix.entries) == 1
    entry = first.psro.payoff_matrix.entries[0]
    assert entry.estimate.marshal_payoff in (0.0, 1.0)
    assert entry.estimate.sample_count == 1
    assert entry.estimate.standard_error is None
    assert first.run_config.payoff_config.max_decisions is None

    assert main(["--resume", str(first_path), "--generations", "0"]) == 0
    resumed = PSROExperimentCheckpoint.from_json(first_path.read_text("utf-8"))
    assert resumed == first
