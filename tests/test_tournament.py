from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import fugitive.tournament as tournament_module
from fugitive.experiment import ExperimentStatus
from fugitive.model import Winner
from fugitive.tournament import (
    TournamentConfig,
    run_tournament,
    tournament_game_seed,
    wilson_interval,
)


class _FakeManifest:
    def __init__(
        self,
        *,
        status: ExperimentStatus,
        winner: Winner | None,
        decision_count: int = 9,
    ) -> None:
        self.status = status
        self.winner = winner
        self.reason = "fake_terminal" if winner is not None else None
        self.rounds = 4 if winner is not None else None
        self.decision_count = decision_count

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(
            {
                "outcome": {
                    "status": self.status.value,
                    "winner": self.winner.value if self.winner else None,
                }
            },
            indent=indent,
        )


def _fake_run(
    *,
    status: ExperimentStatus = ExperimentStatus.COMPLETED,
    winner: Winner | None = Winner.MARSHAL,
):
    return SimpleNamespace(
        manifest=_FakeManifest(status=status, winner=winner),
        inference_events=(),
        inference_diagnostic_failures=(),
        rollout_events=(),
        rollout_diagnostic_failures=(),
    )


class _DiagnosticRecord:
    def __init__(self, kind: str) -> None:
        self.kind = kind

    def to_dict(self) -> dict[str, str]:
        return {"kind": self.kind}


def _config(tmp_path, *, games: int, root_seed: int = 17) -> TournamentConfig:
    return TournamentConfig(
        fugitive_agents=("hierarchical-random",),
        marshal_agents=("hierarchical-random",),
        games_per_matchup=games,
        root_seed=root_seed,
        output_directory=tmp_path / "results",
    )


def test_matrix_reuses_one_master_seed_per_game_index(monkeypatch, tmp_path) -> None:
    calls: list[dict[str, object]] = []

    def fake_registered_experiment(**kwargs):
        calls.append(kwargs)
        winner = (
            Winner.MARSHAL
            if kwargs["marshal_name"] == "route-count-random"
            else Winner.FUGITIVE
        )
        return _fake_run(winner=winner)

    monkeypatch.setattr(
        tournament_module,
        "run_registered_experiment",
        fake_registered_experiment,
    )
    config = TournamentConfig(
        fugitive_agents=("hierarchical-random", "belief-informed-random"),
        marshal_agents=("hierarchical-random", "route-count-random"),
        games_per_matchup=2,
        root_seed=23,
        output_directory=tmp_path / "matrix",
    )

    report = run_tournament(config)

    assert len(calls) == 8
    assert [call["master_seed"] for call in calls[:4]] == [
        tournament_game_seed(23, 0)
    ] * 4
    assert [call["master_seed"] for call in calls[4:]] == [
        tournament_game_seed(23, 1)
    ] * 4
    assert all(call["max_decisions"] is None for call in calls)
    assert all(call["validate_invariants"] is True for call in calls)
    assert len(report.matchups) == 4
    assert all(item.completed == 2 for item in report.matchups)
    route_count_cells = [
        item for item in report.matchups if item.marshal_agent == "route-count-random"
    ]
    assert all(item.marshal_win_rate == 1.0 for item in route_count_cells)

    output = config.output_directory
    assert len((output / "games.jsonl").read_text().splitlines()) == 8
    assert len(tuple((output / "manifests").rglob("*.json"))) == 8
    assert len(tuple((output / "diagnostics").rglob("*.json"))) == 8
    for filename in ("config.json", "summary.json", "summary.csv", "summary.md"):
        assert (output / filename).is_file()


def test_tournament_persists_rollout_diagnostics_and_counts(
    monkeypatch,
    tmp_path,
) -> None:
    def fake_registered_experiment(**_kwargs):
        run = _fake_run()
        run.rollout_events = (
            _DiagnosticRecord("rollout-event-1"),
            _DiagnosticRecord("rollout-event-2"),
        )
        run.rollout_diagnostic_failures = (
            _DiagnosticRecord("rollout-failure"),
        )
        return run

    monkeypatch.setattr(
        tournament_module,
        "run_registered_experiment",
        fake_registered_experiment,
    )
    config = _config(tmp_path, games=1)

    report = run_tournament(config)

    record = report.records[0]
    matchup = report.matchups[0]
    assert record.rollout_events == matchup.rollout_events == 2
    assert record.rollout_diagnostic_failures == (
        matchup.rollout_diagnostic_failures
    ) == 1
    stored = json.loads(
        (config.output_directory / record.diagnostics_path).read_text(
            encoding="utf-8"
        )
    )
    assert stored["schema_version"] == (
        tournament_module.TOURNAMENT_DIAGNOSTICS_SCHEMA_VERSION
    )
    assert stored["configuration_hash"] == config.configuration_hash
    assert stored["game_key"] == record.key
    assert stored["rollout_events"] == [
        {"kind": "rollout-event-1"},
        {"kind": "rollout-event-2"},
    ]
    assert stored["rollout_failures"] == [
        {"kind": "rollout-failure"}
    ]


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    (
        ("schema_version", 999, "unsupported schema"),
        ("configuration_hash", "wrong-configuration", "wrong configuration"),
        ("game_key", "wrong-game", "wrong game key"),
    ),
)
def test_resume_rejects_mismatched_diagnostics_sidecar_metadata(
    monkeypatch,
    tmp_path,
    field,
    replacement,
    message,
) -> None:
    monkeypatch.setattr(
        tournament_module,
        "run_registered_experiment",
        lambda **_kwargs: _fake_run(),
    )
    config = _config(tmp_path, games=1)
    report = run_tournament(config)
    record = report.records[0]
    diagnostics_path = config.output_directory / record.diagnostics_path
    stored = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    stored[field] = replacement
    diagnostics_path.write_text(json.dumps(stored), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        run_tournament(config, resume=True)


def test_resume_can_extend_the_game_target_without_repeating_work(
    monkeypatch,
    tmp_path,
) -> None:
    calls: list[int] = []

    def fake_registered_experiment(**kwargs):
        calls.append(kwargs["master_seed"])
        return _fake_run()

    monkeypatch.setattr(
        tournament_module,
        "run_registered_experiment",
        fake_registered_experiment,
    )

    calibration = _config(tmp_path, games=1)
    first = run_tournament(calibration)
    expanded = _config(tmp_path, games=3)
    second = run_tournament(expanded, resume=True)
    third = run_tournament(expanded, resume=True)

    assert calibration.configuration_hash == expanded.configuration_hash
    assert len(first.records) == 1
    assert len(second.records) == 3
    assert third == second
    assert calls == [tournament_game_seed(17, index) for index in range(3)]
    stored_config = json.loads(
        (expanded.output_directory / "config.json").read_text(encoding="utf-8")
    )
    assert stored_config["games_per_matchup"] == 3

    with pytest.raises(ValueError, match="stay unchanged or increase"):
        run_tournament(_config(tmp_path, games=2), resume=True)


def test_tournament_identity_contains_resolved_agent_specs(tmp_path) -> None:
    config = _config(tmp_path, games=1)
    agents = config.identity_dict()["agents"]
    fugitive = agents["fugitive"]
    marshal = agents["marshal"]

    assert fugitive["names"] == list(config.fugitive_agents)
    assert marshal["names"] == list(config.marshal_agents)
    assert [spec["name"] for spec in fugitive["resolved_specs"]] == list(
        config.fugitive_agents
    )
    assert [spec["name"] for spec in marshal["resolved_specs"]] == list(
        config.marshal_agents
    )
    assert all(
        "algorithm_id" in spec["parameters"]
        for spec in (
            *fugitive["resolved_specs"],
            *marshal["resolved_specs"],
        )
    )


def test_resume_rejects_a_different_experiment_identity(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        tournament_module,
        "run_registered_experiment",
        lambda **_kwargs: _fake_run(),
    )
    run_tournament(_config(tmp_path, games=1, root_seed=1))

    with pytest.raises(ValueError, match="incompatible"):
        run_tournament(
            _config(tmp_path, games=2, root_seed=2),
            resume=True,
        )


def test_errors_are_reported_separately_from_completed_win_rates(
    monkeypatch,
    tmp_path,
) -> None:
    outcomes = iter(
        (
            _fake_run(winner=Winner.MARSHAL),
            _fake_run(
                status=ExperimentStatus.ERROR,
                winner=None,
            ),
            _fake_run(winner=Winner.FUGITIVE),
        )
    )
    monkeypatch.setattr(
        tournament_module,
        "run_registered_experiment",
        lambda **_kwargs: next(outcomes),
    )

    report = run_tournament(_config(tmp_path, games=3))
    summary = report.matchups[0]

    assert summary.recorded_games == 3
    assert summary.completed == 2
    assert summary.errors == 1
    assert summary.fugitive_wins == 1
    assert summary.marshal_wins == 1
    assert summary.marshal_win_rate == 0.5
    assert summary.marshal_win_rate_95_low is not None
    assert summary.marshal_win_rate_95_high is not None


def test_wilson_interval_handles_empty_and_known_samples() -> None:
    assert wilson_interval(0, 0) == (None, None)
    low, high = wilson_interval(5, 10)
    assert low == pytest.approx(0.236593, abs=1e-6)
    assert high == pytest.approx(0.763407, abs=1e-6)

    with pytest.raises(ValueError, match="0 <= successes <= trials"):
        wilson_interval(2, 1)
