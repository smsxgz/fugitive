from __future__ import annotations

import json

import pytest

import fugitive.experiment as experiment_module
from fugitive.driver import DrawDecision, GuessDecision
from fugitive.experiment import ExperimentStatus, replay_manifest, run_experiment
from fugitive.model import FugitiveAction, Phase, Role, Winner
from fugitive.reproducibility import SeedBundle
from fugitive.rollout import RolloutBudget
from fugitive.rollout_diagnostics import RolloutDiagnosticsSnapshot, RolloutEvent


class _OpeningOneTwoFugitive:
    def choose_draw_pile(self, observation):
        return observation.legal_draw_piles[0]

    def choose_fugitive_action(self, observation):
        if observation.phase is Phase.FUGITIVE_OPENING:
            return FugitiveAction(len(observation.route))
        return FugitiveAction(None)


class _DiagnosticMarshal:
    def __init__(self) -> None:
        self._last_action = DrawDecision(0)

    def choose_draw_pile(self, observation):
        pile = observation.legal_draw_piles[0]
        self._last_action = DrawDecision(pile)
        return pile

    def choose_guess(self, _observation):
        numbers = (1, 2)
        self._last_action = GuessDecision(numbers)
        return numbers

    def rollout_diagnostics(self) -> RolloutDiagnosticsSnapshot:
        return RolloutDiagnosticsSnapshot.shortcut(
            algorithm_id="test-rollout-v1",
            rollout_model_id="test-terminal-model-v1",
            role=Role.MARSHAL,
            plan=RolloutBudget(1).plan(1),
            action=self._last_action,
        )


class _FailingDiagnosticMarshal(_DiagnosticMarshal):
    def rollout_diagnostics(self) -> RolloutDiagnosticsSnapshot:
        raise RuntimeError("rollout diagnostics unavailable")


def _run(marshal):
    return run_experiment(
        _OpeningOneTwoFugitive(),
        marshal,
        seeds=SeedBundle(master=123, deck=7, fugitive=11, marshal=13),
        validate_invariants=True,
    )


def test_rollout_side_channel_is_deterministic_and_outside_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _run(_DiagnosticMarshal())
    second = _run(_DiagnosticMarshal())

    assert first.rollout_events == second.rollout_events
    assert [event.decision for event in first.rollout_events] == [3, 4, 5]
    assert all(event.role is Role.MARSHAL for event in first.rollout_events)
    json.dumps(
        [event.to_dict() for event in first.rollout_events],
        allow_nan=False,
    )

    monkeypatch.setattr(
        experiment_module,
        "read_rollout_diagnostics",
        lambda _agent: None,
    )
    without_side_channel = _run(_DiagnosticMarshal())
    assert without_side_channel.rollout_events == ()
    assert first.manifest.to_json(indent=None) == (
        without_side_channel.manifest.to_json(indent=None)
    )
    assert replay_manifest(first.manifest) == replay_manifest(
        without_side_channel.manifest
    )


def test_rollout_diagnostics_failure_after_commit_is_nonfatal() -> None:
    run = _run(_FailingDiagnosticMarshal())

    assert run.status is ExperimentStatus.COMPLETED
    assert run.game_result is not None
    assert run.game_result.winner is Winner.MARSHAL
    assert run.manifest.decision_count == 5
    assert run.rollout_events == ()
    assert [
        failure.decision for failure in run.rollout_diagnostic_failures
    ] == [3, 4, 5]
    assert run.rollout_diagnostic_failures[0].to_dict()["error"] == {
        "type": "RuntimeError",
        "message": "rollout diagnostics unavailable",
    }
    assert "rollout_diagnostic_failures" not in run.manifest.to_dict()


def test_rollout_event_rejects_a_diagnostics_role_mismatch() -> None:
    diagnostics = RolloutDiagnosticsSnapshot.shortcut(
        algorithm_id="test-rollout-v1",
        rollout_model_id="test-terminal-model-v1",
        role=Role.FUGITIVE,
        plan=RolloutBudget(1).plan(1),
        action=FugitiveAction(1),
    )

    with pytest.raises(
        ValueError,
        match="rollout event role must match its diagnostics role",
    ):
        RolloutEvent(
            decision=1,
            round_number=1,
            phase=Phase.MARSHAL_DRAW,
            role=Role.MARSHAL,
            diagnostics=diagnostics,
        )
