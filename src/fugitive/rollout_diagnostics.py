"""Information-safe diagnostics for one rollout action comparison."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .driver import DrawDecision, GuessDecision, PlayerDecision
from .model import FugitiveAction, Phase, Role
from .reproducibility import JSONValue
from .rollout import (
    PairedRolloutResult,
    RolloutActionResult,
    RolloutEvaluation,
    RolloutPlan,
)


@dataclass(frozen=True, slots=True)
class RolloutActionDiagnostics:
    action: PlayerDecision
    wins: int
    samples: int
    win_rate: float | None

    @classmethod
    def from_result(cls, result: RolloutActionResult) -> "RolloutActionDiagnostics":
        return cls(result.action, result.wins, result.samples, result.win_rate)

    @classmethod
    def shortcut(cls, action: PlayerDecision) -> "RolloutActionDiagnostics":
        return cls(action, 0, 0, None)

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "action": _action_dict(self.action),
            "wins": self.wins,
            "samples": self.samples,
            "win_rate": self.win_rate,
        }


@dataclass(frozen=True, slots=True)
class PairedOutcomeDiagnostics:
    first_action: PlayerDecision
    second_action: PlayerDecision
    both_win: int
    first_only_win: int
    second_only_win: int
    both_loss: int
    mean_difference: float
    standard_error: float | None

    @classmethod
    def from_result(cls, result: PairedRolloutResult) -> "PairedOutcomeDiagnostics":
        return cls(
            first_action=result.first_action,
            second_action=result.second_action,
            both_win=result.both_win,
            first_only_win=result.first_only_win,
            second_only_win=result.second_only_win,
            both_loss=result.both_loss,
            mean_difference=result.mean_difference,
            standard_error=result.standard_error,
        )

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "first_action": _action_dict(self.first_action),
            "second_action": _action_dict(self.second_action),
            "both_win": self.both_win,
            "first_only_win": self.first_only_win,
            "second_only_win": self.second_only_win,
            "both_loss": self.both_loss,
            "mean_difference": self.mean_difference,
            "standard_error": self.standard_error,
        }


@dataclass(frozen=True, slots=True)
class RolloutDiagnosticsSnapshot:
    """A decision-local rollout report that contains no sampled world."""

    algorithm_id: str
    rollout_model_id: str
    role: Role
    plan: RolloutPlan
    simulated_decisions: int
    actions: tuple[RolloutActionDiagnostics, ...]
    pairs: tuple[PairedOutcomeDiagnostics, ...]

    @classmethod
    def from_evaluation(
        cls,
        *,
        algorithm_id: str,
        rollout_model_id: str,
        plan: RolloutPlan,
        evaluation: RolloutEvaluation,
    ) -> "RolloutDiagnosticsSnapshot":
        if plan.evaluated_candidates != len(evaluation.action_results):
            raise ValueError("rollout plan and evaluation disagree on candidates")
        if plan.paired_scenarios != evaluation.paired_scenarios:
            raise ValueError("rollout plan and evaluation disagree on scenarios")
        if plan.terminal_simulations != evaluation.terminal_simulations:
            raise ValueError("rollout plan and evaluation disagree on work")
        return cls(
            algorithm_id=algorithm_id,
            rollout_model_id=rollout_model_id,
            role=evaluation.role,
            plan=plan,
            simulated_decisions=evaluation.simulated_decisions,
            actions=tuple(
                RolloutActionDiagnostics.from_result(result)
                for result in evaluation.action_results
            ),
            pairs=tuple(
                PairedOutcomeDiagnostics.from_result(result)
                for result in evaluation.paired_results
            ),
        )

    @classmethod
    def shortcut(
        cls,
        *,
        algorithm_id: str,
        rollout_model_id: str,
        role: Role,
        plan: RolloutPlan,
        action: PlayerDecision,
    ) -> "RolloutDiagnosticsSnapshot":
        if not plan.single_candidate_shortcut:
            raise ValueError("shortcut diagnostics require a one-candidate plan")
        return cls(
            algorithm_id=algorithm_id,
            rollout_model_id=rollout_model_id,
            role=role,
            plan=plan,
            simulated_decisions=0,
            actions=(RolloutActionDiagnostics.shortcut(action),),
            pairs=(),
        )

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "algorithm_id": self.algorithm_id,
            "rollout_model_id": self.rollout_model_id,
            "role": self.role.value,
            "plan": {
                "offered_candidates": self.plan.offered_candidates,
                "evaluated_candidates": self.plan.evaluated_candidates,
                "paired_scenarios": self.plan.paired_scenarios,
                "terminal_simulations": self.plan.terminal_simulations,
                "maximum_terminal_simulations": (
                    self.plan.maximum_terminal_simulations
                ),
                "unused_terminal_simulations": (
                    self.plan.unused_terminal_simulations
                ),
                "single_candidate_shortcut": (
                    self.plan.single_candidate_shortcut
                ),
            },
            "simulated_decisions": self.simulated_decisions,
            "actions": [action.to_dict() for action in self.actions],
            "pairs": [pair.to_dict() for pair in self.pairs],
        }


@dataclass(frozen=True, slots=True)
class RolloutEvent:
    """A decision-indexed rollout side channel, separate from replay."""

    decision: int
    round_number: int
    phase: Phase
    role: Role
    diagnostics: RolloutDiagnosticsSnapshot

    def __post_init__(self) -> None:
        if self.role is not self.diagnostics.role:
            raise ValueError(
                "rollout event role must match its diagnostics role"
            )

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "decision": self.decision,
            "round_number": self.round_number,
            "phase": self.phase.value,
            "role": self.role.value,
            "diagnostics": self.diagnostics.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class RolloutDiagnosticFailure:
    """A non-fatal failure reading diagnostics after a committed action."""

    decision: int
    round_number: int
    phase: Phase
    role: Role
    error_type: str
    message: str

    @classmethod
    def from_exception(
        cls,
        *,
        decision: int,
        round_number: int,
        phase: Phase,
        role: Role,
        error: Exception,
    ) -> "RolloutDiagnosticFailure":
        try:
            message = str(error)
        except Exception:  # pragma: no cover - defensive exception formatting
            message = "<error message unavailable>"
        return cls(
            decision=decision,
            round_number=round_number,
            phase=phase,
            role=role,
            error_type=type(error).__name__,
            message=message,
        )

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "decision": self.decision,
            "round_number": self.round_number,
            "phase": self.phase.value,
            "role": self.role.value,
            "error": {
                "type": self.error_type,
                "message": self.message,
            },
        }


@runtime_checkable
class RolloutDiagnosticsProvider(Protocol):
    def rollout_diagnostics(self) -> RolloutDiagnosticsSnapshot | None: ...


def read_rollout_diagnostics(agent: object) -> RolloutDiagnosticsSnapshot | None:
    if not isinstance(agent, RolloutDiagnosticsProvider):
        return None
    return agent.rollout_diagnostics()


def _action_dict(action: PlayerDecision) -> dict[str, JSONValue]:
    if isinstance(action, DrawDecision):
        return {"kind": "draw", "pile": action.pile}
    if isinstance(action, GuessDecision):
        return {"kind": "guess", "numbers": list(action.numbers)}
    if isinstance(action, FugitiveAction):
        return {
            "kind": "fugitive_action",
            "hideout": action.hideout,
            "sprint_cards": list(action.sprint_cards),
        }
    raise TypeError(f"unsupported rollout action: {type(action).__name__}")


__all__ = [
    "PairedOutcomeDiagnostics",
    "RolloutActionDiagnostics",
    "RolloutDiagnosticFailure",
    "RolloutDiagnosticsProvider",
    "RolloutDiagnosticsSnapshot",
    "RolloutEvent",
    "read_rollout_diagnostics",
]
