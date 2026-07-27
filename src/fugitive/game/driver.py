"""Canonical one-decision driver shared by simulations and experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

from .engine import GameEngine
from .model import (
    FugitiveAction,
    FugitiveAgent,
    IllegalActionError,
    MarshalAgent,
    Phase,
    Role,
)


@dataclass(frozen=True, slots=True)
class DrawDecision:
    pile: int


@dataclass(frozen=True, slots=True)
class GuessDecision:
    numbers: tuple[int, ...]


PlayerDecision: TypeAlias = FugitiveAction | DrawDecision | GuessDecision


@dataclass(frozen=True, slots=True)
class DrawTrace:
    decision: int
    round_number: int
    phase: Phase
    role: Role
    pile: int
    card: int


@dataclass(frozen=True, slots=True)
class FugitiveTrace:
    decision: int
    round_number: int
    phase: Phase
    role: Role
    hideout: int | None
    sprint_cards: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class GuessTrace:
    decision: int
    round_number: int
    phase: Phase
    role: Role
    numbers: tuple[int, ...]
    success: bool


DecisionRecord: TypeAlias = DrawTrace | FugitiveTrace | GuessTrace


class AgentStepError(RuntimeError):
    """An agent decision failure with stable stage and attempted-action data."""

    def __init__(
        self,
        original: Exception,
        *,
        stage: str,
        phase: Phase,
        attempted_action: dict[str, object] | None = None,
    ) -> None:
        super().__init__(str(original))
        self.original = original
        self.stage = stage
        self.phase = phase
        self.attempted_action = attempted_action


def apply_decision(
    engine: GameEngine,
    action: PlayerDecision,
    *,
    decision: int,
) -> DecisionRecord:
    """Apply one normalized human or agent decision and return its trace."""

    if isinstance(decision, bool) or not isinstance(decision, int) or decision < 1:
        raise ValueError("decision must be a positive integer")
    phase = engine.phase
    if phase is Phase.TERMINAL:
        raise RuntimeError("cannot act in a terminal game")
    role = role_for_phase(phase)
    if role is None:  # pragma: no cover - future engine phase guard
        raise RuntimeError(f"unhandled engine phase: {phase}")
    round_number = engine.observation(role).round_number

    if phase in (Phase.FUGITIVE_OPENING, Phase.FUGITIVE_ACTION):
        if not isinstance(action, FugitiveAction):
            raise IllegalActionError("the Fugitive phase requires a Fugitive action")
        normalized = FugitiveAction(action.hideout, tuple(action.sprint_cards))
        engine.apply_fugitive_action(normalized)
        return FugitiveTrace(
            decision,
            round_number,
            phase,
            Role.FUGITIVE,
            normalized.hideout,
            tuple(sorted(normalized.sprint_cards)),
        )

    if phase in (Phase.FUGITIVE_DRAW, Phase.MARSHAL_DRAW):
        if not isinstance(action, DrawDecision):
            raise IllegalActionError("a draw phase requires a pile decision")
        card = engine.draw(action.pile)
        return DrawTrace(
            decision,
            round_number,
            phase,
            role,
            action.pile,
            card,
        )

    if not isinstance(action, GuessDecision):
        raise IllegalActionError("a Marshal guess phase requires a guess decision")
    numbers = tuple(action.numbers)
    success = engine.apply_guess(numbers)
    return GuessTrace(
        decision,
        round_number,
        phase,
        Role.MARSHAL,
        tuple(sorted(numbers)),
        success,
    )


def step_agent(
    engine: GameEngine,
    fugitive_agent: FugitiveAgent,
    marshal_agent: MarshalAgent,
    *,
    decision: int,
) -> DecisionRecord:
    """Choose and apply exactly one action for the engine's current phase."""

    if isinstance(decision, bool) or not isinstance(decision, int) or decision < 1:
        raise ValueError("decision must be a positive integer")
    phase = engine.phase
    if phase is Phase.TERMINAL:
        raise RuntimeError("cannot step a terminal game")

    role = role_for_phase(phase)
    if role is None:  # pragma: no cover - future engine phase guard
        raise RuntimeError(f"unhandled engine phase: {phase}")
    try:
        observation = engine.observation(role)
    except Exception as exc:
        raise AgentStepError(exc, stage="observe", phase=phase) from exc
    if phase in (Phase.FUGITIVE_OPENING, Phase.FUGITIVE_ACTION):
        try:
            raw_action = fugitive_agent.choose_fugitive_action(observation)
            action = FugitiveAction(
                raw_action.hideout,
                tuple(raw_action.sprint_cards),
            )
        except Exception as exc:
            raise AgentStepError(
                exc,
                stage="choose_fugitive_action",
                phase=phase,
            ) from exc
        attempted = {
            "kind": "fugitive_action",
            "hideout": action.hideout,
            "sprint_cards": list(action.sprint_cards),
        }
        try:
            record = apply_decision(engine, action, decision=decision)
        except Exception as exc:
            raise AgentStepError(
                exc,
                stage="apply_fugitive_action",
                phase=phase,
                attempted_action=attempted,
            ) from exc
        assert isinstance(record, FugitiveTrace)
        return record

    if phase in (Phase.FUGITIVE_DRAW, Phase.MARSHAL_DRAW):
        agent = fugitive_agent if role is Role.FUGITIVE else marshal_agent
        choose_stage = (
            "choose_fugitive_draw"
            if role is Role.FUGITIVE
            else "choose_marshal_draw"
        )
        apply_stage = (
            "apply_fugitive_draw"
            if role is Role.FUGITIVE
            else "apply_marshal_draw"
        )
        try:
            pile = agent.choose_draw_pile(observation)
        except Exception as exc:
            raise AgentStepError(exc, stage=choose_stage, phase=phase) from exc
        attempted = {"kind": "draw", "pile": pile}
        try:
            record = apply_decision(
                engine,
                DrawDecision(pile),
                decision=decision,
            )
        except Exception as exc:
            raise AgentStepError(
                exc,
                stage=apply_stage,
                phase=phase,
                attempted_action=attempted,
            ) from exc
        assert isinstance(record, DrawTrace)
        return record

    try:
        numbers = tuple(marshal_agent.choose_guess(observation))
    except Exception as exc:
        raise AgentStepError(
            exc,
            stage="choose_marshal_guess",
            phase=phase,
        ) from exc
    attempted = {"kind": "guess", "numbers": list(numbers)}
    try:
        record = apply_decision(
            engine,
            GuessDecision(numbers),
            decision=decision,
        )
    except Exception as exc:
        raise AgentStepError(
            exc,
            stage="apply_marshal_guess",
            phase=phase,
            attempted_action=attempted,
        ) from exc
    assert isinstance(record, GuessTrace)
    return record


def role_for_phase(phase: Phase) -> Role | None:
    if phase in (
        Phase.FUGITIVE_OPENING,
        Phase.FUGITIVE_DRAW,
        Phase.FUGITIVE_ACTION,
    ):
        return Role.FUGITIVE
    if phase in (Phase.MARSHAL_DRAW, Phase.MARSHAL_GUESS, Phase.MANHUNT):
        return Role.MARSHAL
    return None


__all__ = [
    "AgentStepError",
    "DecisionRecord",
    "DrawDecision",
    "DrawTrace",
    "FugitiveTrace",
    "GuessDecision",
    "GuessTrace",
    "PlayerDecision",
    "apply_decision",
    "role_for_phase",
    "step_agent",
]
