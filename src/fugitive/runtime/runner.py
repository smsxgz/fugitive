"""Headless execution of one complete or watchdog-limited match."""

from __future__ import annotations

from typing import Mapping

from ..agents.marshal.inference.diagnostics import (
    InferenceDiagnosticFailure,
    InferenceEvent,
    read_inference_diagnostics,
)
from ..agents.planning.rollout_diagnostics import (
    RolloutDiagnosticFailure,
    RolloutEvent,
    read_rollout_diagnostics,
)
from ..agents.registry import FUGITIVE_AGENT_REGISTRY, MARSHAL_AGENT_REGISTRY
from ..game.driver import AgentStepError, DecisionRecord, step_agent
from ..game.engine import GameEngine
from ..game.model import FugitiveAgent, GameResult, MarshalAgent, Phase, Role
from ..shared.reproducibility import (
    AgentDescriptor,
    JSONValue,
    SeedBundle,
    derive_seed_bundle,
    normalize_parameters as _normalize_parameters,
)
from .manifest import (
    MANIFEST_SCHEMA_VERSION,
    RULES_SHA256,
    RULES_VERSION,
    MatchError,
    MatchRun,
    MatchStatus,
    ReplayManifest,
    _to_json_value,
    _validate_max_decisions,
)
from .replay import state_sha256


def run_registered_match(
    *,
    master_seed: int,
    fugitive_name: str,
    marshal_name: str,
    fugitive_parameters: Mapping[str, object] | None = None,
    marshal_parameters: Mapping[str, object] | None = None,
    fugitive_profile: str = "default",
    marshal_profile: str = "default",
    max_decisions: int | None = None,
    validate_invariants: bool = True,
) -> MatchRun:
    """Construct registered agents from independent seeds and run one game."""

    try:
        fugitive_registration = FUGITIVE_AGENT_REGISTRY[fugitive_name]
    except KeyError as exc:
        raise ValueError(f"unknown Fugitive agent: {fugitive_name}") from exc
    try:
        marshal_registration = MARSHAL_AGENT_REGISTRY[marshal_name]
    except KeyError as exc:
        raise ValueError(f"unknown Marshal agent: {marshal_name}") from exc

    f_parameters = _normalize_parameters(fugitive_parameters or {})
    m_parameters = _normalize_parameters(marshal_parameters or {})
    seeds = derive_seed_bundle(master_seed)
    assert seeds.fugitive is not None and seeds.marshal is not None
    built_fugitive = fugitive_registration.build(
        seeds.fugitive,
        profile=fugitive_profile,
        overrides=f_parameters,
    )
    built_marshal = marshal_registration.build(
        seeds.marshal,
        profile=marshal_profile,
        overrides=m_parameters,
    )
    return run_match(
        built_fugitive.agent,
        built_marshal.agent,
        seeds=seeds,
        fugitive_descriptor=built_fugitive.spec.descriptor(),
        marshal_descriptor=built_marshal.spec.descriptor(),
        max_decisions=max_decisions,
        validate_invariants=validate_invariants,
    )


def run_match(
    fugitive_agent: FugitiveAgent,
    marshal_agent: MarshalAgent,
    *,
    seeds: SeedBundle,
    fugitive_descriptor: AgentDescriptor | None = None,
    marshal_descriptor: AgentDescriptor | None = None,
    max_decisions: int | None = None,
    validate_invariants: bool = False,
) -> MatchRun:
    """Run a game with optional watchdog and produce a replay manifest.

    ``max_decisions=None`` is the default and runs until the engine declares a
    rules-defined winner, exactly like :func:`fugitive.game.engine.play_game`.
    Reaching an explicit limit records ``truncated`` with no winner.
    """

    _validate_max_decisions(max_decisions)
    if not isinstance(seeds, SeedBundle):
        raise TypeError("seeds must be a SeedBundle")
    fugitive_descriptor = fugitive_descriptor or AgentDescriptor.create(
        _agent_name(fugitive_agent)
    )
    marshal_descriptor = marshal_descriptor or AgentDescriptor.create(
        _agent_name(marshal_agent)
    )

    engine = GameEngine(seed=seeds.deck)
    trace: list[DecisionRecord] = []
    inference_events: list[InferenceEvent] = []
    inference_diagnostic_failures: list[InferenceDiagnosticFailure] = []
    rollout_events: list[RolloutEvent] = []
    rollout_diagnostic_failures: list[RolloutDiagnosticFailure] = []
    error: MatchError | None = None
    status: MatchStatus | None = None

    if validate_invariants:
        try:
            engine.validate_invariants()
        except Exception as exc:  # pragma: no cover - engine construction guard
            error = _run_error(exc, "initial_validate", 0, engine.phase, None)
            status = MatchStatus.ERROR

    while status is None and not engine.is_terminal:
        if max_decisions is not None and len(trace) >= max_decisions:
            status = MatchStatus.TRUNCATED
            break
        decision = len(trace) + 1
        phase = engine.phase
        stage = "step_agent"
        try:
            record = step_agent(
                engine,
                fugitive_agent,
                marshal_agent,
                decision=decision,
            )
            trace.append(record)
            stage = "collect_inference_diagnostics"
            acting_agent = (
                fugitive_agent
                if record.role is Role.FUGITIVE
                else marshal_agent
            )
            try:
                diagnostics = read_inference_diagnostics(acting_agent)
                if diagnostics is not None:
                    inference_events.append(
                        InferenceEvent(
                            decision=record.decision,
                            round_number=record.round_number,
                            phase=record.phase,
                            role=record.role,
                            diagnostics=diagnostics,
                        )
                    )
            except Exception as diagnostics_error:
                inference_diagnostic_failures.append(
                    InferenceDiagnosticFailure.from_exception(
                        decision=record.decision,
                        round_number=record.round_number,
                        phase=record.phase,
                        role=record.role,
                        error=diagnostics_error,
                    )
                )
            stage = "collect_rollout_diagnostics"
            try:
                rollout_diagnostics = read_rollout_diagnostics(acting_agent)
                if rollout_diagnostics is not None:
                    rollout_events.append(
                        RolloutEvent(
                            decision=record.decision,
                            round_number=record.round_number,
                            phase=record.phase,
                            role=record.role,
                            diagnostics=rollout_diagnostics,
                        )
                    )
            except Exception as diagnostics_error:
                rollout_diagnostic_failures.append(
                    RolloutDiagnosticFailure.from_exception(
                        decision=record.decision,
                        round_number=record.round_number,
                        phase=record.phase,
                        role=record.role,
                        error=diagnostics_error,
                    )
                )
            if validate_invariants:
                stage = "validate_invariants"
                engine.validate_invariants()
        except AgentStepError as exc:
            error = _run_error(
                exc.original,
                exc.stage,
                decision,
                exc.phase,
                _safe_attempted_action(exc.attempted_action),
            )
            status = MatchStatus.ERROR
        except Exception as exc:
            error = _run_error(exc, stage, decision, phase, None)
            status = MatchStatus.ERROR

    if status is None:
        status = MatchStatus.COMPLETED

    game_result = engine.result() if status is MatchStatus.COMPLETED else None
    manifest = _build_manifest(
        engine=engine,
        seeds=seeds,
        fugitive_descriptor=fugitive_descriptor,
        marshal_descriptor=marshal_descriptor,
        status=status,
        trace=tuple(trace),
        max_decisions=max_decisions,
        game_result=game_result,
        error=error,
    )
    return MatchRun(
        manifest=manifest,
        game_result=game_result,
        inference_events=tuple(inference_events),
        inference_diagnostic_failures=tuple(inference_diagnostic_failures),
        rollout_events=tuple(rollout_events),
        rollout_diagnostic_failures=tuple(rollout_diagnostic_failures),
    )


def _build_manifest(
    *,
    engine: GameEngine,
    seeds: SeedBundle,
    fugitive_descriptor: AgentDescriptor,
    marshal_descriptor: AgentDescriptor,
    status: MatchStatus,
    trace: tuple[DecisionRecord, ...],
    max_decisions: int | None,
    game_result: GameResult | None,
    error: MatchError | None,
) -> ReplayManifest:
    return ReplayManifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        rules_version=RULES_VERSION,
        rules_sha256=RULES_SHA256,
        seeds=seeds,
        fugitive_agent=fugitive_descriptor,
        marshal_agent=marshal_descriptor,
        status=status,
        decision_count=len(trace),
        max_decisions=max_decisions,
        trace=trace,
        final_state_sha256=state_sha256(engine),
        winner=game_result.winner if game_result is not None else None,
        reason=game_result.reason if game_result is not None else None,
        rounds=game_result.rounds if game_result is not None else None,
        error=error,
    )


def _run_error(
    error: Exception,
    stage: str,
    decision: int,
    phase: Phase,
    attempted: dict[str, JSONValue] | None,
) -> MatchError:
    error_type = f"{type(error).__module__}.{type(error).__qualname__}"
    return MatchError(error_type, str(error), stage, decision, phase, attempted)


def _agent_name(agent: object) -> str:
    name = getattr(agent, "name", None)
    if isinstance(name, str) and name:
        return name
    return f"{type(agent).__module__}.{type(agent).__qualname__}"


def _safe_json_scalar(value: object) -> JSONValue:
    if value is None or isinstance(value, (bool, int, float, str)):
        return _to_json_value(value)
    return {"python_type": f"{type(value).__module__}.{type(value).__qualname__}"}


def _safe_attempted_action(
    attempted: Mapping[str, object] | None,
) -> dict[str, JSONValue] | None:
    if attempted is None:
        return None
    result: dict[str, JSONValue] = {}
    for key, value in attempted.items():
        try:
            result[key] = _to_json_value(value)
        except ValueError:
            result[key] = _safe_json_scalar(value)
    return result


__all__ = [
    "run_match",
    "run_registered_match",
]
