"""Reproducible experiment runner and action-trace replay support.

The rules engine deliberately has no artificial turn or decision limit.  This
module adds an optional experiment-only watchdog without inventing a winner:
an interrupted run is recorded as ``truncated``, while policy or engine
failures are recorded as ``error``.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
import hashlib
import json
import math
from typing import Any, Mapping, cast

from .agents.registry import FUGITIVE_AGENT_REGISTRY, MARSHAL_AGENT_REGISTRY
from .driver import (
    AgentStepError,
    DrawTrace,
    FugitiveTrace,
    GuessTrace,
    DecisionRecord,
    step_agent,
)
from .engine import GameEngine
from .inference_diagnostics import (
    InferenceDiagnosticFailure,
    InferenceEvent,
    read_inference_diagnostics,
)
from .model import (
    FugitiveAction,
    FugitiveAgent,
    GameResult,
    MarshalAgent,
    Phase,
    Role,
    Winner,
)
from .reproducibility import (
    AgentDescriptor,
    JSONValue,
    SeedBundle,
    derive_seed,
    derive_seed_bundle,
    normalize_parameters as _normalize_parameters,
)


MANIFEST_SCHEMA_VERSION = 1
RULES_VERSION = "fugitive-canonical-2026-07-23-v1"
# SHA-256 of docs/rules/CANONICAL_RULES.md after normalizing CRLF to LF.  A
# rules-text change must update both this fingerprint and RULES_VERSION.
RULES_SHA256 = "027b6c902e372bc0f385c1aa899ff1b6265b559342bf45db5550cf81709085b3"


class ExperimentStatus(str, Enum):
    COMPLETED = "completed"
    TRUNCATED = "truncated"
    ERROR = "error"


class ReplayMismatchError(ValueError):
    """Raised when a seed and recorded trace do not reproduce the manifest."""


@dataclass(frozen=True, slots=True)
class RunError:
    """Stable error envelope for a failed experiment decision."""

    error_type: str
    message: str
    stage: str
    decision: int
    phase: Phase
    attempted_action: dict[str, JSONValue] | None = None

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "type": self.error_type,
            "message": self.message,
            "stage": self.stage,
            "decision": self.decision,
            "phase": self.phase.value,
            "attempted_action": self.attempted_action,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "RunError":
        attempted = data.get("attempted_action")
        if attempted is not None and not isinstance(attempted, Mapping):
            raise ValueError("attempted_action must be an object or null")
        return cls(
            error_type=str(data.get("type", "Error")),
            message=str(data.get("message", "")),
            stage=str(data.get("stage", "unknown")),
            decision=_require_int(data.get("decision"), "error decision"),
            phase=Phase(str(data.get("phase"))),
            attempted_action=(
                _normalize_parameters(cast(Mapping[str, object], attempted))
                if attempted is not None
                else None
            ),
        )


@dataclass(frozen=True, slots=True)
class ReplayManifest:
    """Everything needed to audit and replay an experiment trajectory."""

    schema_version: int
    rules_version: str
    rules_sha256: str
    seeds: SeedBundle
    fugitive_agent: AgentDescriptor
    marshal_agent: AgentDescriptor
    status: ExperimentStatus
    decision_count: int
    max_decisions: int | None
    trace: tuple[DecisionRecord, ...]
    final_state_sha256: str
    winner: Winner | None = None
    reason: str | None = None
    rounds: int | None = None
    error: RunError | None = None

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "schema_version": self.schema_version,
            "rules": {
                "version": self.rules_version,
                "sha256": self.rules_sha256,
            },
            "seeds": self.seeds.to_dict(),
            "agents": {
                "fugitive": self.fugitive_agent.to_dict(),
                "marshal": self.marshal_agent.to_dict(),
            },
            "outcome": {
                "status": self.status.value,
                "decision_count": self.decision_count,
                "max_decisions": self.max_decisions,
                "winner": self.winner.value if self.winner is not None else None,
                "reason": self.reason,
                "rounds": self.rounds,
                "error": self.error.to_dict() if self.error is not None else None,
            },
            "trace": [_trace_to_dict(record) for record in self.trace],
            "final_state_sha256": self.final_state_sha256,
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=True,
            allow_nan=False,
            indent=indent,
            sort_keys=True,
            separators=(",", ":") if indent is None else None,
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "ReplayManifest":
        rules = _require_mapping(data.get("rules"), "rules")
        seeds = _require_mapping(data.get("seeds"), "seeds")
        agents = _require_mapping(data.get("agents"), "agents")
        outcome = _require_mapping(data.get("outcome"), "outcome")
        trace_data = data.get("trace")
        if not isinstance(trace_data, list):
            raise ValueError("trace must be an array")
        error_data = outcome.get("error")
        if error_data is not None and not isinstance(error_data, Mapping):
            raise ValueError("outcome.error must be an object or null")
        winner_data = outcome.get("winner")
        max_decisions_data = outcome.get("max_decisions")
        rounds_data = outcome.get("rounds")
        return cls(
            schema_version=_require_int(data.get("schema_version"), "schema_version"),
            rules_version=str(rules.get("version")),
            rules_sha256=str(rules.get("sha256")),
            seeds=SeedBundle.from_dict(seeds),
            fugitive_agent=AgentDescriptor.from_dict(
                _require_mapping(agents.get("fugitive"), "fugitive agent")
            ),
            marshal_agent=AgentDescriptor.from_dict(
                _require_mapping(agents.get("marshal"), "marshal agent")
            ),
            status=ExperimentStatus(str(outcome.get("status"))),
            decision_count=_require_int(
                outcome.get("decision_count"), "decision_count"
            ),
            max_decisions=(
                None
                if max_decisions_data is None
                else _require_int(max_decisions_data, "max_decisions")
            ),
            trace=tuple(
                _trace_from_dict(_require_mapping(item, "trace item"))
                for item in trace_data
            ),
            final_state_sha256=str(data.get("final_state_sha256")),
            winner=None if winner_data is None else Winner(str(winner_data)),
            reason=(
                None if outcome.get("reason") is None else str(outcome.get("reason"))
            ),
            rounds=(
                None if rounds_data is None else _require_int(rounds_data, "rounds")
            ),
            error=(
                None
                if error_data is None
                else RunError.from_dict(cast(Mapping[str, object], error_data))
            ),
        )

    @classmethod
    def from_json(cls, payload: str) -> "ReplayManifest":
        data = json.loads(payload)
        if not isinstance(data, Mapping):
            raise ValueError("manifest JSON must contain an object")
        return cls.from_dict(cast(Mapping[str, object], data))


@dataclass(frozen=True, slots=True)
class ExperimentRun:
    """One completed, truncated, or failed experiment invocation."""

    manifest: ReplayManifest
    game_result: GameResult | None
    inference_events: tuple[InferenceEvent, ...] = ()
    inference_diagnostic_failures: tuple[InferenceDiagnosticFailure, ...] = ()

    @property
    def status(self) -> ExperimentStatus:
        return self.manifest.status


@dataclass(frozen=True, slots=True)
class ReplayVerification:
    """Successful reconstruction of a manifest's recorded trace."""

    status: ExperimentStatus
    decision_count: int
    final_state_sha256: str
    game_result: GameResult | None


def run_registered_experiment(
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
) -> ExperimentRun:
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
    return run_experiment(
        built_fugitive.agent,
        built_marshal.agent,
        seeds=seeds,
        fugitive_descriptor=built_fugitive.spec.descriptor(),
        marshal_descriptor=built_marshal.spec.descriptor(),
        max_decisions=max_decisions,
        validate_invariants=validate_invariants,
    )


def run_experiment(
    fugitive_agent: FugitiveAgent,
    marshal_agent: MarshalAgent,
    *,
    seeds: SeedBundle,
    fugitive_descriptor: AgentDescriptor | None = None,
    marshal_descriptor: AgentDescriptor | None = None,
    max_decisions: int | None = None,
    validate_invariants: bool = False,
) -> ExperimentRun:
    """Run a game with optional watchdog and produce a replay manifest.

    ``max_decisions=None`` is the default and runs until the engine declares a
    rules-defined winner, exactly like :func:`fugitive.engine.play_game`.
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
    error: RunError | None = None
    status: ExperimentStatus | None = None

    if validate_invariants:
        try:
            engine.validate_invariants()
        except Exception as exc:  # pragma: no cover - engine construction guard
            error = _run_error(exc, "initial_validate", 0, engine.phase, None)
            status = ExperimentStatus.ERROR

    while status is None and not engine.is_terminal:
        if max_decisions is not None and len(trace) >= max_decisions:
            status = ExperimentStatus.TRUNCATED
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
            status = ExperimentStatus.ERROR
        except Exception as exc:
            error = _run_error(exc, stage, decision, phase, None)
            status = ExperimentStatus.ERROR

    if status is None:
        status = ExperimentStatus.COMPLETED

    game_result = engine.result() if status is ExperimentStatus.COMPLETED else None
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
    return ExperimentRun(
        manifest,
        game_result,
        tuple(inference_events),
        tuple(inference_diagnostic_failures),
    )


def replay_manifest(
    manifest: ReplayManifest,
    *,
    validate_invariants: bool = True,
) -> ReplayVerification:
    """Replay and strictly verify every recorded action and outcome."""

    if manifest.schema_version != MANIFEST_SCHEMA_VERSION:
        raise ReplayMismatchError(
            f"unsupported manifest schema: {manifest.schema_version}"
        )
    if (
        manifest.rules_version != RULES_VERSION
        or manifest.rules_sha256 != RULES_SHA256
    ):
        raise ReplayMismatchError("manifest rules fingerprint does not match this build")
    if manifest.decision_count != len(manifest.trace):
        raise ReplayMismatchError("decision_count does not match trace length")
    _validate_manifest_outcome(manifest)

    engine = GameEngine(seed=manifest.seeds.deck)
    if validate_invariants:
        engine.validate_invariants()
    for expected_decision, record in enumerate(manifest.trace, start=1):
        if record.decision != expected_decision:
            raise ReplayMismatchError(
                f"trace decision {record.decision} is out of sequence"
            )
        if engine.is_terminal:
            raise ReplayMismatchError("trace contains an action after terminal state")
        if engine.phase is not record.phase:
            raise ReplayMismatchError(
                f"decision {record.decision}: expected phase {record.phase.value}, "
                f"got {engine.phase.value}"
            )
        actual_round = _round_number(engine, engine.phase)
        if actual_round != record.round_number:
            raise ReplayMismatchError(
                f"decision {record.decision}: expected round {record.round_number}, "
                f"got {actual_round}"
            )

        try:
            if isinstance(record, DrawTrace):
                actual_role = (
                    Role.FUGITIVE
                    if record.phase is Phase.FUGITIVE_DRAW
                    else Role.MARSHAL
                )
                if record.role is not actual_role:
                    raise ReplayMismatchError(
                        f"decision {record.decision}: draw role does not match phase"
                    )
                card = engine.draw(record.pile)
                if card != record.card:
                    raise ReplayMismatchError(
                        f"decision {record.decision}: expected draw {record.card}, "
                        f"got {card}"
                    )
            elif isinstance(record, FugitiveTrace):
                if record.role is not Role.FUGITIVE:
                    raise ReplayMismatchError(
                        f"decision {record.decision}: invalid Fugitive action role"
                    )
                engine.apply_fugitive_action(
                    FugitiveAction(record.hideout, record.sprint_cards)
                )
            elif isinstance(record, GuessTrace):
                if record.role is not Role.MARSHAL:
                    raise ReplayMismatchError(
                        f"decision {record.decision}: invalid guess role"
                    )
                success = engine.apply_guess(record.numbers)
                if success is not record.success:
                    raise ReplayMismatchError(
                        f"decision {record.decision}: guess result differs"
                    )
            else:  # pragma: no cover - closed union defensive guard
                raise ReplayMismatchError("unknown trace record")
        except ReplayMismatchError:
            raise
        except Exception as exc:
            raise ReplayMismatchError(
                f"decision {record.decision} could not be replayed: {exc}"
            ) from exc
        if validate_invariants:
            engine.validate_invariants()

    state_hash = _state_sha256(engine)
    if state_hash != manifest.final_state_sha256:
        raise ReplayMismatchError("final state fingerprint differs from manifest")

    game_result: GameResult | None = None
    if manifest.status is ExperimentStatus.COMPLETED:
        if not engine.is_terminal:
            raise ReplayMismatchError("completed manifest does not reach terminal state")
        game_result = engine.result()
        if (
            game_result.winner is not manifest.winner
            or game_result.reason != manifest.reason
            or game_result.rounds != manifest.rounds
        ):
            raise ReplayMismatchError("terminal result differs from manifest")
    elif manifest.status is ExperimentStatus.TRUNCATED:
        if engine.is_terminal:
            raise ReplayMismatchError("truncated manifest already has a rules winner")
        if manifest.winner is not None or manifest.reason is not None:
            raise ReplayMismatchError("truncated manifest must not contain a winner")
    elif manifest.status is ExperimentStatus.ERROR:
        if manifest.error is None:
            raise ReplayMismatchError("error manifest has no error metadata")

    return ReplayVerification(
        manifest.status,
        len(manifest.trace),
        state_hash,
        game_result,
    )


def _validate_manifest_outcome(manifest: ReplayManifest) -> None:
    """Reject internally inconsistent result metadata before replaying."""

    try:
        _validate_max_decisions(manifest.max_decisions)
    except ValueError as exc:
        raise ReplayMismatchError(str(exc)) from exc
    if manifest.decision_count < 0:
        raise ReplayMismatchError("decision_count must be non-negative")

    has_result = (
        manifest.winner is not None
        and manifest.reason is not None
        and manifest.rounds is not None
    )
    has_partial_result = any(
        value is not None
        for value in (manifest.winner, manifest.reason, manifest.rounds)
    )
    if manifest.status is ExperimentStatus.COMPLETED:
        if not has_result or manifest.error is not None:
            raise ReplayMismatchError(
                "completed manifest requires one result and no error metadata"
            )
        if manifest.max_decisions is not None and (
            manifest.decision_count > manifest.max_decisions
        ):
            raise ReplayMismatchError("completed trace exceeds max_decisions")
        return

    if has_partial_result:
        raise ReplayMismatchError(
            f"{manifest.status.value} manifest must not contain a winner"
        )
    if manifest.status is ExperimentStatus.TRUNCATED:
        if manifest.error is not None:
            raise ReplayMismatchError("truncated manifest must not contain an error")
        if (
            manifest.max_decisions is None
            or manifest.decision_count != manifest.max_decisions
        ):
            raise ReplayMismatchError(
                "truncated manifest must stop exactly at max_decisions"
            )
        return

    if manifest.error is None:
        raise ReplayMismatchError("error manifest has no error metadata")
    if manifest.error.decision not in (
        manifest.decision_count,
        manifest.decision_count + 1,
    ):
        raise ReplayMismatchError("error decision does not follow the trace prefix")


def _build_manifest(
    *,
    engine: GameEngine,
    seeds: SeedBundle,
    fugitive_descriptor: AgentDescriptor,
    marshal_descriptor: AgentDescriptor,
    status: ExperimentStatus,
    trace: tuple[DecisionRecord, ...],
    max_decisions: int | None,
    game_result: GameResult | None,
    error: RunError | None,
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
        final_state_sha256=_state_sha256(engine),
        winner=game_result.winner if game_result is not None else None,
        reason=game_result.reason if game_result is not None else None,
        rounds=game_result.rounds if game_result is not None else None,
        error=error,
    )


def _state_sha256(engine: GameEngine) -> str:
    payload: dict[str, Any] = {
        "fugitive": _jsonable(engine.observation(Role.FUGITIVE)),
        "marshal": _jsonable(engine.observation(Role.MARSHAL)),
        "terminal": engine.is_terminal,
    }
    if engine.is_terminal:
        payload["result"] = _jsonable(engine.result())
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def state_sha256(engine: GameEngine) -> str:
    """Return the canonical replay fingerprint for one engine state."""

    return _state_sha256(engine)


def _jsonable(value: object) -> JSONValue:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite values are not JSON serializable")
        return value
    if isinstance(value, Enum):
        return cast(str, value.value)
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _jsonable(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        result: dict[str, JSONValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("JSON object keys must be strings")
            result[key] = _jsonable(item)
        return result
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    raise ValueError(f"unsupported JSON value: {type(value).__name__}")


def _trace_to_dict(record: DecisionRecord) -> dict[str, JSONValue]:
    base: dict[str, JSONValue] = {
        "decision": record.decision,
        "round": record.round_number,
        "phase": record.phase.value,
        "role": record.role.value,
    }
    if isinstance(record, DrawTrace):
        base.update({"kind": "draw", "pile": record.pile, "card": record.card})
    elif isinstance(record, FugitiveTrace):
        base.update(
            {
                "kind": "fugitive_action",
                "hideout": record.hideout,
                "sprint_cards": list(record.sprint_cards),
            }
        )
    else:
        base.update(
            {
                "kind": "guess",
                "numbers": list(record.numbers),
                "success": record.success,
            }
        )
    return base


def _trace_from_dict(data: Mapping[str, object]) -> DecisionRecord:
    decision = _require_int(data.get("decision"), "trace decision")
    round_number = _require_int(data.get("round"), "trace round")
    phase = Phase(str(data.get("phase")))
    role = Role(str(data.get("role")))
    kind = data.get("kind")
    if kind == "draw":
        return DrawTrace(
            decision,
            round_number,
            phase,
            role,
            _require_int(data.get("pile"), "draw pile"),
            _require_int(data.get("card"), "draw card"),
        )
    if kind == "fugitive_action":
        hideout_data = data.get("hideout")
        return FugitiveTrace(
            decision,
            round_number,
            phase,
            role,
            None
            if hideout_data is None
            else _require_int(hideout_data, "hideout"),
            _integer_tuple(data.get("sprint_cards"), "sprint_cards"),
        )
    if kind == "guess":
        success = data.get("success")
        if not isinstance(success, bool):
            raise ValueError("guess success must be a boolean")
        return GuessTrace(
            decision,
            round_number,
            phase,
            role,
            _integer_tuple(data.get("numbers"), "guess numbers"),
            success,
        )
    raise ValueError(f"unknown trace kind: {kind}")


def _integer_tuple(value: object, label: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be an array")
    return tuple(_require_int(item, label) for item in value)


def _run_error(
    error: Exception,
    stage: str,
    decision: int,
    phase: Phase,
    attempted: dict[str, JSONValue] | None,
) -> RunError:
    error_type = f"{type(error).__module__}.{type(error).__qualname__}"
    return RunError(error_type, str(error), stage, decision, phase, attempted)


def _round_number(engine: GameEngine, phase: Phase) -> int:
    role = (
        Role.FUGITIVE
        if phase
        in (Phase.FUGITIVE_OPENING, Phase.FUGITIVE_DRAW, Phase.FUGITIVE_ACTION)
        else Role.MARSHAL
    )
    return engine.observation(role).round_number


def _validate_max_decisions(max_decisions: int | None) -> None:
    if max_decisions is None:
        return
    if (
        isinstance(max_decisions, bool)
        or not isinstance(max_decisions, int)
        or max_decisions < 0
    ):
        raise ValueError("max_decisions must be a non-negative integer or None")


def _agent_name(agent: object) -> str:
    name = getattr(agent, "name", None)
    if isinstance(name, str) and name:
        return name
    return f"{type(agent).__module__}.{type(agent).__qualname__}"


def _safe_json_scalar(value: object) -> JSONValue:
    if value is None or isinstance(value, (bool, int, float, str)):
        return _jsonable(value)
    return {"python_type": f"{type(value).__module__}.{type(value).__qualname__}"}


def _safe_attempted_action(
    attempted: Mapping[str, object] | None,
) -> dict[str, JSONValue] | None:
    if attempted is None:
        return None
    result: dict[str, JSONValue] = {}
    for key, value in attempted.items():
        try:
            result[key] = _jsonable(value)
        except ValueError:
            result[key] = _safe_json_scalar(value)
    return result


def _require_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return cast(Mapping[str, object], value)


def _require_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return value


__all__ = [
    "AgentDescriptor",
    "DrawTrace",
    "ExperimentRun",
    "ExperimentStatus",
    "FugitiveTrace",
    "GuessTrace",
    "MANIFEST_SCHEMA_VERSION",
    "RULES_SHA256",
    "RULES_VERSION",
    "ReplayManifest",
    "ReplayMismatchError",
    "ReplayVerification",
    "RunError",
    "SeedBundle",
    "derive_seed",
    "derive_seed_bundle",
    "replay_manifest",
    "run_experiment",
    "run_registered_experiment",
    "state_sha256",
]
