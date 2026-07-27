"""Serializable contracts for reproducible full-game matches."""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
import json
import math
from typing import Mapping, cast

from ..agents.marshal.inference.diagnostics import (
    InferenceDiagnosticFailure,
    InferenceEvent,
)
from ..agents.planning.rollout_diagnostics import (
    RolloutDiagnosticFailure,
    RolloutEvent,
)
from ..game.driver import DecisionRecord, DrawTrace, FugitiveTrace, GuessTrace
from ..game.model import GameResult, Phase, Role, Winner
from ..shared.reproducibility import (
    AgentDescriptor,
    JSONValue,
    SeedBundle,
    normalize_parameters as _normalize_parameters,
)


MANIFEST_SCHEMA_VERSION = 1
RULES_VERSION = "fugitive-canonical-2026-07-23-v1"
# SHA-256 of docs/rules/CANONICAL_RULES.md after normalizing CRLF to LF.
RULES_SHA256 = "027b6c902e372bc0f385c1aa899ff1b6265b559342bf45db5550cf81709085b3"


class MatchStatus(str, Enum):
    COMPLETED = "completed"
    TRUNCATED = "truncated"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class MatchError:
    """Stable error envelope for a failed match decision."""

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
    def from_dict(cls, data: Mapping[str, object]) -> "MatchError":
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
    """Everything needed to audit and replay a match trajectory."""

    schema_version: int
    rules_version: str
    rules_sha256: str
    seeds: SeedBundle
    fugitive_agent: AgentDescriptor
    marshal_agent: AgentDescriptor
    status: MatchStatus
    decision_count: int
    max_decisions: int | None
    trace: tuple[DecisionRecord, ...]
    final_state_sha256: str
    winner: Winner | None = None
    reason: str | None = None
    rounds: int | None = None
    error: MatchError | None = None

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
            status=MatchStatus(str(outcome.get("status"))),
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
                else MatchError.from_dict(cast(Mapping[str, object], error_data))
            ),
        )

    @classmethod
    def from_json(cls, payload: str) -> "ReplayManifest":
        data = json.loads(payload)
        if not isinstance(data, Mapping):
            raise ValueError("manifest JSON must contain an object")
        return cls.from_dict(cast(Mapping[str, object], data))


@dataclass(frozen=True, slots=True)
class MatchRun:
    """One completed, truncated, or failed match invocation."""

    manifest: ReplayManifest
    game_result: GameResult | None
    inference_events: tuple[InferenceEvent, ...] = ()
    inference_diagnostic_failures: tuple[InferenceDiagnosticFailure, ...] = ()
    rollout_events: tuple[RolloutEvent, ...] = ()
    rollout_diagnostic_failures: tuple[RolloutDiagnosticFailure, ...] = ()

    @property
    def status(self) -> MatchStatus:
        return self.manifest.status


def _validate_max_decisions(max_decisions: int | None) -> None:
    """Validate the optional runtime watchdog stored in a manifest."""

    if max_decisions is None:
        return
    if (
        isinstance(max_decisions, bool)
        or not isinstance(max_decisions, int)
        or max_decisions < 0
    ):
        raise ValueError("max_decisions must be a non-negative integer or None")


def _to_json_value(value: object) -> JSONValue:
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
            field.name: _to_json_value(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        result: dict[str, JSONValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("JSON object keys must be strings")
            result[key] = _to_json_value(item)
        return result
    if isinstance(value, (tuple, list)):
        return [_to_json_value(item) for item in value]
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


def _require_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return cast(Mapping[str, object], value)


def _require_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return value


__all__ = [
    "MANIFEST_SCHEMA_VERSION",
    "RULES_SHA256",
    "RULES_VERSION",
    "MatchError",
    "MatchRun",
    "MatchStatus",
    "ReplayManifest",
]
