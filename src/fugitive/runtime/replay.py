"""Strict replay verification and canonical engine-state fingerprints."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Callable, TypeAlias

from ..game.driver import DecisionRecord, DrawTrace, FugitiveTrace, GuessTrace
from ..game.engine import GameEngine
from ..game.model import FugitiveAction, GameResult, Phase, Role
from .manifest import (
    MANIFEST_SCHEMA_VERSION,
    RULES_SHA256,
    RULES_VERSION,
    MatchStatus,
    ReplayManifest,
    _to_json_value,
    _validate_max_decisions,
)


ReplayInspectionCallback: TypeAlias = Callable[[GameEngine, DecisionRecord], None]


class ReplayMismatchError(ValueError):
    """Raised when a seed and recorded trace do not reproduce the manifest."""


@dataclass(frozen=True, slots=True)
class ReplayVerification:
    """Successful reconstruction of a manifest's recorded trace."""

    status: MatchStatus
    decision_count: int
    final_state_sha256: str
    game_result: GameResult | None


def replay_manifest(
    manifest: ReplayManifest,
    *,
    validate_invariants: bool = True,
    inspect_before_decision: ReplayInspectionCallback | None = None,
) -> ReplayVerification:
    """Replay and strictly verify every recorded action and outcome.

    ``inspect_before_decision`` is an offline, read-only inspection hook.  It is
    called after the recorded phase and round have been checked, immediately
    before that decision is applied.  The callback must not mutate the engine.
    It exists so audit tooling can take immutable snapshots without changing
    the agent-facing observation API or the replay manifest schema.
    """

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
                if inspect_before_decision is not None:
                    inspect_before_decision(engine, record)
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
                if inspect_before_decision is not None:
                    inspect_before_decision(engine, record)
                engine.apply_fugitive_action(
                    FugitiveAction(record.hideout, record.sprint_cards)
                )
            elif isinstance(record, GuessTrace):
                if record.role is not Role.MARSHAL:
                    raise ReplayMismatchError(
                        f"decision {record.decision}: invalid guess role"
                    )
                if inspect_before_decision is not None:
                    inspect_before_decision(engine, record)
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

    state_hash = state_sha256(engine)
    if state_hash != manifest.final_state_sha256:
        raise ReplayMismatchError("final state fingerprint differs from manifest")

    game_result: GameResult | None = None
    if manifest.status is MatchStatus.COMPLETED:
        if not engine.is_terminal:
            raise ReplayMismatchError("completed manifest does not reach terminal state")
        game_result = engine.result()
        if (
            game_result.winner is not manifest.winner
            or game_result.reason != manifest.reason
            or game_result.rounds != manifest.rounds
        ):
            raise ReplayMismatchError("terminal result differs from manifest")
    elif manifest.status is MatchStatus.TRUNCATED:
        if engine.is_terminal:
            raise ReplayMismatchError("truncated manifest already has a rules winner")
        if manifest.winner is not None or manifest.reason is not None:
            raise ReplayMismatchError("truncated manifest must not contain a winner")
    elif manifest.status is MatchStatus.ERROR:
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
    if manifest.status is MatchStatus.COMPLETED:
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
    if manifest.status is MatchStatus.TRUNCATED:
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


def state_sha256(engine: GameEngine) -> str:
    """Return the canonical replay fingerprint for one engine state."""

    payload: dict[str, Any] = {
        "fugitive": _to_json_value(engine.observation(Role.FUGITIVE)),
        "marshal": _to_json_value(engine.observation(Role.MARSHAL)),
        "terminal": engine.is_terminal,
    }
    if engine.is_terminal:
        payload["result"] = _to_json_value(engine.result())
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _round_number(engine: GameEngine, phase: Phase) -> int:
    role = (
        Role.FUGITIVE
        if phase
        in (Phase.FUGITIVE_OPENING, Phase.FUGITIVE_DRAW, Phase.FUGITIVE_ACTION)
        else Role.MARSHAL
    )
    return engine.observation(role).round_number


__all__ = [
    "ReplayInspectionCallback",
    "ReplayMismatchError",
    "ReplayVerification",
    "replay_manifest",
    "state_sha256",
]
