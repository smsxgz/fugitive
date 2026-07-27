"""Web-facing game sessions, perspectives, traces, and agent catalogue.

The API deliberately serializes only an engine :class:`Observation` for a
human viewer.  Spectators can explicitly choose an omniscient, role-specific,
or public perspective.  The session object keeps the engine and agent
instances server-side so changing perspective never mutates game state.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from http import HTTPStatus
import json
import logging
import random
import secrets
import threading
from typing import Callable, Mapping

from ..agents.common.base import bounded_fugitive_actions
from ..agents.registry import (
    DEFAULT_FUGITIVE_AGENT,
    DEFAULT_MARSHAL_AGENT,
    FUGITIVE_AGENT_REGISTRY,
    MARSHAL_AGENT_REGISTRY,
)
from ..game.driver import (
    DrawDecision,
    DrawTrace,
    FugitiveTrace,
    GuessDecision,
    GuessTrace,
    DecisionRecord,
    apply_decision,
    role_for_phase,
    step_agent,
)
from ..game.engine import GameEngine
from ..runtime.manifest import (
    MANIFEST_SCHEMA_VERSION,
    RULES_SHA256,
    RULES_VERSION,
    MatchStatus,
    ReplayManifest,
)
from ..runtime.replay import state_sha256
from ..agents.marshal.inference.diagnostics import (
    InferenceDiagnosticFailure,
    InferenceEvent,
    read_inference_diagnostics,
)
from ..game.model import (
    FugitiveAction,
    IllegalActionError,
    Observation,
    Phase,
    Role,
)
from ..game.observation import observation_to_canonical_data
from ..shared.reproducibility import (
    AgentDescriptor,
    AgentSpec,
    SEED_DERIVATION_VERSION,
    derive_seed_bundle,
    parse_seed,
    thaw_parameters,
)
from ..agents.planning.rollout_diagnostics import (
    RolloutDiagnosticFailure,
    RolloutEvent,
    read_rollout_diagnostics,
)


MODES = ("human_fugitive", "human_marshal", "spectate")
SPECTATOR_VIEWS = ("omniscient", "fugitive", "marshal", "public")
EXECUTION_PROFILES = ("full", "quick")
_REGISTRY_PROFILE_BY_EXECUTION_PROFILE = {
    "full": "default",
    "quick": "interactive",
}
_PROFILE_UNSET = object()
DEFAULT_AUTO_STEP_LIMIT = 10_000
MAX_SAFE_JSON_INTEGER = (1 << 53) - 1
WEB_TRACE_SCHEMA_VERSION = 2

_LOGGER = logging.getLogger(__name__)

_FUGITIVE_LABELS = {
    "hierarchical-random": "Hierarchical Legal Random (HR-1)",
    "belief-informed-random": "Belief-Informed Random (BIR-1)",
    "continuation-count": "Continuation-Count Fugitive",
    "belief-rollout": "Full-Game Belief Rollout Fugitive",
}
_MARSHAL_LABELS = {
    "hierarchical-random": "Hierarchical Legal Random (HR-1)",
    "belief-informed-random": "BIR-1 · Constructive Bootstrap Filter",
    "route-count-random": "Route-Count Random (HR-1.1)",
    "support-catalogue-random": "Shared-Catalogue Support Random (HR-1C)",
    "route-count-catalogue-random": (
        "Shared-Catalogue Route-Count Random (HR-1.1C)"
    ),
    "constructive-belief-informed-random": (
        "BIR-2S · Constructive SNIS · Sequential Sprint"
    ),
    "unweighted-constructive-belief-informed-random": (
        "BIR-2U · Constructive Unweighted · Sequential Sprint"
    ),
    "rollout-bir2u": "BIR-2U Full-Game Rollout Marshal",
    "exact-sprint-belief-informed-random": (
        "BIR-2E · Constructive SNIS · Exact Sprint DP · very slow"
    ),
    "mcmc-belief-informed-random": "BIR-3 · SIR + Independent MH",
}


class WebAPIError(Exception):
    """An expected client error with a stable JSON error code."""

    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


def _parse_execution_profile(value: object) -> str:
    if not isinstance(value, str) or value not in EXECUTION_PROFILES:
        raise WebAPIError(
            HTTPStatus.BAD_REQUEST,
            "invalid_execution_profile",
            f"execution_profile must be one of {', '.join(EXECUTION_PROFILES)}",
        )
    return value


def agent_catalog() -> dict[str, object]:
    """Return role-specific agents without exposing implementation objects."""

    return {
        "defaults": {
            Role.FUGITIVE.value: DEFAULT_FUGITIVE_AGENT,
            Role.MARSHAL.value: DEFAULT_MARSHAL_AGENT,
        },
        "fugitive": [
            {
                "id": name,
                "name": _FUGITIVE_LABELS.get(name, name),
                "label": _FUGITIVE_LABELS.get(name, name),
                "role": Role.FUGITIVE.value,
                "expensive": registration.expensive,
            }
            for name, registration in sorted(FUGITIVE_AGENT_REGISTRY.items())
        ],
        "marshal": [
            {
                "id": name,
                "name": _MARSHAL_LABELS.get(name, name),
                "label": _MARSHAL_LABELS.get(name, name),
                "role": Role.MARSHAL.value,
                "expensive": registration.expensive,
            }
            for name, registration in sorted(MARSHAL_AGENT_REGISTRY.items())
        ],
        "spectator_views": [
            {"id": "omniscient", "label": "Omniscient"},
            {"id": "fugitive", "label": "Fugitive view"},
            {"id": "marshal", "label": "Marshal view"},
            {"id": "public", "label": "Public view"},
        ],
    }


def _is_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def parse_web_seed(value: object) -> int | None:
    """Parse the Web seed protocol without accepting lossy JSON numbers."""

    if value is None:
        return None
    if _is_integer(value):
        if not 0 <= value <= MAX_SAFE_JSON_INTEGER:
            raise WebAPIError(
                HTTPStatus.BAD_REQUEST,
                "invalid_seed",
                "numeric seeds must be JavaScript-safe integers; send larger seeds as decimal strings",
            )
    try:
        return parse_seed(value, "master")
    except ValueError as exc:
        raise WebAPIError(
            HTTPStatus.BAD_REQUEST,
            "invalid_seed",
            "seed must be a canonical uint64 decimal string or a JavaScript-safe integer",
        ) from exc


@dataclass(frozen=True, slots=True)
class _Viewer:
    role: Role | None
    spectator_view: str | None = None

    @property
    def omniscient(self) -> bool:
        return self.spectator_view == "omniscient"


class GameSession:
    """One isolated game plus its persistent agent random state."""

    def __init__(
        self,
        *,
        session_id: str,
        mode: str,
        fugitive_agent: str,
        marshal_agent: str,
        seed: int | str | None,
        spectator_view: str | None = None,
        execution_profile: str = "full",
        auto_step_limit: int = DEFAULT_AUTO_STEP_LIMIT,
    ) -> None:
        if mode not in MODES:
            raise WebAPIError(
                HTTPStatus.BAD_REQUEST,
                "invalid_mode",
                f"mode must be one of {', '.join(MODES)}",
            )
        if fugitive_agent not in FUGITIVE_AGENT_REGISTRY:
            raise WebAPIError(
                HTTPStatus.BAD_REQUEST,
                "unknown_fugitive_agent",
                f"unknown Fugitive agent {fugitive_agent!r}",
            )
        if marshal_agent not in MARSHAL_AGENT_REGISTRY:
            raise WebAPIError(
                HTTPStatus.BAD_REQUEST,
                "unknown_marshal_agent",
                f"unknown Marshal agent {marshal_agent!r}",
            )
        parsed_seed = parse_web_seed(seed)
        parsed_execution_profile = _parse_execution_profile(execution_profile)
        if spectator_view is not None and spectator_view not in SPECTATOR_VIEWS:
            raise WebAPIError(
                HTTPStatus.BAD_REQUEST,
                "invalid_spectator_view",
                f"spectator_view must be one of {', '.join(SPECTATOR_VIEWS)}",
            )
        if mode != "spectate" and spectator_view is not None:
            raise WebAPIError(
                HTTPStatus.CONFLICT,
                "spectator_view_not_available",
                "spectator_view is available only in spectate mode",
            )
        if (
            not _is_integer(auto_step_limit)
            or auto_step_limit < 1
            or auto_step_limit > DEFAULT_AUTO_STEP_LIMIT
        ):
            raise ValueError(
                f"auto_step_limit must be from 1 through {DEFAULT_AUTO_STEP_LIMIT}"
            )

        self.id = session_id
        self.mode = mode
        self.fugitive_agent_name = fugitive_agent
        self.marshal_agent_name = marshal_agent
        self.execution_profile = parsed_execution_profile
        self.spectator_view = (
            spectator_view or "omniscient" if mode == "spectate" else None
        )
        self._lock = threading.RLock()
        self._events: list[dict[str, object]] = []
        self._decision_count = 0
        self.auto_step_limit = auto_step_limit
        self._stalled = False
        self._pause_reason: str | None = None
        self._terminated = False
        self._termination_reason: str | None = None
        self._closed = False
        self._seed_was_supplied = parsed_seed is not None
        self.seed = parsed_seed if parsed_seed is not None else secrets.randbits(64)
        self.engine: GameEngine
        self.fugitive_player: object | None = None
        self.marshal_player: object | None = None
        try:
            self._build_game()
        except Exception:
            self._release_agents()
            raise
        if self.human_role is not None:
            self._advance_until_human(self.auto_step_limit)

    @property
    def human_role(self) -> Role | None:
        if self.mode == "human_fugitive":
            return Role.FUGITIVE
        if self.mode == "human_marshal":
            return Role.MARSHAL
        return None

    @property
    def decision_trace(self) -> tuple[DecisionRecord, ...]:
        """Canonical agent-decision trace, excluding automatic setup draws."""

        with self._lock:
            return tuple(self._decision_trace)

    @property
    def viewer(self) -> _Viewer:
        if self.human_role is not None:
            return _Viewer(self.human_role)
        role = None
        if self.spectator_view == Role.FUGITIVE.value:
            role = Role.FUGITIVE
        elif self.spectator_view == Role.MARSHAL.value:
            role = Role.MARSHAL
        return _Viewer(role, self.spectator_view)

    def _build_game(self) -> None:
        self._seed_bundle = derive_seed_bundle(self.seed)
        assert self._seed_bundle.fugitive is not None
        assert self._seed_bundle.marshal is not None
        self.engine = GameEngine(seed=self._seed_bundle.deck)
        registry_profile = _REGISTRY_PROFILE_BY_EXECUTION_PROFILE[
            self.execution_profile
        ]
        built_fugitive = FUGITIVE_AGENT_REGISTRY[self.fugitive_agent_name].build(
            self._seed_bundle.fugitive,
            profile=registry_profile,
        )
        built_marshal = MARSHAL_AGENT_REGISTRY[self.marshal_agent_name].build(
            self._seed_bundle.marshal,
            profile=registry_profile,
        )
        self.fugitive_player = built_fugitive.agent
        self.marshal_player = built_marshal.agent
        self._agent_specs = {
            Role.FUGITIVE.value: built_fugitive.spec,
            Role.MARSHAL.value: built_marshal.spec,
        }
        self._agent_configurations = {
            Role.FUGITIVE.value: self._agent_configuration(
                self.fugitive_player,
                built_fugitive.spec,
            ),
            Role.MARSHAL.value: self._agent_configuration(
                self.marshal_player,
                built_marshal.spec,
            ),
        }
        self._events = []
        self._inference_events: list[InferenceEvent] = []
        self._inference_diagnostic_failures: list[
            InferenceDiagnosticFailure
        ] = []
        self._rollout_events: list[RolloutEvent] = []
        self._rollout_diagnostic_failures: list[
            RolloutDiagnosticFailure
        ] = []
        self._decision_count = 0
        self._decision_trace: list[DecisionRecord] = []
        # Setup draws are part of the UI history, but serialization below keeps
        # each identity private to its owner.
        setup = self.engine.observation(Role.FUGITIVE).draw_history
        for record in setup:
            self._events.append(
                {
                    "type": "draw",
                    "decision": 0,
                    "phase": "setup",
                    "role": record.role.value,
                    "pile": record.pile,
                    "card": record.card,
                    "round_number": record.round_number,
                }
            )

    @staticmethod
    def _agent_configuration(
        player: object,
        spec: AgentSpec,
    ) -> dict[str, object]:
        return {
            "class": f"{type(player).__module__}.{type(player).__qualname__}",
            "name": getattr(player, "name", type(player).__name__),
            "profile": spec.profile,
            "parameters": thaw_parameters(spec.parameters),
        }

    def _serialized_agents(self) -> dict[str, object]:
        return {
            Role.FUGITIVE.value: {
                "registry_name": self.fugitive_agent_name,
                **copy.deepcopy(
                    self._agent_configurations[Role.FUGITIVE.value]
                ),
            },
            Role.MARSHAL.value: {
                "registry_name": self.marshal_agent_name,
                **copy.deepcopy(self._agent_configurations[Role.MARSHAL.value]),
            },
        }

    def _ensure_open(self) -> None:
        if self._closed:
            raise WebAPIError(
                HTTPStatus.GONE,
                "session_closed",
                "the game session has been released",
            )

    def _ensure_playable(self) -> None:
        self._ensure_open()
        if self._terminated:
            raise WebAPIError(
                HTTPStatus.CONFLICT,
                "session_terminated",
                "the game session has been terminated",
            )
        if self._stalled:
            raise WebAPIError(
                HTTPStatus.CONFLICT,
                "session_stalled",
                "automatic play reached its safety limit; continue or terminate the session",
            )

    @staticmethod
    def _release_player(player: object | None) -> None:
        if player is None:
            return
        # Agents may optionally expose an explicit resource protocol.  The
        # built-in agents need only be dereferenced; their caches then become
        # collectible with the agent instance.
        for method_name in ("clear_cache", "close"):
            method = getattr(player, method_name, None)
            if callable(method):
                try:
                    method()
                except Exception:
                    _LOGGER.exception(
                        "agent cleanup failed method=%s agent=%s",
                        method_name,
                        type(player).__name__,
                    )

    def _release_agents(self) -> None:
        players = (self.fugitive_player, self.marshal_player)
        self.fugitive_player = None
        self.marshal_player = None
        seen: set[int] = set()
        for player in players:
            if player is None or id(player) in seen:
                continue
            seen.add(id(player))
            self._release_player(player)

    def close(self) -> None:
        """Release agent state after store eviction or explicit deletion."""

        with self._lock:
            if self._closed:
                return
            self._release_agents()
            self._stalled = False
            self._pause_reason = None
            self._closed = True

    def reset(
        self,
        *,
        seed: int | str | None = None,
        execution_profile: object = _PROFILE_UNSET,
    ) -> dict[str, object]:
        """Reset this session; omission intentionally deals a fresh game."""

        parsed_seed = parse_web_seed(seed)
        parsed_execution_profile = (
            self.execution_profile
            if execution_profile is _PROFILE_UNSET
            else _parse_execution_profile(execution_profile)
        )
        with self._lock:
            self._ensure_open()
            self._release_agents()
            self._stalled = False
            self._pause_reason = None
            self._terminated = False
            self._termination_reason = None
            self._seed_was_supplied = parsed_seed is not None
            self.seed = parsed_seed if parsed_seed is not None else secrets.randbits(64)
            self.execution_profile = parsed_execution_profile
            try:
                self._build_game()
            except Exception:
                self._release_agents()
                raise
            if self.human_role is not None:
                self._advance_until_human(self.auto_step_limit)
            return self.as_dict()

    def set_spectator_view(self, spectator_view: object) -> dict[str, object]:
        """Change only the serialization perspective for a spectator game."""

        if self.mode != "spectate":
            raise WebAPIError(
                HTTPStatus.CONFLICT,
                "spectator_view_not_available",
                "spectator_view is available only in spectate mode",
            )
        if not isinstance(spectator_view, str) or spectator_view not in SPECTATOR_VIEWS:
            raise WebAPIError(
                HTTPStatus.BAD_REQUEST,
                "invalid_spectator_view",
                f"spectator_view must be one of {', '.join(SPECTATOR_VIEWS)}",
            )
        with self._lock:
            self._ensure_open()
            self.spectator_view = spectator_view
            return self.as_dict()

    def _event_header(self, phase: Phase, round_number: int) -> dict[str, object]:
        self._decision_count += 1
        return {
            "decision": self._decision_count,
            "phase": phase.value,
            "round_number": round_number,
        }

    def _record_draw(
        self,
        role: Role,
        pile: int,
        card: int,
        *,
        phase: Phase,
        round_number: int,
    ) -> None:
        self._events.append(
            {
                **self._event_header(phase, round_number),
                "type": "draw",
                "role": role.value,
                "pile": pile,
                "card": card,
            }
        )

    def _record_fugitive_action(
        self,
        action: FugitiveAction,
        *,
        phase: Phase,
        round_number: int,
    ) -> None:
        self._events.append(
            {
                **self._event_header(phase, round_number),
                "type": "pass" if action.hideout is None else "fugitive_action",
                "role": Role.FUGITIVE.value,
                "hideout": action.hideout,
                "sprint_cards": list(action.sprint_cards),
                "sprint_count": len(action.sprint_cards),
                "route_index": (
                    None if action.hideout is None else len(self.engine.observation(Role.FUGITIVE).route) - 1
                ),
            }
        )

    def _record_guess(
        self,
        numbers: tuple[int, ...],
        success: bool,
        *,
        phase: Phase,
        round_number: int,
    ) -> None:
        observation = self.engine.observation(Role.MARSHAL)
        record = observation.guess_history[-1]
        self._events.append(
            {
                **self._event_header(phase, round_number),
                "type": "guess",
                "role": Role.MARSHAL.value,
                "numbers": list(numbers),
                "success": success,
                "manhunt": record.manhunt,
                "route_length": record.route_length,
            }
        )

    def _record_decision(self, record: DecisionRecord) -> None:
        self._decision_trace.append(record)
        if isinstance(record, DrawTrace):
            self._record_draw(
                record.role,
                record.pile,
                record.card,
                phase=record.phase,
                round_number=record.round_number,
            )
        elif isinstance(record, FugitiveTrace):
            self._record_fugitive_action(
                FugitiveAction(record.hideout, record.sprint_cards),
                phase=record.phase,
                round_number=record.round_number,
            )
        else:
            assert isinstance(record, GuessTrace)
            self._record_guess(
                record.numbers,
                record.success,
                phase=record.phase,
                round_number=record.round_number,
            )

    def _agent_step(self) -> None:
        if self.engine.is_terminal:
            return
        if self.fugitive_player is None or self.marshal_player is None:
            raise RuntimeError("an agent has been released")
        record = step_agent(
            self.engine,
            self.fugitive_player,  # type: ignore[arg-type]
            self.marshal_player,  # type: ignore[arg-type]
            decision=self._decision_count + 1,
        )
        self._record_decision(record)
        acting_agent = (
            self.fugitive_player
            if record.role is Role.FUGITIVE
            else self.marshal_player
        )
        try:
            diagnostics = read_inference_diagnostics(acting_agent)
            if diagnostics is not None:
                self._inference_events.append(
                    InferenceEvent(
                        decision=record.decision,
                        round_number=record.round_number,
                        phase=record.phase,
                        role=record.role,
                        diagnostics=diagnostics,
                    )
                )
        except Exception as diagnostics_error:
            self._inference_diagnostic_failures.append(
                InferenceDiagnosticFailure.from_exception(
                    decision=record.decision,
                    round_number=record.round_number,
                    phase=record.phase,
                    role=record.role,
                    error=diagnostics_error,
                )
            )
        try:
            rollout_diagnostics = read_rollout_diagnostics(acting_agent)
            if rollout_diagnostics is not None:
                self._rollout_events.append(
                    RolloutEvent(
                        decision=record.decision,
                        round_number=record.round_number,
                        phase=record.phase,
                        role=record.role,
                        diagnostics=rollout_diagnostics,
                    )
                )
        except Exception as diagnostics_error:
            self._rollout_diagnostic_failures.append(
                RolloutDiagnosticFailure.from_exception(
                    decision=record.decision,
                    round_number=record.round_number,
                    phase=record.phase,
                    role=record.role,
                    error=diagnostics_error,
                )
            )

    def _advance_until_human(self, max_steps: int) -> tuple[int, bool]:
        steps = 0
        while not self.engine.is_terminal:
            actor = role_for_phase(self.engine.phase)
            if actor is self.human_role:
                break
            if steps >= max_steps:
                self._stalled = True
                self._pause_reason = "auto_step_limit"
                return steps, True
            self._agent_step()
            steps += 1
        self._stalled = False
        self._pause_reason = None
        return steps, False

    def _validate_max_steps(self, max_steps: object) -> int:
        if (
            not _is_integer(max_steps)
            or max_steps < 1
            or max_steps > self.auto_step_limit
        ):
            raise WebAPIError(
                HTTPStatus.BAD_REQUEST,
                "invalid_max_steps",
                f"max_steps must be from 1 through {self.auto_step_limit}",
            )
        return max_steps

    def step(self) -> dict[str, object]:
        """Advance exactly one decision in a spectator game."""

        with self._lock:
            self._ensure_playable()
            if self.mode != "spectate":
                raise WebAPIError(
                    HTTPStatus.CONFLICT,
                    "step_not_available",
                    "step is available only for agent-v-agent spectator games",
                )
            if self.engine.is_terminal:
                raise WebAPIError(
                    HTTPStatus.CONFLICT,
                    "game_finished",
                    "the game has already finished",
                )
            self._agent_step()
            return self.as_dict()

    def auto(self, *, max_steps: int | None = None) -> dict[str, object]:
        """Advance an agent-v-agent game, pausing rather than cutting it off."""

        checked_steps = self._validate_max_steps(
            self.auto_step_limit if max_steps is None else max_steps
        )
        with self._lock:
            self._ensure_playable()
            if self.mode != "spectate":
                raise WebAPIError(
                    HTTPStatus.CONFLICT,
                    "auto_not_available",
                    "auto is available only for agent-v-agent spectator games",
                )
            steps, paused = self._advance_until_human(checked_steps)
            state = self.as_dict()
            state["auto_steps"] = steps
            state["auto_paused"] = paused
            return state

    def continue_after_stall(
        self, *, max_steps: int | None = None
    ) -> dict[str, object]:
        """Resume agent execution after an automatic-step safety pause."""

        checked_steps = self._validate_max_steps(
            self.auto_step_limit if max_steps is None else max_steps
        )
        with self._lock:
            self._ensure_open()
            if self._terminated:
                raise WebAPIError(
                    HTTPStatus.CONFLICT,
                    "session_terminated",
                    "the game session has been terminated",
                )
            if self.engine.is_terminal:
                raise WebAPIError(
                    HTTPStatus.CONFLICT,
                    "game_finished",
                    "the game has already finished",
                )
            if not self._stalled:
                raise WebAPIError(
                    HTTPStatus.CONFLICT,
                    "session_not_stalled",
                    "the game session is not waiting for continuation",
                )
            self._stalled = False
            self._pause_reason = None
            steps, paused = self._advance_until_human(checked_steps)
            state = self.as_dict()
            state["auto_steps"] = steps
            state["auto_paused"] = paused
            return state

    def terminate(self) -> dict[str, object]:
        """Stop a live session without assigning a rules-level winner."""

        with self._lock:
            self._ensure_open()
            if self.engine.is_terminal:
                raise WebAPIError(
                    HTTPStatus.CONFLICT,
                    "game_finished",
                    "the game has already finished",
                )
            if not self._terminated:
                self._terminated = True
                self._termination_reason = "terminated_by_user"
                self._stalled = False
                self._pause_reason = None
                self._release_agents()
            return self.as_dict()

    def apply_human_action(self, payload: Mapping[str, object]) -> dict[str, object]:
        """Validate one human decision, then run agents to the next human turn."""

        with self._lock:
            self._ensure_playable()
            if self.human_role is None:
                raise WebAPIError(
                    HTTPStatus.CONFLICT,
                    "human_action_not_available",
                    "spectator games do not accept human actions",
                )
            if self.engine.is_terminal:
                raise WebAPIError(
                    HTTPStatus.CONFLICT,
                    "game_finished",
                    "the game has already finished",
                )
            actor = role_for_phase(self.engine.phase)
            if actor is not self.human_role:
                raise WebAPIError(
                    HTTPStatus.CONFLICT,
                    "not_human_turn",
                    "the session is currently waiting for an agent",
                )
            action_type = payload.get("type")
            try:
                if action_type == "draw":
                    pile = payload.get("pile")
                    if not _is_integer(pile):
                        raise IllegalActionError("pile must be an integer")
                    decision = DrawDecision(pile)
                elif action_type in ("fugitive_action", "play"):
                    if self.human_role is not Role.FUGITIVE:
                        raise IllegalActionError("only the Fugitive can establish a Hideout")
                    hideout = payload.get("hideout")
                    if not _is_integer(hideout):
                        raise IllegalActionError("hideout must be an integer")
                    sprint_cards = _integer_tuple(payload.get("sprint_cards", []), "sprint_cards")
                    decision = FugitiveAction(hideout, sprint_cards)
                elif action_type == "pass":
                    if self.human_role is not Role.FUGITIVE:
                        raise IllegalActionError("only the Fugitive can pass")
                    decision = FugitiveAction(None)
                elif action_type == "guess":
                    if self.human_role is not Role.MARSHAL:
                        raise IllegalActionError("only the Marshal can guess")
                    numbers = _integer_tuple(payload.get("numbers"), "numbers")
                    decision = GuessDecision(numbers)
                else:
                    raise IllegalActionError(
                        "type must be draw, fugitive_action, pass, or guess"
                    )
                record = apply_decision(
                    self.engine,
                    decision,
                    decision=self._decision_count + 1,
                )
                self._record_decision(record)
            except IllegalActionError as exc:
                raise WebAPIError(
                    HTTPStatus.UNPROCESSABLE_ENTITY,
                    "illegal_action",
                    str(exc),
                ) from exc

            _steps, paused = self._advance_until_human(self.auto_step_limit)
            state = self.as_dict()
            state["auto_paused"] = paused
            return state

    def _viewer_observations(
        self,
    ) -> tuple[Observation, Observation, Observation]:
        """Return the selected data, Fugitive, and Marshal observations."""

        fugitive_view = self.engine.observation(Role.FUGITIVE)
        marshal_view = self.engine.observation(Role.MARSHAL)
        viewer = self.viewer
        if self.human_role is not None:
            selected = (
                fugitive_view
                if self.human_role is Role.FUGITIVE
                else marshal_view
            )
        elif viewer.role is Role.FUGITIVE:
            selected = fugitive_view
        elif viewer.role is Role.MARSHAL:
            selected = marshal_view
        else:
            actor = role_for_phase(self.engine.phase)
            selected = fugitive_view if actor is Role.FUGITIVE else marshal_view
        return selected, fugitive_view, marshal_view

    def _legal_actions(self, observation: Observation) -> dict[str, object]:
        empty: dict[str, object] = {
            "draw_piles": [],
            "fugitive_actions": [],
            "candidate_hideouts": [],
            "sprint_cards": [],
            "previous_hideout": None,
            "can_pass": False,
            "representative_actions_only": False,
            "guess_numbers": [],
            "min_guess_count": 0,
            "max_guess_count": 0,
        }
        if (
            self.engine.is_terminal
            or self._stalled
            or self._terminated
            or self._closed
            or self.human_role is None
        ):
            return empty
        if role_for_phase(self.engine.phase) is not self.human_role:
            return empty
        if self.engine.phase in (Phase.FUGITIVE_DRAW, Phase.MARSHAL_DRAW):
            empty["draw_piles"] = list(observation.legal_draw_piles)
        elif self.engine.phase in (Phase.FUGITIVE_OPENING, Phase.FUGITIVE_ACTION):
            actions = bounded_fugitive_actions(
                observation,
                rng=random.Random(0),
                include_pass=True,
                max_per_hideout=6,
            )
            empty["candidate_hideouts"] = sorted(
                {
                    action.hideout
                    for action in actions
                    if action.hideout is not None
                }
            )
            empty["sprint_cards"] = list(observation.hand)
            empty["previous_hideout"] = observation.route[-1].hideout
            empty["can_pass"] = self.engine.phase is Phase.FUGITIVE_ACTION
            # This list is convenient for one-click choices, but it is an
            # action abstraction only.  The action endpoint accepts every
            # hand-selected Sprint combination and delegates exact legality to
            # GameEngine.
            empty["representative_actions_only"] = True
            empty["fugitive_actions"] = [
                (
                    {"type": "pass"}
                    if action.hideout is None
                    else {
                        "type": "fugitive_action",
                        "hideout": action.hideout,
                        "sprint_cards": list(action.sprint_cards),
                    }
                )
                for action in actions
            ]
        elif self.engine.phase in (Phase.MARSHAL_GUESS, Phase.MANHUNT):
            empty["guess_numbers"] = list(range(1, 42))
            empty["min_guess_count"] = 1
            empty["max_guess_count"] = (
                1 if self.engine.phase is Phase.MANHUNT else 41
            )
        return empty

    def _serialize_history(
        self, viewer: _Viewer, marshal_view: Observation
    ) -> list[dict[str, object]]:
        public_route = {slot.index: slot for slot in marshal_view.route}
        result: list[dict[str, object]] = []
        for event in self._events:
            item = dict(event)
            if item["type"] == "draw":
                if not viewer.omniscient and (
                    viewer.role is None or item["role"] != viewer.role.value
                ):
                    item["card"] = None
            elif (
                item["type"] == "fugitive_action"
                and not viewer.omniscient
                and viewer.role is not Role.FUGITIVE
            ):
                route_index = item.get("route_index")
                slot = public_route.get(route_index) if isinstance(route_index, int) else None
                item["hideout"] = slot.hideout if slot is not None else None
                item["sprint_cards"] = (
                    list(slot.sprint_cards)
                    if slot is not None and slot.sprint_cards is not None
                    else None
                )
            item.pop("route_index", None)
            result.append(item)
        return result

    def export_trace(self) -> dict[str, object]:
        """Return a full replay-oriented trace after play has stopped."""

        with self._lock:
            self._ensure_open()
            if not self.engine.is_terminal and not self._terminated:
                raise WebAPIError(
                    HTTPStatus.CONFLICT,
                    "export_not_ready",
                    "a full trace is available only after the game or session ends",
                )
            fugitive_view = self.engine.observation(Role.FUGITIVE)
            marshal_view = self.engine.observation(Role.MARSHAL)
            if self.engine.is_terminal:
                result = self.engine.result()
                status = "finished"
                winner = result.winner.value
                reason = result.reason
                rounds = result.rounds
                fugitive_descriptor = self._agent_specs[
                    Role.FUGITIVE.value
                ].descriptor()
                marshal_descriptor = self._agent_specs[
                    Role.MARSHAL.value
                ].descriptor()
                if self.human_role is Role.FUGITIVE:
                    fugitive_descriptor = AgentDescriptor.create("human-fugitive")
                elif self.human_role is Role.MARSHAL:
                    marshal_descriptor = AgentDescriptor.create("human-marshal")
                replay = ReplayManifest(
                    schema_version=MANIFEST_SCHEMA_VERSION,
                    rules_version=RULES_VERSION,
                    rules_sha256=RULES_SHA256,
                    seeds=self._seed_bundle,
                    fugitive_agent=fugitive_descriptor,
                    marshal_agent=marshal_descriptor,
                    status=MatchStatus.COMPLETED,
                    decision_count=len(self._decision_trace),
                    max_decisions=None,
                    trace=tuple(self._decision_trace),
                    final_state_sha256=state_sha256(self.engine),
                    winner=result.winner,
                    reason=result.reason,
                    rounds=result.rounds,
                ).to_dict()
            else:
                status = "terminated"
                winner = None
                reason = self._termination_reason
                rounds = fugitive_view.round_number
                replay = None
            return {
                "schema": "fugitive.web-trace",
                "schema_version": WEB_TRACE_SCHEMA_VERSION,
                "rules": {
                    "version": RULES_VERSION,
                    "sha256": RULES_SHA256,
                },
                "seed_derivation": {
                    "version": SEED_DERIVATION_VERSION,
                    "master": str(self.seed),
                    "deck": str(self._seed_bundle.deck),
                    "fugitive": str(self._seed_bundle.fugitive),
                    "marshal": str(self._seed_bundle.marshal),
                },
                "mode": self.mode,
                "execution_profile": self.execution_profile,
                "agents": self._serialized_agents(),
                "outcome": {
                    "status": status,
                    "winner": winner,
                    "reason": reason,
                    "rounds": rounds,
                    "decision_count": self._decision_count,
                },
                "trace": json.loads(json.dumps(self._events)),
                "inference_events": [
                    event.to_dict() for event in self._inference_events
                ],
                "inference_diagnostic_failures": [
                    failure.to_dict()
                    for failure in self._inference_diagnostic_failures
                ],
                "rollout_events": [
                    event.to_dict() for event in self._rollout_events
                ],
                "rollout_diagnostic_failures": [
                    failure.to_dict()
                    for failure in self._rollout_diagnostic_failures
                ],
                "replay_manifest": replay,
                "observations": {
                    "fugitive": observation_to_canonical_data(fugitive_view)[
                        "observation"
                    ],
                    "marshal": observation_to_canonical_data(marshal_view)[
                        "observation"
                    ],
                },
            }

    def as_dict(self) -> dict[str, object]:
        """Serialize state with viewer-appropriate redaction."""

        with self._lock:
            self._ensure_open()
            observation, fugitive_view, marshal_view = self._viewer_observations()
            viewer = self.viewer
            route = (
                fugitive_view.route
                if viewer.omniscient or viewer.role is Role.FUGITIVE
                else marshal_view.route
            )
            fugitive_hand: list[int] | None = None
            marshal_hand: list[int] | None = None
            if viewer.omniscient:
                fugitive_hand = list(fugitive_view.hand)
                marshal_hand = list(marshal_view.hand)
            elif viewer.role is Role.FUGITIVE:
                fugitive_hand = list(fugitive_view.hand)
            elif viewer.role is Role.MARSHAL:
                marshal_hand = list(marshal_view.hand)
            actor = role_for_phase(self.engine.phase)
            if viewer.omniscient and actor is Role.FUGITIVE:
                selected_hand = fugitive_hand
            elif viewer.omniscient and actor is Role.MARSHAL:
                selected_hand = marshal_hand
            elif viewer.role is Role.FUGITIVE:
                selected_hand = fugitive_hand
            elif viewer.role is Role.MARSHAL:
                selected_hand = marshal_hand
            else:
                selected_hand = []
            if self._terminated:
                winner = None
                reason = self._termination_reason
                status = "terminated"
            elif self.engine.is_terminal:
                result = self.engine.result()
                winner = result.winner.value if result.winner is not None else None
                reason = result.reason
                status = "finished"
            elif self._stalled:
                winner = None
                reason = None
                status = "stalled"
            else:
                winner = None
                reason = None
                actor = role_for_phase(self.engine.phase)
                status = "awaiting_human" if actor is self.human_role else "running"
            inference_events = (
                [event.to_dict() for event in self._inference_events]
                if self.mode == "spectate"
                and self.spectator_view in ("omniscient", "marshal")
                else []
            )
            inference_diagnostic_failures = (
                [
                    failure.to_dict()
                    for failure in self._inference_diagnostic_failures
                ]
                if self.mode == "spectate"
                and self.spectator_view in ("omniscient", "marshal")
                else []
            )
            rollout_events = (
                [
                    event.to_dict()
                    for event in self._rollout_events
                    if self.spectator_view == "omniscient"
                    or self.spectator_view == event.role.value
                ]
                if self.mode == "spectate"
                else []
            )
            rollout_diagnostic_failures = (
                [
                    failure.to_dict()
                    for failure in self._rollout_diagnostic_failures
                    if self.spectator_view == "omniscient"
                    or self.spectator_view == failure.role.value
                ]
                if self.mode == "spectate"
                else []
            )
            return {
                "id": self.id,
                "mode": self.mode,
                "execution_profile": self.execution_profile,
                "status": status,
                "phase": observation.phase.value,
                "round_number": observation.round_number,
                "viewer_role": (
                    self.human_role.value
                    if self.human_role is not None
                    else "spectator"
                ),
                "spectator_view": self.spectator_view,
                "available_spectator_views": list(SPECTATOR_VIEWS),
                "actor": (
                    role_for_phase(observation.phase).value
                    if role_for_phase(observation.phase) is not None
                    else None
                ),
                "fugitive_agent": self.fugitive_agent_name,
                "marshal_agent": self.marshal_agent_name,
                "agents": self._serialized_agents(),
                "pile_sizes": list(observation.pile_sizes),
                "route": [
                    {
                        "index": slot.index,
                        "hideout": slot.hideout,
                        "sprint_count": slot.sprint_count,
                        "sprint_cards": (
                            list(slot.sprint_cards)
                            if slot.sprint_cards is not None
                            else None
                        ),
                        "revealed": slot.revealed,
                    }
                    for slot in route
                ],
                "hand": selected_hand or [],
                "hands": {
                    Role.FUGITIVE.value: fugitive_hand,
                    Role.MARSHAL.value: marshal_hand,
                },
                "legal_actions": self._legal_actions(observation),
                "history": self._serialize_history(viewer, marshal_view),
                "inference_events": inference_events,
                "inference_diagnostic_failures": (
                    inference_diagnostic_failures
                ),
                "rollout_events": rollout_events,
                "rollout_diagnostic_failures": (
                    rollout_diagnostic_failures
                ),
                "winner": winner,
                "reason": reason,
                # An omitted random seed must not become a side channel into
                # the shuffled deck.  It is revealed only after the game.
                "seed": (
                    str(self.seed)
                    if self.engine.is_terminal or self._terminated
                    else None
                ),
                "seed_was_supplied": self._seed_was_supplied,
                "can_step": (
                    self.mode == "spectate"
                    and not self.engine.is_terminal
                    and not self._stalled
                    and not self._terminated
                ),
                "can_auto": (
                    self.mode == "spectate"
                    and not self.engine.is_terminal
                    and not self._stalled
                    and not self._terminated
                ),
                "can_continue": (
                    self._stalled
                    and not self.engine.is_terminal
                    and not self._terminated
                ),
                "can_terminate": not self.engine.is_terminal and not self._terminated,
                "can_export": self.engine.is_terminal or self._terminated,
                "auto_paused": self._stalled,
                "pause_reason": self._pause_reason,
                "auto_running": False,
                "auto_delay_ms": 0,
            }


def _integer_tuple(value: object, name: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise IllegalActionError(f"{name} must be a JSON array")
    if any(not _is_integer(item) for item in value):
        raise IllegalActionError(f"every item in {name} must be an integer")
    return tuple(value)


def replay_manifest_from_web_trace(
    payload: Mapping[str, object],
) -> ReplayManifest:
    """Extract a completed replay manifest from the current Web wrapper.

    Web trace v1 used a different seed derivation and had no canonical replay
    payload. It is deliberately rejected instead of being silently interpreted
    under the v2 protocol. Terminated sessions remain auditable prefixes but
    are not represented as completed match manifests.
    """

    if payload.get("schema") != "fugitive.web-trace":
        raise ValueError("not a fugitive.web-trace payload")
    version = payload.get("schema_version")
    if version != WEB_TRACE_SCHEMA_VERSION:
        raise ValueError(f"unsupported Web trace schema: {version}")
    manifest = payload.get("replay_manifest")
    if not isinstance(manifest, Mapping):
        raise ValueError("Web trace has no completed replay manifest")
    return ReplayManifest.from_dict(manifest)


__all__ = [
    "EXECUTION_PROFILES",
    "GameSession",
    "MAX_SAFE_JSON_INTEGER",
    "WEB_TRACE_SCHEMA_VERSION",
    "WebAPIError",
    "agent_catalog",
    "parse_web_seed",
    "replay_manifest_from_web_trace",
]
