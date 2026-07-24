"""Dependency-free local web server and session API for playing Fugitive.

The API deliberately serializes only an engine :class:`Observation` for a
human viewer.  Spectators can explicitly choose an omniscient, role-specific,
or public perspective.  The session object keeps the engine and agent
instances server-side so changing perspective never mutates game state.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import math
from pathlib import Path
import random
import secrets
import threading
import time
from typing import Callable, Mapping
from urllib.parse import unquote, urlsplit
import uuid

from .agents.base import bounded_fugitive_actions
from .agents.registry import (
    DEFAULT_FUGITIVE_AGENT,
    DEFAULT_MARSHAL_AGENT,
    FUGITIVE_AGENT_REGISTRY,
    MARSHAL_AGENT_REGISTRY,
)
from .driver import DrawTrace, FugitiveTrace, GuessTrace, TraceRecord, step_agent
from .engine import GameEngine
from .experiment import (
    MANIFEST_SCHEMA_VERSION,
    RULES_SHA256,
    RULES_VERSION,
    ExperimentStatus,
    ReplayManifest,
    state_sha256,
)
from .model import (
    FugitiveAction,
    IllegalActionError,
    Observation,
    Phase,
    Role,
)
from .observation_protocol import observation_to_canonical_data
from .reproducibility import (
    AgentDescriptor,
    AgentSpec,
    SEED_DERIVATION_VERSION,
    derive_seed as _derive_seed,
    derive_seed_bundle,
)


MODES = ("human_fugitive", "human_marshal", "spectate")
SPECTATOR_VIEWS = ("omniscient", "fugitive", "marshal", "public")
DEFAULT_AUTO_STEP_LIMIT = 10_000
DEFAULT_SESSION_IDLE_TTL_SECONDS = 30 * 60
DEFAULT_MAX_ACTIVE_SESSIONS = 128
MAX_REQUEST_BYTES = 1_000_000
MAX_SEED = (1 << 64) - 1
MAX_SAFE_JSON_INTEGER = (1 << 53) - 1
WEB_TRACE_SCHEMA_VERSION = 2

_LOGGER = logging.getLogger(__name__)

_FUGITIVE_LABELS = {
    "hierarchical-random": "Hierarchical Legal Random (HR-1)",
    "belief-informed-random": "Belief-Informed Random (BIR-1)",
}
_MARSHAL_LABELS = {
    "hierarchical-random": "Hierarchical Legal Random (HR-1)",
    "belief-informed-random": "Belief-Informed Random (BIR-1)",
    "route-count-random": "Route-Count Random (HR-1.1)",
    "constructive-belief-informed-random": (
        "Constructive Belief-Informed Random (BIR-2)"
    ),
    "mcmc-belief-informed-random": "MCMC Belief-Informed Random (BIR-3)",
}


class WebAPIError(Exception):
    """An expected client error with a stable JSON error code."""

    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


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


def _parse_seed(value: object) -> int | None:
    """Parse the Web seed protocol without accepting lossy JSON numbers."""

    if value is None:
        return None
    if _is_integer(value):
        if 0 <= value <= MAX_SAFE_JSON_INTEGER:
            return value
        raise WebAPIError(
            HTTPStatus.BAD_REQUEST,
            "invalid_seed",
            "numeric seeds must be JavaScript-safe integers; send larger seeds as decimal strings",
        )
    if isinstance(value, str):
        if (
            not value
            or len(value) > 20
            or not value.isascii()
            or not value.isdecimal()
            or (len(value) > 1 and value.startswith("0"))
        ):
            raise WebAPIError(
                HTTPStatus.BAD_REQUEST,
                "invalid_seed",
                "seed must be a canonical unsigned decimal string",
            )
        seed = int(value)
        if seed <= MAX_SEED:
            return seed
        raise WebAPIError(
            HTTPStatus.BAD_REQUEST,
            "invalid_seed",
            f"seed must be at most {MAX_SEED}",
        )
    raise WebAPIError(
        HTTPStatus.BAD_REQUEST,
        "invalid_seed",
        "seed must be a decimal string or a JavaScript-safe integer",
    )


def _role_for_phase(phase: Phase) -> Role | None:
    if phase in (
        Phase.FUGITIVE_OPENING,
        Phase.FUGITIVE_DRAW,
        Phase.FUGITIVE_ACTION,
    ):
        return Role.FUGITIVE
    if phase in (Phase.MARSHAL_DRAW, Phase.MARSHAL_GUESS, Phase.MANHUNT):
        return Role.MARSHAL
    return None


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
        parsed_seed = _parse_seed(seed)
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
    def decision_trace(self) -> tuple[TraceRecord, ...]:
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
        built_fugitive = FUGITIVE_AGENT_REGISTRY[self.fugitive_agent_name].build(
            self._seed_bundle.fugitive,
            profile="interactive",
        )
        built_marshal = MARSHAL_AGENT_REGISTRY[self.marshal_agent_name].build(
            self._seed_bundle.marshal,
            profile="interactive",
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
        self._decision_count = 0
        self._decision_trace: list[TraceRecord] = []
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
            "parameters": dict(spec.parameters),
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

    def reset(self, *, seed: int | str | None = None) -> dict[str, object]:
        """Reset this session; omission intentionally deals a fresh game."""

        parsed_seed = _parse_seed(seed)
        with self._lock:
            self._ensure_open()
            self._release_agents()
            self._stalled = False
            self._pause_reason = None
            self._terminated = False
            self._termination_reason = None
            self._seed_was_supplied = parsed_seed is not None
            self.seed = parsed_seed if parsed_seed is not None else secrets.randbits(64)
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

    def _advance_until_human(self, max_steps: int) -> tuple[int, bool]:
        steps = 0
        while not self.engine.is_terminal:
            actor = _role_for_phase(self.engine.phase)
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
            actor = _role_for_phase(self.engine.phase)
            if actor is not self.human_role:
                raise WebAPIError(
                    HTTPStatus.CONFLICT,
                    "not_human_turn",
                    "the session is currently waiting for an agent",
                )
            phase = self.engine.phase
            round_number = self.engine.observation(self.human_role).round_number
            action_type = payload.get("type")
            try:
                if action_type == "draw":
                    pile = payload.get("pile")
                    if not _is_integer(pile):
                        raise IllegalActionError("pile must be an integer")
                    card = self.engine.draw(pile)
                    self._decision_trace.append(
                        DrawTrace(
                            self._decision_count + 1,
                            round_number,
                            phase,
                            actor,
                            pile,
                            card,
                        )
                    )
                    self._record_draw(
                        self.human_role,
                        pile,
                        card,
                        phase=phase,
                        round_number=round_number,
                    )
                elif action_type in ("fugitive_action", "play"):
                    if self.human_role is not Role.FUGITIVE:
                        raise IllegalActionError("only the Fugitive can establish a Hideout")
                    hideout = payload.get("hideout")
                    if not _is_integer(hideout):
                        raise IllegalActionError("hideout must be an integer")
                    sprint_cards = _integer_tuple(payload.get("sprint_cards", []), "sprint_cards")
                    action = FugitiveAction(hideout, sprint_cards)
                    self.engine.apply_fugitive_action(action)
                    self._decision_trace.append(
                        FugitiveTrace(
                            self._decision_count + 1,
                            round_number,
                            phase,
                            Role.FUGITIVE,
                            action.hideout,
                            tuple(sorted(action.sprint_cards)),
                        )
                    )
                    self._record_fugitive_action(
                        action,
                        phase=phase,
                        round_number=round_number,
                    )
                elif action_type == "pass":
                    if self.human_role is not Role.FUGITIVE:
                        raise IllegalActionError("only the Fugitive can pass")
                    action = FugitiveAction(None)
                    self.engine.apply_fugitive_action(action)
                    self._decision_trace.append(
                        FugitiveTrace(
                            self._decision_count + 1,
                            round_number,
                            phase,
                            Role.FUGITIVE,
                            None,
                            (),
                        )
                    )
                    self._record_fugitive_action(
                        action,
                        phase=phase,
                        round_number=round_number,
                    )
                elif action_type == "guess":
                    if self.human_role is not Role.MARSHAL:
                        raise IllegalActionError("only the Marshal can guess")
                    numbers = _integer_tuple(payload.get("numbers"), "numbers")
                    success = self.engine.apply_guess(numbers)
                    self._decision_trace.append(
                        GuessTrace(
                            self._decision_count + 1,
                            round_number,
                            phase,
                            Role.MARSHAL,
                            tuple(sorted(numbers)),
                            success,
                        )
                    )
                    self._record_guess(
                        tuple(sorted(numbers)),
                        success,
                        phase=phase,
                        round_number=round_number,
                    )
                else:
                    raise IllegalActionError(
                        "type must be draw, fugitive_action, pass, or guess"
                    )
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
            actor = _role_for_phase(self.engine.phase)
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
        if _role_for_phase(self.engine.phase) is not self.human_role:
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
                    status=ExperimentStatus.COMPLETED,
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
                "agents": {
                    "fugitive": {
                        "registry_name": self.fugitive_agent_name,
                        **self._agent_configurations[Role.FUGITIVE.value],
                    },
                    "marshal": {
                        "registry_name": self.marshal_agent_name,
                        **self._agent_configurations[Role.MARSHAL.value],
                    },
                },
                "outcome": {
                    "status": status,
                    "winner": winner,
                    "reason": reason,
                    "rounds": rounds,
                    "decision_count": self._decision_count,
                },
                "trace": json.loads(json.dumps(self._events)),
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
            actor = _role_for_phase(self.engine.phase)
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
                actor = _role_for_phase(self.engine.phase)
                status = "awaiting_human" if actor is self.human_role else "running"
            return {
                "id": self.id,
                "mode": self.mode,
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
                    _role_for_phase(observation.phase).value
                    if _role_for_phase(observation.phase) is not None
                    else None
                ),
                "fugitive_agent": self.fugitive_agent_name,
                "marshal_agent": self.marshal_agent_name,
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
    are not represented as completed experiment manifests.
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


@dataclass(slots=True)
class _SessionEntry:
    session: GameSession
    last_access: float


class SessionStore:
    """Thread-safe in-memory owner of active local game sessions."""

    def __init__(
        self,
        *,
        idle_ttl_seconds: float | None = DEFAULT_SESSION_IDLE_TTL_SECONDS,
        max_active_sessions: int = DEFAULT_MAX_ACTIVE_SESSIONS,
        auto_step_limit: int = DEFAULT_AUTO_STEP_LIMIT,
        clock: Callable[[], float] = time.monotonic,
        session_factory: Callable[..., GameSession] = GameSession,
    ) -> None:
        if idle_ttl_seconds is not None and (
            not isinstance(idle_ttl_seconds, (int, float))
            or isinstance(idle_ttl_seconds, bool)
            or not math.isfinite(idle_ttl_seconds)
            or idle_ttl_seconds <= 0
        ):
            raise ValueError("idle_ttl_seconds must be positive or None")
        if (
            not _is_integer(max_active_sessions)
            or max_active_sessions < 1
        ):
            raise ValueError("max_active_sessions must be a positive integer")
        if (
            not _is_integer(auto_step_limit)
            or not 1 <= auto_step_limit <= DEFAULT_AUTO_STEP_LIMIT
        ):
            raise ValueError(
                f"auto_step_limit must be from 1 through {DEFAULT_AUTO_STEP_LIMIT}"
            )
        self.idle_ttl_seconds = idle_ttl_seconds
        self.max_active_sessions = max_active_sessions
        self.auto_step_limit = auto_step_limit
        self._clock = clock
        self._session_factory = session_factory
        self._sessions: OrderedDict[str, _SessionEntry] = OrderedDict()
        self._lock = threading.RLock()

    def _remove_expired_locked(self, now: float) -> list[GameSession]:
        if self.idle_ttl_seconds is None:
            return []
        expired_ids = [
            session_id
            for session_id, entry in self._sessions.items()
            if now - entry.last_access >= self.idle_ttl_seconds
        ]
        return [self._sessions.pop(session_id).session for session_id in expired_ids]

    @staticmethod
    def _close_sessions(sessions: list[GameSession]) -> None:
        for session in sessions:
            session.close()

    def cleanup_expired(self) -> int:
        """Release expired sessions and return the number removed."""

        with self._lock:
            expired = self._remove_expired_locked(self._clock())
        self._close_sessions(expired)
        return len(expired)

    @property
    def active_count(self) -> int:
        self.cleanup_expired()
        with self._lock:
            return len(self._sessions)

    def create(self, payload: Mapping[str, object]) -> GameSession:
        mode = payload.get("mode", "spectate")
        fugitive_agent = payload.get("fugitive_agent", DEFAULT_FUGITIVE_AGENT)
        marshal_agent = payload.get("marshal_agent", DEFAULT_MARSHAL_AGENT)
        seed = payload.get("seed")
        spectator_view = payload.get("spectator_view")
        if not isinstance(mode, str):
            raise WebAPIError(HTTPStatus.BAD_REQUEST, "invalid_mode", "mode must be a string")
        if not isinstance(fugitive_agent, str) or not isinstance(marshal_agent, str):
            raise WebAPIError(
                HTTPStatus.BAD_REQUEST,
                "invalid_agent",
                "agent identifiers must be strings",
            )
        session = self._session_factory(
            session_id=uuid.uuid4().hex,
            mode=mode,
            fugitive_agent=fugitive_agent,
            marshal_agent=marshal_agent,
            seed=seed,  # type: ignore[arg-type]
            spectator_view=spectator_view,  # type: ignore[arg-type]
            auto_step_limit=self.auto_step_limit,
        )
        to_close: list[GameSession]
        with self._lock:
            now = self._clock()
            to_close = self._remove_expired_locked(now)
            self._sessions[session.id] = _SessionEntry(session, now)
            self._sessions.move_to_end(session.id)
            while len(self._sessions) > self.max_active_sessions:
                _session_id, entry = self._sessions.popitem(last=False)
                to_close.append(entry.session)
        self._close_sessions(to_close)
        return session

    def get(self, session_id: str) -> GameSession:
        session: GameSession | None = None
        with self._lock:
            now = self._clock()
            expired = self._remove_expired_locked(now)
            entry = self._sessions.get(session_id)
            if entry is not None:
                entry.last_access = now
                self._sessions.move_to_end(session_id)
                session = entry.session
        self._close_sessions(expired)
        if session is None:
            raise WebAPIError(
                HTTPStatus.NOT_FOUND,
                "game_not_found",
                f"game {session_id!r} does not exist",
            )
        return session

    def delete(self, session_id: str) -> None:
        """Remove one session and promptly release its agent state."""

        deleted: GameSession | None = None
        with self._lock:
            expired = self._remove_expired_locked(self._clock())
            entry = self._sessions.pop(session_id, None)
            if entry is not None:
                deleted = entry.session
        self._close_sessions(expired)
        if deleted is None:
            raise WebAPIError(
                HTTPStatus.NOT_FOUND,
                "game_not_found",
                f"game {session_id!r} does not exist",
            )
        deleted.close()


def make_handler(
    store: SessionStore,
    *,
    static_dir: Path | None = None,
    error_logger: logging.Logger | None = None,
) -> type[BaseHTTPRequestHandler]:
    """Bind a store and static root to a request-handler class."""

    resolved_static = static_dir.resolve() if static_dir is not None else None
    logger = error_logger if error_logger is not None else _LOGGER

    class FugitiveRequestHandler(BaseHTTPRequestHandler):
        server_version = "FugitiveLocal/1.0"

        def do_OPTIONS(self) -> None:  # noqa: N802
            self.send_response(HTTPStatus.NO_CONTENT)
            self._common_headers()
            self.end_headers()

        def do_GET(self) -> None:  # noqa: N802
            try:
                path = urlsplit(self.path).path
                if path == "/api/agents":
                    self._send_json(HTTPStatus.OK, agent_catalog())
                    return
                parts = _api_game_parts(path)
                if parts is not None and len(parts) == 1:
                    self._send_json(HTTPStatus.OK, store.get(parts[0]).as_dict())
                    return
                if (
                    parts is not None
                    and len(parts) == 2
                    and parts[1] == "export"
                ):
                    self._send_json(
                        HTTPStatus.OK,
                        store.get(parts[0]).export_trace(),
                    )
                    return
                if path.startswith("/api/"):
                    raise WebAPIError(
                        HTTPStatus.NOT_FOUND, "endpoint_not_found", "API endpoint not found"
                    )
                self._serve_static(path)
            except WebAPIError as exc:
                self._send_api_error(exc)
            except Exception as exc:
                self._send_internal_error(exc)

        def do_POST(self) -> None:  # noqa: N802
            try:
                payload = self._read_json_object()
                path = urlsplit(self.path).path
                if path == "/api/games":
                    session = store.create(payload)
                    self._send_json(HTTPStatus.CREATED, session.as_dict())
                    return
                parts = _api_game_parts(path)
                if parts is None or len(parts) != 2:
                    raise WebAPIError(
                        HTTPStatus.NOT_FOUND, "endpoint_not_found", "API endpoint not found"
                    )
                session = store.get(parts[0])
                operation = parts[1]
                if operation == "action":
                    state = session.apply_human_action(payload)
                elif operation == "step":
                    state = session.step()
                elif operation == "auto":
                    max_steps = payload.get("max_steps")
                    state = session.auto(max_steps=max_steps)  # type: ignore[arg-type]
                elif operation == "continue":
                    max_steps = payload.get("max_steps")
                    state = session.continue_after_stall(
                        max_steps=max_steps  # type: ignore[arg-type]
                    )
                elif operation == "terminate":
                    state = session.terminate()
                elif operation == "reset":
                    state = session.reset(seed=payload.get("seed"))  # type: ignore[arg-type]
                elif operation == "view":
                    state = session.set_spectator_view(
                        payload.get("spectator_view")
                    )
                else:
                    raise WebAPIError(
                        HTTPStatus.NOT_FOUND, "endpoint_not_found", "API endpoint not found"
                    )
                self._send_json(HTTPStatus.OK, state)
            except WebAPIError as exc:
                self._send_api_error(exc)
            except Exception as exc:
                self._send_internal_error(exc)

        def do_DELETE(self) -> None:  # noqa: N802
            try:
                path = urlsplit(self.path).path
                parts = _api_game_parts(path)
                if parts is None or len(parts) != 1:
                    raise WebAPIError(
                        HTTPStatus.NOT_FOUND,
                        "endpoint_not_found",
                        "API endpoint not found",
                    )
                session_id = parts[0]
                store.delete(session_id)
                self._send_json(
                    HTTPStatus.OK,
                    {"id": session_id, "deleted": True},
                )
            except WebAPIError as exc:
                self._send_api_error(exc)
            except Exception as exc:
                self._send_internal_error(exc)

        def _read_json_object(self) -> dict[str, object]:
            raw_length = self.headers.get("Content-Length", "0")
            try:
                length = int(raw_length)
            except ValueError as exc:
                raise WebAPIError(
                    HTTPStatus.BAD_REQUEST, "invalid_content_length", "invalid Content-Length"
                ) from exc
            if length < 0 or length > MAX_REQUEST_BYTES:
                raise WebAPIError(
                    HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                    "request_too_large",
                    "JSON request body is too large",
                )
            raw = self.rfile.read(length)
            if not raw:
                return {}
            try:
                value = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise WebAPIError(
                    HTTPStatus.BAD_REQUEST, "invalid_json", "request body must be valid JSON"
                ) from exc
            if not isinstance(value, dict):
                raise WebAPIError(
                    HTTPStatus.BAD_REQUEST, "invalid_json", "request JSON must be an object"
                )
            return value

        def _send_api_error(self, exc: WebAPIError) -> None:
            self._send_json(
                exc.status,
                {"error": {"code": exc.code, "message": exc.message}},
            )

        def _send_internal_error(self, exc: Exception) -> None:
            error_id = uuid.uuid4().hex
            logger.exception(
                "unhandled web request error error_id=%s method=%s path=%s",
                error_id,
                self.command,
                urlsplit(self.path).path,
                exc_info=exc,
            )
            self._send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {
                    "error": {
                        "code": "internal_server_error",
                        "message": "an unexpected server error occurred",
                        "error_id": error_id,
                    }
                },
            )

        def _send_json(self, status: int, value: object) -> None:
            body = json.dumps(value, separators=(",", ":"), ensure_ascii=True).encode(
                "utf-8"
            )
            self.send_response(status)
            self._common_headers()
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _serve_static(self, request_path: str) -> None:
            if resolved_static is None or not resolved_static.is_dir():
                raise WebAPIError(
                    HTTPStatus.NOT_FOUND, "static_not_found", "web interface is not installed"
                )
            relative = "index.html" if request_path == "/" else unquote(request_path.lstrip("/"))
            candidate = (resolved_static / relative).resolve()
            try:
                candidate.relative_to(resolved_static)
            except ValueError as exc:
                raise WebAPIError(
                    HTTPStatus.NOT_FOUND, "static_not_found", "static file not found"
                ) from exc
            if not candidate.is_file():
                raise WebAPIError(
                    HTTPStatus.NOT_FOUND, "static_not_found", "static file not found"
                )
            content_types = {
                ".html": "text/html; charset=utf-8",
                ".css": "text/css; charset=utf-8",
                ".js": "text/javascript; charset=utf-8",
                ".json": "application/json; charset=utf-8",
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".svg": "image/svg+xml",
                ".ico": "image/x-icon",
            }
            body = candidate.read_bytes()
            self.send_response(HTTPStatus.OK)
            self._common_headers()
            self.send_header(
                "Content-Type", content_types.get(candidate.suffix.lower(), "application/octet-stream")
            )
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(body)

        def _common_headers(self) -> None:
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Cross-Origin-Resource-Policy", "same-origin")

        def log_message(self, format: str, *args: object) -> None:
            # Keep the normal useful server log while avoiding user-controlled
            # terminal formatting from raw request paths.
            super().log_message(format, *args)

    return FugitiveRequestHandler


def _api_game_parts(path: str) -> tuple[str, ...] | None:
    prefix = "/api/games/"
    if not path.startswith(prefix):
        return None
    suffix = path[len(prefix) :].strip("/")
    if not suffix:
        return ()
    return tuple(suffix.split("/"))


def create_server(
    host: str = "127.0.0.1",
    port: int = 8000,
    *,
    store: SessionStore | None = None,
    static_dir: Path | None = None,
    error_logger: logging.Logger | None = None,
) -> ThreadingHTTPServer:
    """Create, but do not start, a local threaded server."""

    if static_dir is None:
        static_dir = Path(__file__).resolve().with_name("web_static")
    actual_store = store if store is not None else SessionStore()
    return ThreadingHTTPServer(
        (host, port),
        make_handler(
            actual_store,
            static_dir=static_dir,
            error_logger=error_logger,
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the local Fugitive web interface")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--static-dir",
        type=Path,
        default=Path(__file__).resolve().with_name("web_static"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    server = create_server(args.host, args.port, static_dir=args.static_dir)
    host, port = server.server_address[:2]
    print(f"Fugitive web interface: http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through CLI use
    raise SystemExit(main())


__all__ = [
    "GameSession",
    "SessionStore",
    "WebAPIError",
    "agent_catalog",
    "build_parser",
    "create_server",
    "main",
    "make_handler",
    "replay_manifest_from_web_trace",
]
