"""Web sessions backed by the C++ Fugitive game registered in OpenSpiel.

OpenSpiel owns all rules and state transitions.  This module only translates
the browser's macro actions into the game's small primitive action space,
samples explicit chance nodes, and redacts observations for the selected view.
"""

from __future__ import annotations

import copy
from http import HTTPStatus
import json
import random
import secrets
import threading
from typing import Any, Mapping

import pyspiel


MODES = ("human_fugitive", "human_marshal", "spectate")
SPECTATOR_VIEWS = ("omniscient", "fugitive", "marshal", "public")
DEFAULT_FUGITIVE_AGENT = "random"
DEFAULT_MARSHAL_AGENT = "random"
DEFAULT_AUTO_STEP_LIMIT = 10_000
MAX_SAFE_JSON_INTEGER = (1 << 53) - 1
MAX_UINT64 = (1 << 64) - 1
WEB_TRACE_SCHEMA_VERSION = 1

_FUGITIVE = 0
_MARSHAL = 1
_CHANCE = -1
_PASS = 0
_ESCAPE = 42
_FIRST_PILE_ACTION = 43
_LAST_PILE_ACTION = 45
_COMMIT = 46
_ROLE_BY_PLAYER = {_FUGITIVE: "fugitive", _MARSHAL: "marshal"}


class WebAPIError(Exception):
    """An expected client error with a stable JSON error code."""

    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


def _is_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def parse_web_seed(value: object) -> int | None:
    """Parse a uint64 seed while rejecting lossy JSON numeric values."""

    if value is None:
        return None
    if _is_integer(value):
        if not 0 <= value <= MAX_SAFE_JSON_INTEGER:
            raise WebAPIError(
                HTTPStatus.BAD_REQUEST,
                "invalid_seed",
                "numeric seeds must be JavaScript-safe integers; "
                "send larger seeds as decimal strings",
            )
        return value
    if isinstance(value, str) and (
        value == "0" or (value.isascii() and value.isdigit() and not value.startswith("0"))
    ):
        parsed = int(value)
        if parsed <= MAX_UINT64:
            return parsed
    raise WebAPIError(
        HTTPStatus.BAD_REQUEST,
        "invalid_seed",
        "seed must be a canonical uint64 decimal string or a JavaScript-safe integer",
    )


def agent_catalog() -> dict[str, object]:
    """Return the single built-in policy exposed by the compatibility UI."""

    def entry(role: str) -> dict[str, object]:
        return {
            "id": "random",
            "name": "OpenSpiel Uniform Random",
            "label": "OpenSpiel Uniform Random",
            "role": role,
            "expensive": False,
        }

    return {
        "defaults": {"fugitive": "random", "marshal": "random"},
        "fugitive": [entry("fugitive")],
        "marshal": [entry("marshal")],
        "spectator_views": [
            {"id": "omniscient", "label": "Omniscient"},
            {"id": "fugitive", "label": "Fugitive view"},
            {"id": "marshal", "label": "Marshal view"},
            {"id": "public", "label": "Public view"},
        ],
    }


def _integer_list(value: object, name: str) -> list[int]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a JSON array")
    if any(not _is_integer(item) for item in value):
        raise ValueError(f"every item in {name} must be an integer")
    result = list(value)
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicates")
    return result


class GameSession:
    """One isolated OpenSpiel state and its deterministic random streams."""

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
        if fugitive_agent != "random":
            raise WebAPIError(
                HTTPStatus.BAD_REQUEST,
                "unknown_fugitive_agent",
                f"unknown Fugitive agent {fugitive_agent!r}",
            )
        if marshal_agent != "random":
            raise WebAPIError(
                HTTPStatus.BAD_REQUEST,
                "unknown_marshal_agent",
                f"unknown Marshal agent {marshal_agent!r}",
            )
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
        if not _is_integer(auto_step_limit) or not 1 <= auto_step_limit <= DEFAULT_AUTO_STEP_LIMIT:
            raise ValueError(
                f"auto_step_limit must be from 1 through {DEFAULT_AUTO_STEP_LIMIT}"
            )

        parsed_seed = parse_web_seed(seed)
        self.id = session_id
        self.mode = mode
        self.fugitive_agent_name = fugitive_agent
        self.marshal_agent_name = marshal_agent
        self.spectator_view = (
            (spectator_view or "omniscient") if mode == "spectate" else None
        )
        self.auto_step_limit = auto_step_limit
        self._seed_was_supplied = parsed_seed is not None
        self.seed = parsed_seed if parsed_seed is not None else secrets.randbits(64)
        self._lock = threading.RLock()
        self._closed = False
        self._terminated = False
        self._termination_reason: str | None = None
        self._stalled = False
        self._pause_reason: str | None = None
        self._build_game()
        if self.human_player is not None:
            self._advance_until_human(self.auto_step_limit)

    @property
    def human_player(self) -> int | None:
        if self.mode == "human_fugitive":
            return _FUGITIVE
        if self.mode == "human_marshal":
            return _MARSHAL
        return None

    @property
    def human_role(self) -> str | None:
        player = self.human_player
        return _ROLE_BY_PLAYER.get(player) if player is not None else None

    @property
    def decision_trace(self) -> tuple[int, ...]:
        """Primitive OpenSpiel action history retained for compatibility."""

        with self._lock:
            return tuple(int(action) for action in self.state.history())

    def _build_game(self) -> None:
        self.game = pyspiel.load_game("fugitive")
        self.state = self.game.new_initial_state()
        self._chance_rng = random.Random(self.seed)
        self._bots = {
            _FUGITIVE: pyspiel.make_uniform_random_bot(
                _FUGITIVE, (self.seed ^ 0x46554749) & 0x7FFFFFFF
            ),
            _MARSHAL: pyspiel.make_uniform_random_bot(
                _MARSHAL, (self.seed ^ 0x4D415253) & 0x7FFFFFFF
            ),
        }
        self._observation_cache: dict[int, dict[str, Any]] = {}
        self._events: list[dict[str, object]] = []
        self._decision_count = 0
        self._known_history_lengths = {"draw": 0, "play": 0, "guess": 0}
        self._resolve_chance_nodes()

    def _ensure_open(self) -> None:
        if self._closed:
            raise WebAPIError(
                HTTPStatus.GONE, "session_closed", "the game session has been released"
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

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._bots = {}
            self._stalled = False
            self._pause_reason = None
            self._closed = True

    def reset(
        self,
        *,
        seed: int | str | None = None,
    ) -> dict[str, object]:
        parsed_seed = parse_web_seed(seed)
        with self._lock:
            self._ensure_open()
            self._seed_was_supplied = parsed_seed is not None
            self.seed = parsed_seed if parsed_seed is not None else secrets.randbits(64)
            self._terminated = False
            self._termination_reason = None
            self._stalled = False
            self._pause_reason = None
            self._build_game()
            if self.human_player is not None:
                self._advance_until_human(self.auto_step_limit)
            return self.as_dict()

    def set_spectator_view(self, spectator_view: object) -> dict[str, object]:
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

    def _observation(self, player: int) -> dict[str, Any]:
        cached = self._observation_cache.get(player)
        if cached is not None:
            return cached
        try:
            envelope = json.loads(self.state.observation_string(player))
            observation = envelope["observation"]
        except (TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
            raise RuntimeError("fugitive returned an invalid observation payload") from exc
        if not isinstance(observation, dict):
            raise RuntimeError("fugitive observation payload must be a JSON object")
        self._observation_cache[player] = observation
        return observation

    def _observations(self) -> tuple[dict[str, Any], dict[str, Any]]:
        return self._observation(_FUGITIVE), self._observation(_MARSHAL)

    def _invalidate_observations(self) -> None:
        self._observation_cache.clear()

    def _sample_chance_action(self) -> int:
        outcomes = [
            (int(action), float(probability))
            for action, probability in self.state.chance_outcomes()
        ]
        if not outcomes:
            raise RuntimeError("chance node has no outcomes")
        threshold = self._chance_rng.random()
        cumulative = 0.0
        for action, probability in outcomes:
            cumulative += probability
            if threshold < cumulative:
                return action
        return outcomes[-1][0]

    def _resolve_chance_nodes(self) -> None:
        changed = False
        while not self.state.is_terminal() and int(self.state.current_player()) == _CHANCE:
            self.state.apply_action(self._sample_chance_action())
            self._invalidate_observations()
            changed = True
        if changed:
            self._sync_events()

    def _sync_events(self) -> None:
        fugitive, marshal = self._observations()
        histories = {
            "draw": fugitive.get("draw_history", []),
            "play": fugitive.get("play_history", []),
            "guess": marshal.get("guess_history", []),
        }
        for kind, records in histories.items():
            if not isinstance(records, list):
                raise RuntimeError(f"{kind}_history must be an array")
            start = self._known_history_lengths[kind]
            for history_index, raw_record in enumerate(records[start:], start=start):
                if not isinstance(raw_record, dict):
                    raise RuntimeError(f"{kind}_history record must be an object")
                self._decision_count += 1
                record = copy.deepcopy(raw_record)
                round_number = record.get("round_number")
                if kind == "draw":
                    role = str(record.get("role"))
                    owner_history = (
                        fugitive.get("draw_history", [])
                        if role == "fugitive"
                        else marshal.get("draw_history", [])
                    )
                    # Histories have identical ordering; read this record by index
                    # from the drawing player's observation to recover its card.
                    if isinstance(owner_history, list) and history_index < len(owner_history):
                        visible = owner_history[history_index]
                        if isinstance(visible, dict):
                            record["card"] = visible.get("card")
                    event = {
                        "decision": self._decision_count,
                        "type": "draw",
                        "phase": "setup" if round_number == 0 else f"{role}_draw",
                        **record,
                    }
                elif kind == "play":
                    passed = bool(record.pop("passed", False))
                    event = {
                        "decision": self._decision_count,
                        "type": "pass" if passed else "fugitive_action",
                        "phase": "fugitive_action",
                        "role": "fugitive",
                        **record,
                    }
                else:
                    event = {
                        "decision": self._decision_count,
                        "type": "guess",
                        "phase": "manhunt" if record.get("manhunt") else "marshal_guess",
                        "role": "marshal",
                        **record,
                    }
                self._events.append(event)
            self._known_history_lengths[kind] = len(records)

    def _bot_macro_step(self) -> None:
        """Advance one visible decision, consuming its primitive sub-actions."""

        self._resolve_chance_nodes()
        if self.state.is_terminal():
            return
        before = self._decision_count
        for _ in range(100):
            player = int(self.state.current_player())
            if player == _CHANCE:
                self._resolve_chance_nodes()
            elif player in self._bots:
                action = int(self._bots[player].step(self.state))
                if action not in [int(value) for value in self.state.legal_actions()]:
                    raise RuntimeError("OpenSpiel random bot returned an illegal action")
                self.state.apply_action(action)
                self._invalidate_observations()
                if action in (_PASS, _COMMIT):
                    self._sync_events()
            else:
                raise RuntimeError(f"unexpected OpenSpiel player id {player}")
            if self.state.is_terminal() or self._decision_count != before:
                return
        raise RuntimeError("a Fugitive macro action exceeded 100 primitive actions")

    def _advance_until_human(self, max_steps: int) -> tuple[int, bool]:
        steps = 0
        self._resolve_chance_nodes()
        while not self.state.is_terminal():
            if int(self.state.current_player()) == self.human_player:
                self._stalled = False
                self._pause_reason = None
                return steps, False
            if steps >= max_steps:
                self._stalled = True
                self._pause_reason = "auto_step_limit"
                return steps, True
            self._bot_macro_step()
            steps += 1
        self._stalled = False
        self._pause_reason = None
        return steps, False

    def _validate_max_steps(self, max_steps: object) -> int:
        if not _is_integer(max_steps) or not 1 <= max_steps <= self.auto_step_limit:
            raise WebAPIError(
                HTTPStatus.BAD_REQUEST,
                "invalid_max_steps",
                f"max_steps must be from 1 through {self.auto_step_limit}",
            )
        return max_steps

    def step(self) -> dict[str, object]:
        with self._lock:
            self._ensure_playable()
            if self.mode != "spectate":
                raise WebAPIError(
                    HTTPStatus.CONFLICT,
                    "step_not_available",
                    "step is available only for agent-v-agent spectator games",
                )
            if self.state.is_terminal():
                raise WebAPIError(
                    HTTPStatus.CONFLICT, "game_finished", "the game has already finished"
                )
            self._bot_macro_step()
            return self.as_dict()

    def auto(self, *, max_steps: int | None = None) -> dict[str, object]:
        checked = self._validate_max_steps(
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
            steps, paused = self._advance_until_human(checked)
            result = self.as_dict()
            result["auto_steps"] = steps
            result["auto_paused"] = paused
            return result

    def continue_after_stall(self, *, max_steps: int | None = None) -> dict[str, object]:
        checked = self._validate_max_steps(
            self.auto_step_limit if max_steps is None else max_steps
        )
        with self._lock:
            self._ensure_open()
            if not self._stalled or self._terminated or self.state.is_terminal():
                raise WebAPIError(
                    HTTPStatus.CONFLICT,
                    "continue_not_available",
                    "the session is not paused at its automatic-step limit",
                )
            self._stalled = False
            self._pause_reason = None
            steps, paused = self._advance_until_human(checked)
            result = self.as_dict()
            result["auto_steps"] = steps
            result["auto_paused"] = paused
            return result

    def terminate(self) -> dict[str, object]:
        with self._lock:
            self._ensure_open()
            if self.state.is_terminal():
                raise WebAPIError(
                    HTTPStatus.CONFLICT, "game_finished", "the game has already finished"
                )
            self._terminated = True
            self._termination_reason = "terminated_by_user"
            self._stalled = False
            self._pause_reason = None
            return self.as_dict()

    @staticmethod
    def _apply_checked(state: Any, actions: list[int]) -> None:
        for action in actions:
            legal = [int(value) for value in state.legal_actions()]
            if action not in legal:
                raise ValueError(f"primitive action {action} is not legal in this state")
            state.apply_action(action)

    @staticmethod
    def _fugitive_action_prefix(payload: Mapping[str, object]) -> list[int]:
        hideout = payload.get("hideout")
        if not _is_integer(hideout):
            raise ValueError("hideout must be an integer")
        sprints = sorted(
            _integer_list(payload.get("sprint_cards", []), "sprint_cards")
        )
        return [hideout, *sprints]

    def _require_human_turn(self) -> int:
        self._ensure_playable()
        player = self.human_player
        if player is None:
            raise WebAPIError(
                HTTPStatus.CONFLICT,
                "human_action_not_available",
                "spectator games do not accept human actions",
            )
        if self.state.is_terminal():
            raise WebAPIError(
                HTTPStatus.CONFLICT, "game_finished", "the game has already finished"
            )
        if int(self.state.current_player()) != player:
            raise WebAPIError(
                HTTPStatus.CONFLICT,
                "not_human_turn",
                "the session is currently waiting for an agent",
            )
        return player

    def preview_fugitive_action(
        self, payload: Mapping[str, object]
    ) -> dict[str, bool]:
        """Check a browser draft against OpenSpiel without changing the game."""

        with self._lock:
            if self._require_human_turn() != _FUGITIVE:
                raise WebAPIError(
                    HTTPStatus.CONFLICT,
                    "fugitive_preview_not_available",
                    "only a human Fugitive can preview this action",
                )
            try:
                actions = self._fugitive_action_prefix(payload)
            except (TypeError, ValueError) as exc:
                raise WebAPIError(
                    HTTPStatus.UNPROCESSABLE_ENTITY, "illegal_action", str(exc)
                ) from exc

            trial = self.state.clone()
            try:
                self._apply_checked(trial, actions)
            except ValueError:
                return {"valid_prefix": False, "can_commit": False}
            legal = [int(value) for value in trial.legal_actions()]
            return {"valid_prefix": True, "can_commit": _COMMIT in legal}

    def apply_human_action(self, payload: Mapping[str, object]) -> dict[str, object]:
        """Apply one browser macro action atomically, then run the opponent."""

        with self._lock:
            player = self._require_human_turn()

            action_type = payload.get("type")
            try:
                if action_type == "draw":
                    pile = payload.get("pile")
                    if not _is_integer(pile) or not 0 <= pile <= 2:
                        raise ValueError("pile must be an integer from 0 through 2")
                    actions = [_FIRST_PILE_ACTION + pile]
                elif action_type in ("fugitive_action", "play"):
                    if player != _FUGITIVE:
                        raise ValueError("only the Fugitive can establish a Hideout")
                    actions = [*self._fugitive_action_prefix(payload), _COMMIT]
                elif action_type == "pass":
                    if player != _FUGITIVE:
                        raise ValueError("only the Fugitive can pass")
                    actions = [_PASS]
                elif action_type == "guess":
                    if player != _MARSHAL:
                        raise ValueError("only the Marshal can guess")
                    numbers = sorted(_integer_list(payload.get("numbers"), "numbers"))
                    if self._observation(_MARSHAL).get("phase") == "manhunt" and len(numbers) != 1:
                        raise ValueError("Manhunt accepts exactly one number")
                    actions = [*numbers, _COMMIT]
                else:
                    raise ValueError("type must be draw, fugitive_action, pass, or guess")

                trial = self.state.clone()
                self._apply_checked(trial, actions)
            except (TypeError, ValueError) as exc:
                raise WebAPIError(
                    HTTPStatus.UNPROCESSABLE_ENTITY, "illegal_action", str(exc)
                ) from exc

            self.state = trial
            self._invalidate_observations()
            self._sync_events()
            self._resolve_chance_nodes()
            _steps, paused = self._advance_until_human(self.auto_step_limit)
            result = self.as_dict()
            result["auto_paused"] = paused
            return result

    def _viewer_kind(self) -> str:
        if self.human_role is not None:
            return self.human_role
        return self.spectator_view or "public"

    def _legal_actions(self, observation: Mapping[str, Any]) -> dict[str, object]:
        result: dict[str, object] = {
            "draw_piles": [],
            "candidate_hideouts": [],
            "sprint_cards": [],
            "can_pass": False,
            "guess_numbers": [],
            "min_guess_count": 0,
            "max_guess_count": 0,
        }
        if (
            self.state.is_terminal()
            or self._stalled
            or self._terminated
            or self.human_player is None
            or int(self.state.current_player()) != self.human_player
        ):
            return result

        legal = [int(action) for action in self.state.legal_actions()]
        phase = observation.get("phase")
        if phase in ("fugitive_draw", "marshal_draw"):
            result["draw_piles"] = [
                action - _FIRST_PILE_ACTION
                for action in legal
                if _FIRST_PILE_ACTION <= action <= _LAST_PILE_ACTION
            ]
        elif phase in ("fugitive_opening", "fugitive_action"):
            result["candidate_hideouts"] = [
                action for action in legal if 1 <= action <= _ESCAPE
            ]
            result["sprint_cards"] = [
                card for card in observation.get("hand", []) if card != _ESCAPE
            ]
            result["can_pass"] = _PASS in legal
        elif phase in ("marshal_guess", "manhunt"):
            result["guess_numbers"] = [action for action in legal if 1 <= action <= 41]
            result["min_guess_count"] = 1
            result["max_guess_count"] = 1 if phase == "manhunt" else 41
        return result

    def _serialized_history(
        self, viewer_kind: str, marshal_observation: Mapping[str, Any]
    ) -> list[dict[str, object]]:
        public_route = {
            node.get("index"): node
            for node in marshal_observation.get("route", [])
            if isinstance(node, dict)
        }
        result: list[dict[str, object]] = []
        for source in self._events:
            item = copy.deepcopy(source)
            if item.get("type") == "draw":
                if viewer_kind != "omniscient" and item.get("role") != viewer_kind:
                    item["card"] = None
            elif item.get("type") == "fugitive_action" and viewer_kind not in (
                "omniscient",
                "fugitive",
            ):
                route_node = public_route.get(item.get("route_index"), {})
                item["hideout"] = route_node.get("hideout")
                item["sprint_cards"] = route_node.get("sprint_cards")
            item.pop("route_index", None)
            result.append(item)
        return result

    def _agent_data(self) -> dict[str, object]:
        value = {
            "registry_name": "random",
            "class": "open_spiel.UniformRandomBot",
            "name": "OpenSpiel Uniform Random",
            "parameters": {},
        }
        return {"fugitive": copy.deepcopy(value), "marshal": copy.deepcopy(value)}

    def export_trace(self) -> dict[str, object]:
        with self._lock:
            self._ensure_open()
            if not self.state.is_terminal() and not self._terminated:
                raise WebAPIError(
                    HTTPStatus.CONFLICT,
                    "export_not_ready",
                    "a full trace is available only after the game or session ends",
                )
            fugitive, marshal = self._observations()
            terminal = fugitive.get("terminal") if self.state.is_terminal() else None
            return {
                "schema": "fugitive.openspiel-web-trace",
                "schema_version": WEB_TRACE_SCHEMA_VERSION,
                "game": self.game.serialize(),
                "state": self.state.serialize(),
                "game_and_state": pyspiel.serialize_game_and_state(self.game, self.state),
                "action_history": [int(action) for action in self.state.history()],
                "seed": str(self.seed),
                "mode": self.mode,
                "agents": self._agent_data(),
                "outcome": {
                    "status": "finished" if self.state.is_terminal() else "terminated",
                    "winner": terminal.get("winner") if isinstance(terminal, dict) else None,
                    "reason": (
                        terminal.get("reason")
                        if isinstance(terminal, dict)
                        else self._termination_reason
                    ),
                    "rounds": fugitive.get("round_number"),
                    "decision_count": self._decision_count,
                },
                "trace": copy.deepcopy(self._events),
                "observations": {
                    "fugitive": copy.deepcopy(fugitive),
                    "marshal": copy.deepcopy(marshal),
                },
            }

    def as_dict(self) -> dict[str, object]:
        with self._lock:
            self._ensure_open()
            fugitive, marshal = self._observations()
            viewer_kind = self._viewer_kind()
            selected = fugitive if viewer_kind == "fugitive" else marshal
            route_source = fugitive if viewer_kind in ("omniscient", "fugitive") else marshal

            fugitive_hand = (
                list(fugitive.get("hand", []))
                if viewer_kind in ("omniscient", "fugitive")
                else None
            )
            marshal_hand = (
                list(marshal.get("hand", []))
                if viewer_kind in ("omniscient", "marshal")
                else None
            )
            actor = _ROLE_BY_PLAYER.get(int(self.state.current_player()))
            if viewer_kind == "omniscient":
                selected_hand = fugitive_hand if actor == "fugitive" else marshal_hand
            elif viewer_kind == "fugitive":
                selected_hand = fugitive_hand
            elif viewer_kind == "marshal":
                selected_hand = marshal_hand
            else:
                selected_hand = []

            terminal = selected.get("terminal")
            if self._terminated:
                status = "terminated"
                winner = None
                reason = self._termination_reason
            elif self.state.is_terminal():
                status = "finished"
                winner = terminal.get("winner") if isinstance(terminal, dict) else None
                reason = terminal.get("reason") if isinstance(terminal, dict) else None
            elif self._stalled:
                status = "stalled"
                winner = reason = None
            else:
                status = (
                    "awaiting_human"
                    if int(self.state.current_player()) == self.human_player
                    else "running"
                )
                winner = reason = None

            can_play = not self.state.is_terminal() and not self._terminated
            return {
                "id": self.id,
                "mode": self.mode,
                "status": status,
                "phase": selected.get("phase"),
                "round_number": selected.get("round_number"),
                "viewer_role": self.human_role or "spectator",
                "spectator_view": self.spectator_view,
                "available_spectator_views": list(SPECTATOR_VIEWS),
                "actor": actor,
                "fugitive_agent": self.fugitive_agent_name,
                "marshal_agent": self.marshal_agent_name,
                "agents": self._agent_data(),
                "pile_sizes": list(selected.get("pile_sizes", [])),
                "route": copy.deepcopy(route_source.get("route", [])),
                "hand": selected_hand or [],
                "hands": {"fugitive": fugitive_hand, "marshal": marshal_hand},
                "legal_actions": self._legal_actions(selected),
                "history": self._serialized_history(viewer_kind, marshal),
                "winner": winner,
                "reason": reason,
                "seed": str(self.seed) if self.state.is_terminal() or self._terminated else None,
                "seed_was_supplied": self._seed_was_supplied,
                "can_step": self.mode == "spectate" and can_play and not self._stalled,
                "can_auto": self.mode == "spectate" and can_play and not self._stalled,
                "can_continue": self._stalled and can_play,
                "can_terminate": can_play,
                "can_export": self.state.is_terminal() or self._terminated,
                "auto_paused": self._stalled,
                "pause_reason": self._pause_reason,
                "auto_running": False,
                "auto_delay_ms": 0,
            }


__all__ = [
    "DEFAULT_AUTO_STEP_LIMIT",
    "DEFAULT_FUGITIVE_AGENT",
    "DEFAULT_MARSHAL_AGENT",
    "GameSession",
    "MAX_SAFE_JSON_INTEGER",
    "WEB_TRACE_SCHEMA_VERSION",
    "WebAPIError",
    "agent_catalog",
    "parse_web_seed",
]
