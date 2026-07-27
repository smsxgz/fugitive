"""Public data model for the Fugitive rules engine.

The engine deliberately passes immutable observations to agents.  None of the
classes in this module contains the shuffled deck order or an opponent's
private cards.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, TypeAlias


class Role(str, Enum):
    FUGITIVE = "fugitive"
    MARSHAL = "marshal"


class Phase(str, Enum):
    FUGITIVE_OPENING = "fugitive_opening"
    FUGITIVE_DRAW = "fugitive_draw"
    FUGITIVE_ACTION = "fugitive_action"
    MARSHAL_DRAW = "marshal_draw"
    MARSHAL_GUESS = "marshal_guess"
    MANHUNT = "manhunt"
    TERMINAL = "terminal"


class Winner(str, Enum):
    FUGITIVE = "fugitive"
    MARSHAL = "marshal"


class IllegalActionError(ValueError):
    """Raised when an agent returns an action that is illegal in the state."""


@dataclass(frozen=True, slots=True)
class FugitiveAction:
    """A Hideout play, or a pass when ``hideout`` is ``None``."""

    hideout: int | None
    sprint_cards: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class RouteView:
    """One route position as visible to the observing player.

    Index zero is the public starting Hideout 0.  ``hideout`` is ``None`` for
    a Hideout hidden from the observer.  Sprint count is always public, while
    ``sprint_cards`` is ``None`` until their identities are visible.
    ``revealed`` records whether the Marshal has uncovered this route card;
    this is distinct from visibility because Hideout 42 is public when played
    without being uncovered.
    """

    index: int
    hideout: int | None
    sprint_count: int
    sprint_cards: tuple[int, ...] | None
    revealed: bool = False


@dataclass(frozen=True, slots=True)
class GuessRecord:
    """Public result of one Marshal guess action.

    ``route_length`` is the number of Fugitive Hideouts at the time of the
    guess, excluding the starting Hideout 0.  It is therefore
    ``len(observation.route) - 1`` at that moment.
    """

    numbers: tuple[int, ...]
    success: bool
    route_length: int
    round_number: int
    manhunt: bool = False


@dataclass(frozen=True, slots=True)
class DrawRecord:
    """A draw event.

    The selected pile is public.  In an :class:`Observation`, ``card`` is set
    only for draws made by the observing role and is ``None`` for opponent
    draws.  A completed :class:`GameResult` contains the actual card.
    """

    role: Role
    pile: int
    card: int | None
    round_number: int


@dataclass(frozen=True, slots=True)
class PlayRecord:
    """A complete post-game record of a Fugitive play or pass."""

    hideout: int | None
    sprint_cards: tuple[int, ...]
    route_index: int | None
    round_number: int


@dataclass(frozen=True, slots=True)
class PlayView:
    """A role-sanitized Fugitive placement or Pass event.

    ``passed`` distinguishes a Pass from a face-down placement, since both
    have ``hideout=None`` in a Marshal observation.  Sprint count and timing
    are public; card identities follow the same visibility rules as
    :class:`RouteView`.
    """

    hideout: int | None
    sprint_count: int
    sprint_cards: tuple[int, ...] | None
    route_index: int | None
    round_number: int
    passed: bool


@dataclass(frozen=True, slots=True)
class Observation:
    """The complete information state supplied to one agent."""

    role: Role
    hand: tuple[int, ...]
    pile_sizes: tuple[int, int, int]
    route: tuple[RouteView, ...]
    guess_history: tuple[GuessRecord, ...]
    draw_history: tuple[DrawRecord, ...]
    round_number: int
    phase: Phase
    legal_draw_piles: tuple[int, ...]
    play_history: tuple[PlayView, ...] = ()


HistoryRecord: TypeAlias = DrawRecord | PlayRecord | GuessRecord


@dataclass(frozen=True, slots=True)
class GameResult:
    """The result of a game that ended with a rules-defined winner."""

    winner: Winner
    reason: str
    rounds: int
    history: tuple[HistoryRecord, ...]


class FugitiveAgent(Protocol):
    def choose_draw_pile(self, observation: Observation) -> int:
        ...

    def choose_fugitive_action(
        self, observation: Observation
    ) -> FugitiveAction:
        ...


class MarshalAgent(Protocol):
    def choose_draw_pile(self, observation: Observation) -> int:
        ...

    def choose_guess(self, observation: Observation) -> tuple[int, ...]:
        ...


__all__ = [
    "DrawRecord",
    "FugitiveAction",
    "FugitiveAgent",
    "GameResult",
    "GuessRecord",
    "HistoryRecord",
    "IllegalActionError",
    "MarshalAgent",
    "Observation",
    "Phase",
    "PlayRecord",
    "PlayView",
    "Role",
    "RouteView",
    "Winner",
]
