"""Stateful engine and agent-vs-agent game runner for Fugitive."""

from __future__ import annotations

from dataclasses import dataclass, field
import random
from typing import Iterable

from .model import (
    DrawRecord,
    FugitiveAction,
    FugitiveAgent,
    GameResult,
    GuessRecord,
    HistoryRecord,
    IllegalActionError,
    MarshalAgent,
    Observation,
    Phase,
    PlayRecord,
    PlayView,
    Role,
    RouteView,
    Winner,
)
from .rules import (
    PILE_CARDS,
    has_legal_hideout_play,
    is_legal_fugitive_action,
    is_legal_guess,
    sprint_value,
)


@dataclass(slots=True)
class _RouteNode:
    hideout: int
    sprint_cards: tuple[int, ...]
    revealed: bool = False


@dataclass(slots=True)
class _GameState:
    piles: list[list[int]]
    hands: dict[Role, list[int]]
    route: list[_RouteNode]
    phase: Phase = Phase.FUGITIVE_OPENING
    round_number: int = 1
    opening_plays_remaining: int = 2
    draws_remaining: int = 0
    winner: Winner | None = None
    reason: str | None = None
    history: list[HistoryRecord] = field(default_factory=list)
    guess_history: list[GuessRecord] = field(default_factory=list)
    draw_history: list[DrawRecord] = field(default_factory=list)


class GameEngine:
    """Encapsulated, step-by-step Fugitive game state.

    Agents should receive only :meth:`observation`; the private ``_state`` is
    intentionally not part of the public API.  Search code can reproduce a
    trajectory by constructing another engine with the same seed and applying
    the same actions.
    """

    def __init__(
        self,
        *,
        seed: int | None = None,
    ) -> None:
        rng = random.Random(seed)
        piles = [list(cards) for cards in PILE_CARDS]
        for pile in piles:
            rng.shuffle(pile)

        self._state = _GameState(
            piles=piles,
            hands={Role.FUGITIVE: [1, 2, 3, 42], Role.MARSHAL: []},
            route=[_RouteNode(0, (), revealed=True)],
        )

        # Setup draws are private draws with fixed, commonly known pile choices.
        for _ in range(3):
            self._setup_draw(Role.FUGITIVE, 0)
        for _ in range(2):
            self._setup_draw(Role.FUGITIVE, 1)
        self._sort_hands()

    @property
    def phase(self) -> Phase:
        return self._state.phase

    @property
    def is_terminal(self) -> bool:
        return self._state.phase is Phase.TERMINAL

    def validate_invariants(self) -> None:
        """Raise ``AssertionError`` if the private game state is inconsistent.

        This debug/test hook deliberately exposes no private state.  Experiment
        runners can call it after every action to catch engine or agent-harness
        corruption close to its source.
        """

        state = self._state
        assert state.round_number >= 1
        assert state.route
        assert state.route[0] == _RouteNode(0, (), revealed=True)

        card_locations: list[int] = []
        previous_hideout = 0
        for node in state.route[1:]:
            assert previous_hideout < node.hideout <= 42
            assert node.hideout - previous_hideout <= 3 + sum(
                sprint_value(card) for card in node.sprint_cards
            )
            card_locations.append(node.hideout)
            card_locations.extend(node.sprint_cards)
            previous_hideout = node.hideout

        for pile_index, pile in enumerate(state.piles):
            assert all(card in PILE_CARDS[pile_index] for card in pile)
            card_locations.extend(pile)
        for hand in state.hands.values():
            assert hand == sorted(hand)
            card_locations.extend(hand)

        assert len(card_locations) == 42
        assert set(card_locations) == set(range(1, 43))

        if state.phase is Phase.FUGITIVE_OPENING:
            assert state.opening_plays_remaining in (1, 2)
            assert state.draws_remaining == 0
        elif state.phase is Phase.FUGITIVE_DRAW:
            assert state.opening_plays_remaining == 0
            assert state.draws_remaining == 1
            assert self._nonempty_piles()
        elif state.phase is Phase.MARSHAL_DRAW:
            assert state.opening_plays_remaining == 0
            assert state.draws_remaining in (1, 2)
            assert self._nonempty_piles()
        else:
            assert state.draws_remaining == 0

        if state.phase is Phase.TERMINAL:
            assert state.winner is not None
            assert state.reason is not None
        else:
            assert state.winner is None
            assert state.reason is None

    def observation(self, role: Role) -> Observation:
        """Return an immutable observation containing exactly ``role``'s view."""

        if not isinstance(role, Role):
            raise ValueError("role must be Role.FUGITIVE or Role.MARSHAL")

        route = tuple(
            self._route_view(node, index, role)
            for index, node in enumerate(self._state.route)
        )
        draw_history = tuple(
            DrawRecord(
                role=record.role,
                pile=record.pile,
                card=record.card if record.role is role else None,
                round_number=record.round_number,
            )
            for record in self._state.draw_history
        )
        play_history = tuple(
            self._play_view(record, route, role)
            for record in self._state.history
            if isinstance(record, PlayRecord)
        )
        legal_draw_piles: tuple[int, ...] = ()
        if self._role_to_act() is role and self.phase in (
            Phase.FUGITIVE_DRAW,
            Phase.MARSHAL_DRAW,
        ):
            legal_draw_piles = self._nonempty_piles()

        return Observation(
            role=role,
            hand=tuple(sorted(self._state.hands[role])),
            pile_sizes=tuple(len(pile) for pile in self._state.piles),  # type: ignore[arg-type]
            route=route,
            guess_history=tuple(self._state.guess_history),
            draw_history=draw_history,
            round_number=self._state.round_number,
            phase=self._state.phase,
            legal_draw_piles=legal_draw_piles,
            play_history=play_history,
        )

    def draw(self, pile: int) -> int:
        """Apply the current player's required draw and return their card."""

        if self.phase not in (Phase.FUGITIVE_DRAW, Phase.MARSHAL_DRAW):
            raise IllegalActionError(f"cannot draw during phase {self.phase.value}")
        if isinstance(pile, bool) or not isinstance(pile, int) or not 0 <= pile < 3:
            raise IllegalActionError("draw pile must be 0, 1, or 2")
        if not self._state.piles[pile]:
            raise IllegalActionError(f"draw pile {pile} is empty")

        role = self._role_to_act()
        assert role is not None
        card = self._state.piles[pile].pop()
        self._state.hands[role].append(card)
        self._state.hands[role].sort()
        record = DrawRecord(role, pile, card, self._state.round_number)
        self._state.draw_history.append(record)
        self._state.history.append(record)

        self._state.draws_remaining -= 1
        if self._state.draws_remaining <= 0:
            self._state.phase = (
                Phase.FUGITIVE_ACTION
                if role is Role.FUGITIVE
                else Phase.MARSHAL_GUESS
            )
        else:
            self._skip_draw_if_all_piles_empty()
        return card

    def apply_fugitive_action(self, action: FugitiveAction) -> None:
        """Apply a Fugitive Hideout play or pass."""

        if self.phase not in (Phase.FUGITIVE_OPENING, Phase.FUGITIVE_ACTION):
            raise IllegalActionError(
                f"cannot play a Fugitive action during phase {self.phase.value}"
            )
        if not isinstance(action, FugitiveAction):
            raise IllegalActionError("Fugitive agent must return FugitiveAction")

        try:
            sprint_cards = tuple(action.sprint_cards)
        except TypeError as exc:
            raise IllegalActionError("sprint_cards must be an iterable of cards") from exc
        normalized = FugitiveAction(action.hideout, sprint_cards)
        opening = self.phase is Phase.FUGITIVE_OPENING
        previous = self._state.route[-1].hideout
        if not is_legal_fugitive_action(
            normalized,
            self._state.hands[Role.FUGITIVE],
            previous,
            allow_pass=not opening,
        ):
            raise IllegalActionError(
                "illegal Fugitive action: pass is forbidden in the opening; "
                "otherwise check hand ownership, forward movement, and Sprint value"
            )
        if opening and self._state.opening_plays_remaining == 2:
            assert normalized.hideout is not None
            spent = {normalized.hideout, *normalized.sprint_cards}
            remaining = tuple(
                card
                for card in self._state.hands[Role.FUGITIVE]
                if card not in spent
            )
            if not has_legal_hideout_play(remaining, normalized.hideout):
                raise IllegalActionError(
                    "the first opening play must leave a legal second Hideout play"
                )

        # Sprint payment is a set in the rules. Store one canonical ordering so
        # equivalent agent actions do not create different public histories.
        normalized = FugitiveAction(
            normalized.hideout,
            tuple(sorted(normalized.sprint_cards)),
        )

        route_index: int | None = None
        if normalized.hideout is not None:
            used_cards = (normalized.hideout, *normalized.sprint_cards)
            for card in used_cards:
                self._state.hands[Role.FUGITIVE].remove(card)
            route_index = len(self._state.route)
            self._state.route.append(
                _RouteNode(normalized.hideout, normalized.sprint_cards)
            )
        record = PlayRecord(
            normalized.hideout,
            normalized.sprint_cards,
            route_index,
            self._state.round_number,
        )
        self._state.history.append(record)

        if normalized.hideout == 42:
            self._begin_escape_resolution()
            return

        if opening:
            self._state.opening_plays_remaining -= 1
            if self._state.opening_plays_remaining > 0:
                return
            self._start_marshal_turn(first_turn=True)
            return

        self._start_marshal_turn(first_turn=False)

    def apply_guess(self, numbers: Iterable[int]) -> bool:
        """Apply a normal guess or one Manhunt guess and return its success."""

        if self.phase not in (Phase.MARSHAL_GUESS, Phase.MANHUNT):
            raise IllegalActionError(f"cannot guess during phase {self.phase.value}")
        try:
            guess = tuple(numbers)
        except TypeError as exc:
            raise IllegalActionError("guess must be an iterable of card numbers") from exc
        manhunt = self.phase is Phase.MANHUNT
        if not is_legal_guess(guess, manhunt=manhunt):
            constraint = " exactly one" if manhunt else " one or more distinct"
            raise IllegalActionError(
                f"a Marshal guess requires{constraint} integer card number(s) from 1 to 41"
            )
        # A multi-number guess is a set, not an ordered sequence.
        guess = tuple(sorted(guess))

        hidden_by_number = {
            node.hideout: node
            for node in self._state.route[1:]
            if not node.revealed and node.hideout != 42
        }
        success = all(number in hidden_by_number for number in guess)
        if success:
            for number in guess:
                hidden_by_number[number].revealed = True

        record = GuessRecord(
            guess,
            success,
            len(self._state.route) - 1,
            self._state.round_number,
            manhunt,
        )
        self._state.guess_history.append(record)
        self._state.history.append(record)

        if manhunt:
            if not success:
                self._finish(Winner.FUGITIVE, "manhunt_failed")
            elif not self._unrevealed_route_hideouts():
                self._finish(Winner.MARSHAL, "manhunt_success")
            return success

        if success and self._all_current_hideouts_revealed():
            self._finish(Winner.MARSHAL, "marshal_uncovered_route")
        else:
            self._finish_round_or_start_next()
        return success

    def result(self) -> GameResult:
        """Return the rules-defined result, raising if the game is still running."""

        if not self.is_terminal:
            raise RuntimeError("the game has not ended")
        assert self._state.reason is not None
        assert self._state.winner is not None
        return GameResult(
            self._state.winner,
            self._state.reason,
            self._state.round_number,
            tuple(self._state.history),
        )

    def _setup_draw(self, role: Role, pile: int) -> None:
        card = self._state.piles[pile].pop()
        self._state.hands[role].append(card)
        record = DrawRecord(role, pile, card, 0)
        self._state.draw_history.append(record)
        self._state.history.append(record)

    def _route_view(self, node: _RouteNode, index: int, role: Role) -> RouteView:
        if role is Role.FUGITIVE:
            return RouteView(
                index=index,
                hideout=node.hideout,
                sprint_count=len(node.sprint_cards),
                sprint_cards=node.sprint_cards,
                revealed=node.revealed,
            )
        hideout_visible = node.revealed or node.hideout in (0, 42)
        # Hideout 42 is public when played, but its Sprint identities stay hidden.
        sprint_visible = node.revealed and node.hideout != 42
        return RouteView(
            index=index,
            hideout=node.hideout if hideout_visible else None,
            sprint_count=len(node.sprint_cards),
            sprint_cards=node.sprint_cards if sprint_visible else None,
            revealed=node.revealed,
        )

    @staticmethod
    def _play_view(
        record: PlayRecord,
        route: tuple[RouteView, ...],
        role: Role,
    ) -> PlayView:
        if record.route_index is None:
            return PlayView(
                hideout=None,
                sprint_count=0,
                sprint_cards=(),
                route_index=None,
                round_number=record.round_number,
                passed=True,
            )

        slot = route[record.route_index]
        if role is Role.FUGITIVE:
            hideout = record.hideout
            sprint_cards: tuple[int, ...] | None = record.sprint_cards
        else:
            hideout = slot.hideout
            sprint_cards = slot.sprint_cards
        return PlayView(
            hideout=hideout,
            sprint_count=len(record.sprint_cards),
            sprint_cards=sprint_cards,
            route_index=record.route_index,
            round_number=record.round_number,
            passed=False,
        )

    def _role_to_act(self) -> Role | None:
        if self.phase in (
            Phase.FUGITIVE_OPENING,
            Phase.FUGITIVE_DRAW,
            Phase.FUGITIVE_ACTION,
        ):
            return Role.FUGITIVE
        if self.phase in (
            Phase.MARSHAL_DRAW,
            Phase.MARSHAL_GUESS,
            Phase.MANHUNT,
        ):
            return Role.MARSHAL
        return None

    def _nonempty_piles(self) -> tuple[int, ...]:
        return tuple(index for index, pile in enumerate(self._state.piles) if pile)

    def _skip_draw_if_all_piles_empty(self) -> None:
        if self.phase not in (Phase.FUGITIVE_DRAW, Phase.MARSHAL_DRAW):
            return
        if self._nonempty_piles():
            return
        role = self._role_to_act()
        self._state.draws_remaining = 0
        self._state.phase = (
            Phase.FUGITIVE_ACTION
            if role is Role.FUGITIVE
            else Phase.MARSHAL_GUESS
        )

    def _start_marshal_turn(self, *, first_turn: bool) -> None:
        self._state.phase = Phase.MARSHAL_DRAW
        self._state.draws_remaining = 2 if first_turn else 1
        self._skip_draw_if_all_piles_empty()

    def _finish_round_or_start_next(self) -> None:
        self._state.round_number += 1
        self._state.phase = Phase.FUGITIVE_DRAW
        self._state.draws_remaining = 1
        self._skip_draw_if_all_piles_empty()

    def _begin_escape_resolution(self) -> None:
        highest_revealed = max(
            node.hideout
            for node in self._state.route
            if node.revealed and node.hideout != 42
        )
        if highest_revealed >= 30:
            self._finish(Winner.FUGITIVE, "escape_no_manhunt")
            return
        if not self._unrevealed_route_hideouts():
            self._finish(Winner.MARSHAL, "manhunt_success")
            return
        self._state.phase = Phase.MANHUNT
        self._state.draws_remaining = 0

    def _unrevealed_route_hideouts(self) -> tuple[int, ...]:
        return tuple(
            node.hideout
            for node in self._state.route[1:]
            if not node.revealed and node.hideout != 42
        )

    def _all_current_hideouts_revealed(self) -> bool:
        return bool(self._state.route[1:]) and all(
            node.revealed for node in self._state.route[1:]
        )

    def _finish(self, winner: Winner, reason: str) -> None:
        self._state.winner = winner
        self._state.reason = reason
        self._state.phase = Phase.TERMINAL
        self._state.draws_remaining = 0

    def _sort_hands(self) -> None:
        for hand in self._state.hands.values():
            hand.sort()


def play_game(
    fugitive_agent: FugitiveAgent,
    marshal_agent: MarshalAgent,
    seed: int | None = None,
) -> GameResult:
    """Run one complete simulation to a rules-defined winner."""

    # Imported lazily to keep GameEngine independent of the runner module
    # during initialization while sharing one canonical phase dispatcher.
    from .driver import AgentStepError, step_agent

    engine = GameEngine(seed=seed)
    decision = 0
    while not engine.is_terminal:
        decision += 1
        try:
            step_agent(
                engine,
                fugitive_agent,
                marshal_agent,
                decision=decision,
            )
        except AgentStepError as exc:
            # play_game historically propagated policy/legality exceptions.
            # Preserve that API while experiment runners retain stage metadata.
            raise exc.original from exc
    return engine.result()


__all__ = ["GameEngine", "play_game"]
