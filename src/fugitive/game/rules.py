"""Pure legality helpers for the canonical Fugitive rules."""

from __future__ import annotations

from itertools import combinations
from typing import Iterable, Iterator

from .model import FugitiveAction, Observation, Phase, Role


MIN_CARD = 0
MAX_CARD = 42
GUESSABLE_CARDS = tuple(range(1, 42))
PILE_CARDS: tuple[tuple[int, ...], ...] = (
    tuple(range(4, 15)),
    tuple(range(15, 29)),
    tuple(range(29, 42)),
)


def _is_card(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= 42


def sprint_value(card: int) -> int:
    """Return +1 for odd cards and +2 for even cards."""

    if not _is_card(card) or card == 0:
        raise ValueError(f"Sprint card must be an integer from 1 through 42, got {card!r}")
    return 1 if card % 2 else 2


def required_sprint_value(previous_hideout: int, hideout: int) -> int:
    """Return the minimum Sprint value needed for a forward move."""

    if not _is_card(previous_hideout) or not _is_card(hideout):
        raise ValueError("hideouts must be integer cards from 0 through 42")
    if hideout <= previous_hideout:
        raise ValueError("a new hideout must be higher than the previous hideout")
    return max(0, hideout - previous_hideout - 3)


def is_legal_fugitive_action(
    action: FugitiveAction,
    hand: Iterable[int],
    previous_hideout: int,
    *,
    allow_pass: bool = True,
) -> bool:
    """Check a Fugitive action without mutating game state."""

    try:
        hand_tuple = tuple(hand)
        sprint_cards = tuple(action.sprint_cards)
    except (TypeError, AttributeError):
        return False

    if action.hideout is None:
        return allow_pass and not sprint_cards
    if not _is_card(action.hideout) or action.hideout == 0:
        return False
    if not _is_card(previous_hideout) or action.hideout <= previous_hideout:
        return False
    if any(not _is_card(card) or card == 0 for card in hand_tuple):
        return False
    if len(hand_tuple) != len(set(hand_tuple)):
        return False
    if action.hideout not in hand_tuple:
        return False
    if any(not _is_card(card) or card == 0 for card in sprint_cards):
        return False
    if len(sprint_cards) != len(set(sprint_cards)):
        return False
    if action.hideout in sprint_cards:
        return False
    hand_set = set(hand_tuple)
    if any(card not in hand_set for card in sprint_cards):
        return False

    available_distance = 3 + sum(sprint_value(card) for card in sprint_cards)
    return action.hideout - previous_hideout <= available_distance


def has_legal_hideout_play(hand: Iterable[int], previous_hideout: int) -> bool:
    """Return whether ``hand`` can establish at least one forward Hideout."""

    try:
        hand_tuple = tuple(hand)
    except TypeError:
        return False
    for hideout in hand_tuple:
        if not _is_card(hideout) or hideout <= previous_hideout:
            continue
        # All Sprint values are positive, so if the maximum payment fails then
        # no subset can work; if it succeeds, that payment itself is legal.
        sprint_cards = tuple(card for card in hand_tuple if card != hideout)
        if is_legal_fugitive_action(
            FugitiveAction(hideout, sprint_cards),
            hand_tuple,
            previous_hideout,
            allow_pass=False,
        ):
            return True
    return False


def iter_legal_fugitive_actions(
    observation: Observation,
    *,
    max_sprint_cards: int | None = None,
    include_pass: bool | None = None,
) -> Iterator[FugitiveAction]:
    """Yield legal actions in deterministic order.

    With ``max_sprint_cards=None`` this is an exact action enumeration and can
    be exponential in a large hand because all legal overpayments are distinct
    actions.  Search agents should normally set a small limit as an action
    abstraction.
    """

    if observation.role is not Role.FUGITIVE:
        return
    if observation.phase not in (Phase.FUGITIVE_OPENING, Phase.FUGITIVE_ACTION):
        return
    if not observation.route or observation.route[-1].hideout is None:
        return

    if include_pass is None:
        include_pass = observation.phase is Phase.FUGITIVE_ACTION
    if include_pass and observation.phase is Phase.FUGITIVE_ACTION:
        yield FugitiveAction(None)

    if max_sprint_cards is not None:
        if isinstance(max_sprint_cards, bool) or max_sprint_cards < 0:
            raise ValueError("max_sprint_cards must be non-negative or None")

    hand = tuple(sorted(observation.hand))
    previous = observation.route[-1].hideout
    assert previous is not None
    for hideout in hand:
        if hideout <= previous:
            continue
        sprint_candidates = tuple(card for card in hand if card != hideout)
        limit = len(sprint_candidates)
        if max_sprint_cards is not None:
            limit = min(limit, max_sprint_cards)
        for count in range(limit + 1):
            for sprint_cards in combinations(sprint_candidates, count):
                action = FugitiveAction(hideout, sprint_cards)
                if is_legal_fugitive_action(
                    action,
                    hand,
                    previous,
                    allow_pass=False,
                ):
                    if (
                        observation.phase is Phase.FUGITIVE_OPENING
                        and len(observation.route) == 1
                    ):
                        spent = {hideout, *sprint_cards}
                        remaining = tuple(card for card in hand if card not in spent)
                        if not has_legal_hideout_play(remaining, hideout):
                            continue
                    yield action


def legal_fugitive_actions(
    observation: Observation,
    *,
    max_sprint_cards: int | None = None,
    include_pass: bool | None = None,
) -> tuple[FugitiveAction, ...]:
    """Materialize :func:`iter_legal_fugitive_actions` as a tuple."""

    return tuple(
        iter_legal_fugitive_actions(
            observation,
            max_sprint_cards=max_sprint_cards,
            include_pass=include_pass,
        )
    )


def is_legal_guess(numbers: Iterable[int], *, manhunt: bool = False) -> bool:
    """Check the shape of a Marshal guess, independent of hidden truth."""

    try:
        guess = tuple(numbers)
    except TypeError:
        return False
    if not guess or (manhunt and len(guess) != 1):
        return False
    if len(guess) != len(set(guess)):
        return False
    return all(
        isinstance(number, int)
        and not isinstance(number, bool)
        and 1 <= number <= 41
        for number in guess
    )


__all__ = [
    "GUESSABLE_CARDS",
    "MAX_CARD",
    "MIN_CARD",
    "PILE_CARDS",
    "has_legal_hideout_play",
    "is_legal_fugitive_action",
    "is_legal_guess",
    "iter_legal_fugitive_actions",
    "legal_fugitive_actions",
    "required_sprint_value",
    "sprint_value",
]
