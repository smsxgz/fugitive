"""Shared agent protocol and bounded action-generation utilities."""

from __future__ import annotations

from collections import defaultdict
import random
from typing import Iterable, Protocol

from fugitive.model import (
    FugitiveAction,
    FugitiveAgent,
    MarshalAgent,
    Observation,
    Phase,
)
from fugitive.rules import sprint_value


PILE_RANGES: tuple[range, ...] = (range(4, 15), range(15, 29), range(29, 42))


def make_rng(
    seed: int | random.Random | None = None,
    rng: random.Random | None = None,
) -> random.Random:
    """Accept either a normal seed or a caller-owned random generator."""

    if rng is not None:
        if seed is not None:
            raise ValueError("pass either seed or rng, not both")
        return rng
    if isinstance(seed, random.Random):
        return seed
    return random.Random(seed)


class Agent(FugitiveAgent, MarshalAgent, Protocol):
    """Compatibility interface for policies that intentionally play both roles.

    Role-specific policies should satisfy :class:`FugitiveAgent` or
    :class:`MarshalAgent` structurally instead of inheriting this class.  The
    name remains available for existing dual-role policies and imports.
    """

    name = "agent"


def sprint_subsets_by_value(
    cards: Iterable[int],
    *,
    rng: random.Random,
    max_per_value: int = 4,
    max_cards: int | None = None,
) -> dict[int, tuple[tuple[int, ...], ...]]:
    """Keep a bounded sample of card subsets for each Sprint total.

    Sprint cards are worth only one or two, so total-value DP has at most 83
    buckets in the real game.  Bounding each bucket avoids enumerating ``2**H``
    subsets for a large late-game hand.
    """

    ordered = list(cards)
    rng.shuffle(ordered)
    buckets: dict[int, list[tuple[int, ...]]] = {0: [()]}
    for card in ordered:
        value = sprint_value(card)
        additions: dict[int, list[tuple[int, ...]]] = defaultdict(list)
        for total, subsets in tuple(buckets.items()):
            for subset in subsets:
                if max_cards is not None and len(subset) >= max_cards:
                    continue
                additions[total + value].append(tuple(sorted((*subset, card))))
        for total, subsets in additions.items():
            combined = list(dict.fromkeys((*buckets.get(total, ()), *subsets)))
            rng.shuffle(combined)
            combined.sort(key=len)
            buckets[total] = combined[:max_per_value]
    return {total: tuple(subsets) for total, subsets in buckets.items()}


def bounded_fugitive_actions(
    observation: Observation,
    *,
    rng: random.Random,
    include_pass: bool = True,
    include_overpay: bool = True,
    max_per_hideout: int = 5,
) -> list[FugitiveAction]:
    """Generate representative legal actions in polynomial time."""

    if not observation.route or observation.route[-1].hideout is None:
        return []
    previous = observation.route[-1].hideout
    hand = tuple(observation.hand)
    actions: list[FugitiveAction] = []

    for hideout in hand:
        if hideout <= previous:
            continue
        sprint_cards = [card for card in hand if card != hideout]
        subsets = sprint_subsets_by_value(sprint_cards, rng=rng)
        required = max(0, hideout - previous - 3)
        feasible: list[tuple[int, tuple[int, ...]]] = [
            (total, subset)
            for total, candidates in subsets.items()
            if total >= required
            for subset in candidates
        ]
        if not feasible:
            continue

        feasible.sort(key=lambda item: (len(item[1]), item[0], item[1]))
        chosen: list[tuple[int, tuple[int, ...]]] = []
        if feasible:
            chosen.append(feasible[0])
        if include_overpay:
            for option in feasible[1:]:
                if option[0] > required and option not in chosen:
                    chosen.append(option)
                if len(chosen) >= max_per_hideout:
                    break
        for _total, subset in chosen[:max_per_hideout]:
            actions.append(FugitiveAction(hideout=hideout, sprint_cards=subset))

    opening = observation.phase == Phase.FUGITIVE_OPENING
    if opening and len(observation.route) == 1:
        safe = [action for action in actions if _has_opening_continuation(observation, action)]
        if safe:
            actions = safe
    if include_pass and not opening:
        actions.append(FugitiveAction(hideout=None, sprint_cards=()))
    return actions


def _has_opening_continuation(
    observation: Observation, first_action: FugitiveAction
) -> bool:
    assert first_action.hideout is not None
    spent = {first_action.hideout, *first_action.sprint_cards}
    remaining = [card for card in observation.hand if card not in spent]
    for next_hideout in remaining:
        if next_hideout <= first_action.hideout:
            continue
        available_sprint = sum(
            sprint_value(card) for card in remaining if card != next_hideout
        )
        if next_hideout - first_action.hideout <= 3 + available_sprint:
            return True
    return False


def sensible_guess_numbers(observation: Observation) -> list[int]:
    """Numbers that can still be an unrevealed Hideout."""

    unavailable = set(observation.hand)
    for slot in observation.route:
        if slot.hideout is not None:
            unavailable.add(slot.hideout)
        if slot.sprint_cards is not None:
            unavailable.update(slot.sprint_cards)
    current_route_length = len(observation.route) - 1
    unavailable.update(
        record.numbers[0]
        for record in observation.guess_history
        if not record.success
        and len(record.numbers) == 1
        and record.route_length == current_route_length
    )
    candidates = [number for number in range(1, 42) if number not in unavailable]
    if candidates:
        return candidates
    # Guessing a revealed/impossible number is still rules-legal.  This branch
    # simply prevents an agent from crashing in a contradictory test fixture.
    return list(range(1, 42))


__all__ = [
    "Agent",
    "PILE_RANGES",
    "bounded_fugitive_actions",
    "make_rng",
    "sensible_guess_numbers",
    "sprint_subsets_by_value",
    "sprint_value",
]
