"""Shared constraint model for Sprint-card assignment proposals.

The model owns only facts and legal transitions. Exact descendant counting and
the faster sequential importance proposal compose this model independently;
neither algorithm is a subtype of the other.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Sequence

from fugitive.game.rules import sprint_value
from ..world_validation import CARD_TO_PILE, INITIAL_FUGITIVE_CARDS

from .constraints import CompiledMarshalConstraints, DeadlineDemands
from .draw_matching import DrawAssignment, DrawDeadlineMatcher
from .sampling import SamplingCounter


@dataclass(frozen=True, slots=True)
class SprintDrawSample:
    route_sprints: tuple[tuple[int, ...], ...]
    draw_assignment: DrawAssignment
    completion_count: int | None
    log_q: float


_INITIAL_SOURCE = -1
_SPRINT_CATEGORY_KEYS = tuple(
    (source, value)
    for source in (_INITIAL_SOURCE, 0, 1, 2)
    for value in (1, 2)
)
_SPRINT_CATEGORY_INDEX = {
    key: index for index, key in enumerate(_SPRINT_CATEGORY_KEYS)
}
_SPRINT_CATEGORY_VALUES = tuple(value for _source, value in _SPRINT_CATEGORY_KEYS)
_SPRINT_CATEGORY_SOURCES = tuple(source for source, _value in _SPRINT_CATEGORY_KEYS)
CategoryAllocation = tuple[int, ...]


class SprintConstraintModel:
    """Route-specific Sprint resources, deadlines, and legal transitions."""

    def __init__(
        self,
        constraints: CompiledMarshalConstraints,
        route: Sequence[int],
        matcher: DrawDeadlineMatcher,
        budget: SamplingCounter,
    ) -> None:
        self.constraints = constraints
        self.route = tuple(route)
        self.matcher = matcher
        self.budget = budget
        self.base_stacks: list[tuple[int, ...] | None] = [()] * len(self.route)
        self.unknown: list[int] = []
        self.fixed_used: set[int] = set(self.route[1:])
        self.fixed_deadlines: dict[int, int] = {
            card: constraints.creation_rounds[index]
            for index, card in enumerate(self.route)
            if index and card in CARD_TO_PILE
        }
        self.valid = len(self.route) == len(constraints.observation.route)
        if len(set(self.route[1:])) != len(self.route) - 1:
            self.valid = False
        for index in range(1, len(self.route)):
            slot = constraints.observation.route[index]
            stack = slot.sprint_cards
            if stack is None:
                self.base_stacks[index] = None
                self.unknown.append(index)
                continue
            fixed = tuple(sorted(stack))
            if any(card in self.fixed_used for card in fixed):
                self.valid = False
                continue
            self.fixed_used.update(fixed)
            self.base_stacks[index] = fixed
            for card in fixed:
                if card in CARD_TO_PILE:
                    self.fixed_deadlines[card] = constraints.creation_rounds[index]
        if self.fixed_used & set(constraints.marshal_hand):
            self.valid = False

        group_cards: list[list[int]] = [[] for _key in _SPRINT_CATEGORY_KEYS]
        for card in range(1, 43):
            if card in self.fixed_used or card in constraints.marshal_hand:
                continue
            source = (
                _INITIAL_SOURCE
                if card in INITIAL_FUGITIVE_CARDS
                else CARD_TO_PILE[card]
            )
            category = _SPRINT_CATEGORY_INDEX[(source, sprint_value(card))]
            group_cards[category].append(card)
        self.initial_group_cards = tuple(tuple(sorted(cards)) for cards in group_cards)
        self.initial_group_counts = tuple(len(cards) for cards in self.initial_group_cards)

        fixed_demands: list[list[int]] = [[], [], []]
        for card, deadline in self.fixed_deadlines.items():
            fixed_demands[CARD_TO_PILE[card]].append(deadline)
        self.fixed_demands: DeadlineDemands = tuple(
            tuple(sorted(values)) for values in fixed_demands
        )  # type: ignore[assignment]
        self._allocation_cache: dict[
            tuple[int, tuple[int, ...]], tuple[CategoryAllocation, ...]
        ] = {}

    def category_allocations(
        self,
        position: int,
        remaining: tuple[int, ...],
    ) -> tuple[CategoryAllocation, ...]:
        """Enumerate category counts legal for one hidden Sprint stack."""

        cache_key = position, remaining
        cached = self._allocation_cache.get(cache_key)
        if cached is not None:
            return cached
        index = self.unknown[position]
        slot = self.constraints.observation.route[index]
        count = slot.sprint_count
        required = max(0, self.route[index] - self.route[index - 1] - 3)
        allocations: list[CategoryAllocation] = []
        current = [0] * len(remaining)

        def enumerate_groups(group: int, cards_left: int, value: int) -> None:
            self.budget.consume("sprint_category_allocations")
            if group == len(remaining):
                if cards_left == 0 and value >= required:
                    allocations.append(tuple(current))
                return
            if cards_left < 0 or sum(remaining[group:]) < cards_left:
                return
            if value + 2 * cards_left < required:
                return
            upper = min(remaining[group], cards_left)
            group_value = _SPRINT_CATEGORY_VALUES[group]
            for chosen in range(upper + 1):
                current[group] = chosen
                enumerate_groups(
                    group + 1,
                    cards_left - chosen,
                    value + chosen * group_value,
                )
            current[group] = 0

        enumerate_groups(0, count, 0)
        result = tuple(allocations)
        self._allocation_cache[cache_key] = result
        return result

    @staticmethod
    def branch_multiplicity(
        remaining: tuple[int, ...],
        allocation: CategoryAllocation,
    ) -> int:
        multiplicity = 1
        for available, chosen in zip(remaining, allocation, strict=True):
            multiplicity *= math.comb(available, chosen)
        return multiplicity

    def advance_category_state(
        self,
        position: int,
        remaining: tuple[int, ...],
        demands: DeadlineDemands,
        allocation: CategoryAllocation,
    ) -> tuple[tuple[int, ...], DeadlineDemands]:
        index = self.unknown[position]
        deadline = self.constraints.creation_rounds[index]
        next_remaining = tuple(
            available - chosen
            for available, chosen in zip(remaining, allocation, strict=True)
        )
        updated = [list(values) for values in demands]
        for category, chosen in enumerate(allocation):
            source = _SPRINT_CATEGORY_SOURCES[category]
            if source != _INITIAL_SOURCE and chosen:
                updated[source].extend([deadline] * chosen)
        next_demands: DeadlineDemands = tuple(
            tuple(sorted(values)) for values in updated
        )  # type: ignore[assignment]
        return next_remaining, next_demands

    def sample_identities(
        self,
        allocation: CategoryAllocation,
        group_cards: list[list[int]],
        rng: random.Random,
        deadline: int,
        exact_deadlines: dict[int, int],
    ) -> tuple[tuple[int, ...], int]:
        """Materialize category counts and return their identity multiplicity."""

        stack: list[int] = []
        identity_multiplicity = 1
        for category, chosen in enumerate(allocation):
            if not chosen:
                continue
            available = len(group_cards[category])
            identity_multiplicity *= math.comb(available, chosen)
            identities = rng.sample(group_cards[category], chosen)
            selected_identities = set(identities)
            group_cards[category] = [
                card
                for card in group_cards[category]
                if card not in selected_identities
            ]
            stack.extend(identities)
            if _SPRINT_CATEGORY_SOURCES[category] != _INITIAL_SOURCE:
                for card in identities:
                    exact_deadlines[card] = deadline
        return tuple(sorted(stack)), identity_multiplicity


__all__ = ["SprintConstraintModel", "SprintDrawSample"]
