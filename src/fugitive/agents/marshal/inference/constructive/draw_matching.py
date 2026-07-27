"""Count and sample Fugitive draw histories under card-use deadlines."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Mapping, Sequence

from fugitive.game.rules import PILE_CARDS

from .constraints import CompiledMarshalConstraints, DeadlineDemands, DrawSlot
from .sampling import SamplingCounter


@dataclass(frozen=True, slots=True)
class DrawAssignment:
    """One ordered Fugitive draw history and its proposal probability."""

    fugitive_draws: tuple[int, ...]
    remaining_piles: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]
    completion_count: int
    log_q: float


class DrawDeadlineMatcher:
    """Count and sample ordered Fugitive draws under nested use deadlines."""

    def __init__(
        self,
        draw_slots: Sequence[DrawSlot],
        marshal_hand: Sequence[int],
        *,
        pile_cards: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]] = PILE_CARDS,
        budget: SamplingCounter | None = None,
    ) -> None:
        self.draw_slots = tuple(draw_slots)
        self.marshal_hand = frozenset(marshal_hand)
        if len(pile_cards) != 3:
            raise ValueError("pile_cards must contain exactly three piles")
        normalized_piles = tuple(tuple(cards) for cards in pile_cards)
        flat_cards = tuple(card for cards in normalized_piles for card in cards)
        if any(
            isinstance(card, bool)
            or not isinstance(card, int)
            or not 1 <= card <= 42
            for card in flat_cards
        ):
            raise ValueError("pile_cards must contain game cards 1--42")
        if len(flat_cards) != len(set(flat_cards)):
            raise ValueError("pile_cards must not repeat a card across piles")
        self.pile_cards = normalized_piles
        self.card_to_pile = {
            card: pile
            for pile, cards in enumerate(self.pile_cards)
            for card in cards
        }
        self.budget = budget
        self._demand_count_cache: dict[DeadlineDemands, int] = {}

    @classmethod
    def from_constraints(
        cls,
        constraints: CompiledMarshalConstraints,
        budget: SamplingCounter,
    ) -> "DrawDeadlineMatcher":
        return cls(
            constraints.draw_slots,
            constraints.marshal_hand,
            budget=budget,
        )

    def count(self, used_deadlines: Mapping[int, int]) -> int:
        demands: list[list[int]] = [[], [], []]
        for card, deadline in used_deadlines.items():
            pile = self.card_to_pile.get(card)
            if pile is None:
                continue
            if card in self.marshal_hand:
                return 0
            demands[pile].append(deadline)
        return self.count_deadline_demands(
            tuple(tuple(sorted(values)) for values in demands)  # type: ignore[arg-type]
        )

    def count_deadline_demands(self, demands: DeadlineDemands) -> int:
        """Count completions from per-pile multisets of use deadlines.

        Card identities within a pile are exchangeable for this count. A
        caller that already counted identity choices can avoid enumerating
        them only to test whether the draw schedule is feasible.
        """

        if len(demands) != 3:
            raise ValueError("deadline demands must contain three piles")
        normalized: DeadlineDemands = tuple(
            tuple(sorted(values)) for values in demands
        )  # type: ignore[assignment]
        cached = self._demand_count_cache.get(normalized)
        if cached is not None:
            return cached

        total = 1
        for pile, cards in enumerate(self.pile_cards):
            if self.budget is not None:
                self.budget.consume("draw_deadline_count")
            slots = tuple(slot for slot in self.draw_slots if slot.pile == pile)
            required = normalized[pile]
            ways = 1
            for assigned, deadline in enumerate(required):
                eligible = sum(slot.round_number <= deadline for slot in slots)
                choices = eligible - assigned
                if choices <= 0:
                    self._demand_count_cache[normalized] = 0
                    return 0
                ways *= choices
            free_slots = len(slots) - len(required)
            optional_count = len(set(cards) - self.marshal_hand) - len(required)
            if free_slots < 0 or optional_count < free_slots:
                self._demand_count_cache[normalized] = 0
                return 0
            ways *= math.perm(optional_count, free_slots)
            total *= ways
        self._demand_count_cache[normalized] = total
        return total

    def sample(
        self,
        used_deadlines: Mapping[int, int],
        rng: random.Random,
    ) -> DrawAssignment | None:
        total = self.count(used_deadlines)
        if not total:
            return None
        assigned: list[int | None] = [None] * len(self.draw_slots)
        remaining_piles: list[tuple[int, ...]] = []
        for pile, cards in enumerate(self.pile_cards):
            pile_slot_indices = [
                index for index, slot in enumerate(self.draw_slots) if slot.pile == pile
            ]
            available_slots = set(pile_slot_indices)
            required = tuple(
                sorted(
                    (
                        (deadline, card)
                        for card, deadline in used_deadlines.items()
                        if card in cards
                    )
                )
            )
            for deadline, card in required:
                eligible = sorted(
                    index
                    for index in available_slots
                    if self.draw_slots[index].round_number <= deadline
                )
                chosen = rng.choice(eligible)
                assigned[chosen] = card
                available_slots.remove(chosen)
            required_cards = {card for _deadline, card in required}
            optional = tuple(sorted(set(cards) - self.marshal_hand - required_cards))
            free_indices = sorted(available_slots)
            selected_optional = rng.sample(optional, len(free_indices))
            for index, card in zip(free_indices, selected_optional, strict=True):
                assigned[index] = card
            drawn = required_cards | set(selected_optional)
            remaining_piles.append(tuple(sorted(set(cards) - self.marshal_hand - drawn)))
        if any(card is None for card in assigned):  # pragma: no cover - count guards this
            return None
        return DrawAssignment(
            fugitive_draws=tuple(card for card in assigned if card is not None),
            remaining_piles=tuple(remaining_piles),  # type: ignore[arg-type]
            completion_count=total,
            log_q=-math.log(total),
        )


__all__ = ["DrawAssignment", "DrawDeadlineMatcher"]
