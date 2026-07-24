"""Constructive, observation-only hidden-world proposals for the Marshal.

This module backs the registered constructive BIR-2 Marshal agent.  Its target
is explicit and modest: every complete hidden
world consistent with the Marshal observation has equal unnormalised target
density.  This is a policy-agnostic constraint target, not a calibrated
Bayesian posterior over Fugitive behaviour.

The proposal is constructive in three stages:

* sample a legal route using exact suffix completion counts;
* sample Sprint identities with either an exact descendant-count backend or a
  fast sequential importance backend whose complete proposal probability is
  recorded in ``log_q``;
* assign used cards to the observed Fugitive draw slots under nested creation
  deadlines, then fill the remaining slots without replacement.

Route-only proposals can still lead to no exact Sprint/draw completion.  Such
proposals are reported as rejections.  Their probability is part of the raw
proposal, so self-normalised importance weights for accepted worlds remain
valid up to one common acceptance-probability factor.  A future joint route /
resource DP can remove this remaining top-level rejection.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
import random
from typing import Callable, Iterator, Mapping, Protocol, Sequence

from fugitive.belief import PathBelief
from fugitive.model import DrawRecord, Observation, Role
from fugitive.particle_belief import MarshalParticle, particle_matches_observation
from fugitive.rules import PILE_CARDS, sprint_value


INITIAL_FUGITIVE_CARDS = frozenset((1, 2, 3, 42))
CARD_TO_PILE = {
    card: pile for pile, cards in enumerate(PILE_CARDS) for card in cards
}


class SamplingBudgetExceeded(RuntimeError):
    """Internal control flow raised when a deterministic budget is exhausted."""

    def __init__(self, stage: str) -> None:
        super().__init__(f"sampling budget exhausted during {stage}")
        self.stage = stage


@dataclass(frozen=True, slots=True)
class SamplingBudget:
    """Deterministic safety boundary for one constructive batch.

    ``max_nodes`` counts explored DP transitions; ``None`` records work
    without truncating it.  Unlike a wall-clock limit, a finite value preserves
    exact same-seed replay across machines.  ``max_proposals``
    bounds route proposals even when all relevant DP tables are already
    cached and therefore consume no additional nodes.
    """

    max_nodes: int | None = None
    max_proposals: int | None = None

    def __post_init__(self) -> None:
        if self.max_nodes is not None and (
            isinstance(self.max_nodes, bool)
            or not isinstance(self.max_nodes, int)
            or self.max_nodes <= 0
        ):
            raise ValueError("max_nodes must be a positive integer or None")
        if self.max_proposals is not None and (
            isinstance(self.max_proposals, bool)
            or not isinstance(self.max_proposals, int)
            or self.max_proposals <= 0
        ):
            raise ValueError("max_proposals must be a positive integer or None")


@dataclass(slots=True)
class _BudgetCounter:
    specification: SamplingBudget
    nodes: int = 0
    last_stage: str | None = None

    def consume(self, stage: str, amount: int = 1) -> None:
        if amount < 0:
            raise ValueError("budget consumption cannot be negative")
        if (
            self.specification.max_nodes is not None
            and self.nodes + amount > self.specification.max_nodes
        ):
            self.last_stage = stage
            raise SamplingBudgetExceeded(stage)
        self.nodes += amount
        self.last_stage = stage


@dataclass(frozen=True, slots=True)
class DrawSlot:
    """One labelled Fugitive draw position in public-history order."""

    global_index: int
    pile: int
    round_number: int


DeadlineDemands = tuple[
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
]


@dataclass(frozen=True, slots=True)
class CompiledMarshalConstraints:
    """Validated immutable constraints derived only from a Marshal observation."""

    observation: Observation
    fugitive_records: tuple[DrawRecord, ...]
    marshal_records: tuple[DrawRecord, ...]
    draw_slots: tuple[DrawSlot, ...]
    creation_rounds: tuple[int, ...]
    marshal_hand: tuple[int, ...]
    known_fugitive_cards: frozenset[int]
    path_belief: PathBelief = field(repr=False, compare=False)

    @classmethod
    def from_observation(cls, observation: Observation) -> "CompiledMarshalConstraints":
        if observation.role is not Role.MARSHAL:
            raise ValueError("constructive belief requires a Marshal observation")
        if len(observation.pile_sizes) != 3 or any(
            isinstance(size, bool) or not isinstance(size, int) or size < 0
            for size in observation.pile_sizes
        ):
            raise ValueError("pile_sizes must contain three non-negative integers")
        if not observation.route or observation.route[0].hideout != 0:
            raise ValueError("the public route must start at Hideout 0")
        start = observation.route[0]
        if (
            not start.revealed
            or start.sprint_count != 0
            or start.sprint_cards != ()
        ):
            raise ValueError("Hideout 0 must be fully revealed with no Sprint cards")
        if any(slot.index != index for index, slot in enumerate(observation.route)):
            raise ValueError("route indices must be contiguous")

        fugitive_records = tuple(
            record
            for record in observation.draw_history
            if record.role is Role.FUGITIVE
        )
        marshal_records = tuple(
            record
            for record in observation.draw_history
            if record.role is Role.MARSHAL
        )
        if any(not 0 <= record.pile < 3 for record in observation.draw_history):
            raise ValueError("draw records must reference pile 0, 1, or 2")
        if any(record.card is not None for record in fugitive_records):
            raise ValueError("a Marshal observation cannot reveal Fugitive draws")

        marshal_hand = tuple(sorted(observation.hand))
        if len(marshal_hand) != len(set(marshal_hand)) or any(
            card not in CARD_TO_PILE for card in marshal_hand
        ):
            raise ValueError("the Marshal hand must contain distinct pile cards")
        visible_marshal_draws = tuple(
            sorted(record.card for record in marshal_records if record.card is not None)
        )
        if visible_marshal_draws != marshal_hand or any(
            record.card is None or record.card not in PILE_CARDS[record.pile]
            for record in marshal_records
        ):
            raise ValueError("Marshal draw records must exactly identify the Marshal hand")

        seen_public: set[int] = set()
        known_fugitive = set(INITIAL_FUGITIVE_CARDS)
        for index, slot in enumerate(observation.route):
            if slot.sprint_count < 0:
                raise ValueError("Sprint counts cannot be negative")
            if slot.revealed and (
                slot.hideout is None or slot.sprint_cards is None
            ):
                raise ValueError("a revealed route slot must expose all identities")
            if index and slot.revealed and slot.hideout == 42:
                raise ValueError("Hideout 42 is public but cannot be uncovered")
            if (
                index
                and not slot.revealed
                and slot.hideout is not None
                and slot.hideout != 42
            ):
                raise ValueError("only unrevealed Hideout 42 may remain face up")
            if not slot.revealed and slot.sprint_cards is not None:
                raise ValueError("an unrevealed route slot cannot expose Sprint identities")
            if slot.sprint_cards is not None and len(slot.sprint_cards) != slot.sprint_count:
                raise ValueError("visible Sprint identities must match the public count")
            identities = (
                () if not index or slot.hideout is None else (slot.hideout,)
            ) + (slot.sprint_cards or ())
            for card in identities:
                if not 1 <= card <= 42 or card in seen_public:
                    raise ValueError("public card identities must be distinct cards 1--42")
                seen_public.add(card)
                known_fugitive.add(card)
        if set(marshal_hand) & known_fugitive:
            raise ValueError("a card cannot belong to both players")

        for pile, original in enumerate(PILE_CARDS):
            fugitive_count = sum(record.pile == pile for record in fugitive_records)
            marshal_count = sum(record.pile == pile for record in marshal_records)
            if fugitive_count + marshal_count + observation.pile_sizes[pile] != len(original):
                raise ValueError("draw history and pile sizes do not conserve cards")
            forced = sum(card in original for card in known_fugitive)
            if forced > fugitive_count:
                raise ValueError("more public Fugitive cards exist than Fugitive draws")

        rounds: list[int | None] = [0, *([None] * (len(observation.route) - 1))]
        placed: set[int] = set()
        placement_count = 0
        for record in observation.play_history:
            if record.passed:
                if record.route_index is not None or record.sprint_count:
                    raise ValueError("a Pass cannot create a route slot or Sprint stack")
                continue
            if record.route_index is None or not 0 < record.route_index < len(rounds):
                raise ValueError("play history contains an invalid route index")
            placement_count += 1
            placed.add(record.route_index)
            rounds[record.route_index] = record.round_number
            slot = observation.route[record.route_index]
            if slot.sprint_count != record.sprint_count:
                raise ValueError("play history Sprint count disagrees with the route")
            if record.hideout is not None and record.hideout != slot.hideout:
                raise ValueError("play history Hideout disagrees with the route")
            if (
                record.sprint_cards is not None
                and record.sprint_cards != slot.sprint_cards
            ):
                raise ValueError("play history Sprint identities disagree with the route")
        if observation.play_history and (
            placed != set(range(1, len(rounds)))
            or placement_count != len(rounds) - 1
        ):
            raise ValueError("play history must account for every route placement")
        for index in range(1, len(rounds)):
            if rounds[index] is None:
                rounds[index] = next(
                    (
                        record.round_number
                        for record in observation.guess_history
                        if record.route_length >= index
                    ),
                    observation.round_number,
                )

        draw_slots = tuple(
            DrawSlot(index, record.pile, record.round_number)
            for index, record in enumerate(fugitive_records)
        )
        return cls(
            observation=observation,
            fugitive_records=fugitive_records,
            marshal_records=marshal_records,
            draw_slots=draw_slots,
            creation_rounds=tuple(int(value) for value in rounds),
            marshal_hand=marshal_hand,
            known_fugitive_cards=frozenset(known_fugitive),
            path_belief=PathBelief.from_observation(observation),
        )


@dataclass(frozen=True, slots=True)
class RouteSample:
    route: tuple[int, ...]
    log_q: float
    total_paths: int


class RouteSampler(Protocol):
    """Injectable route proposal API for a future public PathBelief sampler."""

    @property
    def total_paths(self) -> int: ...

    def sample(self, rng: random.Random) -> RouteSample | None: ...


class PathBeliefRouteSampler:
    """Completion-count adapter over the current :class:`PathBelief` internals."""

    def __init__(self, belief: PathBelief, budget: _BudgetCounter) -> None:
        self.belief = belief
        self.budget = budget
        self._memo: dict[tuple[int, int, tuple[int, int, int], int], int] = {}
        self._initial: tuple[int, tuple[int, int, int], int] | None = None
        limits_valid = not (
            belief.pile_hideout_limits is not None
            and any(limit < 0 for limit in belief.pile_hideout_limits)
        ) and not (
            belief.pile_hideout_prefix_limits is not None
            and any(
                limit < 0
                for limits in belief.pile_hideout_prefix_limits
                for limit in limits
            )
        )
        if belief._basic_shape_is_valid() and limits_valid:
            values = belief._values_at(0)
            if len(values) == 1:
                value = values[0]
                counts = belief._add_pile_card((0, 0, 0), value, 0)
                multi = belief._advance_multi_state(
                    0, value, belief._initial_multi_state
                )
                if counts is not None and multi is not None:
                    self._initial = value, counts, multi
        self._total_paths: int | None = None

    @property
    def total_paths(self) -> int:
        if self._total_paths is None:
            if self._initial is None:
                self._total_paths = 0
            else:
                previous, counts, multi = self._initial
                self._total_paths = self._count_suffix(1, previous, counts, multi)
        return self._total_paths

    def _count_suffix(
        self,
        index: int,
        previous: int,
        counts: tuple[int, int, int],
        multi_state: int,
    ) -> int:
        if index == len(self.belief.route):
            return 1
        key = index, previous, counts, multi_state
        cached = self._memo.get(key)
        if cached is not None:
            return cached
        total = 0
        cap = self.belief._edge_cap(index)
        for value in self.belief._values_at(index):
            self.budget.consume("route_completion")
            if value <= previous or value - previous > cap:
                continue
            next_counts = self.belief._add_pile_card(counts, value, index)
            next_multi = self.belief._advance_multi_state(index, value, multi_state)
            if next_counts is None or next_multi is None:
                continue
            total += self._count_suffix(index + 1, value, next_counts, next_multi)
        self._memo[key] = total
        return total

    def sample(self, rng: random.Random) -> RouteSample | None:
        total = self.total_paths
        if not total or self._initial is None:
            return None
        previous, counts, multi_state = self._initial
        route = [previous]
        for index in range(1, len(self.belief.route)):
            cap = self.belief._edge_cap(index)
            branches: list[tuple[int, tuple[int, int, int], int, int]] = []
            for value in self.belief._values_at(index):
                if value <= previous or value - previous > cap:
                    continue
                next_counts = self.belief._add_pile_card(counts, value, index)
                next_multi = self.belief._advance_multi_state(
                    index, value, multi_state
                )
                if next_counts is None or next_multi is None:
                    continue
                completions = self._count_suffix(
                    index + 1, value, next_counts, next_multi
                )
                if completions:
                    branches.append((value, next_counts, next_multi, completions))
            selected = _weighted_choice(branches, lambda item: item[3], rng)
            if selected is None:  # pragma: no cover - guarded by the count DP
                return None
            previous, counts, multi_state, _ways = selected
            route.append(previous)
        return RouteSample(tuple(route), -math.log(total), total)


@dataclass(frozen=True, slots=True)
class DrawAssignment:
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
        budget: _BudgetCounter | None = None,
    ) -> None:
        self.draw_slots = tuple(draw_slots)
        self.marshal_hand = frozenset(marshal_hand)
        self.pile_cards = pile_cards
        self.budget = budget
        self._demand_count_cache: dict[DeadlineDemands, int] = {}

    @classmethod
    def from_constraints(
        cls,
        constraints: CompiledMarshalConstraints,
        budget: _BudgetCounter,
    ) -> "DrawDeadlineMatcher":
        return cls(
            constraints.draw_slots,
            constraints.marshal_hand,
            budget=budget,
        )

    def count(self, used_deadlines: Mapping[int, int]) -> int:
        demands: list[list[int]] = [[], [], []]
        for card, deadline in used_deadlines.items():
            pile = CARD_TO_PILE.get(card)
            if pile is None:
                continue
            if card in self.marshal_hand:
                return 0
            demands[pile].append(deadline)
        return self.count_deadline_demands(
            tuple(tuple(sorted(values)) for values in demands)  # type: ignore[arg-type]
        )

    def count_deadline_demands(
        self,
        demands: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]],
    ) -> int:
        """Count draw completions from per-pile multisets of use deadlines.

        Card identities within one pile are exchangeable for this count.  A
        caller that already counted identity choices can therefore avoid
        enumerating them solely to ask whether the draw schedule is feasible.
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
            optional = tuple(
                sorted(set(cards) - self.marshal_hand - required_cards)
            )
            free_indices = sorted(available_slots)
            selected_optional = rng.sample(optional, len(free_indices))
            for index, card in zip(free_indices, selected_optional, strict=True):
                assigned[index] = card
            drawn = required_cards | set(selected_optional)
            remaining_piles.append(
                tuple(sorted(set(cards) - self.marshal_hand - drawn))
            )
        if any(card is None for card in assigned):  # pragma: no cover - count guards this
            return None
        return DrawAssignment(
            fugitive_draws=tuple(card for card in assigned if card is not None),
            remaining_piles=tuple(remaining_piles),  # type: ignore[arg-type]
            completion_count=total,
            log_q=-math.log(total),
        )


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
_SPRINT_CATEGORY_SOURCES = tuple(
    source for source, _value in _SPRINT_CATEGORY_KEYS
)
CategoryAllocation = tuple[int, ...]


class SprintAssignmentDP:
    """Exact category-count DP for Sprint identities and draw assignments.

    Cards are exchangeable inside the eight ``source x Sprint-value`` groups:
    the four fixed initial cards versus each of the three piles, crossed with
    odd/+1 and even/+2.  A branch chooses only counts from these groups and is
    multiplied by ``comb(remaining, chosen)``.  Concrete identities are drawn
    uniformly only after a category branch has been selected.
    """

    def __init__(
        self,
        constraints: CompiledMarshalConstraints,
        route: Sequence[int],
        matcher: DrawDeadlineMatcher,
        budget: _BudgetCounter,
    ) -> None:
        self.constraints = constraints
        self.route = tuple(route)
        self.matcher = matcher
        self.budget = budget
        self._base_stacks: list[tuple[int, ...] | None] = [()] * len(self.route)
        self._unknown: list[int] = []
        self._fixed_used: set[int] = set(self.route[1:])
        self._fixed_deadlines: dict[int, int] = {
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
                self._base_stacks[index] = None
                self._unknown.append(index)
                continue
            fixed = tuple(sorted(stack))
            if any(card in self._fixed_used for card in fixed):
                self.valid = False
                continue
            self._fixed_used.update(fixed)
            self._base_stacks[index] = fixed
            for card in fixed:
                if card in CARD_TO_PILE:
                    self._fixed_deadlines[card] = constraints.creation_rounds[index]
        if self._fixed_used & set(constraints.marshal_hand):
            self.valid = False
        group_cards: list[list[int]] = [[] for _key in _SPRINT_CATEGORY_KEYS]
        for card in range(1, 43):
            if card in self._fixed_used or card in constraints.marshal_hand:
                continue
            source = (
                _INITIAL_SOURCE
                if card in INITIAL_FUGITIVE_CARDS
                else CARD_TO_PILE[card]
            )
            category = _SPRINT_CATEGORY_INDEX[(source, sprint_value(card))]
            group_cards[category].append(card)
        self._initial_group_cards = tuple(
            tuple(sorted(cards)) for cards in group_cards
        )
        self._initial_group_counts = tuple(
            len(cards) for cards in self._initial_group_cards
        )
        fixed_demands: list[list[int]] = [[], [], []]
        for card, deadline in self._fixed_deadlines.items():
            fixed_demands[CARD_TO_PILE[card]].append(deadline)
        self._fixed_demands: DeadlineDemands = tuple(
            tuple(sorted(values)) for values in fixed_demands
        )  # type: ignore[assignment]
        self._memo: dict[
            tuple[int, tuple[int, ...], DeadlineDemands], int
        ] = {}
        self._allocation_cache: dict[
            tuple[int, tuple[int, ...]], tuple[CategoryAllocation, ...]
        ] = {}

    @property
    def completion_count(self) -> int:
        if not self.valid:
            return 0
        return self._count(
            0,
            self._initial_group_counts,
            self._fixed_demands,
        )

    def _category_allocations(
        self,
        position: int,
        remaining: tuple[int, ...],
    ) -> tuple[CategoryAllocation, ...]:
        cache_key = position, remaining
        cached = self._allocation_cache.get(cache_key)
        if cached is not None:
            return cached
        index = self._unknown[position]
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
            maximum_future_value = value + 2 * cards_left
            if maximum_future_value < required:
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
    def _branch_multiplicity(
        remaining: tuple[int, ...],
        allocation: CategoryAllocation,
    ) -> int:
        multiplicity = 1
        for available, chosen in zip(remaining, allocation, strict=True):
            multiplicity *= math.comb(available, chosen)
        return multiplicity

    def _advance_category_state(
        self,
        position: int,
        remaining: tuple[int, ...],
        demands: DeadlineDemands,
        allocation: CategoryAllocation,
    ) -> tuple[tuple[int, ...], DeadlineDemands]:
        index = self._unknown[position]
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

    def _count(
        self,
        position: int,
        remaining: tuple[int, ...],
        demands: DeadlineDemands,
    ) -> int:
        if position == len(self._unknown):
            return self.matcher.count_deadline_demands(demands)
        key = position, remaining, demands
        cached = self._memo.get(key)
        if cached is not None:
            return cached
        total = 0
        for allocation in self._category_allocations(position, remaining):
            next_remaining, next_demands = self._advance_category_state(
                position, remaining, demands, allocation
            )
            if not self.matcher.count_deadline_demands(next_demands):
                continue
            multiplicity = self._branch_multiplicity(remaining, allocation)
            total += multiplicity * self._count(
                position + 1,
                next_remaining,
                next_demands,
            )
        self._memo[key] = total
        return total

    def sample(self, rng: random.Random) -> SprintDrawSample | None:
        total = self.completion_count
        if not total:
            return None
        stacks = list(self._base_stacks)
        group_cards = [list(cards) for cards in self._initial_group_cards]
        remaining = self._initial_group_counts
        demands = self._fixed_demands
        exact_deadlines = dict(self._fixed_deadlines)
        for position, index in enumerate(self._unknown):
            branches: list[
                tuple[CategoryAllocation, tuple[int, ...], DeadlineDemands, int]
            ] = []
            for allocation in self._category_allocations(position, remaining):
                next_remaining, next_demands = self._advance_category_state(
                    position, remaining, demands, allocation
                )
                child = self._count(position + 1, next_remaining, next_demands)
                branch_ways = self._branch_multiplicity(
                    remaining, allocation
                ) * child
                if branch_ways:
                    branches.append(
                        (allocation, next_remaining, next_demands, branch_ways)
                    )
            selected = _weighted_choice(branches, lambda item: item[3], rng)
            if selected is None:  # pragma: no cover - guarded by total
                return None
            allocation, remaining, demands, _ways = selected
            stack: list[int] = []
            deadline = self.constraints.creation_rounds[index]
            for category, chosen in enumerate(allocation):
                if not chosen:
                    continue
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
            stacks[index] = tuple(sorted(stack))
        draw = self.matcher.sample(exact_deadlines, rng)
        if draw is None:  # pragma: no cover - branch counts guarantee feasibility
            return None
        return SprintDrawSample(
            route_sprints=tuple(stack or () for stack in stacks),
            draw_assignment=draw,
            completion_count=total,
            log_q=-math.log(total),
        )


@dataclass(frozen=True, slots=True)
class SequentialAllocationBranch:
    """One auditable branch of the fast Sprint proposal."""

    allocation: CategoryAllocation
    next_remaining: tuple[int, ...]
    next_demands: DeadlineDemands
    identity_multiplicity: int
    feasible_draw_completions: int

    @property
    def proposal_weight(self) -> int:
        return self.identity_multiplicity * self.feasible_draw_completions


class SequentialSprintAssignmentProposal(SprintAssignmentDP):
    """Fast sequential importance proposal for Sprint identities.

    At each unknown stack this proposal enumerates only the current eight-way
    category allocations.  A feasible allocation receives positive weight

    ``identity multiplicity * current draw-deadline completion count``.

    The allocation is sampled from those normalized weights, then concrete
    identities are sampled uniformly inside every category.  Deadlines are
    advanced immediately and checked before proceeding to the next stack.
    The final draw assignment is sampled uniformly by ``DrawDeadlineMatcher``.

    This is deliberately not the exact descendant-count distribution.  Its
    returned ``log_q`` is nevertheless the exact probability of every choice
    made by this proposal.  A later dead end rejects the whole raw proposal;
    conditioning on acceptance contributes one common factor that cancels in
    self-normalized importance weights.
    """

    def _allocation_branches(
        self,
        position: int,
        remaining: tuple[int, ...],
        demands: DeadlineDemands,
    ) -> tuple[SequentialAllocationBranch, ...]:
        branches: list[SequentialAllocationBranch] = []
        for allocation in self._category_allocations(position, remaining):
            next_remaining, next_demands = self._advance_category_state(
                position, remaining, demands, allocation
            )
            feasible_draws = self.matcher.count_deadline_demands(next_demands)
            if not feasible_draws:
                continue
            branches.append(
                SequentialAllocationBranch(
                    allocation=allocation,
                    next_remaining=next_remaining,
                    next_demands=next_demands,
                    identity_multiplicity=self._branch_multiplicity(
                        remaining, allocation
                    ),
                    feasible_draw_completions=feasible_draws,
                )
            )
        return tuple(branches)

    def sample(self, rng: random.Random) -> SprintDrawSample | None:
        if not self.valid:
            return None
        stacks = list(self._base_stacks)
        group_cards = [list(cards) for cards in self._initial_group_cards]
        remaining = self._initial_group_counts
        demands = self._fixed_demands
        exact_deadlines = dict(self._fixed_deadlines)
        log_q = 0.0

        for position, index in enumerate(self._unknown):
            branches = self._allocation_branches(position, remaining, demands)
            total_weight = sum(branch.proposal_weight for branch in branches)
            selected = _weighted_choice(
                branches, lambda branch: branch.proposal_weight, rng
            )
            if selected is None or total_weight <= 0:
                return None
            log_q += math.log(selected.proposal_weight) - math.log(total_weight)
            remaining = selected.next_remaining
            demands = selected.next_demands

            stack: list[int] = []
            deadline = self.constraints.creation_rounds[index]
            for category, chosen in enumerate(selected.allocation):
                if not chosen:
                    continue
                available = len(group_cards[category])
                identity_count = math.comb(available, chosen)
                log_q -= math.log(identity_count)
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
            stacks[index] = tuple(sorted(stack))

        draw = self.matcher.sample(exact_deadlines, rng)
        if draw is None:
            return None
        log_q += draw.log_q
        return SprintDrawSample(
            route_sprints=tuple(stack or () for stack in stacks),
            draw_assignment=draw,
            completion_count=None,
            log_q=log_q,
        )


WorldKey = tuple[
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[tuple[int, ...], ...],
    tuple[int, ...],
    tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]],
]


@dataclass(frozen=True, slots=True)
class ConstructiveWorld:
    fugitive_hand: tuple[int, ...]
    marshal_hand: tuple[int, ...]
    route_hideouts: tuple[int, ...]
    route_sprints: tuple[tuple[int, ...], ...]
    fugitive_draws: tuple[int, ...]
    remaining_piles: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]
    hidden_mask: int
    log_q: float
    log_target: float

    @property
    def world_key(self) -> WorldKey:
        """Canonical physical-world identity, deliberately excluding weights."""

        return (
            self.fugitive_hand,
            self.marshal_hand,
            self.route_hideouts,
            self.route_sprints,
            self.fugitive_draws,
            self.remaining_piles,
        )

    @property
    def importance_log_weight(self) -> float:
        return self.log_target - self.log_q

    @property
    def all_cards_are_unique(self) -> bool:
        return self.to_particle().all_cards_are_unique

    def to_particle(self, *, weight: float = 1.0) -> MarshalParticle:
        return MarshalParticle(
            fugitive_hand=self.fugitive_hand,
            marshal_hand=self.marshal_hand,
            route_hideouts=self.route_hideouts,
            route_sprints=self.route_sprints,
            fugitive_draws=self.fugitive_draws,
            remaining_piles=self.remaining_piles,
            weight=weight,
        )


class ConstraintUniformTarget:
    """Uniform unnormalised density over observation-consistent worlds."""

    name = "constraint-uniform-complete-worlds-v1"

    def log_density(
        self,
        world: ConstructiveWorld,
        constraints: CompiledMarshalConstraints,
    ) -> float:
        particle = world.to_particle()
        if not particle_matches_observation(particle, constraints.observation):
            return -math.inf

        hidden_mask = 0
        for index, hideout in enumerate(world.route_hideouts):
            if (
                index
                and not constraints.observation.route[index].revealed
                and hideout != 42
            ):
                hidden_mask |= 1 << hideout
        if world.hidden_mask != hidden_mask:
            return -math.inf

        used = set(world.route_hideouts[1:])
        used.update(card for stack in world.route_sprints for card in stack)
        owned = set(INITIAL_FUGITIVE_CARDS) | set(world.fugitive_draws)
        if not used <= owned or set(world.fugitive_hand) != owned - used:
            return -math.inf

        expected_remaining = tuple(
            tuple(
                sorted(
                    set(cards)
                    - set(constraints.marshal_hand)
                    - set(world.fugitive_draws)
                )
            )
            for cards in PILE_CARDS
        )
        if world.remaining_piles != expected_remaining:
            return -math.inf

        draw_rounds = {
            card: constraints.fugitive_records[index].round_number
            for index, card in enumerate(world.fugitive_draws)
        }
        for index in range(1, len(world.route_hideouts)):
            deadline = constraints.creation_rounds[index]
            for card in (world.route_hideouts[index], *world.route_sprints[index]):
                if (
                    card in CARD_TO_PILE
                    and draw_rounds.get(card, deadline + 1) > deadline
                ):
                    return -math.inf
        return 0.0


@dataclass(frozen=True, slots=True)
class ConstructiveSamplingReport:
    requested: int
    produced: int
    proposals: int
    rejected_routes: int
    rejected_targets: int
    search_nodes: int
    degraded: bool
    importance_valid: bool
    termination_reason: str | None
    exhausted_stage: str | None
    unique_worlds: int


@dataclass(frozen=True, slots=True)
class ConstructiveSampleBatch:
    worlds: tuple[ConstructiveWorld, ...]
    report: ConstructiveSamplingReport

    @property
    def normalized_weights(self) -> tuple[float, ...]:
        if not self.report.importance_valid:
            raise RuntimeError(
                "importance weights are invalid after a node-budget stop"
            )
        if not self.worlds:
            return ()
        logs = tuple(world.importance_log_weight for world in self.worlds)
        if any(not math.isfinite(value) for value in logs):
            raise RuntimeError("importance weights must all be finite")
        maximum = max(logs)
        raw = tuple(math.exp(value - maximum) for value in logs)
        total = sum(raw)
        return tuple(value / total for value in raw)


RouteSamplerFactory = Callable[
    [CompiledMarshalConstraints, _BudgetCounter], RouteSampler
]


def _default_route_sampler_factory(
    constraints: CompiledMarshalConstraints,
    budget: _BudgetCounter,
) -> RouteSampler:
    return PathBeliefRouteSampler(constraints.path_belief, budget)


SPRINT_ASSIGNMENT_BACKENDS = ("exact", "sequential")


class ConstructiveWorldSampler:
    """Build weighted complete-world proposals without engine-private state.

    ``sprint_backend="exact"`` remains the reference implementation.
    ``"sequential"`` avoids recursive descendant counting and is intended for
    online agents; both backends expose the actual proposal density through
    each world's ``log_q``.
    """

    def __init__(
        self,
        *,
        target: ConstraintUniformTarget | None = None,
        route_sampler_factory: RouteSamplerFactory = _default_route_sampler_factory,
        sprint_backend: str = "exact",
    ) -> None:
        if sprint_backend not in SPRINT_ASSIGNMENT_BACKENDS:
            choices = ", ".join(SPRINT_ASSIGNMENT_BACKENDS)
            raise ValueError(f"sprint_backend must be one of {choices}")
        self.target = target or ConstraintUniformTarget()
        self.route_sampler_factory = route_sampler_factory
        self.sprint_backend = sprint_backend

    def _assignment_sampler(
        self,
        constraints: CompiledMarshalConstraints,
        route: Sequence[int],
        matcher: DrawDeadlineMatcher,
        counter: _BudgetCounter,
    ) -> SprintAssignmentDP:
        sampler_type = (
            SprintAssignmentDP
            if self.sprint_backend == "exact"
            else SequentialSprintAssignmentProposal
        )
        return sampler_type(constraints, route, matcher, counter)

    def sample_batch(
        self,
        observation: Observation,
        *,
        particle_count: int,
        seed: int | None = None,
        rng: random.Random | None = None,
        budget: SamplingBudget | None = None,
    ) -> ConstructiveSampleBatch:
        if (
            isinstance(particle_count, bool)
            or not isinstance(particle_count, int)
            or particle_count <= 0
        ):
            raise ValueError("particle_count must be a positive integer")
        if seed is not None and rng is not None:
            raise ValueError("pass either seed or rng, not both")
        owned_rng = rng if rng is not None else random.Random(seed)
        specification = budget or SamplingBudget()
        counter = _BudgetCounter(specification)
        proposals = 0
        rejected = 0
        rejected_targets = 0
        worlds: list[ConstructiveWorld] = []
        exhausted_stage: str | None = None
        termination_reason: str | None = None
        try:
            constraints = CompiledMarshalConstraints.from_observation(observation)
            route_sampler = self.route_sampler_factory(constraints, counter)
            matcher = DrawDeadlineMatcher.from_constraints(constraints, counter)
            completion_cache: dict[tuple[int, ...], SprintAssignmentDP] = {}
            max_proposals = specification.max_proposals or max(
                256, particle_count * 32
            )
            while len(worlds) < particle_count and proposals < max_proposals:
                proposals += 1
                route = route_sampler.sample(owned_rng)
                if route is None:
                    termination_reason = "no_route_support"
                    break
                assignment_dp = completion_cache.get(route.route)
                if assignment_dp is None:
                    assignment_dp = self._assignment_sampler(
                        constraints, route.route, matcher, counter
                    )
                    completion_cache[route.route] = assignment_dp
                assignment = assignment_dp.sample(owned_rng)
                if assignment is None:
                    rejected += 1
                    continue
                used = set(route.route[1:])
                used.update(
                    card for stack in assignment.route_sprints for card in stack
                )
                owned = set(INITIAL_FUGITIVE_CARDS) | set(
                    assignment.draw_assignment.fugitive_draws
                )
                fugitive_hand = tuple(sorted(owned - used))
                hidden_mask = 0
                for index, hideout in enumerate(route.route):
                    if (
                        index
                        and not observation.route[index].revealed
                        and hideout != 42
                    ):
                        hidden_mask |= 1 << hideout
                provisional = ConstructiveWorld(
                    fugitive_hand=fugitive_hand,
                    marshal_hand=constraints.marshal_hand,
                    route_hideouts=route.route,
                    route_sprints=assignment.route_sprints,
                    fugitive_draws=assignment.draw_assignment.fugitive_draws,
                    remaining_piles=assignment.draw_assignment.remaining_piles,
                    hidden_mask=hidden_mask,
                    log_q=route.log_q + assignment.log_q,
                    log_target=0.0,
                )
                log_target = self.target.log_density(provisional, constraints)
                if not math.isfinite(log_target):
                    rejected_targets += 1
                    continue
                worlds.append(replace(provisional, log_target=log_target))
        except SamplingBudgetExceeded as error:
            exhausted_stage = error.stage
            termination_reason = "node_budget"

        degraded = len(worlds) < particle_count
        if (
            degraded
            and termination_reason is None
            and proposals >= (specification.max_proposals or max(256, particle_count * 32))
        ):
            termination_reason = "max_proposals"
        return ConstructiveSampleBatch(
            worlds=tuple(worlds),
            report=ConstructiveSamplingReport(
                requested=particle_count,
                produced=len(worlds),
                proposals=proposals,
                rejected_routes=rejected,
                rejected_targets=rejected_targets,
                search_nodes=counter.nodes,
                degraded=degraded,
                importance_valid=termination_reason != "node_budget",
                termination_reason=termination_reason,
                exhausted_stage=exhausted_stage,
                unique_worlds=len({world.world_key for world in worlds}),
            ),
        )


def _weighted_choice(
    values: Sequence[object],
    weight: Callable[[object], int],
    rng: random.Random,
):
    total = sum(weight(value) for value in values)
    if total <= 0:
        return None
    target = rng.randrange(total)
    cumulative = 0
    for value in values:
        cumulative += weight(value)
        if target < cumulative:
            return value
    return values[-1]  # pragma: no cover - integer arithmetic is exact


__all__ = [
    "CompiledMarshalConstraints",
    "ConstraintUniformTarget",
    "ConstructiveSampleBatch",
    "ConstructiveSamplingReport",
    "ConstructiveWorld",
    "ConstructiveWorldSampler",
    "DrawAssignment",
    "DrawDeadlineMatcher",
    "DrawSlot",
    "PathBeliefRouteSampler",
    "RouteSample",
    "RouteSampler",
    "SamplingBudget",
    "SPRINT_ASSIGNMENT_BACKENDS",
    "SequentialAllocationBranch",
    "SequentialSprintAssignmentProposal",
    "SprintAssignmentDP",
    "SprintDrawSample",
    "WorldKey",
]
