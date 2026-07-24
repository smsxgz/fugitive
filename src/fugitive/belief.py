"""Belief inference over the Fugitive's hidden route.

The model deliberately reasons only from a Marshal :class:`Observation`.  It
does not try to assign identities to face-down Sprint cards; for an unrevealed
Sprint stack it therefore uses the largest possible edge capacity, ``3 + 2k``.
This makes the belief conservative while retaining the deductions exposed by
the physical game.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from functools import cached_property, lru_cache
import random
from types import MappingProxyType
from typing import Callable, Iterable, Mapping, Sequence

from fugitive.model import Observation, Role, RouteView
from fugitive.rules import PILE_CARDS, sprint_value


# Full public history is exact by default.  Optional limits remain available
# only for callers that deliberately choose an approximate teaching model.
DEFAULT_MAX_FAILED_MULTI_CONSTRAINTS: int | None = None
DEFAULT_MAX_FAILED_MULTI_MASK_BITS: int | None = None
_CARD_TO_PILE = {
    card: pile for pile, cards in enumerate(PILE_CARDS) for card in cards
}


@dataclass(frozen=True)
class BeliefResult:
    """The result of exact dynamic programming over candidate route paths."""

    total_paths: int
    marginals: Mapping[int, float]
    marginal_counts: Mapping[int, int]
    _source: "PathBelief" = field(repr=False, compare=False)

    def count_paths_containing(self, required: Iterable[int]) -> int:
        """Count consistent paths that contain every requested hideout."""

        return self._source.count_paths_containing(required)


@dataclass(frozen=True, slots=True)
class RouteSample:
    """One uniformly sampled route plus an auditable unranking trace.

    ``prefix_completion_counts`` starts with the total number of compatible
    routes.  Every following entry is the number of completions below the
    chosen prefix after one more route card has been fixed.  Likewise,
    ``prefix_ranks`` starts with the sampled global rank and records its local
    rank inside each chosen completion block.  A complete route therefore
    ends with completion count one and local rank zero.
    """

    route: tuple[int, ...]
    rank: int
    total_completions: int
    prefix_completion_counts: tuple[int, ...]
    prefix_ranks: tuple[int, ...]

    @property
    def proposal_probability(self) -> Fraction:
        """Return the exact probability assigned to this route."""

        return Fraction(1, self.total_completions)

    @property
    def conditional_probabilities(self) -> tuple[Fraction, ...]:
        """Return each selected branch probability in route order."""

        return tuple(
            Fraction(child, parent)
            for parent, child in zip(
                self.prefix_completion_counts[:-1],
                self.prefix_completion_counts[1:],
                strict=True,
            )
        )


class PathBelief:
    """Count strictly increasing routes consistent with public information.

    Args:
        route: Public route slots, including the revealed starting card at
            index zero.  A hidden hideout has ``hideout=None``.
        excluded_cards: Cards known not to be hidden hideouts, normally the
            Marshal hand plus all public Sprint and route cards.
        failed_single_guesses: ``(number, route_length)`` pairs.  Route length
            is the number of Fugitive hideouts at the time of the failed guess
            and excludes card zero.
        failed_multi_guesses: ``(numbers, route_length)`` constraints.  Each
            one excludes paths containing every number within the historical
            route prefix.  :meth:`from_observation` preserves their full
            conjunction by default, while removing constraints already
            implied by a stronger failure.
        pile_hideout_limits: Maximum route Hideouts from each draw pile after
            subtracting identities of public Sprint cards.  ``None`` disables
            this optional resource constraint for small synthetic examples.
        pile_hideout_prefix_limits: Optional per-route-index versions of those
            limits.  Production observations use these to prevent a later
            Fugitive draw from supplying an earlier route position.
        candidate_cards: Values allowed in a hidden slot.  The game default is
            1--41 because card 42 is public as soon as it is played.

    The class is useful directly in small teaching examples; production code
    will normally call :meth:`from_observation`.
    """

    def __init__(
        self,
        route: Sequence[RouteView],
        excluded_cards: Iterable[int] = (),
        failed_single_guesses: Iterable[tuple[int, int]] = (),
        failed_multi_guesses: Iterable[tuple[Iterable[int], int]] = (),
        pile_hideout_limits: tuple[int, int, int] | None = None,
        pile_hideout_prefix_limits: Sequence[tuple[int, int, int]] | None = None,
        candidate_cards: Iterable[int] = range(1, 42),
    ) -> None:
        self.route = tuple(route)
        self.excluded_cards = frozenset(excluded_cards)
        self.failed_single_guesses = tuple(
            (int(number), int(route_length))
            for number, route_length in failed_single_guesses
        )
        normalized_multis: list[tuple[tuple[int, ...], int]] = []
        seen_multis: set[tuple[tuple[int, ...], int]] = set()
        for numbers, route_length in failed_multi_guesses:
            distinct_numbers = tuple(sorted(set(numbers)))
            constraint = (distinct_numbers, int(route_length))
            if len(distinct_numbers) > 1 and constraint not in seen_multis:
                normalized_multis.append(constraint)
                seen_multis.add(constraint)
        number_sets = tuple(
            frozenset(numbers) for numbers, _route_length in normalized_multis
        )
        # C(A, LA) implies C(B, LB) when A is a subset of B and LA >= LB:
        # if B had appeared by LB, A would necessarily have appeared by LA.
        self.failed_multi_guesses = tuple(
            constraint
            for index, constraint in enumerate(normalized_multis)
            if not any(
                other_index != index
                and number_sets[other_index].issubset(number_sets[index])
                and normalized_multis[other_index][1] >= constraint[1]
                for other_index in range(len(normalized_multis))
            )
        )
        if pile_hideout_limits is not None and len(pile_hideout_limits) != 3:
            raise ValueError("pile_hideout_limits must contain three values")
        self.pile_hideout_limits = pile_hideout_limits
        if (
            pile_hideout_prefix_limits is not None
            and len(pile_hideout_prefix_limits) != len(self.route)
        ):
            raise ValueError(
                "pile_hideout_prefix_limits must match the route length"
            )
        self.pile_hideout_prefix_limits = (
            tuple(pile_hideout_prefix_limits)
            if pile_hideout_prefix_limits is not None
            else None
        )
        self.candidate_cards = tuple(sorted(set(candidate_cards)))
        self._candidate_set = frozenset(self.candidate_cards)
        # A legal route is strictly increasing, so each failed set can only be
        # matched in sorted order.  Pack its next-needed position into an
        # arbitrary-precision integer: zero means the constraint is already
        # safe, while p + 1 means p values have matched.  As soon as the route
        # passes a missing value, that constraint is cleared permanently.
        packed_constraints: list[tuple[tuple[int, ...], int, int, int]] = []
        bit_offset = 0
        for numbers, route_length in self.failed_multi_guesses:
            width = max(1, len(numbers).bit_length())
            field_mask = ((1 << width) - 1) << bit_offset
            packed_constraints.append(
                (numbers, route_length, bit_offset, field_mask)
            )
            bit_offset += width
        self._packed_multi_constraints = tuple(packed_constraints)
        self._initial_multi_state = sum(
            1 << offset
            for _numbers, _route_length, offset, _field_mask
            in self._packed_multi_constraints
        )

    @classmethod
    def from_observation(
        cls,
        observation: Observation,
        *,
        max_failed_multi_constraints: int | None = (
            DEFAULT_MAX_FAILED_MULTI_CONSTRAINTS
        ),
        max_failed_multi_mask_bits: int | None = DEFAULT_MAX_FAILED_MULTI_MASK_BITS,
    ) -> "PathBelief":
        """Construct a route belief using only public and Marshal information.

        All failed single and informative failed multi guesses are represented
        exactly.  The optional limits are ``None`` by default; supplying one
        explicitly opts into a bounded approximation using the most recent
        constraints.
        """

        if (
            isinstance(max_failed_multi_constraints, bool)
            or (
                max_failed_multi_constraints is not None
                and max_failed_multi_constraints < 0
            )
        ):
            raise ValueError("max_failed_multi_constraints must be non-negative")
        if (
            isinstance(max_failed_multi_mask_bits, bool)
            or (
                max_failed_multi_mask_bits is not None
                and max_failed_multi_mask_bits < 0
            )
        ):
            raise ValueError("max_failed_multi_mask_bits must be non-negative")

        public_route_cards = {
            slot.hideout for slot in observation.route if slot.hideout is not None
        }
        public_sprint_cards = {
            card
            for slot in observation.route
            if slot.sprint_cards is not None
            for card in slot.sprint_cards
        }
        failed_singles: list[tuple[int, int]] = []
        failed_multis: list[tuple[tuple[int, ...], int]] = []
        revealed_before: set[int] = set()
        for record in observation.guess_history:
            if record.success:
                revealed_before.update(record.numbers)
                continue
            # If any member had already been revealed, failure was guaranteed
            # and says nothing about the remaining guessed numbers.
            if any(number in revealed_before for number in record.numbers):
                continue
            if len(record.numbers) == 1:
                failed_singles.append((record.numbers[0], record.route_length))
            elif len(record.numbers) > 1:
                failed_multis.append(
                    (tuple(sorted(record.numbers)), record.route_length)
                )

        informative_multis: list[tuple[tuple[int, ...], int]] = []
        seen_multis: set[tuple[tuple[int, ...], int]] = set()
        for constraint in failed_multis:
            if constraint in seen_multis:
                continue
            seen_multis.add(constraint)
            numbers, route_length = constraint
            # More distinct guesses than historical route slots could never
            # all have hit, so such a failure carries no information.
            if len(numbers) <= route_length:
                informative_multis.append(constraint)

        retained_multis = informative_multis
        if (
            max_failed_multi_constraints is not None
            or max_failed_multi_mask_bits is not None
        ):
            retained_multis = []
            retained_mask_bits = 0
            bounded_constraints = (
                ()
                if max_failed_multi_constraints == 0
                or max_failed_multi_mask_bits == 0
                else reversed(informative_multis)
            )
            for constraint in bounded_constraints:
                numbers, _route_length = constraint
                if (
                    max_failed_multi_mask_bits is not None
                    and retained_mask_bits + len(numbers)
                    > max_failed_multi_mask_bits
                ):
                    continue
                retained_multis.append(constraint)
                retained_mask_bits += len(numbers)
                if (
                    max_failed_multi_constraints is not None
                    and len(retained_multis) >= max_failed_multi_constraints
                ):
                    break
            retained_multis.reverse()

        fugitive_draw_counts = [0, 0, 0]
        for record in observation.draw_history:
            if record.role is Role.FUGITIVE and 0 <= record.pile < 3:
                fugitive_draw_counts[record.pile] += 1
        public_sprint_counts = [0, 0, 0]
        for card in public_sprint_cards:
            pile = _CARD_TO_PILE.get(card)
            if pile is not None:
                public_sprint_counts[pile] += 1
        pile_hideout_limits = tuple(
            draws - sprint_cards
            for draws, sprint_cards in zip(
                fugitive_draw_counts, public_sprint_counts, strict=True
            )
        )

        # Engine invariant: every nonterminal Hideout placement is followed by
        # a Marshal guess in the same round, and observations retain guess
        # history in chronological order.  The first record whose route length
        # reaches a slot therefore identifies that slot's creation round.  A
        # newly placed tail has no record yet and belongs to the current round.
        creation_rounds = [0]
        for index in range(1, len(observation.route)):
            creation_rounds.append(
                next(
                    (
                        record.round_number
                        for record in observation.guess_history
                        if record.route_length >= index
                    ),
                    observation.round_number,
                )
            )
        pile_hideout_prefix_limits: list[tuple[int, int, int]] = []
        public_sprints_so_far = [0, 0, 0]
        for index, (slot, creation_round) in enumerate(
            zip(observation.route, creation_rounds, strict=True)
        ):
            if slot.sprint_cards is not None:
                for card in slot.sprint_cards:
                    pile = _CARD_TO_PILE.get(card)
                    if pile is not None:
                        public_sprints_so_far[pile] += 1
            draws_available = [
                sum(
                    record.role is Role.FUGITIVE
                    and record.pile == pile
                    and record.round_number <= creation_round
                    for record in observation.draw_history
                )
                for pile in range(3)
            ]
            pile_hideout_prefix_limits.append(
                tuple(
                    draws - sprint_cards
                    for draws, sprint_cards in zip(
                        draws_available, public_sprints_so_far, strict=True
                    )
                )
            )
        return cls(
            route=observation.route,
            excluded_cards=(
                set(observation.hand) | public_route_cards | public_sprint_cards
            ),
            failed_single_guesses=failed_singles,
            failed_multi_guesses=retained_multis,
            pile_hideout_limits=pile_hideout_limits,
            pile_hideout_prefix_limits=pile_hideout_prefix_limits,
        )

    @cached_property
    def result(self) -> BeliefResult:
        if (
            self.pile_hideout_limits is not None
            or self.pile_hideout_prefix_limits is not None
            or self.failed_multi_guesses
        ):
            total = self._count_constrained(())
            possible_hidden_cards = {
                card
                for index, slot in enumerate(self.route)
                if slot.hideout is None
                for card in self._values_at(index)
            }
            counts = {
                card: count
                for card in possible_hidden_cards
                if (count := self._count_constrained((card,)))
            } if total else {}
            marginals = {
                card: count / total for card, count in counts.items()
            } if total else {}
            return BeliefResult(
                total_paths=total,
                marginals=MappingProxyType(marginals),
                marginal_counts=MappingProxyType(counts),
                _source=self,
            )

        forward, backward, total = self._forward_backward()
        counts: dict[int, int] = {card: 0 for card in self.candidate_cards}

        if total:
            for index, slot in enumerate(self.route):
                if slot.hideout is not None:
                    continue
                for card, prefix_count in forward[index].items():
                    counts[card] += prefix_count * backward[index].get(card, 0)

        counts = {card: count for card, count in counts.items() if count}
        marginals = {
            card: count / total for card, count in counts.items()
        } if total else {}
        return BeliefResult(
            total_paths=total,
            marginals=MappingProxyType(marginals),
            marginal_counts=MappingProxyType(counts),
            _source=self,
        )

    @property
    def total_paths(self) -> int:
        return self.result.total_paths

    @property
    def marginals(self) -> Mapping[int, float]:
        return self.result.marginals

    def solve(self) -> BeliefResult:
        """Return the cached exact result."""

        return self.result

    def count_paths_containing(self, required: Iterable[int]) -> int:
        """Count paths containing all values in ``required``.

        Required values are consumed in sorted order.  This needs only one
        extra integer per DP state rather than a ``2**len(required)`` mask,
        because every legal route is strictly increasing.
        """

        needed = tuple(sorted(set(required)))
        if not needed:
            return self.total_paths
        if (
            self.pile_hideout_limits is not None
            or self.pile_hideout_prefix_limits is not None
            or self.failed_multi_guesses
        ):
            return self._count_constrained(needed)
        if not self._basic_shape_is_valid():
            return 0

        states: dict[tuple[int, int], int] = {}
        for value in self._values_at(0):
            progress = self._consume_required(0, value, needed)
            if progress is not None:
                states[(value, progress)] = 1

        for index in range(1, len(self.route)):
            cap = self._edge_cap(index)
            next_states: dict[tuple[int, int], int] = {}
            for value in self._values_at(index):
                for (previous, progress), count in states.items():
                    if previous >= value or value - previous > cap:
                        continue
                    next_progress = self._consume_required(progress, value, needed)
                    if next_progress is not None:
                        key = (value, next_progress)
                        next_states[key] = next_states.get(key, 0) + count
            states = next_states
            if not states:
                return 0

        return sum(
            count
            for (_value, progress), count in states.items()
            if progress == len(needed)
        )

    def sample_route(self, rng: random.Random) -> RouteSample:
        """Sample a compatible route exactly in proportion to path count.

        The method draws one uniformly distributed integer rank, then decodes
        it through completion-count blocks.  It never proposes and rejects a
        route.  All constraints represented by this belief, including pile
        prefix limits and historical failed multi-guesses, participate in the
        completion counts.
        """

        total = self._route_completion_count()
        if total <= 0:
            raise ValueError("cannot sample an empty route belief")
        return self.route_from_rank(rng.randrange(total))

    def route_from_rank(self, rank: int) -> RouteSample:
        """Decode one zero-based rank from the compatible-route catalogue."""

        if isinstance(rank, bool) or not isinstance(rank, int):
            raise TypeError("route rank must be an integer")
        total = self._route_completion_count()
        if total <= 0:
            raise ValueError("cannot rank routes in an empty route belief")
        if not 0 <= rank < total:
            raise ValueError(f"route rank must be from zero through {total - 1}")

        original_rank = rank
        route: list[int] = []
        completion_trace = [total]
        rank_trace = [rank]
        previous = -1
        counts = (0, 0, 0)
        multi_state = self._initial_multi_state

        for index in range(len(self.route)):
            selected: tuple[int, tuple[int, int, int], int, int] | None = None
            for value, next_counts, next_multi_state, completions in (
                self._route_completion_options(
                    index,
                    previous,
                    counts,
                    multi_state,
                )
            ):
                if rank < completions:
                    selected = (
                        value,
                        next_counts,
                        next_multi_state,
                        completions,
                    )
                    break
                rank -= completions
            if selected is None:  # pragma: no cover - guards the DP invariant
                raise RuntimeError("route completion counts could not decode rank")

            value, counts, multi_state, completions = selected
            route.append(value)
            previous = value
            completion_trace.append(completions)
            rank_trace.append(rank)

        if completion_trace[-1] != 1 or rank_trace[-1] != 0:
            raise RuntimeError("completed route has an inconsistent rank trace")
        return RouteSample(
            route=tuple(route),
            rank=original_rank,
            total_completions=total,
            prefix_completion_counts=tuple(completion_trace),
            prefix_ranks=tuple(rank_trace),
        )

    def _route_completion_count(self) -> int:
        if not self._sampling_shape_is_valid():
            return 0
        return self._route_completion_counter(
            0,
            -1,
            (0, 0, 0),
            self._initial_multi_state,
        )

    @cached_property
    def _route_completion_counter(
        self,
    ) -> Callable[[int, int, tuple[int, int, int], int], int]:
        @lru_cache(maxsize=None)
        def count(
            index: int,
            previous: int,
            pile_counts: tuple[int, int, int],
            multi_state: int,
        ) -> int:
            if index == len(self.route):
                return 1
            return sum(
                completions
                for _value, _counts, _multi_state, completions
                in self._route_completion_options(
                    index,
                    previous,
                    pile_counts,
                    multi_state,
                )
            )

        return count

    def _route_completion_options(
        self,
        index: int,
        previous: int,
        pile_counts: tuple[int, int, int],
        multi_state: int,
    ) -> tuple[tuple[int, tuple[int, int, int], int, int], ...]:
        options: list[tuple[int, tuple[int, int, int], int, int]] = []
        cap = self._edge_cap(index) if index else None
        for value in self._values_at(index):
            if index and (previous >= value or value - previous > cap):
                continue
            next_counts = self._add_pile_card(pile_counts, value, index)
            next_multi_state = self._advance_multi_state(
                index,
                value,
                multi_state,
            )
            if next_counts is None or next_multi_state is None:
                continue
            completions = self._route_completion_counter(
                index + 1,
                value,
                next_counts,
                next_multi_state,
            )
            if completions:
                options.append(
                    (value, next_counts, next_multi_state, completions)
                )
        return tuple(options)

    def _sampling_shape_is_valid(self) -> bool:
        if not self._basic_shape_is_valid():
            return False
        if self.pile_hideout_limits is not None and any(
            limit < 0 for limit in self.pile_hideout_limits
        ):
            return False
        return not (
            self.pile_hideout_prefix_limits is not None
            and any(
                limit < 0
                for limits in self.pile_hideout_prefix_limits
                for limit in limits
            )
        )

    @lru_cache(maxsize=256)
    def _count_constrained(self, needed: tuple[int, ...]) -> int:
        """Count paths with draw capacities and exact multi-failure state."""

        if not self._basic_shape_is_valid():
            return 0
        if self.pile_hideout_limits is not None and any(
            limit < 0 for limit in self.pile_hideout_limits
        ):
            return 0
        if self.pile_hideout_prefix_limits is not None and any(
            limit < 0
            for limits in self.pile_hideout_prefix_limits
            for limit in limits
        ):
            return 0

        initial_counts = (0, 0, 0)
        initial_multi_state = self._initial_multi_state
        states: dict[tuple[int, int, tuple[int, int, int], int], int] = {}
        for value in self._values_at(0):
            progress = self._consume_required(0, value, needed)
            counts = self._add_pile_card(initial_counts, value, 0)
            multi_state = self._advance_multi_state(
                0, value, initial_multi_state
            )
            if progress is not None and counts is not None and multi_state is not None:
                states[(value, progress, counts, multi_state)] = 1

        for index in range(1, len(self.route)):
            cap = self._edge_cap(index)
            next_states: dict[
                tuple[int, int, tuple[int, int, int], int], int
            ] = {}
            for value in self._values_at(index):
                for (previous, progress, counts, multi_state), ways in states.items():
                    if previous >= value or value - previous > cap:
                        continue
                    next_progress = self._consume_required(progress, value, needed)
                    next_counts = self._add_pile_card(counts, value, index)
                    next_multi_state = self._advance_multi_state(
                        index, value, multi_state
                    )
                    if (
                        next_progress is None
                        or next_counts is None
                        or next_multi_state is None
                    ):
                        continue
                    key = (value, next_progress, next_counts, next_multi_state)
                    next_states[key] = next_states.get(key, 0) + ways
            states = next_states
            if not states:
                return 0

        return sum(
            ways
            for (_value, progress, _counts, _multi_state), ways in states.items()
            if progress == len(needed)
        )

    def _add_pile_card(
        self, counts: tuple[int, int, int], value: int, index: int
    ) -> tuple[int, int, int] | None:
        pile = _CARD_TO_PILE.get(value)
        if pile is None:
            return counts
        updated = list(counts)
        updated[pile] += 1
        limits = self.pile_hideout_limits
        if self.pile_hideout_prefix_limits is not None:
            limits = self.pile_hideout_prefix_limits[index]
        if limits is not None and updated[pile] > limits[pile]:
            return None
        return updated[0], updated[1], updated[2]

    def _advance_multi_state(
        self, index: int, value: int, multi_state: int
    ) -> int | None:
        if index == 0 or not multi_state:
            return multi_state
        updated = multi_state
        for numbers, route_length, offset, field_mask in self._packed_multi_constraints:
            code = (updated & field_mask) >> offset
            if not code:
                continue
            progress = code - 1
            next_number = numbers[progress]
            next_code = code
            if value == next_number:
                progress += 1
                if progress == len(numbers):
                    return None
                next_code = progress + 1
            elif value > next_number:
                next_code = 0
            if index >= route_length:
                next_code = 0
            if next_code != code:
                updated = (updated & ~field_mask) | (next_code << offset)
        return updated

    @staticmethod
    def _consume_required(
        progress: int, value: int, required: tuple[int, ...]
    ) -> int | None:
        if progress == len(required):
            return progress
        next_required = required[progress]
        if value == next_required:
            return progress + 1
        if value > next_required:
            return None
        return progress

    def _basic_shape_is_valid(self) -> bool:
        if not self.route:
            return False
        if self.route[0].index != 0 or self.route[0].hideout != 0:
            return False
        return all(slot.index == index for index, slot in enumerate(self.route))

    def _values_at(self, index: int) -> tuple[int, ...]:
        slot = self.route[index]
        if slot.hideout is not None:
            return (slot.hideout,)

        prohibited_by_history = {
            number
            for number, route_length in self.failed_single_guesses
            if index <= route_length
        }
        return tuple(
            card
            for card in self.candidate_cards
            if card not in self.excluded_cards
            and card not in prohibited_by_history
        )

    def _edge_cap(self, index: int) -> int:
        slot = self.route[index]
        if slot.sprint_cards is None:
            sprint_capacity = 2 * slot.sprint_count
        else:
            sprint_capacity = sum(sprint_value(card) for card in slot.sprint_cards)
        return 3 + sprint_capacity

    def _forward_backward(
        self,
    ) -> tuple[list[dict[int, int]], list[dict[int, int]], int]:
        if not self._basic_shape_is_valid():
            return [], [], 0

        forward: list[dict[int, int]] = [dict() for _ in self.route]
        forward[0] = {value: 1 for value in self._values_at(0)}
        for index in range(1, len(self.route)):
            cap = self._edge_cap(index)
            current: dict[int, int] = {}
            for value in self._values_at(index):
                count = sum(
                    ways
                    for previous, ways in forward[index - 1].items()
                    if previous < value and value - previous <= cap
                )
                if count:
                    current[value] = count
            forward[index] = current

        total = sum(forward[-1].values())
        backward: list[dict[int, int]] = [dict() for _ in self.route]
        backward[-1] = {value: 1 for value in forward[-1]}
        for index in range(len(self.route) - 2, -1, -1):
            cap = self._edge_cap(index + 1)
            current: dict[int, int] = {}
            for value in forward[index]:
                count = sum(
                    ways
                    for following, ways in backward[index + 1].items()
                    if value < following and following - value <= cap
                )
                if count:
                    current[value] = count
            backward[index] = current

        return forward, backward, total


@lru_cache(maxsize=256)
def solve_observation_belief(observation: Observation) -> BeliefResult:
    """Cache the deterministic Marshal belief for one immutable observation."""

    return PathBelief.from_observation(observation).solve()


__all__ = [
    "BeliefResult",
    "PathBelief",
    "RouteSample",
    "solve_observation_belief",
]
