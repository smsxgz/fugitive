"""Hierarchical legal-random baselines for the full Fugitive game.

The policies in this module randomize over rule-level decisions rather than
over the engine's flat action encoding.  In particular, a Hideout is selected
before its Sprint payment and the Marshal first selects a small guess size.
Every decision is a function of the acting player's :class:`Observation` and
an explicitly injected random generator.
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
import random
from typing import Iterable, Mapping, TypeVar

from fugitive.belief import BeliefResult, solve_observation_belief
from fugitive.model import FugitiveAction, Observation, Phase, Role
from fugitive.observation_protocol import stable_observation_seed
from fugitive.rules import is_legal_fugitive_action, required_sprint_value, sprint_value

from .base import make_rng


DEFAULT_MAX_LOW_COST_PAYMENTS = 8
DEFAULT_MAX_EXTRA_OVERPAYMENTS = 8
DEFAULT_OVERPAY_PROBABILITY = 0.05
DEFAULT_MULTI_GUESS_CONTINUATION = 0.10
DEFAULT_MAX_GUESS_SIZE = 4
DEFAULT_MAX_GUESSES_PER_SIZE = 128
HR1_FUGITIVE_ALGORITHM_ID = "hr-1-fugitive-hierarchical-legal-random-v1"
HR1_MARSHAL_ALGORITHM_ID = "hr-1-marshal-hard-support-random-v2"


@dataclass(frozen=True, slots=True)
class SprintPayment:
    """One bounded Sprint payment and its normalized resource cost."""

    cards: tuple[int, ...]
    sprint_total: int
    overpay: int
    cost: float


@dataclass(frozen=True, slots=True)
class SprintPlanGroup:
    """Low-cost payments plus a small, separately sampled overpay branch."""

    low_cost: tuple[SprintPayment, ...]
    extra_overpay: tuple[SprintPayment, ...]

    @property
    def all_payments(self) -> tuple[SprintPayment, ...]:
        return self.low_cost + self.extra_overpay


@dataclass(frozen=True, slots=True)
class OpeningPlan:
    """Both Hideouts and Sprint payments selected for the opening turn."""

    first: FugitiveAction
    second: FugitiveAction


@dataclass(frozen=True, slots=True)
class OpeningPlanGroup:
    """Bounded complete opening plans, split by deliberate overpayment."""

    low_cost: tuple[OpeningPlan, ...]
    extra_overpay: tuple[OpeningPlan, ...]


T = TypeVar("T")


def sprint_cost(
    previous_hideout: int,
    hideout: int,
    sprint_cards: Iterable[int],
) -> float:
    """Return ``|S| + .5*overpay + .75*future_cards`` for a payment."""

    cards = tuple(sprint_cards)
    required = required_sprint_value(previous_hideout, hideout)
    overpay = max(0, sum(sprint_value(card) for card in cards) - required)
    future_cards = sum(card > hideout for card in cards)
    return len(cards) + 0.5 * overpay + 0.75 * future_cards


def bounded_sprint_plans(
    hand: Iterable[int],
    previous_hideout: int,
    hideout: int,
    *,
    max_low_cost: int = DEFAULT_MAX_LOW_COST_PAYMENTS,
    max_extra_overpay: int = DEFAULT_MAX_EXTRA_OVERPAYMENTS,
) -> SprintPlanGroup:
    """Return bounded useful Sprint payments without enumerating ``2**hand``.

    Card 42 is intentionally unavailable as Sprint in this baseline.  Dynamic
    programming retains only the best few subsets for each Sprint total, then
    ranks feasible payments with :func:`sprint_cost`.
    """

    if max_low_cost < 1:
        raise ValueError("max_low_cost must be positive")
    if max_extra_overpay < 0:
        raise ValueError("max_extra_overpay must be non-negative")

    hand_tuple = tuple(sorted(hand))
    if hideout not in hand_tuple or hideout <= previous_hideout:
        return SprintPlanGroup((), ())

    candidates = tuple(
        card for card in hand_tuple if card not in (hideout, 42)
    )
    retained_per_total = max_low_cost + max_extra_overpay
    retained_per_total = max(1, retained_per_total)
    subsets: dict[int, list[tuple[int, ...]]] = {0: [()]}

    for card in candidates:
        additions: dict[int, list[tuple[int, ...]]] = {}
        value = sprint_value(card)
        for total, current_subsets in tuple(subsets.items()):
            additions.setdefault(total + value, []).extend(
                (*subset, card) for subset in current_subsets
            )
        for total, new_subsets in additions.items():
            combined = list(dict.fromkeys((*subsets.get(total, ()), *new_subsets)))
            combined.sort(
                key=lambda subset: (
                    len(subset)
                    + 0.75 * sum(card > hideout for card in subset),
                    len(subset),
                    subset,
                )
            )
            subsets[total] = combined[:retained_per_total]

    required = required_sprint_value(previous_hideout, hideout)
    feasible: list[SprintPayment] = []
    for total, payment_subsets in subsets.items():
        if total < required:
            continue
        for cards in payment_subsets:
            action = FugitiveAction(hideout, cards)
            if not is_legal_fugitive_action(
                action,
                hand_tuple,
                previous_hideout,
                allow_pass=False,
            ):
                continue
            feasible.append(
                SprintPayment(
                    cards=cards,
                    sprint_total=total,
                    overpay=total - required,
                    cost=sprint_cost(previous_hideout, hideout, cards),
                )
            )

    feasible.sort(
        key=lambda payment: (
            payment.cost,
            len(payment.cards),
            payment.overpay,
            payment.cards,
        )
    )
    if not feasible:
        return SprintPlanGroup((), ())

    # The normal branch never deliberately overpays.  An overpay can still be
    # unavoidable (for example, a required value of one with only +2 cards),
    # so compare against the smallest achievable overpay rather than zero.
    minimum_overpay = min(payment.overpay for payment in feasible)
    low_cost = tuple(
        payment for payment in feasible if payment.overpay == minimum_overpay
    )[:max_low_cost]
    low_cards = {payment.cards for payment in low_cost}
    extra_overpay = tuple(
        payment
        for payment in feasible
        if payment.cards not in low_cards and payment.overpay > minimum_overpay
    )[:max_extra_overpay]
    return SprintPlanGroup(low_cost, extra_overpay)


def normalized_play_actions(
    observation: Observation,
    *,
    max_low_cost: int = DEFAULT_MAX_LOW_COST_PAYMENTS,
    max_extra_overpay: int = DEFAULT_MAX_EXTRA_OVERPAYMENTS,
) -> dict[int, SprintPlanGroup]:
    """Group bounded legal plays by Hideout, excluding card 42 as Sprint."""

    if observation.role is not Role.FUGITIVE:
        return {}
    if observation.phase not in (Phase.FUGITIVE_OPENING, Phase.FUGITIVE_ACTION):
        return {}
    if not observation.route or observation.route[-1].hideout is None:
        return {}
    previous = observation.route[-1].hideout
    assert previous is not None

    grouped: dict[int, SprintPlanGroup] = {}
    for hideout in sorted(observation.hand):
        if hideout <= previous:
            continue
        plans = bounded_sprint_plans(
            observation.hand,
            previous,
            hideout,
            max_low_cost=max_low_cost,
            max_extra_overpay=max_extra_overpay,
        )
        if plans.low_cost:
            grouped[hideout] = plans
    return grouped


def enumerate_opening_plans(
    observation: Observation,
    *,
    max_low_cost: int = DEFAULT_MAX_LOW_COST_PAYMENTS,
    max_extra_overpay: int = DEFAULT_MAX_EXTRA_OVERPAYMENTS,
) -> OpeningPlanGroup:
    """Enumerate bounded complete two-Hideout plans for the first turn."""

    if (
        observation.role is not Role.FUGITIVE
        or observation.phase is not Phase.FUGITIVE_OPENING
        or len(observation.route) != 1
        or observation.route[0].hideout != 0
    ):
        return OpeningPlanGroup((), ())

    low_plans: list[OpeningPlan] = []
    extra_plans: list[OpeningPlan] = []
    first_groups = normalized_play_actions(
        observation,
        max_low_cost=max_low_cost,
        max_extra_overpay=max_extra_overpay,
    )
    for first_hideout, first_group in first_groups.items():
        first_choices = (
            *((payment, False) for payment in first_group.low_cost),
            *((payment, True) for payment in first_group.extra_overpay),
        )
        for first_payment, first_is_extra in first_choices:
            spent = {first_hideout, *first_payment.cards}
            remaining = tuple(card for card in observation.hand if card not in spent)
            for second_hideout in sorted(remaining):
                if second_hideout <= first_hideout:
                    continue
                second_group = bounded_sprint_plans(
                    remaining,
                    first_hideout,
                    second_hideout,
                    max_low_cost=max_low_cost,
                    max_extra_overpay=max_extra_overpay,
                )
                second_choices = (
                    *((payment, False) for payment in second_group.low_cost),
                    *((payment, True) for payment in second_group.extra_overpay),
                )
                for second_payment, second_is_extra in second_choices:
                    plan = OpeningPlan(
                        FugitiveAction(first_hideout, first_payment.cards),
                        FugitiveAction(second_hideout, second_payment.cards),
                    )
                    target = (
                        extra_plans
                        if first_is_extra or second_is_extra
                        else low_plans
                    )
                    target.append(plan)

    return OpeningPlanGroup(
        tuple(dict.fromkeys(low_plans)),
        tuple(dict.fromkeys(extra_plans)),
    )


def _hard_constraint_guess_support(
    observation: Observation,
) -> tuple[tuple[int, ...], BeliefResult | None]:
    """Return supported values and the solved belief used to find them."""

    if observation.role is not Role.MARSHAL:
        return (), None
    result = solve_observation_belief(observation)
    if result.total_paths:
        return tuple(sorted(result.marginal_counts)), result

    # Keep malformed teaching fixtures usable without pretending that known
    # cards or failed singletons are viable targets.
    unavailable = set(observation.hand)
    unavailable.update(
        slot.hideout for slot in observation.route if slot.hideout is not None
    )
    unavailable.update(
        card
        for slot in observation.route
        if slot.sprint_cards is not None
        for card in slot.sprint_cards
    )
    route_length = len(observation.route) - 1
    unavailable.update(
        record.numbers[0]
        for record in observation.guess_history
        if not record.success
        and len(record.numbers) == 1
        and record.route_length == route_length
    )
    return tuple(card for card in range(1, 42) if card not in unavailable), result


def hard_constraint_guess_numbers(observation: Observation) -> tuple[int, ...]:
    """Return values supported by at least one information-consistent route.

    This uses only the Marshal observation.  Geometry, known card ownership,
    draw availability, and the correct joint interpretation of failed guesses
    are enforced by :class:`PathBelief`; route counts are used only as a
    Boolean support test, never as action weights.
    """

    candidates, _result = _hard_constraint_guess_support(observation)
    return candidates


def _normalize(weights: Mapping[T, float]) -> dict[T, float]:
    positive = {action: weight for action, weight in weights.items() if weight > 0}
    total = sum(positive.values())
    if total <= 0:
        return {}
    return {action: weight / total for action, weight in positive.items()}


def _sample_distribution(distribution: Mapping[T, float], rng: random.Random) -> T:
    if not distribution:
        raise ValueError("cannot sample an empty distribution")
    threshold = rng.random()
    cumulative = 0.0
    last: T | None = None
    for item, probability in distribution.items():
        last = item
        cumulative += probability
        if threshold < cumulative:
            return item
    assert last is not None
    return last


def _payment_distribution(
    group: SprintPlanGroup,
    overpay_probability: float,
) -> dict[tuple[int, ...], float]:
    weights: dict[tuple[int, ...], float] = {}
    if group.extra_overpay:
        for payment in group.low_cost:
            weights[payment.cards] = (1 - overpay_probability) / len(group.low_cost)
        for payment in group.extra_overpay:
            weights[payment.cards] = overpay_probability / len(group.extra_overpay)
    else:
        for payment in group.low_cost:
            weights[payment.cards] = 1 / len(group.low_cost)
    return weights


class HierarchicalRandomFugitiveAgent:
    """HR-1 Fugitive: random macro-actions with bounded resource payments."""

    name = "hierarchical-random-fugitive"

    def __init__(
        self,
        seed: int | random.Random | None = None,
        *,
        rng: random.Random | None = None,
        overpay_probability: float = DEFAULT_OVERPAY_PROBABILITY,
        max_low_cost_payments: int = DEFAULT_MAX_LOW_COST_PAYMENTS,
        max_extra_overpayments: int = DEFAULT_MAX_EXTRA_OVERPAYMENTS,
    ) -> None:
        if not 0 <= overpay_probability <= 1:
            raise ValueError("overpay_probability must be between zero and one")
        if max_low_cost_payments < 1:
            raise ValueError("max_low_cost_payments must be positive")
        if max_extra_overpayments < 0:
            raise ValueError("max_extra_overpayments must be non-negative")
        self.rng = make_rng(seed, rng)
        self.algorithm_id = HR1_FUGITIVE_ALGORITHM_ID
        self.overpay_probability = overpay_probability
        self.max_low_cost_payments = max_low_cost_payments
        self.max_extra_overpayments = max_extra_overpayments

    def draw_pile_distribution(self, observation: Observation) -> dict[int, float]:
        """Return the HR-1 uniform distribution over nonempty legal piles."""

        piles = observation.legal_draw_piles
        if not piles:
            return {}
        probability = 1 / len(piles)
        return {pile: probability for pile in piles}

    def choose_draw_pile(self, observation: Observation) -> int:
        distribution = self.draw_pile_distribution(observation)
        if not distribution:
            raise ValueError("there is no legal draw pile")
        return _sample_distribution(distribution, self.rng)

    def fugitive_action_distribution(
        self, observation: Observation
    ) -> dict[FugitiveAction, float]:
        """Build the observable action distribution before sampling."""

        if observation.phase is Phase.FUGITIVE_OPENING:
            if len(observation.route) == 1:
                plans = enumerate_opening_plans(
                    observation,
                    max_low_cost=self.max_low_cost_payments,
                    max_extra_overpay=self.max_extra_overpayments,
                )
                plan_weights = self._opening_plan_distribution(plans)
                action_weights: dict[FugitiveAction, float] = {}
                for plan, probability in plan_weights.items():
                    action_weights[plan.first] = (
                        action_weights.get(plan.first, 0.0) + probability
                    )
                return _normalize(action_weights)
            if len(observation.route) == 2:
                return self._second_opening_distribution(observation)
            return {}

        if observation.phase is not Phase.FUGITIVE_ACTION:
            return {}
        return self._play_distribution(observation)

    def choose_fugitive_action(self, observation: Observation) -> FugitiveAction:
        distribution = self.fugitive_action_distribution(observation)
        if not distribution:
            raise ValueError("there is no legal Fugitive action")
        return _sample_distribution(distribution, self.rng)

    def _second_opening_distribution(
        self, observation: Observation
    ) -> dict[FugitiveAction, float]:
        first_slot = observation.route[1]
        if first_slot.hideout is None or first_slot.sprint_cards is None:
            return {}
        first = FugitiveAction(first_slot.hideout, first_slot.sprint_cards)
        original = Observation(
            role=observation.role,
            hand=tuple(
                sorted(
                    (
                        *observation.hand,
                        first_slot.hideout,
                        *first_slot.sprint_cards,
                    )
                )
            ),
            pile_sizes=observation.pile_sizes,
            route=(observation.route[0],),
            guess_history=observation.guess_history,
            draw_history=observation.draw_history,
            round_number=observation.round_number,
            phase=observation.phase,
            legal_draw_piles=observation.legal_draw_piles,
            play_history=(),
        )
        plans = enumerate_opening_plans(
            original,
            max_low_cost=self.max_low_cost_payments,
            max_extra_overpay=self.max_extra_overpayments,
        )
        plan_distribution = self._opening_plan_distribution(plans)
        weights: dict[FugitiveAction, float] = {}
        for plan, probability in plan_distribution.items():
            if plan.first == first:
                weights[plan.second] = weights.get(plan.second, 0.0) + probability
        return _normalize(weights)

    def _play_distribution(
        self,
        observation: Observation,
        *,
        pass_probability: float | None = None,
    ) -> dict[FugitiveAction, float]:
        groups = normalized_play_actions(
            observation,
            max_low_cost=self.max_low_cost_payments,
            max_extra_overpay=self.max_extra_overpayments,
        )
        if not groups:
            if observation.phase is Phase.FUGITIVE_ACTION:
                return {FugitiveAction(None): 1.0}
            return {}

        if pass_probability is None:
            pass_probability = self._state_dependent_pass_probability(
                observation, groups
            )
        destination_mass = (1 - pass_probability) / len(groups)
        weights: dict[FugitiveAction, float] = {}
        if pass_probability:
            weights[FugitiveAction(None)] = pass_probability
        for hideout, group in groups.items():
            for sprint_cards, conditional in _payment_distribution(
                group, self.overpay_probability
            ).items():
                weights[FugitiveAction(hideout, sprint_cards)] = (
                    destination_mass * conditional
                )
        return _normalize(weights)

    def _state_dependent_pass_probability(
        self,
        observation: Observation,
        groups: Mapping[int, SprintPlanGroup],
    ) -> float:
        previous = observation.route[-1].hideout
        assert previous is not None

        highest_revealed = max(
            (
                slot.hideout
                for slot in observation.route
                if slot.revealed and slot.hideout is not None and slot.hideout != 42
            ),
            default=0,
        )
        if 42 in groups and highest_revealed >= 30:
            return 0.0

        low_actions = (
            (hideout, payment)
            for hideout, group in groups.items()
            for payment in group.low_cost
        )
        if any(
            hideout - previous >= 2
            and all(card <= hideout for card in payment.cards)
            for hideout, payment in low_actions
        ):
            return 0.03

        all_actions = [
            (hideout, payment)
            for hideout, group in groups.items()
            for payment in group.low_cost
        ]
        if all_actions and all(
            sum(card > hideout for card in payment.cards) >= 2
            for hideout, payment in all_actions
        ):
            return 0.25
        return 0.10

    def _opening_plan_distribution(
        self, plans: OpeningPlanGroup
    ) -> dict[OpeningPlan, float]:
        weights: dict[OpeningPlan, float] = {}
        if plans.extra_overpay:
            for plan in plans.low_cost:
                weights[plan] = (1 - self.overpay_probability) / len(plans.low_cost)
            for plan in plans.extra_overpay:
                weights[plan] = self.overpay_probability / len(
                    plans.extra_overpay
                )
        elif plans.low_cost:
            for plan in plans.low_cost:
                weights[plan] = 1 / len(plans.low_cost)
        elif plans.extra_overpay:
            for plan in plans.extra_overpay:
                weights[plan] = 1 / len(plans.extra_overpay)
        return _normalize(weights)

class HierarchicalRandomMarshalAgent:
    """HR-1 Marshal: hard-support random guesses with a small-size prior."""

    name = "hierarchical-random-marshal"

    def __init__(
        self,
        seed: int | random.Random | None = None,
        *,
        rng: random.Random | None = None,
        multi_guess_continuation: float = DEFAULT_MULTI_GUESS_CONTINUATION,
        max_guess_size: int = DEFAULT_MAX_GUESS_SIZE,
        max_guesses_per_size: int = DEFAULT_MAX_GUESSES_PER_SIZE,
    ) -> None:
        if not 0 <= multi_guess_continuation <= 1:
            raise ValueError(
                "multi_guess_continuation must be between zero and one"
            )
        if max_guess_size < 1:
            raise ValueError("max_guess_size must be positive")
        if max_guesses_per_size < 1:
            raise ValueError("max_guesses_per_size must be positive")
        self.rng = make_rng(seed, rng)
        self.algorithm_id = HR1_MARSHAL_ALGORITHM_ID
        self.multi_guess_continuation = multi_guess_continuation
        self.max_guess_size = max_guess_size
        self.max_guesses_per_size = max_guesses_per_size

    def draw_pile_distribution(self, observation: Observation) -> dict[int, float]:
        """Return the HR-1 uniform distribution over nonempty legal piles."""

        piles = observation.legal_draw_piles
        if not piles:
            return {}
        probability = 1 / len(piles)
        return {pile: probability for pile in piles}

    def choose_draw_pile(self, observation: Observation) -> int:
        distribution = self.draw_pile_distribution(observation)
        if not distribution:
            raise ValueError("there is no legal draw pile")
        return _sample_distribution(distribution, self.rng)

    def guess_distribution(
        self, observation: Observation
    ) -> dict[tuple[int, ...], float]:
        """Build a deterministic distribution from Marshal-visible support."""

        candidates, belief = _hard_constraint_guess_support(observation)
        if not candidates:
            return {}
        if observation.phase is Phase.MANHUNT:
            probability = 1 / len(candidates)
            return {(card,): probability for card in candidates}

        hidden_count = sum(slot.hideout is None for slot in observation.route)
        size_limit = min(self.max_guess_size, hidden_count, len(candidates))
        if size_limit < 1:
            size_limit = 1

        assert belief is not None
        groups: dict[int, tuple[tuple[int, ...], ...]] = {
            1: tuple((card,) for card in candidates)
        }
        for size in range(2, size_limit + 1):
            guesses = self._bounded_joint_guesses(
                observation,
                belief,
                candidates,
                size,
            )
            if guesses:
                groups[size] = guesses

        size_weights: dict[int, float] = {}
        last_size = max(groups)
        continuation = self.multi_guess_continuation
        for size in groups:
            if size < last_size:
                size_weights[size] = continuation ** (size - 1) * (1 - continuation)
            else:
                size_weights[size] = continuation ** (size - 1)
        size_distribution = _normalize(size_weights)

        distribution: dict[tuple[int, ...], float] = {}
        for size, guesses in groups.items():
            if size not in size_distribution:
                continue
            per_guess = size_distribution[size] / len(guesses)
            for guess in guesses:
                distribution[guess] = per_guess
        return _normalize(distribution)

    def choose_guess(self, observation: Observation) -> tuple[int, ...]:
        distribution = self.guess_distribution(observation)
        if not distribution:
            raise ValueError("there is no information-consistent Marshal guess")
        return _sample_distribution(distribution, self.rng)

    def _bounded_joint_guesses(
        self,
        observation: Observation,
        belief: BeliefResult,
        candidates: tuple[int, ...],
        size: int,
    ) -> tuple[tuple[int, ...], ...]:
        failed_here = {
            tuple(sorted(record.numbers))
            for record in observation.guess_history
            if not record.success
            and len(record.numbers) > 1
            and record.route_length == len(observation.route) - 1
        }
        found: set[tuple[int, ...]] = set()

        # A local generator derived exclusively from the information state
        # makes the candidate action set deterministic and privacy-testable.
        local_rng = random.Random(
            stable_observation_seed(
                observation,
                domain="hr.marshal.joint-candidates.v1",
                salt=str(size),
            )
        )
        combination_count = math.comb(len(candidates), size)
        attempts = min(
            combination_count,
            max(64, self.max_guesses_per_size * 12),
        )
        tested: set[tuple[int, ...]] = set()
        for _ in range(attempts):
            guess = tuple(sorted(local_rng.sample(candidates, size)))
            if guess in tested:
                continue
            tested.add(guess)
            if guess in failed_here:
                continue
            if belief.count_paths_containing(guess):
                found.add(guess)
                if len(found) >= self.max_guesses_per_size:
                    break

        # Sparse supports can be missed by random proposals.  A deterministic
        # fallback ensures a legal joint action whenever one is easy to find.
        if not found and combination_count <= 4_096:
            for guess in itertools.combinations(candidates, size):
                if guess in failed_here:
                    continue
                if belief.count_paths_containing(guess):
                    found.add(guess)
                    if len(found) >= self.max_guesses_per_size:
                        break
        return tuple(sorted(found))


__all__ = [
    "DEFAULT_MAX_EXTRA_OVERPAYMENTS",
    "DEFAULT_MAX_GUESS_SIZE",
    "DEFAULT_MAX_GUESSES_PER_SIZE",
    "DEFAULT_MAX_LOW_COST_PAYMENTS",
    "DEFAULT_MULTI_GUESS_CONTINUATION",
    "DEFAULT_OVERPAY_PROBABILITY",
    "HR1_FUGITIVE_ALGORITHM_ID",
    "HR1_MARSHAL_ALGORITHM_ID",
    "HierarchicalRandomFugitiveAgent",
    "HierarchicalRandomMarshalAgent",
    "OpeningPlan",
    "OpeningPlanGroup",
    "SprintPayment",
    "SprintPlanGroup",
    "bounded_sprint_plans",
    "enumerate_opening_plans",
    "hard_constraint_guess_numbers",
    "normalized_play_actions",
    "sprint_cost",
]
