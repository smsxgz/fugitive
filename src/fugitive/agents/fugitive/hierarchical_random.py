"""Hierarchical legal-random Fugitive policy and action construction.

The policy randomizes over rule-level decisions rather than the engine's flat
action encoding: it selects a Hideout before selecting a bounded Sprint
payment.  The Marshal baseline lives in :mod:`fugitive.agents.marshal`.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Iterable, Mapping

from fugitive.game.model import FugitiveAction, Observation, Phase, Role
from fugitive.game.rules import is_legal_fugitive_action, required_sprint_value, sprint_value

from ..common.base import make_rng
from ..common.baseline_utils import normalize_distribution, sample_distribution


DEFAULT_MAX_LOW_COST_PAYMENTS = 8
DEFAULT_MAX_EXTRA_OVERPAYMENTS = 8
DEFAULT_OVERPAY_PROBABILITY = 0.05
HR1_FUGITIVE_ALGORITHM_ID = "hr-1-fugitive-hierarchical-legal-random-v1"


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
        return sample_distribution(distribution, rng=self.rng)

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
                return normalize_distribution(action_weights)
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
        return sample_distribution(distribution, rng=self.rng)

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
        return normalize_distribution(weights)

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
        return normalize_distribution(weights)

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
        return normalize_distribution(weights)

__all__ = [
    "DEFAULT_MAX_EXTRA_OVERPAYMENTS",
    "DEFAULT_MAX_LOW_COST_PAYMENTS",
    "DEFAULT_OVERPAY_PROBABILITY",
    "HR1_FUGITIVE_ALGORITHM_ID",
    "HierarchicalRandomFugitiveAgent",
    "OpeningPlan",
    "OpeningPlanGroup",
    "SprintPayment",
    "SprintPlanGroup",
    "bounded_sprint_plans",
    "enumerate_opening_plans",
    "normalized_play_actions",
    "sprint_cost",
]
