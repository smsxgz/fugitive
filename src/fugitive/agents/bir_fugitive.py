"""Belief-informed random policy for the Fugitive role."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
import math
import random
from typing import Callable, Iterable, Mapping

from fugitive.belief import BeliefResult, PathBelief
from fugitive.model import (
    DrawRecord,
    FugitiveAction,
    GuessRecord,
    Observation,
    Phase,
    Role,
    RouteView,
)
from fugitive.observation_protocol import stable_observation_seed
from fugitive.rules import sprint_value

from .base import make_rng
from .baseline_utils import epsilon_softmax, possible_draw_cards, sample_distribution
from .bir_common import (
    ChoiceT,
    DEFAULT_EPSILON,
    DEFAULT_MANHUNT_ALPHA,
    DEFAULT_MANHUNT_EPSILON,
    DEFAULT_TEMPERATURE,
    normalized as _normalized,
    validate_manhunt_parameters as _validate_manhunt_parameters,
)
from .hierarchical_random import (
    DEFAULT_MAX_EXTRA_OVERPAYMENTS,
    DEFAULT_MAX_LOW_COST_PAYMENTS,
    DEFAULT_OVERPAY_PROBABILITY,
    OpeningPlan,
    OpeningPlanGroup,
    SprintPayment,
    SprintPlanGroup,
    bounded_sprint_plans,
    enumerate_opening_plans,
    hard_constraint_guess_numbers,
    normalized_play_actions,
    sprint_cost,
)


DEFAULT_MANHUNT_ROLLOUTS = 32
BIR1_FUGITIVE_ALGORITHM_ID = "bir-1-fugitive-information-set-random-v1"


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


class BeliefInformedRandomFugitiveAgent:
    """BIR-1 Fugitive policy with hierarchical, epsilon-softmax choices."""

    name = "belief-informed-random-fugitive"

    def __init__(
        self,
        seed: int | random.Random | None = None,
        *,
        rng: random.Random | None = None,
        epsilon: float = DEFAULT_EPSILON,
        temperature: float = DEFAULT_TEMPERATURE,
        overpay_probability: float = DEFAULT_OVERPAY_PROBABILITY,
        max_low_cost_payments: int = DEFAULT_MAX_LOW_COST_PAYMENTS,
        max_extra_overpayments: int = DEFAULT_MAX_EXTRA_OVERPAYMENTS,
        manhunt_epsilon: float = DEFAULT_MANHUNT_EPSILON,
        manhunt_alpha: float = DEFAULT_MANHUNT_ALPHA,
        manhunt_rollouts: int = DEFAULT_MANHUNT_ROLLOUTS,
    ) -> None:
        if not math.isfinite(epsilon) or not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be between zero and one")
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if (
            not math.isfinite(overpay_probability)
            or not 0.0 <= overpay_probability <= 1.0
        ):
            raise ValueError("overpay_probability must be between zero and one")
        if (
            isinstance(max_low_cost_payments, bool)
            or not isinstance(max_low_cost_payments, int)
            or max_low_cost_payments <= 0
        ):
            raise ValueError("max_low_cost_payments must be a positive integer")
        if (
            isinstance(max_extra_overpayments, bool)
            or not isinstance(max_extra_overpayments, int)
            or max_extra_overpayments < 0
        ):
            raise ValueError("max_extra_overpayments must be a non-negative integer")
        _validate_manhunt_parameters(manhunt_epsilon, manhunt_alpha)
        if (
            isinstance(manhunt_rollouts, bool)
            or not isinstance(manhunt_rollouts, int)
            or manhunt_rollouts <= 0
        ):
            raise ValueError("manhunt_rollouts must be a positive integer")
        self.rng = make_rng(seed, rng)
        self.algorithm_id = BIR1_FUGITIVE_ALGORITHM_ID
        self.epsilon = epsilon
        self.temperature = temperature
        self.overpay_probability = overpay_probability
        self.max_low_cost_payments = max_low_cost_payments
        self.max_extra_overpayments = max_extra_overpayments
        self.manhunt_epsilon = manhunt_epsilon
        self.manhunt_alpha = manhunt_alpha
        self.manhunt_rollouts = manhunt_rollouts
        self._shadow_cache: dict[Observation, BeliefResult] = {}

    def draw_pile_distribution(self, observation: Observation) -> dict[int, float]:
        if observation.role is not Role.FUGITIVE:
            return {}
        scores: dict[int, float] = {}
        previous = observation.route[-1].hideout
        if previous is None:
            return {}
        for pile in observation.legal_draw_piles:
            support = possible_draw_cards(observation, pile)
            if not support:
                continue
            scores[pile] = sum(
                self._hand_value(
                    tuple(sorted((*observation.hand, card))), previous, observation
                )
                for card in support
            ) / len(support)
        if not scores:
            return {}
        return epsilon_softmax(
            scores, epsilon=self.epsilon, temperature=self.temperature
        )

    def choose_draw_pile(self, observation: Observation) -> int:
        distribution = self.draw_pile_distribution(observation)
        if not distribution:
            raise ValueError("there is no legal draw pile")
        return sample_distribution(distribution, rng=self.rng)

    def opening_plan_distribution(
        self, observation: Observation
    ) -> dict[OpeningPlan, float]:
        plans = enumerate_opening_plans(
            observation,
            max_low_cost=self.max_low_cost_payments,
            max_extra_overpay=self.max_extra_overpayments,
        )
        return self._opening_plan_distribution(observation, plans)

    def fugitive_action_distribution(
        self, observation: Observation
    ) -> dict[FugitiveAction, float]:
        if observation.role is not Role.FUGITIVE:
            return {}
        if observation.phase is Phase.FUGITIVE_OPENING:
            if len(observation.route) == 1:
                plans = self.opening_plan_distribution(observation)
                weights: dict[FugitiveAction, float] = defaultdict(float)
                for plan, probability in plans.items():
                    weights[plan.first] += probability
                return _normalized(weights)
            if len(observation.route) == 2:
                return self._second_opening_distribution(observation)
            return {}
        if observation.phase is not Phase.FUGITIVE_ACTION:
            return {}
        return self._normal_action_distribution(observation)

    def choose_fugitive_action(self, observation: Observation) -> FugitiveAction:
        distribution = self.fugitive_action_distribution(observation)
        if not distribution:
            raise ValueError("there is no legal Fugitive action")
        return sample_distribution(distribution, rng=self.rng)

    def _opening_plan_distribution(
        self,
        observation: Observation,
        plans: OpeningPlanGroup,
    ) -> dict[OpeningPlan, float]:
        low = self._softmax_opening_branch(observation, plans.low_cost)
        extra = self._softmax_opening_branch(observation, plans.extra_overpay)
        if not low:
            return extra
        if not extra:
            return low
        weights = {
            **{
                plan: (1.0 - self.overpay_probability) * probability
                for plan, probability in low.items()
            },
            **{
                plan: self.overpay_probability * probability
                for plan, probability in extra.items()
            },
        }
        return _normalized(weights)

    def _softmax_opening_branch(
        self,
        observation: Observation,
        plans: tuple[OpeningPlan, ...],
    ) -> dict[OpeningPlan, float]:
        if not plans:
            return {}
        scores = {plan: self._opening_score(observation, plan) for plan in plans}
        result: dict[OpeningPlan, float] = {}
        first_hideouts = sorted({plan.first.hideout for plan in plans})
        first_hideout_distribution = self._softmax_groups(
            first_hideouts,
            plans,
            scores,
            key=lambda plan: plan.first.hideout,
        )
        for first_hideout, first_hideout_probability in (
            first_hideout_distribution.items()
        ):
            first_plans = tuple(
                plan for plan in plans if plan.first.hideout == first_hideout
            )
            first_actions = tuple(sorted({plan.first for plan in first_plans}, key=repr))
            first_action_distribution = self._softmax_groups(
                first_actions,
                first_plans,
                scores,
                key=lambda plan: plan.first,
            )
            for first_action, first_action_probability in (
                first_action_distribution.items()
            ):
                descendants = tuple(
                    plan for plan in first_plans if plan.first == first_action
                )
                second_hideouts = sorted(
                    {plan.second.hideout for plan in descendants}
                )
                second_hideout_distribution = self._softmax_groups(
                    second_hideouts,
                    descendants,
                    scores,
                    key=lambda plan: plan.second.hideout,
                )
                for second_hideout, second_hideout_probability in (
                    second_hideout_distribution.items()
                ):
                    leaves = tuple(
                        plan
                        for plan in descendants
                        if plan.second.hideout == second_hideout
                    )
                    leaf_distribution = epsilon_softmax(
                        {plan: scores[plan] for plan in leaves},

                        epsilon=self.epsilon,
                        temperature=self.temperature,
                    )
                    prefix_probability = (
                        first_hideout_probability
                        * first_action_probability
                        * second_hideout_probability
                    )
                    for plan, leaf_probability in leaf_distribution.items():
                        result[plan] = prefix_probability * leaf_probability
        return _normalized(result)

    def _softmax_groups(
        self,
        choices: Iterable[ChoiceT],
        plans: tuple[OpeningPlan, ...],
        scores: Mapping[OpeningPlan, float],
        *,
        key: Callable[[OpeningPlan], ChoiceT],
    ) -> dict[ChoiceT, float]:
        group_scores = {
            choice: max(scores[plan] for plan in plans if key(plan) == choice)
            for choice in choices
        }
        return epsilon_softmax(
            group_scores,
            epsilon=self.epsilon,
            temperature=self.temperature,
        )

    def _second_opening_distribution(
        self, observation: Observation
    ) -> dict[FugitiveAction, float]:
        first_slot = observation.route[1]
        if first_slot.hideout is None or first_slot.sprint_cards is None:
            return {}
        first = FugitiveAction(first_slot.hideout, first_slot.sprint_cards)
        original_hand = tuple(
            sorted((*observation.hand, first_slot.hideout, *first_slot.sprint_cards))
        )
        original = replace(
            observation,
            hand=original_hand,
            route=(observation.route[0],),
            play_history=(),
        )
        plans = self.opening_plan_distribution(original)
        weights: dict[FugitiveAction, float] = defaultdict(float)
        for plan, probability in plans.items():
            if plan.first == first:
                weights[plan.second] += probability
        conditional = _normalized(weights)
        if conditional:
            return conditional
        return self._play_action_distribution(observation, include_pass=False)

    def _opening_score(self, observation: Observation, plan: OpeningPlan) -> float:
        assert plan.first.hideout is not None
        assert plan.second.hideout is not None
        first_cost = sprint_cost(0, plan.first.hideout, plan.first.sprint_cards)
        remaining = self._spend(observation.hand, plan.first)
        second_cost = sprint_cost(
            plan.first.hideout, plan.second.hideout, plan.second.sprint_cards
        )
        remaining = self._spend(remaining, plan.second)
        cost_scale = 2.75 * max(1, len(observation.hand) - 2)
        return (
            2.0 * plan.second.hideout / 42.0
            - clamp01((first_cost + second_cost) / cost_scale)
            + 0.5 * self._mobility(remaining, plan.second.hideout)
        )

    def _normal_action_distribution(
        self, observation: Observation
    ) -> dict[FugitiveAction, float]:
        return self._play_action_distribution(observation, include_pass=True)

    def _play_action_distribution(
        self,
        observation: Observation,
        *,
        include_pass: bool,
    ) -> dict[FugitiveAction, float]:
        groups = normalized_play_actions(
            observation,
            max_low_cost=self.max_low_cost_payments,
            max_extra_overpay=self.max_extra_overpayments,
        )
        if not groups:
            return {FugitiveAction(None): 1.0} if include_pass else {}

        macro_scores: dict[int | None, float] = {
            hideout: max(
                self._action_score(observation, FugitiveAction(hideout, payment.cards))
                for payment in group.all_payments
            )
            for hideout, group in groups.items()
        }
        if include_pass:
            macro_scores[None] = self._pass_score(observation)
        macro_distribution = epsilon_softmax(
            macro_scores, epsilon=self.epsilon, temperature=self.temperature
        )

        result: dict[FugitiveAction, float] = {}
        if include_pass:
            pass_mass = macro_distribution.get(None, 0.0)
            if pass_mass:
                result[FugitiveAction(None)] = pass_mass
        for hideout, group in groups.items():
            payment_distribution = self._payment_distribution(
                observation, hideout, group
            )
            for cards, probability in payment_distribution.items():
                result[FugitiveAction(hideout, cards)] = (
                    macro_distribution[hideout] * probability
                )
        return _normalized(result)

    def _payment_distribution(
        self,
        observation: Observation,
        hideout: int,
        group: SprintPlanGroup,
    ) -> dict[tuple[int, ...], float]:
        def branch(payments: tuple[SprintPayment, ...]) -> dict[tuple[int, ...], float]:
            if not payments:
                return {}
            scores = {
                payment.cards: self._action_score(
                    observation, FugitiveAction(hideout, payment.cards)
                )
                for payment in payments
            }
            return epsilon_softmax(
                scores, epsilon=self.epsilon, temperature=self.temperature
            )

        low = branch(group.low_cost)
        extra = branch(group.extra_overpay)
        if not low:
            return extra
        if not extra:
            return low
        result = {
            **{
                cards: (1.0 - self.overpay_probability) * probability
                for cards, probability in low.items()
            },
            **{
                cards: self.overpay_probability * probability
                for cards, probability in extra.items()
            },
        }
        return _normalized(result)

    def _action_score(
        self, observation: Observation, action: FugitiveAction
    ) -> float:
        assert action.hideout is not None
        previous = observation.route[-1].hideout
        assert previous is not None
        progress = (action.hideout - previous) / max(1, 42 - previous)
        cost = sprint_cost(previous, action.hideout, action.sprint_cards)
        cost_scale = 2.75 * max(1, len(observation.hand) - 1)
        remaining = self._spend(observation.hand, action)
        terminal = self._terminal_value(observation, action)
        return (
            2.0 * clamp01(progress)
            - clamp01(cost / cost_scale)
            + 0.6 * self._mobility(remaining, action.hideout)
            + 4.0 * terminal
        )

    def _pass_score(self, observation: Observation) -> float:
        previous = observation.route[-1].hideout
        assert previous is not None
        nonempty = tuple(
            pile for pile, size in enumerate(observation.pile_sizes) if size > 0
        )
        if not nonempty:
            return self._hand_value(observation.hand, previous, observation) - 0.5

        draw_view = replace(
            observation,
            phase=Phase.FUGITIVE_DRAW,
            legal_draw_piles=nonempty,
        )
        future_values: list[float] = []
        for pile in nonempty:
            support = possible_draw_cards(draw_view, pile)
            if support:
                future_values.append(
                    sum(
                        self._quick_hand_value(
                            tuple(sorted((*observation.hand, card))),
                            previous,
                        )
                        for card in support
                    )

                    / len(support)
                )
        catch_risk = self._public_catch_risk(observation)
        return (max(future_values) if future_values else 0.0) - 0.35 - catch_risk

    def _hand_value(
        self,
        hand: tuple[int, ...],
        previous: int,
        observation: Observation,
    ) -> float:
        scores: list[float] = []
        synthetic = replace(
            observation,
            hand=hand,
            phase=Phase.FUGITIVE_ACTION,
        )
        for hideout in hand:
            if hideout <= previous:
                continue
            group = bounded_sprint_plans(
                hand,
                previous,
                hideout,
                max_low_cost=self.max_low_cost_payments,
                max_extra_overpay=self.max_extra_overpayments,
            )
            for payment in group.low_cost[:1]:
                scores.append(
                    self._action_score(
                        synthetic, FugitiveAction(hideout, payment.cards)
                    )
                )
        if not scores:
            return -0.5
        return max(scores) + 0.25 * math.log1p(len(scores))

    def _quick_hand_value(self, hand: tuple[int, ...], previous: int) -> float:
        reachable = []
        for hideout in hand:
            if hideout <= previous:
                continue
            capacity = 3 + sum(
                sprint_value(card)
                for card in hand
                if card not in (hideout, 42)
            )
            if hideout - previous <= capacity:
                reachable.append(hideout)
        if not reachable:
            return -0.5
        progress = (max(reachable) - previous) / max(1, 42 - previous)
        return 2.0 * progress + 0.6 * len(reachable) / len(hand)

    def _terminal_value(
        self, observation: Observation, action: FugitiveAction
    ) -> float:
        if action.hideout != 42:
            return 0.0
        highest_revealed = max(
            (
                slot.hideout
                for slot in observation.route
                if slot.revealed and slot.hideout not in (None, 42)
            ),
            default=0,
        )
        if highest_revealed >= 30:
            return 1.0

        hidden = [
            slot
            for index, slot in enumerate(observation.route)
            if index and not slot.revealed and slot.hideout != 42
        ]
        if not hidden:
            return 0.0
        return self._manhunt_survival_probability(observation, action)

    def _manhunt_survival_probability(
        self,
        observation: Observation,
        action: FugitiveAction,
    ) -> float:
        """Estimate survival under sequential reveal-and-update Manhunt play."""

        truth = {
            slot.hideout: slot
            for index, slot in enumerate(observation.route)
            if index
            and not slot.revealed
            and slot.hideout not in (None, 42)
        }
        if not truth:
            return 0.0
        shadow = self._marshal_shadow(observation, action)
        salt = ",".join(str(card) for card in action.sprint_cards)
        rollout_rng = random.Random(
            stable_observation_seed(
                observation,
                domain="bir.fugitive.manhunt-rollout.v1",
                salt=f"42:{salt}",
            )
        )
        caught = 0
        for _ in range(self.manhunt_rollouts):
            current = shadow
            remaining = set(truth)
            while remaining:
                distribution = self._shadow_manhunt_number_distribution(current)
                if not distribution:
                    break
                guess = sample_distribution(distribution, rng=rollout_rng)
                if guess not in remaining:
                    break
                remaining.remove(guess)
                if not remaining:
                    caught += 1
                    break
                current = self._shadow_after_manhunt_reveal(
                    current,
                    truth[guess],
                )
        return clamp01(1.0 - caught / self.manhunt_rollouts)

    def _shadow_manhunt_number_distribution(
        self,
        shadow: Observation,
    ) -> dict[int, float]:
        result = self._shadow_result(shadow)
        if result.total_paths and result.marginals:
            powered = {
                number: probability**self.manhunt_alpha
                for number, probability in result.marginals.items()
            }
            total = sum(powered.values())
            count = len(powered)
            return {
                number: self.manhunt_epsilon / count
                + (1.0 - self.manhunt_epsilon) * weight / total
                for number, weight in powered.items()
            }
        candidates = hard_constraint_guess_numbers(shadow)
        if not candidates:
            return {}
        probability = 1.0 / len(candidates)
        return {number: probability for number in candidates}

    @staticmethod
    def _shadow_after_manhunt_reveal(
        shadow: Observation,
        truth: RouteView,
    ) -> Observation:
        route = list(shadow.route)
        route[truth.index] = RouteView(
            index=truth.index,
            hideout=truth.hideout,
            sprint_count=truth.sprint_count,
            sprint_cards=truth.sprint_cards,
            revealed=True,
        )
        assert truth.hideout is not None
        guess = GuessRecord(
            numbers=(truth.hideout,),
            success=True,
            route_length=len(shadow.route) - 1,
            round_number=shadow.round_number,
            manhunt=True,
        )
        return replace(
            shadow,
            route=tuple(route),
            guess_history=(*shadow.guess_history, guess),
        )

    def _public_catch_risk(self, observation: Observation) -> float:
        hidden = tuple(
            slot.hideout
            for index, slot in enumerate(observation.route)
            if index and not slot.revealed and slot.hideout not in (None, 42)
        )
        if not hidden:
            return 1.0
        result = self._shadow_result(self._marshal_shadow(observation))
        if not result.total_paths:
            return 1.0 / max(2, 3 * len(hidden))
        return clamp01(
            result.count_paths_containing(hidden) / result.total_paths
        )

    def _shadow_result(self, shadow: Observation) -> BeliefResult:
        cached = self._shadow_cache.get(shadow)
        if cached is not None:
            return cached
        result = PathBelief.from_observation(shadow).solve()
        if len(self._shadow_cache) >= 256:
            self._shadow_cache.pop(next(iter(self._shadow_cache)))
        self._shadow_cache[shadow] = result
        return result


    @staticmethod
    def _marshal_shadow(
        observation: Observation,
        action: FugitiveAction | None = None,
    ) -> Observation:
        route = tuple(
            RouteView(
                index=slot.index,
                hideout=(
                    slot.hideout
                    if slot.revealed or slot.hideout in (0, 42)
                    else None
                ),
                sprint_count=slot.sprint_count,
                sprint_cards=(
                    slot.sprint_cards
                    if slot.revealed and slot.hideout != 42
                    else None
                ),
                revealed=slot.revealed,
            )
            for slot in observation.route
        )
        if action is not None:
            assert action.hideout == 42
            route = (
                *route,
                RouteView(
                    index=len(route),
                    hideout=42,
                    sprint_count=len(action.sprint_cards),
                    sprint_cards=None,
                    revealed=False,
                ),
            )
        return Observation(
            role=Role.MARSHAL,
            hand=(),
            pile_sizes=observation.pile_sizes,
            route=route,
            guess_history=observation.guess_history,
            draw_history=tuple(
                DrawRecord(
                    role=record.role,
                    pile=record.pile,
                    card=None,
                    round_number=record.round_number,
                )
                for record in observation.draw_history
            ),
            round_number=observation.round_number,
            phase=Phase.MANHUNT if action is not None else Phase.MARSHAL_GUESS,
            legal_draw_piles=(),
            play_history=(),
        )

    def _mobility(self, hand: tuple[int, ...], previous: int) -> float:
        candidates = tuple(card for card in hand if card > previous)
        if not candidates or previous >= 42:
            return 0.0
        feasible = tuple(
            hideout
            for hideout in candidates
            if hideout - previous
            <= 3
            + sum(
                sprint_value(card)
                for card in hand
                if card not in (hideout, 42)
            )
        )
        if not feasible:
            return 0.0
        count_score = len(feasible) / len(candidates)
        reach_score = (max(feasible) - previous) / max(1, 42 - previous)
        return clamp01(0.5 * count_score + 0.5 * reach_score)

    @staticmethod
    def _spend(hand: Iterable[int], action: FugitiveAction) -> tuple[int, ...]:
        spent = {action.hideout, *action.sprint_cards}
        return tuple(sorted(card for card in hand if card not in spent))


__all__ = [
    "BIR1_FUGITIVE_ALGORITHM_ID",
    "BeliefInformedRandomFugitiveAgent",
    "DEFAULT_MANHUNT_ROLLOUTS",
    "clamp01",
]
