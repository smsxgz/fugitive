"""Information-set random baselines for both roles.

The policies here stay stochastic, but randomize over a small hierarchy of
meaningful decisions.  Every distribution method consumes only an immutable
``Observation``.  The Marshal's private-world hypotheses are sampled by
``MarshalParticleBelief``; the engine and its deck order are never inspected.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
import hashlib
import itertools
import math
import random
from typing import Callable, Hashable, Iterable, Mapping, TypeVar

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
from fugitive.observation_protocol import (
    canonical_random_state_bytes,
    stable_observation_seed,
)
from fugitive.particle_belief import (
    DEFAULT_PARTICLE_COUNT,
    MarshalDrawOutcomeStatistics,
    MarshalParticleBelief,
)
from fugitive.rules import sprint_value

from .base import make_rng
from .baseline_utils import epsilon_softmax, possible_draw_cards, sample_distribution
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


DEFAULT_EPSILON = 0.15
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MANHUNT_EPSILON = 0.10
DEFAULT_MANHUNT_ALPHA = 2.0
DEFAULT_MANHUNT_ROLLOUTS = 32
DEFAULT_MAX_GUESS_CANDIDATES = 128
DEFAULT_MIN_UNIQUE_PARTICLE_FRACTION = 0.01


ChoiceT = TypeVar("ChoiceT", bound=Hashable)


@dataclass(frozen=True, slots=True)
class MarshalBeliefDiagnostics:
    """Observable counters for incremental particle-belief maintenance."""

    incremental_updates: int
    fresh_resamples: int
    minimum_unique_particles: int
    last_incremental_particle_count: int | None
    last_incremental_unique_particle_count: int | None
    last_recovery_reason: str | None
    particle_count: int
    unique_particle_count: int
    effective_sample_size: float
    sampling_attempts: int
    sampling_acceptance_rate: float
    sampling_exhausted: bool


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _normalized(weights: Mapping[ChoiceT, float]) -> dict[ChoiceT, float]:
    positive = {
        action: weight for action, weight in weights.items() if weight > 0.0
    }
    total = sum(positive.values())
    if total <= 0.0:
        return {}
    return {action: weight / total for action, weight in positive.items()}


def _stable_seed(observation: Observation, salt: str) -> int:
    return stable_observation_seed(
        observation,
        domain="bir.marshal.particle-belief.v1",
        salt=salt,
    )


def _validate_manhunt_parameters(epsilon: float, alpha: float) -> None:
    if not math.isfinite(epsilon) or not 0.0 <= epsilon <= 1.0:
        raise ValueError("manhunt_epsilon must be between zero and one")
    if not math.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("manhunt_alpha must be positive")


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
            - _clamp01((first_cost + second_cost) / cost_scale)
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
            2.0 * _clamp01(progress)
            - _clamp01(cost / cost_scale)
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
        return _clamp01(1.0 - caught / self.manhunt_rollouts)

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
        return _clamp01(
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
        return _clamp01(0.5 * count_score + 0.5 * reach_score)

    @staticmethod
    def _spend(hand: Iterable[int], action: FugitiveAction) -> tuple[int, ...]:
        spent = {action.hideout, *action.sprint_cards}
        return tuple(sorted(card for card in hand if card not in spent))


class BeliefInformedRandomMarshalAgent:
    """BIR-1 Marshal using observation-only complete-world particles."""

    name = "belief-informed-random-marshal"

    def __init__(
        self,
        seed: int | random.Random | None = None,
        *,
        rng: random.Random | None = None,
        epsilon: float = DEFAULT_EPSILON,
        temperature: float = DEFAULT_TEMPERATURE,
        particle_count: int = DEFAULT_PARTICLE_COUNT,
        min_unique_particles: int | None = None,
        max_guess_candidates: int = DEFAULT_MAX_GUESS_CANDIDATES,
        terminal_bonus_scale: float = 1.0,
        manhunt_epsilon: float = DEFAULT_MANHUNT_EPSILON,
        manhunt_alpha: float = DEFAULT_MANHUNT_ALPHA,
    ) -> None:
        if not math.isfinite(epsilon) or not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be between zero and one")
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if (
            isinstance(particle_count, bool)
            or not isinstance(particle_count, int)
            or particle_count <= 0
        ):
            raise ValueError("particle_count must be a positive integer")
        if (
            isinstance(max_guess_candidates, bool)
            or not isinstance(max_guess_candidates, int)
            or not 1 <= max_guess_candidates <= DEFAULT_MAX_GUESS_CANDIDATES
        ):
            raise ValueError("max_guess_candidates must be from 1 through 128")
        if not math.isfinite(terminal_bonus_scale) or terminal_bonus_scale < 0.0:
            raise ValueError("terminal_bonus_scale must be finite and non-negative")
        _validate_manhunt_parameters(manhunt_epsilon, manhunt_alpha)
        if min_unique_particles is None:
            min_unique_particles = min(
                particle_count,
                max(
                    2,
                    math.ceil(
                        particle_count * DEFAULT_MIN_UNIQUE_PARTICLE_FRACTION
                    ),
                ),
            )
        elif (
            isinstance(min_unique_particles, bool)
            or not isinstance(min_unique_particles, int)
            or not 1 <= min_unique_particles <= particle_count
        ):
            raise ValueError(
                "min_unique_particles must be from 1 through particle_count"
            )
        self.rng = make_rng(seed, rng)
        self.epsilon = epsilon
        self.temperature = temperature
        self.particle_count = particle_count
        self.min_unique_particles = min_unique_particles
        self.max_guess_candidates = max_guess_candidates
        self.terminal_bonus_scale = terminal_bonus_scale
        self.manhunt_epsilon = manhunt_epsilon
        self.manhunt_alpha = manhunt_alpha
        self._belief_salt = hashlib.sha256(
            canonical_random_state_bytes(self.rng)
        ).hexdigest()
        self._belief_cache: dict[Observation, MarshalParticleBelief] = {}
        self._latest_belief: MarshalParticleBelief | None = None
        self._belief_incremental_updates = 0
        self._belief_fresh_resamples = 0
        self._last_incremental_particle_count: int | None = None
        self._last_incremental_unique_particle_count: int | None = None
        self._last_recovery_reason: str | None = None

    @property
    def belief_diagnostics(self) -> MarshalBeliefDiagnostics:
        current = self._latest_belief
        return MarshalBeliefDiagnostics(
            incremental_updates=self._belief_incremental_updates,
            fresh_resamples=self._belief_fresh_resamples,
            minimum_unique_particles=self.min_unique_particles,
            last_incremental_particle_count=(
                self._last_incremental_particle_count
            ),
            last_incremental_unique_particle_count=(
                self._last_incremental_unique_particle_count
            ),
            last_recovery_reason=self._last_recovery_reason,
            particle_count=0 if current is None else len(current.particles),
            unique_particle_count=(
                0 if current is None else current.unique_particle_count
            ),
            effective_sample_size=(
                0.0 if current is None else current.effective_sample_size
            ),
            sampling_attempts=(
                0 if current is None else current.sampling_attempts
            ),
            sampling_acceptance_rate=(
                0.0 if current is None else current.sampling_acceptance_rate
            ),
            sampling_exhausted=(
                False if current is None else current.sampling_exhausted
            ),
        )

    def belief(self, observation: Observation) -> MarshalParticleBelief:
        if observation.role is not Role.MARSHAL:
            raise ValueError("Marshal belief requires a Marshal observation")
        cached = self._belief_cache.get(observation)
        if cached is not None:
            self._latest_belief = cached
            return cached

        belief_seed = _stable_seed(observation, self._belief_salt)
        sampled: MarshalParticleBelief
        recovery_reason: str | None = None
        if self._latest_belief is not None:
            try:
                sampled = self._latest_belief.advance_to(
                    observation,
                    particle_count=self.particle_count,
                    seed=belief_seed,
                )
            except ValueError:
                recovery_reason = "incompatible_observation"
            else:
                self._belief_incremental_updates += 1
                self._last_incremental_particle_count = len(sampled.particles)
                self._last_incremental_unique_particle_count = (
                    sampled.unique_particle_count
                )
                if sampled.is_empty:
                    recovery_reason = "empty"
                elif sampled.unique_particle_count < self.min_unique_particles:
                    recovery_reason = "low_unique"

            if recovery_reason is not None:
                sampled = MarshalParticleBelief.from_observation(
                    observation,
                    particle_count=self.particle_count,
                    seed=belief_seed,
                )
                self._belief_fresh_resamples += 1
                self._last_recovery_reason = recovery_reason
        else:
            sampled = MarshalParticleBelief.from_observation(
                observation,
                particle_count=self.particle_count,
                seed=belief_seed,
            )
        if len(self._belief_cache) >= 8:
            self._belief_cache.pop(next(iter(self._belief_cache)))
        self._belief_cache[observation] = sampled
        self._latest_belief = sampled
        return sampled

    def draw_pile_distribution(self, observation: Observation) -> dict[int, float]:
        if observation.role is not Role.MARSHAL:
            return {}
        belief = self.belief(observation)
        if belief.is_empty:
            if not observation.legal_draw_piles:
                return {}
            probability = 1.0 / len(observation.legal_draw_piles)
            return {pile: probability for pile in observation.legal_draw_piles}

        outcome_statistics = belief.draw_outcome_statistics()
        scores: dict[int, float] = {}
        for pile in observation.legal_draw_piles:
            outcomes = outcome_statistics[pile]
            if not outcomes:
                continue
            scores[pile] = sum(
                outcome.probability
                * self._best_guess_score_from_statistics(observation, outcome)
                for outcome in outcomes.values()
            )
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

    def candidate_guess_sets(
        self,
        observation: Observation,
        belief: MarshalParticleBelief | None = None,
    ) -> tuple[tuple[int, ...], ...]:
        belief = belief or self.belief(observation)
        if belief.is_empty:
            return tuple((number,) for number in hard_constraint_guess_numbers(observation))

        hidden_mass: dict[tuple[int, ...], float] = defaultdict(float)
        for particle in belief.particles:
            hidden = tuple(sorted(belief.current_hidden_hideouts(particle)))
            if hidden:
                hidden_mass[hidden] += particle.weight
        return self._candidate_guess_sets_from_masses(
            belief.marginals,
            hidden_mass,
        )

    def _candidate_guess_sets_from_masses(
        self,
        marginals: Mapping[int, float],
        hidden_mass: Mapping[tuple[int, ...], float],
    ) -> tuple[tuple[int, ...], ...]:
        ordered_singles = [
            (number,)
            for number, _probability in sorted(
                marginals.items(), key=lambda item: (-item[1], item[0])
            )
        ]
        single_budget = (
            len(ordered_singles)
            if len(ordered_singles) < self.max_guess_candidates
            else max(1, self.max_guess_candidates // 2)
        )
        candidates: list[tuple[int, ...]] = ordered_singles[:single_budget]
        seen = set(candidates)
        if len(candidates) >= self.max_guess_candidates:
            return tuple(candidates)

        likely_routes = sorted(
            hidden_mass,
            key=lambda route: (-hidden_mass[route], len(route), route),
        )

        def add(guess: Iterable[int]) -> bool:
            item = tuple(sorted(guess))
            if not item or item in seen:
                return False
            candidates.append(item)
            seen.add(item)
            return len(candidates) >= self.max_guess_candidates

        for route in likely_routes[:24]:
            if len(route) > 1 and add(route):
                return tuple(candidates)
            for size in range(2, min(4, len(route)) + 1):
                for subset in itertools.combinations(route, size):
                    if add(subset):
                        return tuple(candidates)

        return tuple(candidates)

    def guess_distribution(
        self, observation: Observation
    ) -> dict[tuple[int, ...], float]:
        if observation.role is not Role.MARSHAL:
            return {}
        belief = self.belief(observation)
        if observation.phase is Phase.MANHUNT:
            return self._manhunt_distribution(observation, belief)
        if observation.phase is not Phase.MARSHAL_GUESS:
            return {}

        candidates = self.candidate_guess_sets(observation, belief)
        if not candidates:
            return {}
        if belief.is_empty:
            probability = 1.0 / len(candidates)
            return {guess: probability for guess in candidates}

        grouped: dict[int, list[tuple[int, ...]]] = defaultdict(list)
        scores: dict[tuple[int, ...], float] = {}
        failure_cost = self._escape_risk_after_draw(belief)
        hidden_count = self._hidden_count(observation)
        for guess in candidates:
            grouped[len(guess)].append(guess)
            scores[guess] = self._guess_score(
                observation,
                belief,
                guess,
                failure_cost=failure_cost,
                hidden_count=hidden_count,
            )
        size_scores = {
            size: max(scores[guess] for guess in guesses)
            for size, guesses in grouped.items()
        }
        size_distribution = epsilon_softmax(
            size_scores, epsilon=self.epsilon, temperature=self.temperature
        )
        result: dict[tuple[int, ...], float] = {}
        for size, guesses in grouped.items():
            conditional = epsilon_softmax(
                {guess: scores[guess] for guess in guesses},
                epsilon=self.epsilon,
                temperature=self.temperature,
            )
            for guess, probability in conditional.items():
                result[guess] = size_distribution[size] * probability
        return _normalized(result)

    def choose_guess(self, observation: Observation) -> tuple[int, ...]:
        distribution = self.guess_distribution(observation)
        if not distribution:
            raise ValueError("there is no information-consistent Marshal guess")
        return sample_distribution(distribution, rng=self.rng)

    def _best_guess_score(
        self,
        observation: Observation,
        belief: MarshalParticleBelief,
    ) -> float:
        candidates = self.candidate_guess_sets(observation, belief)
        if not candidates or belief.is_empty:
            return 0.0
        failure_cost = self._escape_risk_after_draw(belief)
        hidden_count = self._hidden_count(observation)
        return max(
            self._guess_score(
                observation,
                belief,
                guess,
                failure_cost=failure_cost,
                hidden_count=hidden_count,
            )
            for guess in candidates
        )

    def _best_guess_score_from_statistics(
        self,
        observation: Observation,
        statistics: MarshalDrawOutcomeStatistics,
    ) -> float:
        candidates = self._candidate_guess_sets_from_masses(
            statistics.marginals,
            statistics.hidden_route_masses,
        )
        if not candidates:
            return 0.0
        hidden_count = self._hidden_count(observation)
        return max(
            self._guess_score_from_success(
                guess,
                statistics.joint_success(guess),
                failure_cost=statistics.escape_risk,
                hidden_count=hidden_count,
            )
            for guess in candidates
        )

    def _guess_score(
        self,
        observation: Observation,
        belief: MarshalParticleBelief,
        guess: tuple[int, ...],
        *,
        failure_cost: float | None = None,
        hidden_count: int | None = None,
    ) -> float:
        success = belief.joint_success(guess)
        if hidden_count is None:
            hidden_count = self._hidden_count(observation)
        if failure_cost is None:
            failure_cost = self._escape_risk_after_draw(belief)
        return self._guess_score_from_success(
            guess,
            success,
            failure_cost=failure_cost,
            hidden_count=hidden_count,
        )

    def _guess_score_from_success(
        self,
        guess: tuple[int, ...],
        success: float,
        *,
        failure_cost: float,
        hidden_count: int,
    ) -> float:
        terminal_bonus = (
            self.terminal_bonus_scale * hidden_count
            if len(guess) == hidden_count
            else 0.0
        )
        return (
            len(guess) * success
            + terminal_bonus * success
            - failure_cost * (1.0 - success)
        )

    @staticmethod
    def _escape_risk_after_draw(belief: MarshalParticleBelief) -> float:
        if belief.is_empty:
            return 0.0

        pile_risks = [0.0, 0.0, 0.0]
        any_nonempty = False
        immediate_risk = 0.0
        for particle in belief.particles:
            hand = particle.fugitive_hand
            previous = particle.route_hideouts[-1]

            def can_escape(cards: tuple[int, ...]) -> bool:
                return (
                    42 in cards
                    and previous != 42
                    and 42 - previous
                    <= 3
                    + sum(
                        sprint_value(card) for card in cards if card != 42
                    )
                )

            if can_escape(hand):
                immediate_risk += particle.weight
            for pile, contents in enumerate(particle.remaining_piles):
                if not contents:
                    continue
                any_nonempty = True
                favorable = sum(
                    can_escape(tuple(sorted((*hand, card)))) for card in contents
                )
                pile_risks[pile] += (
                    particle.weight * favorable / len(contents)
                )
        return max(pile_risks) if any_nonempty else immediate_risk

    @staticmethod
    def _hidden_count(observation: Observation) -> int:
        return sum(
            index > 0 and not slot.revealed and slot.hideout != 42
            for index, slot in enumerate(observation.route)
        )

    def _manhunt_distribution(
        self,
        observation: Observation,
        belief: MarshalParticleBelief,
    ) -> dict[tuple[int, ...], float]:
        if belief.is_empty:
            candidates = hard_constraint_guess_numbers(observation)
            if not candidates:
                return {}
            probability = 1.0 / len(candidates)
            return {(number,): probability for number in candidates}
        marginals = belief.marginals
        if not marginals:
            return {}
        powered = {
            number: probability**self.manhunt_alpha
            for number, probability in marginals.items()
        }
        total = sum(powered.values())
        count = len(powered)
        return {
            (number,): self.manhunt_epsilon / count
            + (1.0 - self.manhunt_epsilon) * weight / total
            for number, weight in powered.items()
        }


__all__ = [
    "BeliefInformedRandomFugitiveAgent",
    "BeliefInformedRandomMarshalAgent",
    "DEFAULT_EPSILON",
    "DEFAULT_MANHUNT_ALPHA",
    "DEFAULT_MANHUNT_EPSILON",
    "DEFAULT_MANHUNT_ROLLOUTS",
    "DEFAULT_MAX_GUESS_CANDIDATES",
    "DEFAULT_MIN_UNIQUE_PARTICLE_FRACTION",
    "DEFAULT_TEMPERATURE",
    "MarshalBeliefDiagnostics",
]
