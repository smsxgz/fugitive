"""Fugitive policy that values a small action shortlist by full-game rollout."""

from __future__ import annotations

import hashlib
import math
import random
from typing import Iterable, Mapping

from fugitive.game.driver import DrawDecision
from fugitive.game.model import FugitiveAction, Observation, Phase, Role
from fugitive.game.observation import (
    canonical_random_state_bytes,
    stable_observation_seed,
)
from ..planning.rollout import (
    FugitiveWorld,
    RolloutBudget,
    RolloutEvaluator,
    RolloutPlan,
)
from ..planning.rollout_diagnostics import RolloutDiagnosticsSnapshot
from fugitive.game.rules import PILE_CARDS

from ..common.base import make_rng
from ..common.baseline_utils import epsilon_softmax, sample_distribution
from ..common.bir_common import (
    DEFAULT_EPSILON,
    DEFAULT_MANHUNT_ALPHA,
    DEFAULT_MANHUNT_EPSILON,
    DEFAULT_TEMPERATURE,
)
from .bir import (
    DEFAULT_MANHUNT_ROLLOUTS,
    BeliefInformedRandomFugitiveAgent,
)
from .continuation_count import (
    DEFAULT_CATCH_RISK_WEIGHT,
    DEFAULT_CONCEALMENT_WEIGHT,
    DEFAULT_CONTINUATION_DEPTH,
    DEFAULT_CONTINUATION_WEIGHT,
    ContinuationCountFugitiveAgent,
)
from .hierarchical_random import (
    DEFAULT_MAX_EXTRA_OVERPAYMENTS,
    DEFAULT_MAX_LOW_COST_PAYMENTS,
    DEFAULT_OVERPAY_PROBABILITY,
)
from ..planning.policy_mixture import (
    RegisteredPolicyMixtureFactory,
    WeightedPolicySpec,
    default_policy_spec,
)


BELIEF_ROLLOUT_FUGITIVE_ALGORITHM_ID = "belief-rollout-fugitive-budgeted-v2"
ROLLOUT_MODEL_ID = "root-sampled-budgeted-full-game-crn-v2"
DEFAULT_MAX_TERMINAL_SIMULATIONS = 192
DEFAULT_ROLLOUT_CANDIDATE_COUNT = 8
DEFAULT_ROLLOUT_EPSILON = 0.05
DEFAULT_ROLLOUT_TEMPERATURE = 0.25


def _default_marshal_mixture() -> tuple[WeightedPolicySpec, ...]:
    """A non-recursive default model; PSRO supplies its own frozen mixture."""

    return (
        default_policy_spec("support-catalogue-random", Role.MARSHAL),
        default_policy_spec("route-count-catalogue-random", Role.MARSHAL),
    )


class BeliefRolloutFugitiveAgent:
    """Rank public-information candidates using sampled full-game outcomes.

    The proposal policy is the continuation-count Fugitive.  Rollout worlds
    sample only information hidden from the Fugitive: which publicly drawn
    pile cards are in the Marshal hand and the order of each remaining pile.
    Continuation agents receive ordinary role observations, never a world.
    """

    name = "belief-rollout-fugitive"

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
        continuation_weight: float = DEFAULT_CONTINUATION_WEIGHT,
        concealment_weight: float = DEFAULT_CONCEALMENT_WEIGHT,
        catch_risk_weight: float = DEFAULT_CATCH_RISK_WEIGHT,
        continuation_depth: int = DEFAULT_CONTINUATION_DEPTH,
        rollout_candidate_count: int = DEFAULT_ROLLOUT_CANDIDATE_COUNT,
        max_terminal_simulations: int = DEFAULT_MAX_TERMINAL_SIMULATIONS,
        rollout_epsilon: float = DEFAULT_ROLLOUT_EPSILON,
        rollout_temperature: float = DEFAULT_ROLLOUT_TEMPERATURE,
        opponent_policies: Iterable[
            WeightedPolicySpec | Mapping[str, object]
        ] | None = None,
    ) -> None:
        self.rng = make_rng(seed, rng)
        constructor_state = canonical_random_state_bytes(self.rng)
        base_seed = int.from_bytes(
            hashlib.sha256(
                b"belief-rollout-fugitive.constructor.v1\0" + constructor_state
            ).digest()[:8],
            "big",
        )
        base_parameters = {
            "epsilon": epsilon,
            "temperature": temperature,
            "overpay_probability": overpay_probability,
            "max_low_cost_payments": max_low_cost_payments,
            "max_extra_overpayments": max_extra_overpayments,
            "manhunt_epsilon": manhunt_epsilon,
            "manhunt_alpha": manhunt_alpha,
            "manhunt_rollouts": manhunt_rollouts,
        }
        self.base_agent = ContinuationCountFugitiveAgent(
            base_seed,
            **base_parameters,
            continuation_weight=continuation_weight,
            concealment_weight=concealment_weight,
            catch_risk_weight=catch_risk_weight,
            continuation_depth=continuation_depth,
        )
        # Expose every registered policy parameter for complete AgentSpec replay.
        for name, value in {
            **base_parameters,
            "continuation_weight": continuation_weight,
            "concealment_weight": concealment_weight,
            "catch_risk_weight": catch_risk_weight,
            "continuation_depth": continuation_depth,
        }.items():
            setattr(self, name, getattr(self.base_agent, name, value))

        self.rollout_candidate_count = _positive_int(
            "rollout_candidate_count", rollout_candidate_count
        )
        self.rollout_epsilon = _probability("rollout_epsilon", rollout_epsilon)
        if (
            isinstance(rollout_temperature, bool)
            or not isinstance(rollout_temperature, (int, float))
            or not math.isfinite(rollout_temperature)
            or rollout_temperature <= 0.0
        ):
            raise ValueError("rollout_temperature must be finite and positive")
        self.rollout_temperature = float(rollout_temperature)
        self.rollout_budget = RolloutBudget(max_terminal_simulations)
        self.max_terminal_simulations = (
            self.rollout_budget.max_terminal_simulations
        )

        choices = (
            _default_marshal_mixture()
            if opponent_policies is None
            else tuple(opponent_policies)
        )
        self._opponent_factory = RegisteredPolicyMixtureFactory(
            Role.MARSHAL, choices
        )
        self.opponent_policies = self._opponent_factory.serialized_choices
        self.algorithm_id = BELIEF_ROLLOUT_FUGITIVE_ALGORITHM_ID
        self.world_model_id = "uniform-marshal-hand-by-public-pile-count-v1"
        self.rollout_model_id = ROLLOUT_MODEL_ID
        self.rollout_leaf_fugitive_id = (
            "bir-1-fugitive-information-set-random-v1"
        )
        self._rollout_salt = hashlib.sha256(constructor_state).hexdigest()
        self._distribution_cache: dict[Observation, dict[object, float]] = {}
        self._diagnostics_cache: dict[
            Observation, RolloutDiagnosticsSnapshot
        ] = {}
        self._last_rollout_diagnostics: RolloutDiagnosticsSnapshot | None = None

    def draw_pile_distribution(self, observation: Observation) -> dict[int, float]:
        if observation.role is not Role.FUGITIVE or observation.phase is not Phase.FUGITIVE_DRAW:
            return {}
        cached = self._distribution_cache.get(observation)
        if cached is not None:
            self._last_rollout_diagnostics = self._diagnostics_cache.get(
                observation
            )
            return {int(action): probability for action, probability in cached.items()}
        base = self.base_agent.draw_pile_distribution(observation)
        candidates = tuple(DrawDecision(pile) for pile, _ in _top_items(base, len(base)))
        distribution = {
            action.pile: probability
            for action, probability in self._rollout_distribution(
                observation, candidates
            ).items()
            if isinstance(action, DrawDecision)
        }
        self._cache(observation, distribution)
        return distribution

    def choose_draw_pile(self, observation: Observation) -> int:
        distribution = self.draw_pile_distribution(observation)
        if not distribution:
            raise ValueError("there is no rollout-supported Fugitive draw pile")
        return sample_distribution(distribution, rng=self.rng)

    def fugitive_action_distribution(
        self, observation: Observation
    ) -> dict[FugitiveAction, float]:
        if observation.role is not Role.FUGITIVE or observation.phase not in (
            Phase.FUGITIVE_OPENING,
            Phase.FUGITIVE_ACTION,
        ):
            return {}
        cached = self._distribution_cache.get(observation)
        if cached is not None:
            self._last_rollout_diagnostics = self._diagnostics_cache.get(
                observation
            )
            return {
                action: probability
                for action, probability in cached.items()
                if isinstance(action, FugitiveAction)
            }
        base = self.base_agent.fugitive_action_distribution(observation)
        candidates = tuple(
            action
            for action, _ in _top_items(base, self.rollout_candidate_count)
        )
        distribution = {
            action: probability
            for action, probability in self._rollout_distribution(
                observation, candidates
            ).items()
            if isinstance(action, FugitiveAction)
        }
        self._cache(observation, distribution)
        return distribution

    def choose_fugitive_action(self, observation: Observation) -> FugitiveAction:
        distribution = self.fugitive_action_distribution(observation)
        if not distribution:
            raise ValueError("there is no rollout-supported Fugitive action")
        return sample_distribution(distribution, rng=self.rng)

    def _rollout_distribution(self, observation: Observation, candidates):
        if not candidates:
            return {}
        plan, candidates = self._rollout_plan(tuple(candidates))
        if plan.single_candidate_shortcut:
            action = candidates[0]
            self._last_rollout_diagnostics = RolloutDiagnosticsSnapshot.shortcut(
                algorithm_id=self.algorithm_id,
                rollout_model_id=self.rollout_model_id,
                role=Role.FUGITIVE,
                plan=plan,
                action=action,
            )
            return {action: 1.0}

        world_count = plan.paired_scenarios
        world_rng = random.Random(
            stable_observation_seed(
                observation,
                domain="belief-rollout-fugitive.worlds.v1",
                salt=self._rollout_salt,
            )
        )
        worlds = sample_fugitive_worlds(
            observation,
            count=world_count,
            rng=world_rng,
        )
        evaluator = RolloutEvaluator(
            lambda rollout_seed: BeliefInformedRandomFugitiveAgent(
                rollout_seed,
                epsilon=self.epsilon,
                temperature=self.temperature,
                overpay_probability=self.overpay_probability,
                max_low_cost_payments=self.max_low_cost_payments,
                max_extra_overpayments=self.max_extra_overpayments,
                manhunt_epsilon=self.manhunt_epsilon,
                manhunt_alpha=self.manhunt_alpha,
                manhunt_rollouts=self.manhunt_rollouts,
            ),
            self._opponent_factory,
            seed=stable_observation_seed(
                observation,
                domain="belief-rollout-fugitive.full-game-evaluator.v1",
                salt=self._rollout_salt,
            ),
        )
        evaluation = evaluator.evaluate_fugitive_worlds_detailed(
            observation,
            worlds,
            candidates,
            repetitions=1,
        )
        probabilities = epsilon_softmax(
            {
                result.action: result.win_rate
                for result in evaluation.action_results
            },
            epsilon=self.rollout_epsilon,
            temperature=self.rollout_temperature,
        )
        self._last_rollout_diagnostics = (
            RolloutDiagnosticsSnapshot.from_evaluation(
                algorithm_id=self.algorithm_id,
                rollout_model_id=self.rollout_model_id,
                plan=plan,
                evaluation=evaluation,
            )
        )
        return dict(probabilities)

    def _rollout_plan(self, candidates: tuple) -> tuple[RolloutPlan, tuple]:
        plan = self.rollout_budget.plan(len(candidates))
        return plan, candidates[: plan.evaluated_candidates]

    def rollout_diagnostics(self) -> RolloutDiagnosticsSnapshot | None:
        return self._last_rollout_diagnostics

    def _cache(self, observation: Observation, distribution: dict[object, float]) -> None:
        if len(self._distribution_cache) >= 16:
            evicted = next(iter(self._distribution_cache))
            self._distribution_cache.pop(evicted)
            self._diagnostics_cache.pop(evicted, None)
        self._distribution_cache[observation] = dict(distribution)
        if self._last_rollout_diagnostics is not None:
            self._diagnostics_cache[observation] = self._last_rollout_diagnostics


def sample_fugitive_worlds(
    observation: Observation,
    *,
    count: int,
    rng: random.Random,
) -> tuple[FugitiveWorld, ...]:
    """Sample the Fugitive's unknown card allocation from public pile counts."""

    if observation.role is not Role.FUGITIVE:
        raise ValueError("Fugitive worlds require a Fugitive observation")
    count = _positive_int("count", count)
    if not isinstance(rng, random.Random):
        raise TypeError("rng must be random.Random")

    known = set(observation.hand)
    for slot in observation.route:
        if slot.hideout is None or slot.sprint_cards is None:
            raise ValueError("a Fugitive observation must show its own route")
        known.add(slot.hideout)
        known.update(slot.sprint_cards)
    marshal_counts = tuple(
        sum(
            record.role is Role.MARSHAL and record.pile == pile
            for record in observation.draw_history
        )
        for pile in range(3)
    )

    unknown_by_pile: list[tuple[int, ...]] = []
    for pile, original in enumerate(PILE_CARDS):
        unknown = tuple(card for card in original if card not in known)
        expected = marshal_counts[pile] + observation.pile_sizes[pile]
        if len(unknown) != expected:
            raise ValueError(
                "public pile counts do not match the Fugitive's known cards"
            )
        unknown_by_pile.append(unknown)

    worlds: list[FugitiveWorld] = []
    for _ in range(count):
        marshal_hand: list[int] = []
        remaining: list[tuple[int, ...]] = []
        for pile, unknown in enumerate(unknown_by_pile):
            shuffled = list(unknown)
            rng.shuffle(shuffled)
            draw_count = marshal_counts[pile]
            marshal_hand.extend(shuffled[:draw_count])
            remaining.append(tuple(sorted(shuffled[draw_count:])))
        worlds.append(
            FugitiveWorld(
                marshal_hand=tuple(sorted(marshal_hand)),
                remaining_piles=(remaining[0], remaining[1], remaining[2]),
            )
        )
    return tuple(worlds)


def _top_items(distribution, limit: int):
    return tuple(
        sorted(
            distribution.items(),
            key=lambda item: (-item[1], repr(item[0])),
        )[:limit]
    )


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _probability(name: str, value: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0.0 <= value <= 1.0
    ):
        raise ValueError(f"{name} must be finite and between zero and one")
    return float(value)


__all__ = [
    "BELIEF_ROLLOUT_FUGITIVE_ALGORITHM_ID",
    "BeliefRolloutFugitiveAgent",
    "DEFAULT_MAX_TERMINAL_SIMULATIONS",
    "DEFAULT_ROLLOUT_CANDIDATE_COUNT",
    "DEFAULT_ROLLOUT_EPSILON",
    "DEFAULT_ROLLOUT_TEMPERATURE",
    "ROLLOUT_MODEL_ID",
    "sample_fugitive_worlds",
]
