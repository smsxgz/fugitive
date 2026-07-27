"""BIR-2U Marshal whose root actions are valued by full-game rollouts."""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import replace
from typing import Iterable, Mapping

from fugitive.game.driver import DrawDecision, GuessDecision, PlayerDecision
from .inference.diagnostics import InferenceDiagnosticsSnapshot
from fugitive.game.model import Observation, Phase, Role
from fugitive.game.observation import (
    canonical_random_state_bytes,
    stable_observation_seed,
)
from .inference.particle_belief import DEFAULT_PARTICLE_COUNT, MarshalParticleBelief
from ..planning.rollout import RolloutBudget, RolloutEvaluator, RolloutPlan
from ..planning.rollout_diagnostics import RolloutDiagnosticsSnapshot

from ..common.base import make_rng
from ..common.baseline_utils import epsilon_softmax, sample_distribution
from ..planning.policy_mixture import (
    RegisteredPolicyMixtureFactory,
    WeightedPolicySpec,
    default_policy_spec,
)
from .route_count_catalogue import RouteCountCatalogueRandomMarshalAgent
from .bir.unweighted import (
    BIR2U_BELIEF_SEED_DOMAIN,
    BIR2U_PROPOSAL_KERNEL_ID,
    BIR2U_REFERENCE_TARGET_ID,
    BIR2U_WEIGHTING_ID,
    UnweightedConstructiveBeliefInformedRandomMarshalAgent,
)


ROLLOUT_BIR2U_MARSHAL_ALGORITHM_ID = "rollout-bir2u-marshal-budgeted-v2"
ROLLOUT_MODEL_ID = "root-sampled-budgeted-full-game-crn-v2"
DEFAULT_MAX_TERMINAL_SIMULATIONS = 192
DEFAULT_ROLLOUT_CANDIDATE_COUNT = 8
DEFAULT_ROLLOUT_EPSILON = 0.05
DEFAULT_ROLLOUT_TEMPERATURE = 0.25
_CONSTRUCTOR_SEED_DOMAIN = "fugitive.rollout-bir2u.constructor.v1"


def _default_fugitive_mixture() -> tuple[WeightedPolicySpec, ...]:
    return (
        default_policy_spec("hierarchical-random", Role.FUGITIVE),
        default_policy_spec("belief-informed-random", Role.FUGITIVE),
        default_policy_spec("continuation-count", Role.FUGITIVE),
    )


class RolloutBIR2UMarshalAgent:
    """Use a BIR-2U root belief and terminal Monte Carlo action values."""

    name = "rollout-bir2u-marshal"

    def __init__(
        self,
        seed: int | random.Random | None = None,
        *,
        rng: random.Random | None = None,
        particle_count: int = DEFAULT_PARTICLE_COUNT,
        max_guess_candidates: int = 128,
        rollout_candidate_count: int = DEFAULT_ROLLOUT_CANDIDATE_COUNT,
        max_terminal_simulations: int = DEFAULT_MAX_TERMINAL_SIMULATIONS,
        rollout_epsilon: float = DEFAULT_ROLLOUT_EPSILON,
        rollout_temperature: float = DEFAULT_ROLLOUT_TEMPERATURE,
        opponent_policies: Iterable[
            WeightedPolicySpec | Mapping[str, object]
        ] | None = None,
    ) -> None:
        self.rng = make_rng(seed, rng)
        rng_state_hash = _rng_state_hash(self.rng)
        base_seed = _seed_from_rng_state_hash(
            rng_state_hash,
            domain=_CONSTRUCTOR_SEED_DOMAIN,
        )
        self.base_agent = UnweightedConstructiveBeliefInformedRandomMarshalAgent(
            base_seed,
            particle_count=particle_count,
            max_guess_candidates=max_guess_candidates,
        )
        self.particle_count = self.base_agent.particle_count
        self.max_guess_candidates = self.base_agent.max_guess_candidates
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
            _default_fugitive_mixture()
            if opponent_policies is None
            else tuple(opponent_policies)
        )
        self._opponent_factory = RegisteredPolicyMixtureFactory(
            Role.FUGITIVE, choices
        )
        self.opponent_policies = self._opponent_factory.serialized_choices
        self.algorithm_id = ROLLOUT_BIR2U_MARSHAL_ALGORITHM_ID
        self.belief_backend_id = self.base_agent.belief_backend.backend_id
        self.proposal_kernel_id = BIR2U_PROPOSAL_KERNEL_ID
        self.belief_seed_domain = BIR2U_BELIEF_SEED_DOMAIN
        self.reference_target_id = BIR2U_REFERENCE_TARGET_ID
        self.weighting_id = BIR2U_WEIGHTING_ID
        self.rollout_model_id = ROLLOUT_MODEL_ID
        self.rollout_leaf_marshal_id = (
            "hr-1.1c-marshal-shared-catalogue-route-count-v1"
        )
        self._rollout_salt = rng_state_hash
        self._distribution_cache: dict[Observation, dict[object, float]] = {}
        self._diagnostics_cache: dict[
            Observation, RolloutDiagnosticsSnapshot
        ] = {}
        self._last_rollout_diagnostics: RolloutDiagnosticsSnapshot | None = None

    def belief(self, observation: Observation) -> MarshalParticleBelief:
        return self.base_agent.belief(observation)

    def inference_diagnostics(self) -> InferenceDiagnosticsSnapshot | None:
        snapshot = self.base_agent.inference_diagnostics()
        return (
            None
            if snapshot is None
            else replace(snapshot, algorithm_id=self.algorithm_id)
        )

    def draw_pile_distribution(self, observation: Observation) -> dict[int, float]:
        if (
            observation.role is not Role.MARSHAL
            or observation.phase is not Phase.MARSHAL_DRAW
        ):
            return {}
        cached = self._distribution_cache.get(observation)
        if cached is not None:
            self._last_rollout_diagnostics = self._diagnostics_cache.get(
                observation
            )
            return {int(action): probability for action, probability in cached.items()}
        base = self.base_agent.draw_pile_distribution(observation)
        candidates = tuple(
            DrawDecision(pile)
            for pile, _probability in _top_items(base, len(base))
        )
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
            raise ValueError("there is no legal Marshal draw pile")
        return sample_distribution(distribution, rng=self.rng)

    def guess_distribution(
        self, observation: Observation
    ) -> dict[tuple[int, ...], float]:
        if observation.role is not Role.MARSHAL or observation.phase not in (
            Phase.MARSHAL_GUESS,
            Phase.MANHUNT,
        ):
            return {}
        cached = self._distribution_cache.get(observation)
        if cached is not None:
            self._last_rollout_diagnostics = self._diagnostics_cache.get(
                observation
            )
            return {
                tuple(action): probability
                for action, probability in cached.items()
            }
        base = self.base_agent.guess_distribution(observation)
        candidates = tuple(
            GuessDecision(guess)
            for guess, _probability in _top_items(
                base, self.rollout_candidate_count
            )
        )
        distribution = {
            action.numbers: probability
            for action, probability in self._rollout_distribution(
                observation, candidates
            ).items()
            if isinstance(action, GuessDecision)
        }
        self._cache(observation, distribution)
        return distribution

    def choose_guess(self, observation: Observation) -> tuple[int, ...]:
        distribution = self.guess_distribution(observation)
        if not distribution:
            raise ValueError("there is no rollout-supported Marshal guess")
        return sample_distribution(distribution, rng=self.rng)

    def _rollout_distribution(
        self,
        observation: Observation,
        candidates: tuple[PlayerDecision, ...],
    ) -> dict[PlayerDecision, float]:
        if not candidates:
            return {}
        plan, candidates = self._rollout_plan(candidates)
        if plan.single_candidate_shortcut:
            action = candidates[0]
            self._last_rollout_diagnostics = RolloutDiagnosticsSnapshot.shortcut(
                algorithm_id=self.algorithm_id,
                rollout_model_id=self.rollout_model_id,
                role=Role.MARSHAL,
                plan=plan,
                action=action,
            )
            return {action: 1.0}

        belief = self.base_agent.belief(observation)
        if belief.is_empty:
            raise RuntimeError("rollout BIR-2U requires a nonempty root belief")
        world_count = plan.paired_scenarios
        local_rng = random.Random(
            stable_observation_seed(
                observation,
                domain="rollout-bir2u.root-resampling.v1",
                salt=self._rollout_salt,
            )
        )
        worlds = tuple(
            local_rng.choices(
                belief.particles,
                weights=[particle.weight for particle in belief.particles],
                k=world_count,
            )
        )
        evaluator = RolloutEvaluator(
            self._opponent_factory,
            lambda rollout_seed: RouteCountCatalogueRandomMarshalAgent(
                rollout_seed
            ),
            seed=stable_observation_seed(
                observation,
                domain="rollout-bir2u.full-game-evaluator.v1",
                salt=self._rollout_salt,
            ),
        )
        evaluation = evaluator.evaluate_worlds_detailed(
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

    def _rollout_plan(
        self,
        candidates: tuple[PlayerDecision, ...],
    ) -> tuple[RolloutPlan, tuple[PlayerDecision, ...]]:
        plan = self.rollout_budget.plan(len(candidates))
        return plan, candidates[: plan.evaluated_candidates]

    def rollout_diagnostics(self) -> RolloutDiagnosticsSnapshot | None:
        return self._last_rollout_diagnostics

    def _cache(
        self,
        observation: Observation,
        distribution: dict[object, float],
    ) -> None:
        if len(self._distribution_cache) >= 16:
            evicted = next(iter(self._distribution_cache))
            self._distribution_cache.pop(evicted)
            self._diagnostics_cache.pop(evicted, None)
        self._distribution_cache[observation] = dict(distribution)
        if self._last_rollout_diagnostics is not None:
            self._diagnostics_cache[observation] = self._last_rollout_diagnostics


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


def _rng_state_hash(rng: random.Random) -> str:
    """Return a stable full hash of one canonical Python RNG state."""

    digest = hashlib.sha256()
    digest.update(b"fugitive.rollout-bir2u.rng-state\x00v1\x00")
    digest.update(canonical_random_state_bytes(rng))
    return digest.hexdigest()


def _seed_from_rng_state_hash(state_hash: str, *, domain: str) -> int:
    """Domain-separate a recorded RNG-state hash into one 64-bit seed."""

    if not domain:
        raise ValueError("seed domain must be non-empty")
    try:
        state_digest = bytes.fromhex(state_hash)
    except ValueError as exc:
        raise ValueError("state_hash must be hexadecimal") from exc
    if len(state_digest) != hashlib.sha256().digest_size:
        raise ValueError("state_hash must contain one SHA-256 digest")
    digest = hashlib.sha256()
    digest.update(b"fugitive.rollout-bir2u.seed\x00v1\x00")
    for part in (domain.encode("utf-8"), state_digest):
        digest.update(len(part).to_bytes(8, "big"))
        digest.update(part)
    return int.from_bytes(digest.digest()[:8], "big")


__all__ = [
    "DEFAULT_MAX_TERMINAL_SIMULATIONS",
    "DEFAULT_ROLLOUT_CANDIDATE_COUNT",
    "DEFAULT_ROLLOUT_EPSILON",
    "DEFAULT_ROLLOUT_TEMPERATURE",
    "ROLLOUT_MODEL_ID",
    "ROLLOUT_BIR2U_MARSHAL_ALGORITHM_ID",
    "RolloutBIR2UMarshalAgent",
]
