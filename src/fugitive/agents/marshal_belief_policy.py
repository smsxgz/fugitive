"""Marshal action policy shared by the belief-informed agents."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import itertools
import math
import random
from typing import Iterable, Mapping, Protocol

from fugitive.model import Observation, Phase, Role
from fugitive.inference_diagnostics import (
    BeliefBackendDiagnosticsSnapshot,
    InferenceDiagnosticsSnapshot,
)
from fugitive.observation_protocol import canonical_random_state_bytes
from fugitive.particle_belief import (
    MarshalDrawOutcomeStatistics,
    MarshalParticleBelief,
)
from fugitive.rules import sprint_value

from .base import make_rng
from .baseline_utils import epsilon_softmax, sample_distribution
from .bir_common import (
    normalized as _normalized,
    validate_manhunt_parameters as _validate_manhunt_parameters,
)
from .hierarchical_random import hard_constraint_guess_numbers


DEFAULT_MAX_GUESS_CANDIDATES = 128


class MarshalBeliefBackend(Protocol):
    """Observation-only inference component consumed by the action policy."""

    backend_id: str
    particle_count: int

    @property
    def latest_belief(self) -> MarshalParticleBelief | None: ...

    def diagnostic_snapshot(self) -> BeliefBackendDiagnosticsSnapshot | None: ...

    def infer(self, observation: Observation) -> MarshalParticleBelief: ...



class BeliefInformedMarshalActionPolicy:
    """Shared stochastic Marshal action policy over an injected belief backend."""

    def __init__(
        self,
        *,
        rng: random.Random,
        belief_backend: MarshalBeliefBackend,
        epsilon: float,
        temperature: float,
        max_guess_candidates: int,
        terminal_bonus_scale: float,
        manhunt_epsilon: float,
        manhunt_alpha: float,
    ) -> None:
        if not math.isfinite(epsilon) or not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be between zero and one")
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if (
            isinstance(max_guess_candidates, bool)
            or not isinstance(max_guess_candidates, int)
            or not 1 <= max_guess_candidates <= DEFAULT_MAX_GUESS_CANDIDATES
        ):
            raise ValueError("max_guess_candidates must be from 1 through 128")
        if not math.isfinite(terminal_bonus_scale) or terminal_bonus_scale < 0.0:
            raise ValueError("terminal_bonus_scale must be finite and non-negative")
        _validate_manhunt_parameters(manhunt_epsilon, manhunt_alpha)
        self.rng = rng
        self.belief_backend = belief_backend
        self.epsilon = epsilon
        self.temperature = temperature
        self.max_guess_candidates = max_guess_candidates
        self.terminal_bonus_scale = terminal_bonus_scale
        self.manhunt_epsilon = manhunt_epsilon
        self.manhunt_alpha = manhunt_alpha

    def draw_pile_distribution(self, observation: Observation) -> dict[int, float]:
        if observation.role is not Role.MARSHAL:
            return {}
        belief = self.belief_backend.infer(observation)
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
        belief = belief or self.belief_backend.infer(observation)
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
        belief = self.belief_backend.infer(observation)
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


def marshal_rng_and_belief_salt(
    seed: int | random.Random | None,
    rng: random.Random | None,
) -> tuple[random.Random, str]:
    """Create the action RNG and domain salt without consuming that RNG."""

    policy_rng = make_rng(seed, rng)
    salt = hashlib.sha256(
        canonical_random_state_bytes(policy_rng)
    ).hexdigest()
    return policy_rng, salt


class ComposedBeliefInformedRandomMarshalAgent:
    """Thin agent facade combining one action policy with one belief backend."""

    def __init__(self, action_policy: BeliefInformedMarshalActionPolicy) -> None:
        self.action_policy = action_policy
        self.belief_backend = action_policy.belief_backend

    @property
    def rng(self) -> random.Random:
        return self.action_policy.rng

    @property
    def particle_count(self) -> int:
        return self.belief_backend.particle_count

    @property
    def epsilon(self) -> float:
        return self.action_policy.epsilon

    @property
    def temperature(self) -> float:
        return self.action_policy.temperature

    @property
    def max_guess_candidates(self) -> int:
        return self.action_policy.max_guess_candidates

    @property
    def terminal_bonus_scale(self) -> float:
        return self.action_policy.terminal_bonus_scale

    @property
    def manhunt_epsilon(self) -> float:
        return self.action_policy.manhunt_epsilon

    @property
    def manhunt_alpha(self) -> float:
        return self.action_policy.manhunt_alpha

    def inference_diagnostics(self) -> InferenceDiagnosticsSnapshot | None:
        """Label the backend's already-computed snapshot with this algorithm."""

        snapshot = self.belief_backend.diagnostic_snapshot()
        if snapshot is None:
            return None
        algorithm_id = getattr(self, "algorithm_id", None)
        if not isinstance(algorithm_id, str) or not algorithm_id:
            raise RuntimeError("belief-informed agent has no algorithm_id")
        return snapshot.for_algorithm(algorithm_id)

    def belief(self, observation: Observation) -> MarshalParticleBelief:
        return self.belief_backend.infer(observation)

    def draw_pile_distribution(self, observation: Observation) -> dict[int, float]:
        return self.action_policy.draw_pile_distribution(observation)

    def choose_draw_pile(self, observation: Observation) -> int:
        return self.action_policy.choose_draw_pile(observation)

    def candidate_guess_sets(
        self,
        observation: Observation,
        belief: MarshalParticleBelief | None = None,
    ) -> tuple[tuple[int, ...], ...]:
        return self.action_policy.candidate_guess_sets(observation, belief)

    def guess_distribution(
        self,
        observation: Observation,
    ) -> dict[tuple[int, ...], float]:
        return self.action_policy.guess_distribution(observation)

    def choose_guess(self, observation: Observation) -> tuple[int, ...]:
        return self.action_policy.choose_guess(observation)


__all__ = [
    "BeliefInformedMarshalActionPolicy",
    "ComposedBeliefInformedRandomMarshalAgent",
    "DEFAULT_MAX_GUESS_CANDIDATES",
    "MarshalBeliefBackend",
    "marshal_rng_and_belief_salt",
]
