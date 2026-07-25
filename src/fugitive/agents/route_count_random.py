"""Route-count-weighted random Marshal baseline.

This HR-1.1 teaching policy changes only how Marshal guesses are weighted.
Draw-pile choices remain uniform, making comparisons with HR-1 isolate the
effect of using completion mass rather than Boolean route support.
"""

from __future__ import annotations

import math
import random
from typing import Hashable, Mapping, TypeVar

from fugitive.belief import BeliefResult, PathBelief
from fugitive.model import Observation, Phase, Role
from fugitive.observation_protocol import stable_observation_seed

from .base import make_rng
from .baseline_utils import sample_distribution, uniform_distribution


DEFAULT_ALPHA = 1.0
DEFAULT_EPSILON = 0.10
DEFAULT_MULTI_GUESS_CONTINUATION = 0.10
DEFAULT_MAX_GUESS_SIZE = 3
DEFAULT_MAX_JOINT_CANDIDATES = 48
HR11_MARSHAL_ALGORITHM_ID = "hr-1.1-marshal-route-count-random-v1"
_MANHUNT_ALPHA = 2.0
_JOINT_ROUTE_SAMPLE_MULTIPLIER = 8


ChoiceT = TypeVar("ChoiceT", bound=Hashable)


def _validate_probability(name: str, value: float) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0.0 <= value <= 1.0
    ):
        raise ValueError(f"{name} must be finite and between zero and one")


def _count_weighted_distribution(
    counts: Mapping[ChoiceT, int],
    *,
    exponent: float,
    epsilon: float,
) -> dict[ChoiceT, float]:
    """Normalize positive integer counts with a uniform epsilon mixture."""

    positive = {choice: count for choice, count in counts.items() if count > 0}
    if not positive:
        return {}
    maximum_log_weight = max(
        exponent * math.log(count) for count in positive.values()
    )
    scaled = {
        choice: math.exp(exponent * math.log(count) - maximum_log_weight)
        for choice, count in positive.items()
    }
    denominator = sum(scaled.values())
    uniform_mass = epsilon / len(scaled)
    informed_mass = 1.0 - epsilon
    return {
        choice: uniform_mass + informed_mass * weight / denominator
        for choice, weight in scaled.items()
    }


class RouteCountRandomMarshalAgent:
    """HR-1.1 Marshal weighted by exact compatible-route counts.

    The route model is still policy agnostic and does not identify hidden
    Sprint cards.  Its counts are therefore an inspectable route-count model,
    not a calibrated posterior over complete game worlds.
    """

    name = "route-count-random-marshal"

    def __init__(
        self,
        seed: int | random.Random | None = None,
        *,
        rng: random.Random | None = None,
        alpha: float = DEFAULT_ALPHA,
        epsilon: float = DEFAULT_EPSILON,
        multi_guess_continuation: float = DEFAULT_MULTI_GUESS_CONTINUATION,
        max_guess_size: int = DEFAULT_MAX_GUESS_SIZE,
        max_joint_candidates: int = DEFAULT_MAX_JOINT_CANDIDATES,
    ) -> None:
        if (
            isinstance(alpha, bool)
            or not isinstance(alpha, (int, float))
            or not math.isfinite(alpha)
            or alpha <= 0.0
        ):
            raise ValueError("alpha must be positive and finite")
        _validate_probability("epsilon", epsilon)
        _validate_probability(
            "multi_guess_continuation",
            multi_guess_continuation,
        )
        if (
            isinstance(max_guess_size, bool)
            or not isinstance(max_guess_size, int)
            or not 1 <= max_guess_size <= DEFAULT_MAX_GUESS_SIZE
        ):
            raise ValueError("max_guess_size must be an integer from one through three")
        if (
            isinstance(max_joint_candidates, bool)
            or not isinstance(max_joint_candidates, int)
            or not 1 <= max_joint_candidates <= DEFAULT_MAX_JOINT_CANDIDATES
        ):
            raise ValueError(
                "max_joint_candidates must be an integer from one through 48"
            )

        self.rng = make_rng(seed, rng)
        self.algorithm_id = HR11_MARSHAL_ALGORITHM_ID
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.multi_guess_continuation = float(multi_guess_continuation)
        self.max_guess_size = max_guess_size
        self.max_joint_candidates = max_joint_candidates

    def draw_pile_distribution(self, observation: Observation) -> dict[int, float]:
        """Return a uniform distribution over the legal nonempty piles."""

        if observation.role is not Role.MARSHAL or not observation.legal_draw_piles:
            return {}
        return uniform_distribution(observation.legal_draw_piles)

    def choose_draw_pile(self, observation: Observation) -> int:
        distribution = self.draw_pile_distribution(observation)
        if not distribution:
            raise ValueError("there is no legal Marshal draw pile")
        return sample_distribution(distribution, rng=self.rng)

    def guess_distribution(
        self,
        observation: Observation,
    ) -> dict[tuple[int, ...], float]:
        """Return the full HR-1.1 distribution before sampling an action."""

        if observation.role is not Role.MARSHAL or observation.phase not in (
            Phase.MARSHAL_GUESS,
            Phase.MANHUNT,
        ):
            return {}

        belief = PathBelief.from_observation(observation)
        result = belief.solve()
        if not result.total_paths or not result.marginal_counts:
            return {}

        singleton_counts = {
            (card,): count
            for card, count in sorted(result.marginal_counts.items())
        }
        if observation.phase is Phase.MANHUNT:
            return _count_weighted_distribution(
                singleton_counts,
                exponent=_MANHUNT_ALPHA,
                epsilon=self.epsilon,
            )

        hidden_count = sum(slot.hideout is None for slot in observation.route)
        size_limit = min(
            self.max_guess_size,
            hidden_count,
            len(singleton_counts),
        )
        groups: dict[int, dict[tuple[int, ...], float]] = {
            1: _count_weighted_distribution(
                singleton_counts,
                exponent=self.alpha,
                epsilon=self.epsilon,
            )
        }
        candidate_cards = frozenset(result.marginal_counts)
        for size in range(2, size_limit + 1):
            joint_counts = self._sampled_joint_counts(
                observation,
                belief,
                result,
                candidate_cards,
                size,
            )
            if joint_counts:
                groups[size] = _count_weighted_distribution(
                    joint_counts,
                    exponent=1.0,
                    epsilon=0.0,
                )

        size_distribution = self._guess_size_distribution(tuple(groups))
        distribution: dict[tuple[int, ...], float] = {}
        for size, conditional in groups.items():
            size_mass = size_distribution[size]
            for guess, probability in conditional.items():
                distribution[guess] = size_mass * probability
        return distribution

    def choose_guess(self, observation: Observation) -> tuple[int, ...]:
        distribution = self.guess_distribution(observation)
        if not distribution:
            raise ValueError("there is no route-count-supported Marshal guess")
        return sample_distribution(distribution, rng=self.rng)

    def _guess_size_distribution(self, sizes: tuple[int, ...]) -> dict[int, float]:
        last_size = max(sizes)
        continuation = self.multi_guess_continuation
        weights = {
            size: (
                continuation ** (size - 1) * (1.0 - continuation)
                if size < last_size
                else continuation ** (size - 1)
            )
            for size in sizes
        }
        denominator = sum(weights.values())
        return {size: weight / denominator for size, weight in weights.items()}

    def _sampled_joint_counts(
        self,
        observation: Observation,
        belief: PathBelief,
        result: BeliefResult,
        candidate_cards: frozenset[int],
        size: int,
    ) -> dict[tuple[int, ...], int]:
        local_rng = random.Random(
            stable_observation_seed(
                observation,
                domain="route-count-random.marshal.joint-candidates.v1",
                salt=f"{size}:{self.max_joint_candidates}",
            )
        )
        candidates: set[tuple[int, ...]] = set()
        sample_budget = max(
            64,
            self.max_joint_candidates * _JOINT_ROUTE_SAMPLE_MULTIPLIER,
        )
        for _ in range(sample_budget):
            sampled = belief.sample_route(local_rng)
            hidden_values = tuple(
                value
                for slot, value in zip(
                    observation.route,
                    sampled.route,
                    strict=True,
                )
                if slot.hideout is None and value in candidate_cards
            )
            if len(hidden_values) < size:
                continue
            guess = tuple(sorted(local_rng.sample(hidden_values, size)))
            candidates.add(guess)
            if len(candidates) >= self.max_joint_candidates:
                break

        return {
            guess: count
            for guess in sorted(candidates)
            if (count := result.count_paths_containing(guess)) > 0
        }


__all__ = [
    "DEFAULT_ALPHA",
    "DEFAULT_EPSILON",
    "DEFAULT_MAX_GUESS_SIZE",
    "DEFAULT_MAX_JOINT_CANDIDATES",
    "DEFAULT_MULTI_GUESS_CONTINUATION",
    "HR11_MARSHAL_ALGORITHM_ID",
    "RouteCountRandomMarshalAgent",
]
