"""Shared policy mechanics for the controlled Marshal catalogue agents."""

from __future__ import annotations

import math
import random
from typing import Mapping

from fugitive.game.model import Observation, Phase, Role

from ..common.base import make_rng
from ..common.baseline_utils import sample_distribution, uniform_distribution
from .candidates import (
    CATALOGUE_SEED_DOMAIN,
    DEFAULT_CATALOGUE_MAX_GUESS_SIZE,
    DEFAULT_CATALOGUE_MAX_JOINT_CANDIDATES,
    build_marshal_guess_catalogue,
)


DEFAULT_CATALOGUE_EPSILON = 0.10
DEFAULT_CATALOGUE_MULTI_GUESS_CONTINUATION = 0.10


class CatalogueRandomMarshalPolicy:
    """Implementation shared by Boolean-support and route-count agents."""

    def __init__(
        self,
        seed: int | random.Random | None = None,
        *,
        rng: random.Random | None = None,
        use_route_counts: bool,
        epsilon: float = DEFAULT_CATALOGUE_EPSILON,
        multi_guess_continuation: float = (
            DEFAULT_CATALOGUE_MULTI_GUESS_CONTINUATION
        ),
        max_guess_size: int = DEFAULT_CATALOGUE_MAX_GUESS_SIZE,
        max_joint_candidates: int = DEFAULT_CATALOGUE_MAX_JOINT_CANDIDATES,
    ) -> None:
        _validate_probability("epsilon", epsilon)
        _validate_probability(
            "multi_guess_continuation", multi_guess_continuation
        )
        if isinstance(max_guess_size, bool) or not isinstance(max_guess_size, int):
            raise ValueError("max_guess_size must be a positive integer")
        if max_guess_size < 1:
            raise ValueError("max_guess_size must be a positive integer")
        if (
            isinstance(max_joint_candidates, bool)
            or not isinstance(max_joint_candidates, int)
            or max_joint_candidates < 1
        ):
            raise ValueError("max_joint_candidates must be a positive integer")
        self.rng = make_rng(seed, rng)
        self.use_route_counts = use_route_counts
        self.candidate_catalogue_id = CATALOGUE_SEED_DOMAIN
        self.epsilon = float(epsilon)
        self.multi_guess_continuation = float(multi_guess_continuation)
        self.max_guess_size = max_guess_size
        self.max_joint_candidates = max_joint_candidates

    def draw_pile_distribution(self, observation: Observation) -> dict[int, float]:
        if observation.role is not Role.MARSHAL:
            return {}
        return uniform_distribution(observation.legal_draw_piles)

    def choose_draw_pile(self, observation: Observation) -> int:
        distribution = self.draw_pile_distribution(observation)
        if not distribution:
            raise ValueError("there is no legal Marshal draw pile")
        return sample_distribution(distribution, rng=self.rng)

    def guess_distribution(
        self, observation: Observation
    ) -> dict[tuple[int, ...], float]:
        catalogue = build_marshal_guess_catalogue(
            observation,
            max_guess_size=self.max_guess_size,
            max_joint_candidates=self.max_joint_candidates,
        )
        groups = catalogue.counts_by_size()
        if not groups:
            return {}
        if observation.phase is Phase.MANHUNT:
            return self._conditional(groups[1], exponent=2.0)

        size_distribution = self._size_distribution(tuple(groups))
        result: dict[tuple[int, ...], float] = {}
        for size, counts in groups.items():
            conditional = self._conditional(counts, exponent=1.0)
            for guess, probability in conditional.items():
                result[guess] = size_distribution[size] * probability
        return _normalize(result)

    def choose_guess(self, observation: Observation) -> tuple[int, ...]:
        distribution = self.guess_distribution(observation)
        if not distribution:
            raise ValueError("there is no catalogue-supported Marshal guess")
        return sample_distribution(distribution, rng=self.rng)

    def _conditional(
        self,
        counts: Mapping[tuple[int, ...], int],
        *,
        exponent: float,
    ) -> dict[tuple[int, ...], float]:
        if not counts:
            return {}
        if not self.use_route_counts:
            return uniform_distribution(tuple(counts))
        log_weights = {
            guess: exponent * math.log(count)
            for guess, count in counts.items()
            if count > 0
        }
        maximum = max(log_weights.values())
        scaled = {
            guess: math.exp(value - maximum)
            for guess, value in log_weights.items()
        }
        total = sum(scaled.values())
        uniform = self.epsilon / len(scaled)
        return {
            guess: uniform + (1.0 - self.epsilon) * weight / total
            for guess, weight in scaled.items()
        }

    def _size_distribution(self, sizes: tuple[int, ...]) -> dict[int, float]:
        last = max(sizes)
        continuation = self.multi_guess_continuation
        weights = {
            size: (
                continuation ** (size - 1) * (1.0 - continuation)
                if size < last
                else continuation ** (size - 1)
            )
            for size in sizes
        }
        return _normalize(weights)


def _normalize(weights):
    total = sum(weights.values())
    if total <= 0.0:
        return {}
    return {choice: weight / total for choice, weight in weights.items()}


def _validate_probability(name: str, value: float) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0.0 <= value <= 1.0
    ):
        raise ValueError(f"{name} must be finite and between zero and one")


__all__ = [
    "CatalogueRandomMarshalPolicy",
    "DEFAULT_CATALOGUE_EPSILON",
    "DEFAULT_CATALOGUE_MULTI_GUESS_CONTINUATION",
]
