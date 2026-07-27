"""Hierarchical legal-random Marshal policy.

The Marshal draws uniformly from a legal pile, then samples a small guess set
from values supported by at least one route consistent with its observation.
Route counts are deliberately used only as a Boolean support test in HR-1.
"""

from __future__ import annotations

import itertools
import math
import random
from fugitive.game.model import Observation, Phase, Role
from fugitive.game.observation import stable_observation_seed

from ..common.base import make_rng
from ..common.baseline_utils import normalize_distribution, sample_distribution
from .inference.path_belief import BeliefResult, solve_observation_belief


DEFAULT_MULTI_GUESS_CONTINUATION = 0.10
DEFAULT_MAX_GUESS_SIZE = 4
DEFAULT_MAX_GUESSES_PER_SIZE = 128
HR1_MARSHAL_ALGORITHM_ID = "hr-1-marshal-hard-support-random-v2"

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
    """Return values supported by an information-consistent hidden route."""

    candidates, _result = _hard_constraint_guess_support(observation)
    return candidates


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
        return sample_distribution(distribution, rng=self.rng)

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
        size_distribution = normalize_distribution(size_weights)

        distribution: dict[tuple[int, ...], float] = {}
        for size, guesses in groups.items():
            if size not in size_distribution:
                continue
            per_guess = size_distribution[size] / len(guesses)
            for guess in guesses:
                distribution[guess] = per_guess
        return normalize_distribution(distribution)

    def choose_guess(self, observation: Observation) -> tuple[int, ...]:
        distribution = self.guess_distribution(observation)
        if not distribution:
            raise ValueError("there is no information-consistent Marshal guess")
        return sample_distribution(distribution, rng=self.rng)

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

        local_rng = random.Random(
            stable_observation_seed(
                observation,
                domain="hr.marshal.joint-candidates.v1",
                salt=str(size),
            )
        )
        combination_count = math.comb(len(candidates), size)
        attempts = min(combination_count, max(64, self.max_guesses_per_size * 12))
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
    "DEFAULT_MAX_GUESS_SIZE",
    "DEFAULT_MAX_GUESSES_PER_SIZE",
    "DEFAULT_MULTI_GUESS_CONTINUATION",
    "HR1_MARSHAL_ALGORITHM_ID",
    "HierarchicalRandomMarshalAgent",
    "hard_constraint_guess_numbers",
]
