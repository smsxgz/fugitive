"""Stable Marshal guess catalogues for controlled policy comparisons.

The catalogue depends only on the Marshal observation and ``PathBelief``
support.  Policies may assign different weights to the returned guesses
without accidentally changing the action abstraction being compared.
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
import random

from fugitive.belief import PathBelief
from fugitive.model import Observation, Phase, Role
from fugitive.observation_protocol import stable_observation_seed


CATALOGUE_SEED_DOMAIN = "fugitive.marshal.guess-catalogue.v1"
DEFAULT_CATALOGUE_MAX_GUESS_SIZE = 3
DEFAULT_CATALOGUE_MAX_JOINT_CANDIDATES = 48
_ROUTE_SAMPLE_MULTIPLIER = 8


@dataclass(frozen=True, slots=True)
class MarshalGuessCatalogue:
    """One deterministic action catalogue with exact route-support counts."""

    groups: tuple[tuple[int, tuple[tuple[tuple[int, ...], int], ...]], ...]
    total_paths: int

    def counts_by_size(self) -> dict[int, dict[tuple[int, ...], int]]:
        return {
            size: dict(entries)
            for size, entries in self.groups
        }

    @property
    def guesses(self) -> tuple[tuple[int, ...], ...]:
        return tuple(
            guess
            for _size, entries in self.groups
            for guess, _count in entries
        )


def build_marshal_guess_catalogue(
    observation: Observation,
    *,
    max_guess_size: int = DEFAULT_CATALOGUE_MAX_GUESS_SIZE,
    max_joint_candidates: int = DEFAULT_CATALOGUE_MAX_JOINT_CANDIDATES,
) -> MarshalGuessCatalogue:
    """Return guesses shared by support- and count-weighted policies."""

    _validate_limit("max_guess_size", max_guess_size)
    _validate_limit("max_joint_candidates", max_joint_candidates)
    if observation.role is not Role.MARSHAL or observation.phase not in (
        Phase.MARSHAL_GUESS,
        Phase.MANHUNT,
    ):
        return MarshalGuessCatalogue((), 0)

    belief = PathBelief.from_observation(observation)
    result = belief.solve()
    if not result.total_paths or not result.marginal_counts:
        return MarshalGuessCatalogue((), result.total_paths)

    groups: list[tuple[int, tuple[tuple[tuple[int, ...], int], ...]]] = []
    singles = tuple(
        ((number,), count)
        for number, count in sorted(result.marginal_counts.items())
        if count > 0
    )
    if singles:
        groups.append((1, singles))
    if observation.phase is Phase.MANHUNT:
        return MarshalGuessCatalogue(tuple(groups), result.total_paths)

    hidden_count = sum(slot.hideout is None for slot in observation.route)
    size_limit = min(max_guess_size, hidden_count, len(singles))
    candidate_cards = frozenset(result.marginal_counts)
    for size in range(2, size_limit + 1):
        candidates = _joint_candidates(
            observation,
            belief,
            candidate_cards,
            size=size,
            limit=max_joint_candidates,
        )
        entries = tuple(
            (guess, count)
            for guess in candidates
            if (count := result.count_paths_containing(guess)) > 0
        )
        if entries:
            groups.append((size, entries))
    return MarshalGuessCatalogue(tuple(groups), result.total_paths)


def _joint_candidates(
    observation: Observation,
    belief: PathBelief,
    candidate_cards: frozenset[int],
    *,
    size: int,
    limit: int,
) -> tuple[tuple[int, ...], ...]:
    local_rng = random.Random(
        stable_observation_seed(
            observation,
            domain=CATALOGUE_SEED_DOMAIN,
            salt=f"{size}:{limit}",
        )
    )
    candidates: set[tuple[int, ...]] = set()
    sample_budget = max(64, limit * _ROUTE_SAMPLE_MULTIPLIER)
    for _ in range(sample_budget):
        sampled = belief.sample_route(local_rng)
        hidden = tuple(
            value
            for slot, value in zip(observation.route, sampled.route, strict=True)
            if slot.hideout is None and value in candidate_cards
        )
        if len(hidden) < size:
            continue
        candidates.add(tuple(sorted(local_rng.sample(hidden, size))))
        if len(candidates) >= limit:
            break

    if not candidates:
        cards = tuple(sorted(candidate_cards))
        combination_count = math.comb(len(cards), size)
        if combination_count <= 4_096:
            result = belief.solve()
            for guess in itertools.combinations(cards, size):
                if result.count_paths_containing(guess):
                    candidates.add(guess)
                    if len(candidates) >= limit:
                        break
    return tuple(sorted(candidates))


def _validate_limit(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


__all__ = [
    "CATALOGUE_SEED_DOMAIN",
    "DEFAULT_CATALOGUE_MAX_GUESS_SIZE",
    "DEFAULT_CATALOGUE_MAX_JOINT_CANDIDATES",
    "MarshalGuessCatalogue",
    "build_marshal_guess_catalogue",
]
