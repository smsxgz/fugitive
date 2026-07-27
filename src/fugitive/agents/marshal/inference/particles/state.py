"""Particle worlds, normalized beliefs, and deterministic belief queries.

Fresh construction and incremental propagation live in separate modules.  The
two class entry points import those modules lazily so dependencies remain
one-directional: inference algorithms depend on this state model.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
import random
from types import MappingProxyType
from typing import Callable, Iterable, Mapping, Sequence

from fugitive.game.model import FugitiveAction, Observation, Role
from fugitive.game.rules import (
    GUESSABLE_CARDS,
    PILE_CARDS,
    is_legal_fugitive_action,
    sprint_value,
)


DEFAULT_PARTICLE_COUNT = 2_000
BIR1_INCREMENTAL_PLAY_PROPOSAL_LIMIT = 64
BIR1_RESAMPLE_ESS_FRACTION = 0.5


class IncompatibleObservationError(ValueError):
    """Raised when an observation cannot extend an incremental belief."""


@dataclass(frozen=True, slots=True)
class MarshalParticle:
    """One complete hidden-world hypothesis from the Marshal's perspective.

    ``fugitive_draws`` is aligned with the Fugitive records in the public draw
    history.  Remaining piles are represented as unordered contents: because
    the order is unknown and uniformly shuffled, this is a sufficient state
    for the posterior of the next draw.
    """

    fugitive_hand: tuple[int, ...]
    marshal_hand: tuple[int, ...]
    route_hideouts: tuple[int, ...]
    route_sprints: tuple[tuple[int, ...], ...]
    fugitive_draws: tuple[int, ...]
    remaining_piles: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]
    weight: float = 1.0

    @property
    def world_key(self) -> tuple[object, ...]:
        """Return the hidden-world identity without posterior bookkeeping."""

        return (
            self.fugitive_hand,
            self.marshal_hand,
            self.route_hideouts,
            self.route_sprints,
            self.fugitive_draws,
            self.remaining_piles,
        )

    @property
    def all_cards_are_unique(self) -> bool:
        """Whether all card locations form a valid one-card partition."""

        located = [
            *self.fugitive_hand,
            *self.marshal_hand,
            *self.route_hideouts[1:],
            *(card for stack in self.route_sprints for card in stack),
            *(card for pile in self.remaining_piles for card in pile),
        ]
        return len(located) == len(set(located)) and set(located) == set(range(1, 43))


@dataclass(frozen=True, slots=True)
class ParticleBeliefSummary:
    """Deterministic, directly comparable public summary of a belief."""

    particle_count: int
    unique_particle_count: int
    effective_sample_size: float
    pre_resample_effective_sample_size: float
    world_effective_sample_size: float
    max_world_mass: float
    unique_hidden_hypothesis_count: int
    hidden_hypothesis_effective_sample_size: float
    max_hidden_hypothesis_mass: float
    resampling_count: int
    sampling_attempts: int
    sampling_accepted: int
    sampling_acceptance_rate: float
    sampling_exhausted: bool
    marginals: tuple[tuple[int, float], ...]
    draw_posteriors: tuple[tuple[tuple[int, float], ...], ...]


@dataclass(frozen=True, slots=True)
class MarshalDrawOutcomeStatistics:
    """Sufficient statistics after one hypothetical private Marshal draw."""

    pile: int
    card: int
    probability: float
    marginals: Mapping[int, float]
    hidden_route_masses: Mapping[tuple[int, ...], float]
    hidden_mask_masses: Mapping[int, float]
    escape_risk: float

    def joint_success(self, numbers: Iterable[int]) -> float:
        """Return the conditioned mass containing every guessed number."""

        requested = tuple(sorted(set(numbers)))
        if not requested or any(number not in GUESSABLE_CARDS for number in requested):
            return 0.0
        requested_mask = 0
        for number in requested:
            requested_mask |= 1 << number
        return sum(
            mass
            for hidden_mask, mass in self.hidden_mask_masses.items()
            if hidden_mask & requested_mask == requested_mask
        )


class _DrawOutcomeAccumulator:
    __slots__ = (
        "mass",
        "marginals",
        "hidden_routes",
        "hidden_masks",
        "immediate_risk",
        "pile_risks",
        "any_nonempty",
    )

    def __init__(self) -> None:
        self.mass = 0.0
        self.marginals: dict[int, float] = {}
        self.hidden_routes: dict[tuple[int, ...], float] = {}
        self.hidden_masks: dict[int, float] = {}
        self.immediate_risk = 0.0
        self.pile_risks = [0.0, 0.0, 0.0]
        self.any_nonempty = False


def _can_play_42(cards: Sequence[int], previous_hideout: int) -> bool:
    return (
        42 in cards
        and previous_hideout != 42
        and 42 - previous_hideout
        <= 3 + sum(sprint_value(card) for card in cards if card != 42)
    )


class MarshalParticleBelief:
    """Weighted particle approximation to the Marshal's information state."""

    def __init__(
        self,
        observation: Observation,
        particles: Sequence[MarshalParticle],
        *,
        sampling_attempts: int,
        sampling_exhausted: bool,
        sampling_accepted: int | None = None,
        pre_resample_effective_sample_size: float | None = None,
        resampling_count: int = 0,
    ) -> None:
        if observation.role is not Role.MARSHAL:
            raise ValueError("MarshalParticleBelief requires a Marshal observation")
        self.observation = observation
        self.sampling_attempts = sampling_attempts
        self.sampling_accepted = (
            len(particles) if sampling_accepted is None else sampling_accepted
        )
        self.sampling_exhausted = sampling_exhausted
        self.particles = self._normalized(tuple(particles))
        if (
            isinstance(resampling_count, bool)
            or not isinstance(resampling_count, int)
            or resampling_count < 0
        ):
            raise ValueError("resampling_count must be a non-negative integer")
        self.resampling_count = resampling_count
        if pre_resample_effective_sample_size is None:
            pre_resample_effective_sample_size = self.effective_sample_size
        if (
            isinstance(pre_resample_effective_sample_size, bool)
            or not isinstance(pre_resample_effective_sample_size, (int, float))
            or not math.isfinite(pre_resample_effective_sample_size)
            or pre_resample_effective_sample_size < 0.0
        ):
            raise ValueError(
                "pre_resample_effective_sample_size must be finite and non-negative"
            )
        self.pre_resample_effective_sample_size = (
            pre_resample_effective_sample_size
        )
        self._revealed_indices = frozenset(
            index
            for index, slot in enumerate(observation.route)
            if index and slot.revealed
        )
        self._draw_outcome_statistics_cache: tuple[
            Mapping[int, MarshalDrawOutcomeStatistics],
            Mapping[int, MarshalDrawOutcomeStatistics],
            Mapping[int, MarshalDrawOutcomeStatistics],
        ] | None = None

    @classmethod
    def from_observation(
        cls,
        observation: Observation,
        *,
        particle_count: int = DEFAULT_PARTICLE_COUNT,
        seed: int | None = None,
        rng: random.Random | None = None,
    ) -> "MarshalParticleBelief":
        """Build an equal-weight sequential constructive belief."""

        from ..constructive.sampler import (
            ConstructiveWorldSampler,
        )

        from .constructive_fresh import (
            ConstructiveWeightingStrategy,
            SEQUENTIAL_CONSTRUCTIVE_SPRINT_BACKEND,
            build_constructive_belief,
            is_complete_constructive_batch,
        )

        sampler = ConstructiveWorldSampler(
            sprint_backend=SEQUENTIAL_CONSTRUCTIVE_SPRINT_BACKEND
        )
        batch = sampler.sample_batch(
            observation,
            particle_count=particle_count,
            seed=seed,
            rng=rng,
        )
        if not is_complete_constructive_batch(batch, particle_count):
            report = batch.report
            raise RuntimeError(
                "fresh constructive belief requires a complete auditable batch; "
                f"requested={report.requested}, produced={report.produced}, "
                f"termination_reason={report.termination_reason!r}"
            )
        return build_constructive_belief(
            observation,
            batch,
            expected_particles=particle_count,
            weighting=(
                ConstructiveWeightingStrategy.UNIFORM_ACCEPTED_PROPOSALS
            ),
        ).belief

    def advance_to(
        self,
        new_observation: Observation,
        *,
        particle_count: int | None = None,
        seed: int | None = None,
        rng: random.Random | None = None,
        max_play_proposals_per_particle: int = (
            BIR1_INCREMENTAL_PLAY_PROPOSAL_LIMIT
        ),
        resample_ess_fraction: float = BIR1_RESAMPLE_ESS_FRACTION,
    ) -> "MarshalParticleBelief":
        """Advance this belief through newly visible public events."""

        from .bootstrap_filter import advance_belief

        return advance_belief(
            self,
            new_observation,
            particle_count=particle_count,
            seed=seed,
            rng=rng,
            max_play_proposals_per_particle=max_play_proposals_per_particle,
            resample_ess_fraction=resample_ess_fraction,
        )

    @staticmethod
    def _normalized(
        particles: tuple[MarshalParticle, ...],
    ) -> tuple[MarshalParticle, ...]:
        if not particles:
            return ()
        _validate_particle_weights(particles)
        total = math.fsum(particle.weight for particle in particles)
        if not math.isfinite(total) or total <= 0.0:
            raise ValueError("particle weights must have positive finite mass")
        return tuple(
            replace(particle, weight=particle.weight / total)
            for particle in particles
        )

    @property
    def is_empty(self) -> bool:
        return not self.particles

    @property
    def unique_particle_count(self) -> int:
        return len({particle.world_key for particle in self.particles})

    @property
    def effective_sample_size(self) -> float:
        """Return entry-weight ESS; clones remain separate entries.

        After resampling this value can be high even when many entries refer
        to the same physical world.  ``world_effective_sample_size`` is the
        relevant diversity diagnostic in that case.
        """

        denominator = sum(particle.weight**2 for particle in self.particles)
        if denominator <= 0.0:
            return 0.0
        return min(float(len(self.particles)), 1.0 / denominator)

    @property
    def world_effective_sample_size(self) -> float:
        """Return ESS after masses of identical physical worlds are combined."""

        masses = _mass_by_key(self.particles, lambda particle: particle.world_key)
        denominator = sum(mass * mass for mass in masses.values())
        return 0.0 if denominator <= 0.0 else 1.0 / denominator

    @property
    def max_world_mass(self) -> float:
        """Return the posterior mass of the most common physical world."""

        masses = _mass_by_key(self.particles, lambda particle: particle.world_key)
        return max(masses.values(), default=0.0)

    def _hidden_hypothesis_masses(self) -> dict[tuple[int, ...], float]:
        masses: dict[tuple[int, ...], float] = {}
        for particle in self.particles:
            hypothesis = tuple(sorted(self.current_hidden_hideouts(particle)))
            masses[hypothesis] = masses.get(hypothesis, 0.0) + particle.weight
        return masses

    @property
    def unique_hidden_hypothesis_count(self) -> int:
        """Count distinct current unrevealed-Hideout hypotheses."""

        return len(self._hidden_hypothesis_masses())

    @property
    def hidden_hypothesis_effective_sample_size(self) -> float:
        """Return ESS after worlds with the same hidden route are combined."""

        masses = self._hidden_hypothesis_masses().values()
        denominator = sum(mass * mass for mass in masses)
        return 0.0 if denominator <= 0.0 else 1.0 / denominator

    @property
    def max_hidden_hypothesis_mass(self) -> float:
        """Return posterior mass of the most likely hidden-route hypothesis."""

        return max(self._hidden_hypothesis_masses().values(), default=0.0)

    @property
    def sampling_acceptance_rate(self) -> float:
        if self.sampling_attempts <= 0:
            return 0.0
        return self.sampling_accepted / self.sampling_attempts

    @property
    def marginals(self) -> Mapping[int, float]:
        """Posterior probability that each number is currently unrevealed."""

        probabilities: dict[int, float] = {}
        for particle in self.particles:
            hidden = self.current_hidden_hideouts(particle)
            for number in hidden:
                probabilities[number] = probabilities.get(number, 0.0) + particle.weight
        return MappingProxyType(dict(sorted(probabilities.items())))

    def probability_hidden(self, number: int) -> float:
        return self.marginals.get(number, 0.0)

    def joint_success(self, numbers: Iterable[int]) -> float:
        """Return ``P(numbers subset of current hidden route | I_M)``.

        This is evaluated jointly in each particle.  It intentionally does not
        multiply marginal probabilities, because route numbers are strongly
        dependent.
        """

        requested = frozenset(numbers)
        if not requested:
            return 0.0
        if any(number not in GUESSABLE_CARDS for number in requested):
            return 0.0
        return sum(
            particle.weight
            for particle in self.particles
            if requested.issubset(self.current_hidden_hideouts(particle))
        )

    def current_hidden_hideouts(
        self,
        particle: MarshalParticle,
    ) -> frozenset[int]:
        """Return the unrevealed Hideout values in one sampled hypothesis."""

        return frozenset(
            value
            for index, value in enumerate(particle.route_hideouts)
            if index
            and index not in self._revealed_indices
            and value != 42
        )

    def probability_fugitive_can_play_42(self) -> float:
        """Posterior chance that the current Fugitive hand can legally play 42.

        This is an immediate-hand risk estimate; it does not assume which pile
        the Fugitive will choose on a future draw.
        """

        probability = 0.0
        for particle in self.particles:
            if 42 not in particle.fugitive_hand or particle.route_hideouts[-1] == 42:
                continue
            sprint_capacity = 3 + sum(
                sprint_value(card)
                for card in particle.fugitive_hand
                if card != 42
            )
            if 42 - particle.route_hideouts[-1] <= sprint_capacity:
                probability += particle.weight
        return probability

    def draw_card_posterior(self, pile: int) -> Mapping[int, float]:
        """Distribution of the Marshal's next card from ``pile``."""

        if isinstance(pile, bool) or not isinstance(pile, int) or not 0 <= pile < 3:
            raise ValueError("pile must be 0, 1, or 2")
        probabilities: dict[int, float] = {}
        for particle in self.particles:
            contents = particle.remaining_piles[pile]
            if not contents:
                continue
            share = particle.weight / len(contents)
            for card in contents:
                probabilities[card] = probabilities.get(card, 0.0) + share
        total = sum(probabilities.values())
        if total > 0.0:
            probabilities = {
                card: probability / total
                for card, probability in probabilities.items()
            }
        return MappingProxyType(dict(sorted(probabilities.items())))

    def draw_outcome_statistics(
        self,
    ) -> tuple[
        Mapping[int, MarshalDrawOutcomeStatistics],
        Mapping[int, MarshalDrawOutcomeStatistics],
        Mapping[int, MarshalDrawOutcomeStatistics],
    ]:
        """Return cached sufficient statistics for every possible own draw.

        The conditioned-belief reference evaluator constructs one belief
        for every possible card, then repeatedly scans it for marginals,
        routes, joint guesses, and escape risk.  This method computes the same
        conditioned quantities for all three piles in one pass over the base
        particles.  Accumulator masses are unnormalized Bayes likelihoods;
        final values are normalized separately for each observed card.
        """

        cached = self._draw_outcome_statistics_cache
        if cached is not None:
            return cached

        accumulators: list[dict[int, _DrawOutcomeAccumulator]] = [
            {},
            {},
            {},
        ]
        for particle in self.particles:
            hidden_route = tuple(sorted(self.current_hidden_hideouts(particle)))
            hidden_mask = 0
            for number in hidden_route:
                hidden_mask |= 1 << number

            hand = particle.fugitive_hand
            previous = particle.route_hideouts[-1]
            immediate_escape = _can_play_42(hand, previous)
            favorable_by_pile = tuple(
                tuple(
                    _can_play_42(tuple(sorted((*hand, card))), previous)
                    for card in contents
                )
                for contents in particle.remaining_piles
            )
            favorable_counts = tuple(
                sum(flags) for flags in favorable_by_pile
            )

            for draw_pile, contents in enumerate(particle.remaining_piles):
                if not contents:
                    continue
                likelihood = particle.weight / len(contents)
                for card_index, card in enumerate(contents):
                    accumulator = accumulators[draw_pile].setdefault(
                        card,
                        _DrawOutcomeAccumulator(),
                    )
                    accumulator.mass += likelihood
                    for number in hidden_route:
                        accumulator.marginals[number] = (
                            accumulator.marginals.get(number, 0.0)
                            + likelihood
                        )
                    if hidden_route:
                        accumulator.hidden_routes[hidden_route] = (
                            accumulator.hidden_routes.get(hidden_route, 0.0)
                            + likelihood
                        )
                        accumulator.hidden_masks[hidden_mask] = (
                            accumulator.hidden_masks.get(hidden_mask, 0.0)
                            + likelihood
                        )
                    if immediate_escape:
                        accumulator.immediate_risk += likelihood

                    for future_pile, future_contents in enumerate(
                        particle.remaining_piles
                    ):
                        future_size = len(future_contents) - int(
                            future_pile == draw_pile
                        )
                        if future_size <= 0:
                            continue
                        accumulator.any_nonempty = True
                        favorable = favorable_counts[future_pile]
                        if future_pile == draw_pile:
                            favorable -= favorable_by_pile[future_pile][card_index]
                        accumulator.pile_risks[future_pile] += (
                            likelihood * favorable / future_size
                        )

        finalized: list[Mapping[int, MarshalDrawOutcomeStatistics]] = []
        for pile, by_card in enumerate(accumulators):
            pile_mass = sum(item.mass for item in by_card.values())
            outcomes: dict[int, MarshalDrawOutcomeStatistics] = {}
            for card, item in sorted(by_card.items()):
                if item.mass <= 0.0 or pile_mass <= 0.0:
                    continue
                inverse_mass = 1.0 / item.mass
                normalized_marginals = MappingProxyType(
                    {
                        number: mass * inverse_mass
                        for number, mass in sorted(item.marginals.items())
                    }
                )
                normalized_routes = MappingProxyType(
                    {
                        route: mass * inverse_mass
                        for route, mass in sorted(item.hidden_routes.items())
                    }
                )
                normalized_masks = MappingProxyType(
                    {
                        mask: mass * inverse_mass
                        for mask, mass in sorted(item.hidden_masks.items())
                    }
                )
                if item.any_nonempty:
                    escape_risk = max(item.pile_risks) * inverse_mass
                else:
                    escape_risk = item.immediate_risk * inverse_mass
                outcomes[card] = MarshalDrawOutcomeStatistics(
                    pile=pile,
                    card=card,
                    probability=item.mass / pile_mass,
                    marginals=normalized_marginals,
                    hidden_route_masses=normalized_routes,
                    hidden_mask_masses=normalized_masks,
                    escape_risk=escape_risk,
                )
            finalized.append(MappingProxyType(outcomes))

        result = (finalized[0], finalized[1], finalized[2])
        self._draw_outcome_statistics_cache = result
        return result

    def conditioned_on_marshal_draw(
        self,
        pile: int,
        card: int,
    ) -> "MarshalParticleBelief":
        """Bayes-condition particles after the Marshal privately draws ``card``.

        Pile order is integrated out.  Consequently a particle survives when
        the card is in its pile, with likelihood ``1 / pile_size``.
        """

        if isinstance(pile, bool) or not isinstance(pile, int) or not 0 <= pile < 3:
            raise ValueError("pile must be 0, 1, or 2")
        updated: list[MarshalParticle] = []
        for particle in self.particles:
            contents = particle.remaining_piles[pile]
            if card not in contents or not contents:
                continue
            piles = list(particle.remaining_piles)
            piles[pile] = tuple(value for value in contents if value != card)
            updated.append(
                replace(
                    particle,
                    marshal_hand=tuple(sorted((*particle.marshal_hand, card))),
                    remaining_piles=tuple(piles),  # type: ignore[arg-type]
                    weight=particle.weight / len(contents),
                )
            )
        return MarshalParticleBelief(
            self.observation,
            updated,
            sampling_attempts=self.sampling_attempts,
            sampling_exhausted=self.sampling_exhausted or not updated,
            sampling_accepted=self.sampling_accepted,
            pre_resample_effective_sample_size=(
                _effective_sample_size(updated)
            ),
            resampling_count=self.resampling_count,
        )

    def summary(self) -> ParticleBeliefSummary:
        """Return a stable summary suitable for information-leakage tests."""

        return ParticleBeliefSummary(
            particle_count=len(self.particles),
            unique_particle_count=self.unique_particle_count,
            effective_sample_size=self.effective_sample_size,
            pre_resample_effective_sample_size=(
                self.pre_resample_effective_sample_size
            ),
            world_effective_sample_size=self.world_effective_sample_size,
            max_world_mass=self.max_world_mass,
            unique_hidden_hypothesis_count=(
                self.unique_hidden_hypothesis_count
            ),
            hidden_hypothesis_effective_sample_size=(
                self.hidden_hypothesis_effective_sample_size
            ),
            max_hidden_hypothesis_mass=self.max_hidden_hypothesis_mass,
            resampling_count=self.resampling_count,
            sampling_attempts=self.sampling_attempts,
            sampling_accepted=self.sampling_accepted,
            sampling_acceptance_rate=self.sampling_acceptance_rate,
            sampling_exhausted=self.sampling_exhausted,
            marginals=tuple(self.marginals.items()),
            draw_posteriors=tuple(
                tuple(self.draw_card_posterior(pile).items()) for pile in range(3)
            ),
        )


def _validate_particle_weights(particles: Sequence[MarshalParticle]) -> None:
    for particle in particles:
        weight = particle.weight
        if (
            isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not math.isfinite(weight)
            or weight < 0.0
        ):
            raise ValueError(
                "particle weights must be finite non-negative numbers"
            )


def _aggregate_particles(
    particles: Sequence[MarshalParticle],
) -> tuple[MarshalParticle, ...]:
    """Combine duplicate physical worlds while preserving their total mass."""

    if not particles:
        return ()
    _validate_particle_weights(particles)
    representatives: dict[tuple[object, ...], MarshalParticle] = {}
    masses: dict[tuple[object, ...], float] = {}
    for particle in particles:
        key = particle.world_key
        representatives.setdefault(key, particle)
        masses[key] = masses.get(key, 0.0) + particle.weight
    if any(not math.isfinite(mass) for mass in masses.values()):
        raise ValueError("aggregated particle mass must be finite")
    return tuple(
        replace(representatives[key], weight=mass)
        for key, mass in masses.items()
    )


def _effective_sample_size(particles: Sequence[MarshalParticle]) -> float:
    if not particles:
        return 0.0
    _validate_particle_weights(particles)
    total = math.fsum(particle.weight for particle in particles)
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("particle weights must have positive finite mass")
    denominator = sum((particle.weight / total) ** 2 for particle in particles)
    return 0.0 if denominator <= 0.0 else 1.0 / denominator


def _mass_by_key(
    particles: Sequence[MarshalParticle],
    key: Callable[[MarshalParticle], object],
) -> dict[object, float]:
    if not particles:
        return {}
    _validate_particle_weights(particles)
    total = math.fsum(particle.weight for particle in particles)
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("particle weights must have positive finite mass")
    masses: dict[object, float] = {}
    for particle in particles:
        identity = key(particle)
        masses[identity] = masses.get(identity, 0.0) + particle.weight / total
    return masses


def _matches_public_projection_fields(
    particle: MarshalParticle,
    observation: Observation,
    *,
    revealed_numbers: frozenset[int],
) -> bool:
    if particle.marshal_hand != tuple(sorted(observation.hand)):
        return False
    if tuple(map(len, particle.remaining_piles)) != observation.pile_sizes:
        return False
    if len(particle.route_hideouts) != len(observation.route):
        return False
    if len(particle.route_sprints) != len(observation.route):
        return False
    fugitive_records = tuple(
        record
        for record in observation.draw_history
        if record.role is Role.FUGITIVE
    )
    if len(particle.fugitive_draws) != len(fugitive_records):
        return False
    if any(
        card not in PILE_CARDS[record.pile]
        for card, record in zip(
            particle.fugitive_draws, fugitive_records, strict=True
        )
    ):
        return False

    for index, slot in enumerate(observation.route):
        hideout = particle.route_hideouts[index]
        sprints = particle.route_sprints[index]
        if slot.hideout is not None and slot.hideout != hideout:
            return False
        if slot.sprint_count != len(sprints):
            return False
        if slot.sprint_cards is not None and tuple(slot.sprint_cards) != sprints:
            return False
        if index:
            if slot.revealed != (hideout in revealed_numbers):
                return False
            if not is_legal_fugitive_action(
                FugitiveAction(hideout, sprints),
                (hideout, *sprints),
                particle.route_hideouts[index - 1],
                allow_pass=False,
            ):
                return False
    return particle.all_cards_are_unique


def particle_matches_public_projection(
    particle: MarshalParticle,
    observation: Observation,
) -> bool:
    """Return whether a particle projects to the visible observation.

    The revealed-number set is derived from the observation here so external
    callers do not need to duplicate that bookkeeping.  This deliberately
    does not validate hidden ownership, pile provenance, or draw deadlines;
    use ``is_complete_world_consistent`` for a complete-world check.
    """

    revealed_numbers = frozenset(
        slot.hideout
        for index, slot in enumerate(observation.route)
        if index and slot.revealed and slot.hideout is not None
    )
    if not _matches_public_projection_fields(
        particle,
        observation,
        revealed_numbers=revealed_numbers,
    ):
        return False

    revealed_at_time: set[int] = set()
    route_length = len(particle.route_hideouts) - 1
    for record in observation.guess_history:
        if not 0 <= record.route_length <= route_length:
            return False
        hidden_at_time = (
            set(particle.route_hideouts[1 : record.route_length + 1])
            - revealed_at_time
            - {42}
        )
        success = bool(record.numbers) and all(
            number in hidden_at_time for number in record.numbers
        )
        if success != record.success:
            return False
        if success:
            revealed_at_time.update(record.numbers)
    return revealed_at_time == set(revealed_numbers)


__all__ = [
    "BIR1_INCREMENTAL_PLAY_PROPOSAL_LIMIT",
    "BIR1_RESAMPLE_ESS_FRACTION",
    "DEFAULT_PARTICLE_COUNT",
    "IncompatibleObservationError",
    "MarshalDrawOutcomeStatistics",
    "MarshalParticle",
    "MarshalParticleBelief",
    "ParticleBeliefSummary",
    "particle_matches_public_projection",
]
