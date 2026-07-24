"""Observation-only particle belief for the Marshal.

The sampler in this module never receives a :class:`~fugitive.engine.GameEngine`
or any private engine state.  A particle is a complete policy-agnostic
hypothesis for the hidden Fugitive cards that is consistent with the public
history and the Marshal's own cards.

This is a constraint-guided rejection sampler, not an exactly calibrated
Bayesian posterior. Accepted proposals receive equal weight. This deliberate
approximation is suitable for a baseline because observationally
indistinguishable worlds induce exactly the same estimate with the same seed.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import combinations
import math
import random
from types import MappingProxyType
from typing import Iterable, Mapping, Sequence

from fugitive.model import (
    DrawRecord,
    FugitiveAction,
    GuessRecord,
    Observation,
    PlayView,
    Role,
)
from fugitive.rules import (
    GUESSABLE_CARDS,
    PILE_CARDS,
    is_legal_fugitive_action,
    sprint_value,
)


DEFAULT_PARTICLE_COUNT = 2_000
DEFAULT_MAX_INCREMENTAL_PLAY_PROPOSALS = 64
_MAX_EXACT_PLAY_COMBINATIONS = 4_096
_INITIAL_FUGITIVE_CARDS = frozenset((1, 2, 3, 42))
_CARD_TO_PILE = {
    card: pile for pile, cards in enumerate(PILE_CARDS) for card in cards
}
_INFINITY = 10**9


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
        max_attempts: int | None = None,
    ) -> "MarshalParticleBelief":
        """Sample complete hypotheses using only ``observation``.

        When rejection support is sparse, every valid sample found is kept and
        resampled to the requested population size.  A contradictory
        observation safely produces an empty belief instead of manufacturing
        an information-leaking hypothesis.  ``sampling_exhausted`` reports
        either fallback.
        """

        if observation.role is not Role.MARSHAL:
            raise ValueError("particle sampling requires a Marshal observation")
        if (
            isinstance(particle_count, bool)
            or not isinstance(particle_count, int)
            or particle_count <= 0
        ):
            raise ValueError("particle_count must be a positive integer")
        if seed is not None and rng is not None:
            raise ValueError("pass either seed or rng, not both")
        owned_rng = rng if rng is not None else random.Random(seed)
        if max_attempts is None:
            max_attempts = max(250, particle_count * 24)
        if (
            isinstance(max_attempts, bool)
            or not isinstance(max_attempts, int)
            or max_attempts <= 0
        ):
            raise ValueError("max_attempts must be a positive integer")

        sampler = _ObservationSampler(observation, owned_rng)
        particles: list[MarshalParticle] = []
        attempts = 0
        while sampler.valid and len(particles) < particle_count and attempts < max_attempts:
            attempts += 1
            particle = sampler.sample()
            if particle is not None:
                particles.append(particle)

        accepted = len(particles)
        exhausted = accepted < particle_count
        if particles and exhausted:
            source = tuple(particles)
            while len(particles) < particle_count:
                particles.append(owned_rng.choice(source))

        return cls(
            observation,
            particles,
            sampling_attempts=attempts,
            sampling_exhausted=exhausted,
            sampling_accepted=accepted,
        )

    def advance_to(
        self,
        new_observation: Observation,
        *,
        particle_count: int | None = None,
        seed: int | None = None,
        rng: random.Random | None = None,
        max_play_proposals_per_particle: int = (
            DEFAULT_MAX_INCREMENTAL_PLAY_PROPOSALS
        ),
    ) -> "MarshalParticleBelief":
        """Advance this belief through newly visible public events.

        The old draw and guess histories must be literal prefixes of the new
        histories.  Play history and route entries may refine formerly hidden
        identities after a successful guess, but their stable public fields
        must remain unchanged.

        Hidden draws are propagated over every possible card in each particle.
        Hidden placements enumerate small action sets, then retain a bounded
        uniform reservoir; large action sets use deterministic-RNG proposals.
        This bounded placement proposal is the only additional approximation
        beyond the particle population itself. Systematic resampling follows
        every new event.
        """

        if new_observation.role is not Role.MARSHAL:
            raise ValueError("incremental particle filtering requires a Marshal observation")
        if seed is not None and rng is not None:
            raise ValueError("pass either seed or rng, not both")
        if particle_count is None:
            particle_count = len(self.particles) or DEFAULT_PARTICLE_COUNT
        if (
            isinstance(particle_count, bool)
            or not isinstance(particle_count, int)
            or particle_count <= 0
        ):
            raise ValueError("particle_count must be a positive integer")
        if (
            isinstance(max_play_proposals_per_particle, bool)
            or not isinstance(max_play_proposals_per_particle, int)
            or max_play_proposals_per_particle <= 0
        ):
            raise ValueError(
                "max_play_proposals_per_particle must be a positive integer"
            )

        _validate_observation_extension(self.observation, new_observation)
        owned_rng = rng if rng is not None else random.Random(seed)
        if not self.particles:
            return MarshalParticleBelief(
                new_observation,
                (),
                sampling_attempts=self.sampling_attempts,
                sampling_exhausted=True,
                sampling_accepted=self.sampling_accepted,
            )

        particles = tuple(self.particles)
        revealed_numbers = {
            slot.hideout
            for index, slot in enumerate(self.observation.route)
            if index and slot.revealed and slot.hideout is not None
        }
        events = _new_public_events(self.observation, new_observation)
        for kind, record in events:
            if kind == "draw":
                assert isinstance(record, DrawRecord)
                particles = _advance_draw(particles, record)
            elif kind == "play":
                assert isinstance(record, PlayView)
                particles = _advance_play(
                    particles,
                    record,
                    new_observation,
                    rng=owned_rng,
                    max_proposals=max_play_proposals_per_particle,
                )
            else:
                assert kind == "guess" and isinstance(record, GuessRecord)
                particles = _advance_guess(
                    particles,
                    record,
                    revealed_numbers=frozenset(revealed_numbers),
                )
                if record.success:
                    revealed_numbers.update(record.numbers)

            particles = _systematic_resample(
                particles,
                particle_count,
                rng=owned_rng,
            )
            if not particles:
                return MarshalParticleBelief(
                    new_observation,
                    (),
                    sampling_attempts=self.sampling_attempts,
                    sampling_exhausted=True,
                    sampling_accepted=self.sampling_accepted,
                )

        particles = tuple(
            particle
            for particle in particles
            if _particle_matches_observation(
                particle,
                new_observation,
                revealed_numbers=frozenset(revealed_numbers),
            )
        )
        particles = _systematic_resample(
            particles,
            particle_count,
            rng=owned_rng,
        )
        return MarshalParticleBelief(
            new_observation,
            particles,
            sampling_attempts=self.sampling_attempts,
            sampling_exhausted=self.sampling_exhausted or not particles,
            sampling_accepted=self.sampling_accepted,
        )

    @staticmethod
    def _normalized(
        particles: tuple[MarshalParticle, ...],
    ) -> tuple[MarshalParticle, ...]:
        if not particles:
            return ()
        total = sum(max(0.0, particle.weight) for particle in particles)
        if total <= 0.0:
            uniform = 1.0 / len(particles)
            return tuple(replace(particle, weight=uniform) for particle in particles)
        return tuple(
            replace(particle, weight=max(0.0, particle.weight) / total)
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
        """Return ``1 / sum(w^2)`` for the normalized particle weights."""

        denominator = sum(particle.weight**2 for particle in self.particles)
        if denominator <= 0.0:
            return 0.0
        return min(float(len(self.particles)), 1.0 / denominator)

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

        The legacy draw evaluator constructs one conditioned particle belief
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
        )

    def summary(self) -> ParticleBeliefSummary:
        """Return a stable summary suitable for information-leakage tests."""

        return ParticleBeliefSummary(
            particle_count=len(self.particles),
            unique_particle_count=self.unique_particle_count,
            effective_sample_size=self.effective_sample_size,
            sampling_attempts=self.sampling_attempts,
            sampling_accepted=self.sampling_accepted,
            sampling_acceptance_rate=self.sampling_acceptance_rate,
            sampling_exhausted=self.sampling_exhausted,
            marginals=tuple(self.marginals.items()),
            draw_posteriors=tuple(
                tuple(self.draw_card_posterior(pile).items()) for pile in range(3)
            ),
        )


def _validate_observation_extension(
    old: Observation,
    new: Observation,
) -> None:
    if old.role is not Role.MARSHAL or new.role is not Role.MARSHAL:
        raise ValueError("both observations must belong to the Marshal")
    if old.round_number > new.round_number:
        raise ValueError("the new observation precedes the old round")
    if len(old.draw_history) > len(new.draw_history) or tuple(
        new.draw_history[: len(old.draw_history)]
    ) != old.draw_history:
        raise ValueError("old draw history is not a prefix of the new history")
    if len(old.guess_history) > len(new.guess_history) or tuple(
        new.guess_history[: len(old.guess_history)]
    ) != old.guess_history:
        raise ValueError("old guess history is not a prefix of the new history")
    if len(old.play_history) > len(new.play_history):
        raise ValueError("old play history is longer than the new history")
    for before, after in zip(old.play_history, new.play_history, strict=False):
        if not _play_view_refines(before, after):
            raise ValueError("new play history does not refine its old prefix")

    if len(old.route) > len(new.route):
        raise ValueError("the new route is shorter than the old route")
    for before, after in zip(old.route, new.route, strict=False):
        if (
            before.index != after.index
            or before.sprint_count != after.sprint_count
            or (before.hideout is not None and before.hideout != after.hideout)
            or (
                before.sprint_cards is not None
                and before.sprint_cards != after.sprint_cards
            )
            or (before.revealed and not after.revealed)
        ):
            raise ValueError("new route does not refine the old visible route")
    if not set(old.hand).issubset(new.hand):
        raise ValueError("cards disappeared from the Marshal hand")
    if any(after > before for before, after in zip(old.pile_sizes, new.pile_sizes)):
        raise ValueError("a draw pile grew between observations")


def _play_view_refines(before: PlayView, after: PlayView) -> bool:
    return (
        before.route_index == after.route_index
        and before.round_number == after.round_number
        and before.passed == after.passed
        and before.sprint_count == after.sprint_count
        and (before.hideout is None or before.hideout == after.hideout)
        and (
            before.sprint_cards is None
            or before.sprint_cards == after.sprint_cards
        )
    )


def _new_public_events(
    old: Observation,
    new: Observation,
) -> tuple[tuple[str, DrawRecord | PlayView | GuessRecord], ...]:
    ordered: list[
        tuple[int, int, int, str, DrawRecord | PlayView | GuessRecord]
    ] = []
    for index, record in enumerate(
        new.draw_history[len(old.draw_history) :],
        start=len(old.draw_history),
    ):
        stage = 0 if record.role is Role.FUGITIVE else 2
        ordered.append((record.round_number, stage, index, "draw", record))
    for index, record in enumerate(
        new.play_history[len(old.play_history) :],
        start=len(old.play_history),
    ):
        ordered.append((record.round_number, 1, index, "play", record))
    for index, record in enumerate(
        new.guess_history[len(old.guess_history) :],
        start=len(old.guess_history),
    ):
        ordered.append((record.round_number, 3, index, "guess", record))
    ordered.sort(key=lambda item: item[:3])
    return tuple((kind, record) for _round, _stage, _index, kind, record in ordered)


def _advance_draw(
    particles: Sequence[MarshalParticle],
    record: DrawRecord,
) -> tuple[MarshalParticle, ...]:
    if not 0 <= record.pile < 3:
        return ()
    children: list[MarshalParticle] = []
    for particle in particles:
        contents = particle.remaining_piles[record.pile]
        if not contents:
            continue
        if record.role is Role.MARSHAL:
            if record.card is None or record.card not in contents:
                continue
            piles = list(particle.remaining_piles)
            piles[record.pile] = tuple(
                card for card in contents if card != record.card
            )
            children.append(
                replace(
                    particle,
                    marshal_hand=tuple(
                        sorted((*particle.marshal_hand, record.card))
                    ),
                    remaining_piles=tuple(piles),  # type: ignore[arg-type]
                    weight=particle.weight / len(contents),
                )
            )
            continue

        if record.card is not None:
            continue
        share = particle.weight / len(contents)
        for card in contents:
            piles = list(particle.remaining_piles)
            piles[record.pile] = tuple(
                value for value in contents if value != card
            )
            children.append(
                replace(
                    particle,
                    fugitive_hand=tuple(sorted((*particle.fugitive_hand, card))),
                    fugitive_draws=(*particle.fugitive_draws, card),
                    remaining_piles=tuple(piles),  # type: ignore[arg-type]
                    weight=share,
                )
            )
    return tuple(children)


def _advance_play(
    particles: Sequence[MarshalParticle],
    record: PlayView,
    target: Observation,
    *,
    rng: random.Random,
    max_proposals: int,
) -> tuple[MarshalParticle, ...]:
    if record.passed:
        if record.route_index is not None or record.sprint_count != 0:
            return ()
        return tuple(particles)
    if record.route_index is None or not 0 < record.route_index < len(target.route):
        return ()
    slot = target.route[record.route_index]
    if record.sprint_count != slot.sprint_count:
        return ()

    children: list[MarshalParticle] = []
    for particle in particles:
        if record.route_index != len(particle.route_hideouts):
            continue
        actions = _matching_incremental_plays(
            particle,
            record,
            slot_hideout=slot.hideout,
            slot_sprint_cards=slot.sprint_cards,
            rng=rng,
            limit=max_proposals,
        )
        if not actions:
            continue
        share = particle.weight / len(actions)
        for action in actions:
            assert action.hideout is not None
            spent = {action.hideout, *action.sprint_cards}
            children.append(
                replace(
                    particle,
                    fugitive_hand=tuple(
                        card for card in particle.fugitive_hand if card not in spent
                    ),
                    route_hideouts=(*particle.route_hideouts, action.hideout),
                    route_sprints=(
                        *particle.route_sprints,
                        tuple(sorted(action.sprint_cards)),
                    ),
                    weight=share,
                )
            )
    return tuple(children)


def _matching_incremental_plays(
    particle: MarshalParticle,
    record: PlayView,
    *,
    slot_hideout: int | None,
    slot_sprint_cards: tuple[int, ...] | None,
    rng: random.Random,
    limit: int,
) -> tuple[FugitiveAction, ...]:
    hand = particle.fugitive_hand
    previous = particle.route_hideouts[-1]
    fixed_hideout = record.hideout if record.hideout is not None else slot_hideout
    if (
        record.hideout is not None
        and slot_hideout is not None
        and record.hideout != slot_hideout
    ):
        return ()
    fixed_sprints = (
        record.sprint_cards
        if record.sprint_cards is not None
        else slot_sprint_cards
    )
    if (
        record.sprint_cards is not None
        and slot_sprint_cards is not None
        and record.sprint_cards != slot_sprint_cards
    ):
        return ()

    if fixed_hideout is not None:
        hideouts = (fixed_hideout,)
    else:
        # Card 42 is public as soon as it is a Hideout.
        hideouts = tuple(card for card in hand if previous < card < 42)
    hideouts = tuple(card for card in hideouts if card in hand and card > previous)
    if not hideouts:
        return ()

    def legal(action: FugitiveAction) -> bool:
        return (
            len(action.sprint_cards) == record.sprint_count
            and is_legal_fugitive_action(
                action,
                hand,
                previous,
                allow_pass=False,
            )
        )

    if fixed_sprints is not None:
        if len(fixed_sprints) != record.sprint_count:
            return ()
        actions = tuple(
            FugitiveAction(hideout, tuple(sorted(fixed_sprints)))
            for hideout in hideouts
            if legal(FugitiveAction(hideout, tuple(sorted(fixed_sprints))))
        )
        return actions[:limit]

    total_combinations = sum(
        math.comb(len(hand) - 1, record.sprint_count)
        if record.sprint_count <= len(hand) - 1
        else 0
        for _hideout in hideouts
    )
    if total_combinations <= _MAX_EXACT_PLAY_COMBINATIONS:
        reservoir: list[FugitiveAction] = []
        valid_seen = 0
        for hideout in hideouts:
            sprint_candidates = tuple(card for card in hand if card != hideout)
            for sprint_cards in combinations(
                sprint_candidates, record.sprint_count
            ):
                action = FugitiveAction(hideout, tuple(sorted(sprint_cards)))
                if not legal(action):
                    continue
                valid_seen += 1
                if len(reservoir) < limit:
                    reservoir.append(action)
                else:
                    replacement = rng.randrange(valid_seen)
                    if replacement < limit:
                        reservoir[replacement] = action
        return tuple(sorted(set(reservoir), key=_action_key))

    # Large hands can make C(H, k) enormous.  Sample a bounded proposal set.
    sampled: set[FugitiveAction] = set()
    attempts = max(256, limit * 24)
    for _ in range(attempts):
        hideout = rng.choice(hideouts)
        sprint_candidates = tuple(card for card in hand if card != hideout)
        if record.sprint_count > len(sprint_candidates):
            continue
        action = FugitiveAction(
            hideout,
            tuple(sorted(rng.sample(sprint_candidates, record.sprint_count))),
        )
        if legal(action):
            sampled.add(action)
            if len(sampled) >= limit:
                break
    return tuple(sorted(sampled, key=_action_key))


def _action_key(action: FugitiveAction) -> tuple[int, tuple[int, ...]]:
    assert action.hideout is not None
    return action.hideout, action.sprint_cards


def _advance_guess(
    particles: Sequence[MarshalParticle],
    record: GuessRecord,
    *,
    revealed_numbers: frozenset[int],
) -> tuple[MarshalParticle, ...]:
    requested = set(record.numbers)
    survivors: list[MarshalParticle] = []
    for particle in particles:
        if not 0 <= record.route_length < len(particle.route_hideouts):
            continue
        hidden = (
            set(particle.route_hideouts[1 : record.route_length + 1])
            - set(revealed_numbers)
            - {42}
        )
        success = requested.issubset(hidden)
        if success == record.success:
            survivors.append(particle)
    return tuple(survivors)


def _systematic_resample(
    particles: Sequence[MarshalParticle],
    count: int,
    *,
    rng: random.Random,
) -> tuple[MarshalParticle, ...]:
    if not particles:
        return ()
    weights = [max(0.0, particle.weight) for particle in particles]
    total = sum(weights)
    if total <= 0.0:
        weights = [1.0] * len(particles)
        total = float(len(particles))
    cumulative: list[float] = []
    running = 0.0
    for weight in weights:
        running += weight / total
        cumulative.append(running)
    cumulative[-1] = 1.0

    step = 1.0 / count
    threshold = rng.random() * step
    source_index = 0
    result: list[MarshalParticle] = []
    uniform = 1.0 / count
    for sample_index in range(count):
        point = threshold + sample_index * step
        while source_index < len(cumulative) - 1 and point > cumulative[source_index]:
            source_index += 1
        result.append(replace(particles[source_index], weight=uniform))
    return tuple(result)


def _particle_matches_observation(
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


def particle_matches_observation(
    particle: MarshalParticle,
    observation: Observation,
) -> bool:
    """Return whether a complete Marshal particle has the public support.

    The revealed-number set is derived from the observation here so external
    inference backends do not need to duplicate that bookkeeping.
    """

    revealed_numbers = frozenset(
        slot.hideout
        for index, slot in enumerate(observation.route)
        if index and slot.revealed and slot.hideout is not None
    )
    if not _particle_matches_observation(
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


class _ObservationSampler:
    """Constraint-guided proposal sampler internal to this module."""

    def __init__(self, observation: Observation, rng: random.Random) -> None:
        self.observation = observation
        self.rng = rng
        self.valid = True
        self.route_length = len(observation.route) - 1
        self.fugitive_records = tuple(
            record for record in observation.draw_history if record.role is Role.FUGITIVE
        )
        self.marshal_records = tuple(
            record for record in observation.draw_history if record.role is Role.MARSHAL
        )
        self.fugitive_draw_counts = tuple(
            sum(record.pile == pile for record in self.fugitive_records)
            for pile in range(3)
        )
        self.marshal_draw_counts = tuple(
            sum(record.pile == pile for record in self.marshal_records)
            for pile in range(3)
        )
        self.creation_rounds = self._creation_rounds()
        self.known_fugitive_cards = self._known_fugitive_cards()
        self.marshal_hand = tuple(sorted(observation.hand))
        self._validate_public_shape()

    def _creation_rounds(self) -> tuple[int, ...]:
        rounds: list[int | None] = [0, *([None] * max(0, self.route_length))]
        for record in getattr(self.observation, "play_history", ()):
            route_index = getattr(record, "route_index", None)
            passed = bool(getattr(record, "passed", route_index is None))
            if passed or route_index is None:
                continue
            if 0 < route_index < len(rounds):
                rounds[route_index] = int(record.round_number)
        for index in range(1, len(rounds)):
            if rounds[index] is None:
                rounds[index] = next(
                    (
                        record.round_number
                        for record in self.observation.guess_history
                        if record.route_length >= index
                    ),
                    self.observation.round_number,
                )
        return tuple(int(round_number) for round_number in rounds)

    def _known_fugitive_cards(self) -> frozenset[int]:
        cards: set[int] = set(_INITIAL_FUGITIVE_CARDS)
        for index, slot in enumerate(self.observation.route):
            if index and slot.hideout is not None:
                cards.add(slot.hideout)
            if slot.sprint_cards is not None:
                cards.update(slot.sprint_cards)
        return frozenset(cards)

    def _validate_public_shape(self) -> None:
        observation = self.observation
        if (
            len(observation.pile_sizes) != 3
            or any(
                isinstance(size, bool) or not isinstance(size, int) or size < 0
                for size in observation.pile_sizes
            )
            or any(
                isinstance(record.pile, bool)
                or not isinstance(record.pile, int)
                or not 0 <= record.pile < 3
                for record in observation.draw_history
            )
        ):
            self.valid = False
            return
        if not observation.route or observation.route[0].hideout != 0:
            self.valid = False
            return
        if len(self.marshal_hand) != len(set(self.marshal_hand)):
            self.valid = False
            return
        if any(card not in _CARD_TO_PILE for card in self.marshal_hand):
            self.valid = False
            return
        if self.known_fugitive_cards & set(self.marshal_hand):
            self.valid = False
            return

        for pile, original in enumerate(PILE_CARDS):
            if (
                self.fugitive_draw_counts[pile]
                + self.marshal_draw_counts[pile]
                + observation.pile_sizes[pile]
                != len(original)
            ):
                self.valid = False
                return
            marshal_cards = {card for card in self.marshal_hand if card in original}
            if len(marshal_cards) != self.marshal_draw_counts[pile]:
                self.valid = False
                return
            forced_fugitive = {
                card for card in self.known_fugitive_cards if card in original
            }
            if len(forced_fugitive) > self.fugitive_draw_counts[pile]:
                self.valid = False
                return

        visible_marshal_draws = {
            record.card for record in self.marshal_records if record.card is not None
        }
        if visible_marshal_draws != set(self.marshal_hand):
            self.valid = False
            return
        for record in self.marshal_records:
            if record.card is not None and record.card not in PILE_CARDS[record.pile]:
                self.valid = False
                return

        seen: set[int] = set()
        for index, slot in enumerate(observation.route):
            if slot.index != index or slot.sprint_count < 0:
                self.valid = False
                return
            identities = slot.sprint_cards
            if identities is not None and len(identities) != slot.sprint_count:
                self.valid = False
                return
            known = (() if slot.hideout is None or not index else (slot.hideout,))
            stack = identities or ()
            for card in (*known, *stack):
                if card in seen or card == 0 or not 1 <= card <= 42:
                    self.valid = False
                    return
                seen.add(card)

        play_history = getattr(observation, "play_history", ())
        if play_history:
            placed = {
                record.route_index
                for record in play_history
                if not record.passed and record.route_index is not None
            }
            if placed != set(range(1, len(observation.route))):
                self.valid = False
                return
            for record in play_history:
                if record.passed:
                    if record.route_index is not None or record.sprint_count:
                        self.valid = False
                        return
                    continue
                if record.route_index is None:
                    self.valid = False
                    return
                slot = observation.route[record.route_index]
                if record.sprint_count != slot.sprint_count:
                    self.valid = False
                    return
                if record.hideout is not None and record.hideout != slot.hideout:
                    self.valid = False
                    return
                if (
                    record.sprint_cards is not None
                    and record.sprint_cards != slot.sprint_cards
                ):
                    self.valid = False
                    return

    def sample(self) -> MarshalParticle | None:
        owned = self._sample_fugitive_owned_cards()
        if owned is None:
            return None
        assignment = self._sample_route_and_sprints(owned)
        if assignment is None:
            return None
        route, sprints = assignment
        deadlines = self._used_card_deadlines(route, sprints)
        fugitive_draws = self._assign_fugitive_draws(owned, deadlines)
        if fugitive_draws is None:
            return None

        used = set(route[1:])
        used.update(card for stack in sprints for card in stack)
        fugitive_hand = tuple(sorted(owned - used))
        remaining_piles = tuple(
            tuple(
                sorted(
                    set(original)
                    - {card for card in owned if card in original}
                    - {card for card in self.marshal_hand if card in original}
                )
            )
            for original in PILE_CARDS
        )
        if tuple(map(len, remaining_piles)) != self.observation.pile_sizes:
            return None
        particle = MarshalParticle(
            fugitive_hand=fugitive_hand,
            marshal_hand=self.marshal_hand,
            route_hideouts=route,
            route_sprints=sprints,
            fugitive_draws=fugitive_draws,
            remaining_piles=remaining_piles,  # type: ignore[arg-type]
        )
        return particle if particle.all_cards_are_unique else None

    def _sample_fugitive_owned_cards(self) -> set[int] | None:
        owned = set(_INITIAL_FUGITIVE_CARDS)
        marshal = set(self.marshal_hand)
        for pile, original in enumerate(PILE_CARDS):
            forced = {card for card in self.known_fugitive_cards if card in original}
            need = self.fugitive_draw_counts[pile] - len(forced)
            candidates = sorted(set(original) - marshal - forced)
            if need < 0 or need > len(candidates):
                return None
            owned.update(forced)
            owned.update(self.rng.sample(candidates, need))
        return owned

    def _sample_route_and_sprints(
        self,
        owned: set[int],
    ) -> tuple[tuple[int, ...], tuple[tuple[int, ...], ...]] | None:
        known_sprints = {
            card
            for slot in self.observation.route
            if slot.sprint_cards is not None
            for card in slot.sprint_cards
        }
        available_route = owned - known_sprints - {0}
        route: list[int] = [0]

        def search(
            index: int,
            available: set[int],
        ) -> tuple[tuple[int, ...], tuple[tuple[int, ...], ...]] | None:
            if index == len(self.observation.route):
                route_tuple = tuple(route)
                if not self._guess_history_consistent(route_tuple):
                    return None
                sprints = self._sample_sprints(owned, route_tuple)
                if sprints is None:
                    return None
                return route_tuple, sprints

            slot = self.observation.route[index]
            previous = route[-1]
            if slot.sprint_cards is None:
                edge_capacity = 3 + 2 * slot.sprint_count
            else:
                edge_capacity = 3 + sum(sprint_value(card) for card in slot.sprint_cards)
            if slot.hideout is not None:
                candidates = [slot.hideout]
            else:
                candidates = [
                    card
                    for card in available
                    if card != 42 and previous < card <= previous + edge_capacity
                ]
                self.rng.shuffle(candidates)

            next_known_index, next_known = self._next_known_hideout(index)
            for card in candidates:
                if card not in available or card <= previous or card - previous > edge_capacity:
                    continue
                if next_known is not None:
                    remaining_edges = next_known_index - index
                    if card + remaining_edges > next_known:
                        continue
                    maximum_reach = card + sum(
                        self._public_edge_capacity(edge)
                        for edge in range(index + 1, next_known_index + 1)
                    )
                    if maximum_reach < next_known:
                        continue
                route.append(card)
                selected = search(index + 1, available - {card})
                if selected is not None:
                    return selected
                route.pop()
            return None

        return search(1, available_route)

    def _next_known_hideout(self, after_index: int) -> tuple[int, int | None]:
        for index in range(after_index + 1, len(self.observation.route)):
            hideout = self.observation.route[index].hideout
            if hideout is not None:
                return index, hideout
        return len(self.observation.route), None

    def _public_edge_capacity(self, index: int) -> int:
        slot = self.observation.route[index]
        if slot.sprint_cards is None:
            return 3 + 2 * slot.sprint_count
        return 3 + sum(sprint_value(card) for card in slot.sprint_cards)

    def _sample_sprints(
        self,
        owned: set[int],
        route: tuple[int, ...],
    ) -> tuple[tuple[int, ...], ...] | None:
        stacks: list[tuple[int, ...] | None] = [()] * len(route)
        used = set(route[1:])
        available = owned - used
        unknown: list[tuple[int, int, int]] = []
        selected_deadlines: dict[int, int] = {
            card: self.creation_rounds[index]
            for index, card in enumerate(route)
            if index and card in _CARD_TO_PILE
        }

        for index in range(1, len(route)):
            slot = self.observation.route[index]
            required = max(0, route[index] - route[index - 1] - 3)
            if slot.sprint_cards is not None:
                stack = tuple(sorted(slot.sprint_cards))
                if any(card not in available for card in stack):
                    return None
                if sum(sprint_value(card) for card in stack) < required:
                    return None
                stacks[index] = stack
                available -= set(stack)
                for card in stack:
                    if card in _CARD_TO_PILE:
                        selected_deadlines[card] = self.creation_rounds[index]
            else:
                unknown.append((index, slot.sprint_count, required))

        if not self._timing_feasible(selected_deadlines):
            return None
        # Earliest and tightest stacks first sharply reduce rejection while the
        # random candidate order retains diversity.
        unknown.sort(key=lambda item: (self.creation_rounds[item[0]], -item[2], -item[1]))

        def assign(position: int, remaining: set[int]) -> bool:
            if position == len(unknown):
                return True
            index, count, required = unknown[position]
            candidates = self._candidate_combinations(remaining, count)
            self.rng.shuffle(candidates)
            for combo in candidates:
                if sum(sprint_value(card) for card in combo) < required:
                    continue
                deadlines = dict(selected_deadlines)
                for previous_index in range(position):
                    stack_index = unknown[previous_index][0]
                    assert stacks[stack_index] is not None
                    for card in stacks[stack_index] or ():
                        if card in _CARD_TO_PILE:
                            deadlines[card] = self.creation_rounds[stack_index]
                for card in combo:
                    if card in _CARD_TO_PILE:
                        deadlines[card] = self.creation_rounds[index]
                if not self._timing_feasible(deadlines):
                    continue
                stacks[index] = tuple(sorted(combo))
                if assign(position + 1, remaining - set(combo)):
                    return True
                stacks[index] = None
            return False

        if not assign(0, available):
            return None
        return tuple(stack or () for stack in stacks)

    def _candidate_combinations(
        self,
        cards: set[int],
        count: int,
        *,
        limit: int = 512,
    ) -> list[tuple[int, ...]]:
        if count < 0 or count > len(cards):
            return []
        ordered = tuple(sorted(cards))
        if count == 0:
            return [()]
        total = math.comb(len(ordered), count)
        if total <= limit:
            return list(combinations(ordered, count))
        sampled: set[tuple[int, ...]] = set()
        attempts = 0
        while len(sampled) < limit and attempts < limit * 8:
            attempts += 1
            sampled.add(tuple(sorted(self.rng.sample(ordered, count))))
        return list(sampled)

    def _used_card_deadlines(
        self,
        route: tuple[int, ...],
        sprints: tuple[tuple[int, ...], ...],
    ) -> dict[int, int]:
        deadlines: dict[int, int] = {}
        for index in range(1, len(route)):
            deadline = self.creation_rounds[index]
            for card in (route[index], *sprints[index]):
                if card in _CARD_TO_PILE:
                    deadlines[card] = deadline
        return deadlines

    def _timing_feasible(self, deadlines: Mapping[int, int]) -> bool:
        for pile in range(3):
            draw_rounds = sorted(
                record.round_number
                for record in self.fugitive_records
                if record.pile == pile
            )
            card_deadlines = sorted(
                deadline
                for card, deadline in deadlines.items()
                if _CARD_TO_PILE.get(card) == pile
            )
            if len(card_deadlines) > len(draw_rounds):
                return False
            for position, deadline in enumerate(card_deadlines):
                if draw_rounds[position] > deadline:
                    return False
        return True

    def _assign_fugitive_draws(
        self,
        owned: set[int],
        deadlines: Mapping[int, int],
    ) -> tuple[int, ...] | None:
        assignments: list[int | None] = [None] * len(self.fugitive_records)
        for pile, original in enumerate(PILE_CARDS):
            record_indices = [
                index
                for index, record in enumerate(self.fugitive_records)
                if record.pile == pile
            ]
            record_indices.sort(
                key=lambda index: (self.fugitive_records[index].round_number, index)
            )
            cards = [card for card in owned if card in original]
            cards.sort(key=lambda card: (deadlines.get(card, _INFINITY), self.rng.random()))
            if len(cards) != len(record_indices):
                return None
            for card, record_index in zip(cards, record_indices, strict=True):
                if self.fugitive_records[record_index].round_number > deadlines.get(
                    card, _INFINITY
                ):
                    return None
                assignments[record_index] = card
        if any(card is None for card in assignments):
            return None
        return tuple(card for card in assignments if card is not None)

    def _guess_history_consistent(self, route: tuple[int, ...]) -> bool:
        revealed: set[int] = set()
        for record in self.observation.guess_history:
            if not 0 <= record.route_length <= self.route_length:
                return False
            prefix = set(route[1 : record.route_length + 1])
            hidden_at_time = prefix - revealed - {42}
            hit = set(record.numbers).issubset(hidden_at_time)
            if hit != record.success:
                return False
            if hit:
                revealed.update(record.numbers)

        for index, slot in enumerate(self.observation.route):
            if not index:
                continue
            value = route[index]
            if slot.hideout is not None and slot.hideout != value:
                return False
            if slot.revealed != (value in revealed):
                return False
        return True


__all__ = [
    "DEFAULT_MAX_INCREMENTAL_PLAY_PROPOSALS",
    "DEFAULT_PARTICLE_COUNT",
    "MarshalDrawOutcomeStatistics",
    "MarshalParticle",
    "MarshalParticleBelief",
    "ParticleBeliefSummary",
    "particle_matches_observation",
]
