"""Incremental bootstrap filtering over newly public game events."""

from __future__ import annotations

from dataclasses import replace
from itertools import combinations
import math
import random
from typing import Sequence

from fugitive.game.model import (
    DrawRecord,
    FugitiveAction,
    GuessRecord,
    Observation,
    PlayView,
    Role,
)
from fugitive.game.rules import is_legal_fugitive_action
from ..world_validation import is_complete_world_consistent

from .state import (
    BIR1_INCREMENTAL_PLAY_PROPOSAL_LIMIT,
    BIR1_RESAMPLE_ESS_FRACTION,
    DEFAULT_PARTICLE_COUNT,
    IncompatibleObservationError,
    MarshalParticle,
    MarshalParticleBelief,
    _aggregate_particles,
    _effective_sample_size,
    _validate_particle_weights,
)


BIR1_INCREMENTAL_EXACT_COMBINATION_THRESHOLD = 4_096


def advance_belief(
    belief: MarshalParticleBelief,
    new_observation: Observation,
    *,
    particle_count: int | None = None,
    seed: int | None = None,
    rng: random.Random | None = None,
    max_play_proposals_per_particle: int = (
        BIR1_INCREMENTAL_PLAY_PROPOSAL_LIMIT
    ),
    resample_ess_fraction: float = BIR1_RESAMPLE_ESS_FRACTION,
) -> MarshalParticleBelief:
    """Advance this belief through newly visible public events.

    The old draw and guess histories must be literal prefixes of the new
    histories.  Play history and route entries may refine formerly hidden
    identities after a successful guess, but their stable public fields
    must remain unchanged.

    Hidden draws are propagated over every possible card in each particle.
    Hidden placements enumerate small action sets, then retain a bounded
    uniform reservoir; large action sets use deterministic-RNG proposals.
    This bounded placement proposal is the only additional approximation
    beyond the particle population itself. Duplicate worlds are combined,
    and systematic resampling is used only when the population exceeds its
    bound or pre-resample ESS crosses the configured threshold.
    """

    if new_observation.role is not Role.MARSHAL:
        raise ValueError("incremental particle filtering requires a Marshal observation")
    if seed is not None and rng is not None:
        raise ValueError("pass either seed or rng, not both")
    if particle_count is None:
        particle_count = len(belief.particles) or DEFAULT_PARTICLE_COUNT
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
    if (
        isinstance(resample_ess_fraction, bool)
        or not isinstance(resample_ess_fraction, (int, float))
        or not math.isfinite(resample_ess_fraction)
        or not 0.0 < resample_ess_fraction <= 1.0
    ):
        raise ValueError("resample_ess_fraction must be in (0, 1]")

    _validate_observation_extension(belief.observation, new_observation)
    owned_rng = rng if rng is not None else random.Random(seed)
    if not belief.particles:
        return MarshalParticleBelief(
            new_observation,
            (),
            sampling_attempts=belief.sampling_attempts,
            sampling_exhausted=True,
            sampling_accepted=belief.sampling_accepted,
            pre_resample_effective_sample_size=0.0,
            resampling_count=belief.resampling_count,
        )

    particles = _aggregate_particles(belief.particles)
    minimum_pre_resample_ess = math.inf
    resampling_count = belief.resampling_count
    revealed_numbers = {
        slot.hideout
        for index, slot in enumerate(belief.observation.route)
        if index and slot.revealed and slot.hideout is not None
    }
    events = _new_public_events(belief.observation, new_observation)
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

        if not particles:
            return MarshalParticleBelief(
                new_observation,
                (),
                sampling_attempts=belief.sampling_attempts,
                sampling_exhausted=True,
                sampling_accepted=belief.sampling_accepted,
                pre_resample_effective_sample_size=0.0,
                resampling_count=resampling_count,
            )
        particles = _aggregate_particles(particles)
        pre_resample_ess = _effective_sample_size(particles)
        minimum_pre_resample_ess = min(
            minimum_pre_resample_ess, pre_resample_ess
        )
        event_changed_information = not (
            kind == "play"
            and isinstance(record, PlayView)
            and record.passed
        )
        if (
            len(particles) > particle_count
            or (
                event_changed_information
                and pre_resample_ess
                <= resample_ess_fraction * particle_count
            )
        ):
            particles = _aggregate_particles(
                _systematic_resample(
                    particles,
                    particle_count,
                    rng=owned_rng,
                )
            )
            resampling_count += 1

    particles = tuple(
        particle
        for particle in particles
        if is_complete_world_consistent(particle, new_observation)
    )
    particles = _aggregate_particles(particles)
    final_pre_resample_ess = _effective_sample_size(particles)
    minimum_pre_resample_ess = min(
        minimum_pre_resample_ess, final_pre_resample_ess
    )
    if minimum_pre_resample_ess == math.inf:
        minimum_pre_resample_ess = final_pre_resample_ess
    return MarshalParticleBelief(
        new_observation,
        particles,
        sampling_attempts=belief.sampling_attempts,
        sampling_exhausted=belief.sampling_exhausted or not particles,
        sampling_accepted=belief.sampling_accepted,
        pre_resample_effective_sample_size=minimum_pre_resample_ess,
        resampling_count=resampling_count,
    )


def _validate_observation_extension(
    old: Observation,
    new: Observation,
) -> None:
    if old.role is not Role.MARSHAL or new.role is not Role.MARSHAL:
        raise IncompatibleObservationError(
            "both observations must belong to the Marshal"
        )
    if old.round_number > new.round_number:
        raise IncompatibleObservationError("the new observation precedes the old round")
    if len(old.draw_history) > len(new.draw_history) or tuple(
        new.draw_history[: len(old.draw_history)]
    ) != old.draw_history:
        raise IncompatibleObservationError(
            "old draw history is not a prefix of the new history"
        )
    if len(old.guess_history) > len(new.guess_history) or tuple(
        new.guess_history[: len(old.guess_history)]
    ) != old.guess_history:
        raise IncompatibleObservationError(
            "old guess history is not a prefix of the new history"
        )
    if len(old.play_history) > len(new.play_history):
        raise IncompatibleObservationError(
            "old play history is longer than the new history"
        )
    for before, after in zip(old.play_history, new.play_history, strict=False):
        if not _play_view_refines(before, after):
            raise IncompatibleObservationError(
                "new play history does not refine its old prefix"
            )

    if len(old.route) > len(new.route):
        raise IncompatibleObservationError(
            "the new route is shorter than the old route"
        )
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
            raise IncompatibleObservationError(
                "new route does not refine the old visible route"
            )
    if not set(old.hand).issubset(new.hand):
        raise IncompatibleObservationError(
            "cards disappeared from the Marshal hand"
        )
    if any(after > before for before, after in zip(old.pile_sizes, new.pile_sizes)):
        raise IncompatibleObservationError(
            "a draw pile grew between observations"
        )


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
    if total_combinations <= BIR1_INCREMENTAL_EXACT_COMBINATION_THRESHOLD:
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
    _validate_particle_weights(particles)
    weights = [particle.weight for particle in particles]
    total = math.fsum(weights)
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("particle weights must have positive finite mass")
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


__all__ = [
    "BIR1_INCREMENTAL_EXACT_COMBINATION_THRESHOLD",
    "advance_belief",
]
