from __future__ import annotations

from dataclasses import replace
import math

import pytest

from fugitive.agents.marshal.bir.snis import (
    ConstructiveBeliefInformedRandomMarshalAgent,
)
from fugitive.agents.marshal.bir.bootstrap import BeliefInformedRandomMarshalAgent
from fugitive.agents.marshal.bir.unweighted import (
    UnweightedConstructiveBeliefInformedRandomMarshalAgent,
)
from fugitive.game.engine import GameEngine
from fugitive.agents.marshal.inference.constructive.sampler import ConstructiveWorldSampler
from fugitive.agents.marshal.inference.particles.constructive_fresh import (
    ConstructiveWeightingStrategy,
    build_constructive_belief,
)
from fugitive.game.model import FugitiveAction, Role


def _opening_observation():
    engine = GameEngine(seed=7)
    engine.apply_fugitive_action(FugitiveAction(1))
    engine.apply_fugitive_action(FugitiveAction(2))
    return engine.observation(Role.MARSHAL)


def test_bir1_fresh_and_bir2u_share_the_bir2s_proposal_batch() -> None:
    observation = _opening_observation()
    bootstrap = BeliefInformedRandomMarshalAgent(
        91,
        particle_count=32,
        max_guess_candidates=16,
    )
    corrected = ConstructiveBeliefInformedRandomMarshalAgent(
        91,
        particle_count=32,
        max_guess_candidates=16,
    )
    unweighted = UnweightedConstructiveBeliefInformedRandomMarshalAgent(
        91,
        particle_count=32,
        max_guess_candidates=16,
    )

    bootstrap_belief = bootstrap.belief(observation)
    corrected_belief = corrected.belief(observation)
    unweighted_belief = unweighted.belief(observation)

    ordered_worlds = tuple(
        particle.world_key for particle in bootstrap_belief.particles
    )
    assert ordered_worlds == tuple(
        particle.world_key for particle in corrected_belief.particles
    )
    assert ordered_worlds == tuple(
        particle.world_key for particle in unweighted_belief.particles
    )
    bootstrap_weights = tuple(
        particle.weight for particle in bootstrap_belief.particles
    )
    corrected_weights = tuple(
        particle.weight for particle in corrected_belief.particles
    )
    unweighted_weights = tuple(
        particle.weight for particle in unweighted_belief.particles
    )
    assert bootstrap_weights == pytest.approx((1.0 / 32,) * 32)
    assert bootstrap_weights == pytest.approx(unweighted_weights)
    assert any(
        not math.isclose(weight, 1.0 / 32)
        for weight in corrected_weights
    )

def test_unweighted_builder_preserves_duplicate_proposal_multiplicity() -> None:
    observation = _opening_observation()
    sampler = ConstructiveWorldSampler(sprint_backend="sequential")
    source = sampler.sample_batch(
        observation,
        particle_count=16,
        seed=314,
    )
    first = source.worlds[0]
    second = next(
        world for world in source.worlds if world.world_key != first.world_key
    )
    batch = replace(
        source,
        worlds=(first, first, second),
        report=replace(
            source.report,
            requested=3,
            produced=3,
            unique_worlds=2,
        ),
    )

    belief = build_constructive_belief(
        observation,
        batch,
        expected_particles=3,
        weighting=ConstructiveWeightingStrategy.UNIFORM_ACCEPTED_PROPOSALS,
    ).belief
    masses: dict[tuple[object, ...], float] = {}
    for particle in belief.particles:
        masses[particle.world_key] = (
            masses.get(particle.world_key, 0.0) + particle.weight
        )

    assert masses[first.world_key] == pytest.approx(2.0 / 3.0)
    assert masses[second.world_key] == pytest.approx(1.0 / 3.0)
