from __future__ import annotations

from dataclasses import replace
import math

import pytest

from fugitive.agents.constructive_bir import (
    BIR2_PROPOSAL_KERNEL_ID,
    BIR2_REFERENCE_TARGET_ID,
    BIR2_WEIGHTING_ID,
    ConstructiveBeliefInformedRandomMarshalAgent,
)
from fugitive.agents.bootstrap_bir import (
    BIR1_WEIGHTING_ID,
    BeliefInformedRandomMarshalAgent,
)
from fugitive.agents.hierarchical_random import (
    HierarchicalRandomFugitiveAgent,
)
from fugitive.agents.registry import MARSHAL_AGENT_REGISTRY
from fugitive.agents.unweighted_constructive_bir import (
    BIR2U_ALGORITHM_ID,
    BIR2U_WEIGHTING_ID,
    UnweightedConstructiveBeliefConstructionError,
    UnweightedConstructiveBeliefInformedRandomMarshalAgent,
)
from fugitive.engine import GameEngine, play_game
from fugitive.inference.constructive_sampler import ConstructiveWorldSampler
from fugitive.particle_inference.constructive_fresh import (
    ConstructiveWeightingStrategy,
    build_constructive_belief,
)
from fugitive.inference_diagnostics import (
    ConstructiveInferenceWorkDiagnostics,
)
from fugitive.model import FugitiveAction, Role


def _opening_observation():
    engine = GameEngine(seed=7)
    engine.apply_fugitive_action(FugitiveAction(1))
    engine.apply_fugitive_action(FugitiveAction(2))
    return engine.observation(Role.MARSHAL)


class _ImportanceInvalidSampler(ConstructiveWorldSampler):
    def sample_batch(self, *args, **kwargs):
        batch = super().sample_batch(*args, **kwargs)
        return replace(
            batch,
            report=replace(batch.report, importance_valid=False),
        )


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

    corrected_diagnostics = corrected.inference_diagnostics()
    unweighted_diagnostics = unweighted.inference_diagnostics()
    assert corrected_diagnostics is not None
    assert unweighted_diagnostics is not None
    assert isinstance(
        corrected_diagnostics.work,
        ConstructiveInferenceWorkDiagnostics,
    )
    assert isinstance(
        unweighted_diagnostics.work,
        ConstructiveInferenceWorkDiagnostics,
    )
    assert corrected_diagnostics.work.weighting_id == BIR2_WEIGHTING_ID
    assert unweighted_diagnostics.work.weighting_id == BIR2U_WEIGHTING_ID
    assert (
        corrected_diagnostics.work.sampling
        == unweighted_diagnostics.work.sampling
    )
    assert corrected_diagnostics.quality.entry_ess < 32.0
    assert unweighted_diagnostics.quality.entry_ess == pytest.approx(32.0)
    assert bootstrap.weighting_id == BIR1_WEIGHTING_ID


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


def test_constructive_builder_rejects_a_batch_from_another_observation() -> None:
    engine = GameEngine(seed=7)
    engine.apply_fugitive_action(FugitiveAction(1))
    engine.apply_fugitive_action(FugitiveAction(2))
    first_observation = engine.observation(Role.MARSHAL)
    batch = ConstructiveWorldSampler(sprint_backend="sequential").sample_batch(
        first_observation,
        particle_count=4,
        seed=19,
    )

    engine.draw(0)
    second_observation = engine.observation(Role.MARSHAL)
    with pytest.raises(ValueError, match="belongs to a different observation"):
        build_constructive_belief(
            second_observation,
            batch,
            expected_particles=4,
            weighting=(
                ConstructiveWeightingStrategy.UNIFORM_ACCEPTED_PROPOSALS
            ),
        )


def test_bir2u_metadata_profile_and_draw_distribution_are_explicit() -> None:
    corrected_registration = MARSHAL_AGENT_REGISTRY[
        "constructive-belief-informed-random"
    ]
    unweighted_registration = MARSHAL_AGENT_REGISTRY[
        "unweighted-constructive-belief-informed-random"
    ]
    corrected = corrected_registration.build(17, profile="interactive")
    unweighted = unweighted_registration.build(17, profile="interactive")

    assert corrected.spec.parameters["weighting_id"] == BIR2_WEIGHTING_ID
    assert unweighted.spec.parameters["algorithm_id"] == BIR2U_ALGORITHM_ID
    assert unweighted.spec.parameters["weighting_id"] == BIR2U_WEIGHTING_ID
    assert unweighted.spec.parameters["belief_seed_domain"] == (
        "fugitive.marshal.sequential-constructive-belief.v1"
    )
    assert unweighted.spec.parameters["proposal_kernel_id"] == (
        BIR2_PROPOSAL_KERNEL_ID
    )
    assert unweighted.spec.parameters["reference_target_id"] == (
        BIR2_REFERENCE_TARGET_ID
    )
    for parameter in (
        "particle_count",
        "max_guess_candidates",
        "epsilon",
        "temperature",
        "terminal_bonus_scale",
        "manhunt_epsilon",
        "manhunt_alpha",
    ):
        assert unweighted.spec.parameters[parameter] == (
            corrected.spec.parameters[parameter]
        )

    observation = _opening_observation()
    distribution = unweighted.agent.draw_pile_distribution(observation)
    assert set(distribution) == set(observation.legal_draw_piles)
    assert all(math.isfinite(probability) and probability >= 0.0
               for probability in distribution.values())
    assert math.fsum(distribution.values()) == pytest.approx(1.0)


def test_bir2u_rejects_a_batch_without_auditable_importance_weights() -> None:
    agent = UnweightedConstructiveBeliefInformedRandomMarshalAgent(
        91,
        particle_count=8,
        max_guess_candidates=16,
        _sampler=_ImportanceInvalidSampler(sprint_backend="sequential"),
    )

    with pytest.raises(
        UnweightedConstructiveBeliefConstructionError,
        match="importance_valid=False",
    ):
        agent.belief(_opening_observation())

    assert agent.belief_backend.cache_size == 0


def test_bir2u_small_particle_full_game_finishes() -> None:
    result = play_game(
        HierarchicalRandomFugitiveAgent(100),
        UnweightedConstructiveBeliefInformedRandomMarshalAgent(
            200,
            particle_count=8,
            max_guess_candidates=16,
        ),
        seed=0,
    )

    assert result.winner is not None
    assert result.reason
