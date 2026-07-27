from __future__ import annotations

import json

import pytest

from fugitive.agents.marshal.bir.bootstrap import BeliefInformedRandomMarshalAgent
from fugitive.agents.marshal.bir.snis import (
    ConstructiveBeliefInformedRandomMarshalAgent,
)
from fugitive.agents.marshal.bir.mcmc import (
    BIR3_INITIAL_WEIGHTING_ID,
    MCMCBeliefInformedRandomMarshalAgent,
)
from fugitive.agents.marshal.inference.path_belief import PathBelief
from fugitive.game.engine import GameEngine
from fugitive.agents.marshal.inference.diagnostics import (
    BootstrapInferenceWorkDiagnostics,
    ConstructiveInferenceWorkDiagnostics,
    IndependentMHInferenceWorkDiagnostics,
)
from fugitive.game.model import FugitiveAction, Role


def _opening_observation():
    engine = GameEngine(seed=7)
    engine.apply_fugitive_action(FugitiveAction(1))
    engine.apply_fugitive_action(FugitiveAction(2))
    return engine.observation(Role.MARSHAL)


@pytest.mark.parametrize(
    ("agent", "work_type", "kind"),
    (
        (
            BeliefInformedRandomMarshalAgent(11, particle_count=16),
            BootstrapInferenceWorkDiagnostics,
            "bootstrap",
        ),
        (
            ConstructiveBeliefInformedRandomMarshalAgent(
                11,
                particle_count=16,
                max_guess_candidates=16,
            ),
            ConstructiveInferenceWorkDiagnostics,
            "constructive",
        ),
        (
            MCMCBeliefInformedRandomMarshalAgent(
                11,
                particle_count=16,
                max_guess_candidates=16,
                mh_steps_per_chain=1,
            ),
            IndependentMHInferenceWorkDiagnostics,
            "sir_independent_mh",
        ),
    ),
)
def test_bir_backends_publish_comparable_read_only_snapshots(
    agent,
    work_type,
    kind: str,
) -> None:
    observation = _opening_observation()
    assert agent.inference_diagnostics() is None

    belief = agent.belief(observation)
    rng_state = agent.rng.getstate()
    first = agent.inference_diagnostics()
    repeated = agent.inference_diagnostics()

    assert first is not None
    assert repeated == first
    assert agent.rng.getstate() == rng_state
    assert first.operation == "fresh_build"
    assert first.backend_id == agent.belief_backend.backend_id
    assert first.algorithm_id == agent.algorithm_id
    assert isinstance(first.work, work_type)
    assert first.quality.requested_particles == 16
    assert first.quality.particle_entries == len(belief.particles)
    assert first.quality.unique_worlds == belief.unique_particle_count
    assert first.quality.entry_ess == belief.effective_sample_size
    assert first.quality.world_ess == belief.world_effective_sample_size
    assert first.quality.unique_hidden_routes == (
        belief.unique_hidden_hypothesis_count
    )
    assert first.quality.hidden_route_ess == (
        belief.hidden_hypothesis_effective_sample_size
    )
    assert first.quality.hard_route_support_count == (
        PathBelief.from_observation(observation).total_paths
    )
    encoded = first.to_dict()
    assert encoded["work"]["kind"] == kind
    json.dumps(encoded, allow_nan=False)

    same_belief = agent.belief(observation)
    cached = agent.inference_diagnostics()
    assert same_belief is belief
    assert cached is not None
    assert cached.operation == "cache_hit"


def test_constructive_and_mh_work_keep_distinct_weight_semantics() -> None:
    observation = _opening_observation()
    constructive = ConstructiveBeliefInformedRandomMarshalAgent(
        21,
        particle_count=16,
        max_guess_candidates=16,
    )
    constructive.belief(observation)
    constructive_snapshot = constructive.inference_diagnostics()
    assert constructive_snapshot is not None
    assert isinstance(
        constructive_snapshot.work,
        ConstructiveInferenceWorkDiagnostics,
    )
    assert constructive_snapshot.work.weighting_id == (
        "self-normalized-importance-v1"
    )
    sampling = constructive_snapshot.work.sampling
    assert 1.0 <= sampling.proposal_importance_ess <= 16.0
    assert 0.0 < sampling.max_normalized_importance_weight <= 1.0

    mcmc = MCMCBeliefInformedRandomMarshalAgent(
        21,
        particle_count=16,
        max_guess_candidates=16,
        mh_steps_per_chain=1,
    )
    mcmc.belief(observation)
    mcmc_snapshot = mcmc.inference_diagnostics()
    assert mcmc_snapshot is not None
    assert isinstance(
        mcmc_snapshot.work,
        IndependentMHInferenceWorkDiagnostics,
    )
    work = mcmc_snapshot.work
    assert work.initial_weighting_id == BIR3_INITIAL_WEIGHTING_ID
    assert work.to_dict()["initial_weighting_id"] == BIR3_INITIAL_WEIGHTING_ID
    assert 1.0 <= work.initial_sampling.proposal_importance_ess <= 16.0
    assert work.mh_proposal_sampling.proposal_importance_ess is None
    assert work.mh_proposal_sampling.max_normalized_importance_weight is None
    assert work.mh_proposals == 16
    assert 0 <= work.changed <= work.accepted <= work.mh_proposals
