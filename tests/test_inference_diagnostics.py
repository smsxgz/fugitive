from __future__ import annotations

import json

import pytest

import fugitive.experiment as experiment_module
from fugitive.agents.bootstrap_bir import BeliefInformedRandomMarshalAgent
from fugitive.agents.constructive_bir import (
    ConstructiveBeliefInformedRandomMarshalAgent,
)
from fugitive.agents.mcmc_bir import (
    BIR3_INITIAL_WEIGHTING_ID,
    MCMCBeliefInformedRandomMarshalAgent,
)
from fugitive.belief import PathBelief
from fugitive.engine import GameEngine
from fugitive.experiment import (
    ExperimentStatus,
    SeedBundle,
    replay_manifest,
    run_experiment,
    run_registered_experiment,
)
from fugitive.inference_diagnostics import (
    BootstrapInferenceWorkDiagnostics,
    ConstructiveInferenceWorkDiagnostics,
    IndependentMHInferenceWorkDiagnostics,
)
from fugitive.model import FugitiveAction, Phase, Role, Winner


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


def test_bootstrap_operation_distinguishes_incremental_work_from_cache_hits() -> None:
    engine = GameEngine(seed=7)
    engine.apply_fugitive_action(FugitiveAction(1))
    engine.apply_fugitive_action(FugitiveAction(2))
    agent = BeliefInformedRandomMarshalAgent(33, particle_count=16)

    agent.belief(engine.observation(Role.MARSHAL))
    engine.draw(engine.observation(Role.MARSHAL).legal_draw_piles[0])
    agent.belief(engine.observation(Role.MARSHAL))
    incremental = agent.inference_diagnostics()

    assert incremental is not None
    assert incremental.operation == "incremental_update"
    assert isinstance(incremental.work, BootstrapInferenceWorkDiagnostics)
    assert incremental.work.inference_count == 2
    assert incremental.work.incremental_update_count == 1
    assert incremental.work.cache_hit_count == 0

    agent.belief(engine.observation(Role.MARSHAL))
    cached = agent.inference_diagnostics()
    assert cached is not None
    assert cached.operation == "cache_hit"
    assert isinstance(cached.work, BootstrapInferenceWorkDiagnostics)
    assert cached.work.inference_count == 2
    assert cached.work.cache_hit_count == 1


def _diagnostic_experiment():
    return run_registered_experiment(
        master_seed=91,
        fugitive_name="hierarchical-random",
        marshal_name="constructive-belief-informed-random",
        marshal_parameters={
            "particle_count": 8,
            "max_guess_candidates": 16,
        },
        max_decisions=5,
    )


def test_experiment_diagnostics_are_deterministic_and_outside_replay() -> None:
    first = _diagnostic_experiment()
    second = _diagnostic_experiment()

    assert first.inference_events
    assert first.inference_events == second.inference_events
    assert [event.decision for event in first.inference_events] == [3, 4, 5]
    assert all(event.role is Role.MARSHAL for event in first.inference_events)
    assert all("round_number" in event.to_dict() for event in first.inference_events)
    assert all("round" not in event.to_dict() for event in first.inference_events)
    json.dumps(
        [event.to_dict() for event in first.inference_events],
        allow_nan=False,
    )
    assert first.manifest == second.manifest
    assert first.manifest.to_json(indent=None) == second.manifest.to_json(indent=None)
    assert replay_manifest(first.manifest).final_state_sha256 == (
        first.manifest.final_state_sha256
    )


def test_collecting_diagnostics_does_not_change_manifest_or_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instrumented = _diagnostic_experiment()
    monkeypatch.setattr(
        experiment_module,
        "read_inference_diagnostics",
        lambda _agent: None,
    )
    without_side_channel = _diagnostic_experiment()

    assert without_side_channel.inference_events == ()
    assert instrumented.manifest.to_json(indent=None) == (
        without_side_channel.manifest.to_json(indent=None)
    )
    assert instrumented.manifest.final_state_sha256 == (
        without_side_channel.manifest.final_state_sha256
    )
    assert replay_manifest(instrumented.manifest) == replay_manifest(
        without_side_channel.manifest
    )


def test_agents_without_diagnostics_leave_the_side_channel_empty() -> None:
    run = run_registered_experiment(
        master_seed=12,
        fugitive_name="hierarchical-random",
        marshal_name="hierarchical-random",
        max_decisions=5,
    )

    assert run.inference_events == ()


class _OpeningOneTwoFugitive:
    def choose_draw_pile(self, observation):
        return observation.legal_draw_piles[0]

    def choose_fugitive_action(self, observation):
        if observation.phase is Phase.FUGITIVE_OPENING:
            return FugitiveAction(len(observation.route))
        return FugitiveAction(None)


class _FailingDiagnosticsMarshal:
    def choose_draw_pile(self, observation):
        return observation.legal_draw_piles[0]

    def choose_guess(self, _observation):
        return (1, 2)

    def inference_diagnostics(self):
        raise RuntimeError("diagnostics unavailable")


def test_diagnostics_failure_after_commit_does_not_fail_experiment() -> None:
    run = run_experiment(
        _OpeningOneTwoFugitive(),
        _FailingDiagnosticsMarshal(),
        seeds=SeedBundle(master=123, deck=7, fugitive=11, marshal=13),
        validate_invariants=True,
    )

    assert run.status is ExperimentStatus.COMPLETED
    assert run.game_result is not None
    assert run.game_result.winner is Winner.MARSHAL
    assert run.manifest.decision_count == 5
    assert run.inference_events == ()
    assert [failure.decision for failure in run.inference_diagnostic_failures] == [
        3,
        4,
        5,
    ]
    failure = run.inference_diagnostic_failures[0].to_dict()
    assert failure == {
        "decision": 3,
        "round_number": 1,
        "phase": Phase.MARSHAL_DRAW.value,
        "role": Role.MARSHAL.value,
        "error": {
            "type": "RuntimeError",
            "message": "diagnostics unavailable",
        },
    }
    assert "inference_diagnostic_failures" not in run.manifest.to_dict()
    assert replay_manifest(run.manifest).final_state_sha256 == (
        run.manifest.final_state_sha256
    )
