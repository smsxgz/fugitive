from __future__ import annotations

import math

import pytest

from fugitive.agents.hierarchical_random import HierarchicalRandomFugitiveAgent
from fugitive.agents.mcmc_bir import (
    DEFAULT_MCMC_PARTICLE_COUNT,
    DEFAULT_MH_STEPS_PER_CHAIN,
    MCMCBeliefConstructionError,
    MCMCBeliefInformedRandomMarshalAgent,
)
from fugitive.constructive_belief import (
    ConstructiveSampleBatch,
    ConstructiveSamplingReport,
    ConstructiveWorldSampler,
)
from fugitive.engine import GameEngine, play_game
from fugitive.experiment import (
    ExperimentStatus,
    replay_manifest,
    run_registered_experiment,
)
from fugitive.model import FugitiveAction, Role
from fugitive.rejuvenation import IncompleteProposalBatchError


def _opening_observation(
    first: FugitiveAction = FugitiveAction(1),
    second: FugitiveAction = FugitiveAction(2),
    *,
    seed: int = 7,
):
    engine = GameEngine(seed=seed)
    engine.apply_fugitive_action(first)
    engine.apply_fugitive_action(second)
    return engine, engine.observation(Role.MARSHAL)


class _RecordingSampler:
    def __init__(self) -> None:
        self.delegate = ConstructiveWorldSampler(sprint_backend="sequential")
        self.sprint_backend = self.delegate.sprint_backend
        self.target = self.delegate.target
        self.requested: list[int] = []

    def sample_batch(self, observation, *, particle_count, **kwargs):
        self.requested.append(particle_count)
        return self.delegate.sample_batch(
            observation,
            particle_count=particle_count,
            **kwargs,
        )


def _incomplete_batch(particle_count: int) -> ConstructiveSampleBatch:
    return ConstructiveSampleBatch(
        worlds=(),
        report=ConstructiveSamplingReport(
            requested=particle_count,
            produced=0,
            proposals=1,
            rejected_routes=1,
            rejected_targets=0,
            search_nodes=1,
            degraded=True,
            importance_valid=True,
            termination_reason="max_proposals",
            exhausted_stage=None,
            unique_worlds=0,
        ),
    )


class _FailFirstBatchSampler(_RecordingSampler):
    def sample_batch(self, _observation, *, particle_count, **_kwargs):
        self.requested.append(particle_count)
        return _incomplete_batch(particle_count)


class _FailSecondBatchSampler(_RecordingSampler):
    def sample_batch(self, observation, *, particle_count, **kwargs):
        self.requested.append(particle_count)
        if len(self.requested) == 1:
            return self.delegate.sample_batch(
                observation,
                particle_count=particle_count,
                **kwargs,
            )
        return _incomplete_batch(particle_count)


def _agent(seed: int, **kwargs) -> MCMCBeliefInformedRandomMarshalAgent:
    return MCMCBeliefInformedRandomMarshalAgent(
        seed,
        particle_count=24,
        max_guess_candidates=16,
        **kwargs,
    )


def test_mcmc_bir_defaults_are_balanced() -> None:
    agent = MCMCBeliefInformedRandomMarshalAgent(3)

    assert agent.particle_count == DEFAULT_MCMC_PARTICLE_COUNT == 1_000
    assert agent.mh_steps_per_chain == DEFAULT_MH_STEPS_PER_CHAIN == 1


@pytest.mark.parametrize("steps", (True, 0, -1, 1.5, "1"))
def test_mcmc_bir_requires_a_positive_integer_step_count(steps: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        MCMCBeliefInformedRandomMarshalAgent(
            3,
            particle_count=8,
            mh_steps_per_chain=steps,  # type: ignore[arg-type]
        )


def test_mcmc_bir_is_deterministic_cached_and_batches_all_mh_proposals() -> None:
    _engine, observation = _opening_observation()
    sampler = _RecordingSampler()
    first = _agent(91, _sampler=sampler)
    second = _agent(91)

    belief = first.belief(observation)
    repeated = first.belief(observation)
    comparison = second.belief(observation)

    assert repeated is belief
    assert sampler.requested == [24, 24]
    assert belief.particles == comparison.particles
    assert first.draw_pile_distribution(
        observation
    ) == second.draw_pile_distribution(observation)


def test_mcmc_bir_depends_only_on_the_marshal_observation() -> None:
    first_engine, first_observation = _opening_observation(
        FugitiveAction(1), FugitiveAction(2), seed=23
    )
    second_engine, second_observation = _opening_observation(
        FugitiveAction(2), FugitiveAction(3), seed=23
    )
    assert first_observation == second_observation
    assert first_engine.observation(Role.FUGITIVE) != second_engine.observation(
        Role.FUGITIVE
    )

    first = _agent(147)
    second = _agent(147)

    assert first.belief(first_observation).particles == second.belief(
        second_observation
    ).particles


def test_mcmc_bir_publishes_equal_weights_and_auditable_diagnostics() -> None:
    _engine, observation = _opening_observation()
    agent = _agent(251)

    belief = agent.belief(observation)
    weights = tuple(particle.weight for particle in belief.particles)
    diagnostics = agent.last_rejuvenation_diagnostics
    initial_report = agent.last_initial_sampling_report
    proposal_report = agent.last_mh_proposal_report

    assert len(weights) == 24
    assert weights == pytest.approx((1.0 / 24,) * 24)
    assert math.isclose(sum(weights), 1.0)
    assert diagnostics is not None
    assert diagnostics.chains == 24
    assert diagnostics.steps_per_chain == 1
    assert diagnostics.proposals == 24
    assert 0 <= diagnostics.changed <= diagnostics.accepted <= 24
    assert 1 <= diagnostics.final_unique_worlds <= 24
    assert initial_report is not None and initial_report.produced == 24
    assert proposal_report is not None and proposal_report.produced == 24


def test_mcmc_bir_fails_closed_when_initial_batch_is_incomplete() -> None:
    _engine, observation = _opening_observation()
    agent = MCMCBeliefInformedRandomMarshalAgent(
        9,
        particle_count=8,
        _sampler=_FailFirstBatchSampler(),
    )

    with pytest.raises(MCMCBeliefConstructionError, match="BIR-3"):
        agent.belief(observation)

    assert observation not in agent._belief_cache
    assert agent.last_initial_sampling_report is None
    assert agent.last_rejuvenation_diagnostics is None


def test_mcmc_bir_fails_closed_when_mh_proposal_batch_is_incomplete() -> None:
    _engine, observation = _opening_observation()
    sampler = _FailSecondBatchSampler()
    agent = _agent(12, _sampler=sampler)

    with pytest.raises(IncompleteProposalBatchError) as caught:
        agent.belief(observation)

    assert sampler.requested == [24, 24]
    assert caught.value.report.termination_reason == "max_proposals"
    assert observation not in agent._belief_cache
    assert agent.last_initial_sampling_report is None
    assert agent.last_mh_proposal_report is None
    assert agent.last_rejuvenation_diagnostics is None


def test_small_mcmc_bir_marshal_finishes_a_complete_game() -> None:
    fugitive = HierarchicalRandomFugitiveAgent(100)
    marshal = MCMCBeliefInformedRandomMarshalAgent(
        200,
        particle_count=8,
        max_guess_candidates=16,
    )

    result = play_game(fugitive, marshal, seed=0)

    assert result.winner is not None
    assert result.reason
    assert marshal.last_rejuvenation_diagnostics is not None


def test_registered_small_mcmc_bir_game_replays_exactly() -> None:
    run = run_registered_experiment(
        master_seed=41,
        fugitive_name="hierarchical-random",
        marshal_name="mcmc-belief-informed-random",
        marshal_parameters={
            "particle_count": 8,
            "max_guess_candidates": 16,
            "mh_steps_per_chain": 1,
        },
    )

    assert run.status is ExperimentStatus.COMPLETED
    assert run.game_result is not None
    assert run.manifest.max_decisions is None
    assert run.manifest.marshal_agent.name == "mcmc-belief-informed-random"
    verification = replay_manifest(run.manifest)
    assert verification.status is ExperimentStatus.COMPLETED
    assert verification.game_result == run.game_result
