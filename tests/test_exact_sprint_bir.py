from __future__ import annotations

import pytest

from fugitive.agents.exact_sprint_bir import (
    EXACT_SPRINT_ALGORITHM_ID,
    ExactSprintBeliefInformedRandomMarshalAgent,
)
from fugitive.agents.bootstrap_bir import BeliefInformedRandomMarshalAgent
from fugitive.agents.marshal_belief_policy import BeliefInformedMarshalActionPolicy
from fugitive.agents.constructive_bir import ConstructiveMarshalBeliefBackend
from fugitive.agents.hierarchical_random import HierarchicalRandomFugitiveAgent
from fugitive.engine import GameEngine, play_game
from fugitive.inference_diagnostics import ConstructiveInferenceWorkDiagnostics
from fugitive.inference.constructive_sampler import ConstructiveWorldSampler
from fugitive.model import FugitiveAction, Role


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


class _RecordingExactSampler:
    sprint_backend = "exact"

    def __init__(self) -> None:
        self.delegate = ConstructiveWorldSampler(sprint_backend="exact")
        self.calls = 0

    @property
    def proposal_kernel_id(self):
        return self.delegate.proposal_kernel_id

    @property
    def target_id(self):
        return self.delegate.target_id

    def sample_batch(self, *args, **kwargs):
        self.calls += 1
        return self.delegate.sample_batch(*args, **kwargs)


def _agent(seed: int, **kwargs) -> ExactSprintBeliefInformedRandomMarshalAgent:
    return ExactSprintBeliefInformedRandomMarshalAgent(
        seed,
        particle_count=4,
        max_guess_candidates=8,
        **kwargs,
    )


def test_exact_sprint_agent_has_a_fixed_algorithm_identity_and_backend() -> None:
    agent = _agent(3)

    assert agent.algorithm_id == EXACT_SPRINT_ALGORITHM_ID
    assert agent.sprint_backend == "exact"
    assert isinstance(agent.action_policy, BeliefInformedMarshalActionPolicy)
    assert isinstance(agent.belief_backend, ConstructiveMarshalBeliefBackend)
    assert agent.belief_backend.sampler.sprint_backend == "exact"
    assert not isinstance(agent, BeliefInformedRandomMarshalAgent)
    with pytest.raises(ValueError, match="exact Sprint backend"):
        _agent(
            3,
            _sampler=ConstructiveWorldSampler(sprint_backend="sequential"),
        )


def test_exact_sprint_agent_is_deterministic_and_caches_observations() -> None:
    _engine, observation = _opening_observation()
    sampler = _RecordingExactSampler()
    first = _agent(91, _sampler=sampler)  # type: ignore[arg-type]
    second = _agent(91)

    belief = first.belief(observation)
    repeated = first.belief(observation)

    assert repeated is belief
    assert sampler.calls == 1
    assert belief.particles == second.belief(observation).particles
    assert first.draw_pile_distribution(
        observation
    ) == second.draw_pile_distribution(observation)
    diagnostics = first.inference_diagnostics()
    assert diagnostics is not None
    assert isinstance(diagnostics.work, ConstructiveInferenceWorkDiagnostics)
    assert diagnostics.work.sampling.produced == 4
    assert diagnostics.backend_id == (
        "constructive-snis-exact-sprint-v1"
    )


def test_exact_sprint_agent_depends_only_on_the_marshal_observation() -> None:
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


def test_small_exact_sprint_agent_finishes_a_complete_game() -> None:
    result = play_game(
        HierarchicalRandomFugitiveAgent(100),
        ExactSprintBeliefInformedRandomMarshalAgent(
            200,
            particle_count=1,
            max_guess_candidates=8,
        ),
        seed=0,
    )

    assert result.winner is not None
    assert result.reason
