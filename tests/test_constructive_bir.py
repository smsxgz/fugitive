from __future__ import annotations

import math
import time

import pytest

from fugitive.agents.constructive_bir import (
    ConstructiveBeliefConstructionError,
    ConstructiveBeliefInformedRandomMarshalAgent,
    ConstructiveMarshalBeliefBackend,
)
from fugitive.agents.bootstrap_bir import BeliefInformedRandomMarshalAgent
from fugitive.agents.marshal_belief_policy import BeliefInformedMarshalActionPolicy
from fugitive.agents.hierarchical_random import (
    HierarchicalRandomFugitiveAgent,
    HierarchicalRandomMarshalAgent,
)
from fugitive.engine import GameEngine, play_game
from fugitive.inference_diagnostics import ConstructiveInferenceWorkDiagnostics
from fugitive.inference.constructive_metadata import constructive_observation_hash
from fugitive.inference.constructive_sampler import ConstructiveWorldSampler
from fugitive.inference.worlds import (
    ConstructiveSampleBatch,
    ConstructiveSamplingReport,
)
from fugitive.model import FugitiveAction, Phase, Role
from fugitive.particle_belief import MarshalParticleBelief


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


def _review_high_sprint_observation():
    engine = GameEngine(seed=0)
    fugitive = HierarchicalRandomFugitiveAgent(100)
    marshal = HierarchicalRandomMarshalAgent(200)
    while True:
        phase = engine.phase
        role = (
            Role.FUGITIVE
            if phase
            in (Phase.FUGITIVE_OPENING, Phase.FUGITIVE_DRAW, Phase.FUGITIVE_ACTION)
            else Role.MARSHAL
        )
        observation = engine.observation(role)
        if phase is Phase.MARSHAL_DRAW and observation.round_number == 2:
            return observation
        if phase in (Phase.FUGITIVE_OPENING, Phase.FUGITIVE_ACTION):
            engine.apply_fugitive_action(
                fugitive.choose_fugitive_action(observation)
            )
        elif phase is Phase.FUGITIVE_DRAW:
            engine.draw(fugitive.choose_draw_pile(observation))
        elif phase is Phase.MARSHAL_DRAW:
            engine.draw(marshal.choose_draw_pile(observation))
        else:
            engine.apply_guess(marshal.choose_guess(observation))


class _RecordingSampler:
    def __init__(self, sprint_backend: str = "sequential") -> None:
        self.sprint_backend = sprint_backend
        self.delegate = ConstructiveWorldSampler(sprint_backend=sprint_backend)
        self.calls = 0
        self.last_batch = None

    @property
    def proposal_kernel_id(self):
        return self.delegate.proposal_kernel_id

    @property
    def target_id(self):
        return self.delegate.target_id

    def sample_batch(self, *args, **kwargs):
        self.calls += 1
        self.last_batch = self.delegate.sample_batch(*args, **kwargs)
        return self.last_batch


class _IncompleteSampler(_RecordingSampler):
    def sample_batch(self, observation, *, particle_count, **_kwargs):
        self.calls += 1
        self.last_batch = ConstructiveSampleBatch(
            worlds=(),
            report=ConstructiveSamplingReport(
                requested=particle_count,
                produced=0,
                proposals=1,
                dead_end_route_proposals=1,
                rejected_targets=0,
                search_nodes=0,
                degraded=True,
                importance_valid=True,
                termination_reason="max_proposals",
                exhausted_stage=None,
                unique_worlds=0,
            ),
            proposal_kernel_id=self.delegate.proposal_kernel_id,
            target_id=self.delegate.target_id,
            observation_hash=constructive_observation_hash(observation),
        )
        return self.last_batch


def _agent(seed: int, **kwargs) -> ConstructiveBeliefInformedRandomMarshalAgent:
    return ConstructiveBeliefInformedRandomMarshalAgent(
        seed,
        particle_count=24,
        max_guess_candidates=16,
        **kwargs,
    )


def test_constructive_bir_is_deterministic_and_caches_observations() -> None:
    _engine, observation = _opening_observation()
    first_sampler = _RecordingSampler()
    first = _agent(91, _sampler=first_sampler)
    second = _agent(91)

    first_belief = first.belief(observation)
    repeated = first.belief(observation)

    assert repeated is first_belief
    assert first_sampler.calls == 1
    assert first_belief.summary() == second.belief(observation).summary()
    assert first.draw_pile_distribution(observation) == second.draw_pile_distribution(
        observation
    )
    diagnostics = first.inference_diagnostics()
    assert diagnostics is not None
    assert isinstance(diagnostics.work, ConstructiveInferenceWorkDiagnostics)
    assert diagnostics.work.sampling.produced == 24


def test_constructive_bir_defaults_to_sequential_sprint_proposals() -> None:
    agent = ConstructiveBeliefInformedRandomMarshalAgent(
        1, particle_count=8
    )

    assert agent.sprint_backend == "sequential"
    assert isinstance(agent.action_policy, BeliefInformedMarshalActionPolicy)
    assert isinstance(agent.belief_backend, ConstructiveMarshalBeliefBackend)
    assert agent.belief_backend.sampler.sprint_backend == "sequential"
    assert not isinstance(agent, BeliefInformedRandomMarshalAgent)


def test_constructive_bir_depends_only_on_the_marshal_observation() -> None:
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

    assert first.belief(first_observation).summary() == second.belief(
        second_observation
    ).summary()
    assert first.draw_pile_distribution(
        first_observation
    ) == second.draw_pile_distribution(second_observation)


def test_constructive_bir_preserves_normalized_importance_weights() -> None:
    _engine, observation = _opening_observation()
    sampler = _RecordingSampler()
    agent = _agent(251, _sampler=sampler)

    belief = agent.belief(observation)
    assert sampler.last_batch is not None
    expected = sampler.last_batch.normalized_weights
    actual = tuple(particle.weight for particle in belief.particles)

    assert actual == pytest.approx(expected)
    assert len(actual) == agent.particle_count
    assert all(math.isfinite(weight) and weight >= 0.0 for weight in actual)
    assert sum(actual) == pytest.approx(1.0)
    assert not belief.sampling_exhausted


def test_constructive_bir_rejects_budget_partial_without_fresh_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _engine, observation = _opening_observation()
    sampler = _IncompleteSampler()
    agent = ConstructiveBeliefInformedRandomMarshalAgent(
        9,
        particle_count=8,
        _sampler=sampler,
    )

    def forbidden_fresh_fallback(*_args, **_kwargs):
        raise AssertionError("BIR-2 must not use a second fresh sampler")

    monkeypatch.setattr(
        MarshalParticleBelief,
        "from_observation",
        forbidden_fresh_fallback,
    )
    with pytest.raises(
        ConstructiveBeliefConstructionError,
        match=r"requested=8, produced=0.*termination_reason='max_proposals'",
    ) as caught:
        agent.belief(observation)

    assert caught.value.report.degraded
    assert caught.value.report.importance_valid
    assert sampler.calls == 1
    assert agent.inference_diagnostics() is None
    assert agent.belief_backend.cache_size == 0

def test_constructive_bir_small_particle_full_game_finishes() -> None:
    fugitive = HierarchicalRandomFugitiveAgent(100)
    marshal = ConstructiveBeliefInformedRandomMarshalAgent(
        200,
        particle_count=8,
        max_guess_candidates=16,
    )

    result = play_game(fugitive, marshal, seed=0)

    assert result.winner is not None
    assert result.reason
    diagnostics = marshal.inference_diagnostics()
    assert diagnostics is not None
    assert isinstance(diagnostics.work, ConstructiveInferenceWorkDiagnostics)
    assert diagnostics.work.sampling.produced == 8
    assert diagnostics.work.weighting_id == marshal.weighting_id

@pytest.mark.parametrize("particle_count", (128, 2_000))
def test_registered_particle_profiles_complete_high_sprint_fixture(
    particle_count: int,
) -> None:
    observation = _review_high_sprint_observation()
    assert tuple(slot.sprint_count for slot in observation.route[1:]) == (0, 3, 3)
    agent = ConstructiveBeliefInformedRandomMarshalAgent(
        999,
        particle_count=particle_count,
    )

    started = time.perf_counter()
    belief = agent.belief(observation)
    elapsed = time.perf_counter() - started

    diagnostics = agent.inference_diagnostics()
    assert diagnostics is not None
    assert isinstance(diagnostics.work, ConstructiveInferenceWorkDiagnostics)
    report = diagnostics.work.sampling
    assert len(belief.particles) == particle_count
    assert report.produced == particle_count
    assert report.search_nodes < 2_000_000
    assert elapsed < 15.0
