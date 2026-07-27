from __future__ import annotations

import math

import pytest

from fugitive.agents.marshal.bir.snis import (
    ConstructiveBeliefConstructionError,
    ConstructiveBeliefInformedRandomMarshalAgent,
)
from fugitive.game.engine import GameEngine
from fugitive.agents.marshal.inference.constructive.metadata import constructive_observation_hash
from fugitive.agents.marshal.inference.constructive.sampler import ConstructiveWorldSampler
from fugitive.agents.marshal.inference.constructive.worlds import (
    ConstructiveSampleBatch,
    ConstructiveSamplingReport,
)
from fugitive.game.model import FugitiveAction, Role
from fugitive.agents.marshal.inference.particle_belief import MarshalParticleBelief


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
