from __future__ import annotations

import math
import time

import pytest

from fugitive.agents.constructive_bir import (
    ConstructiveBeliefConstructionError,
    ConstructiveBeliefInformedRandomMarshalAgent,
)
from fugitive.agents.hierarchical_random import (
    HierarchicalRandomFugitiveAgent,
    HierarchicalRandomMarshalAgent,
)
from fugitive.constructive_belief import (
    ConstructiveSampleBatch,
    ConstructiveSamplingReport,
    ConstructiveWorldSampler,
)
from fugitive.engine import GameEngine, play_game
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

    def sample_batch(self, *args, **kwargs):
        self.calls += 1
        self.last_batch = self.delegate.sample_batch(*args, **kwargs)
        return self.last_batch


class _IncompleteSampler(_RecordingSampler):
    def sample_batch(self, _observation, *, particle_count, **_kwargs):
        self.calls += 1
        self.last_batch = ConstructiveSampleBatch(
            worlds=(),
            report=ConstructiveSamplingReport(
                requested=particle_count,
                produced=0,
                proposals=1,
                rejected_routes=1,
                rejected_targets=0,
                search_nodes=0,
                degraded=True,
                importance_valid=True,
                termination_reason="max_proposals",
                exhausted_stage=None,
                unique_worlds=0,
            ),
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
    assert first.last_sampling_report is not None
    assert first.last_sampling_report.produced == 24


def test_constructive_bir_defaults_to_sequential_sprint_proposals() -> None:
    agent = ConstructiveBeliefInformedRandomMarshalAgent(
        1, particle_count=8
    )

    assert agent.sprint_backend == "sequential"
    assert agent._constructive_sampler.sprint_backend == "sequential"


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


def test_constructive_bir_rejects_budget_partial_without_legacy_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _engine, observation = _opening_observation()
    sampler = _IncompleteSampler()
    agent = ConstructiveBeliefInformedRandomMarshalAgent(
        9,
        particle_count=8,
        _sampler=sampler,
    )

    def forbidden_legacy_sampler(*_args, **_kwargs):
        raise AssertionError("BIR-2 must not call the legacy rejection sampler")

    monkeypatch.setattr(
        MarshalParticleBelief,
        "from_observation",
        forbidden_legacy_sampler,
    )
    with pytest.raises(
        ConstructiveBeliefConstructionError,
        match=r"requested=8, produced=0.*termination_reason='max_proposals'",
    ) as caught:
        agent.belief(observation)

    assert caught.value.report.degraded
    assert caught.value.report.importance_valid
    assert sampler.calls == 1
    assert agent.last_sampling_report is None
    assert observation not in agent._belief_cache

@pytest.mark.parametrize("seed", (0, 1, 2))
def test_constructive_bir_full_game_fixtures_finish(seed: int) -> None:
    fugitive = HierarchicalRandomFugitiveAgent(100 + seed)
    marshal = ConstructiveBeliefInformedRandomMarshalAgent(
        200 + seed,
        particle_count=8,
        max_guess_candidates=16,
    )

    result = play_game(fugitive, marshal, seed=seed)

    assert result.winner is not None
    assert result.reason
    assert marshal.last_sampling_report is not None
    assert marshal.last_sampling_report.produced == 8
    assert not marshal.last_sampling_report.degraded

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

    report = agent.last_sampling_report
    assert report is not None
    assert len(belief.particles) == particle_count
    assert report.produced == particle_count
    assert not report.degraded
    assert report.importance_valid
    assert report.search_nodes < 2_000_000
    assert elapsed < 15.0
