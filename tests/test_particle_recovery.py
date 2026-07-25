from __future__ import annotations

import math

import pytest

from fugitive.agents.bootstrap_bir import (
    BeliefInformedRandomMarshalAgent,
    BootstrapParticleBeliefBackend,
    _stable_seed,
)
from fugitive.agents.hierarchical_random import (
    HierarchicalRandomFugitiveAgent,
)
from fugitive.agents.unweighted_constructive_bir import (
    UnweightedConstructiveBeliefInformedRandomMarshalAgent,
)
from fugitive.driver import step_agent
from fugitive.engine import GameEngine
from fugitive.inference_diagnostics import (
    BootstrapInferenceWorkDiagnostics,
    InferenceDiagnosticsSnapshot,
)
from fugitive.model import FugitiveAction, Role
from fugitive.particle_belief import (
    IncompatibleObservationError,
    MarshalParticleBelief,
)


def _opening_engine(seed: int = 13) -> GameEngine:
    engine = GameEngine(seed=seed)
    engine.apply_fugitive_action(FugitiveAction(1))
    engine.apply_fugitive_action(FugitiveAction(2))
    return engine


def _diagnostics(
    agent: BeliefInformedRandomMarshalAgent,
) -> tuple[InferenceDiagnosticsSnapshot, BootstrapInferenceWorkDiagnostics]:
    snapshot = agent.inference_diagnostics()
    assert snapshot is not None
    assert isinstance(snapshot.work, BootstrapInferenceWorkDiagnostics)
    return snapshot, snapshot.work


def test_bir_marshal_resets_after_legal_reveal_extinguishes_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A high-information reveal can eliminate every incremental particle."""

    original = MarshalParticleBelief.from_observation.__func__
    fresh_sample_calls = 0

    def counted_from_observation(
        cls: type[MarshalParticleBelief],
        *args: object,
        **kwargs: object,
    ) -> MarshalParticleBelief:
        nonlocal fresh_sample_calls
        fresh_sample_calls += 1
        return original(cls, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        MarshalParticleBelief,
        "from_observation",
        classmethod(counted_from_observation),
    )

    engine = GameEngine(seed=0)
    agent = BeliefInformedRandomMarshalAgent(2000, particle_count=2000)
    engine.apply_fugitive_action(FugitiveAction(4, (3,)))
    engine.apply_fugitive_action(FugitiveAction(14, (1, 2, 10, 24)))
    agent.belief(engine.observation(Role.MARSHAL))

    engine.draw(1)
    agent.belief(engine.observation(Role.MARSHAL))
    engine.draw(2)
    agent.belief(engine.observation(Role.MARSHAL))
    assert not engine.apply_guess((2,))

    engine.draw(0)
    engine.apply_fugitive_action(FugitiveAction(18, (8,)))
    agent.belief(engine.observation(Role.MARSHAL))
    engine.draw(0)
    before_reveal = agent.belief(engine.observation(Role.MARSHAL))
    assert not before_reveal.is_empty
    _snapshot, work = _diagnostics(agent)
    assert work.support_extinction_reset_count == 0

    assert engine.apply_guess((14,))
    engine.draw(1)
    engine.apply_fugitive_action(FugitiveAction(None))
    recovered = agent.belief(engine.observation(Role.MARSHAL))
    diagnostics, work = _diagnostics(agent)

    assert len(recovered.particles) == 2000
    assert fresh_sample_calls == 2
    assert diagnostics.operation == "support_extinction_reset"
    assert work.fresh_build_count == 2
    assert work.support_extinction_reset_count == 1
    assert all(
        particle.route_hideouts[2] == 14
        and particle.route_sprints[2] == (1, 2, 10, 24)
        for particle in recovered.particles
    )


def test_nonempty_concentrated_incremental_population_is_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _opening_engine()
    agent = BeliefInformedRandomMarshalAgent(5, particle_count=64)
    initial = agent.belief(engine.observation(Role.MARSHAL))

    engine.draw(2)
    target = engine.observation(Role.MARSHAL)

    def concentrated_advance(
        self: MarshalParticleBelief,
        *_args: object,
        **_kwargs: object,
    ) -> MarshalParticleBelief:
        return MarshalParticleBelief(
            target,
            (self.particles[0],) * 64,
            sampling_attempts=self.sampling_attempts,
            sampling_accepted=self.sampling_accepted,
            sampling_exhausted=False,
            resampling_count=self.resampling_count,
        )

    monkeypatch.setattr(
        MarshalParticleBelief,
        "advance_to",
        concentrated_advance,
    )
    accepted = agent.belief(target)
    diagnostics, work = _diagnostics(agent)

    assert accepted.unique_particle_count == 1
    assert accepted.world_effective_sample_size == pytest.approx(1.0)
    assert diagnostics.operation == "incremental_update"
    assert work.fresh_build_count == 1
    assert work.support_extinction_reset_count == 0
    assert diagnostics.quality.particle_entries == 64
    assert diagnostics.quality.unique_worlds == 1


def test_empty_incremental_population_causes_one_fresh_reset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _opening_engine()
    agent = BeliefInformedRandomMarshalAgent(5, particle_count=32)
    initial = agent.belief(engine.observation(Role.MARSHAL))

    engine.draw(2)
    target = engine.observation(Role.MARSHAL)

    def empty_advance(
        self: MarshalParticleBelief,
        *_args: object,
        **_kwargs: object,
    ) -> MarshalParticleBelief:
        return MarshalParticleBelief(
            target,
            (),
            sampling_attempts=self.sampling_attempts,
            sampling_accepted=self.sampling_accepted,
            sampling_exhausted=True,
            resampling_count=self.resampling_count,
        )

    monkeypatch.setattr(MarshalParticleBelief, "advance_to", empty_advance)
    recovered = agent.belief(target)
    observation_local = (
        UnweightedConstructiveBeliefInformedRandomMarshalAgent(
            5,
            particle_count=32,
        ).belief(target)
    )
    diagnostics, work = _diagnostics(agent)

    assert recovered is not initial
    assert len(recovered.particles) == 32
    assert diagnostics.operation == "support_extinction_reset"
    assert work.fresh_build_count == 2
    assert work.support_extinction_reset_count == 1
    assert recovered.particles == observation_local.particles


def test_fresh_construction_is_one_complete_constructive_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _opening_engine()
    observation = engine.observation(Role.MARSHAL)
    source = MarshalParticleBelief.from_observation(
        observation,
        particle_count=1,
        seed=11,
    )
    calls: list[dict[str, object]] = []

    def concentrated_complete_batch(
        cls: type[MarshalParticleBelief],
        _observation: object,
        **kwargs: object,
    ) -> MarshalParticleBelief:
        calls.append(kwargs)
        return cls(
            observation,
            (source.particles[0],) * 64,
            sampling_attempts=64,
            sampling_accepted=64,
            sampling_exhausted=False,
        )

    monkeypatch.setattr(
        MarshalParticleBelief,
        "from_observation",
        classmethod(concentrated_complete_batch),
    )
    agent = BeliefInformedRandomMarshalAgent(5, particle_count=64)

    belief = agent.belief(observation)
    _diagnostic, work = _diagnostics(agent)

    assert len(calls) == 1
    assert calls[0]["particle_count"] == 64
    assert belief.unique_particle_count == 1
    assert work.fresh_build_count == 1
    assert work.support_extinction_reset_count == 0


def test_constructive_fresh_completes_on_legal_manhunt_long_tail_state() -> None:
    """Regression for a state where the removed legacy DFS stalled."""

    engine = GameEngine(seed=21202)
    fugitive = HierarchicalRandomFugitiveAgent(22202)
    marshal = BeliefInformedRandomMarshalAgent(
        23202,
        particle_count=4,
        max_guess_candidates=6,
    )
    for decision in range(1, 64):
        step_agent(engine, fugitive, marshal, decision=decision)

    observation = engine.observation(Role.MARSHAL)
    assert observation.phase.value == "manhunt"
    seed = _stable_seed(observation, marshal.belief_backend.belief_salt)

    first = MarshalParticleBelief.from_observation(
        observation,
        particle_count=1,
        seed=seed,
    )
    second = MarshalParticleBelief.from_observation(
        observation,
        particle_count=1,
        seed=seed,
    )

    assert len(first.particles) == 1
    assert first.sampling_accepted == 1
    assert first.sampling_attempts >= first.sampling_accepted
    assert not first.sampling_exhausted
    assert second.particles == first.particles
    assert second.sampling_attempts == first.sampling_attempts


def test_incompatible_incremental_observation_fails_without_mutating_backend() -> None:
    first = _opening_engine()
    second = _opening_engine()
    agent = BeliefInformedRandomMarshalAgent(8, particle_count=32)

    agent.belief(first.observation(Role.MARSHAL))
    first.draw(0)
    latest = agent.belief(first.observation(Role.MARSHAL))
    before = agent.inference_diagnostics()

    second.draw(1)
    incompatible = second.observation(Role.MARSHAL)
    with pytest.raises(IncompatibleObservationError):
        agent.belief(incompatible)

    backend = agent.belief_backend
    assert isinstance(backend, BootstrapParticleBeliefBackend)
    assert backend.latest_belief is latest
    assert agent.inference_diagnostics() == before


def test_diagnostics_report_world_and_hidden_hypothesis_quality_separately() -> None:
    engine = _opening_engine()
    agent = BeliefInformedRandomMarshalAgent(5, particle_count=64)
    belief = agent.belief(engine.observation(Role.MARSHAL))
    diagnostics, work = _diagnostics(agent)

    hidden_masses: dict[tuple[int, ...], float] = {}
    for particle in belief.particles:
        hypothesis = tuple(sorted(belief.current_hidden_hideouts(particle)))
        hidden_masses[hypothesis] = (
            hidden_masses.get(hypothesis, 0.0) + particle.weight
        )
    hidden_ess = 1.0 / math.fsum(
        mass * mass for mass in hidden_masses.values()
    )

    quality = diagnostics.quality
    assert quality.requested_particles == 64
    assert quality.particle_entries == len(belief.particles)
    assert quality.unique_worlds == belief.unique_particle_count
    assert quality.entry_ess == (
        belief.effective_sample_size
    )
    assert work.minimum_pre_resample_entry_ess == (
        belief.pre_resample_effective_sample_size
    )
    assert quality.world_ess == (
        belief.world_effective_sample_size
    )
    assert quality.max_world_mass == belief.max_world_mass
    assert quality.unique_hidden_routes == len(hidden_masses)
    assert quality.hidden_route_ess == pytest.approx(
        hidden_ess
    )
    assert quality.max_hidden_route_mass == pytest.approx(
        max(hidden_masses.values())
    )
    assert work.origin_fresh_sampling_accepted == 64
    assert work.origin_fresh_sampling_proposals >= 64
