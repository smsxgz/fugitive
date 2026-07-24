from __future__ import annotations

import pytest

from fugitive.agents.belief_informed_random import (
    BeliefInformedRandomMarshalAgent,
)
from fugitive.engine import GameEngine
from fugitive.model import FugitiveAction, Role
from fugitive.particle_belief import MarshalParticleBelief


def test_bir_marshal_recovers_after_legal_reveal_collapses_particles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression for a reveal whose Sprint identities kill every old particle."""

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
    assert agent.belief_diagnostics.fresh_resamples == 0

    assert engine.apply_guess((14,))
    engine.draw(1)
    engine.apply_fugitive_action(FugitiveAction(None))
    recovered = agent.belief(engine.observation(Role.MARSHAL))

    assert not recovered.is_empty
    assert len(recovered.particles) == 2000
    assert recovered.unique_particle_count > agent.min_unique_particles
    assert fresh_sample_calls == 2  # Initial sample plus the recovery.
    assert agent.belief_diagnostics.fresh_resamples == 1
    assert agent.belief_diagnostics.last_incremental_particle_count == 0
    assert agent.belief_diagnostics.last_incremental_unique_particle_count == 0
    assert agent.belief_diagnostics.last_recovery_reason == "empty"
    assert all(
        particle.route_hideouts[2] == 14
        and particle.route_sprints[2] == (1, 2, 10, 24)
        for particle in recovered.particles
    )


def test_bir_marshal_recovers_from_low_particle_diversity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = GameEngine(seed=13)
    engine.apply_fugitive_action(FugitiveAction(1))
    engine.apply_fugitive_action(FugitiveAction(2))
    agent = BeliefInformedRandomMarshalAgent(
        5,
        particle_count=64,
        min_unique_particles=2,
    )
    initial = agent.belief(engine.observation(Role.MARSHAL))
    assert not initial.is_empty

    engine.draw(2)
    target = engine.observation(Role.MARSHAL)

    def sparse_advance(
        self: MarshalParticleBelief,
        *_args: object,
        **_kwargs: object,
    ) -> MarshalParticleBelief:
        return MarshalParticleBelief(
            target,
            (self.particles[0],) * 64,
            sampling_attempts=self.sampling_attempts,
            sampling_exhausted=True,
        )

    monkeypatch.setattr(MarshalParticleBelief, "advance_to", sparse_advance)
    recovered = agent.belief(target)

    assert not recovered.is_empty
    assert recovered.unique_particle_count >= agent.min_unique_particles
    assert agent.belief_diagnostics.fresh_resamples == 1
    assert agent.belief_diagnostics.last_incremental_unique_particle_count == 1
    assert agent.belief_diagnostics.last_recovery_reason == "low_unique"
