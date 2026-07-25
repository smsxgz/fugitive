"""Focused golden checks for particle order and end-to-end BIR replay."""

from __future__ import annotations

import hashlib
import json

import pytest

from fugitive.agents.bir_fugitive import BeliefInformedRandomFugitiveAgent
from fugitive.agents.bootstrap_bir import BeliefInformedRandomMarshalAgent
from fugitive.driver import DrawTrace, FugitiveTrace, GuessTrace
from fugitive.engine import GameEngine
from fugitive.experiment import ExperimentStatus, replay_manifest, run_experiment
from fugitive.model import FugitiveAction, Phase, Role
from fugitive.particle_belief import MarshalParticleBelief
from fugitive.reproducibility import SeedBundle


def _sha256(rows: object) -> str:
    encoded = json.dumps(
        rows,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _particle_population_sha256(belief: MarshalParticleBelief) -> str:
    rows = [
        (particle.world_key, particle.weight.hex())
        for particle in belief.particles
    ]
    return _sha256(rows)


def test_bootstrap_particle_fresh_and_incremental_populations_golden() -> None:
    engine = GameEngine(seed=1)
    engine.apply_fugitive_action(FugitiveAction(1))
    engine.apply_fugitive_action(FugitiveAction(6, (2,)))
    opening = engine.observation(Role.MARSHAL)

    fresh = MarshalParticleBelief.from_observation(
        opening,
        particle_count=32,
        seed=19,
    )

    assert (
        len(fresh.particles),
        fresh.unique_particle_count,
        fresh.sampling_attempts,
        fresh.sampling_accepted,
        fresh.resampling_count,
    ) == (32, 32, 32, 32, 0)
    assert not fresh.sampling_exhausted
    # BIR-1 and BIR-2 use the same canonical constructive proposal stream.
    # This digest protects that stream's ordered, duplicate-preserving output.
    assert _particle_population_sha256(fresh) == (
        "d91da35ab2ce3a4113a61b5db98ddcd03"
        "ece258ee619e158c05dfa66874a11c1"
    )

    # The observed Deck 1 draw removes worlds in which card 8 was already
    # owned by the Fugitive.  No resampling is needed, so the digest protects
    # both the filtering result and the surviving constructive order.
    assert engine.draw(0) == 8
    advanced = fresh.advance_to(
        engine.observation(Role.MARSHAL),
        particle_count=32,
        seed=29,
    )

    assert (
        len(advanced.particles),
        advanced.unique_particle_count,
        advanced.sampling_attempts,
        advanced.sampling_accepted,
        advanced.resampling_count,
    ) == (27, 27, 32, 32, 0)
    assert advanced.pre_resample_effective_sample_size == pytest.approx(27.0)
    assert _particle_population_sha256(advanced) == (
        "aa9a3296c32eb4da2fa684fda9b0cb54"
        "3ea80e5782afdc892c2a7a554641e6d5"
    )


def test_bir_fixed_prefix_trace_and_replay_golden() -> None:
    run = run_experiment(
        BeliefInformedRandomFugitiveAgent(11),
        BeliefInformedRandomMarshalAgent(
            13,
            particle_count=32,
            max_guess_candidates=16,
        ),
        seeds=SeedBundle(master=5, deck=73, fugitive=11, marshal=13),
        max_decisions=12,
        validate_invariants=True,
    )

    assert run.status is ExperimentStatus.TRUNCATED
    assert run.game_result is None
    assert run.manifest.final_state_sha256 == (
        "d36cb1f3d99ee4e065bb94e69e67f57"
        "b4c377a45296413f24bd1eba8062960f1"
    )
    assert run.manifest.trace == (
        FugitiveTrace(1, 1, Phase.FUGITIVE_OPENING, Role.FUGITIVE, 3, ()),
        FugitiveTrace(2, 1, Phase.FUGITIVE_OPENING, Role.FUGITIVE, 8, (22,)),
        DrawTrace(3, 1, Phase.MARSHAL_DRAW, Role.MARSHAL, 0, 11),
        DrawTrace(4, 1, Phase.MARSHAL_DRAW, Role.MARSHAL, 0, 14),
        GuessTrace(5, 1, Phase.MARSHAL_GUESS, Role.MARSHAL, (2, 4), False),
        DrawTrace(6, 2, Phase.FUGITIVE_DRAW, Role.FUGITIVE, 1, 23),
        FugitiveTrace(7, 2, Phase.FUGITIVE_ACTION, Role.FUGITIVE, 12, (1,)),
        DrawTrace(8, 2, Phase.MARSHAL_DRAW, Role.MARSHAL, 1, 26),
        GuessTrace(9, 2, Phase.MARSHAL_GUESS, Role.MARSHAL, (2, 3, 4), False),
        DrawTrace(10, 3, Phase.FUGITIVE_DRAW, Role.FUGITIVE, 1, 18),
        FugitiveTrace(11, 3, Phase.FUGITIVE_ACTION, Role.FUGITIVE, None, ()),
        DrawTrace(12, 3, Phase.MARSHAL_DRAW, Role.MARSHAL, 0, 13),
    )

    verification = replay_manifest(run.manifest)
    assert verification.status is ExperimentStatus.TRUNCATED
    assert verification.decision_count == 12
    assert verification.final_state_sha256 == run.manifest.final_state_sha256
