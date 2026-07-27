from __future__ import annotations

from fugitive.agents.marshal.bir.exact_sprint import (
    ExactSprintBeliefInformedRandomMarshalAgent,
)
from fugitive.game.engine import GameEngine
from fugitive.game.model import FugitiveAction, Role


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


def _agent(seed: int, **kwargs) -> ExactSprintBeliefInformedRandomMarshalAgent:
    return ExactSprintBeliefInformedRandomMarshalAgent(
        seed,
        particle_count=4,
        max_guess_candidates=8,
        **kwargs,
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
