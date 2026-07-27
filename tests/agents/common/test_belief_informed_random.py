from __future__ import annotations

from dataclasses import replace

import pytest

from fugitive.agents.common.baseline_utils import epsilon_softmax
from fugitive.agents.fugitive.bir import BeliefInformedRandomFugitiveAgent
from fugitive.agents.marshal.bir.bootstrap import BeliefInformedRandomMarshalAgent
from fugitive.game.engine import GameEngine, play_game
from fugitive.game.model import FugitiveAction, Phase, Role, Winner
from fugitive.game.rules import is_legal_fugitive_action, is_legal_guess


def assert_distribution(distribution: dict[object, float]) -> None:
    assert distribution
    assert sum(distribution.values()) == pytest.approx(1.0)
    assert all(probability > 0.0 for probability in distribution.values())


def finish_opening(
    engine: GameEngine,
    first: FugitiveAction = FugitiveAction(1),
    second: FugitiveAction = FugitiveAction(2),
) -> None:
    engine.apply_fugitive_action(first)
    engine.apply_fugitive_action(second)


def test_epsilon_softmax_is_normalized_and_preserves_exploration() -> None:
    distribution = epsilon_softmax(
        {"low": -10.0, "middle": 0.0, "high": 10.0},
        epsilon=0.15,
        temperature=0.7,
    )

    assert_distribution(distribution)
    assert distribution["high"] > distribution["middle"] > distribution["low"]
    assert all(probability >= 0.05 - 1e-12 for probability in distribution.values())

    tied = epsilon_softmax({"a": 4.0, "b": 4.0, "c": 4.0})
    assert tied == pytest.approx({"a": 1 / 3, "b": 1 / 3, "c": 1 / 3})


def test_fugitive_opening_distribution_conditions_a_legal_second_play() -> None:
    engine = GameEngine(seed=19)
    agent = BeliefInformedRandomFugitiveAgent(7)
    first_observation = engine.observation(Role.FUGITIVE)
    plans = agent.opening_plan_distribution(first_observation)

    assert_distribution(plans)
    assert all(42 not in plan.first.sprint_cards for plan in plans)
    assert all(42 not in plan.second.sprint_cards for plan in plans)

    first_distribution = agent.fugitive_action_distribution(first_observation)
    assert_distribution(first_distribution)
    assert all(42 not in action.sprint_cards for action in first_distribution)
    first = agent.choose_fugitive_action(first_observation)
    engine.apply_fugitive_action(first)

    second_observation = engine.observation(Role.FUGITIVE)
    second_distribution = agent.fugitive_action_distribution(second_observation)
    assert_distribution(second_distribution)
    assert all(42 not in action.sprint_cards for action in second_distribution)
    assert all(
        is_legal_fugitive_action(
            action,
            second_observation.hand,
            first.hideout,  # type: ignore[arg-type]
            allow_pass=False,
        )
        for action in second_distribution
    )

    second = agent.choose_fugitive_action(second_observation)
    engine.apply_fugitive_action(second)
    assert engine.phase is Phase.MARSHAL_DRAW
    assert len(engine.observation(Role.FUGITIVE).route) == 3


def test_second_opening_has_fallback_for_any_engine_legal_first_play() -> None:
    engine = GameEngine(seed=0)
    first = FugitiveAction(1, (2, 4))
    engine.apply_fugitive_action(first)
    observation = engine.observation(Role.FUGITIVE)

    distribution = BeliefInformedRandomFugitiveAgent(
        7
    ).fugitive_action_distribution(observation)

    assert_distribution(distribution)
    assert all(
        is_legal_fugitive_action(
            action,
            observation.hand,
            1,
            allow_pass=False,
        )
        for action in distribution
    )


def test_fugitive_draw_distribution_ignores_hidden_marshal_cards() -> None:
    # These seeds deal the same ordered private cards to the Fugitive, but the
    # untouched deck order and the Marshal's later private cards differ.
    engines = [GameEngine(seed=566), GameEngine(seed=1080)]
    for engine in engines:
        finish_opening(engine)
        engine.draw(2)
        engine.draw(2)
        assert not engine.apply_guess((41,))

    first_observation = engines[0].observation(Role.FUGITIVE)
    second_observation = engines[1].observation(Role.FUGITIVE)
    assert first_observation == second_observation
    assert engines[0].observation(Role.MARSHAL).hand != engines[1].observation(
        Role.MARSHAL
    ).hand

    first = BeliefInformedRandomFugitiveAgent(31).draw_pile_distribution(
        first_observation
    )
    second = BeliefInformedRandomFugitiveAgent(31).draw_pile_distribution(
        second_observation
    )
    assert_distribution(first)
    assert first == second
    assert set(first) == set(first_observation.legal_draw_piles)


def test_marshal_uses_joint_success_and_manhunt_is_singleton_only() -> None:
    engine = GameEngine(seed=5)
    finish_opening(engine, FugitiveAction(1), FugitiveAction(3))
    engine.draw(2)
    engine.draw(2)
    assert not engine.apply_guess((1, 2))
    engine.draw(2)
    engine.apply_fugitive_action(FugitiveAction(None))
    engine.draw(2)
    observation = engine.observation(Role.MARSHAL)

    agent = BeliefInformedRandomMarshalAgent(
        53, particle_count=128, max_guess_candidates=64
    )
    belief = agent.belief(observation)
    distribution = agent.guess_distribution(observation)

    assert not belief.is_empty
    assert belief.probability_hidden(1) > 0.0
    assert belief.probability_hidden(2) > 0.0
    assert belief.joint_success((1, 2)) == 0.0
    assert (1, 2) not in distribution
    assert_distribution(distribution)
    assert all(belief.joint_success(guess) > 0.0 for guess in distribution)
    assert all(is_legal_guess(guess) for guess in distribution)
    assert any(len(guess) > 1 for guess in distribution)

    manhunt = replace(
        observation,
        phase=Phase.MANHUNT,
        legal_draw_piles=(),
    )
    manhunt_distribution = agent.guess_distribution(manhunt)
    assert_distribution(manhunt_distribution)
    assert all(len(guess) == 1 for guess in manhunt_distribution)
    assert all(is_legal_guess(guess, manhunt=True) for guess in manhunt_distribution)


def test_indistinguishable_marshal_worlds_have_identical_guess_distribution() -> None:
    first_engine = GameEngine(seed=29)
    second_engine = GameEngine(seed=29)
    finish_opening(first_engine, FugitiveAction(1), FugitiveAction(2))
    finish_opening(second_engine, FugitiveAction(2), FugitiveAction(3))
    for engine in (first_engine, second_engine):
        engine.draw(2)
        engine.draw(2)

    first_observation = first_engine.observation(Role.MARSHAL)
    second_observation = second_engine.observation(Role.MARSHAL)
    assert first_observation == second_observation
    assert first_engine.observation(Role.FUGITIVE) != second_engine.observation(
        Role.FUGITIVE
    )

    first = BeliefInformedRandomMarshalAgent(
        67, particle_count=64, max_guess_candidates=48
    ).guess_distribution(first_observation)
    second = BeliefInformedRandomMarshalAgent(
        67, particle_count=64, max_guess_candidates=48
    ).guess_distribution(second_observation)
    assert_distribution(first)
    assert first == second


def test_small_particle_bir_agents_finish_a_complete_game() -> None:
    result = play_game(
        BeliefInformedRandomFugitiveAgent(8),
        BeliefInformedRandomMarshalAgent(
            9,
            particle_count=32,
            max_guess_candidates=32,
        ),
        seed=7,
    )

    assert result.winner in (Winner.FUGITIVE, Winner.MARSHAL)
    assert result.reason
    assert result.rounds > 0
