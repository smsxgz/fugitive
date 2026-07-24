from __future__ import annotations

from dataclasses import replace
import math
import random

import pytest

from fugitive.agents.route_count_random import RouteCountRandomMarshalAgent
from fugitive.belief import PathBelief
from fugitive.engine import GameEngine
from fugitive.model import DrawRecord, FugitiveAction, Observation, Phase, Role, RouteView
from fugitive.rules import is_legal_guess


SETUP_DRAWS = (
    DrawRecord(Role.FUGITIVE, 0, None, 0),
    DrawRecord(Role.FUGITIVE, 0, None, 0),
    DrawRecord(Role.FUGITIVE, 0, None, 0),
    DrawRecord(Role.FUGITIVE, 1, None, 0),
    DrawRecord(Role.FUGITIVE, 1, None, 0),
)


def _observation(
    *,
    phase: Phase = Phase.MARSHAL_GUESS,
    hidden_slots: int = 2,
    legal_draw_piles: tuple[int, ...] = (),
) -> Observation:
    route = (
        RouteView(0, 0, 0, (), True),
        RouteView(1, 3, 0, (), True),
        *(
            RouteView(index, None, 0, None, False)
            for index in range(2, hidden_slots + 2)
        ),
    )
    return Observation(
        role=Role.MARSHAL,
        hand=(),
        pile_sizes=(8, 12, 13),
        route=route,
        guess_history=(),
        draw_history=SETUP_DRAWS,
        round_number=3,
        phase=phase,
        legal_draw_piles=legal_draw_piles,
    )


def _assert_count_power_distribution(
    distribution: dict[tuple[int, ...], float],
    counts: dict[tuple[int, ...], int],
    *,
    exponent: float,
    epsilon: float,
) -> None:
    denominator = sum(count**exponent for count in counts.values())
    uniform = epsilon / len(counts)
    for choice, count in counts.items():
        expected = uniform + (1.0 - epsilon) * count**exponent / denominator
        assert distribution[choice] == pytest.approx(expected)


def test_draw_choice_stays_uniform_to_isolate_guess_weighting() -> None:
    observation = _observation(
        phase=Phase.MARSHAL_DRAW,
        legal_draw_piles=(0, 2),
    )
    first = RouteCountRandomMarshalAgent(13)
    second = RouteCountRandomMarshalAgent(13)

    assert first.draw_pile_distribution(observation) == {0: 0.5, 2: 0.5}
    assert first.choose_draw_pile(observation) == second.choose_draw_pile(observation)


def test_singletons_use_marginal_count_power_with_epsilon() -> None:
    observation = _observation(hidden_slots=2)
    result = PathBelief.from_observation(observation).solve()
    agent = RouteCountRandomMarshalAgent(
        5,
        alpha=1.0,
        epsilon=0.2,
        max_guess_size=1,
    )
    distribution = agent.guess_distribution(observation)
    counts = {(card,): count for card, count in result.marginal_counts.items()}

    assert set(distribution) == set(counts)
    _assert_count_power_distribution(
        distribution,
        counts,
        exponent=1.0,
        epsilon=0.2,
    )
    assert sum(distribution.values()) == pytest.approx(1.0)


def test_manhunt_uses_squared_marginals_and_single_guesses_only() -> None:
    observation = _observation(phase=Phase.MANHUNT, hidden_slots=2)
    result = PathBelief.from_observation(observation).solve()
    agent = RouteCountRandomMarshalAgent(7, epsilon=0.15)
    distribution = agent.guess_distribution(observation)
    counts = {(card,): count for card, count in result.marginal_counts.items()}

    assert all(len(guess) == 1 for guess in distribution)
    _assert_count_power_distribution(
        distribution,
        counts,
        exponent=2.0,
        epsilon=0.15,
    )


def test_joint_candidates_are_sampled_from_routes_then_exactly_weighted() -> None:
    observation = _observation(hidden_slots=3)
    result = PathBelief.from_observation(observation).solve()
    agent = RouteCountRandomMarshalAgent(
        11,
        alpha=1.0,
        epsilon=0.0,
        multi_guess_continuation=0.2,
        max_guess_size=3,
        max_joint_candidates=8,
    )
    distribution = agent.guess_distribution(observation)

    by_size = {
        size: {
            guess: probability
            for guess, probability in distribution.items()
            if len(guess) == size
        }
        for size in (1, 2, 3)
    }
    assert all(by_size.values())
    assert len(by_size[2]) <= 8
    assert len(by_size[3]) <= 8
    assert sum(by_size[1].values()) == pytest.approx(0.8)
    assert sum(by_size[2].values()) == pytest.approx(0.16)
    assert sum(by_size[3].values()) == pytest.approx(0.04)

    for size in (2, 3):
        mass = sum(by_size[size].values())
        exact_counts = {
            guess: result.count_paths_containing(guess)
            for guess in by_size[size]
        }
        count_total = sum(exact_counts.values())
        for guess, probability in by_size[size].items():
            assert probability / mass == pytest.approx(
                exact_counts[guess] / count_total
            )
            assert is_legal_guess(guess)


def test_distribution_depends_only_on_marshal_observation() -> None:
    first_engine = GameEngine(seed=1)
    second_engine = GameEngine(seed=999)
    first_engine.apply_fugitive_action(FugitiveAction(1))
    first_engine.apply_fugitive_action(FugitiveAction(2))
    second_engine.apply_fugitive_action(FugitiveAction(2))
    second_engine.apply_fugitive_action(FugitiveAction(3))
    first_observation = replace(
        first_engine.observation(Role.MARSHAL),
        phase=Phase.MARSHAL_GUESS,
        legal_draw_piles=(),
    )
    second_observation = replace(
        second_engine.observation(Role.MARSHAL),
        phase=Phase.MARSHAL_GUESS,
        legal_draw_piles=(),
    )

    assert first_observation == second_observation
    assert RouteCountRandomMarshalAgent(23).guess_distribution(
        first_observation
    ) == RouteCountRandomMarshalAgent(999).guess_distribution(second_observation)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"alpha": 0.0}, "alpha"),
        ({"alpha": math.inf}, "alpha"),
        ({"alpha": True}, "alpha"),
        ({"epsilon": -0.1}, "epsilon"),
        ({"epsilon": math.nan}, "epsilon"),
        ({"epsilon": True}, "epsilon"),
        ({"multi_guess_continuation": 1.1}, "multi_guess_continuation"),
        ({"max_guess_size": 0}, "max_guess_size"),
        ({"max_guess_size": 4}, "max_guess_size"),
        ({"max_joint_candidates": 0}, "max_joint_candidates"),
        ({"max_joint_candidates": 49}, "max_joint_candidates"),
    ),
)
def test_invalid_parameters_are_rejected(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        RouteCountRandomMarshalAgent(**kwargs)  # type: ignore[arg-type]


def test_injected_rng_and_wrong_role_handling() -> None:
    rng = random.Random(31)
    agent = RouteCountRandomMarshalAgent(rng=rng)
    wrong_role = replace(_observation(), role=Role.FUGITIVE)

    assert agent.rng is rng
    assert agent.guess_distribution(wrong_role) == {}
    with pytest.raises(ValueError, match="either seed or rng"):
        RouteCountRandomMarshalAgent(1, rng=random.Random(1))
