from __future__ import annotations

import pytest

from fugitive.agents.fugitive.hierarchical_random import (
    HierarchicalRandomFugitiveAgent,
    normalized_play_actions,
)
from fugitive.agents.marshal.hierarchical_random import (
    HierarchicalRandomMarshalAgent,
    hard_constraint_guess_numbers,
)
from fugitive.game.engine import GameEngine
from fugitive.game.model import (
    DrawRecord,
    FugitiveAction,
    GuessRecord,
    Observation,
    Phase,
    Role,
    RouteView,
)
from fugitive.game.rules import is_legal_guess


SETUP_DRAWS = (
    DrawRecord(Role.FUGITIVE, 0, None, 0),
    DrawRecord(Role.FUGITIVE, 0, None, 0),
    DrawRecord(Role.FUGITIVE, 0, None, 0),
    DrawRecord(Role.FUGITIVE, 1, None, 0),
    DrawRecord(Role.FUGITIVE, 1, None, 0),
)


def make_observation(
    *,
    role: Role,
    phase: Phase,
    hand: tuple[int, ...],
    route: tuple[RouteView, ...],
    guesses: tuple[GuessRecord, ...] = (),
    legal_draw_piles: tuple[int, ...] = (),
) -> Observation:
    return Observation(
        role=role,
        hand=hand,
        pile_sizes=(8, 12, 13),
        route=route,
        guess_history=guesses,
        draw_history=SETUP_DRAWS,
        round_number=3,
        phase=phase,
        legal_draw_piles=legal_draw_piles,
    )


def test_normal_actions_are_grouped_by_hideout_not_payment_count() -> None:
    observation = make_observation(
        role=Role.FUGITIVE,
        phase=Phase.FUGITIVE_ACTION,
        hand=(11, 12, 13, 18, 42),
        route=(
            RouteView(0, 0, 0, (), True),
            RouteView(1, 10, 0, (), False),
        ),
    )
    groups = normalized_play_actions(observation)
    distribution = HierarchicalRandomFugitiveAgent(
        1, overpay_probability=0.0
    ).fugitive_action_distribution(observation)

    assert set(groups) >= {11, 12, 13}
    play_mass = {
        hideout: sum(
            probability
            for action, probability in distribution.items()
            if action.hideout == hideout
        )
        for hideout in groups
    }
    assert len(set(play_mass.values())) == 1
    assert sum(distribution.values()) == pytest.approx(1.0)


def test_marshal_support_uses_route_geometry_and_known_cards() -> None:
    observation = make_observation(
        role=Role.MARSHAL,
        phase=Phase.MARSHAL_GUESS,
        hand=(5,),
        route=(
            RouteView(0, 0, 0, (), True),
            RouteView(1, 3, 0, (), True),
            RouteView(2, None, 0, None, False),
        ),
    )

    assert hard_constraint_guess_numbers(observation) == (4, 6)


def test_guess_size_is_truncated_geometric_and_jointly_possible() -> None:
    observation = make_observation(
        role=Role.MARSHAL,
        phase=Phase.MARSHAL_GUESS,
        hand=(),
        route=(
            RouteView(0, 0, 0, (), True),
            RouteView(1, 3, 0, (), True),
            RouteView(2, None, 0, None, False),
            RouteView(3, None, 0, None, False),
        ),
    )
    distribution = HierarchicalRandomMarshalAgent(
        2,
        multi_guess_continuation=0.35,
        max_guess_size=4,
    ).guess_distribution(observation)

    singleton_mass = sum(
        probability for guess, probability in distribution.items() if len(guess) == 1
    )
    pair_mass = sum(
        probability for guess, probability in distribution.items() if len(guess) == 2
    )
    assert singleton_mass == pytest.approx(0.65)
    assert pair_mass == pytest.approx(0.35)
    assert all(is_legal_guess(guess) for guess in distribution)
    assert sum(distribution.values()) == pytest.approx(1.0)


def test_manhunt_recomputes_a_singleton_distribution_after_reveal() -> None:
    before = make_observation(
        role=Role.MARSHAL,
        phase=Phase.MANHUNT,
        hand=(),
        route=(
            RouteView(0, 0, 0, (), True),
            RouteView(1, 3, 0, (), True),
            RouteView(2, None, 0, None, False),
            RouteView(3, None, 0, None, False),
        ),
    )
    after = make_observation(
        role=Role.MARSHAL,
        phase=Phase.MANHUNT,
        hand=(),
        route=(
            RouteView(0, 0, 0, (), True),
            RouteView(1, 3, 0, (), True),
            RouteView(2, 4, 0, (), True),
            RouteView(3, None, 0, None, False),
        ),
        guesses=(GuessRecord((4,), True, 3, 3, True),),
    )
    agent = HierarchicalRandomMarshalAgent(17)
    before_distribution = agent.guess_distribution(before)
    after_distribution = agent.guess_distribution(after)

    assert all(len(guess) == 1 for guess in before_distribution)
    assert all(len(guess) == 1 for guess in after_distribution)
    assert (4,) in before_distribution
    assert (4,) not in after_distribution
    assert before_distribution != after_distribution


def test_marshal_distribution_cannot_see_hidden_route_or_deck_order() -> None:
    first_engine = GameEngine(seed=1)
    second_engine = GameEngine(seed=999)
    first_engine.apply_fugitive_action(FugitiveAction(1))
    first_engine.apply_fugitive_action(FugitiveAction(2))
    second_engine.apply_fugitive_action(FugitiveAction(2))
    second_engine.apply_fugitive_action(FugitiveAction(3))
    first_observation = first_engine.observation(Role.MARSHAL)
    second_observation = second_engine.observation(Role.MARSHAL)

    assert first_observation == second_observation
    first_distribution = HierarchicalRandomMarshalAgent(23).guess_distribution(
        first_observation
    )
    second_distribution = HierarchicalRandomMarshalAgent(23).guess_distribution(
        second_observation
    )
    assert first_distribution == second_distribution
