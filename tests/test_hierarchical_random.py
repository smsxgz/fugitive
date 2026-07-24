from __future__ import annotations

import random

import pytest

from fugitive.agents.hierarchical_random import (
    HierarchicalRandomFugitiveAgent,
    HierarchicalRandomMarshalAgent,
    bounded_sprint_plans,
    enumerate_opening_plans,
    hard_constraint_guess_numbers,
    normalized_play_actions,
    sprint_cost,
)
from fugitive.engine import GameEngine
from fugitive.model import (
    DrawRecord,
    FugitiveAction,
    GuessRecord,
    Observation,
    Phase,
    Role,
    RouteView,
)
from fugitive.rules import is_legal_fugitive_action, is_legal_guess


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


def test_sprint_cost_and_bounded_plans_protect_42() -> None:
    assert sprint_cost(10, 16, (8, 12)) == pytest.approx(2.5)

    group = bounded_sprint_plans((11, 12, 14, 42), 10, 14)

    assert 1 <= len(group.low_cost) <= 8
    assert len(group.extra_overpay) <= 8
    assert all(42 not in payment.cards for payment in group.all_payments)
    assert all(payment.overpay == 0 for payment in group.low_cost)
    assert all(payment.overpay > 0 for payment in group.extra_overpay)
    assert all(
        is_legal_fugitive_action(
            FugitiveAction(14, payment.cards),
            (11, 12, 14, 42),
            10,
            allow_pass=False,
        )
        for payment in group.all_payments
    )


def test_opening_is_selected_as_a_complete_two_step_plan() -> None:
    engine = GameEngine(seed=19)
    observation = engine.observation(Role.FUGITIVE)
    plans = enumerate_opening_plans(observation)

    assert plans.low_cost
    assert all(42 not in plan.first.sprint_cards for plan in plans.low_cost)
    assert all(42 not in plan.second.sprint_cards for plan in plans.low_cost)

    agent = HierarchicalRandomFugitiveAgent(7)
    first = agent.choose_fugitive_action(observation)
    engine.apply_fugitive_action(first)
    second_observation = engine.observation(Role.FUGITIVE)
    second_distribution = agent.fugitive_action_distribution(second_observation)

    assert second_distribution
    assert second_distribution == HierarchicalRandomFugitiveAgent(
        999
    ).fugitive_action_distribution(second_observation)
    assert sum(second_distribution.values()) == pytest.approx(1.0)
    assert all(
        any(plan.first == first and plan.second == action for plan in plans.low_cost)
        or any(plan.first == first and plan.second == action for plan in plans.extra_overpay)
        for action in second_distribution
    )
    second = agent.choose_fugitive_action(second_observation)
    assert second in second_distribution
    assert 42 not in first.sprint_cards
    assert 42 not in second.sprint_cards
    assert is_legal_fugitive_action(
        second,
        second_observation.hand,
        first.hideout,  # type: ignore[arg-type]
        allow_pass=False,
    )
    engine.apply_fugitive_action(second)
    assert len(engine.observation(Role.FUGITIVE).route) == 3
    assert engine.phase is Phase.MARSHAL_DRAW


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


@pytest.mark.parametrize(
    ("hand", "route", "expected_pass"),
    (
        (
            (12, 20, 42),
            (
                RouteView(0, 0, 0, (), True),
                RouteView(1, 10, 0, (), False),
            ),
            0.03,
        ),
        (
            (16, 17, 19, 21, 42),
            (
                RouteView(0, 0, 0, (), True),
                RouteView(1, 10, 0, (), False),
            ),
            0.25,
        ),
        (
            (1, 2),
            (
                RouteView(0, 0, 0, (), True),
                RouteView(1, 41, 0, (), False),
            ),
            1.0,
        ),
        (
            (42,),
            (
                RouteView(0, 0, 0, (), True),
                RouteView(1, 40, 0, (), True),
            ),
            0.0,
        ),
    ),
)
def test_pass_is_a_state_dependent_macro_action(
    hand: tuple[int, ...],
    route: tuple[RouteView, ...],
    expected_pass: float,
) -> None:
    observation = make_observation(
        role=Role.FUGITIVE,
        phase=Phase.FUGITIVE_ACTION,
        hand=hand,
        route=route,
    )
    distribution = HierarchicalRandomFugitiveAgent(
        3
    ).fugitive_action_distribution(observation)

    assert distribution.get(FugitiveAction(None), 0.0) == pytest.approx(
        expected_pass
    )
    assert sum(distribution.values()) == pytest.approx(1.0)


def test_hr_draw_choice_is_uniform_and_seed_reproducible() -> None:
    observation = make_observation(
        role=Role.FUGITIVE,
        phase=Phase.FUGITIVE_DRAW,
        hand=(1, 2, 3, 42),
        route=(RouteView(0, 0, 0, (), True),),
        legal_draw_piles=(0, 2),
    )
    first = HierarchicalRandomFugitiveAgent(11)
    second = HierarchicalRandomFugitiveAgent(11)

    assert first.draw_pile_distribution(observation) == {0: 0.5, 2: 0.5}
    assert first.choose_draw_pile(observation) == second.choose_draw_pile(observation)


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


def test_failed_single_excludes_a_value_but_failed_multi_is_joint_only() -> None:
    route = (
        RouteView(0, 0, 0, (), True),
        RouteView(1, 3, 0, (), True),
        RouteView(2, None, 0, None, False),
        RouteView(3, None, 0, None, False),
    )
    failed_single = make_observation(
        role=Role.MARSHAL,
        phase=Phase.MARSHAL_GUESS,
        hand=(),
        route=route,
        guesses=(GuessRecord((4,), False, 3, 2),),
    )
    failed_multi = make_observation(
        role=Role.MARSHAL,
        phase=Phase.MARSHAL_GUESS,
        hand=(),
        route=route,
        guesses=(GuessRecord((4, 7), False, 3, 2),),
    )

    assert 4 not in hard_constraint_guess_numbers(failed_single)
    assert 4 in hard_constraint_guess_numbers(failed_multi)
    assert 7 in hard_constraint_guess_numbers(failed_multi)

    distribution = HierarchicalRandomMarshalAgent(
        5,
        multi_guess_continuation=0.5,
        max_guess_size=2,
    ).guess_distribution(failed_multi)
    assert (4,) in distribution
    assert (7,) in distribution
    assert (4, 7) not in distribution


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


def test_agents_accept_injected_random_generators() -> None:
    fugitive_rng = random.Random(3)
    marshal_rng = random.Random(4)

    assert HierarchicalRandomFugitiveAgent(rng=fugitive_rng).rng is fugitive_rng
    assert HierarchicalRandomMarshalAgent(rng=marshal_rng).rng is marshal_rng
    with pytest.raises(ValueError, match="either seed or rng"):
        HierarchicalRandomFugitiveAgent(1, rng=random.Random(1))
