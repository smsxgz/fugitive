from __future__ import annotations

from unittest.mock import patch

import pytest

from fugitive.agents.hierarchical_random import (
    DEFAULT_MULTI_GUESS_CONTINUATION,
    HierarchicalRandomFugitiveAgent,
    HierarchicalRandomMarshalAgent,
    bounded_sprint_plans,
    sprint_cost,
)
from fugitive.belief import solve_observation_belief
from fugitive.model import (
    DrawRecord,
    FugitiveAction,
    Observation,
    Phase,
    Role,
    RouteView,
)


SETUP_DRAWS = (
    DrawRecord(Role.FUGITIVE, 0, None, 0),
    DrawRecord(Role.FUGITIVE, 0, None, 0),
    DrawRecord(Role.FUGITIVE, 0, None, 0),
    DrawRecord(Role.FUGITIVE, 1, None, 0),
    DrawRecord(Role.FUGITIVE, 1, None, 0),
)


def observation(
    role: Role,
    phase: Phase,
    hand: tuple[int, ...],
    route: tuple[RouteView, ...],
) -> Observation:
    return Observation(
        role=role,
        hand=hand,
        pile_sizes=(8, 12, 13),
        route=route,
        guess_history=(),
        draw_history=SETUP_DRAWS,
        round_number=3,
        phase=phase,
        legal_draw_piles=(),
    )


def test_sprint_cost_protects_only_cards_beyond_the_destination() -> None:
    assert sprint_cost(4, 8, (5,)) == pytest.approx(1.0)
    assert sprint_cost(4, 8, (11,)) == pytest.approx(1.75)

    plans = bounded_sprint_plans(
        (5, 8, 11, 42),
        previous_hideout=4,
        hideout=8,
        max_low_cost=2,
        max_extra_overpay=0,
    )
    assert [(plan.cards, plan.cost) for plan in plans.low_cost] == [
        ((5,), 1.0),
        ((11,), 1.75),
    ]


def test_pass_treats_sprint_cards_below_destination_as_spendable() -> None:
    state = observation(
        Role.FUGITIVE,
        Phase.FUGITIVE_ACTION,
        (5, 8, 11, 42),
        (
            RouteView(0, 0, 0, (), True),
            RouteView(1, 4, 0, (), False),
        ),
    )

    distribution = HierarchicalRandomFugitiveAgent(1).fugitive_action_distribution(
        state
    )

    assert distribution[FugitiveAction(None)] == pytest.approx(0.03)


def test_default_multi_guess_continuation_is_conservative() -> None:
    assert DEFAULT_MULTI_GUESS_CONTINUATION == pytest.approx(0.10)
    assert HierarchicalRandomMarshalAgent().multi_guess_continuation == pytest.approx(
        0.10
    )


def test_marshal_reuses_one_solved_path_belief_per_distribution() -> None:
    state = observation(
        Role.MARSHAL,
        Phase.MARSHAL_GUESS,
        (),
        (
            RouteView(0, 0, 0, (), True),
            RouteView(1, 3, 0, (), True),
            RouteView(2, None, 0, None, False),
            RouteView(3, None, 0, None, False),
        ),
    )
    solve_observation_belief.cache_clear()

    with patch(
        "fugitive.agents.hierarchical_random.solve_observation_belief",
        wraps=solve_observation_belief,
    ) as solve:
        distribution = HierarchicalRandomMarshalAgent(2).guess_distribution(state)

    assert distribution
    solve.assert_called_once_with(state)
