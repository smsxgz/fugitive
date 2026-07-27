from __future__ import annotations

import pytest

from fugitive.agents.fugitive.hierarchical_random import (
    HierarchicalRandomFugitiveAgent,
    bounded_sprint_plans,
    sprint_cost,
)
from fugitive.game.model import (
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
