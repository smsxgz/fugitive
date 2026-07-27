from __future__ import annotations

from dataclasses import replace

import pytest

from fugitive.agents.marshal.candidates import build_marshal_guess_catalogue
from fugitive.agents.marshal.route_count_catalogue import (
    RouteCountCatalogueRandomMarshalAgent,
)
from fugitive.agents.marshal.support_catalogue import (
    SupportCatalogueRandomMarshalAgent,
)
from fugitive.game.engine import GameEngine
from fugitive.game.model import DrawRecord, FugitiveAction, Observation, Phase, Role, RouteView


def _observation(*, phase: Phase = Phase.MARSHAL_GUESS) -> Observation:
    setup = (
        DrawRecord(Role.FUGITIVE, 0, None, 0),
        DrawRecord(Role.FUGITIVE, 0, None, 0),
        DrawRecord(Role.FUGITIVE, 0, None, 0),
        DrawRecord(Role.FUGITIVE, 1, None, 0),
        DrawRecord(Role.FUGITIVE, 1, None, 0),
    )
    return Observation(
        role=Role.MARSHAL,
        hand=(),
        pile_sizes=(8, 12, 13),
        route=(
            RouteView(0, 0, 0, (), True),
            RouteView(1, 3, 0, (), True),
            RouteView(2, None, 0, None, False),
            RouteView(3, None, 1, None, False),
            RouteView(4, None, 0, None, False),
        ),
        guess_history=(),
        draw_history=setup,
        round_number=3,
        phase=phase,
        legal_draw_piles=(),
    )


def test_controlled_agents_use_exactly_the_same_catalogue_and_size_prior() -> None:
    observation = _observation()
    support = SupportCatalogueRandomMarshalAgent(
        1, multi_guess_continuation=0.2, max_joint_candidates=12
    ).guess_distribution(observation)
    counted = RouteCountCatalogueRandomMarshalAgent(
        1, multi_guess_continuation=0.2, max_joint_candidates=12
    ).guess_distribution(observation)

    assert set(support) == set(counted)
    assert set(support) == set(
        build_marshal_guess_catalogue(
            observation, max_joint_candidates=12
        ).guesses
    )
    for size in {len(guess) for guess in support}:
        support_mass = sum(p for guess, p in support.items() if len(guess) == size)
        counted_mass = sum(p for guess, p in counted.items() if len(guess) == size)
        assert support_mass == pytest.approx(counted_mass)


def test_only_conditional_weighting_changes() -> None:
    observation = _observation()
    catalogue = build_marshal_guess_catalogue(
        observation, max_guess_size=1
    ).counts_by_size()[1]
    support = SupportCatalogueRandomMarshalAgent(
        2, max_guess_size=1
    ).guess_distribution(observation)
    counted = RouteCountCatalogueRandomMarshalAgent(
        2, epsilon=0.0, max_guess_size=1
    ).guess_distribution(observation)

    assert len(set(support.values())) == 1
    total = sum(catalogue.values())
    for guess, count in catalogue.items():
        assert counted[guess] == pytest.approx(count / total)


def test_manhunt_catalogue_contains_singletons_only() -> None:
    observation = _observation(phase=Phase.MANHUNT)
    catalogue = build_marshal_guess_catalogue(observation)

    assert catalogue.guesses
    assert all(len(guess) == 1 for guess in catalogue.guesses)
    assert set(
        SupportCatalogueRandomMarshalAgent(4).guess_distribution(observation)
    ) == set(catalogue.guesses)


def test_catalogue_depends_only_on_marshal_observation() -> None:
    first = GameEngine(seed=1)
    second = GameEngine(seed=999)
    first.apply_fugitive_action(FugitiveAction(1))
    first.apply_fugitive_action(FugitiveAction(2))
    second.apply_fugitive_action(FugitiveAction(2))
    second.apply_fugitive_action(FugitiveAction(3))
    first_observation = replace(
        first.observation(Role.MARSHAL),
        phase=Phase.MARSHAL_GUESS,
        legal_draw_piles=(),
    )
    second_observation = replace(
        second.observation(Role.MARSHAL),
        phase=Phase.MARSHAL_GUESS,
        legal_draw_piles=(),
    )

    assert first_observation == second_observation
    assert build_marshal_guess_catalogue(
        first_observation
    ) == build_marshal_guess_catalogue(second_observation)
