from __future__ import annotations

from itertools import combinations

import pytest

from fugitive.belief import PathBelief
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


SETUP_DRAWS = (
    DrawRecord(Role.FUGITIVE, 0, None, 0),
    DrawRecord(Role.FUGITIVE, 0, None, 0),
    DrawRecord(Role.FUGITIVE, 0, None, 0),
    DrawRecord(Role.FUGITIVE, 1, None, 0),
    DrawRecord(Role.FUGITIVE, 1, None, 0),
)


def slot(
    index: int,
    hideout: int | None,
    sprint_count: int = 0,
    sprint_cards: tuple[int, ...] | None = None,
) -> RouteView:
    return RouteView(index, hideout, sprint_count, sprint_cards)


def marshal_observation(
    route: tuple[RouteView, ...],
    *,
    hand: tuple[int, ...] = (),
    guesses: tuple[GuessRecord, ...] = (),
    draws: tuple[DrawRecord, ...] = SETUP_DRAWS,
) -> Observation:
    return Observation(
        role=Role.MARSHAL,
        hand=hand,
        pile_sizes=(11, 14, 13),
        route=route,
        guess_history=guesses,
        draw_history=draws,
        round_number=2,
        phase=Phase.MARSHAL_GUESS,
        legal_draw_piles=(),
    )


def test_simple_path_count_and_joint_queries() -> None:
    # 0 < a < b, with each edge at most three: each a in 1..3 has
    # three possible successors in the candidate range.
    belief = PathBelief(
        (slot(0, 0, sprint_cards=()), slot(1, None), slot(2, None)),
        candidate_cards=range(1, 7),
    )

    assert belief.total_paths == 9
    assert belief.marginals[3] == pytest.approx(5 / 9)
    assert belief.count_paths_containing({2}) == 4
    assert belief.count_paths_containing({2, 4}) == 1
    assert belief.count_paths_containing(set()) == 9


def test_revealed_hideout_constrains_both_sides() -> None:
    belief = PathBelief(
        (
            slot(0, 0, sprint_cards=()),
            slot(1, None),
            slot(2, 5, sprint_cards=()),
            slot(3, None),
        ),
        candidate_cards=range(1, 9),
    )

    # Before 5 only 2 or 3 can also reach it; after 5, 6/7/8 are legal.
    assert belief.total_paths == 6
    assert belief.marginals[2] == pytest.approx(1 / 2)
    assert belief.marginals[3] == pytest.approx(1 / 2)
    assert belief.marginals[6] == pytest.approx(1 / 3)
    assert 1 not in belief.marginals
    assert 5 not in belief.marginals


def test_known_and_unknown_sprint_stacks_have_different_edge_caps() -> None:
    exact = PathBelief(
        (slot(0, 0, sprint_cards=()), slot(1, None, 1, (1,))),
        candidate_cards=range(1, 7),
    )
    hidden = PathBelief(
        (slot(0, 0, sprint_cards=()), slot(1, None, 1, None)),
        candidate_cards=range(1, 7),
    )

    assert exact.total_paths == 4  # cap = 3 + sprint_value(1)
    assert hidden.total_paths == 5  # conservative cap = 3 + 2 * count


def test_failed_single_guess_only_excludes_positions_that_existed_then() -> None:
    route = (slot(0, 0, sprint_cards=()), slot(1, None), slot(2, None))
    belief = PathBelief(
        route,
        failed_single_guesses=((2, 1),),
        candidate_cards=range(1, 7),
    )

    assert belief.total_paths == 6
    # Card 2 is forbidden at index 1, but the Fugitive may play it later at
    # index 2.  The sole remaining path containing it is (1, 2).
    assert belief.count_paths_containing({2}) == 1
    assert belief.marginals[2] == pytest.approx(1 / 6)

    observation = marshal_observation(
        route,
        guesses=(GuessRecord((2,), False, 1, 1),),
    )
    assert PathBelief.from_observation(observation).total_paths == 6


def test_observation_excludes_marshal_hand_and_public_sprint_cards() -> None:
    observation = marshal_observation(
        (
            slot(0, 0, sprint_cards=()),
            slot(1, 3, 1, (5,)),
            slot(2, None),
        ),
        hand=(4,),
    )

    result = PathBelief.from_observation(observation).solve()
    assert result.total_paths == 1
    assert dict(result.marginals) == {6: 1.0}


def test_undrawn_pile_cards_cannot_be_current_hideouts() -> None:
    observation = marshal_observation(
        (slot(0, 0, sprint_cards=()), slot(1, None, 20, None)),
    )

    result = PathBelief.from_observation(observation).solve()
    assert result.total_paths > 0
    assert not set(range(29, 42)) & result.marginals.keys()


def test_real_engine_setup_history_produces_nonempty_320_belief() -> None:
    engine = GameEngine(seed=17)
    engine.apply_fugitive_action(FugitiveAction(1))
    engine.apply_fugitive_action(FugitiveAction(2))

    observation = engine.observation(Role.MARSHAL)
    belief = PathBelief.from_observation(observation)

    assert belief.pile_hideout_limits == (3, 2, 0)
    assert belief.pile_hideout_prefix_limits == (
        (3, 2, 0),
        (3, 2, 0),
        (3, 2, 0),
    )
    assert belief.total_paths > 0


def test_fugitive_draw_count_caps_hideouts_from_each_pile() -> None:
    only_two_pile_zero_draws = (
        DrawRecord(Role.FUGITIVE, 0, None, 0),
        DrawRecord(Role.FUGITIVE, 0, None, 0),
    )
    observation = marshal_observation(
        (
            slot(0, 0, sprint_cards=()),
            slot(1, 4, sprint_cards=()),
            slot(2, 5, sprint_cards=()),
            slot(3, 6, sprint_cards=()),
        ),
        draws=only_two_pile_zero_draws,
    )

    assert PathBelief.from_observation(observation).total_paths == 0


def test_public_sprint_card_consumes_draw_capacity() -> None:
    two_draws = (
        DrawRecord(Role.FUGITIVE, 0, None, 0),
        DrawRecord(Role.FUGITIVE, 0, None, 0),
    )
    observation = marshal_observation(
        (
            slot(0, 0, sprint_cards=()),
            slot(1, 4, 1, (6,)),
            slot(2, 5, 0, ()),
        ),
        draws=two_draws,
    )

    assert PathBelief.from_observation(observation).total_paths == 0


def test_later_draw_cannot_supply_an_earlier_route_prefix() -> None:
    draws = (
        DrawRecord(Role.FUGITIVE, 0, None, 0),
        DrawRecord(Role.FUGITIVE, 0, None, 2),
    )
    observation = marshal_observation(
        (
            slot(0, 0, sprint_cards=()),
            slot(1, 4, sprint_cards=()),
            slot(2, 5, sprint_cards=()),
        ),
        guesses=(GuessRecord((1,), False, 2, 1),),
        draws=draws,
    )

    belief = PathBelief.from_observation(observation)
    assert belief.pile_hideout_limits == (2, 0, 0)
    assert belief.pile_hideout_prefix_limits[2] == (1, 0, 0)
    assert belief.total_paths == 0


def test_failed_multi_guess_excludes_joint_historical_hit() -> None:
    route = (slot(0, 0, sprint_cards=()), slot(1, None), slot(2, None))
    unconstrained = PathBelief(route, candidate_cards=range(1, 7))
    constrained = PathBelief(
        route,
        failed_multi_guesses=(((2, 3), 2),),
        candidate_cards=range(1, 7),
    )

    assert unconstrained.total_paths == 9
    assert constrained.total_paths == 8
    assert constrained.count_paths_containing({2, 3}) == 0

    observation = marshal_observation(
        route,
        guesses=(GuessRecord((2, 3), False, 2, 1),),
    )
    observed = PathBelief.from_observation(observation)
    assert observed.total_paths == 8
    assert observed.count_paths_containing({2, 3}) == 0


def test_failed_multi_only_constrains_the_route_prefix_at_that_time() -> None:
    route = (
        slot(0, 0, sprint_cards=()),
        slot(1, None),
        slot(2, None),
        slot(3, None),
    )
    belief = PathBelief(
        route,
        failed_multi_guesses=(((2, 3), 2),),
        candidate_cards=range(1, 8),
    )

    # The pair can become true after the route grows: path (1, 2, 3) was a
    # failed pair at length two but contains both values at length three.
    assert belief.count_paths_containing({2, 3}) > 0


def test_failed_multi_dominance_eliminates_same_and_earlier_prefix_supersets() -> None:
    route = tuple(
        [slot(0, 0, sprint_cards=())]
        + [slot(index, None, 3, None) for index in range(1, 5)]
    )
    belief = PathBelief(
        route,
        failed_multi_guesses=(
            ((2, 4, 6), 4),  # Same-prefix superset: redundant.
            ((1, 3, 7), 3),  # Earlier-prefix superset: redundant.
            ((2, 4), 4),
            ((1, 3), 4),
        ),
        candidate_cards=range(1, 9),
    )

    assert belief.failed_multi_guesses == (((2, 4), 4), ((1, 3), 4))


def test_failed_multi_dominance_does_not_reverse_the_prefix_direction() -> None:
    route = tuple(
        [slot(0, 0, sprint_cards=())]
        + [slot(index, None, 3, None) for index in range(1, 5)]
    )
    constraints = (((2, 4), 2), ((2, 4, 6), 4))
    belief = PathBelief(
        route,
        failed_multi_guesses=constraints,
        candidate_cards=range(1, 9),
    )

    assert belief.failed_multi_guesses == constraints


def test_failed_multi_dominance_matches_brute_force_path_semantics() -> None:
    cards = tuple(range(1, 9))
    route_length = 4
    route = tuple(
        [slot(0, 0, sprint_cards=())]
        + [slot(index, None, 3, None) for index in range(1, route_length + 1)]
    )
    constraint_cases = (
        (((2, 4, 6), 4), ((2, 4), 4)),
        (((1, 3, 7), 3), ((1, 3), 4)),
        (((2, 4), 2), ((2, 4, 6), 4)),
        (((1, 5, 7), 4), ((1, 5), 3), ((2, 6), 4)),
    )

    for constraints in constraint_cases:
        valid_paths = tuple(
            path
            for path in combinations(cards, route_length)
            if not any(
                set(numbers).issubset(path[:prefix_length])
                for numbers, prefix_length in constraints
            )
        )
        belief = PathBelief(
            route,
            failed_multi_guesses=constraints,
            candidate_cards=cards,
        )

        assert belief.total_paths == len(valid_paths)
        for required in combinations(cards, 2):
            expected = sum(set(required).issubset(path) for path in valid_paths)
            assert belief.count_paths_containing(required) == expected


def test_observation_bounds_number_of_retained_multi_constraints() -> None:
    route = (slot(0, 0, sprint_cards=()), slot(1, None), slot(2, None))
    guesses = (
        GuessRecord((1, 2), False, 2, 1),
        GuessRecord((2, 3), False, 2, 2),
    )
    observation = marshal_observation(route, guesses=guesses)

    assert len(
        PathBelief.from_observation(
            observation, max_failed_multi_constraints=1
        ).failed_multi_guesses
    ) == 1
    assert not PathBelief.from_observation(
        observation, max_failed_multi_constraints=0
    ).failed_multi_guesses


def test_observation_retains_more_than_eight_multi_membership_bits() -> None:
    route = tuple(
        [slot(0, 0, sprint_cards=())]
        + [slot(index, None, 2, None) for index in range(1, 11)]
    )
    guesses = tuple(
        GuessRecord((2 * index + 1, 2 * index + 2), False, 10, index)
        for index in range(5)
    )
    observation = marshal_observation(route, guesses=guesses)

    belief = PathBelief.from_observation(observation)
    assert belief.failed_multi_guesses == tuple(
        (record.numbers, record.route_length) for record in guesses
    )
    assert sum(len(numbers) for numbers, _length in belief.failed_multi_guesses) == 10


def test_old_failed_pair_still_constrains_paths_after_four_newer_pairs() -> None:
    route = tuple(
        [slot(0, 0, sprint_cards=())]
        + [slot(index, None, 2, None) for index in range(1, 11)]
    )
    guesses = tuple(
        GuessRecord((2 * index + 1, 2 * index + 2), False, 10, index)
        for index in range(5)
    )
    draws = tuple(
        DrawRecord(Role.FUGITIVE, 0, None, 0) for _index in range(11)
    )
    observation = marshal_observation(route, guesses=guesses, draws=draws)

    exact = PathBelief.from_observation(observation)
    explicitly_bounded = PathBelief.from_observation(
        observation, max_failed_multi_mask_bits=8
    )

    assert exact.count_paths_containing((1, 2)) == 0
    assert explicitly_bounded.count_paths_containing((1, 2)) > 0


def test_contradictory_public_route_returns_safe_empty_result() -> None:
    belief = PathBelief(
        (slot(0, 0, sprint_cards=()), slot(1, 7, sprint_cards=())),
        candidate_cards=range(1, 8),
    )

    result = belief.solve()
    assert result.total_paths == 0
    assert dict(result.marginals) == {}
    assert result.count_paths_containing({7}) == 0
