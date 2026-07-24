from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
import math
import random

import pytest

from fugitive.belief import PathBelief, solve_observation_belief
from fugitive.model import DrawRecord, GuessRecord, Observation, Phase, Role, RouteView


def _slot(index: int, hideout: int | None) -> RouteView:
    return RouteView(index, hideout, 0, () if hideout is not None else None)


def test_route_unranking_exhausts_exact_constrained_distribution() -> None:
    route = (
        _slot(0, 0),
        _slot(1, None),
        _slot(2, 4),
        _slot(3, None),
    )
    belief = PathBelief(
        route,
        failed_single_guesses=((2, 1),),
        failed_multi_guesses=(((1, 5), 3),),
        pile_hideout_limits=(2, 0, 0),
        pile_hideout_prefix_limits=(
            (0, 0, 0),
            (0, 0, 0),
            (1, 0, 0),
            (2, 0, 0),
        ),
        candidate_cards=range(1, 9),
    )
    expected = {
        (0, 1, 4, 6),
        (0, 1, 4, 7),
        (0, 3, 4, 5),
        (0, 3, 4, 6),
        (0, 3, 4, 7),
    }

    assert belief.total_paths == len(expected)
    samples = tuple(
        belief.route_from_rank(rank) for rank in range(belief.total_paths)
    )

    assert {sample.route for sample in samples} == expected
    assert len({sample.route for sample in samples}) == belief.total_paths
    for rank, sample in enumerate(samples):
        assert sample.rank == rank
        assert sample.total_completions == belief.total_paths
        assert sample.prefix_completion_counts[0] == belief.total_paths
        assert sample.prefix_completion_counts[-1] == 1
        assert sample.prefix_ranks[0] == rank
        assert sample.prefix_ranks[-1] == 0
        assert all(
            0 <= local_rank < completions
            for local_rank, completions in zip(
                sample.prefix_ranks,
                sample.prefix_completion_counts,
                strict=True,
            )
        )
        assert math.prod(
            sample.conditional_probabilities,
            start=Fraction(1),
        ) == Fraction(1, belief.total_paths)
        assert sample.proposal_probability == Fraction(1, belief.total_paths)

    sampled = belief.sample_route(random.Random(91))
    repeated = belief.sample_route(random.Random(91))
    assert sampled == repeated
    assert sampled.route in expected


def test_route_sampling_rejects_empty_beliefs_and_invalid_ranks() -> None:
    empty = PathBelief((_slot(0, 0), _slot(1, 7)), candidate_cards=range(1, 8))
    nonempty = PathBelief(
        (_slot(0, 0), _slot(1, None)),
        candidate_cards=range(1, 4),
    )

    with pytest.raises(ValueError, match="empty route belief"):
        empty.sample_route(random.Random(1))
    with pytest.raises(ValueError, match="route rank"):
        nonempty.route_from_rank(-1)
    with pytest.raises(ValueError, match="route rank"):
        nonempty.route_from_rank(nonempty.total_paths)
    with pytest.raises(TypeError, match="integer"):
        nonempty.route_from_rank(True)


def test_solve_observation_belief_cache_is_bounded_to_256() -> None:
    observation = Observation(
        role=Role.MARSHAL,
        hand=(),
        pile_sizes=(8, 12, 13),
        route=(_slot(0, 0), _slot(1, None)),
        guess_history=(GuessRecord((9,), False, 1, 1),),
        draw_history=(
            DrawRecord(Role.FUGITIVE, 0, None, 0),
            DrawRecord(Role.FUGITIVE, 1, None, 0),
        ),
        round_number=1,
        phase=Phase.MARSHAL_GUESS,
        legal_draw_piles=(),
    )
    solve_observation_belief.cache_clear()
    try:
        for round_number in range(300):
            solve_observation_belief(
                replace(observation, round_number=round_number + 1)
            )
        info = solve_observation_belief.cache_info()
        assert info.maxsize == 256
        assert info.currsize == 256
    finally:
        solve_observation_belief.cache_clear()
