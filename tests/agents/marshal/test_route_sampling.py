from __future__ import annotations

from fractions import Fraction
import math
import random

import pytest

from fugitive.agents.marshal.inference.path_belief import (
    PathBelief,
)
from fugitive.game.model import RouteView


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


def test_compiled_route_catalogue_samples_its_exact_distribution() -> None:
    belief = PathBelief(
        (_slot(0, 0), _slot(1, None), _slot(2, None)),
        candidate_cards=range(1, 7),
    )
    catalogue = belief.compile_route_catalogue()

    assert catalogue.total_paths == belief.total_paths == 9
    sample = catalogue.sample(random.Random(91))
    assert sample is not None
    assert sample == catalogue.route_from_rank(sample.rank)
    assert sample.total_completions == catalogue.total_paths
    assert sample.log_q == pytest.approx(-math.log(sample.total_completions))
