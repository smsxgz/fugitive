from __future__ import annotations

from collections import Counter
from dataclasses import replace
from itertools import permutations
import math
import random
import time

import pytest

from fugitive.belief import PathBelief
from fugitive.agents.hierarchical_random import (
    HierarchicalRandomFugitiveAgent,
    HierarchicalRandomMarshalAgent,
)
from fugitive.constructive_belief import (
    CARD_TO_PILE,
    INITIAL_FUGITIVE_CARDS,
    CompiledMarshalConstraints,
    ConstraintUniformTarget,
    ConstructiveWorldSampler,
    DrawDeadlineMatcher,
    DrawSlot,
    PathBeliefRouteSampler,
    SamplingBudget,
    SequentialSprintAssignmentProposal,
    SprintAssignmentDP,
    _BudgetCounter,
    _INITIAL_SOURCE,
    _SPRINT_CATEGORY_INDEX,
)
from fugitive.engine import GameEngine
from fugitive.model import FugitiveAction, Phase, Role, RouteView
from fugitive.particle_belief import _particle_matches_observation
from fugitive.rules import PILE_CARDS, sprint_value


def _opening_observation(
    first: FugitiveAction = FugitiveAction(1),
    second: FugitiveAction = FugitiveAction(2),
    *,
    seed: int = 7,
):
    engine = GameEngine(seed=seed)
    engine.apply_fugitive_action(first)
    engine.apply_fugitive_action(second)
    return engine, engine.observation(Role.MARSHAL)


def _revealed_numbers(observation) -> frozenset[int]:
    return frozenset(
        slot.hideout
        for index, slot in enumerate(observation.route)
        if index and slot.revealed and slot.hideout is not None
    )


def test_route_sampler_count_and_raw_proposal_match_bruteforce() -> None:
    route = (
        RouteView(0, 0, 0, (), True),
        RouteView(1, None, 0, None, False),
        RouteView(2, None, 0, None, False),
    )
    candidates = tuple(range(1, 6))
    expected = tuple(
        (0, first, second)
        for first in candidates
        for second in candidates
        if 0 < first < second and first <= 3 and second - first <= 3
    )
    belief = PathBelief(route, candidate_cards=candidates)
    counter = _BudgetCounter(SamplingBudget(max_nodes=10_000))
    sampler = PathBeliefRouteSampler(belief, counter)

    assert sampler.total_paths == len(expected)
    observed = Counter()
    for seed in range(2_000):
        sample = sampler.sample(random.Random(seed))
        assert sample is not None
        assert sample.route in expected
        assert sample.total_paths == len(expected)
        assert sample.log_q == pytest.approx(-math.log(len(expected)))
        observed[sample.route] += 1

    # This is intentionally broad: exact count/q checks carry correctness;
    # this catches an accidentally deterministic branch choice.
    expected_frequency = 2_000 / len(expected)
    assert all(
        abs(count - expected_frequency) < expected_frequency * 0.30
        for count in observed.values()
    )


def test_route_adapter_matches_constrained_path_belief_count() -> None:
    engine, _observation = _opening_observation(
        FugitiveAction(1), FugitiveAction(3), seed=5
    )
    engine.draw(2)
    engine.draw(2)
    assert not engine.apply_guess((1, 2))
    engine.draw(2)
    engine.apply_fugitive_action(FugitiveAction(None))
    engine.draw(2)
    observation = engine.observation(Role.MARSHAL)
    constraints = CompiledMarshalConstraints.from_observation(observation)
    sampler = PathBeliefRouteSampler(
        constraints.path_belief,
        _BudgetCounter(SamplingBudget(max_nodes=50_000)),
    )

    assert sampler.total_paths == constraints.path_belief.total_paths


def test_draw_deadline_matcher_count_and_q_match_enumeration() -> None:
    cards = ((4, 5, 6, 7), (), ())
    slots = (DrawSlot(0, 0, 0), DrawSlot(1, 0, 2))
    counter = _BudgetCounter(SamplingBudget(max_nodes=100))
    matcher = DrawDeadlineMatcher(slots, (7,), pile_cards=cards, budget=counter)
    deadlines = {4: 0}
    available = (4, 5, 6)
    brute = tuple(
        assignment
        for assignment in permutations(available, len(slots))
        if 4 in assignment
        and slots[assignment.index(4)].round_number <= deadlines[4]
    )

    assert matcher.count(deadlines) == len(brute) == 2
    first_count_nodes = counter.nodes
    assert matcher.count(deadlines) == len(brute)
    assert counter.nodes == first_count_nodes
    observed = set()
    for seed in range(20):
        result = matcher.sample(deadlines, random.Random(seed))
        assert result is not None
        assert result.fugitive_draws in brute
        assert result.completion_count == len(brute)
        assert result.log_q == pytest.approx(-math.log(len(brute)))
        observed.add(result.fugitive_draws)
    assert observed == set(brute)


def test_sprint_assignment_counts_every_identity_and_draw_completion() -> None:
    engine, _observation = _opening_observation(
        FugitiveAction(1), FugitiveAction(6, (2,)), seed=1
    )
    engine.draw(2)
    engine.draw(2)
    assert engine.apply_guess((1,))
    observation = engine.observation(Role.MARSHAL)
    constraints = CompiledMarshalConstraints.from_observation(observation)
    counter = _BudgetCounter(SamplingBudget(max_nodes=500_000))
    matcher = DrawDeadlineMatcher.from_constraints(constraints, counter)
    route = (0, 1, 6)
    assignment = SprintAssignmentDP(constraints, route, matcher, counter)

    used_route_deadlines = {6: constraints.creation_rounds[2]}
    brute_count = 0
    for card in range(1, 43):
        if card in {1, 6, *constraints.marshal_hand}:
            continue
        if sprint_value(card) < 2:
            continue
        if card not in (1, 2, 3, 42):
            pile = next(
                pile for pile, pile_cards in enumerate(PILE_CARDS) if card in pile_cards
            )
            if not any(
                slot.pile == pile
                and slot.round_number <= constraints.creation_rounds[2]
                for slot in constraints.draw_slots
            ):
                continue
        deadlines = dict(used_route_deadlines)
        if card not in (1, 2, 3, 42):
            deadlines[card] = constraints.creation_rounds[2]
        brute_count += matcher.count(deadlines)

    assert assignment.completion_count == brute_count
    sample = assignment.sample(random.Random(91))
    assert sample is not None
    assert len(sample.route_sprints[2]) == 1
    assert sprint_value(sample.route_sprints[2][0]) == 2
    assert sample.completion_count == brute_count
    assert sample.log_q == pytest.approx(-math.log(brute_count))


def test_category_dp_matches_two_stack_identity_bruteforce() -> None:
    _engine, observation = _opening_observation()
    constraints = CompiledMarshalConstraints.from_observation(observation)
    synthetic_observation = replace(
        observation,
        route=(
            RouteView(0, 0, 0, (), True),
            RouteView(1, None, 1, None, False),
            RouteView(2, None, 1, None, False),
        ),
    )
    synthetic = replace(constraints, observation=synthetic_observation)
    counter = _BudgetCounter(SamplingBudget(max_nodes=200_000))
    matcher = DrawDeadlineMatcher.from_constraints(synthetic, counter)
    route = (0, 2, 7)
    assignment = SprintAssignmentDP(synthetic, route, matcher, counter)

    brute_count = 0
    available = tuple(card for card in range(1, 43) if card not in route[1:])
    for first in available:
        for second in available:
            if second == first or sprint_value(second) < 2:
                continue
            deadlines = {7: synthetic.creation_rounds[2]}
            for card, route_index in ((first, 1), (second, 2)):
                if card not in (1, 2, 3, 42):
                    deadlines[card] = synthetic.creation_rounds[route_index]
            brute_count += matcher.count(deadlines)

    assert assignment.completion_count == brute_count
    sample = assignment.sample(random.Random(123))
    assert sample is not None
    assert len(sample.route_sprints[1]) == len(sample.route_sprints[2]) == 1
    assert sample.route_sprints[1] != sample.route_sprints[2]
    assert sprint_value(sample.route_sprints[2][0]) == 2
    assert sample.log_q == pytest.approx(-math.log(brute_count))


def test_sequential_proposal_normalizes_support_and_log_q_by_bruteforce() -> None:
    _engine, observation = _opening_observation()
    constraints = CompiledMarshalConstraints.from_observation(observation)
    synthetic_observation = replace(
        observation,
        route=(
            observation.route[0],
            replace(observation.route[1], sprint_count=1),
            observation.route[2],
        ),
    )
    synthetic = replace(constraints, observation=synthetic_observation)
    counter = _BudgetCounter(SamplingBudget(max_nodes=50_000))
    matcher = DrawDeadlineMatcher.from_constraints(synthetic, counter)
    proposal = SequentialSprintAssignmentProposal(
        synthetic, (0, 4, 5), matcher, counter
    )

    branches = proposal._allocation_branches(
        0, proposal._initial_group_counts, proposal._fixed_demands
    )
    branch_by_allocation = {branch.allocation: branch for branch in branches}
    total_weight = sum(branch.proposal_weight for branch in branches)
    support_counts: Counter[tuple[int, ...]] = Counter()
    probability_mass = 0.0
    deadline = synthetic.creation_rounds[1]

    for category, cards in enumerate(proposal._initial_group_cards):
        for card in cards:
            allocation = [0] * len(proposal._initial_group_counts)
            allocation[category] = 1
            allocation_key = tuple(allocation)
            exact_deadlines = dict(proposal._fixed_deadlines)
            if card in CARD_TO_PILE:
                exact_deadlines[card] = deadline
            if not matcher.count(exact_deadlines):
                assert allocation_key not in branch_by_allocation
                continue
            branch = branch_by_allocation[allocation_key]
            support_counts[allocation_key] += 1
            probability_mass += (
                branch.proposal_weight
                / total_weight
                / branch.identity_multiplicity
            )

    assert support_counts
    assert set(support_counts) == set(branch_by_allocation)
    assert all(
        support_counts[allocation] == branch.identity_multiplicity
        for allocation, branch in branch_by_allocation.items()
    )
    assert probability_mass == pytest.approx(1.0)

    sample = proposal.sample(random.Random(17))
    assert sample is not None
    assert sample.completion_count is None
    selected_card = sample.route_sprints[1][0]
    source = (
        _INITIAL_SOURCE
        if selected_card in INITIAL_FUGITIVE_CARDS
        else CARD_TO_PILE[selected_card]
    )
    category = _SPRINT_CATEGORY_INDEX[(source, sprint_value(selected_card))]
    allocation = [0] * len(proposal._initial_group_counts)
    allocation[category] = 1
    selected_branch = branch_by_allocation[tuple(allocation)]
    exact_deadlines = dict(proposal._fixed_deadlines)
    if selected_card in CARD_TO_PILE:
        exact_deadlines[selected_card] = deadline
    draw_count = matcher.count(exact_deadlines)
    expected_log_q = (
        math.log(selected_branch.proposal_weight)
        - math.log(total_weight)
        - math.log(selected_branch.identity_multiplicity)
        - math.log(draw_count)
    )
    assert sample.log_q == pytest.approx(expected_log_q)


def test_sequential_two_stack_support_q_and_late_rejection_are_explicit() -> None:
    _engine, observation = _opening_observation()
    constraints = CompiledMarshalConstraints.from_observation(observation)
    route_view = (
        RouteView(0, 0, 0, (), True),
        RouteView(1, None, 1, None, False),
        RouteView(2, None, 1, None, False),
    )
    synthetic = replace(
        constraints,
        observation=replace(observation, route=route_view),
        draw_slots=(DrawSlot(0, 0, 0), DrawSlot(1, 0, 0)),
        creation_rounds=(0, 1, 1),
        marshal_hand=tuple(card for card in range(4, 42) if card not in (5, 7)),
    )
    counter = _BudgetCounter(SamplingBudget(max_nodes=20_000))
    matcher = DrawDeadlineMatcher(
        synthetic.draw_slots,
        synthetic.marshal_hand,
        pile_cards=((5, 7), (), ()),
        budget=counter,
    )
    proposal = SequentialSprintAssignmentProposal(
        synthetic, (0, 2, 7), matcher, counter
    )

    # The first stack can use 1, 3, 5, or 42. The second move needs +2,
    # therefore choosing 42 first is the sole late dead end. Every accepted
    # Sprint identity history has two equally likely draw orders.
    expected_support = {
        ((first,), (42,), draws)
        for first in (1, 3, 5)
        for draws in ((5, 7), (7, 5))
    }
    observed_support = set()
    rejected = 0
    for seed in range(2_000):
        sample = proposal.sample(random.Random(seed))
        if sample is None:
            rejected += 1
            continue
        key = (
            sample.route_sprints[1],
            sample.route_sprints[2],
            sample.draw_assignment.fugitive_draws,
        )
        observed_support.add(key)
        assert key in expected_support
        assert sample.log_q == pytest.approx(-math.log(8))

    raw_accepted_mass = len(expected_support) * (1 / 8)
    raw_rejection_mass = 1 / 4
    assert raw_accepted_mass + raw_rejection_mass == pytest.approx(1.0)
    assert observed_support == expected_support
    assert rejected > 0

    # The target is uniform, so inverse raw-q weights recover equal mass over
    # the six accepted complete worlds; the common acceptance factor cancels.
    inverse_q = [1 / (1 / 8) for _world in expected_support]
    total_inverse_q = sum(inverse_q)
    assert [weight / total_inverse_q for weight in inverse_q] == pytest.approx(
        [1 / 6] * 6
    )


def test_sprint_categories_represent_more_than_512_identity_sets() -> None:
    _engine, observation = _opening_observation()
    constraints = CompiledMarshalConstraints.from_observation(observation)
    synthetic_observation = replace(
        observation,
        route=(
            RouteView(0, 0, 0, (), True),
            RouteView(1, None, 2, None, False),
        ),
    )
    synthetic = replace(
        constraints,
        observation=synthetic_observation,
        draw_slots=(
            DrawSlot(0, 0, 0),
            DrawSlot(1, 1, 0),
            DrawSlot(2, 2, 0),
        ),
        creation_rounds=(0, 1),
        marshal_hand=(),
    )
    counter = _BudgetCounter(SamplingBudget(max_nodes=2_000))
    matcher = DrawDeadlineMatcher.from_constraints(synthetic, counter)
    assignment = SprintAssignmentDP(synthetic, (0, 4), matcher, counter)

    allocations = assignment._category_allocations(
        0, assignment._initial_group_counts
    )
    identity_sets = sum(
        assignment._branch_multiplicity(
            assignment._initial_group_counts, allocation
        )
        for allocation in allocations
    )
    assert identity_sets == math.comb(41, 2)
    assert identity_sets > 512
    assert len(allocations) < 512


def test_constructive_worlds_match_observation_cards_and_deadlines() -> None:
    _engine, observation = _opening_observation()
    constraints = CompiledMarshalConstraints.from_observation(observation)
    batch = ConstructiveWorldSampler().sample_batch(
        observation,
        particle_count=64,
        seed=801,
        budget=SamplingBudget(max_nodes=100_000),
    )

    assert not batch.report.degraded
    assert batch.report.produced == 64
    assert len(batch.normalized_weights) == 64
    assert sum(batch.normalized_weights) == pytest.approx(1.0)
    revealed = _revealed_numbers(observation)
    for world in batch.worlds:
        assert world.all_cards_are_unique
        assert _particle_matches_observation(
            world.to_particle(), observation, revealed_numbers=revealed
        )
        hidden = {
            hideout
            for index, hideout in enumerate(world.route_hideouts)
            if index and not observation.route[index].revealed and hideout != 42
        }
        assert world.hidden_mask == sum(1 << card for card in hidden)

        draw_round = {
            card: constraints.fugitive_records[index].round_number
            for index, card in enumerate(world.fugitive_draws)
        }
        for index in range(1, len(world.route_hideouts)):
            for card in (
                world.route_hideouts[index],
                *world.route_sprints[index],
            ):
                if card in draw_round:
                    assert draw_round[card] <= constraints.creation_rounds[index]

        changed_scores = replace(
            world,
            log_q=world.log_q - 3.0,
            log_target=world.log_target + 4.0,
        )
        assert changed_scores.world_key == world.world_key


def test_constructive_worlds_handle_an_unknown_sprint_stack() -> None:
    engine, _observation = _opening_observation(
        FugitiveAction(1), FugitiveAction(6, (2,)), seed=1
    )
    engine.draw(2)
    engine.draw(2)
    assert engine.apply_guess((1,))
    observation = engine.observation(Role.MARSHAL)

    batch = ConstructiveWorldSampler().sample_batch(
        observation,
        particle_count=16,
        seed=12,
        budget=SamplingBudget(max_nodes=100_000),
    )

    assert not batch.report.degraded
    revealed = _revealed_numbers(observation)
    assert all(
        len(world.route_sprints[2]) == 1
        and _particle_matches_observation(
            world.to_particle(), observation, revealed_numbers=revealed
        )
        for world in batch.worlds
    )


def test_constructive_sampling_is_reproducible_and_noninterfering() -> None:
    first_engine, first_observation = _opening_observation(
        FugitiveAction(1), FugitiveAction(2), seed=3
    )
    second_engine, second_observation = _opening_observation(
        FugitiveAction(2), FugitiveAction(3), seed=31
    )
    assert first_engine.observation(Role.FUGITIVE) != second_engine.observation(
        Role.FUGITIVE
    )
    assert first_observation == second_observation

    sampler = ConstructiveWorldSampler()
    first = sampler.sample_batch(first_observation, particle_count=24, seed=991)
    repeated = sampler.sample_batch(first_observation, particle_count=24, seed=991)
    indistinguishable = sampler.sample_batch(
        second_observation, particle_count=24, seed=991
    )

    assert first == repeated == indistinguishable


def test_sequential_worlds_are_reproducible_and_observation_consistent() -> None:
    engine, _observation = _opening_observation(
        FugitiveAction(1), FugitiveAction(3, (2,)), seed=13
    )
    engine.draw(2)
    engine.draw(2)
    assert engine.apply_guess((1,))
    observation = engine.observation(Role.MARSHAL)
    sampler = ConstructiveWorldSampler(sprint_backend="sequential")

    first = sampler.sample_batch(
        observation,
        particle_count=32,
        seed=404,
        budget=SamplingBudget(max_nodes=100_000, max_proposals=1_024),
    )
    repeated = sampler.sample_batch(
        observation,
        particle_count=32,
        seed=404,
        budget=SamplingBudget(max_nodes=100_000, max_proposals=1_024),
    )

    assert first == repeated
    assert first.report.produced == 32
    assert not first.report.degraded
    assert first.report.importance_valid
    assert sum(first.normalized_weights) == pytest.approx(1.0)
    revealed = _revealed_numbers(observation)
    assert all(
        world.all_cards_are_unique
        and math.isfinite(world.log_q)
        and _particle_matches_observation(
            world.to_particle(), observation, revealed_numbers=revealed
        )
        for world in first.worlds
    )


def test_compiler_rejects_malformed_marshal_route_visibility() -> None:
    _engine, observation = _opening_observation()
    start, first, second = observation.route
    malformed_routes = (
        (replace(start, revealed=False, sprint_cards=None), first, second),
        (start, replace(first, revealed=True, hideout=None, sprint_cards=()), second),
        (start, replace(first, hideout=1), second),
        (start, replace(first, sprint_cards=()), second),
        (
            start,
            replace(first, hideout=42, sprint_count=1, sprint_cards=(2,)),
            second,
        ),
    )

    for route in malformed_routes:
        with pytest.raises(ValueError):
            CompiledMarshalConstraints.from_observation(
                replace(observation, route=route)
            )


def test_node_budget_returns_an_explicit_partial_degraded_batch() -> None:
    _engine, observation = _opening_observation()
    batch = ConstructiveWorldSampler().sample_batch(
        observation,
        particle_count=8,
        seed=5,
        budget=SamplingBudget(max_nodes=1, max_proposals=8),
    )

    assert batch.worlds == ()
    assert batch.report.produced == 0
    assert batch.report.degraded
    assert not batch.report.importance_valid
    assert batch.report.termination_reason == "node_budget"
    assert batch.report.exhausted_stage == "route_completion"
    assert batch.report.search_nodes == 1


def test_unbounded_node_budget_still_counts_search_work() -> None:
    _engine, observation = _opening_observation()
    batch = ConstructiveWorldSampler(sprint_backend="sequential").sample_batch(
        observation,
        particle_count=8,
        seed=5,
        budget=SamplingBudget(max_nodes=None, max_proposals=256),
    )

    assert batch.report.produced == 8
    assert batch.report.search_nodes > 0
    assert batch.report.importance_valid
    assert batch.report.termination_reason is None


def test_nonempty_budget_partial_cannot_expose_importance_weights() -> None:
    _engine, observation = _opening_observation()
    sampler = ConstructiveWorldSampler()
    first_two = sampler.sample_batch(
        observation,
        particle_count=2,
        seed=801,
        budget=SamplingBudget(max_nodes=100_000, max_proposals=20),
    )
    partial = sampler.sample_batch(
        observation,
        particle_count=3,
        seed=801,
        budget=SamplingBudget(
            max_nodes=first_two.report.search_nodes,
            max_proposals=20,
        ),
    )

    assert partial.report.produced == 2
    assert partial.report.degraded
    assert partial.report.termination_reason == "node_budget"
    assert not partial.report.importance_valid
    with pytest.raises(RuntimeError, match="importance weights are invalid"):
        _ = partial.normalized_weights


def _review_round_two_marshal_draw_observation():
    engine = GameEngine(seed=0)
    fugitive = HierarchicalRandomFugitiveAgent(100)
    marshal = HierarchicalRandomMarshalAgent(200)
    while True:
        phase = engine.phase
        role = (
            Role.FUGITIVE
            if phase
            in (Phase.FUGITIVE_OPENING, Phase.FUGITIVE_DRAW, Phase.FUGITIVE_ACTION)
            else Role.MARSHAL
        )
        observation = engine.observation(role)
        if phase is Phase.MARSHAL_DRAW and observation.round_number == 2:
            return observation
        if phase in (Phase.FUGITIVE_OPENING, Phase.FUGITIVE_ACTION):
            engine.apply_fugitive_action(
                fugitive.choose_fugitive_action(observation)
            )
        elif phase is Phase.FUGITIVE_DRAW:
            engine.draw(fugitive.choose_draw_pile(observation))
        elif phase is Phase.MARSHAL_DRAW:
            engine.draw(marshal.choose_draw_pile(observation))
        else:
            engine.apply_guess(marshal.choose_guess(observation))


def test_review_three_by_three_sprint_fixture_fits_two_million_nodes() -> None:
    observation = _review_round_two_marshal_draw_observation()
    assert tuple(slot.sprint_count for slot in observation.route[1:]) == (0, 3, 3)

    started = time.perf_counter()
    batch = ConstructiveWorldSampler().sample_batch(
        observation,
        particle_count=1,
        seed=999,
        budget=SamplingBudget(max_nodes=2_000_000, max_proposals=128),
    )
    elapsed = time.perf_counter() - started

    assert batch.report.produced == 1
    assert not batch.report.degraded
    assert batch.report.importance_valid
    assert batch.report.search_nodes < 2_000_000
    assert elapsed < 5.0
    assert _particle_matches_observation(
        batch.worlds[0].to_particle(),
        observation,
        revealed_numbers=_revealed_numbers(observation),
    )


def test_sequential_high_sprint_batch_avoids_exact_dp_long_tail() -> None:
    observation = _review_round_two_marshal_draw_observation()
    started = time.perf_counter()
    batch = ConstructiveWorldSampler(sprint_backend="sequential").sample_batch(
        observation,
        particle_count=24,
        seed=999,
        budget=SamplingBudget(max_nodes=100_000, max_proposals=768),
    )
    elapsed = time.perf_counter() - started

    assert batch.report.produced == 24
    assert not batch.report.degraded
    assert batch.report.importance_valid
    assert batch.report.search_nodes < 100_000
    assert elapsed < 2.0
    weights = batch.normalized_weights
    effective_sample_size = 1.0 / sum(weight**2 for weight in weights)
    assert 1.0 <= effective_sample_size <= 24.0


def test_constructive_sampler_validates_sprint_backend() -> None:
    with pytest.raises(ValueError, match="sprint_backend"):
        ConstructiveWorldSampler(sprint_backend="unknown")


def test_constraint_uniform_target_is_explicitly_policy_agnostic() -> None:
    target = ConstraintUniformTarget()
    assert target.name == "constraint-uniform-complete-worlds-v1"


def test_constraint_uniform_target_rejects_worlds_outside_public_support() -> None:
    _engine, observation = _opening_observation()
    constraints = CompiledMarshalConstraints.from_observation(observation)
    batch = ConstructiveWorldSampler(sprint_backend="sequential").sample_batch(
        observation,
        particle_count=1,
        seed=41,
        budget=SamplingBudget(max_nodes=100_000, max_proposals=64),
    )
    world = batch.worlds[0]
    target = ConstraintUniformTarget()

    assert target.log_density(world, constraints) == 0.0
    assert target.log_density(
        replace(world, hidden_mask=world.hidden_mask ^ (1 << 41)), constraints
    ) == -math.inf
    assert target.log_density(
        replace(world, remaining_piles=((), (), ())), constraints
    ) == -math.inf
    impossible_deadlines = replace(
        constraints,
        creation_rounds=(0, *([-1] * (len(constraints.creation_rounds) - 1))),
    )
    assert target.log_density(world, impossible_deadlines) == -math.inf


def test_target_rejects_route_that_contradicts_failed_guess_history() -> None:
    engine, _observation = _opening_observation()
    engine.draw(2)
    engine.draw(2)
    assert not engine.apply_guess((3,))
    observation = engine.observation(Role.MARSHAL)
    constraints = CompiledMarshalConstraints.from_observation(observation)
    batch = ConstructiveWorldSampler(sprint_backend="sequential").sample_batch(
        observation,
        particle_count=1,
        seed=31,
        budget=SamplingBudget(max_nodes=100_000, max_proposals=64),
    )
    world = batch.worlds[0]
    assert 3 in world.fugitive_hand
    old_tail = world.route_hideouts[2]
    assert old_tail != 3

    contradictory = replace(
        world,
        fugitive_hand=tuple(
            sorted(old_tail if card == 3 else card for card in world.fugitive_hand)
        ),
        route_hideouts=(*world.route_hideouts[:2], 3),
        hidden_mask=(world.hidden_mask & ~(1 << old_tail)) | (1 << 3),
    )
    assert contradictory.all_cards_are_unique
    assert ConstraintUniformTarget().log_density(
        contradictory, constraints
    ) == -math.inf


def test_target_zero_proposals_are_rejected_before_weight_normalization() -> None:
    class RejectAllTarget(ConstraintUniformTarget):
        def log_density(self, world, constraints) -> float:
            return -math.inf

    _engine, observation = _opening_observation()
    batch = ConstructiveWorldSampler(
        target=RejectAllTarget(), sprint_backend="sequential"
    ).sample_batch(
        observation,
        particle_count=2,
        seed=9,
        budget=SamplingBudget(max_nodes=100_000, max_proposals=5),
    )

    assert batch.worlds == ()
    assert batch.report.rejected_targets == 5
    assert batch.report.termination_reason == "max_proposals"
    assert batch.normalized_weights == ()

    valid = ConstructiveWorldSampler(sprint_backend="sequential").sample_batch(
        observation,
        particle_count=1,
        seed=9,
        budget=SamplingBudget(max_nodes=100_000, max_proposals=32),
    )
    all_infinite = replace(
        valid,
        worlds=(replace(valid.worlds[0], log_target=-math.inf),),
    )
    with pytest.raises(RuntimeError, match="must all be finite"):
        _ = all_infinite.normalized_weights


@pytest.mark.parametrize("particle_count", (True, 0, -1, 1.5, "8"))
def test_constructive_particle_count_is_strict(particle_count: object) -> None:
    _engine, observation = _opening_observation()
    with pytest.raises(ValueError, match="positive integer"):
        ConstructiveWorldSampler().sample_batch(
            observation,
            particle_count=particle_count,  # type: ignore[arg-type]
        )
