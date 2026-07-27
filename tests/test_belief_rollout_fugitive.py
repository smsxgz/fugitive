from __future__ import annotations

from collections import Counter
import json
import random

import pytest

from fugitive.agents.belief_rollout_fugitive import (
    BeliefRolloutFugitiveAgent,
    sample_fugitive_worlds,
)
from fugitive.engine import GameEngine
from fugitive.model import FugitiveAction, Role
from fugitive.rules import PILE_CARDS, is_legal_fugitive_action
from fugitive.world_validation import CARD_TO_PILE


def _normal_observation():
    engine = GameEngine(seed=13)
    engine.apply_fugitive_action(FugitiveAction(1))
    engine.apply_fugitive_action(FugitiveAction(2))
    engine.draw(0)
    engine.draw(1)
    engine.apply_guess((3,))
    engine.draw(0)
    return engine.observation(Role.FUGITIVE)


def test_world_sampler_uses_public_draw_counts_and_partitions_every_pile() -> None:
    observation = _normal_observation()
    worlds = sample_fugitive_worlds(
        observation,
        count=12,
        rng=random.Random(91),
    )
    expected_draw_counts = Counter(
        record.pile
        for record in observation.draw_history
        if record.role is Role.MARSHAL
    )
    known = set(observation.hand)
    for slot in observation.route:
        assert slot.hideout is not None and slot.sprint_cards is not None
        known.add(slot.hideout)
        known.update(slot.sprint_cards)

    for world in worlds:
        actual_draw_counts = Counter(CARD_TO_PILE[card] for card in world.marshal_hand)
        assert actual_draw_counts == expected_draw_counts
        for pile, original in enumerate(PILE_CARDS):
            owned = {card for card in known if CARD_TO_PILE.get(card) == pile}
            marshal = {card for card in world.marshal_hand if CARD_TO_PILE[card] == pile}
            assert owned | marshal | set(world.remaining_piles[pile]) == set(original)
            assert not (owned & marshal)
            assert not (owned & set(world.remaining_piles[pile]))
            assert not (marshal & set(world.remaining_piles[pile]))


def test_world_sampler_is_reproducible_and_depends_only_on_observation() -> None:
    observation = _normal_observation()
    first = sample_fugitive_worlds(observation, count=8, rng=random.Random(7))
    second = sample_fugitive_worlds(observation, count=8, rng=random.Random(7))
    assert first == second


def test_tiny_real_rollout_returns_a_legal_normalized_action_distribution() -> None:
    observation = GameEngine(seed=3).observation(Role.FUGITIVE)
    agent = BeliefRolloutFugitiveAgent(
        5,
        rollout_candidate_count=1,
        max_terminal_simulations=1,
        manhunt_rollouts=1,
        continuation_depth=1,
    )
    distribution = agent.fugitive_action_distribution(observation)

    assert len(distribution) == 1
    assert sum(distribution.values()) == pytest.approx(1.0)
    assert all(
        is_legal_fugitive_action(
            action,
            observation.hand,
            0,
            allow_pass=False,
        )
        for action in distribution
    )


def test_single_candidate_shortcut_never_samples_a_world(monkeypatch) -> None:
    observation = GameEngine(seed=3).observation(Role.FUGITIVE)
    agent = BeliefRolloutFugitiveAgent(
        5,
        rollout_candidate_count=1,
        manhunt_rollouts=1,
        continuation_depth=1,
    )

    def fail_world_sampling(*_args, **_kwargs):
        raise AssertionError("single-candidate decisions must not sample worlds")

    monkeypatch.setattr(
        "fugitive.agents.belief_rollout_fugitive.sample_fugitive_worlds",
        fail_world_sampling,
    )
    distribution = agent.fugitive_action_distribution(observation)
    diagnostics = agent.rollout_diagnostics()

    assert tuple(distribution.values()) == (1.0,)
    assert diagnostics is not None
    assert diagnostics.plan.single_candidate_shortcut
    assert diagnostics.plan.terminal_simulations == 0
    assert diagnostics.simulated_decisions == 0
    assert diagnostics.actions[0].samples == 0
    json.dumps(diagnostics.to_dict(), allow_nan=False)


def test_budgeted_fugitive_rollout_obeys_total_terminal_game_cap() -> None:
    observation = GameEngine(seed=4).observation(Role.FUGITIVE)
    agent = BeliefRolloutFugitiveAgent(
        6,
        rollout_candidate_count=3,
        max_terminal_simulations=5,
        manhunt_rollouts=1,
        continuation_depth=1,
    )

    distribution = agent.fugitive_action_distribution(observation)
    diagnostics = agent.rollout_diagnostics()

    assert diagnostics is not None
    assert diagnostics.plan.offered_candidates == 3
    assert diagnostics.plan.evaluated_candidates == 3
    assert diagnostics.plan.paired_scenarios == 1
    assert diagnostics.plan.terminal_simulations == 3
    assert diagnostics.plan.unused_terminal_simulations == 2
    assert len(distribution) == 3
    assert len(diagnostics.pairs) == 3
    assert diagnostics.rollout_model_id.endswith("v2")


def test_agent_seed_reproduces_full_rollout_distribution() -> None:
    observation = GameEngine(seed=9).observation(Role.FUGITIVE)
    parameters = {
        "rollout_candidate_count": 1,
        "max_terminal_simulations": 1,
        "manhunt_rollouts": 1,
        "continuation_depth": 1,
    }
    first = BeliefRolloutFugitiveAgent(19, **parameters)
    second = BeliefRolloutFugitiveAgent(19, **parameters)
    assert first.fugitive_action_distribution(observation) == (
        second.fugitive_action_distribution(observation)
    )


@pytest.mark.parametrize(
    "kwargs",
    (
        {"rollout_candidate_count": -1},
        {"max_terminal_simulations": 0},
        {"rollout_epsilon": 1.1},
        {"rollout_temperature": 0.0},
        {"opponent_policies": ()},
    ),
)
def test_invalid_rollout_parameters_are_rejected(kwargs) -> None:
    with pytest.raises(ValueError):
        BeliefRolloutFugitiveAgent(**kwargs)


@pytest.mark.parametrize("value", (0, -1, True))
def test_invalid_terminal_simulation_budget_is_rejected(value) -> None:
    with pytest.raises(ValueError):
        BeliefRolloutFugitiveAgent(max_terminal_simulations=value)


def test_fugitive_rollout_exposes_its_budgeted_identity() -> None:
    agent = BeliefRolloutFugitiveAgent(1)

    assert agent.algorithm_id.endswith("v2")
    assert agent.rollout_model_id.endswith("v2")
    assert agent.max_terminal_simulations > 0
