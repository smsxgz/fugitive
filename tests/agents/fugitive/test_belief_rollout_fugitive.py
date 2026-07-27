from __future__ import annotations

from collections import Counter
import json
import random

from fugitive.agents.fugitive.belief_rollout import (
    BeliefRolloutFugitiveAgent,
    sample_fugitive_worlds,
)
from fugitive.game.engine import GameEngine
from fugitive.game.model import FugitiveAction, Role
from fugitive.game.rules import PILE_CARDS, is_legal_fugitive_action
from fugitive.agents.marshal.inference.world_validation import CARD_TO_PILE


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
    assert worlds == sample_fugitive_worlds(
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
        "fugitive.agents.fugitive.belief_rollout.sample_fugitive_worlds",
        fail_world_sampling,
    )
    distribution = agent.fugitive_action_distribution(observation)
    diagnostics = agent.rollout_diagnostics()

    assert tuple(distribution.values()) == (1.0,)
    assert all(
        is_legal_fugitive_action(
            action,
            observation.hand,
            0,
            allow_pass=False,
        )
        for action in distribution
    )
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
