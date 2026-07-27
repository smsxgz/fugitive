from __future__ import annotations

from dataclasses import replace

import pytest

from fugitive.agents.continuation_count_fugitive import (
    CATCH_RISK_FEATURE_ID,
    CONCEALMENT_FEATURE_ID,
    CONTINUATION_COUNT_FUGITIVE_ALGORITHM_ID,
    CONTINUATION_MASS_ID,
    CONTINUATION_TRANSFORM_ID,
    ContinuationCountFugitiveAgent,
    candidate_relative_log1p_features,
)
from fugitive.agents.hierarchical_random import HierarchicalRandomFugitiveAgent
from fugitive.engine import GameEngine
from fugitive.model import FugitiveAction, Observation, Phase, PlayView, Role
from fugitive.rules import is_legal_fugitive_action


def _normal_observation(*, reveal_first: bool = False) -> Observation:
    engine = GameEngine(seed=13)
    engine.apply_fugitive_action(FugitiveAction(1))
    engine.apply_fugitive_action(FugitiveAction(2))
    engine.draw(0)
    engine.draw(1)
    engine.apply_guess((1 if reveal_first else 3,))
    engine.draw(0)
    return engine.observation(Role.FUGITIVE)


def test_has_precise_algorithm_metadata() -> None:
    agent = ContinuationCountFugitiveAgent(1, continuation_depth=1)

    assert agent.algorithm_id == CONTINUATION_COUNT_FUGITIVE_ALGORITHM_ID
    assert "action-tree-node-mass" in CONTINUATION_MASS_ID
    assert "candidate-relative-log1p" in CONTINUATION_TRANSFORM_ID
    assert "marginal-entropy" in CONCEALMENT_FEATURE_ID
    assert "single-guess" in CATCH_RISK_FEATURE_ID


def test_candidate_relative_log1p_does_not_saturate_at_a_fixed_count() -> None:
    alone = candidate_relative_log1p_features({"small": 256})
    together = candidate_relative_log1p_features(
        {"small": 256, "large": 1_024}
    )

    assert alone == {"small": 1.0}
    assert 0.0 < together["small"] < 1.0
    assert together["large"] == 1.0


@pytest.mark.parametrize("bad_count", (-1, 1.5, True))
def test_candidate_relative_log1p_rejects_invalid_counts(bad_count) -> None:
    with pytest.raises(ValueError):
        candidate_relative_log1p_features({"candidate": bad_count})


def test_continuation_memo_is_exact_agent_local_and_canonical() -> None:
    hand = (4, 5, 6, 8, 42)
    agent = ContinuationCountFugitiveAgent(2, continuation_depth=2)

    actual = agent._continuation_count(hand, 2, 2)
    entries_after_first = agent.continuation_cache_entries
    hits_after_first = agent.continuation_cache_hits

    assert actual > 0
    assert entries_after_first > 1
    assert agent._continuation_count(tuple(reversed(hand)), 2, 2) == actual
    assert agent.continuation_cache_entries == entries_after_first
    assert agent.continuation_cache_hits == hits_after_first + 1
    assert ContinuationCountFugitiveAgent(2).continuation_cache_entries == 0


def test_distribution_keeps_the_same_bounded_legal_candidates() -> None:
    observation = _normal_observation()
    distribution = ContinuationCountFugitiveAgent(
        4,
        continuation_depth=1,
        concealment_weight=0.0,
        catch_risk_weight=0.0,
    ).fugitive_action_distribution(observation)
    hr = HierarchicalRandomFugitiveAgent(4).fugitive_action_distribution(observation)

    assert set(distribution) == set(hr)
    assert sum(distribution.values()) == pytest.approx(1.0)
    previous = observation.route[-1].hideout
    assert previous is not None
    assert all(
        is_legal_fugitive_action(
            action,
            observation.hand,
            previous,
            allow_pass=True,
        )
        for action in distribution
    )


def test_play_candidates_share_one_relative_continuation_denominator(
    monkeypatch,
) -> None:
    observation = _normal_observation()
    agent = ContinuationCountFugitiveAgent(4, continuation_depth=1)
    captured: dict[FugitiveAction, float] = {}
    pass_action = FugitiveAction(None)

    monkeypatch.setattr(
        agent,
        "_action_continuation_log",
        lambda _observation, action: float(action.hideout),
    )
    monkeypatch.setattr(
        agent,
        "_pass_continuation_log",
        lambda _observation: 84.0,
    )

    def action_score(_observation, action, feature):
        captured[action] = feature
        return feature

    def pass_score(_observation, feature):
        captured[pass_action] = feature
        return feature

    monkeypatch.setattr(agent, "_action_score_with_feature", action_score)
    monkeypatch.setattr(agent, "_pass_score_with_feature", pass_score)

    distribution = agent.fugitive_action_distribution(observation)

    assert distribution
    assert captured[pass_action] == 1.0
    for action, feature in captured.items():
        if action.hideout is not None:
            assert feature == pytest.approx(action.hideout / 84.0)


def test_shadow_preserves_public_existing_and_hypothetical_play_timeline() -> None:
    observation = _normal_observation(reveal_first=True)
    action = FugitiveAction(5, (4,))
    shadow, truth = ContinuationCountFugitiveAgent._public_shadow_after(
        observation,
        (action,),
    )

    assert len(shadow.play_history) == len(observation.play_history) + 1
    first, second, hypothetical = shadow.play_history
    assert first == PlayView(1, 0, (), 1, 1, False)
    assert second == PlayView(None, 0, None, 2, 1, False)
    assert hypothetical == PlayView(
        None,
        1,
        None,
        len(observation.route),
        observation.round_number,
        False,
    )
    assert shadow.route[-1].hideout is None
    assert shadow.route[-1].sprint_count == 1
    assert action.hideout in truth
    assert all(record.card is None for record in shadow.draw_history)


def test_empty_hypothesis_appends_public_pass_without_changing_route() -> None:
    observation = _normal_observation()
    shadow, truth = ContinuationCountFugitiveAgent._public_shadow_after(
        observation,
        (),
    )

    assert len(shadow.route) == len(observation.route)
    assert len(shadow.play_history) == len(observation.play_history) + 1
    assert shadow.play_history[-1] == PlayView(
        None,
        0,
        (),
        None,
        observation.round_number,
        True,
    )
    assert truth == (1, 2)


def test_existing_pass_is_preserved_before_a_hypothetical_play() -> None:
    engine = GameEngine(seed=21)
    engine.apply_fugitive_action(FugitiveAction(1))
    engine.apply_fugitive_action(FugitiveAction(2))
    engine.draw(0)
    engine.draw(1)
    engine.apply_guess((3,))
    engine.draw(0)
    engine.apply_fugitive_action(FugitiveAction(None))
    observation = engine.observation(Role.FUGITIVE)

    shadow, _ = ContinuationCountFugitiveAgent._public_shadow_after(
        observation,
        (FugitiveAction(4),),
    )

    assert shadow.play_history[-2].passed is True
    assert shadow.play_history[-2].round_number == 2
    assert shadow.play_history[-1].passed is False
    assert shadow.play_history[-1].round_number == observation.round_number


def test_manhunt_shadow_uses_the_same_public_hypothetical_timeline() -> None:
    observation = _normal_observation()
    action = FugitiveAction(42, (4,))

    shadow = ContinuationCountFugitiveAgent._marshal_shadow(observation, action)

    assert shadow.phase is Phase.MANHUNT
    assert shadow.route[-1].hideout == 42
    assert shadow.route[-1].sprint_cards is None
    assert shadow.play_history[-1] == PlayView(
        42,
        1,
        None,
        len(observation.route),
        observation.round_number,
        False,
    )


def test_public_shadow_is_invariant_to_private_hidden_identities() -> None:
    observation = _normal_observation()
    observation = replace(
        observation,
        route=(
            observation.route[0],
            replace(observation.route[1], sprint_count=1, sprint_cards=(8,)),
            observation.route[2],
        ),
        play_history=(
            replace(
                observation.play_history[0],
                sprint_count=1,
                sprint_cards=(8,),
            ),
            observation.play_history[1],
        ),
    )
    changed_route = (
        observation.route[0],
        replace(observation.route[1], hideout=4, sprint_cards=(10,)),
        replace(observation.route[2], hideout=5, sprint_count=0, sprint_cards=()),
    )
    changed_history = (
        replace(observation.play_history[0], hideout=4, sprint_cards=(10,)),
        replace(observation.play_history[1], hideout=5),
    )
    changed = replace(
        observation,
        hand=tuple(reversed(observation.hand)),
        route=changed_route,
        play_history=changed_history,
    )
    action = FugitiveAction(7, (12,))

    first_shadow, first_truth = ContinuationCountFugitiveAgent._public_shadow_after(
        observation,
        (action,),
    )
    second_shadow, second_truth = ContinuationCountFugitiveAgent._public_shadow_after(
        changed,
        (action,),
    )

    assert first_shadow == second_shadow
    assert first_truth != second_truth


def test_opening_distribution_is_seed_reproducible() -> None:
    observation = GameEngine(seed=8).observation(Role.FUGITIVE)
    first = ContinuationCountFugitiveAgent(
        11,
        continuation_depth=1,
        concealment_weight=0.0,
        catch_risk_weight=0.0,
    )
    second = ContinuationCountFugitiveAgent(
        11,
        continuation_depth=1,
        concealment_weight=0.0,
        catch_risk_weight=0.0,
    )

    assert first.fugitive_action_distribution(observation) == (
        second.fugitive_action_distribution(observation)
    )
    assert first.choose_fugitive_action(observation) == (
        second.choose_fugitive_action(observation)
    )
