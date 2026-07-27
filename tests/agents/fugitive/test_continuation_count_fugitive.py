from __future__ import annotations

from dataclasses import replace

import pytest

from fugitive.agents.fugitive.continuation_count import (
    ContinuationCountFugitiveAgent,
    candidate_relative_log1p_features,
)
from fugitive.game.engine import GameEngine
from fugitive.game.model import FugitiveAction, Observation, Phase, PlayView, Role


def _normal_observation(*, reveal_first: bool = False) -> Observation:
    engine = GameEngine(seed=13)
    engine.apply_fugitive_action(FugitiveAction(1))
    engine.apply_fugitive_action(FugitiveAction(2))
    engine.draw(0)
    engine.draw(1)
    engine.apply_guess((1 if reveal_first else 3,))
    engine.draw(0)
    return engine.observation(Role.FUGITIVE)


def test_candidate_relative_log1p_does_not_saturate_at_a_fixed_count() -> None:
    alone = candidate_relative_log1p_features({"small": 256})
    together = candidate_relative_log1p_features(
        {"small": 256, "large": 1_024}
    )

    assert alone == {"small": 1.0}
    assert 0.0 < together["small"] < 1.0
    assert together["large"] == 1.0


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
