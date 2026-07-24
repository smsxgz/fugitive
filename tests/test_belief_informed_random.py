from __future__ import annotations

from dataclasses import replace

import pytest

from fugitive.agents.baseline_utils import epsilon_softmax
from fugitive.agents.belief_informed_random import (
    BeliefInformedRandomFugitiveAgent,
    BeliefInformedRandomMarshalAgent,
)
from fugitive.engine import GameEngine, play_game
from fugitive.model import FugitiveAction, Phase, Role, Winner
from fugitive.particle_belief import DEFAULT_PARTICLE_COUNT, MarshalParticleBelief
from fugitive.rules import is_legal_fugitive_action, is_legal_guess


def assert_distribution(distribution: dict[object, float]) -> None:
    assert distribution
    assert sum(distribution.values()) == pytest.approx(1.0)
    assert all(probability > 0.0 for probability in distribution.values())


def finish_opening(
    engine: GameEngine,
    first: FugitiveAction = FugitiveAction(1),
    second: FugitiveAction = FugitiveAction(2),
) -> None:
    engine.apply_fugitive_action(first)
    engine.apply_fugitive_action(second)


def test_epsilon_softmax_is_normalized_and_preserves_exploration() -> None:
    distribution = epsilon_softmax(
        {"low": -10.0, "middle": 0.0, "high": 10.0},
        epsilon=0.15,
        temperature=0.7,
    )

    assert_distribution(distribution)
    assert distribution["high"] > distribution["middle"] > distribution["low"]
    assert all(probability >= 0.05 - 1e-12 for probability in distribution.values())

    tied = epsilon_softmax({"a": 4.0, "b": 4.0, "c": 4.0})
    assert tied == pytest.approx({"a": 1 / 3, "b": 1 / 3, "c": 1 / 3})


def test_fugitive_opening_distribution_conditions_a_legal_second_play() -> None:
    engine = GameEngine(seed=19)
    agent = BeliefInformedRandomFugitiveAgent(7)
    first_observation = engine.observation(Role.FUGITIVE)
    plans = agent.opening_plan_distribution(first_observation)

    assert_distribution(plans)
    assert all(42 not in plan.first.sprint_cards for plan in plans)
    assert all(42 not in plan.second.sprint_cards for plan in plans)

    first_distribution = agent.fugitive_action_distribution(first_observation)
    assert_distribution(first_distribution)
    assert all(42 not in action.sprint_cards for action in first_distribution)
    first = agent.choose_fugitive_action(first_observation)
    engine.apply_fugitive_action(first)

    second_observation = engine.observation(Role.FUGITIVE)
    second_distribution = agent.fugitive_action_distribution(second_observation)
    assert_distribution(second_distribution)
    assert all(42 not in action.sprint_cards for action in second_distribution)
    assert all(
        is_legal_fugitive_action(
            action,
            second_observation.hand,
            first.hideout,  # type: ignore[arg-type]
            allow_pass=False,
        )
        for action in second_distribution
    )

    second = agent.choose_fugitive_action(second_observation)
    engine.apply_fugitive_action(second)
    assert engine.phase is Phase.MARSHAL_DRAW
    assert len(engine.observation(Role.FUGITIVE).route) == 3


def test_opening_randomization_is_hierarchical_not_plan_count_weighted() -> None:
    observation = GameEngine(seed=0).observation(Role.FUGITIVE)
    agent = BeliefInformedRandomFugitiveAgent(
        7,
        epsilon=1.0,
        overpay_probability=0.0,
    )
    distribution = agent.opening_plan_distribution(observation)

    hideout_mass: dict[int, float] = {}
    action_mass: dict[FugitiveAction, float] = {}
    for plan, probability in distribution.items():
        assert plan.first.hideout is not None
        hideout_mass[plan.first.hideout] = (
            hideout_mass.get(plan.first.hideout, 0.0) + probability
        )
        action_mass[plan.first] = action_mass.get(plan.first, 0.0) + probability

    assert len(hideout_mass) > 1
    assert len(set(round(mass, 12) for mass in hideout_mass.values())) == 1
    for hideout, mass in hideout_mass.items():
        conditional = [
            probability / mass
            for action, probability in action_mass.items()
            if action.hideout == hideout
        ]
        assert len(set(round(probability, 12) for probability in conditional)) == 1


def test_second_opening_has_fallback_for_any_engine_legal_first_play() -> None:
    engine = GameEngine(seed=0)
    first = FugitiveAction(1, (2, 4))
    engine.apply_fugitive_action(first)
    observation = engine.observation(Role.FUGITIVE)

    distribution = BeliefInformedRandomFugitiveAgent(
        7
    ).fugitive_action_distribution(observation)

    assert_distribution(distribution)
    assert all(
        is_legal_fugitive_action(
            action,
            observation.hand,
            1,
            allow_pass=False,
        )
        for action in distribution
    )


def test_marshal_advances_the_previous_belief_for_real_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = GameEngine(seed=13)
    finish_opening(engine)
    agent = BeliefInformedRandomMarshalAgent(5, particle_count=64)
    before = engine.observation(Role.MARSHAL)
    initial = agent.belief(before)
    assert not initial.is_empty

    engine.draw(2)
    after = engine.observation(Role.MARSHAL)

    def forbidden_resample(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("incremental update rebuilt the complete history")

    monkeypatch.setattr(
        MarshalParticleBelief,
        "from_observation",
        forbidden_resample,
    )
    advanced = agent.belief(after)

    assert not advanced.is_empty
    assert advanced.observation == after
    assert all(particle.marshal_hand == after.hand for particle in advanced.particles)


def test_fugitive_draw_distribution_ignores_hidden_marshal_cards() -> None:
    # These seeds deal the same ordered private cards to the Fugitive, but the
    # untouched deck order and the Marshal's later private cards differ.
    engines = [GameEngine(seed=566), GameEngine(seed=1080)]
    for engine in engines:
        finish_opening(engine)
        engine.draw(2)
        engine.draw(2)
        assert not engine.apply_guess((41,))

    first_observation = engines[0].observation(Role.FUGITIVE)
    second_observation = engines[1].observation(Role.FUGITIVE)
    assert first_observation == second_observation
    assert engines[0].observation(Role.MARSHAL).hand != engines[1].observation(
        Role.MARSHAL
    ).hand

    first = BeliefInformedRandomFugitiveAgent(31).draw_pile_distribution(
        first_observation
    )
    second = BeliefInformedRandomFugitiveAgent(31).draw_pile_distribution(
        second_observation
    )
    assert_distribution(first)
    assert first == second
    assert set(first) == set(first_observation.legal_draw_piles)


def test_marshal_draw_distribution_ignores_hidden_route_and_fugitive_hand() -> None:
    first_engine = GameEngine(seed=23)
    second_engine = GameEngine(seed=23)
    finish_opening(first_engine, FugitiveAction(1), FugitiveAction(2))
    finish_opening(second_engine, FugitiveAction(2), FugitiveAction(3))

    first_observation = first_engine.observation(Role.MARSHAL)
    second_observation = second_engine.observation(Role.MARSHAL)
    assert first_observation == second_observation
    assert first_engine.observation(Role.FUGITIVE) != second_engine.observation(
        Role.FUGITIVE
    )

    first = BeliefInformedRandomMarshalAgent(
        41, particle_count=64, max_guess_candidates=32
    ).draw_pile_distribution(first_observation)
    second = BeliefInformedRandomMarshalAgent(
        41, particle_count=64, max_guess_candidates=32
    ).draw_pile_distribution(second_observation)
    assert_distribution(first)
    assert first == second
    assert set(first) == set(first_observation.legal_draw_piles)


def test_marshal_uses_joint_success_and_manhunt_is_singleton_only() -> None:
    engine = GameEngine(seed=5)
    finish_opening(engine, FugitiveAction(1), FugitiveAction(3))
    engine.draw(2)
    engine.draw(2)
    assert not engine.apply_guess((1, 2))
    engine.draw(2)
    engine.apply_fugitive_action(FugitiveAction(None))
    engine.draw(2)
    observation = engine.observation(Role.MARSHAL)

    agent = BeliefInformedRandomMarshalAgent(
        53, particle_count=128, max_guess_candidates=64
    )
    belief = agent.belief(observation)
    distribution = agent.guess_distribution(observation)

    assert not belief.is_empty
    assert belief.probability_hidden(1) > 0.0
    assert belief.probability_hidden(2) > 0.0
    assert belief.joint_success((1, 2)) == 0.0
    assert (1, 2) not in distribution
    assert_distribution(distribution)
    assert all(belief.joint_success(guess) > 0.0 for guess in distribution)
    assert all(is_legal_guess(guess) for guess in distribution)
    assert any(len(guess) > 1 for guess in distribution)

    manhunt = replace(
        observation,
        phase=Phase.MANHUNT,
        legal_draw_piles=(),
    )
    manhunt_distribution = agent.guess_distribution(manhunt)
    assert_distribution(manhunt_distribution)
    assert all(len(guess) == 1 for guess in manhunt_distribution)
    assert all(is_legal_guess(guess, manhunt=True) for guess in manhunt_distribution)


def test_indistinguishable_marshal_worlds_have_identical_guess_distribution() -> None:
    first_engine = GameEngine(seed=29)
    second_engine = GameEngine(seed=29)
    finish_opening(first_engine, FugitiveAction(1), FugitiveAction(2))
    finish_opening(second_engine, FugitiveAction(2), FugitiveAction(3))
    for engine in (first_engine, second_engine):
        engine.draw(2)
        engine.draw(2)

    first_observation = first_engine.observation(Role.MARSHAL)
    second_observation = second_engine.observation(Role.MARSHAL)
    assert first_observation == second_observation
    assert first_engine.observation(Role.FUGITIVE) != second_engine.observation(
        Role.FUGITIVE
    )

    first = BeliefInformedRandomMarshalAgent(
        67, particle_count=64, max_guess_candidates=48
    ).guess_distribution(first_observation)
    second = BeliefInformedRandomMarshalAgent(
        67, particle_count=64, max_guess_candidates=48
    ).guess_distribution(second_observation)
    assert_distribution(first)
    assert first == second


def test_small_particle_bir_agents_finish_a_complete_game() -> None:
    result = play_game(
        BeliefInformedRandomFugitiveAgent(8),
        BeliefInformedRandomMarshalAgent(
            9,
            particle_count=32,
            max_guess_candidates=32,
        ),
        seed=7,
    )

    assert result.winner in (Winner.FUGITIVE, Winner.MARSHAL)
    assert result.reason
    assert result.rounds > 0


def test_non_degenerate_incremental_update_samples_history_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = MarshalParticleBelief.from_observation.__func__
    calls = 0

    def counted_from_observation(
        cls: type[MarshalParticleBelief],
        *args: object,
        **kwargs: object,
    ) -> MarshalParticleBelief:
        nonlocal calls
        calls += 1
        return original(cls, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        MarshalParticleBelief,
        "from_observation",
        classmethod(counted_from_observation),
    )
    engine = GameEngine(seed=13)
    finish_opening(engine)
    agent = BeliefInformedRandomMarshalAgent(8, particle_count=64)

    initial = agent.belief(engine.observation(Role.MARSHAL))
    assert not initial.is_empty
    engine.draw(2)
    advanced = agent.belief(engine.observation(Role.MARSHAL))

    assert not advanced.is_empty
    assert calls == 1
    assert agent.belief_diagnostics.incremental_updates == 1
    assert agent.belief_diagnostics.fresh_resamples == 0
    assert agent.belief_diagnostics.last_recovery_reason is None


def test_cached_belief_becomes_the_parent_of_a_new_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_engine = GameEngine(seed=13)
    finish_opening(first_engine)
    opening = first_engine.observation(Role.MARSHAL)

    second_engine = GameEngine(seed=13)
    finish_opening(second_engine)
    assert second_engine.observation(Role.MARSHAL) == opening

    agent = BeliefInformedRandomMarshalAgent(8, particle_count=64)
    opening_belief = agent.belief(opening)
    first_engine.draw(2)
    first_branch = first_engine.observation(Role.MARSHAL)
    agent.belief(first_branch)

    assert agent.belief(opening) is opening_belief
    assert agent._latest_belief is opening_belief
    assert agent.belief_diagnostics.unique_particle_count == (
        opening_belief.unique_particle_count
    )

    advance_parents: list[MarshalParticleBelief] = []
    original_advance = MarshalParticleBelief.advance_to

    def tracked_advance(
        self: MarshalParticleBelief,
        *args: object,
        **kwargs: object,
    ) -> MarshalParticleBelief:
        advance_parents.append(self)
        return original_advance(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(MarshalParticleBelief, "advance_to", tracked_advance)
    second_engine.draw(1)
    second_branch = second_engine.observation(Role.MARSHAL)
    assert second_branch != first_branch
    agent.belief(second_branch)

    assert advance_parents == [opening_belief]


@pytest.mark.parametrize(
    ("keyword", "value"),
    (
        ("particle_count", 1.5),
        ("particle_count", True),
        ("min_unique_particles", 0),
        ("min_unique_particles", True),
        ("min_unique_particles", DEFAULT_PARTICLE_COUNT + 1),
        ("max_guess_candidates", 0),
        ("max_guess_candidates", 129),
        ("max_guess_candidates", 3.5),
        ("terminal_bonus_scale", -1.0),
    ),
)
def test_marshal_configuration_is_strict(keyword: str, value: object) -> None:
    with pytest.raises(ValueError):
        BeliefInformedRandomMarshalAgent(**{keyword: value})  # type: ignore[arg-type]
