from __future__ import annotations

from collections import Counter
import random

import pytest

from fugitive.driver import DrawDecision, GuessDecision, step_agent
from fugitive.engine import GameEngine
from fugitive.inference.worlds import ConstructiveWorld
from fugitive.model import FugitiveAction, Observation, Phase, Role, Winner
from fugitive.particle_belief import MarshalParticle
from fugitive.rollout import (
    FugitiveWorld,
    RolloutBudget,
    RolloutEvaluator,
    determinize_fugitive_game,
    determinize_game,
)
from fugitive.rules import PILE_CARDS, legal_fugitive_actions, sprint_value


def _marshal_guess_root() -> GameEngine:
    engine = GameEngine(seed=7)
    engine.apply_fugitive_action(FugitiveAction(1))
    engine.apply_fugitive_action(FugitiveAction(2))
    engine.draw(2)
    engine.draw(2)
    assert engine.phase is Phase.MARSHAL_GUESS
    return engine


def _round_two_draw_root() -> GameEngine:
    engine = _marshal_guess_root()
    assert not engine.apply_guess((41,))
    assert engine.phase is Phase.FUGITIVE_DRAW
    return engine


def _ground_truth_world(engine: GameEngine) -> MarshalParticle:
    fugitive = engine.observation(Role.FUGITIVE)
    marshal = engine.observation(Role.MARSHAL)
    fugitive_draws = tuple(
        record.card
        for record in fugitive.draw_history
        if record.role is Role.FUGITIVE and record.card is not None
    )
    marshal_draws = tuple(
        record.card
        for record in marshal.draw_history
        if record.role is Role.MARSHAL and record.card is not None
    )
    drawn = set((*fugitive_draws, *marshal_draws))
    remaining = tuple(
        tuple(card for card in pile if card not in drawn)
        for pile in PILE_CARDS
    )
    return MarshalParticle(
        fugitive_hand=fugitive.hand,
        marshal_hand=marshal.hand,
        route_hideouts=tuple(
            slot.hideout for slot in fugitive.route if slot.hideout is not None
        ),
        route_sprints=tuple(
            slot.sprint_cards
            for slot in fugitive.route
            if slot.sprint_cards is not None
        ),
        fugitive_draws=fugitive_draws,
        remaining_piles=remaining,  # type: ignore[arg-type]
    )


def _as_constructive_world(particle: MarshalParticle) -> ConstructiveWorld:
    hidden_mask = 0
    for hideout in particle.route_hideouts[1:]:
        hidden_mask |= 1 << hideout
    return ConstructiveWorld(
        fugitive_hand=particle.fugitive_hand,
        marshal_hand=particle.marshal_hand,
        route_hideouts=particle.route_hideouts,
        route_sprints=particle.route_sprints,
        fugitive_draws=particle.fugitive_draws,
        remaining_piles=particle.remaining_piles,
        hidden_mask=hidden_mask,
        log_q=0.0,
        log_target=0.0,
    )


def _fugitive_world(particle: MarshalParticle) -> FugitiveWorld:
    return FugitiveWorld(
        marshal_hand=particle.marshal_hand,
        remaining_piles=particle.remaining_piles,
    )


class _EscapeFugitive:
    def __init__(self, observations: list[Observation] | None = None) -> None:
        self.observations = observations

    def _record(self, observation: Observation) -> None:
        if self.observations is not None:
            self.observations.append(observation)

    def choose_draw_pile(self, observation: Observation) -> int:
        self._record(observation)
        previous = observation.route[-1].hideout
        assert previous is not None
        target = 0 if previous < 15 else 1 if previous < 29 else 2
        return next(
            (pile for pile in observation.legal_draw_piles if pile >= target),
            observation.legal_draw_piles[-1],
        )

    def choose_fugitive_action(self, observation: Observation) -> FugitiveAction:
        self._record(observation)
        if observation.phase is Phase.FUGITIVE_OPENING:
            return FugitiveAction(len(observation.route))
        previous = observation.route[-1].hideout
        assert previous is not None
        required = max(0, 42 - previous - 3)
        candidates = sorted(
            (card for card in observation.hand if card != 42),
            key=lambda card: (-sprint_value(card), card),
        )
        payment: list[int] = []
        paid = 0
        for card in candidates:
            if paid >= required:
                break
            payment.append(card)
            paid += sprint_value(card)
        if 42 in observation.hand and paid >= required:
            return FugitiveAction(42, tuple(payment))

        actions = [
            action
            for action in legal_fugitive_actions(
                observation,
                max_sprint_cards=2,
                include_pass=False,
            )
            if action.hideout != 42 and 42 not in action.sprint_cards
        ]
        if not actions:
            return FugitiveAction(None)
        return max(
            actions,
            key=lambda action: (
                action.hideout,
                -len(action.sprint_cards),
                tuple(-card for card in action.sprint_cards),
            ),
        )


class _MissingMarshal:
    def __init__(self, observations: list[Observation] | None = None) -> None:
        self.observations = observations

    def _record(self, observation: Observation) -> None:
        if self.observations is not None:
            self.observations.append(observation)

    def choose_draw_pile(self, observation: Observation) -> int:
        self._record(observation)
        return observation.legal_draw_piles[0]

    def choose_guess(self, observation: Observation) -> tuple[int, ...]:
        self._record(observation)
        # A card in the Marshal hand cannot be a hidden Fugitive Hideout.
        return (observation.hand[0],)


def test_determinization_reproduces_both_role_observations_and_rng_pile_order() -> None:
    original = _round_two_draw_root()
    marshal_observation = original.observation(Role.MARSHAL)
    fugitive_observation = original.observation(Role.FUGITIVE)
    particle = _ground_truth_world(original)

    pile_rng = random.Random(91)
    expected_first_pile = list(particle.remaining_piles[0])
    pile_rng.shuffle(expected_first_pile)

    reconstructed = determinize_game(
        marshal_observation,
        particle,
        rng=random.Random(91),
    )
    assert reconstructed.observation(Role.MARSHAL) == marshal_observation
    assert reconstructed.observation(Role.FUGITIVE) == fugitive_observation
    assert reconstructed.draw(0) == expected_first_pile[-1]

    constructive = determinize_game(
        marshal_observation,
        _as_constructive_world(particle),
        rng=random.Random(91),
    )
    assert constructive.observation(Role.MARSHAL) == marshal_observation
    assert constructive.draw(0) == expected_first_pile[-1]


def test_determinization_restores_both_first_marshal_draw_subphases() -> None:
    engine = GameEngine(seed=17)
    engine.apply_fugitive_action(FugitiveAction(1))
    engine.apply_fugitive_action(FugitiveAction(2))

    for seed in (1, 2):
        observation = engine.observation(Role.MARSHAL)
        reconstructed = determinize_game(
            observation,
            _ground_truth_world(engine),
            rng=random.Random(seed),
        )
        assert reconstructed.observation(Role.MARSHAL) == observation
        assert reconstructed.phase is Phase.MARSHAL_DRAW
        engine.draw(2)

    assert engine.phase is Phase.MARSHAL_GUESS


def test_fugitive_determinization_reproduces_observation_and_samples_piles() -> None:
    original = _round_two_draw_root()
    observation = original.observation(Role.FUGITIVE)
    particle = _ground_truth_world(original)
    world = _fugitive_world(particle)
    reconstructed = determinize_fugitive_game(
        observation,
        world,
        rng=random.Random(121),
    )
    same_seed = determinize_fugitive_game(
        observation,
        world,
        rng=random.Random(121),
    )

    assert reconstructed.observation(Role.FUGITIVE) == observation
    assert reconstructed.draw(0) == same_seed.draw(0)


def test_determinization_restores_a_manhunt_root_with_public_42() -> None:
    engine = GameEngine(seed=7)
    fugitive = _EscapeFugitive()
    marshal = _MissingMarshal()
    decision = 0
    while engine.phase is not Phase.MANHUNT:
        decision += 1
        step_agent(engine, fugitive, marshal, decision=decision)

    observation = engine.observation(Role.MARSHAL)
    reconstructed = determinize_game(
        observation,
        _ground_truth_world(engine),
        rng=random.Random(3),
    )

    assert reconstructed.phase is Phase.MANHUNT
    assert reconstructed.observation(Role.MARSHAL) == observation
    assert reconstructed.observation(Role.FUGITIVE) == engine.observation(
        Role.FUGITIVE
    )
    assert observation.route[-1].hideout == 42
    assert observation.route[-1].sprint_cards is None


def test_clone_is_independent_and_preserves_draw_order_and_terminal_result() -> None:
    engine = _round_two_draw_root()
    clone = engine.clone()
    before = engine.observation(Role.FUGITIVE)

    clone_card = clone.draw(0)
    assert engine.observation(Role.FUGITIVE) == before
    assert engine.draw(0) == clone_card

    terminal = _marshal_guess_root()
    assert terminal.apply_guess((1, 2))
    terminal_clone = terminal.clone()
    assert terminal_clone.result() == terminal.result()
    assert terminal_clone.result().winner is Winner.MARSHAL


def test_rollout_runs_to_terminal_and_pairs_policy_randomness_across_actions() -> None:
    engine = _marshal_guess_root()
    fugitive_seeds: list[int] = []
    marshal_seeds: list[int] = []

    def fugitive_factory(seed: int) -> _EscapeFugitive:
        fugitive_seeds.append(seed)
        return _EscapeFugitive()

    def marshal_factory(seed: int) -> _MissingMarshal:
        marshal_seeds.append(seed)
        return _MissingMarshal()

    evaluator = RolloutEvaluator(
        fugitive_factory,
        marshal_factory,
        seed=1234,
    )
    correct = GuessDecision((1, 2))
    miss = GuessDecision((41,))
    results = evaluator.evaluate_engines(
        (engine,),
        (correct, miss),
        repetitions=2,
    )

    assert results[0].action == correct
    assert results[0].wins == results[0].samples == 2
    assert results[0].win_rate == 1.0
    assert results[1].action == miss
    assert results[1].wins == 0
    assert results[1].samples == 2
    assert results[1].win_rate == 0.0
    assert sorted(Counter(fugitive_seeds).values()) == [2, 2]
    assert sorted(Counter(marshal_seeds).values()) == [2, 2]

    reversed_results = evaluator.evaluate_engines(
        (engine,),
        (miss, correct),
        repetitions=2,
    )
    by_action = {result.action: result for result in results}
    reversed_by_action = {result.action: result for result in reversed_results}
    assert {
        action: (result.wins, result.samples)
        for action, result in by_action.items()
    } == {
        action: (result.wins, result.samples)
        for action, result in reversed_by_action.items()
    }


@pytest.mark.parametrize(
    ("budget", "offered", "evaluated", "scenarios", "used", "unused"),
    (
        (64, 8, 8, 8, 64, 0),
        (65, 8, 8, 8, 64, 1),
        (5, 3, 3, 1, 3, 2),
        (2, 8, 2, 1, 2, 0),
        (1, 8, 1, 0, 0, 1),
    ),
)
def test_rollout_budget_allocates_only_complete_paired_scenarios(
    budget: int,
    offered: int,
    evaluated: int,
    scenarios: int,
    used: int,
    unused: int,
) -> None:
    plan = RolloutBudget(budget).plan(offered)

    assert plan.evaluated_candidates == evaluated
    assert plan.paired_scenarios == scenarios
    assert plan.terminal_simulations == used
    assert plan.unused_terminal_simulations == unused
    assert plan.single_candidate_shortcut is (evaluated == 1)
    assert used <= budget
    if evaluated > 1:
        assert used % evaluated == 0


@pytest.mark.parametrize("value", (0, -1, True))
def test_rollout_budget_rejects_nonpositive_integer_limits(value) -> None:
    with pytest.raises(ValueError):
        RolloutBudget(value)


def test_detailed_rollout_reports_paired_outcomes_and_terminal_work() -> None:
    engine = _marshal_guess_root()
    evaluator = RolloutEvaluator(
        lambda _seed: _EscapeFugitive(),
        lambda _seed: _MissingMarshal(),
        seed=4321,
    )
    correct = GuessDecision((1, 2))
    miss = GuessDecision((41,))

    evaluation = evaluator.evaluate_engines_detailed(
        (engine,),
        (correct, miss),
        repetitions=2,
    )

    assert evaluation.paired_scenarios == 2
    assert evaluation.terminal_simulations == 4
    assert evaluation.simulated_decisions >= evaluation.terminal_simulations
    assert tuple(result.wins for result in evaluation.action_results) == (2, 0)
    assert len(evaluation.paired_results) == 1
    pair = evaluation.paired_results[0]
    assert pair.first_action == correct
    assert pair.second_action == miss
    assert (pair.both_win, pair.first_only_win) == (0, 2)
    assert (pair.second_only_win, pair.both_loss) == (0, 0)
    assert pair.mean_difference == 1.0
    assert pair.standard_error == 0.0


def test_world_rollout_uses_only_role_observations_for_continuation_policies() -> None:
    engine = _marshal_guess_root()
    observation = engine.observation(Role.MARSHAL)
    world = _ground_truth_world(engine)
    fugitive_observations: list[Observation] = []
    marshal_observations: list[Observation] = []

    evaluator = RolloutEvaluator(
        lambda _seed: _EscapeFugitive(fugitive_observations),
        lambda _seed: _MissingMarshal(marshal_observations),
        seed=55,
    )
    result = evaluator.evaluate_worlds(
        observation,
        (world,),
        (GuessDecision((41,)),),
    )[0]

    assert result.samples == 1
    assert result.wins == 0
    assert fugitive_observations
    assert marshal_observations
    assert all(item.role is Role.FUGITIVE for item in fugitive_observations)
    assert all(item.role is Role.MARSHAL for item in marshal_observations)
    assert all(
        record.card is None
        for item in fugitive_observations
        for record in item.draw_history
        if record.role is Role.MARSHAL
    )
    assert all(
        record.card is None
        for item in marshal_observations
        for record in item.draw_history
        if record.role is Role.FUGITIVE
    )
    assert marshal_observations[0].route[1].hideout is None
    assert all(
        not hasattr(item, "piles") and not hasattr(item, "opponent_hand")
        for item in (*fugitive_observations, *marshal_observations)
    )


def test_world_rollout_reports_world_times_repetition_sample_count() -> None:
    engine = _marshal_guess_root()
    observation = engine.observation(Role.MARSHAL)
    world = _ground_truth_world(engine)
    evaluator = RolloutEvaluator(
        lambda _seed: _EscapeFugitive(),
        lambda _seed: _MissingMarshal(),
        seed=8,
    )

    result = evaluator.evaluate_worlds(
        observation,
        (world, world),
        (GuessDecision((1, 2)),),
        repetitions=3,
    )[0]

    assert result.samples == 6
    assert result.wins == 6


def test_fugitive_world_rollout_is_role_generic_and_information_safe() -> None:
    engine = _round_two_draw_root()
    observation = engine.observation(Role.FUGITIVE)
    world = _fugitive_world(_ground_truth_world(engine))
    fugitive_observations: list[Observation] = []
    marshal_observations: list[Observation] = []
    evaluator = RolloutEvaluator(
        lambda _seed: _EscapeFugitive(fugitive_observations),
        lambda _seed: _MissingMarshal(marshal_observations),
        seed=89,
    )

    results = evaluator.evaluate_fugitive_worlds(
        observation,
        (world,),
        (DrawDecision(0), DrawDecision(1)),
        repetitions=2,
    )

    assert tuple(result.samples for result in results) == (2, 2)
    assert tuple(result.wins for result in results) == (2, 2)
    assert all(item.role is Role.FUGITIVE for item in fugitive_observations)
    assert all(item.role is Role.MARSHAL for item in marshal_observations)
    assert all(
        record.card is None
        for item in marshal_observations
        for record in item.draw_history
        if record.role is Role.FUGITIVE
    )
    assert marshal_observations[0].route[1].hideout is None


def test_fugitive_world_rollout_accepts_a_hideout_phase_macro_action() -> None:
    engine = _round_two_draw_root()
    engine.draw(0)
    observation = engine.observation(Role.FUGITIVE)
    world = _fugitive_world(_ground_truth_world(engine))
    evaluator = RolloutEvaluator(
        lambda _seed: _EscapeFugitive(),
        lambda _seed: _MissingMarshal(),
        seed=144,
    )

    result = evaluator.evaluate_fugitive_worlds(
        observation,
        (world,),
        (FugitiveAction(None),),
    )[0]

    assert result.samples == 1
    assert result.wins == 1
