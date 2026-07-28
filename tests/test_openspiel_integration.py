from __future__ import annotations

import json

import pytest

pyspiel = pytest.importorskip("pyspiel")

from examples.train_outcome_sampling import sample_policy_episode


def test_registered_game_type_and_serialization_round_trip() -> None:
    game = pyspiel.load_game("fugitive", {"max_rounds": 3})
    game_type = game.get_type()

    assert game.num_players() == 2
    assert game.num_distinct_actions() == 47
    assert game.max_game_length() == 400
    assert game_type.dynamics == pyspiel.GameType.Dynamics.SEQUENTIAL
    assert game_type.chance_mode == pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC
    assert game_type.information == pyspiel.GameType.Information.IMPERFECT_INFORMATION

    state = game.new_initial_state()
    for card in (4, 5, 6, 15, 16):
        state.apply_action(card)
    serialized = pyspiel.serialize_game_and_state(game, state)
    restored_game, restored_state = pyspiel.deserialize_game_and_state(serialized)

    assert str(restored_game) == str(game)
    assert restored_state.history() == state.history()
    assert restored_state.observation_string(0) == state.observation_string(0)
    assert json.loads(restored_state.information_state_string(1))["schema"] == (
        "fugitive.information_state"
    )


def test_max_rounds_rejects_values_that_would_overflow_game_length() -> None:
    with pytest.raises(pyspiel.SpielError):
        pyspiel.load_game("fugitive", {"max_rounds": 21_474_836})


def test_outcome_sampling_mccfr_runs_on_fugitive() -> None:
    game = pyspiel.load_game("fugitive", {"max_rounds": 1})
    solver = pyspiel.OutcomeSamplingMCCFRSolver(game, seed=17)

    solver.run_iteration()
    returns = sample_policy_episode(game, solver.average_policy(), seed=17)

    assert len(returns) == 2
    assert returns[0] + returns[1] == pytest.approx(0.0)
