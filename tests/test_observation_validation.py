from dataclasses import replace

import pytest

from fugitive.belief import PathBelief
from fugitive.engine import GameEngine
from fugitive.inference.constraints import CompiledMarshalConstraints
from fugitive.inference.constructive_sampler import ConstructiveWorldSampler
from fugitive.model import (
    DrawRecord,
    FugitiveAction,
    GuessRecord,
    Observation,
    Phase,
    PlayView,
    Role,
    RouteView,
)
from fugitive.observation_validation import (
    ObservationContractError,
    is_valid_marshal_observation_contract,
    validate_marshal_observation_contract,
)


SETUP_DRAWS = (
    DrawRecord(Role.FUGITIVE, 0, None, 0),
    DrawRecord(Role.FUGITIVE, 0, None, 0),
    DrawRecord(Role.FUGITIVE, 0, None, 0),
    DrawRecord(Role.FUGITIVE, 1, None, 0),
    DrawRecord(Role.FUGITIVE, 1, None, 0),
)


def _assert_contract(engine: GameEngine) -> None:
    observation = engine.observation(Role.MARSHAL)
    validate_marshal_observation_contract(observation)
    assert is_valid_marshal_observation_contract(observation)


def test_contract_accepts_engine_observations_across_a_normal_round() -> None:
    engine = GameEngine(seed=7)
    _assert_contract(engine)
    engine.apply_fugitive_action(FugitiveAction(1))
    _assert_contract(engine)
    engine.apply_fugitive_action(FugitiveAction(2))
    _assert_contract(engine)

    engine.draw(engine.observation(Role.MARSHAL).legal_draw_piles[0])
    _assert_contract(engine)
    engine.draw(engine.observation(Role.MARSHAL).legal_draw_piles[0])
    _assert_contract(engine)
    assert not engine.apply_guess((41,))
    _assert_contract(engine)

    engine.draw(engine.observation(Role.FUGITIVE).legal_draw_piles[0])
    _assert_contract(engine)
    engine.apply_fugitive_action(FugitiveAction(None))
    _assert_contract(engine)


def _resource_impossible_observation() -> Observation:
    return Observation(
        role=Role.MARSHAL,
        hand=(),
        pile_sizes=(8, 12, 13),
        route=(
            RouteView(0, 0, 0, (), True),
            RouteView(1, None, 10, None, False),
            RouteView(2, None, 10, None, False),
        ),
        guess_history=(),
        draw_history=SETUP_DRAWS,
        round_number=1,
        phase=Phase.MARSHAL_DRAW,
        legal_draw_piles=(0, 1, 2),
        play_history=(
            PlayView(None, 10, None, 1, 1, False),
            PlayView(None, 10, None, 2, 1, False),
        ),
    )


def test_card_resource_prefix_rejects_zero_support_reproduction_before_sampling() -> None:
    observation = _resource_impossible_observation()

    with pytest.raises(ObservationContractError, match="more cards than"):
        validate_marshal_observation_contract(observation)
    assert not is_valid_marshal_observation_contract(observation)
    with pytest.raises(ObservationContractError, match="more cards than"):
        CompiledMarshalConstraints.from_observation(observation)
    with pytest.raises(ObservationContractError, match="more cards than"):
        ConstructiveWorldSampler().sample_batch(
            observation,
            particle_count=1,
            seed=0,
        )


def test_contract_does_not_claim_to_prove_complete_world_support() -> None:
    engine = GameEngine(seed=7)
    engine.apply_fugitive_action(FugitiveAction(1))
    engine.apply_fugitive_action(FugitiveAction(2))
    observation = engine.observation(Role.MARSHAL)
    impossible_route = (
        observation.route[0],
        observation.route[1],
        RouteView(2, 10, 0, (), True),
    )
    play_history = (
        observation.play_history[0],
        replace(
            observation.play_history[1],
            hideout=10,
            sprint_cards=(),
        ),
    )
    structurally_coherent = replace(
        observation,
        route=impossible_route,
        play_history=play_history,
        guess_history=(GuessRecord((10,), True, 2, 1),),
    )

    validate_marshal_observation_contract(structurally_coherent)
    assert PathBelief.from_observation(structurally_coherent).total_paths == 0
