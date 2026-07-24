from __future__ import annotations

import hashlib
import random

from fugitive.engine import GameEngine
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
from fugitive.observation_protocol import (
    OBSERVATION_SCHEMA,
    OBSERVATION_SCHEMA_VERSION,
    canonical_observation_bytes,
    canonical_random_state_bytes,
    observation_to_canonical_data,
    stable_observation_seed,
)


def populated_observation() -> Observation:
    return Observation(
        role=Role.MARSHAL,
        hand=(8, 12),
        pile_sizes=(5, 9, 13),
        route=(
            RouteView(0, 0, 0, (), True),
            RouteView(1, None, 2, None, False),
        ),
        guess_history=(GuessRecord((4, 7), False, 1, 2, False),),
        draw_history=(DrawRecord(Role.FUGITIVE, 0, None, 1),),
        round_number=2,
        phase=Phase.MARSHAL_GUESS,
        legal_draw_piles=(0, 2),
        play_history=(PlayView(None, 2, None, 1, 1, False),),
    )


def test_canonical_observation_schema_has_all_explicit_public_fields() -> None:
    data = observation_to_canonical_data(populated_observation())

    assert data["schema"] == OBSERVATION_SCHEMA
    assert data["schema_version"] == OBSERVATION_SCHEMA_VERSION == 1
    assert data["observation"] == {
        "role": "marshal",
        "hand": [8, 12],
        "pile_sizes": [5, 9, 13],
        "route": [
            {
                "index": 0,
                "hideout": 0,
                "sprint_count": 0,
                "sprint_cards": [],
                "revealed": True,
            },
            {
                "index": 1,
                "hideout": None,
                "sprint_count": 2,
                "sprint_cards": None,
                "revealed": False,
            },
        ],
        "guess_history": [
            {
                "numbers": [4, 7],
                "success": False,
                "route_length": 1,
                "round_number": 2,
                "manhunt": False,
            }
        ],
        "draw_history": [
            {
                "role": "fugitive",
                "pile": 0,
                "card": None,
                "round_number": 1,
            }
        ],
        "round_number": 2,
        "phase": "marshal_guess",
        "legal_draw_piles": [0, 2],
        "play_history": [
            {
                "hideout": None,
                "sprint_count": 2,
                "sprint_cards": None,
                "route_index": 1,
                "round_number": 1,
                "passed": False,
            }
        ],
    }


def test_canonical_bytes_hash_and_rng_state_are_repeatable() -> None:
    observation = populated_observation()
    encoded = canonical_observation_bytes(observation)

    assert canonical_observation_bytes(observation) == encoded
    assert hashlib.sha256(encoded).hexdigest() == (
        "4acd9fefc437a9e5abd6e0335a92687aca7cec6a53ba2e5da7f8d33e5db31852"
    )
    assert stable_observation_seed(
        observation, domain="test.v1", salt="agent-9"
    ) == stable_observation_seed(observation, domain="test.v1", salt=b"agent-9")
    assert stable_observation_seed(
        observation, domain="test.v1", salt="agent-9"
    ) != stable_observation_seed(observation, domain="other.v1", salt="agent-9")
    assert canonical_random_state_bytes(random.Random(17)) == (
        canonical_random_state_bytes(random.Random(17))
    )


def test_canonical_encoding_cannot_distinguish_marshal_private_worlds() -> None:
    first_engine = GameEngine(seed=3)
    second_engine = GameEngine(seed=31)
    first_engine.apply_fugitive_action(FugitiveAction(1))
    first_engine.apply_fugitive_action(FugitiveAction(2))
    second_engine.apply_fugitive_action(FugitiveAction(2))
    second_engine.apply_fugitive_action(FugitiveAction(3))

    first = first_engine.observation(Role.MARSHAL)
    second = second_engine.observation(Role.MARSHAL)
    assert first == second
    assert first_engine.observation(Role.FUGITIVE) != second_engine.observation(
        Role.FUGITIVE
    )
    assert canonical_observation_bytes(first) == canonical_observation_bytes(second)
    assert stable_observation_seed(first, domain="privacy.v1") == (
        stable_observation_seed(second, domain="privacy.v1")
    )
