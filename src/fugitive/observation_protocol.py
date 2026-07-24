"""Stable serialization and hashing for agent-visible information states.

The protocol is intentionally separate from dataclass ``repr`` output.  Any
change to the fields or their meaning must introduce a new schema version so
recorded experiment seeds do not silently acquire different semantics.
"""

from __future__ import annotations

import hashlib
import json
import random
from typing import Any

from fugitive.model import Observation


OBSERVATION_SCHEMA = "fugitive.observation"
OBSERVATION_SCHEMA_VERSION = 1
RANDOM_STATE_SCHEMA = "python.random-state"
RANDOM_STATE_SCHEMA_VERSION = 1


def observation_to_canonical_data(observation: Observation) -> dict[str, Any]:
    """Return the complete, versioned wire representation of an observation.

    This function can only serialize fields present in ``Observation``.  It
    therefore cannot inspect the engine, deck order, or the opponent's private
    cards and is safe to use for information-set-local random streams.
    """

    return {
        "schema": OBSERVATION_SCHEMA,
        "schema_version": OBSERVATION_SCHEMA_VERSION,
        "observation": {
            "role": observation.role.value,
            "hand": list(observation.hand),
            "pile_sizes": list(observation.pile_sizes),
            "route": [
                {
                    "index": slot.index,
                    "hideout": slot.hideout,
                    "sprint_count": slot.sprint_count,
                    "sprint_cards": (
                        None if slot.sprint_cards is None else list(slot.sprint_cards)
                    ),
                    "revealed": slot.revealed,
                }
                for slot in observation.route
            ],
            "guess_history": [
                {
                    "numbers": list(record.numbers),
                    "success": record.success,
                    "route_length": record.route_length,
                    "round_number": record.round_number,
                    "manhunt": record.manhunt,
                }
                for record in observation.guess_history
            ],
            "draw_history": [
                {
                    "role": record.role.value,
                    "pile": record.pile,
                    "card": record.card,
                    "round_number": record.round_number,
                }
                for record in observation.draw_history
            ],
            "round_number": observation.round_number,
            "phase": observation.phase.value,
            "legal_draw_piles": list(observation.legal_draw_piles),
            "play_history": [
                {
                    "hideout": record.hideout,
                    "sprint_count": record.sprint_count,
                    "sprint_cards": (
                        None
                        if record.sprint_cards is None
                        else list(record.sprint_cards)
                    ),
                    "route_index": record.route_index,
                    "round_number": record.round_number,
                    "passed": record.passed,
                }
                for record in observation.play_history
            ],
        },
    }


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")


def canonical_observation_bytes(observation: Observation) -> bytes:
    """Serialize an observation deterministically under the current schema."""

    return _canonical_json_bytes(observation_to_canonical_data(observation))


def stable_observation_seed(
    observation: Observation,
    *,
    domain: str,
    salt: str | bytes = b"",
) -> int:
    """Return a domain-separated 64-bit seed for one information state."""

    if not domain:
        raise ValueError("domain must be non-empty")
    domain_bytes = domain.encode("utf-8")
    salt_bytes = salt.encode("utf-8") if isinstance(salt, str) else bytes(salt)
    payload = canonical_observation_bytes(observation)
    digest = hashlib.sha256()
    digest.update(b"fugitive.observation-seed\x00v1\x00")
    for part in (domain_bytes, salt_bytes, payload):
        digest.update(len(part).to_bytes(8, "big"))
        digest.update(part)
    return int.from_bytes(digest.digest()[:8], "big")


def canonical_random_state_bytes(rng: random.Random) -> bytes:
    """Encode a ``random.Random`` state without relying on Python ``repr``."""

    state = rng.getstate()
    if not (
        isinstance(state, tuple)
        and len(state) == 3
        and isinstance(state[0], int)
        and isinstance(state[1], tuple)
        and all(isinstance(item, int) for item in state[1])
        and (state[2] is None or isinstance(state[2], float))
    ):
        raise TypeError("unsupported random.Random state shape")
    return _canonical_json_bytes(
        {
            "schema": RANDOM_STATE_SCHEMA,
            "schema_version": RANDOM_STATE_SCHEMA_VERSION,
            "generator_version": state[0],
            "internal_state": list(state[1]),
            "gauss_next": None if state[2] is None else state[2].hex(),
        }
    )


__all__ = [
    "OBSERVATION_SCHEMA",
    "OBSERVATION_SCHEMA_VERSION",
    "canonical_observation_bytes",
    "canonical_random_state_bytes",
    "observation_to_canonical_data",
    "stable_observation_seed",
]
