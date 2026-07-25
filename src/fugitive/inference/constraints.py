"""Compile Marshal-visible facts and the route creation timeline.

This module is the observation boundary of constructive inference. Everything
downstream consumes :class:`CompiledMarshalConstraints` and therefore cannot
inspect engine-private Fugitive state.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from fugitive.belief import PathBelief
from fugitive.model import DrawRecord, Observation, Role
from fugitive.observation_validation import validate_marshal_observation_contract
from fugitive.world_validation import (
    INITIAL_FUGITIVE_CARDS,
    compile_route_creation_rounds,
)


@dataclass(frozen=True, slots=True)
class DrawSlot:
    """One labelled Fugitive draw position in public-history order."""

    global_index: int
    pile: int
    round_number: int


DeadlineDemands = tuple[
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
]


def compile_creation_timeline(observation: Observation) -> tuple[int, ...]:
    """Map each public route position to the round in which it was created."""

    return compile_route_creation_rounds(observation)


@dataclass(frozen=True, slots=True)
class CompiledMarshalConstraints:
    """Validated immutable constraints derived only from a Marshal observation."""

    observation: Observation
    fugitive_records: tuple[DrawRecord, ...]
    marshal_records: tuple[DrawRecord, ...]
    draw_slots: tuple[DrawSlot, ...]
    creation_rounds: tuple[int, ...]
    marshal_hand: tuple[int, ...]
    known_fugitive_cards: frozenset[int]
    path_belief: PathBelief = field(repr=False, compare=False)

    @classmethod
    def from_observation(cls, observation: Observation) -> "CompiledMarshalConstraints":
        validate_marshal_observation_contract(observation)

        fugitive_records = tuple(
            record
            for record in observation.draw_history
            if record.role is Role.FUGITIVE
        )
        marshal_records = tuple(
            record
            for record in observation.draw_history
            if record.role is Role.MARSHAL
        )
        marshal_hand = tuple(sorted(observation.hand))
        known_fugitive = set(INITIAL_FUGITIVE_CARDS)
        for index, slot in enumerate(observation.route):
            identities = (
                () if not index or slot.hideout is None else (slot.hideout,)
            ) + (slot.sprint_cards or ())
            for card in identities:
                known_fugitive.add(card)

        creation_rounds = compile_creation_timeline(observation)
        draw_slots = tuple(
            DrawSlot(index, record.pile, record.round_number)
            for index, record in enumerate(fugitive_records)
        )
        return cls(
            observation=observation,
            fugitive_records=fugitive_records,
            marshal_records=marshal_records,
            draw_slots=draw_slots,
            creation_rounds=creation_rounds,
            marshal_hand=marshal_hand,
            known_fugitive_cards=frozenset(known_fugitive),
            path_belief=PathBelief.from_observation(observation),
        )


__all__ = [
    "CompiledMarshalConstraints",
    "DeadlineDemands",
    "DrawSlot",
    "compile_creation_timeline",
]
