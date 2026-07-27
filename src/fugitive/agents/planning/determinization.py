"""Reconstruct complete simulation states from player information sets."""

from __future__ import annotations

from dataclasses import dataclass
import random

from fugitive.game.engine import GameEngine
from fugitive.game.model import (
    DrawRecord,
    HistoryRecord,
    Observation,
    Phase,
    PlayRecord,
    PlayView,
    Role,
    RouteView,
)
from ..marshal.inference.particles.state import MarshalParticle
from ..marshal.inference.world_validation import (
    CARD_TO_PILE,
    CompleteMarshalWorld,
    validate_complete_world,
)


@dataclass(frozen=True, slots=True)
class FugitiveWorld:
    """One hypothesis for the state hidden from the Fugitive.

    The Fugitive observation already contains its hand, route, Sprint cards,
    and private draws. Only the Marshal hand and unordered remaining pile
    contents must be supplied by a Fugitive-side belief model.
    """

    marshal_hand: tuple[int, ...]
    remaining_piles: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]


def determinize_game(
    observation: Observation,
    world: CompleteMarshalWorld,
    *,
    rng: random.Random,
) -> GameEngine:
    """Reconstruct one engine consistent with a Marshal information state.

    ``MarshalParticle`` and ``ConstructiveWorld`` both implement
    :class:`CompleteMarshalWorld`. Their remaining piles are unordered. This
    function samples a uniform order for every pile with the caller-owned RNG;
    no deck-order knowledge is introduced by the world representation.
    """

    if not isinstance(rng, random.Random):
        raise TypeError("rng must be random.Random")
    validate_complete_world(world, observation)

    engine = GameEngine.from_determinized_state(
        piles=_ordered_piles(world.remaining_piles, rng),
        fugitive_hand=world.fugitive_hand,
        marshal_hand=world.marshal_hand,
        route=tuple(
            (
                world.route_hideouts[index],
                world.route_sprints[index],
                slot.revealed,
            )
            for index, slot in enumerate(observation.route)
        ),
        phase=observation.phase,
        round_number=observation.round_number,
        history=_reconstruct_history(observation, world),
    )
    if engine.observation(Role.MARSHAL) != observation:
        raise ValueError("determinized engine does not reproduce the observation")
    return engine


def determinize_fugitive_game(
    observation: Observation,
    world: FugitiveWorld,
    *,
    rng: random.Random,
) -> GameEngine:
    """Reconstruct a game from a Fugitive information state and one world."""

    if observation.role is not Role.FUGITIVE:
        raise ValueError("Fugitive determinization requires a Fugitive observation")
    if not isinstance(world, FugitiveWorld):
        raise TypeError("world must be FugitiveWorld")
    if not isinstance(rng, random.Random):
        raise TypeError("rng must be random.Random")

    marshal_observation, complete_world = _complete_fugitive_world(
        observation,
        world,
        rng,
    )
    validate_complete_world(complete_world, marshal_observation)

    engine = GameEngine.from_determinized_state(
        piles=_ordered_piles(complete_world.remaining_piles, rng),
        fugitive_hand=complete_world.fugitive_hand,
        marshal_hand=complete_world.marshal_hand,
        route=tuple(
            (
                complete_world.route_hideouts[index],
                complete_world.route_sprints[index],
                slot.revealed,
            )
            for index, slot in enumerate(observation.route)
        ),
        phase=observation.phase,
        round_number=observation.round_number,
        history=_reconstruct_history(marshal_observation, complete_world),
    )
    if engine.observation(Role.FUGITIVE) != observation:
        raise ValueError("determinized engine does not reproduce the observation")
    return engine


def _ordered_piles(
    remaining_piles: tuple[
        tuple[int, ...],
        tuple[int, ...],
        tuple[int, ...],
    ],
    rng: random.Random,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    piles: list[tuple[int, ...]] = []
    for contents in remaining_piles:
        ordered = list(contents)
        rng.shuffle(ordered)
        piles.append(tuple(ordered))
    return (piles[0], piles[1], piles[2])


def _reconstruct_history(
    observation: Observation,
    world: CompleteMarshalWorld,
) -> tuple[HistoryRecord, ...]:
    fugitive_draws = iter(world.fugitive_draws)
    draws = tuple(
        DrawRecord(
            record.role,
            record.pile,
            next(fugitive_draws) if record.role is Role.FUGITIVE else record.card,
            record.round_number,
        )
        for record in observation.draw_history
    )
    plays: list[PlayRecord] = []
    for record in observation.play_history:
        if record.passed:
            plays.append(PlayRecord(None, (), None, record.round_number))
            continue
        assert record.route_index is not None
        plays.append(
            PlayRecord(
                world.route_hideouts[record.route_index],
                world.route_sprints[record.route_index],
                record.route_index,
                record.round_number,
            )
        )
    guesses = tuple(observation.guess_history)

    indexed: list[tuple[tuple[int, int, int], HistoryRecord]] = []
    indexed.extend(
        (
            (
                record.round_number,
                0 if record.role is Role.FUGITIVE else 2,
                index,
            ),
            record,
        )
        for index, record in enumerate(draws)
    )
    indexed.extend(
        ((record.round_number, 1, index), record)
        for index, record in enumerate(plays)
    )
    indexed.extend(
        ((record.round_number, 3, index), record)
        for index, record in enumerate(guesses)
    )
    indexed.sort(key=lambda item: item[0])
    return tuple(record for _key, record in indexed)


def _complete_fugitive_world(
    observation: Observation,
    world: FugitiveWorld,
    rng: random.Random,
) -> tuple[Observation, MarshalParticle]:
    """Reuse complete-world validation via a sampled Marshal projection."""

    cards_by_pile: list[list[int]] = []
    for pile in range(3):
        cards = sorted(
            card
            for card in world.marshal_hand
            if CARD_TO_PILE.get(card) == pile
        )
        rng.shuffle(cards)
        cards_by_pile.append(cards)
    marshal_draw_counts = tuple(
        sum(
            record.role is Role.MARSHAL and record.pile == pile
            for record in observation.draw_history
        )
        for pile in range(3)
    )
    if tuple(map(len, cards_by_pile)) != marshal_draw_counts:
        raise ValueError("sampled Marshal hand does not match public draw piles")

    draw_offsets = [0, 0, 0]
    projected_draws: list[DrawRecord] = []
    fugitive_draws: list[int] = []
    for record in observation.draw_history:
        if record.role is Role.FUGITIVE:
            if record.card is None:
                raise ValueError("Fugitive observation hides its own draw")
            fugitive_draws.append(record.card)
            projected_draws.append(
                DrawRecord(
                    Role.FUGITIVE,
                    record.pile,
                    None,
                    record.round_number,
                )
            )
            continue
        offset = draw_offsets[record.pile]
        card = cards_by_pile[record.pile][offset]
        draw_offsets[record.pile] += 1
        projected_draws.append(
            DrawRecord(
                Role.MARSHAL,
                record.pile,
                card,
                record.round_number,
            )
        )

    projected_route: list[RouteView] = []
    route_hideouts: list[int] = []
    route_sprints: list[tuple[int, ...]] = []
    for slot in observation.route:
        if slot.hideout is None or slot.sprint_cards is None:
            raise ValueError("Fugitive observation hides its own route")
        hideout = slot.hideout
        sprints = slot.sprint_cards
        route_hideouts.append(hideout)
        route_sprints.append(sprints)
        projected_route.append(
            RouteView(
                index=slot.index,
                hideout=(hideout if slot.revealed or hideout in (0, 42) else None),
                sprint_count=slot.sprint_count,
                sprint_cards=(
                    sprints if slot.revealed and hideout != 42 else None
                ),
                revealed=slot.revealed,
            )
        )

    projected_plays: list[PlayView] = []
    for record in observation.play_history:
        if record.passed:
            projected_plays.append(
                PlayView(
                    hideout=None,
                    sprint_count=0,
                    sprint_cards=(),
                    route_index=None,
                    round_number=record.round_number,
                    passed=True,
                )
            )
            continue
        assert record.route_index is not None
        route_slot = projected_route[record.route_index]
        projected_plays.append(
            PlayView(
                hideout=route_slot.hideout,
                sprint_count=record.sprint_count,
                sprint_cards=route_slot.sprint_cards,
                route_index=record.route_index,
                round_number=record.round_number,
                passed=False,
            )
        )

    legal_draw_piles = (
        tuple(index for index, pile in enumerate(world.remaining_piles) if pile)
        if observation.phase is Phase.MARSHAL_DRAW
        else ()
    )
    marshal_observation = Observation(
        role=Role.MARSHAL,
        hand=tuple(sorted(world.marshal_hand)),
        pile_sizes=observation.pile_sizes,
        route=tuple(projected_route),
        guess_history=observation.guess_history,
        draw_history=tuple(projected_draws),
        round_number=observation.round_number,
        phase=observation.phase,
        legal_draw_piles=legal_draw_piles,
        play_history=tuple(projected_plays),
    )
    complete_world = MarshalParticle(
        fugitive_hand=observation.hand,
        marshal_hand=tuple(sorted(world.marshal_hand)),
        route_hideouts=tuple(route_hideouts),
        route_sprints=tuple(route_sprints),
        fugitive_draws=tuple(fugitive_draws),
        remaining_piles=world.remaining_piles,
    )
    return marshal_observation, complete_world


__all__ = [
    "FugitiveWorld",
    "determinize_fugitive_game",
    "determinize_game",
]
