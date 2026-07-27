from dataclasses import replace

import pytest

from fugitive.game.engine import GameEngine
from fugitive.agents.marshal.inference.constructive.sampler import ConstructiveWorldSampler
from fugitive.game.model import FugitiveAction, Role
from fugitive.agents.marshal.inference.particle_belief import (
    MarshalParticle,
    particle_matches_public_projection,
)
from fugitive.game.rules import PILE_CARDS
from fugitive.agents.marshal.inference.world_validation import (
    CompleteWorldConsistencyError,
    is_complete_world_consistent,
    validate_complete_world,
)


def _world_and_observation():
    engine = GameEngine(seed=7)
    engine.apply_fugitive_action(FugitiveAction(1))
    engine.apply_fugitive_action(FugitiveAction(2))
    observation = engine.observation(Role.MARSHAL)
    batch = ConstructiveWorldSampler(sprint_backend="exact").sample_batch(
        observation,
        particle_count=1,
        seed=91,
    )
    assert len(batch.worlds) == 1
    return batch.worlds[0], observation


def _world_after_normal_guess():
    engine = GameEngine(seed=7)
    engine.apply_fugitive_action(FugitiveAction(1))
    engine.apply_fugitive_action(FugitiveAction(2))
    engine.draw(2)
    engine.draw(2)
    assert not engine.apply_guess((3,))
    observation = engine.observation(Role.MARSHAL)
    batch = ConstructiveWorldSampler(sprint_backend="exact").sample_batch(
        observation,
        particle_count=1,
        seed=92,
    )
    return batch.worlds[0], observation


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
        route_hideouts=tuple(slot.hideout for slot in fugitive.route),
        route_sprints=tuple(slot.sprint_cards for slot in fugitive.route),
        fugitive_draws=fugitive_draws,
        remaining_piles=remaining,
    )


def test_constructive_world_passes_complete_validation() -> None:
    world, observation = _world_and_observation()

    validate_complete_world(world, observation)
    assert is_complete_world_consistent(world, observation)


def test_draw_identity_cannot_disagree_with_fugitive_ownership() -> None:
    world, observation = _world_and_observation()
    drawn = world.fugitive_draws[0]
    replacement = next(
        card
        for card in world.remaining_piles[0]
        if card != drawn
    )
    mutated_hand = tuple(
        sorted(replacement if card == drawn else card for card in world.fugitive_hand)
    )
    mutated_pile = tuple(
        sorted(drawn if card == replacement else card for card in world.remaining_piles[0])
    )
    mutated = replace(
        world,
        fugitive_hand=mutated_hand,
        remaining_piles=(mutated_pile, *world.remaining_piles[1:]),
    )

    assert particle_matches_public_projection(mutated.to_particle(), observation)
    with pytest.raises(
        CompleteWorldConsistencyError,
        match="owned cards|draw history",
    ):
        validate_complete_world(mutated, observation)
    assert not is_complete_world_consistent(mutated, observation)


def test_remaining_cards_cannot_move_between_piles() -> None:
    world, observation = _world_and_observation()
    first = world.remaining_piles[0][0]
    second = world.remaining_piles[1][0]
    mutated = replace(
        world,
        remaining_piles=(
            tuple(sorted((second, *world.remaining_piles[0][1:]))),
            tuple(sorted((first, *world.remaining_piles[1][1:]))),
            world.remaining_piles[2],
        ),
    )

    with pytest.raises(CompleteWorldConsistencyError, match="different pile"):
        validate_complete_world(mutated, observation)


def test_card_cannot_be_played_before_its_draw_round() -> None:
    engine = GameEngine(seed=5)
    engine.apply_fugitive_action(FugitiveAction(1))
    engine.apply_fugitive_action(FugitiveAction(2))
    engine.draw(2)
    engine.draw(2)
    assert not engine.apply_guess((41,))
    assert engine.draw(0) == 4
    engine.apply_fugitive_action(FugitiveAction(4))
    observation = engine.observation(Role.MARSHAL)
    world = _ground_truth_world(engine)
    validate_complete_world(world, observation)
    records = list(observation.draw_history)
    record_index = next(
        index
        for index, record in enumerate(records)
        if record.role is Role.FUGITIVE and record.round_number == 2
    )
    records[record_index] = replace(records[record_index], round_number=3)
    mutated_observation = replace(
        observation,
        draw_history=tuple(records),
        round_number=3,
    )

    with pytest.raises(CompleteWorldConsistencyError, match="before"):
        validate_complete_world(world, mutated_observation)


def test_complete_validation_requires_explicit_play_timeline() -> None:
    world, observation = _world_and_observation()

    with pytest.raises(CompleteWorldConsistencyError, match="every route placement"):
        validate_complete_world(world, replace(observation, play_history=()))
def test_manhunt_cannot_appear_before_hideout_42() -> None:
    world, observation = _world_after_normal_guess()
    guess = replace(observation.guess_history[0], manhunt=True)

    with pytest.raises(
        CompleteWorldConsistencyError,
        match="only after Hideout 42",
    ):
        validate_complete_world(
            world,
            replace(observation, guess_history=(guess,)),
        )


def test_marshal_projection_never_exposes_sprints_beneath_42() -> None:
    world, observation = _world_and_observation()
    payment = tuple(range(2, 42, 2))
    mutated_world = replace(
        world,
        route_hideouts=(*world.route_hideouts[:-1], 42),
        route_sprints=(*world.route_sprints[:-1], payment),
    )
    leaked_slot = replace(
        observation.route[-1],
        hideout=42,
        sprint_count=len(payment),
        sprint_cards=payment,
        revealed=False,
    )

    with pytest.raises(CompleteWorldConsistencyError, match="without exposing"):
        validate_complete_world(
            mutated_world,
            replace(
                observation,
                route=(*observation.route[:-1], leaked_slot),
            ),
        )
