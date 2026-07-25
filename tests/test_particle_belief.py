from __future__ import annotations

from dataclasses import replace
import random

import pytest

from fugitive.engine import GameEngine
from fugitive.model import FugitiveAction, Phase, Role
from fugitive.observation_validation import ObservationContractError
from fugitive.particle_belief import (
    DEFAULT_PARTICLE_COUNT,
    MarshalParticle,
    MarshalParticleBelief,
)
from fugitive.particle_inference.bootstrap_filter import (
    _advance_play,
    _matching_incremental_plays,
)
from fugitive.rules import (
    PILE_CARDS,
    is_legal_fugitive_action,
    legal_fugitive_actions,
    sprint_value,
)


def opening_observation(
    first: FugitiveAction = FugitiveAction(1),
    second: FugitiveAction = FugitiveAction(2),
    *,
    seed: int = 7,
):
    engine = GameEngine(seed=seed)
    engine.apply_fugitive_action(first)
    engine.apply_fugitive_action(second)
    return engine, engine.observation(Role.MARSHAL)


@pytest.mark.parametrize("particle_count", (True, 0, -1, 1.5, "100"))
def test_particle_count_must_be_a_positive_integer(particle_count: object) -> None:
    _engine, observation = opening_observation()
    with pytest.raises(ValueError, match="positive integer"):
        MarshalParticleBelief.from_observation(
            observation,
            particle_count=particle_count,  # type: ignore[arg-type]
        )


def test_default_population_and_complete_particle_invariants() -> None:
    # Seed 1 gives the Fugitive card 6.  The hidden second edge therefore has
    # one actual Sprint card whose parity must be represented in every sample.
    engine, observation = opening_observation(
        FugitiveAction(1), FugitiveAction(6, (2,)), seed=1
    )
    assert engine.phase is Phase.MARSHAL_DRAW

    belief = MarshalParticleBelief.from_observation(observation, seed=19)

    assert len(belief.particles) == DEFAULT_PARTICLE_COUNT
    assert not belief.sampling_exhausted
    assert belief.sampling_accepted == len(belief.particles)
    assert belief.sampling_acceptance_rate == pytest.approx(
        belief.sampling_accepted / belief.sampling_attempts
    )
    assert 0.0 < belief.effective_sample_size <= len(belief.particles)
    assert sum(particle.weight for particle in belief.particles) == pytest.approx(1.0)
    for particle in belief.particles:
        assert particle.all_cards_are_unique
        assert len(particle.route_hideouts) == len(observation.route)
        assert tuple(map(len, particle.route_sprints)) == tuple(
            slot.sprint_count for slot in observation.route
        )
        for index in range(1, len(particle.route_hideouts)):
            distance = (
                particle.route_hideouts[index]
                - particle.route_hideouts[index - 1]
            )
            capacity = 3 + sum(
                sprint_value(card) for card in particle.route_sprints[index]
            )
            assert 1 <= distance <= capacity
        assert tuple(map(len, particle.remaining_piles)) == observation.pile_sizes
        assert belief.current_hidden_hideouts(particle)
        for card, record in zip(
            particle.fugitive_draws,
            (
                record
                for record in observation.draw_history
                if record.role is Role.FUGITIVE
            ),
            strict=True,
        ):
            assert card in PILE_CARDS[record.pile]

    assert 0.0 <= belief.probability_fugitive_can_play_42() <= 1.0


def test_fresh_constructive_belief_is_complete_and_deterministic() -> None:
    _, observation = opening_observation()
    first = MarshalParticleBelief.from_observation(
        observation,
        particle_count=32,
        seed=7,
    )
    second = MarshalParticleBelief.from_observation(
        observation,
        particle_count=32,
        seed=7,
    )

    assert first.particles == second.particles
    assert len(first.particles) == 32
    assert first.sampling_accepted == 32
    assert first.sampling_attempts >= 32
    assert not first.sampling_exhausted


def test_world_identity_excludes_particle_weight() -> None:
    _engine, observation = opening_observation()
    sampled = MarshalParticleBelief.from_observation(
        observation,
        particle_count=1,
        seed=23,
    )
    particle = sampled.particles[0]
    belief = MarshalParticleBelief(
        observation,
        (
            replace(particle, weight=0.25),
            replace(particle, weight=0.75),
        ),
        sampling_attempts=2,
        sampling_accepted=2,
        sampling_exhausted=False,
    )

    assert belief.particles[0].world_key == belief.particles[1].world_key
    assert belief.unique_particle_count == 1
    assert belief.effective_sample_size == pytest.approx(1.6)
    assert belief.world_effective_sample_size == pytest.approx(1.0)
    assert belief.max_world_mass == pytest.approx(1.0)
    assert belief.unique_hidden_hypothesis_count == 1


@pytest.mark.parametrize(
    "weight",
    (-1.0, float("nan"), float("inf"), float("-inf")),
)
def test_particle_weights_fail_fast(weight: float) -> None:
    _engine, observation = opening_observation()
    particle = MarshalParticleBelief.from_observation(
        observation, particle_count=1, seed=23
    ).particles[0]

    with pytest.raises(ValueError, match="finite non-negative"):
        MarshalParticleBelief(
            observation,
            (replace(particle, weight=weight),),
            sampling_attempts=1,
            sampling_exhausted=False,
        )


def test_zero_total_particle_mass_fails_fast() -> None:
    _engine, observation = opening_observation()
    particle = MarshalParticleBelief.from_observation(
        observation, particle_count=1, seed=23
    ).particles[0]

    with pytest.raises(ValueError, match="positive finite mass"):
        MarshalParticleBelief(
            observation,
            (replace(particle, weight=0.0),),
            sampling_attempts=1,
            sampling_exhausted=False,
        )


def test_incremental_update_aggregates_duplicate_world_mass() -> None:
    _engine, observation = opening_observation()
    particle = MarshalParticleBelief.from_observation(
        observation, particle_count=1, seed=23
    ).particles[0]
    belief = MarshalParticleBelief(
        observation,
        (
            replace(particle, weight=0.25),
            replace(particle, weight=0.75),
        ),
        sampling_attempts=2,
        sampling_exhausted=False,
    )

    advanced = belief.advance_to(observation, particle_count=2, seed=9)

    assert len(advanced.particles) == 1
    assert advanced.particles[0].weight == pytest.approx(1.0)
    assert advanced.resampling_count == 0
    assert advanced.world_effective_sample_size == pytest.approx(1.0)


def test_indistinguishable_private_worlds_have_identical_summary() -> None:
    first_engine, first_observation = opening_observation(
        FugitiveAction(1), FugitiveAction(2), seed=3
    )
    second_engine, second_observation = opening_observation(
        FugitiveAction(2), FugitiveAction(3), seed=31
    )

    # The true hands, hidden routes, and pile contents differ.  None of those
    # differences is present in the Marshal information state.
    assert first_engine.observation(Role.FUGITIVE) != second_engine.observation(
        Role.FUGITIVE
    )
    assert first_observation == second_observation

    first = MarshalParticleBelief.from_observation(
        first_observation, particle_count=256, seed=991
    )
    second = MarshalParticleBelief.from_observation(
        second_observation, particle_count=256, seed=991
    )

    assert first.summary() == second.summary()
    assert first.particles == second.particles


def test_failed_multi_guess_excludes_only_the_joint_event() -> None:
    engine, _ = opening_observation(
        FugitiveAction(1), FugitiveAction(3), seed=5
    )
    engine.draw(2)
    engine.draw(2)
    assert not engine.apply_guess((1, 2))
    engine.draw(2)
    engine.apply_fugitive_action(FugitiveAction(None))
    engine.draw(2)
    observation = engine.observation(Role.MARSHAL)
    assert observation.phase is Phase.MARSHAL_GUESS

    belief = MarshalParticleBelief.from_observation(
        observation, particle_count=512, seed=101
    )

    assert not belief.is_empty
    assert belief.joint_success((1, 2)) == 0.0
    # Failure means "not both".  It does not individually eliminate either
    # number, and the particles retain support for both alternatives.
    assert belief.probability_hidden(1) > 0.0
    assert belief.probability_hidden(2) > 0.0
    assert any(1 in particle.route_hideouts for particle in belief.particles)
    assert any(2 in particle.route_hideouts for particle in belief.particles)


def test_failed_single_applies_only_to_the_historical_route_prefix() -> None:
    engine, _ = opening_observation(
        FugitiveAction(1), FugitiveAction(3), seed=0
    )
    engine.draw(2)
    engine.draw(2)
    assert not engine.apply_guess((4,))
    engine.draw(2)
    engine.apply_fugitive_action(FugitiveAction(4))
    engine.draw(2)
    observation = engine.observation(Role.MARSHAL)

    belief = MarshalParticleBelief.from_observation(
        observation, particle_count=512, seed=81
    )

    assert not belief.is_empty
    assert all(4 not in particle.route_hideouts[1:3] for particle in belief.particles)
    assert any(particle.route_hideouts[3] == 4 for particle in belief.particles)
    assert belief.probability_hidden(4) > 0.0


def test_successful_guess_is_revealed_in_every_particle() -> None:
    engine, _ = opening_observation(
        FugitiveAction(1), FugitiveAction(3), seed=17
    )
    engine.draw(2)
    engine.draw(2)
    assert engine.apply_guess((1,))
    engine.draw(2)
    engine.apply_fugitive_action(FugitiveAction(None))
    engine.draw(2)
    observation = engine.observation(Role.MARSHAL)

    belief = MarshalParticleBelief.from_observation(
        observation, particle_count=256, seed=82
    )

    assert not belief.is_empty
    assert all(particle.route_hideouts[1] == 1 for particle in belief.particles)
    assert belief.probability_hidden(1) == 0.0
    assert belief.joint_success((1,)) == 0.0


def test_draw_posterior_and_private_draw_conditioning() -> None:
    _, observation = opening_observation(seed=11)
    belief = MarshalParticleBelief.from_observation(
        observation, particle_count=512, seed=22
    )
    posterior = belief.draw_card_posterior(0)

    assert sum(posterior.values()) == pytest.approx(1.0)
    card = max(posterior, key=posterior.get)
    conditioned = belief.conditioned_on_marshal_draw(0, card)
    assert not conditioned.is_empty
    assert sum(particle.weight for particle in conditioned.particles) == pytest.approx(1.0)
    assert all(card in particle.marshal_hand for particle in conditioned.particles)
    assert all(
        card not in particle.remaining_piles[0] for particle in conditioned.particles
    )


def test_fresh_constructive_belief_rejects_bad_input_contract() -> None:
    _, observation = opening_observation(seed=13)
    contradictory = replace(observation, pile_sizes=(0, 0, 0))
    with pytest.raises(ObservationContractError):
        MarshalParticleBelief.from_observation(
            contradictory,
            particle_count=16,
            seed=4,
        )


@pytest.mark.parametrize("particle_count", (True, 0, -1, 1.5, "64"))
def test_incremental_particle_count_requires_a_positive_integer(
    particle_count: object,
) -> None:
    _, observation = opening_observation()
    belief = MarshalParticleBelief.from_observation(
        observation, particle_count=16, seed=1
    )

    with pytest.raises(ValueError, match="positive integer"):
        belief.advance_to(
            observation,
            particle_count=particle_count,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("proposal_count", (True, 0, -1, 1.5, "64"))
def test_incremental_play_proposal_count_requires_a_positive_integer(
    proposal_count: object,
) -> None:
    _, observation = opening_observation()
    belief = MarshalParticleBelief.from_observation(
        observation, particle_count=16, seed=1
    )

    with pytest.raises(ValueError, match="positive integer"):
        belief.advance_to(
            observation,
            max_play_proposals_per_particle=proposal_count,  # type: ignore[arg-type]
        )


def test_incremental_own_draw_conditions_the_remaining_pile() -> None:
    engine, observation = opening_observation(seed=7)
    belief = MarshalParticleBelief.from_observation(
        observation, particle_count=64, seed=2
    )

    card = engine.draw(0)
    target = engine.observation(Role.MARSHAL)
    advanced = belief.advance_to(target, particle_count=64, seed=3)

    assert not advanced.is_empty
    assert advanced.observation == target
    # Adaptive filtering keeps the surviving weighted support when ESS is
    # healthy instead of resampling back to a cosmetic fixed entry count.
    assert 0 < len(advanced.particles) <= 64
    assert all(card in particle.marshal_hand for particle in advanced.particles)
    assert all(
        card not in particle.remaining_piles[0]
        for particle in advanced.particles
    )
    assert all(
        tuple(map(len, particle.remaining_piles)) == target.pile_sizes
        for particle in advanced.particles
    )


def test_incremental_hidden_fugitive_draw_propagates_every_possible_identity() -> None:
    engine, _ = opening_observation(seed=11)
    engine.draw(2)
    engine.draw(2)
    assert not engine.apply_guess((41,))
    observation = engine.observation(Role.MARSHAL)
    belief = MarshalParticleBelief.from_observation(
        observation, particle_count=64, seed=4
    )
    old_draw_count = len(belief.particles[0].fugitive_draws)

    engine.draw(0)
    target = engine.observation(Role.MARSHAL)
    advanced = belief.advance_to(target, particle_count=64, seed=5)

    assert not advanced.is_empty
    assert advanced.resampling_count > belief.resampling_count
    assert advanced.pre_resample_effective_sample_size > 0.0
    assert all(
        len(particle.fugitive_draws) == old_draw_count + 1
        for particle in advanced.particles
    )
    assert all(
        particle.fugitive_draws[-1] in PILE_CARDS[0]
        and particle.fugitive_draws[-1] in particle.fugitive_hand
        for particle in advanced.particles
    )
    assert all(
        tuple(map(len, particle.remaining_piles)) == target.pile_sizes
        for particle in advanced.particles
    )


def test_incremental_failed_multi_filters_only_the_joint_event() -> None:
    engine, _ = opening_observation(
        FugitiveAction(1), FugitiveAction(3), seed=5
    )
    engine.draw(2)
    engine.draw(2)
    observation = engine.observation(Role.MARSHAL)
    belief = MarshalParticleBelief.from_observation(
        observation, particle_count=128, seed=6
    )
    assert belief.joint_success((1, 2)) > 0.0

    assert not engine.apply_guess((1, 2))
    target = engine.observation(Role.MARSHAL)
    advanced = belief.advance_to(target, particle_count=128, seed=7)

    assert not advanced.is_empty
    assert advanced.joint_success((1, 2)) == 0.0
    assert advanced.probability_hidden(1) > 0.0
    assert advanced.probability_hidden(2) > 0.0


def test_incremental_hidden_play_uses_hand_and_actual_sprint_parity() -> None:
    engine, _ = opening_observation(seed=1)
    engine.draw(2)
    engine.draw(2)
    assert not engine.apply_guess((41,))
    engine.draw(2)
    observation = engine.observation(Role.MARSHAL)
    belief = MarshalParticleBelief.from_observation(
        observation, particle_count=128, seed=8
    )

    engine.apply_fugitive_action(FugitiveAction(6, (3,)))
    target = engine.observation(Role.MARSHAL)
    advanced = belief.advance_to(target, particle_count=128, seed=9)

    assert not advanced.is_empty
    assert target.route[-1].hideout is None
    assert target.route[-1].sprint_count == 1
    for particle in advanced.particles:
        action = FugitiveAction(
            particle.route_hideouts[-1], particle.route_sprints[-1]
        )
        assert len(action.sprint_cards) == 1
        assert is_legal_fugitive_action(
            action,
            (action.hideout, *action.sprint_cards),
            particle.route_hideouts[-2],
            allow_pass=False,
        )


def test_incremental_hidden_play_conserves_each_parent_mass() -> None:
    engine, _ = opening_observation(seed=41)
    engine.draw(2)
    engine.draw(2)
    assert not engine.apply_guess((41,))
    engine.draw(0)
    observation = engine.observation(Role.MARSHAL)
    belief = MarshalParticleBelief.from_observation(
        observation,
        particle_count=128,
        seed=44,
    )

    fugitive_observation = engine.observation(Role.FUGITIVE)
    public_action = next(
        action
        for action in legal_fugitive_actions(
            fugitive_observation,
            max_sprint_cards=1,
            include_pass=False,
        )
        if action.hideout != 42 and len(action.sprint_cards) == 1
    )
    engine.apply_fugitive_action(public_action)
    target = engine.observation(Role.MARSHAL)
    record = target.play_history[-1]
    slot = target.route[record.route_index or 0]

    candidates: dict[int, MarshalParticle] = {}
    for particle in belief.particles:
        actions = _matching_incremental_plays(
            particle,
            record,
            slot_hideout=slot.hideout,
            slot_sprint_cards=slot.sprint_cards,
            rng=random.Random(17),
            limit=64,
        )
        if actions:
            candidates.setdefault(len(actions), particle)
        if len(candidates) >= 2:
            break
    assert len(candidates) >= 2

    for parent_weight, particle in zip(
        (0.25, 0.75),
        candidates.values(),
        strict=True,
    ):
        children = _advance_play(
            (replace(particle, weight=parent_weight),),
            record,
            target,
            rng=random.Random(17),
            max_proposals=64,
        )
        assert len(children) > 0
        assert sum(child.weight for child in children) == pytest.approx(
            parent_weight
        )


def test_incremental_success_filters_newly_revealed_sprint_identities() -> None:
    engine, _ = opening_observation(
        FugitiveAction(1), FugitiveAction(6, (2,)), seed=1
    )
    engine.draw(2)
    engine.draw(2)
    observation = engine.observation(Role.MARSHAL)
    belief = MarshalParticleBelief.from_observation(
        observation, particle_count=128, seed=6
    )

    assert engine.apply_guess((6,))
    target = engine.observation(Role.MARSHAL)
    advanced = belief.advance_to(target, particle_count=128, seed=10)

    assert not advanced.is_empty
    assert target.route[2].sprint_cards == (2,)
    assert all(
        particle.route_hideouts[2] == 6
        and particle.route_sprints[2] == (2,)
        for particle in advanced.particles
    )


def _escape_action(observation):
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
            observation, max_sprint_cards=2, include_pass=False
        )
        if action.hideout != 42 and 42 not in action.sprint_cards
    ]
    if not actions:
        return FugitiveAction(None)
    return max(actions, key=lambda action: (action.hideout, -len(action.sprint_cards)))


def _advance_engine_to_escape_play(engine: GameEngine) -> FugitiveAction:
    for _step in range(200):
        if engine.phase in (Phase.FUGITIVE_OPENING, Phase.FUGITIVE_ACTION):
            observation = engine.observation(Role.FUGITIVE)
            action = _escape_action(observation)
            if action.hideout == 42:
                return action
            engine.apply_fugitive_action(action)
        elif engine.phase in (Phase.FUGITIVE_DRAW, Phase.MARSHAL_DRAW):
            role = (
                Role.FUGITIVE
                if engine.phase is Phase.FUGITIVE_DRAW
                else Role.MARSHAL
            )
            observation = engine.observation(role)
            if role is Role.MARSHAL:
                pile = observation.legal_draw_piles[0]
            else:
                previous = observation.route[-1].hideout
                assert previous is not None
                preferred = 0 if previous < 15 else 1 if previous < 29 else 2
                pile = next(
                    (
                        candidate
                        for candidate in observation.legal_draw_piles
                        if candidate >= preferred
                    ),
                    observation.legal_draw_piles[-1],
                )
            engine.draw(pile)
        else:
            observation = engine.observation(Role.MARSHAL)
            engine.apply_guess((observation.hand[0],))
    raise AssertionError("escape planner did not reach card 42")


def _truth_particle(engine: GameEngine) -> MarshalParticle:
    state = engine._state
    return MarshalParticle(
        fugitive_hand=tuple(sorted(state.hands[Role.FUGITIVE])),
        marshal_hand=tuple(sorted(state.hands[Role.MARSHAL])),
        route_hideouts=tuple(node.hideout for node in state.route),
        route_sprints=tuple(tuple(node.sprint_cards) for node in state.route),
        fugitive_draws=tuple(
            record.card
            for record in state.draw_history
            if record.role is Role.FUGITIVE and record.card is not None
        ),
        remaining_piles=tuple(
            tuple(sorted(pile)) for pile in state.piles
        ),  # type: ignore[arg-type]
    )


def test_incremental_card_42_play_enters_manhunt_and_leaves_hand() -> None:
    engine = GameEngine(seed=7)
    action = _advance_engine_to_escape_play(engine)
    observation = engine.observation(Role.MARSHAL)
    particle = _truth_particle(engine)
    assert particle.all_cards_are_unique
    belief = MarshalParticleBelief(
        observation,
        (particle,),
        sampling_attempts=0,
        sampling_exhausted=False,
    )

    engine.apply_fugitive_action(action)
    target = engine.observation(Role.MARSHAL)
    advanced = belief.advance_to(target, particle_count=32, seed=11)

    assert target.phase is Phase.MANHUNT
    assert target.route[-1].hideout == 42
    assert not target.route[-1].revealed
    assert not advanced.is_empty
    assert all(42 not in particle.fugitive_hand for particle in advanced.particles)
    assert all(particle.route_hideouts[-1] == 42 for particle in advanced.particles)
    assert all(
        len(particle.route_sprints[-1]) == target.route[-1].sprint_count
        for particle in advanced.particles
    )


def test_incremental_filter_is_deterministic_for_indistinguishable_worlds() -> None:
    first_engine = GameEngine(seed=23)
    second_engine = GameEngine(seed=23)
    finish_first = (FugitiveAction(1), FugitiveAction(2))
    finish_second = (FugitiveAction(2), FugitiveAction(3))
    for action in finish_first:
        first_engine.apply_fugitive_action(action)
    for action in finish_second:
        second_engine.apply_fugitive_action(action)
    first_observation = first_engine.observation(Role.MARSHAL)
    second_observation = second_engine.observation(Role.MARSHAL)
    assert first_observation == second_observation
    assert first_engine.observation(Role.FUGITIVE) != second_engine.observation(
        Role.FUGITIVE
    )

    first_belief = MarshalParticleBelief.from_observation(
        first_observation, particle_count=64, seed=12
    )
    second_belief = MarshalParticleBelief.from_observation(
        second_observation, particle_count=64, seed=12
    )
    first_engine.draw(2)
    second_engine.draw(2)
    first_target = first_engine.observation(Role.MARSHAL)
    second_target = second_engine.observation(Role.MARSHAL)
    assert first_target == second_target

    first = first_belief.advance_to(first_target, particle_count=64, seed=13)
    second = second_belief.advance_to(second_target, particle_count=64, seed=13)
    assert first.particles == second.particles
    assert first.summary() == second.summary()


def test_incremental_filter_never_calls_full_history_sampler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, observation = opening_observation(seed=31)
    belief = MarshalParticleBelief.from_observation(
        observation, particle_count=32, seed=14
    )
    engine.draw(0)
    target = engine.observation(Role.MARSHAL)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("full-history sampler was called")

    monkeypatch.setattr(MarshalParticleBelief, "from_observation", forbidden)
    advanced = belief.advance_to(target, particle_count=32, seed=15)

    assert not advanced.is_empty
    assert advanced.sampling_attempts == belief.sampling_attempts


def test_incremental_filter_rejects_nonprefix_history() -> None:
    _, observation = opening_observation(seed=37)
    belief = MarshalParticleBelief.from_observation(
        observation, particle_count=16, seed=16
    )
    incompatible = replace(observation, draw_history=())

    with pytest.raises(ValueError, match="not a prefix"):
        belief.advance_to(incompatible, particle_count=16, seed=17)
