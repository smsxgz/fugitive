"""Auditable validation for complete Marshal hidden-world hypotheses.

The game has only 42 cards, so inference code uses one direct validator rather
than composing several partial fast paths.  Given a trusted engine observation,
a complete world is checked down to card ownership and draw deadlines.  This is
not a standalone validator for arbitrary game transcripts.
"""

from __future__ import annotations

from typing import Protocol

from fugitive.game.model import FugitiveAction, Observation, Role
from fugitive.game.rules import GUESSABLE_CARDS, PILE_CARDS, is_legal_fugitive_action


INITIAL_FUGITIVE_CARDS = frozenset((1, 2, 3, 42))
ALL_CARDS = frozenset(range(1, 43))
CARD_TO_PILE = {
    card: pile for pile, cards in enumerate(PILE_CARDS) for card in cards
}


class CompleteMarshalWorld(Protocol):
    """Structural interface shared by complete-world inference backends."""

    fugitive_hand: tuple[int, ...]
    marshal_hand: tuple[int, ...]
    route_hideouts: tuple[int, ...]
    route_sprints: tuple[tuple[int, ...], ...]
    fugitive_draws: tuple[int, ...]
    remaining_piles: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]


class CompleteWorldConsistencyError(ValueError):
    """Raised when a proposed complete world contradicts public information."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CompleteWorldConsistencyError(message)


def _valid_card_sequence(cards: tuple[int, ...]) -> bool:
    return all(
        not isinstance(card, bool) and isinstance(card, int) and card in ALL_CARDS
        for card in cards
    )


def compile_route_creation_rounds(
    observation: Observation,
    *,
    require_complete_timeline: bool = True,
) -> tuple[int, ...]:
    """Return the public placement round for every route slot.

    Production observations contain one non-Pass play record for every route
    slot after zero.  Complete-world validation requires that exact timeline.
    Lightweight route-only fixtures may explicitly request the historical
    guess-based fallback by setting ``require_complete_timeline=False``.
    """

    _require(bool(observation.route), "the route must contain Hideout 0")
    rounds: list[int | None] = [0, *([None] * (len(observation.route) - 1))]
    placements = 0
    previous_round = 0
    for record in observation.play_history:
        _require(
            0 <= record.round_number <= observation.round_number,
            "play history contains an invalid round",
        )
        _require(
            previous_round <= record.round_number,
            "play history is not chronological",
        )
        previous_round = record.round_number
        if record.passed:
            _require(
                record.route_index is None
                and record.hideout is None
                and record.sprint_count == 0
                and record.sprint_cards == (),
                "a Pass cannot place cards",
            )
            continue
        _require(
            record.route_index is not None
            and 0 < record.route_index < len(rounds),
            "play history contains an invalid route index",
        )
        index = record.route_index
        _require(rounds[index] is None, "a route slot was placed more than once")
        rounds[index] = record.round_number
        placements += 1
        slot = observation.route[index]
        _require(
            record.sprint_count == slot.sprint_count,
            "play history Sprint count disagrees with the route",
        )
        if record.hideout is not None:
            _require(
                record.hideout == slot.hideout,
                "play history Hideout disagrees with the route",
            )
        if record.sprint_cards is not None:
            _require(
                record.sprint_cards == slot.sprint_cards,
                "play history Sprint cards disagree with the route",
            )

    complete = placements == len(rounds) - 1 and all(
        value is not None for value in rounds
    )
    if require_complete_timeline:
        _require(complete, "play history must account for every route placement")
    elif not complete:
        for index in range(1, len(rounds)):
            if rounds[index] is not None:
                continue
            rounds[index] = next(
                (
                    record.round_number
                    for record in observation.guess_history
                    if record.route_length >= index
                ),
                observation.round_number,
            )
    result = tuple(int(value) for value in rounds)
    _require(
        all(left <= right for left, right in zip(result, result[1:])),
        "route placements must be in chronological order",
    )
    return result


def validate_complete_world(
    world: CompleteMarshalWorld,
    observation: Observation,
) -> None:
    """Validate a hidden world conditional on a trusted engine observation."""

    _require(
        observation.role is Role.MARSHAL,
        "complete Marshal worlds require a Marshal observation",
    )
    _require(
        not isinstance(observation.round_number, bool)
        and isinstance(observation.round_number, int)
        and observation.round_number >= 1,
        "the observation has an invalid round",
    )
    _require(
        len(observation.pile_sizes) == 3
        and len(world.remaining_piles) == 3,
        "the game must contain exactly three piles",
    )
    _require(
        bool(observation.route)
        and observation.route[0].index == 0
        and observation.route[0].hideout == 0
        and observation.route[0].revealed
        and observation.route[0].sprint_count == 0
        and observation.route[0].sprint_cards == (),
        "the public route must start with revealed Hideout 0",
    )
    _require(
        all(slot.index == index for index, slot in enumerate(observation.route)),
        "public route indices must be contiguous",
    )

    for name, cards in (
        ("Fugitive hand", world.fugitive_hand),
        ("Marshal hand", world.marshal_hand),
        ("route", world.route_hideouts[1:]),
        ("Fugitive draws", world.fugitive_draws),
    ):
        _require(_valid_card_sequence(cards), f"{name} contains an invalid card")
    for stack in world.route_sprints:
        _require(_valid_card_sequence(stack), "a Sprint stack contains an invalid card")
        _require(stack == tuple(sorted(stack)), "Sprint stacks must be canonical")
    for pile in world.remaining_piles:
        _require(_valid_card_sequence(pile), "a remaining pile contains an invalid card")
        _require(pile == tuple(sorted(pile)), "remaining piles must be canonical")
    _require(
        world.fugitive_hand == tuple(sorted(world.fugitive_hand))
        and world.marshal_hand == tuple(sorted(world.marshal_hand)),
        "hands must be canonical",
    )

    _require(
        len(world.route_hideouts) == len(observation.route)
        and len(world.route_sprints) == len(observation.route),
        "world route length disagrees with the observation",
    )
    _require(
        world.route_hideouts[0] == 0 and world.route_sprints[0] == (),
        "world route must start with an empty Hideout 0 stack",
    )
    for index in range(1, len(world.route_hideouts)):
        previous = world.route_hideouts[index - 1]
        hideout = world.route_hideouts[index]
        sprints = world.route_sprints[index]
        _require(previous < hideout <= 42, "Hideouts must strictly increase")
        _require(
            is_legal_fugitive_action(
                FugitiveAction(hideout, sprints),
                (hideout, *sprints),
                previous,
                allow_pass=False,
            ),
            "a route segment has insufficient Sprint value",
        )

    _require(
        world.marshal_hand == tuple(sorted(observation.hand)),
        "Marshal hand disagrees with the observation",
    )
    _require(
        tuple(map(len, world.remaining_piles)) == observation.pile_sizes,
        "remaining pile sizes disagree with the observation",
    )
    for index, slot in enumerate(observation.route):
        hideout = world.route_hideouts[index]
        sprints = world.route_sprints[index]
        if slot.hideout is not None:
            _require(slot.hideout == hideout, "a visible Hideout identity disagrees")
        _require(slot.sprint_count == len(sprints), "a Sprint count disagrees")
        if slot.sprint_cards is not None:
            _require(slot.sprint_cards == sprints, "visible Sprint cards disagree")
        if index and hideout == 42:
            _require(
                slot.hideout == 42
                and not slot.revealed
                and slot.sprint_cards is None,
                "Hideout 42 must be public without exposing its Sprint cards",
            )
        elif index and not slot.revealed:
            _require(slot.hideout is None, "a hidden Hideout identity is public")
            _require(slot.sprint_cards is None, "hidden Sprint identities are public")
        elif index:
            _require(
                slot.hideout == hideout and slot.sprint_cards == sprints,
                "an uncovered route slot must expose all identities",
            )

    fugitive_records = tuple(
        record for record in observation.draw_history if record.role is Role.FUGITIVE
    )
    marshal_records = tuple(
        record for record in observation.draw_history if record.role is Role.MARSHAL
    )
    _require(
        all(
            0 <= record.round_number <= observation.round_number
            for record in observation.draw_history
        ),
        "draw history contains an invalid round",
    )
    _require(
        all(
            left.round_number <= right.round_number
            for left, right in zip(
                observation.draw_history, observation.draw_history[1:]
            )
        ),
        "draw history is not chronological",
    )
    setup_signature = tuple(
        (record.role, record.pile)
        for record in observation.draw_history
        if record.round_number == 0
    )
    _require(
        setup_signature
        == (
            (Role.FUGITIVE, 0),
            (Role.FUGITIVE, 0),
            (Role.FUGITIVE, 0),
            (Role.FUGITIVE, 1),
            (Role.FUGITIVE, 1),
        ),
        "draw history has an invalid setup prefix",
    )
    _require(
        len(world.fugitive_draws) == len(fugitive_records),
        "Fugitive draw identities do not align with public draw slots",
    )
    _require(
        all(record.card is None for record in fugitive_records),
        "a Marshal observation cannot reveal Fugitive draws",
    )
    for card, record in zip(world.fugitive_draws, fugitive_records, strict=True):
        _require(
            0 <= record.pile < 3 and card in PILE_CARDS[record.pile],
            "a Fugitive draw came from the wrong pile",
        )
    _require(
        all(
            0 <= record.pile < 3
            and record.card is not None
            and record.card in PILE_CARDS[record.pile]
            for record in marshal_records
        ),
        "Marshal draw history contains an invalid card",
    )
    _require(
        tuple(sorted(record.card for record in marshal_records if record.card is not None))
        == world.marshal_hand,
        "Marshal draw history does not produce the Marshal hand",
    )

    locations = (
        *world.fugitive_hand,
        *world.marshal_hand,
        *world.route_hideouts[1:],
        *(card for stack in world.route_sprints for card in stack),
        *(card for pile in world.remaining_piles for card in pile),
    )
    _require(
        len(locations) == 42 and set(locations) == ALL_CARDS,
        "card locations must form one exact partition of cards 1--42",
    )
    for pile_index, pile in enumerate(world.remaining_piles):
        _require(
            all(card in PILE_CARDS[pile_index] for card in pile),
            "a remaining card belongs to a different pile",
        )

    used = set(world.route_hideouts[1:])
    used.update(card for stack in world.route_sprints for card in stack)
    owned = set(INITIAL_FUGITIVE_CARDS) | set(world.fugitive_draws)
    _require(used <= owned, "the Fugitive used a card they never owned")
    _require(
        set(world.fugitive_hand) == owned - used,
        "Fugitive hand does not equal owned cards minus played cards",
    )
    expected_remaining = tuple(
        tuple(
            sorted(
                set(cards)
                - set(world.marshal_hand)
                - set(world.fugitive_draws)
            )
        )
        for cards in PILE_CARDS
    )
    _require(
        world.remaining_piles == expected_remaining,
        "remaining piles do not match the complete draw history",
    )

    creation_rounds = compile_route_creation_rounds(observation)
    draw_rounds = {
        card: record.round_number
        for card, record in zip(world.fugitive_draws, fugitive_records, strict=True)
    }
    for index in range(1, len(world.route_hideouts)):
        for card in (world.route_hideouts[index], *world.route_sprints[index]):
            if card in CARD_TO_PILE:
                _require(card in draw_rounds, "a played pile card was never drawn")
                _require(
                    draw_rounds[card] <= creation_rounds[index],
                    "a card was played before the Fugitive drew it",
                )

    revealed: set[int] = set()
    route_length = len(world.route_hideouts) - 1
    previous_guess_round = 0
    normal_guess_rounds: set[int] = set()
    manhunt_round: int | None = None
    manhunt_failed = False
    for record in observation.guess_history:
        _require(not manhunt_failed, "guess history continues after a failed Manhunt")
        _require(
            bool(record.numbers)
            and len(record.numbers) == len(set(record.numbers))
            and all(number in GUESSABLE_CARDS for number in record.numbers),
            "guess history contains an illegal guess",
        )
        _require(
            1 <= record.round_number <= observation.round_number
            and previous_guess_round <= record.round_number,
            "guess history contains an invalid round",
        )
        previous_guess_round = record.round_number
        expected_route_length = sum(
            round_number <= record.round_number
            for round_number in creation_rounds[1:]
        )
        _require(
            record.route_length == expected_route_length <= route_length,
            "guess history contains an invalid route length",
        )
        route_at_time = world.route_hideouts[1 : record.route_length + 1]
        if record.manhunt:
            _require(
                len(record.numbers) == 1,
                "a Manhunt guess must contain exactly one number",
            )
            _require(
                bool(route_at_time)
                and route_at_time[-1] == 42
                and record.route_length == route_length,
                "Manhunt can begin only after Hideout 42 is played",
            )
            if manhunt_round is None:
                manhunt_round = record.round_number
                _require(
                    max(revealed, default=0) < 30,
                    "Manhunt is forbidden after revealing Hideout 30 or higher",
                )
            else:
                _require(
                    record.round_number == manhunt_round,
                    "all Manhunt guesses must occur in the escape round",
                )
        else:
            _require(manhunt_round is None, "a normal guess follows Manhunt")
            _require(
                42 not in route_at_time,
                "a normal guess cannot occur after Hideout 42 is played",
            )
            _require(
                record.round_number not in normal_guess_rounds,
                "a normal round contains more than one guess action",
            )
            normal_guess_rounds.add(record.round_number)
        hidden_at_time = (
            set(route_at_time)
            - revealed
            - {42}
        )
        success = set(record.numbers) <= hidden_at_time
        _require(success == record.success, "guess result contradicts the route")
        if success:
            revealed.update(record.numbers)
        elif record.manhunt:
            manhunt_failed = True
    observed_revealed = {
        world.route_hideouts[index]
        for index, slot in enumerate(observation.route)
        if index and slot.revealed
    }
    _require(
        revealed == observed_revealed,
        "revealed route cards disagree with guess history",
    )


def is_complete_world_consistent(
    world: CompleteMarshalWorld,
    observation: Observation,
) -> bool:
    """Return whether ``world`` passes :func:`validate_complete_world`."""

    try:
        validate_complete_world(world, observation)
    except (CompleteWorldConsistencyError, AttributeError, IndexError, TypeError):
        return False
    return True


__all__ = [
    "ALL_CARDS",
    "CARD_TO_PILE",
    "CompleteMarshalWorld",
    "CompleteWorldConsistencyError",
    "INITIAL_FUGITIVE_CARDS",
    "compile_route_creation_rounds",
    "is_complete_world_consistent",
    "validate_complete_world",
]
