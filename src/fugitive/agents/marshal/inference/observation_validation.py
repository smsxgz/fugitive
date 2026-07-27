"""Lightweight input-contract checks for Marshal belief inference.

The game engine is the authority on game legality.  This module only checks
facts that are directly present in a Marshal observation: public visibility,
history shape, card conservation, and whether the Fugitive had enough cards
by each recorded placement.  It deliberately does not replay the complete
phase machine or prove that a compatible complete hidden world exists.
"""

from __future__ import annotations

from collections import Counter

from fugitive.game.model import Observation, Role
from fugitive.game.rules import GUESSABLE_CARDS, PILE_CARDS
from .world_validation import (
    CARD_TO_PILE,
    INITIAL_FUGITIVE_CARDS,
    CompleteWorldConsistencyError,
    compile_route_creation_rounds,
)


class ObservationContractError(ValueError):
    """Raised when public data violates the inference input contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ObservationContractError(message)


def _valid_round(value: object, current_round: int) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, int)
        and 0 <= value <= current_round
    )


def _validate_route(observation: Observation) -> None:
    _require(bool(observation.route), "the public route must contain Hideout 0")
    start = observation.route[0]
    _require(
        start.index == 0
        and start.hideout == 0
        and start.revealed
        and start.sprint_count == 0
        and start.sprint_cards == (),
        "the public route must start with revealed Hideout 0",
    )
    _require(
        all(slot.index == index for index, slot in enumerate(observation.route)),
        "route indices must be contiguous",
    )

    seen_public: set[int] = set()
    for index, slot in enumerate(observation.route[1:], start=1):
        _require(
            not isinstance(slot.sprint_count, bool)
            and isinstance(slot.sprint_count, int)
            and slot.sprint_count >= 0,
            "Sprint counts must be non-negative integers",
        )
        if slot.revealed:
            _require(
                slot.hideout is not None and slot.sprint_cards is not None,
                "a revealed route slot must expose all identities",
            )
            _require(slot.hideout != 42, "Hideout 42 cannot be uncovered")
        else:
            _require(
                slot.hideout in (None, 42),
                "only unrevealed Hideout 42 may be face up",
            )
            _require(
                slot.sprint_cards is None,
                "an unrevealed route slot cannot expose Sprint identities",
            )
        if slot.hideout == 42:
            _require(
                index == len(observation.route) - 1,
                "Hideout 42 must be the final route slot",
            )
        if slot.sprint_cards is not None:
            _require(
                isinstance(slot.sprint_cards, tuple)
                and len(slot.sprint_cards) == slot.sprint_count,
                "visible Sprint identities must match the public count",
            )
            _require(
                slot.sprint_cards == tuple(sorted(slot.sprint_cards)),
                "visible Sprint cards must be in canonical order",
            )

        identities = (
            (() if slot.hideout is None else (slot.hideout,))
            + (slot.sprint_cards or ())
        )
        for card in identities:
            _require(
                not isinstance(card, bool)
                and isinstance(card, int)
                and 1 <= card <= 42
                and card not in seen_public,
                "public card identities must be distinct cards 1--42",
            )
            seen_public.add(card)


def _validate_draws(observation: Observation) -> None:
    previous_round = 0
    draw_counts: Counter[tuple[Role, int]] = Counter()
    for record in observation.draw_history:
        _require(
            _valid_round(record.round_number, observation.round_number),
            "draw history contains an invalid round",
        )
        _require(
            previous_round <= record.round_number,
            "draw history is not chronological",
        )
        previous_round = record.round_number
        _require(
            not isinstance(record.pile, bool)
            and isinstance(record.pile, int)
            and 0 <= record.pile < 3,
            "draw records must reference pile 0, 1, or 2",
        )
        draw_counts[(record.role, record.round_number)] += 1
        if record.role is Role.FUGITIVE:
            _require(
                record.card is None,
                "a Marshal observation cannot reveal Fugitive draws",
            )
        elif record.role is Role.MARSHAL:
            _require(
                record.card is not None and record.card in PILE_CARDS[record.pile],
                "a Marshal draw must reveal a card from the selected pile",
            )
        else:
            raise ObservationContractError("draw records contain an invalid role")

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
        draw_counts[(Role.FUGITIVE, 1)] == 0,
        "the Fugitive cannot draw during the opening round",
    )
    for (role, round_number), count in draw_counts.items():
        if round_number == 0:
            continue
        maximum = 2 if role is Role.MARSHAL and round_number == 1 else 1
        _require(count <= maximum, "a turn contains too many public draw records")

    marshal_records = tuple(
        record for record in observation.draw_history if record.role is Role.MARSHAL
    )
    marshal_hand = tuple(sorted(observation.hand))
    _require(
        observation.hand == marshal_hand
        and len(marshal_hand) == len(set(marshal_hand))
        and all(card in CARD_TO_PILE for card in marshal_hand),
        "the Marshal hand must contain canonical distinct pile cards",
    )
    _require(
        tuple(sorted(record.card for record in marshal_records)) == marshal_hand,
        "Marshal draw records must exactly identify the Marshal hand",
    )

    for pile, original in enumerate(PILE_CARDS):
        draws = sum(record.pile == pile for record in observation.draw_history)
        _require(
            draws + observation.pile_sizes[pile] == len(original),
            "draw history and pile sizes do not conserve cards",
        )


def _validate_plays_and_resources(
    observation: Observation,
    creation_rounds: tuple[int, ...],
) -> None:
    opening_plays = tuple(
        record for record in observation.play_history if record.round_number == 1
    )
    _require(
        len(opening_plays) <= 2 and all(not record.passed for record in opening_plays),
        "the opening contains at most two placements and no Pass",
    )
    normal_counts = Counter(
        record.round_number
        for record in observation.play_history
        if record.round_number > 1
    )
    _require(
        all(count <= 1 for count in normal_counts.values()),
        "a normal Fugitive turn contains more than one action",
    )
    if any(record.round_number > 1 for record in observation.play_history):
        _require(
            len(opening_plays) == 2,
            "normal play cannot begin before two opening placements",
        )

    fugitive_draws = tuple(
        record for record in observation.draw_history if record.role is Role.FUGITIVE
    )
    used_cards = 0
    visible_pile_cards = [0, 0, 0]
    for index, creation_round in enumerate(creation_rounds[1:], start=1):
        slot = observation.route[index]
        used_cards += 1 + slot.sprint_count
        draws_by_then = sum(
            record.round_number <= creation_round for record in fugitive_draws
        )
        _require(
            used_cards <= len(INITIAL_FUGITIVE_CARDS) + draws_by_then,
            "the Fugitive placed more cards than they could own by that round",
        )

        for card in (
            (() if slot.hideout is None else (slot.hideout,))
            + (slot.sprint_cards or ())
        ):
            pile = CARD_TO_PILE.get(card)
            if pile is not None:
                visible_pile_cards[pile] += 1
        for pile, visible_count in enumerate(visible_pile_cards):
            available = sum(
                record.pile == pile and record.round_number <= creation_round
                for record in fugitive_draws
            )
            _require(
                visible_count <= available,
                "a public Fugitive card was placed before a matching pile draw",
            )


def _validate_guesses(
    observation: Observation,
    creation_rounds: tuple[int, ...],
) -> None:
    previous_round = 0
    previous_route_length = 0
    successful_numbers: set[int] = set()
    for record in observation.guess_history:
        _require(
            bool(record.numbers)
            and record.numbers == tuple(sorted(record.numbers))
            and len(record.numbers) == len(set(record.numbers))
            and all(number in GUESSABLE_CARDS for number in record.numbers),
            "guess history contains an illegal guess",
        )
        _require(
            _valid_round(record.round_number, observation.round_number)
            and record.round_number >= 1
            and previous_round <= record.round_number,
            "guess history contains an invalid round",
        )
        previous_round = record.round_number
        expected_route_length = sum(
            round_number <= record.round_number
            for round_number in creation_rounds[1:]
        )
        _require(
            not isinstance(record.route_length, bool)
            and isinstance(record.route_length, int)
            and previous_route_length <= record.route_length == expected_route_length,
            "guess history contains an invalid route length",
        )
        previous_route_length = record.route_length
        if record.manhunt:
            _require(
                len(record.numbers) == 1,
                "a Manhunt guess must contain exactly one number",
            )
        if record.success:
            successful_numbers.update(record.numbers)

    revealed_numbers = {
        slot.hideout
        for slot in observation.route[1:]
        if slot.revealed and slot.hideout is not None
    }
    _require(
        successful_numbers == revealed_numbers,
        "revealed route cards disagree with successful guesses",
    )


def validate_marshal_observation_contract(observation: Observation) -> None:
    """Validate the public input contract used by Marshal inference.

    Passing this check means that the observation is structurally coherent and
    respects card-resource prefixes visible to the Marshal.  It does not prove
    full rules legality or guarantee non-empty complete-world support.
    """

    _require(
        observation.role is Role.MARSHAL,
        "Marshal inference requires a Marshal observation",
    )
    _require(
        not isinstance(observation.round_number, bool)
        and isinstance(observation.round_number, int)
        and observation.round_number >= 1,
        "the observation has an invalid round",
    )
    _require(
        len(observation.pile_sizes) == 3
        and all(
            not isinstance(size, bool) and isinstance(size, int) and size >= 0
            for size in observation.pile_sizes
        ),
        "pile_sizes must contain three non-negative integers",
    )

    _validate_route(observation)
    _validate_draws(observation)
    try:
        creation_rounds = compile_route_creation_rounds(observation)
    except CompleteWorldConsistencyError as error:
        raise ObservationContractError(str(error)) from error
    _validate_plays_and_resources(observation, creation_rounds)
    _validate_guesses(observation, creation_rounds)

    known_fugitive = set(INITIAL_FUGITIVE_CARDS)
    for slot in observation.route[1:]:
        if slot.hideout is not None:
            known_fugitive.add(slot.hideout)
        known_fugitive.update(slot.sprint_cards or ())
    _require(
        not (set(observation.hand) & known_fugitive),
        "a card cannot belong to both players",
    )
    for pile, cards in enumerate(PILE_CARDS):
        forced = sum(card in cards for card in known_fugitive)
        fugitive_draws = sum(
            record.role is Role.FUGITIVE and record.pile == pile
            for record in observation.draw_history
        )
        _require(
            forced <= fugitive_draws,
            "more public Fugitive cards exist than matching pile draws",
        )


def is_valid_marshal_observation_contract(observation: Observation) -> bool:
    """Return whether ``observation`` passes the lightweight contract check."""

    try:
        validate_marshal_observation_contract(observation)
    except (ObservationContractError, AttributeError, IndexError, TypeError):
        return False
    return True


__all__ = [
    "ObservationContractError",
    "is_valid_marshal_observation_contract",
    "validate_marshal_observation_contract",
]
