from __future__ import annotations

from http import HTTPStatus

import pytest

from fugitive.game.model import Winner
from fugitive.shared.reproducibility import MAX_SEED
from fugitive.web.session import (
    MAX_SAFE_JSON_INTEGER,
    GameSession,
    WebAPIError,
    parse_web_seed,
)


def make_session(seed: int | str | None) -> GameSession:
    return GameSession(
        session_id="seed-test",
        mode="spectate",
        fugitive_agent="hierarchical-random",
        marshal_agent="hierarchical-random",
        seed=seed,
    )


def finish_for_serialization(session: GameSession) -> None:
    session.engine._finish(Winner.FUGITIVE, "seed serialization test")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("0", 0),
        (str(MAX_SEED), MAX_SEED),
        (0, 0),
        (MAX_SAFE_JSON_INTEGER, MAX_SAFE_JSON_INTEGER),
        (None, None),
    ],
)
def test_seed_parser_accepts_canonical_strings_and_safe_json_integers(
    value: object, expected: int | None
) -> None:
    assert parse_web_seed(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        True,
        1.0,
        -1,
        MAX_SAFE_JSON_INTEGER + 1,
        "",
        "00",
        "01",
        "+1",
        "-1",
        " 1",
        "1 ",
        "1.0",
        "1e3",
        "１",
        str(MAX_SEED + 1),
        "1" * 21,
        "1" * 5_000,
    ],
)
def test_seed_parser_rejects_lossy_or_noncanonical_values(value: object) -> None:
    with pytest.raises(WebAPIError) as caught:
        parse_web_seed(value)

    assert caught.value.status == HTTPStatus.BAD_REQUEST
    assert caught.value.code == "invalid_seed"


def test_terminal_and_reset_seed_values_are_exact_decimal_strings() -> None:
    session = make_session(str(MAX_SEED))

    assert session.seed == MAX_SEED
    assert session.as_dict()["seed"] is None
    finish_for_serialization(session)
    terminal = session.as_dict()
    assert terminal["seed"] == str(MAX_SEED)
    assert isinstance(terminal["seed"], str)

    reset = session.reset(seed="9007199254740993")
    assert session.seed == 9_007_199_254_740_993
    assert reset["seed"] is None
    assert reset["seed_was_supplied"] is True
    finish_for_serialization(session)
    assert session.as_dict()["seed"] == "9007199254740993"
