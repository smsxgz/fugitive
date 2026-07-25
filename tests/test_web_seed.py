from __future__ import annotations

from http import HTTPStatus
import json
from pathlib import Path
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from fugitive.engine import GameEngine
from fugitive.model import Role, Winner
from fugitive.reproducibility import MAX_SEED, derive_seed
from fugitive.web import (
    MAX_SAFE_JSON_INTEGER,
    GameSession,
    SessionStore,
    WebAPIError,
    _parse_seed,
    create_server,
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
    assert _parse_seed(value) == expected


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
        _parse_seed(value)

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


def test_omitted_random_seed_uses_the_same_string_protocol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("fugitive.web.secrets.randbits", lambda _bits: MAX_SEED)

    session = make_session(None)
    assert session.seed == MAX_SEED
    assert session.as_dict()["seed"] is None
    assert session.as_dict()["seed_was_supplied"] is False

    finish_for_serialization(session)
    assert session.as_dict()["seed"] == str(MAX_SEED)


def test_master_seed_derives_distinct_replay_stable_random_streams() -> None:
    master = MAX_SEED
    derived = {
        domain: derive_seed(master, domain)
        for domain in ("deck", "fugitive", "marshal")
    }

    assert len(set(derived.values())) == 3
    assert all(0 <= seed <= MAX_SEED for seed in derived.values())
    assert derived == {
        domain: derive_seed(master, domain)
        for domain in ("deck", "fugitive", "marshal")
    }

    first = make_session(str(master))
    second = make_session(str(master))
    expected_engine = GameEngine(seed=derived["deck"])
    assert first.engine.observation(Role.FUGITIVE) == expected_engine.observation(
        Role.FUGITIVE
    )
    assert first.step() == second.step()
    assert first.step() == second.step()


def request_json(url: str, *, method: str = "GET", body: object | None = None):
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=5) as response:
        return response.status, json.loads(response.read())


def test_http_seed_round_trip_never_serializes_a_64_bit_json_number(
    tmp_path: Path,
) -> None:
    (tmp_path / "index.html").write_text("<!doctype html>", encoding="ascii")
    store = SessionStore()
    server = create_server("127.0.0.1", 0, store=store, static_dir=tmp_path)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        status, created = request_json(
            f"{base}/api/games",
            method="POST",
            body={
                "mode": "spectate",
                "fugitive_agent": "hierarchical-random",
                "marshal_agent": "hierarchical-random",
                "seed": str(MAX_SEED),
            },
        )
        assert status == HTTPStatus.CREATED
        assert created["seed"] is None

        session = store.get(created["id"])
        assert session.seed == MAX_SEED
        finish_for_serialization(session)
        status, terminal = request_json(f"{base}/api/games/{created['id']}")
        assert status == HTTPStatus.OK
        assert terminal["seed"] == str(MAX_SEED)
        assert isinstance(terminal["seed"], str)

        with pytest.raises(HTTPError) as unsafe_number:
            request_json(
                f"{base}/api/games",
                method="POST",
                body={"seed": MAX_SAFE_JSON_INTEGER + 1},
            )
        assert unsafe_number.value.code == HTTPStatus.BAD_REQUEST
        error = json.loads(unsafe_number.value.read())
        assert error["error"]["code"] == "invalid_seed"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_frontend_keeps_seed_as_decimal_text() -> None:
    static_dir = Path(__file__).parents[1] / "src" / "fugitive" / "web_static"
    javascript = (static_dir / "app.js").read_text(encoding="utf-8")
    html = (static_dir / "index.html").read_text(encoding="utf-8")

    assert "const value = Number(raw)" not in javascript
    assert "return raw;" in javascript
    assert 'typeof game.seed === "string"' in javascript
    assert 'id="gameSeed" name="seed" type="text"' in html
