from __future__ import annotations

from collections.abc import Iterator
from http import HTTPStatus
from http.client import HTTPConnection
import json
import threading
from typing import Any

import pytest

pyspiel = pytest.importorskip("pyspiel")

from fugitive.web.server import create_server


@pytest.fixture
def live_server() -> Iterator[tuple[str, int]]:
    server = create_server("127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    try:
        yield str(host), int(port)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def request(
    server_address: tuple[str, int],
    method: str,
    path: str,
    payload: object | None = None,
) -> tuple[int, dict[str, str], bytes]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"} if body is not None else {}
    connection = HTTPConnection(*server_address, timeout=5)
    try:
        connection.request(method, path, body=body, headers=headers)
        response = connection.getresponse()
        return response.status, dict(response.getheaders()), response.read()
    finally:
        connection.close()


def request_json(
    server_address: tuple[str, int],
    method: str,
    path: str,
    payload: object | None = None,
) -> tuple[int, dict[str, str], dict[str, Any]]:
    status, headers, body = request(server_address, method, path, payload)
    return status, headers, json.loads(body)


def create_spectator_game(server_address: tuple[str, int]) -> dict[str, Any]:
    status, headers, game = request_json(
        server_address,
        "POST",
        "/api/games",
        {
            "mode": "spectate",
            "fugitive_agent": "random",
            "marshal_agent": "random",
            "seed": 17,
            "spectator_view": "public",
        },
    )
    assert status == HTTPStatus.CREATED
    assert headers["X-Content-Type-Options"] == "nosniff"
    assert headers["Cache-Control"] == "no-store"
    return game


def test_http_step_can_drive_repeated_auto_ticks_without_stalling(
    live_server: tuple[str, int],
) -> None:
    game = create_spectator_game(live_server)
    history_length = len(game["history"])

    for _ in range(4):
        status, _headers, game = request_json(
            live_server,
            "POST",
            f"/api/games/{game['id']}/step",
            {},
        )

        assert status == HTTPStatus.OK
        assert game["status"] == "running"
        assert game["can_auto"] is True
        assert game["can_continue"] is False
        assert game["auto_paused"] is False
        assert len(game["history"]) == history_length + 1
        history_length += 1


def test_http_auto_endpoint_retains_batch_safety_pause(
    live_server: tuple[str, int],
) -> None:
    game = create_spectator_game(live_server)

    status, _headers, paused = request_json(
        live_server,
        "POST",
        f"/api/games/{game['id']}/auto",
        {"max_steps": 1},
    )

    assert status == HTTPStatus.OK
    assert paused["status"] == "stalled"
    assert paused["can_auto"] is False
    assert paused["can_continue"] is True
    assert paused["auto_paused"] is True


def test_served_auto_loop_uses_the_single_step_endpoint(
    live_server: tuple[str, int],
) -> None:
    status, headers, body = request(live_server, "GET", "/app.js")
    source = body.decode("utf-8")

    assert status == HTTPStatus.OK
    assert headers["Content-Type"] == "text/javascript; charset=utf-8"
    assert 'await mutateGame("step", {});' in source
    assert 'await mutateGame("auto", { max_steps: 1 });' not in source
