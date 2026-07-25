from __future__ import annotations

from http import HTTPStatus
import json
import logging
from pathlib import Path
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from fugitive.agents.registry import INTERACTIVE_BIR_PARTICLE_COUNT
from fugitive.model import Phase
from fugitive.particle_inference.state import DEFAULT_PARTICLE_COUNT
from fugitive.web import GameSession, SessionStore, WebAPIError, create_server


HR_GAME = {
    "mode": "spectate",
    "fugitive_agent": "hierarchical-random",
    "marshal_agent": "hierarchical-random",
    "seed": 7,
}


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def make_session(*, auto_step_limit: int = 10_000) -> GameSession:
    return GameSession(
        session_id="resilience-test",
        mode="spectate",
        fugitive_agent="hierarchical-random",
        marshal_agent="hierarchical-random",
        seed=7,
        auto_step_limit=auto_step_limit,
    )


def test_web_session_defaults_to_full_agent_profile() -> None:
    session = GameSession(
        session_id="full-profile-test",
        mode="spectate",
        fugitive_agent="belief-informed-random",
        marshal_agent="belief-informed-random",
        seed=7,
    )

    assert session.marshal_player is not None
    assert session.marshal_player.particle_count == DEFAULT_PARTICLE_COUNT
    state = session.as_dict()
    assert state["execution_profile"] == "full"
    assert state["agents"]["marshal"]["profile"] == "default"
    assert (
        state["agents"]["marshal"]["parameters"]["particle_count"]
        == DEFAULT_PARTICLE_COUNT
    )


def test_quick_profile_is_explicit_and_reset_preserves_it() -> None:
    session = GameSession(
        session_id="quick-profile-test",
        mode="spectate",
        fugitive_agent="belief-informed-random",
        marshal_agent="belief-informed-random",
        seed=7,
        execution_profile="quick",
    )

    assert session.marshal_player is not None
    assert session.marshal_player.particle_count == INTERACTIVE_BIR_PARTICLE_COUNT
    assert session.as_dict()["agents"]["marshal"]["profile"] == "interactive"

    reset = session.reset(seed=8)

    assert reset["execution_profile"] == "quick"
    assert reset["agents"]["marshal"]["profile"] == "interactive"
    assert session.marshal_player is not None
    assert session.marshal_player.particle_count == INTERACTIVE_BIR_PARTICLE_COUNT


def test_invalid_execution_profile_is_rejected_without_resetting_session() -> None:
    session = make_session()
    original_engine = session.engine

    with pytest.raises(WebAPIError) as created:
        GameSession(
            session_id="invalid-profile-test",
            mode="spectate",
            fugitive_agent="hierarchical-random",
            marshal_agent="hierarchical-random",
            seed=7,
            execution_profile="slow",
        )
    assert created.value.status == HTTPStatus.BAD_REQUEST
    assert created.value.code == "invalid_execution_profile"

    with pytest.raises(WebAPIError) as reset:
        session.reset(execution_profile="slow")
    assert reset.value.status == HTTPStatus.BAD_REQUEST
    assert reset.value.code == "invalid_execution_profile"
    assert session.engine is original_engine
    assert session.execution_profile == "full"

    with pytest.raises(WebAPIError) as null_reset:
        session.reset(execution_profile=None)
    assert null_reset.value.code == "invalid_execution_profile"


def test_session_store_validates_and_forwards_execution_profile_payload() -> None:
    store = SessionStore()
    quick = store.create({**HR_GAME, "execution_profile": "quick"})

    assert quick.execution_profile == "quick"
    assert quick.as_dict()["agents"]["marshal"]["profile"] == "interactive"

    with pytest.raises(WebAPIError) as invalid:
        store.create({**HR_GAME, "execution_profile": "benchmark"})
    assert invalid.value.status == HTTPStatus.BAD_REQUEST
    assert invalid.value.code == "invalid_execution_profile"


def request_json(
    url: str, *, method: str = "GET", body: object | None = None
) -> tuple[int, dict[str, object]]:
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=5) as response:
        return response.status, json.loads(response.read())


def start_server(
    tmp_path: Path,
    *,
    store: SessionStore,
    logger: logging.Logger | None = None,
):
    (tmp_path / "index.html").write_text("<!doctype html>", encoding="ascii")
    server = create_server(
        "127.0.0.1",
        0,
        store=store,
        static_dir=tmp_path,
        error_logger=logger,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread, f"http://127.0.0.1:{server.server_address[1]}"


def stop_server(server, thread: threading.Thread) -> None:
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


def test_auto_limit_is_persistent_stalled_state_until_continue_or_terminate() -> None:
    session = make_session(auto_step_limit=2)
    session._agent_step = lambda: None  # type: ignore[method-assign]

    stalled = session.auto(max_steps=2)

    assert stalled["status"] == "stalled"
    assert stalled["auto_paused"] is True
    assert stalled["pause_reason"] == "auto_step_limit"
    assert stalled["can_continue"] is True
    assert stalled["can_step"] is False
    assert session.as_dict()["status"] == "stalled"

    with pytest.raises(WebAPIError) as blocked:
        session.step()
    assert blocked.value.code == "session_stalled"

    continued = session.continue_after_stall(max_steps=1)
    assert continued["status"] == "stalled"
    assert continued["auto_steps"] == 1

    terminated = session.terminate()
    assert terminated["status"] == "terminated"
    assert terminated["reason"] == "terminated_by_user"
    assert terminated["winner"] is None
    assert terminated["can_continue"] is False
    assert terminated["can_terminate"] is False
    assert session.fugitive_player is None
    assert session.marshal_player is None

    reset = session.reset(seed=8)
    assert reset["status"] == "running"
    assert reset["phase"] == Phase.FUGITIVE_OPENING.value
    assert session.fugitive_player is not None
    assert session.marshal_player is not None


def test_reset_invokes_optional_agent_cleanup_protocol() -> None:
    session = make_session()

    class DisposableAgent:
        def __init__(self) -> None:
            self.clears = 0
            self.closes = 0

        def clear_cache(self) -> None:
            self.clears += 1

        def close(self) -> None:
            self.closes += 1

    fugitive = DisposableAgent()
    marshal = DisposableAgent()
    session.fugitive_player = fugitive
    session.marshal_player = marshal

    session.reset(seed=9)

    assert (fugitive.clears, fugitive.closes) == (1, 1)
    assert (marshal.clears, marshal.closes) == (1, 1)
    assert session.fugitive_player is not fugitive
    assert session.marshal_player is not marshal


def test_session_store_enforces_lru_capacity_and_releases_evicted_agents() -> None:
    clock = FakeClock()
    store = SessionStore(
        idle_ttl_seconds=None,
        max_active_sessions=2,
        clock=clock,
    )
    first = store.create(HR_GAME)
    clock.advance(1)
    second = store.create({**HR_GAME, "seed": 8})
    clock.advance(1)
    assert store.get(first.id) is first
    clock.advance(1)

    third = store.create({**HR_GAME, "seed": 9})

    assert store.active_count == 2
    assert store.get(first.id) is first
    assert store.get(third.id) is third
    assert second.fugitive_player is None
    assert second.marshal_player is None
    with pytest.raises(WebAPIError) as evicted:
        store.get(second.id)
    assert evicted.value.code == "game_not_found"


def test_session_delete_waits_for_an_active_request_lease() -> None:
    store = SessionStore(idle_ttl_seconds=None)
    session = store.create(HR_GAME)

    with store.lease(session.id) as leased:
        assert leased is session
        store.delete(session.id)
        assert session.fugitive_player is not None
        assert session.marshal_player is not None
        with pytest.raises(WebAPIError) as removed:
            store.get(session.id)
        assert removed.value.code == "game_not_found"

    assert session.fugitive_player is None
    assert session.marshal_player is None


def test_expiry_counts_a_leased_entry_before_its_deferred_close() -> None:
    clock = FakeClock()
    store = SessionStore(
        idle_ttl_seconds=10,
        clock=clock,
    )
    session = store.create(HR_GAME)

    with store.lease(session.id):
        clock.advance(10)

        assert store.cleanup_expired() == 1
        assert store.active_count == 0
        assert store.cleanup_expired() == 0
        assert session.fugitive_player is not None
        assert session.marshal_player is not None

    assert session.fugitive_player is None
    assert session.marshal_player is None


def test_create_lease_keeps_the_creation_response_safe_during_lru_eviction() -> None:
    store = SessionStore(
        idle_ttl_seconds=None,
        max_active_sessions=1,
    )

    with store.create_lease(HR_GAME) as first:
        second = store.create({**HR_GAME, "seed": 8})

        with pytest.raises(WebAPIError) as evicted:
            store.get(first.id)
        assert evicted.value.code == "game_not_found"
        assert first.fugitive_player is not None
        assert first.marshal_player is not None
        assert first.as_dict()["id"] == first.id

    assert first.fugitive_player is None
    assert first.marshal_player is None
    assert store.get(second.id) is second


def test_http_creation_holds_lease_until_initial_state_is_serialized(
    tmp_path: Path,
) -> None:
    serialization_started = threading.Event()
    allow_serialization = threading.Event()

    class BlockingCreationSession(GameSession):
        block_initial_state = False

        def as_dict(self) -> dict[str, object]:
            if self.block_initial_state:
                serialization_started.set()
                if not allow_serialization.wait(timeout=5):
                    raise TimeoutError("test did not release creation serialization")
            return super().as_dict()

    sessions: list[BlockingCreationSession] = []

    def session_factory(**kwargs) -> BlockingCreationSession:
        session = BlockingCreationSession(**kwargs)
        session.block_initial_state = not sessions
        sessions.append(session)
        return session

    store = SessionStore(
        idle_ttl_seconds=None,
        max_active_sessions=1,
        session_factory=session_factory,
    )
    server, server_thread, base = start_server(tmp_path, store=store)
    response: list[tuple[int, dict[str, object]]] = []
    request_error: list[BaseException] = []

    def create_over_http() -> None:
        try:
            response.append(
                request_json(f"{base}/api/games", method="POST", body=HR_GAME)
            )
        except BaseException as exc:  # pragma: no cover - assertion reports it
            request_error.append(exc)

    request_thread = threading.Thread(target=create_over_http)
    request_thread.start()
    try:
        assert serialization_started.wait(timeout=5)
        first = sessions[0]
        store.create({**HR_GAME, "seed": 8})

        assert first.fugitive_player is not None
        assert first.marshal_player is not None
        allow_serialization.set()
        request_thread.join(timeout=5)

        assert not request_thread.is_alive()
        assert request_error == []
        assert response[0][0] == HTTPStatus.CREATED
        assert response[0][1]["id"] == first.id
        assert first.fugitive_player is None
        assert first.marshal_player is None
    finally:
        allow_serialization.set()
        request_thread.join(timeout=5)
        stop_server(server, server_thread)


def test_session_store_expires_idle_sessions_with_injected_clock() -> None:
    clock = FakeClock()
    store = SessionStore(
        idle_ttl_seconds=10,
        max_active_sessions=4,
        clock=clock,
    )
    session = store.create(HR_GAME)
    clock.advance(9)
    assert store.get(session.id) is session
    clock.advance(10)

    assert store.cleanup_expired() == 1
    assert store.active_count == 0
    assert session.fugitive_player is None
    assert session.marshal_player is None
    with pytest.raises(WebAPIError) as expired:
        store.get(session.id)
    assert expired.value.code == "game_not_found"


def test_http_continue_terminate_and_delete_lifecycle(tmp_path: Path) -> None:
    store = SessionStore(auto_step_limit=1)
    server, thread, base = start_server(tmp_path, store=store)
    try:
        status, created = request_json(
            f"{base}/api/games", method="POST", body=HR_GAME
        )
        assert status == HTTPStatus.CREATED
        session_id = str(created["id"])
        session = store.get(session_id)
        session._agent_step = lambda: None  # type: ignore[method-assign]

        _, stalled = request_json(
            f"{base}/api/games/{session_id}/auto",
            method="POST",
            body={"max_steps": 1},
        )
        assert stalled["status"] == "stalled"
        _, persisted = request_json(f"{base}/api/games/{session_id}")
        assert persisted["status"] == "stalled"

        _, continued = request_json(
            f"{base}/api/games/{session_id}/continue",
            method="POST",
            body={"max_steps": 1},
        )
        assert continued["status"] == "stalled"

        _, terminated = request_json(
            f"{base}/api/games/{session_id}/terminate",
            method="POST",
            body={},
        )
        assert terminated["status"] == "terminated"

        status, deleted = request_json(
            f"{base}/api/games/{session_id}", method="DELETE"
        )
        assert status == HTTPStatus.OK
        assert deleted == {"id": session_id, "deleted": True}
        with pytest.raises(HTTPError) as missing:
            request_json(f"{base}/api/games/{session_id}")
        assert missing.value.code == HTTPStatus.NOT_FOUND
    finally:
        stop_server(server, thread)


@pytest.mark.parametrize(
    ("store_method", "http_method", "path", "body"),
    [
        ("lease", "GET", "/api/games/boom", None),
        ("create_lease", "POST", "/api/games", {}),
        ("delete", "DELETE", "/api/games/boom", None),
    ],
)
def test_unexpected_http_exception_returns_stable_500_and_logs_error_id(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    store_method: str,
    http_method: str,
    path: str,
    body: object | None,
) -> None:
    store = SessionStore()

    def fail(*_args, **_kwargs):
        raise RuntimeError("private failure detail")

    setattr(store, store_method, fail)
    logger = logging.getLogger(
        f"fugitive.web.resilience-test.{store_method}"
    )
    caplog.set_level(logging.ERROR, logger=logger.name)
    server, thread, base = start_server(tmp_path, store=store, logger=logger)
    try:
        with pytest.raises(HTTPError) as failed:
            request_json(
                f"{base}{path}",
                method=http_method,
                body=body,
            )
        assert failed.value.code == HTTPStatus.INTERNAL_SERVER_ERROR
        payload = json.loads(failed.value.read())
        error = payload["error"]
        assert error["code"] == "internal_server_error"
        assert error["message"] == "an unexpected server error occurred"
        assert len(error["error_id"]) == 32
        assert "private failure detail" not in error["message"]
        record = next(
            item for item in caplog.records if error["error_id"] in item.message
        )
        assert record.exc_info is not None
        assert record.exc_info[0] is RuntimeError
    finally:
        stop_server(server, thread)


def test_guess_history_records_public_route_length_at_guess_time() -> None:
    session = GameSession(
        session_id="route-length-test",
        mode="human_marshal",
        fugitive_agent="hierarchical-random",
        marshal_agent="hierarchical-random",
        seed=3,
    )
    while session.engine.phase is Phase.MARSHAL_DRAW:
        pile = session.as_dict()["legal_actions"]["draw_piles"][0]
        session.apply_human_action({"type": "draw", "pile": pile})
    route_length = len(session.engine.observation(session.human_role).route) - 1

    state = session.apply_human_action({"type": "guess", "numbers": [41]})

    event = next(item for item in reversed(state["history"]) if item["type"] == "guess")
    assert event["route_length"] == route_length
