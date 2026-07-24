from __future__ import annotations

from http import HTTPStatus
import json
from pathlib import Path
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from fugitive.agents.registry import (
    INTERACTIVE_BIR_MAX_GUESS_CANDIDATES,
    INTERACTIVE_BIR_PARTICLE_COUNT,
)
from fugitive.experiment import RULES_SHA256, RULES_VERSION
from fugitive.model import DrawRecord, GuessRecord, PlayRecord, Role
from fugitive.observation_protocol import observation_to_canonical_data
from fugitive.web import (
    MAX_SEED,
    SEED_DERIVATION_VERSION,
    WEB_TRACE_SCHEMA_VERSION,
    GameSession,
    SessionStore,
    WebAPIError,
    _derive_seed,
    create_server,
)


def make_session(
    *,
    seed: int | str = str(MAX_SEED),
    fugitive_agent: str = "hierarchical-random",
    marshal_agent: str = "hierarchical-random",
) -> GameSession:
    return GameSession(
        session_id="export-test",
        mode="spectate",
        fugitive_agent=fugitive_agent,
        marshal_agent=marshal_agent,
        seed=seed,
    )


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


def assert_trace_matches_engine_history(session: GameSession, trace: list) -> None:
    history = session.engine.result().history
    assert len(trace) == len(history)
    for event, record in zip(trace, history, strict=True):
        assert event["round_number"] == record.round_number
        if isinstance(record, DrawRecord):
            assert event["type"] == "draw"
            assert event["role"] == record.role.value
            assert event["pile"] == record.pile
            assert event["card"] == record.card
        elif isinstance(record, PlayRecord):
            assert event["type"] == (
                "pass" if record.hideout is None else "fugitive_action"
            )
            assert event["hideout"] == record.hideout
            assert event["sprint_cards"] == list(record.sprint_cards)
            assert event["route_index"] == record.route_index
        else:
            assert isinstance(record, GuessRecord)
            assert event["type"] == "guess"
            assert event["numbers"] == list(record.numbers)
            assert event["success"] == record.success
            assert event["route_length"] == record.route_length
            assert event["manhunt"] == record.manhunt


def test_active_game_refuses_full_private_trace_export() -> None:
    session = make_session()

    assert session.as_dict()["can_export"] is False
    with pytest.raises(WebAPIError) as caught:
        session.export_trace()

    assert caught.value.status == HTTPStatus.CONFLICT
    assert caught.value.code == "export_not_ready"


def test_finished_game_export_is_strict_json_and_complete_private_trace() -> None:
    session = make_session()
    session.set_spectator_view("public")
    final = session.auto(max_steps=10_000)
    assert final["status"] == "finished"

    exported = session.export_trace()
    encoded = json.dumps(exported, allow_nan=False, sort_keys=True)
    assert json.loads(encoded) == exported
    assert exported["schema"] == "fugitive.web-trace"
    assert exported["schema_version"] == WEB_TRACE_SCHEMA_VERSION
    assert exported["rules"] == {
        "version": RULES_VERSION,
        "sha256": RULES_SHA256,
    }
    assert exported["seed_derivation"] == {
        "version": SEED_DERIVATION_VERSION,
        "master": str(MAX_SEED),
        "deck": str(_derive_seed(MAX_SEED, "deck")),
        "fugitive": str(_derive_seed(MAX_SEED, "fugitive")),
        "marshal": str(_derive_seed(MAX_SEED, "marshal")),
    }

    agents = exported["agents"]
    assert agents["fugitive"]["registry_name"] == "hierarchical-random"
    assert agents["fugitive"]["parameters"] == {
        "overpay_probability": 0.05,
        "max_low_cost_payments": 8,
        "max_extra_overpayments": 8,
    }
    assert agents["marshal"]["registry_name"] == "hierarchical-random"
    assert agents["marshal"]["parameters"] == {
        "multi_guess_continuation": 0.1,
        "max_guess_size": 4,
        "max_guesses_per_size": 128,
    }

    trace = exported["trace"]
    assert_trace_matches_engine_history(session, trace)
    setup = trace[:5]
    assert all(event["decision"] == 0 for event in setup)
    decisions = [event["decision"] for event in trace[5:]]
    assert decisions == list(range(1, len(decisions) + 1))
    assert all(
        event["card"] is not None
        for event in trace
        if event["type"] == "draw"
    )
    assert exported["outcome"]["decision_count"] == len(decisions)

    expected_fugitive = observation_to_canonical_data(
        session.engine.observation(Role.FUGITIVE)
    )["observation"]
    expected_marshal = observation_to_canonical_data(
        session.engine.observation(Role.MARSHAL)
    )["observation"]
    assert exported["observations"] == {
        "fugitive": expected_fugitive,
        "marshal": expected_marshal,
    }


def test_terminated_game_exports_saved_bir_parameters_after_agent_release() -> None:
    seed = 9_007_199_254_740_993
    session = make_session(
        seed=str(seed),
        fugitive_agent="belief-informed-random",
        marshal_agent="belief-informed-random",
    )
    session.step()
    terminated = session.terminate()

    assert terminated["can_export"] is True
    assert session.fugitive_player is None
    assert session.marshal_player is None
    exported = session.export_trace()
    json.dumps(exported, allow_nan=False)
    assert exported["seed_derivation"]["master"] == str(seed)
    assert exported["outcome"] == {
        "status": "terminated",
        "winner": None,
        "reason": "terminated_by_user",
        "rounds": 1,
        "decision_count": 1,
    }
    marshal_parameters = exported["agents"]["marshal"]["parameters"]
    assert marshal_parameters["particle_count"] == INTERACTIVE_BIR_PARTICLE_COUNT
    assert (
        marshal_parameters["max_guess_candidates"]
        == INTERACTIVE_BIR_MAX_GUESS_CANDIDATES
    )
    assert marshal_parameters["min_unique_particles"] <= marshal_parameters[
        "particle_count"
    ]
    assert len(exported["trace"]) == 6
    assert all(
        event["card"] is not None
        for event in exported["trace"]
        if event["type"] == "draw"
    )


def test_http_export_rejects_active_game_then_serves_terminated_trace(
    tmp_path: Path,
) -> None:
    (tmp_path / "index.html").write_text("<!doctype html>", encoding="ascii")
    store = SessionStore()
    server = create_server("127.0.0.1", 0, store=store, static_dir=tmp_path)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        _, created = request_json(
            f"{base}/api/games",
            method="POST",
            body={
                "mode": "spectate",
                "fugitive_agent": "hierarchical-random",
                "marshal_agent": "hierarchical-random",
                "seed": str(MAX_SEED),
            },
        )
        session_id = created["id"]
        with pytest.raises(HTTPError) as active:
            request_json(f"{base}/api/games/{session_id}/export")
        assert active.value.code == HTTPStatus.CONFLICT
        error = json.loads(active.value.read())
        assert error["error"]["code"] == "export_not_ready"

        request_json(
            f"{base}/api/games/{session_id}/terminate",
            method="POST",
            body={},
        )
        status, exported = request_json(
            f"{base}/api/games/{session_id}/export"
        )
        assert status == HTTPStatus.OK
        assert exported["outcome"]["status"] == "terminated"
        assert exported["seed_derivation"]["master"] == str(MAX_SEED)
        assert exported == store.get(session_id).export_trace()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
