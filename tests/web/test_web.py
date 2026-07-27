from __future__ import annotations

from http import HTTPStatus
import json
from pathlib import Path
import threading
from urllib.request import Request, urlopen

import pytest

from fugitive.game.model import FugitiveAction, Phase, Role
from fugitive.shared.reproducibility import MAX_SEED
from fugitive.web.server import SessionStore, create_server
from fugitive.web.session import GameSession, WebAPIError


def make_session(mode: str, *, seed: int = 73) -> GameSession:
    return GameSession(
        session_id="test-game",
        mode=mode,
        fugitive_agent="hierarchical-random",
        marshal_agent="hierarchical-random",
        seed=seed,
    )


def test_human_fugitive_can_play_a_legal_action() -> None:
    session = make_session("human_fugitive")
    state = session.as_dict()

    assert state["status"] == "awaiting_human"
    assert state["viewer_role"] == Role.FUGITIVE.value
    assert len(state["hand"]) == 9
    actions = state["legal_actions"]["fugitive_actions"]
    assert actions
    assert state["legal_actions"]["can_pass"] is False

    before = session.as_dict()
    with pytest.raises(WebAPIError) as caught:
        session.apply_human_action({"type": "pass"})
    assert caught.value.code == "illegal_action"
    assert session.as_dict() == before

    advanced = session.apply_human_action(actions[0])
    assert advanced["phase"] == Phase.FUGITIVE_OPENING.value
    assert len(advanced["route"]) == 2


def test_human_fugitive_sees_which_hideout_was_uncovered() -> None:
    session = make_session("human_fugitive", seed=3)

    class GuessTwoMarshal:
        @staticmethod
        def choose_draw_pile(observation):
            return observation.legal_draw_piles[0]

        @staticmethod
        def choose_guess(_observation):
            return (2,)

    session.marshal_player = GuessTwoMarshal()
    session.apply_human_action(
        {"type": "fugitive_action", "hideout": 1, "sprint_cards": []}
    )
    state = session.apply_human_action(
        {"type": "fugitive_action", "hideout": 2, "sprint_cards": []}
    )

    assert [slot["hideout"] for slot in state["route"]] == [0, 1, 2]
    assert [slot["revealed"] for slot in state["route"]] == [True, False, True]
    successful_guess = next(
        event
        for event in reversed(state["history"])
        if event["type"] == "guess" and event["success"]
    )
    assert successful_guess["numbers"] == [2]


def test_human_marshal_can_draw_and_guess_without_seeing_private_state() -> None:
    session = make_session("human_marshal")
    state = session.as_dict()
    fugitive_view = session.engine.observation(Role.FUGITIVE)

    assert state["status"] == "awaiting_human"
    assert state["phase"] == Phase.MARSHAL_DRAW.value
    assert state["hands"] == {"fugitive": None, "marshal": []}
    assert fugitive_view.route[1].hideout is not None
    assert state["route"][1]["hideout"] is None
    assert state["route"][1]["sprint_cards"] is None
    assert all(
        event["card"] is None
        for event in state["history"]
        if event["type"] == "draw" and event["role"] == "fugitive"
    )

    while state["phase"] == Phase.MARSHAL_DRAW.value:
        pile = state["legal_actions"]["draw_piles"][0]
        state = session.apply_human_action({"type": "draw", "pile": pile})
    assert len(state["hand"]) == 2

    route_length = len(state["route"]) - 1
    state = session.apply_human_action({"type": "guess", "numbers": [41]})
    guess = next(
        event for event in reversed(state["history"]) if event["type"] == "guess"
    )
    assert guess["route_length"] == route_length


def test_agents_can_finish_a_complete_game() -> None:
    session = make_session("spectate")
    final = session.auto(max_steps=10_000)

    assert final["status"] == "finished"
    assert final["winner"] in ("fugitive", "marshal")
    assert final["reason"] is not None
    assert final["seed"] == "73"


def test_spectator_perspectives_reveal_only_the_selected_information() -> None:
    session = make_session("spectate", seed=0)

    class SprintOpeningFugitive:
        @staticmethod
        def choose_draw_pile(observation):
            return observation.legal_draw_piles[0]

        @staticmethod
        def choose_fugitive_action(observation):
            if len(observation.route) == 1:
                return FugitiveAction(1, (3,))
            return FugitiveAction(2)

    session.fugitive_player = SprintOpeningFugitive()
    session.step()
    session.step()
    fugitive_view = session.engine.observation(Role.FUGITIVE)
    marshal_view = session.engine.observation(Role.MARSHAL)

    omniscient = session.set_spectator_view("omniscient")
    assert omniscient["hands"] == {
        "fugitive": list(fugitive_view.hand),
        "marshal": list(marshal_view.hand),
    }
    assert omniscient["route"][1]["hideout"] == 1
    assert omniscient["route"][1]["sprint_cards"] == [3]

    fugitive = session.set_spectator_view("fugitive")
    assert fugitive["hands"] == {
        "fugitive": list(fugitive_view.hand),
        "marshal": None,
    }
    assert fugitive["route"][1]["hideout"] == 1

    marshal = session.set_spectator_view("marshal")
    assert marshal["hands"] == {
        "fugitive": None,
        "marshal": list(marshal_view.hand),
    }
    assert marshal["route"][1]["hideout"] is None
    assert marshal["route"][1]["sprint_cards"] is None
    assert all(
        event["card"] is None
        for event in marshal["history"]
        if event["type"] == "draw" and event["role"] == "fugitive"
    )

    public = session.set_spectator_view("public")
    assert public["hands"] == {"fugitive": None, "marshal": None}
    assert public["route"][1]["hideout"] is None
    assert public["route"][1]["sprint_cards"] is None
    assert all(
        event["card"] is None
        for event in public["history"]
        if event["type"] == "draw"
    )


def test_finished_session_can_reset_and_start_again() -> None:
    session = make_session("spectate")
    assert session.auto(max_steps=10_000)["status"] == "finished"

    reset = session.reset(seed=74)
    assert reset["status"] == "running"
    assert reset["phase"] == Phase.FUGITIVE_OPENING.value
    assert len(reset["route"]) == 1

    advanced = session.step()
    assert advanced["status"] == "running"
    assert len(advanced["route"]) == 2


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


def test_http_server_serves_the_interface_and_one_game_flow(tmp_path: Path) -> None:
    (tmp_path / "index.html").write_text(
        "<!doctype html><title>Fugitive</title>", encoding="ascii"
    )
    store = SessionStore()
    server = create_server("127.0.0.1", 0, store=store, static_dir=tmp_path)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with urlopen(f"{base}/", timeout=5) as response:
            assert response.status == HTTPStatus.OK
            assert b"Fugitive" in response.read()

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
        assert created["spectator_view"] == "omniscient"

        status, stepped = request_json(
            f"{base}/api/games/{created['id']}/step", method="POST", body={}
        )
        assert status == HTTPStatus.OK
        assert len(stepped["route"]) == 2

        status, public = request_json(
            f"{base}/api/games/{created['id']}/view",
            method="POST",
            body={"spectator_view": "public"},
        )
        assert status == HTTPStatus.OK
        assert public["hands"] == {"fugitive": None, "marshal": None}
        assert public["route"][1]["hideout"] is None
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
