from __future__ import annotations

from http import HTTPStatus
import json
from pathlib import Path
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from fugitive.agents.registry import MARSHAL_AGENT_REGISTRY
from fugitive.model import FugitiveAction, Phase, Role
from fugitive.rules import legal_fugitive_actions
from fugitive.web import (
    DEFAULT_FUGITIVE_AGENT,
    DEFAULT_MARSHAL_AGENT,
    GameSession,
    SessionStore,
    WebAPIError,
    agent_catalog,
    create_server,
)


def make_session(mode: str, *, seed: int = 73) -> GameSession:
    return GameSession(
        session_id="test-game",
        mode=mode,
        fugitive_agent="hierarchical-random",
        marshal_agent="hierarchical-random",
        seed=seed,
    )


def test_agent_catalog_is_role_specific_and_marks_expensive_agents() -> None:
    catalog = agent_catalog()

    assert catalog["defaults"] == {
        "fugitive": "belief-informed-random",
        "marshal": "belief-informed-random",
    }
    assert DEFAULT_FUGITIVE_AGENT == "belief-informed-random"
    assert DEFAULT_MARSHAL_AGENT == "belief-informed-random"
    assert {item["id"] for item in catalog["marshal"]} == {
        "hierarchical-random",
        "belief-informed-random",
        "route-count-random",
        "constructive-belief-informed-random",
        "mcmc-belief-informed-random",
    }
    fugitive = {item["id"]: item for item in catalog["fugitive"]}
    assert fugitive["hierarchical-random"]["label"] == (
        "Hierarchical Legal Random (HR-1)"
    )
    assert fugitive["belief-informed-random"]["label"] == (
        "Belief-Informed Random (BIR-1)"
    )
    assert fugitive["belief-informed-random"]["role"] == "fugitive"
    marshal = {item["id"]: item for item in catalog["marshal"]}
    assert marshal["route-count-random"]["label"] == (
        "Route-Count Random (HR-1.1)"
    )
    assert marshal["constructive-belief-informed-random"]["label"] == (
        "Constructive Belief-Informed Random (BIR-2)"
    )
    assert marshal["mcmc-belief-informed-random"]["label"] == (
        "MCMC Belief-Informed Random (BIR-3)"
    )
    assert all(item["expensive"] is False for item in catalog["fugitive"])
    assert all(item["role"] == "marshal" for item in catalog["marshal"])
    assert all(
        item["expensive"] == MARSHAL_AGENT_REGISTRY[item["id"]].expensive
        for item in catalog["marshal"]
    )
    assert [item["id"] for item in catalog["spectator_views"]] == [
        "omniscient",
        "fugitive",
        "marshal",
        "public",
    ]


def test_session_store_uses_bir_for_both_default_players() -> None:
    session = SessionStore().create({})

    assert session.mode == "spectate"
    assert session.fugitive_agent_name == "belief-informed-random"
    assert session.marshal_agent_name == "belief-informed-random"
    assert len(session.step()["route"]) == 2


def test_human_fugitive_gets_bounded_choices_but_engine_validates_actions() -> None:
    session = make_session("human_fugitive")
    state = session.as_dict()

    assert state["status"] == "awaiting_human"
    assert state["phase"] == Phase.FUGITIVE_OPENING.value
    assert state["viewer_role"] == Role.FUGITIVE.value
    assert len(state["hand"]) == 9
    legal = state["legal_actions"]
    actions = legal["fugitive_actions"]
    assert actions
    assert legal["candidate_hideouts"]
    assert legal["can_pass"] is False
    assert legal["representative_actions_only"] is True
    assert all(action["type"] != "pass" for action in actions)

    before = session.as_dict()
    with pytest.raises(WebAPIError) as caught:
        session.apply_human_action({"type": "pass"})
    assert caught.value.status == HTTPStatus.UNPROCESSABLE_ENTITY
    assert caught.value.code == "illegal_action"
    assert session.as_dict() == before

    next_state = session.apply_human_action(actions[0])
    assert next_state["phase"] == Phase.FUGITIVE_OPENING.value
    assert len(next_state["route"]) == 2

    # The displayed list is deliberately bounded for UI size.  A different
    # exact legal overpayment is still accepted by the endpoint.
    another = make_session("human_fugitive")
    exact = legal_fugitive_actions(another.engine.observation(Role.FUGITIVE))
    displayed = {
        (action.get("hideout"), tuple(action.get("sprint_cards", ())))
        for action in another.as_dict()["legal_actions"]["fugitive_actions"]
    }
    non_representative = next(
        action
        for action in exact
        if (action.hideout, action.sprint_cards) not in displayed
    )
    accepted = another.apply_human_action(
        {
            "type": "fugitive_action",
            "hideout": non_representative.hideout,
            "sprint_cards": list(non_representative.sprint_cards),
        }
    )
    assert len(accepted["route"]) == 2


def test_human_fugitive_route_identifies_the_hideout_uncovered_by_agent() -> None:
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

    successful_guess = next(
        event
        for event in reversed(state["history"])
        if event["type"] == "guess" and event["success"]
    )
    assert successful_guess["numbers"] == [2]
    assert [slot["hideout"] for slot in state["route"]] == [0, 1, 2]
    assert [slot["revealed"] for slot in state["route"]] == [True, False, True]


def test_human_marshal_view_never_contains_fugitive_private_state() -> None:
    session = make_session("human_marshal")
    state = session.as_dict()
    fugitive_view = session.engine.observation(Role.FUGITIVE)
    marshal_view = session.engine.observation(Role.MARSHAL)

    # The Fugitive agent has completed both opening plays, then automation
    # stops before the human Marshal's first draw.
    assert state["phase"] == Phase.MARSHAL_DRAW.value
    assert state["status"] == "awaiting_human"
    assert state["hand"] == list(marshal_view.hand) == []
    assert state["hands"] == {"fugitive": None, "marshal": []}
    assert state["spectator_view"] is None
    assert len(state["route"]) == 3
    assert fugitive_view.route[1].hideout is not None
    assert state["route"][1]["hideout"] is None
    assert state["route"][1]["sprint_cards"] is None
    fugitive_draws = [
        event for event in state["history"] if event["type"] == "draw" and event["role"] == "fugitive"
    ]
    assert len(fugitive_draws) == 5
    assert all(event["card"] is None for event in fugitive_draws)

    pile = state["legal_actions"]["draw_piles"][0]
    after_draw = session.apply_human_action({"type": "draw", "pile": pile})
    assert len(after_draw["hand"]) == 1
    own_draws = [
        event for event in after_draw["history"] if event["type"] == "draw" and event["role"] == "marshal"
    ]
    assert own_draws[-1]["card"] == after_draw["hand"][0]


def test_spectator_defaults_to_omniscient_and_can_run_to_a_winner() -> None:
    session = make_session("spectate")
    initial = session.as_dict()
    fugitive_view = session.engine.observation(Role.FUGITIVE)
    marshal_view = session.engine.observation(Role.MARSHAL)

    assert initial["viewer_role"] == "spectator"
    assert initial["spectator_view"] == "omniscient"
    assert initial["hand"] == list(fugitive_view.hand)
    assert initial["hands"] == {
        "fugitive": list(fugitive_view.hand),
        "marshal": list(marshal_view.hand),
    }
    assert initial["seed"] is None
    assert all(
        event["card"] is not None
        for event in initial["history"]
        if event["type"] == "draw"
    )

    after_step = session.step()
    assert len(after_step["route"]) == 2
    assert after_step["route"][1]["hideout"] is not None
    play = [event for event in after_step["history"] if event["type"] == "fugitive_action"][-1]
    assert play["hideout"] == after_step["route"][1]["hideout"]
    assert play["sprint_cards"] == after_step["route"][1]["sprint_cards"]

    final = session.auto(max_steps=10_000)
    assert final["status"] == "finished"
    assert final["winner"] in ("fugitive", "marshal")
    assert final["reason"] is not None
    assert final["auto_paused"] is False
    assert final["seed"] == "73"


def test_switching_spectator_perspective_is_read_only_and_role_safe() -> None:
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
    before_fugitive = session.engine.observation(Role.FUGITIVE)
    before_marshal = session.engine.observation(Role.MARSHAL)
    before_events = tuple(dict(event) for event in session._events)

    omniscient = session.set_spectator_view("omniscient")
    assert omniscient["route"][1]["hideout"] == before_fugitive.route[1].hideout
    assert omniscient["hands"] == {
        "fugitive": list(before_fugitive.hand),
        "marshal": list(before_marshal.hand),
    }
    sprinted_index = next(
        slot.index for slot in before_fugitive.route if slot.sprint_count > 0
    )
    assert omniscient["route"][sprinted_index]["sprint_cards"] == list(
        before_fugitive.route[sprinted_index].sprint_cards
    )
    assert all(
        event["card"] is not None
        for event in omniscient["history"]
        if event["type"] == "draw"
    )

    fugitive = session.set_spectator_view("fugitive")
    assert fugitive["viewer_role"] == "spectator"
    assert fugitive["hand"] == list(before_fugitive.hand)
    assert fugitive["hands"] == {
        "fugitive": list(before_fugitive.hand),
        "marshal": None,
    }
    assert fugitive["route"][1]["hideout"] == before_fugitive.route[1].hideout

    marshal = session.set_spectator_view("marshal")
    assert marshal["hand"] == list(before_marshal.hand)
    assert marshal["hands"] == {
        "fugitive": None,
        "marshal": list(before_marshal.hand),
    }
    assert marshal["route"][1]["hideout"] is None
    assert all(
        event["card"] is None
        for event in marshal["history"]
        if event["type"] == "draw" and event["role"] == "fugitive"
    )

    public = session.set_spectator_view("public")
    assert public["hand"] == []
    assert public["hands"] == {"fugitive": None, "marshal": None}
    assert public["route"][1]["hideout"] is None
    assert public["route"][sprinted_index]["sprint_cards"] is None
    assert all(
        event["card"] is None
        for event in public["history"]
        if event["type"] == "draw"
    )
    for state in (omniscient, fugitive, marshal, public):
        assert [slot["revealed"] for slot in state["route"]] == [
            True,
            False,
            False,
        ]
        assert all(isinstance(slot["revealed"], bool) for slot in state["route"])

    assert session.engine.observation(Role.FUGITIVE) == before_fugitive
    assert session.engine.observation(Role.MARSHAL) == before_marshal
    assert tuple(dict(event) for event in session._events) == before_events


def test_invalid_or_human_spectator_view_switch_is_rejected() -> None:
    spectator = make_session("spectate")
    with pytest.raises(WebAPIError) as invalid:
        spectator.set_spectator_view("xray")
    assert invalid.value.status == HTTPStatus.BAD_REQUEST
    assert invalid.value.code == "invalid_spectator_view"

    human = make_session("human_fugitive")
    before = human.as_dict()
    with pytest.raises(WebAPIError) as unavailable:
        human.set_spectator_view("omniscient")
    assert unavailable.value.status == HTTPStatus.CONFLICT
    assert unavailable.value.code == "spectator_view_not_available"
    assert human.as_dict() == before


def test_spectator_rejects_human_actions_and_human_game_rejects_step() -> None:
    spectator = make_session("spectate")
    with pytest.raises(WebAPIError) as action_error:
        spectator.apply_human_action({"type": "pass"})
    assert action_error.value.code == "human_action_not_available"

    human = make_session("human_fugitive")
    with pytest.raises(WebAPIError) as step_error:
        human.step()
    assert step_error.value.code == "step_not_available"


def test_reset_without_seed_uses_fresh_unrevealed_seed() -> None:
    session = make_session("spectate")
    original_seed = session.seed

    state = session.reset()

    assert session.seed != original_seed
    assert state["seed"] is None
    assert state["seed_was_supplied"] is False
    assert state["phase"] == Phase.FUGITIVE_OPENING.value


def _request_json(url: str, *, method: str = "GET", body: object | None = None):
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=5) as response:
        return response.status, json.loads(response.read())


def test_http_api_and_static_file_serving(tmp_path: Path) -> None:
    (tmp_path / "index.html").write_text("<!doctype html><title>Fugitive</title>", encoding="ascii")
    store = SessionStore()
    server = create_server("127.0.0.1", 0, store=store, static_dir=tmp_path)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with urlopen(f"{base}/", timeout=5) as response:
            assert response.status == HTTPStatus.OK
            assert b"Fugitive" in response.read()

        status, catalog = _request_json(f"{base}/api/agents")
        assert status == HTTPStatus.OK
        assert catalog["fugitive"]
        assert catalog["defaults"]["fugitive"] == "belief-informed-random"

        status, game = _request_json(
            f"{base}/api/games",
            method="POST",
            body={
                "mode": "spectate",
                "fugitive_agent": "hierarchical-random",
                "marshal_agent": "hierarchical-random",
                "seed": 19,
            },
        )
        assert status == HTTPStatus.CREATED
        assert game["seed"] is None
        assert game["spectator_view"] == "omniscient"

        status, stepped = _request_json(
            f"{base}/api/games/{game['id']}/step", method="POST", body={}
        )
        assert status == HTTPStatus.OK
        assert len(stepped["route"]) == 2

        status, public = _request_json(
            f"{base}/api/games/{game['id']}/view",
            method="POST",
            body={"spectator_view": "public"},
        )
        assert status == HTTPStatus.OK
        assert public["spectator_view"] == "public"
        assert public["hands"] == {"fugitive": None, "marshal": None}
        assert public["route"][1]["hideout"] is None

        with pytest.raises(HTTPError) as invalid:
            _request_json(
                f"{base}/api/games/{game['id']}/action",
                method="POST",
                body={"type": "pass"},
            )
        assert invalid.value.code == HTTPStatus.CONFLICT
        error = json.loads(invalid.value.read())
        assert error["error"]["code"] == "human_action_not_available"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_default_server_serves_packaged_interface() -> None:
    server = create_server("127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}/"
        with urlopen(url, timeout=5) as response:
            assert response.status == HTTPStatus.OK
            assert b"<!doctype html>" in response.read().lower()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
