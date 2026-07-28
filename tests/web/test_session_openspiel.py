from __future__ import annotations

import pytest

pyspiel = pytest.importorskip("pyspiel")

from fugitive.web.session import GameSession, WebAPIError, agent_catalog
from fugitive.web.server import SessionStore


def make_session(
    *, mode: str = "human_fugitive", seed: int = 17, **kwargs: object
) -> GameSession:
    return GameSession(
        session_id="test-session",
        mode=mode,
        fugitive_agent="random",
        marshal_agent="random",
        seed=seed,
        **kwargs,
    )


def first_complete_fugitive_play(session: GameSession) -> tuple[int, list[int]]:
    trial = session.state.clone()
    hideout = next(action for action in trial.legal_actions() if 1 <= action <= 42)
    trial.apply_action(hideout)
    sprint_cards: list[int] = []
    while 46 not in trial.legal_actions():
        sprint = next(action for action in trial.legal_actions() if 1 <= action <= 42)
        sprint_cards.append(sprint)
        trial.apply_action(sprint)
    return hideout, sprint_cards


def test_catalog_exposes_only_openspiel_random_bot() -> None:
    catalog = agent_catalog()

    assert catalog["defaults"] == {"fugitive": "random", "marshal": "random"}
    assert [entry["id"] for entry in catalog["fugitive"]] == ["random"]
    assert [entry["id"] for entry in catalog["marshal"]] == ["random"]


def test_setup_is_seeded_and_internal_chance_nodes_are_consumed() -> None:
    first = make_session(seed=1234).as_dict()
    second = make_session(seed=1234).as_dict()

    assert first["phase"] == "fugitive_opening"
    assert first["actor"] == "fugitive"
    assert first["hand"] == second["hand"]
    assert len(first["history"]) == 5
    assert all(event["type"] == "draw" for event in first["history"])


def test_reset_rebuilds_same_state_for_same_seed() -> None:
    session = make_session(seed=1234)
    initial = session.as_dict()

    reset = session.reset(seed="1234")

    assert reset["hand"] == initial["hand"]
    assert reset["history"] == initial["history"]
    assert reset["phase"] == "fugitive_opening"
    assert reset["seed_was_supplied"] is True


def test_observation_cache_is_reused_until_state_changes() -> None:
    session = make_session(mode="spectate", spectator_view="public")
    cached = session._observation(0)

    assert session._observation(0) is cached

    session.step()

    assert session._observation(0) is not cached


def test_fugitive_composite_action_commits_one_route_node() -> None:
    session = make_session()
    before = session.as_dict()
    hideout, sprint_cards = first_complete_fugitive_play(session)

    after = session.apply_human_action(
        {
            "type": "fugitive_action",
            "hideout": hideout,
            "sprint_cards": list(reversed(sprint_cards)),
        }
    )

    assert len(after["route"]) == len(before["route"]) + 1
    assert after["route"][-1]["hideout"] == hideout
    assert after["route"][-1]["sprint_cards"] == sorted(sprint_cards)
    assert after["phase"] == "fugitive_opening"
    assert after["actor"] == "fugitive"


def test_fugitive_draft_preview_checks_commit_without_mutating_state() -> None:
    session = make_session(seed=17)
    history_before = tuple(session.state.history())

    incomplete = session.preview_fugitive_action(
        {"hideout": 9, "sprint_cards": []}
    )
    hideout, sprint_cards = first_complete_fugitive_play(session)
    complete = session.preview_fugitive_action(
        {"hideout": hideout, "sprint_cards": sprint_cards}
    )

    assert incomplete == {"valid_prefix": True, "can_commit": False}
    assert complete == {"valid_prefix": True, "can_commit": True}
    assert tuple(session.state.history()) == history_before


def test_illegal_composite_action_does_not_partially_mutate_state() -> None:
    session = make_session()
    history_before = tuple(session.state.history())
    route_before = session.as_dict()["route"]

    with pytest.raises(WebAPIError) as error:
        session.apply_human_action(
            {"type": "fugitive_action", "hideout": 42, "sprint_cards": []}
        )

    assert error.value.code == "illegal_action"
    assert tuple(session.state.history()) == history_before
    assert session.as_dict()["route"] == route_before


def test_spectator_step_advances_exactly_one_visible_decision() -> None:
    session = make_session(mode="spectate", spectator_view="omniscient")
    before = session.as_dict()

    after = session.step()

    assert len(after["history"]) == len(before["history"]) + 1
    assert after["history"][-1]["type"] in {
        "draw",
        "fugitive_action",
        "pass",
        "guess",
    }


def test_human_marshal_draws_and_submits_composite_guess() -> None:
    session = make_session(mode="human_marshal", seed=3)
    state = session.as_dict()

    for _ in range(2):
        pile = state["legal_actions"]["draw_piles"][0]
        state = session.apply_human_action({"type": "draw", "pile": pile})

    assert state["phase"] == "marshal_guess"
    number = state["legal_actions"]["guess_numbers"][0]
    state = session.apply_human_action({"type": "guess", "numbers": [number]})

    assert any(event["type"] == "guess" for event in state["history"])
    assert state["status"] in {"awaiting_human", "finished"}


def test_public_spectator_view_redacts_private_cards() -> None:
    session = make_session(mode="spectate", spectator_view="omniscient")
    omniscient = session.as_dict()
    public = session.set_spectator_view("public")

    assert omniscient["hands"]["fugitive"]
    assert public["hand"] == []
    assert public["hands"] == {"fugitive": None, "marshal": None}
    assert all(event.get("card") is None for event in public["history"])


def test_export_contains_native_openspiel_state_and_action_history() -> None:
    session = make_session(mode="spectate", spectator_view="public")
    session.step()
    session.terminate()

    exported = session.export_trace()

    assert exported["schema"] == "fugitive.openspiel-web-trace"
    assert exported["game"]
    assert exported["state"]
    assert exported["game_and_state"]
    assert exported["action_history"] == list(session.state.history())


def test_session_store_creates_and_steps_spectator_game() -> None:
    store = SessionStore()
    session = store.create(
        {
            "mode": "spectate",
            "fugitive_agent": "random",
            "marshal_agent": "random",
            "seed": 19,
            "spectator_view": "public",
        }
    )
    created = session.as_dict()
    advanced = session.step()

    assert created["phase"] == "fugitive_opening"
    assert len(advanced["history"]) == len(created["history"]) + 1
