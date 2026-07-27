from __future__ import annotations

from http import HTTPStatus
import json

import pytest

from fugitive.game.model import DrawRecord, GuessRecord, PlayRecord
from fugitive.runtime.manifest import RULES_SHA256, RULES_VERSION
from fugitive.runtime.replay import replay_manifest
from fugitive.shared.reproducibility import (
    MAX_SEED,
    SEED_DERIVATION_VERSION,
    derive_seed,
)
from fugitive.web.session import (
    WEB_TRACE_SCHEMA_VERSION,
    GameSession,
    WebAPIError,
    replay_manifest_from_web_trace,
)


def assert_trace_matches_engine_history(session: GameSession, trace: list) -> None:
    history = session.engine.result().history
    assert len(trace) == len(history)
    for event, record in zip(trace, history, strict=True):
        assert event["round_number"] == record.round_number
        if isinstance(record, DrawRecord):
            assert (event["type"], event["pile"], event["card"]) == (
                "draw",
                record.pile,
                record.card,
            )
        elif isinstance(record, PlayRecord):
            assert event["hideout"] == record.hideout
            assert event["sprint_cards"] == list(record.sprint_cards)
        else:
            assert isinstance(record, GuessRecord)
            assert event["type"] == "guess"
            assert event["numbers"] == list(record.numbers)
            assert event["success"] == record.success


def test_completed_game_exports_a_private_trace_that_replays() -> None:
    session = GameSession(
        session_id="export-test",
        mode="spectate",
        fugitive_agent="hierarchical-random",
        marshal_agent="hierarchical-random",
        seed=str(MAX_SEED),
    )
    assert session.as_dict()["can_export"] is False
    with pytest.raises(WebAPIError) as active:
        session.export_trace()
    assert active.value.status == HTTPStatus.CONFLICT
    assert active.value.code == "export_not_ready"

    session.set_spectator_view("public")
    assert session.auto(max_steps=10_000)["status"] == "finished"
    exported = session.export_trace()

    assert json.loads(json.dumps(exported, allow_nan=False)) == exported
    assert exported["schema"] == "fugitive.web-trace"
    assert exported["schema_version"] == WEB_TRACE_SCHEMA_VERSION
    assert exported["rules"] == {
        "version": RULES_VERSION,
        "sha256": RULES_SHA256,
    }
    assert exported["seed_derivation"] == {
        "version": SEED_DERIVATION_VERSION,
        "master": str(MAX_SEED),
        "deck": str(derive_seed(MAX_SEED, "deck")),
        "fugitive": str(derive_seed(MAX_SEED, "fugitive")),
        "marshal": str(derive_seed(MAX_SEED, "marshal")),
    }
    assert exported["agents"]["fugitive"]["registry_name"] == (
        "hierarchical-random"
    )
    assert exported["agents"]["marshal"]["registry_name"] == (
        "hierarchical-random"
    )

    trace = exported["trace"]
    assert_trace_matches_engine_history(session, trace)
    assert all(
        event["card"] is not None
        for event in trace
        if event["type"] == "draw"
    )

    manifest = replay_manifest_from_web_trace(exported)
    verification = replay_manifest(manifest)
    assert manifest.to_dict() == exported["replay_manifest"]
    assert verification.game_result == session.engine.result()
    assert verification.final_state_sha256 == manifest.final_state_sha256
