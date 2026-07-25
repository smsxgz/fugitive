from __future__ import annotations

import pytest

from fugitive.experiment import (
    replay_manifest,
    run_registered_experiment,
    state_sha256,
)
from fugitive.reproducibility import (
    MAX_SEED,
    SEED_DERIVATION_VERSION,
    SeedBundle,
    derive_seed_bundle,
)
from fugitive.web import (
    GameSession,
    replay_manifest_from_web_trace,
)


def test_shared_seed_protocol_preserves_the_published_experiment_mapping() -> None:
    assert SEED_DERIVATION_VERSION == "fugitive-experiment-v1"
    assert derive_seed_bundle(2**63 + 123) == SeedBundle(
        master=9_223_372_036_854_775_931,
        deck=11_566_804_124_543_284_205,
        fugitive=2_516_367_888_458_312_592,
        marshal=12_117_919_150_439_490_662,
    )


def test_web_and_experiment_match_with_the_same_master_profile_and_spec() -> None:
    session = GameSession(
        session_id="cross-entrypoint-golden",
        mode="spectate",
        fugitive_agent="hierarchical-random",
        marshal_agent="hierarchical-random",
        seed=str(MAX_SEED),
    )
    web_state = session.auto(max_steps=10_000)
    assert web_state["status"] == "finished"

    experiment = run_registered_experiment(
        master_seed=MAX_SEED,
        fugitive_name="hierarchical-random",
        marshal_name="hierarchical-random",
        fugitive_profile="default",
        marshal_profile="default",
    )
    assert experiment.game_result is not None

    assert session._seed_bundle == experiment.manifest.seeds
    assert session.decision_trace == experiment.manifest.trace
    assert state_sha256(session.engine) == experiment.manifest.final_state_sha256
    assert session.engine.result() == experiment.game_result

    exported = session.export_trace()
    manifest = replay_manifest_from_web_trace(exported)
    assert manifest == experiment.manifest
    assert replay_manifest(manifest).game_result == experiment.game_result


def test_web_trace_adapter_rejects_legacy_and_terminated_wrappers() -> None:
    with pytest.raises(ValueError, match="unsupported Web trace schema"):
        replay_manifest_from_web_trace(
            {
                "schema": "fugitive.web-trace",
                "schema_version": 1,
            }
        )

    session = GameSession(
        session_id="terminated-prefix",
        mode="spectate",
        fugitive_agent="hierarchical-random",
        marshal_agent="hierarchical-random",
        seed="7",
    )
    session.step()
    session.terminate()
    with pytest.raises(ValueError, match="no completed replay manifest"):
        replay_manifest_from_web_trace(session.export_trace())
