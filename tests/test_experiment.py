from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path

import pytest

from fugitive.engine import GameEngine
from fugitive.experiment import (
    AgentDescriptor,
    DrawTrace,
    ExperimentStatus,
    RULES_SHA256,
    ReplayManifest,
    ReplayMismatchError,
    SeedBundle,
    derive_seed_bundle,
    replay_manifest,
    run_experiment,
    run_registered_experiment,
)
from fugitive.model import FugitiveAction, Phase, Role, Winner


class OpeningOneTwoFugitive:
    name = "test-opening-one-two"

    def choose_draw_pile(self, observation):
        return observation.legal_draw_piles[0]

    def choose_fugitive_action(self, observation):
        if observation.phase is Phase.FUGITIVE_OPENING:
            return FugitiveAction(len(observation.route))
        return FugitiveAction(None)


class ImmediateMarshal:
    name = "test-immediate-marshal"

    def choose_draw_pile(self, observation):
        return observation.legal_draw_piles[0]

    def choose_guess(self, observation):
        return (1, 2)


class MissingMarshal:
    name = "test-missing-marshal"

    def choose_draw_pile(self, observation):
        return observation.legal_draw_piles[0]

    def choose_guess(self, observation):
        return (observation.hand[0],)


class BrokenMarshal(ImmediateMarshal):
    name = "test-broken-marshal"

    def choose_guess(self, observation):
        raise RuntimeError("deliberate policy failure")


TEST_SEEDS = SeedBundle(master=123, deck=7, fugitive=11, marshal=13)


def test_completed_run_round_trips_and_replays_exactly():
    run = run_experiment(
        OpeningOneTwoFugitive(),
        ImmediateMarshal(),
        seeds=TEST_SEEDS,
        fugitive_descriptor=AgentDescriptor.create("opening", {"variant": 1}),
        marshal_descriptor=AgentDescriptor.create("immediate", {"pairs": True}),
        validate_invariants=True,
    )

    assert run.status is ExperimentStatus.COMPLETED
    assert run.game_result is not None
    assert run.game_result.winner is Winner.MARSHAL
    assert run.manifest.winner is Winner.MARSHAL
    assert run.manifest.decision_count == 5
    assert len(run.manifest.trace) == 5
    assert run.manifest.max_decisions is None

    payload = run.manifest.to_json(indent=None)
    raw = run.manifest.to_dict()
    assert raw["seeds"] == {
        "master": "123",
        "deck": "7",
        "fugitive": "11",
        "marshal": "13",
    }
    restored = ReplayManifest.from_json(payload)
    assert restored == run.manifest

    verification = replay_manifest(restored)
    assert verification.status is ExperimentStatus.COMPLETED
    assert verification.game_result == run.game_result
    assert verification.final_state_sha256 == run.manifest.final_state_sha256


def test_explicit_watchdog_truncates_without_inventing_a_winner():
    run = run_experiment(
        OpeningOneTwoFugitive(),
        MissingMarshal(),
        seeds=TEST_SEEDS,
        max_decisions=3,
    )

    assert run.status is ExperimentStatus.TRUNCATED
    assert run.game_result is None
    assert run.manifest.winner is None
    assert run.manifest.reason is None
    assert run.manifest.rounds is None
    assert run.manifest.error is None
    assert run.manifest.decision_count == 3
    assert replay_manifest(run.manifest).status is ExperimentStatus.TRUNCATED


def test_policy_exception_returns_error_metadata_and_replayable_prefix():
    run = run_experiment(
        OpeningOneTwoFugitive(),
        BrokenMarshal(),
        seeds=TEST_SEEDS,
        validate_invariants=True,
    )

    assert run.status is ExperimentStatus.ERROR
    assert run.game_result is None
    assert run.manifest.winner is None
    assert run.manifest.error is not None
    assert run.manifest.error.error_type == "builtins.RuntimeError"
    assert run.manifest.error.stage == "choose_marshal_guess"
    assert run.manifest.error.decision == 5
    assert "deliberate policy failure" in run.manifest.error.message
    assert replay_manifest(run.manifest).status is ExperimentStatus.ERROR


def test_replay_rejects_a_tampered_draw_outcome():
    run = run_experiment(
        OpeningOneTwoFugitive(),
        ImmediateMarshal(),
        seeds=TEST_SEEDS,
    )
    records = list(run.manifest.trace)
    draw_index = next(
        index for index, record in enumerate(records) if isinstance(record, DrawTrace)
    )
    draw = records[draw_index]
    assert isinstance(draw, DrawTrace)
    records[draw_index] = replace(draw, card=draw.card + 1)
    tampered = replace(run.manifest, trace=tuple(records))

    with pytest.raises(ReplayMismatchError, match="expected draw"):
        replay_manifest(tampered)


def test_registered_runner_domain_separates_all_random_streams():
    run = run_registered_experiment(
        master_seed=2**63 + 123,
        fugitive_name="hierarchical-random",
        marshal_name="hierarchical-random",
        fugitive_parameters={"max_low_cost_payments": 3},
        marshal_parameters={"max_guess_size": 2},
        max_decisions=0,
    )

    expected = derive_seed_bundle(2**63 + 123)
    assert run.manifest.seeds == expected
    assert len({expected.deck, expected.fugitive, expected.marshal}) == 3
    assert run.status is ExperimentStatus.TRUNCATED
    assert run.manifest.fugitive_agent.parameters == {
        "algorithm_id": "hr-1-fugitive-hierarchical-legal-random-v1",
        "max_extra_overpayments": 8,
        "max_low_cost_payments": 3,
        "overpay_probability": 0.05,
    }
    assert run.manifest.marshal_agent.parameters == {
        "algorithm_id": "hr-1-marshal-hard-support-random-v2",
        "max_guess_size": 2,
        "max_guesses_per_size": 128,
        "multi_guess_continuation": 0.1,
    }
    assert run.manifest.to_dict()["seeds"]["master"] == str(2**63 + 123)


def test_registered_runner_default_has_no_cutoff_and_full_game_replays():
    run = run_registered_experiment(
        master_seed=20260724,
        fugitive_name="hierarchical-random",
        marshal_name="hierarchical-random",
    )

    assert run.status is ExperimentStatus.COMPLETED
    assert run.manifest.max_decisions is None
    assert run.manifest.winner in (Winner.FUGITIVE, Winner.MARSHAL)
    assert replay_manifest(run.manifest).game_result == run.game_result


def test_invalid_watchdog_is_rejected_instead_of_changing_game_rules():
    with pytest.raises(ValueError, match="max_decisions"):
        run_experiment(
            OpeningOneTwoFugitive(),
            ImmediateMarshal(),
            seeds=TEST_SEEDS,
            max_decisions=True,
        )


def test_replay_rejects_inconsistent_outcome_and_rules_metadata():
    run = run_experiment(
        OpeningOneTwoFugitive(),
        MissingMarshal(),
        seeds=TEST_SEEDS,
        max_decisions=3,
    )

    with pytest.raises(ReplayMismatchError, match="rules fingerprint"):
        replay_manifest(replace(run.manifest, rules_sha256="0" * 64))
    with pytest.raises(ReplayMismatchError, match="must not contain a winner"):
        replay_manifest(
            replace(
                run.manifest,
                winner=Winner.FUGITIVE,
                reason="invented",
                rounds=1,
            )
        )
    with pytest.raises(ReplayMismatchError, match="exactly at max_decisions"):
        replay_manifest(replace(run.manifest, max_decisions=4))


def test_all_empty_piles_skip_required_draws_without_ending_the_game():
    engine = GameEngine(seed=19)
    engine.apply_fugitive_action(FugitiveAction(1))
    engine.apply_fugitive_action(FugitiveAction(2))

    steps = 0
    while sum(engine.observation(Role.FUGITIVE).pile_sizes) > 0:
        steps += 1
        assert steps < 100
        if engine.phase is Phase.FUGITIVE_DRAW:
            observation = engine.observation(Role.FUGITIVE)
            engine.draw(observation.legal_draw_piles[0])
        elif engine.phase is Phase.FUGITIVE_ACTION:
            engine.apply_fugitive_action(FugitiveAction(None))
        elif engine.phase is Phase.MARSHAL_DRAW:
            observation = engine.observation(Role.MARSHAL)
            engine.draw(observation.legal_draw_piles[0])
        elif engine.phase is Phase.MARSHAL_GUESS:
            observation = engine.observation(Role.MARSHAL)
            engine.apply_guess((observation.hand[0],))
        else:  # pragma: no cover - the passive route never reaches Manhunt
            raise AssertionError(f"unexpected phase {engine.phase}")
        engine.validate_invariants()

    # Move to the Fugitive action if the final physical card was drawn by M.
    if engine.phase is Phase.MARSHAL_GUESS:
        observation = engine.observation(Role.MARSHAL)
        engine.apply_guess((observation.hand[0],))
    assert engine.phase is Phase.FUGITIVE_ACTION
    assert not engine.is_terminal

    # Both nominal draws are now skipped, with no winner or bonus action.
    engine.apply_fugitive_action(FugitiveAction(None))
    assert engine.phase is Phase.MARSHAL_GUESS
    observation = engine.observation(Role.MARSHAL)
    engine.apply_guess((observation.hand[0],))
    assert engine.phase is Phase.FUGITIVE_ACTION
    assert not engine.is_terminal
    engine.validate_invariants()


def test_embedded_rules_fingerprint_matches_the_canonical_rules_file():
    rules_path = (
        Path(__file__).resolve().parents[1]
        / "docs"
        / "rules"
        / "CANONICAL_RULES.md"
    )
    normalized = rules_path.read_text(encoding="utf-8").replace("\r\n", "\n")
    assert hashlib.sha256(normalized.encode("utf-8")).hexdigest() == RULES_SHA256
