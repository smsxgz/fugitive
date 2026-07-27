from __future__ import annotations

import json
import math

import pytest

from fugitive.agents.constructive_bir import (
    ConstructiveBeliefInformedRandomMarshalAgent,
)
from fugitive.agents.unweighted_constructive_bir import (
    UnweightedConstructiveBeliefInformedRandomMarshalAgent,
)
from fugitive.engine import GameEngine
from fugitive.experiment import SeedBundle, run_experiment
from fugitive.inference_benchmark import (
    MarshalBackendRequest,
    main,
    run_inference_benchmark,
)
from fugitive.model import FugitiveAction, Observation, Role


class OpeningOneTwoFugitive:
    def choose_draw_pile(self, observation: Observation) -> int:
        return observation.legal_draw_piles[0]

    def choose_fugitive_action(self, observation: Observation) -> FugitiveAction:
        return FugitiveAction(len(observation.route))


class GuessBothMarshal:
    def choose_draw_pile(self, observation: Observation) -> int:
        return observation.legal_draw_piles[0]

    def choose_guess(self, observation: Observation) -> tuple[int, ...]:
        return (1, 2)


class GuessOneMarshal:
    def choose_draw_pile(self, observation: Observation) -> int:
        return observation.legal_draw_piles[0]

    def choose_guess(self, observation: Observation) -> tuple[int, ...]:
        return (1,)


@pytest.fixture
def manifest_path(tmp_path):
    manifest = run_experiment(
        OpeningOneTwoFugitive(),
        GuessBothMarshal(),
        seeds=SeedBundle(master=101, deck=7, fugitive=11, marshal=13),
        validate_invariants=True,
    ).manifest
    path = tmp_path / "manifest.json"
    path.write_text(manifest.to_json(), encoding="utf-8")
    return path


def _opening_observation() -> Observation:
    engine = GameEngine(seed=7)
    engine.apply_fugitive_action(FugitiveAction(1))
    engine.apply_fugitive_action(FugitiveAction(2))
    return engine.observation(Role.MARSHAL)


@pytest.mark.parametrize(
    ("alias", "canonical_name"),
    (
        ("BIR-1", "belief-informed-random"),
        ("BIR-2U", "unweighted-constructive-belief-informed-random"),
        ("BIR-2S", "constructive-belief-informed-random"),
        ("BIR-2E", "exact-sprint-belief-informed-random"),
        ("BIR-3", "mcmc-belief-informed-random"),
    ),
)
def test_each_bir_backend_runs_on_a_fixed_manifest(
    manifest_path,
    alias,
    canonical_name,
):
    parameters = {
        "particle_count": 2,
        "max_guess_candidates": 2,
    }
    if alias == "BIR-3":
        parameters["mh_steps_per_chain"] = 1

    payload = run_inference_benchmark(
        [manifest_path],
        [MarshalBackendRequest.create(alias, parameters)],
        seed=73,
        top_k=(1, 2),
        calibration_bins=4,
        catalogue_max_guess_size=2,
        catalogue_max_joint_candidates=2,
    )

    assert payload["schema_version"] == 3
    assert payload["settings"]["replicates"] == 1
    assert len(payload["runs"]) == 1
    run = payload["runs"][0]
    assert run["replicate"] == 0
    assert "backend_seed" not in run
    assert len(run["request"]["parameter_sha256"]) == 64
    assert run["agent"]["name"] == canonical_name
    assert run["decision_count"] == 3
    assert len(run["decisions"]) == 3
    assert run["summary"]["hidden_cards"]["evaluated_cards"] == 3 * 41
    assert run["summary"]["joint_calibration"]["forecast_count"] > 0
    assert len(run["summary"]["joint_calibration"]["bins"]) == 4


def test_cli_json_is_deterministic_and_repeated_manifests_are_isolated(
    manifest_path,
    capsys,
):
    arguments = [
        str(manifest_path),
        str(manifest_path),
        "--agent",
        "BIR-1",
        "--parameters",
        '{"particle_count": 4, "max_guess_candidates": 4}',
        "--top-k",
        "1",
        "2",
        "--calibration-bins",
        "4",
        "--catalogue-max-guess-size",
        "2",
        "--catalogue-max-joint-candidates",
        "2",
        "--seed",
        "991",
        "--replicates",
        "2",
    ]

    assert main(arguments) == 0
    first_text = capsys.readouterr().out
    assert main(arguments) == 0
    second_text = capsys.readouterr().out

    assert first_text == second_text
    payload = json.loads(first_text)
    assert payload["benchmark_version"] == (
        "fixed-observation-marshal-inference-v3"
    )
    assert payload["settings"]["replicates"] == 2
    assert payload["settings"]["seed_protocol"] == {
        "id": "paired-backend-comparison-seed-v2",
        "pairing_key": ["root_seed", "manifest_sha256", "replicate"],
    }
    first_zero, first_one, second_zero, second_one = payload["runs"]
    assert first_zero["replicate"] == second_zero["replicate"] == 0
    assert first_one["replicate"] == second_one["replicate"] == 1
    assert first_zero["comparison_seed"] == second_zero["comparison_seed"]
    assert first_one["comparison_seed"] == second_one["comparison_seed"]
    assert first_zero["comparison_seed"] != first_one["comparison_seed"]
    assert first_zero["summary"] == second_zero["summary"]
    assert first_zero["decisions"] == second_zero["decisions"]
    assert first_one["summary"] == second_one["summary"]
    assert first_one["decisions"] == second_one["decisions"]
    assert payload["game_first"]["aggregation_id"] == (
        "equal-source-game-then-replicate-v1"
    )
    assert payload["game_first"]["within_game_weighting"] == (
        "equal_unique_replicates"
    )


def test_game_first_summary_gives_each_source_game_equal_weight(
    manifest_path,
    tmp_path,
) -> None:
    longer_manifest = run_experiment(
        OpeningOneTwoFugitive(),
        GuessOneMarshal(),
        seeds=SeedBundle(master=202, deck=17, fugitive=19, marshal=23),
        max_decisions=8,
        validate_invariants=True,
    ).manifest
    longer_path = tmp_path / "longer-manifest.json"
    longer_path.write_text(longer_manifest.to_json(), encoding="utf-8")

    payload = run_inference_benchmark(
        [manifest_path, manifest_path, longer_path],
        [
            MarshalBackendRequest.create(
                "BIR-1",
                {"particle_count": 2, "max_guess_candidates": 2},
            )
        ],
        seed=505,
        replicates=2,
        top_k=(1, 2),
        calibration_bins=2,
        catalogue_max_guess_size=2,
        catalogue_max_joint_candidates=2,
    )

    group = payload["game_first"]["groups"][0]
    assert group["source_game_count"] == 2
    assert group["replicate_count"] == 2
    assert group["summary"]["averaging_unit"] == "source_game"
    assert group["summary"]["unit_count"] == 2

    game_rows = group["source_games"]
    repeated = next(
        row for row in game_rows if row["input_manifest_count"] == 2
    )
    longer = next(
        row for row in game_rows if row["input_manifest_count"] == 1
    )
    assert repeated["summary"]["mean_decision_count"] == 3.0
    assert longer["summary"]["mean_decision_count"] == 4.0
    assert group["summary"]["mean_decision_count"] == 3.5
    assert [
        row["summary"]["mean_decision_count"]
        for row in group["replicate_summaries"]
    ] == [3.5, 3.5]

    input_weighted_mean = (3.0 + 3.0 + 4.0) / 3.0
    assert group["summary"]["mean_decision_count"] != pytest.approx(
        input_weighted_mean
    )


def test_benchmark_rejects_a_manifest_without_marshal_decisions(tmp_path) -> None:
    manifest = run_experiment(
        OpeningOneTwoFugitive(),
        GuessBothMarshal(),
        seeds=SeedBundle(master=303, deck=7, fugitive=11, marshal=13),
        max_decisions=2,
        validate_invariants=True,
    ).manifest
    path = tmp_path / "fugitive-opening-only.json"
    path.write_text(manifest.to_json(), encoding="utf-8")

    with pytest.raises(ValueError, match="no Marshal inference decisions"):
        run_inference_benchmark(
            [path],
            [
                MarshalBackendRequest.create(
                    "BIR-1",
                    {"particle_count": 2, "max_guess_candidates": 2},
                )
            ],
            seed=606,
            top_k=(1, 2),
            calibration_bins=2,
            catalogue_max_guess_size=2,
            catalogue_max_joint_candidates=2,
        )


@pytest.mark.parametrize("replicates", (0, -1, True))
def test_replicates_must_be_a_positive_integer(manifest_path, replicates) -> None:
    with pytest.raises(ValueError, match="replicates must be a positive integer"):
        run_inference_benchmark(
            [manifest_path],
            [
                MarshalBackendRequest.create(
                    "BIR-1",
                    {"particle_count": 2, "max_guess_candidates": 2},
                )
            ],
            replicates=replicates,
        )


def test_bir2u_and_bir2s_use_a_common_comparison_seed_and_proposals(
    manifest_path,
) -> None:
    parameters = {"particle_count": 2, "max_guess_candidates": 2}
    payload = run_inference_benchmark(
        [manifest_path],
        [
            MarshalBackendRequest.create("BIR-2U", parameters),
            MarshalBackendRequest.create("BIR-2S", parameters),
        ],
        seed=431,
        catalogue_max_guess_size=2,
        catalogue_max_joint_candidates=2,
    )

    unweighted_run, corrected_run = payload["runs"]
    assert unweighted_run["comparison_seed"] == corrected_run["comparison_seed"]
    assert (
        unweighted_run["request"]["parameter_sha256"]
        == corrected_run["request"]["parameter_sha256"]
    )
    assert unweighted_run["request"]["name"] != corrected_run["request"]["name"]

    comparison_seed = int(unweighted_run["comparison_seed"])
    observation = _opening_observation()
    unweighted = UnweightedConstructiveBeliefInformedRandomMarshalAgent(
        comparison_seed,
        particle_count=32,
        max_guess_candidates=16,
    ).belief(observation)
    corrected = ConstructiveBeliefInformedRandomMarshalAgent(
        comparison_seed,
        particle_count=32,
        max_guess_candidates=16,
    ).belief(observation)

    assert tuple(particle.world_key for particle in unweighted.particles) == tuple(
        particle.world_key for particle in corrected.particles
    )
    assert tuple(particle.weight for particle in unweighted.particles) == (
        pytest.approx((1.0 / 32,) * 32)
    )
    assert any(
        not math.isclose(particle.weight, 1.0 / 32)
        for particle in corrected.particles
    )


def test_parameters_are_metadata_but_do_not_change_comparison_seed(
    manifest_path,
) -> None:
    payload = run_inference_benchmark(
        [manifest_path],
        [
            MarshalBackendRequest.create(
                "BIR-1",
                {"particle_count": 2, "max_guess_candidates": 2},
            ),
            MarshalBackendRequest.create(
                "BIR-1",
                {"particle_count": 3, "max_guess_candidates": 2},
            ),
        ],
        seed=29,
        catalogue_max_guess_size=2,
        catalogue_max_joint_candidates=2,
    )

    first, second = payload["runs"]
    assert first["comparison_seed"] == second["comparison_seed"]
    assert (
        first["request"]["parameter_sha256"]
        != second["request"]["parameter_sha256"]
    )
