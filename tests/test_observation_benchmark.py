from __future__ import annotations

import math

import pytest

from fugitive.agents.bootstrap_bir import BootstrapParticleBeliefBackend
from fugitive.experiment import SeedBundle, run_experiment
from fugitive.model import FugitiveAction, Observation, Phase
from fugitive.observation_benchmark import (
    evaluate_backend_on_manifest,
    extract_marshal_decision_points,
    hidden_card_brier_score,
    hidden_card_log_loss,
    joint_calibration_bins,
    joint_calibration_record,
    route_forecast_metrics,
)
from fugitive.particle_inference.state import MarshalParticleBelief


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


class RecordingEmptyBackend:
    backend_id = "recording-empty-v1"

    def __init__(self) -> None:
        self.observations: list[Observation] = []

    def infer(self, observation: Observation) -> MarshalParticleBelief:
        assert isinstance(observation, Observation)
        self.observations.append(observation)
        return MarshalParticleBelief(
            observation,
            (),
            sampling_attempts=0,
            sampling_exhausted=True,
        )


def _manifest():
    return run_experiment(
        OpeningOneTwoFugitive(),
        GuessBothMarshal(),
        seeds=SeedBundle(master=101, deck=7, fugitive=11, marshal=13),
        validate_invariants=True,
    ).manifest


def test_replay_extracts_complete_truth_before_each_marshal_decision():
    points = extract_marshal_decision_points(_manifest())

    assert [point.decision for point in points] == [3, 4, 5]
    assert [point.phase for point in points] == [
        Phase.MARSHAL_DRAW,
        Phase.MARSHAL_DRAW,
        Phase.MARSHAL_GUESS,
    ]
    assert [len(point.observation.hand) for point in points] == [0, 1, 2]

    first = points[0]
    assert first.truth.route_hideouts == (0, 1, 2)
    assert first.truth.route_sprints == ((), (), ())
    assert first.truth.hidden_route == (1, 2)
    assert tuple(slot.hideout for slot in first.observation.route) == (0, None, None)
    assert first.truth.marshal_hand == first.observation.hand
    assert tuple(map(len, first.truth.remaining_piles)) == first.observation.pile_sizes
    assert first.truth.remaining_piles[first.record.pile][-1] == first.record.card

    for point in points:
        assert point.truth.marshal_hand == point.observation.hand
        assert (
            tuple(map(len, point.truth.remaining_piles))
            == point.observation.pile_sizes
        )


def test_backend_observations_are_ordered_and_candidate_catalogue_is_injected():
    backend = RecordingEmptyBackend()
    catalogue_inputs: list[Observation] = []

    def catalogue(observation: Observation):
        assert isinstance(observation, Observation)
        catalogue_inputs.append(observation)
        return ((1,), (1, 2))

    result = evaluate_backend_on_manifest(
        _manifest(),
        backend,
        candidate_set_provider=catalogue,
    )

    assert result.backend_id == "recording-empty-v1"
    assert tuple(backend.observations) == tuple(
        decision.point.observation for decision in result.decisions
    )
    assert [observation.phase for observation in backend.observations] == [
        Phase.MARSHAL_DRAW,
        Phase.MARSHAL_DRAW,
        Phase.MARSHAL_GUESS,
    ]
    assert [observation.phase for observation in catalogue_inputs] == [
        Phase.MARSHAL_GUESS
    ]

    guess_evaluation = result.decisions[-1]
    rows = {row.guess: row for row in guess_evaluation.joint_calibration}
    assert set(rows) == {(1,), (1, 2)}
    assert rows[(1,)].observed_success
    assert rows[(1, 2)].observed_success
    assert rows[(1, 2)].recorded_action
    assert rows[(1, 2)].predicted_success == 0.0


def test_actual_bir1_backend_can_be_evaluated_incrementally_on_the_trace():
    backend = BootstrapParticleBeliefBackend(
        particle_count=8,
        belief_salt="observation-benchmark-test",
    )

    result = evaluate_backend_on_manifest(_manifest(), backend)

    assert len(result.decisions) == 3
    assert backend.cache_size == 3
    assert backend.latest_belief is not None
    assert backend.latest_belief.observation == result.decisions[-1].point.observation


def test_hidden_card_metrics_have_explicit_binary_semantics():
    probabilities = {1: 0.8, 2: 0.3, 3: 0.1}

    brier = hidden_card_brier_score(
        probabilities,
        (1, 2),
        card_universe=(1, 2, 3),
    )
    log_loss = hidden_card_log_loss(
        probabilities,
        (1, 2),
        card_universe=(1, 2, 3),
    )

    assert brier == pytest.approx((0.2**2 + 0.7**2 + 0.1**2) / 3)
    assert log_loss == pytest.approx(
        (-math.log(0.8) - math.log(0.3) - math.log(0.9)) / 3
    )


def test_route_metrics_report_mass_rank_and_top_k_coverage():
    metrics = route_forecast_metrics(
        {
            (1, 2): 2.0,
            (1, 3): 5.0,
            (2, 3): 3.0,
        },
        (2, 3),
        top_k=(1, 2, 5),
    )

    assert metrics.true_mass == pytest.approx(0.3)
    assert metrics.rank == 2
    assert metrics.support_size == 3
    assert metrics.top_k_coverage == ((1, False), (2, True), (5, True))

    absent = route_forecast_metrics({(1, 2): 1.0}, (2, 3), top_k=(2,))
    assert absent.true_mass == 0.0
    assert absent.rank == 2
    assert absent.top_k_coverage == ((2, False),)


def test_joint_forecasts_form_standard_reliability_bins():
    rows = (
        joint_calibration_record((1,), 0.05, (2,)),
        joint_calibration_record((2,), 0.15, (2,)),
        joint_calibration_record((2, 3), 1.0, (2, 3)),
    )

    bins = joint_calibration_bins(rows, bin_count=10)

    assert bins[0].count == 1
    assert bins[0].mean_prediction == pytest.approx(0.05)
    assert bins[0].observed_frequency == 0.0
    assert bins[1].count == 1
    assert bins[1].observed_frequency == 1.0
    assert bins[9].count == 1
    assert bins[9].mean_prediction == 1.0
    assert bins[9].observed_frequency == 1.0
