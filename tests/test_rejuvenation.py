from __future__ import annotations

from dataclasses import replace
import math
import random

import pytest

from fugitive.engine import GameEngine
from fugitive.inference.constraints import CompiledMarshalConstraints
from fugitive.inference.constructive_metadata import constructive_observation_hash
from fugitive.inference.constructive_sampler import ConstructiveWorldSampler
from fugitive.inference.sampling import SamplingBudget
from fugitive.inference.worlds import (
    ConstraintUniformTarget,
    ConstructiveSampleBatch,
    ConstructiveSamplingReport,
    ConstructiveWorld,
)
from fugitive.model import FugitiveAction, Role
from fugitive.rejuvenation import (
    IncompleteProposalBatchError,
    IndependentMHRejuvenator,
    InvalidRejuvenationBatchError,
    _accept_log_probability,
    independent_mh_log_acceptance,
)


def _opening_observation():
    engine = GameEngine(seed=7)
    engine.apply_fugitive_action(FugitiveAction(1))
    engine.apply_fugitive_action(FugitiveAction(2))
    return engine.observation(Role.MARSHAL)


def _report(
    requested: int,
    produced: int,
    *,
    degraded: bool = False,
    importance_valid: bool = True,
    termination_reason: str | None = None,
) -> ConstructiveSamplingReport:
    return ConstructiveSamplingReport(
        requested=requested,
        produced=produced,
        proposals=produced,
        dead_end_route_proposals=0,
        rejected_targets=0,
        search_nodes=0,
        degraded=degraded,
        importance_valid=importance_valid,
        termination_reason=termination_reason,
        exhausted_stage=None,
        unique_worlds=produced,
    )


_TEST_SAMPLER = ConstructiveWorldSampler(sprint_backend="sequential")


def _batch(
    worlds: tuple[ConstructiveWorld, ...],
    *,
    observation=None,
    report: ConstructiveSamplingReport | None = None,
    proposal_kernel_id: str = _TEST_SAMPLER.proposal_kernel_id,
    target_id: str = _TEST_SAMPLER.target_id,
) -> ConstructiveSampleBatch:
    observation = observation or _opening_observation()
    return ConstructiveSampleBatch(
        worlds,
        report or _report(len(worlds), len(worlds)),
        proposal_kernel_id=proposal_kernel_id,
        target_id=target_id,
        observation_hash=constructive_observation_hash(observation),
    )


def _distinct_worlds(count: int = 8):
    observation = _opening_observation()
    sampled = ConstructiveWorldSampler(sprint_backend="sequential").sample_batch(
        observation,
        particle_count=64,
        seed=317,
        budget=SamplingBudget(max_nodes=100_000, max_proposals=2_048),
    )
    by_key = {world.world_key: world for world in sampled.worlds}
    assert len(by_key) >= count
    return observation, tuple(by_key.values())[:count]


class _FixedSampler:
    target = ConstraintUniformTarget()
    sprint_backend = "sequential"
    target_id = _TEST_SAMPLER.target_id
    proposal_kernel_id = _TEST_SAMPLER.proposal_kernel_id

    def __init__(
        self,
        worlds: tuple[ConstructiveWorld, ...],
        *,
        report: ConstructiveSamplingReport | None = None,
    ) -> None:
        self.worlds = worlds
        self.report = report
        self.calls = 0

    def sample_batch(
        self,
        observation,
        *,
        particle_count: int,
        seed=None,
        rng=None,
        budget=None,
    ) -> ConstructiveSampleBatch:
        del seed, rng, budget
        self.calls += 1
        worlds = self.worlds[:particle_count]
        report = self.report or _report(particle_count, len(worlds))
        return _batch(
            worlds,
            observation=observation,
            report=report,
            proposal_kernel_id=self.proposal_kernel_id,
            target_id=self.target_id,
        )


class _ForbiddenSampler(_FixedSampler):
    def sample_batch(self, *_args, **_kwargs):
        raise AssertionError("zero MH steps must not request proposals")


def test_rejuvenation_is_same_seed_deterministic_and_equally_weighted() -> None:
    observation = _opening_observation()
    sampler = ConstructiveWorldSampler(sprint_backend="sequential")
    initial = sampler.sample_batch(
        observation,
        particle_count=12,
        seed=991,
        budget=SamplingBudget(max_nodes=100_000, max_proposals=512),
    )
    rejuvenator = IndependentMHRejuvenator(sampler=sampler)

    first = rejuvenator.rejuvenate(
        observation,
        initial,
        steps_per_chain=2,
        seed=443,
        proposal_budget=SamplingBudget(max_nodes=200_000, max_proposals=1_024),
    )
    repeated = rejuvenator.rejuvenate(
        observation,
        initial,
        steps_per_chain=2,
        seed=443,
        proposal_budget=SamplingBudget(max_nodes=200_000, max_proposals=1_024),
    )

    assert first == repeated
    assert len(first.worlds) == 12
    assert first.normalized_weights == pytest.approx((1 / 12,) * 12)
    assert tuple(particle.weight for particle in first.to_particles()) == pytest.approx(
        (1 / 12,) * 12
    )
    diagnostics = first.diagnostics
    assert 1.0 <= diagnostics.initial_effective_sample_size <= 12.0
    assert diagnostics.proposals == 24
    assert 0 <= diagnostics.accepted <= diagnostics.proposals
    assert 0 <= diagnostics.changed <= diagnostics.accepted
    assert diagnostics.acceptance_rate == pytest.approx(
        diagnostics.accepted / diagnostics.proposals
    )
    assert diagnostics.change_rate == pytest.approx(
        diagnostics.changed / diagnostics.proposals
    )
    assert 1 <= diagnostics.initial_unique_worlds <= 12
    assert 1 / 12 <= diagnostics.final_max_clone_fraction <= 1.0
    assert len(first.ancestor_indices) == 12
    assert all(0 <= index < 12 for index in first.ancestor_indices)
    assert first.proposal_report is not None
    assert first.proposal_report.produced == 24
    assert first.proposal_kernel_id == initial.proposal_kernel_id
    assert first.target_id == initial.target_id
    assert first.observation_hash == initial.observation_hash


def test_zero_steps_only_performs_systematic_resampling() -> None:
    observation, worlds = _distinct_worlds(3)
    initial = _batch(tuple(replace(world, log_q=-2.0) for world in worlds))
    rejuvenator = IndependentMHRejuvenator(
        sampler=_ForbiddenSampler(())  # type: ignore[arg-type]
    )

    result = rejuvenator.rejuvenate(
        observation,
        initial,
        steps_per_chain=0,
        seed=19,
    )

    assert result.diagnostics.proposals == 0
    assert result.diagnostics.accepted == 0
    assert result.diagnostics.changed == 0
    assert result.diagnostics.acceptance_rate == 0.0
    assert result.diagnostics.change_rate == 0.0
    assert len(result.worlds) == 3
    assert len(result.ancestor_indices) == 3
    assert result.proposal_report is None


@pytest.mark.parametrize("steps", (True, -1, 1.5, "1"))
def test_steps_per_chain_is_a_non_negative_integer(steps: object) -> None:
    observation, worlds = _distinct_worlds(1)
    with pytest.raises(ValueError, match="non-negative integer"):
        IndependentMHRejuvenator(
            sampler=ConstructiveWorldSampler(sprint_backend="sequential")
        ).rejuvenate(
            observation,
            _batch(worlds),
            steps_per_chain=steps,  # type: ignore[arg-type]
        )


def test_seed_and_rng_are_mutually_exclusive() -> None:
    observation, worlds = _distinct_worlds(1)
    with pytest.raises(ValueError, match="either seed or rng"):
        IndependentMHRejuvenator().rejuvenate(
            observation,
            _batch(worlds),
            steps_per_chain=0,
            seed=1,
            rng=random.Random(1),
        )


def test_rejuvenator_requires_sampler_and_target_identity_to_match() -> None:
    class AlternateTarget(ConstraintUniformTarget):
        target_id = "test-alternate-target-v1"

    sampler = ConstructiveWorldSampler(
        target=AlternateTarget(),
        sprint_backend="sequential",
    )
    inherited = IndependentMHRejuvenator(sampler=sampler)
    assert inherited.target is sampler.target
    assert inherited.target_id == sampler.target_id

    with pytest.raises(ValueError, match="target IDs must match"):
        IndependentMHRejuvenator(
            sampler=sampler,
            target=ConstraintUniformTarget(),
        )


@pytest.mark.parametrize(
    ("field_name", "wrong_value"),
    (
        ("proposal_kernel_id", "wrong-proposal-kernel-v1"),
        ("target_id", "wrong-target-v1"),
        ("observation_hash", "wrong-observation-hash"),
    ),
)
def test_initial_batch_fingerprint_is_machine_checked(
    field_name: str,
    wrong_value: str,
) -> None:
    observation, worlds = _distinct_worlds(1)
    initial = replace(_batch(worlds), **{field_name: wrong_value})

    with pytest.raises(
        InvalidRejuvenationBatchError,
        match=field_name,
    ):
        IndependentMHRejuvenator(
            sampler=ConstructiveWorldSampler(sprint_backend="sequential")
        ).rejuvenate(
            observation,
            initial,
            steps_per_chain=0,
            seed=1,
        )


@pytest.mark.parametrize(
    ("field_name", "wrong_value"),
    (
        ("proposal_kernel_id", "different-kernel-v1"),
        ("target_id", "different-target-v1"),
        ("observation_hash", "different-observation-hash"),
    ),
)
def test_proposal_batch_fingerprint_is_machine_checked(
    field_name: str,
    wrong_value: str,
) -> None:
    observation, worlds = _distinct_worlds(2)

    class MislabelingSampler(_FixedSampler):
        def sample_batch(self, *args, **kwargs):
            batch = super().sample_batch(*args, **kwargs)
            return replace(batch, **{field_name: wrong_value})

    with pytest.raises(
        InvalidRejuvenationBatchError,
        match=field_name,
    ):
        IndependentMHRejuvenator(
            sampler=MislabelingSampler((worlds[1],))  # type: ignore[arg-type]
        ).rejuvenate(
            observation,
            _batch((worlds[0],)),
            steps_per_chain=1,
            seed=2,
        )


@pytest.mark.parametrize(
    "report",
    (
        _report(2, 1, degraded=True, termination_reason="max_proposals"),
        _report(1, 1, importance_valid=False, termination_reason="node_budget"),
    ),
)
def test_initial_batch_must_be_complete_and_importance_valid(report) -> None:
    observation, worlds = _distinct_worlds(1)
    initial = _batch(worlds, report=report)
    with pytest.raises(InvalidRejuvenationBatchError):
        IndependentMHRejuvenator().rejuvenate(
            observation,
            initial,
            steps_per_chain=0,
            seed=1,
        )


def test_incomplete_proposal_batch_is_rejected() -> None:
    observation, worlds = _distinct_worlds(2)
    partial_report = _report(
        2,
        1,
        degraded=True,
        termination_reason="max_proposals",
    )
    sampler = _FixedSampler((worlds[1],), report=partial_report)

    with pytest.raises(IncompleteProposalBatchError) as caught:
        IndependentMHRejuvenator(sampler=sampler).rejuvenate(  # type: ignore[arg-type]
            observation,
            _batch((worlds[0],)),
            steps_per_chain=2,
            seed=2,
        )

    assert caught.value.report is partial_report


def test_independent_mh_ratio_and_seeded_acceptance_match_hand_calculation() -> None:
    observation, worlds = _distinct_worlds(2)
    constraints = CompiledMarshalConstraints.from_observation(observation)
    current = replace(worlds[0], log_q=-math.log(4), log_target=0.0)
    proposal = replace(worlds[1], log_q=0.0, log_target=0.0)
    target = ConstraintUniformTarget()

    assert independent_mh_log_acceptance(
        current,
        proposal,
        constraints=constraints,
        target=target,
    ) == pytest.approx(math.log(0.25))

    class FixedRng:
        def __init__(self, value: float) -> None:
            self.value = value

        def random(self) -> float:
            return self.value

    from fugitive.rejuvenation import _accept_log_probability

    assert not _accept_log_probability(  # type: ignore[arg-type]
        math.log(0.25), FixedRng(0.30)
    )
    assert _accept_log_probability(  # type: ignore[arg-type]
        math.log(0.25), FixedRng(0.20)
    )


def test_mh_moves_can_restore_diversity_after_sir_cloning() -> None:
    observation, worlds = _distinct_worlds(5)
    common_log_q = -math.log(5)
    initial_world = replace(worlds[0], log_q=common_log_q, log_target=0.0)
    proposals = tuple(
        replace(world, log_q=common_log_q, log_target=0.0)
        for world in worlds[1:5]
    )
    initial = _batch((initial_world,) * 4)

    result = IndependentMHRejuvenator(
        sampler=_FixedSampler(proposals)  # type: ignore[arg-type]
    ).rejuvenate(
        observation,
        initial,
        steps_per_chain=1,
        seed=99,
    )

    assert result.diagnostics.resampled_unique_worlds == 1
    assert result.diagnostics.accepted == 4
    assert result.diagnostics.changed == 4
    assert result.diagnostics.final_unique_worlds == 4
    assert result.diagnostics.final_max_clone_fraction == pytest.approx(0.25)
    assert len({world.world_key for world in result.worlds}) == 4
    assert result.proposal_report is not None


def test_world_identity_and_output_weights_ignore_proposal_scores() -> None:
    observation, worlds = _distinct_worlds(1)
    world = worlds[0]
    altered = replace(world, log_q=world.log_q - 7.0, log_target=world.log_target)
    assert altered.world_key == world.world_key

    result = IndependentMHRejuvenator(
        sampler=_ForbiddenSampler(())  # type: ignore[arg-type]
    ).rejuvenate(
        observation,
        _batch((altered,)),
        steps_per_chain=0,
        seed=3,
    )
    assert result.normalized_weights == (1.0,)


def test_duplicate_world_requires_one_aggregated_proposal_density() -> None:
    observation, worlds = _distinct_worlds(1)
    world = worlds[0]
    conflicting = replace(world, log_q=world.log_q - 1.0)
    initial = _batch((world, conflicting))

    with pytest.raises(
        InvalidRejuvenationBatchError,
        match="aggregated world-level proposal mass",
    ):
        IndependentMHRejuvenator(
            sampler=ConstructiveWorldSampler(sprint_backend="sequential")
        ).rejuvenate(
            observation,
            initial,
            steps_per_chain=0,
            seed=3,
        )


def test_three_world_independent_mh_transition_matrix_is_exact() -> None:
    observation, base_worlds = _distinct_worlds(3)
    constraints = CompiledMarshalConstraints.from_observation(observation)
    target = ConstraintUniformTarget()
    raw_masses = (3 / 8, 1 / 4, 1 / 8)
    accepted_mass = sum(raw_masses)
    proposal_masses = tuple(mass / accepted_mass for mass in raw_masses)
    worlds = tuple(
        replace(world, log_q=math.log(raw_mass), log_target=0.0)
        for world, raw_mass in zip(base_worlds, raw_masses, strict=True)
    )

    transition = [[0.0] * 3 for _ in range(3)]
    stationary_acceptance = 0.0
    stationary_moves = 0.0
    for current_index, current in enumerate(worlds):
        row_acceptance = 0.0
        row_moves = 0.0
        for proposal_index, proposal in enumerate(worlds):
            log_alpha = independent_mh_log_acceptance(
                current,
                proposal,
                constraints=constraints,
                target=target,
            )
            alpha = math.exp(log_alpha)
            attempted_mass = proposal_masses[proposal_index]
            row_acceptance += attempted_mass * alpha
            if proposal_index == current_index:
                transition[current_index][current_index] += attempted_mass
            else:
                move_mass = attempted_mass * alpha
                transition[current_index][proposal_index] += move_mass
                transition[current_index][current_index] += attempted_mass - move_mass
                row_moves += move_mass

            # Dividing every raw mass by its common 3/4 retry normalizer does
            # not change any MH ratio.
            normalized_log_alpha = min(
                0.0,
                math.log(proposal_masses[current_index])
                - math.log(proposal_masses[proposal_index]),
            )
            assert log_alpha == pytest.approx(normalized_log_alpha)
        stationary_acceptance += row_acceptance / 3
        stationary_moves += row_moves / 3

    expected = (
        (1 / 2, 1 / 3, 1 / 6),
        (1 / 3, 1 / 2, 1 / 6),
        (1 / 6, 1 / 6, 2 / 3),
    )
    for row, expected_row in zip(transition, expected, strict=True):
        assert row == pytest.approx(expected_row)
        assert sum(row) == pytest.approx(1.0)
    for row in range(3):
        for column in range(3):
            assert transition[row][column] == pytest.approx(
                transition[column][row]
            )
    stationary_after_step = tuple(
        sum((1 / 3) * transition[row][column] for row in range(3))
        for column in range(3)
    )
    assert stationary_after_step == pytest.approx((1 / 3,) * 3)
    assert stationary_acceptance == pytest.approx(7 / 9)
    assert stationary_moves == pytest.approx(4 / 9)


def test_zero_random_draw_is_safe_for_a_nonzero_acceptance_probability() -> None:
    class ZeroRng:
        @staticmethod
        def random() -> float:
            return 0.0

    assert _accept_log_probability(-1.0, ZeroRng())  # type: ignore[arg-type]
