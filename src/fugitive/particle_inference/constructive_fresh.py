"""Build particle beliefs from one auditable constructive sample batch.

The sampler decides which complete worlds are proposed. This module decides
how those proposal draws become particle mass. Keeping that conversion in one
place makes the BIR-1/BIR-2U/BIR-2S comparison literal instead of relying on
three similar implementations.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

from fugitive.inference.constructive_metadata import (
    constructive_observation_hash,
)
from fugitive.inference.worlds import ConstructiveSampleBatch
from fugitive.model import Observation

from .state import MarshalParticleBelief


SELF_NORMALIZED_IMPORTANCE_WEIGHTING_ID = "self-normalized-importance-v1"
UNIFORM_ACCEPTED_PROPOSALS_WEIGHTING_ID = "uniform-accepted-proposals-v1"
SEQUENTIAL_CONSTRUCTIVE_SPRINT_BACKEND = "sequential"
SEQUENTIAL_CONSTRUCTIVE_BELIEF_SEED_DOMAIN = (
    "fugitive.marshal.sequential-constructive-belief.v1"
)


class ConstructiveWeightingStrategy(str, Enum):
    """How accepted proposal draws contribute to the empirical belief."""

    SELF_NORMALIZED_IMPORTANCE = SELF_NORMALIZED_IMPORTANCE_WEIGHTING_ID
    UNIFORM_ACCEPTED_PROPOSALS = UNIFORM_ACCEPTED_PROPOSALS_WEIGHTING_ID

    def particle_weights(
        self,
        batch_size: int,
        importance_weights: tuple[float, ...],
    ) -> tuple[float, ...]:
        if self is ConstructiveWeightingStrategy.SELF_NORMALIZED_IMPORTANCE:
            return importance_weights
        if batch_size <= 0:  # pragma: no cover - complete-batch invariant
            return ()
        return (1.0 / batch_size,) * batch_size


@dataclass(frozen=True, slots=True)
class ConstructiveBeliefBuild:
    """A particle belief plus reference SNIS weights for its proposals."""

    belief: MarshalParticleBelief
    importance_weights: tuple[float, ...]


def is_complete_constructive_batch(
    batch: ConstructiveSampleBatch,
    expected_particles: int,
) -> bool:
    """Whether a batch is complete and its proposal densities are auditable."""

    report = batch.report
    return (
        report.requested == expected_particles
        and report.produced == expected_particles
        and len(batch.worlds) == expected_particles
        and not report.degraded
        and report.importance_valid
        and report.termination_reason is None
    )


def build_constructive_belief(
    observation: Observation,
    batch: ConstructiveSampleBatch,
    *,
    expected_particles: int,
    weighting: ConstructiveWeightingStrategy,
) -> ConstructiveBeliefBuild:
    """Convert every accepted proposal draw into one particle entry.

    Repeated physical worlds intentionally remain repeated entries. Under
    unweighted sampling, ``A, A, B`` therefore has aggregated masses
    ``2/3, 1/3``, exactly preserving proposal multiplicity.
    """

    if batch.observation_hash != constructive_observation_hash(observation):
        raise ValueError(
            "constructive sample batch belongs to a different observation"
        )
    if not is_complete_constructive_batch(batch, expected_particles):
        raise ValueError("constructive sample batch is incomplete or unauditable")
    importance_weights = tuple(batch.normalized_weights)
    if (
        len(importance_weights) != expected_particles
        or any(
            not math.isfinite(weight) or weight < 0.0
            for weight in importance_weights
        )
        or not math.isclose(
            math.fsum(importance_weights),
            1.0,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    ):
        raise RuntimeError(
            "constructive sampler returned invalid normalized importance weights"
        )
    weights = weighting.particle_weights(
        expected_particles,
        importance_weights,
    )
    if (
        len(weights) != expected_particles
        or any(not math.isfinite(weight) or weight < 0.0 for weight in weights)
        or not math.isclose(
            math.fsum(weights),
            1.0,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    ):  # pragma: no cover - fixed weighting strategies are total
        raise RuntimeError("constructive weighting strategy returned invalid weights")

    report = batch.report
    belief = MarshalParticleBelief(
        observation,
        tuple(
            world.to_particle(weight=weight)
            for world, weight in zip(batch.worlds, weights, strict=True)
        ),
        sampling_attempts=report.proposals,
        sampling_accepted=report.produced,
        sampling_exhausted=False,
    )
    return ConstructiveBeliefBuild(belief, importance_weights)


__all__ = [
    "ConstructiveBeliefBuild",
    "ConstructiveWeightingStrategy",
    "SEQUENTIAL_CONSTRUCTIVE_BELIEF_SEED_DOMAIN",
    "SEQUENTIAL_CONSTRUCTIVE_SPRINT_BACKEND",
    "SELF_NORMALIZED_IMPORTANCE_WEIGHTING_ID",
    "UNIFORM_ACCEPTED_PROPOSALS_WEIGHTING_ID",
    "build_constructive_belief",
    "is_complete_constructive_batch",
]
