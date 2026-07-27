"""Comparable, information-safe diagnostics for Marshal belief inference.

The rule trace answers *what happened in the game*.  These records answer a
different teaching question: *what approximation did the Marshal use for one
decision?*  They are deliberately kept outside replay manifests and state
fingerprints.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, Protocol, TypeAlias, runtime_checkable

from fugitive.game.model import Phase, Role
from fugitive.shared.reproducibility import JSONValue

from .constructive.worlds import ConstructiveSamplingReport
from .particle_belief import MarshalParticleBelief
from .path_belief import PathBelief


InferenceOperation: TypeAlias = Literal[
    "fresh_build",
    "incremental_update",
    "support_extinction_reset",
    "cache_hit",
]


@dataclass(frozen=True, slots=True)
class BeliefQualityDiagnostics:
    """Quality of the particle population actually used for a decision.

    ``hard_route_support_count`` counts route-value paths admitted by
    :class:`PathBelief`.  It is useful context for posterior concentration, but
    it is neither a count of complete worlds nor a posterior probability.
    """

    requested_particles: int
    particle_entries: int
    unique_worlds: int
    entry_ess: float
    world_ess: float
    max_world_mass: float
    unique_hidden_routes: int
    hidden_route_ess: float
    max_hidden_route_mass: float
    hard_route_support_count: int

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "requested_particles": self.requested_particles,
            "particle_entries": self.particle_entries,
            "unique_worlds": self.unique_worlds,
            "entry_ess": self.entry_ess,
            "world_ess": self.world_ess,
            "max_world_mass": self.max_world_mass,
            "unique_hidden_routes": self.unique_hidden_routes,
            "hidden_route_ess": self.hidden_route_ess,
            "max_hidden_route_mass": self.max_hidden_route_mass,
            "hard_route_support_count": self.hard_route_support_count,
        }


@dataclass(frozen=True, slots=True)
class BootstrapInferenceWorkDiagnostics:
    """Cumulative work plus latest-population sampling data for BIR-1."""

    inference_count: int
    cache_hit_count: int
    fresh_build_count: int
    incremental_update_count: int
    support_extinction_reset_count: int
    systematic_resampling_count: int
    origin_fresh_sampling_proposals: int
    origin_fresh_sampling_accepted: int
    origin_fresh_sampling_acceptance_rate: float
    minimum_pre_resample_entry_ess: float
    origin_fresh_sampling_exhausted: bool
    cache_size: int

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "kind": "bootstrap",
            "inference_count": self.inference_count,
            "cache_hit_count": self.cache_hit_count,
            "fresh_build_count": self.fresh_build_count,
            "incremental_update_count": self.incremental_update_count,
            "support_extinction_reset_count": (
                self.support_extinction_reset_count
            ),
            "systematic_resampling_count": self.systematic_resampling_count,
            "origin_fresh_sampling_proposals": (
                self.origin_fresh_sampling_proposals
            ),
            "origin_fresh_sampling_accepted": (
                self.origin_fresh_sampling_accepted
            ),
            "origin_fresh_sampling_acceptance_rate": (
                self.origin_fresh_sampling_acceptance_rate
            ),
            "minimum_pre_resample_entry_ess": (
                self.minimum_pre_resample_entry_ess
            ),
            "origin_fresh_sampling_exhausted": (
                self.origin_fresh_sampling_exhausted
            ),
            "cache_size": self.cache_size,
        }


@dataclass(frozen=True, slots=True)
class ConstructiveSamplingWorkDiagnostics:
    """Work and importance-weight quality for one constructive sample batch."""

    requested: int
    produced: int
    proposals: int
    dead_end_route_proposals: int
    rejected_targets: int
    search_nodes: int
    unique_proposed_worlds: int
    proposal_importance_ess: float | None
    max_normalized_importance_weight: float | None

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "requested": self.requested,
            "produced": self.produced,
            "proposals": self.proposals,
            "dead_end_route_proposals": self.dead_end_route_proposals,
            "rejected_targets": self.rejected_targets,
            "search_nodes": self.search_nodes,
            "unique_proposed_worlds": self.unique_proposed_worlds,
            "proposal_importance_ess": self.proposal_importance_ess,
            "max_normalized_importance_weight": (
                self.max_normalized_importance_weight
            ),
        }


@dataclass(frozen=True, slots=True)
class ConstructiveInferenceWorkDiagnostics:
    """Observation-local constructive work plus the decision weighting rule.

    The sampling diagnostics always describe the constructive proposal and
    its reference importance weights.  ``weighting_id`` separately records
    which weights the agent actually placed on the accepted worlds.
    """

    weighting_id: str
    inference_count: int
    cache_hit_count: int
    cache_size: int
    sampling: ConstructiveSamplingWorkDiagnostics

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "kind": "constructive",
            "weighting_id": self.weighting_id,
            "inference_count": self.inference_count,
            "cache_hit_count": self.cache_hit_count,
            "cache_size": self.cache_size,
            "sampling": self.sampling.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class IndependentMHInferenceWorkDiagnostics:
    """Initial SIR and finite-step independent-MH work for BIR-3."""

    initial_weighting_id: str
    inference_count: int
    cache_hit_count: int
    cache_size: int
    initial_sampling: ConstructiveSamplingWorkDiagnostics
    mh_proposal_sampling: ConstructiveSamplingWorkDiagnostics
    resampled_unique_worlds: int
    chains: int
    steps_per_chain: int
    mh_proposals: int
    accepted: int
    changed: int
    acceptance_rate: float
    change_rate: float

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "kind": "sir_independent_mh",
            "initial_weighting_id": self.initial_weighting_id,
            "inference_count": self.inference_count,
            "cache_hit_count": self.cache_hit_count,
            "cache_size": self.cache_size,
            "initial_sampling": self.initial_sampling.to_dict(),
            "mh_proposal_sampling": self.mh_proposal_sampling.to_dict(),
            "resampled_unique_worlds": self.resampled_unique_worlds,
            "chains": self.chains,
            "steps_per_chain": self.steps_per_chain,
            "mh_proposals": self.mh_proposals,
            "accepted": self.accepted,
            "changed": self.changed,
            "acceptance_rate": self.acceptance_rate,
            "change_rate": self.change_rate,
        }


InferenceWorkDiagnostics: TypeAlias = (
    BootstrapInferenceWorkDiagnostics
    | ConstructiveInferenceWorkDiagnostics
    | IndependentMHInferenceWorkDiagnostics
)


@dataclass(frozen=True, slots=True)
class BeliefBackendDiagnosticsSnapshot:
    """Algorithm-independent snapshot returned by a belief backend."""

    backend_id: str
    operation: InferenceOperation
    quality: BeliefQualityDiagnostics
    work: InferenceWorkDiagnostics

    def for_algorithm(self, algorithm_id: str) -> "InferenceDiagnosticsSnapshot":
        return InferenceDiagnosticsSnapshot(
            algorithm_id=algorithm_id,
            backend_id=self.backend_id,
            operation=self.operation,
            quality=self.quality,
            work=self.work,
        )


@dataclass(frozen=True, slots=True)
class InferenceDiagnosticsSnapshot:
    """One agent-labelled belief snapshot, ready for an experiment event."""

    algorithm_id: str
    backend_id: str
    operation: InferenceOperation
    quality: BeliefQualityDiagnostics
    work: InferenceWorkDiagnostics

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "algorithm_id": self.algorithm_id,
            "backend_id": self.backend_id,
            "operation": self.operation,
            "quality": self.quality.to_dict(),
            "work": self.work.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class InferenceEvent:
    """A decision-indexed diagnostics side channel, separate from replay."""

    decision: int
    round_number: int
    phase: Phase
    role: Role
    diagnostics: InferenceDiagnosticsSnapshot

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "decision": self.decision,
            "round_number": self.round_number,
            "phase": self.phase.value,
            "role": self.role.value,
            "diagnostics": self.diagnostics.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class InferenceDiagnosticFailure:
    """A non-fatal failure while reading the diagnostics side channel.

    The associated game decision has already been committed.  This record is
    therefore deliberately separate from replay manifests and run errors.
    """

    decision: int
    round_number: int
    phase: Phase
    role: Role
    error_type: str
    message: str

    @classmethod
    def from_exception(
        cls,
        *,
        decision: int,
        round_number: int,
        phase: Phase,
        role: Role,
        error: Exception,
    ) -> "InferenceDiagnosticFailure":
        try:
            message = str(error)
        except Exception:  # pragma: no cover - defensive exception formatting
            message = "<error message unavailable>"
        return cls(
            decision=decision,
            round_number=round_number,
            phase=phase,
            role=role,
            error_type=type(error).__name__,
            message=message,
        )

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "decision": self.decision,
            "round_number": self.round_number,
            "phase": self.phase.value,
            "role": self.role.value,
            "error": {
                "type": self.error_type,
                "message": self.message,
            },
        }


@runtime_checkable
class InferenceDiagnosticsProvider(Protocol):
    """Agent interface for reading an already-computed diagnostics snapshot."""

    def inference_diagnostics(self) -> InferenceDiagnosticsSnapshot | None: ...


def belief_quality_diagnostics(
    belief: MarshalParticleBelief,
    *,
    requested_particles: int,
) -> BeliefQualityDiagnostics:
    """Summarize a belief without sampling, mutation, or random-number use."""

    hard_route_support = PathBelief.from_observation(
        belief.observation
    ).total_paths
    return BeliefQualityDiagnostics(
        requested_particles=requested_particles,
        particle_entries=len(belief.particles),
        unique_worlds=belief.unique_particle_count,
        entry_ess=belief.effective_sample_size,
        world_ess=belief.world_effective_sample_size,
        max_world_mass=belief.max_world_mass,
        unique_hidden_routes=belief.unique_hidden_hypothesis_count,
        hidden_route_ess=belief.hidden_hypothesis_effective_sample_size,
        max_hidden_route_mass=belief.max_hidden_hypothesis_mass,
        hard_route_support_count=hard_route_support,
    )


def constructive_sampling_work_diagnostics(
    report: ConstructiveSamplingReport,
    normalized_weights: tuple[float, ...] | None = None,
) -> ConstructiveSamplingWorkDiagnostics:
    """Convert one sampler report into compact, comparable work diagnostics.

    Importance statistics are optional because independent-MH proposals use
    their proposal densities individually rather than as a normalized SNIS
    population.
    """

    importance_ess: float | None = None
    max_weight: float | None = None
    if normalized_weights is not None:
        denominator = math.fsum(weight * weight for weight in normalized_weights)
        importance_ess = 0.0 if denominator <= 0.0 else 1.0 / denominator
        max_weight = max(normalized_weights, default=0.0)
    return ConstructiveSamplingWorkDiagnostics(
        requested=report.requested,
        produced=report.produced,
        proposals=report.proposals,
        dead_end_route_proposals=report.dead_end_route_proposals,
        rejected_targets=report.rejected_targets,
        search_nodes=report.search_nodes,
        unique_proposed_worlds=report.unique_worlds,
        proposal_importance_ess=importance_ess,
        max_normalized_importance_weight=max_weight,
    )


def read_inference_diagnostics(
    agent: object,
) -> InferenceDiagnosticsSnapshot | None:
    """Read diagnostics only from agents that explicitly implement the API."""

    if not isinstance(agent, InferenceDiagnosticsProvider):
        return None
    return agent.inference_diagnostics()


__all__ = [
    "BeliefBackendDiagnosticsSnapshot",
    "BeliefQualityDiagnostics",
    "BootstrapInferenceWorkDiagnostics",
    "ConstructiveInferenceWorkDiagnostics",
    "ConstructiveSamplingWorkDiagnostics",
    "IndependentMHInferenceWorkDiagnostics",
    "InferenceDiagnosticsProvider",
    "InferenceDiagnosticsSnapshot",
    "InferenceDiagnosticFailure",
    "InferenceEvent",
    "InferenceOperation",
    "InferenceWorkDiagnostics",
    "belief_quality_diagnostics",
    "constructive_sampling_work_diagnostics",
    "read_inference_diagnostics",
]
