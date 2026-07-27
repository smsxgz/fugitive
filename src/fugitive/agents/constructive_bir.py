"""BIR-2S Marshal backed by constructive, importance-weighted worlds.

The action scoring and stochastic choice hierarchy are supplied by the shared
belief-informed action-policy component.  Belief construction is an injected
backend: every new Marshal observation is sampled independently by
:class:`ConstructiveWorldSampler`, and a belief is published only when the
sampler produced the complete requested batch with valid importance weights.
"""

from __future__ import annotations

import random
from typing import Callable

from fugitive.inference_diagnostics import (
    BeliefBackendDiagnosticsSnapshot,
    ConstructiveInferenceWorkDiagnostics,
    ConstructiveSamplingWorkDiagnostics,
    InferenceOperation,
    belief_quality_diagnostics,
    constructive_sampling_work_diagnostics,
)
from fugitive.inference.constructive_metadata import (
    constructive_proposal_kernel_id,
)
from fugitive.inference.constructive_sampler import ConstructiveWorldSampler
from fugitive.inference.worlds import (
    ConstraintUniformTarget,
    ConstructiveSamplingReport,
)
from fugitive.model import Observation, Role
from fugitive.observation_protocol import stable_observation_seed
from fugitive.particle_belief import DEFAULT_PARTICLE_COUNT, MarshalParticleBelief
from fugitive.particle_inference.constructive_fresh import (
    ConstructiveWeightingStrategy,
    SEQUENTIAL_CONSTRUCTIVE_BELIEF_SEED_DOMAIN,
    SEQUENTIAL_CONSTRUCTIVE_SPRINT_BACKEND,
    SELF_NORMALIZED_IMPORTANCE_WEIGHTING_ID,
    UNIFORM_ACCEPTED_PROPOSALS_WEIGHTING_ID,
    build_constructive_belief,
    is_complete_constructive_batch,
)

from .bir_common import (
    DEFAULT_EPSILON,
    DEFAULT_MANHUNT_ALPHA,
    DEFAULT_MANHUNT_EPSILON,
    DEFAULT_TEMPERATURE,
)
from .marshal_belief_policy import (
    DEFAULT_MAX_GUESS_CANDIDATES,
    BeliefInformedMarshalActionPolicy,
    ComposedBeliefInformedRandomMarshalAgent,
    marshal_rng_and_belief_salt,
)


CONSTRUCTIVE_SPRINT_BACKEND = SEQUENTIAL_CONSTRUCTIVE_SPRINT_BACKEND
BIR2S_ALGORITHM_ID = "bir-2s-constructive-snis-v1"
BIR2_REFERENCE_TARGET_ID = ConstraintUniformTarget.target_id
BIR2_PROPOSAL_KERNEL_ID = constructive_proposal_kernel_id(
    CONSTRUCTIVE_SPRINT_BACKEND
)
BIR2_BELIEF_SEED_DOMAIN = SEQUENTIAL_CONSTRUCTIVE_BELIEF_SEED_DOMAIN
BIR2_WEIGHTING_ID = SELF_NORMALIZED_IMPORTANCE_WEIGHTING_ID
UNWEIGHTED_ACCEPTED_PROPOSALS_ID = (
    UNIFORM_ACCEPTED_PROPOSALS_WEIGHTING_ID
)


class ConstructiveBeliefConstructionError(RuntimeError):
    """Raised when BIR-2S cannot construct its complete particle batch."""

    def __init__(self, report: ConstructiveSamplingReport) -> None:
        self.report = report
        super().__init__(
            "BIR-2S requires a complete importance-valid constructive belief; "
            f"requested={report.requested}, produced={report.produced}, "
            f"degraded={report.degraded}, "
            f"importance_valid={report.importance_valid}, "
            f"termination_reason={report.termination_reason!r}, "
            f"exhausted_stage={report.exhausted_stage!r}, "
            f"proposals={report.proposals}, search_nodes={report.search_nodes}"
        )


class ConstructiveMarshalBeliefBackend:
    """Strict observation-local constructive inference with bounded caching."""

    def __init__(
        self,
        *,
        sampler: ConstructiveWorldSampler,
        particle_count: int,
        belief_salt: str,
        seed_domain: str,
        backend_id: str,
        weighting: ConstructiveWeightingStrategy,
        error_factory: Callable[[ConstructiveSamplingReport], RuntimeError],
    ) -> None:
        if (
            isinstance(particle_count, bool)
            or not isinstance(particle_count, int)
            or particle_count <= 0
        ):
            raise ValueError("particle_count must be a positive integer")
        for name, value in (
            ("belief_salt", belief_salt),
            ("seed_domain", seed_domain),
            ("backend_id", backend_id),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        self.sampler = sampler
        self.particle_count = particle_count
        self.belief_salt = belief_salt
        self.seed_domain = seed_domain
        self.backend_id = backend_id
        if not isinstance(weighting, ConstructiveWeightingStrategy):
            raise TypeError("weighting must be a ConstructiveWeightingStrategy")
        self.weighting = weighting
        self.weighting_id = weighting.value
        self.error_factory = error_factory
        self._cache: dict[Observation, MarshalParticleBelief] = {}
        self._sampling_work: dict[
            Observation, ConstructiveSamplingWorkDiagnostics
        ] = {}
        self._latest: MarshalParticleBelief | None = None
        self._last_operation: InferenceOperation | None = None
        self._inference_count = 0
        self._cache_hit_count = 0

    @property
    def latest_belief(self) -> MarshalParticleBelief | None:
        return self._latest

    @property
    def cache_size(self) -> int:
        return len(self._cache)

    def diagnostic_snapshot(self) -> BeliefBackendDiagnosticsSnapshot | None:
        """Return the latest belief quality and work without sampling again."""

        current = self._latest
        operation = self._last_operation
        if current is None or operation is None:
            return None
        sampling = self._sampling_work.get(current.observation)
        if sampling is None:  # pragma: no cover - cache ownership invariant
            raise RuntimeError("constructive belief has no sampling diagnostics")
        return BeliefBackendDiagnosticsSnapshot(
            backend_id=self.backend_id,
            operation=operation,
            quality=belief_quality_diagnostics(
                current,
                requested_particles=self.particle_count,
            ),
            work=ConstructiveInferenceWorkDiagnostics(
                weighting_id=self.weighting_id,
                inference_count=self._inference_count,
                cache_hit_count=self._cache_hit_count,
                cache_size=len(self._cache),
                sampling=sampling,
            ),
        )

    def infer(self, observation: Observation) -> MarshalParticleBelief:
        if observation.role is not Role.MARSHAL:
            raise ValueError("Marshal belief requires a Marshal observation")
        cached = self._cache.get(observation)
        if cached is not None:
            self._latest = cached
            self._last_operation = "cache_hit"
            self._cache_hit_count += 1
            return cached

        belief_seed = stable_observation_seed(
            observation,
            domain=self.seed_domain,
            salt=self.belief_salt,
        )
        batch = self.sampler.sample_batch(
            observation,
            particle_count=self.particle_count,
            seed=belief_seed,
        )
        report = batch.report
        if not is_complete_constructive_batch(batch, self.particle_count):
            raise self.error_factory(report)
        built = build_constructive_belief(
            observation,
            batch,
            expected_particles=self.particle_count,
            weighting=self.weighting,
        )
        sampled = built.belief
        sampling_work = constructive_sampling_work_diagnostics(
            report,
            built.importance_weights,
        )

        if len(self._cache) >= 8:
            oldest = next(iter(self._cache))
            self._cache.pop(oldest)
            self._sampling_work.pop(oldest, None)
        self._cache[observation] = sampled
        self._sampling_work[observation] = sampling_work
        self._latest = sampled
        self._last_operation = "fresh_build"
        self._inference_count += 1
        return sampled


def make_sequential_constructive_policy(
    seed: int | random.Random | None,
    *,
    rng: random.Random | None,
    epsilon: float,
    temperature: float,
    particle_count: int,
    max_guess_candidates: int,
    terminal_bonus_scale: float,
    manhunt_epsilon: float,
    manhunt_alpha: float,
    weighting: ConstructiveWeightingStrategy,
    backend_id: str,
    error_factory: Callable[[ConstructiveSamplingReport], RuntimeError],
    sampler: ConstructiveWorldSampler | None,
) -> BeliefInformedMarshalActionPolicy:
    """Wire the one shared proposal and action policy used by BIR-2S/2U."""

    policy_rng, belief_salt = marshal_rng_and_belief_salt(seed, rng)
    resolved_sampler = sampler or ConstructiveWorldSampler(
        sprint_backend=CONSTRUCTIVE_SPRINT_BACKEND
    )
    if resolved_sampler.sprint_backend != CONSTRUCTIVE_SPRINT_BACKEND:
        raise ValueError("injected sampler must use the sequential Sprint backend")
    if (
        resolved_sampler.proposal_kernel_id != BIR2_PROPOSAL_KERNEL_ID
        or resolved_sampler.target_id != BIR2_REFERENCE_TARGET_ID
    ):
        raise ValueError(
            "sequential constructive BIR requires its fixed proposal and target"
        )
    backend = ConstructiveMarshalBeliefBackend(
        sampler=resolved_sampler,
        particle_count=particle_count,
        belief_salt=belief_salt,
        seed_domain=BIR2_BELIEF_SEED_DOMAIN,
        backend_id=backend_id,
        weighting=weighting,
        error_factory=error_factory,
    )
    return BeliefInformedMarshalActionPolicy(
        rng=policy_rng,
        belief_backend=backend,
        epsilon=epsilon,
        temperature=temperature,
        max_guess_candidates=max_guess_candidates,
        terminal_bonus_scale=terminal_bonus_scale,
        manhunt_epsilon=manhunt_epsilon,
        manhunt_alpha=manhunt_alpha,
    )


class ConstructiveBeliefInformedRandomMarshalAgent(
    ComposedBeliefInformedRandomMarshalAgent
):
    """BIR-2S: shared action policy plus sequential constructive SNIS."""

    name = "constructive-belief-informed-random-marshal"

    def __init__(
        self,
        seed: int | random.Random | None = None,
        *,
        rng: random.Random | None = None,
        epsilon: float = DEFAULT_EPSILON,
        temperature: float = DEFAULT_TEMPERATURE,
        particle_count: int = DEFAULT_PARTICLE_COUNT,
        max_guess_candidates: int = DEFAULT_MAX_GUESS_CANDIDATES,
        terminal_bonus_scale: float = 1.0,
        manhunt_epsilon: float = DEFAULT_MANHUNT_EPSILON,
        manhunt_alpha: float = DEFAULT_MANHUNT_ALPHA,
        _sampler: ConstructiveWorldSampler | None = None,
    ) -> None:
        policy = make_sequential_constructive_policy(
            seed,
            rng=rng,
            epsilon=epsilon,
            temperature=temperature,
            particle_count=particle_count,
            max_guess_candidates=max_guess_candidates,
            terminal_bonus_scale=terminal_bonus_scale,
            manhunt_epsilon=manhunt_epsilon,
            manhunt_alpha=manhunt_alpha,
            weighting=ConstructiveWeightingStrategy.SELF_NORMALIZED_IMPORTANCE,
            backend_id="constructive-snis-sequential-v1",
            error_factory=ConstructiveBeliefConstructionError,
            sampler=_sampler,
        )
        super().__init__(policy)
        self.sprint_backend = CONSTRUCTIVE_SPRINT_BACKEND
        self.algorithm_id = BIR2S_ALGORITHM_ID
        self.proposal_kernel_id = BIR2_PROPOSAL_KERNEL_ID
        self.belief_seed_domain = BIR2_BELIEF_SEED_DOMAIN
        self.reference_target_id = BIR2_REFERENCE_TARGET_ID
        self.weighting_id = BIR2_WEIGHTING_ID

__all__ = [
    "BIR2S_ALGORITHM_ID",
    "BIR2_BELIEF_SEED_DOMAIN",
    "BIR2_PROPOSAL_KERNEL_ID",
    "BIR2_REFERENCE_TARGET_ID",
    "BIR2_WEIGHTING_ID",
    "ConstructiveBeliefConstructionError",
    "ConstructiveBeliefInformedRandomMarshalAgent",
    "ConstructiveMarshalBeliefBackend",
    "ConstructiveWeightingStrategy",
    "CONSTRUCTIVE_SPRINT_BACKEND",
    "UNWEIGHTED_ACCEPTED_PROPOSALS_ID",
    "make_sequential_constructive_policy",
]
