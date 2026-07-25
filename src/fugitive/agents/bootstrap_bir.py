"""BIR-1 Marshal agent and its heuristic bootstrap particle backend."""

from __future__ import annotations

import random

from fugitive.inference.constructive_metadata import (
    constructive_proposal_kernel_id,
)
from fugitive.inference.worlds import ConstraintUniformTarget
from fugitive.inference_diagnostics import (
    BeliefBackendDiagnosticsSnapshot,
    BootstrapInferenceWorkDiagnostics,
    InferenceOperation,
    belief_quality_diagnostics,
)
from fugitive.model import Observation, Role
from fugitive.observation_protocol import stable_observation_seed
from fugitive.particle_belief import (
    BIR1_INCREMENTAL_EXACT_COMBINATION_THRESHOLD,
    BIR1_INCREMENTAL_PLAY_PROPOSAL_LIMIT,
    BIR1_RESAMPLE_ESS_FRACTION,
    DEFAULT_PARTICLE_COUNT,
    MarshalParticleBelief,
)
from fugitive.particle_inference.constructive_fresh import (
    SEQUENTIAL_CONSTRUCTIVE_BELIEF_SEED_DOMAIN,
    SEQUENTIAL_CONSTRUCTIVE_SPRINT_BACKEND,
    UNIFORM_ACCEPTED_PROPOSALS_WEIGHTING_ID,
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
    _marshal_rng_and_belief_salt,
)


BIR1_MARSHAL_ALGORITHM_ID = "bir-1-constructive-bootstrap-pf-marshal-v1"
BIR1_BOOTSTRAP_BACKEND_ID = "constructive-bootstrap-particle-filter-v1"
BIR1_PROPOSAL_KERNEL_ID = constructive_proposal_kernel_id(
    SEQUENTIAL_CONSTRUCTIVE_SPRINT_BACKEND
)
BIR1_BELIEF_SEED_DOMAIN = SEQUENTIAL_CONSTRUCTIVE_BELIEF_SEED_DOMAIN
BIR1_REFERENCE_TARGET_ID = ConstraintUniformTarget.target_id
BIR1_WEIGHTING_ID = UNIFORM_ACCEPTED_PROPOSALS_WEIGHTING_ID
BIR1_PUBLIC_EVENT_MODEL_ID = "constraint-feasibility-only-v1"
BIR1_INCREMENTAL_TRANSITION_ID = "bounded-uniform-legal-extension-v1"


def _stable_seed(observation: Observation, salt: str) -> int:
    return stable_observation_seed(
        observation,
        domain=BIR1_BELIEF_SEED_DOMAIN,
        salt=salt,
    )


class BootstrapParticleBeliefBackend:
    """Constructive initialization plus constraint-only incremental filtering."""

    backend_id = BIR1_BOOTSTRAP_BACKEND_ID
    max_incremental_play_proposals_per_particle = (
        BIR1_INCREMENTAL_PLAY_PROPOSAL_LIMIT
    )
    exact_incremental_play_combination_threshold = (
        BIR1_INCREMENTAL_EXACT_COMBINATION_THRESHOLD
    )
    resample_ess_fraction = BIR1_RESAMPLE_ESS_FRACTION

    def __init__(
        self,
        *,
        particle_count: int,
        belief_salt: str,
    ) -> None:
        if (
            isinstance(particle_count, bool)
            or not isinstance(particle_count, int)
            or particle_count <= 0
        ):
            raise ValueError("particle_count must be a positive integer")
        if not isinstance(belief_salt, str) or not belief_salt:
            raise ValueError("belief_salt must be a non-empty string")
        self.particle_count = particle_count
        self.belief_salt = belief_salt
        self._cache: dict[Observation, MarshalParticleBelief] = {}
        self._latest: MarshalParticleBelief | None = None
        self._last_operation: InferenceOperation | None = None
        self._inference_count = 0
        self._cache_hit_count = 0
        self._incremental_update_count = 0
        self._fresh_constructions = 0
        self._recovery_resets = 0
        self._systematic_resampling_count = 0

    @property
    def latest_belief(self) -> MarshalParticleBelief | None:
        return self._latest

    @property
    def cache_size(self) -> int:
        return len(self._cache)

    def diagnostic_snapshot(self) -> BeliefBackendDiagnosticsSnapshot | None:
        """Return the latest belief and work summary without rerunning inference."""

        current = self._latest
        operation = self._last_operation
        if current is None or operation is None:
            return None
        return BeliefBackendDiagnosticsSnapshot(
            backend_id=self.backend_id,
            operation=operation,
            quality=belief_quality_diagnostics(
                current,
                requested_particles=self.particle_count,
            ),
            work=BootstrapInferenceWorkDiagnostics(
                inference_count=self._inference_count,
                cache_hit_count=self._cache_hit_count,
                fresh_build_count=self._fresh_constructions,
                incremental_update_count=self._incremental_update_count,
                support_extinction_reset_count=self._recovery_resets,
                systematic_resampling_count=(
                    self._systematic_resampling_count
                ),
                origin_fresh_sampling_proposals=current.sampling_attempts,
                origin_fresh_sampling_accepted=current.sampling_accepted,
                origin_fresh_sampling_acceptance_rate=(
                    current.sampling_acceptance_rate
                ),
                minimum_pre_resample_entry_ess=(
                    current.pre_resample_effective_sample_size
                ),
                origin_fresh_sampling_exhausted=(
                    current.sampling_exhausted
                ),
                cache_size=len(self._cache),
            ),
        )

    def _fresh_belief(
        self,
        observation: Observation,
        *,
        belief_seed: int,
    ) -> MarshalParticleBelief:
        sampled = MarshalParticleBelief.from_observation(
            observation,
            particle_count=self.particle_count,
            seed=belief_seed,
        )
        if len(sampled.particles) != self.particle_count:
            raise RuntimeError(
                "constructive initialization returned an incomplete population"
            )
        self._fresh_constructions += 1
        return sampled

    def infer(self, observation: Observation) -> MarshalParticleBelief:
        if observation.role is not Role.MARSHAL:
            raise ValueError("Marshal belief requires a Marshal observation")
        cached = self._cache.get(observation)
        if cached is not None:
            self._latest = cached
            self._last_operation = "cache_hit"
            self._cache_hit_count += 1
            return cached

        belief_seed = _stable_seed(observation, self.belief_salt)
        sampled: MarshalParticleBelief
        operation: InferenceOperation
        if self._latest is not None:
            previous = self._latest
            sampled = previous.advance_to(
                observation,
                particle_count=self.particle_count,
                seed=belief_seed,
                max_play_proposals_per_particle=(
                    self.max_incremental_play_proposals_per_particle
                ),
                resample_ess_fraction=self.resample_ess_fraction,
            )
            self._incremental_update_count += 1
            self._systematic_resampling_count += max(
                0,
                sampled.resampling_count - previous.resampling_count,
            )
            if sampled.is_empty:
                sampled = self._fresh_belief(
                    observation,
                    belief_seed=belief_seed,
                )
                self._recovery_resets += 1
                operation = "support_extinction_reset"
            else:
                operation = "incremental_update"
        else:
            sampled = self._fresh_belief(
                observation,
                belief_seed=belief_seed,
            )
            operation = "fresh_build"
        if len(self._cache) >= 8:
            self._cache.pop(next(iter(self._cache)))
        self._cache[observation] = sampled
        self._latest = sampled
        self._last_operation = operation
        self._inference_count += 1
        return sampled


class BeliefInformedRandomMarshalAgent(
    ComposedBeliefInformedRandomMarshalAgent
):
    """BIR-1 agent composed from the shared policy and bootstrap backend."""

    name = "belief-informed-random-marshal"

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
    ) -> None:
        policy_rng, belief_salt = _marshal_rng_and_belief_salt(seed, rng)
        backend = BootstrapParticleBeliefBackend(
            particle_count=particle_count,
            belief_salt=belief_salt,
        )
        policy = BeliefInformedMarshalActionPolicy(
            rng=policy_rng,
            belief_backend=backend,
            epsilon=epsilon,
            temperature=temperature,
            max_guess_candidates=max_guess_candidates,
            terminal_bonus_scale=terminal_bonus_scale,
            manhunt_epsilon=manhunt_epsilon,
            manhunt_alpha=manhunt_alpha,
        )
        super().__init__(policy)
        self.algorithm_id = BIR1_MARSHAL_ALGORITHM_ID
        self.belief_backend_id = BIR1_BOOTSTRAP_BACKEND_ID
        self.max_incremental_play_proposals_per_particle = (
            BIR1_INCREMENTAL_PLAY_PROPOSAL_LIMIT
        )
        self.exact_incremental_play_combination_threshold = (
            BIR1_INCREMENTAL_EXACT_COMBINATION_THRESHOLD
        )
        self.resample_ess_fraction = BIR1_RESAMPLE_ESS_FRACTION
        self.sprint_backend = SEQUENTIAL_CONSTRUCTIVE_SPRINT_BACKEND
        self.proposal_kernel_id = BIR1_PROPOSAL_KERNEL_ID
        self.belief_seed_domain = BIR1_BELIEF_SEED_DOMAIN
        self.reference_target_id = BIR1_REFERENCE_TARGET_ID
        self.weighting_id = BIR1_WEIGHTING_ID
        self.public_event_model_id = BIR1_PUBLIC_EVENT_MODEL_ID
        self.incremental_transition_id = BIR1_INCREMENTAL_TRANSITION_ID

__all__ = [
    "BIR1_BOOTSTRAP_BACKEND_ID",
    "BIR1_BELIEF_SEED_DOMAIN",
    "BIR1_INCREMENTAL_TRANSITION_ID",
    "BIR1_MARSHAL_ALGORITHM_ID",
    "BIR1_PROPOSAL_KERNEL_ID",
    "BIR1_PUBLIC_EVENT_MODEL_ID",
    "BIR1_REFERENCE_TARGET_ID",
    "BIR1_WEIGHTING_ID",
    "BeliefInformedRandomMarshalAgent",
    "BootstrapParticleBeliefBackend",
]
