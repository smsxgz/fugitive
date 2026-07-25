"""BIR-3 Marshal using a finite-step independent-MH belief backend.

The agent composes the same action-policy component as BIR-1/BIR-2 with an
independent MCMC belief backend.  A constructive importance batch is
systematically resampled, then each chain receives a fixed number of
independent Metropolis-Hastings moves toward the policy-agnostic
constraint-uniform complete-world target.

This is a finite-step MCMC approximation, not a calibrated behavioural
posterior and not a convergence claim.  The registered offline and interactive
profiles leave the path-dependent node budget unlimited so it does not
cost-censor proposal density.
"""

from __future__ import annotations

import random

from fugitive.inference_diagnostics import (
    BeliefBackendDiagnosticsSnapshot,
    ConstructiveSamplingWorkDiagnostics,
    IndependentMHInferenceWorkDiagnostics,
    InferenceOperation,
    belief_quality_diagnostics,
    constructive_sampling_work_diagnostics,
)
from fugitive.inference.constructive_sampler import ConstructiveWorldSampler
from fugitive.inference.worlds import ConstructiveSamplingReport
from fugitive.model import Observation, Role
from fugitive.observation_protocol import stable_observation_seed
from fugitive.particle_belief import MarshalParticleBelief
from fugitive.rejuvenation import (
    IndependentMHRejuvenator,
    RejuvenationDiagnostics,
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
from .constructive_bir import (
    BIR2_PROPOSAL_KERNEL_ID,
    BIR2_REFERENCE_TARGET_ID,
    BIR2_WEIGHTING_ID,
    CONSTRUCTIVE_SPRINT_BACKEND,
)


DEFAULT_MCMC_PARTICLE_COUNT = 1_000
DEFAULT_MH_STEPS_PER_CHAIN = 1
BIR3_ALGORITHM_ID = "bir-3-sir-independent-mh-v1"
BIR3_MCMC_KERNEL_ID = "independent-metropolis-hastings-v1"
BIR3_INITIAL_WEIGHTING_ID = BIR2_WEIGHTING_ID
BIR3_REFERENCE_TARGET_ID = BIR2_REFERENCE_TARGET_ID
BIR3_INITIAL_SEED_DOMAIN = "fugitive.bir3.initial-constructive-belief.v1"
BIR3_REJUVENATION_SEED_DOMAIN = "fugitive.bir3.independent-mh.v1"


class MCMCBeliefConstructionError(RuntimeError):
    """Raised when BIR-3 cannot build its complete initial population."""

    def __init__(self, report: ConstructiveSamplingReport) -> None:
        self.report = report
        super().__init__(
            "BIR-3 requires a complete importance-valid initial batch; "
            f"requested={report.requested}, produced={report.produced}, "
            f"degraded={report.degraded}, "
            f"importance_valid={report.importance_valid}, "
            f"termination_reason={report.termination_reason!r}, "
            f"exhausted_stage={report.exhausted_stage!r}, "
            f"proposals={report.proposals}, search_nodes={report.search_nodes}"
        )


class MCMCMarshalBeliefBackend:
    """Observation-local SIR plus independent-MH inference backend."""

    backend_id = "sir-independent-mh-v1"

    def __init__(
        self,
        *,
        sampler: ConstructiveWorldSampler,
        particle_count: int,
        mh_steps_per_chain: int,
        belief_salt: str,
    ) -> None:
        if (
            isinstance(particle_count, bool)
            or not isinstance(particle_count, int)
            or particle_count <= 0
        ):
            raise ValueError("particle_count must be a positive integer")
        if (
            isinstance(mh_steps_per_chain, bool)
            or not isinstance(mh_steps_per_chain, int)
            or mh_steps_per_chain < 1
        ):
            raise ValueError("mh_steps_per_chain must be a positive integer")
        if not isinstance(belief_salt, str) or not belief_salt:
            raise ValueError("belief_salt must be a non-empty string")
        self.sampler = sampler
        self.particle_count = particle_count
        self.mh_steps_per_chain = mh_steps_per_chain
        self.belief_salt = belief_salt
        self.rejuvenator = IndependentMHRejuvenator(
            sampler=sampler,
            target=sampler.target,
        )
        self._cache: dict[Observation, MarshalParticleBelief] = {}
        self._initial_work: dict[
            Observation, ConstructiveSamplingWorkDiagnostics
        ] = {}
        self._proposal_work: dict[
            Observation, ConstructiveSamplingWorkDiagnostics
        ] = {}
        self._rejuvenation_diagnostics: dict[
            Observation, RejuvenationDiagnostics
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
        """Return final-chain quality and the latest SIR/MH work summary."""

        current = self._latest
        operation = self._last_operation
        if current is None or operation is None:
            return None
        observation = current.observation
        initial = self._initial_work.get(observation)
        proposals = self._proposal_work.get(observation)
        rejuvenation = self._rejuvenation_diagnostics.get(observation)
        if initial is None or proposals is None or rejuvenation is None:
            raise RuntimeError("MCMC belief has incomplete diagnostics")
        return BeliefBackendDiagnosticsSnapshot(
            backend_id=self.backend_id,
            operation=operation,
            quality=belief_quality_diagnostics(
                current,
                requested_particles=self.particle_count,
            ),
            work=IndependentMHInferenceWorkDiagnostics(
                initial_weighting_id=BIR3_INITIAL_WEIGHTING_ID,
                inference_count=self._inference_count,
                cache_hit_count=self._cache_hit_count,
                cache_size=len(self._cache),
                initial_sampling=initial,
                mh_proposal_sampling=proposals,
                resampled_unique_worlds=(
                    rejuvenation.resampled_unique_worlds
                ),
                chains=rejuvenation.chains,
                steps_per_chain=rejuvenation.steps_per_chain,
                mh_proposals=rejuvenation.proposals,
                accepted=rejuvenation.accepted,
                changed=rejuvenation.changed,
                acceptance_rate=rejuvenation.acceptance_rate,
                change_rate=rejuvenation.change_rate,
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

        initial_seed = stable_observation_seed(
            observation,
            domain=BIR3_INITIAL_SEED_DOMAIN,
            salt=self.belief_salt,
        )
        initial_batch = self.sampler.sample_batch(
            observation,
            particle_count=self.particle_count,
            seed=initial_seed,
        )
        initial_report = initial_batch.report
        initial_complete = (
            initial_report.requested == self.particle_count
            and initial_report.produced == self.particle_count
            and len(initial_batch.worlds) == self.particle_count
            and not initial_report.degraded
            and initial_report.importance_valid
            and initial_report.termination_reason is None
        )
        if not initial_complete:
            raise MCMCBeliefConstructionError(initial_report)

        rejuvenation_seed = stable_observation_seed(
            observation,
            domain=BIR3_REJUVENATION_SEED_DOMAIN,
            salt=self.belief_salt,
        )
        rejuvenated = self.rejuvenator.rejuvenate(
            observation,
            initial_batch,
            steps_per_chain=self.mh_steps_per_chain,
            seed=rejuvenation_seed,
        )
        proposal_report = rejuvenated.proposal_report
        if proposal_report is None:  # pragma: no cover - steps are positive
            raise RuntimeError("positive-step MCMC returned no proposal report")

        sampled = MarshalParticleBelief(
            observation,
            rejuvenated.to_particles(),
            sampling_attempts=initial_report.proposals,
            sampling_accepted=initial_report.produced,
            sampling_exhausted=False,
        )
        initial_work = constructive_sampling_work_diagnostics(
            initial_report,
            tuple(initial_batch.normalized_weights),
        )
        proposal_work = constructive_sampling_work_diagnostics(
            proposal_report,
        )
        if len(self._cache) >= 8:
            oldest = next(iter(self._cache))
            self._cache.pop(oldest)
            self._initial_work.pop(oldest, None)
            self._proposal_work.pop(oldest, None)
            self._rejuvenation_diagnostics.pop(oldest, None)
        self._cache[observation] = sampled
        self._initial_work[observation] = initial_work
        self._proposal_work[observation] = proposal_work
        self._rejuvenation_diagnostics[observation] = rejuvenated.diagnostics
        self._latest = sampled
        self._last_operation = "fresh_build"
        self._inference_count += 1
        return sampled


class MCMCBeliefInformedRandomMarshalAgent(
    ComposedBeliefInformedRandomMarshalAgent
):
    """BIR-3: shared action policy plus a finite-step SIR/MH backend."""

    name = "mcmc-belief-informed-random-marshal"

    def __init__(
        self,
        seed: int | random.Random | None = None,
        *,
        rng: random.Random | None = None,
        epsilon: float = DEFAULT_EPSILON,
        temperature: float = DEFAULT_TEMPERATURE,
        particle_count: int = DEFAULT_MCMC_PARTICLE_COUNT,
        max_guess_candidates: int = DEFAULT_MAX_GUESS_CANDIDATES,
        terminal_bonus_scale: float = 1.0,
        manhunt_epsilon: float = DEFAULT_MANHUNT_EPSILON,
        manhunt_alpha: float = DEFAULT_MANHUNT_ALPHA,
        mh_steps_per_chain: int = DEFAULT_MH_STEPS_PER_CHAIN,
        _sampler: ConstructiveWorldSampler | None = None,
    ) -> None:
        if (
            isinstance(mh_steps_per_chain, bool)
            or not isinstance(mh_steps_per_chain, int)
            or mh_steps_per_chain < 1
        ):
            raise ValueError("mh_steps_per_chain must be a positive integer")
        policy_rng, belief_salt = _marshal_rng_and_belief_salt(seed, rng)
        sampler = _sampler or ConstructiveWorldSampler(
            sprint_backend=CONSTRUCTIVE_SPRINT_BACKEND
        )
        if sampler.sprint_backend != CONSTRUCTIVE_SPRINT_BACKEND:
            raise ValueError(
                "injected sampler must use the sequential Sprint backend"
            )
        if (
            sampler.proposal_kernel_id != BIR2_PROPOSAL_KERNEL_ID
            or sampler.target_id != BIR3_REFERENCE_TARGET_ID
        ):
            raise ValueError("BIR-3 requires the BIR-2S proposal and target")
        backend = MCMCMarshalBeliefBackend(
            sampler=sampler,
            particle_count=particle_count,
            mh_steps_per_chain=mh_steps_per_chain,
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
        self.mh_steps_per_chain = mh_steps_per_chain
        self.sprint_backend = CONSTRUCTIVE_SPRINT_BACKEND
        self.algorithm_id = BIR3_ALGORITHM_ID
        self.initial_seed_domain = BIR3_INITIAL_SEED_DOMAIN
        self.rejuvenation_seed_domain = BIR3_REJUVENATION_SEED_DOMAIN
        self.proposal_kernel_id = BIR2_PROPOSAL_KERNEL_ID
        self.reference_target_id = BIR3_REFERENCE_TARGET_ID
        self.initial_weighting_id = BIR3_INITIAL_WEIGHTING_ID
        self.mcmc_kernel_id = BIR3_MCMC_KERNEL_ID

__all__ = [
    "BIR3_ALGORITHM_ID",
    "BIR3_INITIAL_WEIGHTING_ID",
    "BIR3_INITIAL_SEED_DOMAIN",
    "BIR3_MCMC_KERNEL_ID",
    "BIR3_REFERENCE_TARGET_ID",
    "BIR3_REJUVENATION_SEED_DOMAIN",
    "DEFAULT_MCMC_PARTICLE_COUNT",
    "DEFAULT_MH_STEPS_PER_CHAIN",
    "MCMCBeliefConstructionError",
    "MCMCBeliefInformedRandomMarshalAgent",
    "MCMCMarshalBeliefBackend",
]
