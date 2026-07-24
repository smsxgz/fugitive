"""BIR-3 Marshal using a finite-step independent-MH belief backend.

The policy scores and samples actions exactly as BIR-2 does.  Only belief
construction changes: a constructive importance batch is systematically
resampled, then each chain receives a fixed number of independent
Metropolis-Hastings moves toward the policy-agnostic constraint-uniform
complete-world target.

This is a finite-step MCMC approximation, not a calibrated behavioural
posterior and not a convergence claim.  The registered offline and interactive
profiles leave the path-dependent node budget unlimited so it does not
cost-censor proposal density.
"""

from __future__ import annotations

import random

from fugitive.constructive_belief import (
    ConstructiveSamplingReport,
    ConstructiveWorldSampler,
)
from fugitive.model import Observation, Role
from fugitive.observation_protocol import stable_observation_seed
from fugitive.particle_belief import MarshalParticleBelief
from fugitive.rejuvenation import (
    IndependentMHRejuvenator,
    RejuvenationDiagnostics,
)

from .belief_informed_random import (
    DEFAULT_EPSILON,
    DEFAULT_MANHUNT_ALPHA,
    DEFAULT_MANHUNT_EPSILON,
    DEFAULT_MAX_GUESS_CANDIDATES,
    DEFAULT_TEMPERATURE,
)
from .constructive_bir import ConstructiveBeliefInformedRandomMarshalAgent


DEFAULT_MCMC_PARTICLE_COUNT = 1_000
DEFAULT_MH_STEPS_PER_CHAIN = 1
_INITIAL_SEED_DOMAIN = "fugitive.bir3.initial-constructive-belief.v1"
_REJUVENATION_SEED_DOMAIN = "fugitive.bir3.independent-mh.v1"


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


class MCMCBeliefInformedRandomMarshalAgent(
    ConstructiveBeliefInformedRandomMarshalAgent
):
    """Marshal policy with a SIR warm start and finite-step independent MH."""

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
        super().__init__(
            seed,
            rng=rng,
            epsilon=epsilon,
            temperature=temperature,
            particle_count=particle_count,
            max_guess_candidates=max_guess_candidates,
            terminal_bonus_scale=terminal_bonus_scale,
            manhunt_epsilon=manhunt_epsilon,
            manhunt_alpha=manhunt_alpha,
            _sampler=_sampler,
        )
        self.mh_steps_per_chain = mh_steps_per_chain
        # Initial worlds and MH proposals must carry one proposal density.
        # Sharing this exact sampler object enforces that precondition here.
        self._rejuvenator = IndependentMHRejuvenator(
            sampler=self._constructive_sampler,
            target=self._constructive_sampler.target,
        )
        self._rejuvenation_diagnostics: dict[
            Observation, RejuvenationDiagnostics
        ] = {}
        self._proposal_reports: dict[
            Observation, ConstructiveSamplingReport
        ] = {}

    @property
    def last_initial_sampling_report(self) -> ConstructiveSamplingReport | None:
        """Constructive report for the latest successful initial population."""

        return self.last_sampling_report

    @property
    def last_mh_proposal_report(
        self,
    ) -> ConstructiveSamplingReport | None:
        """Constructive report for the latest successful MH proposal batch."""

        if self._latest_belief is None:
            return None
        return self._proposal_reports.get(self._latest_belief.observation)

    @property
    def last_proposal_sampling_report(
        self,
    ) -> ConstructiveSamplingReport | None:
        """Backward-readable alias for :attr:`last_mh_proposal_report`."""

        return self.last_mh_proposal_report

    @property
    def last_rejuvenation_diagnostics(
        self,
    ) -> RejuvenationDiagnostics | None:
        """SIR/MH quality counters for the latest successful belief."""

        if self._latest_belief is None:
            return None
        return self._rejuvenation_diagnostics.get(
            self._latest_belief.observation
        )

    def belief(self, observation: Observation) -> MarshalParticleBelief:
        """Construct, rejuvenate, and cache one equal-weight belief."""

        if observation.role is not Role.MARSHAL:
            raise ValueError("Marshal belief requires a Marshal observation")
        cached = self._belief_cache.get(observation)
        if cached is not None:
            self._latest_belief = cached
            return cached

        initial_seed = stable_observation_seed(
            observation,
            domain=_INITIAL_SEED_DOMAIN,
            salt=self._belief_salt,
        )
        initial_batch = self._constructive_sampler.sample_batch(
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
            domain=_REJUVENATION_SEED_DOMAIN,
            salt=self._belief_salt,
        )
        rejuvenated = self._rejuvenator.rejuvenate(
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
            # Keep generic particle diagnostics about the initial constructive
            # stage.  MH acceptance has separate, non-comparable counters.
            sampling_attempts=initial_report.proposals,
            sampling_accepted=initial_report.produced,
            sampling_exhausted=False,
        )

        if len(self._belief_cache) >= 8:
            oldest = next(iter(self._belief_cache))
            self._belief_cache.pop(oldest)
            self._constructive_reports.pop(oldest, None)
            self._proposal_reports.pop(oldest, None)
            self._rejuvenation_diagnostics.pop(oldest, None)
        self._belief_cache[observation] = sampled
        self._constructive_reports[observation] = initial_report
        self._proposal_reports[observation] = proposal_report
        self._rejuvenation_diagnostics[observation] = (
            rejuvenated.diagnostics
        )
        self._latest_belief = sampled
        self._belief_fresh_resamples += 1
        return sampled


__all__ = [
    "DEFAULT_MCMC_PARTICLE_COUNT",
    "DEFAULT_MH_STEPS_PER_CHAIN",
    "MCMCBeliefConstructionError",
    "MCMCBeliefInformedRandomMarshalAgent",
]
