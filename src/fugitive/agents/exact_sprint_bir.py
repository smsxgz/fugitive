"""BIR-2E Marshal backed by the exact Sprint-assignment DP.

This is a deliberately slow reference agent.  It composes the shared Marshal
action policy with a constructive backend whose Sprint sampler is fixed to
``exact``.  The registered BIR-2S agent remains fixed to the sequential
importance proposal in :mod:`fugitive.agents.constructive_bir`.
"""

from __future__ import annotations

import random

from fugitive.inference.constructive_metadata import (
    constructive_proposal_kernel_id,
)
from fugitive.inference.constructive_sampler import ConstructiveWorldSampler
from fugitive.inference.worlds import (
    ConstraintUniformTarget,
    ConstructiveSamplingReport,
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
from .constructive_bir import (
    BIR2_WEIGHTING_ID,
    ConstructiveMarshalBeliefBackend,
    ConstructiveWeightingStrategy,
)


DEFAULT_EXACT_SPRINT_PARTICLE_COUNT = 128
EXACT_SPRINT_BACKEND = "exact"
EXACT_SPRINT_ALGORITHM_ID = "bir-2e-exact-sprint-dp-v1"
EXACT_SPRINT_REFERENCE_TARGET_ID = ConstraintUniformTarget.target_id
EXACT_SPRINT_PROPOSAL_KERNEL_ID = constructive_proposal_kernel_id(
    EXACT_SPRINT_BACKEND
)
EXACT_SPRINT_BELIEF_SEED_DOMAIN = (
    "fugitive.bir2e.exact-sprint-belief.v1"
)


class ExactSprintBeliefConstructionError(RuntimeError):
    """Raised when BIR-2E cannot construct its complete reference batch."""

    def __init__(self, report: ConstructiveSamplingReport) -> None:
        self.report = report
        super().__init__(
            "BIR-2E requires a complete importance-valid exact Sprint batch; "
            f"requested={report.requested}, produced={report.produced}, "
            f"degraded={report.degraded}, "
            f"importance_valid={report.importance_valid}, "
            f"termination_reason={report.termination_reason!r}, "
            f"proposals={report.proposals}, search_nodes={report.search_nodes}"
        )


class ExactSprintBeliefInformedRandomMarshalAgent(
    ComposedBeliefInformedRandomMarshalAgent
):
    """BIR-2E: shared action policy plus exact Sprint constructive SNIS."""

    name = "exact-sprint-belief-informed-random-marshal"

    def __init__(
        self,
        seed: int | random.Random | None = None,
        *,
        rng: random.Random | None = None,
        epsilon: float = DEFAULT_EPSILON,
        temperature: float = DEFAULT_TEMPERATURE,
        particle_count: int = DEFAULT_EXACT_SPRINT_PARTICLE_COUNT,
        max_guess_candidates: int = DEFAULT_MAX_GUESS_CANDIDATES,
        terminal_bonus_scale: float = 1.0,
        manhunt_epsilon: float = DEFAULT_MANHUNT_EPSILON,
        manhunt_alpha: float = DEFAULT_MANHUNT_ALPHA,
        _sampler: ConstructiveWorldSampler | None = None,
    ) -> None:
        policy_rng, belief_salt = marshal_rng_and_belief_salt(seed, rng)
        sampler = _sampler or ConstructiveWorldSampler(
            sprint_backend=EXACT_SPRINT_BACKEND
        )
        if sampler.sprint_backend != EXACT_SPRINT_BACKEND:
            raise ValueError("injected sampler must use the exact Sprint backend")
        if (
            sampler.proposal_kernel_id != EXACT_SPRINT_PROPOSAL_KERNEL_ID
            or sampler.target_id != EXACT_SPRINT_REFERENCE_TARGET_ID
        ):
            raise ValueError(
                "BIR-2E requires its fixed proposal kernel and target"
            )
        backend = ConstructiveMarshalBeliefBackend(
            sampler=sampler,
            particle_count=particle_count,
            belief_salt=belief_salt,
            seed_domain=EXACT_SPRINT_BELIEF_SEED_DOMAIN,
            backend_id="constructive-snis-exact-sprint-v1",
            weighting=(
                ConstructiveWeightingStrategy.SELF_NORMALIZED_IMPORTANCE
            ),
            error_factory=ExactSprintBeliefConstructionError,
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
        self.sprint_backend = EXACT_SPRINT_BACKEND
        self.algorithm_id = EXACT_SPRINT_ALGORITHM_ID
        self.belief_seed_domain = EXACT_SPRINT_BELIEF_SEED_DOMAIN
        self.proposal_kernel_id = EXACT_SPRINT_PROPOSAL_KERNEL_ID
        self.reference_target_id = EXACT_SPRINT_REFERENCE_TARGET_ID
        self.weighting_id = BIR2_WEIGHTING_ID

__all__ = [
    "DEFAULT_EXACT_SPRINT_PARTICLE_COUNT",
    "EXACT_SPRINT_ALGORITHM_ID",
    "EXACT_SPRINT_BACKEND",
    "EXACT_SPRINT_BELIEF_SEED_DOMAIN",
    "EXACT_SPRINT_PROPOSAL_KERNEL_ID",
    "EXACT_SPRINT_REFERENCE_TARGET_ID",
    "ExactSprintBeliefConstructionError",
    "ExactSprintBeliefInformedRandomMarshalAgent",
]
