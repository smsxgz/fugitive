"""BIR-2U: the unweighted-proposal ablation for constructive BIR.

BIR-2U deliberately uses the same sequential constructive proposal, reference
target, observation-derived seed, and Marshal action policy as BIR-2S.  Its
only algorithmic change is to give every accepted proposal weight ``1 / N``
instead of applying the self-normalized importance correction.  Comparing the
two agents therefore isolates the effect of importance weighting.
"""

from __future__ import annotations

import random

from ..inference.constructive.sampler import ConstructiveWorldSampler
from ..inference.constructive.worlds import ConstructiveSamplingReport
from ..inference.particle_belief import DEFAULT_PARTICLE_COUNT

from ...common.bir_common import (
    DEFAULT_EPSILON,
    DEFAULT_MANHUNT_ALPHA,
    DEFAULT_MANHUNT_EPSILON,
    DEFAULT_TEMPERATURE,
)
from .snis import (
    BIR2_BELIEF_SEED_DOMAIN,
    BIR2_PROPOSAL_KERNEL_ID,
    BIR2_REFERENCE_TARGET_ID,
    CONSTRUCTIVE_SPRINT_BACKEND,
    UNWEIGHTED_ACCEPTED_PROPOSALS_ID,
    ConstructiveWeightingStrategy,
    make_sequential_constructive_policy,
)
from ..action_policy import (
    DEFAULT_MAX_GUESS_CANDIDATES,
    ComposedBeliefInformedRandomMarshalAgent,
)


BIR2U_ALGORITHM_ID = "bir-2u-constructive-unweighted-v1"
BIR2U_BELIEF_SEED_DOMAIN = BIR2_BELIEF_SEED_DOMAIN
BIR2U_PROPOSAL_KERNEL_ID = BIR2_PROPOSAL_KERNEL_ID
BIR2U_REFERENCE_TARGET_ID = BIR2_REFERENCE_TARGET_ID
BIR2U_WEIGHTING_ID = UNWEIGHTED_ACCEPTED_PROPOSALS_ID


class UnweightedConstructiveBeliefConstructionError(RuntimeError):
    """Raised when BIR-2U cannot construct a complete, auditable batch."""

    def __init__(self, report: ConstructiveSamplingReport) -> None:
        self.report = report
        super().__init__(
            "BIR-2U requires a complete importance-valid constructive batch; "
            f"requested={report.requested}, produced={report.produced}, "
            f"degraded={report.degraded}, "
            f"importance_valid={report.importance_valid}, "
            f"termination_reason={report.termination_reason!r}, "
            f"exhausted_stage={report.exhausted_stage!r}, "
            f"proposals={report.proposals}, search_nodes={report.search_nodes}"
        )


class UnweightedConstructiveBeliefInformedRandomMarshalAgent(
    ComposedBeliefInformedRandomMarshalAgent
):
    """Shared BIR-2 policy using equal weights on accepted proposals."""

    name = "unweighted-constructive-belief-informed-random-marshal"

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
            weighting=ConstructiveWeightingStrategy.UNIFORM_ACCEPTED_PROPOSALS,
            backend_id="constructive-unweighted-sequential-v1",
            error_factory=UnweightedConstructiveBeliefConstructionError,
            sampler=_sampler,
        )
        super().__init__(policy)
        self.sprint_backend = CONSTRUCTIVE_SPRINT_BACKEND
        self.algorithm_id = BIR2U_ALGORITHM_ID
        self.belief_seed_domain = BIR2U_BELIEF_SEED_DOMAIN
        self.proposal_kernel_id = BIR2U_PROPOSAL_KERNEL_ID
        self.reference_target_id = BIR2U_REFERENCE_TARGET_ID
        self.weighting_id = BIR2U_WEIGHTING_ID

__all__ = [
    "BIR2U_ALGORITHM_ID",
    "BIR2U_BELIEF_SEED_DOMAIN",
    "BIR2U_PROPOSAL_KERNEL_ID",
    "BIR2U_REFERENCE_TARGET_ID",
    "BIR2U_WEIGHTING_ID",
    "UnweightedConstructiveBeliefConstructionError",
    "UnweightedConstructiveBeliefInformedRandomMarshalAgent",
]
