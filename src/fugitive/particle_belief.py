"""Stable public surface for Marshal particle beliefs.

Implementation details are separated by teaching concern:

* :mod:`fugitive.particle_inference.state` defines worlds and queries;
* :mod:`fugitive.particle_inference.bootstrap_filter` advances a belief;
* :mod:`fugitive.particle_inference.constructive_fresh` turns constructive
  proposal batches into fresh particle beliefs.
"""

from fugitive.particle_inference.bootstrap_filter import (
    BIR1_INCREMENTAL_EXACT_COMBINATION_THRESHOLD,
)
from fugitive.particle_inference.state import (
    BIR1_INCREMENTAL_PLAY_PROPOSAL_LIMIT,
    BIR1_RESAMPLE_ESS_FRACTION,
    DEFAULT_PARTICLE_COUNT,
    IncompatibleObservationError,
    MarshalDrawOutcomeStatistics,
    MarshalParticle,
    MarshalParticleBelief,
    ParticleBeliefSummary,
    particle_matches_public_projection,
)


__all__ = [
    "BIR1_INCREMENTAL_EXACT_COMBINATION_THRESHOLD",
    "BIR1_INCREMENTAL_PLAY_PROPOSAL_LIMIT",
    "BIR1_RESAMPLE_ESS_FRACTION",
    "DEFAULT_PARTICLE_COUNT",
    "IncompatibleObservationError",
    "MarshalDrawOutcomeStatistics",
    "MarshalParticle",
    "MarshalParticleBelief",
    "ParticleBeliefSummary",
    "particle_matches_public_projection",
]
