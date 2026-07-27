"""Stable public surface for Marshal particle beliefs.

Implementation details are separated by teaching concern:

* :mod:`fugitive.agents.marshal.inference.particles.state` defines worlds and queries;
* :mod:`fugitive.agents.marshal.inference.particles.bootstrap_filter` advances a belief;
* :mod:`fugitive.agents.marshal.inference.particles.constructive_fresh` turns constructive
  proposal batches into fresh particle beliefs.
"""

from .particles.bootstrap_filter import (
    BIR1_INCREMENTAL_EXACT_COMBINATION_THRESHOLD,
)
from .particles.state import (
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
