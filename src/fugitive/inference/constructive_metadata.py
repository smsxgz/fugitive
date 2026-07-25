"""Stable identities for constructive proposal kernels and observations."""

from __future__ import annotations

import hashlib

from fugitive.model import Observation
from fugitive.observation_protocol import canonical_observation_bytes


CONSTRUCTIVE_PROPOSAL_SCHEMA = "fugitive.constructive-world-proposal.v1"
DEFAULT_ROUTE_PROPOSAL_ID = "path-belief-completion-count.v1"
DRAW_PROPOSAL_ID = "draw-deadline-matcher.v1"
SPRINT_PROPOSAL_IDS = {
    "exact": "sprint-category-descendant-count.v1",
    "sequential": "sprint-category-sequential-importance.v1",
}


def constructive_proposal_kernel_id(
    sprint_backend: str,
    *,
    route_proposal_id: str = DEFAULT_ROUTE_PROPOSAL_ID,
) -> str:
    """Return the stable identity of one constructive proposal kernel."""

    if sprint_backend not in SPRINT_PROPOSAL_IDS:
        choices = ", ".join(sorted(SPRINT_PROPOSAL_IDS))
        raise ValueError(f"sprint_backend must be one of {choices}")
    if not isinstance(route_proposal_id, str) or not route_proposal_id:
        raise ValueError("route_proposal_id must be a non-empty string")
    return "|".join(
        (
            CONSTRUCTIVE_PROPOSAL_SCHEMA,
            f"route={route_proposal_id}",
            f"sprint={SPRINT_PROPOSAL_IDS[sprint_backend]}",
            f"draw={DRAW_PROPOSAL_ID}",
        )
    )


def constructive_observation_hash(observation: Observation) -> str:
    """Return a stable hash of the complete agent-visible observation."""

    digest = hashlib.sha256()
    digest.update(b"fugitive.constructive-observation\x00v1\x00")
    payload = canonical_observation_bytes(observation)
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)
    return digest.hexdigest()


__all__ = [
    "CONSTRUCTIVE_PROPOSAL_SCHEMA",
    "DEFAULT_ROUTE_PROPOSAL_ID",
    "DRAW_PROPOSAL_ID",
    "SPRINT_PROPOSAL_IDS",
    "constructive_observation_hash",
    "constructive_proposal_kernel_id",
]
