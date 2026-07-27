"""Controlled HR support baseline using the shared Marshal catalogue."""

from __future__ import annotations

import random

from .catalogue_random_common import (
    DEFAULT_CATALOGUE_MULTI_GUESS_CONTINUATION,
    CatalogueRandomMarshalPolicy,
)
from .marshal_candidates import (
    DEFAULT_CATALOGUE_MAX_GUESS_SIZE,
    DEFAULT_CATALOGUE_MAX_JOINT_CANDIDATES,
)


CONTROLLED_SUPPORT_ALGORITHM_ID = "hr-1c-marshal-shared-catalogue-support-v1"


class SupportCatalogueRandomMarshalAgent(CatalogueRandomMarshalPolicy):
    """Uniformly weight supported guesses in a shared action catalogue."""

    name = "support-catalogue-random-marshal"

    def __init__(
        self,
        seed: int | random.Random | None = None,
        *,
        rng: random.Random | None = None,
        multi_guess_continuation: float = (
            DEFAULT_CATALOGUE_MULTI_GUESS_CONTINUATION
        ),
        max_guess_size: int = DEFAULT_CATALOGUE_MAX_GUESS_SIZE,
        max_joint_candidates: int = DEFAULT_CATALOGUE_MAX_JOINT_CANDIDATES,
    ) -> None:
        super().__init__(
            seed,
            rng=rng,
            use_route_counts=False,
            epsilon=0.0,
            multi_guess_continuation=multi_guess_continuation,
            max_guess_size=max_guess_size,
            max_joint_candidates=max_joint_candidates,
        )
        self.algorithm_id = CONTROLLED_SUPPORT_ALGORITHM_ID
        self.weighting_id = "boolean-support-uniform-v1"


__all__ = [
    "CONTROLLED_SUPPORT_ALGORITHM_ID",
    "SupportCatalogueRandomMarshalAgent",
]
