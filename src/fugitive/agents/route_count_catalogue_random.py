"""Controlled route-count baseline using the shared Marshal catalogue."""

from __future__ import annotations

import random

from .catalogue_random_common import CatalogueRandomMarshalPolicy


CONTROLLED_ROUTE_COUNT_ALGORITHM_ID = (
    "hr-1.1c-marshal-shared-catalogue-route-count-v1"
)


class RouteCountCatalogueRandomMarshalAgent(CatalogueRandomMarshalPolicy):
    """Weight the same catalogue by compatible-route completion counts."""

    name = "route-count-catalogue-random-marshal"

    def __init__(
        self,
        seed: int | random.Random | None = None,
        **kwargs,
    ) -> None:
        super().__init__(seed, use_route_counts=True, **kwargs)
        self.algorithm_id = CONTROLLED_ROUTE_COUNT_ALGORITHM_ID
        self.weighting_id = "compatible-route-count-v1"


__all__ = [
    "CONTROLLED_ROUTE_COUNT_ALGORITHM_ID",
    "RouteCountCatalogueRandomMarshalAgent",
]
