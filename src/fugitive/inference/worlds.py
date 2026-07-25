"""Complete hidden-world data, target density, and sample-batch reports."""

from __future__ import annotations

from dataclasses import dataclass
import math

from fugitive.particle_inference.state import MarshalParticle
from fugitive.world_validation import (
    compile_route_creation_rounds,
    is_complete_world_consistent,
)

from .constraints import CompiledMarshalConstraints


WorldKey = tuple[
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[tuple[int, ...], ...],
    tuple[int, ...],
    tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]],
]


@dataclass(frozen=True, slots=True)
class ConstructiveWorld:
    fugitive_hand: tuple[int, ...]
    marshal_hand: tuple[int, ...]
    route_hideouts: tuple[int, ...]
    route_sprints: tuple[tuple[int, ...], ...]
    fugitive_draws: tuple[int, ...]
    remaining_piles: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]
    hidden_mask: int
    log_q: float
    log_target: float

    @property
    def world_key(self) -> WorldKey:
        """Canonical physical-world identity, deliberately excluding weights."""

        return (
            self.fugitive_hand,
            self.marshal_hand,
            self.route_hideouts,
            self.route_sprints,
            self.fugitive_draws,
            self.remaining_piles,
        )

    @property
    def importance_log_weight(self) -> float:
        return self.log_target - self.log_q

    @property
    def all_cards_are_unique(self) -> bool:
        return self.to_particle().all_cards_are_unique

    def to_particle(self, *, weight: float = 1.0) -> MarshalParticle:
        return MarshalParticle(
            fugitive_hand=self.fugitive_hand,
            marshal_hand=self.marshal_hand,
            route_hideouts=self.route_hideouts,
            route_sprints=self.route_sprints,
            fugitive_draws=self.fugitive_draws,
            remaining_piles=self.remaining_piles,
            weight=weight,
        )


class ConstraintUniformTarget:
    """Uniform unnormalised density over observation-consistent worlds."""

    target_id = "constraint-uniform-complete-worlds-v1"
    name = target_id

    def log_density(
        self,
        world: ConstructiveWorld,
        constraints: CompiledMarshalConstraints,
    ) -> float:
        if constraints.creation_rounds != compile_route_creation_rounds(
            constraints.observation
        ):
            return -math.inf
        if not is_complete_world_consistent(world, constraints.observation):
            return -math.inf

        hidden_mask = 0
        for index, hideout in enumerate(world.route_hideouts):
            if (
                index
                and not constraints.observation.route[index].revealed
                and hideout != 42
            ):
                hidden_mask |= 1 << hideout
        if world.hidden_mask != hidden_mask:
            return -math.inf
        return 0.0


@dataclass(frozen=True, slots=True)
class ConstructiveSamplingReport:
    requested: int
    produced: int
    proposals: int
    dead_end_route_proposals: int
    rejected_targets: int
    search_nodes: int
    degraded: bool
    importance_valid: bool
    termination_reason: str | None
    exhausted_stage: str | None
    unique_worlds: int

@dataclass(frozen=True, slots=True)
class ConstructiveSampleBatch:
    worlds: tuple[ConstructiveWorld, ...]
    report: ConstructiveSamplingReport
    proposal_kernel_id: str
    target_id: str
    observation_hash: str

    def __post_init__(self) -> None:
        for field_name, value in (
            ("proposal_kernel_id", self.proposal_kernel_id),
            ("target_id", self.target_id),
            ("observation_hash", self.observation_hash),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{field_name} must be a non-empty string")

    @property
    def normalized_weights(self) -> tuple[float, ...]:
        if not self.report.importance_valid:
            raise RuntimeError("importance weights are invalid after a node-budget stop")
        if not self.worlds:
            return ()
        logs = tuple(world.importance_log_weight for world in self.worlds)
        if any(not math.isfinite(value) for value in logs):
            raise RuntimeError("importance weights must all be finite")
        maximum = max(logs)
        raw = tuple(math.exp(value - maximum) for value in logs)
        total = sum(raw)
        return tuple(value / total for value in raw)


__all__ = [
    "ConstraintUniformTarget",
    "ConstructiveSampleBatch",
    "ConstructiveSamplingReport",
    "ConstructiveWorld",
    "WorldKey",
]
