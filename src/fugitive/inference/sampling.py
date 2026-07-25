"""Deterministic sampling budgets and small shared sampling utilities."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Callable, Sequence, TypeVar


class SamplingBudgetExceeded(RuntimeError):
    """Internal control flow raised when a deterministic budget is exhausted."""

    def __init__(self, stage: str) -> None:
        super().__init__(f"sampling budget exhausted during {stage}")
        self.stage = stage


@dataclass(frozen=True, slots=True)
class SamplingBudget:
    """Deterministic safety boundary for one constructive batch.

    ``max_nodes`` counts explored DP transitions; ``None`` records work
    without truncating it. Unlike a wall-clock limit, a finite value preserves
    exact same-seed replay across machines. ``max_proposals`` bounds route
    proposals even when cached DP tables consume no additional nodes.
    """

    max_nodes: int | None = None
    max_proposals: int | None = None

    def __post_init__(self) -> None:
        if self.max_nodes is not None and (
            isinstance(self.max_nodes, bool)
            or not isinstance(self.max_nodes, int)
            or self.max_nodes <= 0
        ):
            raise ValueError("max_nodes must be a positive integer or None")
        if self.max_proposals is not None and (
            isinstance(self.max_proposals, bool)
            or not isinstance(self.max_proposals, int)
            or self.max_proposals <= 0
        ):
            raise ValueError("max_proposals must be a positive integer or None")


@dataclass(slots=True)
class SamplingCounter:
    """Count deterministic inference work against a :class:`SamplingBudget`.

    The counter is an explicit dependency of the route, draw, and Sprint
    stages.  Keeping it public makes those stages independently executable in
    small teaching examples without reaching into sampler internals.
    """

    specification: SamplingBudget
    nodes: int = 0
    last_stage: str | None = None

    def consume(self, stage: str, amount: int = 1) -> None:
        if amount < 0:
            raise ValueError("budget consumption cannot be negative")
        if (
            self.specification.max_nodes is not None
            and self.nodes + amount > self.specification.max_nodes
        ):
            self.last_stage = stage
            raise SamplingBudgetExceeded(stage)
        self.nodes += amount
        self.last_stage = stage


ChoiceT = TypeVar("ChoiceT")


def weighted_choice(
    values: Sequence[ChoiceT],
    weight: Callable[[ChoiceT], int],
    rng: random.Random,
) -> ChoiceT | None:
    """Choose from integer-weighted values without floating-point drift."""

    total = sum(weight(value) for value in values)
    if total <= 0:
        return None
    target = rng.randrange(total)
    cumulative = 0
    for value in values:
        cumulative += weight(value)
        if target < cumulative:
            return value
    return values[-1]  # pragma: no cover - integer arithmetic is exact


__all__ = ["SamplingBudget", "SamplingBudgetExceeded", "SamplingCounter"]
