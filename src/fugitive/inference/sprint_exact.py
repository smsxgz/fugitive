"""Exact descendant-count dynamic program for Sprint assignments."""

from __future__ import annotations

import math
import random
from typing import Sequence

from .constraints import CompiledMarshalConstraints, DeadlineDemands
from .draw_matching import DrawDeadlineMatcher
from .sampling import SamplingCounter, weighted_choice
from .sprint_constraints import (
    CategoryAllocation,
    SprintConstraintModel,
    SprintDrawSample,
)


class SprintAssignmentDP:
    """Exact category-count DP for Sprint identities and draw assignments.

    Cards are exchangeable inside eight ``source x Sprint-value`` groups. The
    exact DP weights a category allocation by all feasible descendant Sprint
    allocations and draw completions before sampling concrete identities.
    """

    def __init__(
        self,
        constraints: CompiledMarshalConstraints,
        route: Sequence[int],
        matcher: DrawDeadlineMatcher,
        budget: SamplingCounter,
    ) -> None:
        self.model = SprintConstraintModel(constraints, route, matcher, budget)
        self.constraints = constraints
        self.route = self.model.route
        self.matcher = matcher
        self.budget = budget
        self.valid = self.model.valid
        self._memo: dict[
            tuple[int, tuple[int, ...], DeadlineDemands], int
        ] = {}

    @property
    def completion_count(self) -> int:
        if not self.valid:
            return 0
        return self._count(
            0, self.model.initial_group_counts, self.model.fixed_demands
        )

    def _count(
        self,
        position: int,
        remaining: tuple[int, ...],
        demands: DeadlineDemands,
    ) -> int:
        if position == len(self.model.unknown):
            return self.matcher.count_deadline_demands(demands)
        key = position, remaining, demands
        cached = self._memo.get(key)
        if cached is not None:
            return cached
        total = 0
        for allocation in self.model.category_allocations(position, remaining):
            next_remaining, next_demands = self.model.advance_category_state(
                position, remaining, demands, allocation
            )
            if not self.matcher.count_deadline_demands(next_demands):
                continue
            multiplicity = self.model.branch_multiplicity(remaining, allocation)
            total += multiplicity * self._count(
                position + 1, next_remaining, next_demands
            )
        self._memo[key] = total
        return total

    def sample(self, rng: random.Random) -> SprintDrawSample | None:
        total = self.completion_count
        if not total:
            return None
        stacks = list(self.model.base_stacks)
        group_cards = [list(cards) for cards in self.model.initial_group_cards]
        remaining = self.model.initial_group_counts
        demands = self.model.fixed_demands
        exact_deadlines = dict(self.model.fixed_deadlines)
        for position, index in enumerate(self.model.unknown):
            branches: list[
                tuple[CategoryAllocation, tuple[int, ...], DeadlineDemands, int]
            ] = []
            for allocation in self.model.category_allocations(position, remaining):
                next_remaining, next_demands = self.model.advance_category_state(
                    position, remaining, demands, allocation
                )
                child = self._count(position + 1, next_remaining, next_demands)
                branch_ways = (
                    self.model.branch_multiplicity(remaining, allocation) * child
                )
                if branch_ways:
                    branches.append(
                        (allocation, next_remaining, next_demands, branch_ways)
                    )
            selected = weighted_choice(branches, lambda item: item[3], rng)
            if selected is None:  # pragma: no cover - guarded by total
                return None
            allocation, remaining, demands, _ways = selected
            deadline = self.constraints.creation_rounds[index]
            stack, _identity_count = self.model.sample_identities(
                allocation,
                group_cards,
                rng,
                deadline,
                exact_deadlines,
            )
            stacks[index] = stack
        draw = self.matcher.sample(exact_deadlines, rng)
        if draw is None:  # pragma: no cover - branch counts guarantee feasibility
            return None
        return SprintDrawSample(
            route_sprints=tuple(stack or () for stack in stacks),
            draw_assignment=draw,
            completion_count=total,
            log_q=-math.log(total),
        )


__all__ = ["SprintAssignmentDP"]
