"""Fast sequential importance proposal for Sprint assignments."""

from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(frozen=True, slots=True)
class SequentialAllocationBranch:
    """One auditable branch of the fast Sprint proposal."""

    allocation: CategoryAllocation
    next_remaining: tuple[int, ...]
    next_demands: DeadlineDemands
    identity_multiplicity: int
    feasible_draw_completions: int

    @property
    def proposal_weight(self) -> int:
        return self.identity_multiplicity * self.feasible_draw_completions


class SequentialSprintAssignmentProposal:
    """Sequential importance proposal composed with shared Sprint constraints.

    At each hidden stack, legal category allocations are weighted only by
    current identity multiplicity and current draw feasibility. Unlike the
    exact reference DP, this proposal does not recursively count descendants.
    Its returned ``log_q`` records every sequential and identity choice.
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

    def allocation_branches(
        self,
        position: int,
        remaining: tuple[int, ...],
        demands: DeadlineDemands,
    ) -> tuple[SequentialAllocationBranch, ...]:
        branches: list[SequentialAllocationBranch] = []
        for allocation in self.model.category_allocations(position, remaining):
            next_remaining, next_demands = self.model.advance_category_state(
                position, remaining, demands, allocation
            )
            feasible_draws = self.matcher.count_deadline_demands(next_demands)
            if not feasible_draws:
                continue
            branches.append(
                SequentialAllocationBranch(
                    allocation=allocation,
                    next_remaining=next_remaining,
                    next_demands=next_demands,
                    identity_multiplicity=self.model.branch_multiplicity(
                        remaining, allocation
                    ),
                    feasible_draw_completions=feasible_draws,
                )
            )
        return tuple(branches)

    def sample(self, rng: random.Random) -> SprintDrawSample | None:
        if not self.valid:
            return None
        stacks = list(self.model.base_stacks)
        group_cards = [list(cards) for cards in self.model.initial_group_cards]
        remaining = self.model.initial_group_counts
        demands = self.model.fixed_demands
        exact_deadlines = dict(self.model.fixed_deadlines)
        log_q = 0.0

        for position, index in enumerate(self.model.unknown):
            branches = self.allocation_branches(position, remaining, demands)
            total_weight = sum(branch.proposal_weight for branch in branches)
            selected = weighted_choice(
                branches, lambda branch: branch.proposal_weight, rng
            )
            if selected is None or total_weight <= 0:
                return None
            log_q += math.log(selected.proposal_weight) - math.log(total_weight)
            remaining = selected.next_remaining
            demands = selected.next_demands

            deadline = self.constraints.creation_rounds[index]
            stack, identity_count = self.model.sample_identities(
                selected.allocation,
                group_cards,
                rng,
                deadline,
                exact_deadlines,
            )
            log_q -= math.log(identity_count)
            stacks[index] = stack

        draw = self.matcher.sample(exact_deadlines, rng)
        if draw is None:
            return None
        log_q += draw.log_q
        return SprintDrawSample(
            route_sprints=tuple(stack or () for stack in stacks),
            draw_assignment=draw,
            completion_count=None,
            log_q=log_q,
        )


__all__ = [
    "SequentialAllocationBranch",
    "SequentialSprintAssignmentProposal",
]
