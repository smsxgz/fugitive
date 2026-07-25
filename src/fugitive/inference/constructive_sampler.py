"""Orchestrate route, Sprint, draw, and target stages into world samples."""

from __future__ import annotations

from dataclasses import replace
import math
import random
from typing import Callable, Sequence

from fugitive.belief import RouteSampler
from fugitive.model import Observation
from fugitive.world_validation import INITIAL_FUGITIVE_CARDS

from .constraints import CompiledMarshalConstraints
from .constructive_metadata import (
    DEFAULT_ROUTE_PROPOSAL_ID,
    constructive_observation_hash,
    constructive_proposal_kernel_id,
)
from .draw_matching import DrawDeadlineMatcher
from .sampling import (
    SamplingBudget,
    SamplingBudgetExceeded,
    SamplingCounter,
)
from .sprint_exact import SprintAssignmentDP
from .sprint_sequential import SequentialSprintAssignmentProposal
from .worlds import (
    ConstraintUniformTarget,
    ConstructiveSampleBatch,
    ConstructiveSamplingReport,
    ConstructiveWorld,
)


RouteSamplerFactory = Callable[
    [CompiledMarshalConstraints, SamplingCounter], RouteSampler
]
SprintAssignmentSampler = SprintAssignmentDP | SequentialSprintAssignmentProposal


def _default_route_sampler_factory(
    constraints: CompiledMarshalConstraints,
    budget: SamplingCounter,
) -> RouteSampler:
    return constraints.path_belief.compile_route_catalogue(budget)


SPRINT_ASSIGNMENT_BACKENDS = ("exact", "sequential")


class ConstructiveWorldSampler:
    """Build weighted complete-world proposals without engine-private state.

    The ``exact`` and ``sequential`` Sprint backends are peer implementations
    composed with the same constraint model. Exact recursively counts all
    descendants; sequential records its cheaper local proposal probability.
    """

    def __init__(
        self,
        *,
        target: ConstraintUniformTarget | None = None,
        route_sampler_factory: RouteSamplerFactory = _default_route_sampler_factory,
        sprint_backend: str = "exact",
        route_proposal_id: str | None = None,
    ) -> None:
        if sprint_backend not in SPRINT_ASSIGNMENT_BACKENDS:
            choices = ", ".join(SPRINT_ASSIGNMENT_BACKENDS)
            raise ValueError(f"sprint_backend must be one of {choices}")
        self.target = target or ConstraintUniformTarget()
        target_id = getattr(self.target, "target_id", None)
        if not isinstance(target_id, str) or not target_id:
            raise ValueError("constructive targets must define a stable target_id")
        if (
            type(self.target) is not ConstraintUniformTarget
            and "target_id" not in type(self.target).__dict__
        ):
            raise ValueError("custom constructive targets must override target_id")
        if route_proposal_id is None:
            if route_sampler_factory is not _default_route_sampler_factory:
                raise ValueError("custom route samplers must define route_proposal_id")
            route_proposal_id = DEFAULT_ROUTE_PROPOSAL_ID
        if not isinstance(route_proposal_id, str) or not route_proposal_id:
            raise ValueError("route_proposal_id must be a non-empty string")
        self.route_sampler_factory = route_sampler_factory
        self.sprint_backend = sprint_backend
        self.route_proposal_id = route_proposal_id
        self.target_id = target_id
        self.proposal_kernel_id = constructive_proposal_kernel_id(
            sprint_backend,
            route_proposal_id=route_proposal_id,
        )

    def _assignment_sampler(
        self,
        constraints: CompiledMarshalConstraints,
        route: Sequence[int],
        matcher: DrawDeadlineMatcher,
        counter: SamplingCounter,
    ) -> SprintAssignmentSampler:
        if self.sprint_backend == "exact":
            return SprintAssignmentDP(constraints, route, matcher, counter)
        return SequentialSprintAssignmentProposal(
            constraints, route, matcher, counter
        )

    def sample_batch(
        self,
        observation: Observation,
        *,
        particle_count: int,
        seed: int | None = None,
        rng: random.Random | None = None,
        budget: SamplingBudget | None = None,
    ) -> ConstructiveSampleBatch:
        if (
            isinstance(particle_count, bool)
            or not isinstance(particle_count, int)
            or particle_count <= 0
        ):
            raise ValueError("particle_count must be a positive integer")
        if seed is not None and rng is not None:
            raise ValueError("pass either seed or rng, not both")
        owned_rng = rng if rng is not None else random.Random(seed)
        specification = budget or SamplingBudget()
        counter = SamplingCounter(specification)
        proposals = 0
        dead_end_route_proposals = 0
        rejected_targets = 0
        worlds: list[ConstructiveWorld] = []
        exhausted_stage: str | None = None
        termination_reason: str | None = None
        max_proposals = specification.max_proposals
        try:
            constraints = CompiledMarshalConstraints.from_observation(observation)
            route_sampler = self.route_sampler_factory(constraints, counter)
            matcher = DrawDeadlineMatcher.from_constraints(constraints, counter)
            completion_cache: dict[tuple[int, ...], SprintAssignmentSampler] = {}
            while len(worlds) < particle_count and (
                max_proposals is None or proposals < max_proposals
            ):
                proposals += 1
                route = route_sampler.sample(owned_rng)
                if route is None:
                    termination_reason = "no_route_support"
                    break
                assignment_sampler = completion_cache.get(route.route)
                if assignment_sampler is None:
                    assignment_sampler = self._assignment_sampler(
                        constraints, route.route, matcher, counter
                    )
                    completion_cache[route.route] = assignment_sampler
                assignment = assignment_sampler.sample(owned_rng)
                if assignment is None:
                    dead_end_route_proposals += 1
                    continue
                used = set(route.route[1:])
                used.update(card for stack in assignment.route_sprints for card in stack)
                owned = set(INITIAL_FUGITIVE_CARDS) | set(
                    assignment.draw_assignment.fugitive_draws
                )
                fugitive_hand = tuple(sorted(owned - used))
                hidden_mask = 0
                for index, hideout in enumerate(route.route):
                    if (
                        index
                        and not observation.route[index].revealed
                        and hideout != 42
                    ):
                        hidden_mask |= 1 << hideout
                provisional = ConstructiveWorld(
                    fugitive_hand=fugitive_hand,
                    marshal_hand=constraints.marshal_hand,
                    route_hideouts=route.route,
                    route_sprints=assignment.route_sprints,
                    fugitive_draws=assignment.draw_assignment.fugitive_draws,
                    remaining_piles=assignment.draw_assignment.remaining_piles,
                    hidden_mask=hidden_mask,
                    log_q=route.log_q + assignment.log_q,
                    log_target=0.0,
                )
                log_target = self.target.log_density(provisional, constraints)
                if not math.isfinite(log_target):
                    rejected_targets += 1
                    continue
                worlds.append(replace(provisional, log_target=log_target))
        except SamplingBudgetExceeded as error:
            exhausted_stage = error.stage
            termination_reason = "node_budget"

        degraded = len(worlds) < particle_count
        if (
            degraded
            and termination_reason is None
            and max_proposals is not None
            and proposals >= max_proposals
        ):
            termination_reason = "max_proposals"
        return ConstructiveSampleBatch(
            worlds=tuple(worlds),
            report=ConstructiveSamplingReport(
                requested=particle_count,
                produced=len(worlds),
                proposals=proposals,
                dead_end_route_proposals=dead_end_route_proposals,
                rejected_targets=rejected_targets,
                search_nodes=counter.nodes,
                degraded=degraded,
                importance_valid=termination_reason != "node_budget",
                termination_reason=termination_reason,
                exhausted_stage=exhausted_stage,
                unique_worlds=len({world.world_key for world in worlds}),
            ),
            proposal_kernel_id=self.proposal_kernel_id,
            target_id=self.target_id,
            observation_hash=constructive_observation_hash(observation),
        )


__all__ = ["ConstructiveWorldSampler", "SPRINT_ASSIGNMENT_BACKENDS"]
