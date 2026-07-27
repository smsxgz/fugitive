"""Project online-search policies to non-search rollout leaves.

An agent may be used in two different contexts:

* as the real policy, where its declared online search is allowed; or
* inside another policy's hypothetical full-game rollout, where starting a
  second search would make planning recursive.

The small explicit table below defines the latter interpretation.  Ordinary
agents project to themselves.  Search agents project to the policy that
already supplies their continuation play.  Registry defaults are resolved at
projection time, and shared user parameters are copied by name.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..game.model import Role
from ..shared.reproducibility import AgentSpec, thaw_parameters


PLANNING_LEAF_PROJECTION_ID = "non-recursive-planning-leaf-projection-v1"


@dataclass(frozen=True, slots=True)
class PlanningLeafRule:
    role: Role
    search_agent: str
    leaf_agent: str
    opponent_parameter: str


PLANNING_LEAF_RULES = (
    PlanningLeafRule(
        Role.FUGITIVE,
        "belief-rollout",
        "continuation-count",
        "opponent_policies",
    ),
    PlanningLeafRule(
        Role.MARSHAL,
        "rollout-bir2u",
        "unweighted-constructive-belief-informed-random",
        "opponent_policies",
    ),
)


_RULE_BY_POLICY = {
    (rule.role, rule.search_agent): rule for rule in PLANNING_LEAF_RULES
}


def is_online_search_spec(spec: AgentSpec) -> bool:
    """Return whether ``spec`` has a distinct non-search planning leaf."""

    if not isinstance(spec, AgentSpec):
        raise ValueError("online-search classification requires an AgentSpec")
    return (spec.role, spec.name) in _RULE_BY_POLICY


def search_opponent_parameter(role: Role, agent_name: str) -> str | None:
    """Return the mixture parameter declared by one online-search policy."""

    if not isinstance(role, Role):
        raise ValueError("search classification requires a player role")
    if not isinstance(agent_name, str) or not agent_name:
        raise ValueError("search classification requires an agent name")
    rule = _RULE_BY_POLICY.get((role, agent_name))
    return None if rule is None else rule.opponent_parameter


def planning_leaf_spec(spec: AgentSpec) -> AgentSpec:
    """Return the complete non-search spec used inside another planner.

    Parameter copying is deliberately structural: only names documented by
    the target leaf's registry schema are retained.  For example, a Marshal
    rollout keeps the BIR-2U particle and guess-candidate settings but drops
    ``max_terminal_simulations``, ``rollout_candidate_count``,
    ``opponent_policies``, and ``rollout_temperature``.
    """

    if not isinstance(spec, AgentSpec):
        raise ValueError("planning-leaf projection requires an AgentSpec")
    rule = _RULE_BY_POLICY.get((spec.role, spec.name))
    if rule is None:
        return spec

    # Lazy import avoids registry -> rollout agent -> planning leaf -> registry
    # during module initialization.
    from ..agents.registry import FUGITIVE_AGENT_REGISTRY, MARSHAL_AGENT_REGISTRY

    registry = (
        FUGITIVE_AGENT_REGISTRY
        if spec.role is Role.FUGITIVE
        else MARSHAL_AGENT_REGISTRY
    )
    try:
        registration = registry[rule.leaf_agent]
    except KeyError as exc:
        raise ValueError(
            f"planning leaf {rule.leaf_agent!r} is not registered for "
            f"{spec.name!r}"
        ) from exc

    source_parameters = thaw_parameters(spec.parameters)
    overrides = {
        name: source_parameters[name]
        for name in registration.user_parameter_defaults
        if name in source_parameters
    }
    return registration.build(
        0,
        profile=spec.profile,
        overrides=overrides,
    ).spec


__all__ = [
    "PLANNING_LEAF_PROJECTION_ID",
    "PLANNING_LEAF_RULES",
    "PlanningLeafRule",
    "is_online_search_spec",
    "planning_leaf_spec",
    "search_opponent_parameter",
]
