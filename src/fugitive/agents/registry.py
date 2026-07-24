"""Typed registry for the active information-safe agent families.

Concrete policies remain independent of this module.  The local Web app uses
the registry as the single place that assigns stable public names.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import inspect
from typing import Callable, Generic, Mapping, TypeVar, cast

from fugitive.model import FugitiveAgent, MarshalAgent, Role
from fugitive.reproducibility import (
    AGENT_PROFILES,
    AgentSpec,
    normalize_parameters,
)

from .belief_informed_random import (
    BeliefInformedRandomFugitiveAgent,
    BeliefInformedRandomMarshalAgent,
)
from .constructive_bir import ConstructiveBeliefInformedRandomMarshalAgent
from .hierarchical_random import (
    HierarchicalRandomFugitiveAgent,
    HierarchicalRandomMarshalAgent,
)
from .mcmc_bir import MCMCBeliefInformedRandomMarshalAgent
from .route_count_random import RouteCountRandomMarshalAgent


AgentT = TypeVar("AgentT")
FugitiveAgentFactory = Callable[..., FugitiveAgent]
MarshalAgentFactory = Callable[..., MarshalAgent]
DEFAULT_FUGITIVE_AGENT = "belief-informed-random"
DEFAULT_MARSHAL_AGENT = "belief-informed-random"
INTERACTIVE_BIR_PARTICLE_COUNT = 384
INTERACTIVE_BIR_MAX_GUESS_CANDIDATES = 64
INTERACTIVE_BIR2_PARTICLE_COUNT = 128
INTERACTIVE_BIR2_MAX_GUESS_CANDIDATES = 64
INTERACTIVE_MCMC_BIR_PARTICLE_COUNT = 256
INTERACTIVE_MCMC_BIR_MAX_GUESS_CANDIDATES = 128
INTERACTIVE_MCMC_BIR_STEPS_PER_CHAIN = 1


@dataclass(frozen=True, slots=True)
class BuiltAgent(Generic[AgentT]):
    """A fresh policy together with its complete resolved configuration."""

    agent: AgentT
    spec: AgentSpec


@dataclass(frozen=True, slots=True)
class AgentRegistration(Generic[AgentT]):
    """A stable policy name plus runner-relevant construction metadata."""

    name: str
    role: Role
    factory: Callable[..., AgentT]
    expensive: bool = False
    profile_overrides: Mapping[str, Mapping[str, object]] = field(
        default_factory=dict
    )

    def build(
        self,
        seed: int,
        *,
        profile: str = "default",
        overrides: Mapping[str, object] | None = None,
    ) -> BuiltAgent[AgentT]:
        """Construct an agent and record every effective policy parameter."""

        if profile not in AGENT_PROFILES:
            raise ValueError(
                f"agent profile must be one of {', '.join(AGENT_PROFILES)}"
            )
        supplied = normalize_parameters(overrides or {})
        if "seed" in supplied or "rng" in supplied:
            raise ValueError("agent parameters must not override seed or rng")

        signature = inspect.signature(self.factory)
        # Underscored constructor arguments are internal dependency-injection
        # hooks.  They are never user-overridable policy parameters.
        constructor_parameters = {
            name: parameter
            for name, parameter in signature.parameters.items()
            if name not in ("seed", "rng")
            and not name.startswith("_")
            and parameter.kind
            in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        }
        unknown = set(supplied) - set(constructor_parameters)
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"unknown parameters for agent {self.name}: {names}")

        effective: dict[str, object] = {}
        for name, parameter in constructor_parameters.items():
            if parameter.default is inspect.Parameter.empty:
                if name not in supplied:
                    raise ValueError(
                        f"missing required parameter for agent {self.name}: {name}"
                    )
                continue
            effective[name] = parameter.default

        profile_values = self.profile_overrides.get(profile, {})
        unknown_profile = set(profile_values) - set(constructor_parameters)
        if unknown_profile:  # pragma: no cover - registry definition guard
            names = ", ".join(sorted(unknown_profile))
            raise RuntimeError(
                f"invalid {profile} profile for agent {self.name}: {names}"
            )
        effective.update(profile_values)
        effective.update(supplied)

        agent = self.factory(seed, **effective)
        # Constructors may resolve sentinel defaults such as
        # min_unique_particles=None. Prefer the resulting public value so the
        # spec can reconstruct the exact policy rather than just its request.
        resolved = {
            name: getattr(agent, name, effective[name])
            for name in constructor_parameters
            if name in effective
        }
        spec = AgentSpec(
            self.name,
            self.role,
            profile,
            normalize_parameters(resolved),
        )
        return BuiltAgent(agent, spec)

    def create(self, seed: int, *, interactive: bool = False) -> AgentT:
        """Create a fresh policy, optionally using its responsive UI profile."""

        profile = "interactive" if interactive else "default"
        return self.build(seed, profile=profile).agent


FUGITIVE_AGENT_REGISTRY: dict[str, AgentRegistration[FugitiveAgent]] = {
    registration.name: registration
    for registration in (
        AgentRegistration(
            "hierarchical-random",
            Role.FUGITIVE,
            cast(FugitiveAgentFactory, HierarchicalRandomFugitiveAgent),
        ),
        AgentRegistration(
            "belief-informed-random",
            Role.FUGITIVE,
            cast(FugitiveAgentFactory, BeliefInformedRandomFugitiveAgent),
        ),
    )
}

MARSHAL_AGENT_REGISTRY: dict[str, AgentRegistration[MarshalAgent]] = {
    registration.name: registration
    for registration in (
        AgentRegistration(
            "hierarchical-random",
            Role.MARSHAL,
            cast(MarshalAgentFactory, HierarchicalRandomMarshalAgent),
        ),
        AgentRegistration(
            "belief-informed-random",
            Role.MARSHAL,
            cast(MarshalAgentFactory, BeliefInformedRandomMarshalAgent),
            expensive=True,
            profile_overrides={
                "interactive": {
                    "particle_count": INTERACTIVE_BIR_PARTICLE_COUNT,
                    "max_guess_candidates": INTERACTIVE_BIR_MAX_GUESS_CANDIDATES,
                }
            },
        ),
        AgentRegistration(
            "route-count-random",
            Role.MARSHAL,
            cast(MarshalAgentFactory, RouteCountRandomMarshalAgent),
        ),
        AgentRegistration(
            "constructive-belief-informed-random",
            Role.MARSHAL,
            cast(
                MarshalAgentFactory,
                ConstructiveBeliefInformedRandomMarshalAgent,
            ),
            expensive=True,
            profile_overrides={
                "interactive": {
                    "particle_count": INTERACTIVE_BIR2_PARTICLE_COUNT,
                    "max_guess_candidates": (
                        INTERACTIVE_BIR2_MAX_GUESS_CANDIDATES
                    ),
                }
            },
        ),
        AgentRegistration(
            "mcmc-belief-informed-random",
            Role.MARSHAL,
            cast(
                MarshalAgentFactory,
                MCMCBeliefInformedRandomMarshalAgent,
            ),
            expensive=True,
            profile_overrides={
                "interactive": {
                    "particle_count": INTERACTIVE_MCMC_BIR_PARTICLE_COUNT,
                    "max_guess_candidates": (
                        INTERACTIVE_MCMC_BIR_MAX_GUESS_CANDIDATES
                    ),
                    "mh_steps_per_chain": (
                        INTERACTIVE_MCMC_BIR_STEPS_PER_CHAIN
                    ),
                }
            },
        ),
    )
}

__all__ = [
    "AgentRegistration",
    "BuiltAgent",
    "DEFAULT_FUGITIVE_AGENT",
    "DEFAULT_MARSHAL_AGENT",
    "FUGITIVE_AGENT_REGISTRY",
    "INTERACTIVE_BIR2_MAX_GUESS_CANDIDATES",
    "INTERACTIVE_BIR2_PARTICLE_COUNT",
    "INTERACTIVE_BIR_MAX_GUESS_CANDIDATES",
    "INTERACTIVE_BIR_PARTICLE_COUNT",
    "INTERACTIVE_MCMC_BIR_MAX_GUESS_CANDIDATES",
    "INTERACTIVE_MCMC_BIR_PARTICLE_COUNT",
    "INTERACTIVE_MCMC_BIR_STEPS_PER_CHAIN",
    "MARSHAL_AGENT_REGISTRY",
]
