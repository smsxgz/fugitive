"""Typed registry for the active information-safe agent families.

Concrete policies remain independent of this module.  The local Web app uses
the registry as the single place that assigns stable public names.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Callable, Generic, Mapping, TypeVar, cast

from fugitive.model import FugitiveAgent, MarshalAgent, Role
from fugitive.particle_belief import (
    BIR1_INCREMENTAL_EXACT_COMBINATION_THRESHOLD,
    BIR1_INCREMENTAL_PLAY_PROPOSAL_LIMIT,
    BIR1_RESAMPLE_ESS_FRACTION,
    DEFAULT_PARTICLE_COUNT,
)
from fugitive.reproducibility import (
    AGENT_PROFILES,
    AgentSpec,
    normalize_parameters,
    validate_seed,
)

from .bir_common import (
    DEFAULT_EPSILON as DEFAULT_BIR_EPSILON,
    DEFAULT_MANHUNT_ALPHA,
    DEFAULT_MANHUNT_EPSILON,
    DEFAULT_TEMPERATURE,
)
from .bir_fugitive import (
    BIR1_FUGITIVE_ALGORITHM_ID,
    BeliefInformedRandomFugitiveAgent,
    DEFAULT_MANHUNT_ROLLOUTS,
)
from .bootstrap_bir import (
    BIR1_BOOTSTRAP_BACKEND_ID,
    BIR1_BELIEF_SEED_DOMAIN,
    BIR1_INCREMENTAL_TRANSITION_ID,
    BIR1_MARSHAL_ALGORITHM_ID,
    BIR1_PROPOSAL_KERNEL_ID,
    BIR1_PUBLIC_EVENT_MODEL_ID,
    BIR1_REFERENCE_TARGET_ID,
    BIR1_WEIGHTING_ID,
    BeliefInformedRandomMarshalAgent,
)
from .marshal_belief_policy import (
    DEFAULT_MAX_GUESS_CANDIDATES,
)
from .constructive_bir import (
    BIR2_ALGORITHM_ID,
    BIR2_BELIEF_SEED_DOMAIN,
    BIR2_PROPOSAL_KERNEL_ID,
    BIR2_REFERENCE_TARGET_ID,
    BIR2_WEIGHTING_ID,
    CONSTRUCTIVE_SPRINT_BACKEND,
    ConstructiveBeliefInformedRandomMarshalAgent,
)
from .exact_sprint_bir import (
    DEFAULT_EXACT_SPRINT_PARTICLE_COUNT,
    EXACT_SPRINT_ALGORITHM_ID,
    EXACT_SPRINT_BACKEND,
    EXACT_SPRINT_BELIEF_SEED_DOMAIN,
    EXACT_SPRINT_PROPOSAL_KERNEL_ID,
    EXACT_SPRINT_REFERENCE_TARGET_ID,
    ExactSprintBeliefInformedRandomMarshalAgent,
)
from .hierarchical_random import (
    DEFAULT_MAX_EXTRA_OVERPAYMENTS,
    DEFAULT_MAX_GUESSES_PER_SIZE,
    DEFAULT_MAX_GUESS_SIZE as DEFAULT_HR_MAX_GUESS_SIZE,
    DEFAULT_MAX_LOW_COST_PAYMENTS,
    DEFAULT_MULTI_GUESS_CONTINUATION as DEFAULT_HR_MULTI_GUESS_CONTINUATION,
    DEFAULT_OVERPAY_PROBABILITY,
    HR1_FUGITIVE_ALGORITHM_ID,
    HR1_MARSHAL_ALGORITHM_ID,
    HierarchicalRandomFugitiveAgent,
    HierarchicalRandomMarshalAgent,
)
from .mcmc_bir import (
    BIR3_ALGORITHM_ID,
    BIR3_INITIAL_SEED_DOMAIN,
    BIR3_INITIAL_WEIGHTING_ID,
    BIR3_MCMC_KERNEL_ID,
    BIR3_REFERENCE_TARGET_ID,
    BIR3_REJUVENATION_SEED_DOMAIN,
    DEFAULT_MCMC_PARTICLE_COUNT,
    DEFAULT_MH_STEPS_PER_CHAIN,
    MCMCBeliefInformedRandomMarshalAgent,
)
from .route_count_random import (
    DEFAULT_ALPHA as DEFAULT_ROUTE_COUNT_ALPHA,
    DEFAULT_EPSILON as DEFAULT_ROUTE_COUNT_EPSILON,
    DEFAULT_MAX_GUESS_SIZE as DEFAULT_ROUTE_COUNT_MAX_GUESS_SIZE,
    DEFAULT_MAX_JOINT_CANDIDATES,
    DEFAULT_MULTI_GUESS_CONTINUATION as DEFAULT_ROUTE_COUNT_CONTINUATION,
    HR11_MARSHAL_ALGORITHM_ID,
    RouteCountRandomMarshalAgent,
)
from .unweighted_constructive_bir import (
    BIR2U_ALGORITHM_ID,
    BIR2U_BELIEF_SEED_DOMAIN,
    BIR2U_PROPOSAL_KERNEL_ID,
    BIR2U_REFERENCE_TARGET_ID,
    BIR2U_WEIGHTING_ID,
    UnweightedConstructiveBeliefInformedRandomMarshalAgent,
)


AgentT = TypeVar("AgentT")
FugitiveAgentFactory = Callable[..., FugitiveAgent]
MarshalAgentFactory = Callable[..., MarshalAgent]
DEFAULT_FUGITIVE_AGENT = "belief-informed-random"
DEFAULT_MARSHAL_AGENT = "belief-informed-random"
INTERACTIVE_BIR_PARTICLE_COUNT = 384
INTERACTIVE_BIR_MAX_GUESS_CANDIDATES = 64
INTERACTIVE_BIR2_PARTICLE_COUNT = 128
INTERACTIVE_BIR2_MAX_GUESS_CANDIDATES = 64
INTERACTIVE_BIR2E_PARTICLE_COUNT = 32
INTERACTIVE_BIR2E_MAX_GUESS_CANDIDATES = 32
INTERACTIVE_MCMC_BIR_PARTICLE_COUNT = 256
INTERACTIVE_MCMC_BIR_MAX_GUESS_CANDIDATES = 128
INTERACTIVE_MCMC_BIR_STEPS_PER_CHAIN = 1


# These tables are intentionally explicit.  They are the teaching/replay
# contract for parameters a user may set, independent of Python signatures or
# internal dependency-injection hooks on the concrete agent classes.
_HR_FUGITIVE_DEFAULTS = {
    "overpay_probability": DEFAULT_OVERPAY_PROBABILITY,
    "max_low_cost_payments": DEFAULT_MAX_LOW_COST_PAYMENTS,
    "max_extra_overpayments": DEFAULT_MAX_EXTRA_OVERPAYMENTS,
}
_BIR_FUGITIVE_DEFAULTS = {
    "epsilon": DEFAULT_BIR_EPSILON,
    "temperature": DEFAULT_TEMPERATURE,
    **_HR_FUGITIVE_DEFAULTS,
    "manhunt_epsilon": DEFAULT_MANHUNT_EPSILON,
    "manhunt_alpha": DEFAULT_MANHUNT_ALPHA,
    "manhunt_rollouts": DEFAULT_MANHUNT_ROLLOUTS,
}
_HR_MARSHAL_DEFAULTS = {
    "multi_guess_continuation": DEFAULT_HR_MULTI_GUESS_CONTINUATION,
    "max_guess_size": DEFAULT_HR_MAX_GUESS_SIZE,
    "max_guesses_per_size": DEFAULT_MAX_GUESSES_PER_SIZE,
}
_BIR_MARSHAL_ACTION_DEFAULTS = {
    "epsilon": DEFAULT_BIR_EPSILON,
    "temperature": DEFAULT_TEMPERATURE,
    "max_guess_candidates": DEFAULT_MAX_GUESS_CANDIDATES,
    "terminal_bonus_scale": 1.0,
    "manhunt_epsilon": DEFAULT_MANHUNT_EPSILON,
    "manhunt_alpha": DEFAULT_MANHUNT_ALPHA,
}
_BIR1_MARSHAL_DEFAULTS = {
    **_BIR_MARSHAL_ACTION_DEFAULTS,
    "particle_count": DEFAULT_PARTICLE_COUNT,
}
_ROUTE_COUNT_MARSHAL_DEFAULTS = {
    "alpha": DEFAULT_ROUTE_COUNT_ALPHA,
    "epsilon": DEFAULT_ROUTE_COUNT_EPSILON,
    "multi_guess_continuation": DEFAULT_ROUTE_COUNT_CONTINUATION,
    "max_guess_size": DEFAULT_ROUTE_COUNT_MAX_GUESS_SIZE,
    "max_joint_candidates": DEFAULT_MAX_JOINT_CANDIDATES,
}


def _constructive_marshal_defaults(particle_count: int) -> dict[str, object]:
    return {
        **_BIR_MARSHAL_ACTION_DEFAULTS,
        "particle_count": particle_count,
    }


@dataclass(frozen=True, slots=True)
class BuiltAgent(Generic[AgentT]):
    """A fresh policy together with its complete resolved configuration."""

    agent: AgentT
    spec: AgentSpec


@dataclass(frozen=True, slots=True)
class AgentRegistration(Generic[AgentT]):
    """A stable policy name with an explicit, inspectable construction schema.

    ``user_parameter_defaults`` is both the user-facing allowlist and the
    complete constructor-default table.  ``profile_overrides`` changes those
    defaults for named execution profiles.  ``fixed_algorithm_metadata`` is
    recorded in :class:`AgentSpec` and checked on the constructed policy, but
    is never forwarded to its constructor.
    """

    name: str
    role: Role
    factory: Callable[..., AgentT]
    user_parameter_defaults: Mapping[str, object] = field(default_factory=dict)
    expensive: bool = False
    profile_overrides: Mapping[str, Mapping[str, object]] = field(
        default_factory=dict
    )
    fixed_algorithm_metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("registration name must be a non-empty string")
        if not isinstance(self.role, Role):
            raise ValueError("registration role must be Fugitive or Marshal")

        defaults = normalize_parameters(self.user_parameter_defaults)
        fixed = normalize_parameters(self.fixed_algorithm_metadata)
        reserved = {
            name
            for name in defaults
            if name in ("seed", "rng") or name.startswith("_")
        }
        if reserved:
            names = ", ".join(sorted(reserved))
            raise ValueError(
                f"user parameter schema contains reserved names: {names}"
            )
        collision = set(defaults) & set(fixed)
        if collision:
            names = ", ".join(sorted(collision))
            raise ValueError(
                f"fixed algorithm metadata overlaps user parameters: {names}"
            )

        profiles: dict[str, Mapping[str, object]] = {}
        for profile, values in self.profile_overrides.items():
            if profile not in AGENT_PROFILES:
                raise ValueError(
                    f"unknown profile {profile!r} for agent {self.name}"
                )
            normalized = normalize_parameters(values)
            unknown = set(normalized) - set(defaults)
            if unknown:
                names = ", ".join(sorted(unknown))
                raise ValueError(
                    f"invalid {profile} profile for agent {self.name}: {names}"
                )
            profiles[profile] = MappingProxyType(normalized)

        object.__setattr__(
            self,
            "user_parameter_defaults",
            MappingProxyType(defaults),
        )
        object.__setattr__(
            self,
            "profile_overrides",
            MappingProxyType(profiles),
        )
        object.__setattr__(
            self,
            "fixed_algorithm_metadata",
            MappingProxyType(fixed),
        )

    def build(
        self,
        seed: int,
        *,
        profile: str = "default",
        overrides: Mapping[str, object] | None = None,
    ) -> BuiltAgent[AgentT]:
        """Construct an agent and record every effective policy parameter."""

        seed = validate_seed(seed, "agent seed")
        if profile not in AGENT_PROFILES:
            raise ValueError(
                f"agent profile must be one of {', '.join(AGENT_PROFILES)}"
            )
        supplied = normalize_parameters(overrides or {})
        if "seed" in supplied or "rng" in supplied:
            raise ValueError("agent parameters must not override seed or rng")
        fixed = self.fixed_algorithm_metadata
        for name, value in fixed.items():
            if name not in supplied:
                continue
            if supplied[name] != value:
                raise ValueError(
                    f"parameter {name} is fixed for agent {self.name}"
                )
            supplied.pop(name)

        unknown = set(supplied) - set(self.user_parameter_defaults)
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"unknown parameters for agent {self.name}: {names}")

        effective = dict(self.user_parameter_defaults)
        profile_values = self.profile_overrides.get(profile, {})
        effective.update(profile_values)
        effective.update(supplied)

        agent = self.factory(seed, **effective)
        for name, expected in fixed.items():
            if getattr(agent, name, None) != expected:
                raise RuntimeError(
                    f"agent {self.name} does not implement fixed parameter {name}"
                )
        # Constructors may resolve documented sentinel defaults. Prefer the
        # resulting public value so the spec records the exact policy.
        resolved = {
            name: getattr(agent, name, effective[name])
            for name in self.user_parameter_defaults
        }
        resolved = {**dict(fixed), **resolved}
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
            user_parameter_defaults=_HR_FUGITIVE_DEFAULTS,
            fixed_algorithm_metadata={
                "algorithm_id": HR1_FUGITIVE_ALGORITHM_ID,
            },
        ),
        AgentRegistration(
            "belief-informed-random",
            Role.FUGITIVE,
            cast(FugitiveAgentFactory, BeliefInformedRandomFugitiveAgent),
            user_parameter_defaults=_BIR_FUGITIVE_DEFAULTS,
            fixed_algorithm_metadata={
                "algorithm_id": BIR1_FUGITIVE_ALGORITHM_ID,
            },
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
            user_parameter_defaults=_HR_MARSHAL_DEFAULTS,
            fixed_algorithm_metadata={
                "algorithm_id": HR1_MARSHAL_ALGORITHM_ID,
            },
        ),
        AgentRegistration(
            "belief-informed-random",
            Role.MARSHAL,
            cast(MarshalAgentFactory, BeliefInformedRandomMarshalAgent),
            user_parameter_defaults=_BIR1_MARSHAL_DEFAULTS,
            expensive=True,
            profile_overrides={
                "interactive": {
                    "particle_count": INTERACTIVE_BIR_PARTICLE_COUNT,
                    "max_guess_candidates": INTERACTIVE_BIR_MAX_GUESS_CANDIDATES,
                }
            },
            fixed_algorithm_metadata={
                "algorithm_id": BIR1_MARSHAL_ALGORITHM_ID,
                "belief_backend_id": BIR1_BOOTSTRAP_BACKEND_ID,
                "max_incremental_play_proposals_per_particle": (
                    BIR1_INCREMENTAL_PLAY_PROPOSAL_LIMIT
                ),
                "exact_incremental_play_combination_threshold": (
                    BIR1_INCREMENTAL_EXACT_COMBINATION_THRESHOLD
                ),
                "resample_ess_fraction": BIR1_RESAMPLE_ESS_FRACTION,
                "sprint_backend": CONSTRUCTIVE_SPRINT_BACKEND,
                "proposal_kernel_id": BIR1_PROPOSAL_KERNEL_ID,
                "belief_seed_domain": BIR1_BELIEF_SEED_DOMAIN,
                "reference_target_id": BIR1_REFERENCE_TARGET_ID,
                "weighting_id": BIR1_WEIGHTING_ID,
                "public_event_model_id": BIR1_PUBLIC_EVENT_MODEL_ID,
                "incremental_transition_id": (
                    BIR1_INCREMENTAL_TRANSITION_ID
                ),
            },
        ),
        AgentRegistration(
            "route-count-random",
            Role.MARSHAL,
            cast(MarshalAgentFactory, RouteCountRandomMarshalAgent),
            user_parameter_defaults=_ROUTE_COUNT_MARSHAL_DEFAULTS,
            fixed_algorithm_metadata={
                "algorithm_id": HR11_MARSHAL_ALGORITHM_ID,
            },
        ),
        AgentRegistration(
            "constructive-belief-informed-random",
            Role.MARSHAL,
            cast(
                MarshalAgentFactory,
                ConstructiveBeliefInformedRandomMarshalAgent,
            ),
            user_parameter_defaults=_constructive_marshal_defaults(
                DEFAULT_PARTICLE_COUNT
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
            fixed_algorithm_metadata={
                "algorithm_id": BIR2_ALGORITHM_ID,
                "proposal_kernel_id": BIR2_PROPOSAL_KERNEL_ID,
                "sprint_backend": CONSTRUCTIVE_SPRINT_BACKEND,
                "belief_seed_domain": BIR2_BELIEF_SEED_DOMAIN,
                "reference_target_id": BIR2_REFERENCE_TARGET_ID,
                "weighting_id": BIR2_WEIGHTING_ID,
            },
        ),
        AgentRegistration(
            "unweighted-constructive-belief-informed-random",
            Role.MARSHAL,
            cast(
                MarshalAgentFactory,
                UnweightedConstructiveBeliefInformedRandomMarshalAgent,
            ),
            user_parameter_defaults=_constructive_marshal_defaults(
                DEFAULT_PARTICLE_COUNT
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
            fixed_algorithm_metadata={
                "algorithm_id": BIR2U_ALGORITHM_ID,
                "proposal_kernel_id": BIR2U_PROPOSAL_KERNEL_ID,
                "sprint_backend": CONSTRUCTIVE_SPRINT_BACKEND,
                "belief_seed_domain": BIR2U_BELIEF_SEED_DOMAIN,
                "reference_target_id": BIR2U_REFERENCE_TARGET_ID,
                "weighting_id": BIR2U_WEIGHTING_ID,
            },
        ),
        AgentRegistration(
            "exact-sprint-belief-informed-random",
            Role.MARSHAL,
            cast(
                MarshalAgentFactory,
                ExactSprintBeliefInformedRandomMarshalAgent,
            ),
            user_parameter_defaults=_constructive_marshal_defaults(
                DEFAULT_EXACT_SPRINT_PARTICLE_COUNT
            ),
            expensive=True,
            profile_overrides={
                "interactive": {
                    "particle_count": INTERACTIVE_BIR2E_PARTICLE_COUNT,
                    "max_guess_candidates": (
                        INTERACTIVE_BIR2E_MAX_GUESS_CANDIDATES
                    ),
                }
            },
            fixed_algorithm_metadata={
                "algorithm_id": EXACT_SPRINT_ALGORITHM_ID,
                "proposal_kernel_id": EXACT_SPRINT_PROPOSAL_KERNEL_ID,
                "sprint_backend": EXACT_SPRINT_BACKEND,
                "belief_seed_domain": EXACT_SPRINT_BELIEF_SEED_DOMAIN,
                "reference_target_id": EXACT_SPRINT_REFERENCE_TARGET_ID,
                "weighting_id": BIR2_WEIGHTING_ID,
            },
        ),
        AgentRegistration(
            "mcmc-belief-informed-random",
            Role.MARSHAL,
            cast(
                MarshalAgentFactory,
                MCMCBeliefInformedRandomMarshalAgent,
            ),
            user_parameter_defaults={
                **_constructive_marshal_defaults(DEFAULT_MCMC_PARTICLE_COUNT),
                "mh_steps_per_chain": DEFAULT_MH_STEPS_PER_CHAIN,
            },
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
            fixed_algorithm_metadata={
                "algorithm_id": BIR3_ALGORITHM_ID,
                "mcmc_kernel_id": BIR3_MCMC_KERNEL_ID,
                "initial_weighting_id": BIR3_INITIAL_WEIGHTING_ID,
                "initial_seed_domain": BIR3_INITIAL_SEED_DOMAIN,
                "rejuvenation_seed_domain": BIR3_REJUVENATION_SEED_DOMAIN,
                "proposal_kernel_id": BIR2_PROPOSAL_KERNEL_ID,
                "sprint_backend": CONSTRUCTIVE_SPRINT_BACKEND,
                "reference_target_id": BIR3_REFERENCE_TARGET_ID,
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
    "INTERACTIVE_BIR2E_MAX_GUESS_CANDIDATES",
    "INTERACTIVE_BIR2E_PARTICLE_COUNT",
    "INTERACTIVE_BIR_MAX_GUESS_CANDIDATES",
    "INTERACTIVE_BIR_PARTICLE_COUNT",
    "INTERACTIVE_MCMC_BIR_MAX_GUESS_CANDIDATES",
    "INTERACTIVE_MCMC_BIR_PARTICLE_COUNT",
    "INTERACTIVE_MCMC_BIR_STEPS_PER_CHAIN",
    "MARSHAL_AGENT_REGISTRY",
]
