"""Registry-backed policies and response oracles for full-game PSRO."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Mapping, Sequence, cast

from ..game.model import Role
from .planning_leaf import (
    PLANNING_LEAF_PROJECTION_ID,
    is_online_search_spec,
    planning_leaf_spec,
)
from .algorithm import ResponseOracleRequest
from .population import PolicyIdentifier, PolicyPopulation, PolicySpec
from ..shared.reproducibility import (
    AgentSpec,
    FrozenJSONValue,
    JSONValue,
    freeze_parameters,
    normalize_parameters,
    thaw_parameters,
    validate_seed,
)


REGISTERED_POLICY_ADAPTER_ID = "registered-agent-and-planning-leaf-v2"


def _require_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return cast(Mapping[str, object], value)


def _require_sequence(value: object, label: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{label} must be a JSON array")
    return cast(Sequence[object], value)


def _registry_for(role: Role):
    # Lazy import keeps the generic PSRO types independent of agent modules.
    from ..agents.registry import FUGITIVE_AGENT_REGISTRY, MARSHAL_AGENT_REGISTRY

    return (
        FUGITIVE_AGENT_REGISTRY
        if role is Role.FUGITIVE
        else MARSHAL_AGENT_REGISTRY
    )


def _canonical_agent_spec(agent_spec: AgentSpec) -> str:
    return json.dumps(
        agent_spec.to_dict(),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def registered_policy_spec(
    agent_spec: AgentSpec,
    *,
    identifier_name: str | None = None,
    metadata: Mapping[str, object] | None = None,
) -> PolicySpec:
    """Store one executable policy and its non-search planning leaf."""

    if not isinstance(agent_spec, AgentSpec):
        raise ValueError("registered policy requires an AgentSpec")
    digest = hashlib.sha256(_canonical_agent_spec(agent_spec).encode("utf-8"))
    default_name = f"{agent_spec.name}@{digest.hexdigest()[:12]}"
    leaf_spec = planning_leaf_spec(agent_spec)
    return PolicySpec.create(
        agent_spec.role,
        identifier_name or default_name,
        {
            "adapter_id": REGISTERED_POLICY_ADAPTER_ID,
            "agent_spec": agent_spec.to_dict(),
            "planning_leaf_spec": leaf_spec.to_dict(),
            "planning_leaf_projection_id": PLANNING_LEAF_PROJECTION_ID,
            "online_search": is_online_search_spec(agent_spec),
            "metadata": dict(metadata or {}),
        },
    )


def agent_spec_from_policy(
    policy: PolicySpec,
    repository: PolicyPopulation | None = None,
) -> AgentSpec:
    """Recover, and when needed materialize, a registered agent spec.

    Mixture-conditioned search responses keep only opponent policy IDs in the
    checkpoint.  Their concrete non-search leaf mixture is resolved from the
    population immediately before a payoff game is constructed.
    """

    if not isinstance(policy, PolicySpec):
        raise ValueError("registered policy must be a PolicySpec")
    parameters = thaw_parameters(policy.parameters)
    if parameters.get("adapter_id") != REGISTERED_POLICY_ADAPTER_ID:
        raise ValueError("policy is not a registered-agent PSRO policy")
    agent_data = _require_mapping(parameters.get("agent_spec"), "agent spec")
    agent_spec = AgentSpec.from_dict(agent_data)
    if agent_spec.role is not policy.role:
        raise ValueError("PSRO policy role disagrees with its AgentSpec")
    metadata = _require_mapping(parameters.get("metadata", {}), "policy metadata")
    opponent_parameter = metadata.get("deferred_opponent_parameter")
    if opponent_parameter is None:
        return agent_spec
    if not isinstance(opponent_parameter, str) or not opponent_parameter:
        raise ValueError("deferred opponent parameter must be non-empty text")
    if repository is None:
        raise ValueError(
            "mixture-conditioned policy requires its PolicyPopulation repository"
        )
    references = _require_sequence(
        metadata.get("opponent_leaf_mixture"), "opponent leaf mixture"
    )
    choices: list[dict[str, object]] = []
    for item in references:
        reference = _require_mapping(item, "opponent leaf reference")
        identifiers = tuple(
            PolicyIdentifier.from_dict(_require_mapping(value, "policy ID"))
            for value in _require_sequence(
                reference.get("policy_ids"), "opponent policy IDs"
            )
        )
        if not identifiers:
            raise ValueError("opponent leaf reference has no policy IDs")
        leaves = tuple(
            planning_leaf_spec_from_policy(repository.policy(identifier))
            for identifier in identifiers
        )
        canonical = {_canonical_agent_spec(leaf) for leaf in leaves}
        if len(canonical) != 1:
            raise ValueError("coalesced opponent policy IDs have different leaves")
        expected_hash = reference.get("leaf_spec_sha256")
        actual_hash = hashlib.sha256(
            _canonical_agent_spec(leaves[0]).encode("utf-8")
        ).hexdigest()
        if expected_hash != actual_hash:
            raise ValueError("opponent planning leaf no longer matches checkpoint")
        weight = reference.get("weight")
        if (
            isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not math.isfinite(weight)
            or weight <= 0.0
        ):
            raise ValueError("opponent leaf weight must be finite and positive")
        choices.append({"spec": leaves[0].to_dict(), "weight": float(weight)})

    concrete_parameters = thaw_parameters(agent_spec.parameters)
    concrete_parameters[opponent_parameter] = choices
    registration = _registry_for(policy.role).get(agent_spec.name)
    if registration is None:
        raise ValueError(f"unknown deferred response agent: {agent_spec.name}")
    return registration.build(
        0,
        profile=agent_spec.profile,
        overrides=concrete_parameters,
    ).spec


def planning_leaf_spec_from_policy(policy: PolicySpec) -> AgentSpec:
    """Recover the frozen non-search leaf stored with a PSRO policy."""

    if not isinstance(policy, PolicySpec):
        raise ValueError("registered policy must be a PolicySpec")
    parameters = thaw_parameters(policy.parameters)
    if parameters.get("adapter_id") != REGISTERED_POLICY_ADAPTER_ID:
        raise ValueError("policy is not a registered-agent PSRO policy")
    if parameters.get("planning_leaf_projection_id") != (
        PLANNING_LEAF_PROJECTION_ID
    ):
        raise ValueError("policy uses an unsupported planning-leaf projection")
    leaf_data = _require_mapping(
        parameters.get("planning_leaf_spec"), "planning leaf spec"
    )
    leaf_spec = AgentSpec.from_dict(leaf_data)
    if leaf_spec.role is not policy.role:
        raise ValueError("PSRO planning leaf has the wrong role")
    if is_online_search_spec(leaf_spec):
        raise ValueError("a PSRO planning leaf cannot itself be an online search")
    return leaf_spec


def resolve_registered_policy(
    role: Role,
    agent_name: str,
    *,
    identifier_name: str | None = None,
    profile: str = "default",
    overrides: Mapping[str, object] | None = None,
    resolution_seed: int = 0,
) -> PolicySpec:
    """Resolve registry defaults and return a fully specified PSRO policy."""

    if not isinstance(role, Role):
        raise ValueError("registered policy role must be Fugitive or Marshal")
    try:
        registration = _registry_for(role)[agent_name]
    except KeyError as exc:
        raise ValueError(f"unknown {role.value} agent: {agent_name}") from exc
    built = registration.build(
        validate_seed(resolution_seed, "resolution seed"),
        profile=profile,
        overrides=normalize_parameters(overrides or {}),
    )
    return registered_policy_spec(
        built.spec,
        identifier_name=identifier_name,
    )


@dataclass(frozen=True, slots=True)
class MixtureResponseTemplate:
    """A registered response candidate with optional opponent conditioning.

    When ``opponent_parameter`` is set, the real response policy receives a
    flat mixture of non-search leaf specs.  It may search at the root, but a
    hypothetical continuation can never start another online search.  When it
    is ``None``, the template is a cheap mixture-independent candidate.
    """

    role: Role
    agent_name: str
    profile: str = "default"
    base_parameters: Mapping[str, FrozenJSONValue] = field(default_factory=dict)
    opponent_parameter: str | None = "opponent_policies"
    identifier_prefix: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.role, Role):
            raise ValueError("response template role must be Fugitive or Marshal")
        if not isinstance(self.agent_name, str) or not self.agent_name:
            raise ValueError("response template requires an agent name")
        if not isinstance(self.profile, str) or not self.profile:
            raise ValueError("response template requires a profile")
        if self.opponent_parameter is not None and (
            not isinstance(self.opponent_parameter, str)
            or not self.opponent_parameter
        ):
            raise ValueError("response opponent parameter must be non-empty")
        if self.identifier_prefix is not None and (
            not isinstance(self.identifier_prefix, str)
            or not self.identifier_prefix
        ):
            raise ValueError("response identifier prefix must be non-empty")
        object.__setattr__(
            self,
            "base_parameters",
            freeze_parameters(self.base_parameters),
        )

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "role": self.role.value,
            "agent_name": self.agent_name,
            "profile": self.profile,
            "base_parameters": thaw_parameters(self.base_parameters),
            "opponent_parameter": self.opponent_parameter,
            "identifier_prefix": self.identifier_prefix,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "MixtureResponseTemplate":
        role = data.get("role")
        agent_name = data.get("agent_name")
        profile = data.get("profile")
        opponent_parameter = data.get("opponent_parameter")
        identifier_prefix = data.get("identifier_prefix")
        if not isinstance(role, str):
            raise ValueError("response template role must be a string")
        if not isinstance(agent_name, str):
            raise ValueError("response template agent_name must be a string")
        if not isinstance(profile, str):
            raise ValueError("response template profile must be a string")
        if opponent_parameter is not None and not isinstance(
            opponent_parameter, str
        ):
            raise ValueError(
                "response template opponent_parameter must be a string or null"
            )
        if identifier_prefix is not None and not isinstance(
            identifier_prefix, str
        ):
            raise ValueError("response template identifier_prefix must be a string")
        return cls(
            role=Role(role),
            agent_name=agent_name,
            profile=profile,
            base_parameters=_require_mapping(
                data.get("base_parameters"), "response base parameters"
            ),
            opponent_parameter=opponent_parameter,
            identifier_prefix=identifier_prefix,
        )


class MixtureConditionedResponseOracle:
    """Adapt a rollout/search template to the current opponent meta-mixture.

    The template itself is the approximate-response algorithm.  This adapter
    freezes the current opponent mixture into its constructor parameters and
    assigns a deterministic generation-specific PSRO identity.
    """

    def __init__(self, template: MixtureResponseTemplate) -> None:
        if not isinstance(template, MixtureResponseTemplate):
            raise ValueError("response oracle requires a mixture template")
        self.template = template

    def propose_response(self, request: ResponseOracleRequest) -> PolicySpec:
        if request.role is not self.template.role:
            raise ValueError("response request has the wrong template role")
        weighted_specs: dict[
            str, tuple[AgentSpec, float, list[dict[str, str]]]
        ] = {}
        if self.template.opponent_parameter is not None:
            for entry in request.opponent_mixture.entries:
                if entry.probability <= 0.0:
                    continue
                opponent = request.population.policy(entry.identifier)
                opponent_spec = planning_leaf_spec_from_policy(opponent)
                key = _canonical_agent_spec(opponent_spec)
                previous = weighted_specs.get(key)
                weighted_specs[key] = (
                    opponent_spec,
                    entry.probability
                    + (0.0 if previous is None else previous[1]),
                    [
                        *([] if previous is None else previous[2]),
                        entry.identifier.to_dict(),
                    ],
                )
            if not weighted_specs:
                raise ValueError("opponent meta-mixture has no positive support")
        choices = [
            {"spec": spec.to_dict(), "weight": weight}
            for spec, weight, _identifiers in weighted_specs.values()
        ]

        parameters = thaw_parameters(self.template.base_parameters)
        if self.template.opponent_parameter is not None:
            parameters[self.template.opponent_parameter] = choices
        agent_spec = self._resolve_agent_spec(parameters)
        for incumbent in request.population.policies(request.role):
            try:
                incumbent_spec = agent_spec_from_policy(
                    incumbent,
                    request.population,
                )
            except ValueError:
                continue
            if incumbent_spec == agent_spec:
                return incumbent
        digest = hashlib.sha256(
            _canonical_agent_spec(agent_spec).encode("utf-8")
        ).hexdigest()[:12]
        prefix = self.template.identifier_prefix or self.template.agent_name
        identifier = f"{prefix}-g{request.generation}-{digest}"
        opponent_references = [
            {
                "policy_ids": identifiers,
                "leaf_spec_sha256": hashlib.sha256(
                    _canonical_agent_spec(spec).encode("utf-8")
                ).hexdigest(),
                "weight": weight,
            }
            for spec, weight, identifiers in weighted_specs.values()
        ]
        stored_spec = agent_spec
        if self.template.opponent_parameter is not None:
            stored_parameters = thaw_parameters(agent_spec.parameters)
            stored_parameters.pop(self.template.opponent_parameter, None)
            stored_spec = AgentSpec(
                agent_spec.name,
                agent_spec.role,
                agent_spec.profile,
                stored_parameters,
            )
        return registered_policy_spec(
            stored_spec,
            identifier_name=identifier,
            metadata={
                "candidate_kind": (
                    "mixture-conditioned-search-candidate"
                    if self.template.opponent_parameter is not None
                    else "registered-policy-candidate"
                ),
                "planning_depth": 1 if is_online_search_spec(agent_spec) else 0,
                "opponent_leaf_mixture": opponent_references,
                "deferred_opponent_parameter": self.template.opponent_parameter,
            },
        )

    def _resolve_agent_spec(self, parameters: Mapping[str, object]) -> AgentSpec:
        registration = _registry_for(self.template.role).get(
            self.template.agent_name
        )
        if registration is None:
            raise ValueError(
                f"unknown {self.template.role.value} response agent: "
                f"{self.template.agent_name}"
            )
        return registration.build(
            0,
            profile=self.template.profile,
            overrides=normalize_parameters(parameters),
        ).spec


__all__ = [
    "MixtureConditionedResponseOracle",
    "MixtureResponseTemplate",
    "REGISTERED_POLICY_ADAPTER_ID",
    "agent_spec_from_policy",
    "planning_leaf_spec_from_policy",
    "registered_policy_spec",
    "resolve_registered_policy",
]
