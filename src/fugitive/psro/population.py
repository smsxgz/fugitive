"""Pure-policy identities, populations, and mixed strategies."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from typing import Iterable, Mapping, Sequence

from ..game.model import Role
from ..shared.reproducibility import (
    FrozenJSONValue,
    JSONValue,
    freeze_parameters,
    thaw_parameters,
)
from ._json_validation import (
    require_finite as _require_finite,
    require_mapping as _require_mapping,
    require_sequence as _require_sequence,
)

@dataclass(frozen=True, slots=True)
class PolicyIdentifier:
    """A stable policy name qualified by the role that can use it."""

    role: Role
    name: str

    def __post_init__(self) -> None:
        if not isinstance(self.role, Role):
            raise ValueError("policy role must be Fugitive or Marshal")
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("policy name must be a non-empty string")
        if self.name != self.name.strip():
            raise ValueError("policy name must not have surrounding whitespace")

    def to_dict(self) -> dict[str, JSONValue]:
        return {"role": self.role.value, "name": self.name}

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "PolicyIdentifier":
        role = data.get("role")
        name = data.get("name")
        if not isinstance(role, str) or not isinstance(name, str):
            raise ValueError("policy identifier requires string role and name")
        return cls(Role(role), name)


@dataclass(frozen=True, slots=True)
class PolicySpec:
    """An immutable construction specification for one pure policy."""

    identifier: PolicyIdentifier
    parameters: Mapping[str, FrozenJSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.identifier, PolicyIdentifier):
            raise ValueError("policy spec requires a PolicyIdentifier")
        object.__setattr__(self, "parameters", freeze_parameters(self.parameters))

    @property
    def role(self) -> Role:
        return self.identifier.role

    @classmethod
    def create(
        cls,
        role: Role,
        name: str,
        parameters: Mapping[str, object] | None = None,
    ) -> "PolicySpec":
        return cls(PolicyIdentifier(role, name), parameters or {})

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "identifier": self.identifier.to_dict(),
            "parameters": thaw_parameters(self.parameters),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "PolicySpec":
        identifier = PolicyIdentifier.from_dict(
            _require_mapping(data.get("identifier"), "policy identifier")
        )
        parameters = _require_mapping(
            data.get("parameters", {}), "policy parameters"
        )
        return cls(identifier, parameters)


def _validate_policy_bucket(
    policies: tuple[PolicySpec, ...], role: Role
) -> None:
    seen: set[PolicyIdentifier] = set()
    for policy in policies:
        if not isinstance(policy, PolicySpec) or policy.role is not role:
            raise ValueError(f"{role.value} population contains a wrong-role policy")
        if policy.identifier in seen:
            raise ValueError(f"duplicate policy identifier: {policy.identifier.name}")
        seen.add(policy.identifier)


@dataclass(frozen=True, slots=True)
class PolicyPopulation:
    """Role-separated ordered pure policies in one restricted game."""

    fugitive_policies: tuple[PolicySpec, ...] = ()
    marshal_policies: tuple[PolicySpec, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "fugitive_policies", tuple(self.fugitive_policies))
        object.__setattr__(self, "marshal_policies", tuple(self.marshal_policies))
        _validate_policy_bucket(self.fugitive_policies, Role.FUGITIVE)
        _validate_policy_bucket(self.marshal_policies, Role.MARSHAL)

    @classmethod
    def create(
        cls,
        *,
        fugitive: Iterable[PolicySpec] = (),
        marshal: Iterable[PolicySpec] = (),
    ) -> "PolicyPopulation":
        return cls(tuple(fugitive), tuple(marshal))

    def policies(self, role: Role) -> tuple[PolicySpec, ...]:
        if role is Role.FUGITIVE:
            return self.fugitive_policies
        if role is Role.MARSHAL:
            return self.marshal_policies
        raise ValueError("population role must be Fugitive or Marshal")

    def identifiers(self, role: Role) -> tuple[PolicyIdentifier, ...]:
        return tuple(policy.identifier for policy in self.policies(role))

    def policy(self, identifier: PolicyIdentifier) -> PolicySpec:
        for policy in self.policies(identifier.role):
            if policy.identifier == identifier:
                return policy
        raise KeyError(identifier)

    def with_policies(self, *policies: PolicySpec) -> "PolicyPopulation":
        fugitive = list(self.fugitive_policies)
        marshal = list(self.marshal_policies)
        existing = {
            policy.identifier: policy
            for policy in (*self.fugitive_policies, *self.marshal_policies)
        }
        for policy in policies:
            if not isinstance(policy, PolicySpec):
                raise ValueError("population additions must be PolicySpec values")
            previous = existing.get(policy.identifier)
            if previous is not None:
                if previous != policy:
                    raise ValueError(
                        "a policy identifier cannot be reused with new parameters: "
                        f"{policy.identifier.name}"
                    )
                continue
            existing[policy.identifier] = policy
            if policy.role is Role.FUGITIVE:
                fugitive.append(policy)
            else:
                marshal.append(policy)
        return PolicyPopulation(tuple(fugitive), tuple(marshal))

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "fugitive": [policy.to_dict() for policy in self.fugitive_policies],
            "marshal": [policy.to_dict() for policy in self.marshal_policies],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "PolicyPopulation":
        fugitive = tuple(
            PolicySpec.from_dict(_require_mapping(item, "Fugitive policy"))
            for item in _require_sequence(data.get("fugitive"), "Fugitive policies")
        )
        marshal = tuple(
            PolicySpec.from_dict(_require_mapping(item, "Marshal policy"))
            for item in _require_sequence(data.get("marshal"), "Marshal policies")
        )
        return cls(fugitive, marshal)


@dataclass(frozen=True, slots=True)
class PolicyProbability:
    identifier: PolicyIdentifier
    probability: float

    def __post_init__(self) -> None:
        if not isinstance(self.identifier, PolicyIdentifier):
            raise ValueError("mixture entry requires a policy identifier")
        probability = _require_finite(self.probability, "policy probability")
        if probability < 0.0:
            raise ValueError("policy probability must be non-negative")
        object.__setattr__(self, "probability", probability)

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "identifier": self.identifier.to_dict(),
            "probability": self.probability,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "PolicyProbability":
        return cls(
            PolicyIdentifier.from_dict(
                _require_mapping(data.get("identifier"), "mixture identifier")
            ),
            _require_finite(data.get("probability"), "policy probability"),
        )


@dataclass(frozen=True, slots=True)
class PolicyMixture:
    """An immutable mixed policy for exactly one role."""

    role: Role
    entries: tuple[PolicyProbability, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.role, Role):
            raise ValueError("mixture role must be Fugitive or Marshal")
        object.__setattr__(self, "entries", tuple(self.entries))
        if not self.entries:
            raise ValueError("a policy mixture cannot be empty")
        seen: set[PolicyIdentifier] = set()
        for entry in self.entries:
            if entry.identifier.role is not self.role:
                raise ValueError("mixture entry has the wrong role")
            if entry.identifier in seen:
                raise ValueError("mixture contains a duplicate policy")
            seen.add(entry.identifier)
        total = math.fsum(entry.probability for entry in self.entries)
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("policy mixture probabilities must sum to one")

    @classmethod
    def from_probabilities(
        cls,
        role: Role,
        identifiers: Sequence[PolicyIdentifier],
        probabilities: Sequence[float],
    ) -> "PolicyMixture":
        if len(identifiers) != len(probabilities) or not identifiers:
            raise ValueError("mixture identifiers and probabilities must align")
        checked = [
            _require_finite(probability, "policy probability")
            for probability in probabilities
        ]
        if any(probability < 0.0 for probability in checked):
            raise ValueError("policy probability must be non-negative")
        total = math.fsum(checked)
        if total <= 0.0:
            raise ValueError("a policy mixture must have positive mass")
        normalized = [probability / total for probability in checked]
        normalized[-1] += 1.0 - math.fsum(normalized)
        return cls(
            role,
            tuple(
                PolicyProbability(identifier, probability)
                for identifier, probability in zip(
                    identifiers, normalized, strict=True
                )
            ),
        )

    def probability(self, identifier: PolicyIdentifier) -> float:
        if identifier.role is not self.role:
            return 0.0
        return next(
            (
                entry.probability
                for entry in self.entries
                if entry.identifier == identifier
            ),
            0.0,
        )

    @property
    def support(self) -> tuple[PolicyIdentifier, ...]:
        return tuple(
            entry.identifier for entry in self.entries if entry.probability > 0.0
        )

    def sample(self, rng: random.Random) -> PolicyIdentifier:
        threshold = rng.random()
        cumulative = 0.0
        for entry in self.entries:
            cumulative += entry.probability
            if threshold < cumulative:
                return entry.identifier
        return self.entries[-1].identifier

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "role": self.role.value,
            "entries": [entry.to_dict() for entry in self.entries],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "PolicyMixture":
        role = data.get("role")
        if not isinstance(role, str):
            raise ValueError("mixture requires a string role")
        entries = tuple(
            PolicyProbability.from_dict(_require_mapping(item, "mixture entry"))
            for item in _require_sequence(data.get("entries"), "mixture entries")
        )
        return cls(Role(role), entries)

__all__ = [
    "PolicyIdentifier",
    "PolicyMixture",
    "PolicyPopulation",
    "PolicyProbability",
    "PolicySpec",
]
