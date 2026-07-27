"""Replayable factories for opponent-policy mixtures used by rollout agents."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import random
from typing import Iterable, Mapping

from fugitive.game.model import FugitiveAgent, MarshalAgent, Role
from fugitive.shared.reproducibility import AgentSpec, derive_seed, normalize_parameters


@dataclass(frozen=True, slots=True)
class WeightedPolicySpec:
    spec: AgentSpec
    weight: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.weight, bool)
            or not isinstance(self.weight, (int, float))
            or not math.isfinite(self.weight)
            or self.weight <= 0.0
        ):
            raise ValueError("policy weight must be finite and positive")

    def to_dict(self) -> dict[str, object]:
        return {"spec": self.spec.to_dict(), "weight": float(self.weight)}

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "WeightedPolicySpec":
        spec = data.get("spec")
        if not isinstance(spec, Mapping):
            raise ValueError("weighted policy requires an AgentSpec object")
        weight = data.get("weight")
        if isinstance(weight, bool) or not isinstance(weight, (int, float)):
            raise ValueError("weighted policy requires a numeric weight")
        return cls(AgentSpec.from_dict(spec), float(weight))


class RegisteredPolicyMixtureFactory:
    """Build one registered policy sampled from a frozen meta-strategy."""

    def __init__(
        self,
        role: Role,
        choices: Iterable[WeightedPolicySpec | Mapping[str, object]],
    ) -> None:
        if not isinstance(role, Role):
            raise ValueError("role must be Fugitive or Marshal")
        parsed = tuple(
            choice
            if isinstance(choice, WeightedPolicySpec)
            else WeightedPolicySpec.from_dict(choice)
            for choice in choices
        )
        if not parsed:
            raise ValueError("a policy mixture requires at least one policy")
        if any(choice.spec.role is not role for choice in parsed):
            raise ValueError("every policy in a mixture must have the requested role")
        identities = {
            json.dumps(choice.spec.to_dict(), sort_keys=True, separators=(",", ":"))
            for choice in parsed
        }
        if len(identities) != len(parsed):
            raise ValueError("duplicate policy specs are not allowed in a mixture")
        self.role = role
        self.choices = parsed
        total = math.fsum(choice.weight for choice in parsed)
        self.normalized_weights = tuple(choice.weight / total for choice in parsed)

    @property
    def serialized_choices(self) -> tuple[dict[str, object], ...]:
        return tuple(choice.to_dict() for choice in self.choices)

    def __call__(self, seed: int) -> FugitiveAgent | MarshalAgent:
        rng = random.Random(seed)
        threshold = rng.random()
        cumulative = 0.0
        selected = self.choices[-1]
        for choice, weight in zip(
            self.choices, self.normalized_weights, strict=True
        ):
            cumulative += weight
            if threshold < cumulative:
                selected = choice
                break

        # Imported lazily to avoid a registry -> rollout agent -> registry
        # import cycle.  Construction happens only after registry import ends.
        from ..registry import FUGITIVE_AGENT_REGISTRY, MARSHAL_AGENT_REGISTRY

        registry = (
            FUGITIVE_AGENT_REGISTRY
            if self.role is Role.FUGITIVE
            else MARSHAL_AGENT_REGISTRY
        )
        try:
            registration = registry[selected.spec.name]
        except KeyError as exc:
            raise ValueError(
                f"unknown {self.role.value} mixture policy: {selected.spec.name}"
            ) from exc
        policy_seed = derive_seed(
            seed,
            f"rollout-mixture:{self.role.value}:{selected.spec.name}",
        )
        return registration.build(
            policy_seed,
            profile=selected.spec.profile,
            overrides=normalize_parameters(selected.spec.parameters),
        ).agent


def default_policy_spec(name: str, role: Role) -> WeightedPolicySpec:
    """Resolve and freeze one registry policy's complete default AgentSpec."""

    from ..registry import FUGITIVE_AGENT_REGISTRY, MARSHAL_AGENT_REGISTRY

    registry = (
        FUGITIVE_AGENT_REGISTRY
        if role is Role.FUGITIVE
        else MARSHAL_AGENT_REGISTRY
    )
    try:
        registration = registry[name]
    except KeyError as exc:
        raise ValueError(f"unknown {role.value} policy: {name}") from exc
    return WeightedPolicySpec(registration.build(0).spec, 1.0)


__all__ = [
    "RegisteredPolicyMixtureFactory",
    "WeightedPolicySpec",
    "default_policy_spec",
]
