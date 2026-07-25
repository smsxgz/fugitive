"""Shared deterministic seed and agent-description protocol.

This module is intentionally independent of the Web and experiment runners.
Both entry points use the same domain-separated random streams and the same
JSON-safe descriptions of constructed policies.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from types import MappingProxyType
from typing import Mapping, TypeAlias, cast

from .model import Role


# Preserve the mapping originally published by fugitive.experiment.  Changing
# this value is a replay-protocol change, not a normal implementation detail.
SEED_DERIVATION_VERSION = "fugitive-experiment-v1"
AGENT_PROFILES = ("default", "interactive")
MIN_SEED = 0
MAX_SEED = (1 << 64) - 1

JSONValue: TypeAlias = (
    None | bool | int | float | str | list["JSONValue"] | dict[str, "JSONValue"]
)
FrozenJSONValue: TypeAlias = (
    None
    | bool
    | int
    | float
    | str
    | tuple["FrozenJSONValue", ...]
    | Mapping[str, "FrozenJSONValue"]
)


@dataclass(frozen=True, slots=True)
class SeedBundle:
    """Independent random streams used by one game."""

    master: int | None
    deck: int
    fugitive: int | None
    marshal: int | None

    def __post_init__(self) -> None:
        for label, value, optional in (
            ("master", self.master, True),
            ("deck", self.deck, False),
            ("fugitive", self.fugitive, True),
            ("marshal", self.marshal, True),
        ):
            if value is None and optional:
                continue
            validate_seed(value, f"{label} seed")  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, str | None]:
        return {
            "master": seed_to_text(self.master),
            "deck": str(self.deck),
            "fugitive": seed_to_text(self.fugitive),
            "marshal": seed_to_text(self.marshal),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "SeedBundle":
        return cls(
            master=parse_optional_seed(data.get("master"), "master"),
            deck=parse_seed(data.get("deck"), "deck"),
            fugitive=parse_optional_seed(data.get("fugitive"), "fugitive"),
            marshal=parse_optional_seed(data.get("marshal"), "marshal"),
        )


@dataclass(frozen=True, slots=True)
class AgentDescriptor:
    """Legacy manifest identity and constructor parameters for one policy."""

    name: str
    parameters: Mapping[str, FrozenJSONValue]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("agent name must be a non-empty string")
        object.__setattr__(self, "parameters", freeze_parameters(self.parameters))

    @classmethod
    def create(
        cls,
        name: str,
        parameters: Mapping[str, object] | None = None,
    ) -> "AgentDescriptor":
        return cls(name=name, parameters=parameters or {})

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "name": self.name,
            "parameters": thaw_parameters(self.parameters),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "AgentDescriptor":
        name = data.get("name")
        parameters = data.get("parameters", {})
        if not isinstance(name, str):
            raise ValueError("agent descriptor has no valid name")
        if not isinstance(parameters, Mapping):
            raise ValueError("agent parameters must be an object")
        return cls.create(name, cast(Mapping[str, object], parameters))


@dataclass(frozen=True, slots=True)
class AgentSpec:
    """Complete resolved registry configuration for one constructed agent."""

    name: str
    role: Role
    profile: str
    parameters: Mapping[str, FrozenJSONValue]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("agent name must be a non-empty string")
        if not isinstance(self.role, Role):
            raise ValueError("agent role must be Role.FUGITIVE or Role.MARSHAL")
        if self.profile not in AGENT_PROFILES:
            raise ValueError(
                f"agent profile must be one of {', '.join(AGENT_PROFILES)}"
            )
        object.__setattr__(self, "parameters", freeze_parameters(self.parameters))

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "name": self.name,
            "role": self.role.value,
            "profile": self.profile,
            "parameters": thaw_parameters(self.parameters),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "AgentSpec":
        name = data.get("name")
        role = data.get("role")
        profile = data.get("profile")
        parameters = data.get("parameters")
        if not isinstance(name, str):
            raise ValueError("agent spec has no valid name")
        if not isinstance(profile, str):
            raise ValueError("agent spec has no valid profile")
        if not isinstance(parameters, Mapping):
            raise ValueError("agent spec parameters must be an object")
        return cls(
            name,
            Role(str(role)),
            profile,
            cast(Mapping[str, object], parameters),
        )

    def descriptor(self) -> AgentDescriptor:
        """Return the schema-v1 replay descriptor for this resolved spec."""

        return AgentDescriptor.create(self.name, self.parameters)


def derive_seed(master_seed: int, domain: str) -> int:
    """Derive a deterministic 64-bit seed with explicit domain separation."""

    master_seed = validate_seed(master_seed, "master_seed")
    if not isinstance(domain, str) or not domain:
        raise ValueError("domain must be a non-empty string")
    payload = f"{SEED_DERIVATION_VERSION}\0{master_seed}\0{domain}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def derive_seed_bundle(master_seed: int) -> SeedBundle:
    """Derive unrelated deck, Fugitive-policy, and Marshal-policy streams."""

    return SeedBundle(
        master=master_seed,
        deck=derive_seed(master_seed, "deck"),
        fugitive=derive_seed(master_seed, "fugitive"),
        marshal=derive_seed(master_seed, "marshal"),
    )


def normalize_parameters(parameters: Mapping[str, object]) -> dict[str, JSONValue]:
    """Detach and validate a mapping for stable JSON serialization."""

    normalized = json_value(parameters)
    assert isinstance(normalized, dict)
    return cast(
        dict[str, JSONValue],
        json.loads(json.dumps(normalized, allow_nan=False, sort_keys=True)),
    )


def freeze_parameters(
    parameters: Mapping[str, object],
) -> Mapping[str, FrozenJSONValue]:
    """Return a detached, recursively immutable JSON mapping."""

    normalized = normalize_parameters(parameters)
    frozen = _freeze_json_value(normalized)
    assert isinstance(frozen, Mapping)
    return frozen


def thaw_parameters(
    parameters: Mapping[str, FrozenJSONValue],
) -> dict[str, JSONValue]:
    """Return a mutable JSON copy suitable for serialization."""

    thawed = _thaw_json_value(parameters)
    assert isinstance(thawed, dict)
    return thawed


def _freeze_json_value(value: JSONValue) -> FrozenJSONValue:
    if isinstance(value, dict):
        return MappingProxyType(
            {key: _freeze_json_value(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json_value(item) for item in value)
    return value


def _thaw_json_value(value: FrozenJSONValue) -> JSONValue:
    if isinstance(value, Mapping):
        return {
            key: _thaw_json_value(item)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return [_thaw_json_value(item) for item in value]
    return value


def json_value(value: object) -> JSONValue:
    """Convert plain JSON-compatible values while rejecting lossy values."""

    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite values are not JSON serializable")
        return value
    if isinstance(value, Mapping):
        result: dict[str, JSONValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("JSON object keys must be strings")
            result[key] = json_value(item)
        return result
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    raise ValueError(f"unsupported JSON value: {type(value).__name__}")


def seed_to_text(value: int | None) -> str | None:
    return None if value is None else str(value)


def validate_seed(value: int, label: str = "seed") -> int:
    """Validate the shared programmatic seed contract without parsing text."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    if not MIN_SEED <= value <= MAX_SEED:
        raise ValueError(
            f"{label} must be between {MIN_SEED} and {MAX_SEED}"
        )
    return value


def parse_seed(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError(f"{label} seed must be a decimal string")
    if isinstance(value, str):
        if (
            not value
            or len(value) > 20
            or not value.isascii()
            or not value.isdecimal()
            or (len(value) > 1 and value.startswith("0"))
        ):
            raise ValueError(f"{label} seed must be a canonical decimal string")
        parsed = int(value)
    else:
        parsed = value
    return validate_seed(parsed, f"{label} seed")


def parse_optional_seed(value: object, label: str) -> int | None:
    return None if value is None else parse_seed(value, label)


__all__ = [
    "AGENT_PROFILES",
    "AgentDescriptor",
    "AgentSpec",
    "FrozenJSONValue",
    "JSONValue",
    "MAX_SEED",
    "MIN_SEED",
    "SEED_DERIVATION_VERSION",
    "SeedBundle",
    "derive_seed",
    "derive_seed_bundle",
    "freeze_parameters",
    "json_value",
    "normalize_parameters",
    "parse_optional_seed",
    "parse_seed",
    "seed_to_text",
    "thaw_parameters",
    "validate_seed",
]
