from __future__ import annotations

from collections.abc import Mapping

import pytest

from fugitive.game.model import Role
from fugitive.shared.reproducibility import (
    MAX_SEED,
    AgentDescriptor,
    AgentSpec,
    SeedBundle,
    derive_seed_bundle,
    parse_seed,
    validate_seed,
)


def test_agent_parameters_are_deeply_immutable_but_serialize_as_json() -> None:
    source = {"nested": {"values": [1, 2]}}
    spec = AgentSpec("example", Role.MARSHAL, "default", source)
    source["nested"]["values"].append(3)  # type: ignore[index,union-attr]

    nested = spec.parameters["nested"]
    assert isinstance(nested, Mapping)
    assert nested["values"] == (1, 2)
    with pytest.raises(TypeError):
        spec.parameters["new"] = 1  # type: ignore[index]
    with pytest.raises(TypeError):
        nested["new"] = 1  # type: ignore[index]

    serialized = spec.to_dict()
    assert serialized["parameters"] == {"nested": {"values": [1, 2]}}
    serialized["parameters"]["nested"]["values"].append(4)  # type: ignore[index,union-attr]
    assert nested["values"] == (1, 2)
    assert AgentSpec.from_dict(spec.to_dict()) == spec

    descriptor = AgentDescriptor.create("example", source)
    assert descriptor.to_dict()["parameters"] == {
        "nested": {"values": [1, 2, 3]}
    }


@pytest.mark.parametrize("value", (-1, MAX_SEED + 1))
def test_every_seed_entry_point_rejects_values_outside_uint64(value: int) -> None:
    with pytest.raises(ValueError, match="between"):
        parse_seed(value, "master")
    with pytest.raises(ValueError, match="between"):
        derive_seed_bundle(value)
    with pytest.raises(ValueError, match="between"):
        SeedBundle(value, 0, 0, 0)


@pytest.mark.parametrize("value", ("", "00", "+1", "-1", " 1", "1 "))
def test_shared_seed_parser_requires_canonical_decimal_text(value: str) -> None:
    with pytest.raises(ValueError, match="canonical"):
        parse_seed(value, "master")


def test_shared_seed_parser_accepts_the_complete_uint64_boundary() -> None:
    assert parse_seed(str(MAX_SEED), "master") == MAX_SEED
    assert parse_seed(MAX_SEED, "master") == MAX_SEED


@pytest.mark.parametrize("value", (0, MAX_SEED))
def test_programmatic_seed_validator_accepts_uint64_boundaries(value: int) -> None:
    assert validate_seed(value) == value


@pytest.mark.parametrize("value", (True, 1.0, "1", None))
def test_programmatic_seed_validator_accepts_only_integers(value: object) -> None:
    with pytest.raises(TypeError, match="integer"):
        validate_seed(value)  # type: ignore[arg-type]
