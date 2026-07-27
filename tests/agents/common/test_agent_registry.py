from __future__ import annotations

import pytest

from fugitive.agents.registry import (
    AgentRegistration,
    DEFAULT_FUGITIVE_AGENT,
    DEFAULT_MARSHAL_AGENT,
    FUGITIVE_AGENT_REGISTRY,
    MARSHAL_AGENT_REGISTRY,
)
from fugitive.game.model import Role
from fugitive.shared.reproducibility import AgentSpec


def test_registrations_publish_explicit_schemas() -> None:
    registrations = (
        *FUGITIVE_AGENT_REGISTRY.values(),
        *MARSHAL_AGENT_REGISTRY.values(),
    )
    assert registrations
    for registration in registrations:
        user_names = set(registration.user_parameter_defaults)
        fixed_names = set(registration.fixed_algorithm_metadata)
        assert not user_names & {"seed", "rng"}
        assert not any(name.startswith("_") for name in user_names)
        assert not user_names & fixed_names
        assert all(
            set(overrides) <= user_names
            for overrides in registration.profile_overrides.values()
        )

    engineering_controls = {
        "sampling_max_nodes",
        "sampling_max_proposals",
        "route_sampler_factory",
        "sampler",
        "_sampler",
    }
    for name in (
        "constructive-belief-informed-random",
        "mcmc-belief-informed-random",
    ):
        assert engineering_controls.isdisjoint(
            MARSHAL_AGENT_REGISTRY[name].user_parameter_defaults
        )


def test_every_registration_matches_its_native_constructor() -> None:
    registrations = (
        *FUGITIVE_AGENT_REGISTRY.values(),
        *MARSHAL_AGENT_REGISTRY.values(),
    )
    for registration in registrations:
        native = registration.factory(41)
        registered = registration.build(41)
        for parameter in registration.user_parameter_defaults:
            assert getattr(registered.agent, parameter) == getattr(native, parameter)
        for parameter, expected in registration.fixed_algorithm_metadata.items():
            assert getattr(native, parameter) == expected


def test_registration_builds_resolved_roundtrippable_specs() -> None:
    class TeachingAgent:
        def __init__(
            self,
            _seed: int,
            *,
            sample_count: int | None = None,
            engineering_switch: bool = False,
        ) -> None:
            self.sample_count = 12 if sample_count is None else sample_count
            self.engineering_switch = engineering_switch
            self.algorithm_id = "teaching-agent-v1"

    registration = AgentRegistration(
        "teaching-agent",
        Role.MARSHAL,
        TeachingAgent,
        user_parameter_defaults={"sample_count": None},
        profile_overrides={"interactive": {"sample_count": 4}},
        fixed_algorithm_metadata={"algorithm_id": "teaching-agent-v1"},
    )

    default = registration.build(3)
    interactive = registration.build(3, profile="interactive")
    assert default.spec.parameters == {
        "algorithm_id": "teaching-agent-v1",
        "sample_count": 12,
    }
    assert interactive.spec.parameters["sample_count"] == 4
    assert AgentSpec.from_dict(default.spec.to_dict()) == default.spec
    rebuilt = registration.build(3, overrides=default.spec.parameters)
    assert rebuilt.spec == default.spec

    with pytest.raises(ValueError, match="unknown parameters"):
        registration.build(3, overrides={"engineering_switch": True})


def test_typed_registries_contain_only_role_appropriate_agents() -> None:
    common = {"hierarchical-random", "belief-informed-random"}
    fugitive_only = {"continuation-count", "belief-rollout"}
    marshal_only = {
        "route-count-random",
        "support-catalogue-random",
        "route-count-catalogue-random",
        "constructive-belief-informed-random",
        "unweighted-constructive-belief-informed-random",
        "rollout-bir2u",
        "exact-sprint-belief-informed-random",
        "mcmc-belief-informed-random",
    }
    assert set(FUGITIVE_AGENT_REGISTRY) == common | fugitive_only
    assert set(MARSHAL_AGENT_REGISTRY) == common | marshal_only
    assert DEFAULT_FUGITIVE_AGENT == "belief-informed-random"
    assert DEFAULT_MARSHAL_AGENT == "belief-informed-random"

    for name, registration in FUGITIVE_AGENT_REGISTRY.items():
        assert registration.name == name
        assert registration.role is Role.FUGITIVE
    for name, registration in MARSHAL_AGENT_REGISTRY.items():
        assert registration.name == name
        assert registration.role is Role.MARSHAL
