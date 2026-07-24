from __future__ import annotations

import pytest

from fugitive.agents.registry import (
    DEFAULT_FUGITIVE_AGENT,
    DEFAULT_MARSHAL_AGENT,
    FUGITIVE_AGENT_REGISTRY,
    MARSHAL_AGENT_REGISTRY,
    INTERACTIVE_BIR2_MAX_GUESS_CANDIDATES,
    INTERACTIVE_BIR2_PARTICLE_COUNT,
    INTERACTIVE_BIR_MAX_GUESS_CANDIDATES,
    INTERACTIVE_BIR_PARTICLE_COUNT,
    INTERACTIVE_MCMC_BIR_MAX_GUESS_CANDIDATES,
    INTERACTIVE_MCMC_BIR_PARTICLE_COUNT,
    INTERACTIVE_MCMC_BIR_STEPS_PER_CHAIN,
)
from fugitive.engine import GameEngine
from fugitive.model import FugitiveAction, Role
from fugitive.reproducibility import AgentSpec


def test_typed_registries_contain_only_role_appropriate_agents() -> None:
    common = {"hierarchical-random", "belief-informed-random"}
    marshal_only = {
        "route-count-random",
        "constructive-belief-informed-random",
        "mcmc-belief-informed-random",
    }
    assert set(FUGITIVE_AGENT_REGISTRY) == common
    assert set(MARSHAL_AGENT_REGISTRY) == common | marshal_only

    for name, registration in FUGITIVE_AGENT_REGISTRY.items():
        assert registration.name == name
        assert registration.role is Role.FUGITIVE
        assert not registration.expensive

    for name, registration in MARSHAL_AGENT_REGISTRY.items():
        assert registration.name == name
        assert registration.role is Role.MARSHAL
        assert registration.expensive is (name in {
            "belief-informed-random",
            "constructive-belief-informed-random",
            "mcmc-belief-informed-random",
        })


def test_new_random_baselines_are_role_specific_and_default_to_bir() -> None:
    common = {"hierarchical-random", "belief-informed-random"}

    assert common == set(FUGITIVE_AGENT_REGISTRY)
    assert common < set(MARSHAL_AGENT_REGISTRY)
    assert DEFAULT_FUGITIVE_AGENT == "belief-informed-random"
    assert DEFAULT_MARSHAL_AGENT == "belief-informed-random"

    for name in common:
        fugitive = FUGITIVE_AGENT_REGISTRY[name].create(7)
        marshal = MARSHAL_AGENT_REGISTRY[name].create(7)
        assert hasattr(fugitive, "choose_fugitive_action")
        assert not hasattr(fugitive, "choose_guess")
        assert hasattr(marshal, "choose_guess")
        assert not hasattr(marshal, "choose_fugitive_action")

    for name in set(MARSHAL_AGENT_REGISTRY) - common:
        marshal = MARSHAL_AGENT_REGISTRY[name].create(7)
        assert hasattr(marshal, "choose_guess")
        assert name not in FUGITIVE_AGENT_REGISTRY


def test_build_returns_complete_round_trippable_resolved_agent_specs() -> None:
    default = MARSHAL_AGENT_REGISTRY["belief-informed-random"].build(
        7,
        profile="default",
    )
    interactive = MARSHAL_AGENT_REGISTRY["belief-informed-random"].build(
        7,
        profile="interactive",
    )

    assert AgentSpec.from_dict(default.spec.to_dict()) == default.spec
    assert AgentSpec.from_dict(interactive.spec.to_dict()) == interactive.spec
    assert default.spec.profile == "default"
    assert default.spec.parameters["particle_count"] == 2_000
    assert default.spec.parameters["max_guess_candidates"] == 128
    assert interactive.spec.profile == "interactive"
    assert (
        interactive.spec.parameters["particle_count"]
        == INTERACTIVE_BIR_PARTICLE_COUNT
    )
    assert (
        interactive.spec.parameters["max_guess_candidates"]
        == INTERACTIVE_BIR_MAX_GUESS_CANDIDATES
    )
    assert isinstance(interactive.spec.parameters["min_unique_particles"], int)


def test_build_applies_overrides_after_profile_and_rejects_seed_override() -> None:
    registration = MARSHAL_AGENT_REGISTRY["belief-informed-random"]
    built = registration.build(
        11,
        profile="interactive",
        overrides={"particle_count": 96, "max_guess_candidates": 24},
    )

    assert built.spec.parameters["particle_count"] == 96
    assert built.spec.parameters["max_guess_candidates"] == 24

    with pytest.raises(ValueError, match="must not override"):
        registration.build(11, overrides={"seed": 12})
    with pytest.raises(ValueError, match="unknown parameters"):
        registration.build(11, overrides={"not_a_parameter": 1})


@pytest.mark.parametrize(
    ("name", "parameter", "value"),
    (
        ("constructive-belief-informed-random", "sampling_max_nodes", None),
        ("constructive-belief-informed-random", "sampling_max_proposals", 1),
        ("constructive-belief-informed-random", "sprint_backend", "exact"),
        ("constructive-belief-informed-random", "sampler", None),
        ("mcmc-belief-informed-random", "sampling_max_nodes", None),
        ("mcmc-belief-informed-random", "sampling_max_proposals", 1),
        ("mcmc-belief-informed-random", "sprint_backend", "exact"),
        ("mcmc-belief-informed-random", "_sampler", None),
    ),
)
def test_engineering_controls_are_not_registered_policy_parameters(
    name: str,
    parameter: str,
    value: object,
) -> None:
    with pytest.raises(ValueError, match="unknown parameters"):
        MARSHAL_AGENT_REGISTRY[name].build(
            13,
            overrides={parameter: value},
        )


def test_bir2_interactive_profile_is_explicit_and_round_trippable() -> None:
    built = MARSHAL_AGENT_REGISTRY[
        "constructive-belief-informed-random"
    ].build(17, profile="interactive")

    assert AgentSpec.from_dict(built.spec.to_dict()) == built.spec
    assert built.spec.parameters["particle_count"] == INTERACTIVE_BIR2_PARTICLE_COUNT
    assert (
        built.spec.parameters["max_guess_candidates"]
        == INTERACTIVE_BIR2_MAX_GUESS_CANDIDATES
    )
    assert "sampling_max_proposals" not in built.spec.parameters
    assert "sprint_backend" not in built.spec.parameters
    assert "sampler" not in built.spec.parameters
    assert "_sampler" not in built.spec.parameters


def test_bir2_serialized_spec_rebuilds_identical_beliefs_and_distributions() -> None:
    registration = MARSHAL_AGENT_REGISTRY[
        "constructive-belief-informed-random"
    ]
    overrides = {
        "particle_count": 16,
        "max_guess_candidates": 16,
    }
    original = registration.build(73, profile="default", overrides=overrides)
    restored_spec = AgentSpec.from_dict(original.spec.to_dict())
    rebuilt = registration.build(
        73,
        profile=restored_spec.profile,
        overrides=restored_spec.parameters,
    )
    assert rebuilt.spec == restored_spec == original.spec

    engine = GameEngine(seed=19)
    engine.apply_fugitive_action(FugitiveAction(1))
    engine.apply_fugitive_action(FugitiveAction(2))
    draw_observation = engine.observation(Role.MARSHAL)
    original_draw = original.agent.draw_pile_distribution(draw_observation)
    rebuilt_draw = rebuilt.agent.draw_pile_distribution(draw_observation)
    assert original.agent.belief(draw_observation).summary() == rebuilt.agent.belief(
        draw_observation
    ).summary()
    assert original_draw == rebuilt_draw

    engine.draw(draw_observation.legal_draw_piles[0])
    second_draw = engine.observation(Role.MARSHAL)
    engine.draw(second_draw.legal_draw_piles[0])
    guess_observation = engine.observation(Role.MARSHAL)
    original_guess = original.agent.guess_distribution(guess_observation)
    rebuilt_guess = rebuilt.agent.guess_distribution(guess_observation)
    assert original.agent.belief(guess_observation).summary() == rebuilt.agent.belief(
        guess_observation
    ).summary()
    assert original_guess == rebuilt_guess


def test_bir3_interactive_profile_is_distinct_and_round_trippable() -> None:
    built = MARSHAL_AGENT_REGISTRY["mcmc-belief-informed-random"].build(
        29,
        profile="interactive",
    )

    assert AgentSpec.from_dict(built.spec.to_dict()) == built.spec
    assert built.spec.name == "mcmc-belief-informed-random"
    assert built.spec.parameters["particle_count"] == (
        INTERACTIVE_MCMC_BIR_PARTICLE_COUNT
    )
    assert built.spec.parameters["max_guess_candidates"] == (
        INTERACTIVE_MCMC_BIR_MAX_GUESS_CANDIDATES
    )
    assert "sampling_max_proposals" not in built.spec.parameters
    assert "sprint_backend" not in built.spec.parameters
    assert "sampler" not in built.spec.parameters
    assert "_sampler" not in built.spec.parameters
    assert built.spec.parameters["mh_steps_per_chain"] == (
        INTERACTIVE_MCMC_BIR_STEPS_PER_CHAIN
    )


def test_bir3_default_spec_records_uncensored_mcmc_inference() -> None:
    built = MARSHAL_AGENT_REGISTRY["mcmc-belief-informed-random"].build(31)

    assert built.spec.parameters["particle_count"] == 1_000
    assert "sampling_max_nodes" not in built.spec.parameters
    assert built.spec.parameters["mh_steps_per_chain"] == 1
