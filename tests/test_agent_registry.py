from __future__ import annotations

import pytest

import fugitive.agents.registry as registry_module
from fugitive.agents.registry import (
    AgentRegistration,
    DEFAULT_FUGITIVE_AGENT,
    DEFAULT_MARSHAL_AGENT,
    FUGITIVE_AGENT_REGISTRY,
    MARSHAL_AGENT_REGISTRY,
    INTERACTIVE_BIR2E_MAX_GUESS_CANDIDATES,
    INTERACTIVE_BIR2E_PARTICLE_COUNT,
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
from fugitive.reproducibility import MAX_SEED, AgentSpec


def test_registrations_publish_explicit_user_and_algorithm_schemas() -> None:
    assert "inspect" not in vars(registry_module)

    registrations = (
        *FUGITIVE_AGENT_REGISTRY.values(),
        *MARSHAL_AGENT_REGISTRY.values(),
    )
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

    assert FUGITIVE_AGENT_REGISTRY[
        "hierarchical-random"
    ].fixed_algorithm_metadata["algorithm_id"] == (
        "hr-1-fugitive-hierarchical-legal-random-v1"
    )
    assert MARSHAL_AGENT_REGISTRY[
        "hierarchical-random"
    ].fixed_algorithm_metadata["algorithm_id"] == (
        "hr-1-marshal-hard-support-random-v2"
    )
    assert MARSHAL_AGENT_REGISTRY[
        "route-count-random"
    ].fixed_algorithm_metadata["algorithm_id"] == (
        "hr-1.1-marshal-route-count-random-v1"
    )

    bir1 = MARSHAL_AGENT_REGISTRY["belief-informed-random"]
    assert "min_unique_particles" not in bir1.user_parameter_defaults
    assert bir1.fixed_algorithm_metadata["algorithm_id"] == (
        "bir-1-constructive-bootstrap-pf-marshal-v1"
    )
    assert bir1.fixed_algorithm_metadata["belief_backend_id"] == (
        "constructive-bootstrap-particle-filter-v1"
    )
    assert (
        bir1.fixed_algorithm_metadata[
            "max_incremental_play_proposals_per_particle"
        ]
        == 64
    )
    assert (
        bir1.fixed_algorithm_metadata[
            "exact_incremental_play_combination_threshold"
        ]
        == 4_096
    )
    assert bir1.fixed_algorithm_metadata["resample_ess_fraction"] == 0.5
    assert bir1.fixed_algorithm_metadata["sprint_backend"] == "sequential"
    assert bir1.fixed_algorithm_metadata["belief_seed_domain"] == (
        "fugitive.marshal.sequential-constructive-belief.v1"
    )
    assert bir1.fixed_algorithm_metadata["weighting_id"] == (
        "uniform-accepted-proposals-v1"
    )
    assert bir1.fixed_algorithm_metadata["public_event_model_id"] == (
        "constraint-feasibility-only-v1"
    )
    mcmc = MARSHAL_AGENT_REGISTRY["mcmc-belief-informed-random"]
    assert mcmc.user_parameter_defaults["mh_steps_per_chain"] == 1
    assert mcmc.fixed_algorithm_metadata["mcmc_kernel_id"] == (
        "independent-metropolis-hastings-v1"
    )
    assert mcmc.fixed_algorithm_metadata["initial_seed_domain"] == (
        "fugitive.bir3.initial-constructive-belief.v1"
    )
    assert mcmc.fixed_algorithm_metadata["rejuvenation_seed_domain"] == (
        "fugitive.bir3.independent-mh.v1"
    )


def test_every_registration_default_table_matches_native_constructor() -> None:
    """Catch drift while keeping production registry schemas explicit."""

    registrations = (
        *FUGITIVE_AGENT_REGISTRY.values(),
        *MARSHAL_AGENT_REGISTRY.values(),
    )
    for registration in registrations:
        native = registration.factory(41)
        registered = registration.build(41)
        for parameter in registration.user_parameter_defaults:
            assert getattr(registered.agent, parameter) == getattr(native, parameter), (
                registration.name,
                parameter,
            )
        for parameter, expected in registration.fixed_algorithm_metadata.items():
            assert getattr(native, parameter) == expected, (
                registration.name,
                parameter,
            )


def test_registry_delegates_seed_range_validation() -> None:
    registration = FUGITIVE_AGENT_REGISTRY["hierarchical-random"]
    with pytest.raises(ValueError, match="between"):
        registration.build(-1)
    with pytest.raises(ValueError, match="between"):
        registration.create(-1)


def test_registry_accepts_uint64_upper_boundary() -> None:
    registration = FUGITIVE_AGENT_REGISTRY["hierarchical-random"]
    assert registration.build(MAX_SEED).agent is not None


def test_registry_delegates_seed_type_validation() -> None:
    registration = FUGITIVE_AGENT_REGISTRY["hierarchical-random"]
    with pytest.raises(TypeError, match="integer"):
        registration.build(True)


def test_explicit_schema_controls_input_and_records_resolved_sentinels() -> None:
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

    rebuilt = registration.build(3, overrides=default.spec.parameters)
    assert rebuilt.spec == default.spec
    with pytest.raises(ValueError, match="unknown parameters"):
        registration.build(3, overrides={"engineering_switch": True})


def test_registration_rejects_ambiguous_schema_definitions() -> None:
    def factory(_seed: int, **_parameters: object) -> object:
        return object()

    with pytest.raises(ValueError, match="reserved names"):
        AgentRegistration(
            "reserved",
            Role.MARSHAL,
            factory,
            user_parameter_defaults={"rng": None},
        )
    with pytest.raises(ValueError, match="overlaps user parameters"):
        AgentRegistration(
            "collision",
            Role.MARSHAL,
            factory,
            user_parameter_defaults={"algorithm_id": "user-choice"},
            fixed_algorithm_metadata={"algorithm_id": "fixed-choice"},
        )
    with pytest.raises(ValueError, match="invalid interactive profile"):
        AgentRegistration(
            "bad-profile",
            Role.MARSHAL,
            factory,
            user_parameter_defaults={"particle_count": 8},
            profile_overrides={"interactive": {"unknown": 4}},
        )


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

    for name, registration in FUGITIVE_AGENT_REGISTRY.items():
        assert registration.name == name
        assert registration.role is Role.FUGITIVE
        assert registration.expensive is (name in fugitive_only)

    for name, registration in MARSHAL_AGENT_REGISTRY.items():
        assert registration.name == name
        assert registration.role is Role.MARSHAL
        assert registration.expensive is (name in {
            "belief-informed-random",
            "constructive-belief-informed-random",
            "unweighted-constructive-belief-informed-random",
            "rollout-bir2u",
            "exact-sprint-belief-informed-random",
            "mcmc-belief-informed-random",
        })


def test_controlled_catalogue_registry_has_no_noop_parameters() -> None:
    support = MARSHAL_AGENT_REGISTRY["support-catalogue-random"]
    route_count = MARSHAL_AGENT_REGISTRY["route-count-catalogue-random"]

    assert "epsilon" not in support.user_parameter_defaults
    assert "epsilon" in route_count.user_parameter_defaults


def test_new_random_baselines_are_role_specific_and_default_to_bir() -> None:
    common = {"hierarchical-random", "belief-informed-random"}

    assert common < set(FUGITIVE_AGENT_REGISTRY)
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

    for name in set(FUGITIVE_AGENT_REGISTRY) - common:
        fugitive = FUGITIVE_AGENT_REGISTRY[name].create(7)
        assert hasattr(fugitive, "choose_fugitive_action")
        assert name not in MARSHAL_AGENT_REGISTRY


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
    assert interactive.spec.parameters["algorithm_id"] == (
        "bir-1-constructive-bootstrap-pf-marshal-v1"
    )
    assert interactive.spec.parameters["belief_backend_id"] == (
        "constructive-bootstrap-particle-filter-v1"
    )
    assert "min_unique_particles" not in interactive.spec.parameters


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
        ("constructive-belief-informed-random", "route_sampler_factory", None),
        ("constructive-belief-informed-random", "sampler", None),
        ("mcmc-belief-informed-random", "sampling_max_nodes", None),
        ("mcmc-belief-informed-random", "sampling_max_proposals", 1),
        ("mcmc-belief-informed-random", "route_sampler_factory", None),
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
    assert built.spec.parameters["algorithm_id"] == "bir-2s-constructive-snis-v1"
    assert built.spec.parameters["sprint_backend"] == "sequential"
    assert built.spec.parameters["belief_seed_domain"] == (
        "fugitive.marshal.sequential-constructive-belief.v1"
    )
    assert built.spec.parameters["reference_target_id"] == (
        "constraint-uniform-complete-worlds-v1"
    )
    assert "sprint=sprint-category-sequential-importance.v1" in (
        built.spec.parameters["proposal_kernel_id"]
    )
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


def test_bir2e_spec_records_fixed_exact_backend_and_round_trips() -> None:
    registration = MARSHAL_AGENT_REGISTRY[
        "exact-sprint-belief-informed-random"
    ]
    built = registration.build(23, profile="interactive")

    assert AgentSpec.from_dict(built.spec.to_dict()) == built.spec
    assert built.spec.parameters["algorithm_id"] == (
        "bir-2e-exact-sprint-dp-v1"
    )
    assert built.spec.parameters["sprint_backend"] == "exact"
    assert built.spec.parameters["belief_seed_domain"] == (
        "fugitive.bir2e.exact-sprint-belief.v1"
    )
    assert built.spec.parameters["reference_target_id"] == (
        "constraint-uniform-complete-worlds-v1"
    )
    assert "sprint=sprint-category-descendant-count.v1" in (
        built.spec.parameters["proposal_kernel_id"]
    )
    assert built.spec.parameters["particle_count"] == (
        INTERACTIVE_BIR2E_PARTICLE_COUNT
    )
    assert built.spec.parameters["max_guess_candidates"] == (
        INTERACTIVE_BIR2E_MAX_GUESS_CANDIDATES
    )

    rebuilt = registration.build(
        23,
        profile=built.spec.profile,
        overrides=built.spec.parameters,
    )
    assert rebuilt.spec == built.spec
    assert rebuilt.agent.sprint_backend == "exact"



@pytest.mark.parametrize(
    ("parameter", "invalid_value"),
    (
        ("algorithm_id", "another-algorithm"),
        ("proposal_kernel_id", "another-proposal"),
        ("sprint_backend", "sequential"),
        ("reference_target_id", "another-target"),
    ),
)
def test_bir2e_algorithm_definition_cannot_be_overridden(
    parameter: str,
    invalid_value: str,
) -> None:
    registration = MARSHAL_AGENT_REGISTRY[
        "exact-sprint-belief-informed-random"
    ]

    with pytest.raises(ValueError, match="is fixed"):
        registration.build(23, overrides={parameter: invalid_value})


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
    assert built.spec.parameters["algorithm_id"] == (
        "bir-3-sir-independent-mh-v1"
    )
    assert built.spec.parameters["mcmc_kernel_id"] == (
        "independent-metropolis-hastings-v1"
    )
    assert built.spec.parameters["initial_weighting_id"] == (
        "self-normalized-importance-v1"
    )
    assert built.agent.initial_weighting_id == (
        built.spec.parameters["initial_weighting_id"]
    )
    assert built.spec.parameters["sprint_backend"] == "sequential"
    assert built.spec.parameters["initial_seed_domain"] == (
        "fugitive.bir3.initial-constructive-belief.v1"
    )
    assert built.spec.parameters["rejuvenation_seed_domain"] == (
        "fugitive.bir3.independent-mh.v1"
    )
    assert built.spec.parameters["reference_target_id"] == (
        "constraint-uniform-complete-worlds-v1"
    )
    assert "sprint=sprint-category-sequential-importance.v1" in (
        built.spec.parameters["proposal_kernel_id"]
    )
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
