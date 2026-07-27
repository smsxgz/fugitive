from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from fugitive.experiment import RULES_SHA256, RULES_VERSION, ExperimentStatus
from fugitive.model import GameResult, Role, Winner
from fugitive.payoff_ledger import PayoffLedger
from fugitive.psro import (
    EmpiricalPayoffMatrix,
    MetaSolverConfig,
    MultiplicativeWeightsMetaSolver,
    PayoffEstimate,
    PayoffEvaluation,
    PayoffPair,
    PayoffRequest,
    PolicyPopulation,
    PolicySpec,
    ResponseOracleRequest,
    evaluate_missing_payoffs,
    initialize_psro,
    run_psro,
)
from fugitive.psro_experiment import (
    PSROExperimentCheckpoint,
    PSROExperimentRunConfig,
    PSROExperimentRunResult,
    finalize_registered_psro,
    initialize_registered_psro,
    run_registered_psro,
    run_registered_psro_iteration,
)
from fugitive.psro_payoff import (
    PSRO_RESPONSE_VALIDATION_SEED_DOMAIN,
    RegisteredGamePayoffConfig,
    RegisteredGamePayoffEvaluator,
    ResponseValidationReport,
)
from fugitive.psro_policy_adapter import (
    MixtureConditionedResponseOracle,
    MixtureResponseTemplate,
    agent_spec_from_policy,
    planning_leaf_spec_from_policy,
    registered_policy_spec,
    resolve_registered_policy,
)
from fugitive.reproducibility import AgentSpec, derive_seed, thaw_parameters


def wrapped(role: Role, name: str):
    return registered_policy_spec(
        AgentSpec(name, role, "default", {"variant": name}),
        identifier_name=f"policy-{name}",
    )


class StaticEvaluator:
    def __init__(self, values: dict[tuple[str, str], float]) -> None:
        self.values = values

    def evaluate(self, requests):
        return tuple(
            PayoffEvaluation(
                request.pair,
                PayoffEstimate(
                    self.values[
                        (
                            request.pair.marshal.name,
                            request.pair.fugitive.name,
                        )
                    ]
                ),
            )
            for request in requests
        )


class NoneOracle:
    def propose_response(self, request):
        return None


class ConstantEvaluator:
    def __init__(self, value: float = 0.5) -> None:
        self.value = value

    def evaluate(self, requests):
        return tuple(
            PayoffEvaluation(request.pair, PayoffEstimate(self.value))
            for request in requests
        )


class AlwaysAcceptEvaluator(ConstantEvaluator):
    def validate_response(self, **kwargs):
        opponent = kwargs["opponent_mixture"].support[0]
        return ResponseValidationReport(
            role=kwargs["role"],
            generation=kwargs["generation"],
            candidate=kwargs["candidate"].identifier,
            incumbent=kwargs["incumbent"].identifier,
            opponent_draws=(opponent,),
            paired_gains=(1.0,),
            mean_gain=1.0,
            confidence_lower_bound=1.0,
            confidence_upper_bound=1.0,
            confidence_level=0.95,
            bootstrap_replicates=1,
            bootstrap_seed=0,
            minimum_gain=0.0,
            admitted=True,
        )


def experiment_run_config(
    payoff_config: RegisteredGamePayoffConfig,
) -> PSROExperimentRunConfig:
    return PSROExperimentRunConfig(
        payoff_config=payoff_config,
        meta_solver_config=MetaSolverConfig(
            max_iterations=200,
            tolerance=0.1,
            check_interval=1,
        ),
        marshal_response_template=MixtureResponseTemplate(
            Role.MARSHAL,
            "hierarchical-random",
            base_parameters={"max_guess_size": 1},
            opponent_parameter=None,
            identifier_prefix="marshal-response",
        ),
        fugitive_response_template=MixtureResponseTemplate(
            Role.FUGITIVE,
            "hierarchical-random",
            opponent_parameter=None,
            identifier_prefix="fugitive-response",
        ),
    )


def test_registered_policy_round_trip_separates_psro_id_from_agent_spec():
    agent_spec = AgentSpec(
        "hierarchical-random",
        Role.MARSHAL,
        "default",
        {"max_guess_size": 1},
    )

    policy = registered_policy_spec(agent_spec, identifier_name="marshal-seed-0")

    assert policy.identifier.name == "marshal-seed-0"
    assert agent_spec_from_policy(policy) == agent_spec
    assert policy.identifier.name != agent_spec.name


def test_payoff_evaluator_uses_the_same_paired_seeds_for_every_cell(monkeypatch):
    marshals = [wrapped(Role.MARSHAL, name) for name in ("m0", "m1")]
    fugitives = [wrapped(Role.FUGITIVE, name) for name in ("f0", "f1")]
    population = PolicyPopulation.create(marshal=marshals, fugitive=fugitives)
    config = RegisteredGamePayoffConfig((11, 22), validate_invariants=False)
    calls: list[tuple[str, str, int]] = []

    def fake_run_registered_experiment(**kwargs):
        calls.append(
            (
                kwargs["marshal_name"],
                kwargs["fugitive_name"],
                kwargs["master_seed"],
            )
        )
        winner = (
            Winner.MARSHAL
            if kwargs["master_seed"] == 11
            else Winner.FUGITIVE
        )
        return SimpleNamespace(
            status=ExperimentStatus.COMPLETED,
            game_result=GameResult(winner, "test", 1, ()),
            manifest=SimpleNamespace(
                to_dict=lambda: {"status": "completed", "seed": str(kwargs["master_seed"])}
            ),
        )

    monkeypatch.setattr(
        "fugitive.psro_payoff.run_registered_experiment",
        fake_run_registered_experiment,
    )
    collected = evaluate_missing_payoffs(
        population,
        EmpiricalPayoffMatrix(),
        RegisteredGamePayoffEvaluator(config),
    )

    assert len(calls) == 8
    for marshal in ("m0", "m1"):
        for fugitive in ("f0", "f1"):
            assert [
                seed
                for seen_marshal, seen_fugitive, seed in calls
                if seen_marshal == marshal and seen_fugitive == fugitive
            ] == [11, 22]
    assert {
        entry.estimate.marshal_payoff for entry in collected.matrix.entries
    } == {0.5}
    assert {entry.estimate.sample_count for entry in collected.matrix.entries} == {
        2
    }
    assert {
        entry.estimate.standard_error for entry in collected.matrix.entries
    } == {0.5}


def test_seed_schedules_are_domain_separated_and_round_trip() -> None:
    config = RegisteredGamePayoffConfig.derived(
        20260727,
        2,
        response_validation_games=3,
        final_holdout_games=4,
    )

    train = set(config.meta_train_seeds)
    validation = set(config.response_validation_seeds)
    holdout = set(config.final_holdout_seeds)
    assert (len(train), len(validation), len(holdout)) == (2, 3, 4)
    assert train.isdisjoint(validation)
    assert train.isdisjoint(holdout)
    assert validation.isdisjoint(holdout)
    assert RegisteredGamePayoffConfig.from_dict(config.to_dict()) == config


def test_payoff_ledger_reuses_old_games_and_only_appends_new_seeds(
    monkeypatch,
    tmp_path,
) -> None:
    marshal = wrapped(Role.MARSHAL, "m")
    fugitive = wrapped(Role.FUGITIVE, "f")
    pair = PayoffPair(marshal.identifier, fugitive.identifier)
    request = PayoffRequest(pair, marshal, fugitive)
    calls: list[int] = []

    def fake_run_registered_experiment(**kwargs):
        calls.append(kwargs["master_seed"])
        winner = (
            Winner.MARSHAL
            if kwargs["master_seed"] == 11
            else Winner.FUGITIVE
        )
        return SimpleNamespace(
            status=ExperimentStatus.COMPLETED,
            game_result=GameResult(winner, "test", 1, ()),
            manifest=SimpleNamespace(
                to_dict=lambda: {
                    "status": "completed",
                    "seed": str(kwargs["master_seed"]),
                }
            ),
        )

    monkeypatch.setattr(
        "fugitive.psro_payoff.run_registered_experiment",
        fake_run_registered_experiment,
    )
    ledger = PayoffLedger(tmp_path)
    first = RegisteredGamePayoffEvaluator(
        RegisteredGamePayoffConfig((11,), validate_invariants=False),
        ledger=ledger,
    ).evaluate((request,))
    progress = []
    second = RegisteredGamePayoffEvaluator(
        RegisteredGamePayoffConfig((11, 22), validate_invariants=False),
        ledger=ledger,
        progress=progress.append,
    ).evaluate((request,))

    assert first[0].estimate.sample_count == 1
    assert second[0].estimate.sample_count == 2
    assert second[0].estimate.marshal_payoff == 0.5
    assert calls == [11, 22]
    assert len(ledger.samples()) == 2
    assert progress[-1].reused_games == 1


def test_registered_payoff_games_run_in_parallel_and_resume(tmp_path) -> None:
    marshal = resolve_registered_policy(
        Role.MARSHAL,
        "hierarchical-random",
        identifier_name="parallel-marshal",
        overrides={"multi_guess_continuation": 0.0, "max_guess_size": 1},
    )
    fugitive = resolve_registered_policy(
        Role.FUGITIVE,
        "hierarchical-random",
        identifier_name="parallel-fugitive",
    )
    pair = PayoffPair(marshal.identifier, fugitive.identifier)
    request = PayoffRequest(pair, marshal, fugitive)
    config = RegisteredGamePayoffConfig.derived(
        991,
        2,
        response_validation_games=0,
        final_holdout_games=0,
    )
    ledger = PayoffLedger(tmp_path)

    first = RegisteredGamePayoffEvaluator(
        config,
        ledger=ledger,
        workers=2,
    ).evaluate((request,))
    progress = []
    resumed = RegisteredGamePayoffEvaluator(
        config,
        ledger=ledger,
        workers=2,
        progress=progress.append,
    ).evaluate((request,))

    assert first == resumed
    assert first[0].estimate.sample_count == 2
    assert len(ledger.samples()) == 2
    assert progress[-1].completed_games == 2
    assert progress[-1].reused_games == 2


def test_mixture_response_adapter_serializes_opponents_and_unique_generation():
    marshals = [wrapped(Role.MARSHAL, name) for name in ("m0", "m1")]
    fugitives = [wrapped(Role.FUGITIVE, name) for name in ("f0", "f1")]
    population = PolicyPopulation.create(marshal=marshals, fugitive=fugitives)
    values = {
        ("policy-m0", "policy-f0"): 1.0,
        ("policy-m0", "policy-f1"): -1.0,
        ("policy-m1", "policy-f0"): -1.0,
        ("policy-m1", "policy-f1"): 1.0,
    }
    checkpoint = initialize_psro(
        population,
        evaluator=StaticEvaluator(values),
        meta_solver=MultiplicativeWeightsMetaSolver(),
    )
    request = ResponseOracleRequest(
        Role.MARSHAL,
        1,
        checkpoint.population,
        checkpoint.payoff_matrix,
        checkpoint.meta_solution,
    )
    template = MixtureResponseTemplate(
        Role.MARSHAL,
        "rollout-bir2u",
        base_parameters={
            "particle_count": 8,
            "max_guess_candidates": 8,
            "rollout_candidate_count": 1,
            "max_terminal_simulations": 1,
        },
        identifier_prefix="psro-rollout-marshal",
    )
    adapter = MixtureConditionedResponseOracle(template)

    first = adapter.propose_response(request)
    repeated = adapter.propose_response(request)
    next_generation = adapter.propose_response(replace(request, generation=2))
    concrete = agent_spec_from_policy(first, population)
    parameters = thaw_parameters(concrete.parameters)
    opponents = parameters["opponent_policies"]

    assert first == repeated
    assert first.identifier.name.startswith("psro-rollout-marshal-g1-")
    assert next_generation.identifier != first.identifier
    assert concrete.name == "rollout-bir2u"
    assert [choice["weight"] for choice in opponents] == [0.5, 0.5]
    assert [choice["spec"]["name"] for choice in opponents] == ["f0", "f1"]

    from fugitive.agents.registry import MARSHAL_AGENT_REGISTRY

    rebuilt = MARSHAL_AGENT_REGISTRY[concrete.name].build(
        0,
        profile=concrete.profile,
        overrides=thaw_parameters(concrete.parameters),
    ).spec
    restored = PolicySpec.from_dict(first.to_dict())
    assert rebuilt == concrete
    assert agent_spec_from_policy(restored, population) == concrete
    assert MixtureResponseTemplate.from_dict(template.to_dict()) == template


def test_registered_belief_rollout_fugitive_template_round_trips_completely():
    marshal = wrapped(Role.MARSHAL, "m0")
    fugitive = wrapped(Role.FUGITIVE, "f0")
    checkpoint = initialize_psro(
        PolicyPopulation.create(marshal=[marshal], fugitive=[fugitive]),
        evaluator=StaticEvaluator({("policy-m0", "policy-f0"): 0.25}),
    )
    request = ResponseOracleRequest(
        Role.FUGITIVE,
        1,
        checkpoint.population,
        checkpoint.payoff_matrix,
        checkpoint.meta_solution,
    )
    template = MixtureResponseTemplate(
        Role.FUGITIVE,
        "belief-rollout",
        base_parameters={
            "rollout_candidate_count": 2,
            "max_terminal_simulations": 2,
        },
    )
    adapter = MixtureConditionedResponseOracle(template)

    response = adapter.propose_response(request)
    concrete = agent_spec_from_policy(response, checkpoint.population)
    opponents = thaw_parameters(concrete.parameters)["opponent_policies"]

    assert response.role is Role.FUGITIVE
    assert concrete.name == "belief-rollout"
    assert opponents[0]["spec"]["role"] == Role.MARSHAL.value
    assert opponents[0]["weight"] == 1.0

    from fugitive.agents.registry import FUGITIVE_AGENT_REGISTRY

    rebuilt = FUGITIVE_AGENT_REGISTRY[concrete.name].build(
        0,
        profile=concrete.profile,
        overrides=thaw_parameters(concrete.parameters),
    ).spec
    restored = PolicySpec.from_dict(response.to_dict())
    assert rebuilt == concrete
    assert agent_spec_from_policy(restored, checkpoint.population) == concrete
    assert MixtureResponseTemplate.from_dict(template.to_dict()) == template


def test_search_responses_store_ids_but_execute_non_search_opponent_leaves():
    fugitive_search = resolve_registered_policy(
        Role.FUGITIVE,
        "belief-rollout",
        identifier_name="fugitive-search",
        overrides={
            "rollout_candidate_count": 1,
            "max_terminal_simulations": 1,
        },
    )
    marshal = resolve_registered_policy(
        Role.MARSHAL,
        "hierarchical-random",
        identifier_name="marshal-base",
    )
    checkpoint = initialize_psro(
        PolicyPopulation.create(
            marshal=[marshal],
            fugitive=[fugitive_search],
        ),
        evaluator=StaticEvaluator({("marshal-base", "fugitive-search"): 0.5}),
    )
    response = MixtureConditionedResponseOracle(
        MixtureResponseTemplate(
            Role.MARSHAL,
            "rollout-bir2u",
            base_parameters={
                "particle_count": 1,
                "max_guess_candidates": 1,
                "rollout_candidate_count": 1,
                "max_terminal_simulations": 1,
            },
        )
    ).propose_response(
        ResponseOracleRequest(
            Role.MARSHAL,
            1,
            checkpoint.population,
            checkpoint.payoff_matrix,
            checkpoint.meta_solution,
        )
    )

    stored = thaw_parameters(response.parameters)["agent_spec"]
    assert "opponent_policies" not in stored["parameters"]
    execution = agent_spec_from_policy(response, checkpoint.population)
    execution_parameters = thaw_parameters(execution.parameters)
    opponent_specs = execution_parameters["opponent_policies"]
    assert [choice["spec"]["name"] for choice in opponent_specs] == [
        "continuation-count"
    ]
    assert "opponent_policies" not in opponent_specs[0]["spec"]["parameters"]

    response_leaf = planning_leaf_spec_from_policy(response)
    assert response_leaf.name == "unweighted-constructive-belief-informed-random"
    assert "opponent_policies" not in response_leaf.parameters

    metadata = thaw_parameters(response.parameters)["metadata"]
    references = metadata["opponent_leaf_mixture"][0]["policy_ids"]
    assert references == [fugitive_search.identifier.to_dict()]
    assert metadata["planning_depth"] == 1


def test_response_template_coalesces_duplicate_underlying_opponent_specs():
    marshal = wrapped(Role.MARSHAL, "m0")
    underlying = AgentSpec("same-f", Role.FUGITIVE, "default", {})
    fugitive_a = registered_policy_spec(underlying, identifier_name="f-a")
    fugitive_b = registered_policy_spec(underlying, identifier_name="f-b")
    population = PolicyPopulation.create(
        marshal=[marshal],
        fugitive=[fugitive_a, fugitive_b],
    )
    checkpoint = initialize_psro(
        population,
        evaluator=StaticEvaluator(
            {
                ("policy-m0", "f-a"): 0.0,
                ("policy-m0", "f-b"): 0.0,
            }
        ),
    )
    request = ResponseOracleRequest(
        Role.MARSHAL,
        1,
        checkpoint.population,
        checkpoint.payoff_matrix,
        checkpoint.meta_solution,
    )
    adapter = MixtureConditionedResponseOracle(
        MixtureResponseTemplate(
            Role.MARSHAL,
            "rollout-bir2u",
            base_parameters={
                "particle_count": 1,
                "max_guess_candidates": 1,
                "rollout_candidate_count": 1,
                "max_terminal_simulations": 1,
            },
        )
    )

    response = adapter.propose_response(request)
    opponents = thaw_parameters(
        agent_spec_from_policy(response, population).parameters
    )["opponent_policies"]

    assert len(opponents) == 1
    assert opponents[0]["spec"]["name"] == "same-f"
    assert opponents[0]["weight"] == pytest.approx(1.0)


def test_stable_mixture_returns_incumbent_and_psro_stops_without_new_policy():
    marshal = wrapped(Role.MARSHAL, "m0")
    fugitive = wrapped(Role.FUGITIVE, "f0")
    initial = initialize_psro(
        PolicyPopulation.create(marshal=[marshal], fugitive=[fugitive]),
        evaluator=StaticEvaluator({("policy-m0", "policy-f0"): 0.5}),
    )
    adapter = MixtureConditionedResponseOracle(
        MixtureResponseTemplate(
            Role.MARSHAL,
            "rollout-bir2u",
            base_parameters={
                "particle_count": 8,
                "max_guess_candidates": 8,
                "rollout_candidate_count": 1,
                "max_terminal_simulations": 1,
            },
        )
    )
    first = adapter.propose_response(
        ResponseOracleRequest(
            Role.MARSHAL,
            1,
            initial.population,
            initial.payoff_matrix,
            initial.meta_solution,
        )
    )
    expanded = initialize_psro(
        PolicyPopulation.create(
            marshal=[marshal, first],
            fugitive=[fugitive],
        ),
        evaluator=ConstantEvaluator(),
    )

    repeated = adapter.propose_response(
        ResponseOracleRequest(
            Role.MARSHAL,
            2,
            expanded.population,
            expanded.payoff_matrix,
            expanded.meta_solution,
        )
    )
    stopped = run_psro(
        expanded,
        generations=3,
        marshal_oracle=adapter,
        fugitive_oracle=NoneOracle(),
        evaluator=None,
    )

    assert repeated is first
    assert stopped.stop_reason == "no_new_policies"
    assert stopped.final_checkpoint.last_generation is not None
    assert stopped.final_checkpoint.last_generation.added_policies == ()


def test_current_checkpoint_alone_reproduces_the_next_generation():
    population = PolicyPopulation.create(
        marshal=[wrapped(Role.MARSHAL, "m0")],
        fugitive=[wrapped(Role.FUGITIVE, "f0")],
    )
    run_config = experiment_run_config(
        RegisteredGamePayoffConfig((101, 202), validate_invariants=False)
    )
    checkpoint = initialize_registered_psro(
        population,
        run_config=run_config,
        evaluator=ConstantEvaluator(),
    )
    payload = checkpoint.to_json(indent=None)
    restored = PSROExperimentCheckpoint.from_json(payload)

    first_next = run_registered_psro_iteration(
        checkpoint,
        evaluator=AlwaysAcceptEvaluator(),
    )
    restored_next = run_registered_psro_iteration(
        restored,
        evaluator=AlwaysAcceptEvaluator(),
    )
    resumed_run = run_registered_psro(
        restored,
        generations=1,
        evaluator=AlwaysAcceptEvaluator(),
    )
    continued_run = run_registered_psro(
        first_next,
        generations=1,
        evaluator=AlwaysAcceptEvaluator(),
    )

    assert restored == checkpoint
    assert restored.run_config == run_config
    assert restored.to_json(indent=None) == payload
    assert restored_next == first_next
    assert resumed_run.final_checkpoint == first_next
    assert resumed_run.stop_reason == "generation_limit"
    assert first_next.psro.generation == 1
    assert len(first_next.psro.population.marshal_policies) == 2
    assert len(first_next.psro.population.fugitive_policies) == 2
    assert continued_run.final_checkpoint.response_validations == (
        first_next.response_validations
    )


def test_checkpoint_rejects_a_different_rules_fingerprint():
    population = PolicyPopulation.create(
        marshal=[wrapped(Role.MARSHAL, "m0")],
        fugitive=[wrapped(Role.FUGITIVE, "f0")],
    )
    checkpoint = initialize_registered_psro(
        population,
        run_config=experiment_run_config(
            RegisteredGamePayoffConfig((101,), validate_invariants=False)
        ),
        evaluator=ConstantEvaluator(),
    )
    payload = checkpoint.to_dict()
    payload["run_config"]["rules"]["sha256"] = "0" * 64

    with pytest.raises(ValueError, match="rules fingerprint"):
        PSROExperimentCheckpoint.from_dict(payload)


def test_one_registered_full_game_per_cell_is_reproducible_and_checkpointed():
    fugitive = resolve_registered_policy(
        Role.FUGITIVE,
        "hierarchical-random",
        identifier_name="hr-fugitive",
        overrides={
            "overpay_probability": 0.0,
            "max_low_cost_payments": 4,
            "max_extra_overpayments": 0,
        },
    )
    marshal = resolve_registered_policy(
        Role.MARSHAL,
        "hierarchical-random",
        identifier_name="hr-marshal",
        overrides={
            "multi_guess_continuation": 0.0,
            "max_guess_size": 1,
            "max_guesses_per_size": 32,
        },
    )
    population = PolicyPopulation.create(marshal=[marshal], fugitive=[fugitive])
    payoff_config = RegisteredGamePayoffConfig.derived(20260726, 1)
    run_config = experiment_run_config(payoff_config)

    first = initialize_registered_psro(population, run_config=run_config)
    second = initialize_registered_psro(population, run_config=run_config)

    assert first == second
    entry = first.psro.payoff_matrix.entries[0]
    assert entry.estimate.marshal_payoff in (0.0, 1.0)
    assert entry.estimate.sample_count == 1
    assert entry.estimate.standard_error is None
    payload = first.to_json(indent=None)
    assert PSROExperimentCheckpoint.from_json(payload) == first
    assert PSROExperimentCheckpoint.from_json(payload).to_json(indent=None) == payload
    assert first.run_config == run_config


def test_fresh_paired_validation_controls_admission_and_holdout_is_reporting_only(
    monkeypatch,
    tmp_path,
) -> None:
    calls: list[tuple[int, int, float]] = []

    def fake_run_registered_experiment(**kwargs):
        marshal_size = kwargs["marshal_parameters"]["max_guess_size"]
        fugitive_overpay = kwargs["fugitive_parameters"]["overpay_probability"]
        calls.append((kwargs["master_seed"], marshal_size, fugitive_overpay))
        marshal_wins = marshal_size == 1 or fugitive_overpay == 0.1
        winner = Winner.MARSHAL if marshal_wins else Winner.FUGITIVE
        return SimpleNamespace(
            status=ExperimentStatus.COMPLETED,
            game_result=GameResult(winner, "test", 1, ()),
            manifest=SimpleNamespace(
                to_dict=lambda: {
                    "status": "completed",
                    "seed": str(kwargs["master_seed"]),
                }
            ),
        )

    monkeypatch.setattr(
        "fugitive.psro_payoff.run_registered_experiment",
        fake_run_registered_experiment,
    )
    marshal = resolve_registered_policy(
        Role.MARSHAL,
        "hierarchical-random",
        identifier_name="incumbent-marshal",
        overrides={"max_guess_size": 2},
    )
    fugitive = resolve_registered_policy(
        Role.FUGITIVE,
        "hierarchical-random",
        identifier_name="incumbent-fugitive",
        overrides={"overpay_probability": 0.05},
    )
    config = RegisteredGamePayoffConfig(
        meta_train_seeds=(11,),
        response_validation_seeds=(21, 22, 23),
        final_holdout_seeds=(31, 32),
        bootstrap_replicates=200,
        validate_invariants=False,
    )
    run_config = PSROExperimentRunConfig(
        config,
        MetaSolverConfig(max_iterations=20, tolerance=1.0, check_interval=1),
        MixtureResponseTemplate(
            Role.MARSHAL,
            "hierarchical-random",
            base_parameters={"max_guess_size": 1},
            opponent_parameter=None,
            identifier_prefix="candidate-marshal",
        ),
        MixtureResponseTemplate(
            Role.FUGITIVE,
            "hierarchical-random",
            base_parameters={"overpay_probability": 0.1},
            opponent_parameter=None,
            identifier_prefix="candidate-fugitive",
        ),
    )
    ledger = PayoffLedger(tmp_path / "ledger")
    evaluator = RegisteredGamePayoffEvaluator(config, ledger=ledger)
    initial = initialize_registered_psro(
        PolicyPopulation.create(marshal=[marshal], fugitive=[fugitive]),
        run_config=run_config,
        evaluator=evaluator,
    )

    selected = run_registered_psro_iteration(initial, evaluator=evaluator)
    reports = {report.role: report for report in selected.response_validations}

    assert reports[Role.MARSHAL].paired_gains == (1.0, 1.0, 1.0)
    assert reports[Role.MARSHAL].admitted is True
    assert reports[Role.FUGITIVE].paired_gains == (-1.0, -1.0, -1.0)
    assert reports[Role.FUGITIVE].admitted is False
    assert len(selected.psro.population.marshal_policies) == 2
    assert len(selected.psro.population.fugitive_policies) == 1
    assert reports[Role.FUGITIVE].candidate not in set(
        selected.psro.population.identifiers(Role.FUGITIVE)
    )
    calls_before_resume = len(calls)
    repeated_marshal_report = evaluator.validate_response(
        role=Role.MARSHAL,
        generation=1,
        candidate=selected.psro.population.policy(
            reports[Role.MARSHAL].candidate
        ),
        incumbent=marshal,
        opponent_mixture=initial.psro.meta_solution.fugitive_mixture,
        population=initial.psro.population,
    )
    assert repeated_marshal_report == reports[Role.MARSHAL]
    assert len(calls) == calls_before_resume

    matrix_before_holdout = selected.psro.payoff_matrix
    finalized = finalize_registered_psro(selected, evaluator=evaluator)
    assert finalized.psro.payoff_matrix == matrix_before_holdout
    assert finalized.final_holdout is not None
    assert finalized.final_holdout.sample_count == 2
    assert finalized.final_holdout.to_dict()["selection_feedback"] is False
    assert PSROExperimentCheckpoint.from_json(finalized.to_json()) == finalized
    result = PSROExperimentRunResult((initial, finalized), "generation_limit")
    assert PSROExperimentRunResult.from_json(result.to_json()) == result

    validation_seeds = tuple(
        derive_seed(
            seed,
            f"{PSRO_RESPONSE_VALIDATION_SEED_DOMAIN}:generation=1",
        )
        for seed in (21, 22, 23)
    )
    validation_calls = [
        call for call in calls if call[0] in set(validation_seeds)
    ]
    # The incumbent/incumbent game is shared by both role comparisons, so the
    # ledger needs only three distinct games per seed rather than four.
    assert len(validation_calls) == 9
    assert {
        seed: sum(call[0] == seed for call in validation_calls)
        for seed in validation_seeds
    } == {
        seed: 3 for seed in validation_seeds
    }
    samples = ledger.samples()
    assert {sample.metadata["rules_version"] for sample in samples} == {
        RULES_VERSION
    }
    assert {sample.metadata["rules_sha256"] for sample in samples} == {
        RULES_SHA256
    }
    assert {sample.metadata["schedule"] for sample in samples} == {
        "meta-train",
        "response-validation",
        "final-holdout",
    }


def test_response_validation_seeds_are_paired_resumable_and_fresh_per_generation(
    monkeypatch,
    tmp_path,
) -> None:
    calls: list[int] = []

    def fake_run_registered_experiment(**kwargs):
        calls.append(kwargs["master_seed"])
        return SimpleNamespace(
            status=ExperimentStatus.COMPLETED,
            game_result=GameResult(Winner.MARSHAL, "test", 1, ()),
            manifest=SimpleNamespace(
                to_dict=lambda: {
                    "status": "completed",
                    "seed": str(kwargs["master_seed"]),
                }
            ),
        )

    monkeypatch.setattr(
        "fugitive.psro_payoff.run_registered_experiment",
        fake_run_registered_experiment,
    )
    incumbent = wrapped(Role.MARSHAL, "validation-incumbent")
    candidate_one = wrapped(Role.MARSHAL, "validation-candidate-one")
    candidate_two = wrapped(Role.MARSHAL, "validation-candidate-two")
    fugitive = wrapped(Role.FUGITIVE, "validation-opponent")
    population = PolicyPopulation.create(
        marshal=[incumbent],
        fugitive=[fugitive],
    )
    checkpoint = initialize_psro(
        population,
        evaluator=ConstantEvaluator(),
    )
    base_seeds = (41, 42)
    evaluator = RegisteredGamePayoffEvaluator(
        RegisteredGamePayoffConfig(
            meta_train_seeds=(11,),
            response_validation_seeds=base_seeds,
            final_holdout_seeds=(),
            bootstrap_replicates=20,
            validate_invariants=False,
        ),
        ledger=PayoffLedger(tmp_path / "ledger"),
    )

    first = evaluator.validate_response(
        role=Role.MARSHAL,
        generation=1,
        candidate=candidate_one,
        incumbent=incumbent,
        opponent_mixture=checkpoint.meta_solution.fugitive_mixture,
        population=population,
    )
    calls_after_first = tuple(calls)
    repeated = evaluator.validate_response(
        role=Role.MARSHAL,
        generation=1,
        candidate=candidate_one,
        incumbent=incumbent,
        opponent_mixture=checkpoint.meta_solution.fugitive_mixture,
        population=population,
    )
    second = evaluator.validate_response(
        role=Role.MARSHAL,
        generation=2,
        candidate=candidate_two,
        incumbent=incumbent,
        opponent_mixture=checkpoint.meta_solution.fugitive_mixture,
        population=population,
    )

    generation_one = tuple(
        derive_seed(
            seed,
            f"{PSRO_RESPONSE_VALIDATION_SEED_DOMAIN}:generation=1",
        )
        for seed in base_seeds
    )
    generation_two = tuple(
        derive_seed(
            seed,
            f"{PSRO_RESPONSE_VALIDATION_SEED_DOMAIN}:generation=2",
        )
        for seed in base_seeds
    )
    assert calls_after_first == tuple(
        seed for seed in generation_one for _ in range(2)
    )
    assert repeated == first
    assert tuple(calls[len(calls_after_first) :]) == tuple(
        seed for seed in generation_two for _ in range(2)
    )
    assert set(generation_one).isdisjoint(generation_two)
    assert set(generation_one).isdisjoint(base_seeds)
    assert set(generation_two).isdisjoint(base_seeds)
    assert first.opponent_draws == second.opponent_draws
