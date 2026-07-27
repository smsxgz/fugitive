from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

from fugitive.game.model import GameResult, Role, Winner
from fugitive.psro.algorithm import ResponseOracleRequest, initialize_psro
from fugitive.psro.checkpoint import PSROExperimentRunConfig
from fugitive.psro.ledger import PayoffLedger
from fugitive.psro.payoff import (
    RegisteredGamePayoffConfig,
    RegisteredGamePayoffEvaluator,
)
from fugitive.psro.payoff_matrix import (
    EmpiricalPayoffMatrix,
    PayoffEstimate,
    PayoffEvaluation,
    evaluate_missing_payoffs,
)
from fugitive.psro.policy_adapter import (
    MixtureConditionedResponseOracle,
    MixtureResponseTemplate,
    agent_spec_from_policy,
    planning_leaf_spec_from_policy,
    registered_policy_spec,
    resolve_registered_policy,
)
from fugitive.psro.population import PolicyPopulation
from fugitive.psro.runner import (
    finalize_registered_psro,
    initialize_registered_psro,
    run_registered_psro_iteration,
)
from fugitive.psro.solver import MetaSolverConfig
from fugitive.runtime.manifest import RULES_SHA256, RULES_VERSION, MatchStatus
from fugitive.shared.reproducibility import AgentSpec, thaw_parameters


def wrapped(role: Role, name: str):
    return registered_policy_spec(
        AgentSpec(name, role, "default", {"variant": name}),
        identifier_name=f"policy-{name}",
    )


def fake_completed_match(winner: Winner, seed: int):
    return SimpleNamespace(
        status=MatchStatus.COMPLETED,
        game_result=GameResult(winner, "test", 1, ()),
        manifest=SimpleNamespace(
            to_dict=lambda: {"status": "completed", "seed": str(seed)}
        ),
    )


class ConstantEvaluator:
    def evaluate(self, requests):
        return tuple(
            PayoffEvaluation(request.pair, PayoffEstimate(0.5))
            for request in requests
        )


def test_payoff_cells_use_the_same_paired_seeds(monkeypatch) -> None:
    marshals = [wrapped(Role.MARSHAL, name) for name in ("m0", "m1")]
    fugitives = [wrapped(Role.FUGITIVE, name) for name in ("f0", "f1")]
    population = PolicyPopulation.create(marshal=marshals, fugitive=fugitives)
    calls: list[tuple[str, str, int]] = []

    def fake_run_registered_match(**kwargs):
        calls.append(
            (
                kwargs["marshal_name"],
                kwargs["fugitive_name"],
                kwargs["master_seed"],
            )
        )
        winner = Winner.MARSHAL if kwargs["master_seed"] == 11 else Winner.FUGITIVE
        return fake_completed_match(winner, kwargs["master_seed"])

    monkeypatch.setattr(
        "fugitive.psro.payoff.run_registered_match",
        fake_run_registered_match,
    )
    result = evaluate_missing_payoffs(
        population,
        EmpiricalPayoffMatrix(),
        RegisteredGamePayoffEvaluator(
            RegisteredGamePayoffConfig((11, 22), validate_invariants=False)
        ),
    )

    assert len(calls) == 8
    for marshal in ("m0", "m1"):
        for fugitive in ("f0", "f1"):
            assert [
                seed
                for seen_marshal, seen_fugitive, seed in calls
                if seen_marshal == marshal and seen_fugitive == fugitive
            ] == [11, 22]
    assert {entry.estimate.marshal_payoff for entry in result.matrix.entries} == {
        0.5
    }
    assert {entry.estimate.sample_count for entry in result.matrix.entries} == {2}


def test_mixture_oracle_creates_role_correct_generation_responses() -> None:
    marshal = wrapped(Role.MARSHAL, "m0")
    fugitive = wrapped(Role.FUGITIVE, "f0")
    checkpoint = initialize_psro(
        PolicyPopulation.create(marshal=[marshal], fugitive=[fugitive]),
        evaluator=ConstantEvaluator(),
    )
    request = ResponseOracleRequest(
        Role.MARSHAL,
        1,
        checkpoint.population,
        checkpoint.payoff_matrix,
        checkpoint.meta_solution,
    )
    oracle = MixtureConditionedResponseOracle(
        MixtureResponseTemplate(
            Role.MARSHAL,
            "rollout-bir2u",
            base_parameters={
                "particle_count": 1,
                "max_guess_candidates": 1,
                "rollout_candidate_count": 1,
                "max_terminal_simulations": 1,
            },
            identifier_prefix="response-marshal",
        )
    )

    first = oracle.propose_response(request)
    second = oracle.propose_response(replace(request, generation=2))
    concrete = agent_spec_from_policy(first, checkpoint.population)
    opponents = thaw_parameters(concrete.parameters)["opponent_policies"]

    assert first.role is Role.MARSHAL
    assert first.identifier.name.startswith("response-marshal-g1-")
    assert second.identifier != first.identifier
    assert concrete.name == "rollout-bir2u"
    assert len(opponents) == 1
    assert opponents[0]["spec"]["name"] == "f0"
    assert opponents[0]["spec"]["role"] == Role.FUGITIVE.value
    assert opponents[0]["weight"] == 1.0


def test_search_response_executes_declared_non_search_planning_leaves() -> None:
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
        PolicyPopulation.create(marshal=[marshal], fugitive=[fugitive_search]),
        evaluator=ConstantEvaluator(),
    )
    oracle = MixtureConditionedResponseOracle(
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
    response = oracle.propose_response(
        ResponseOracleRequest(
            Role.MARSHAL,
            1,
            checkpoint.population,
            checkpoint.payoff_matrix,
            checkpoint.meta_solution,
        )
    )

    stored = thaw_parameters(response.parameters)
    assert "opponent_policies" not in stored["agent_spec"]["parameters"]
    execution = agent_spec_from_policy(response, checkpoint.population)
    opponent = thaw_parameters(execution.parameters)["opponent_policies"][0]
    assert opponent["spec"]["name"] == "continuation-count"
    assert "opponent_policies" not in opponent["spec"]["parameters"]

    response_leaf = planning_leaf_spec_from_policy(response)
    assert response_leaf.name == "unweighted-constructive-belief-informed-random"
    assert "opponent_policies" not in response_leaf.parameters
    assert stored["metadata"]["opponent_leaf_mixture"][0]["policy_ids"] == [
        fugitive_search.identifier.to_dict()
    ]
    assert stored["metadata"]["planning_depth"] == 1


def test_validation_selects_responses_before_the_holdout_is_reported(
    monkeypatch,
    tmp_path,
) -> None:
    def fake_run_registered_match(**kwargs):
        marshal_size = kwargs["marshal_parameters"]["max_guess_size"]
        fugitive_overpay = kwargs["fugitive_parameters"]["overpay_probability"]
        marshal_wins = marshal_size == 1 or fugitive_overpay == 0.1
        winner = Winner.MARSHAL if marshal_wins else Winner.FUGITIVE
        return fake_completed_match(winner, kwargs["master_seed"])

    monkeypatch.setattr(
        "fugitive.psro.payoff.run_registered_match",
        fake_run_registered_match,
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
    payoff_config = RegisteredGamePayoffConfig(
        meta_train_seeds=(11,),
        response_validation_seeds=(21, 22, 23),
        final_holdout_seeds=(31, 32),
        bootstrap_replicates=200,
        validate_invariants=False,
    )
    run_config = PSROExperimentRunConfig(
        payoff_config,
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
    evaluator = RegisteredGamePayoffEvaluator(payoff_config, ledger=ledger)
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

    matrix_before_holdout = selected.psro.payoff_matrix
    finalized = finalize_registered_psro(selected, evaluator=evaluator)

    assert finalized.psro.payoff_matrix == matrix_before_holdout
    assert finalized.final_holdout is not None
    assert finalized.final_holdout.sample_count == 2
    assert finalized.final_holdout.to_dict()["selection_feedback"] is False
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
