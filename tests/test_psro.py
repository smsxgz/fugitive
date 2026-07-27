from __future__ import annotations

from collections import deque
from dataclasses import replace

import pytest

from fugitive.model import Role
from fugitive.psro import (
    EmpiricalPayoffMatrix,
    MetaSolverConfig,
    MultiplicativeWeightsMetaSolver,
    PSROCheckpoint,
    PayoffEstimate,
    PayoffEvaluation,
    PayoffPair,
    PolicyPopulation,
    PolicySpec,
    ResponseOracleRequest,
    evaluate_missing_payoffs,
    initialize_psro,
    run_psro,
    run_psro_iteration,
)


def policy(role: Role, name: str, **parameters: object) -> PolicySpec:
    return PolicySpec.create(role, name, parameters)


class TableEvaluator:
    def __init__(self, values: dict[tuple[str, str], float]) -> None:
        self.values = values
        self.calls: list[tuple[PayoffPair, ...]] = []

    def evaluate(self, requests):
        self.calls.append(tuple(request.pair for request in requests))
        # Returning another order verifies that the batch API is parallel-safe.
        return [
            PayoffEvaluation(
                request.pair,
                PayoffEstimate(
                    self.values[
                        (
                            request.pair.marshal.name,
                            request.pair.fugitive.name,
                        )
                    ],
                    sample_count=200,
                    standard_error=0.03,
                ),
            )
            for request in reversed(requests)
        ]


class QueueOracle:
    def __init__(self, role: Role, *responses: PolicySpec | None) -> None:
        self.role = role
        self.responses = deque(responses)
        self.requests: list[ResponseOracleRequest] = []

    def propose_response(self, request: ResponseOracleRequest) -> PolicySpec | None:
        assert request.role is self.role
        assert request.opponent_mixture.role is not self.role
        assert request.incumbent_mixture.role is self.role
        self.requests.append(request)
        return self.responses.popleft() if self.responses else None


def complete_matrix(
    population: PolicyPopulation,
    values: list[list[float]],
) -> EmpiricalPayoffMatrix:
    evaluations = []
    for row, marshal in enumerate(population.marshal_policies):
        for column, fugitive in enumerate(population.fugitive_policies):
            evaluations.append(
                PayoffEvaluation(
                    PayoffPair(marshal.identifier, fugitive.identifier),
                    PayoffEstimate(values[row][column]),
                )
            )
    return EmpiricalPayoffMatrix().with_evaluations(evaluations)


def test_policy_identity_is_role_separated_and_spec_is_deeply_immutable():
    marshal = policy(Role.MARSHAL, "same-name", nested={"values": [1, 2]})
    fugitive = policy(Role.FUGITIVE, "same-name")

    assert marshal.identifier != fugitive.identifier
    assert marshal.role is Role.MARSHAL
    assert marshal.parameters["nested"]["values"] == (1, 2)
    with pytest.raises(TypeError):
        marshal.parameters["new"] = 3
    with pytest.raises(TypeError):
        marshal.parameters["nested"]["new"] = 3


def test_matching_pennies_returns_the_restricted_equilibrium_and_zero_gap():
    marshals = [policy(Role.MARSHAL, name) for name in ("heads", "tails")]
    fugitives = [policy(Role.FUGITIVE, name) for name in ("heads", "tails")]
    population = PolicyPopulation.create(marshal=marshals, fugitive=fugitives)
    matrix = complete_matrix(population, [[1.0, -1.0], [-1.0, 1.0]])
    solver = MultiplicativeWeightsMetaSolver(
        MetaSolverConfig(
            max_iterations=1_000,
            tolerance=1e-12,
            check_interval=1,
        )
    )

    solution = solver.solve(population, matrix)

    assert [entry.probability for entry in solution.marshal_mixture.entries] == [
        0.5,
        0.5,
    ]
    assert [entry.probability for entry in solution.fugitive_mixture.entries] == [
        0.5,
        0.5,
    ]
    assert solution.restricted_game_value == pytest.approx(0.0)
    assert solution.profile_payoff == pytest.approx(0.0)
    assert solution.lower_bound == pytest.approx(0.0)
    assert solution.upper_bound == pytest.approx(0.0)
    assert solution.diagnostics.duality_gap == pytest.approx(0.0)
    assert solution.diagnostics.converged


def test_dominated_policies_lose_mass_and_residual_bounds_are_auditable():
    strong_marshal = policy(Role.MARSHAL, "strong")
    weak_marshal = policy(Role.MARSHAL, "dominated")
    weak_fugitive = policy(Role.FUGITIVE, "dominated")
    strong_fugitive = policy(Role.FUGITIVE, "strong")
    population = PolicyPopulation.create(
        marshal=[strong_marshal, weak_marshal],
        fugitive=[weak_fugitive, strong_fugitive],
    )
    matrix = complete_matrix(population, [[1.0, 0.0], [0.0, -1.0]])
    solver = MultiplicativeWeightsMetaSolver(
        MetaSolverConfig(
            max_iterations=50_000,
            tolerance=0.006,
            check_interval=100,
        )
    )

    solution = solver.solve(population, matrix)

    assert solution.marshal_mixture.probability(strong_marshal.identifier) > 0.997
    assert solution.fugitive_mixture.probability(strong_fugitive.identifier) > 0.997
    assert solution.restricted_game_value == pytest.approx(0.0, abs=1e-12)
    assert solution.lower_bound <= solution.restricted_game_value
    assert solution.restricted_game_value <= solution.upper_bound
    assert solution.diagnostics.duality_gap <= 0.006
    assert solution.diagnostics.marshal_residual > 0.0
    assert solution.diagnostics.fugitive_residual > 0.0
    assert solution.diagnostics.converged


def test_solver_is_deterministic_and_reports_non_convergence_honestly():
    marshals = [policy(Role.MARSHAL, name) for name in ("m0", "m1")]
    fugitives = [policy(Role.FUGITIVE, name) for name in ("f0", "f1")]
    population = PolicyPopulation.create(marshal=marshals, fugitive=fugitives)
    matrix = complete_matrix(population, [[2.0, 0.0], [0.0, -1.0]])
    solver = MultiplicativeWeightsMetaSolver(
        MetaSolverConfig(
            max_iterations=1,
            minimum_iterations=1,
            check_interval=1,
            tolerance=0.0,
        )
    )

    first = solver.solve(population, matrix)
    second = solver.solve(population, matrix)

    assert first == second
    assert first.diagnostics.iterations == 1
    assert not first.diagnostics.converged
    assert first.diagnostics.duality_gap > 0.0


def test_checkpoint_rejects_an_inconsistent_restricted_value_midpoint():
    marshal = policy(Role.MARSHAL, "m")
    fugitive = policy(Role.FUGITIVE, "f")
    checkpoint = initialize_psro(
        PolicyPopulation.create(marshal=[marshal], fugitive=[fugitive]),
        evaluator=TableEvaluator({("m", "f"): 0.25}),
    )

    with pytest.raises(ValueError, match="restricted_game_value"):
        replace(
            checkpoint,
            meta_solution=replace(
                checkpoint.meta_solution,
                restricted_game_value=0.5,
            ),
        )


def test_incremental_expansion_requests_only_the_new_row_and_column():
    m0 = policy(Role.MARSHAL, "m0")
    m1 = policy(Role.MARSHAL, "m1", generation=1)
    f0 = policy(Role.FUGITIVE, "f0")
    f1 = policy(Role.FUGITIVE, "f1", generation=1)
    evaluator = TableEvaluator(
        {
            ("m0", "f0"): 0.0,
            ("m1", "f0"): 1.0,
            ("m0", "f1"): -1.0,
            ("m1", "f1"): 0.0,
        }
    )
    population = PolicyPopulation.create(marshal=[m0], fugitive=[f0])
    solver = MultiplicativeWeightsMetaSolver(
        MetaSolverConfig(max_iterations=5_000, tolerance=0.01, check_interval=10)
    )
    initial = initialize_psro(
        population,
        evaluator=evaluator,
        meta_solver=solver,
    )
    assert evaluator.calls == [(PayoffPair(m0.identifier, f0.identifier),)]

    expanded = run_psro_iteration(
        initial,
        marshal_oracle=QueueOracle(Role.MARSHAL, m1),
        fugitive_oracle=QueueOracle(Role.FUGITIVE, f1),
        evaluator=evaluator,
        meta_solver=solver,
    )

    expected_new = {
        PayoffPair(m0.identifier, f1.identifier),
        PayoffPair(m1.identifier, f0.identifier),
        PayoffPair(m1.identifier, f1.identifier),
    }
    assert len(evaluator.calls) == 2
    assert set(evaluator.calls[1]) == expected_new
    assert PayoffPair(m0.identifier, f0.identifier) not in evaluator.calls[1]
    assert len(expanded.payoff_matrix.entries) == 4
    assert expanded.last_generation is not None
    assert set(expanded.last_generation.evaluated_pairs) == expected_new
    assert expanded.last_generation.marshal_response_gain == pytest.approx(1.0)
    assert expanded.last_generation.fugitive_response_gain == pytest.approx(1.0)


def test_multiple_generations_stop_when_both_oracles_add_nothing():
    m0 = policy(Role.MARSHAL, "m0")
    m1 = policy(Role.MARSHAL, "m1")
    f0 = policy(Role.FUGITIVE, "f0")
    f1 = policy(Role.FUGITIVE, "f1")
    evaluator = TableEvaluator(
        {
            ("m0", "f0"): 0.0,
            ("m1", "f0"): 1.0,
            ("m0", "f1"): -1.0,
            ("m1", "f1"): 0.0,
        }
    )
    initial = initialize_psro(
        PolicyPopulation.create(marshal=[m0], fugitive=[f0]),
        evaluator=evaluator,
    )
    marshal_oracle = QueueOracle(Role.MARSHAL, m1, None)
    fugitive_oracle = QueueOracle(Role.FUGITIVE, f1, None)

    result = run_psro(
        initial,
        generations=10,
        marshal_oracle=marshal_oracle,
        fugitive_oracle=fugitive_oracle,
        evaluator=evaluator,
    )

    assert result.stop_reason == "no_new_policies"
    assert [checkpoint.generation for checkpoint in result.checkpoints] == [0, 1, 2]
    assert result.final_checkpoint.last_generation is not None
    assert result.final_checkpoint.last_generation.added_policies == ()
    # Initial matrix plus one expansion; the empty generation never calls it.
    assert len(evaluator.calls) == 2


def test_generation_checkpoint_json_round_trip_is_exact_and_deterministic():
    m0 = policy(Role.MARSHAL, "m0", profile="default")
    m1 = policy(Role.MARSHAL, "m1", search={"rollouts": 64})
    f0 = policy(Role.FUGITIVE, "f0", epsilon=0.1)
    f1 = policy(Role.FUGITIVE, "f1", tags=["response", "generation-1"])
    evaluator = TableEvaluator(
        {
            ("m0", "f0"): 0.0,
            ("m1", "f0"): 1.0,
            ("m0", "f1"): -1.0,
            ("m1", "f1"): 0.0,
        }
    )
    initial = initialize_psro(
        PolicyPopulation.create(marshal=[m0], fugitive=[f0]),
        evaluator=evaluator,
    )
    checkpoint = run_psro_iteration(
        initial,
        marshal_oracle=QueueOracle(Role.MARSHAL, m1),
        fugitive_oracle=QueueOracle(Role.FUGITIVE, f1),
        evaluator=evaluator,
    )

    payload = checkpoint.to_json(indent=None)
    restored = PSROCheckpoint.from_json(payload)

    assert restored == checkpoint
    assert restored.to_json(indent=None) == payload


def test_payoff_batch_rejects_partial_results_without_mutating_the_matrix():
    marshal = policy(Role.MARSHAL, "m")
    fugitive = policy(Role.FUGITIVE, "f")
    population = PolicyPopulation.create(marshal=[marshal], fugitive=[fugitive])
    original = EmpiricalPayoffMatrix()

    class EmptyEvaluator:
        def evaluate(self, requests):
            return ()

    with pytest.raises(ValueError, match="omitted 1"):
        evaluate_missing_payoffs(population, original, EmptyEvaluator())
    assert original.entries == ()
