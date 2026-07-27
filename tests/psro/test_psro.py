from __future__ import annotations

from collections import deque

import pytest

from fugitive.game.model import Role
from fugitive.psro.algorithm import (
    ResponseOracleRequest,
    initialize_psro,
    run_psro,
    run_psro_iteration,
)
from fugitive.psro.payoff_matrix import (
    EmpiricalPayoffMatrix,
    PayoffEstimate,
    PayoffEvaluation,
    PayoffPair,
)
from fugitive.psro.population import PolicyPopulation, PolicySpec
from fugitive.psro.solver import MetaSolverConfig, MultiplicativeWeightsMetaSolver


def policy(role: Role, name: str) -> PolicySpec:
    return PolicySpec.create(role, name)


class TableEvaluator:
    def __init__(self, values: dict[tuple[str, str], float]) -> None:
        self.values = values
        self.calls: list[tuple[PayoffPair, ...]] = []

    def evaluate(self, requests):
        self.calls.append(tuple(request.pair for request in requests))
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


class QueueOracle:
    def __init__(self, role: Role, *responses: PolicySpec | None) -> None:
        self.role = role
        self.responses = deque(responses)

    def propose_response(self, request: ResponseOracleRequest) -> PolicySpec | None:
        assert request.role is self.role
        assert request.opponent_mixture.role is not self.role
        return self.responses.popleft() if self.responses else None


def complete_matrix(
    population: PolicyPopulation,
    values: list[list[float]],
) -> EmpiricalPayoffMatrix:
    evaluations = (
        PayoffEvaluation(
            PayoffPair(marshal.identifier, fugitive.identifier),
            PayoffEstimate(values[row][column]),
        )
        for row, marshal in enumerate(population.marshal_policies)
        for column, fugitive in enumerate(population.fugitive_policies)
    )
    return EmpiricalPayoffMatrix().with_evaluations(evaluations)


def test_matching_pennies_returns_the_restricted_equilibrium() -> None:
    marshals = [policy(Role.MARSHAL, name) for name in ("heads", "tails")]
    fugitives = [policy(Role.FUGITIVE, name) for name in ("heads", "tails")]
    population = PolicyPopulation.create(marshal=marshals, fugitive=fugitives)
    matrix = complete_matrix(population, [[1.0, -1.0], [-1.0, 1.0]])
    solver = MultiplicativeWeightsMetaSolver(
        MetaSolverConfig(max_iterations=1_000, tolerance=1e-12, check_interval=1)
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
    assert solution.diagnostics.duality_gap == pytest.approx(0.0)
    assert solution.diagnostics.converged


def test_iteration_evaluates_only_the_new_row_and_column() -> None:
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

    expanded = run_psro_iteration(
        initial,
        marshal_oracle=QueueOracle(Role.MARSHAL, m1),
        fugitive_oracle=QueueOracle(Role.FUGITIVE, f1),
        evaluator=evaluator,
    )

    expected = {
        PayoffPair(m0.identifier, f1.identifier),
        PayoffPair(m1.identifier, f0.identifier),
        PayoffPair(m1.identifier, f1.identifier),
    }
    assert set(evaluator.calls[1]) == expected
    assert PayoffPair(m0.identifier, f0.identifier) not in evaluator.calls[1]
    assert len(expanded.payoff_matrix.entries) == 4
    assert expanded.last_generation is not None
    assert set(expanded.last_generation.evaluated_pairs) == expected


def test_psro_stops_when_both_response_oracles_add_nothing() -> None:
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

    result = run_psro(
        initial,
        generations=10,
        marshal_oracle=QueueOracle(Role.MARSHAL, m1, None),
        fugitive_oracle=QueueOracle(Role.FUGITIVE, f1, None),
        evaluator=evaluator,
    )

    assert result.stop_reason == "no_new_policies"
    assert [checkpoint.generation for checkpoint in result.checkpoints] == [0, 1, 2]
    assert result.final_checkpoint.last_generation is not None
    assert result.final_checkpoint.last_generation.added_policies == ()
