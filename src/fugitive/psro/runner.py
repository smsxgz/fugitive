"""Initialization, response admission, and iteration of registered PSRO."""

from __future__ import annotations

import math

from ..game.model import Role
from .algorithm import (
    ResponseOracleRequest,
    initialize_psro,
    run_psro_iteration,
)
from .checkpoint import (
    PSROExperimentCheckpoint,
    PSROExperimentRunConfig,
    PSROExperimentRunResult,
)
from .payoff import RegisteredGamePayoffEvaluator
from .payoff_matrix import EmpiricalPayoffMatrix, PayoffEvaluator
from .policy_adapter import MixtureConditionedResponseOracle
from .population import PolicyMixture, PolicyPopulation, PolicySpec
from .solver import MultiplicativeWeightsMetaSolver
from .validation import FinalHoldoutReport, ResponseValidationReport


def initialize_registered_psro(
    population: PolicyPopulation,
    *,
    run_config: PSROExperimentRunConfig,
    payoff_matrix: EmpiricalPayoffMatrix | None = None,
    evaluator: PayoffEvaluator | None = None,
) -> PSROExperimentCheckpoint:
    concrete_evaluator = evaluator or RegisteredGamePayoffEvaluator(
        run_config.payoff_config
    )
    if isinstance(concrete_evaluator, RegisteredGamePayoffEvaluator):
        concrete_evaluator.bind_population(population)
    checkpoint = initialize_psro(
        population,
        evaluator=concrete_evaluator,
        payoff_matrix=payoff_matrix,
        meta_solver=MultiplicativeWeightsMetaSolver(
            run_config.meta_solver_config
        ),
    )
    return PSROExperimentCheckpoint(checkpoint, run_config)


class _FixedResponseOracle:
    """Return a response already proposed and independently validated."""

    def __init__(self, response: PolicySpec | None) -> None:
        self.response = response

    def propose_response(self, request: ResponseOracleRequest) -> PolicySpec | None:
        del request
        return self.response


def _restricted_best_response(
    role: Role,
    population: PolicyPopulation,
    matrix: EmpiricalPayoffMatrix,
    opponent_mixture: PolicyMixture,
) -> PolicySpec:
    """Return the deterministic best incumbent pure response."""

    policies = population.policies(role)
    if role is Role.MARSHAL:
        value = lambda policy: math.fsum(  # noqa: E731 - formula reads directly
            entry.probability
            * matrix.marshal_payoff(policy.identifier, entry.identifier)
            for entry in opponent_mixture.entries
        )
        return max(policies, key=value)
    value = lambda policy: math.fsum(  # noqa: E731 - formula reads directly
        entry.probability
        * matrix.marshal_payoff(entry.identifier, policy.identifier)
        for entry in opponent_mixture.entries
    )
    return min(policies, key=value)


def _validate_candidate(
    evaluator: PayoffEvaluator,
    *,
    checkpoint: PSROExperimentCheckpoint,
    role: Role,
    generation: int,
    candidate: PolicySpec | None,
) -> tuple[PolicySpec | None, ResponseValidationReport | None]:
    if candidate is None:
        return None, None
    current_ids = set(checkpoint.psro.population.identifiers(role))
    if candidate.identifier in current_ids:
        return candidate, None
    validate = getattr(evaluator, "validate_response", None)
    if not callable(validate):
        raise ValueError(
            "registered PSRO response admission requires a validation evaluator"
        )
    opponent_mixture = (
        checkpoint.psro.meta_solution.fugitive_mixture
        if role is Role.MARSHAL
        else checkpoint.psro.meta_solution.marshal_mixture
    )
    incumbent = _restricted_best_response(
        role,
        checkpoint.psro.population,
        checkpoint.psro.payoff_matrix,
        opponent_mixture,
    )
    report = validate(
        role=role,
        generation=generation,
        candidate=candidate,
        incumbent=incumbent,
        opponent_mixture=opponent_mixture,
        population=checkpoint.psro.population,
    )
    if not isinstance(report, ResponseValidationReport):
        raise ValueError("response validator returned a malformed report")
    return (candidate if report.admitted else None), report


def run_registered_psro_iteration(
    checkpoint: PSROExperimentCheckpoint,
    *,
    evaluator: PayoffEvaluator | None = None,
) -> PSROExperimentCheckpoint:
    concrete_evaluator = evaluator or RegisteredGamePayoffEvaluator(
        checkpoint.payoff_config
    )
    if isinstance(concrete_evaluator, RegisteredGamePayoffEvaluator):
        concrete_evaluator.bind_population(checkpoint.psro.population)
    generation = checkpoint.psro.generation + 1
    marshal_request = ResponseOracleRequest(
        Role.MARSHAL,
        generation,
        checkpoint.psro.population,
        checkpoint.psro.payoff_matrix,
        checkpoint.psro.meta_solution,
    )
    fugitive_request = ResponseOracleRequest(
        Role.FUGITIVE,
        generation,
        checkpoint.psro.population,
        checkpoint.psro.payoff_matrix,
        checkpoint.psro.meta_solution,
    )
    marshal_candidate = MixtureConditionedResponseOracle(
        checkpoint.run_config.marshal_response_template
    ).propose_response(marshal_request)
    fugitive_candidate = MixtureConditionedResponseOracle(
        checkpoint.run_config.fugitive_response_template
    ).propose_response(fugitive_request)
    marshal_response, marshal_report = _validate_candidate(
        concrete_evaluator,
        checkpoint=checkpoint,
        role=Role.MARSHAL,
        generation=generation,
        candidate=marshal_candidate,
    )
    fugitive_response, fugitive_report = _validate_candidate(
        concrete_evaluator,
        checkpoint=checkpoint,
        role=Role.FUGITIVE,
        generation=generation,
        candidate=fugitive_candidate,
    )
    updated = run_psro_iteration(
        checkpoint.psro,
        marshal_oracle=_FixedResponseOracle(marshal_response),
        fugitive_oracle=_FixedResponseOracle(fugitive_response),
        evaluator=concrete_evaluator,
        meta_solver=MultiplicativeWeightsMetaSolver(
            checkpoint.run_config.meta_solver_config
        ),
    )
    reports = tuple(
        report
        for report in (marshal_report, fugitive_report)
        if report is not None
    )
    return PSROExperimentCheckpoint(
        updated,
        checkpoint.run_config,
        (*checkpoint.response_validations, *reports),
    )


def finalize_registered_psro(
    checkpoint: PSROExperimentCheckpoint,
    *,
    evaluator: PayoffEvaluator | None = None,
) -> PSROExperimentCheckpoint:
    """Attach a fresh final-profile report without changing selection state."""

    concrete = evaluator or RegisteredGamePayoffEvaluator(checkpoint.payoff_config)
    if not checkpoint.payoff_config.final_holdout_seeds:
        report = None
    else:
        evaluate_holdout = getattr(concrete, "evaluate_final_holdout", None)
        if not callable(evaluate_holdout):
            raise ValueError(
                "registered PSRO finalization requires a holdout evaluator"
            )
        report = evaluate_holdout(
            checkpoint.psro.population,
            checkpoint.psro.meta_solution.marshal_mixture,
            checkpoint.psro.meta_solution.fugitive_mixture,
        )
        if report is not None and not isinstance(report, FinalHoldoutReport):
            raise ValueError("final holdout evaluator returned a malformed report")
    return PSROExperimentCheckpoint(
        checkpoint.psro,
        checkpoint.run_config,
        checkpoint.response_validations,
        report,
    )


def run_registered_psro(
    initial_checkpoint: PSROExperimentCheckpoint,
    *,
    generations: int,
    evaluator: PayoffEvaluator | None = None,
    stop_when_no_new_policies: bool = True,
) -> PSROExperimentRunResult:
    if isinstance(generations, bool) or not isinstance(generations, int):
        raise ValueError("PSRO generations must be an integer")
    if generations < 0:
        raise ValueError("PSRO generations must be non-negative")
    current = initial_checkpoint
    checkpoints = [current]
    stop_reason = "generation_limit"
    for _ in range(generations):
        current = run_registered_psro_iteration(current, evaluator=evaluator)
        checkpoints.append(current)
        summary = current.psro.last_generation
        if (
            stop_when_no_new_policies
            and summary is not None
            and not summary.added_policies
        ):
            stop_reason = "no_new_policies"
            break
    checkpoints[-1] = finalize_registered_psro(current, evaluator=evaluator)
    return PSROExperimentRunResult(tuple(checkpoints), stop_reason)


__all__ = [
    "finalize_registered_psro",
    "initialize_registered_psro",
    "run_registered_psro",
    "run_registered_psro_iteration",
]
