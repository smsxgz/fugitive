"""Meta-strategy models and solvers for finite empirical games."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Protocol, Sequence

from ..game.model import Role
from ..shared.reproducibility import (
    FrozenJSONValue,
    JSONValue,
    freeze_parameters,
    thaw_parameters,
)
from ._json_validation import (
    require_finite as _require_finite,
    require_mapping as _require_mapping,
    require_non_negative_int as _require_non_negative_int,
    require_positive_int as _require_positive_int,
)
from .payoff_matrix import EmpiricalPayoffMatrix
from .population import PolicyMixture, PolicyPopulation


MULTIPLICATIVE_WEIGHTS_SOLVER_ID = "simultaneous-multiplicative-weights-v1"

@dataclass(frozen=True, slots=True)
class MetaSolverConfig:
    # One payoff percentage point is a practical default for empirical games;
    # callers can request a tighter restricted-game residual explicitly.
    max_iterations: int = 100_000
    tolerance: float = 1e-2
    minimum_iterations: int = 1
    check_interval: int = 100
    learning_rate: float | None = None

    def __post_init__(self) -> None:
        _require_positive_int(self.max_iterations, "max_iterations")
        tolerance = _require_finite(self.tolerance, "tolerance")
        if tolerance < 0.0:
            raise ValueError("tolerance must be non-negative")
        object.__setattr__(self, "tolerance", tolerance)
        minimum = _require_positive_int(self.minimum_iterations, "minimum_iterations")
        if minimum > self.max_iterations:
            raise ValueError("minimum_iterations cannot exceed max_iterations")
        _require_positive_int(self.check_interval, "check_interval")
        if self.learning_rate is not None:
            rate = _require_finite(self.learning_rate, "learning_rate")
            if rate <= 0.0:
                raise ValueError("learning_rate must be positive")
            object.__setattr__(self, "learning_rate", rate)

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "max_iterations": self.max_iterations,
            "tolerance": self.tolerance,
            "minimum_iterations": self.minimum_iterations,
            "check_interval": self.check_interval,
            "learning_rate": self.learning_rate,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "MetaSolverConfig":
        learning_rate = data.get("learning_rate")
        return cls(
            max_iterations=_require_positive_int(
                data.get("max_iterations"), "max_iterations"
            ),
            tolerance=_require_finite(data.get("tolerance"), "tolerance"),
            minimum_iterations=_require_positive_int(
                data.get("minimum_iterations"), "minimum_iterations"
            ),
            check_interval=_require_positive_int(
                data.get("check_interval"), "check_interval"
            ),
            learning_rate=(
                None
                if learning_rate is None
                else _require_finite(learning_rate, "learning_rate")
            ),
        )


@dataclass(frozen=True, slots=True)
class MetaSolverDiagnostics:
    """Restricted-matrix best-response residuals for one solver run."""

    solver_id: str
    solver_parameters: Mapping[str, FrozenJSONValue]
    iterations: int
    converged: bool
    tolerance: float
    learning_rate: float
    payoff_min: float
    payoff_max: float
    marshal_best_response_value: float
    fugitive_best_response_value: float
    marshal_residual: float
    fugitive_residual: float
    duality_gap: float

    def __post_init__(self) -> None:
        if not isinstance(self.solver_id, str) or not self.solver_id:
            raise ValueError("solver diagnostics require a solver_id")
        object.__setattr__(
            self,
            "solver_parameters",
            freeze_parameters(self.solver_parameters),
        )
        _require_non_negative_int(self.iterations, "iterations")
        if not isinstance(self.converged, bool):
            raise ValueError("converged must be a Boolean")
        for label in (
            "tolerance",
            "learning_rate",
            "payoff_min",
            "payoff_max",
            "marshal_best_response_value",
            "fugitive_best_response_value",
            "marshal_residual",
            "fugitive_residual",
            "duality_gap",
        ):
            object.__setattr__(
                self,
                label,
                _require_finite(getattr(self, label), label),
            )
        if self.tolerance < 0.0:
            raise ValueError("solver tolerance must be non-negative")
        if self.learning_rate <= 0.0:
            raise ValueError("solver learning_rate must be positive")
        if self.payoff_min > self.payoff_max:
            raise ValueError("solver payoff bounds are reversed")
        if min(
            self.marshal_residual,
            self.fugitive_residual,
            self.duality_gap,
        ) < 0.0:
            raise ValueError("solver residuals must be non-negative")

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "solver_id": self.solver_id,
            "solver_parameters": thaw_parameters(self.solver_parameters),
            "iterations": self.iterations,
            "converged": self.converged,
            "tolerance": self.tolerance,
            "learning_rate": self.learning_rate,
            "payoff_min": self.payoff_min,
            "payoff_max": self.payoff_max,
            "marshal_best_response_value": self.marshal_best_response_value,
            "fugitive_best_response_value": self.fugitive_best_response_value,
            "marshal_residual": self.marshal_residual,
            "fugitive_residual": self.fugitive_residual,
            "duality_gap": self.duality_gap,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "MetaSolverDiagnostics":
        solver_id = data.get("solver_id")
        converged = data.get("converged")
        if not isinstance(solver_id, str) or not solver_id:
            raise ValueError("solver diagnostics require a solver_id")
        if not isinstance(converged, bool):
            raise ValueError("solver diagnostics require a convergence flag")
        return cls(
            solver_id=solver_id,
            solver_parameters=freeze_parameters(
                _require_mapping(
                    data.get("solver_parameters", {}), "solver parameters"
                )
            ),
            iterations=_require_non_negative_int(data.get("iterations"), "iterations"),
            converged=converged,
            tolerance=_require_finite(data.get("tolerance"), "tolerance"),
            learning_rate=_require_finite(
                data.get("learning_rate"), "learning_rate"
            ),
            payoff_min=_require_finite(data.get("payoff_min"), "payoff_min"),
            payoff_max=_require_finite(data.get("payoff_max"), "payoff_max"),
            marshal_best_response_value=_require_finite(
                data.get("marshal_best_response_value"),
                "marshal_best_response_value",
            ),
            fugitive_best_response_value=_require_finite(
                data.get("fugitive_best_response_value"),
                "fugitive_best_response_value",
            ),
            marshal_residual=_require_finite(
                data.get("marshal_residual"), "marshal_residual"
            ),
            fugitive_residual=_require_finite(
                data.get("fugitive_residual"), "fugitive_residual"
            ),
            duality_gap=_require_finite(data.get("duality_gap"), "duality_gap"),
        )


@dataclass(frozen=True, slots=True)
class MetaStrategySolution:
    """Approximate equilibrium of one finite empirical restricted game.

    ``lower_bound`` is the Marshal payoff guaranteed by the returned Marshal
    mixture, and ``upper_bound`` is the best restricted Marshal response to
    the returned Fugitive mixture.  ``restricted_game_value`` is their
    midpoint; ``profile_payoff`` is the payoff when the returned mixtures play
    each other.
    """

    marshal_mixture: PolicyMixture
    fugitive_mixture: PolicyMixture
    restricted_game_value: float
    profile_payoff: float
    lower_bound: float
    upper_bound: float
    diagnostics: MetaSolverDiagnostics

    def __post_init__(self) -> None:
        if self.marshal_mixture.role is not Role.MARSHAL:
            raise ValueError("meta solution requires a Marshal mixture")
        if self.fugitive_mixture.role is not Role.FUGITIVE:
            raise ValueError("meta solution requires a Fugitive mixture")
        for name in (
            "restricted_game_value",
            "profile_payoff",
            "lower_bound",
            "upper_bound",
        ):
            object.__setattr__(self, name, _require_finite(getattr(self, name), name))
        if self.lower_bound > self.upper_bound + 1e-10:
            raise ValueError("meta solution lower bound exceeds its upper bound")
        midpoint = 0.5 * (self.lower_bound + self.upper_bound)
        if not math.isclose(
            self.restricted_game_value,
            midpoint,
            rel_tol=1e-10,
            abs_tol=1e-10,
        ):
            raise ValueError(
                "restricted_game_value must be the midpoint of the bounds"
            )

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "marshal_mixture": self.marshal_mixture.to_dict(),
            "fugitive_mixture": self.fugitive_mixture.to_dict(),
            "restricted_game_value": self.restricted_game_value,
            "profile_payoff": self.profile_payoff,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "diagnostics": self.diagnostics.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "MetaStrategySolution":
        return cls(
            marshal_mixture=PolicyMixture.from_dict(
                _require_mapping(data.get("marshal_mixture"), "Marshal mixture")
            ),
            fugitive_mixture=PolicyMixture.from_dict(
                _require_mapping(data.get("fugitive_mixture"), "Fugitive mixture")
            ),
            restricted_game_value=_require_finite(
                data.get("restricted_game_value"), "restricted_game_value"
            ),
            profile_payoff=_require_finite(
                data.get("profile_payoff"), "profile_payoff"
            ),
            lower_bound=_require_finite(data.get("lower_bound"), "lower_bound"),
            upper_bound=_require_finite(data.get("upper_bound"), "upper_bound"),
            diagnostics=MetaSolverDiagnostics.from_dict(
                _require_mapping(data.get("diagnostics"), "solver diagnostics")
            ),
        )


class MetaStrategySolver(Protocol):
    def solve(
        self,
        population: PolicyPopulation,
        payoff_matrix: EmpiricalPayoffMatrix,
    ) -> MetaStrategySolution:
        ...


def _softmax(log_weights: Sequence[float]) -> list[float]:
    largest = max(log_weights)
    weights = [math.exp(weight - largest) for weight in log_weights]
    total = math.fsum(weights)
    result = [weight / total for weight in weights]
    result[-1] += 1.0 - math.fsum(result)
    return result


def _normalized(vector: Sequence[float]) -> list[float]:
    total = math.fsum(vector)
    if total <= 0.0:
        raise ValueError("a strategy accumulator must have positive mass")
    result = [value / total for value in vector]
    result[-1] += 1.0 - math.fsum(result)
    return result


def profile_metrics(
    matrix: Sequence[Sequence[float]],
    marshal: Sequence[float],
    fugitive: Sequence[float],
) -> tuple[float, float, float, float, float, float]:
    row_values = [
        math.fsum(value * fugitive[j] for j, value in enumerate(row))
        for row in matrix
    ]
    column_values = [
        math.fsum(marshal[i] * matrix[i][j] for i in range(len(matrix)))
        for j in range(len(matrix[0]))
    ]
    upper = max(row_values)
    lower = min(column_values)
    profile = math.fsum(marshal[i] * row_values[i] for i in range(len(matrix)))
    marshal_residual = max(0.0, upper - profile)
    fugitive_residual = max(0.0, profile - lower)
    gap = max(0.0, upper - lower)
    return profile, lower, upper, marshal_residual, fugitive_residual, gap


class MultiplicativeWeightsMetaSolver:
    """Auditable no-regret solver for a finite zero-sum payoff matrix.

    Both players update log weights simultaneously.  The Marshal maximizes
    normalized payoff and the Fugitive minimizes it.  Returned mixtures are
    time averages, while convergence is determined by exact pure best
    responses to those averages in the restricted matrix.
    """

    solver_id = MULTIPLICATIVE_WEIGHTS_SOLVER_ID

    def __init__(self, config: MetaSolverConfig | None = None) -> None:
        self.config = config or MetaSolverConfig()

    def solve(
        self,
        population: PolicyPopulation,
        payoff_matrix: EmpiricalPayoffMatrix,
    ) -> MetaStrategySolution:
        if not population.marshal_policies or not population.fugitive_policies:
            raise ValueError("a meta-game requires at least one policy per role")
        payoff_matrix.validate_population(population)
        matrix = payoff_matrix.dense(population)
        row_count = len(matrix)
        column_count = len(matrix[0])
        flat = [value for row in matrix for value in row]
        payoff_min = min(flat)
        payoff_max = max(flat)
        payoff_range = payoff_max - payoff_min
        config = self.config
        rate = config.learning_rate
        if rate is None:
            rate = math.sqrt(
                8.0
                * math.log(max(2, row_count, column_count))
                / config.max_iterations
            )

        if payoff_range == 0.0:
            marshal = [1.0 / row_count] * row_count
            fugitive = [1.0 / column_count] * column_count
            iterations = 0
            converged = True
        else:
            scaled = tuple(
                tuple((value - payoff_min) / payoff_range for value in row)
                for row in matrix
            )
            marshal_logs = [0.0] * row_count
            fugitive_logs = [0.0] * column_count
            marshal_sum = [0.0] * row_count
            fugitive_sum = [0.0] * column_count
            marshal = [1.0 / row_count] * row_count
            fugitive = [1.0 / column_count] * column_count
            converged = False
            iterations = 0
            for iteration in range(1, config.max_iterations + 1):
                current_marshal = _softmax(marshal_logs)
                current_fugitive = _softmax(fugitive_logs)
                for index, probability in enumerate(current_marshal):
                    marshal_sum[index] += probability
                for index, probability in enumerate(current_fugitive):
                    fugitive_sum[index] += probability

                should_check = (
                    iteration >= config.minimum_iterations
                    and (
                        iteration % config.check_interval == 0
                        or iteration == config.max_iterations
                    )
                )
                if should_check:
                    marshal = _normalized(marshal_sum)
                    fugitive = _normalized(fugitive_sum)
                    *_, gap = profile_metrics(matrix, marshal, fugitive)
                    if gap <= config.tolerance:
                        converged = True
                        iterations = iteration
                        break

                row_gains = [
                    math.fsum(
                        scaled[i][j] * current_fugitive[j]
                        for j in range(column_count)
                    )
                    for i in range(row_count)
                ]
                column_payoffs = [
                    math.fsum(
                        current_marshal[i] * scaled[i][j]
                        for i in range(row_count)
                    )
                    for j in range(column_count)
                ]
                for index, gain in enumerate(row_gains):
                    marshal_logs[index] += rate * gain
                for index, payoff in enumerate(column_payoffs):
                    fugitive_logs[index] -= rate * payoff

                if iteration % 1_000 == 0:
                    marshal_offset = max(marshal_logs)
                    fugitive_offset = max(fugitive_logs)
                    marshal_logs = [value - marshal_offset for value in marshal_logs]
                    fugitive_logs = [value - fugitive_offset for value in fugitive_logs]
                iterations = iteration

            marshal = _normalized(marshal_sum)
            fugitive = _normalized(fugitive_sum)

        profile, lower, upper, marshal_residual, fugitive_residual, gap = (
            profile_metrics(matrix, marshal, fugitive)
        )
        value = 0.5 * (lower + upper)
        diagnostics = MetaSolverDiagnostics(
            solver_id=self.solver_id,
            solver_parameters={
                "max_iterations": config.max_iterations,
                "minimum_iterations": config.minimum_iterations,
                "check_interval": config.check_interval,
                "configured_learning_rate": config.learning_rate,
            },
            iterations=iterations,
            converged=converged or gap <= config.tolerance,
            tolerance=config.tolerance,
            learning_rate=rate,
            payoff_min=payoff_min,
            payoff_max=payoff_max,
            marshal_best_response_value=upper,
            fugitive_best_response_value=lower,
            marshal_residual=marshal_residual,
            fugitive_residual=fugitive_residual,
            duality_gap=gap,
        )
        return MetaStrategySolution(
            marshal_mixture=PolicyMixture.from_probabilities(
                Role.MARSHAL,
                population.identifiers(Role.MARSHAL),
                marshal,
            ),
            fugitive_mixture=PolicyMixture.from_probabilities(
                Role.FUGITIVE,
                population.identifiers(Role.FUGITIVE),
                fugitive,
            ),
            restricted_game_value=value,
            profile_payoff=profile,
            lower_bound=lower,
            upper_bound=upper,
            diagnostics=diagnostics,
        )

__all__ = [
    "MULTIPLICATIVE_WEIGHTS_SOLVER_ID",
    "MetaSolverConfig",
    "MetaSolverDiagnostics",
    "MetaStrategySolution",
    "MetaStrategySolver",
    "MultiplicativeWeightsMetaSolver",
    "profile_metrics",
]
