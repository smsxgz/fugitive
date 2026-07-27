"""The response-oracle loop and replayable PSRO checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Mapping, Protocol

from ..game.model import Role
from ..shared.reproducibility import JSONValue
from ._json_validation import (
    require_finite as _require_finite,
    require_mapping as _require_mapping,
    require_non_negative_int as _require_non_negative_int,
    require_positive_int as _require_positive_int,
    require_sequence as _require_sequence,
)
from .payoff_matrix import (
    EmpiricalPayoffMatrix,
    PayoffEvaluator,
    PayoffPair,
    evaluate_missing_payoffs,
)
from .population import (
    PolicyIdentifier,
    PolicyMixture,
    PolicyPopulation,
    PolicySpec,
)
from .solver import (
    MetaStrategySolution,
    MetaStrategySolver,
    MultiplicativeWeightsMetaSolver,
    profile_metrics,
)


PSRO_CHECKPOINT_SCHEMA = "fugitive.psro-checkpoint"
PSRO_CHECKPOINT_VERSION = 1

@dataclass(frozen=True, slots=True)
class ResponseOracleRequest:
    role: Role
    generation: int
    population: PolicyPopulation
    payoff_matrix: EmpiricalPayoffMatrix
    meta_solution: MetaStrategySolution

    def __post_init__(self) -> None:
        if not isinstance(self.role, Role):
            raise ValueError("response role must be Fugitive or Marshal")
        _require_positive_int(self.generation, "response generation")

    @property
    def opponent_mixture(self) -> PolicyMixture:
        if self.role is Role.MARSHAL:
            return self.meta_solution.fugitive_mixture
        return self.meta_solution.marshal_mixture

    @property
    def incumbent_mixture(self) -> PolicyMixture:
        if self.role is Role.MARSHAL:
            return self.meta_solution.marshal_mixture
        return self.meta_solution.fugitive_mixture


class ApproximateResponseOracle(Protocol):
    """Produce a new role-correct policy, or ``None`` when none is available."""

    def propose_response(self, request: ResponseOracleRequest) -> PolicySpec | None:
        ...


@dataclass(frozen=True, slots=True)
class GenerationSummary:
    """Audit record for one simultaneous response expansion.

    Response gains are measured against the best incumbent pure response to
    the previous opponent mixture: the prior upper bound for Marshal and the
    prior lower bound for Fugitive.  This avoids treating meta-solver residual
    as improvement by a new policy.
    """

    generation: int
    marshal_response: PolicyIdentifier | None
    fugitive_response: PolicyIdentifier | None
    added_policies: tuple[PolicyIdentifier, ...]
    evaluated_pairs: tuple[PayoffPair, ...]
    prior_restricted_game_value: float
    prior_lower_bound: float
    prior_upper_bound: float
    marshal_response_value: float | None = None
    marshal_response_gain: float | None = None
    fugitive_response_value: float | None = None
    fugitive_response_gain: float | None = None

    def __post_init__(self) -> None:
        _require_positive_int(self.generation, "generation")
        if (
            self.marshal_response is not None
            and self.marshal_response.role is not Role.MARSHAL
        ):
            raise ValueError("Marshal response has the wrong role")
        if (
            self.fugitive_response is not None
            and self.fugitive_response.role is not Role.FUGITIVE
        ):
            raise ValueError("Fugitive response has the wrong role")
        object.__setattr__(self, "added_policies", tuple(self.added_policies))
        object.__setattr__(self, "evaluated_pairs", tuple(self.evaluated_pairs))
        if len(set(self.added_policies)) != len(self.added_policies):
            raise ValueError("generation contains duplicate added policies")
        object.__setattr__(
            self,
            "prior_restricted_game_value",
            _require_finite(
                self.prior_restricted_game_value,
                "prior_restricted_game_value",
            ),
        )
        object.__setattr__(
            self,
            "prior_lower_bound",
            _require_finite(self.prior_lower_bound, "prior_lower_bound"),
        )
        object.__setattr__(
            self,
            "prior_upper_bound",
            _require_finite(self.prior_upper_bound, "prior_upper_bound"),
        )
        if self.prior_lower_bound > self.prior_upper_bound + 1e-10:
            raise ValueError("prior meta-solution bounds are reversed")
        for label in (
            "marshal_response_value",
            "marshal_response_gain",
            "fugitive_response_value",
            "fugitive_response_gain",
        ):
            value = getattr(self, label)
            if value is not None:
                object.__setattr__(self, label, _require_finite(value, label))

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "generation": self.generation,
            "marshal_response": (
                None
                if self.marshal_response is None
                else self.marshal_response.to_dict()
            ),
            "fugitive_response": (
                None
                if self.fugitive_response is None
                else self.fugitive_response.to_dict()
            ),
            "added_policies": [policy.to_dict() for policy in self.added_policies],
            "evaluated_pairs": [pair.to_dict() for pair in self.evaluated_pairs],
            "prior_restricted_game_value": self.prior_restricted_game_value,
            "prior_lower_bound": self.prior_lower_bound,
            "prior_upper_bound": self.prior_upper_bound,
            "marshal_response_value": self.marshal_response_value,
            "marshal_response_gain": self.marshal_response_gain,
            "fugitive_response_value": self.fugitive_response_value,
            "fugitive_response_gain": self.fugitive_response_gain,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "GenerationSummary":
        marshal_response = data.get("marshal_response")
        fugitive_response = data.get("fugitive_response")

        def optional_float(label: str) -> float | None:
            value = data.get(label)
            return None if value is None else _require_finite(value, label)

        return cls(
            generation=_require_positive_int(data.get("generation"), "generation"),
            marshal_response=(
                None
                if marshal_response is None
                else PolicyIdentifier.from_dict(
                    _require_mapping(marshal_response, "Marshal response")
                )
            ),
            fugitive_response=(
                None
                if fugitive_response is None
                else PolicyIdentifier.from_dict(
                    _require_mapping(fugitive_response, "Fugitive response")
                )
            ),
            added_policies=tuple(
                PolicyIdentifier.from_dict(_require_mapping(item, "added policy"))
                for item in _require_sequence(
                    data.get("added_policies"), "added policies"
                )
            ),
            evaluated_pairs=tuple(
                PayoffPair.from_dict(_require_mapping(item, "evaluated pair"))
                for item in _require_sequence(
                    data.get("evaluated_pairs"), "evaluated pairs"
                )
            ),
            prior_restricted_game_value=_require_finite(
                data.get("prior_restricted_game_value"),
                "prior_restricted_game_value",
            ),
            prior_lower_bound=_require_finite(
                data.get("prior_lower_bound"), "prior_lower_bound"
            ),
            prior_upper_bound=_require_finite(
                data.get("prior_upper_bound"), "prior_upper_bound"
            ),
            marshal_response_value=optional_float("marshal_response_value"),
            marshal_response_gain=optional_float("marshal_response_gain"),
            fugitive_response_value=optional_float("fugitive_response_value"),
            fugitive_response_gain=optional_float("fugitive_response_gain"),
        )


@dataclass(frozen=True, slots=True)
class PSROCheckpoint:
    """One complete, replayable restricted-game generation."""

    generation: int
    population: PolicyPopulation
    payoff_matrix: EmpiricalPayoffMatrix
    meta_solution: MetaStrategySolution
    last_generation: GenerationSummary | None = None

    def __post_init__(self) -> None:
        _require_non_negative_int(self.generation, "checkpoint generation")
        if not isinstance(self.population, PolicyPopulation):
            raise ValueError("checkpoint requires a policy population")
        if not isinstance(self.payoff_matrix, EmpiricalPayoffMatrix):
            raise ValueError("checkpoint requires an empirical payoff matrix")
        if not isinstance(self.meta_solution, MetaStrategySolution):
            raise ValueError("checkpoint requires a meta-strategy solution")
        self.payoff_matrix.validate_population(self.population)
        if not self.payoff_matrix.is_complete_for(self.population):
            raise ValueError("checkpoint payoff matrix must be complete")
        marshal_ids = set(self.population.identifiers(Role.MARSHAL))
        fugitive_ids = set(self.population.identifiers(Role.FUGITIVE))
        if {
            entry.identifier for entry in self.meta_solution.marshal_mixture.entries
        } != marshal_ids:
            raise ValueError("checkpoint Marshal mixture does not match its population")
        if {
            entry.identifier for entry in self.meta_solution.fugitive_mixture.entries
        } != fugitive_ids:
            raise ValueError(
                "checkpoint Fugitive mixture does not match its population"
            )
        self._validate_solution_metrics()
        if self.last_generation is not None:
            if self.last_generation.generation != self.generation:
                raise ValueError(
                    "generation summary does not match checkpoint generation"
                )
            all_ids = marshal_ids | fugitive_ids
            responses = {
                identifier
                for identifier in (
                    self.last_generation.marshal_response,
                    self.last_generation.fugitive_response,
                )
                if identifier is not None
            }
            if not responses.issubset(all_ids) or not set(
                self.last_generation.added_policies
            ).issubset(all_ids):
                raise ValueError(
                    "generation summary references a policy outside the population"
                )
            matrix_pairs = {entry.pair for entry in self.payoff_matrix.entries}
            if not set(self.last_generation.evaluated_pairs).issubset(matrix_pairs):
                raise ValueError(
                    "generation summary references an absent payoff cell"
                )
        elif self.generation != 0:
            raise ValueError("non-initial checkpoint requires a generation summary")

    def _validate_solution_metrics(self) -> None:
        matrix = self.payoff_matrix.dense(self.population)
        marshal = [
            self.meta_solution.marshal_mixture.probability(identifier)
            for identifier in self.population.identifiers(Role.MARSHAL)
        ]
        fugitive = [
            self.meta_solution.fugitive_mixture.probability(identifier)
            for identifier in self.population.identifiers(Role.FUGITIVE)
        ]
        profile, lower, upper, marshal_residual, fugitive_residual, gap = (
            profile_metrics(matrix, marshal, fugitive)
        )
        expected = {
            "restricted_game_value": (lower + upper) / 2.0,
            "profile_payoff": profile,
            "lower_bound": lower,
            "upper_bound": upper,
            "marshal_best_response_value": upper,
            "fugitive_best_response_value": lower,
            "marshal_residual": marshal_residual,
            "fugitive_residual": fugitive_residual,
            "duality_gap": gap,
        }
        actual = {
            "restricted_game_value": self.meta_solution.restricted_game_value,
            "profile_payoff": self.meta_solution.profile_payoff,
            "lower_bound": self.meta_solution.lower_bound,
            "upper_bound": self.meta_solution.upper_bound,
            "marshal_best_response_value": (
                self.meta_solution.diagnostics.marshal_best_response_value
            ),
            "fugitive_best_response_value": (
                self.meta_solution.diagnostics.fugitive_best_response_value
            ),
            "marshal_residual": self.meta_solution.diagnostics.marshal_residual,
            "fugitive_residual": self.meta_solution.diagnostics.fugitive_residual,
            "duality_gap": self.meta_solution.diagnostics.duality_gap,
        }
        for label, value in expected.items():
            if not math.isclose(
                value,
                actual[label],
                rel_tol=1e-10,
                abs_tol=1e-10,
            ):
                raise ValueError(
                    f"checkpoint meta solution has inconsistent {label}"
                )

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "schema": PSRO_CHECKPOINT_SCHEMA,
            "version": PSRO_CHECKPOINT_VERSION,
            "generation": self.generation,
            "population": self.population.to_dict(),
            "payoff_matrix": self.payoff_matrix.to_dict(),
            "meta_solution": self.meta_solution.to_dict(),
            "last_generation": (
                None
                if self.last_generation is None
                else self.last_generation.to_dict()
            ),
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(
            self.to_dict(),
            allow_nan=False,
            ensure_ascii=True,
            indent=indent,
            sort_keys=True,
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "PSROCheckpoint":
        if data.get("schema") != PSRO_CHECKPOINT_SCHEMA:
            raise ValueError("unsupported PSRO checkpoint schema")
        if data.get("version") != PSRO_CHECKPOINT_VERSION:
            raise ValueError("unsupported PSRO checkpoint version")
        last_generation = data.get("last_generation")
        return cls(
            generation=_require_non_negative_int(
                data.get("generation"), "checkpoint generation"
            ),
            population=PolicyPopulation.from_dict(
                _require_mapping(data.get("population"), "policy population")
            ),
            payoff_matrix=EmpiricalPayoffMatrix.from_dict(
                _require_mapping(data.get("payoff_matrix"), "payoff matrix")
            ),
            meta_solution=MetaStrategySolution.from_dict(
                _require_mapping(data.get("meta_solution"), "meta solution")
            ),
            last_generation=(
                None
                if last_generation is None
                else GenerationSummary.from_dict(
                    _require_mapping(last_generation, "generation summary")
                )
            ),
        )

    @classmethod
    def from_json(cls, payload: str) -> "PSROCheckpoint":
        try:
            data = json.loads(payload)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("invalid PSRO checkpoint JSON") from exc
        return cls.from_dict(_require_mapping(data, "PSRO checkpoint"))


def initialize_psro(
    population: PolicyPopulation,
    *,
    evaluator: PayoffEvaluator | None,
    payoff_matrix: EmpiricalPayoffMatrix | None = None,
    meta_solver: MetaStrategySolver | None = None,
) -> PSROCheckpoint:
    """Fill the initial restricted matrix and solve generation zero."""

    matrix = payoff_matrix or EmpiricalPayoffMatrix()
    collected = evaluate_missing_payoffs(population, matrix, evaluator)
    solver = meta_solver or MultiplicativeWeightsMetaSolver()
    solution = solver.solve(population, collected.matrix)
    return PSROCheckpoint(0, population, collected.matrix, solution)


def _validate_response(
    response: PolicySpec | None, role: Role
) -> PolicySpec | None:
    if response is None:
        return None
    if not isinstance(response, PolicySpec):
        raise ValueError("response oracle must return PolicySpec or None")
    if response.role is not role:
        raise ValueError(f"{role.value} response oracle returned a wrong-role policy")
    return response


def _expected_new_marshal_value(
    policy: PolicyIdentifier,
    opponent: PolicyMixture,
    matrix: EmpiricalPayoffMatrix,
) -> float:
    return math.fsum(
        entry.probability * matrix.marshal_payoff(policy, entry.identifier)
        for entry in opponent.entries
    )


def _expected_new_fugitive_value(
    policy: PolicyIdentifier,
    opponent: PolicyMixture,
    matrix: EmpiricalPayoffMatrix,
) -> float:
    return math.fsum(
        entry.probability * matrix.marshal_payoff(entry.identifier, policy)
        for entry in opponent.entries
    )


def run_psro_iteration(
    checkpoint: PSROCheckpoint,
    *,
    marshal_oracle: ApproximateResponseOracle,
    fugitive_oracle: ApproximateResponseOracle,
    evaluator: PayoffEvaluator | None,
    meta_solver: MetaStrategySolver | None = None,
) -> PSROCheckpoint:
    """Run one simultaneous two-sided response-and-expand generation."""

    generation = checkpoint.generation + 1
    marshal_response = _validate_response(
        marshal_oracle.propose_response(
            ResponseOracleRequest(
                Role.MARSHAL,
                generation,
                checkpoint.population,
                checkpoint.payoff_matrix,
                checkpoint.meta_solution,
            )
        ),
        Role.MARSHAL,
    )
    fugitive_response = _validate_response(
        fugitive_oracle.propose_response(
            ResponseOracleRequest(
                Role.FUGITIVE,
                generation,
                checkpoint.population,
                checkpoint.payoff_matrix,
                checkpoint.meta_solution,
            )
        ),
        Role.FUGITIVE,
    )

    proposed = tuple(
        response
        for response in (marshal_response, fugitive_response)
        if response is not None
    )
    old_ids = {
        policy.identifier
        for policy in (
            *checkpoint.population.marshal_policies,
            *checkpoint.population.fugitive_policies,
        )
    }
    expanded = checkpoint.population.with_policies(*proposed)
    added = tuple(
        policy.identifier for policy in proposed if policy.identifier not in old_ids
    )
    collected = evaluate_missing_payoffs(
        expanded, checkpoint.payoff_matrix, evaluator
    )
    if added:
        solver = meta_solver or MultiplicativeWeightsMetaSolver()
        solution = solver.solve(expanded, collected.matrix)
    else:
        solution = checkpoint.meta_solution

    prior_value = checkpoint.meta_solution.restricted_game_value
    marshal_value = (
        None
        if marshal_response is None or marshal_response.identifier not in added
        else _expected_new_marshal_value(
            marshal_response.identifier,
            checkpoint.meta_solution.fugitive_mixture,
            collected.matrix,
        )
    )
    fugitive_value = (
        None
        if fugitive_response is None or fugitive_response.identifier not in added
        else _expected_new_fugitive_value(
            fugitive_response.identifier,
            checkpoint.meta_solution.marshal_mixture,
            collected.matrix,
        )
    )
    summary = GenerationSummary(
        generation=generation,
        marshal_response=(
            None if marshal_response is None else marshal_response.identifier
        ),
        fugitive_response=(
            None if fugitive_response is None else fugitive_response.identifier
        ),
        added_policies=added,
        evaluated_pairs=collected.requested_pairs,
        prior_restricted_game_value=prior_value,
        prior_lower_bound=checkpoint.meta_solution.lower_bound,
        prior_upper_bound=checkpoint.meta_solution.upper_bound,
        marshal_response_value=marshal_value,
        marshal_response_gain=(
            None
            if marshal_value is None
            else marshal_value - checkpoint.meta_solution.upper_bound
        ),
        fugitive_response_value=fugitive_value,
        fugitive_response_gain=(
            None
            if fugitive_value is None
            else checkpoint.meta_solution.lower_bound - fugitive_value
        ),
    )
    return PSROCheckpoint(
        generation,
        expanded,
        collected.matrix,
        solution,
        summary,
    )


@dataclass(frozen=True, slots=True)
class PSRORunResult:
    """The immutable checkpoints produced by a bounded PSRO run."""

    checkpoints: tuple[PSROCheckpoint, ...]
    stop_reason: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "checkpoints", tuple(self.checkpoints))
        if not self.checkpoints:
            raise ValueError("a PSRO run must contain its initial checkpoint")
        if self.stop_reason not in {"generation_limit", "no_new_policies"}:
            raise ValueError("unknown PSRO stop reason")

    @property
    def final_checkpoint(self) -> PSROCheckpoint:
        return self.checkpoints[-1]


def run_psro(
    initial_checkpoint: PSROCheckpoint,
    *,
    generations: int,
    marshal_oracle: ApproximateResponseOracle,
    fugitive_oracle: ApproximateResponseOracle,
    evaluator: PayoffEvaluator | None,
    meta_solver: MetaStrategySolver | None = None,
    stop_when_no_new_policies: bool = True,
) -> PSRORunResult:
    """Run up to ``generations`` expansions from an existing checkpoint."""

    _require_non_negative_int(generations, "generations")
    checkpoints = [initial_checkpoint]
    stop_reason = "generation_limit"
    for _ in range(generations):
        checkpoint = run_psro_iteration(
            checkpoints[-1],
            marshal_oracle=marshal_oracle,
            fugitive_oracle=fugitive_oracle,
            evaluator=evaluator,
            meta_solver=meta_solver,
        )
        checkpoints.append(checkpoint)
        if (
            stop_when_no_new_policies
            and checkpoint.last_generation is not None
            and not checkpoint.last_generation.added_policies
        ):
            stop_reason = "no_new_policies"
            break
    return PSRORunResult(tuple(checkpoints), stop_reason)

__all__ = [
    "ApproximateResponseOracle",
    "GenerationSummary",
    "PSRO_CHECKPOINT_SCHEMA",
    "PSRO_CHECKPOINT_VERSION",
    "PSROCheckpoint",
    "PSRORunResult",
    "ResponseOracleRequest",
    "initialize_psro",
    "run_psro",
    "run_psro_iteration",
]
