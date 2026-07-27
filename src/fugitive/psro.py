"""Policy-Space Response Oracles for empirical zero-sum strategy research.

This module deliberately knows nothing about the Fugitive game engine or the
agent registry.  A policy is an immutable, role-qualified specification, and
an external evaluator supplies empirical payoffs for requested policy pairs.
The PSRO loop only maintains the restricted policy population, solves its
empirical zero-sum meta-game, and asks role-specific approximate-response
oracles for possible expansions.

The built-in meta solver is simultaneous multiplicative weights.  It reports
an exact best-response gap *inside the current empirical policy population*.
Neither that gap nor the returned value is a claim about a Nash equilibrium of
the full Fugitive game.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import random
from typing import Iterable, Mapping, Protocol, Sequence, cast

from .model import Role
from .reproducibility import (
    FrozenJSONValue,
    JSONValue,
    freeze_parameters,
    thaw_parameters,
)


PSRO_CHECKPOINT_SCHEMA = "fugitive.psro-checkpoint"
PSRO_CHECKPOINT_VERSION = 1
MULTIPLICATIVE_WEIGHTS_SOLVER_ID = "simultaneous-multiplicative-weights-v1"


def _require_non_negative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _require_positive_int(value: object, label: str) -> int:
    value = _require_non_negative_int(value, label)
    if value == 0:
        raise ValueError(f"{label} must be positive")
    return value


def _require_finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


def _require_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return cast(Mapping[str, object], value)


def _require_sequence(value: object, label: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{label} must be a JSON array")
    return cast(Sequence[object], value)


@dataclass(frozen=True, slots=True)
class PolicyIdentifier:
    """A stable policy name qualified by the role that can use it."""

    role: Role
    name: str

    def __post_init__(self) -> None:
        if not isinstance(self.role, Role):
            raise ValueError("policy role must be Fugitive or Marshal")
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("policy name must be a non-empty string")
        if self.name != self.name.strip():
            raise ValueError("policy name must not have surrounding whitespace")

    def to_dict(self) -> dict[str, JSONValue]:
        return {"role": self.role.value, "name": self.name}

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "PolicyIdentifier":
        role = data.get("role")
        name = data.get("name")
        if not isinstance(role, str) or not isinstance(name, str):
            raise ValueError("policy identifier requires string role and name")
        return cls(Role(role), name)


@dataclass(frozen=True, slots=True)
class PolicySpec:
    """An immutable construction specification for one pure policy."""

    identifier: PolicyIdentifier
    parameters: Mapping[str, FrozenJSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.identifier, PolicyIdentifier):
            raise ValueError("policy spec requires a PolicyIdentifier")
        object.__setattr__(self, "parameters", freeze_parameters(self.parameters))

    @property
    def role(self) -> Role:
        return self.identifier.role

    @classmethod
    def create(
        cls,
        role: Role,
        name: str,
        parameters: Mapping[str, object] | None = None,
    ) -> "PolicySpec":
        return cls(PolicyIdentifier(role, name), parameters or {})

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "identifier": self.identifier.to_dict(),
            "parameters": thaw_parameters(self.parameters),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "PolicySpec":
        identifier = PolicyIdentifier.from_dict(
            _require_mapping(data.get("identifier"), "policy identifier")
        )
        parameters = _require_mapping(
            data.get("parameters", {}), "policy parameters"
        )
        return cls(identifier, parameters)


def _validate_policy_bucket(
    policies: tuple[PolicySpec, ...], role: Role
) -> None:
    seen: set[PolicyIdentifier] = set()
    for policy in policies:
        if not isinstance(policy, PolicySpec) or policy.role is not role:
            raise ValueError(f"{role.value} population contains a wrong-role policy")
        if policy.identifier in seen:
            raise ValueError(f"duplicate policy identifier: {policy.identifier.name}")
        seen.add(policy.identifier)


@dataclass(frozen=True, slots=True)
class PolicyPopulation:
    """Role-separated ordered pure policies in one restricted game."""

    fugitive_policies: tuple[PolicySpec, ...] = ()
    marshal_policies: tuple[PolicySpec, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "fugitive_policies", tuple(self.fugitive_policies))
        object.__setattr__(self, "marshal_policies", tuple(self.marshal_policies))
        _validate_policy_bucket(self.fugitive_policies, Role.FUGITIVE)
        _validate_policy_bucket(self.marshal_policies, Role.MARSHAL)

    @classmethod
    def create(
        cls,
        *,
        fugitive: Iterable[PolicySpec] = (),
        marshal: Iterable[PolicySpec] = (),
    ) -> "PolicyPopulation":
        return cls(tuple(fugitive), tuple(marshal))

    def policies(self, role: Role) -> tuple[PolicySpec, ...]:
        if role is Role.FUGITIVE:
            return self.fugitive_policies
        if role is Role.MARSHAL:
            return self.marshal_policies
        raise ValueError("population role must be Fugitive or Marshal")

    def identifiers(self, role: Role) -> tuple[PolicyIdentifier, ...]:
        return tuple(policy.identifier for policy in self.policies(role))

    def policy(self, identifier: PolicyIdentifier) -> PolicySpec:
        for policy in self.policies(identifier.role):
            if policy.identifier == identifier:
                return policy
        raise KeyError(identifier)

    def with_policies(self, *policies: PolicySpec) -> "PolicyPopulation":
        fugitive = list(self.fugitive_policies)
        marshal = list(self.marshal_policies)
        existing = {
            policy.identifier: policy
            for policy in (*self.fugitive_policies, *self.marshal_policies)
        }
        for policy in policies:
            if not isinstance(policy, PolicySpec):
                raise ValueError("population additions must be PolicySpec values")
            previous = existing.get(policy.identifier)
            if previous is not None:
                if previous != policy:
                    raise ValueError(
                        "a policy identifier cannot be reused with new parameters: "
                        f"{policy.identifier.name}"
                    )
                continue
            existing[policy.identifier] = policy
            if policy.role is Role.FUGITIVE:
                fugitive.append(policy)
            else:
                marshal.append(policy)
        return PolicyPopulation(tuple(fugitive), tuple(marshal))

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "fugitive": [policy.to_dict() for policy in self.fugitive_policies],
            "marshal": [policy.to_dict() for policy in self.marshal_policies],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "PolicyPopulation":
        fugitive = tuple(
            PolicySpec.from_dict(_require_mapping(item, "Fugitive policy"))
            for item in _require_sequence(data.get("fugitive"), "Fugitive policies")
        )
        marshal = tuple(
            PolicySpec.from_dict(_require_mapping(item, "Marshal policy"))
            for item in _require_sequence(data.get("marshal"), "Marshal policies")
        )
        return cls(fugitive, marshal)


@dataclass(frozen=True, slots=True)
class PolicyProbability:
    identifier: PolicyIdentifier
    probability: float

    def __post_init__(self) -> None:
        if not isinstance(self.identifier, PolicyIdentifier):
            raise ValueError("mixture entry requires a policy identifier")
        probability = _require_finite(self.probability, "policy probability")
        if probability < 0.0:
            raise ValueError("policy probability must be non-negative")
        object.__setattr__(self, "probability", probability)

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "identifier": self.identifier.to_dict(),
            "probability": self.probability,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "PolicyProbability":
        return cls(
            PolicyIdentifier.from_dict(
                _require_mapping(data.get("identifier"), "mixture identifier")
            ),
            _require_finite(data.get("probability"), "policy probability"),
        )


@dataclass(frozen=True, slots=True)
class PolicyMixture:
    """An immutable mixed policy for exactly one role."""

    role: Role
    entries: tuple[PolicyProbability, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.role, Role):
            raise ValueError("mixture role must be Fugitive or Marshal")
        object.__setattr__(self, "entries", tuple(self.entries))
        if not self.entries:
            raise ValueError("a policy mixture cannot be empty")
        seen: set[PolicyIdentifier] = set()
        for entry in self.entries:
            if entry.identifier.role is not self.role:
                raise ValueError("mixture entry has the wrong role")
            if entry.identifier in seen:
                raise ValueError("mixture contains a duplicate policy")
            seen.add(entry.identifier)
        total = math.fsum(entry.probability for entry in self.entries)
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("policy mixture probabilities must sum to one")

    @classmethod
    def from_probabilities(
        cls,
        role: Role,
        identifiers: Sequence[PolicyIdentifier],
        probabilities: Sequence[float],
    ) -> "PolicyMixture":
        if len(identifiers) != len(probabilities) or not identifiers:
            raise ValueError("mixture identifiers and probabilities must align")
        checked = [
            _require_finite(probability, "policy probability")
            for probability in probabilities
        ]
        if any(probability < 0.0 for probability in checked):
            raise ValueError("policy probability must be non-negative")
        total = math.fsum(checked)
        if total <= 0.0:
            raise ValueError("a policy mixture must have positive mass")
        normalized = [probability / total for probability in checked]
        normalized[-1] += 1.0 - math.fsum(normalized)
        return cls(
            role,
            tuple(
                PolicyProbability(identifier, probability)
                for identifier, probability in zip(
                    identifiers, normalized, strict=True
                )
            ),
        )

    def probability(self, identifier: PolicyIdentifier) -> float:
        if identifier.role is not self.role:
            return 0.0
        return next(
            (
                entry.probability
                for entry in self.entries
                if entry.identifier == identifier
            ),
            0.0,
        )

    @property
    def support(self) -> tuple[PolicyIdentifier, ...]:
        return tuple(
            entry.identifier for entry in self.entries if entry.probability > 0.0
        )

    def sample(self, rng: random.Random) -> PolicyIdentifier:
        threshold = rng.random()
        cumulative = 0.0
        for entry in self.entries:
            cumulative += entry.probability
            if threshold < cumulative:
                return entry.identifier
        return self.entries[-1].identifier

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "role": self.role.value,
            "entries": [entry.to_dict() for entry in self.entries],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "PolicyMixture":
        role = data.get("role")
        if not isinstance(role, str):
            raise ValueError("mixture requires a string role")
        entries = tuple(
            PolicyProbability.from_dict(_require_mapping(item, "mixture entry"))
            for item in _require_sequence(data.get("entries"), "mixture entries")
        )
        return cls(Role(role), entries)


@dataclass(frozen=True, slots=True)
class PayoffPair:
    """One matrix cell, with Marshal as row and Fugitive as column."""

    marshal: PolicyIdentifier
    fugitive: PolicyIdentifier

    def __post_init__(self) -> None:
        if not isinstance(self.marshal, PolicyIdentifier) or not isinstance(
            self.fugitive, PolicyIdentifier
        ):
            raise ValueError("payoff pair requires policy identifiers")
        if self.marshal.role is not Role.MARSHAL:
            raise ValueError("payoff row must be a Marshal policy")
        if self.fugitive.role is not Role.FUGITIVE:
            raise ValueError("payoff column must be a Fugitive policy")

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "marshal": self.marshal.to_dict(),
            "fugitive": self.fugitive.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "PayoffPair":
        return cls(
            PolicyIdentifier.from_dict(
                _require_mapping(data.get("marshal"), "Marshal payoff policy")
            ),
            PolicyIdentifier.from_dict(
                _require_mapping(data.get("fugitive"), "Fugitive payoff policy")
            ),
        )


@dataclass(frozen=True, slots=True)
class PayoffEstimate:
    """An empirical estimate of the Marshal's zero-sum payoff."""

    marshal_payoff: float
    sample_count: int = 1
    standard_error: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "marshal_payoff",
            _require_finite(self.marshal_payoff, "Marshal payoff"),
        )
        _require_positive_int(self.sample_count, "payoff sample_count")
        if self.standard_error is not None:
            standard_error = _require_finite(
                self.standard_error, "payoff standard_error"
            )
            if standard_error < 0.0:
                raise ValueError("payoff standard_error must be non-negative")
            object.__setattr__(self, "standard_error", standard_error)

    @property
    def fugitive_payoff(self) -> float:
        return -self.marshal_payoff

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "marshal_payoff": self.marshal_payoff,
            "sample_count": self.sample_count,
            "standard_error": self.standard_error,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "PayoffEstimate":
        standard_error = data.get("standard_error")
        return cls(
            _require_finite(data.get("marshal_payoff"), "Marshal payoff"),
            _require_positive_int(data.get("sample_count"), "payoff sample_count"),
            (
                None
                if standard_error is None
                else _require_finite(standard_error, "payoff standard_error")
            ),
        )


@dataclass(frozen=True, slots=True)
class PayoffEntry:
    pair: PayoffPair
    estimate: PayoffEstimate

    def __post_init__(self) -> None:
        if not isinstance(self.pair, PayoffPair) or not isinstance(
            self.estimate, PayoffEstimate
        ):
            raise ValueError("payoff entry requires a pair and an estimate")

    def to_dict(self) -> dict[str, JSONValue]:
        return {"pair": self.pair.to_dict(), "estimate": self.estimate.to_dict()}

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "PayoffEntry":
        return cls(
            PayoffPair.from_dict(_require_mapping(data.get("pair"), "payoff pair")),
            PayoffEstimate.from_dict(
                _require_mapping(data.get("estimate"), "payoff estimate")
            ),
        )


def _pair_sort_key(pair: PayoffPair) -> tuple[str, str]:
    return pair.marshal.name, pair.fugitive.name


@dataclass(frozen=True, slots=True)
class EmpiricalPayoffMatrix:
    """A possibly incomplete immutable cache of empirical payoff cells."""

    entries: tuple[PayoffEntry, ...] = ()

    def __post_init__(self) -> None:
        entries = tuple(self.entries)
        seen: set[PayoffPair] = set()
        for entry in entries:
            if not isinstance(entry, PayoffEntry):
                raise ValueError("payoff matrix entries must be PayoffEntry values")
            if entry.pair in seen:
                raise ValueError("payoff matrix contains a duplicate cell")
            seen.add(entry.pair)
        object.__setattr__(
            self,
            "entries",
            tuple(sorted(entries, key=lambda entry: _pair_sort_key(entry.pair))),
        )

    def estimate(self, pair: PayoffPair) -> PayoffEstimate | None:
        return next(
            (entry.estimate for entry in self.entries if entry.pair == pair),
            None,
        )

    def marshal_payoff(
        self, marshal: PolicyIdentifier, fugitive: PolicyIdentifier
    ) -> float:
        pair = PayoffPair(marshal, fugitive)
        estimate = self.estimate(pair)
        if estimate is None:
            raise KeyError(pair)
        return estimate.marshal_payoff

    def missing_pairs(self, population: PolicyPopulation) -> tuple[PayoffPair, ...]:
        known = {entry.pair for entry in self.entries}
        return tuple(
            PayoffPair(marshal.identifier, fugitive.identifier)
            for marshal in population.marshal_policies
            for fugitive in population.fugitive_policies
            if PayoffPair(marshal.identifier, fugitive.identifier) not in known
        )

    def is_complete_for(self, population: PolicyPopulation) -> bool:
        return not self.missing_pairs(population)

    def with_evaluations(
        self, evaluations: Iterable["PayoffEvaluation"]
    ) -> "EmpiricalPayoffMatrix":
        by_pair = {entry.pair: entry.estimate for entry in self.entries}
        for evaluation in evaluations:
            if not isinstance(evaluation, PayoffEvaluation):
                raise ValueError("matrix updates must be PayoffEvaluation values")
            previous = by_pair.get(evaluation.pair)
            if previous is not None and previous != evaluation.estimate:
                raise ValueError("an existing payoff cannot be silently overwritten")
            by_pair[evaluation.pair] = evaluation.estimate
        return EmpiricalPayoffMatrix(
            tuple(PayoffEntry(pair, estimate) for pair, estimate in by_pair.items())
        )

    def dense(self, population: PolicyPopulation) -> tuple[tuple[float, ...], ...]:
        missing = self.missing_pairs(population)
        if missing:
            raise ValueError(
                f"empirical payoff matrix is missing {len(missing)} cells"
            )
        return tuple(
            tuple(
                self.marshal_payoff(marshal.identifier, fugitive.identifier)
                for fugitive in population.fugitive_policies
            )
            for marshal in population.marshal_policies
        )

    def validate_population(self, population: PolicyPopulation) -> None:
        marshal_ids = set(population.identifiers(Role.MARSHAL))
        fugitive_ids = set(population.identifiers(Role.FUGITIVE))
        for entry in self.entries:
            if (
                entry.pair.marshal not in marshal_ids
                or entry.pair.fugitive not in fugitive_ids
            ):
                raise ValueError(
                    "payoff matrix contains a policy outside its population"
                )

    def to_dict(self) -> dict[str, JSONValue]:
        return {"entries": [entry.to_dict() for entry in self.entries]}

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "EmpiricalPayoffMatrix":
        return cls(
            tuple(
                PayoffEntry.from_dict(_require_mapping(item, "payoff entry"))
                for item in _require_sequence(data.get("entries"), "payoff entries")
            )
        )


@dataclass(frozen=True, slots=True)
class PayoffRequest:
    """A self-contained matrix cell request suitable for parallel workers."""

    pair: PayoffPair
    marshal_policy: PolicySpec
    fugitive_policy: PolicySpec

    def __post_init__(self) -> None:
        if self.pair.marshal != self.marshal_policy.identifier:
            raise ValueError("payoff request has a mismatched Marshal policy")
        if self.pair.fugitive != self.fugitive_policy.identifier:
            raise ValueError("payoff request has a mismatched Fugitive policy")


@dataclass(frozen=True, slots=True)
class PayoffEvaluation:
    pair: PayoffPair
    estimate: PayoffEstimate

    def __post_init__(self) -> None:
        if not isinstance(self.pair, PayoffPair) or not isinstance(
            self.estimate, PayoffEstimate
        ):
            raise ValueError("payoff evaluation requires a pair and an estimate")


class PayoffEvaluator(Protocol):
    """Batch payoff interface; implementations may evaluate cells in parallel."""

    def evaluate(
        self, requests: tuple[PayoffRequest, ...]
    ) -> Iterable[PayoffEvaluation]:
        ...


@dataclass(frozen=True, slots=True)
class PayoffCollection:
    matrix: EmpiricalPayoffMatrix
    requested_pairs: tuple[PayoffPair, ...]


def evaluate_missing_payoffs(
    population: PolicyPopulation,
    matrix: EmpiricalPayoffMatrix,
    evaluator: PayoffEvaluator | None,
) -> PayoffCollection:
    """Evaluate exactly the matrix cells absent from ``matrix``.

    The evaluator receives one deterministic batch and may process or return
    it in any order.  A malformed, partial, or extra result batch is rejected
    before a new immutable matrix is produced.
    """

    matrix.validate_population(population)
    pairs = matrix.missing_pairs(population)
    if not pairs:
        return PayoffCollection(matrix, ())
    if evaluator is None:
        raise ValueError("a payoff evaluator is required for missing cells")
    requests = tuple(
        PayoffRequest(
            pair,
            population.policy(pair.marshal),
            population.policy(pair.fugitive),
        )
        for pair in pairs
    )
    evaluations = tuple(evaluator.evaluate(requests))
    expected = set(pairs)
    seen: set[PayoffPair] = set()
    for evaluation in evaluations:
        if not isinstance(evaluation, PayoffEvaluation):
            raise ValueError("payoff evaluator returned a non-evaluation value")
        if evaluation.pair not in expected:
            raise ValueError("payoff evaluator returned an unrequested cell")
        if evaluation.pair in seen:
            raise ValueError("payoff evaluator returned a duplicate cell")
        seen.add(evaluation.pair)
    missing_results = expected - seen
    if missing_results:
        raise ValueError(
            f"payoff evaluator omitted {len(missing_results)} requested cells"
        )
    return PayoffCollection(matrix.with_evaluations(evaluations), pairs)


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


def _profile_metrics(
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
                    *_, gap = _profile_metrics(matrix, marshal, fugitive)
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
            _profile_metrics(matrix, marshal, fugitive)
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
            _profile_metrics(matrix, marshal, fugitive)
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
    "EmpiricalPayoffMatrix",
    "GenerationSummary",
    "MULTIPLICATIVE_WEIGHTS_SOLVER_ID",
    "MetaSolverConfig",
    "MetaSolverDiagnostics",
    "MetaStrategySolution",
    "MetaStrategySolver",
    "MultiplicativeWeightsMetaSolver",
    "PSRO_CHECKPOINT_SCHEMA",
    "PSRO_CHECKPOINT_VERSION",
    "PSROCheckpoint",
    "PSRORunResult",
    "PayoffCollection",
    "PayoffEntry",
    "PayoffEstimate",
    "PayoffEvaluation",
    "PayoffEvaluator",
    "PayoffPair",
    "PayoffRequest",
    "PolicyIdentifier",
    "PolicyMixture",
    "PolicyPopulation",
    "PolicyProbability",
    "PolicySpec",
    "ResponseOracleRequest",
    "evaluate_missing_payoffs",
    "initialize_psro",
    "run_psro",
    "run_psro_iteration",
]
