"""Empirical payoff cells, matrices, and batch evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Protocol

from ..game.model import Role
from ..shared.reproducibility import JSONValue
from ._json_validation import (
    require_finite as _require_finite,
    require_mapping as _require_mapping,
    require_positive_int as _require_positive_int,
    require_sequence as _require_sequence,
)
from .population import PolicyIdentifier, PolicyPopulation, PolicySpec

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

__all__ = [
    "EmpiricalPayoffMatrix",
    "PayoffCollection",
    "PayoffEntry",
    "PayoffEstimate",
    "PayoffEvaluation",
    "PayoffEvaluator",
    "PayoffPair",
    "PayoffRequest",
    "evaluate_missing_payoffs",
]
