"""Bootstrap intervals and validation reports for PSRO experiments."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Mapping, Sequence, cast

from ..game.model import Role
from ..shared.reproducibility import JSONValue, parse_seed, validate_seed
from ._json_validation import (
    require_mapping as _require_mapping,
    require_sequence as _require_sequence,
)
from .payoff_matrix import PayoffPair
from .population import PolicyIdentifier


BOOTSTRAP_INTERVAL_ID = "deterministic-percentile-bootstrap-v1"


def _quantile(values: Sequence[float], probability: float) -> float:
    """Linearly interpolate one quantile of an already sorted sample."""

    if not values:
        raise ValueError("a quantile requires at least one value")
    position = probability * (len(values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    fraction = position - lower
    return values[lower] * (1.0 - fraction) + values[upper] * fraction


def bootstrap_mean_interval(
    values: Sequence[float],
    *,
    confidence_level: float,
    replicates: int,
    seed: int,
) -> tuple[float, float, float]:
    """Return mean and a deterministic percentile-bootstrap interval."""

    checked = tuple(float(value) for value in values)
    if not checked or any(not math.isfinite(value) for value in checked):
        raise ValueError("bootstrap values must be finite and non-empty")
    rng = random.Random(validate_seed(seed, "bootstrap seed"))
    count = len(checked)
    means = sorted(
        math.fsum(checked[rng.randrange(count)] for _ in range(count)) / count
        for _ in range(replicates)
    )
    tail = (1.0 - confidence_level) / 2.0
    return (
        math.fsum(checked) / count,
        _quantile(means, tail),
        _quantile(means, 1.0 - tail),
    )


@dataclass(frozen=True, slots=True)
class ResponseValidationReport:
    """Fresh paired evidence used for one candidate admission decision."""

    role: Role
    generation: int
    candidate: PolicyIdentifier
    incumbent: PolicyIdentifier
    opponent_draws: tuple[PolicyIdentifier, ...]
    paired_gains: tuple[float, ...]
    mean_gain: float
    confidence_lower_bound: float
    confidence_upper_bound: float
    confidence_level: float
    bootstrap_replicates: int
    bootstrap_seed: int
    minimum_gain: float
    admitted: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "opponent_draws", tuple(self.opponent_draws))
        object.__setattr__(self, "paired_gains", tuple(self.paired_gains))
        if self.role not in (Role.MARSHAL, Role.FUGITIVE):
            raise ValueError("response validation requires a player role")
        if self.candidate.role is not self.role or self.incumbent.role is not self.role:
            raise ValueError("response validation policies have the wrong role")
        if not self.opponent_draws or len(self.opponent_draws) != len(
            self.paired_gains
        ):
            raise ValueError("response validation samples must align")
        opponent_role = (
            Role.FUGITIVE if self.role is Role.MARSHAL else Role.MARSHAL
        )
        if any(policy.role is not opponent_role for policy in self.opponent_draws):
            raise ValueError("response validation opponent has the wrong role")
        if isinstance(self.generation, bool) or self.generation <= 0:
            raise ValueError("response validation generation must be positive")
        for name in (
            "mean_gain",
            "confidence_lower_bound",
            "confidence_upper_bound",
            "confidence_level",
            "minimum_gain",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.confidence_lower_bound > self.confidence_upper_bound:
            raise ValueError("response validation confidence bounds are reversed")
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("response validation confidence level is invalid")
        if self.bootstrap_replicates <= 0:
            raise ValueError("response validation needs bootstrap replicates")
        validate_seed(self.bootstrap_seed, "response bootstrap seed")
        expected = self.confidence_lower_bound > self.minimum_gain
        if self.admitted is not expected:
            raise ValueError("response admission disagrees with its lower bound")

    @property
    def sample_count(self) -> int:
        return len(self.paired_gains)

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "interval_id": BOOTSTRAP_INTERVAL_ID,
            "role": self.role.value,
            "generation": self.generation,
            "candidate": self.candidate.to_dict(),
            "incumbent": self.incumbent.to_dict(),
            "opponent_draws": [item.to_dict() for item in self.opponent_draws],
            "paired_gains": list(self.paired_gains),
            "mean_gain": self.mean_gain,
            "confidence_lower_bound": self.confidence_lower_bound,
            "confidence_upper_bound": self.confidence_upper_bound,
            "confidence_level": self.confidence_level,
            "bootstrap_replicates": self.bootstrap_replicates,
            "bootstrap_seed": str(self.bootstrap_seed),
            "minimum_gain": self.minimum_gain,
            "admitted": self.admitted,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "ResponseValidationReport":
        if data.get("interval_id") != BOOTSTRAP_INTERVAL_ID:
            raise ValueError("unsupported response confidence interval")
        return cls(
            role=Role(cast(str, data.get("role"))),
            generation=cast(int, data.get("generation")),
            candidate=PolicyIdentifier.from_dict(
                _require_mapping(data.get("candidate"), "candidate")
            ),
            incumbent=PolicyIdentifier.from_dict(
                _require_mapping(data.get("incumbent"), "incumbent")
            ),
            opponent_draws=tuple(
                PolicyIdentifier.from_dict(_require_mapping(item, "opponent draw"))
                for item in _require_sequence(
                    data.get("opponent_draws"), "opponent draws"
                )
            ),
            paired_gains=tuple(
                cast(float, value)
                for value in _require_sequence(data.get("paired_gains"), "paired gains")
            ),
            mean_gain=cast(float, data.get("mean_gain")),
            confidence_lower_bound=cast(
                float, data.get("confidence_lower_bound")
            ),
            confidence_upper_bound=cast(
                float, data.get("confidence_upper_bound")
            ),
            confidence_level=cast(float, data.get("confidence_level")),
            bootstrap_replicates=cast(int, data.get("bootstrap_replicates")),
            bootstrap_seed=parse_seed(data.get("bootstrap_seed"), "bootstrap seed"),
            minimum_gain=cast(float, data.get("minimum_gain")),
            admitted=cast(bool, data.get("admitted")),
        )


@dataclass(frozen=True, slots=True)
class FinalHoldoutReport:
    """Fresh evaluation of the final mixed profile, never used for selection."""

    sampled_pairs: tuple[PayoffPair, ...]
    marshal_payoffs: tuple[float, ...]
    mean_marshal_payoff: float
    confidence_lower_bound: float
    confidence_upper_bound: float
    confidence_level: float
    bootstrap_replicates: int
    bootstrap_seed: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "sampled_pairs", tuple(self.sampled_pairs))
        object.__setattr__(self, "marshal_payoffs", tuple(self.marshal_payoffs))
        if not self.sampled_pairs or len(self.sampled_pairs) != len(
            self.marshal_payoffs
        ):
            raise ValueError("final holdout samples must align")
        for name in (
            "mean_marshal_payoff",
            "confidence_lower_bound",
            "confidence_upper_bound",
            "confidence_level",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.confidence_lower_bound > self.confidence_upper_bound:
            raise ValueError("final holdout confidence bounds are reversed")
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("final holdout confidence level is invalid")
        if self.bootstrap_replicates <= 0:
            raise ValueError("final holdout needs bootstrap replicates")
        validate_seed(self.bootstrap_seed, "final holdout bootstrap seed")

    @property
    def sample_count(self) -> int:
        return len(self.marshal_payoffs)

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "interval_id": BOOTSTRAP_INTERVAL_ID,
            "selection_feedback": False,
            "sampled_pairs": [pair.to_dict() for pair in self.sampled_pairs],
            "marshal_payoffs": list(self.marshal_payoffs),
            "mean_marshal_payoff": self.mean_marshal_payoff,
            "confidence_lower_bound": self.confidence_lower_bound,
            "confidence_upper_bound": self.confidence_upper_bound,
            "confidence_level": self.confidence_level,
            "bootstrap_replicates": self.bootstrap_replicates,
            "bootstrap_seed": str(self.bootstrap_seed),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "FinalHoldoutReport":
        if data.get("interval_id") != BOOTSTRAP_INTERVAL_ID:
            raise ValueError("unsupported final holdout confidence interval")
        if data.get("selection_feedback") is not False:
            raise ValueError("final holdout cannot feed policy selection")
        return cls(
            sampled_pairs=tuple(
                PayoffPair.from_dict(_require_mapping(item, "holdout pair"))
                for item in _require_sequence(
                    data.get("sampled_pairs"), "holdout pairs"
                )
            ),
            marshal_payoffs=tuple(
                cast(float, value)
                for value in _require_sequence(
                    data.get("marshal_payoffs"), "holdout payoffs"
                )
            ),
            mean_marshal_payoff=cast(float, data.get("mean_marshal_payoff")),
            confidence_lower_bound=cast(
                float, data.get("confidence_lower_bound")
            ),
            confidence_upper_bound=cast(
                float, data.get("confidence_upper_bound")
            ),
            confidence_level=cast(float, data.get("confidence_level")),
            bootstrap_replicates=cast(int, data.get("bootstrap_replicates")),
            bootstrap_seed=parse_seed(data.get("bootstrap_seed"), "bootstrap seed"),
        )


__all__ = [
    "BOOTSTRAP_INTERVAL_ID",
    "FinalHoldoutReport",
    "ResponseValidationReport",
    "bootstrap_mean_interval",
]
