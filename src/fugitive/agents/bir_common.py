"""Small shared definitions for the belief-informed random policies."""

from __future__ import annotations

from collections.abc import Hashable, Mapping
import math
from typing import TypeVar


DEFAULT_EPSILON = 0.15
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MANHUNT_EPSILON = 0.10
DEFAULT_MANHUNT_ALPHA = 2.0


ChoiceT = TypeVar("ChoiceT", bound=Hashable)


def normalized(weights: Mapping[ChoiceT, float]) -> dict[ChoiceT, float]:
    """Normalize the positive entries of a weighted choice map."""

    positive = {
        action: weight for action, weight in weights.items() if weight > 0.0
    }
    total = sum(positive.values())
    if total <= 0.0:
        return {}
    return {action: weight / total for action, weight in positive.items()}


def validate_manhunt_parameters(epsilon: float, alpha: float) -> None:
    if not math.isfinite(epsilon) or not 0.0 <= epsilon <= 1.0:
        raise ValueError("manhunt_epsilon must be between zero and one")
    if not math.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("manhunt_alpha must be positive")


__all__ = [
    "DEFAULT_EPSILON",
    "DEFAULT_MANHUNT_ALPHA",
    "DEFAULT_MANHUNT_EPSILON",
    "DEFAULT_TEMPERATURE",
    "normalized",
    "validate_manhunt_parameters",
]
