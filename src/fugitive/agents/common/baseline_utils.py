"""Shared probability and information-set helpers for the new baselines."""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
import math
import random
from statistics import fmean, pstdev
from typing import TypeVar

from fugitive.game.model import Observation
from fugitive.game.rules import PILE_CARDS


ChoiceT = TypeVar("ChoiceT", bound=Hashable)


def visible_card_identities(observation: Observation) -> frozenset[int]:
    """Return identities that this role knows cannot remain in a draw pile."""

    visible = set(observation.hand)
    for slot in observation.route:
        if slot.hideout is not None:
            visible.add(slot.hideout)
        if slot.sprint_cards is not None:
            visible.update(slot.sprint_cards)
    return frozenset(visible)


def possible_draw_cards(observation: Observation, pile: int) -> tuple[int, ...]:
    """Return an information-set support for the next card from ``pile``.

    Opponent-private cards remain in the support.  The helper deliberately
    does not inspect or reconstruct the engine's actual pile order.
    """

    if pile not in observation.legal_draw_piles:
        return ()
    visible = visible_card_identities(observation)
    possible = tuple(card for card in PILE_CARDS[pile] if card not in visible)
    return possible or tuple(PILE_CARDS[pile])


def epsilon_softmax(
    scores: Mapping[ChoiceT, float],
    *,
    epsilon: float = 0.15,
    temperature: float = 0.7,
) -> dict[ChoiceT, float]:
    """Standardize scores and return an epsilon-softmax distribution."""

    if not scores:
        raise ValueError("epsilon_softmax requires at least one action")
    if not math.isfinite(epsilon) or not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon must be finite and in [0, 1]")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be positive and finite")
    if any(not math.isfinite(score) for score in scores.values()):
        raise ValueError("all action scores must be finite")

    actions = tuple(scores)
    values = tuple(float(scores[action]) for action in actions)
    spread = pstdev(values)
    if spread <= 1e-12:
        standardized = (0.0,) * len(values)
    else:
        center = fmean(values)
        standardized = tuple((value - center) / spread for value in values)

    logits = tuple(value / temperature for value in standardized)
    maximum = max(logits)
    exponentials = tuple(math.exp(value - maximum) for value in logits)
    denominator = sum(exponentials)
    uniform = epsilon / len(actions)
    informed_scale = 1.0 - epsilon
    return {
        action: uniform + informed_scale * exponential / denominator
        for action, exponential in zip(actions, exponentials, strict=True)
    }


def sample_distribution(
    probabilities: Mapping[ChoiceT, float],
    *,
    rng: random.Random,
) -> ChoiceT:
    """Sample one key from a normalized distribution in insertion order."""

    if not probabilities:
        raise ValueError("cannot sample an empty distribution")
    total = sum(probabilities.values())
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("probability mass must be positive and finite")
    threshold = rng.random() * total
    cumulative = 0.0
    last: ChoiceT | None = None
    for action, probability in probabilities.items():
        if not math.isfinite(probability) or probability < 0.0:
            raise ValueError("probabilities must be finite and non-negative")
        last = action
        cumulative += probability
        if threshold < cumulative:
            return action
    assert last is not None
    return last


def normalize_distribution(
    weights: Mapping[ChoiceT, float],
) -> dict[ChoiceT, float]:
    """Normalize finite non-negative weights, dropping zero-mass entries."""

    if any(not math.isfinite(weight) or weight < 0.0 for weight in weights.values()):
        raise ValueError("weights must be finite and non-negative")
    positive = {choice: weight for choice, weight in weights.items() if weight > 0.0}
    total = math.fsum(positive.values())
    if total <= 0.0:
        return {}
    return {choice: weight / total for choice, weight in positive.items()}


def uniform_distribution(actions: Sequence[ChoiceT]) -> dict[ChoiceT, float]:
    if not actions:
        raise ValueError("uniform_distribution requires at least one action")
    probability = 1.0 / len(actions)
    return {action: probability for action in actions}


__all__ = [
    "epsilon_softmax",
    "normalize_distribution",
    "possible_draw_cards",
    "sample_distribution",
    "uniform_distribution",
    "visible_card_identities",
]
