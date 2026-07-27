"""Offline, fixed-observation benchmarks for Marshal belief backends.

This module deliberately keeps inference and scoring separate.  A backend sees
only the same immutable Marshal :class:`~fugitive.game.model.Observation` that it
would receive during a game.  Complete state is captured by the replay
inspector and used only after inference returns.

The evaluator visits every Marshal draw, normal guess, and Manhunt decision in
trace order.  Stateful backends such as BIR-1 therefore receive the same public
event sequence as they do during live play.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
from typing import Callable, Iterable, Mapping, Protocol, TypeAlias

from fugitive.game.driver import DecisionRecord, DrawTrace, GuessTrace
from fugitive.game.engine import GameEngine
from fugitive.game.model import Observation, Phase, Role
from fugitive.game.rules import GUESSABLE_CARDS
from fugitive.agents.marshal.inference.particles.state import (
    MarshalParticleBelief,
)
from fugitive.runtime.manifest import ReplayManifest
from fugitive.runtime.replay import replay_manifest


MarshalDecisionRecord: TypeAlias = DrawTrace | GuessTrace
CandidateSetProvider: TypeAlias = Callable[
    [Observation], Iterable[Iterable[int]]
]


class MarshalBeliefInferencer(Protocol):
    """The narrow interface needed by the offline benchmark."""

    backend_id: str

    def infer(self, observation: Observation) -> MarshalParticleBelief: ...


@dataclass(frozen=True, slots=True)
class OfflineMarshalTruth:
    """Complete state at one Marshal decision, never supplied to a backend.

    ``remaining_piles`` preserves the engine's order; the next card drawn is
    the last item of a pile.  This is useful for later continuation studies,
    while inference code remains free to treat pile contents as unordered.
    """

    fugitive_hand: tuple[int, ...]
    marshal_hand: tuple[int, ...]
    route_hideouts: tuple[int, ...]
    route_sprints: tuple[tuple[int, ...], ...]
    route_revealed: tuple[bool, ...]
    remaining_piles: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]

    @property
    def hidden_route(self) -> tuple[int, ...]:
        """Return the currently unrevealed, guessable Hideouts in route order."""

        return tuple(
            hideout
            for index, hideout in enumerate(self.route_hideouts)
            if index and not self.route_revealed[index] and hideout != 42
        )


@dataclass(frozen=True, slots=True)
class MarshalDecisionPoint:
    """One decision-before-action Observation paired with offline truth."""

    decision: int
    round_number: int
    phase: Phase
    record: MarshalDecisionRecord
    observation: Observation
    truth: OfflineMarshalTruth


@dataclass(frozen=True, slots=True)
class HiddenCardMetrics:
    brier_score: float
    log_loss: float
    card_count: int
    positive_count: int


@dataclass(frozen=True, slots=True)
class RouteForecastMetrics:
    true_route: tuple[int, ...]
    true_mass: float
    rank: int
    support_size: int
    top_k_coverage: tuple[tuple[int, bool], ...]


@dataclass(frozen=True, slots=True)
class JointCalibrationRecord:
    """One predicted joint-guess probability and its realized truth label."""

    guess: tuple[int, ...]
    predicted_success: float
    observed_success: bool
    decision: int | None = None
    phase: Phase | None = None
    recorded_action: bool = False


@dataclass(frozen=True, slots=True)
class JointCalibrationBin:
    lower_bound: float
    upper_bound: float
    count: int
    mean_prediction: float | None
    observed_frequency: float | None


@dataclass(frozen=True, slots=True)
class BackendDecisionEvaluation:
    point: MarshalDecisionPoint
    marginals: tuple[tuple[int, float], ...]
    hidden_route_masses: tuple[tuple[tuple[int, ...], float], ...]
    hidden_cards: HiddenCardMetrics
    route: RouteForecastMetrics
    joint_calibration: tuple[JointCalibrationRecord, ...]


@dataclass(frozen=True, slots=True)
class BackendBenchmarkResult:
    backend_id: str
    decisions: tuple[BackendDecisionEvaluation, ...]

    @property
    def joint_calibration(self) -> tuple[JointCalibrationRecord, ...]:
        return tuple(
            record
            for decision in self.decisions
            for record in decision.joint_calibration
        )


def extract_marshal_decision_points(
    manifest: ReplayManifest,
    *,
    validate_invariants: bool = True,
) -> tuple[MarshalDecisionPoint, ...]:
    """Replay ``manifest`` and snapshot every Marshal decision before action."""

    points: list[MarshalDecisionPoint] = []

    def inspect(engine: GameEngine, record: DecisionRecord) -> None:
        if engine.phase not in (
            Phase.MARSHAL_DRAW,
            Phase.MARSHAL_GUESS,
            Phase.MANHUNT,
        ):
            return
        if not isinstance(record, (DrawTrace, GuessTrace)):
            raise AssertionError("a Marshal phase must contain a draw or guess record")
        if record.role is not Role.MARSHAL:
            raise AssertionError("a Marshal phase record must have the Marshal role")

        observation = engine.observation(Role.MARSHAL)
        points.append(
            MarshalDecisionPoint(
                decision=record.decision,
                round_number=record.round_number,
                phase=record.phase,
                record=record,
                observation=observation,
                truth=_offline_truth(engine, observation),
            )
        )

    replay_manifest(
        manifest,
        validate_invariants=validate_invariants,
        inspect_before_decision=inspect,
    )
    return tuple(points)


def hidden_card_brier_score(
    probabilities: Mapping[int, float],
    true_hidden_cards: Iterable[int],
    *,
    card_universe: Iterable[int] = GUESSABLE_CARDS,
) -> float:
    """Return mean binary Brier score for current hidden-Hideout membership."""

    cards, truth = _card_metric_inputs(true_hidden_cards, card_universe)
    return math.fsum(
        (_mapping_probability(probabilities, card) - float(card in truth)) ** 2
        for card in cards
    ) / len(cards)


def hidden_card_log_loss(
    probabilities: Mapping[int, float],
    true_hidden_cards: Iterable[int],
    *,
    card_universe: Iterable[int] = GUESSABLE_CARDS,
    epsilon: float = 1e-12,
) -> float:
    """Return mean binary log loss, clipping only to keep zero mass finite."""

    if not math.isfinite(epsilon) or not 0.0 < epsilon <= 0.5:
        raise ValueError("epsilon must be finite and in (0, 0.5]")
    cards, truth = _card_metric_inputs(true_hidden_cards, card_universe)
    losses: list[float] = []
    for card in cards:
        probability = _mapping_probability(probabilities, card)
        clipped = min(1.0 - epsilon, max(epsilon, probability))
        losses.append(
            -math.log(clipped) if card in truth else -math.log1p(-clipped)
        )
    return math.fsum(losses) / len(losses)


def hidden_card_metrics(
    probabilities: Mapping[int, float],
    true_hidden_cards: Iterable[int],
    *,
    card_universe: Iterable[int] = GUESSABLE_CARDS,
    log_epsilon: float = 1e-12,
) -> HiddenCardMetrics:
    """Compute the two card-marginal metrics with one explicit universe."""

    cards, truth = _card_metric_inputs(true_hidden_cards, card_universe)
    return HiddenCardMetrics(
        brier_score=hidden_card_brier_score(
            probabilities,
            truth,
            card_universe=cards,
        ),
        log_loss=hidden_card_log_loss(
            probabilities,
            truth,
            card_universe=cards,
            epsilon=log_epsilon,
        ),
        card_count=len(cards),
        positive_count=len(truth),
    )


def route_forecast_metrics(
    route_masses: Mapping[tuple[int, ...], float],
    true_route: Iterable[int],
    *,
    top_k: Iterable[int] = (1, 5, 10),
) -> RouteForecastMetrics:
    """Score one hidden-route distribution using mass, rank, and top-k coverage.

    Rank is the usual competition rank: one plus the number of hypotheses with
    strictly greater mass.  A truth absent from support is ranked immediately
    after all positive-mass hypotheses and is never counted as top-k covered.
    """

    truth = tuple(true_route)
    positive: dict[tuple[int, ...], float] = {}
    for route, mass in route_masses.items():
        route_key = tuple(route)
        value = _probability_mass(mass, f"route mass for {route_key}")
        if value > 0.0:
            positive[route_key] = positive.get(route_key, 0.0) + value
    total = math.fsum(positive.values())
    normalized = (
        {route: mass / total for route, mass in positive.items()}
        if total > 0.0
        else {}
    )

    true_mass = normalized.get(truth, 0.0)
    in_support = truth in normalized
    rank = (
        1 + sum(mass > true_mass for mass in normalized.values())
        if in_support
        else len(normalized) + 1
    )
    requested_top_k = _normalize_top_k(top_k)
    return RouteForecastMetrics(
        true_route=truth,
        true_mass=true_mass,
        rank=rank,
        support_size=len(normalized),
        top_k_coverage=tuple(
            (k, in_support and rank <= k) for k in requested_top_k
        ),
    )


def belief_hidden_route_masses(
    belief: MarshalParticleBelief,
) -> dict[tuple[int, ...], float]:
    """Aggregate particle mass by current hidden-route hypothesis."""

    masses: dict[tuple[int, ...], float] = defaultdict(float)
    for particle in belief.particles:
        route = tuple(sorted(belief.current_hidden_hideouts(particle)))
        masses[route] += particle.weight
    return dict(sorted(masses.items()))


def joint_calibration_record(
    guess: Iterable[int],
    predicted_success: float,
    true_hidden_route: Iterable[int],
    *,
    decision: int | None = None,
    phase: Phase | None = None,
    recorded_action: bool = False,
) -> JointCalibrationRecord:
    """Create one calibration row for the event ``guess subset of route``."""

    normalized_guess = _normalize_guess(guess)
    probability = _probability(predicted_success, "predicted joint success")
    truth = frozenset(true_hidden_route)
    return JointCalibrationRecord(
        guess=normalized_guess,
        predicted_success=probability,
        observed_success=frozenset(normalized_guess).issubset(truth),
        decision=decision,
        phase=phase,
        recorded_action=recorded_action,
    )


def joint_calibration_bins(
    records: Iterable[JointCalibrationRecord],
    *,
    bin_count: int = 10,
) -> tuple[JointCalibrationBin, ...]:
    """Aggregate joint forecasts into equal-width reliability bins."""

    if isinstance(bin_count, bool) or not isinstance(bin_count, int) or bin_count < 1:
        raise ValueError("bin_count must be a positive integer")
    grouped: list[list[JointCalibrationRecord]] = [[] for _ in range(bin_count)]
    for record in records:
        probability = _probability(record.predicted_success, "predicted joint success")
        index = min(bin_count - 1, int(probability * bin_count))
        grouped[index].append(record)

    result: list[JointCalibrationBin] = []
    for index, group in enumerate(grouped):
        count = len(group)
        result.append(
            JointCalibrationBin(
                lower_bound=index / bin_count,
                upper_bound=(index + 1) / bin_count,
                count=count,
                mean_prediction=(
                    math.fsum(item.predicted_success for item in group) / count
                    if count
                    else None
                ),
                observed_frequency=(
                    sum(item.observed_success for item in group) / count
                    if count
                    else None
                ),
            )
        )
    return tuple(result)


def evaluate_backend_on_manifest(
    manifest: ReplayManifest,
    backend: MarshalBeliefInferencer,
    *,
    candidate_set_provider: CandidateSetProvider | None = None,
    include_recorded_guess: bool = True,
    card_universe: Iterable[int] = GUESSABLE_CARDS,
    top_k: Iterable[int] = (1, 5, 10),
    validate_invariants: bool = True,
) -> BackendBenchmarkResult:
    """Evaluate one backend on a fixed replay without exposing offline truth.

    Candidate sets are supplied by an Observation-only callable.  A shared
    catalogue can therefore be injected later without coupling this benchmark
    to any agent's action generator.  Set ``include_recorded_guess=False`` when
    only catalogue-defined candidates should enter calibration.
    """

    points = extract_marshal_decision_points(
        manifest,
        validate_invariants=validate_invariants,
    )
    cards = tuple(card_universe)
    requested_top_k = tuple(top_k)
    evaluations: list[BackendDecisionEvaluation] = []
    for point in points:
        # This is the only backend call.  In particular, ``point.truth`` is not
        # an argument and is not captured by a callback passed to the backend.
        belief = backend.infer(point.observation)
        if belief.observation != point.observation:
            raise ValueError("backend returned a belief for a different observation")

        marginals = dict(belief.marginals)
        route_masses = belief_hidden_route_masses(belief)
        candidates: dict[tuple[int, ...], bool] = {}
        if point.phase in (Phase.MARSHAL_GUESS, Phase.MANHUNT):
            if candidate_set_provider is not None:
                for guess in candidate_set_provider(point.observation):
                    candidates[_normalize_guess(guess)] = False
            if include_recorded_guess and isinstance(point.record, GuessTrace):
                candidates[_normalize_guess(point.record.numbers)] = True

        joint_records = tuple(
            joint_calibration_record(
                guess,
                belief.joint_success(guess),
                point.truth.hidden_route,
                decision=point.decision,
                phase=point.phase,
                recorded_action=recorded,
            )
            for guess, recorded in sorted(candidates.items())
        )
        if isinstance(point.record, GuessTrace):
            actual = next(
                (
                    row.observed_success
                    for row in joint_records
                    if row.guess == tuple(sorted(point.record.numbers))
                ),
                frozenset(point.record.numbers).issubset(point.truth.hidden_route),
            )
            if actual is not point.record.success:
                raise AssertionError(
                    "offline truth disagrees with replayed guess result"
                )

        evaluations.append(
            BackendDecisionEvaluation(
                point=point,
                marginals=tuple(sorted(marginals.items())),
                hidden_route_masses=tuple(sorted(route_masses.items())),
                hidden_cards=hidden_card_metrics(
                    marginals,
                    point.truth.hidden_route,
                    card_universe=cards,
                ),
                route=route_forecast_metrics(
                    route_masses,
                    point.truth.hidden_route,
                    top_k=requested_top_k,
                ),
                joint_calibration=joint_records,
            )
        )

    backend_id = getattr(backend, "backend_id", type(backend).__qualname__)
    return BackendBenchmarkResult(str(backend_id), tuple(evaluations))


def _offline_truth(
    engine: GameEngine,
    marshal_observation: Observation,
) -> OfflineMarshalTruth:
    fugitive_observation = engine.observation(Role.FUGITIVE)
    hideouts: list[int] = []
    sprints: list[tuple[int, ...]] = []
    for slot in fugitive_observation.route:
        if slot.hideout is None or slot.sprint_cards is None:
            raise AssertionError("the Fugitive view must contain its complete route")
        hideouts.append(slot.hideout)
        sprints.append(tuple(slot.sprint_cards))

    # Remaining identities are intentionally unavailable through Observation.
    # This is the sole private-state read and is confined to offline replay.
    piles = engine._state.piles
    return OfflineMarshalTruth(
        fugitive_hand=fugitive_observation.hand,
        marshal_hand=marshal_observation.hand,
        route_hideouts=tuple(hideouts),
        route_sprints=tuple(sprints),
        route_revealed=tuple(slot.revealed for slot in marshal_observation.route),
        remaining_piles=(tuple(piles[0]), tuple(piles[1]), tuple(piles[2])),
    )


def _card_metric_inputs(
    true_hidden_cards: Iterable[int],
    card_universe: Iterable[int],
) -> tuple[tuple[int, ...], frozenset[int]]:
    cards = tuple(card_universe)
    if not cards or len(cards) != len(set(cards)):
        raise ValueError("card_universe must contain distinct cards")
    if any(isinstance(card, bool) or not isinstance(card, int) for card in cards):
        raise ValueError("card_universe must contain integers")
    truth = frozenset(true_hidden_cards)
    if any(isinstance(card, bool) or not isinstance(card, int) for card in truth):
        raise ValueError("true hidden cards must contain integers")
    if not truth.issubset(cards):
        raise ValueError("true hidden cards must be contained in card_universe")
    return cards, truth


def _mapping_probability(probabilities: Mapping[int, float], card: int) -> float:
    return _probability(probabilities.get(card, 0.0), f"probability for card {card}")


def _probability(value: float, label: str) -> float:
    tolerance = 1e-12
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not -tolerance <= value <= 1.0 + tolerance
    ):
        raise ValueError(f"{label} must be finite and between zero and one")
    return min(1.0, max(0.0, float(value)))


def _probability_mass(value: float, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0.0
    ):
        raise ValueError(f"{label} must be finite and non-negative")
    return float(value)


def _normalize_top_k(top_k: Iterable[int]) -> tuple[int, ...]:
    values = tuple(top_k)
    if (
        not values
        or len(values) != len(set(values))
        or any(isinstance(k, bool) or not isinstance(k, int) or k < 1 for k in values)
    ):
        raise ValueError("top_k must contain distinct positive integers")
    return tuple(sorted(values))


def _normalize_guess(guess: Iterable[int]) -> tuple[int, ...]:
    values = tuple(guess)
    if (
        not values
        or len(values) != len(set(values))
        or any(number not in GUESSABLE_CARDS for number in values)
    ):
        raise ValueError("candidate guesses must contain distinct cards from 1 to 41")
    return tuple(sorted(values))


__all__ = [
    "BackendBenchmarkResult",
    "BackendDecisionEvaluation",
    "CandidateSetProvider",
    "HiddenCardMetrics",
    "JointCalibrationBin",
    "JointCalibrationRecord",
    "MarshalBeliefInferencer",
    "MarshalDecisionPoint",
    "MarshalDecisionRecord",
    "OfflineMarshalTruth",
    "RouteForecastMetrics",
    "belief_hidden_route_masses",
    "evaluate_backend_on_manifest",
    "extract_marshal_decision_points",
    "hidden_card_brier_score",
    "hidden_card_log_loss",
    "hidden_card_metrics",
    "joint_calibration_bins",
    "joint_calibration_record",
    "route_forecast_metrics",
]
