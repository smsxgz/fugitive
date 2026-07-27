"""Candidate-relative continuation-count Fugitive teaching agent.

The agent keeps the BIR Fugitive's bounded action abstraction and adds two
inspectable features:

* continuation is ``log1p`` of a bounded continuation *action-tree node mass*,
  normalized relative to the candidates at the current decision; and
* the recursive continuation calculation uses an exact, agent-local memo.

The count is not a count of distinct future routes.  Different bounded Sprint
payments are different action-tree branches, and every reachable action-tree
prefix contributes one node to the mass.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from dataclasses import replace
import math
import random
from typing import TypeVar

from fugitive.belief import PathBelief
from fugitive.model import (
    DrawRecord,
    FugitiveAction,
    Observation,
    Phase,
    PlayView,
    Role,
    RouteView,
)

from .baseline_utils import epsilon_softmax, possible_draw_cards
from .bir_common import normalized as _normalized
from .bir_fugitive import BeliefInformedRandomFugitiveAgent, clamp01
from .hierarchical_random import (
    OpeningPlan,
    OpeningPlanGroup,
    SprintPayment,
    SprintPlanGroup,
    bounded_sprint_plans,
    normalized_play_actions,
    sprint_cost,
)


CONTINUATION_COUNT_FUGITIVE_ALGORITHM_ID = (
    "continuation-count-fugitive-candidate-relative-v2"
)
DEFAULT_CONTINUATION_WEIGHT = 0.75
DEFAULT_CONCEALMENT_WEIGHT = 0.50
DEFAULT_CATCH_RISK_WEIGHT = 1.00
DEFAULT_CONTINUATION_DEPTH = 2

# These identifiers are deliberately descriptive metadata rather than claims
# that the features are calibrated probabilities or exact future-route counts.
CONTINUATION_MASS_ID = "bounded-sprint-action-tree-node-mass-v1"
CONTINUATION_TRANSFORM_ID = "candidate-relative-log1p-max-normalization-v1"
CONTINUATION_CACHE_ID = "agent-local-exact-subproblem-memo-v1"
CONCEALMENT_FEATURE_ID = "normalized-hidden-card-marginal-entropy-v1"
CATCH_RISK_FEATURE_ID = (
    "public-path-count-single-guess-true-hidden-hideout-mass-v1"
)
PUBLIC_TIMELINE_ID = "marshal-visible-existing-and-hypothetical-play-timeline-v1"


CandidateT = TypeVar("CandidateT", bound=Hashable)


def _normalize_nonnegative_candidate_values(
    values: Mapping[CandidateT, float],
) -> dict[CandidateT, float]:
    """Normalize non-negative candidate features by their current maximum."""

    if not values:
        return {}
    if any(not math.isfinite(value) or value < 0.0 for value in values.values()):
        raise ValueError("candidate feature values must be finite and non-negative")
    maximum = max(values.values())
    if maximum <= 0.0:
        return {candidate: 0.0 for candidate in values}
    return {candidate: value / maximum for candidate, value in values.items()}


def candidate_relative_log1p_features(
    counts: Mapping[CandidateT, int],
) -> dict[CandidateT, float]:
    """Return ``log1p(count) / max_candidate(log1p(count))``.

    The denominator is recomputed for each decision candidate set.  There is
    no fixed saturation count, so a candidate with count 256 is not forced to
    tie a candidate with count 1,024.
    """

    if any(
        isinstance(count, bool) or not isinstance(count, int) or count < 0
        for count in counts.values()
    ):
        raise ValueError("continuation counts must be non-negative integers")
    return _normalize_nonnegative_candidate_values(
        {candidate: math.log1p(count) for candidate, count in counts.items()}
    )


class ContinuationCountFugitiveAgent(BeliefInformedRandomFugitiveAgent):
    """Continuation count with relative features and exact local caching.

    The defensive terms use a public ``PathBelief`` response model:
    concealment is normalized entropy of hidden-card *marginals*, while catch
    risk is the response model's one-guess mass on the true hidden Hideouts.
    Neither term is a joint route entropy or a calibrated catch probability.
    """

    name = "continuation-count-fugitive"

    def __init__(
        self,
        seed: int | random.Random | None = None,
        *,
        continuation_weight: float = DEFAULT_CONTINUATION_WEIGHT,
        concealment_weight: float = DEFAULT_CONCEALMENT_WEIGHT,
        catch_risk_weight: float = DEFAULT_CATCH_RISK_WEIGHT,
        continuation_depth: int = DEFAULT_CONTINUATION_DEPTH,
        **kwargs,
    ) -> None:
        super().__init__(seed, **kwargs)
        for name, value in (
            ("continuation_weight", continuation_weight),
            ("concealment_weight", concealment_weight),
            ("catch_risk_weight", catch_risk_weight),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0.0
            ):
                raise ValueError(f"{name} must be finite and non-negative")
        if (
            isinstance(continuation_depth, bool)
            or not isinstance(continuation_depth, int)
            or continuation_depth < 1
        ):
            raise ValueError("continuation_depth must be a positive integer")
        self.continuation_weight = float(continuation_weight)
        self.concealment_weight = float(concealment_weight)
        self.catch_risk_weight = float(catch_risk_weight)
        self.continuation_depth = continuation_depth
        self.algorithm_id = CONTINUATION_COUNT_FUGITIVE_ALGORITHM_ID
        self.continuation_mass_id = CONTINUATION_MASS_ID
        self.continuation_transform_id = CONTINUATION_TRANSFORM_ID
        self.continuation_cache_id = CONTINUATION_CACHE_ID
        self.concealment_feature_id = CONCEALMENT_FEATURE_ID
        self.catch_risk_feature_id = CATCH_RISK_FEATURE_ID
        self.public_timeline_id = PUBLIC_TIMELINE_ID
        self.response_model_id = (
            "public-path-count-epsilon-soft-single-guess-response-v1"
        )

        # The memo is exact for this agent's fixed bounded-payment settings.
        # It deliberately has no eviction policy: identical mathematical
        # subproblems remain identical for the lifetime of this agent.
        self._continuation_count_cache: dict[
            tuple[tuple[int, ...], int, int], int
        ] = {}
        self._continuation_cache_hits = 0
        self._continuation_cache_misses = 0
        self._defence_cache: dict[
            tuple[Observation, tuple[int, ...]], tuple[float, float]
        ] = {}

    @property
    def continuation_cache_entries(self) -> int:
        return len(self._continuation_count_cache)

    @property
    def continuation_cache_hits(self) -> int:
        return self._continuation_cache_hits

    @property
    def continuation_cache_misses(self) -> int:
        return self._continuation_cache_misses

    def clear_continuation_cache(self) -> None:
        """Clear the exact continuation memo and its counters."""

        self._continuation_count_cache.clear()
        self._continuation_cache_hits = 0
        self._continuation_cache_misses = 0

    def _opening_plan_distribution(
        self,
        observation: Observation,
        plans: OpeningPlanGroup,
    ) -> dict[OpeningPlan, float]:
        all_plans = tuple(dict.fromkeys((*plans.low_cost, *plans.extra_overpay)))
        counts = {
            plan: self._opening_continuation_count(observation, plan)
            for plan in all_plans
        }
        continuation_features = candidate_relative_log1p_features(counts)
        scores = {
            plan: self._opening_score_with_feature(
                observation,
                plan,
                continuation_features[plan],
            )
            for plan in all_plans
        }

        low = self._opening_branch_distribution(plans.low_cost, scores)
        extra = self._opening_branch_distribution(plans.extra_overpay, scores)
        if not low:
            return extra
        if not extra:
            return low
        return _normalized(
            {
                **{
                    plan: (1.0 - self.overpay_probability) * probability
                    for plan, probability in low.items()
                },
                **{
                    plan: self.overpay_probability * probability
                    for plan, probability in extra.items()
                },
            }
        )

    def _opening_branch_distribution(
        self,
        plans: tuple[OpeningPlan, ...],
        scores: Mapping[OpeningPlan, float],
    ) -> dict[OpeningPlan, float]:
        """Apply BIR's opening hierarchy to already-scored plans."""

        if not plans:
            return {}
        result: dict[OpeningPlan, float] = {}
        first_hideouts = sorted({plan.first.hideout for plan in plans})
        first_hideout_distribution = self._softmax_groups(
            first_hideouts,
            plans,
            scores,
            key=lambda plan: plan.first.hideout,
        )
        for first_hideout, first_hideout_probability in (
            first_hideout_distribution.items()
        ):
            first_plans = tuple(
                plan for plan in plans if plan.first.hideout == first_hideout
            )
            first_actions = tuple(
                sorted({plan.first for plan in first_plans}, key=repr)
            )
            first_action_distribution = self._softmax_groups(
                first_actions,
                first_plans,
                scores,
                key=lambda plan: plan.first,
            )
            for first_action, first_action_probability in (
                first_action_distribution.items()
            ):
                descendants = tuple(
                    plan for plan in first_plans if plan.first == first_action
                )
                second_hideouts = sorted(
                    {plan.second.hideout for plan in descendants}
                )
                second_hideout_distribution = self._softmax_groups(
                    second_hideouts,
                    descendants,
                    scores,
                    key=lambda plan: plan.second.hideout,
                )
                for second_hideout, second_hideout_probability in (
                    second_hideout_distribution.items()
                ):
                    leaves = tuple(
                        plan
                        for plan in descendants
                        if plan.second.hideout == second_hideout
                    )
                    leaf_distribution = epsilon_softmax(
                        {plan: scores[plan] for plan in leaves},
                        epsilon=self.epsilon,
                        temperature=self.temperature,
                    )
                    prefix_probability = (
                        first_hideout_probability
                        * first_action_probability
                        * second_hideout_probability
                    )
                    for plan, leaf_probability in leaf_distribution.items():
                        result[plan] = prefix_probability * leaf_probability
        return _normalized(result)

    def _play_action_distribution(
        self,
        observation: Observation,
        *,
        include_pass: bool,
    ) -> dict[FugitiveAction, float]:
        groups = normalized_play_actions(
            observation,
            max_low_cost=self.max_low_cost_payments,
            max_extra_overpay=self.max_extra_overpayments,
        )
        if not groups:
            return {FugitiveAction(None): 1.0} if include_pass else {}

        actions = tuple(
            FugitiveAction(hideout, payment.cards)
            for hideout, group in groups.items()
            for payment in group.all_payments
        )
        continuation_logs = {
            action: self._action_continuation_log(observation, action)
            for action in actions
        }
        pass_action = FugitiveAction(None)
        if include_pass:
            continuation_logs[pass_action] = self._pass_continuation_log(observation)
        continuation_features = _normalize_nonnegative_candidate_values(
            continuation_logs
        )

        action_scores = {
            action: self._action_score_with_feature(
                observation,
                action,
                continuation_features[action],
            )
            for action in actions
        }
        macro_scores: dict[int | None, float] = {
            hideout: max(
                action_scores[FugitiveAction(hideout, payment.cards)]
                for payment in group.all_payments
            )
            for hideout, group in groups.items()
        }
        if include_pass:
            macro_scores[None] = self._pass_score_with_feature(
                observation,
                continuation_features[pass_action],
            )
        macro_distribution = epsilon_softmax(
            macro_scores,
            epsilon=self.epsilon,
            temperature=self.temperature,
        )

        result: dict[FugitiveAction, float] = {}
        if include_pass and macro_distribution.get(None, 0.0):
            result[pass_action] = macro_distribution[None]
        for hideout, group in groups.items():
            payment_distribution = self._payment_distribution_from_scores(
                hideout,
                group,
                action_scores,
            )
            for cards, probability in payment_distribution.items():
                result[FugitiveAction(hideout, cards)] = (
                    macro_distribution[hideout] * probability
                )
        return _normalized(result)

    def _payment_distribution_from_scores(
        self,
        hideout: int,
        group: SprintPlanGroup,
        scores: Mapping[FugitiveAction, float],
    ) -> dict[tuple[int, ...], float]:
        def branch(
            payments: tuple[SprintPayment, ...],
        ) -> dict[tuple[int, ...], float]:
            if not payments:
                return {}
            return epsilon_softmax(
                {
                    payment.cards: scores[
                        FugitiveAction(hideout, payment.cards)
                    ]
                    for payment in payments
                },
                epsilon=self.epsilon,
                temperature=self.temperature,
            )

        low = branch(group.low_cost)
        extra = branch(group.extra_overpay)
        if not low:
            return extra
        if not extra:
            return low
        return _normalized(
            {
                **{
                    cards: (1.0 - self.overpay_probability) * probability
                    for cards, probability in low.items()
                },
                **{
                    cards: self.overpay_probability * probability
                    for cards, probability in extra.items()
                },
            }
        )

    def _opening_continuation_count(
        self,
        observation: Observation,
        plan: OpeningPlan,
    ) -> int:
        assert plan.first.hideout is not None
        assert plan.second.hideout is not None
        remaining = self._spend(observation.hand, plan.first)
        remaining = self._spend(remaining, plan.second)
        return self._continuation_count(
            remaining,
            plan.second.hideout,
            self.continuation_depth,
        )

    def _opening_score_with_feature(
        self,
        observation: Observation,
        plan: OpeningPlan,
        continuation_feature: float,
    ) -> float:
        assert plan.first.hideout is not None
        assert plan.second.hideout is not None
        first_cost = sprint_cost(0, plan.first.hideout, plan.first.sprint_cards)
        second_cost = sprint_cost(
            plan.first.hideout,
            plan.second.hideout,
            plan.second.sprint_cards,
        )
        cost_scale = 2.75 * max(1, len(observation.hand) - 2)
        concealment, catch_risk = self._defensive_features(
            observation,
            (plan.first, plan.second),
        )
        return (
            2.0 * plan.second.hideout / 42.0
            - clamp01((first_cost + second_cost) / cost_scale)
            + self.continuation_weight * continuation_feature
            + self.concealment_weight * concealment
            - self.catch_risk_weight * catch_risk
        )

    def _action_score_with_feature(
        self,
        observation: Observation,
        action: FugitiveAction,
        continuation_feature: float,
    ) -> float:
        assert action.hideout is not None
        previous = observation.route[-1].hideout
        assert previous is not None
        progress = (action.hideout - previous) / max(1, 42 - previous)
        cost = sprint_cost(previous, action.hideout, action.sprint_cards)
        cost_scale = 2.75 * max(1, len(observation.hand) - 1)
        terminal = self._terminal_value(observation, action)
        concealment, catch_risk = (0.0, 0.0)
        if action.hideout != 42:
            concealment, catch_risk = self._defensive_features(observation, (action,))
        return (
            2.0 * clamp01(progress)
            - clamp01(cost / cost_scale)
            + self.continuation_weight * continuation_feature
            + self.concealment_weight * concealment
            - self.catch_risk_weight * catch_risk
            + 4.0 * terminal
        )

    def _pass_score_with_feature(
        self,
        observation: Observation,
        continuation_feature: float,
    ) -> float:
        concealment, catch_risk = self._defensive_features(observation, ())
        return (
            self.continuation_weight * continuation_feature
            + self.concealment_weight * concealment
            - self.catch_risk_weight * catch_risk
            - 0.35
        )

    def _hand_value(
        self,
        hand: tuple[int, ...],
        previous: int,
        observation: Observation,
    ) -> float:
        """Evaluate a possible draw using its future candidate-relative set."""

        synthetic = replace(
            observation,
            hand=tuple(sorted(hand)),
            phase=Phase.FUGITIVE_ACTION,
        )
        groups = normalized_play_actions(
            synthetic,
            max_low_cost=self.max_low_cost_payments,
            max_extra_overpay=self.max_extra_overpayments,
        )
        actions = tuple(
            FugitiveAction(hideout, payment.cards)
            for hideout, group in groups.items()
            for payment in group.all_payments
        )
        if not actions:
            return -0.5
        features = _normalize_nonnegative_candidate_values(
            {
                action: self._action_continuation_log(synthetic, action)
                for action in actions
            }
        )
        scores = [
            self._action_score_with_feature(synthetic, action, features[action])
            for action in actions
        ]
        return max(scores) + 0.25 * math.log1p(len(scores))

    def _action_continuation_log(
        self,
        observation: Observation,
        action: FugitiveAction,
    ) -> float:
        assert action.hideout is not None
        remaining = self._spend(observation.hand, action)
        return self._continuation_log_value(remaining, action.hideout)

    def _pass_continuation_log(self, observation: Observation) -> float:
        """Best expected next-draw log continuation mass for a candidate Pass."""

        previous = observation.route[-1].hideout
        assert previous is not None
        current = self._continuation_log_value(observation.hand, previous)
        nonempty = tuple(
            pile for pile, size in enumerate(observation.pile_sizes) if size > 0
        )
        if not nonempty:
            return current
        draw_view = replace(
            observation,
            phase=Phase.FUGITIVE_DRAW,
            legal_draw_piles=nonempty,
        )
        expected_logs: list[float] = []
        for pile in nonempty:
            support = possible_draw_cards(draw_view, pile)
            if support:
                expected_logs.append(
                    sum(
                        self._continuation_log_value(
                            tuple(sorted((*observation.hand, card))),
                            previous,
                        )
                        for card in support
                    )
                    / len(support)
                )
        return max(expected_logs, default=current)

    def _continuation_log_value(
        self,
        hand: tuple[int, ...],
        previous: int,
    ) -> float:
        return math.log1p(
            self._continuation_count(
                tuple(sorted(hand)),
                previous,
                self.continuation_depth,
            )
        )

    def _continuation_count(
        self,
        hand: tuple[int, ...],
        previous: int,
        depth: int,
    ) -> int:
        canonical_hand = tuple(sorted(hand))
        key = (canonical_hand, previous, depth)
        cached = self._continuation_count_cache.get(key)
        if cached is not None:
            self._continuation_cache_hits += 1
            return cached

        self._continuation_cache_misses += 1
        if depth <= 0:
            total = 1
        else:
            total = 0
            for hideout in canonical_hand:
                if hideout <= previous:
                    continue
                plans = bounded_sprint_plans(
                    canonical_hand,
                    previous,
                    hideout,
                    max_low_cost=self.max_low_cost_payments,
                    max_extra_overpay=self.max_extra_overpayments,
                )
                for payment in plans.all_payments:
                    spent = {hideout, *payment.cards}
                    remaining = tuple(
                        card for card in canonical_hand if card not in spent
                    )
                    total += 1
                    if depth > 1 and hideout != 42:
                        total += self._continuation_count(
                            remaining,
                            hideout,
                            depth - 1,
                        )
        self._continuation_count_cache[key] = total
        return total

    def _defensive_features(
        self,
        observation: Observation,
        actions: tuple[FugitiveAction, ...],
    ) -> tuple[float, float]:
        shadow, truth = self._public_shadow_after(observation, actions)
        key = (shadow, truth)
        cached = self._defence_cache.get(key)
        if cached is not None:
            return cached
        result = PathBelief.from_observation(shadow).solve()
        if not result.total_paths or not result.marginal_counts:
            value = (0.0, 1.0 if truth else 0.0)
        else:
            weights = {
                number: count / result.total_paths
                for number, count in result.marginal_counts.items()
            }
            total = sum(weights.values())
            probabilities = {
                number: weight / total for number, weight in weights.items()
            }
            entropy = -sum(
                probability * math.log(probability)
                for probability in probabilities.values()
                if probability > 0.0
            )
            concealment = (
                entropy / math.log(len(probabilities))
                if len(probabilities) > 1
                else 0.0
            )
            epsilon = self.manhunt_epsilon
            powered = {
                number: weight**self.manhunt_alpha
                for number, weight in weights.items()
            }
            denominator = sum(powered.values())
            uniform = epsilon / len(powered)
            response = {
                number: uniform + (1.0 - epsilon) * weight / denominator
                for number, weight in powered.items()
            }
            catch_risk = sum(
                probability
                for number, probability in response.items()
                if number in truth
            )
            value = (clamp01(concealment), clamp01(catch_risk))
        if len(self._defence_cache) >= 256:
            self._defence_cache.pop(next(iter(self._defence_cache)))
        self._defence_cache[key] = value
        return value

    @staticmethod
    def _marshal_shadow(
        observation: Observation,
        action: FugitiveAction | None = None,
    ) -> Observation:
        """Use the same public timeline for the inherited Manhunt response model."""

        if action is not None:
            shadow, _ = ContinuationCountFugitiveAgent._public_shadow_after(
                observation,
                (action,),
            )
            return replace(shadow, phase=Phase.MANHUNT)

        # ``_public_shadow_after(..., ())`` represents a candidate Pass.  The
        # inherited no-action caller needs only the existing public timeline.
        shadow, _ = ContinuationCountFugitiveAgent._public_shadow_after(
            observation,
            (),
        )
        return replace(shadow, play_history=shadow.play_history[:-1])

    @staticmethod
    def _public_shadow_after(
        observation: Observation,
        actions: tuple[FugitiveAction, ...],
    ) -> tuple[Observation, tuple[int, ...]]:
        """Project existing history and a candidate action to Marshal vision.

        An empty ``actions`` tuple denotes the hypothetical Pass candidate.
        This convention is used by ``_pass_score_with_feature``.
        """

        true_route = list(observation.route)
        public_route = [
            RouteView(
                index=slot.index,
                hideout=(
                    slot.hideout
                    if slot.revealed or slot.hideout in (0, 42)
                    else None
                ),
                sprint_count=slot.sprint_count,
                sprint_cards=(
                    slot.sprint_cards
                    if slot.revealed and slot.hideout != 42
                    else None
                ),
                revealed=slot.revealed,
            )
            for slot in observation.route
        ]

        public_plays: list[PlayView] = []
        for record in observation.play_history:
            if record.passed:
                public_plays.append(
                    PlayView(
                        hideout=None,
                        sprint_count=0,
                        sprint_cards=(),
                        route_index=None,
                        round_number=record.round_number,
                        passed=True,
                    )
                )
                continue
            assert record.route_index is not None
            slot = public_route[record.route_index]
            public_plays.append(
                PlayView(
                    hideout=slot.hideout,
                    sprint_count=record.sprint_count,
                    sprint_cards=slot.sprint_cards,
                    route_index=record.route_index,
                    round_number=record.round_number,
                    passed=False,
                )
            )

        hypothetical_actions = actions or (FugitiveAction(None),)
        for action in hypothetical_actions:
            if action.hideout is None:
                public_plays.append(
                    PlayView(
                        hideout=None,
                        sprint_count=0,
                        sprint_cards=(),
                        route_index=None,
                        round_number=observation.round_number,
                        passed=True,
                    )
                )
                continue

            route_index = len(true_route)
            true_route.append(
                RouteView(
                    index=route_index,
                    hideout=action.hideout,
                    sprint_count=len(action.sprint_cards),
                    sprint_cards=tuple(sorted(action.sprint_cards)),
                    revealed=False,
                )
            )
            public_slot = RouteView(
                index=route_index,
                hideout=42 if action.hideout == 42 else None,
                sprint_count=len(action.sprint_cards),
                sprint_cards=None,
                revealed=False,
            )
            public_route.append(public_slot)
            public_plays.append(
                PlayView(
                    hideout=public_slot.hideout,
                    sprint_count=public_slot.sprint_count,
                    sprint_cards=None,
                    route_index=route_index,
                    round_number=observation.round_number,
                    passed=False,
                )
            )

        truth = tuple(
            slot.hideout
            for index, slot in enumerate(true_route)
            if index
            and not slot.revealed
            and slot.hideout not in (None, 42)
        )
        shadow = Observation(
            role=Role.MARSHAL,
            hand=(),
            pile_sizes=observation.pile_sizes,
            route=tuple(public_route),
            guess_history=observation.guess_history,
            draw_history=tuple(
                DrawRecord(
                    role=record.role,
                    pile=record.pile,
                    card=None,
                    round_number=record.round_number,
                )
                for record in observation.draw_history
            ),
            round_number=observation.round_number,
            phase=Phase.MARSHAL_GUESS,
            legal_draw_piles=(),
            play_history=tuple(public_plays),
        )
        return shadow, truth


__all__ = [
    "CATCH_RISK_FEATURE_ID",
    "CONCEALMENT_FEATURE_ID",
    "CONTINUATION_CACHE_ID",
    "CONTINUATION_COUNT_FUGITIVE_ALGORITHM_ID",
    "CONTINUATION_MASS_ID",
    "CONTINUATION_TRANSFORM_ID",
    "ContinuationCountFugitiveAgent",
    "PUBLIC_TIMELINE_ID",
    "candidate_relative_log1p_features",
]
