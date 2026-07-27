"""Paired, information-safe full-game rollout evaluation."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
import math
import random
from typing import TypeAlias

from .determinization import (
    FugitiveWorld,
    determinize_fugitive_game,
    determinize_game,
)
from .driver import PlayerDecision, apply_decision, role_for_phase, step_agent
from .engine import GameEngine
from .model import (
    FugitiveAgent,
    MarshalAgent,
    Observation,
    Role,
    Winner,
)
from .reproducibility import derive_seed, validate_seed
from .world_validation import CompleteMarshalWorld


FugitivePolicyFactory: TypeAlias = Callable[[int], FugitiveAgent]
MarshalPolicyFactory: TypeAlias = Callable[[int], MarshalAgent]

_ROLLOUT_SEED_DOMAIN = "fugitive.full-game-rollout.v1"


@dataclass(frozen=True, slots=True)
class RolloutActionResult:
    """Empirical value of one root action for the player making that action."""

    action: PlayerDecision
    wins: int
    samples: int

    @property
    def win_rate(self) -> float:
        return self.wins / self.samples


@dataclass(frozen=True, slots=True)
class RolloutBudget:
    """A deterministic cap on complete terminal simulations.

    The budget is divided into equally paired scenarios.  A remainder smaller
    than the evaluated candidate count is deliberately left unused so every
    candidate sees exactly the same roots and continuation-policy seeds.
    """

    max_terminal_simulations: int

    def __post_init__(self) -> None:
        _validate_positive_int(
            self.max_terminal_simulations,
            "max_terminal_simulations",
        )

    def plan(self, candidate_count: int) -> "RolloutPlan":
        candidate_count = _validate_positive_int(
            candidate_count,
            "candidate_count",
        )
        evaluated = min(candidate_count, self.max_terminal_simulations)
        if evaluated == 1:
            scenarios = 0
            simulations = 0
        else:
            scenarios = self.max_terminal_simulations // evaluated
            simulations = evaluated * scenarios
        return RolloutPlan(
            offered_candidates=candidate_count,
            evaluated_candidates=evaluated,
            paired_scenarios=scenarios,
            terminal_simulations=simulations,
            maximum_terminal_simulations=self.max_terminal_simulations,
            unused_terminal_simulations=(
                self.max_terminal_simulations - simulations
            ),
            single_candidate_shortcut=evaluated == 1,
        )


@dataclass(frozen=True, slots=True)
class RolloutPlan:
    """Resolved candidate/scenario allocation for one rollout decision."""

    offered_candidates: int
    evaluated_candidates: int
    paired_scenarios: int
    terminal_simulations: int
    maximum_terminal_simulations: int
    unused_terminal_simulations: int
    single_candidate_shortcut: bool

    def __post_init__(self) -> None:
        _validate_positive_int(self.offered_candidates, "offered_candidates")
        _validate_positive_int(self.evaluated_candidates, "evaluated_candidates")
        if self.evaluated_candidates > self.offered_candidates:
            raise ValueError("evaluated candidates cannot exceed offered candidates")
        for name in (
            "paired_scenarios",
            "terminal_simulations",
            "unused_terminal_simulations",
        ):
            _validate_non_negative_int(getattr(self, name), name)
        _validate_positive_int(
            self.maximum_terminal_simulations,
            "maximum_terminal_simulations",
        )
        if (
            self.terminal_simulations
            + self.unused_terminal_simulations
            != self.maximum_terminal_simulations
        ):
            raise ValueError("rollout budget accounting is inconsistent")
        if self.single_candidate_shortcut != (self.evaluated_candidates == 1):
            raise ValueError("single-candidate shortcut flag is inconsistent")
        expected = self.evaluated_candidates * self.paired_scenarios
        if self.terminal_simulations != expected:
            raise ValueError("terminal simulation count is inconsistent")

@dataclass(frozen=True, slots=True)
class PairedRolloutResult:
    """Paired binary outcomes for two actions on the same scenarios."""

    first_action: PlayerDecision
    second_action: PlayerDecision
    both_win: int
    first_only_win: int
    second_only_win: int
    both_loss: int

    def __post_init__(self) -> None:
        for name in (
            "both_win",
            "first_only_win",
            "second_only_win",
            "both_loss",
        ):
            _validate_non_negative_int(getattr(self, name), name)
        if self.samples < 1:
            raise ValueError("paired rollout result requires at least one sample")

    @property
    def samples(self) -> int:
        return (
            self.both_win
            + self.first_only_win
            + self.second_only_win
            + self.both_loss
        )

    @property
    def mean_difference(self) -> float:
        return (self.first_only_win - self.second_only_win) / self.samples

    @property
    def standard_error(self) -> float | None:
        if self.samples < 2:
            return None
        mean = self.mean_difference
        sum_squares = self.first_only_win + self.second_only_win
        sample_variance = (sum_squares - self.samples * mean * mean) / (
            self.samples - 1
        )
        return math.sqrt(max(0.0, sample_variance) / self.samples)


@dataclass(frozen=True, slots=True)
class RolloutEvaluation:
    """Detailed result of one common-random-numbers action comparison."""

    role: Role
    action_results: tuple[RolloutActionResult, ...]
    paired_results: tuple[PairedRolloutResult, ...]
    paired_scenarios: int
    terminal_simulations: int
    simulated_decisions: int

    def __post_init__(self) -> None:
        if not isinstance(self.role, Role):
            raise TypeError("rollout evaluation role must be a Role")
        object.__setattr__(self, "action_results", tuple(self.action_results))
        object.__setattr__(self, "paired_results", tuple(self.paired_results))
        if not self.action_results:
            raise ValueError("rollout evaluation requires action results")
        _validate_positive_int(self.paired_scenarios, "paired_scenarios")
        _validate_positive_int(self.terminal_simulations, "terminal_simulations")
        _validate_positive_int(self.simulated_decisions, "simulated_decisions")
        if any(
            result.samples != self.paired_scenarios
            for result in self.action_results
        ):
            raise ValueError("each action must use every paired scenario")
        if self.terminal_simulations != (
            len(self.action_results) * self.paired_scenarios
        ):
            raise ValueError("terminal simulation accounting is inconsistent")


class RolloutEvaluator:
    """Compare root actions with paired full-game simulations.

    Every candidate is applied to a clone of the same determinized root. For a
    given scenario and repetition, every candidate also receives continuation
    policies built from the same pair of seeds. These common random numbers
    reduce comparison noise without exposing the complete world to a policy.

    There is deliberately no horizon or emergency cutoff. Continuation
    policies must therefore eventually reach one of the two rules-defined
    terminal outcomes.
    """

    def __init__(
        self,
        fugitive_policy_factory: FugitivePolicyFactory,
        marshal_policy_factory: MarshalPolicyFactory,
        *,
        seed: int,
    ) -> None:
        if not callable(fugitive_policy_factory):
            raise TypeError("fugitive_policy_factory must be callable")
        if not callable(marshal_policy_factory):
            raise TypeError("marshal_policy_factory must be callable")
        self.fugitive_policy_factory = fugitive_policy_factory
        self.marshal_policy_factory = marshal_policy_factory
        self.seed = validate_seed(seed, "rollout seed")

    def evaluate_worlds(
        self,
        observation: Observation,
        worlds: Sequence[CompleteMarshalWorld],
        candidate_actions: Iterable[PlayerDecision],
        *,
        repetitions: int = 1,
    ) -> tuple[RolloutActionResult, ...]:
        """Evaluate Marshal actions over fixed complete-world samples.

        Worlds are treated as empirical samples. A caller with weighted
        particles should resample them before this method; importance weights
        are intentionally not reinterpreted here.
        """

        return self.evaluate_worlds_detailed(
            observation,
            worlds,
            candidate_actions,
            repetitions=repetitions,
        ).action_results

    def evaluate_worlds_detailed(
        self,
        observation: Observation,
        worlds: Sequence[CompleteMarshalWorld],
        candidate_actions: Iterable[PlayerDecision],
        *,
        repetitions: int = 1,
    ) -> RolloutEvaluation:
        """Return action values plus paired outcomes and work counts."""

        if observation.role is not Role.MARSHAL:
            raise ValueError("complete Marshal worlds require a Marshal observation")
        if role_for_phase(observation.phase) is not Role.MARSHAL:
            raise ValueError("the observation must be a Marshal decision state")
        roots = tuple(worlds)
        if not roots:
            raise ValueError("at least one complete world is required")
        repetitions = _validate_repetitions(repetitions)
        actions = _validate_candidate_actions(candidate_actions)

        def scenarios() -> Iterable[tuple[int, int, GameEngine]]:
            for world_index, world in enumerate(roots):
                for repetition in range(repetitions):
                    pile_seed = self._sample_seed(
                        world_index,
                        repetition,
                        "pile-order",
                    )
                    yield (
                        world_index,
                        repetition,
                        determinize_game(
                            observation,
                            world,
                            rng=random.Random(pile_seed),
                        ),
                    )

        samples = len(roots) * repetitions
        return self._evaluate(Role.MARSHAL, scenarios(), actions, samples)

    def evaluate_fugitive_worlds(
        self,
        observation: Observation,
        worlds: Sequence[FugitiveWorld],
        candidate_actions: Iterable[PlayerDecision],
        *,
        repetitions: int = 1,
    ) -> tuple[RolloutActionResult, ...]:
        """Evaluate Fugitive actions over sampled opponent/deck worlds."""

        return self.evaluate_fugitive_worlds_detailed(
            observation,
            worlds,
            candidate_actions,
            repetitions=repetitions,
        ).action_results

    def evaluate_fugitive_worlds_detailed(
        self,
        observation: Observation,
        worlds: Sequence[FugitiveWorld],
        candidate_actions: Iterable[PlayerDecision],
        *,
        repetitions: int = 1,
    ) -> RolloutEvaluation:
        """Return Fugitive action values plus paired outcomes and work."""

        if observation.role is not Role.FUGITIVE:
            raise ValueError("Fugitive worlds require a Fugitive observation")
        if role_for_phase(observation.phase) is not Role.FUGITIVE:
            raise ValueError("the observation must be a Fugitive decision state")
        roots = tuple(worlds)
        if not roots:
            raise ValueError("at least one Fugitive world is required")
        repetitions = _validate_repetitions(repetitions)
        actions = _validate_candidate_actions(candidate_actions)

        def scenarios() -> Iterable[tuple[int, int, GameEngine]]:
            for world_index, world in enumerate(roots):
                for repetition in range(repetitions):
                    pile_seed = self._sample_seed(
                        world_index,
                        repetition,
                        "pile-order",
                    )
                    yield (
                        world_index,
                        repetition,
                        determinize_fugitive_game(
                            observation,
                            world,
                            rng=random.Random(pile_seed),
                        ),
                    )

        samples = len(roots) * repetitions
        return self._evaluate(Role.FUGITIVE, scenarios(), actions, samples)

    def evaluate_engines(
        self,
        engines: Sequence[GameEngine],
        candidate_actions: Iterable[PlayerDecision],
        *,
        repetitions: int = 1,
    ) -> tuple[RolloutActionResult, ...]:
        """Evaluate actions from one or more known true engine states.

        This entry point is useful for tests, oracle diagnostics, and games in
        which the harness legitimately knows both players' private state. The
        true state is never passed to either continuation policy.
        """

        return self.evaluate_engines_detailed(
            engines,
            candidate_actions,
            repetitions=repetitions,
        ).action_results

    def evaluate_engines_detailed(
        self,
        engines: Sequence[GameEngine],
        candidate_actions: Iterable[PlayerDecision],
        *,
        repetitions: int = 1,
    ) -> RolloutEvaluation:
        """Return known-root action values plus paired outcomes and work."""

        roots = tuple(engines)
        if not roots:
            raise ValueError("at least one engine is required")
        repetitions = _validate_repetitions(repetitions)
        actions = _validate_candidate_actions(candidate_actions)

        role = role_for_phase(roots[0].phase)
        if role is None:
            raise ValueError("rollout roots must be non-terminal")
        if any(
            engine.is_terminal or role_for_phase(engine.phase) is not role
            for engine in roots
        ):
            raise ValueError("all rollout roots must share one acting role")

        scenarios = (
            (engine_index, repetition, engine.clone())
            for engine_index, engine in enumerate(roots)
            for repetition in range(repetitions)
        )
        samples = len(roots) * repetitions
        return self._evaluate(role, scenarios, actions, samples)

    def _evaluate(
        self,
        role: Role,
        scenarios: Iterable[tuple[int, int, GameEngine]],
        actions: tuple[PlayerDecision, ...],
        samples: int,
    ) -> RolloutEvaluation:
        outcomes: list[list[bool]] = [[] for _ in actions]
        expected_winner = Winner(role.value)
        simulated_decisions = 0

        for scenario_index, repetition, root in scenarios:
            fugitive_seed = self._sample_seed(
                scenario_index,
                repetition,
                "fugitive-policy",
            )
            marshal_seed = self._sample_seed(
                scenario_index,
                repetition,
                "marshal-policy",
            )
            for action_index, action in enumerate(actions):
                engine = root.clone()
                fugitive = self.fugitive_policy_factory(fugitive_seed)
                marshal = self.marshal_policy_factory(marshal_seed)
                apply_decision(engine, action, decision=1)

                decision = 1
                while not engine.is_terminal:
                    decision += 1
                    step_agent(
                        engine,
                        fugitive,
                        marshal,
                        decision=decision,
                    )
                simulated_decisions += decision
                outcomes[action_index].append(
                    engine.result().winner is expected_winner
                )

        action_results = tuple(
            RolloutActionResult(action, sum(outcomes[index]), samples)
            for index, action in enumerate(actions)
        )
        paired_results = tuple(
            _paired_result(
                actions[first],
                outcomes[first],
                actions[second],
                outcomes[second],
            )
            for first in range(len(actions))
            for second in range(first + 1, len(actions))
        )
        return RolloutEvaluation(
            role=role,
            action_results=action_results,
            paired_results=paired_results,
            paired_scenarios=samples,
            terminal_simulations=len(actions) * samples,
            simulated_decisions=simulated_decisions,
        )

    def _sample_seed(
        self,
        scenario_index: int,
        repetition: int,
        stream: str,
    ) -> int:
        return derive_seed(
            self.seed,
            (
                f"{_ROLLOUT_SEED_DOMAIN}:scenario={scenario_index}:"
                f"repetition={repetition}:{stream}"
            ),
        )


def _validate_candidate_actions(
    candidate_actions: Iterable[PlayerDecision],
) -> tuple[PlayerDecision, ...]:
    actions = tuple(candidate_actions)
    if not actions:
        raise ValueError("at least one candidate action is required")
    if len(set(actions)) != len(actions):
        raise ValueError("candidate actions must be distinct")
    return actions


def _validate_repetitions(repetitions: int) -> int:
    if (
        isinstance(repetitions, bool)
        or not isinstance(repetitions, int)
        or repetitions < 1
    ):
        raise ValueError("repetitions must be a positive integer")
    return repetitions


def _paired_result(
    first_action: PlayerDecision,
    first_outcomes: Sequence[bool],
    second_action: PlayerDecision,
    second_outcomes: Sequence[bool],
) -> PairedRolloutResult:
    if len(first_outcomes) != len(second_outcomes) or not first_outcomes:
        raise ValueError("paired actions must share a nonempty scenario set")
    both_win = first_only = second_only = both_loss = 0
    for first, second in zip(first_outcomes, second_outcomes, strict=True):
        if first and second:
            both_win += 1
        elif first:
            first_only += 1
        elif second:
            second_only += 1
        else:
            both_loss += 1
    return PairedRolloutResult(
        first_action=first_action,
        second_action=second_action,
        both_win=both_win,
        first_only_win=first_only,
        second_only_win=second_only,
        both_loss=both_loss,
    )


def _validate_non_negative_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _validate_positive_int(value: int, name: str) -> int:
    value = _validate_non_negative_int(value, name)
    if value == 0:
        raise ValueError(f"{name} must be positive")
    return value


__all__ = [
    "FugitiveWorld",
    "FugitivePolicyFactory",
    "MarshalPolicyFactory",
    "PairedRolloutResult",
    "RolloutActionResult",
    "RolloutBudget",
    "RolloutEvaluation",
    "RolloutEvaluator",
    "RolloutPlan",
    "determinize_fugitive_game",
    "determinize_game",
]
