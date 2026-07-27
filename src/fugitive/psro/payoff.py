"""Full-game payoff tasks, sampling schedules, and parallel evaluation."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Callable, Mapping, Sequence, cast

from ..runtime.manifest import (
    RULES_SHA256,
    RULES_VERSION,
    MatchStatus,
)
from ..runtime.runner import run_registered_match
from ..game.model import Role, Winner
from .ledger import PayoffLedger, PayoffSample, deterministic_sample_id
from .payoff_matrix import (
    PayoffEstimate,
    PayoffEvaluation,
    PayoffPair,
    PayoffRequest,
)
from .population import (
    PolicyIdentifier,
    PolicyMixture,
    PolicyPopulation,
    PolicySpec,
)
from .policy_adapter import agent_spec_from_policy
from .validation import (
    FinalHoldoutReport,
    ResponseValidationReport,
    bootstrap_mean_interval,
)
from ..shared.reproducibility import (
    AgentSpec,
    JSONValue,
    derive_seed,
    parse_seed,
    thaw_parameters,
    validate_seed,
)


PSRO_PAYOFF_SEED_DOMAIN = "fugitive.psro.payoff-game.v3"
PSRO_META_TRAIN_SEED_DOMAIN = f"{PSRO_PAYOFF_SEED_DOMAIN}:meta-train"
PSRO_RESPONSE_VALIDATION_SEED_DOMAIN = (
    f"{PSRO_PAYOFF_SEED_DOMAIN}:response-validation"
)
PSRO_FINAL_HOLDOUT_SEED_DOMAIN = f"{PSRO_PAYOFF_SEED_DOMAIN}:final-holdout"


def _require_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return cast(Mapping[str, object], value)


def _require_sequence(value: object, label: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{label} must be a JSON array")
    return cast(Sequence[object], value)


def _optional_game_count(
    value: int | None,
    *,
    default: int,
    label: str,
) -> int:
    resolved = default if value is None else value
    if (
        isinstance(resolved, bool)
        or not isinstance(resolved, int)
        or resolved < 0
    ):
        raise ValueError(f"{label} must be a non-negative integer or None")
    return resolved


@dataclass(frozen=True, slots=True)
class RegisteredGamePayoffConfig:
    """Three domain-separated common-random-number schedules.

    Only ``meta_train_seeds`` populate the restricted payoff matrix.
    Validation and holdout schedules are kept separate so adaptive policy
    selection never silently reuses the final reporting games.
    """

    meta_train_seeds: tuple[int, ...]
    response_validation_seeds: tuple[int, ...] = ()
    final_holdout_seeds: tuple[int, ...] = ()
    response_minimum_gain: float = 0.0
    confidence_level: float = 0.95
    bootstrap_replicates: int = 10_000
    validate_invariants: bool = True
    max_decisions: int | None = None

    def __post_init__(self) -> None:
        schedules = {
            "meta-train": tuple(self.meta_train_seeds),
            "response-validation": tuple(self.response_validation_seeds),
            "final-holdout": tuple(self.final_holdout_seeds),
        }
        checked: dict[str, tuple[int, ...]] = {}
        for name, values in schedules.items():
            checked[name] = tuple(
                validate_seed(seed, f"PSRO {name} seed") for seed in values
            )
            if len(set(checked[name])) != len(checked[name]):
                raise ValueError(f"PSRO {name} seeds must be distinct")
        if not checked["meta-train"]:
            raise ValueError("PSRO meta-train evaluation requires at least one seed")
        all_seeds = tuple(seed for values in checked.values() for seed in values)
        if len(set(all_seeds)) != len(all_seeds):
            raise ValueError("PSRO seed schedules must be domain-separated")
        for value, label in (
            (self.response_minimum_gain, "response_minimum_gain"),
            (self.confidence_level, "confidence_level"),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"{label} must be finite")
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must be between zero and one")
        if (
            isinstance(self.bootstrap_replicates, bool)
            or not isinstance(self.bootstrap_replicates, int)
            or self.bootstrap_replicates <= 0
        ):
            raise ValueError("bootstrap_replicates must be a positive integer")
        if not isinstance(self.validate_invariants, bool):
            raise ValueError("validate_invariants must be Boolean")
        if self.max_decisions is not None and (
            isinstance(self.max_decisions, bool)
            or not isinstance(self.max_decisions, int)
            or self.max_decisions < 0
        ):
            raise ValueError("max_decisions must be a non-negative integer or None")
        object.__setattr__(self, "meta_train_seeds", checked["meta-train"])
        object.__setattr__(
            self,
            "response_validation_seeds",
            checked["response-validation"],
        )
        object.__setattr__(
            self,
            "final_holdout_seeds",
            checked["final-holdout"],
        )

    @classmethod
    def derived(
        cls,
        master_seed: int,
        games_per_cell: int,
        *,
        response_validation_games: int | None = None,
        final_holdout_games: int | None = None,
        response_minimum_gain: float = 0.0,
        confidence_level: float = 0.95,
        bootstrap_replicates: int = 10_000,
        validate_invariants: bool = True,
        max_decisions: int | None = None,
    ) -> "RegisteredGamePayoffConfig":
        master_seed = validate_seed(master_seed, "PSRO schedule seed")
        if (
            isinstance(games_per_cell, bool)
            or not isinstance(games_per_cell, int)
            or games_per_cell <= 0
        ):
            raise ValueError("games_per_cell must be a positive integer")
        validation_count = _optional_game_count(
            response_validation_games,
            default=games_per_cell,
            label="response_validation_games",
        )
        holdout_count = _optional_game_count(
            final_holdout_games,
            default=games_per_cell,
            label="final_holdout_games",
        )

        def schedule(domain: str, count: int) -> tuple[int, ...]:
            return tuple(
                derive_seed(master_seed, f"{domain}:{index}")
                for index in range(count)
            )

        return cls(
            meta_train_seeds=schedule(
                PSRO_META_TRAIN_SEED_DOMAIN, games_per_cell
            ),
            response_validation_seeds=schedule(
                PSRO_RESPONSE_VALIDATION_SEED_DOMAIN, validation_count
            ),
            final_holdout_seeds=schedule(
                PSRO_FINAL_HOLDOUT_SEED_DOMAIN, holdout_count
            ),
            response_minimum_gain=response_minimum_gain,
            confidence_level=confidence_level,
            bootstrap_replicates=bootstrap_replicates,
            validate_invariants=validate_invariants,
            max_decisions=max_decisions,
        )

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "seed_protocol": PSRO_PAYOFF_SEED_DOMAIN,
            "meta_train_seeds": [str(seed) for seed in self.meta_train_seeds],
            "response_validation_seeds": [
                str(seed) for seed in self.response_validation_seeds
            ],
            "final_holdout_seeds": [
                str(seed) for seed in self.final_holdout_seeds
            ],
            "response_minimum_gain": self.response_minimum_gain,
            "confidence_level": self.confidence_level,
            "bootstrap_replicates": self.bootstrap_replicates,
            "validate_invariants": self.validate_invariants,
            "max_decisions": self.max_decisions,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "RegisteredGamePayoffConfig":
        if data.get("seed_protocol") != PSRO_PAYOFF_SEED_DOMAIN:
            raise ValueError("unsupported PSRO payoff seed protocol")
        validate_invariants = data.get("validate_invariants")
        if not isinstance(validate_invariants, bool):
            raise ValueError("validate_invariants must be Boolean")
        max_decisions = data.get("max_decisions")
        return cls(
            tuple(
                parse_seed(seed, "PSRO meta-train")
                for seed in _require_sequence(
                    data.get("meta_train_seeds"), "PSRO meta-train seeds"
                )
            ),
            tuple(
                parse_seed(seed, "PSRO response-validation")
                for seed in _require_sequence(
                    data.get("response_validation_seeds"),
                    "PSRO response-validation seeds",
                )
            ),
            tuple(
                parse_seed(seed, "PSRO final-holdout")
                for seed in _require_sequence(
                    data.get("final_holdout_seeds"),
                    "PSRO final-holdout seeds",
                )
            ),
            response_minimum_gain=cast(float, data.get("response_minimum_gain")),
            confidence_level=cast(float, data.get("confidence_level")),
            bootstrap_replicates=cast(int, data.get("bootstrap_replicates")),
            validate_invariants=validate_invariants,
            max_decisions=(
                (
                    None
                    if max_decisions is None
                    else cast(int, max_decisions)
                )
            ),
        )


class RegisteredPayoffEvaluationError(RuntimeError):
    """A requested full game failed to produce a rules-defined winner."""


@dataclass(frozen=True, slots=True)
class PayoffEvaluationProgress:
    completed_games: int
    total_games: int
    reused_games: int
    pair: PayoffPair
    sample_key: str
    schedule: str


@dataclass(frozen=True, slots=True)
class _RegisteredPayoffGame:
    pair: PayoffPair
    marshal_spec: AgentSpec
    fugitive_spec: AgentSpec
    sample_key: str
    schedule: str
    master_seed: int


@dataclass(frozen=True, slots=True)
class _RegisteredPayoffGameTask:
    experiment_id: str
    pair: dict[str, JSONValue]
    marshal_spec: dict[str, JSONValue]
    fugitive_spec: dict[str, JSONValue]
    sample_key: str
    schedule: str
    master_seed: int
    validate_invariants: bool
    max_decisions: int | None


def _payoff_experiment_id(
    marshal_spec: AgentSpec,
    fugitive_spec: AgentSpec,
    config: RegisteredGamePayoffConfig,
) -> str:
    identity = {
        "protocol": "registered-full-game-payoff-v3",
        "seed_protocol": PSRO_PAYOFF_SEED_DOMAIN,
        "rules_version": RULES_VERSION,
        "rules_sha256": RULES_SHA256,
        "marshal_spec": marshal_spec.to_dict(),
        "fugitive_spec": fugitive_spec.to_dict(),
        "validate_invariants": config.validate_invariants,
        "max_decisions": config.max_decisions,
    }
    encoded = json.dumps(
        identity,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return f"registered-full-game-v3:{hashlib.sha256(encoded).hexdigest()}"


def _run_registered_payoff_game(
    task: _RegisteredPayoffGameTask,
) -> dict[str, JSONValue]:
    pair = PayoffPair.from_dict(task.pair)
    marshal_spec = AgentSpec.from_dict(task.marshal_spec)
    fugitive_spec = AgentSpec.from_dict(task.fugitive_spec)
    run = run_registered_match(
        master_seed=task.master_seed,
        fugitive_name=fugitive_spec.name,
        marshal_name=marshal_spec.name,
        fugitive_parameters=thaw_parameters(fugitive_spec.parameters),
        marshal_parameters=thaw_parameters(marshal_spec.parameters),
        fugitive_profile=fugitive_spec.profile,
        marshal_profile=marshal_spec.profile,
        max_decisions=task.max_decisions,
        validate_invariants=task.validate_invariants,
    )
    if run.status is not MatchStatus.COMPLETED or run.game_result is None:
        raise RegisteredPayoffEvaluationError(
            "PSRO payoff game did not complete: "
            f"pair={pair.marshal.name}/{pair.fugitive.name}, "
            f"seed={task.master_seed}, status={run.status.value}"
        )
    result = run.game_result
    sample = PayoffSample(
        experiment_id=task.experiment_id,
        pair=pair,
        sample_key=task.sample_key,
        marshal_payoff=float(result.winner is Winner.MARSHAL),
        outcome={
            "winner": result.winner.value,
            "reason": result.reason,
            "rounds": result.rounds,
            "manifest": run.manifest.to_dict(),
        },
        metadata={
            "schedule": task.schedule,
            "master_seed": str(task.master_seed),
            "rules_version": RULES_VERSION,
            "rules_sha256": RULES_SHA256,
        },
    )
    return sample.to_dict()


def _sample_mixture_policy(
    mixture: PolicyMixture,
    seed: int,
) -> PolicyIdentifier:
    """Draw one pure policy from a mixture with an isolated random stream."""

    return mixture.sample(random.Random(validate_seed(seed, "mixture seed")))


class RegisteredGamePayoffEvaluator:
    """Evaluate, persist, resume, and optionally parallelize full games."""

    def __init__(
        self,
        config: RegisteredGamePayoffConfig,
        *,
        ledger: PayoffLedger | str | Path | None = None,
        workers: int = 1,
        progress: Callable[[PayoffEvaluationProgress], None] | None = None,
        repository: PolicyPopulation | None = None,
    ) -> None:
        if not isinstance(config, RegisteredGamePayoffConfig):
            raise ValueError("registered payoff evaluator requires its config")
        if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
            raise ValueError("payoff workers must be a positive integer")
        if progress is not None and not callable(progress):
            raise ValueError("payoff progress callback must be callable")
        self.config = config
        self.ledger = (
            ledger
            if isinstance(ledger, PayoffLedger) or ledger is None
            else PayoffLedger(ledger)
        )
        self.workers = workers
        self.progress = progress
        self.repository = repository

    def bind_population(self, population: PolicyPopulation) -> None:
        """Set the ID-to-policy repository used by deferred search specs."""

        if not isinstance(population, PolicyPopulation):
            raise ValueError("payoff repository must be a PolicyPopulation")
        self.repository = population

    def evaluate(
        self, requests: tuple[PayoffRequest, ...]
    ) -> tuple[PayoffEvaluation, ...]:
        requests = tuple(requests)
        if not requests:
            return ()
        games: list[_RegisteredPayoffGame] = []
        for request in requests:
            marshal_spec = agent_spec_from_policy(
                request.marshal_policy,
                self.repository,
            )
            fugitive_spec = agent_spec_from_policy(
                request.fugitive_policy,
                self.repository,
            )
            for index, master_seed in enumerate(self.config.meta_train_seeds):
                games.append(
                    _RegisteredPayoffGame(
                        request.pair,
                        marshal_spec,
                        fugitive_spec,
                        f"meta-train:{index}:{master_seed}",
                        "meta-train",
                        master_seed,
                    )
                )
        samples = self._run_games(tuple(games))
        by_pair: dict[PayoffPair, list[float]] = {
            request.pair: [] for request in requests
        }
        for sample in samples:
            by_pair[sample.pair].append(sample.marshal_payoff)
        evaluations: list[PayoffEvaluation] = []
        for request in requests:
            values = by_pair[request.pair]
            if len(values) != len(self.config.meta_train_seeds):
                raise RuntimeError("PSRO payoff ledger is missing completed games")
            mean = math.fsum(values) / len(values)
            standard_error = None
            if len(values) > 1:
                variance = math.fsum(
                    (value - mean) ** 2 for value in values
                ) / (len(values) - 1)
                standard_error = math.sqrt(variance / len(values))
            evaluations.append(
                PayoffEvaluation(
                    request.pair,
                    PayoffEstimate(mean, len(values), standard_error),
                )
            )
        return tuple(evaluations)

    def validate_response(
        self,
        *,
        role: Role,
        generation: int,
        candidate: PolicySpec,
        incumbent: PolicySpec,
        opponent_mixture: PolicyMixture,
        population: PolicyPopulation,
    ) -> ResponseValidationReport:
        """Compare a candidate and incumbent on paired fresh games."""

        base_seeds = self.config.response_validation_seeds
        if not base_seeds:
            raise ValueError(
                "a PSRO response iteration requires response-validation games"
            )
        seeds = tuple(
            derive_seed(
                seed,
                f"{PSRO_RESPONSE_VALIDATION_SEED_DOMAIN}:generation={generation}",
            )
            for seed in base_seeds
        )
        self.bind_population(population)
        candidate_spec = agent_spec_from_policy(candidate, population)
        incumbent_spec = agent_spec_from_policy(incumbent, population)
        games: list[_RegisteredPayoffGame] = []
        opponent_draws: list[PolicyIdentifier] = []
        for index, master_seed in enumerate(seeds):
            opponent_id = _sample_mixture_policy(
                opponent_mixture,
                derive_seed(
                    master_seed,
                    f"response-validation-opponent:{generation}:{role.value}",
                ),
            )
            opponent_draws.append(opponent_id)
            opponent = population.policy(opponent_id)
            opponent_spec = agent_spec_from_policy(opponent, population)
            sample_key = f"response-validation:{index}:{master_seed}"
            if role is Role.MARSHAL:
                games.extend(
                    (
                        _RegisteredPayoffGame(
                            PayoffPair(candidate.identifier, opponent_id),
                            candidate_spec,
                            opponent_spec,
                            sample_key,
                            "response-validation",
                            master_seed,
                        ),
                        _RegisteredPayoffGame(
                            PayoffPair(incumbent.identifier, opponent_id),
                            incumbent_spec,
                            opponent_spec,
                            sample_key,
                            "response-validation",
                            master_seed,
                        ),
                    )
                )
            else:
                games.extend(
                    (
                        _RegisteredPayoffGame(
                            PayoffPair(opponent_id, candidate.identifier),
                            opponent_spec,
                            candidate_spec,
                            sample_key,
                            "response-validation",
                            master_seed,
                        ),
                        _RegisteredPayoffGame(
                            PayoffPair(opponent_id, incumbent.identifier),
                            opponent_spec,
                            incumbent_spec,
                            sample_key,
                            "response-validation",
                            master_seed,
                        ),
                    )
                )
        samples = self._run_games(tuple(games))
        gains = tuple(
            (
                samples[index].marshal_payoff - samples[index + 1].marshal_payoff
                if role is Role.MARSHAL
                else samples[index + 1].marshal_payoff
                - samples[index].marshal_payoff
            )
            for index in range(0, len(samples), 2)
        )
        bootstrap_seed = derive_seed(
            seeds[0],
            "response-validation-bootstrap:"
            f"{generation}:{role.value}:"
            f"{candidate.identifier.name}:{incumbent.identifier.name}",
        )
        mean, lower, upper = bootstrap_mean_interval(
            gains,
            confidence_level=self.config.confidence_level,
            replicates=self.config.bootstrap_replicates,
            seed=bootstrap_seed,
        )
        return ResponseValidationReport(
            role=role,
            generation=generation,
            candidate=candidate.identifier,
            incumbent=incumbent.identifier,
            opponent_draws=tuple(opponent_draws),
            paired_gains=gains,
            mean_gain=mean,
            confidence_lower_bound=lower,
            confidence_upper_bound=upper,
            confidence_level=self.config.confidence_level,
            bootstrap_replicates=self.config.bootstrap_replicates,
            bootstrap_seed=bootstrap_seed,
            minimum_gain=self.config.response_minimum_gain,
            admitted=lower > self.config.response_minimum_gain,
        )

    def evaluate_final_holdout(
        self,
        population: PolicyPopulation,
        marshal_mixture: PolicyMixture,
        fugitive_mixture: PolicyMixture,
    ) -> FinalHoldoutReport | None:
        """Evaluate the final mixed profile without updating selection data."""

        seeds = self.config.final_holdout_seeds
        if not seeds:
            return None
        self.bind_population(population)
        games: list[_RegisteredPayoffGame] = []
        pairs: list[PayoffPair] = []
        for index, master_seed in enumerate(seeds):
            marshal_id = _sample_mixture_policy(
                marshal_mixture,
                derive_seed(master_seed, "final-holdout-marshal-mixture"),
            )
            fugitive_id = _sample_mixture_policy(
                fugitive_mixture,
                derive_seed(master_seed, "final-holdout-fugitive-mixture"),
            )
            pair = PayoffPair(marshal_id, fugitive_id)
            pairs.append(pair)
            marshal = population.policy(marshal_id)
            fugitive = population.policy(fugitive_id)
            games.append(
                _RegisteredPayoffGame(
                    pair,
                    agent_spec_from_policy(marshal, population),
                    agent_spec_from_policy(fugitive, population),
                    f"final-holdout:{index}:{master_seed}",
                    "final-holdout",
                    master_seed,
                )
            )
        samples = self._run_games(tuple(games))
        payoffs = tuple(sample.marshal_payoff for sample in samples)
        bootstrap_seed = derive_seed(seeds[0], "final-holdout-bootstrap")
        mean, lower, upper = bootstrap_mean_interval(
            payoffs,
            confidence_level=self.config.confidence_level,
            replicates=self.config.bootstrap_replicates,
            seed=bootstrap_seed,
        )
        return FinalHoldoutReport(
            sampled_pairs=tuple(pairs),
            marshal_payoffs=payoffs,
            mean_marshal_payoff=mean,
            confidence_lower_bound=lower,
            confidence_upper_bound=upper,
            confidence_level=self.config.confidence_level,
            bootstrap_replicates=self.config.bootstrap_replicates,
            bootstrap_seed=bootstrap_seed,
        )

    def _run_games(
        self,
        games: tuple[_RegisteredPayoffGame, ...],
    ) -> tuple[PayoffSample, ...]:
        """Run each distinct logical game once and preserve requested order."""

        if not games:
            return ()
        sample_ids: list[str] = []
        samples: dict[str, PayoffSample] = {}
        tasks: dict[str, _RegisteredPayoffGameTask] = {}
        for game in games:
            experiment_id = _payoff_experiment_id(
                game.marshal_spec,
                game.fugitive_spec,
                self.config,
            )
            sample_id = deterministic_sample_id(
                experiment_id, game.pair, game.sample_key
            )
            sample_ids.append(sample_id)
            existing = None if self.ledger is None else self.ledger.get(sample_id)
            if existing is not None:
                samples[sample_id] = existing
                continue
            task = _RegisteredPayoffGameTask(
                experiment_id=experiment_id,
                pair=game.pair.to_dict(),
                marshal_spec=game.marshal_spec.to_dict(),
                fugitive_spec=game.fugitive_spec.to_dict(),
                sample_key=game.sample_key,
                schedule=game.schedule,
                master_seed=game.master_seed,
                validate_invariants=self.config.validate_invariants,
                max_decisions=self.config.max_decisions,
            )
            previous = tasks.get(sample_id)
            if previous is not None and previous != task:
                raise ValueError("one payoff sample ID describes two games")
            tasks[sample_id] = task

        reused = len(samples)
        completed = 0
        total = reused + len(tasks)
        for sample_id in samples:
            sample = samples[sample_id]
            completed += 1
            self._report(completed, total, reused, sample)
        if self.workers == 1 or len(tasks) < 2:
            produced = (
                _run_registered_payoff_game(task) for task in tasks.values()
            )
            for payload in produced:
                completed += 1
                sample = PayoffSample.from_dict(payload)
                self._record(sample)
                samples[sample.sample_id] = sample
                self._report(completed, total, reused, sample)
        else:
            with ProcessPoolExecutor(
                max_workers=min(self.workers, len(tasks))
            ) as executor:
                futures = {
                    executor.submit(_run_registered_payoff_game, task): sample_id
                    for sample_id, task in tasks.items()
                }
                for future in as_completed(futures):
                    completed += 1
                    sample = PayoffSample.from_dict(future.result())
                    self._record(sample)
                    samples[sample.sample_id] = sample
                    self._report(completed, total, reused, sample)
        return tuple(samples[sample_id] for sample_id in sample_ids)

    def _record(self, sample: PayoffSample) -> None:
        if self.ledger is not None:
            self.ledger.append(sample)

    def _report(
        self,
        completed: int,
        total: int,
        reused: int,
        sample: PayoffSample,
    ) -> None:
        if self.progress is not None:
            self.progress(
                PayoffEvaluationProgress(
                    completed_games=completed,
                    total_games=total,
                    reused_games=reused,
                    pair=sample.pair,
                    sample_key=sample.sample_key,
                    schedule=cast(str, sample.metadata["schedule"]),
                )
            )


__all__ = [
    "PSRO_FINAL_HOLDOUT_SEED_DOMAIN",
    "PSRO_META_TRAIN_SEED_DOMAIN",
    "PSRO_PAYOFF_SEED_DOMAIN",
    "PSRO_RESPONSE_VALIDATION_SEED_DOMAIN",
    "PayoffEvaluationProgress",
    "RegisteredGamePayoffConfig",
    "RegisteredGamePayoffEvaluator",
    "RegisteredPayoffEvaluationError",
]
