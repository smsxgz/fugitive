"""Checkpointing and iteration orchestration for registered full-game PSRO."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Mapping, Sequence, cast

from .experiment import RULES_SHA256, RULES_VERSION
from .model import Role
from .psro import (
    EmpiricalPayoffMatrix,
    MetaSolverConfig,
    MultiplicativeWeightsMetaSolver,
    PSROCheckpoint,
    PayoffEvaluator,
    PolicyMixture,
    PolicyPopulation,
    PolicySpec,
    ResponseOracleRequest,
    initialize_psro,
    run_psro_iteration,
)
from .psro_payoff import (
    FinalHoldoutReport,
    RegisteredGamePayoffConfig,
    RegisteredGamePayoffEvaluator,
    ResponseValidationReport,
)
from .psro_policy_adapter import (
    MixtureConditionedResponseOracle,
    MixtureResponseTemplate,
)
from .reproducibility import JSONValue


PSRO_EXPERIMENT_CHECKPOINT_SCHEMA = "fugitive.psro-experiment-checkpoint"
PSRO_EXPERIMENT_CHECKPOINT_VERSION = 5
PSRO_EXPERIMENT_RUN_SCHEMA = "fugitive.psro-experiment-run"
PSRO_EXPERIMENT_RUN_VERSION = 1


def _require_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return cast(Mapping[str, object], value)


def _require_sequence(value: object, label: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{label} must be a JSON array")
    return cast(Sequence[object], value)


@dataclass(frozen=True, slots=True)
class PSROExperimentRunConfig:
    """Everything needed to resume the next registered PSRO generation."""

    payoff_config: RegisteredGamePayoffConfig
    meta_solver_config: MetaSolverConfig
    marshal_response_template: MixtureResponseTemplate
    fugitive_response_template: MixtureResponseTemplate

    def __post_init__(self) -> None:
        if not isinstance(self.payoff_config, RegisteredGamePayoffConfig):
            raise ValueError("run config requires a payoff config")
        if not isinstance(self.meta_solver_config, MetaSolverConfig):
            raise ValueError("run config requires a meta-solver config")
        if (
            not isinstance(self.marshal_response_template, MixtureResponseTemplate)
            or self.marshal_response_template.role is not Role.MARSHAL
        ):
            raise ValueError("run config requires a Marshal response template")
        if (
            not isinstance(self.fugitive_response_template, MixtureResponseTemplate)
            or self.fugitive_response_template.role is not Role.FUGITIVE
        ):
            raise ValueError("run config requires a Fugitive response template")

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "rules": {
                "version": RULES_VERSION,
                "sha256": RULES_SHA256,
            },
            "payoff_config": self.payoff_config.to_dict(),
            "meta_solver_config": self.meta_solver_config.to_dict(),
            "marshal_response_template": self.marshal_response_template.to_dict(),
            "fugitive_response_template": self.fugitive_response_template.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "PSROExperimentRunConfig":
        rules = _require_mapping(data.get("rules"), "rules fingerprint")
        if (
            rules.get("version") != RULES_VERSION
            or rules.get("sha256") != RULES_SHA256
        ):
            raise ValueError("PSRO checkpoint uses a different rules fingerprint")
        return cls(
            payoff_config=RegisteredGamePayoffConfig.from_dict(
                _require_mapping(data.get("payoff_config"), "payoff config")
            ),
            meta_solver_config=MetaSolverConfig.from_dict(
                _require_mapping(data.get("meta_solver_config"), "meta-solver config")
            ),
            marshal_response_template=MixtureResponseTemplate.from_dict(
                _require_mapping(
                    data.get("marshal_response_template"),
                    "Marshal response template",
                )
            ),
            fugitive_response_template=MixtureResponseTemplate.from_dict(
                _require_mapping(
                    data.get("fugitive_response_template"),
                    "Fugitive response template",
                )
            ),
        )


@dataclass(frozen=True, slots=True)
class PSROExperimentCheckpoint:
    """A generation plus its complete immutable registered-run definition."""

    psro: PSROCheckpoint
    run_config: PSROExperimentRunConfig
    response_validations: tuple[ResponseValidationReport, ...] = ()
    final_holdout: FinalHoldoutReport | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.psro, PSROCheckpoint):
            raise ValueError("experiment checkpoint requires a PSRO checkpoint")
        if not isinstance(self.run_config, PSROExperimentRunConfig):
            raise ValueError("experiment checkpoint requires a run config")
        object.__setattr__(
            self, "response_validations", tuple(self.response_validations)
        )
        population_ids = set(
            (*self.psro.population.identifiers(Role.MARSHAL),
             *self.psro.population.identifiers(Role.FUGITIVE))
        )
        if any(
            not isinstance(report, ResponseValidationReport)
            for report in self.response_validations
        ):
            raise ValueError("checkpoint response validations are malformed")
        admitted_candidates = {
            report.candidate
            for report in self.response_validations
            if report.admitted
        }
        previous_generation = 0
        roles_by_generation: dict[int, set[Role]] = {}
        for report in self.response_validations:
            if not 1 <= report.generation <= self.psro.generation:
                raise ValueError("response validation has the wrong generation")
            if report.generation < previous_generation:
                raise ValueError("response validations must be chronological")
            previous_generation = report.generation
            roles = roles_by_generation.setdefault(report.generation, set())
            if report.role in roles:
                raise ValueError("a generation has duplicate response validation")
            roles.add(report.role)
            if report.admitted and report.candidate not in population_ids:
                raise ValueError("an admitted response is absent from the population")
            if (
                not report.admitted
                and report.candidate not in admitted_candidates
                and report.candidate in population_ids
            ):
                raise ValueError("a rejected response entered the population")
        if self.final_holdout is not None:
            if not isinstance(self.final_holdout, FinalHoldoutReport):
                raise ValueError("checkpoint final holdout is malformed")
            if any(
                pair.marshal not in population_ids or pair.fugitive not in population_ids
                for pair in self.final_holdout.sampled_pairs
            ):
                raise ValueError("final holdout references a policy outside population")

    @property
    def payoff_config(self) -> RegisteredGamePayoffConfig:
        return self.run_config.payoff_config

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "schema": PSRO_EXPERIMENT_CHECKPOINT_SCHEMA,
            "version": PSRO_EXPERIMENT_CHECKPOINT_VERSION,
            "psro": self.psro.to_dict(),
            "run_config": self.run_config.to_dict(),
            "response_validations": [
                report.to_dict() for report in self.response_validations
            ],
            "final_holdout": (
                None if self.final_holdout is None else self.final_holdout.to_dict()
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
    def from_dict(cls, data: Mapping[str, object]) -> "PSROExperimentCheckpoint":
        if data.get("schema") != PSRO_EXPERIMENT_CHECKPOINT_SCHEMA:
            raise ValueError("unsupported PSRO experiment checkpoint schema")
        if data.get("version") != PSRO_EXPERIMENT_CHECKPOINT_VERSION:
            raise ValueError("unsupported PSRO experiment checkpoint version")
        final_holdout = data.get("final_holdout")
        return cls(
            PSROCheckpoint.from_dict(
                _require_mapping(data.get("psro"), "PSRO checkpoint")
            ),
            PSROExperimentRunConfig.from_dict(
                _require_mapping(data.get("run_config"), "run config")
            ),
            tuple(
                ResponseValidationReport.from_dict(
                    _require_mapping(item, "response validation")
                )
                for item in _require_sequence(
                    data.get("response_validations"), "response validations"
                )
            ),
            (
                None
                if final_holdout is None
                else FinalHoldoutReport.from_dict(
                    _require_mapping(final_holdout, "final holdout")
                )
            ),
        )

    @classmethod
    def from_json(cls, payload: str) -> "PSROExperimentCheckpoint":
        try:
            data = json.loads(payload)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("invalid PSRO experiment checkpoint JSON") from exc
        return cls.from_dict(_require_mapping(data, "experiment checkpoint"))


@dataclass(frozen=True, slots=True)
class PSROExperimentRunResult:
    checkpoints: tuple[PSROExperimentCheckpoint, ...]
    stop_reason: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "checkpoints", tuple(self.checkpoints))
        if not self.checkpoints:
            raise ValueError("experiment run requires its initial checkpoint")

    @property
    def final_checkpoint(self) -> PSROExperimentCheckpoint:
        return self.checkpoints[-1]

    @property
    def final_holdout(self) -> FinalHoldoutReport | None:
        return self.final_checkpoint.final_holdout

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "schema": PSRO_EXPERIMENT_RUN_SCHEMA,
            "version": PSRO_EXPERIMENT_RUN_VERSION,
            "checkpoints": [checkpoint.to_dict() for checkpoint in self.checkpoints],
            "stop_reason": self.stop_reason,
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
    def from_dict(cls, data: Mapping[str, object]) -> "PSROExperimentRunResult":
        if data.get("schema") != PSRO_EXPERIMENT_RUN_SCHEMA:
            raise ValueError("unsupported PSRO experiment run schema")
        if data.get("version") != PSRO_EXPERIMENT_RUN_VERSION:
            raise ValueError("unsupported PSRO experiment run version")
        stop_reason = data.get("stop_reason")
        if not isinstance(stop_reason, str) or not stop_reason:
            raise ValueError("PSRO experiment run requires a stop reason")
        return cls(
            tuple(
                PSROExperimentCheckpoint.from_dict(
                    _require_mapping(item, "experiment checkpoint")
                )
                for item in _require_sequence(
                    data.get("checkpoints"), "experiment checkpoints"
                )
            ),
            stop_reason,
        )

    @classmethod
    def from_json(cls, payload: str) -> "PSROExperimentRunResult":
        try:
            data = json.loads(payload)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("invalid PSRO experiment run JSON") from exc
        return cls.from_dict(_require_mapping(data, "experiment run"))


def initialize_registered_psro(
    population: PolicyPopulation,
    *,
    run_config: PSROExperimentRunConfig,
    payoff_matrix: EmpiricalPayoffMatrix | None = None,
    evaluator: PayoffEvaluator | None = None,
) -> PSROExperimentCheckpoint:
    concrete_evaluator = evaluator or RegisteredGamePayoffEvaluator(
        run_config.payoff_config
    )
    if isinstance(concrete_evaluator, RegisteredGamePayoffEvaluator):
        concrete_evaluator.bind_population(population)
    checkpoint = initialize_psro(
        population,
        evaluator=concrete_evaluator,
        payoff_matrix=payoff_matrix,
        meta_solver=MultiplicativeWeightsMetaSolver(
            run_config.meta_solver_config
        ),
    )
    return PSROExperimentCheckpoint(checkpoint, run_config)


class _FixedResponseOracle:
    """Return a response already proposed and independently validated."""

    def __init__(self, response: PolicySpec | None) -> None:
        self.response = response

    def propose_response(self, request: ResponseOracleRequest) -> PolicySpec | None:
        del request
        return self.response


def _restricted_best_response(
    role: Role,
    population: PolicyPopulation,
    matrix: EmpiricalPayoffMatrix,
    opponent_mixture: PolicyMixture,
) -> PolicySpec:
    """Return the deterministic best incumbent pure response."""

    policies = population.policies(role)
    if role is Role.MARSHAL:
        value = lambda policy: math.fsum(  # noqa: E731 - formula reads directly
            entry.probability
            * matrix.marshal_payoff(policy.identifier, entry.identifier)
            for entry in opponent_mixture.entries
        )
        return max(policies, key=value)
    value = lambda policy: math.fsum(  # noqa: E731 - formula reads directly
        entry.probability
        * matrix.marshal_payoff(entry.identifier, policy.identifier)
        for entry in opponent_mixture.entries
    )
    return min(policies, key=value)


def _validate_candidate(
    evaluator: PayoffEvaluator,
    *,
    checkpoint: PSROExperimentCheckpoint,
    role: Role,
    generation: int,
    candidate: PolicySpec | None,
) -> tuple[PolicySpec | None, ResponseValidationReport | None]:
    if candidate is None:
        return None, None
    current_ids = set(checkpoint.psro.population.identifiers(role))
    if candidate.identifier in current_ids:
        return candidate, None
    validate = getattr(evaluator, "validate_response", None)
    if not callable(validate):
        raise ValueError(
            "registered PSRO response admission requires a validation evaluator"
        )
    opponent_mixture = (
        checkpoint.psro.meta_solution.fugitive_mixture
        if role is Role.MARSHAL
        else checkpoint.psro.meta_solution.marshal_mixture
    )
    incumbent = _restricted_best_response(
        role,
        checkpoint.psro.population,
        checkpoint.psro.payoff_matrix,
        opponent_mixture,
    )
    report = validate(
        role=role,
        generation=generation,
        candidate=candidate,
        incumbent=incumbent,
        opponent_mixture=opponent_mixture,
        population=checkpoint.psro.population,
    )
    if not isinstance(report, ResponseValidationReport):
        raise ValueError("response validator returned a malformed report")
    return (candidate if report.admitted else None), report


def run_registered_psro_iteration(
    checkpoint: PSROExperimentCheckpoint,
    *,
    evaluator: PayoffEvaluator | None = None,
) -> PSROExperimentCheckpoint:
    concrete_evaluator = evaluator or RegisteredGamePayoffEvaluator(
        checkpoint.payoff_config
    )
    if isinstance(concrete_evaluator, RegisteredGamePayoffEvaluator):
        concrete_evaluator.bind_population(checkpoint.psro.population)
    generation = checkpoint.psro.generation + 1
    marshal_request = ResponseOracleRequest(
        Role.MARSHAL,
        generation,
        checkpoint.psro.population,
        checkpoint.psro.payoff_matrix,
        checkpoint.psro.meta_solution,
    )
    fugitive_request = ResponseOracleRequest(
        Role.FUGITIVE,
        generation,
        checkpoint.psro.population,
        checkpoint.psro.payoff_matrix,
        checkpoint.psro.meta_solution,
    )
    marshal_candidate = MixtureConditionedResponseOracle(
        checkpoint.run_config.marshal_response_template
    ).propose_response(marshal_request)
    fugitive_candidate = MixtureConditionedResponseOracle(
        checkpoint.run_config.fugitive_response_template
    ).propose_response(fugitive_request)
    marshal_response, marshal_report = _validate_candidate(
        concrete_evaluator,
        checkpoint=checkpoint,
        role=Role.MARSHAL,
        generation=generation,
        candidate=marshal_candidate,
    )
    fugitive_response, fugitive_report = _validate_candidate(
        concrete_evaluator,
        checkpoint=checkpoint,
        role=Role.FUGITIVE,
        generation=generation,
        candidate=fugitive_candidate,
    )
    updated = run_psro_iteration(
        checkpoint.psro,
        marshal_oracle=_FixedResponseOracle(marshal_response),
        fugitive_oracle=_FixedResponseOracle(fugitive_response),
        evaluator=concrete_evaluator,
        meta_solver=MultiplicativeWeightsMetaSolver(
            checkpoint.run_config.meta_solver_config
        ),
    )
    reports = tuple(
        report
        for report in (marshal_report, fugitive_report)
        if report is not None
    )
    return PSROExperimentCheckpoint(
        updated,
        checkpoint.run_config,
        (*checkpoint.response_validations, *reports),
    )


def finalize_registered_psro(
    checkpoint: PSROExperimentCheckpoint,
    *,
    evaluator: PayoffEvaluator | None = None,
) -> PSROExperimentCheckpoint:
    """Attach a fresh final-profile report without changing selection state."""

    concrete = evaluator or RegisteredGamePayoffEvaluator(checkpoint.payoff_config)
    if not checkpoint.payoff_config.final_holdout_seeds:
        report = None
    else:
        evaluate_holdout = getattr(concrete, "evaluate_final_holdout", None)
        if not callable(evaluate_holdout):
            raise ValueError(
                "registered PSRO finalization requires a holdout evaluator"
            )
        report = evaluate_holdout(
            checkpoint.psro.population,
            checkpoint.psro.meta_solution.marshal_mixture,
            checkpoint.psro.meta_solution.fugitive_mixture,
        )
        if report is not None and not isinstance(report, FinalHoldoutReport):
            raise ValueError("final holdout evaluator returned a malformed report")
    return PSROExperimentCheckpoint(
        checkpoint.psro,
        checkpoint.run_config,
        checkpoint.response_validations,
        report,
    )


def run_registered_psro(
    initial_checkpoint: PSROExperimentCheckpoint,
    *,
    generations: int,
    evaluator: PayoffEvaluator | None = None,
    stop_when_no_new_policies: bool = True,
) -> PSROExperimentRunResult:
    if isinstance(generations, bool) or not isinstance(generations, int):
        raise ValueError("PSRO generations must be an integer")
    if generations < 0:
        raise ValueError("PSRO generations must be non-negative")
    current = initial_checkpoint
    checkpoints = [current]
    stop_reason = "generation_limit"
    for _ in range(generations):
        current = run_registered_psro_iteration(current, evaluator=evaluator)
        checkpoints.append(current)
        summary = current.psro.last_generation
        if (
            stop_when_no_new_policies
            and summary is not None
            and not summary.added_policies
        ):
            stop_reason = "no_new_policies"
            break
    checkpoints[-1] = finalize_registered_psro(current, evaluator=evaluator)
    return PSROExperimentRunResult(tuple(checkpoints), stop_reason)


__all__ = [
    "PSRO_EXPERIMENT_CHECKPOINT_SCHEMA",
    "PSRO_EXPERIMENT_CHECKPOINT_VERSION",
    "PSRO_EXPERIMENT_RUN_SCHEMA",
    "PSRO_EXPERIMENT_RUN_VERSION",
    "PSROExperimentCheckpoint",
    "PSROExperimentRunConfig",
    "PSROExperimentRunResult",
    "finalize_registered_psro",
    "initialize_registered_psro",
    "run_registered_psro",
    "run_registered_psro_iteration",
]
