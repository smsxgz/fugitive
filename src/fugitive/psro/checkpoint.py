"""Serializable configuration and checkpoints for registered PSRO runs."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Mapping

from ..game.model import Role
from ..runtime.manifest import RULES_SHA256, RULES_VERSION
from ..shared.reproducibility import JSONValue
from ._json_validation import (
    require_mapping as _require_mapping,
    require_sequence as _require_sequence,
)
from .algorithm import PSROCheckpoint
from .payoff import RegisteredGamePayoffConfig
from .policy_adapter import MixtureResponseTemplate
from .solver import MetaSolverConfig
from .validation import FinalHoldoutReport, ResponseValidationReport


PSRO_EXPERIMENT_CHECKPOINT_SCHEMA = "fugitive.psro-experiment-checkpoint"
PSRO_EXPERIMENT_CHECKPOINT_VERSION = 5
PSRO_EXPERIMENT_RUN_SCHEMA = "fugitive.psro-experiment-run"
PSRO_EXPERIMENT_RUN_VERSION = 1


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

__all__ = [
    "PSRO_EXPERIMENT_CHECKPOINT_SCHEMA",
    "PSRO_EXPERIMENT_CHECKPOINT_VERSION",
    "PSRO_EXPERIMENT_RUN_SCHEMA",
    "PSRO_EXPERIMENT_RUN_VERSION",
    "PSROExperimentCheckpoint",
    "PSROExperimentRunConfig",
    "PSROExperimentRunResult",
]
