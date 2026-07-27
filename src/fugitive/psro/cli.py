"""Command-line runner for reproducible full-game PSRO experiments."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Mapping, Sequence

from ..game.model import Role
from .planning_leaf import search_opponent_parameter
from .checkpoint import (
    PSROExperimentCheckpoint,
    PSROExperimentRunConfig,
)
from .runner import (
    finalize_registered_psro,
    initialize_registered_psro,
    run_registered_psro_iteration,
)
from .population import PolicyPopulation
from .solver import MetaSolverConfig
from .payoff import (
    RegisteredGamePayoffConfig,
    RegisteredGamePayoffEvaluator,
)
from .policy_adapter import (
    MixtureResponseTemplate,
    resolve_registered_policy,
)
from ..shared.reproducibility import JSONValue, normalize_parameters


DEFAULT_FUGITIVE_POPULATION = (
    "hierarchical-random",
    "belief-informed-random",
)
DEFAULT_MARSHAL_POPULATION = (
    "support-catalogue-random",
    "route-count-catalogue-random",
)
DEFAULT_FUGITIVE_RESPONSE = "continuation-count"
DEFAULT_MARSHAL_RESPONSE = "unweighted-constructive-belief-informed-random"


def _json_object(payload: str, label: str) -> dict[str, JSONValue]:
    try:
        decoded = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON: {exc.msg}") from exc
    if not isinstance(decoded, dict):
        raise ValueError(f"{label} must be a JSON object")
    return normalize_parameters(decoded)


def _agent_parameter_map(
    entries: Sequence[str],
    *,
    label: str,
) -> dict[str, dict[str, JSONValue]]:
    result: dict[str, dict[str, JSONValue]] = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"{label} must have the form AGENT=JSON_OBJECT")
        name, payload = entry.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"{label} requires a non-empty agent name")
        if name in result:
            raise ValueError(f"duplicate {label} for {name!r}")
        result[name] = _json_object(payload, f"{label} for {name!r}")
    return result


def _resolve_population(
    fugitive_names: Sequence[str],
    marshal_names: Sequence[str],
    *,
    profile: str,
    fugitive_parameters: Mapping[str, Mapping[str, object]],
    marshal_parameters: Mapping[str, Mapping[str, object]],
) -> PolicyPopulation:
    unknown_fugitive = set(fugitive_parameters) - set(fugitive_names)
    unknown_marshal = set(marshal_parameters) - set(marshal_names)
    if unknown_fugitive or unknown_marshal:
        unknown = sorted((*unknown_fugitive, *unknown_marshal))
        raise ValueError(
            "parameters supplied for an unselected initial agent: "
            + ", ".join(unknown)
        )
    fugitive = tuple(
        resolve_registered_policy(
            Role.FUGITIVE,
            name,
            profile=profile,
            overrides=fugitive_parameters.get(name),
        )
        for name in fugitive_names
    )
    marshal = tuple(
        resolve_registered_policy(
            Role.MARSHAL,
            name,
            profile=profile,
            overrides=marshal_parameters.get(name),
        )
        for name in marshal_names
    )
    return PolicyPopulation.create(fugitive=fugitive, marshal=marshal)


def _response_parameters(
    payload: str | None,
    *,
    label: str,
) -> dict[str, JSONValue]:
    return {} if payload is None else _json_object(payload, label)


def _new_checkpoint(
    arguments: argparse.Namespace,
    *,
    ledger: Path,
    progress,
) -> PSROExperimentCheckpoint:
    fugitive_names = tuple(arguments.fugitive or DEFAULT_FUGITIVE_POPULATION)
    marshal_names = tuple(arguments.marshal or DEFAULT_MARSHAL_POPULATION)
    population = _resolve_population(
        fugitive_names,
        marshal_names,
        profile=arguments.initial_profile,
        fugitive_parameters=_agent_parameter_map(
            arguments.fugitive_parameters,
            label="--fugitive-parameters",
        ),
        marshal_parameters=_agent_parameter_map(
            arguments.marshal_parameters,
            label="--marshal-parameters",
        ),
    )
    run_config = PSROExperimentRunConfig(
        payoff_config=RegisteredGamePayoffConfig.derived(
            arguments.seed,
            arguments.games_per_cell,
            response_validation_games=arguments.response_validation_games,
            final_holdout_games=arguments.final_holdout_games,
            response_minimum_gain=arguments.response_minimum_gain,
            confidence_level=arguments.confidence_level,
            bootstrap_replicates=arguments.bootstrap_replicates,
            validate_invariants=not arguments.no_validate_invariants,
            max_decisions=None,
        ),
        meta_solver_config=MetaSolverConfig(
            max_iterations=arguments.solver_iterations,
            tolerance=arguments.solver_tolerance,
            minimum_iterations=arguments.solver_minimum_iterations,
            check_interval=arguments.solver_check_interval,
            learning_rate=arguments.solver_learning_rate,
        ),
        marshal_response_template=MixtureResponseTemplate(
            Role.MARSHAL,
            arguments.marshal_response_agent,
            profile=arguments.response_profile,
            base_parameters=_response_parameters(
                arguments.marshal_response_parameters,
                label="--marshal-response-parameters",
            ),
            opponent_parameter=search_opponent_parameter(
                Role.MARSHAL,
                arguments.marshal_response_agent,
            ),
            identifier_prefix="psro-marshal-candidate",
        ),
        fugitive_response_template=MixtureResponseTemplate(
            Role.FUGITIVE,
            arguments.fugitive_response_agent,
            profile=arguments.response_profile,
            base_parameters=_response_parameters(
                arguments.fugitive_response_parameters,
                label="--fugitive-response-parameters",
            ),
            opponent_parameter=search_opponent_parameter(
                Role.FUGITIVE,
                arguments.fugitive_response_agent,
            ),
            identifier_prefix="psro-fugitive-candidate",
        ),
    )
    evaluator = RegisteredGamePayoffEvaluator(
        run_config.payoff_config,
        ledger=ledger,
        workers=arguments.workers,
        progress=progress,
    )
    return initialize_registered_psro(
        population,
        run_config=run_config,
        evaluator=evaluator,
    )


def _load_checkpoint(path: Path) -> PSROExperimentCheckpoint:
    return PSROExperimentCheckpoint.from_json(path.read_text(encoding="utf-8"))


def _write_checkpoint(path: Path, checkpoint: PSROExperimentCheckpoint) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(checkpoint.to_json() + "\n", encoding="utf-8")
    temporary.replace(path)


def checkpoint_summary(
    checkpoint: PSROExperimentCheckpoint,
    *,
    stop_reason: str,
) -> dict[str, JSONValue]:
    psro = checkpoint.psro
    solution = psro.meta_solution
    return {
        "generation": psro.generation,
        "fugitive_policies": len(psro.population.fugitive_policies),
        "marshal_policies": len(psro.population.marshal_policies),
        "payoff_cells": len(psro.payoff_matrix.entries),
        "restricted_game_value": solution.restricted_game_value,
        "lower_bound": solution.lower_bound,
        "upper_bound": solution.upper_bound,
        "restricted_duality_gap": solution.diagnostics.duality_gap,
        "solver_converged": solution.diagnostics.converged,
        "stop_reason": stop_reason,
        "response_validations": [
            report.to_dict() for report in checkpoint.response_validations
        ],
        "final_holdout": (
            None
            if checkpoint.final_holdout is None
            else checkpoint.final_holdout.to_dict()
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run or resume full-game, two-sided empirical PSRO."
    )
    parser.add_argument("--resume", type=Path, help="resume a checkpoint")
    parser.add_argument("--output", type=Path, help="checkpoint output path")
    parser.add_argument("--generations", type=int, default=1)
    parser.add_argument("--games-per-cell", type=int, default=20)
    parser.add_argument("--response-validation-games", type=int)
    parser.add_argument("--final-holdout-games", type=int)
    parser.add_argument("--response-minimum-gain", type=float, default=0.0)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fugitive", action="append")
    parser.add_argument("--marshal", action="append")
    parser.add_argument(
        "--fugitive-parameters",
        action="append",
        default=[],
        metavar="AGENT=JSON",
    )
    parser.add_argument(
        "--marshal-parameters",
        action="append",
        default=[],
        metavar="AGENT=JSON",
    )
    parser.add_argument(
        "--initial-profile", choices=("default", "interactive"), default="default"
    )
    parser.add_argument(
        "--response-profile", choices=("default", "interactive"), default="default"
    )
    parser.add_argument(
        "--fugitive-response-agent",
        default=DEFAULT_FUGITIVE_RESPONSE,
    )
    parser.add_argument(
        "--marshal-response-agent",
        default=DEFAULT_MARSHAL_RESPONSE,
    )
    parser.add_argument("--fugitive-response-parameters")
    parser.add_argument("--marshal-response-parameters")
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(4, os.cpu_count() or 1)),
    )
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--quiet-progress", action="store_true")
    parser.add_argument("--solver-iterations", type=int, default=100_000)
    parser.add_argument("--solver-tolerance", type=float, default=0.01)
    parser.add_argument("--solver-minimum-iterations", type=int, default=1)
    parser.add_argument("--solver-check-interval", type=int, default=100)
    parser.add_argument("--solver-learning-rate", type=float)
    parser.add_argument("--no-validate-invariants", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    try:
        if arguments.generations < 0:
            raise ValueError("--generations must be non-negative")
        output = arguments.output or arguments.resume or Path("psro-checkpoint.json")
        ledger = arguments.ledger or output.with_name(
            f"{output.stem}-payoffs"
        )

        def report(progress) -> None:
            if arguments.quiet_progress:
                return
            print(
                f"payoff {progress.completed_games}/{progress.total_games} "
                f"reused={progress.reused_games} "
                f"schedule={progress.schedule} "
                f"{progress.pair.marshal.name}/{progress.pair.fugitive.name}",
                file=sys.stderr,
                flush=True,
            )

        if arguments.resume is not None:
            checkpoint = _load_checkpoint(arguments.resume)
            evaluator = RegisteredGamePayoffEvaluator(
                checkpoint.payoff_config,
                ledger=ledger,
                workers=arguments.workers,
                progress=report,
            )
        else:
            checkpoint = _new_checkpoint(
                arguments,
                ledger=ledger,
                progress=report,
            )
            evaluator = RegisteredGamePayoffEvaluator(
                checkpoint.payoff_config,
                ledger=ledger,
                workers=arguments.workers,
                progress=report,
            )
        _write_checkpoint(output, checkpoint)
        stop_reason = "generation_limit"
        for _ in range(arguments.generations):
            checkpoint = run_registered_psro_iteration(
                checkpoint,
                evaluator=evaluator,
            )
            _write_checkpoint(output, checkpoint)
            last = checkpoint.psro.last_generation
            if last is not None and not last.added_policies:
                stop_reason = "no_new_policies"
                break
        checkpoint = finalize_registered_psro(
            checkpoint,
            evaluator=evaluator,
        )
        _write_checkpoint(output, checkpoint)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        parser.error(str(exc))

    summary = checkpoint_summary(checkpoint, stop_reason=stop_reason)
    summary["checkpoint"] = str(output)
    summary["payoff_ledger"] = str(ledger)
    print(json.dumps(summary, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_FUGITIVE_POPULATION",
    "DEFAULT_FUGITIVE_RESPONSE",
    "DEFAULT_MARSHAL_POPULATION",
    "DEFAULT_MARSHAL_RESPONSE",
    "checkpoint_summary",
    "main",
]
