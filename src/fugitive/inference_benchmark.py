"""Command-line adapter for fixed-Observation Marshal inference benchmarks.

The benchmark replays recorded experiment manifests, gives each belief backend
only the Marshal Observation available before a decision, and scores the
returned belief against offline truth afterwards.  Every backend is rebuilt
for every manifest.  This is important for BIR-1, whose backend consumes the
public event sequence incrementally and must not carry state between games.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence, cast

from .agents.marshal_candidates import (
    DEFAULT_CATALOGUE_MAX_GUESS_SIZE,
    DEFAULT_CATALOGUE_MAX_JOINT_CANDIDATES,
    build_marshal_guess_catalogue,
)
from .agents.registry import MARSHAL_AGENT_REGISTRY
from .experiment import ReplayManifest
from .observation_benchmark import (
    BackendBenchmarkResult,
    joint_calibration_bins,
    evaluate_backend_on_manifest,
)
from .reproducibility import JSONValue, derive_seed, normalize_parameters


INFERENCE_BENCHMARK_SCHEMA_VERSION = 3
INFERENCE_BENCHMARK_VERSION = "fixed-observation-marshal-inference-v3"
_COMPARISON_SEED_PROTOCOL = "paired-backend-comparison-seed-v2"
_GAME_FIRST_AGGREGATION_ID = "equal-source-game-then-replicate-v1"

_AGENT_ALIASES = {
    "bir-1": "belief-informed-random",
    "bir1": "belief-informed-random",
    "bir-2u": "unweighted-constructive-belief-informed-random",
    "bir2u": "unweighted-constructive-belief-informed-random",
    "bir-2s": "constructive-belief-informed-random",
    "bir2s": "constructive-belief-informed-random",
    "bir-2e": "exact-sprint-belief-informed-random",
    "bir2e": "exact-sprint-belief-informed-random",
    "bir-3": "mcmc-belief-informed-random",
    "bir3": "mcmc-belief-informed-random",
}
SUPPORTED_BIR_AGENTS = (
    "belief-informed-random",
    "unweighted-constructive-belief-informed-random",
    "constructive-belief-informed-random",
    "exact-sprint-belief-informed-random",
    "mcmc-belief-informed-random",
)


@dataclass(frozen=True, slots=True)
class MarshalBackendRequest:
    """One registry Marshal agent whose public ``belief_backend`` is scored."""

    name: str
    parameters: Mapping[str, JSONValue]

    @classmethod
    def create(
        cls,
        name: str,
        parameters: Mapping[str, object] | None = None,
    ) -> "MarshalBackendRequest":
        return cls(_canonical_agent_name(name), normalize_parameters(parameters or {}))


def run_inference_benchmark(
    manifest_paths: Sequence[str | Path],
    agent_requests: Sequence[MarshalBackendRequest],
    *,
    seed: int = 0,
    replicates: int = 1,
    profile: str = "default",
    top_k: Iterable[int] = (1, 5, 10),
    calibration_bins: int = 10,
    catalogue_max_guess_size: int = DEFAULT_CATALOGUE_MAX_GUESS_SIZE,
    catalogue_max_joint_candidates: int = (
        DEFAULT_CATALOGUE_MAX_JOINT_CANDIDATES
    ),
    validate_invariants: bool = True,
) -> dict[str, JSONValue]:
    """Evaluate requested backends on one or more immutable replay manifests."""

    if not manifest_paths:
        raise ValueError("at least one experiment manifest is required")
    if not agent_requests:
        raise ValueError("at least one Marshal belief agent is required")
    if (
        isinstance(replicates, bool)
        or not isinstance(replicates, int)
        or replicates < 1
    ):
        raise ValueError("replicates must be a positive integer")
    requested_top_k = _normalize_top_k(top_k)
    if (
        isinstance(calibration_bins, bool)
        or not isinstance(calibration_bins, int)
        or calibration_bins < 1
    ):
        raise ValueError("calibration_bins must be a positive integer")

    loaded_manifests = tuple(_load_manifest(path) for path in manifest_paths)

    def candidate_sets(observation):
        return build_marshal_guess_catalogue(
            observation,
            max_guess_size=catalogue_max_guess_size,
            max_joint_candidates=catalogue_max_joint_candidates,
        ).guesses

    prepared_requests = tuple(
        (
            request,
            _registration(request.name),
            _request_parameter_hash(request),
        )
        for request in agent_requests
    )

    runs: list[dict[str, JSONValue]] = []
    for manifest_index, (manifest_path, manifest, manifest_hash) in enumerate(
        loaded_manifests
    ):
        for replicate in range(replicates):
            comparison_seed = derive_seed(
                seed,
                _comparison_seed_domain(manifest_hash, replicate),
            )
            for request, registration, parameter_hash in prepared_requests:
                built = registration.build(
                    comparison_seed,
                    profile=profile,
                    overrides=request.parameters,
                )
                backend = getattr(built.agent, "belief_backend", None)
                if backend is None or not callable(getattr(backend, "infer", None)):
                    raise TypeError(
                        f"Marshal agent {request.name!r} has no belief_backend"
                    )

                # evaluate_backend_on_manifest calls infer(observation) before
                # its offline truth is read for scoring. The shared catalogue
                # also receives Observation only.
                evaluated = evaluate_backend_on_manifest(
                    manifest,
                    backend,
                    candidate_set_provider=candidate_sets,
                    include_recorded_guess=False,
                    top_k=requested_top_k,
                    validate_invariants=validate_invariants,
                )
                if not evaluated.decisions:
                    raise ValueError(
                        f"manifest {manifest_path!r} has no Marshal inference decisions"
                    )
                runs.append(
                    _run_payload(
                        manifest_index=manifest_index,
                        manifest_path=manifest_path,
                        manifest=manifest,
                        manifest_hash=manifest_hash,
                        request_name=request.name,
                        request_parameter_hash=parameter_hash,
                        replicate=replicate,
                        comparison_seed=comparison_seed,
                        agent_spec=built.spec.to_dict(),
                        result=evaluated,
                        top_k=requested_top_k,
                        calibration_bin_count=calibration_bins,
                    )
                )

    return {
        "schema_version": INFERENCE_BENCHMARK_SCHEMA_VERSION,
        "benchmark_version": INFERENCE_BENCHMARK_VERSION,
        "settings": {
            "seed": str(seed),
            "replicates": replicates,
            "seed_protocol": {
                "id": _COMPARISON_SEED_PROTOCOL,
                "pairing_key": [
                    "root_seed",
                    "manifest_sha256",
                    "replicate",
                ],
            },
            "profile": profile,
            "top_k": list(requested_top_k),
            "calibration_bins": calibration_bins,
            "validate_invariants": validate_invariants,
            "candidate_catalogue": {
                "max_guess_size": catalogue_max_guess_size,
                "max_joint_candidates": catalogue_max_joint_candidates,
                "include_recorded_guess": False,
            },
        },
        "runs": runs,
        "game_first": _game_first_payload(runs),
    }


def _run_payload(
    *,
    manifest_index: int,
    manifest_path: str,
    manifest: ReplayManifest,
    manifest_hash: str,
    request_name: str,
    request_parameter_hash: str,
    replicate: int,
    comparison_seed: int,
    agent_spec: dict[str, JSONValue],
    result: BackendBenchmarkResult,
    top_k: tuple[int, ...],
    calibration_bin_count: int,
) -> dict[str, JSONValue]:
    decisions = result.decisions
    decision_count = len(decisions)
    hidden_brier = [item.hidden_cards.brier_score for item in decisions]
    hidden_log = [item.hidden_cards.log_loss for item in decisions]
    route_mass = [item.route.true_mass for item in decisions]
    route_rank = [item.route.rank for item in decisions]
    route_support = [item.route.support_size for item in decisions]

    top_k_rows: list[dict[str, JSONValue]] = []
    for k in top_k:
        covered = sum(
            dict(item.route.top_k_coverage)[k]
            for item in decisions
        )
        top_k_rows.append(
            {
                "k": k,
                "covered": covered,
                "rate": covered / decision_count if decision_count else None,
            }
        )

    calibration = result.joint_calibration
    calibration_rows = [
        {
            "lower_bound": item.lower_bound,
            "upper_bound": item.upper_bound,
            "count": item.count,
            "mean_prediction": item.mean_prediction,
            "observed_frequency": item.observed_frequency,
        }
        for item in joint_calibration_bins(
            calibration,
            bin_count=calibration_bin_count,
        )
    ]

    decision_rows: list[dict[str, JSONValue]] = []
    for item in decisions:
        decision_rows.append(
            {
                "decision": item.point.decision,
                "round": item.point.round_number,
                "phase": item.point.phase.value,
                "hidden_cards": {
                    "brier_score": item.hidden_cards.brier_score,
                    "log_loss": item.hidden_cards.log_loss,
                    "card_count": item.hidden_cards.card_count,
                    "positive_count": item.hidden_cards.positive_count,
                },
                "route": {
                    "true_mass": item.route.true_mass,
                    "rank": item.route.rank,
                    "support_size": item.route.support_size,
                    "top_k_coverage": [
                        {"k": k, "covered": covered}
                        for k, covered in item.route.top_k_coverage
                    ],
                },
                "joint_forecast_count": len(item.joint_calibration),
            }
        )

    return {
        "manifest": {
            "input_index": manifest_index,
            "path": manifest_path,
            "sha256": manifest_hash,
            "final_state_sha256": manifest.final_state_sha256,
            "recorded_decision_count": manifest.decision_count,
        },
        "request": {
            "name": request_name,
            "parameter_sha256": request_parameter_hash,
        },
        "replicate": replicate,
        "agent": agent_spec,
        "backend_id": result.backend_id,
        "comparison_seed": str(comparison_seed),
        "decision_count": decision_count,
        "summary": {
            "hidden_cards": {
                "mean_brier_score": _mean(hidden_brier),
                "mean_log_loss": _mean(hidden_log),
                "evaluated_cards": sum(
                    item.hidden_cards.card_count for item in decisions
                ),
                "positive_labels": sum(
                    item.hidden_cards.positive_count for item in decisions
                ),
            },
            "route": {
                "mean_true_mass": _mean(route_mass),
                "mean_rank": _mean(route_rank),
                "mean_support_size": _mean(route_support),
                "top_k_coverage": top_k_rows,
            },
            "joint_calibration": {
                "forecast_count": len(calibration),
                "bins": calibration_rows,
            },
        },
        "decisions": decision_rows,
    }


@dataclass(frozen=True, slots=True)
class _RunMetrics:
    decision_count: float
    mean_brier_score: float | None
    mean_log_loss: float | None
    evaluated_cards: float
    positive_labels: float
    mean_true_mass: float | None
    mean_rank: float | None
    mean_support_size: float | None
    top_k_rates: tuple[tuple[int, float | None], ...]
    forecast_count: float


def _game_first_payload(
    runs: Sequence[Mapping[str, JSONValue]],
) -> dict[str, JSONValue]:
    """Aggregate each source game before giving games equal outer weight.

    A repeated input path is still preserved in ``runs`` for auditability, but
    identical source games and replicate indices contribute only once here.
    This also exposes replicate-level game means, so paired backend comparisons
    do not have to reconstruct the pairing from pooled decision rows.
    """

    grouped: dict[
        tuple[str, str, str],
        dict[str, list[Mapping[str, JSONValue]]],
    ] = {}
    for run in runs:
        request = cast(Mapping[str, JSONValue], run["request"])
        key = (
            cast(str, request["name"]),
            cast(str, request["parameter_sha256"]),
            cast(str, run["backend_id"]),
        )
        manifest = cast(Mapping[str, JSONValue], run["manifest"])
        game_id = cast(str, manifest["sha256"])
        grouped.setdefault(key, {}).setdefault(game_id, []).append(run)

    group_rows: list[dict[str, JSONValue]] = []
    for request_key in sorted(grouped):
        request_name, parameter_hash, backend_id = request_key
        games = grouped[request_key]
        game_rows: list[dict[str, JSONValue]] = []
        game_metrics: list[_RunMetrics] = []
        replicate_metrics: dict[int, list[_RunMetrics]] = {}

        for game_id in sorted(games):
            game_runs = games[game_id]
            by_replicate: dict[int, Mapping[str, JSONValue]] = {}
            comparison_seeds: dict[int, str] = {}
            input_indices: set[int] = set()
            manifest_paths: set[str] = set()
            final_state_hashes: set[str] = set()

            for run in game_runs:
                manifest = cast(Mapping[str, JSONValue], run["manifest"])
                replicate = cast(int, run["replicate"])
                comparison_seed = cast(str, run["comparison_seed"])
                by_replicate.setdefault(replicate, run)
                comparison_seeds.setdefault(replicate, comparison_seed)
                input_indices.add(cast(int, manifest["input_index"]))
                manifest_paths.add(cast(str, manifest["path"]))
                final_state_hashes.add(
                    cast(str, manifest["final_state_sha256"])
                )

            ordered_replicates = sorted(by_replicate)
            per_replicate = [
                _metrics_from_run(by_replicate[index])
                for index in ordered_replicates
            ]
            averaged_game = _average_metrics(per_replicate)
            game_metrics.append(averaged_game)
            for replicate, metric in zip(ordered_replicates, per_replicate):
                replicate_metrics.setdefault(replicate, []).append(metric)

            game_rows.append(
                {
                    "source_game": {
                        "manifest_sha256": game_id,
                        "final_state_sha256": sorted(final_state_hashes)[0],
                    },
                    "input_manifest_count": len(input_indices),
                    "manifest_paths": sorted(manifest_paths),
                    "replicates": [
                        {
                            "replicate": replicate,
                            "comparison_seed": comparison_seeds[replicate],
                        }
                        for replicate in ordered_replicates
                    ],
                    "summary": _metrics_payload(
                        averaged_game,
                        averaging_unit="unique_inference_replicate",
                        unit_count=len(per_replicate),
                    ),
                }
            )

        replicate_rows = [
            {
                "replicate": replicate,
                "source_game_count": len(metrics),
                "summary": _metrics_payload(
                    _average_metrics(metrics),
                    averaging_unit="source_game",
                    unit_count=len(metrics),
                ),
            }
            for replicate, metrics in sorted(replicate_metrics.items())
        ]
        group_rows.append(
            {
                "request": {
                    "name": request_name,
                    "parameter_sha256": parameter_hash,
                },
                "backend_id": backend_id,
                "source_game_count": len(game_rows),
                "replicate_count": len(replicate_rows),
                "summary": _metrics_payload(
                    _average_metrics(game_metrics),
                    averaging_unit="source_game",
                    unit_count=len(game_metrics),
                ),
                "replicate_summaries": replicate_rows,
                "source_games": game_rows,
            }
        )

    return {
        "aggregation_id": _GAME_FIRST_AGGREGATION_ID,
        "source_game_key": "manifest.sha256",
        "within_game_weighting": "equal_unique_replicates",
        "across_game_weighting": "equal_source_games",
        "calibration_bins": "reported per run; not averaged across games",
        "groups": group_rows,
    }


def _metrics_from_run(run: Mapping[str, JSONValue]) -> _RunMetrics:
    summary = cast(Mapping[str, JSONValue], run["summary"])
    hidden = cast(Mapping[str, JSONValue], summary["hidden_cards"])
    route = cast(Mapping[str, JSONValue], summary["route"])
    calibration = cast(Mapping[str, JSONValue], summary["joint_calibration"])
    top_k_rows = cast(Sequence[Mapping[str, JSONValue]], route["top_k_coverage"])
    return _RunMetrics(
        decision_count=float(cast(int, run["decision_count"])),
        mean_brier_score=_optional_float(hidden["mean_brier_score"]),
        mean_log_loss=_optional_float(hidden["mean_log_loss"]),
        evaluated_cards=float(cast(int, hidden["evaluated_cards"])),
        positive_labels=float(cast(int, hidden["positive_labels"])),
        mean_true_mass=_optional_float(route["mean_true_mass"]),
        mean_rank=_optional_float(route["mean_rank"]),
        mean_support_size=_optional_float(route["mean_support_size"]),
        top_k_rates=tuple(
            (
                cast(int, row["k"]),
                _optional_float(row["rate"]),
            )
            for row in top_k_rows
        ),
        forecast_count=float(cast(int, calibration["forecast_count"])),
    )


def _average_metrics(metrics: Sequence[_RunMetrics]) -> _RunMetrics:
    if not metrics:
        raise ValueError("cannot aggregate an empty metric sequence")
    top_k = tuple(k for k, _ in metrics[0].top_k_rates)
    if any(tuple(k for k, _ in item.top_k_rates) != top_k for item in metrics):
        raise ValueError("benchmark runs use inconsistent top-k settings")

    return _RunMetrics(
        decision_count=_required_mean(item.decision_count for item in metrics),
        mean_brier_score=_optional_mean(
            item.mean_brier_score for item in metrics
        ),
        mean_log_loss=_optional_mean(item.mean_log_loss for item in metrics),
        evaluated_cards=_required_mean(item.evaluated_cards for item in metrics),
        positive_labels=_required_mean(item.positive_labels for item in metrics),
        mean_true_mass=_optional_mean(item.mean_true_mass for item in metrics),
        mean_rank=_optional_mean(item.mean_rank for item in metrics),
        mean_support_size=_optional_mean(
            item.mean_support_size for item in metrics
        ),
        top_k_rates=tuple(
            (
                k,
                _optional_mean(
                    dict(item.top_k_rates)[k] for item in metrics
                ),
            )
            for k in top_k
        ),
        forecast_count=_required_mean(item.forecast_count for item in metrics),
    )


def _metrics_payload(
    metrics: _RunMetrics,
    *,
    averaging_unit: str,
    unit_count: int,
) -> dict[str, JSONValue]:
    return {
        "averaging_unit": averaging_unit,
        "unit_count": unit_count,
        "mean_decision_count": metrics.decision_count,
        "hidden_cards": {
            "mean_brier_score": metrics.mean_brier_score,
            "mean_log_loss": metrics.mean_log_loss,
            "mean_evaluated_cards": metrics.evaluated_cards,
            "mean_positive_labels": metrics.positive_labels,
        },
        "route": {
            "mean_true_mass": metrics.mean_true_mass,
            "mean_rank": metrics.mean_rank,
            "mean_support_size": metrics.mean_support_size,
            "top_k_coverage": [
                {"k": k, "mean_rate": rate}
                for k, rate in metrics.top_k_rates
            ],
        },
        "joint_calibration": {
            "mean_forecast_count": metrics.forecast_count,
        },
    }


def _optional_float(value: JSONValue) -> float | None:
    return None if value is None else float(cast(float | int, value))


def _required_mean(values: Iterable[float]) -> float:
    materialized = tuple(values)
    if not materialized:
        raise ValueError("cannot average an empty sequence")
    return math.fsum(materialized) / len(materialized)


def _optional_mean(values: Iterable[float | None]) -> float | None:
    present = tuple(value for value in values if value is not None)
    return math.fsum(present) / len(present) if present else None


def _load_manifest(path: str | Path) -> tuple[str, ReplayManifest, str]:
    source = Path(path)
    manifest = ReplayManifest.from_json(source.read_text(encoding="utf-8"))
    canonical = manifest.to_json(indent=None).encode("utf-8")
    return str(path), manifest, hashlib.sha256(canonical).hexdigest()


def _request_parameter_hash(request: MarshalBackendRequest) -> str:
    parameters = json.dumps(
        request.parameters,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(parameters.encode("utf-8")).hexdigest()


def _comparison_seed_domain(manifest_hash: str, replicate: int) -> str:
    return (
        f"{INFERENCE_BENCHMARK_VERSION}:"
        f"{manifest_hash}:replicate={replicate}"
    )


def _registration(name: str):
    canonical = _canonical_agent_name(name)
    if canonical not in SUPPORTED_BIR_AGENTS:
        supported = ", ".join(SUPPORTED_BIR_AGENTS)
        raise ValueError(
            f"agent {name!r} is not a supported BIR backend; choose {supported}"
        )
    try:
        return MARSHAL_AGENT_REGISTRY[canonical]
    except KeyError as exc:
        raise ValueError(f"Marshal agent {canonical!r} is not registered") from exc


def _canonical_agent_name(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("agent name must be a non-empty string")
    stripped = name.strip()
    return _AGENT_ALIASES.get(stripped.lower(), stripped)


def _normalize_top_k(top_k: Iterable[int]) -> tuple[int, ...]:
    values = tuple(top_k)
    if (
        not values
        or len(values) != len(set(values))
        or any(isinstance(k, bool) or not isinstance(k, int) or k < 1 for k in values)
    ):
        raise ValueError("top_k must contain distinct positive integers")
    return tuple(sorted(values))


def _mean(values: Sequence[float | int]) -> float | None:
    return math.fsum(values) / len(values) if values else None


def _json_object(payload: str, label: str) -> dict[str, JSONValue]:
    try:
        decoded = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON: {exc.msg}") from exc
    if not isinstance(decoded, dict):
        raise ValueError(f"{label} must be a JSON object")
    return normalize_parameters(decoded)


def _parse_agent_requests(
    names: Sequence[str],
    shared_parameters: str,
    per_agent_parameters: Sequence[str],
) -> tuple[MarshalBackendRequest, ...]:
    shared = _json_object(shared_parameters, "--parameters")
    specific: dict[str, dict[str, JSONValue]] = {}
    for item in per_agent_parameters:
        if "=" not in item:
            raise ValueError(
                "--agent-parameters must have the form AGENT=JSON_OBJECT"
            )
        raw_name, raw_parameters = item.split("=", 1)
        name = _canonical_agent_name(raw_name)
        if name in specific:
            raise ValueError(f"duplicate --agent-parameters for {raw_name!r}")
        specific[name] = _json_object(
            raw_parameters,
            f"--agent-parameters for {raw_name!r}",
        )

    canonical_names = tuple(_canonical_agent_name(name) for name in names)
    unused = set(specific) - set(canonical_names)
    if unused:
        raise ValueError(
            "--agent-parameters supplied for an unselected agent: "
            + ", ".join(sorted(unused))
        )
    return tuple(
        MarshalBackendRequest.create(
            name,
            {**shared, **specific.get(name, {})},
        )
        for name in canonical_names
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Score Marshal belief backends on fixed experiment Observations."
        )
    )
    parser.add_argument("manifests", nargs="+", help="ExperimentManifest JSON files")
    parser.add_argument(
        "--agent",
        action="append",
        required=True,
        help=(
            "Marshal BIR agent name (repeatable); short aliases BIR-1, "
            "BIR-2U, BIR-2S, BIR-2E, and BIR-3 are accepted"
        ),
    )
    parser.add_argument(
        "--parameters",
        default="{}",
        help="JSON object of parameters shared by every selected agent",
    )
    parser.add_argument(
        "--agent-parameters",
        action="append",
        default=[],
        metavar="AGENT=JSON",
        help="JSON parameter overrides for one selected agent (repeatable)",
    )
    parser.add_argument("--seed", type=int, default=0, help="root inference seed")
    parser.add_argument(
        "--replicates",
        type=int,
        default=1,
        help="independent paired inference replicates per manifest",
    )
    parser.add_argument(
        "--profile",
        choices=("default", "interactive"),
        default="default",
    )
    parser.add_argument("--top-k", nargs="+", type=int, default=[1, 5, 10])
    parser.add_argument("--calibration-bins", type=int, default=10)
    parser.add_argument(
        "--catalogue-max-guess-size",
        type=int,
        default=DEFAULT_CATALOGUE_MAX_GUESS_SIZE,
    )
    parser.add_argument(
        "--catalogue-max-joint-candidates",
        type=int,
        default=DEFAULT_CATALOGUE_MAX_JOINT_CANDIDATES,
    )
    parser.add_argument(
        "--no-validate-invariants",
        action="store_true",
        help="skip replay invariant checks",
    )
    parser.add_argument("--output", help="write JSON to this path instead of stdout")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    try:
        requests = _parse_agent_requests(
            arguments.agent,
            arguments.parameters,
            arguments.agent_parameters,
        )
        payload = run_inference_benchmark(
            arguments.manifests,
            requests,
            seed=arguments.seed,
            replicates=arguments.replicates,
            profile=arguments.profile,
            top_k=arguments.top_k,
            calibration_bins=arguments.calibration_bins,
            catalogue_max_guess_size=arguments.catalogue_max_guess_size,
            catalogue_max_joint_candidates=(
                arguments.catalogue_max_joint_candidates
            ),
            validate_invariants=not arguments.no_validate_invariants,
        )
    except (OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))

    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    )
    if arguments.output:
        Path(arguments.output).write_text(encoded + "\n", encoding="utf-8")
    else:
        print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "INFERENCE_BENCHMARK_SCHEMA_VERSION",
    "INFERENCE_BENCHMARK_VERSION",
    "MarshalBackendRequest",
    "SUPPORTED_BIR_AGENTS",
    "main",
    "run_inference_benchmark",
]
