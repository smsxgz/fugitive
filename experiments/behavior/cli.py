"""Command-line interface for fixed-Observation inference benchmarks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from fugitive.agents.marshal.candidates import (
    DEFAULT_CATALOGUE_MAX_GUESS_SIZE,
    DEFAULT_CATALOGUE_MAX_JOINT_CANDIDATES,
)
from fugitive.shared.reproducibility import JSONValue, normalize_parameters

from .benchmark import MarshalBackendRequest, run_inference_benchmark


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score Marshal belief backends on fixed experiment Observations."
    )
    parser.add_argument("manifests", nargs="+", help="ReplayManifest JSON files")
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
        "--profile", choices=("default", "interactive"), default="default"
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
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        requests = parse_agent_requests(
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


def parse_agent_requests(
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
        name = MarshalBackendRequest.create(raw_name).name
        if name in specific:
            raise ValueError(f"duplicate --agent-parameters for {raw_name!r}")
        specific[name] = _json_object(
            raw_parameters,
            f"--agent-parameters for {raw_name!r}",
        )

    canonical_names = tuple(MarshalBackendRequest.create(name).name for name in names)
    unused = set(specific) - set(canonical_names)
    if unused:
        raise ValueError(
            "--agent-parameters supplied for an unselected agent: "
            + ", ".join(sorted(unused))
        )
    return tuple(
        MarshalBackendRequest.create(name, {**shared, **specific.get(name, {})})
        for name in canonical_names
    )


def _json_object(payload: str, label: str) -> dict[str, JSONValue]:
    try:
        decoded = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON: {exc.msg}") from exc
    if not isinstance(decoded, dict):
        raise ValueError(f"{label} must be a JSON object")
    return normalize_parameters(decoded)


__all__ = ["build_parser", "main", "parse_agent_requests"]
