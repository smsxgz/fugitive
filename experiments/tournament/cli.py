"""Command-line interface for resumable tournaments."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from fugitive.agents.registry import FUGITIVE_AGENT_REGISTRY, MARSHAL_AGENT_REGISTRY
from fugitive.shared.reproducibility import AGENT_PROFILES, parse_seed

from .model import GameRecord, TournamentConfig
from .runner import run_tournament


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a sequential, resumable Fugitive agent tournament",
    )
    parser.add_argument(
        "--fugitive",
        action="append",
        choices=tuple(sorted(FUGITIVE_AGENT_REGISTRY)),
        help="Fugitive registry ID; repeat for multiple agents (default: all)",
    )
    parser.add_argument(
        "--marshal",
        action="append",
        choices=tuple(sorted(MARSHAL_AGENT_REGISTRY)),
        help="Marshal registry ID; repeat for multiple agents (default: all)",
    )
    parser.add_argument("--games", type=int, default=1, help="games per matchup")
    parser.add_argument(
        "--root-seed", default="0", help="unsigned 64-bit root seed (decimal)"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--fugitive-profile", choices=AGENT_PROFILES, default="default"
    )
    parser.add_argument(
        "--marshal-profile", choices=AGENT_PROFILES, default="default"
    )
    parser.add_argument(
        "--max-decisions",
        type=int,
        default=None,
        help="experiment watchdog; default runs to a rules-defined winner",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume or extend a compatible output directory",
    )
    parser.add_argument(
        "--no-validate-invariants",
        action="store_true",
        help="skip debug invariant checks after decisions",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = TournamentConfig(
            fugitive_agents=tuple(
                args.fugitive or sorted(FUGITIVE_AGENT_REGISTRY)
            ),
            marshal_agents=tuple(args.marshal or sorted(MARSHAL_AGENT_REGISTRY)),
            games_per_matchup=args.games,
            root_seed=parse_seed(args.root_seed, "root"),
            output_directory=args.output,
            fugitive_profile=args.fugitive_profile,
            marshal_profile=args.marshal_profile,
            max_decisions=args.max_decisions,
            validate_invariants=not args.no_validate_invariants,
        )
        report = run_tournament(config, resume=args.resume, progress=_print_progress)
    except (OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(f"completed {len(report.records)} recorded games")
    print(f"summary: {config.output_directory / 'summary.md'}")
    return 0


def _print_progress(record: GameRecord) -> None:
    winner = record.winner.value if record.winner is not None else "-"
    print(
        f"[{record.key}] {record.status.value} winner={winner} "
        f"decisions={record.decision_count} seconds={record.wall_seconds:.2f}",
        flush=True,
    )


__all__ = ["build_parser", "main"]
