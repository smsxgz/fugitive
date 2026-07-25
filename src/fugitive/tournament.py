"""Sequential, resumable tournaments built from the single-game runner.

This module deliberately does not drive the rules engine itself.  Every game
is delegated to :func:`fugitive.experiment.run_registered_experiment`, so a
tournament uses the same AgentSpec, seed, error, manifest, and replay protocol
as an individual experiment.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Callable, Iterable, Mapping, Sequence

from .agents.registry import FUGITIVE_AGENT_REGISTRY, MARSHAL_AGENT_REGISTRY
from .experiment import (
    RULES_SHA256,
    RULES_VERSION,
    ExperimentRun,
    ExperimentStatus,
    run_registered_experiment,
)
from .model import Winner
from .reproducibility import AGENT_PROFILES, derive_seed, parse_seed, validate_seed


TOURNAMENT_SCHEMA_VERSION = 1
TOURNAMENT_GAME_SEED_DOMAIN = "fugitive.tournament.game.v1"
_WILSON_Z_95 = 1.959963984540054


@dataclass(frozen=True, slots=True)
class TournamentConfig:
    """The explicit protocol and requested size of one matchup matrix.

    ``games_per_matchup`` is an extendable run target, not part of
    :attr:`configuration_hash`.  This permits a one-game calibration run to be
    resumed later with a larger target without changing any scientific input.
    """

    fugitive_agents: tuple[str, ...]
    marshal_agents: tuple[str, ...]
    games_per_matchup: int
    root_seed: int
    output_directory: Path
    fugitive_profile: str = "default"
    marshal_profile: str = "default"
    max_decisions: int | None = None
    validate_invariants: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "fugitive_agents", tuple(self.fugitive_agents))
        object.__setattr__(self, "marshal_agents", tuple(self.marshal_agents))
        object.__setattr__(self, "output_directory", Path(self.output_directory))
        _validate_agent_names(
            self.fugitive_agents,
            FUGITIVE_AGENT_REGISTRY,
            "Fugitive",
        )
        _validate_agent_names(
            self.marshal_agents,
            MARSHAL_AGENT_REGISTRY,
            "Marshal",
        )
        if (
            isinstance(self.games_per_matchup, bool)
            or not isinstance(self.games_per_matchup, int)
            or self.games_per_matchup < 1
        ):
            raise ValueError("games_per_matchup must be a positive integer")
        validate_seed(self.root_seed, "tournament root seed")
        for label, profile in (
            ("fugitive_profile", self.fugitive_profile),
            ("marshal_profile", self.marshal_profile),
        ):
            if profile not in AGENT_PROFILES:
                choices = ", ".join(AGENT_PROFILES)
                raise ValueError(f"{label} must be one of {choices}")
        if self.max_decisions is not None and (
            isinstance(self.max_decisions, bool)
            or not isinstance(self.max_decisions, int)
            or self.max_decisions < 0
        ):
            raise ValueError("max_decisions must be a non-negative integer or None")
        if not isinstance(self.validate_invariants, bool):
            raise TypeError("validate_invariants must be a boolean")

    def identity_dict(self) -> dict[str, object]:
        """Return inputs that must remain fixed when extending a tournament."""

        return {
            "schema_version": TOURNAMENT_SCHEMA_VERSION,
            "rules": {"version": RULES_VERSION, "sha256": RULES_SHA256},
            "seed_protocol": {
                "root_seed": str(self.root_seed),
                "game_seed_domain": TOURNAMENT_GAME_SEED_DOMAIN,
            },
            "agents": {
                "fugitive": list(self.fugitive_agents),
                "marshal": list(self.marshal_agents),
            },
            "profiles": {
                "fugitive": self.fugitive_profile,
                "marshal": self.marshal_profile,
            },
            "max_decisions": self.max_decisions,
            "validate_invariants": self.validate_invariants,
        }

    @property
    def configuration_hash(self) -> str:
        payload = _canonical_json(self.identity_dict()).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def to_dict(self) -> dict[str, object]:
        return {
            "configuration_hash": self.configuration_hash,
            "experiment": self.identity_dict(),
            "games_per_matchup": self.games_per_matchup,
            "output_directory": str(self.output_directory),
        }


@dataclass(frozen=True, slots=True)
class GameRecord:
    """Compact tournament index for one separately stored replay manifest."""

    configuration_hash: str
    key: str
    game_index: int
    fugitive_agent: str
    marshal_agent: str
    master_seed: int
    status: ExperimentStatus
    winner: Winner | None
    reason: str | None
    rounds: int | None
    decision_count: int
    wall_seconds: float
    inference_events: int
    inference_diagnostic_failures: int
    manifest_path: str
    diagnostics_path: str

    def to_dict(self) -> dict[str, object]:
        return {
            "configuration_hash": self.configuration_hash,
            "key": self.key,
            "game_index": self.game_index,
            "matchup": {
                "fugitive": self.fugitive_agent,
                "marshal": self.marshal_agent,
            },
            "master_seed": str(self.master_seed),
            "outcome": {
                "status": self.status.value,
                "winner": self.winner.value if self.winner is not None else None,
                "reason": self.reason,
                "rounds": self.rounds,
                "decision_count": self.decision_count,
            },
            "wall_seconds": self.wall_seconds,
            "diagnostics": {
                "events": self.inference_events,
                "failures": self.inference_diagnostic_failures,
                "path": self.diagnostics_path,
            },
            "manifest_path": self.manifest_path,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "GameRecord":
        matchup = _mapping(data.get("matchup"), "matchup")
        outcome = _mapping(data.get("outcome"), "outcome")
        diagnostics = _mapping(data.get("diagnostics"), "diagnostics")
        winner = outcome.get("winner")
        return cls(
            configuration_hash=_string(data.get("configuration_hash"), "configuration_hash"),
            key=_string(data.get("key"), "key"),
            game_index=_integer(data.get("game_index"), "game_index"),
            fugitive_agent=_string(matchup.get("fugitive"), "Fugitive agent"),
            marshal_agent=_string(matchup.get("marshal"), "Marshal agent"),
            master_seed=parse_seed(data.get("master_seed"), "master"),
            status=ExperimentStatus(str(outcome.get("status"))),
            winner=None if winner is None else Winner(str(winner)),
            reason=None if outcome.get("reason") is None else str(outcome.get("reason")),
            rounds=_optional_integer(outcome.get("rounds"), "rounds"),
            decision_count=_integer(outcome.get("decision_count"), "decision_count"),
            wall_seconds=_finite_float(data.get("wall_seconds"), "wall_seconds"),
            inference_events=_integer(diagnostics.get("events"), "diagnostic events"),
            inference_diagnostic_failures=_integer(
                diagnostics.get("failures"), "diagnostic failures"
            ),
            manifest_path=_string(data.get("manifest_path"), "manifest_path"),
            diagnostics_path=_string(diagnostics.get("path"), "diagnostics path"),
        )


@dataclass(frozen=True, slots=True)
class MatchupSummary:
    fugitive_agent: str
    marshal_agent: str
    target_games: int
    recorded_games: int
    completed: int
    truncated: int
    errors: int
    fugitive_wins: int
    marshal_wins: int
    marshal_win_rate: float | None
    marshal_win_rate_95_low: float | None
    marshal_win_rate_95_high: float | None
    average_rounds: float | None
    average_decisions: float | None
    average_wall_seconds: float | None
    total_wall_seconds: float
    inference_events: int
    inference_diagnostic_failures: int
    reasons: Mapping[str, int]

    def to_dict(self) -> dict[str, object]:
        return {
            "fugitive_agent": self.fugitive_agent,
            "marshal_agent": self.marshal_agent,
            "target_games": self.target_games,
            "recorded_games": self.recorded_games,
            "completed": self.completed,
            "truncated": self.truncated,
            "errors": self.errors,
            "fugitive_wins": self.fugitive_wins,
            "marshal_wins": self.marshal_wins,
            "marshal_win_rate": self.marshal_win_rate,
            "marshal_win_rate_95": {
                "low": self.marshal_win_rate_95_low,
                "high": self.marshal_win_rate_95_high,
            },
            "average_rounds": self.average_rounds,
            "average_decisions": self.average_decisions,
            "average_wall_seconds": self.average_wall_seconds,
            "total_wall_seconds": self.total_wall_seconds,
            "inference_events": self.inference_events,
            "inference_diagnostic_failures": self.inference_diagnostic_failures,
            "reasons": dict(sorted(self.reasons.items())),
        }


@dataclass(frozen=True, slots=True)
class TournamentReport:
    configuration_hash: str
    records: tuple[GameRecord, ...]
    matchups: tuple[MatchupSummary, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": TOURNAMENT_SCHEMA_VERSION,
            "configuration_hash": self.configuration_hash,
            "recorded_games": len(self.records),
            "matchups": [matchup.to_dict() for matchup in self.matchups],
        }


ProgressCallback = Callable[[GameRecord], None]


def run_tournament(
    config: TournamentConfig,
    *,
    resume: bool = False,
    progress: ProgressCallback | None = None,
) -> TournamentReport:
    """Run missing games sequentially and continuously checkpoint results."""

    if not isinstance(config, TournamentConfig):
        raise TypeError("config must be a TournamentConfig")
    output = config.output_directory
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "config.json"
    games_path = output / "games.jsonl"

    _prepare_output(config, config_path, games_path, resume=resume)
    existing = _load_records(games_path, config.configuration_hash)
    _validate_existing_records(config, existing, output)
    records_by_key = {record.key: record for record in existing}

    with games_path.open("a", encoding="utf-8", newline="\n") as stream:
        # Game index is outermost so an interrupted matrix has balanced seeds
        # across cells before it begins the next replicate.
        for game_index in range(config.games_per_matchup):
            master_seed = tournament_game_seed(config.root_seed, game_index)
            for fugitive_name in config.fugitive_agents:
                for marshal_name in config.marshal_agents:
                    key = tournament_game_key(
                        fugitive_name,
                        marshal_name,
                        game_index,
                    )
                    if key in records_by_key:
                        continue
                    started = time.perf_counter()
                    run = run_registered_experiment(
                        master_seed=master_seed,
                        fugitive_name=fugitive_name,
                        marshal_name=marshal_name,
                        fugitive_profile=config.fugitive_profile,
                        marshal_profile=config.marshal_profile,
                        max_decisions=config.max_decisions,
                        validate_invariants=config.validate_invariants,
                    )
                    wall_seconds = time.perf_counter() - started
                    record = _store_run(
                        config,
                        run,
                        game_index=game_index,
                        fugitive_name=fugitive_name,
                        marshal_name=marshal_name,
                        master_seed=master_seed,
                        wall_seconds=wall_seconds,
                    )
                    stream.write(_canonical_json(record.to_dict()) + "\n")
                    stream.flush()
                    records_by_key[key] = record
                    report = summarize_tournament(config, records_by_key.values())
                    _write_summaries(config, report)
                    if progress is not None:
                        progress(record)

    report = summarize_tournament(config, records_by_key.values())
    _write_summaries(config, report)
    return report


def tournament_game_seed(root_seed: int, game_index: int) -> int:
    """Derive the master seed shared by every cell at ``game_index``."""

    validate_seed(root_seed, "tournament root seed")
    if isinstance(game_index, bool) or not isinstance(game_index, int) or game_index < 0:
        raise ValueError("game_index must be a non-negative integer")
    return derive_seed(root_seed, f"{TOURNAMENT_GAME_SEED_DOMAIN}/{game_index}")


def tournament_game_key(
    fugitive_agent: str,
    marshal_agent: str,
    game_index: int,
) -> str:
    return f"{fugitive_agent}__vs__{marshal_agent}__{game_index:06d}"


def summarize_tournament(
    config: TournamentConfig,
    records: Iterable[GameRecord],
) -> TournamentReport:
    ordered = tuple(
        sorted(
            records,
            key=lambda item: (
                item.game_index,
                item.fugitive_agent,
                item.marshal_agent,
            ),
        )
    )
    matchups: list[MatchupSummary] = []
    for fugitive_name in config.fugitive_agents:
        for marshal_name in config.marshal_agents:
            cell = tuple(
                record
                for record in ordered
                if record.fugitive_agent == fugitive_name
                and record.marshal_agent == marshal_name
            )
            completed = tuple(
                record
                for record in cell
                if record.status is ExperimentStatus.COMPLETED
            )
            marshal_wins = sum(
                record.winner is Winner.MARSHAL for record in completed
            )
            fugitive_wins = sum(
                record.winner is Winner.FUGITIVE for record in completed
            )
            low, high = wilson_interval(marshal_wins, len(completed))
            matchups.append(
                MatchupSummary(
                    fugitive_agent=fugitive_name,
                    marshal_agent=marshal_name,
                    target_games=config.games_per_matchup,
                    recorded_games=len(cell),
                    completed=len(completed),
                    truncated=sum(
                        record.status is ExperimentStatus.TRUNCATED
                        for record in cell
                    ),
                    errors=sum(
                        record.status is ExperimentStatus.ERROR for record in cell
                    ),
                    fugitive_wins=fugitive_wins,
                    marshal_wins=marshal_wins,
                    marshal_win_rate=(
                        marshal_wins / len(completed) if completed else None
                    ),
                    marshal_win_rate_95_low=low,
                    marshal_win_rate_95_high=high,
                    average_rounds=_average(
                        record.rounds
                        for record in completed
                        if record.rounds is not None
                    ),
                    average_decisions=_average(
                        record.decision_count for record in completed
                    ),
                    average_wall_seconds=_average(
                        record.wall_seconds for record in cell
                    ),
                    total_wall_seconds=sum(record.wall_seconds for record in cell),
                    inference_events=sum(record.inference_events for record in cell),
                    inference_diagnostic_failures=sum(
                        record.inference_diagnostic_failures for record in cell
                    ),
                    reasons=Counter(
                        record.reason
                        for record in completed
                        if record.reason is not None
                    ),
                )
            )
    return TournamentReport(
        config.configuration_hash,
        ordered,
        tuple(matchups),
    )


def wilson_interval(successes: int, trials: int) -> tuple[float | None, float | None]:
    """Return the two-sided 95% Wilson score interval."""

    if (
        isinstance(successes, bool)
        or isinstance(trials, bool)
        or not isinstance(successes, int)
        or not isinstance(trials, int)
        or trials < 0
        or successes < 0
        or successes > trials
    ):
        raise ValueError("successes and trials must satisfy 0 <= successes <= trials")
    if trials == 0:
        return None, None
    proportion = successes / trials
    z_squared = _WILSON_Z_95 * _WILSON_Z_95
    denominator = 1.0 + z_squared / trials
    center = (proportion + z_squared / (2.0 * trials)) / denominator
    radius = (
        _WILSON_Z_95
        * math.sqrt(
            proportion * (1.0 - proportion) / trials
            + z_squared / (4.0 * trials * trials)
        )
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


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
        "--root-seed",
        default="0",
        help="unsigned 64-bit root seed (decimal)",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--fugitive-profile",
        choices=AGENT_PROFILES,
        default="default",
    )
    parser.add_argument(
        "--marshal-profile",
        choices=AGENT_PROFILES,
        default="default",
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
    args = build_parser().parse_args(argv)
    try:
        root_seed = parse_seed(args.root_seed, "root")
        config = TournamentConfig(
            fugitive_agents=tuple(
                args.fugitive or sorted(FUGITIVE_AGENT_REGISTRY)
            ),
            marshal_agents=tuple(args.marshal or sorted(MARSHAL_AGENT_REGISTRY)),
            games_per_matchup=args.games,
            root_seed=root_seed,
            output_directory=args.output,
            fugitive_profile=args.fugitive_profile,
            marshal_profile=args.marshal_profile,
            max_decisions=args.max_decisions,
            validate_invariants=not args.no_validate_invariants,
        )
        report = run_tournament(
            config,
            resume=args.resume,
            progress=_print_progress,
        )
    except (OSError, TypeError, ValueError) as exc:
        build_parser().error(str(exc))
    print(f"completed {len(report.records)} recorded games")
    print(f"summary: {config.output_directory / 'summary.md'}")
    return 0


def _prepare_output(
    config: TournamentConfig,
    config_path: Path,
    games_path: Path,
    *,
    resume: bool,
) -> None:
    if config_path.exists():
        if not resume:
            raise FileExistsError(
                f"{config_path} already exists; pass --resume to continue"
            )
        raw = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(raw, Mapping):
            raise ValueError("existing config.json must contain an object")
        if raw.get("configuration_hash") != config.configuration_hash:
            raise ValueError("existing tournament configuration is incompatible")
        old_target = _integer(raw.get("games_per_matchup"), "games_per_matchup")
        if config.games_per_matchup < old_target:
            raise ValueError(
                "games_per_matchup may stay unchanged or increase when resuming"
            )
    elif games_path.exists():
        raise ValueError("games.jsonl exists without config.json")
    elif resume:
        # Starting an empty directory with --resume is harmless and convenient
        # for scripts that always request resumability.
        pass
    _write_text_atomic(config_path, _pretty_json(config.to_dict()) + "\n")


def _load_records(path: Path, configuration_hash: str) -> tuple[GameRecord, ...]:
    if not path.exists():
        return ()
    records: list[GameRecord] = []
    keys: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid games.jsonl line {line_number}: {exc}") from exc
        if not isinstance(raw, Mapping):
            raise ValueError(f"games.jsonl line {line_number} is not an object")
        record = GameRecord.from_dict(raw)
        if record.configuration_hash != configuration_hash:
            raise ValueError("games.jsonl contains a different configuration hash")
        if record.key in keys:
            raise ValueError(f"games.jsonl contains duplicate key {record.key}")
        keys.add(record.key)
        records.append(record)
    return tuple(records)


def _validate_existing_records(
    config: TournamentConfig,
    records: Iterable[GameRecord],
    output: Path,
) -> None:
    fugitive_names = set(config.fugitive_agents)
    marshal_names = set(config.marshal_agents)
    for record in records:
        expected_key = tournament_game_key(
            record.fugitive_agent,
            record.marshal_agent,
            record.game_index,
        )
        if record.key != expected_key:
            raise ValueError(f"record key does not match its fields: {record.key}")
        if (
            record.fugitive_agent not in fugitive_names
            or record.marshal_agent not in marshal_names
        ):
            raise ValueError(f"record does not belong to configured matrix: {record.key}")
        expected_seed = tournament_game_seed(config.root_seed, record.game_index)
        if record.master_seed != expected_seed:
            raise ValueError(f"record has the wrong paired seed: {record.key}")
        if not (output / record.manifest_path).is_file():
            raise ValueError(f"record manifest is missing: {record.manifest_path}")
        if not (output / record.diagnostics_path).is_file():
            raise ValueError(f"record diagnostics are missing: {record.diagnostics_path}")


def _store_run(
    config: TournamentConfig,
    run: ExperimentRun,
    *,
    game_index: int,
    fugitive_name: str,
    marshal_name: str,
    master_seed: int,
    wall_seconds: float,
) -> GameRecord:
    cell = f"{fugitive_name}__vs__{marshal_name}"
    manifest_relative = Path("manifests") / cell / f"{game_index:06d}.json"
    diagnostics_relative = Path("diagnostics") / cell / f"{game_index:06d}.json"
    _write_text_atomic(
        config.output_directory / manifest_relative,
        run.manifest.to_json() + "\n",
    )
    diagnostics = {
        "events": [event.to_dict() for event in run.inference_events],
        "failures": [
            failure.to_dict() for failure in run.inference_diagnostic_failures
        ],
    }
    _write_text_atomic(
        config.output_directory / diagnostics_relative,
        _pretty_json(diagnostics) + "\n",
    )
    manifest = run.manifest
    return GameRecord(
        configuration_hash=config.configuration_hash,
        key=tournament_game_key(fugitive_name, marshal_name, game_index),
        game_index=game_index,
        fugitive_agent=fugitive_name,
        marshal_agent=marshal_name,
        master_seed=master_seed,
        status=manifest.status,
        winner=manifest.winner,
        reason=manifest.reason,
        rounds=manifest.rounds,
        decision_count=manifest.decision_count,
        wall_seconds=wall_seconds,
        inference_events=len(run.inference_events),
        inference_diagnostic_failures=len(run.inference_diagnostic_failures),
        manifest_path=manifest_relative.as_posix(),
        diagnostics_path=diagnostics_relative.as_posix(),
    )


def _write_summaries(config: TournamentConfig, report: TournamentReport) -> None:
    _write_text_atomic(
        config.output_directory / "summary.json",
        _pretty_json(report.to_dict()) + "\n",
    )
    _write_csv(config.output_directory / "summary.csv", report.matchups)
    _write_text_atomic(
        config.output_directory / "summary.md",
        _summary_markdown(config, report),
    )


def _write_csv(path: Path, matchups: Iterable[MatchupSummary]) -> None:
    rows = tuple(matchups)
    fieldnames = (
        "fugitive_agent",
        "marshal_agent",
        "target_games",
        "recorded_games",
        "completed",
        "truncated",
        "errors",
        "fugitive_wins",
        "marshal_wins",
        "marshal_win_rate",
        "marshal_win_rate_95_low",
        "marshal_win_rate_95_high",
        "average_rounds",
        "average_decisions",
        "average_wall_seconds",
        "total_wall_seconds",
        "inference_events",
        "inference_diagnostic_failures",
        "reasons_json",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for item in rows:
            row = {
                name: getattr(item, name)
                for name in fieldnames
                if name != "reasons_json"
            }
            row["reasons_json"] = _canonical_json(dict(sorted(item.reasons.items())))
            writer.writerow(row)
    temporary.replace(path)


def _summary_markdown(config: TournamentConfig, report: TournamentReport) -> str:
    lines = [
        "# Fugitive Tournament Summary",
        "",
        f"Configuration: `{report.configuration_hash}`",
        "",
        f"Root seed: `{config.root_seed}`",
        "",
        f"Target games per matchup: {config.games_per_matchup}",
        "",
        "| Fugitive | Marshal | Done | Error | F wins | M wins | M win rate (95% Wilson) | Avg rounds | Avg decisions | Avg seconds |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in report.matchups:
        interval = (
            "-"
            if item.marshal_win_rate is None
            else (
                f"{item.marshal_win_rate:.3f} "
                f"[{item.marshal_win_rate_95_low:.3f}, "
                f"{item.marshal_win_rate_95_high:.3f}]"
            )
        )
        lines.append(
            "| "
            + " | ".join(
                (
                    item.fugitive_agent,
                    item.marshal_agent,
                    f"{item.completed}/{item.target_games}",
                    str(item.errors),
                    str(item.fugitive_wins),
                    str(item.marshal_wins),
                    interval,
                    _format_optional(item.average_rounds, 2),
                    _format_optional(item.average_decisions, 1),
                    _format_optional(item.average_wall_seconds, 2),
                )
            )
            + " |"
        )
    lines.append("")
    lines.append(
        "Win rates use completed games only. Errors and explicit experiment "
        "watchdog truncations are never converted into draws or wins."
    )
    lines.append("")
    return "\n".join(lines)


def _print_progress(record: GameRecord) -> None:
    winner = record.winner.value if record.winner is not None else "-"
    print(
        f"[{record.key}] {record.status.value} winner={winner} "
        f"decisions={record.decision_count} seconds={record.wall_seconds:.2f}",
        flush=True,
    )


def _validate_agent_names(
    names: tuple[str, ...],
    registry: Mapping[str, object],
    role: str,
) -> None:
    if not names:
        raise ValueError(f"at least one {role} agent is required")
    if len(set(names)) != len(names):
        raise ValueError(f"{role} agents must not contain duplicates")
    unknown = [name for name in names if name not in registry]
    if unknown:
        raise ValueError(f"unknown {role} agent: {unknown[0]}")


def _average(values: Iterable[int | float]) -> float | None:
    collected = tuple(values)
    return sum(collected) / len(collected) if collected else None


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(path)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _pretty_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        indent=2,
    )


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return value


def _optional_integer(value: object, label: str) -> int | None:
    return None if value is None else _integer(value, label)


def _finite_float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{label} must be finite and non-negative")
    return result


def _format_optional(value: float | None, digits: int) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "GameRecord",
    "MatchupSummary",
    "TOURNAMENT_GAME_SEED_DOMAIN",
    "TOURNAMENT_SCHEMA_VERSION",
    "TournamentConfig",
    "TournamentReport",
    "build_parser",
    "main",
    "run_tournament",
    "summarize_tournament",
    "tournament_game_key",
    "tournament_game_seed",
    "wilson_interval",
]
