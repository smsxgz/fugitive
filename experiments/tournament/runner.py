"""Sequential, resumable tournament execution and per-game storage."""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Callable, Iterable, Mapping

from fugitive.runtime.manifest import MatchRun
from fugitive.runtime.runner import run_registered_match

from .model import (
    GameRecord,
    TOURNAMENT_DIAGNOSTICS_SCHEMA_VERSION,
    TournamentConfig,
    TournamentReport,
    canonical_json,
    summarize_tournament,
    tournament_game_key,
    tournament_game_seed,
)
from .reporting import pretty_json, write_summaries, write_text_atomic


ProgressCallback = Callable[[GameRecord], None]


def run_tournament(
    config: TournamentConfig,
    *,
    resume: bool = False,
    progress: ProgressCallback | None = None,
) -> TournamentReport:
    """Run missing games and checkpoint every replay and summary."""

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
        # The outer game index gives every matchup the same paired seed before
        # the tournament advances to the next replicate.
        for game_index in range(config.games_per_matchup):
            master_seed = tournament_game_seed(config.root_seed, game_index)
            for fugitive_name in config.fugitive_agents:
                for marshal_name in config.marshal_agents:
                    key = tournament_game_key(
                        fugitive_name, marshal_name, game_index
                    )
                    if key in records_by_key:
                        continue
                    started = time.perf_counter()
                    run = run_registered_match(
                        master_seed=master_seed,
                        fugitive_name=fugitive_name,
                        marshal_name=marshal_name,
                        fugitive_profile=config.fugitive_profile,
                        marshal_profile=config.marshal_profile,
                        max_decisions=config.max_decisions,
                        validate_invariants=config.validate_invariants,
                    )
                    record = _store_run(
                        config,
                        run,
                        game_index=game_index,
                        fugitive_name=fugitive_name,
                        marshal_name=marshal_name,
                        master_seed=master_seed,
                        wall_seconds=time.perf_counter() - started,
                    )
                    stream.write(canonical_json(record.to_dict()) + "\n")
                    stream.flush()
                    records_by_key[key] = record
                    report = summarize_tournament(
                        config, records_by_key.values()
                    )
                    write_summaries(config, report)
                    if progress is not None:
                        progress(record)

    report = summarize_tournament(config, records_by_key.values())
    write_summaries(config, report)
    return report


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
    write_text_atomic(config_path, pretty_json(config.to_dict()) + "\n")


def _load_records(path: Path, configuration_hash: str) -> tuple[GameRecord, ...]:
    if not path.exists():
        return ()
    records: list[GameRecord] = []
    keys: set[str] = set()
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"invalid games.jsonl line {line_number}: {exc}"
            ) from exc
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
            raise ValueError(
                f"record does not belong to configured matrix: {record.key}"
            )
        expected_seed = tournament_game_seed(config.root_seed, record.game_index)
        if record.master_seed != expected_seed:
            raise ValueError(f"record has the wrong paired seed: {record.key}")
        if not (output / record.manifest_path).is_file():
            raise ValueError(f"record manifest is missing: {record.manifest_path}")

        diagnostics_path = output / record.diagnostics_path
        if not diagnostics_path.is_file():
            raise ValueError(
                f"record diagnostics are missing: {record.diagnostics_path}"
            )
        try:
            diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"record diagnostics are invalid: {record.diagnostics_path}"
            ) from exc
        if not isinstance(diagnostics, Mapping):
            raise ValueError("record diagnostics must contain an object")
        if diagnostics.get("schema_version") != (
            TOURNAMENT_DIAGNOSTICS_SCHEMA_VERSION
        ):
            raise ValueError("record diagnostics use an unsupported schema")
        if diagnostics.get("configuration_hash") != config.configuration_hash:
            raise ValueError("record diagnostics have the wrong configuration")
        if diagnostics.get("game_key") != record.key:
            raise ValueError("record diagnostics have the wrong game key")

        diagnostic_counts = {
            "events": record.inference_events,
            "failures": record.inference_diagnostic_failures,
            "rollout_events": record.rollout_events,
            "rollout_failures": record.rollout_diagnostic_failures,
        }
        for field, expected_count in diagnostic_counts.items():
            entries = diagnostics.get(field)
            if not isinstance(entries, list) or len(entries) != expected_count:
                raise ValueError(
                    f"record diagnostics {field} count does not match its index"
                )


def _store_run(
    config: TournamentConfig,
    run: MatchRun,
    *,
    game_index: int,
    fugitive_name: str,
    marshal_name: str,
    master_seed: int,
    wall_seconds: float,
) -> GameRecord:
    cell = f"{fugitive_name}__vs__{marshal_name}"
    key = tournament_game_key(fugitive_name, marshal_name, game_index)
    manifest_relative = Path("manifests") / cell / f"{game_index:06d}.json"
    diagnostics_relative = Path("diagnostics") / cell / f"{game_index:06d}.json"
    write_text_atomic(
        config.output_directory / manifest_relative,
        run.manifest.to_json() + "\n",
    )
    diagnostics = {
        "schema_version": TOURNAMENT_DIAGNOSTICS_SCHEMA_VERSION,
        "configuration_hash": config.configuration_hash,
        "game_key": key,
        "events": [event.to_dict() for event in run.inference_events],
        "failures": [
            failure.to_dict() for failure in run.inference_diagnostic_failures
        ],
        "rollout_events": [event.to_dict() for event in run.rollout_events],
        "rollout_failures": [
            failure.to_dict() for failure in run.rollout_diagnostic_failures
        ],
    }
    write_text_atomic(
        config.output_directory / diagnostics_relative,
        pretty_json(diagnostics) + "\n",
    )
    manifest = run.manifest
    return GameRecord(
        configuration_hash=config.configuration_hash,
        key=key,
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
        rollout_events=len(run.rollout_events),
        rollout_diagnostic_failures=len(run.rollout_diagnostic_failures),
        manifest_path=manifest_relative.as_posix(),
        diagnostics_path=diagnostics_relative.as_posix(),
    )


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return value


__all__ = ["ProgressCallback", "run_tournament"]
