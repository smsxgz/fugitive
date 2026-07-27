"""JSON, CSV, and Markdown output for tournament reports."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

from .model import MatchupSummary, TournamentConfig, TournamentReport, canonical_json


def write_summaries(config: TournamentConfig, report: TournamentReport) -> None:
    """Write the three human- and machine-readable tournament summaries."""

    write_text_atomic(
        config.output_directory / "summary.json",
        pretty_json(report.to_dict()) + "\n",
    )
    write_csv(config.output_directory / "summary.csv", report.matchups)
    write_text_atomic(
        config.output_directory / "summary.md",
        summary_markdown(config, report),
    )


def write_csv(path: Path, matchups: Iterable[MatchupSummary]) -> None:
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
        "rollout_events",
        "rollout_diagnostic_failures",
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
            row["reasons_json"] = canonical_json(dict(sorted(item.reasons.items())))
            writer.writerow(row)
    temporary.replace(path)


def summary_markdown(config: TournamentConfig, report: TournamentReport) -> str:
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
    lines.extend(
        (
            "",
            "Win rates use completed games only. Errors and explicit experiment "
            "watchdog truncations are never converted into draws or wins.",
            "",
        )
    )
    return "\n".join(lines)


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(path)


def pretty_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        indent=2,
    )


def _format_optional(value: float | None, digits: int) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


__all__ = [
    "pretty_json",
    "summary_markdown",
    "write_csv",
    "write_summaries",
    "write_text_atomic",
]
