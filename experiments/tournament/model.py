"""Data model, seed protocol, and summary mathematics for tournaments."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Mapping

from fugitive.agents.registry import FUGITIVE_AGENT_REGISTRY, MARSHAL_AGENT_REGISTRY
from fugitive.game.model import Winner
from fugitive.runtime.manifest import RULES_SHA256, RULES_VERSION, MatchStatus
from fugitive.shared.reproducibility import (
    AGENT_PROFILES,
    derive_seed,
    parse_seed,
    validate_seed,
)


TOURNAMENT_SCHEMA_VERSION = 2
TOURNAMENT_DIAGNOSTICS_SCHEMA_VERSION = 1
TOURNAMENT_GAME_SEED_DOMAIN = "fugitive.tournament.game.v1"
_WILSON_Z_95 = 1.959963984540054


@dataclass(frozen=True, slots=True)
class TournamentConfig:
    """The fixed protocol and extendable target for one matchup matrix."""

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
            self.fugitive_agents, FUGITIVE_AGENT_REGISTRY, "Fugitive"
        )
        _validate_agent_names(
            self.marshal_agents, MARSHAL_AGENT_REGISTRY, "Marshal"
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
                "fugitive": {
                    "names": list(self.fugitive_agents),
                    "resolved_specs": [
                        FUGITIVE_AGENT_REGISTRY[name]
                        .build(0, profile=self.fugitive_profile)
                        .spec.to_dict()
                        for name in self.fugitive_agents
                    ],
                },
                "marshal": {
                    "names": list(self.marshal_agents),
                    "resolved_specs": [
                        MARSHAL_AGENT_REGISTRY[name]
                        .build(0, profile=self.marshal_profile)
                        .spec.to_dict()
                        for name in self.marshal_agents
                    ],
                },
            },
            "max_decisions": self.max_decisions,
            "validate_invariants": self.validate_invariants,
        }

    @property
    def configuration_hash(self) -> str:
        payload = canonical_json(self.identity_dict()).encode("utf-8")
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
    status: MatchStatus
    winner: Winner | None
    reason: str | None
    rounds: int | None
    decision_count: int
    wall_seconds: float
    inference_events: int
    inference_diagnostic_failures: int
    rollout_events: int
    rollout_diagnostic_failures: int
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
                "rollout_events": self.rollout_events,
                "rollout_failures": self.rollout_diagnostic_failures,
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
            configuration_hash=_string(
                data.get("configuration_hash"), "configuration_hash"
            ),
            key=_string(data.get("key"), "key"),
            game_index=_integer(data.get("game_index"), "game_index"),
            fugitive_agent=_string(matchup.get("fugitive"), "Fugitive agent"),
            marshal_agent=_string(matchup.get("marshal"), "Marshal agent"),
            master_seed=parse_seed(data.get("master_seed"), "master"),
            status=MatchStatus(str(outcome.get("status"))),
            winner=None if winner is None else Winner(str(winner)),
            reason=(
                None if outcome.get("reason") is None else str(outcome.get("reason"))
            ),
            rounds=_optional_integer(outcome.get("rounds"), "rounds"),
            decision_count=_integer(outcome.get("decision_count"), "decision_count"),
            wall_seconds=_finite_float(data.get("wall_seconds"), "wall_seconds"),
            inference_events=_integer(diagnostics.get("events"), "diagnostic events"),
            inference_diagnostic_failures=_integer(
                diagnostics.get("failures"), "diagnostic failures"
            ),
            rollout_events=_integer(
                diagnostics.get("rollout_events", 0), "rollout diagnostic events"
            ),
            rollout_diagnostic_failures=_integer(
                diagnostics.get("rollout_failures", 0),
                "rollout diagnostic failures",
            ),
            manifest_path=_string(data.get("manifest_path"), "manifest_path"),
            diagnostics_path=_string(
                diagnostics.get("path"), "diagnostics path"
            ),
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
    rollout_events: int
    rollout_diagnostic_failures: int
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
            "rollout_events": self.rollout_events,
            "rollout_diagnostic_failures": self.rollout_diagnostic_failures,
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
                record for record in cell if record.status is MatchStatus.COMPLETED
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
                        record.status is MatchStatus.TRUNCATED for record in cell
                    ),
                    errors=sum(record.status is MatchStatus.ERROR for record in cell),
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
                    rollout_events=sum(record.rollout_events for record in cell),
                    rollout_diagnostic_failures=sum(
                        record.rollout_diagnostic_failures for record in cell
                    ),
                    reasons=Counter(
                        record.reason
                        for record in completed
                        if record.reason is not None
                    ),
                )
            )
    return TournamentReport(config.configuration_hash, ordered, tuple(matchups))


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


def canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
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


__all__ = [
    "GameRecord",
    "MatchupSummary",
    "TOURNAMENT_DIAGNOSTICS_SCHEMA_VERSION",
    "TOURNAMENT_GAME_SEED_DOMAIN",
    "TOURNAMENT_SCHEMA_VERSION",
    "TournamentConfig",
    "TournamentReport",
    "canonical_json",
    "summarize_tournament",
    "tournament_game_key",
    "tournament_game_seed",
    "wilson_interval",
]
