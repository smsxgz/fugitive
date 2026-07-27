"""Paired full-game tournament experiments."""

from .model import (
    GameRecord,
    MatchupSummary,
    TournamentConfig,
    TournamentReport,
    summarize_tournament,
    tournament_game_key,
    tournament_game_seed,
    wilson_interval,
)
from .runner import run_tournament

__all__ = [
    "GameRecord",
    "MatchupSummary",
    "TournamentConfig",
    "TournamentReport",
    "run_tournament",
    "summarize_tournament",
    "tournament_game_key",
    "tournament_game_seed",
    "wilson_interval",
]
