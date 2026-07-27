"""Fugitive rules engine, information-set agents, and local Web app."""

from .game.engine import GameEngine, play_game
from .game.model import FugitiveAction, Observation, Role, Winner

__version__ = "0.1.0"

__all__ = [
    "FugitiveAction",
    "GameEngine",
    "Observation",
    "Role",
    "Winner",
    "play_game",
]
