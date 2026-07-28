"""Local Web application for playing and observing Fugitive games."""

from .server import SessionStore, build_parser, create_server, main, make_handler
from .session import (
    MAX_SAFE_JSON_INTEGER,
    WEB_TRACE_SCHEMA_VERSION,
    GameSession,
    WebAPIError,
    agent_catalog,
    parse_web_seed,
)

__all__ = [
    "GameSession",
    "MAX_SAFE_JSON_INTEGER",
    "SessionStore",
    "WEB_TRACE_SCHEMA_VERSION",
    "WebAPIError",
    "agent_catalog",
    "build_parser",
    "create_server",
    "main",
    "make_handler",
    "parse_web_seed",
]
