"""HTTP transport and in-memory ownership for Fugitive Web sessions."""

from __future__ import annotations

import argparse
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import math
from pathlib import Path
import threading
import time
from typing import Callable, Iterator, Mapping
from urllib.parse import unquote, urlsplit
import uuid

from ..agents.registry import DEFAULT_FUGITIVE_AGENT, DEFAULT_MARSHAL_AGENT
from .session import (
    DEFAULT_AUTO_STEP_LIMIT,
    GameSession,
    WebAPIError,
    _PROFILE_UNSET,
    _is_integer,
    agent_catalog,
)


DEFAULT_SESSION_IDLE_TTL_SECONDS = 30 * 60
DEFAULT_MAX_ACTIVE_SESSIONS = 128
MAX_REQUEST_BYTES = 1_000_000

_LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class _SessionEntry:
    session: GameSession
    last_access: float
    active_leases: int = 0
    retired: bool = False


class SessionStore:
    """Thread-safe in-memory owner of active local game sessions."""

    def __init__(
        self,
        *,
        idle_ttl_seconds: float | None = DEFAULT_SESSION_IDLE_TTL_SECONDS,
        max_active_sessions: int = DEFAULT_MAX_ACTIVE_SESSIONS,
        auto_step_limit: int = DEFAULT_AUTO_STEP_LIMIT,
        clock: Callable[[], float] = time.monotonic,
        session_factory: Callable[..., GameSession] = GameSession,
    ) -> None:
        if idle_ttl_seconds is not None and (
            not isinstance(idle_ttl_seconds, (int, float))
            or isinstance(idle_ttl_seconds, bool)
            or not math.isfinite(idle_ttl_seconds)
            or idle_ttl_seconds <= 0
        ):
            raise ValueError("idle_ttl_seconds must be positive or None")
        if (
            not _is_integer(max_active_sessions)
            or max_active_sessions < 1
        ):
            raise ValueError("max_active_sessions must be a positive integer")
        if (
            not _is_integer(auto_step_limit)
            or not 1 <= auto_step_limit <= DEFAULT_AUTO_STEP_LIMIT
        ):
            raise ValueError(
                f"auto_step_limit must be from 1 through {DEFAULT_AUTO_STEP_LIMIT}"
            )
        self.idle_ttl_seconds = idle_ttl_seconds
        self.max_active_sessions = max_active_sessions
        self.auto_step_limit = auto_step_limit
        self._clock = clock
        self._session_factory = session_factory
        self._sessions: OrderedDict[str, _SessionEntry] = OrderedDict()
        self._lock = threading.RLock()

    def _remove_expired_locked(self, now: float) -> tuple[int, list[GameSession]]:
        if self.idle_ttl_seconds is None:
            return 0, []
        expired_ids = [
            session_id
            for session_id, entry in self._sessions.items()
            if now - entry.last_access >= self.idle_ttl_seconds
        ]
        to_close: list[GameSession] = []
        for session_id in expired_ids:
            entry = self._sessions.pop(session_id)
            session = self._retire_entry_locked(entry)
            if session is not None:
                to_close.append(session)
        return len(expired_ids), to_close

    @staticmethod
    def _retire_entry_locked(entry: _SessionEntry) -> GameSession | None:
        entry.retired = True
        return entry.session if entry.active_leases == 0 else None

    @staticmethod
    def _close_sessions(sessions: list[GameSession]) -> None:
        for session in sessions:
            session.close()

    def cleanup_expired(self) -> int:
        """Remove expired store entries and close sessions no longer leased."""

        with self._lock:
            removed_count, expired = self._remove_expired_locked(self._clock())
        self._close_sessions(expired)
        return removed_count

    @property
    def active_count(self) -> int:
        self.cleanup_expired()
        with self._lock:
            return len(self._sessions)

    def _new_session(self, payload: Mapping[str, object]) -> GameSession:
        mode = payload.get("mode", "spectate")
        fugitive_agent = payload.get("fugitive_agent", DEFAULT_FUGITIVE_AGENT)
        marshal_agent = payload.get("marshal_agent", DEFAULT_MARSHAL_AGENT)
        seed = payload.get("seed")
        spectator_view = payload.get("spectator_view")
        execution_profile = payload.get("execution_profile", "full")
        if not isinstance(mode, str):
            raise WebAPIError(HTTPStatus.BAD_REQUEST, "invalid_mode", "mode must be a string")
        if not isinstance(fugitive_agent, str) or not isinstance(marshal_agent, str):
            raise WebAPIError(
                HTTPStatus.BAD_REQUEST,
                "invalid_agent",
                "agent identifiers must be strings",
            )
        return self._session_factory(
            session_id=uuid.uuid4().hex,
            mode=mode,
            fugitive_agent=fugitive_agent,
            marshal_agent=marshal_agent,
            seed=seed,  # type: ignore[arg-type]
            spectator_view=spectator_view,  # type: ignore[arg-type]
            execution_profile=execution_profile,  # type: ignore[arg-type]
            auto_step_limit=self.auto_step_limit,
        )

    def _register_session(
        self,
        session: GameSession,
        *,
        initial_leases: int,
    ) -> _SessionEntry:
        to_close: list[GameSession]
        with self._lock:
            now = self._clock()
            _removed_count, to_close = self._remove_expired_locked(now)
            registered_entry = _SessionEntry(
                session,
                now,
                active_leases=initial_leases,
            )
            self._sessions[session.id] = registered_entry
            self._sessions.move_to_end(session.id)
            while len(self._sessions) > self.max_active_sessions:
                _session_id, entry = self._sessions.popitem(last=False)
                evicted = self._retire_entry_locked(entry)
                if evicted is not None:
                    to_close.append(evicted)
        self._close_sessions(to_close)
        return registered_entry

    def create(self, payload: Mapping[str, object]) -> GameSession:
        """Create and register a session for direct, non-request use."""

        session = self._new_session(payload)
        self._register_session(session, initial_leases=0)
        return session

    @contextmanager
    def create_lease(self, payload: Mapping[str, object]) -> Iterator[GameSession]:
        """Create a session whose first request owns a lease immediately.

        The session is constructed exactly once.  Its lease is installed in
        the same critical section that publishes it to the store, so an LRU
        eviction can retire but cannot close it before the caller serializes
        the creation response.
        """

        session = self._new_session(payload)
        entry = self._register_session(session, initial_leases=1)
        try:
            yield session
        finally:
            self._release_lease(entry)

    def get(self, session_id: str) -> GameSession:
        """Return a session for local inspection; HTTP handlers use ``lease``."""

        session: GameSession | None = None
        with self._lock:
            now = self._clock()
            _removed_count, expired = self._remove_expired_locked(now)
            entry = self._sessions.get(session_id)
            if entry is not None:
                entry.last_access = now
                self._sessions.move_to_end(session_id)
                session = entry.session
        self._close_sessions(expired)
        if session is None:
            raise WebAPIError(
                HTTPStatus.NOT_FOUND,
                "game_not_found",
                f"game {session_id!r} does not exist",
            )
        return session

    @contextmanager
    def lease(self, session_id: str) -> Iterator[GameSession]:
        """Keep a session alive for the duration of one concurrent request."""

        entry: _SessionEntry | None = None
        with self._lock:
            now = self._clock()
            _removed_count, expired = self._remove_expired_locked(now)
            entry = self._sessions.get(session_id)
            if entry is not None:
                entry.last_access = now
                entry.active_leases += 1
                self._sessions.move_to_end(session_id)
        self._close_sessions(expired)
        if entry is None:
            raise WebAPIError(
                HTTPStatus.NOT_FOUND,
                "game_not_found",
                f"game {session_id!r} does not exist",
            )

        try:
            yield entry.session
        finally:
            self._release_lease(entry)

    def _release_lease(self, entry: _SessionEntry) -> None:
        to_close: GameSession | None = None
        with self._lock:
            entry.active_leases -= 1
            if entry.active_leases < 0:  # pragma: no cover - ownership guard
                raise RuntimeError("session lease count became negative")
            if entry.retired and entry.active_leases == 0:
                to_close = entry.session
        if to_close is not None:
            to_close.close()

    def delete(self, session_id: str) -> None:
        """Remove one session and promptly release its agent state."""

        deleted: GameSession | None = None
        found = False
        with self._lock:
            _removed_count, expired = self._remove_expired_locked(self._clock())
            entry = self._sessions.pop(session_id, None)
            if entry is not None:
                found = True
                deleted = self._retire_entry_locked(entry)
        self._close_sessions(expired)
        if not found:
            raise WebAPIError(
                HTTPStatus.NOT_FOUND,
                "game_not_found",
                f"game {session_id!r} does not exist",
            )
        if deleted is not None:
            deleted.close()


def make_handler(
    store: SessionStore,
    *,
    static_dir: Path | None = None,
    error_logger: logging.Logger | None = None,
) -> type[BaseHTTPRequestHandler]:
    """Bind a store and static root to a request-handler class."""

    resolved_static = static_dir.resolve() if static_dir is not None else None
    logger = error_logger if error_logger is not None else _LOGGER

    class FugitiveRequestHandler(BaseHTTPRequestHandler):
        server_version = "FugitiveLocal/1.0"

        def do_OPTIONS(self) -> None:  # noqa: N802
            self.send_response(HTTPStatus.NO_CONTENT)
            self._common_headers()
            self.end_headers()

        def do_GET(self) -> None:  # noqa: N802
            try:
                path = urlsplit(self.path).path
                if path == "/api/agents":
                    self._send_json(HTTPStatus.OK, agent_catalog())
                    return
                parts = _api_game_parts(path)
                if parts is not None and len(parts) == 1:
                    with store.lease(parts[0]) as session:
                        state = session.as_dict()
                    self._send_json(HTTPStatus.OK, state)
                    return
                if (
                    parts is not None
                    and len(parts) == 2
                    and parts[1] == "export"
                ):
                    with store.lease(parts[0]) as session:
                        exported = session.export_trace()
                    self._send_json(
                        HTTPStatus.OK,
                        exported,
                    )
                    return
                if path.startswith("/api/"):
                    raise WebAPIError(
                        HTTPStatus.NOT_FOUND, "endpoint_not_found", "API endpoint not found"
                    )
                self._serve_static(path)
            except WebAPIError as exc:
                self._send_api_error(exc)
            except Exception as exc:
                self._send_internal_error(exc)

        def do_POST(self) -> None:  # noqa: N802
            try:
                payload = self._read_json_object()
                path = urlsplit(self.path).path
                if path == "/api/games":
                    with store.create_lease(payload) as session:
                        state = session.as_dict()
                    self._send_json(HTTPStatus.CREATED, state)
                    return
                parts = _api_game_parts(path)
                if parts is None or len(parts) != 2:
                    raise WebAPIError(
                        HTTPStatus.NOT_FOUND, "endpoint_not_found", "API endpoint not found"
                    )
                operation = parts[1]
                with store.lease(parts[0]) as session:
                    if operation == "action":
                        state = session.apply_human_action(payload)
                    elif operation == "step":
                        state = session.step()
                    elif operation == "auto":
                        max_steps = payload.get("max_steps")
                        state = session.auto(max_steps=max_steps)  # type: ignore[arg-type]
                    elif operation == "continue":
                        max_steps = payload.get("max_steps")
                        state = session.continue_after_stall(
                            max_steps=max_steps  # type: ignore[arg-type]
                        )
                    elif operation == "terminate":
                        state = session.terminate()
                    elif operation == "reset":
                        state = session.reset(
                            seed=payload.get("seed"),  # type: ignore[arg-type]
                            execution_profile=(
                                payload["execution_profile"]
                                if "execution_profile" in payload
                                else _PROFILE_UNSET
                            ),
                        )
                    elif operation == "view":
                        state = session.set_spectator_view(
                            payload.get("spectator_view")
                        )
                    else:
                        raise WebAPIError(
                            HTTPStatus.NOT_FOUND,
                            "endpoint_not_found",
                            "API endpoint not found",
                        )
                self._send_json(HTTPStatus.OK, state)
            except WebAPIError as exc:
                self._send_api_error(exc)
            except Exception as exc:
                self._send_internal_error(exc)

        def do_DELETE(self) -> None:  # noqa: N802
            try:
                path = urlsplit(self.path).path
                parts = _api_game_parts(path)
                if parts is None or len(parts) != 1:
                    raise WebAPIError(
                        HTTPStatus.NOT_FOUND,
                        "endpoint_not_found",
                        "API endpoint not found",
                    )
                session_id = parts[0]
                store.delete(session_id)
                self._send_json(
                    HTTPStatus.OK,
                    {"id": session_id, "deleted": True},
                )
            except WebAPIError as exc:
                self._send_api_error(exc)
            except Exception as exc:
                self._send_internal_error(exc)

        def _read_json_object(self) -> dict[str, object]:
            raw_length = self.headers.get("Content-Length", "0")
            try:
                length = int(raw_length)
            except ValueError as exc:
                raise WebAPIError(
                    HTTPStatus.BAD_REQUEST, "invalid_content_length", "invalid Content-Length"
                ) from exc
            if length < 0 or length > MAX_REQUEST_BYTES:
                raise WebAPIError(
                    HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                    "request_too_large",
                    "JSON request body is too large",
                )
            raw = self.rfile.read(length)
            if not raw:
                return {}
            try:
                value = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise WebAPIError(
                    HTTPStatus.BAD_REQUEST, "invalid_json", "request body must be valid JSON"
                ) from exc
            if not isinstance(value, dict):
                raise WebAPIError(
                    HTTPStatus.BAD_REQUEST, "invalid_json", "request JSON must be an object"
                )
            return value

        def _send_api_error(self, exc: WebAPIError) -> None:
            self._send_json(
                exc.status,
                {"error": {"code": exc.code, "message": exc.message}},
            )

        def _send_internal_error(self, exc: Exception) -> None:
            error_id = uuid.uuid4().hex
            logger.exception(
                "unhandled web request error error_id=%s method=%s path=%s",
                error_id,
                self.command,
                urlsplit(self.path).path,
                exc_info=exc,
            )
            self._send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {
                    "error": {
                        "code": "internal_server_error",
                        "message": "an unexpected server error occurred",
                        "error_id": error_id,
                    }
                },
            )

        def _send_json(self, status: int, value: object) -> None:
            body = json.dumps(value, separators=(",", ":"), ensure_ascii=True).encode(
                "utf-8"
            )
            self.send_response(status)
            self._common_headers()
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _serve_static(self, request_path: str) -> None:
            if resolved_static is None or not resolved_static.is_dir():
                raise WebAPIError(
                    HTTPStatus.NOT_FOUND, "static_not_found", "web interface is not installed"
                )
            relative = "index.html" if request_path == "/" else unquote(request_path.lstrip("/"))
            candidate = (resolved_static / relative).resolve()
            try:
                candidate.relative_to(resolved_static)
            except ValueError as exc:
                raise WebAPIError(
                    HTTPStatus.NOT_FOUND, "static_not_found", "static file not found"
                ) from exc
            if not candidate.is_file():
                raise WebAPIError(
                    HTTPStatus.NOT_FOUND, "static_not_found", "static file not found"
                )
            content_types = {
                ".html": "text/html; charset=utf-8",
                ".css": "text/css; charset=utf-8",
                ".js": "text/javascript; charset=utf-8",
                ".json": "application/json; charset=utf-8",
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".svg": "image/svg+xml",
                ".ico": "image/x-icon",
            }
            body = candidate.read_bytes()
            self.send_response(HTTPStatus.OK)
            self._common_headers()
            self.send_header(
                "Content-Type", content_types.get(candidate.suffix.lower(), "application/octet-stream")
            )
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(body)

        def _common_headers(self) -> None:
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Cross-Origin-Resource-Policy", "same-origin")

        def log_message(self, format: str, *args: object) -> None:
            # Keep the normal useful server log while avoiding user-controlled
            # terminal formatting from raw request paths.
            super().log_message(format, *args)

    return FugitiveRequestHandler


def _api_game_parts(path: str) -> tuple[str, ...] | None:
    prefix = "/api/games/"
    if not path.startswith(prefix):
        return None
    suffix = path[len(prefix) :].strip("/")
    if not suffix:
        return ()
    return tuple(suffix.split("/"))


def create_server(
    host: str = "127.0.0.1",
    port: int = 8000,
    *,
    store: SessionStore | None = None,
    static_dir: Path | None = None,
    error_logger: logging.Logger | None = None,
) -> ThreadingHTTPServer:
    """Create, but do not start, a local threaded server."""

    if static_dir is None:
        static_dir = Path(__file__).resolve().with_name("static")
    actual_store = store if store is not None else SessionStore()
    return ThreadingHTTPServer(
        (host, port),
        make_handler(
            actual_store,
            static_dir=static_dir,
            error_logger=error_logger,
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the local Fugitive web interface")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--static-dir",
        type=Path,
        default=Path(__file__).resolve().with_name("static"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    server = create_server(args.host, args.port, static_dir=args.static_dir)
    host, port = server.server_address[:2]
    print(f"Fugitive web interface: http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0

__all__ = [
    "SessionStore",
    "build_parser",
    "create_server",
    "main",
    "make_handler",
]
