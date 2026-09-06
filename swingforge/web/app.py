"""Read-mostly FastAPI dashboard over the per-venue DuckDB stores.

One route writes: ``PUT /api/settings``, which validates against
:class:`~swingforge.core.settings.Settings` (so the 2% ``risk_pct`` cap holds no matter what
a client sends) and appends exactly one row to the venue's ``settings_log`` via
:meth:`~swingforge.adapters.store.Store.write_settings`. Every other route opens its venue
store(s) read-only, builds a response, and closes them again -- no store is held open across
requests. That alone does not avoid contention, though: DuckDB's file lock is *exclusive*
across processes even for a read-only open, so if the paper-trading CLI happens to have a
venue's store open at the exact moment a request arrives, the open call raises
``duckdb.IOException`` regardless of how briefly either side holds the file -- this is not a
"keep requests short" problem. The paper runner's side of the bargain is to open its venue
store only briefly, at each 4H bar close, and close it again immediately; this module's side
is ``_open_store_with_retry``: retry a failed open a few times with a short backoff
(``OPEN_RETRIES`` attempts, ``OPEN_BACKOFF_S`` seconds apart) and, if the file is still locked
after every attempt, return ``503 {"detail": "venue busy"}`` instead of a raw 500 -- for every
GET route and for the settings PUT.

``create_app(data_dir=None, token=None)`` resolves ``data_dir`` from the ``SWINGFORGE_DATA_DIR``
env var (default ``"data"``) and ``token`` from ``SWINGFORGE_TOKEN``, so a bare
``uvicorn swingforge.web.app:app`` picks up both from the environment. ``GET /`` (the static
HTML shell, with no data of its own) is always open; when a token is configured, every request
under ``/api/*`` must carry ``Authorization: Bearer <token>`` (checked in constant time via
:func:`secrets.compare_digest`, so no early exit leaks how much of the token matched) or gets
a 401, and the auto-generated ``/docs``, ``/redoc`` and ``/openapi.json`` are disabled outright
so they can't leak the API's shape to an unauthenticated caller. With no token configured, the
dashboard (docs included) is open to whoever can reach it, which is intended only for binding
to ``127.0.0.1`` (see the handoff).
"""

from __future__ import annotations

import os
import secrets
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import duckdb
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from swingforge.adapters.store import Store
from swingforge.core.settings import Settings
from swingforge.web import queries

__all__ = ["app", "create_app"]

_STATIC_DIR = Path(__file__).parent / "static"

OPEN_RETRIES = 5
"""How many times `_open_store_with_retry` attempts to open a locked venue store."""

OPEN_BACKOFF_S = 0.2
"""Seconds to sleep between retries in `_open_store_with_retry` -- comfortably longer than the
paper runner's per-bar-close open/close window, short enough that five attempts stay well
under a second."""


class SettingsPutBody(BaseModel):
    """Body of ``PUT /api/settings``. Nesting `Settings` gives the 422 for free on an invalid field."""

    venue: Literal["hyperliquid", "oanda"]
    settings: Settings


def _open_store_with_retry(path: Path, *, read_only: bool) -> Store:
    """Open ``path`` as a :class:`Store`, retrying past a transient DuckDB file-lock conflict.

    See the module docstring: DuckDB's file lock is exclusive across processes even for a
    read-only open, so a concurrent paper-runner open makes this raise ``duckdb.IOException``
    outright. Retried up to ``OPEN_RETRIES`` times, ``OPEN_BACKOFF_S`` apart; if the file is
    still locked after the last attempt, raises ``HTTPException(503)`` (body
    ``{"detail": "venue busy"}``) instead of letting the raw ``duckdb.IOException`` become a
    500, so every caller -- every GET route and the settings PUT -- gets the same response.
    """
    last_exc: duckdb.IOException | None = None
    for attempt in range(OPEN_RETRIES):
        try:
            return Store(path, read_only=read_only)
        except duckdb.IOException as exc:
            last_exc = exc
            if attempt < OPEN_RETRIES - 1:
                time.sleep(OPEN_BACKOFF_S)
    raise HTTPException(status_code=503, detail="venue busy") from last_exc


def _open_present_stores(data_dir: Path) -> dict[str, Store]:
    """Every present venue's store, opened read-only. Caller must close each one.

    Opens venues one at a time rather than all at once: if a later venue's open fails (after
    exhausting its retries), every store already opened earlier in this call is closed before
    the exception propagates, so a locked second venue never leaks the first venue's file
    handle for the rest of the process's life.
    """
    stores: dict[str, Store] = {}
    try:
        for venue in queries.present_venues(data_dir):
            stores[venue] = _open_store_with_retry(queries.venue_db_path(data_dir, venue), read_only=True)
    except Exception:
        _close_all(stores)
        raise
    return stores


def _close_all(stores: dict[str, Store]) -> None:
    for store in stores.values():
        store.close()


def _bearer_token_matches(expected: str, header_value: str | None) -> bool:
    """Constant-time check of ``header_value`` against ``"Bearer {expected}"``.

    Both sides are encoded to UTF-8 with ``errors="surrogateescape"`` before comparing, so a
    header that doesn't even decode cleanly as UTF-8 (still handed back as a ``str`` -- ASGI
    header values are raw bytes, and Starlette decodes them permissively) compares as a plain
    mismatch (401) rather than raising (500); `secrets.compare_digest` needs bytes (or
    ASCII-only `str`) to do its constant-time comparison and raises `TypeError` on a non-ASCII
    `str`, which the encode step here avoids entirely. Compared in (close to) constant time so
    a byte-by-byte early exit on the wrong prefix can't be used to guess the token.
    """
    if header_value is None:
        return False
    expected_bytes = f"Bearer {expected}".encode("utf-8", errors="surrogateescape")
    actual_bytes = header_value.encode("utf-8", errors="surrogateescape")
    return secrets.compare_digest(expected_bytes, actual_bytes)


def create_app(data_dir: str | Path | None = None, token: str | None = None) -> FastAPI:
    """Build the dashboard app. ``data_dir``/``token`` default from the environment."""
    env_data_dir = os.environ.get("SWINGFORGE_DATA_DIR", "data")
    resolved_data_dir = Path(data_dir if data_dir is not None else env_data_dir)
    resolved_token = token if token is not None else (os.environ.get("SWINGFORGE_TOKEN") or None)

    app = FastAPI(
        title="swingforge dashboard",
        # A token is meant to keep the API private; leaving the docs (which enumerate every
        # route and schema) reachable without auth would defeat that, so they're disabled
        # outright rather than merely middleware-guarded like the API routes below.
        docs_url="/docs" if not resolved_token else None,
        redoc_url="/redoc" if not resolved_token else None,
        openapi_url="/openapi.json" if not resolved_token else None,
    )

    @app.middleware("http")
    async def bearer_auth(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        # "/" is the static HTML shell -- no data of its own, so it stays open even with a
        # token configured; only the API surface underneath it is guarded.
        protected = request.url.path.startswith("/api/")
        if (
            resolved_token
            and protected
            and not _bearer_token_matches(resolved_token, request.headers.get("authorization"))
        ):
            return JSONResponse({"detail": "unauthorized"}, status_code=401)
        return await call_next(request)

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(_STATIC_DIR / "index.html")

    @app.get("/api/overview")
    def get_overview() -> dict:
        now = datetime.now(UTC)
        stores = _open_present_stores(resolved_data_dir)
        try:
            venues = {venue: queries.overview_for_venue(store, venue, now) for venue, store in stores.items()}
        finally:
            _close_all(stores)
        return {"venues": venues, "generated_at": now}

    @app.get("/api/tournament/latest")
    def get_tournament_latest() -> dict:
        stores = _open_present_stores(resolved_data_dir)
        try:
            found = queries.latest_tournament_run(stores)
            if found is None:
                raise HTTPException(status_code=404, detail="no tournament results")
            run_id, _ts = found
            return queries.tournament_payload(stores, run_id)
        finally:
            _close_all(stores)

    @app.get("/api/trades/{run_id}/{trade_id}")
    def get_trade(run_id: str, trade_id: str) -> dict:
        stores = _open_present_stores(resolved_data_dir)
        try:
            detail = queries.trade_detail(stores, run_id, trade_id)
            if detail is None:
                raise HTTPException(status_code=404, detail="trade not found")
            return detail
        finally:
            _close_all(stores)

    @app.get("/api/settings")
    def get_settings() -> dict:
        stores = _open_present_stores(resolved_data_dir)
        try:
            return queries.settings_payload(stores)
        finally:
            _close_all(stores)

    @app.put("/api/settings")
    def put_settings(body: SettingsPutBody) -> dict:
        path = queries.venue_db_path(resolved_data_dir, body.venue)
        path.parent.mkdir(parents=True, exist_ok=True)
        store = _open_store_with_retry(path, read_only=False)
        try:
            version = store.write_settings(body.settings, actor="dashboard")
        finally:
            store.close()
        return {"venue": body.venue, "version": version}

    return app


app = create_app()
