"""A minimal client for OKX's public v5 REST API: candles, funding and instrument metadata.

Public market data only - no key, no account, nothing signed. It is built on the standard
library (`urllib`), so the venue adds no dependency; `fetch`, `sleep` and `clock` are injectable
so the unit tests never touch the network and never wait.

`OkxLike` is the structural (duck-typed) surface the rest of the adapter needs, mirroring
`adapters.hyperliquid.bars.InfoLike`: tests pass a fake, production passes an `OkxClient`.

What the venue does, as read off the live API on 2026-09-20:

* every response is `{"code": "0", "msg": "", "data": [...]}`; a non-zero `code` is an error
  even when the HTTP status is 200 (`OkxError`);
* a burst over the per-endpoint budget (20 requests per 2 seconds for `history-candles`) is
  answered with HTTP 429 or with `code == "50011"`. A four-year backfill is some eight hundred
  pages, so the client spaces its requests (`min_interval_s`) instead of finding the limit,
  and backs off exponentially if it is told off anyway;
* it sits behind a CDN, which now and then answers a 5xx with an HTML page, resets a
  connection or times out. Over eight hundred requests that is likely rather than exotic, and
  losing the run at request 700 helps nobody, so those are retried on the same ladder. A 4xx
  other than 429 is the caller's mistake and is let through at once.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from typing import Any, Protocol, runtime_checkable

__all__ = ["MAX_RETRIES", "RATE_LIMIT_CODE", "OkxClient", "OkxError", "OkxLike"]

BASE_URL = "https://www.okx.com"
RATE_LIMIT_CODE = "50011"
NON_JSON_CODE = "non-json"
_RATE_LIMIT_STATUS = 429
_SERVER_ERROR_STATUS = 500
MAX_RETRIES = 6
"""Backoff is 1, 2, 4, ... seconds: about a minute in total before the error is let through."""

_TIMEOUT_S = 30.0


class OkxError(RuntimeError):
    """The venue answered with a non-zero `code` (its own error space, e.g. 51001 unknown
    instrument), or with something that is not the JSON envelope at all (`NON_JSON_CODE`)."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"OKX error {code}: {message}")
        self.code = code


@runtime_checkable
class OkxLike(Protocol):
    """The three public endpoints the adapter reads."""

    def history_candles(
        self, inst: str, bar: str, *, after_ms: int | None = None, limit: int = 300
    ) -> list[list[str]]: ...

    def funding_rate_history(
        self, inst: str, *, after_ms: int | None = None, limit: int = 100
    ) -> list[dict[str, str]]: ...

    def instrument(self, inst: str) -> dict[str, str]: ...


def _urlopen(url: str) -> bytes:
    # https, and the host is this module's own constant: nothing user-supplied reaches the scheme
    request = urllib.request.Request(url, headers={"User-Agent": "swingforge/okx-research"})
    with urllib.request.urlopen(request, timeout=_TIMEOUT_S) as response:
        body: bytes = response.read()
    return body


def _is_transient(exc: Exception) -> bool:
    """Worth another try: a rate limit, a server-side failure, or a connection that did not hold."""
    if isinstance(exc, OkxError):
        return exc.code in (RATE_LIMIT_CODE, NON_JSON_CODE)
    if isinstance(exc, urllib.error.HTTPError):  # a subclass of URLError: test it first
        return exc.code == _RATE_LIMIT_STATUS or exc.code >= _SERVER_ERROR_STATUS
    return isinstance(exc, urllib.error.URLError | TimeoutError | ConnectionError)


class OkxClient:
    """`OkxLike` over HTTPS. Stateless apart from the instant of its last request."""

    def __init__(
        self,
        *,
        base_url: str = BASE_URL,
        fetch: Callable[[str], bytes] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        min_interval_s: float = 0.12,
        max_retries: int = MAX_RETRIES,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._fetch = fetch or _urlopen
        self._sleep = sleep
        self._clock = clock
        self._min_interval_s = min_interval_s
        self._max_retries = max_retries
        self._last_request_at: float | None = None

    def _pace(self) -> None:
        """Start-to-start spacing: a slow request uses up its own interval rather than adding to it."""
        if self._last_request_at is not None:
            wait = self._min_interval_s - (self._clock() - self._last_request_at)
            if wait > 0:
                self._sleep(wait)
        self._last_request_at = self._clock()

    def _once(self, url: str) -> list[Any]:
        self._pace()
        raw = self._fetch(url)
        try:
            body = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise OkxError(NON_JSON_CODE, f"{url} answered with a non-JSON body: {raw[:120]!r}") from exc
        code = str(body.get("code", "missing")) if isinstance(body, dict) else "missing"
        if code != "0":
            raise OkxError(code, str(body.get("msg", "")) if isinstance(body, dict) else repr(body)[:120])
        data: list[Any] = body.get("data", [])
        return data

    def get(self, path: str, params: Mapping[str, str | int]) -> list[Any]:
        """`data` of one GET; a transient failure (`_is_transient`) is retried with exponential
        backoff, `max_retries` times, and anything else - or the last failure - is raised."""
        url = f"{self._base_url}{path}?{urllib.parse.urlencode(params)}"
        delay = 1.0
        attempt = 0
        while True:
            try:
                return self._once(url)
            except Exception as exc:
                if not _is_transient(exc) or attempt >= self._max_retries:
                    raise
            attempt += 1
            self._sleep(delay)
            delay *= 2

    def history_candles(
        self, inst: str, bar: str, *, after_ms: int | None = None, limit: int = 300
    ) -> list[list[str]]:
        """Up to `limit` candles strictly older than `after_ms`, newest first.

        Rows are `[ts, open, high, low, close, vol (contracts), volCcy (base coin),
        volCcyQuote, confirm]`, every field a string; `confirm == "0"` marks the candle that
        is still forming.
        """
        params: dict[str, str | int] = {"instId": inst, "bar": bar, "limit": limit}
        if after_ms is not None:
            params["after"] = after_ms
        return self.get("/api/v5/market/history-candles", params)

    def funding_rate_history(
        self, inst: str, *, after_ms: int | None = None, limit: int = 100
    ) -> list[dict[str, str]]:
        """Up to `limit` settlements strictly older than `after_ms`, newest first. The venue
        serves roughly the last three months and nothing older."""
        params: dict[str, str | int] = {"instId": inst, "limit": limit}
        if after_ms is not None:
            params["after"] = after_ms
        return self.get("/api/v5/public/funding-rate-history", params)

    def instrument(self, inst: str) -> dict[str, str]:
        """The venue's metadata for one swap (`tickSz`, `ctVal`, `state`, ...).

        An unknown id is normally the venue's own error 51001; an empty answer instead is
        reported as a `LookupError` rather than dressed up as one of the venue's codes.
        """
        data = self.get("/api/v5/public/instruments", {"instType": "SWAP", "instId": inst})
        if not data:
            raise LookupError(f"OKX lists no swap called {inst}")
        row: dict[str, str] = data[0]
        return row
