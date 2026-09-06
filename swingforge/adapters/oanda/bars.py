"""OANDA bar source: v20 REST candles via a duck-typed API client.

`ApiLike` mirrors the one method this module calls on `oandapyV20.API` —
`request(endpoint) -> dict` — so unit tests can pass a fake that returns canned OANDA-shaped
payloads without touching the network. When no `api` is given, a real
`oandapyV20.API(access_token=token, environment=environment)` is constructed lazily.

Daily bars are a deliberate deviation from a literal `"D"` granularity request: OANDA's own
daily candles roll over at 21:00 UTC (the NY trading-day close via broker convention), which
would misalign Daily bias against Hyperliquid's midnight-UTC perp bars. Instead, `history`
fetches H1 candles and `recut_daily` re-cuts them to 00:00-UTC calendar days.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from typing import Any, Protocol, runtime_checkable

from swingforge.core.types import TF, Bar, Instrument

__all__ = [
    "MIN_HOURS_FOR_DAILY_RECUT",
    "ApiLike",
    "OandaBars",
    "instrument_from_account",
    "oanda_instrument",
    "recut_daily",
]

_GRANULARITY: dict[str, str] = {"1h": "H1", "4h": "H4"}
"""swingforge TF -> OANDA candle granularity. `1d` is handled separately (see module docstring)."""

_TF_DURATION: dict[str, timedelta] = {
    "1h": timedelta(hours=1),
    "4h": timedelta(hours=4),
    "1d": timedelta(hours=24),
}

MIN_HOURS_FOR_DAILY_RECUT = 12
"""A re-cut UTC day is emitted only if at least this many 1H bars back it.

Documented threshold: a day with fewer 1H bars (a holiday, a data gap, a partial day at a
query boundary) is too sparse to trust as a Daily bar and is dropped rather than emitted
from incomplete data.
"""

_TZ_SUFFIX = re.compile(r"(Z|[+-]\d{2}:\d{2})$")

_STATIC_INSTRUMENTS: dict[str, tuple[Decimal, str]] = {
    "EUR_USD": (Decimal("0.00001"), "USD"),
    "GBP_USD": (Decimal("0.00001"), "USD"),
    "USD_JPY": (Decimal("0.001"), "JPY"),
    "AUD_USD": (Decimal("0.00001"), "USD"),
    "GBP_JPY": (Decimal("0.001"), "JPY"),
    # OANDA's own account metadata reports displayPrecision=3 (tick 0.001) for XAU_USD;
    # 0.01 here is deliberately coarser/more conservative than that, not a typo.
    "XAU_USD": (Decimal("0.01"), "USD"),
}
"""Tick sizes and quote currencies for the six spec instruments (design spec section 5)."""


@runtime_checkable
class ApiLike(Protocol):
    """Structural mirror of the one `oandapyV20.API` method this module calls."""

    def request(self, endpoint: Any) -> dict[str, Any]: ...


def _parse_oanda_ts(value: str) -> datetime:
    """Parse OANDA's nanosecond RFC3339 (`...000000000Z`) into a tz-aware UTC datetime."""
    match = _TZ_SUFFIX.search(value)
    tz = match.group(1) if match else "Z"
    body = value[: match.start()] if match else value
    if "." in body:
        date_part, frac = body.split(".", 1)
        body = f"{date_part}.{(frac + '000000')[:6]}"
    if tz == "Z":
        tz = "+00:00"
    return datetime.fromisoformat(body + tz).astimezone(UTC)


def _to_oanda_iso(value: datetime) -> str:
    return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def recut_daily(h1_bars: list[Bar], *, now: datetime | None = None) -> list[Bar]:
    """Re-cut 1H bars into 00:00-UTC calendar days.

    A day is emitted only when at least `MIN_HOURS_FOR_DAILY_RECUT` 1H bars back it: open
    of the first bar, close of the last, max high, min low, summed volume, and bid/ask
    close from the last bar. `h1_bars` need not be sorted or complete; the result is
    oldest-first. Every bar in `h1_bars` must share the same `instrument`; a mixed list
    raises `ValueError` (a caller bug — one `recut_daily` call always operates on a single
    instrument's history).

    I1: a day whose close (`day + timedelta(hours=24)`) is still after `now` is still
    forming and is dropped even when it already has enough 1H bars to pass the threshold
    above — `OandaBars.history` calls this with `now=min(query_end, current_time)`, the
    same `min(end, now)` pattern `HyperliquidBars.history` uses for its own forming-candle
    guard. `now=None` (the default, for callers outside `OandaBars.history`) disables this
    check entirely, so a direct `recut_daily(bars)` call keeps its old, unguarded shape.

    FX Sunday sessions (I2): OANDA's FX market reopens at 21:00 UTC on Sunday, so a real
    Sunday typically has only 3 H1 bars (21:00-23:59) — below `MIN_HOURS_FOR_DAILY_RECUT`
    — and is dropped by the day-bar-count check above rather than emitted from a
    near-empty session; Monday's day then starts fresh at 00:00 UTC with its own full 24
    H1 bars. This intentionally aligns Daily bars with Hyperliquid's midnight-UTC perp
    days; analysing the weekend session gap itself is out of scope here.
    """
    if not h1_bars:
        return []
    instrument = h1_bars[0].instrument
    if any(bar.instrument != instrument for bar in h1_bars):
        raise ValueError("recut_daily requires every bar to share the same instrument")
    groups: dict[date, list[Bar]] = {}
    for bar in h1_bars:
        groups.setdefault(bar.ts_open.date(), []).append(bar)

    days: list[Bar] = []
    for day in sorted(groups):
        day_bars = sorted(groups[day], key=lambda b: b.ts_open)
        if len(day_bars) < MIN_HOURS_FOR_DAILY_RECUT:
            continue
        day_start = datetime.combine(day, time.min, tzinfo=UTC)
        if now is not None and day_start + timedelta(hours=24) > now:
            continue
        days.append(
            Bar(
                instrument=instrument,
                tf="1d",
                ts_open=day_start,
                open=day_bars[0].open,
                high=max(b.high for b in day_bars),
                low=min(b.low for b in day_bars),
                close=day_bars[-1].close,
                volume=sum(b.volume for b in day_bars),
                bid_close=day_bars[-1].bid_close,
                ask_close=day_bars[-1].ask_close,
            )
        )
    return days


class OandaBars:
    """`BarSource` over OANDA v20 REST candles.

    `api` is duck-typed against `ApiLike` so tests can pass a fake; when `None`, a real
    `oandapyV20.API` is constructed lazily on first use from `token`/`environment`.
    """

    def __init__(
        self,
        api: ApiLike | None = None,
        *,
        account_id: str | None = None,
        token: str | None = None,
        environment: str = "practice",
        page_size: int = 5000,
        poll_delay_s: float = 5.0,
        now: Callable[[], datetime] | None = None,
        sleep: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        if page_size <= 0:
            raise ValueError(f"page_size must be > 0, got {page_size}")
        self._api = api
        self.account_id = account_id
        self._token = token
        self.environment = environment
        self.page_size = page_size
        self.poll_delay_s = poll_delay_s
        self._now = now or (lambda: datetime.now(UTC))
        self._sleep = sleep or asyncio.sleep

    def _get_api(self) -> ApiLike:
        if self._api is None:
            import oandapyV20  # type: ignore[import-untyped]

            self._api = oandapyV20.API(access_token=self._token, environment=self.environment)
        return self._api

    def _fetch_candles(
        self, symbol: str, granularity: str, start: datetime, end: datetime
    ) -> list[dict[str, Any]]:
        """Page candles with `from`+`count` only (B1).

        OANDA's v20 API rejects a request that combines `from`, `to`, and `count` in the
        same call. Each page therefore carries only `from` (advanced to the last returned
        candle's time plus one granularity step) and `count` (capped at 5000, the venue's
        own per-request limit) — never `to`. Paging stops once a page comes back short
        (fewer than `count` candles) or its last candle's time is at or past `end`. The
        result is filtered to `start <= ts_open < end` client-side, since a page's last
        candle can land on or after `end`.
        """
        from oandapyV20.endpoints.instruments import InstrumentsCandles  # type: ignore[import-untyped]

        api = self._get_api()
        raw: dict[str, dict[str, Any]] = {}
        window_start = start
        step = _TF_DURATION["1h" if granularity == "H1" else "4h"]
        count = min(self.page_size, 5000)
        while window_start < end:
            params = {
                "granularity": granularity,
                "price": "MBA",
                "from": _to_oanda_iso(window_start),
                "count": count,
            }
            endpoint = InstrumentsCandles(instrument=symbol, params=params)
            response = api.request(endpoint)
            candles = response["candles"]
            for candle in candles:
                raw[candle["time"]] = candle
            if not candles:
                break
            last_open = _parse_oanda_ts(candles[-1]["time"])
            if len(candles) < count or last_open >= end:
                break
            next_start = last_open + step
            if next_start <= window_start:
                break
            window_start = next_start
        return [
            candle
            for candle in (raw[key] for key in sorted(raw))
            if start <= _parse_oanda_ts(candle["time"]) < end
        ]

    def history(self, instrument: Instrument, tf: TF, start: datetime, end: datetime) -> list[Bar]:
        """Closed candles with `start <= ts_open < end`, oldest first.

        `tf="1d"` fetches H1 candles and re-cuts them via `recut_daily` rather than asking
        OANDA for `"D"` granularity (see module docstring). `recut_daily` is called with
        `now=min(end, self._now())` (I1) so a day that has not actually closed yet — even
        one already backed by enough 1H bars to pass `recut_daily`'s own threshold — is
        never emitted as a closed Daily bar.
        """
        if tf == "1d":
            h1_bars = self.history(instrument, "1h", start, end)
            cutoff = min(end, self._now())
            return [bar for bar in recut_daily(h1_bars, now=cutoff) if start <= bar.ts_open < end]

        granularity = _GRANULARITY[tf]
        candles = self._fetch_candles(instrument.symbol, granularity, start, end)
        bars: list[Bar] = []
        for candle in candles:
            if not candle.get("complete", False):
                continue
            ts_open = _parse_oanda_ts(candle["time"])
            if not (start <= ts_open < end):
                continue
            mid = candle["mid"]
            bid = candle.get("bid")
            ask = candle.get("ask")
            bars.append(
                Bar(
                    instrument=instrument,
                    tf=tf,
                    ts_open=ts_open,
                    open=float(mid["o"]),
                    high=float(mid["h"]),
                    low=float(mid["l"]),
                    close=float(mid["c"]),
                    volume=float(candle.get("volume", 0)),
                    bid_close=float(bid["c"]) if bid else None,
                    ask_close=float(ask["c"]) if ask else None,
                )
            )
        bars.sort(key=lambda bar: bar.ts_open)
        return bars

    async def stream(self, instrument: Instrument, tf: TF) -> AsyncIterator[Bar]:
        """Yield each closed bar once, indefinitely.

        Same fake-clock design as `HyperliquidBars.stream`: sleeps until the next bar
        boundary plus `poll_delay_s`, re-fetches the last closed bar via `history`, and
        yields it only if newer than the last bar yielded.

        The "next boundary" is `ceil(now / span) * span` (N7): when `now` lands exactly on
        a boundary, that boundary is itself the just-closed bar and must not be skipped in
        favour of the one a full span later.

        Any exception the underlying `ApiLike.request` call raises (a network error, an
        SDK exception, a malformed response) propagates straight out of this generator;
        `stream` itself never retries or reconnects. A caller that wants the stream to
        survive a transient failure must catch around its own iteration and restart
        `stream()` — that reconnection policy belongs to the caller, not here.
        """
        span = _TF_DURATION[tf]
        span_ms = int(span.total_seconds() * 1000)
        last_ts_open: datetime | None = None
        while True:
            now = self._now()
            now_ms = int(now.timestamp() * 1000)
            next_boundary_ms = -(-now_ms // span_ms) * span_ms
            wait_s = (next_boundary_ms - now_ms) / 1000 + self.poll_delay_s
            await self._sleep(wait_s)
            next_boundary = datetime.fromtimestamp(next_boundary_ms / 1000, tz=UTC)
            bars = self.history(instrument, tf, next_boundary - span, next_boundary)
            if bars:
                bar = bars[-1]
                if last_ts_open is None or bar.ts_open > last_ts_open:
                    last_ts_open = bar.ts_open
                    yield bar


def oanda_instrument(name: str) -> Instrument:
    """Build the `Instrument` for one of the six spec OANDA instruments from a static table."""
    if name not in _STATIC_INSTRUMENTS:
        raise KeyError(f"{name!r} is not one of the six spec instruments; use instrument_from_account")
    tick_size, quote_ccy = _STATIC_INSTRUMENTS[name]
    return Instrument(
        venue="oanda",
        symbol=name,
        tick_size=tick_size,
        contract_multiplier=Decimal("1"),
        quote_ccy=quote_ccy,
        session_profile="fx",
    )


def instrument_from_account(api: ApiLike, account_id: str, name: str) -> Instrument:
    """Build an `Instrument` from `AccountInstruments`, deriving `tick_size` from `displayPrecision`.

    Raises `ValueError` (I6) if `name` is not among the account's instruments, rather than
    letting a bare `next()` over an empty match raise an opaque `StopIteration`.
    """
    from oandapyV20.endpoints.accounts import AccountInstruments  # type: ignore[import-untyped]

    endpoint = AccountInstruments(accountID=account_id, params={"instruments": name})
    response = api.request(endpoint)
    entry = next((e for e in response["instruments"] if e["name"] == name), None)
    if entry is None:
        raise ValueError(f"{name!r} not in AccountInstruments for account {account_id}")
    tick_size = Decimal(1).scaleb(-int(entry["displayPrecision"]))
    quote_ccy = name.split("_")[1] if "_" in name else ""
    return Instrument(
        venue="oanda",
        symbol=name,
        tick_size=tick_size,
        contract_multiplier=Decimal("1"),
        quote_ccy=quote_ccy,
        session_profile="fx",
    )
