"""Hyperliquid bar source: candle snapshots via the `hyperliquid-python-sdk` Info client.

`InfoLike` is a structural (duck-typed) mirror of the pieces of `hyperliquid.info.Info`
this module needs, so unit tests can pass a fake without touching the network. When no
`info` is given, `HyperliquidBars` builds a real `Info(base_url, skip_ws=True)` lazily, on
first use, so constructing a `HyperliquidBars()` never requires a connection.

Deviation (⚠ documented for the orchestrator): the spec asks for "90-day median volume" to
pick the extra perp instruments beyond BTC/ETH/SOL, which would need 90 daily candles per
coin. `select_perp_symbols` uses `dayNtlVlm` from `meta_and_asset_ctxs` instead — a
single call, current-day notional volume — as a cheaper proxy. See the handoff.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any, Protocol, runtime_checkable

from swingforge.core.types import TF, Bar, Instrument

__all__ = [
    "DEFAULT_PERPS",
    "HyperliquidBars",
    "InfoLike",
    "closed_only",
    "hl_instrument",
    "hl_instruments",
    "select_perp_symbols",
]

_INTERVAL_MAP: dict[str, str] = {"1h": "1h", "4h": "4h", "1d": "1d"}
"""swingforge TF -> Hyperliquid candle interval string (identity, documented explicitly)."""

_TF_DURATION: dict[str, timedelta] = {
    "1h": timedelta(hours=1),
    "4h": timedelta(hours=4),
    "1d": timedelta(hours=24),
}

DEFAULT_PERPS: dict[str, Decimal] = {
    "BTC": Decimal("1"),
    "ETH": Decimal("0.1"),
    "SOL": Decimal("0.01"),
}
"""Conservative tick sizes for the always-on perps.

Hyperliquid prices carry five significant figures; these ticks are deliberately coarser
than the venue's actual minimum so `round_to_tick` never rounds away real precision. Tick
sizes are configurable per instrument — override with `hl_instrument` directly if a
tighter tick is needed.
"""

_MS_PER_HOUR = 60 * 60 * 1000
DEFAULT_PAGE_MS = 5000 * 4 * _MS_PER_HOUR
"""5000 candles' worth of 4H bars, in milliseconds — Hyperliquid returns at most 5000
candles per `candles_snapshot` call.

This default is sized for 4H bars specifically; `history` never uses it directly for
another timeframe — it caps the per-page window to `min(page_ms, _VENUE_CANDLE_CAP *
span_ms(tf))` (C1), so a 1H or Daily query is paged correctly even though this constant
was tuned for 4H.
"""

_VENUE_CANDLE_CAP = 5000
"""Hyperliquid's own per-`candles_snapshot`-call limit, oldest-first."""


@runtime_checkable
class InfoLike(Protocol):
    """Structural mirror of the `hyperliquid.info.Info` methods this module calls."""

    def candles_snapshot(
        self, name: str, interval: str, startTime: int, endTime: int
    ) -> list[dict[str, Any]]: ...

    def funding_history(
        self, name: str, startTime: int, endTime: int | None = None
    ) -> list[dict[str, Any]]: ...

    def meta_and_asset_ctxs(self) -> Any: ...


def _to_ms(value: datetime) -> int:
    return int(value.timestamp() * 1000)


def _from_ms(value: int) -> datetime:
    return datetime.fromtimestamp(value / 1000, tz=UTC)


def closed_only(candles: list[dict[str, Any]], end_ms: int) -> list[dict[str, Any]]:
    """Drop a candle whose close time `T` is in the future relative to `end_ms`.

    Hyperliquid's `candles_snapshot` can include the still-forming current candle; only
    `T <= end_ms` is a closed candle as of the query's `end`.
    """
    return [c for c in candles if int(c["T"]) <= end_ms]


class HyperliquidBars:
    """`BarSource` over Hyperliquid perp candles.

    `info` is duck-typed against `InfoLike` so tests can pass a fake; when `None`, a real
    `hyperliquid.info.Info(base_url, skip_ws=True)` is constructed lazily on first use.
    """

    def __init__(
        self,
        info: InfoLike | None = None,
        *,
        base_url: str | None = None,
        page_ms: int = DEFAULT_PAGE_MS,
        poll_delay_s: float = 5.0,
        now: Callable[[], datetime] | None = None,
        sleep: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        if page_ms <= 0:
            raise ValueError(f"page_ms must be > 0, got {page_ms}")
        self._info = info
        self._base_url = base_url
        self.page_ms = page_ms
        self.poll_delay_s = poll_delay_s
        self._now = now or (lambda: datetime.now(UTC))
        self._sleep = sleep or asyncio.sleep

    def _get_info(self) -> InfoLike:
        if self._info is None:
            from hyperliquid.info import Info  # type: ignore[import-untyped]

            self._info = Info(self._base_url, skip_ws=True)
        return self._info

    def history(self, instrument: Instrument, tf: TF, start: datetime, end: datetime) -> list[Bar]:
        """Closed candles with `start <= ts_open < end`, oldest first.

        Paginates by time window, capped to `min(page_ms, _VENUE_CANDLE_CAP * span_ms(tf))`
        (C1): `page_ms`'s default is sized for 4H bars, and would ask for far more than
        5000 1H (or Daily) candles in one call otherwise, silently losing whatever the
        venue truncated.

        A page that comes back at the venue's cap is resumed from that page's own last
        candle's open time plus one span (C2) — not blindly advanced to the window's
        requested end — so the remaining candles the venue didn't return this call are
        picked up by the next one instead of being skipped. If a capped page's last
        candle sits more than one span short of the window end it was asked for, and the
        resumed request then comes back completely empty, the gap cannot be resumed and
        this raises `RuntimeError` rather than silently returning a truncated series.

        Candles are de-duplicated on the open-time key `t` so overlapping page windows
        never yield the same bar twice.

        A candle counts as closed only if its close time `T` is at or before
        `min(end, now)` (N2): a query whose `end` reaches into the future must not include
        a candle that has not actually closed yet just because `end` alone would allow it.
        """
        interval = _INTERVAL_MAP[tf]
        info = self._get_info()
        start_ms = _to_ms(start)
        end_ms = _to_ms(end)
        now_ms = _to_ms(self._now())
        closed_end_ms = min(end_ms, now_ms)
        span_ms = int(_TF_DURATION[tf].total_seconds() * 1000)
        page_ms = min(self.page_ms, _VENUE_CANDLE_CAP * span_ms)

        raw: dict[int, dict[str, Any]] = {}
        window_start = start_ms
        gap_pending = False
        while window_start < end_ms:
            window_end = min(window_start + page_ms, end_ms)
            page = info.candles_snapshot(instrument.symbol, interval, window_start, window_end)
            if gap_pending and not page:
                raise RuntimeError(
                    f"Hyperliquid candle paging stalled for {instrument.symbol} {tf}: a page "
                    f"hit the {_VENUE_CANDLE_CAP}-candle cap short of its window end and the "
                    "resumed request came back completely empty, between "
                    f"{_from_ms(window_start).isoformat()} and {_from_ms(end_ms).isoformat()} "
                    "— history may be truncated and cannot be safely resumed"
                )
            gap_pending = False
            for candle in page:
                raw[int(candle["t"])] = candle
            if len(page) >= _VENUE_CANDLE_CAP:
                last_t = max(int(c["t"]) for c in page)
                if window_end - last_t > span_ms:
                    gap_pending = True
                next_start = last_t + span_ms
                if next_start <= window_start:
                    break
                window_start = next_start
            else:
                window_start = window_end

        candles = closed_only([raw[t] for t in sorted(raw)], closed_end_ms)
        bars: list[Bar] = []
        for candle in candles:
            ts_open = _from_ms(int(candle["t"]))
            if not (start <= ts_open < end):
                continue
            bars.append(
                Bar(
                    instrument=instrument,
                    tf=tf,
                    ts_open=ts_open,
                    open=float(candle["o"]),
                    high=float(candle["h"]),
                    low=float(candle["l"]),
                    close=float(candle["c"]),
                    volume=float(candle["v"]),
                )
            )
        return bars

    async def stream(self, instrument: Instrument, tf: TF) -> AsyncIterator[Bar]:
        """Yield each closed bar once, indefinitely.

        Sleeps until the next bar boundary plus `poll_delay_s` (letting the venue finish
        publishing the candle), then re-fetches the last closed bar via `history` and
        yields it only if it is newer than the last bar yielded. `now`/`sleep` are
        injectable so tests can drive this with a fake clock and no real waiting.

        The "next boundary" is `ceil(now / span) * span` (N7): when `now` lands exactly on
        a boundary, that boundary is itself the just-closed bar and must not be skipped in
        favour of the one a full span later.

        Any exception the underlying `InfoLike.candles_snapshot` call raises (a network
        error, an SDK exception, a malformed response) propagates straight out of this
        generator; `stream` itself never retries or reconnects. A caller that wants the
        stream to survive a transient failure must catch around its own iteration and
        restart `stream()` — that reconnection policy belongs to the caller, not here.
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
            next_boundary = _from_ms(next_boundary_ms)
            bars = self.history(instrument, tf, next_boundary - span, next_boundary)
            if bars:
                bar = bars[-1]
                if last_ts_open is None or bar.ts_open > last_ts_open:
                    last_ts_open = bar.ts_open
                    yield bar


def hl_instrument(coin: str, tick_size: Decimal) -> Instrument:
    """Build the `Instrument` for a Hyperliquid perp."""
    return Instrument(
        venue="hyperliquid",
        symbol=coin,
        tick_size=tick_size,
        contract_multiplier=Decimal("1"),
        quote_ccy="USDC",
        session_profile="perp",
    )


def _rank_perps(info: InfoLike, extra: int) -> tuple[list[str], dict[str, dict[str, Any]]]:
    """Shared ranking logic for `select_perp_symbols` and `hl_instruments`: a single
    `meta_and_asset_ctxs()` call, so the two never issue it twice for the same selection.

    Returns the selected symbols plus a per-coin dict merging that coin's `universe` entry
    (`szDecimals`, ...) and asset-ctx entry (`dayNtlVlm`, `markPx`, ...) together, keyed by
    coin name, so callers needing both (e.g. `hl_instruments`'s tick fallback) don't need a
    second pass over the raw API shapes.
    """
    base: Sequence[str] = ("BTC", "ETH", "SOL")
    meta, ctxs = info.meta_and_asset_ctxs()
    universe = meta["universe"]
    combined = {asset["name"]: {**asset, **ctx} for asset, ctx in zip(universe, ctxs, strict=True)}
    ranked = sorted(
        ((float(data["dayNtlVlm"]), name) for name, data in combined.items() if name not in base),
        key=lambda pair: pair[0],
        reverse=True,
    )
    symbols = [*base, *(name for _, name in ranked[:extra])]
    return symbols, combined


def select_perp_symbols(info: InfoLike, extra: int = 5) -> list[str]:
    """BTC, ETH, SOL plus the `extra` highest-`dayNtlVlm` other perps.

    ⚠ Deviation from the spec's "90-day median volume": that needs 90 daily candles per
    coin (one `candles_snapshot` call each). `dayNtlVlm` from a single
    `meta_and_asset_ctxs()` call is used as a cheaper current-day-notional-volume proxy.
    """
    symbols, _combined = _rank_perps(info, extra)
    return symbols


def hl_instruments(
    info: InfoLike, extra: int = 5, ticks: Mapping[str, Decimal] = DEFAULT_PERPS
) -> list[Instrument]:
    """`Instrument`s for `select_perp_symbols(info, extra)` (N3).

    Tick size comes from `ticks.get(coin)` when the coin is listed there (`DEFAULT_PERPS`
    by default) — an explicit entry always takes precedence over the fallback below.

    Otherwise (I4) the tick is the coarser (numerically larger) of two candidates:

    * a 5-significant-figure tick derived from the coin's current `markPx` (from
      `meta_and_asset_ctxs`'s asset ctx): `Decimal(1).scaleb(floor(log10(markPx)) - 4)`;
    * the previous `szDecimals`-derived tick (Hyperliquid's size-decimals field, from the
      `universe` entry): `Decimal(1).scaleb(-(5 - szDecimals))`.

    Using the coarser of the two matters for a coin whose `szDecimals` alone would imply
    an unrealistically fine tick relative to its actual price (e.g. a $60,000 coin with
    `szDecimals=0` would otherwise round to a tick of 0.00001) — the `markPx`-derived tick
    catches that case, while `szDecimals` still dominates for a low-priced coin whose
    5-sig-fig tick would be too fine. This still reproduces `DEFAULT_PERPS` exactly for
    BTC/ETH/SOL at their typical price levels.
    """
    symbols, by_name = _rank_perps(info, extra)
    instruments: list[Instrument] = []
    for coin in symbols:
        tick = ticks.get(coin)
        if tick is None:
            data = by_name[coin]
            sz_decimals = int(data["szDecimals"])
            tick_from_sz = Decimal(1).scaleb(-(5 - sz_decimals))
            mark_px = float(data["markPx"])
            tick_from_mark = (
                Decimal(1).scaleb(math.floor(math.log10(mark_px)) - 4) if mark_px > 0 else Decimal(0)
            )
            tick = max(tick_from_sz, tick_from_mark)
        instruments.append(hl_instrument(coin, tick))
    return instruments
