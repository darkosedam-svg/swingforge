"""OKX bar source: USDT-margined perpetual swaps, history only.

A **research-only** venue. Hyperliquid serves no more than its newest 5,000 candles per
interval - 27 months of 4H bars - which leaves 15 months out of sample and a tournament that
cannot tell a small edge from none. OKX serves 4H candles back to within weeks of each swap's
listing (BTC from 2019-12-16, ETH from 2019-12-25, their `1Dutc` bars from 2020-01; SOL 2021-01), so the same
strategies can be graded over the four years the design spec asks for. Nothing here can stream:
`swingforge paper` refuses the venue.

What the venue does that this module has to know (read off the live API, 2026-09-20):

* `history-candles` pages **backwards**: up to 300 rows, newest first, strictly older than the
  `after` cursor. `history` starts at `end` and walks back until it is past `start`. The 300
  is what the venue serves, not what it documents (100), so a short page is never read as
  the end of history: only an empty page is;
* it has never omitted a candle (no gap in any timeframe over four years of the five swaps),
  so a gap in what comes back is an anomaly, and `history` refuses it rather than store it;
* the candle that is still forming is served too, with `confirm == "0"`;
* 1H and 4H candles open on the UTC grid, but the plain `1D` candle opens at 16:00 UTC (Hong
  Kong midnight). The rest of the system assumes Daily bars open at 00:00 UTC, so Daily is
  asked for as `1Dutc`, and a Daily row that does not open at midnight is refused rather than
  stored;
* volume comes three ways; `volCcy`, the base-coin amount, is the one comparable across venues.

Instruments are keyed by coin (`okx:BTC`) like Hyperliquid's, which keeps config ids comparable
across the two stores; `inst_id` maps a coin to the venue's `BTC-USDT-SWAP`. Positions are sized
in coins (`contract_multiplier == 1`): the venue's own contract size (`ctVal`, 0.01 BTC) is an
order-entry convention that a price-path backtest has no use for.
"""

from __future__ import annotations

import itertools
from collections.abc import AsyncIterator, Callable, Sequence
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from swingforge.adapters.okx.client import OkxClient, OkxLike
from swingforge.core.types import TF, Bar, Instrument

__all__ = ["DEFAULT_SWAPS", "OkxBars", "inst_id", "okx_instrument", "okx_instruments"]

DEFAULT_SWAPS: tuple[str, ...] = ("BTC", "ETH", "SOL", "ARB", "HYPE")
"""The five instruments the Hyperliquid tournament grades, so the two venues' results line up."""

_BAR: dict[str, str] = {"1h": "1H", "4h": "4H", "1d": "1Dutc"}
"""swingforge TF -> OKX `bar`. `1Dutc`, not `1D`: see the module docstring."""

_TF_DURATION: dict[str, timedelta] = {
    "1h": timedelta(hours=1),
    "4h": timedelta(hours=4),
    "1d": timedelta(hours=24),
}

_PAGE_LIMIT = 300
"""What `history-candles` is asked for per call. Its documentation says 100 and it serves 300;
nothing below depends on which, because a short page is not a stop condition."""

_ROW_FIELDS = 9

_CONFIRMED = "1"


def _to_ms(value: datetime) -> int:
    return int(value.timestamp() * 1000)


def _from_ms(value: int) -> datetime:
    return datetime.fromtimestamp(value / 1000, tz=UTC)


def inst_id(instrument: Instrument) -> str:
    """The venue's id for `instrument`'s USDT-margined perpetual swap."""
    return f"{instrument.symbol}-USDT-SWAP"


def okx_instrument(client: OkxLike, coin: str) -> Instrument:
    """The `Instrument` for `coin`'s USDT swap, with the venue's own tick size."""
    row = client.instrument(f"{coin}-USDT-SWAP")  # an unknown or delisted coin is the venue's 51001
    if row.get("state") != "live":
        raise ValueError(f"OKX swap {coin}-USDT-SWAP is not live (state={row.get('state')!r})")
    return Instrument(
        venue="okx",
        symbol=coin,
        tick_size=Decimal(row["tickSz"]),
        contract_multiplier=Decimal("1"),
        quote_ccy="USDT",
        session_profile="perp",
    )


def okx_instruments(client: OkxLike, coins: Sequence[str] = DEFAULT_SWAPS) -> list[Instrument]:
    return [okx_instrument(client, coin) for coin in coins]


class OkxBars:
    """`BarSource` over OKX swap candles - `history` only.

    `client` is duck-typed against `OkxLike` so tests can pass a fake; when `None`, a real
    `OkxClient` is built lazily, so constructing an `OkxBars()` never needs a connection.
    """

    def __init__(self, client: OkxLike | None = None, *, now: Callable[[], datetime] | None = None) -> None:
        self._client = client
        self._now = now or (lambda: datetime.now(UTC))

    @property
    def client(self) -> OkxLike:
        if self._client is None:
            self._client = OkxClient()
        return self._client

    def history(self, instrument: Instrument, tf: TF, start: datetime, end: datetime) -> list[Bar]:
        """Closed candles with `start <= ts_open < end`, oldest first.

        Walks backwards from `end`: each page's oldest candle is the next cursor, and the walk
        stops once a page reaches `start`, comes back **empty** (the venue has nothing older)
        or makes no progress (a venue ignoring the cursor must not become an endless loop). A
        page merely shorter than asked for is not a stop: read as one, a venue that lowered its
        page size would hand back the newest hundred bars of every window without a murmur.
        Candles are de-duplicated on their open time.

        Raises `ValueError` for a row that is not the nine fields the venue documents or a
        Daily candle off the UTC grid, and `RuntimeError` for a gap inside the returned series
        - the counterpart of `HyperliquidBars.history` refusing to truncate silently.

        A candle counts as closed only if the venue has confirmed it *and* its close time is
        not in the future: a query whose `end` reaches past now must not return the candle
        that is still forming.
        """
        bar = _BAR[tf]
        span = _TF_DURATION[tf]
        start_ms, end_ms = _to_ms(start), _to_ms(end)
        inst = inst_id(instrument)

        raw: dict[int, list[str]] = {}
        cursor = end_ms
        while cursor > start_ms:
            page = self.client.history_candles(inst, bar, after_ms=cursor, limit=_PAGE_LIMIT)
            if not page:
                break
            for row in page:
                if len(row) < _ROW_FIELDS:
                    raise ValueError(f"OKX served a malformed {bar} candle for {inst}: {row!r}")
                raw[int(row[0])] = row
            oldest = min(int(row[0]) for row in page)
            if oldest >= cursor:
                break
            cursor = oldest

        now = self._now()
        bars: list[Bar] = []
        for ts_ms in sorted(raw):
            row = raw[ts_ms]
            ts_open = _from_ms(ts_ms)
            if not (start <= ts_open < end) or row[8] != _CONFIRMED or ts_open + span > now:
                continue
            if tf == "1d" and (ts_open.hour, ts_open.minute) != (0, 0):
                raise ValueError(
                    f"OKX served a Daily candle for {inst} opening at {ts_open.isoformat()}, not at "
                    "00:00 UTC; Daily bars must be asked for as `1Dutc`"
                )
            bars.append(
                Bar(
                    instrument=instrument,
                    tf=tf,
                    ts_open=ts_open,
                    open=float(row[1]),
                    high=float(row[2]),
                    low=float(row[3]),
                    close=float(row[4]),
                    volume=float(row[6]),
                )
            )
        for previous, current in itertools.pairwise(bars):
            if current.ts_open - previous.ts_open != span:
                raise RuntimeError(
                    f"OKX {bar} history for {inst} has a gap between {previous.ts_open.isoformat()} and "
                    f"{current.ts_open.isoformat()}; the venue has not omitted a candle before, so the "
                    "series is refused rather than stored"
                )
        return bars

    async def stream(self, instrument: Instrument, tf: TF) -> AsyncIterator[Bar]:
        """Not offered: OKX is a research-only venue (history for the tournament, no paper)."""
        raise NotImplementedError("okx is a research-only venue: it serves history, not a live bar stream")
        yield  # pragma: no cover - makes this an async generator, as `BarSource.stream` is
