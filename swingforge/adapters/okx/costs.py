"""OKX cost model: taker commission, tick slippage, and eight-hourly funding carry.

Funding settles at 00:00, 08:00 and 16:00 UTC on the swaps this adapter trades, and `carry`
charges a position once for every settlement a bar's span crosses, priced at the bar's close
(the same proxy for the mark price the Hyperliquid model uses).

⚠ **The venue serves only about three months of funding history** (`funding-rate-history`
returns nothing older; verified 2026-09-20), and the backtest runs over four years. Every
settlement before the first recorded one is therefore charged `DEFAULT_FUNDING_RATE` - the
venue's own baseline (its `interestRate`) of 0.01% per eight hours, which longs pay and shorts
receive. That is an assumption, not data, and what matters is not the charge but the error:

* the charge itself is small - a three-day hold crosses nine settlements, 0.09% of notional
  against a stop a few percent away, roughly 0.03R;
* the error is not bounded by it. The venue caps funding at 0.375% per settlement for BTC
  and up to 1% for the alts; a rally regime of 0.03% per settlement costs a long about 0.06R
  more than is charged here - a quarter of the +0.2R the tournament is trying to resolve, and
  in one direction, because funding is highest exactly when longs are winning;
* gate rule 6 does not cover it: it multiplies the *assumed* charge by 1.5 (0.01% to
  0.015%), which cannot span a real 0.05%;
* over the 95 days the venue does serve (2026-06 to 2026-09) the baseline *over*-charges a
  long: mean settled rate 0.003% to 0.005% across the five swaps, and 0.01% was the observed
  maximum for four of them. A short is over-credited by the same amount.

So a result on this venue that holds by less than about 0.1R per trade is unresolved until
real funding replaces the assumption. The venue publishes it, as one small zip per swap per
day (`https://www.okx.com/cdn/okex/traderecords/swaprate/daily/YYYYMMDD/
{instId}-swaprate-YYYY-MM-DD.zip`); loading four years of those is the obvious follow-up.
"""

from __future__ import annotations

import bisect
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

from swingforge.adapters.okx.client import OkxLike
from swingforge.core.types import Bar, CostBreakdown, Order, Position

__all__ = ["DEFAULT_FUNDING_RATE", "FUNDING_INTERVAL", "OkxCosts", "load_funding"]

FUNDING_INTERVAL = timedelta(hours=8)

DEFAULT_FUNDING_RATE = 0.0001
"""0.01% per settlement: the venue's baseline, charged where no funding history exists."""

_TAKER_FEE_RATE = 0.0005
"""The venue's regular-tier taker fee on USDT swaps (maker is 0.02%). Entries are limit orders,
but every fill is charged as a taker, as on Hyperliquid: the pessimistic side."""

_TF_DURATION: dict[str, timedelta] = {
    "1h": timedelta(hours=1),
    "4h": timedelta(hours=4),
    "1d": timedelta(hours=24),
}

_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
_MS_PER_HOUR = 3_600_000
_FUNDING_PAGE_LIMIT = 100


def _to_ms(value: datetime) -> int:
    return int(value.timestamp() * 1000)


def _funding_boundaries(ts_open: datetime, duration: timedelta) -> list[datetime]:
    """Every settlement on the eight-hour grid in `(ts_open, ts_open + duration]`.

    A bar's own opening instant never counts, so a 4H bar crosses one settlement or none and a
    Daily bar crosses three.
    """
    end = ts_open + duration
    intervals = (ts_open - _EPOCH) // FUNDING_INTERVAL
    boundary = _EPOCH + (intervals + 1) * FUNDING_INTERVAL
    boundaries = []
    while boundary <= end:
        boundaries.append(boundary)
        boundary += FUNDING_INTERVAL
    return boundaries


class OkxCosts:
    """Entry costs (commission + tick slippage) and eight-hourly funding carry."""

    def __init__(
        self,
        taker_fee_rate: float = _TAKER_FEE_RATE,
        slippage_ticks: float = 0.5,
        funding: Sequence[tuple[datetime, float]] = (),
        default_funding_rate: float = DEFAULT_FUNDING_RATE,
    ) -> None:
        self.taker_fee_rate = taker_fee_rate
        self.slippage_ticks = slippage_ticks
        self.default_funding_rate = default_funding_rate
        self._funding = sorted(funding, key=lambda entry: entry[0])
        self._funding_times = [ts for ts, _ in self._funding]

    def entry(self, order: Order, bar: Bar) -> CostBreakdown:
        price = order.price if order.price is not None else bar.close
        commission = self.taker_fee_rate * order.qty * price
        slippage = self.slippage_ticks * float(order.instrument.tick_size) * order.qty
        return CostBreakdown(spread=0.0, commission=commission, funding=0.0, slippage=slippage)

    def _rate_at(self, boundary: datetime) -> float:
        """The latest recorded rate at or before `boundary`; the baseline before any record."""
        idx = bisect.bisect_right(self._funding_times, boundary) - 1
        return self.default_funding_rate if idx < 0 else self._funding[idx][1]

    def carry(self, position: Position, bar: Bar) -> float:
        """Funding over every settlement in `bar`'s span: a long pays a positive rate, a short
        receives it (the sign follows `position.direction`)."""
        total = 0.0
        for boundary in _funding_boundaries(bar.ts_open, _TF_DURATION[bar.tf]):
            total += self._rate_at(boundary) * position.qty * bar.close * position.direction
        return total


def load_funding(client: OkxLike, inst: str, start: datetime, end: datetime) -> list[tuple[datetime, float]]:
    """Every settlement the venue still serves in `[start, end]` as `(timestamp, rate)`, oldest first.

    Pages backwards from `end` (the cursor is exclusive, so it starts one millisecond past it)
    until a page reaches `start`, comes back empty or makes no progress. `realizedRate` - what
    was actually settled - is preferred over `fundingRate`, the prediction it settled from.
    Timestamps are floored to the hour, as `OkxCosts` looks rates up on exact boundaries.
    """
    start_ms, end_ms = _to_ms(start), _to_ms(end)
    raw: dict[int, float] = {}
    cursor = end_ms + 1
    while cursor > start_ms:
        page = client.funding_rate_history(inst, after_ms=cursor, limit=_FUNDING_PAGE_LIMIT)
        if not page:
            break
        for row in page:
            ts_ms = int(row["fundingTime"])
            raw[ts_ms - ts_ms % _MS_PER_HOUR] = float(row.get("realizedRate") or row["fundingRate"])
        oldest = min(int(row["fundingTime"]) for row in page)
        if oldest >= cursor:
            break
        cursor = oldest
    return sorted(
        (datetime.fromtimestamp(ts_ms / 1000, tz=UTC), rate)
        for ts_ms, rate in raw.items()
        if start_ms <= ts_ms <= end_ms
    )
