"""Hyperliquid cost model: taker commission, tick slippage, and hourly funding carry.

⚠ Orchestrator override of design spec section 5 (I3): the spec's cost paragraph assumed
funding settles only at the 00:00/08:00/16:00 UTC boundaries (every 8h), but Hyperliquid
actually settles funding every hour, and `funding_history`'s `fundingRate` is already the
*hourly* rate (not an 8h rate) — charging a position only at the three 8h marks would
understate its carry by roughly 8x. `carry` accrues once for every hourly boundary a bar's
span crosses: 1 boundary for a 1H bar, 4 for a 4H bar, 24 for a Daily bar.

N4 (documented, not implemented): `carry` prices every funding boundary at `bar.close` —
the fill bar's closing price is used as a proxy for the mark price actually in effect at
the boundary instant, rather than fetching a mark price specifically at that timestamp.
This is a simplification; a boundary that lands mid-bar is priced off the bar's close, not
its own instantaneous mark price.
"""

from __future__ import annotations

import bisect
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

from swingforge.adapters.hyperliquid.bars import InfoLike
from swingforge.core.types import Bar, CostBreakdown, Order, Position

__all__ = ["FUNDING_INTERVAL", "HyperliquidCosts", "load_funding"]

FUNDING_INTERVAL = timedelta(hours=1)
"""Hyperliquid's actual funding-settlement cadence (I3) — hourly, not the spec's assumed
8h. Kept as a named constant for the same reason the old 8h value was: so the grid spacing
used by `_funding_boundaries` is documented in one place rather than a bare literal."""

_TF_DURATION: dict[str, timedelta] = {
    "1h": timedelta(hours=1),
    "4h": timedelta(hours=4),
    "1d": timedelta(hours=24),
}

_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


def _to_ms(value: datetime) -> int:
    return int(value.timestamp() * 1000)


def _funding_boundaries(ts_open: datetime, duration: timedelta) -> list[datetime]:
    """Every hourly-grid boundary in `(ts_open, ts_open + duration]` (I3).

    A bar's own opening instant never counts as one of its boundaries, even when `ts_open`
    itself lands exactly on the hour — only a boundary strictly after it does. Since every
    bar spans at least one full hour (`_TF_DURATION`'s shortest entry is 1h), this always
    returns at least one boundary: 1 for a 1H bar, 4 for a 4H bar, 24 for a Daily bar.
    """
    end = ts_open + duration
    hours_since_epoch = (ts_open - _EPOCH).total_seconds() / 3600
    next_index = int(hours_since_epoch // 1) + 1
    boundary = _EPOCH + timedelta(hours=next_index)
    boundaries = []
    while boundary <= end:
        boundaries.append(boundary)
        boundary += FUNDING_INTERVAL
    return boundaries


class HyperliquidCosts:
    """Entry costs (commission + tick slippage) and hourly funding carry (I3)."""

    def __init__(
        self,
        taker_fee_rate: float = 0.00045,
        slippage_ticks: float = 0.5,
        funding: Sequence[tuple[datetime, float]] = (),
    ) -> None:
        self.taker_fee_rate = taker_fee_rate
        self.slippage_ticks = slippage_ticks
        self._funding = sorted(funding, key=lambda entry: entry[0])
        self._funding_times = [ts for ts, _ in self._funding]

    def entry(self, order: Order, bar: Bar) -> CostBreakdown:
        price = order.price if order.price is not None else bar.close
        commission = self.taker_fee_rate * order.qty * price
        slippage = self.slippage_ticks * float(order.instrument.tick_size) * order.qty
        return CostBreakdown(spread=0.0, commission=commission, funding=0.0, slippage=slippage)

    def _rate_at(self, boundary: datetime) -> float:
        if not self._funding:
            return 0.0
        idx = bisect.bisect_right(self._funding_times, boundary) - 1
        if idx < 0:
            return 0.0
        return self._funding[idx][1]

    def carry(self, position: Position, bar: Bar) -> float:
        """Sum funding charged over every hourly boundary in `bar`'s span (I3: Hyperliquid
        settles funding hourly, not every 8h — see the module docstring).

        Each boundary is priced at `bar.close` (N4, a proxy for the mark price actually in
        effect at that instant — see the module docstring), using the latest funding rate
        recorded at or before that boundary; a long pays a positive rate, a short receives
        it (sign follows `position.direction`).
        """
        duration = _TF_DURATION[bar.tf]
        total = 0.0
        for boundary in _funding_boundaries(bar.ts_open, duration):
            rate = self._rate_at(boundary)
            total += rate * position.qty * bar.close * position.direction
        return total


def load_funding(info: InfoLike, coin: str, start: datetime, end: datetime) -> list[tuple[datetime, float]]:
    """Fetch and shape `info.funding_history` into `(timestamp, rate)` pairs, oldest first."""
    raw = info.funding_history(coin, _to_ms(start), _to_ms(end))
    entries = [
        (datetime.fromtimestamp(int(entry["time"]) / 1000, tz=UTC), float(entry["fundingRate"]))
        for entry in raw
    ]
    entries.sort(key=lambda entry: entry[0])
    return entries
