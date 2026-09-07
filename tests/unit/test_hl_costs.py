"""Unit tests for `swingforge.adapters.hyperliquid.costs`."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from swingforge.adapters.hyperliquid.bars import DEFAULT_PERPS, hl_instrument
from swingforge.adapters.hyperliquid.costs import HyperliquidCosts, load_funding
from swingforge.core.types import Bar, Order, Position

BTC = hl_instrument("BTC", DEFAULT_PERPS["BTC"])


def _bar(ts_open: datetime, tf: str = "4h", close: float = 50_000.0) -> Bar:
    return Bar(
        instrument=BTC,
        tf=tf,
        ts_open=ts_open,
        open=close,
        high=close + 10,
        low=close - 10,
        close=close,
        volume=1.0,
    )


def _order(price: float | None, qty: float = 2.0, direction: int = 1) -> Order:
    return Order(
        id="o1",
        instrument=BTC,
        direction=direction,
        qty=qty,
        kind="limit" if price is not None else "market",
        price=price,
        expires_at_bar=None,
        leg="entry",
    )


def test_entry_commission_and_slippage_arithmetic() -> None:
    costs = HyperliquidCosts(taker_fee_rate=0.0005, slippage_ticks=0.5)
    order = _order(price=50_000.0, qty=2.0)
    bar = _bar(datetime(2024, 1, 1, 1, tzinfo=UTC))

    breakdown = costs.entry(order, bar)

    assert breakdown.commission == pytest.approx(0.0005 * 2.0 * 50_000.0)
    assert breakdown.slippage == pytest.approx(0.5 * float(BTC.tick_size) * 2.0)
    assert breakdown.spread == 0.0
    assert breakdown.funding == 0.0


def test_entry_uses_bar_close_for_market_order_with_no_price() -> None:
    costs = HyperliquidCosts(taker_fee_rate=0.001)
    order = _order(price=None, qty=1.0)
    bar = _bar(datetime(2024, 1, 1, 1, tzinfo=UTC), close=51_000.0)

    breakdown = costs.entry(order, bar)

    assert breakdown.commission == pytest.approx(0.001 * 1.0 * 51_000.0)


def test_carry_accrues_every_hourly_boundary_in_a_4h_bar() -> None:
    """I3 (orchestrator override of design spec section 5): Hyperliquid settles funding
    every hour, not just at the three 8h marks the spec assumed, so a 4H bar accrues 4
    separate hourly charges -- at 05:00, 06:00, 07:00 and 08:00 -- not a single charge at
    08:00. Worked example: BTC long 1.0 @ 60,000, rate 1.25e-5 -> 0.75 per hour, 3.0 for
    the whole 4H bar.
    """
    rate = 1.25e-5
    costs = HyperliquidCosts(funding=[(datetime(2024, 1, 1, 0, tzinfo=UTC), rate)])
    bar = _bar(datetime(2024, 1, 1, 4, tzinfo=UTC), close=60_000.0)  # 04:00-08:00
    position = Position(instrument=BTC, direction=1, qty=1.0, avg_price=59_000.0, stop=58_000.0, target=None)

    funding = costs.carry(position, bar)

    per_hour = rate * 1.0 * 60_000.0
    assert per_hour == pytest.approx(0.75)
    assert funding == pytest.approx(per_hour * 4)
    assert funding == pytest.approx(3.0)
    assert funding > 0  # long pays a positive rate


def test_carry_accrues_a_single_boundary_for_a_1h_bar() -> None:
    """I3: a 1H bar spans exactly one hourly boundary, so it accrues the worked example's
    per-hour charge (0.75) exactly once."""
    rate = 1.25e-5
    costs = HyperliquidCosts(funding=[(datetime(2024, 1, 1, 0, tzinfo=UTC), rate)])
    bar = _bar(datetime(2024, 1, 1, 4, tzinfo=UTC), tf="1h", close=60_000.0)  # 04:00-05:00
    position = Position(instrument=BTC, direction=1, qty=1.0, avg_price=59_000.0, stop=58_000.0, target=None)

    funding = costs.carry(position, bar)

    assert funding == pytest.approx(0.75)


def test_carry_bar_opening_exactly_on_the_hour_does_not_accrue_its_own_open() -> None:
    """I3: funding boundaries are `(ts_open, ts_open + duration]` — strictly greater than
    `ts_open` on the left — so a 1H bar that *opens* exactly on the hour does not itself
    accrue funding for that opening instant; only the boundary at its own close counts.
    """
    open_boundary = datetime(2024, 1, 1, 8, tzinfo=UTC)
    close_boundary = datetime(2024, 1, 1, 9, tzinfo=UTC)
    costs = HyperliquidCosts(funding=[(open_boundary, 0.0001), (close_boundary, 0.0002)])
    bar = _bar(open_boundary, tf="1h", close=50_000.0)  # opens exactly at 08:00, spans 08:00-09:00
    position = Position(instrument=BTC, direction=1, qty=1.0, avg_price=49_000.0, stop=48_000.0, target=None)

    funding = costs.carry(position, bar)

    # Only the 09:00 boundary counts; if 08:00 (the bar's own open) wrongly accrued too,
    # this would be `(0.0001 + 0.0002) * 50_000.0` instead.
    assert funding == pytest.approx(0.0002 * 1.0 * 50_000.0)


def test_carry_short_receives_across_all_boundaries_in_the_bar() -> None:
    rate = 0.0002
    costs = HyperliquidCosts(funding=[(datetime(2024, 1, 1, 0, tzinfo=UTC), rate)])
    bar = _bar(datetime(2024, 1, 1, 4, tzinfo=UTC), close=50_000.0)  # 04:00-08:00, 4 boundaries
    position = Position(instrument=BTC, direction=-1, qty=2.0, avg_price=49_000.0, stop=51_000.0, target=None)

    funding = costs.carry(position, bar)

    assert funding == pytest.approx(-rate * 2.0 * 50_000.0 * 4)
    assert funding < 0  # short is paid


def test_carry_uses_latest_rate_at_or_before_each_boundary() -> None:
    """I3: each of the 4 hourly boundaries in a 4H bar independently looks up the latest
    rate at or before its own timestamp — a rate change partway through the bar's span
    must apply only to the boundaries at or after it, not the whole bar.
    """
    costs = HyperliquidCosts(
        funding=[
            (datetime(2024, 1, 1, 0, tzinfo=UTC), 0.0001),
            (datetime(2024, 1, 1, 6, tzinfo=UTC), 0.0003),
        ]
    )
    bar = _bar(datetime(2024, 1, 1, 4, tzinfo=UTC), close=50_000.0)  # boundaries: 05, 06, 07, 08
    position = Position(instrument=BTC, direction=1, qty=1.0, avg_price=49_000.0, stop=48_000.0, target=None)

    funding = costs.carry(position, bar)

    # 05:00 -> 0.0001 (only the 00:00 entry applies yet); 06:00/07:00/08:00 -> 0.0003.
    expected = (0.0001 + 0.0003 + 0.0003 + 0.0003) * 1.0 * 50_000.0
    assert funding == pytest.approx(expected)


def test_carry_zero_when_all_boundaries_are_before_the_earliest_recorded_funding() -> None:
    costs = HyperliquidCosts(funding=[(datetime(2024, 1, 2, 0, tzinfo=UTC), 0.0002)])
    bar = _bar(datetime(2024, 1, 1, 4, tzinfo=UTC), close=50_000.0)  # boundaries 05-08, day before
    position = Position(instrument=BTC, direction=1, qty=1.0, avg_price=49_000.0, stop=48_000.0, target=None)

    assert costs.carry(position, bar) == 0.0


def test_carry_zero_when_no_funding_history_loaded() -> None:
    costs = HyperliquidCosts()
    bar = _bar(datetime(2024, 1, 1, 4, tzinfo=UTC))
    position = Position(instrument=BTC, direction=1, qty=1.0, avg_price=49_000.0, stop=48_000.0, target=None)

    assert costs.carry(position, bar) == 0.0


class FakeInfo:
    def __init__(self, entries: list[dict]) -> None:
        self.entries = entries

    def funding_history(self, name: str, startTime: int, endTime: int | None = None) -> list[dict]:
        return self.entries


def test_load_funding_shapes_and_sorts_history() -> None:
    entries = [
        {"coin": "BTC", "fundingRate": "0.0002", "premium": "0.0001", "time": 1704099600000},  # later
        {"coin": "BTC", "fundingRate": "0.0001", "premium": "0.0000", "time": 1704067200000},  # earlier
    ]
    info = FakeInfo(entries)

    result = load_funding(info, "BTC", datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 1, 2, tzinfo=UTC))

    assert [rate for _, rate in result] == [0.0001, 0.0002]
    assert result[0][0] < result[1][0]


class PagedFakeInfo:
    """A `funding_history` that behaves like the real endpoint: at most `cap` entries per call,
    oldest first, filtered to `startTime <= time <= endTime`; optionally raising a 429 first."""

    def __init__(self, times: list[int], *, cap: int = 500, rate_limit_first: bool = False) -> None:
        self.times = sorted(times)
        self.cap = cap
        self.calls: list[tuple[int, int | None]] = []
        self._rate_limit_pending = rate_limit_first

    def funding_history(self, name: str, startTime: int, endTime: int | None = None) -> list[dict]:  # noqa: N803
        self.calls.append((startTime, endTime))
        if self._rate_limit_pending:
            self._rate_limit_pending = False
            raise _RateLimited()
        selected = [t for t in self.times if t >= startTime and (endTime is None or t <= endTime)]
        return [
            {"coin": name, "fundingRate": f"{t % 7:.0f}e-5", "premium": "0", "time": t}
            for t in selected[: self.cap]
        ]


class _RateLimited(Exception):
    status_code = 429


_HOUR_MS = 3_600_000
_T0 = 1704067200000  # 2024-01-01T00:00Z


def test_load_funding_pages_past_the_venue_cap() -> None:
    times = [_T0 + i * _HOUR_MS for i in range(1203)]  # ~50 days of hourly funding, 3 pages
    info = PagedFakeInfo(times)

    result = load_funding(
        info, "BTC", datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 3, 1, tzinfo=UTC), sleep=lambda _s: None
    )

    assert len(result) == 1203
    assert [ts for ts, _ in result] == sorted({ts for ts, _ in result})
    assert result[0][0] == datetime(2024, 1, 1, tzinfo=UTC)
    assert result[-1][0] == datetime(2024, 1, 1, tzinfo=UTC) + timedelta(hours=1202)
    assert len(info.calls) == 3
    assert info.calls[1][0] == times[499] + 1  # resumes just past the previous page's last entry


def test_load_funding_single_short_page_makes_one_call() -> None:
    info = PagedFakeInfo([_T0 + i * _HOUR_MS for i in range(10)])

    result = load_funding(
        info, "BTC", datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 1, 2, tzinfo=UTC), sleep=lambda _s: None
    )

    assert len(result) == 10
    assert len(info.calls) == 1


def test_load_funding_backs_off_and_retries_on_rate_limit() -> None:
    info = PagedFakeInfo([_T0 + i * _HOUR_MS for i in range(10)], rate_limit_first=True)
    sleeps: list[float] = []

    result = load_funding(
        info, "BTC", datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 1, 2, tzinfo=UTC), sleep=sleeps.append
    )

    assert len(result) == 10
    assert len(info.calls) == 2
    assert sleeps == [1.0]


def test_load_funding_stops_when_a_full_page_does_not_advance() -> None:
    class StuckInfo:
        def __init__(self) -> None:
            self.calls = 0

        def funding_history(self, name: str, startTime: int, endTime: int | None = None) -> list[dict]:  # noqa: N803
            self.calls += 1
            # A misbehaving venue: the same full page regardless of startTime.
            return [
                {"coin": name, "fundingRate": "0.0001", "premium": "0", "time": _T0 + i * _HOUR_MS}
                for i in range(500)
            ]

    info = StuckInfo()
    result = load_funding(
        info, "BTC", datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 6, 1, tzinfo=UTC), sleep=lambda _s: None
    )

    assert len(result) == 500
    assert info.calls == 2
