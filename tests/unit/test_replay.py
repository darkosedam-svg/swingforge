"""Tests for `swingforge.adapters.replay.ReplaySource`."""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from pydantic import ValidationError

from swingforge.adapters.replay import ReplaySource
from swingforge.adapters.store import Store
from swingforge.core.types import Bar, Instrument

BTC = Instrument(
    venue="hyperliquid",
    symbol="BTC",
    tick_size=Decimal("0.5"),
    contract_multiplier=Decimal("1"),
    quote_ccy="USD",
    session_profile="perp",
)


def _bar(ts_open: datetime, tf: str, low: float = 90.0, high: float = 110.0) -> Bar:
    return Bar(
        instrument=BTC,
        tf=tf,
        ts_open=ts_open,
        open=100.0,
        high=high,
        low=low,
        close=102.0,
        volume=1.0,
    )


def test_history_attaches_subbars_within_window_when_all_four_present() -> None:
    with Store(":memory:") as store:
        ts = datetime(2026, 1, 1, tzinfo=UTC)
        four_h = _bar(ts, "4h")
        one_h = [_bar(ts + timedelta(hours=i), "1h", low=95.0, high=105.0) for i in range(4)]
        store.upsert_bars([four_h, *one_h])

        source = ReplaySource(store)
        got = source.history(BTC, "4h", ts, ts + timedelta(hours=4))

        assert len(got) == 1
        assert len(got[0].subbars) == 4
        for i, sub in enumerate(got[0].subbars):
            assert sub.ts_open == ts + timedelta(hours=i)
            assert ts <= sub.ts_open < ts + timedelta(hours=4)


def test_history_leaves_subbars_empty_on_partial_coverage() -> None:
    with Store(":memory:") as store:
        ts = datetime(2026, 1, 1, tzinfo=UTC)
        four_h = _bar(ts, "4h")
        # Only 3 of the 4 matching 1H bars: hour index 2 is missing.
        one_h = [_bar(ts + timedelta(hours=i), "1h", low=95.0, high=105.0) for i in (0, 1, 3)]
        store.upsert_bars([four_h, *one_h])

        source = ReplaySource(store)
        got = source.history(BTC, "4h", ts, ts + timedelta(hours=4))

        assert len(got) == 1
        assert got[0].subbars == ()


def test_history_query_count_is_two_for_4h(monkeypatch: pytest.MonkeyPatch) -> None:
    """One `Store.bars` call for the 4H bars, one for their 1H subbars -- not one per bar."""
    with Store(":memory:") as store:
        ts = datetime(2026, 1, 1, tzinfo=UTC)
        four_h = [_bar(ts + timedelta(hours=4 * i), "4h") for i in range(5)]
        one_h = [_bar(ts + timedelta(hours=i), "1h", low=95.0, high=105.0) for i in range(20)]
        store.upsert_bars([*four_h, *one_h])

        calls = []
        real_bars = store.bars

        def _counting(*args: object, **kwargs: object) -> list[Bar]:
            calls.append((args, kwargs))
            return real_bars(*args, **kwargs)

        monkeypatch.setattr(store, "bars", _counting)
        source = ReplaySource(store)
        got = source.history(BTC, "4h", ts, ts + timedelta(hours=20))

        assert len(got) == 5
        assert all(len(bar.subbars) == 4 for bar in got)
        assert len(calls) == 2


@pytest.mark.slow
def test_history_attaches_subbars_for_2000_4h_bars_under_2s() -> None:
    with Store(":memory:") as store:
        ts = datetime(2020, 1, 1, tzinfo=UTC)
        n = 2_000
        four_h = [_bar(ts + timedelta(hours=4 * i), "4h") for i in range(n)]
        one_h = [_bar(ts + timedelta(hours=i), "1h", low=95.0, high=105.0) for i in range(4 * n)]
        store.upsert_bars([*four_h, *one_h])

        source = ReplaySource(store)
        start = time.perf_counter()
        got = source.history(BTC, "4h", ts, ts + timedelta(hours=4 * n))
        elapsed = time.perf_counter() - start

        assert len(got) == n
        assert all(len(bar.subbars) == 4 for bar in got)
        assert elapsed < 2.0, f"history('4h') over {n} bars took {elapsed:.2f}s, expected < 2s"


def test_history_raises_on_subbar_that_exceeds_parent_range() -> None:
    """A stored 1H bar whose high exceeds its 4H parent's high must fail loudly.

    `model_copy` (the old implementation) would have silently produced an invalid `Bar`;
    `model_validate` re-runs `Bar._check_subbars` and must raise instead.
    """
    with Store(":memory:") as store:
        ts = datetime(2026, 1, 1, tzinfo=UTC)
        four_h = _bar(ts, "4h", low=90.0, high=110.0)
        bad_hour = _bar(ts, "1h", low=95.0, high=200.0)  # 200 > parent's high of 110
        rest = [_bar(ts + timedelta(hours=i), "1h", low=95.0, high=105.0) for i in (1, 2, 3)]
        store.upsert_bars([four_h, bad_hour, *rest])

        source = ReplaySource(store)
        with pytest.raises(ValidationError, match="subbar high"):
            source.history(BTC, "4h", ts, ts + timedelta(hours=4))


def test_history_non_4h_timeframes_are_plain() -> None:
    with Store(":memory:") as store:
        ts = datetime(2026, 1, 1, tzinfo=UTC)
        store.upsert_bars([_bar(ts, "1d"), _bar(ts, "1h")])
        source = ReplaySource(store)
        assert source.history(BTC, "1d", ts, ts + timedelta(days=1))[0].subbars == ()
        assert source.history(BTC, "1h", ts, ts + timedelta(hours=1))[0].subbars == ()


def _seed_two_days(store: Store, day0: datetime) -> None:
    day1 = day0 + timedelta(hours=24)
    for day in (day0, day1):
        store.upsert_bars([_bar(day, "1d")])
        for h in range(0, 24, 4):
            store.upsert_bars([_bar(day + timedelta(hours=h), "4h")])


def test_merged_orders_by_close_time_with_4h_before_daily_on_ties() -> None:
    with Store(":memory:") as store:
        day0 = datetime(2026, 1, 1, tzinfo=UTC)
        _seed_two_days(store, day0)
        source = ReplaySource(store)

        merged = source.merged(BTC, day0, day0 + timedelta(hours=48))

        assert len(merged) == 14  # 6 4H + 1 Daily, per day, for 2 days

        # Fully explicit expected order: within each day, six 4H bars ascending, then
        # that day's Daily bar (tie on close time, 4H first).
        expected = [(day0 + timedelta(hours=4 * i), "4h") for i in range(6)]
        expected.append((day0, "1d"))
        day1 = day0 + timedelta(hours=24)
        expected += [(day1 + timedelta(hours=4 * i), "4h") for i in range(6)]
        expected.append((day1, "1d"))
        assert [(bar.ts_open, bar.tf) for bar in merged] == expected

        # The Daily bar for day0 comes right after the last 4H bar of day0.
        daily0_index = next(i for i, bar in enumerate(merged) if bar.tf == "1d" and bar.ts_open == day0)
        assert merged[daily0_index - 1].tf == "4h"
        assert merged[daily0_index - 1].ts_open == day0 + timedelta(hours=20)


async def _collect(agen: object) -> list[Bar]:
    return [bar async for bar in agen]  # type: ignore[union-attr]


def test_stream_yields_same_bars_as_history() -> None:
    with Store(":memory:") as store:
        ts = datetime(2026, 1, 1, tzinfo=UTC)
        four_h = [_bar(ts + timedelta(hours=4 * i), "4h") for i in range(3)]
        one_h = [_bar(ts + timedelta(hours=i), "1h", low=95.0, high=105.0) for i in range(12)]
        store.upsert_bars([*four_h, *one_h])

        source = ReplaySource(store)
        expected = source.history(BTC, "4h", ts, ts + timedelta(hours=12))

        streamed = asyncio.run(_collect(source.stream(BTC, "4h")))

        assert streamed == expected
        assert len(streamed) == 3
        assert all(len(bar.subbars) == 4 for bar in streamed)


def test_stream_yields_nothing_when_store_is_empty() -> None:
    with Store(":memory:") as store:
        source = ReplaySource(store)
        streamed = asyncio.run(_collect(source.stream(BTC, "4h")))
        assert streamed == []
