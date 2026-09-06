"""Unit tests for `swingforge.adapters.oanda.bars`, all against a fake `api`."""

from __future__ import annotations

import asyncio
import itertools
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from swingforge.adapters.oanda.bars import (
    MIN_HOURS_FOR_DAILY_RECUT,
    OandaBars,
    _parse_oanda_ts,
    instrument_from_account,
    oanda_instrument,
    recut_daily,
)

EUR_USD = oanda_instrument("EUR_USD")


def _oanda_time(dt: datetime, nanos: bool = True) -> str:
    if nanos:
        return dt.strftime("%Y-%m-%dT%H:%M:%S.000000000Z")
    return dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")


def _h1_candle(
    dt: datetime,
    o: float = 1.1000,
    h: float = 1.1010,
    low: float = 1.0990,
    c: float = 1.1005,
    complete: bool = True,
    volume: int = 100,
) -> dict:
    def _fmt(x: float, delta: float) -> str:
        return f"{x + delta:.5f}"

    return {
        "time": _oanda_time(dt),
        "complete": complete,
        "volume": volume,
        "mid": {"o": _fmt(o, 0), "h": _fmt(h, 0), "l": _fmt(low, 0), "c": _fmt(c, 0)},
        "bid": {"o": _fmt(o, -0.0001), "h": _fmt(h, -0.0001), "l": _fmt(low, -0.0001), "c": _fmt(c, -0.0001)},
        "ask": {"o": _fmt(o, 0.0001), "h": _fmt(h, 0.0001), "l": _fmt(low, 0.0001), "c": _fmt(c, 0.0001)},
    }


class FakeApi:
    """Pages a canned candle list by the endpoint's `from`+`count` params (B1: OANDA's v20
    API rejects a request that combines `from`, `to`, and `count` together, so production
    paging must never send `to` alongside the other two — see the `request` guard below,
    which pins that regression).
    """

    def __init__(self, candles: list[dict]) -> None:
        self.candles = sorted(candles, key=lambda c: _parse_oanda_ts(c["time"]))
        self.calls: list[dict] = []

    def request(self, endpoint) -> dict:
        params = endpoint.params
        self.calls.append(dict(params))
        if "from" in params and "to" in params and "count" in params:
            raise ValueError("OANDA v20 API rejects a request combining from + to + count (B1 regression)")
        start = _parse_oanda_ts(params["from"])
        count = params.get("count", len(self.candles))
        selected = [c for c in self.candles if _parse_oanda_ts(c["time"]) >= start]
        return {"candles": selected[:count]}


def test_h1_bar_shape_bid_ask_populated() -> None:
    start = datetime(2024, 1, 2, 0, tzinfo=UTC)
    candles = [_h1_candle(start, o=1.1, h=1.12, low=1.09, c=1.11)]
    src = OandaBars(api=FakeApi(candles))

    bars = src.history(EUR_USD, "1h", start, start + timedelta(hours=1))

    assert len(bars) == 1
    bar = bars[0]
    assert bar.open == pytest.approx(1.1)
    assert bar.high == pytest.approx(1.12)
    assert bar.low == pytest.approx(1.09)
    assert bar.close == pytest.approx(1.11)
    assert bar.bid_close == pytest.approx(1.11 - 0.0001)
    assert bar.ask_close == pytest.approx(1.11 + 0.0001)


def test_incomplete_candle_is_dropped() -> None:
    start = datetime(2024, 1, 2, 0, tzinfo=UTC)
    candles = [
        _h1_candle(start, complete=True),
        _h1_candle(start + timedelta(hours=1), complete=False),
    ]
    src = OandaBars(api=FakeApi(candles))

    bars = src.history(EUR_USD, "1h", start, start + timedelta(hours=2))

    assert len(bars) == 1
    assert bars[0].ts_open == start


def test_h4_bars_align_to_4h_utc_boundaries() -> None:
    start = datetime(2024, 1, 2, 0, tzinfo=UTC)
    candles = [_h1_candle(start + timedelta(hours=4 * i)) for i in range(6)]
    # Fake the H4 candle times directly: production requests granularity="H4".
    for i, candle in enumerate(candles):
        candle["time"] = _oanda_time(start + timedelta(hours=4 * i))
    src = OandaBars(api=FakeApi(candles))

    bars = src.history(EUR_USD, "4h", start, start + timedelta(hours=24))

    assert [b.ts_open.hour for b in bars] == [0, 4, 8, 12, 16, 20]


class _ReturnsEverythingApi:
    """Ignores from/to/count and always returns the full candle list.

    Used to exercise `history`'s own `start <= ts_open < end` filter independently of
    `_fetch_candles`'s windowing (a real venue would never hand back candles this far
    outside the requested window, but the filter must still guard against it).
    """

    def __init__(self, candles: list[dict]) -> None:
        self.candles = candles

    def request(self, endpoint) -> dict:
        return {"candles": self.candles}


def test_history_excludes_bars_outside_the_requested_range() -> None:
    start = datetime(2024, 1, 2, 0, tzinfo=UTC)
    candles = [_h1_candle(start + timedelta(hours=i)) for i in range(5)]
    src = OandaBars(api=_ReturnsEverythingApi(candles))
    query_start = start + timedelta(hours=3)

    bars = src.history(EUR_USD, "1h", query_start, start + timedelta(hours=5))

    assert [b.ts_open for b in bars] == [query_start, query_start + timedelta(hours=1)]


def test_recut_daily_of_empty_list_is_empty() -> None:
    assert recut_daily([]) == []


def test_get_api_builds_a_real_client_lazily_when_none_given() -> None:
    src = OandaBars(token="dummy-token")
    api = src._get_api()
    import oandapyV20

    assert isinstance(api, oandapyV20.API)
    assert src._get_api() is api  # cached, not rebuilt


def test_timestamp_parsing_nanosecond_rfc3339() -> None:
    ts = _parse_oanda_ts("2024-01-02T03:04:05.123456789Z")
    assert ts.tzinfo is UTC
    assert ts == datetime(2024, 1, 2, 3, 4, 5, 123456, tzinfo=UTC)


def test_timestamp_parsing_offset_form() -> None:
    ts = _parse_oanda_ts("2024-01-02T03:04:05.000000000+00:00")
    assert ts == datetime(2024, 1, 2, 3, 4, 5, tzinfo=UTC)


def test_pagination_stitches_pages_without_duplicates() -> None:
    start = datetime(2024, 1, 2, 0, tzinfo=UTC)
    candles = [_h1_candle(start + timedelta(hours=i)) for i in range(12)]
    fake = FakeApi(candles)
    src = OandaBars(api=fake, page_size=5)

    bars = src.history(EUR_USD, "1h", start, start + timedelta(hours=12))

    assert len(fake.calls) >= 3  # 5 + 5 + 2
    assert len(bars) == 12
    ts_opens = [b.ts_open for b in bars]
    assert len(ts_opens) == len(set(ts_opens))
    assert ts_opens == sorted(ts_opens)
    for prev, curr in itertools.pairwise(ts_opens):
        assert curr - prev == timedelta(hours=1)  # no gaps
    # B1: paging must never combine from+to+count — only from+count.
    for call in fake.calls:
        assert "to" not in call
        assert "from" in call and "count" in call


def test_pagination_never_requests_more_than_5000_candles_per_page() -> None:
    start = datetime(2024, 1, 2, 0, tzinfo=UTC)
    candles = [_h1_candle(start + timedelta(hours=i)) for i in range(3)]
    fake = FakeApi(candles)
    src = OandaBars(api=fake, page_size=10_000)

    bars = src.history(EUR_USD, "1h", start, start + timedelta(hours=3))

    assert len(bars) == 3
    assert fake.calls
    assert all(call["count"] <= 5000 for call in fake.calls)


def test_pagination_stops_once_a_page_reaches_or_passes_end_and_filters_client_side() -> None:
    start = datetime(2024, 1, 2, 0, tzinfo=UTC)
    # 20 hourly candles available from the venue, but only an 8-hour window is requested;
    # `_fetch_candles` must stop paging once a returned page's last candle is >= `end`,
    # and `history` must still drop anything at or after `end` client-side.
    candles = [_h1_candle(start + timedelta(hours=i)) for i in range(20)]
    fake = FakeApi(candles)
    src = OandaBars(api=fake, page_size=5)
    end = start + timedelta(hours=8)

    bars = src.history(EUR_USD, "1h", start, end)

    assert [b.ts_open for b in bars] == [start + timedelta(hours=i) for i in range(8)]
    assert len(fake.calls) <= 3  # did not page through all 20 available hours


def test_fake_api_rejects_a_from_to_count_combination() -> None:
    """Pins the B1 regression directly: mirrors the real v20 API's rejection of a request
    that combines `from`, `to`, and `count`, so this cannot silently return candles again.
    """
    fake = FakeApi([])

    class _Endpoint:
        params = {
            "from": "2024-01-01T00:00:00.000000000Z",
            "to": "2024-01-02T00:00:00.000000000Z",
            "count": 5000,
        }

    with pytest.raises(ValueError):
        fake.request(_Endpoint())


def test_recut_daily_high_low_open_close_volume() -> None:
    day = datetime(2024, 1, 2, tzinfo=UTC)
    day_candles = [
        _h1_candle(
            day + timedelta(hours=h),
            o=1.10,
            h=1.10 + h * 0.001,
            low=1.05 - h * 0.0005,
            c=1.10 + h * 0.0001,
        )
        for h in range(24)
    ]
    src = OandaBars(api=FakeApi(day_candles))
    bars = src.history(EUR_USD, "1h", day, day + timedelta(hours=24))

    daily = recut_daily(bars)

    assert len(daily) == 1
    d = daily[0]
    assert d.ts_open == day
    assert d.open == pytest.approx(1.10)
    assert d.close == pytest.approx(bars[-1].close)
    assert d.high == pytest.approx(max(b.high for b in bars))
    assert d.low == pytest.approx(min(b.low for b in bars))
    assert d.volume == pytest.approx(sum(b.volume for b in bars))
    assert d.bid_close == bars[-1].bid_close
    assert d.ask_close == bars[-1].ask_close


def test_recut_daily_skips_a_day_with_too_few_bars() -> None:
    day = datetime(2024, 1, 2, tzinfo=UTC)
    assert MIN_HOURS_FOR_DAILY_RECUT == 12
    sparse_candles = [_h1_candle(day + timedelta(hours=h)) for h in range(5)]
    src = OandaBars(api=FakeApi(sparse_candles))
    bars = src.history(EUR_USD, "1h", day, day + timedelta(hours=5))

    daily = recut_daily(bars)

    assert daily == []


def test_recut_daily_aggregates_a_day_with_a_missing_hour_above_the_threshold() -> None:
    """N5: a day need not have all 24 hourly bars to be re-cut — only >= the threshold.
    Here hour 5 is missing (23 bars), still well above `MIN_HOURS_FOR_DAILY_RECUT` (12).
    """
    day = datetime(2024, 1, 2, tzinfo=UTC)
    hours = [h for h in range(24) if h != 5]
    day_candles = [_h1_candle(day + timedelta(hours=h)) for h in hours]
    src = OandaBars(api=FakeApi(day_candles))
    bars = src.history(EUR_USD, "1h", day, day + timedelta(hours=24))

    daily = recut_daily(bars)

    assert len(daily) == 1
    assert daily[0].ts_open == day
    assert daily[0].open == pytest.approx(bars[0].open)
    assert daily[0].close == pytest.approx(bars[-1].close)
    assert daily[0].high == pytest.approx(max(b.high for b in bars))
    assert daily[0].low == pytest.approx(min(b.low for b in bars))
    assert daily[0].volume == pytest.approx(sum(b.volume for b in bars))


def test_recut_daily_exact_threshold_of_12_bars_emits() -> None:
    day = datetime(2024, 1, 2, tzinfo=UTC)
    assert MIN_HOURS_FOR_DAILY_RECUT == 12
    twelve_candles = [_h1_candle(day + timedelta(hours=h)) for h in range(12)]
    src = OandaBars(api=FakeApi(twelve_candles))
    bars = src.history(EUR_USD, "1h", day, day + timedelta(hours=12))

    daily = recut_daily(bars)

    assert len(daily) == 1
    assert daily[0].ts_open == day


def test_recut_daily_11_bars_below_threshold_does_not_emit() -> None:
    day = datetime(2024, 1, 2, tzinfo=UTC)
    eleven_candles = [_h1_candle(day + timedelta(hours=h)) for h in range(11)]
    src = OandaBars(api=FakeApi(eleven_candles))
    bars = src.history(EUR_USD, "1h", day, day + timedelta(hours=11))

    daily = recut_daily(bars)

    assert daily == []


def test_history_1d_recuts_from_h1_and_filters_range() -> None:
    day1 = datetime(2024, 1, 2, tzinfo=UTC)
    day2 = day1 + timedelta(days=1)
    candles = [_h1_candle(day1 + timedelta(hours=h)) for h in range(24)]
    candles += [_h1_candle(day2 + timedelta(hours=h)) for h in range(24)]
    src = OandaBars(api=FakeApi(candles))

    bars = src.history(EUR_USD, "1d", day1, day1 + timedelta(days=2))

    assert [b.ts_open for b in bars] == [day1, day2]


class _FakeClock:
    """A fake clock whose `sleep_fn` advances `now_fn` by the slept duration, so a single
    clock instance can drive `stream()` across multiple loop iterations without exhausting
    a one-shot iterator (see the matching helper in `test_hl_bars.py`).
    """

    def __init__(self, start: datetime) -> None:
        self.now = start
        self.sleeps: list[float] = []

    def now_fn(self) -> datetime:
        return self.now

    async def sleep_fn(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += timedelta(seconds=seconds)


def test_stream_yields_exactly_the_new_bar_under_a_fake_clock() -> None:
    start = datetime(2024, 1, 2, 0, tzinfo=UTC)
    boundary = start + timedelta(hours=1)
    candles = [_h1_candle(start)]
    src = OandaBars(api=FakeApi(candles))

    clock = _FakeClock(boundary - timedelta(seconds=2))
    src._now = clock.now_fn
    src._sleep = clock.sleep_fn

    async def _first():
        gen = src.stream(EUR_USD, "1h")
        return await gen.__anext__()

    bar = asyncio.run(_first())

    assert bar.ts_open == start
    assert clock.sleeps == [2.0 + src.poll_delay_s]


def test_stream_does_not_skip_a_bar_when_now_lands_exactly_on_the_boundary() -> None:
    """N7: when `now` is itself exactly a bar boundary (the bar just closed), `stream`
    must treat that boundary as "next" and yield its just-closed bar immediately (after
    `poll_delay_s`), not compute the boundary after it and wait a full extra span.
    """
    start = datetime(2024, 1, 2, 0, tzinfo=UTC)
    boundary = start + timedelta(hours=1)
    candles = [_h1_candle(start)]
    src = OandaBars(api=FakeApi(candles))

    clock = _FakeClock(boundary)  # `now` IS the boundary
    src._now = clock.now_fn
    src._sleep = clock.sleep_fn

    async def _first():
        gen = src.stream(EUR_USD, "1h")
        return await gen.__anext__()

    bar = asyncio.run(_first())

    assert bar.ts_open == start
    assert clock.sleeps == [src.poll_delay_s]  # no extra full-span wait


def test_stream_yields_two_consecutive_bars_across_loop_iterations() -> None:
    """N5: exercise the `while True` loop-back branch for OANDA's `stream` too."""
    start = datetime(2024, 1, 2, 0, tzinfo=UTC)
    span = timedelta(hours=1)
    candles = [_h1_candle(start), _h1_candle(start + span)]
    src = OandaBars(api=FakeApi(candles))

    clock = _FakeClock(start + span - timedelta(seconds=1))
    src._now = clock.now_fn
    src._sleep = clock.sleep_fn

    async def _first_two():
        gen = src.stream(EUR_USD, "1h")
        first = await gen.__anext__()
        second = await gen.__anext__()
        return [first, second]

    bars = asyncio.run(_first_two())

    assert [b.ts_open for b in bars] == [start, start + span]


def test_oanda_instrument_static_table() -> None:
    inst = oanda_instrument("USD_JPY")
    assert inst.venue == "oanda"
    assert inst.tick_size == Decimal("0.001")
    assert inst.quote_ccy == "JPY"
    assert inst.session_profile == "fx"
    assert inst.contract_multiplier == Decimal("1")


def test_oanda_instrument_rejects_unknown_symbol() -> None:
    with pytest.raises(KeyError):
        oanda_instrument("NOT_A_SYMBOL")


class FakeAccountApi:
    def __init__(self, entries: list[dict]) -> None:
        self.entries = entries

    def request(self, endpoint) -> dict:
        return {"instruments": self.entries}


def test_instrument_from_account_derives_tick_from_display_precision() -> None:
    entries = [{"name": "EUR_USD", "displayPrecision": 5, "type": "CURRENCY"}]
    inst = instrument_from_account(FakeAccountApi(entries), "acct-1", "EUR_USD")
    assert inst.tick_size == Decimal("0.00001")
    assert inst.quote_ccy == "USD"


def test_instrument_from_account_raises_on_unknown_instrument_name() -> None:
    """I6: an instrument name absent from `AccountInstruments` must raise a clear
    `ValueError`, not let a bare `next()` over an empty match raise `StopIteration`."""
    entries = [{"name": "EUR_USD", "displayPrecision": 5, "type": "CURRENCY"}]
    with pytest.raises(ValueError, match="NOT_LISTED"):
        instrument_from_account(FakeAccountApi(entries), "acct-1", "NOT_LISTED")


def test_history_1d_drops_a_forming_day_per_injected_now() -> None:
    """I1 (reviewer's repro2): a re-cut day whose 24h close is still in the future
    relative to `now` must never be emitted as a closed Daily bar, even when it already
    has enough 1H bars to pass `MIN_HOURS_FOR_DAILY_RECUT` -- the OANDA daily recut needs
    the same forming-bar guard `HyperliquidBars.history` already has via `min(end, now)`.
    """
    day1 = datetime(2024, 1, 1, tzinfo=UTC)
    day2 = day1 + timedelta(days=1)
    candles = [_h1_candle(day1 + timedelta(hours=h)) for h in range(24)]
    candles += [_h1_candle(day2 + timedelta(hours=h)) for h in range(14)]
    now = day2 + timedelta(hours=14)  # day2 only 00:00-13:59 printed so far
    src = OandaBars(api=FakeApi(candles), now=lambda: now)

    bars = src.history(EUR_USD, "1d", day1, now)

    assert [b.ts_open for b in bars] == [day1]


def test_history_1d_includes_a_day_once_now_passes_its_close() -> None:
    """I1, the flip side: once `now` has passed a day's own 24h close, it is no longer
    forming and is emitted normally, even though it's still bounded by a `now` that is
    itself before the query's `end`.
    """
    day1 = datetime(2024, 1, 1, tzinfo=UTC)
    day2 = day1 + timedelta(days=1)
    candles = [_h1_candle(day1 + timedelta(hours=h)) for h in range(24)]
    candles += [_h1_candle(day2 + timedelta(hours=h)) for h in range(24)]
    now = day2 + timedelta(days=1)  # well past day2's own close
    src = OandaBars(api=FakeApi(candles), now=lambda: now)

    bars = src.history(EUR_USD, "1d", day1, day1 + timedelta(days=2))

    assert [b.ts_open for b in bars] == [day1, day2]


def test_recut_daily_drops_sunday_session_below_threshold_monday_opens_at_midnight() -> None:
    """I2: FX markets reopen Sunday 21:00 UTC, so a real week has only 3 H1 bars for
    Sunday (21:00, 22:00, 23:00) -- below `MIN_HOURS_FOR_DAILY_RECUT` (12) -- so no Sunday
    Daily bar is emitted; Monday's day starts fresh at 00:00 UTC and gets its own full
    Daily bar once 24 H1 bars have accumulated.
    """
    sunday = datetime(2024, 1, 7, tzinfo=UTC)  # a Sunday
    assert sunday.weekday() == 6
    monday = sunday + timedelta(days=1)
    sunday_candles = [_h1_candle(sunday + timedelta(hours=h)) for h in (21, 22, 23)]
    monday_candles = [_h1_candle(monday + timedelta(hours=h)) for h in range(24)]
    src = OandaBars(api=FakeApi(sunday_candles + monday_candles))

    bars = src.history(EUR_USD, "1h", sunday, monday + timedelta(hours=24))
    daily = recut_daily(bars)

    assert [b.ts_open for b in daily] == [monday]
    assert daily[0].ts_open.hour == 0


def test_recut_daily_rejects_mixed_instruments() -> None:
    """Minor: `recut_daily` is meant to operate on one instrument's history; a caller that
    hands it bars from more than one instrument gets a loud `ValueError`, not a Daily bar
    silently built from a blend of two different instruments."""
    day = datetime(2024, 1, 2, tzinfo=UTC)
    candles = [_h1_candle(day + timedelta(hours=h)) for h in range(12)]
    src = OandaBars(api=FakeApi(candles))
    bars = src.history(EUR_USD, "1h", day, day + timedelta(hours=12))
    other = oanda_instrument("USD_JPY")
    mixed = [*bars[:-1], bars[-1].model_copy(update={"instrument": other})]

    with pytest.raises(ValueError, match="same instrument"):
        recut_daily(mixed)


def test_page_size_must_be_positive() -> None:
    with pytest.raises(ValueError):
        OandaBars(api=FakeApi([]), page_size=0)
