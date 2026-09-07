"""Unit tests for `swingforge.adapters.hyperliquid.bars`, all against a fake `Info`."""

from __future__ import annotations

import asyncio
import itertools
import math
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from swingforge.adapters.hyperliquid.bars import (
    DEFAULT_PERPS,
    MAX_RATE_LIMIT_RETRIES,
    HyperliquidBars,
    closed_only,
    hl_instrument,
    hl_instruments,
    hl_instruments_for,
    select_perp_symbols,
    with_rate_limit_backoff,
)

SPAN_MS = 4 * 60 * 60 * 1000
HOUR_MS = 60 * 60 * 1000
DAY_MS = 24 * HOUR_MS
START = datetime(2024, 1, 1, tzinfo=UTC)
BTC = hl_instrument("BTC", DEFAULT_PERPS["BTC"])


def _candle(
    index: int,
    o: float = 100.0,
    h: float = 101.0,
    low: float = 99.0,
    c: float = 100.5,
    v: float = 10.0,
    span_ms: int = SPAN_MS,
) -> dict:
    t_ms = int(START.timestamp() * 1000) + index * span_ms
    return {
        "t": t_ms,
        "T": t_ms + span_ms,
        "s": "BTC",
        "i": "4h",
        "o": str(o),
        "h": str(h),
        "l": str(low),
        "c": str(c),
        "v": str(v),
        "n": "1",
    }


class FakeInfo:
    """Overlap-filters a canned candle list the way the real endpoint would, capping at
    the venue's real 5000-candle-per-call limit (oldest first, like the OANDA fake's own
    `count` cap) -- C1/C2's paging tests exercise the same truncation the production code
    must page and resume through.
    """

    VENUE_CAP = 5000

    def __init__(self, candles: list[dict]) -> None:
        self.candles = candles
        self.calls: list[tuple[int, int]] = []

    def candles_snapshot(self, name: str, interval: str, startTime: int, endTime: int) -> list[dict]:
        self.calls.append((startTime, endTime))
        matches = [c for c in self.candles if c["t"] < endTime and c["T"] > startTime]
        return matches[: self.VENUE_CAP]

    def funding_history(self, name: str, startTime: int, endTime: int | None = None) -> list[dict]:
        return []

    def meta_and_asset_ctxs(self):
        coins = ("BTC", "ETH", "SOL", "ARB", "DOGE", "AVAX", "LINK")
        universe = [{"name": n, "szDecimals": 5} for n in coins]
        vols = (1_000_000, 900_000, 800_000, 50_000, 700_000, 10_000, 600_000)
        mark_pxs = (60_000, 3_000, 150, 0.8, 0.08, 25, 15)
        ctxs = [{"dayNtlVlm": vol, "markPx": px} for vol, px in zip(vols, mark_pxs, strict=True)]
        return {"universe": universe}, ctxs


def test_bars_open_every_4h_utc_with_no_gaps() -> None:
    candles = [_candle(i) for i in range(30)]
    src = HyperliquidBars(info=FakeInfo(candles))
    end = START + timedelta(hours=4 * 30)

    bars = src.history(BTC, "4h", START, end)

    assert len(bars) == 30
    hours = {b.ts_open.hour for b in bars}
    assert hours <= {0, 4, 8, 12, 16, 20}
    for prev, curr in itertools.pairwise(bars):
        assert curr.ts_open - prev.ts_open == timedelta(hours=4)


def test_pagination_stitches_two_pages_without_duplicates() -> None:
    candles = [_candle(i) for i in range(30)]
    # Deliberately not aligned to a candle boundary, so the boundary candle overlaps both
    # page windows and the fake returns it twice - proving `history` de-duplicates on `t`.
    page_ms = int(16.5 * SPAN_MS)
    fake = FakeInfo(candles)
    src = HyperliquidBars(info=fake, page_ms=page_ms)
    end = START + timedelta(hours=4 * 30)

    bars = src.history(BTC, "4h", START, end)

    assert len(fake.calls) >= 2
    assert len(bars) == 30
    ts_opens = [b.ts_open for b in bars]
    assert len(ts_opens) == len(set(ts_opens))


def test_string_values_are_converted_to_float() -> None:
    candles = [_candle(0, o=100.25, h=101.5, low=99.75, c=100.9, v=12.5)]
    src = HyperliquidBars(info=FakeInfo(candles))
    end = START + timedelta(hours=4)

    bars = src.history(BTC, "4h", START, end)

    assert len(bars) == 1
    bar = bars[0]
    assert bar.open == 100.25
    assert bar.high == 101.5
    assert bar.low == 99.75
    assert bar.close == 100.9
    assert bar.volume == 12.5
    assert bar.bid_close is None
    assert bar.ask_close is None


def test_still_forming_candle_is_excluded() -> None:
    candles = [_candle(i) for i in range(30)]
    src = HyperliquidBars(info=FakeInfo(candles))
    # Query end lands mid-candle for the last one: it opened before `end` but has not
    # closed by `end`, so `closed_only` must drop it even though its ts_open is in range.
    end = START + timedelta(hours=4 * 29) + timedelta(hours=2)

    bars = src.history(BTC, "4h", START, end)

    assert len(bars) == 29
    assert bars[-1].ts_open == START + timedelta(hours=4 * 28)


def test_history_excludes_a_bar_that_overlaps_but_opens_before_start() -> None:
    candles = [_candle(i) for i in range(5)]
    src = HyperliquidBars(info=FakeInfo(candles))
    # `query_start` lands mid-candle for index 3 (opens at 12h, closes at 16h): the fake
    # still returns it (its close is after `query_start`), so only the final ts_open
    # filter can drop it.
    query_start = START + timedelta(hours=4 * 3) + timedelta(hours=2)
    end = START + timedelta(hours=4 * 5)

    bars = src.history(BTC, "4h", query_start, end)

    assert [b.ts_open for b in bars] == [START + timedelta(hours=4 * 4)]


def test_closed_only_helper_drops_open_candle() -> None:
    candles = [_candle(0), _candle(1)]
    end_ms = candles[0]["T"]  # exactly the close of the first candle only
    kept = closed_only(candles, end_ms)
    assert [c["t"] for c in kept] == [candles[0]["t"]]


def test_forming_candle_excluded_when_end_is_beyond_a_fake_now() -> None:
    """N2: `closed_only`'s threshold must be `min(end, now)`, not `end` alone — a query
    whose `end` reaches into the future of the injectable `now` must still exclude a
    candle that has not actually closed yet.
    """
    candles = [_candle(i) for i in range(5)]
    fake_now = START + timedelta(hours=4 * 4) + timedelta(hours=1)  # mid-candle for index 4
    src = HyperliquidBars(info=FakeInfo(candles), now=lambda: fake_now)
    end = START + timedelta(hours=4 * 5)  # `end` alone would include candle 4

    bars = src.history(BTC, "4h", START, end)

    assert len(bars) == 4
    assert bars[-1].ts_open == START + timedelta(hours=4 * 3)


def test_forming_candle_included_once_a_fake_now_is_past_its_close() -> None:
    """N2, the flip side: once `now` has passed the candle's close, it counts as closed
    even though `end` sits right at that close time.
    """
    candles = [_candle(i) for i in range(5)]
    fake_now = START + timedelta(hours=4 * 5) + timedelta(hours=1)  # well past candle 4's close
    src = HyperliquidBars(info=FakeInfo(candles), now=lambda: fake_now)
    end = START + timedelta(hours=4 * 5)

    bars = src.history(BTC, "4h", START, end)

    assert len(bars) == 5
    assert bars[-1].ts_open == START + timedelta(hours=4 * 4)


class _FakeClock:
    """A fake clock whose `sleep_fn` advances `now_fn` by the slept duration.

    Lets a single clock drive `stream()` across its own `_now()` call *and* the `_now()`
    call inside `history()` (added for N2) without exhausting a one-shot iterator — the
    second read naturally lands after the first sleep, matching a real clock.
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
    boundary = START + timedelta(hours=4)
    candle = _candle(0)  # opens at START, closes exactly at `boundary`
    fake_info = FakeInfo([candle])
    src = HyperliquidBars(info=fake_info)

    clock = _FakeClock(boundary - timedelta(seconds=1))
    src._now = clock.now_fn
    src._sleep = clock.sleep_fn

    async def _first() -> object:
        gen = src.stream(BTC, "4h")
        return await gen.__anext__()

    bar = asyncio.run(_first())

    assert bar.ts_open == START
    assert clock.sleeps == [1.0 + src.poll_delay_s]


def test_stream_does_not_skip_a_bar_when_now_lands_exactly_on_the_boundary() -> None:
    """N7: when `now` is itself exactly a bar boundary (the bar just closed), `stream`
    must treat that boundary as the "next" one and yield its just-closed bar immediately
    (after `poll_delay_s`), rather than computing the boundary *after* it and waiting a
    full extra span.
    """
    boundary = START + timedelta(hours=4)
    candle = _candle(0)  # opens at START, closes exactly at `boundary`
    fake_info = FakeInfo([candle])
    src = HyperliquidBars(info=fake_info)

    clock = _FakeClock(boundary)  # `now` IS the boundary
    src._now = clock.now_fn
    src._sleep = clock.sleep_fn

    async def _first() -> object:
        gen = src.stream(BTC, "4h")
        return await gen.__anext__()

    bar = asyncio.run(_first())

    assert bar.ts_open == START
    assert clock.sleeps == [src.poll_delay_s]  # no extra full-span wait


def test_stream_yields_two_consecutive_bars_across_loop_iterations() -> None:
    """N5: exercise the `while True` loop-back branch — a second `__anext__()` must yield
    the following bar, not repeat or skip.
    """
    span = timedelta(hours=4)
    candles = [_candle(0), _candle(1)]  # START->+4h, +4h->+8h
    fake_info = FakeInfo(candles)
    src = HyperliquidBars(info=fake_info)

    clock = _FakeClock(START + span - timedelta(seconds=1))
    src._now = clock.now_fn
    src._sleep = clock.sleep_fn

    async def _first_two() -> list[object]:
        gen = src.stream(BTC, "4h")
        first = await gen.__anext__()
        second = await gen.__anext__()
        return [first, second]

    bars = asyncio.run(_first_two())

    assert [b.ts_open for b in bars] == [START, START + span]


def test_hl_instrument_shape() -> None:
    inst = hl_instrument("ETH", Decimal("0.1"))
    assert inst.venue == "hyperliquid"
    assert inst.symbol == "ETH"
    assert inst.tick_size == Decimal("0.1")
    assert inst.contract_multiplier == Decimal("1")
    assert inst.quote_ccy == "USDC"
    assert inst.session_profile == "perp"


def test_select_perp_symbols_ranks_by_day_notional_volume() -> None:
    fake_info = FakeInfo([])
    selected = select_perp_symbols(fake_info, extra=3)
    assert selected[:3] == ["BTC", "ETH", "SOL"]
    # ARB(50k), AVAX(10k) excluded; DOGE(700k), LINK(600k), and one more expected by rank.
    assert selected[3:] == ["DOGE", "LINK", "ARB"]


def test_hl_instruments_uses_default_perps_ticks_for_btc_eth_sol() -> None:
    fake_info = FakeInfo([])
    instruments = hl_instruments(fake_info, extra=3)
    by_symbol = {inst.symbol: inst for inst in instruments}
    assert by_symbol["BTC"].tick_size == DEFAULT_PERPS["BTC"]
    assert by_symbol["ETH"].tick_size == DEFAULT_PERPS["ETH"]
    assert by_symbol["SOL"].tick_size == DEFAULT_PERPS["SOL"]
    for inst in instruments:
        assert inst.venue == "hyperliquid"
        assert inst.contract_multiplier == Decimal("1")
        assert inst.quote_ccy == "USDC"
        assert inst.session_profile == "perp"


def test_hl_instruments_falls_back_to_a_sz_decimals_derived_tick_for_unlisted_perps() -> None:
    """N3: a perp not in `ticks` falls back to `Decimal(1).scaleb(-(5 - szDecimals))`.
    `FakeInfo.meta_and_asset_ctxs` gives every coin `szDecimals=5`, so the fallback is
    `Decimal(1).scaleb(0) == Decimal("1")` for DOGE/LINK/ARB here.
    """
    fake_info = FakeInfo([])
    instruments = hl_instruments(fake_info, extra=3)
    by_symbol = {inst.symbol: inst for inst in instruments}
    assert by_symbol["DOGE"].tick_size == Decimal("1")
    assert by_symbol["LINK"].tick_size == Decimal("1")
    assert by_symbol["ARB"].tick_size == Decimal("1")


def test_hl_instruments_respects_a_custom_ticks_mapping() -> None:
    fake_info = FakeInfo([])
    instruments = hl_instruments(fake_info, extra=3, ticks={"DOGE": Decimal("0.00001")})
    by_symbol = {inst.symbol: inst for inst in instruments}
    assert by_symbol["DOGE"].tick_size == Decimal("0.00001")
    # BTC isn't in this custom mapping, so it too falls back to szDecimals (5 -> tick 1).
    assert by_symbol["BTC"].tick_size == Decimal("1")


def test_hl_instruments_tick_fallback_uses_the_coarser_of_mark_price_and_sz_decimals() -> None:
    """I4: the szDecimals-derived tick alone can be unrealistically fine for a high-priced
    coin (a $60,000 coin with szDecimals=0 would otherwise get a tick of 0.00001); the
    fallback must use the coarser (numerically larger) of that and a 5-sig-fig tick derived
    from `markPx`: `Decimal(1).scaleb(floor(log10(markPx)) - 4)`.
    """

    class _MarkPriceInfo:
        def __init__(self, coins: list[tuple[str, int, float]]) -> None:
            self._coins = coins

        def meta_and_asset_ctxs(self):
            universe = [{"name": n, "szDecimals": sz} for n, sz, _ in self._coins]
            ctxs = [{"dayNtlVlm": 1.0, "markPx": px} for _, _, px in self._coins]
            return {"universe": universe}, ctxs

    info = _MarkPriceInfo([("EXPENSIVE", 0, 60_000.0), ("CHEAP", 0, 0.5), ("MID", 0, 3_200.0)])

    instruments = hl_instruments(info, extra=3)

    by_symbol = {inst.symbol: inst for inst in instruments}
    assert by_symbol["EXPENSIVE"].tick_size == Decimal("1")
    assert by_symbol["CHEAP"].tick_size == Decimal("0.00001")
    assert by_symbol["MID"].tick_size == Decimal("0.1")


def test_hl_instruments_tick_fallback_discriminates_on_sz_decimals() -> None:
    """I5: with `markPx` fixed low enough that its own derived tick never dominates,
    varying `szDecimals` (0, 2, 5) across three fake coins must still yield three distinct
    ticks -- proving the fallback genuinely depends on `szDecimals` and not only on
    `markPx` (a test that used the same `szDecimals` for every coin, as the original I4
    test did, could not catch a fallback that ignored `szDecimals` entirely).
    """

    class _MarkPriceInfo:
        def __init__(self, coins: list[tuple[str, int]]) -> None:
            self._coins = coins

        def meta_and_asset_ctxs(self):
            universe = [{"name": n, "szDecimals": sz} for n, sz in self._coins]
            ctxs = [{"dayNtlVlm": 1.0, "markPx": 1e-8} for _ in self._coins]
            return {"universe": universe}, ctxs

    info = _MarkPriceInfo([("A", 0), ("B", 2), ("C", 5)])

    instruments = hl_instruments(info, extra=3)

    by_symbol = {inst.symbol: inst for inst in instruments}
    assert by_symbol["A"].tick_size == Decimal("0.00001")
    assert by_symbol["B"].tick_size == Decimal("0.001")
    assert by_symbol["C"].tick_size == Decimal("1")
    assert len({by_symbol["A"].tick_size, by_symbol["B"].tick_size, by_symbol["C"].tick_size}) == 3


def test_page_ms_must_be_positive() -> None:
    with pytest.raises(ValueError):
        HyperliquidBars(info=FakeInfo([]), page_ms=0)


def test_history_pages_4_years_of_1h_bars_completely_with_correct_call_count() -> None:
    """C1+C2: `page_ms` must be sized to the timeframe's own span
    (`min(self.page_ms, 5000 * span_ms)`), not left at the 4H-tuned default, and paging
    must resume from the last returned candle's `t` (not blindly jump to the window end)
    once `FakeInfo` enforces the venue's real 5000-candle-per-call cap. Without both
    fixes, a 1H query this long would silently drop most of its history (reviewer's
    `repro.py`).
    """
    n_hours = 4 * 365 * 24 + 24  # ~4 years including a leap day (spec's "~35,064")
    candles = [_candle(i, span_ms=HOUR_MS) for i in range(n_hours)]
    fake = FakeInfo(candles)
    end = START + timedelta(hours=n_hours)
    # Fixed fake clock, well after `end`: keeps this deterministic regardless of the real
    # wall-clock date (this window can otherwise reach into the future relative to it).
    src = HyperliquidBars(info=fake, now=lambda: end + timedelta(days=1))

    bars = src.history(BTC, "1h", START, end)

    assert len(bars) == n_hours
    ts_opens = [b.ts_open for b in bars]
    assert len(ts_opens) == len(set(ts_opens))  # no duplicates
    assert ts_opens == sorted(ts_opens)  # complete, contiguous, oldest-first
    for prev, curr in itertools.pairwise(ts_opens):
        assert curr - prev == timedelta(hours=1)
    expected_calls = math.ceil(n_hours / FakeInfo.VENUE_CAP)
    assert len(fake.calls) == expected_calls


def test_history_pages_1d_bars_without_exceeding_the_5000_day_cap() -> None:
    """C1+C2, the Daily side: `page_ms`'s configured default (sized for 4H bars) is
    already well under `5000 * span_ms("1d")`, so a Daily query never approaches the
    venue's 5000-candle cap in practice -- pin that it still pages and resumes correctly
    over a multi-page range.
    """
    n_days = 1000
    candles = [_candle(i, span_ms=DAY_MS) for i in range(n_days)]
    fake = FakeInfo(candles)
    end = START + timedelta(days=n_days)
    # Fixed fake clock, well after `end`: keeps this deterministic regardless of the real
    # wall-clock date (this window can otherwise reach into the future relative to it).
    src = HyperliquidBars(info=fake, now=lambda: end + timedelta(days=1))

    bars = src.history(BTC, "1d", START, end)

    assert len(bars) == n_days
    ts_opens = [b.ts_open for b in bars]
    assert len(ts_opens) == len(set(ts_opens))
    assert ts_opens == sorted(ts_opens)
    for prev, curr in itertools.pairwise(ts_opens):
        assert curr - prev == timedelta(days=1)
    assert len(fake.calls) >= 2  # DEFAULT_PAGE_MS caps each window under 1000 days
    for call_start, call_end in fake.calls:
        assert call_end - call_start <= 5000 * DAY_MS


def test_history_raises_on_a_capped_page_with_an_unresumable_gap() -> None:
    """C1+C2: a page that hits the venue's 5000-candle cap with its last candle more than
    one span short of the window it was asked for, followed by a resumed request that
    comes back completely empty, means the gap cannot be resumed -- `history` must raise
    loud (`RuntimeError`) instead of silently returning a truncated series.
    """

    class _StalledPagingInfo:
        def __init__(self) -> None:
            self.calls: list[tuple[int, int]] = []

        def candles_snapshot(self, name, interval, startTime, endTime):
            self.calls.append((startTime, endTime))
            if len(self.calls) > 1:
                return []
            # Exactly at the cap (5000 candles), but tightly packed from `startTime` so the
            # last one's `t` sits far short of `endTime` -- more than one span below it.
            tight_step = HOUR_MS // 10
            return [
                {
                    "t": startTime + i * tight_step,
                    "T": startTime + i * tight_step + HOUR_MS,
                    "o": "1",
                    "h": "1",
                    "l": "1",
                    "c": "1",
                    "v": "1",
                }
                for i in range(5000)
            ]

        def funding_history(self, *args, **kwargs):
            return []

        def meta_and_asset_ctxs(self):
            return {"universe": []}, []

    info = _StalledPagingInfo()
    src = HyperliquidBars(info=info)
    end = START + timedelta(hours=20000)

    with pytest.raises(RuntimeError):
        src.history(BTC, "1h", START, end)


def test_history_stops_paging_when_a_capped_page_cannot_advance_past_window_start() -> None:
    """Defensive: if a page hits the venue cap but its own last candle's `t` would resume
    the window at or before the window's own start (a malformed or stale-timestamp
    response), `history` must stop paging rather than loop forever re-requesting the same
    window.
    """

    class _NoProgressInfo:
        def __init__(self) -> None:
            self.calls: list[tuple[int, int]] = []

        def candles_snapshot(self, name, interval, startTime, endTime):
            self.calls.append((startTime, endTime))
            # Always returns the same 5000 stale candles regardless of the window asked
            # for; a naive resume-from-last-candle implementation would spin on this.
            return [
                {"t": i, "T": i + HOUR_MS, "o": "1", "h": "1", "l": "1", "c": "1", "v": "1"}
                for i in range(5000)
            ]

        def funding_history(self, *args, **kwargs):
            return []

        def meta_and_asset_ctxs(self):
            return {"universe": []}, []

    info = _NoProgressInfo()
    src = HyperliquidBars(info=info)
    end = START + timedelta(hours=20000)

    bars = src.history(BTC, "1h", START, end)

    assert len(info.calls) == 1  # stopped after the first page, did not loop forever
    assert bars == []  # the stale candles all fall well before `start` anyway


class _RateLimited(Exception):
    status_code = 429


class _RateLimitingFakeInfo(FakeInfo):
    """`FakeInfo` whose first `candles_snapshot` call answers with an HTTP 429."""

    def __init__(self, candles: list[dict]) -> None:
        super().__init__(candles)
        self._pending = 1

    def candles_snapshot(self, name: str, interval: str, startTime: int, endTime: int) -> list[dict]:
        if self._pending:
            self._pending -= 1
            raise _RateLimited()
        return super().candles_snapshot(name, interval, startTime, endTime)


def test_history_backs_off_and_retries_a_rate_limited_page() -> None:
    candles = [_candle(i) for i in range(30)]
    sleeps: list[float] = []
    src = HyperliquidBars(info=_RateLimitingFakeInfo(candles), rate_limit_sleep=sleeps.append)
    end = START + timedelta(hours=4 * 30)

    bars = src.history(BTC, "4h", START, end)

    assert len(bars) == 30
    assert sleeps == [1.0]


def test_with_rate_limit_backoff_doubles_the_delay_then_gives_up() -> None:
    sleeps: list[float] = []

    def always_limited() -> None:
        raise _RateLimited()

    with pytest.raises(_RateLimited):
        with_rate_limit_backoff(always_limited, sleep=sleeps.append)

    assert sleeps == [2.0**i for i in range(MAX_RATE_LIMIT_RETRIES)]


def test_with_rate_limit_backoff_propagates_other_errors_immediately() -> None:
    sleeps: list[float] = []

    class _Forbidden(Exception):
        status_code = 403

    def forbidden() -> None:
        raise _Forbidden()

    with pytest.raises(_Forbidden):
        with_rate_limit_backoff(forbidden, sleep=sleeps.append)

    assert sleeps == []


def test_hl_instruments_for_derives_ticks_like_hl_instruments() -> None:
    fake = FakeInfo([])
    by_symbol = {inst.symbol: inst for inst in hl_instruments(fake, extra=4)}

    result = hl_instruments_for(fake, ["BTC", "ARB", "DOGE"])

    assert [inst.symbol for inst in result] == ["BTC", "ARB", "DOGE"]
    assert result[0].tick_size == DEFAULT_PERPS["BTC"]
    assert result[1].tick_size == by_symbol["ARB"].tick_size
    assert result[2].tick_size == by_symbol["DOGE"].tick_size


def test_hl_instruments_for_rejects_an_unlisted_symbol_up_front() -> None:
    with pytest.raises(ValueError, match="BOGUS"):
        hl_instruments_for(FakeInfo([]), ["BTC", "BOGUS"])
