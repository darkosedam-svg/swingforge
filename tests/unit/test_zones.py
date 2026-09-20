"""Unit tests for `strategies/zones.py` — Daily supply/demand zone entries."""

from __future__ import annotations

import math
import time
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from swingforge.core.context import Context
from swingforge.core.types import Bar, Instrument
from swingforge.strategies.zones import _MAX_ZONES, Zones, _Zone
from tests.unit.perf_budget import load_factor, scaled_budget

PERP = Instrument(
    venue="hyperliquid",
    symbol="BTC",
    tick_size=Decimal("0.01"),
    contract_multiplier=Decimal("1"),
    quote_ccy="USD",
    session_profile="perp",
)
ETH_PERP = Instrument(
    venue="hyperliquid",
    symbol="ETH",
    tick_size=Decimal("0.01"),
    contract_multiplier=Decimal("1"),
    quote_ccy="USD",
    session_profile="perp",
)
FX = Instrument(
    venue="oanda",
    symbol="EUR_USD",
    tick_size=Decimal("0.0001"),
    contract_multiplier=Decimal("1"),
    quote_ccy="USD",
    session_profile="fx",
)
FX_JPY = Instrument(
    venue="oanda",
    symbol="USD_JPY",
    tick_size=Decimal("0.001"),
    contract_multiplier=Decimal("1"),
    quote_ccy="JPY",
    session_profile="fx",
)
START = datetime(2026, 1, 1, tzinfo=UTC)


def _daily(
    index: int, open_: float, high: float, low: float, close: float, instrument: Instrument = PERP
) -> Bar:
    return Bar(
        instrument=instrument,
        tf="1d",
        ts_open=START + timedelta(hours=24 * index),
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=1.0,
    )


def _flat_daily(n: int, close: float = 100.0, rng: float = 2.0, start: int = 0) -> list[Bar]:
    return [_daily(start + i, close, close + rng / 2, close - rng / 2, close) for i in range(n)]


H4_START = START + timedelta(days=17)
"""Where the 4H fixtures begin: the instant the impulse fixtures' last Daily bar (day 16)
closes. A zone can only be touched by a 4H bar that opened at or after its Daily bar
closed - a 4H bar dated before the Daily bars that spawned the zone could never touch it.
A test whose Daily series runs past day 16 must offset its 4H indices accordingly."""


def _h4(index: int, close: float, instrument: Instrument = PERP) -> Bar:
    return Bar(
        instrument=instrument,
        tf="4h",
        ts_open=H4_START + timedelta(hours=4 * index),
        open=close,
        high=close,
        low=close,
        close=close,
        volume=1.0,
    )


def _demand_impulse_context() -> Context:
    """15 flat bars (ATR seed to 2.0) + a bearish base candle + a bullish impulse bar.

    Bar 15 (open=100, close=98, high=101, low=97) is the last opposing (bearish) candle
    before bar 16's close-to-close jump of +12, which comfortably clears
    `2 * ATR(14)` (ATR is a little over 2 by bar 16) within a single bar (<= impulse_bars).
    Expect exactly one demand zone at [97, 101].
    """
    ctx = Context(PERP)
    for bar in _flat_daily(15):  # indices 0..14
        ctx.push(bar)
    ctx.push(_daily(15, open_=100.0, high=101.0, low=97.0, close=98.0))
    ctx.push(_daily(16, open_=98.0, high=111.0, low=97.0, close=110.0))
    return ctx


def test_impulse_detection_finds_the_expected_demand_zone() -> None:
    ctx = _demand_impulse_context()
    strategy = Zones()
    ctx.push(_h4(0, close=200.0))  # outside the zone: no signal, just triggers detection
    assert strategy.on_bar(ctx) is None
    assert len(strategy._zones) == 1
    zone = strategy._zones[0]
    assert zone.direction == 1
    assert zone.low == pytest.approx(97.0)
    assert zone.high == pytest.approx(101.0)
    assert zone.fresh is True


def test_first_touch_inside_zone_signals_at_midpoint() -> None:
    ctx = _demand_impulse_context()
    strategy = Zones()
    ctx.push(_h4(0, close=200.0))  # detect the zone, well outside it
    strategy.on_bar(ctx)
    zone = strategy._zones[0]
    atr = ctx.atr("1d")

    ctx.push(_h4(1, close=99.0))  # inside [97, 101]
    signal = strategy.on_bar(ctx)

    assert signal is not None
    assert signal.direction == 1
    assert signal.tag == "zone_demand"
    mid = (zone.low + zone.high) / 2
    assert signal.entry == pytest.approx(mid)
    expected_stop = zone.low - 0.25 * atr
    assert signal.stop == pytest.approx(expected_stop, abs=0.01)
    assert signal.expires_in_bars == 3
    assert zone.fresh is False


def test_second_touch_of_the_same_zone_returns_none() -> None:
    ctx = _demand_impulse_context()
    strategy = Zones()
    ctx.push(_h4(0, close=200.0))
    strategy.on_bar(ctx)
    ctx.push(_h4(1, close=99.0))
    first = strategy.on_bar(ctx)
    assert first is not None

    ctx.push(_h4(2, close=99.5))  # inside the same zone again
    second = strategy.on_bar(ctx)
    assert second is None


def test_touch_without_close_inside_still_consumes_freshness() -> None:
    ctx = _demand_impulse_context()
    strategy = Zones()
    ctx.push(_h4(0, close=200.0))
    strategy.on_bar(ctx)
    zone = strategy._zones[0]
    assert zone.fresh is True

    # This bar's range overlaps [97, 101] (low=96 < 101, high=98 > 97) but closes at 98,
    # which IS inside the zone actually - use a bar that overlaps but closes outside instead.
    overlapping_close_outside = Bar(
        instrument=PERP,
        tf="4h",
        ts_open=H4_START + timedelta(hours=4),
        open=105.0,
        high=105.0,
        low=96.0,
        close=105.0,
        volume=1.0,
    )
    ctx.push(overlapping_close_outside)
    signal = strategy.on_bar(ctx)
    assert signal is None
    assert zone.fresh is False

    # Even though the next bar closes squarely inside the zone, it is already stale.
    ctx.push(_h4(2, close=99.0))
    assert strategy.on_bar(ctx) is None


def test_supply_zone_signals_short() -> None:
    ctx = Context(PERP)
    for bar in _flat_daily(15):
        ctx.push(bar)
    # A bullish base candle before a bearish impulse -> supply zone.
    ctx.push(_daily(15, open_=98.0, high=101.0, low=97.0, close=100.0))
    ctx.push(_daily(16, open_=100.0, high=101.0, low=88.0, close=89.0))
    strategy = Zones()
    ctx.push(_h4(0, close=200.0))
    strategy.on_bar(ctx)
    assert len(strategy._zones) == 1
    zone = strategy._zones[0]
    assert zone.direction == -1
    atr = ctx.atr("1d")

    ctx.push(_h4(1, close=99.0))  # inside [97, 101]
    signal = strategy.on_bar(ctx)
    assert signal is not None
    assert signal.direction == -1
    assert signal.tag == "zone_supply"
    expected_stop = zone.high + 0.25 * atr
    assert signal.stop == pytest.approx(expected_stop, abs=0.01)


def test_no_impulse_in_a_flat_series_creates_no_zones() -> None:
    ctx = Context(PERP)
    for bar in _flat_daily(40):
        ctx.push(bar)
    strategy = Zones()
    ctx.push(_h4(0, close=100.0))
    assert strategy.on_bar(ctx) is None
    assert strategy._zones == []


def test_rejected_signals_counter_increments_when_stop_collapses_onto_entry() -> None:
    """A zero-width zone plus a flat (ATR == 0) Daily series makes stop == entry, which the
    Signal validator rejects since the stop is no longer strictly on the wrong side."""
    ctx = Context(PERP)
    for bar in _flat_daily(20, rng=0.0):
        ctx.push(bar)
    strategy = Zones()
    zone = _Zone(direction=1, low=100.0, high=100.0, born_ts=START, source_ts=START)
    assert strategy.rejected_signals == 0
    assert strategy._build_signal(zone, ctx) is None
    assert strategy.rejected_signals == 1

    strategy.reset()
    assert strategy.rejected_signals == 0


def test_missing_atr_blocks_signal_construction() -> None:
    # Only 5 daily bars: ctx.atr("1d") is NaN, so a zone touch cannot size a stop.
    ctx = Context(PERP)
    for bar in _flat_daily(5):
        ctx.push(bar)
    strategy = Zones()
    zone = _Zone(direction=1, low=97.0, high=101.0, born_ts=START, source_ts=START)
    assert strategy._build_signal(zone, ctx) is None


def test_round_number_filter_fx_blocks_and_allows() -> None:
    assert Zones._passes_round_filter(1.2003, FX) is True
    assert Zones._passes_round_filter(1.2031, FX) is False


def test_round_number_filter_fx_jpy_scales_step_and_tolerance_by_100() -> None:
    """JPY quotes scale both the step and the tolerance x100 vs. a non-JPY fx pair."""
    assert Zones._passes_round_filter(150.00, FX_JPY) is True
    assert Zones._passes_round_filter(150.31, FX_JPY) is False


def test_round_number_filter_perp_blocks_and_allows() -> None:
    # BTC ~64,200: step = 10**(floor(log10(64200)) - 1) = 1_000; round levels are multiples
    # of 1_000 and 500; tolerance is 0.1% of price.
    assert Zones._passes_round_filter(64_200.0, PERP) is False  # 200 from 64,000 > 64.2 tol
    assert Zones._passes_round_filter(64_050.0, PERP) is True  # 50 from 64,000 <= 64.05 tol
    assert Zones._passes_round_filter(64_480.0, PERP) is True  # 20 from 64,500 <= 64.48 tol


def test_round_number_filter_perp_rescales_per_symbol_price_scale() -> None:
    # ETH ~3,200: step = 10**(floor(log10(3215)) - 1) = 100; round levels are multiples of
    # 100 and 50.
    assert Zones._passes_round_filter(3_215.0, ETH_PERP) is False  # 15 from 3,200 > 3.215 tol
    assert Zones._passes_round_filter(3_203.0, ETH_PERP) is True  # 3 from 3,200 <= 3.203 tol


def test_round_number_filter_gate_blocks_a_signal_end_to_end() -> None:
    # Zone [96.7, 101.0] -> mid 98.85: the nearest perp round level (granularity step/2 =
    # 0.5 at this price scale) is 99.0, 0.15 away vs. a 0.1%-of-price tolerance of ~0.099.
    ctx = Context(PERP)
    for bar in _flat_daily(15):
        ctx.push(bar)
    ctx.push(_daily(15, open_=100.0, high=101.0, low=96.7, close=98.0))
    ctx.push(_daily(16, open_=98.0, high=111.0, low=96.7, close=110.0))
    strategy = Zones(round_number_filter=True)
    ctx.push(_h4(0, close=200.0))
    strategy.on_bar(ctx)
    zone = strategy._zones[0]
    mid = (zone.low + zone.high) / 2
    assert mid == pytest.approx(98.85)
    assert Zones._passes_round_filter(mid, PERP) is False

    ctx.push(_h4(1, close=mid))
    assert strategy.on_bar(ctx) is None  # touch still consumes freshness
    assert zone.fresh is False


def test_reset_clears_zones_and_rescan_state() -> None:
    ctx = _demand_impulse_context()
    strategy = Zones()
    ctx.push(_h4(0, close=200.0))
    strategy.on_bar(ctx)
    assert len(strategy._zones) == 1

    strategy.reset()
    assert strategy._zones == []
    assert strategy._daily_last_ts is None

    # Re-running from a Context already holding all the same bars re-detects the zone.
    ctx.push(_h4(1, close=200.0))
    strategy.on_bar(ctx)
    assert len(strategy._zones) == 1


def test_keeps_only_the_newest_20_zones() -> None:
    # Prime the incremental ATR state through the bar before the impulse (day 15) for real,
    # so injecting fake zones and then scanning just the impulse bar (day 16) stays internally
    # consistent - jumping `_daily_last_ts` ahead without having run the intermediate bars
    # through `_advance_atr` would leave the local ATR state stale.
    ctx = Context(PERP)
    for bar in _flat_daily(15):
        ctx.push(bar)
    ctx.push(_daily(15, open_=100.0, high=101.0, low=97.0, close=98.0))
    strategy = Zones()
    strategy._detect_new_zones(ctx)

    strategy._zones = [
        # Source timestamps before the fixture's Daily bars, so none collides with the real zone.
        _Zone(
            direction=1,
            low=float(i),
            high=float(i) + 1,
            born_ts=START,
            source_ts=START - timedelta(days=i + 1),
        )
        for i in range(25)
    ]
    ctx.push(_daily(16, open_=98.0, high=111.0, low=97.0, close=110.0))
    strategy._detect_new_zones(ctx)
    assert len(strategy._zones) == _MAX_ZONES
    # The oldest 6 of the original 25 (indices 0..4, i.e. lows 0.0..4.0) were dropped.
    remaining_lows = {zone.low for zone in strategy._zones}
    assert 0.0 not in remaining_lows
    assert 24.0 in remaining_lows
    assert 97.0 in remaining_lows  # the newly detected demand zone survives


def test_zones_keep_detecting_after_the_daily_series_saturates() -> None:
    """`_daily_last_ts` is keyed on `ts_open`, not a row count: a row count would stop
    growing once the Daily buffer is trimmed down to `max_bars`, at which point a
    `daily.shape[0] > self._daily_scanned`-style check would never fire again and detection
    would silently stop dead - the strategy must keep finding impulses long after that."""
    ctx = Context(PERP, max_bars=40)
    strategy = Zones()
    for bar in _flat_daily(45):  # > max_bars: the Daily buffer saturates partway through
        ctx.push(bar)
        assert strategy.on_bar(ctx) is None
    assert ctx.bars("1d").shape[0] == 40  # saturated; row count has stopped growing

    # A fresh opposing candle + impulse, entirely after saturation.
    ctx.push(_daily(45, open_=100.0, high=101.0, low=97.0, close=98.0))
    assert strategy.on_bar(ctx) is None
    ctx.push(_daily(46, open_=98.0, high=111.0, low=97.0, close=110.0))
    strategy.on_bar(ctx)

    assert len(strategy._zones) == 1
    zone = strategy._zones[0]
    assert zone.direction == 1
    assert zone.low == pytest.approx(97.0)
    assert zone.high == pytest.approx(101.0)


# --- freshness is a property of the bars, not of when the engine consulted the strategy -----


def test_a_bar_the_strategy_never_saw_still_consumes_freshness() -> None:
    """The engine consults a strategy only when flat, with no pending entry and inside the
    session (core/engine.py), but a zone is used up by the first 4H bar that overlaps it
    whether or not anyone was watching (spec: the session filter gates entries only). A bar
    pushed without a consult must burn the zone, so the next bar closing inside it cannot
    signal. Measured on the 2026-09-07 run: 38 (BTC) / 47 (SOL) such unconsumed touches
    under 'active', which produced more zone trades than 'none'."""
    ctx = _demand_impulse_context()
    strategy = Zones()
    ctx.push(_h4(0, close=200.0))
    strategy.on_bar(ctx)
    zone = strategy._zones[0]
    assert zone.fresh is True

    # Overlaps [97, 101] but closes outside; the strategy is NOT consulted on this bar.
    ctx.push(
        Bar(
            instrument=PERP,
            tf="4h",
            ts_open=H4_START + timedelta(hours=4),
            open=105.0,
            high=105.0,
            low=96.0,
            close=105.0,
            volume=1.0,
        )
    )

    ctx.push(_h4(2, close=99.0))  # closes squarely inside the zone
    assert strategy.on_bar(ctx) is None
    assert zone.fresh is False


def test_bars_that_closed_before_the_zone_existed_do_not_consume_it() -> None:
    """Replaying unseen bars must respect birth order: the opposing candle's own 4H bars
    overlap the zone by construction, and they closed before the impulse completed. Only a
    bar that opened at or after the zone's Daily bar closed can touch it."""
    ctx = Context(PERP)
    for bar in _flat_daily(15):
        ctx.push(bar)
    ctx.push(_daily(15, open_=100.0, high=101.0, low=97.0, close=98.0))
    # Day 16's own 4H bars sit inside the future zone [97, 101] (never consulted).
    for i in range(6):
        ctx.push(
            Bar(
                instrument=PERP,
                tf="4h",
                ts_open=START + timedelta(days=16, hours=4 * i),
                open=99.0,
                high=99.5,
                low=98.5,
                close=99.0,
                volume=1.0,
            )
        )
    ctx.push(_daily(16, open_=98.0, high=111.0, low=97.0, close=110.0))  # closes on day 17
    strategy = Zones()

    ctx.push(_h4(0, close=200.0))  # day 17 00:00: first consult, detection, replay
    assert strategy.on_bar(ctx) is None
    zone = strategy._zones[0]
    assert zone.fresh is True

    ctx.push(_h4(1, close=99.0))
    signal = strategy.on_bar(ctx)
    assert signal is not None
    assert signal.direction == 1


def test_consecutive_impulses_from_the_same_opposing_candle_yield_one_zone() -> None:
    """Day 17 continues day 16's impulse: the close-to-close move from the same start bar
    still clears the threshold, so `_find_opposing_candle` lands on the same bearish candle.
    That must not append a second, identical zone (26 of BTC's 54 zones on the 2026-09-07
    run were exact copies, crowding the 20-zone cap)."""
    ctx = Context(PERP)
    for bar in _flat_daily(15):
        ctx.push(bar)
    ctx.push(_daily(15, open_=100.0, high=101.0, low=97.0, close=98.0))
    ctx.push(_daily(16, open_=98.0, high=111.0, low=97.0, close=110.0))
    ctx.push(_daily(17, open_=110.0, high=113.0, low=109.0, close=112.0))
    strategy = Zones()

    ctx.push(_h4(6, close=200.0))  # day 18 00:00, after day 17 closed
    strategy.on_bar(ctx)

    assert [(z.direction, z.low, z.high) for z in strategy._zones] == [(1, 97.0, 101.0)]


def test_a_re_detected_zone_does_not_regain_freshness() -> None:
    """The zone from day 16's impulse is touched (and used) on day 17; day 17's bar then
    re-selects the same opposing candle. The zone must stay consumed, not come back fresh."""
    ctx = Context(PERP)
    for bar in _flat_daily(15):
        ctx.push(bar)
    ctx.push(_daily(15, open_=100.0, high=101.0, low=97.0, close=98.0))
    ctx.push(_daily(16, open_=98.0, high=111.0, low=97.0, close=110.0))
    strategy = Zones()
    ctx.push(_h4(0, close=99.0))  # day 17 00:00, inside the zone: signals and consumes it
    assert strategy.on_bar(ctx) is not None

    ctx.push(_daily(17, open_=110.0, high=113.0, low=109.0, close=112.0))
    ctx.push(_h4(6, close=200.0))  # day 18 00:00: re-detection
    assert strategy.on_bar(ctx) is None
    assert [zone.fresh for zone in strategy._zones] == [False]

    ctx.push(_h4(7, close=99.0))
    assert strategy.on_bar(ctx) is None


def test_the_first_bar_after_the_zone_is_born_can_touch_it() -> None:
    """The first 4H bar the strategy sees after a Daily close opens exactly at `born_ts`
    (replay pushes the Daily bar before the 4H bar that opens at the same instant), so the
    birth rule is `>=`: that bar must be able to touch and use the zone."""
    ctx = _demand_impulse_context()
    strategy = Zones()
    ctx.push(_h4(0, close=99.0))  # ts_open == H4_START == the zone's born_ts
    signal = strategy.on_bar(ctx)
    assert signal is not None
    assert signal.direction == 1


def test_touches_are_replayed_from_the_retained_window_after_a_long_consult_gap() -> None:
    """More than `max_bars` 4H bars pass between two consults: the bar the strategy last saw
    has been trimmed away, so the whole retained window is replayed - and a touch inside
    that window still burns the zone."""
    ctx = Context(PERP, max_bars=40)
    for bar in _flat_daily(15):
        ctx.push(bar)
    ctx.push(_daily(15, open_=100.0, high=101.0, low=97.0, close=98.0))
    ctx.push(_daily(16, open_=98.0, high=111.0, low=97.0, close=110.0))
    strategy = Zones()
    ctx.push(_h4(0, close=200.0))
    strategy.on_bar(ctx)  # first consult: the zone is detected, fresh

    for i in range(1, 60):  # never consulted; index 45 is the only bar that overlaps the zone
        ctx.push(_h4(i, close=99.0 if i == 45 else 200.0))
    assert ctx.offset_of(0) is None  # the last-seen bar is out of the window

    ctx.push(_h4(60, close=99.0))  # closes inside the zone
    assert strategy.on_bar(ctx) is None
    assert [zone.fresh for zone in strategy._zones] == [False]


# --- performance (C1): the local Wilder ATR series must be incremental ----------------------


def test_on_bar_stays_fast_over_a_multi_year_history() -> None:
    """Pushing ~4 years of bars (8,760 4H + 1,460 Daily) through `on_bar` must complete well
    under the generous 5s budget - a regression to recomputing the local Wilder ATR series
    from scratch on every newly-scanned Daily bar (or tracking "new" via a row count that
    saturates once the Daily buffer is trimmed) takes well over 60s instead."""
    ctx = Context(PERP)
    strategy = Zones()
    start = datetime(2026, 1, 1, tzinfo=UTC)
    daily_bars = [
        Bar(
            instrument=PERP,
            tf="1d",
            ts_open=start + timedelta(days=i),
            open=100.0 + math.sin(i / 23.0) * 5.0,
            high=100.0 + math.sin(i / 23.0) * 5.0 + 2.0,
            low=100.0 + math.sin(i / 23.0) * 5.0 - 2.0,
            close=100.0 + math.sin(i / 23.0) * 5.0,
            volume=1.0,
        )
        for i in range(1_460)
    ]
    h4_bars = [
        Bar(
            instrument=PERP,
            tf="4h",
            ts_open=start + timedelta(hours=4 * i),
            open=100.0 + math.sin(i / 40.0) * 5.0,
            high=100.0 + math.sin(i / 40.0) * 5.0 + 1.0,
            low=100.0 + math.sin(i / 40.0) * 5.0 - 1.0,
            close=100.0 + math.sin(i / 40.0) * 5.0,
            volume=1.0,
        )
        for i in range(8_760)
    ]

    daily_iter = iter(daily_bars)
    next_daily = next(daily_iter, None)
    load_before = load_factor()
    started = time.perf_counter()
    for h4_bar in h4_bars:
        while next_daily is not None and next_daily.ts_open.date() < h4_bar.ts_open.date():
            ctx.push(next_daily)
            next_daily = next(daily_iter, None)
        ctx.push(h4_bar)
        strategy.on_bar(ctx)
    elapsed = time.perf_counter() - started

    budget = scaled_budget(5.0, load_before)
    assert elapsed < budget, (
        f"Zones.on_bar took {elapsed:.2f}s for 8,760 4H bars "
        f"(budget: {budget:.1f}s, scaled for the machine's load)"
    )
