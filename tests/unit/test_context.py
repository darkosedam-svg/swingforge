"""Contract tests for Context: bar series, bar_index, and Wilder ATR/ADX.

The expected ATR/ADX values are produced by the plain-Python reference loops below, which
are written from Wilder's definitions rather than by calling the implementation under test.
Two exact anchors (constant true range, and a pure one-directional trend) pin the reference
itself, so a shared misunderstanding between the two implementations would still be caught.
"""

from __future__ import annotations

import json
import math
import time
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import numpy as np
import pytest

from swingforge.core.context import Context
from swingforge.core.types import Bar, Instrument, Position

BTC = Instrument(
    venue="hyperliquid",
    symbol="BTC",
    tick_size=Decimal("0.5"),
    contract_multiplier=Decimal("1"),
    quote_ccy="USD",
    session_profile="perp",
)
EUR = Instrument(
    venue="oanda",
    symbol="EUR_USD",
    tick_size=Decimal("0.00001"),
    contract_multiplier=Decimal("1"),
    quote_ccy="USD",
    session_profile="fx",
)
START = datetime(2026, 1, 1, tzinfo=UTC)
TF_HOURS = {"1h": 1, "4h": 4, "1d": 24}


def make_bar(
    index: int,
    open_: float,
    high: float,
    low: float,
    close: float,
    volume: float = 1.0,
    tf: str = "4h",
    instrument: Instrument = BTC,
) -> Bar:
    return Bar(
        instrument=instrument,
        tf=tf,  # type: ignore[arg-type]
        ts_open=START + timedelta(hours=TF_HOURS[tf] * index),
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
    )


def walk(n_bars: int, tf: str = "4h") -> list[Bar]:
    """A deterministic, irregular OHLC walk — no randomness, no shared code with Context."""
    bars: list[Bar] = []
    price = 100.0
    for i in range(n_bars):
        step = math.sin(i / 3.0) * 2.0 + (i % 5) * 0.3 - 0.4
        open_ = price
        close = price + step
        high = max(open_, close) + 0.7 + (i % 3) * 0.4
        low = min(open_, close) - 0.5 - (i % 4) * 0.3
        bars.append(make_bar(i, open_, high, low, close, volume=float(i + 1), tf=tf))
        price = close
    return bars


# --- Independent Wilder reference implementations --------------------------


def true_ranges(bars: list[Bar]) -> list[float]:
    trs = []
    for prev, cur in zip(bars, bars[1:], strict=False):
        trs.append(
            max(
                cur.high - cur.low,
                abs(cur.high - prev.close),
                abs(cur.low - prev.close),
            )
        )
    return trs


def rma(values: list[float], n: int) -> list[float]:
    """Wilder's smoothing, seeded with the simple average of the first n values."""
    out: list[float] = [math.nan] * (n - 1)
    acc = sum(values[:n]) / n
    out.append(acc)
    for value in values[n:]:
        acc = (acc * (n - 1) + value) / n
        out.append(acc)
    return out


def reference_atr(bars: list[Bar], n: int) -> float:
    if len(bars) < n + 1:
        return math.nan
    return rma(true_ranges(bars), n)[-1]


def reference_adx(bars: list[Bar], n: int) -> float:
    if len(bars) < 2 * n + 1:
        return math.nan
    plus_dm: list[float] = []
    minus_dm: list[float] = []
    for prev, cur in zip(bars, bars[1:], strict=False):
        up = cur.high - prev.high
        down = prev.low - cur.low
        plus_dm.append(up if (up > down and up > 0) else 0.0)
        minus_dm.append(down if (down > up and down > 0) else 0.0)

    tr_s = rma(true_ranges(bars), n)
    plus_s = rma(plus_dm, n)
    minus_s = rma(minus_dm, n)

    dx: list[float] = []
    for tr, plus, minus in zip(tr_s[n - 1 :], plus_s[n - 1 :], minus_s[n - 1 :], strict=False):
        if tr <= 0:
            dx.append(0.0)
            continue
        plus_di = 100.0 * plus / tr
        minus_di = 100.0 * minus / tr
        total = plus_di + minus_di
        dx.append(0.0 if total <= 0 else 100.0 * abs(plus_di - minus_di) / total)
    return rma(dx, n)[-1]


def filled(bars: list[Bar], instrument: Instrument = BTC, max_bars: int = 5000) -> Context:
    ctx = Context(instrument, max_bars=max_bars)
    for b in bars:
        ctx.push(b)
    return ctx


# --- Series, index and accessors -------------------------------------------


def test_empty_context() -> None:
    ctx = Context(BTC)
    assert ctx.instrument == BTC
    assert ctx.position is None
    assert ctx.bar_index == -1
    assert ctx.last("4h") is None
    assert ctx.history("4h") == ()
    assert ctx.ts() is None
    assert ctx.bars("4h").shape == (0, 5)
    assert ctx.bars("4h").dtype == np.float64


def test_bars_shape_columns_and_ordering() -> None:
    bars = walk(5)
    ctx = filled(bars)
    arr = ctx.bars("4h")
    assert arr.shape == (5, 5)
    assert arr.dtype == np.float64
    # Oldest first, columns open, high, low, close, volume.
    for row, bar in zip(arr, bars, strict=True):
        assert list(row) == [bar.open, bar.high, bar.low, bar.close, bar.volume]


def test_series_are_kept_per_timeframe() -> None:
    ctx = Context(BTC)
    ctx.push(make_bar(0, 100.0, 101.0, 99.0, 100.5, tf="1d"))
    for b in walk(3, tf="4h"):
        ctx.push(b)
    ctx.push(make_bar(0, 100.0, 100.2, 99.8, 100.1, tf="1h"))

    assert ctx.bars("1d").shape == (1, 5)
    assert ctx.bars("4h").shape == (3, 5)
    assert ctx.bars("1h").shape == (1, 5)


def test_bar_index_advances_only_on_4h_bars() -> None:
    ctx = Context(BTC)
    assert ctx.bar_index == -1

    ctx.push(make_bar(0, 100.0, 101.0, 99.0, 100.5, tf="1d"))
    ctx.push(make_bar(0, 100.0, 100.2, 99.8, 100.1, tf="1h"))
    assert ctx.bar_index == -1

    ctx.push(make_bar(0, 100.0, 101.0, 99.0, 100.5, tf="4h"))
    assert ctx.bar_index == 0
    ctx.push(make_bar(1, 100.5, 102.0, 100.0, 101.0, tf="4h"))
    assert ctx.bar_index == 1

    ctx.push(make_bar(1, 101.0, 101.2, 100.8, 101.1, tf="1h"))
    assert ctx.bar_index == 1


def test_last_history_and_ts() -> None:
    bars = walk(4)
    ctx = filled(bars)
    assert ctx.last("4h") == bars[-1]
    assert ctx.history("4h") == tuple(bars)
    assert ctx.ts() == bars[-1].ts_open
    assert ctx.last("1d") is None


def test_ts_follows_the_4h_series_only() -> None:
    ctx = Context(BTC)
    ctx.push(make_bar(9, 100.0, 101.0, 99.0, 100.5, tf="1d"))
    assert ctx.ts() is None
    four_h = make_bar(2, 100.0, 101.0, 99.0, 100.5, tf="4h")
    ctx.push(four_h)
    assert ctx.ts() == four_h.ts_open


def test_position_is_assignable_by_the_engine() -> None:
    ctx = Context(BTC)
    ctx.position = Position(instrument=BTC, direction=1, qty=1.0, avg_price=100.0, stop=97.0, target=None)
    assert ctx.position is not None
    assert ctx.position.direction == 1


def test_max_bars_trims_oldest_but_not_bar_index() -> None:
    bars = walk(6)
    ctx = filled(bars, max_bars=3)
    assert ctx.bars("4h").shape == (3, 5)
    assert ctx.history("4h") == tuple(bars[-3:])
    assert ctx.last("4h") == bars[-1]
    assert ctx.bar_index == 5


def test_push_rejects_a_foreign_instrument() -> None:
    ctx = Context(BTC)
    with pytest.raises(ValueError):
        ctx.push(make_bar(0, 1.1, 1.2, 1.0, 1.15, tf="4h", instrument=EUR))


def test_unknown_timeframe_rejected() -> None:
    ctx = Context(BTC)
    with pytest.raises(ValueError):
        ctx.bars("15m")  # type: ignore[arg-type]


# --- ATR -------------------------------------------------------------------


def test_atr_is_nan_below_n_plus_one_bars() -> None:
    assert math.isnan(filled(walk(14)).atr("4h", 14))
    assert not math.isnan(filled(walk(15)).atr("4h", 14))
    assert math.isnan(Context(BTC).atr("4h"))


def test_atr_matches_wilder_reference() -> None:
    bars = walk(40)
    ctx = filled(bars)
    for n in (2, 5, 14):
        assert ctx.atr("4h", n) == pytest.approx(reference_atr(bars, n), abs=1e-9, rel=1e-12)


def test_atr_of_constant_true_range_is_that_range() -> None:
    # Each bar spans exactly 2.0 and never gaps, so every true range is 2.0 and Wilder's
    # smoothing of a constant is that constant.
    bars = [make_bar(i, 100.0, 101.0, 99.0, 100.0) for i in range(30)]
    assert filled(bars).atr("4h", 14) == pytest.approx(2.0, abs=1e-12)


def test_atr_is_computed_per_timeframe() -> None:
    daily = walk(40, tf="1d")
    four_h = walk(40, tf="4h")
    ctx = Context(BTC)
    for b in daily:
        ctx.push(b)
    for b in four_h:
        ctx.push(b)
    assert ctx.atr("1d", 14) == pytest.approx(reference_atr(daily, 14), abs=1e-9)
    assert ctx.atr("4h", 14) == pytest.approx(reference_atr(four_h, 14), abs=1e-9)


# --- ADX -------------------------------------------------------------------


def test_adx_is_nan_below_two_n_plus_one_bars() -> None:
    assert math.isnan(filled(walk(28)).adx("4h", 14))
    assert not math.isnan(filled(walk(29)).adx("4h", 14))


def test_adx_matches_wilder_reference() -> None:
    bars = walk(60)
    ctx = filled(bars)
    for n in (3, 7, 14):
        assert ctx.adx("4h", n) == pytest.approx(reference_adx(bars, n), abs=1e-9, rel=1e-12)


def test_adx_of_a_pure_uptrend_is_one_hundred() -> None:
    # Every bar steps up by exactly 1: +DM is 1 on every bar, -DM is 0, so -DI is 0 and
    # DX is 100 on every bar; smoothing 100s gives 100.
    bars = [make_bar(i, i + 0.5, i + 1.0, float(i), i + 0.5) for i in range(40)]
    assert filled(bars).adx("4h", 14) == pytest.approx(100.0, abs=1e-9)


def test_adx_of_a_flat_series_is_zero() -> None:
    bars = [make_bar(i, 100.0, 100.0, 100.0, 100.0) for i in range(40)]
    assert filled(bars).adx("4h", 14) == pytest.approx(0.0, abs=1e-12)


def test_adx_stays_within_zero_and_one_hundred() -> None:
    ctx = filled(walk(120))
    value = ctx.adx("4h", 14)
    assert 0.0 <= value <= 100.0


# --- Position ownership ----------------------------------------------------


def test_position_rejects_a_foreign_instrument() -> None:
    ctx = Context(BTC)
    with pytest.raises(ValueError):
        ctx.position = Position(instrument=EUR, direction=1, qty=1.0, avg_price=1.1, stop=1.05, target=None)
    assert ctx.position is None


def test_position_accepts_none() -> None:
    ctx = Context(BTC)
    ctx.position = Position(instrument=BTC, direction=1, qty=1.0, avg_price=100.0, stop=97.0, target=None)
    ctx.position = None
    assert ctx.position is None


# --- Absolute bar index -> row offset --------------------------------------


def test_offset_of_before_trimming() -> None:
    ctx = filled(walk(5))
    assert ctx.bar_index == 4
    assert [ctx.offset_of(i) for i in range(5)] == [0, 1, 2, 3, 4]
    assert ctx.offset_of(5) is None
    assert ctx.offset_of(-1) is None


def test_offset_of_after_trimming() -> None:
    ctx = filled(walk(8), max_bars=3)
    assert ctx.bar_index == 7
    assert ctx.offset_of(7) == 2
    assert ctx.offset_of(6) == 1
    assert ctx.offset_of(5) == 0
    assert ctx.offset_of(4) is None
    assert ctx.offset_of(8) is None


def test_offset_of_is_none_before_any_bar() -> None:
    assert Context(BTC).offset_of(0) is None


# --- Snapshot --------------------------------------------------------------


def test_snapshot_round_trips_through_json() -> None:
    four_h = walk(40)
    ctx = filled(four_h)
    for b in walk(40, tf="1d"):
        ctx.push(b)

    payload = ctx.snapshot()
    assert isinstance(payload, bytes)
    data = json.loads(payload.decode("utf-8"))

    assert data["bar_index"] == 39
    assert data["ts"] == four_h[-1].ts_open.isoformat()
    assert data["atr_4h"] == pytest.approx(ctx.atr("4h"), abs=1e-12)
    assert data["atr_1d"] == pytest.approx(ctx.atr("1d"), abs=1e-12)
    assert data["adx_4h"] == pytest.approx(ctx.adx("4h"), abs=1e-12)
    assert data["adx_1d"] == pytest.approx(ctx.adx("1d"), abs=1e-12)
    assert data["closes_4h"] == [pytest.approx(b.close) for b in four_h[-20:]]


def test_snapshot_serialises_nan_indicators_as_null() -> None:
    assert json.loads(Context(BTC).snapshot()) == {
        "bar_index": -1,
        "ts": None,
        "atr_4h": None,
        "atr_1d": None,
        "adx_4h": None,
        "adx_1d": None,
        "closes_4h": [],
    }


# --- Read path cost --------------------------------------------------------


def test_default_max_bars_spans_a_multi_year_run() -> None:
    assert Context(BTC).max_bars == 50_000


def test_bars_returns_the_same_view_until_a_bar_is_pushed() -> None:
    ctx = filled(walk(5))
    view = ctx.bars("4h")
    assert ctx.bars("4h") is view
    ctx.push(make_bar(5, 100.0, 101.0, 99.0, 100.5))
    assert ctx.bars("4h") is not view


def test_bars_view_stays_read_only_and_oldest_first_after_trimming() -> None:
    bars = walk(8)
    ctx = filled(bars, max_bars=3)
    arr = ctx.bars("4h")
    assert arr.shape == (3, 5)
    assert not arr.flags.writeable
    for row, bar in zip(arr, bars[-3:], strict=True):
        assert list(row) == [bar.open, bar.high, bar.low, bar.close, bar.volume]
    with pytest.raises(ValueError):
        arr[0, 0] = 1.0


def test_repeated_reads_on_one_bar_do_not_advance_the_smoothing() -> None:
    bars = walk(40)
    ctx = filled(bars)
    for _ in range(3):
        assert ctx.atr("4h", 14) == pytest.approx(reference_atr(bars, 14), abs=1e-9, rel=1e-12)
        assert ctx.adx("4h", 14) == pytest.approx(reference_adx(bars, 14), abs=1e-9, rel=1e-12)


def test_atr_matches_the_reference_bar_by_bar_including_after_trimming() -> None:
    bars = walk(60)
    ctx = Context(BTC, max_bars=20)
    for i, bar in enumerate(bars):
        ctx.push(bar)
        value = ctx.atr("4h", 5)
        expected = reference_atr(bars[: i + 1], 5)
        if math.isnan(expected):
            assert math.isnan(value)
        else:
            assert value == pytest.approx(expected, abs=1e-9, rel=1e-12)
    assert ctx.bars("4h").shape == (20, 5)


def test_adx_matches_the_reference_bar_by_bar_including_after_trimming() -> None:
    bars = walk(60)
    ctx = Context(BTC, max_bars=20)
    for i, bar in enumerate(bars):
        ctx.push(bar)
        value = ctx.adx("4h", 5)
        expected = reference_adx(bars[: i + 1], 5)
        if math.isnan(expected):
            assert math.isnan(value)
        else:
            assert value == pytest.approx(expected, abs=1e-9, rel=1e-12)
    assert ctx.bars("4h").shape == (20, 5)


def test_indicator_reads_stay_linear_over_a_four_year_run() -> None:
    # 8,760 4H bars is four years. A rebuild-per-read implementation is O(n^2) and takes
    # ~2 minutes here; the bound is deliberately loose so only that regression trips it.
    bars = walk(8760)
    ctx = Context(BTC)
    start = time.perf_counter()
    for bar in bars:
        ctx.push(bar)
        ctx.atr("4h")
        ctx.adx("4h")
    elapsed = time.perf_counter() - start
    assert elapsed < 5.0, f"atr+adx over {len(bars)} bars took {elapsed:.1f}s"


# --- Multi-step memo (sparse reads) -----------------------------------------


def test_sparse_reads_match_dense_reads_bar_by_bar_including_after_trimming() -> None:
    # Reading every 7th bar must advance the memo by applying the one-step Wilder update
    # k times (not recompute from scratch), and must land on exactly the same float as
    # reading every bar -- including once trimming has started discarding old rows.
    bars = walk(600)
    dense = Context(BTC, max_bars=50)
    sparse = Context(BTC, max_bars=50)
    dense_atr: list[float] = []
    dense_adx: list[float] = []
    for bar in bars:
        dense.push(bar)
        dense_atr.append(dense.atr("4h", 5))
        dense_adx.append(dense.adx("4h", 5))

    for i, bar in enumerate(bars):
        sparse.push(bar)
        if i % 7 == 0 or i == len(bars) - 1:
            atr_value = sparse.atr("4h", 5)
            adx_value = sparse.adx("4h", 5)
            if math.isnan(dense_atr[i]):
                assert math.isnan(atr_value)
            else:
                assert atr_value == pytest.approx(dense_atr[i], abs=1e-12, rel=1e-12)
            if math.isnan(dense_adx[i]):
                assert math.isnan(adx_value)
            else:
                assert adx_value == pytest.approx(dense_adx[i], abs=1e-12, rel=1e-12)


def test_sparse_reads_survive_a_gap_wider_than_the_retained_window() -> None:
    # A read gap (k new pushes since the last read) wider than the retained window (here
    # max_bars=10, so at most 10 bars are ever available to replay) can no longer apply the
    # one-step update k times -- there aren't k+1 old rows left to anchor it -- so it must
    # fall back to a full recompute over whatever is currently retained, exactly as a full
    # rebuild always has (trimming has already discarded the bars a "true" continuation
    # would need, independent of this memo). That recompute is over the last `max_bars`
    # bars only, so the reference must be taken over that same trailing window, not the
    # full untrimmed history.
    bars = walk(120)
    ctx = Context(BTC, max_bars=15)
    for bar in bars[:20]:
        ctx.push(bar)
    ctx.atr("4h", 5)
    ctx.adx("4h", 5)
    for bar in bars[20:]:
        ctx.push(bar)
    # 100 bars pushed since the last read, far more than max_bars=15 can replay.
    retained = bars[-15:]
    assert ctx.atr("4h", 5) == pytest.approx(reference_atr(retained, 5), abs=1e-9, rel=1e-12)
    assert ctx.adx("4h", 5) == pytest.approx(reference_adx(retained, 5), abs=1e-9, rel=1e-12)


def test_interleaved_daily_pushes_do_not_disturb_the_4h_memo() -> None:
    four_h = walk(60, tf="4h")
    daily = walk(60, tf="1d")
    ctx = Context(BTC)

    # Read the 4h memo up to date first, sparsely.
    for i, bar in enumerate(four_h[:30]):
        ctx.push(bar)
        if i % 5 == 0:
            ctx.atr("4h", 5)
            ctx.adx("4h", 5)

    # Interleave the remaining daily and 4h pushes (each bumps only its own timeframe's
    # push counter), reading both memos sparsely and at different cadences.
    for i in range(30):
        ctx.push(daily[i])
        if i % 4 == 0:
            ctx.atr("1d", 5)
            ctx.adx("1d", 5)
        ctx.push(four_h[30 + i])
        if i % 3 == 0:
            ctx.atr("4h", 5)
            ctx.adx("4h", 5)

    for bar in daily[30:]:
        ctx.push(bar)

    assert ctx.atr("4h", 5) == pytest.approx(reference_atr(four_h, 5), abs=1e-9, rel=1e-12)
    assert ctx.adx("4h", 5) == pytest.approx(reference_adx(four_h, 5), abs=1e-9, rel=1e-12)
    assert ctx.atr("1d", 5) == pytest.approx(reference_atr(daily, 5), abs=1e-9, rel=1e-12)
    assert ctx.adx("1d", 5) == pytest.approx(reference_adx(daily, 5), abs=1e-9, rel=1e-12)


def test_sparse_indicator_reads_stay_fast_over_a_four_year_run() -> None:
    # Reading atr/adx every 25th bar (as the tournament does at entries) must cost roughly
    # one smoothing step per *pushed* bar, not a full O(n) rebuild per *read* -- otherwise
    # sparse reads over a long run are O(n^2). Today (rebuild-on-gap) this takes >4s here;
    # the multi-step memo must bring it under the same bound as the dense-read test.
    bars = walk(8760)
    ctx = Context(BTC)
    start = time.perf_counter()
    for i, bar in enumerate(bars):
        ctx.push(bar)
        if i % 25 == 0:
            ctx.atr("4h")
            ctx.adx("4h")
    elapsed = time.perf_counter() - start
    assert elapsed < 2.0, f"sparse atr+adx over {len(bars)} bars took {elapsed:.1f}s"
