"""Unit tests for `strategies/ict.py` — the sweep / MSS / OB-FVG entry strategy.

Two layers: direct unit tests of the module-level helpers (ports of the SHURIKEN v0.3
reference tests, `docs/superpowers/reference/shuriken_test_ict_analyzer_v03.py`), and
fixture-driven tests that drive `ICT().on_bar(ctx)` bar-by-bar over hand-built histories in
`tests/fixtures/ict_*.json`, asserting that only the intended bar (or none) ever signals.
"""

from __future__ import annotations

import json
import math
import sys
import time
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest

from swingforge.core.context import Context
from swingforge.core.types import Bar, Instrument, Signal
from swingforge.strategies.ict import (
    ICT,
    find_bos,
    find_fvg,
    find_order_block,
    find_sweep,
    find_sweep_against,
)

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"

INSTRUMENT = Instrument(
    venue="hyperliquid",
    symbol="BTC",
    tick_size=Decimal("0.01"),
    contract_multiplier=Decimal("1"),
    quote_ccy="USD",
    session_profile="perp",
)


def _ohlcv(rows: list[tuple[float, float, float, float, float]]) -> np.ndarray:
    return np.array(rows, dtype=np.float64)


# --- module-level helper tests (ported from the SHURIKEN v0.3 reference suite) -------------


def test_sweep_of_daily_low_is_long_and_sweep_of_high_is_short() -> None:
    """Regression: v0.3 fixed the direction mapping (low sweep -> long, high sweep -> short)."""
    level_low, level_high = 100.0, 105.0
    # 10 bars comfortably inside the range, then one wicking below the low, closing inside.
    inside = [(102.0, 103.0, 101.0, 102.0, 1.0) for _ in range(10)]
    sweep_low = inside + [(102.0, 103.0, 99.5, 101.0, 1.0)]
    bars = _ohlcv(sweep_low)
    idx, direction = find_sweep(bars, level_low, level_high, lookback=12, min_pierce_pct=0.0003)
    assert idx == 10 and direction == 1

    sweep_high = inside + [(102.0, 105.5, 101.0, 104.0, 1.0)]
    bars2 = _ohlcv(sweep_high)
    idx2, direction2 = find_sweep(bars2, level_low, level_high, lookback=12, min_pierce_pct=0.0003)
    assert idx2 == 10 and direction2 == -1


def test_pullback_candle_that_opened_outside_the_range_is_not_a_sweep() -> None:
    """Regression: a candle must OPEN inside the range, not just close there."""
    level_low, level_high = 100.0, 105.0
    inside = [(102.0, 103.0, 101.0, 102.0, 1.0) for _ in range(10)]
    # Opened above the range (108), wicked down, closed back inside (104).
    pullback = inside + [(108.0, 108.5, 103.0, 104.0, 1.0)]
    bars = _ohlcv(pullback)
    idx, direction = find_sweep(bars, level_low, level_high, lookback=12, min_pierce_pct=0.0003)
    assert idx is None and direction is None


def test_fvg_and_order_block_helpers() -> None:
    long_gap_bars = _ohlcv(
        [
            (100.0, 101.0, 99.0, 100.0, 1.0),
            (100.0, 105.0, 100.0, 104.0, 1.0),
            (104.0, 108.0, 102.0, 107.0, 1.0),
        ]
    )
    gap = find_fvg(long_gap_bars, 0, 3, 1)
    assert gap == (101.0, 102.0)  # (prev.high, next.low) = (bottom, top)

    no_gap_bars = _ohlcv(
        [
            (100.0, 103.0, 99.0, 102.0, 1.0),
            (102.0, 106.0, 101.0, 105.0, 1.0),
            (105.0, 107.0, 102.0, 106.0, 1.0),
        ]
    )
    assert find_fvg(no_gap_bars, 0, 3, 1) is None

    ob_bars = _ohlcv(
        [
            (105.0, 106.0, 104.0, 104.0, 1.0),  # bearish
            (104.0, 105.0, 103.0, 103.0, 1.0),  # bearish (the LAST one -> picked)
            (103.0, 108.0, 103.0, 107.0, 1.0),  # bullish (BOS bar)
        ]
    )
    ob = find_order_block(ob_bars, 0, 2, 1)
    assert ob == (104.0, 103.0)  # (body_high, body_low)


def test_bos_neckline_uses_bars_before_the_current_candle() -> None:
    """Regression: the neckline must exclude the current candle's own high/low.

    This is the discriminating shape (a non-discriminating one - own high just above its own
    close - would fire under a buggy `max(prior, own_high)` implementation too, since the
    fold-in barely moves the threshold): the candle clears the true prior neckline (102, from
    the sweep bar) comfortably, but its OWN high (110) is far above its close (103) - a
    pre-fix implementation that folds the candle's own high into the neckline BEFORE
    comparing would raise the threshold to 110 - min_delta and this candle would NOT clear
    it, so it discriminates: fixed fires here, `max(prior, own_high)` does not. Verified
    against both implementations in a scratch script (not committed).
    """
    bars = _ohlcv(
        [
            (101.0, 102.0, 99.0, 100.0, 1.0),  # sweep bar, high=102
            (100.0, 100.5, 99.5, 100.2, 1.0),  # i=0
            (100.2, 100.8, 100.0, 100.5, 1.0),  # i=1
            (100.5, 110.0, 100.0, 103.0, 1.0),  # i=2: own high 110, close 103 -> clears the
            #                                     bars-before neckline (102) but not 110
        ]
    )
    bos_idx, neckline, _swing_low = find_bos(bars, 0, 1, min_bos_pct=0.0002, within=6)
    assert bos_idx == 3
    assert neckline == 110.0  # max(running_high_before=102, this candle's own high=110)


def test_bos_does_not_fire_before_the_i_gte_2_and_min_delta_gate_is_cleared() -> None:
    """Covers the `i >= 2` confirmation-delay gate together with the `min_delta` margin: at
    i=1 the close (104.0) already exceeds the running high built from bars before it (103),
    but `i >= _BOS_MIN_INDEX` blocks it; at i=2 the close (103.99) sits below running_high
    (106) minus min_delta, so neither gate is satisfied anywhere in this fixture."""
    bars = _ohlcv(
        [
            (101.0, 102.0, 99.0, 100.0, 1.0),  # sweep bar
            (100.0, 103.0, 99.0, 101.0, 1.0),  # i=0, running_high -> 103
            (101.0, 106.0, 100.0, 104.0, 1.0),  # i=1, running_high -> 106
            (103.0, 104.0, 102.0, 103.99, 1.0),  # i=2, close near own high but below 106
        ]
    )
    bos_idx, _neckline, _swing_low = find_bos(bars, 0, 1, min_bos_pct=0.0002, within=6)
    assert bos_idx is None


def test_find_sweep_on_an_empty_array_returns_none() -> None:
    assert find_sweep(_ohlcv([]).reshape(0, 5), 100.0, 105.0, lookback=12, min_pierce_pct=0.0003) == (
        None,
        None,
    )


def test_find_bos_short_direction_mirrors_the_long_case() -> None:
    # Sweep of the high (bearish setup): running_low starts at the sweep bar's low and a
    # later candle must close BELOW it (plus min_delta) to register the break down.
    bars = _ohlcv(
        [
            (104.0, 106.0, 103.0, 105.0, 1.0),  # sweep bar (short), low=103
            (105.0, 105.5, 101.0, 102.0, 1.0),  # i=0, running_low -> 101
            (102.0, 103.0, 99.0, 100.0, 1.0),  # i=1, running_low -> 99
            (100.0, 101.0, 95.0, 96.0, 1.0),  # i=2: closes below running_low(99) - min_delta
        ]
    )
    bos_idx, swing_high, running_low_extreme = find_bos(bars, 0, -1, min_bos_pct=0.0002, within=6)
    assert bos_idx == 3
    # The second return value is the max high of the up-to-10 bars right after the sweep
    # (105.5, 103.0, 101.0), mirroring `swing_low` for the long case.
    assert swing_high == 105.5
    # The third is min(running_low so far, this bar's own low) = min(99.0, 95.0).
    assert running_low_extreme == 95.0


def test_find_fvg_short_direction_gap() -> None:
    bearish_gap_bars = _ohlcv(
        [
            (108.0, 109.0, 107.0, 108.0, 1.0),
            (107.0, 107.5, 103.0, 104.0, 1.0),
            (104.0, 105.0, 101.0, 102.0, 1.0),
        ]
    )
    gap = find_fvg(bearish_gap_bars, 0, 3, -1)
    assert gap == (105.0, 107.0)  # (next.high, prev.low) = (bottom, top)


# --- fixture loading ---------------------------------------------------------------------


def _row_to_bar(row: list, tf: str) -> Bar:
    ts_iso, o, h, low, c, v = row
    return Bar(
        instrument=INSTRUMENT,
        tf=tf,  # type: ignore[arg-type]
        ts_open=datetime.fromisoformat(ts_iso),
        open=o,
        high=h,
        low=low,
        close=c,
        volume=v,
    )


def _load_fixture(name: str) -> tuple[list[Bar], list[Bar]]:
    data = json.loads((FIXTURES_DIR / f"ict_{name}.json").read_text())
    daily_bars = sorted((_row_to_bar(r, "1d") for r in data["daily"]), key=lambda b: b.ts_open)
    h4_bars = sorted((_row_to_bar(r, "4h") for r in data["h4"]), key=lambda b: b.ts_open)
    return daily_bars, h4_bars


def _run_fixture(name: str, strategy: ICT | None = None) -> list[Signal | None]:
    """Push a fixture's bars into a fresh Context, calling `on_bar` after every 4H bar.

    A Daily bar for day D is pushed after the last 4H bar whose `ts_open` falls on day D (it
    only becomes known once that day's own 4H bars have all closed) - here every Daily bar
    precedes the whole 4H window, so this reduces to: push all Daily bars, then the 4H bars
    one at a time, recording what `on_bar` returns after each.
    """
    daily_bars, h4_bars = _load_fixture(name)
    ctx = Context(INSTRUMENT)
    strategy = strategy if strategy is not None else ICT()
    daily_iter = iter(daily_bars)
    next_daily = next(daily_iter, None)
    results: list[Signal | None] = []
    for h4_bar in h4_bars:
        while next_daily is not None and next_daily.ts_open.date() < h4_bar.ts_open.date():
            ctx.push(next_daily)
            next_daily = next(daily_iter, None)
        ctx.push(h4_bar)
        results.append(strategy.on_bar(ctx))
    while next_daily is not None:
        ctx.push(next_daily)
        next_daily = next(daily_iter, None)
    return results


# --- fixture-driven tests ------------------------------------------------------------------


def test_clean_sweep_mss_ob_signals_a_long_at_the_ob_midpoint() -> None:
    # This fixture (and bos_neckline_boundary below) is the primary guard for the neckline
    # fix end-to-end through on_bar, in addition to the direct find_bos unit tests above.
    results = _run_fixture("clean_sweep_mss_ob")
    non_none = [(i, s) for i, s in enumerate(results) if s is not None]
    assert len(non_none) == 1
    idx, signal = non_none[0]
    assert idx == 33  # the BOS bar (0-indexed within the fixture's 4H series)
    assert signal.direction == 1
    assert signal.tag == "ict_sweep_mss_ob"
    assert signal.entry == pytest.approx(100.5)
    assert signal.stop == pytest.approx(87.99)
    assert signal.structure_target == pytest.approx(115.0)
    assert signal.expires_in_bars == 3


def test_pullback_not_sweep_never_signals() -> None:
    results = _run_fixture("pullback_not_sweep")
    assert all(s is None for s in results)


def test_sweep_no_mss_within_6_never_signals() -> None:
    results = _run_fixture("sweep_no_mss_within_6")
    assert all(s is None for s in results)


def test_bos_neckline_boundary_signals_only_on_the_true_break() -> None:
    # Primary guard for the neckline fix end-to-end: idx33 clears the running_high built
    # from bars before it only if that running high already folds in idx33's own high (the
    # pre-fix bug), so a regression here would surface as a spurious signal at idx33 (or
    # none at all) instead of the single true signal at idx34. See the fixture's own "note".
    results = _run_fixture("bos_neckline_boundary")
    non_none = [(i, s) for i, s in enumerate(results) if s is not None]
    assert len(non_none) == 1
    idx, signal = non_none[0]
    assert idx == 34  # the genuine BOS bar, one past the boundary candle at idx 33
    assert signal.direction == 1
    assert signal.tag == "ict_sweep_mss_ob"


def test_a_signalled_sweep_is_never_re_emitted() -> None:
    """`_last_signalled_sweep` guards against a re-scan finding the same old sweep again."""
    strategy = ICT()
    results = _run_fixture("clean_sweep_mss_ob", strategy=strategy)
    fired_at = next(i for i, s in enumerate(results) if s is not None)
    assert strategy._last_signalled_sweep == 30  # the sweep bar's index
    # Re-running on_bar again on the same, unchanged context must not re-signal.
    daily_bars, h4_bars = _load_fixture("clean_sweep_mss_ob")
    ctx = Context(INSTRUMENT)
    for bar in daily_bars:
        ctx.push(bar)
    for bar in h4_bars[: fired_at + 1]:
        ctx.push(bar)
    assert strategy.on_bar(ctx) is None


def test_reset_clears_last_signalled_sweep() -> None:
    strategy = ICT()
    strategy._last_signalled_sweep = 5
    strategy.reset()
    assert strategy._last_signalled_sweep is None


# --- on_bar edge cases (built directly on the clean_sweep_mss_ob Daily levels) --------------


def _quiet_h4_rows(n: int) -> list[tuple[float, float, float, float, float]]:
    rows = []
    for i in range(n):
        c = 100.0 if i % 2 == 0 else 102.0
        rows.append((c, c + 1.0, c - 1.0, c, 1.0))
    return rows


def _push_daily_levels(ctx: Context) -> None:
    """The Daily bars from `clean_sweep_mss_ob` establish swing levels [90, 115]."""
    daily_bars, _h4_bars = _load_fixture("clean_sweep_mss_ob")
    for bar in daily_bars:
        ctx.push(bar)


def _push_h4_rows(ctx: Context, rows: list[tuple[float, float, float, float, float]]) -> None:
    start = datetime(2026, 1, 12, tzinfo=UTC)
    for i, (o, h, low, c, v) in enumerate(rows):
        ctx.push(
            Bar(
                instrument=INSTRUMENT,
                tf="4h",
                ts_open=start + timedelta(hours=4 * i),
                open=o,
                high=h,
                low=low,
                close=c,
                volume=v,
            )
        )


def test_degenerate_signal_is_rejected_when_ob_midpoint_crosses_the_stop() -> None:
    """A later opposing candle can form an OB priced below the sweep low itself, putting the
    entry on the wrong side of `stop` - the Signal validator rejects it and `on_bar` returns
    None rather than raising."""
    rows = _quiet_h4_rows(30) + [
        (101.0, 102.0, 88.0, 100.0, 1.0),  # idx30: sweep (bearish), low=88 -> stop=87.99
        (100.0, 104.0, 100.0, 103.0, 1.0),  # idx31: i=0, bullish
        (90.0, 92.0, 84.0, 85.0, 1.0),  # idx32: i=1, bearish, body (90, 85) - below the stop!
        (85.0, 109.0, 84.0, 108.0, 1.0),  # idx33: i=2 - BOS fires here
    ]
    ctx = Context(INSTRUMENT)
    _push_daily_levels(ctx)
    _push_h4_rows(ctx, rows)
    strategy = ICT()
    assert strategy.rejected_signals == 0
    assert strategy.on_bar(ctx) is None
    assert strategy.rejected_signals == 1
    strategy.reset()
    assert strategy.rejected_signals == 0


def test_no_ob_and_no_fvg_returns_none() -> None:
    """A sweep + BOS with every candle bullish (no opposing candle for an OB) and no genuine
    3-bar gap either - `on_bar` finds nothing to enter on and returns None."""
    rows = _quiet_h4_rows(30) + [
        (98.0, 99.0, 88.0, 99.0, 1.0),  # idx30: sweep, itself bullish (close >= open)
        (99.0, 101.0, 98.0, 100.0, 1.0),  # idx31: i=0, bullish
        (100.0, 103.0, 99.0, 102.0, 1.0),  # idx32: i=1, bullish
        (102.0, 106.0, 101.0, 105.0, 1.0),  # idx33: i=2 - BOS fires here, still no bearish candle
    ]
    ctx = Context(INSTRUMENT)
    _push_daily_levels(ctx)
    _push_h4_rows(ctx, rows)
    assert ICT().on_bar(ctx) is None


def test_short_signal_uses_the_mirrored_stop_and_target() -> None:
    """A sweep of the Daily swing HIGH, mirroring `clean_sweep_mss_ob`: short direction,
    stop = sweep high + 1 tick, structure_target = the nearest Daily swing low below entry."""
    rows = _quiet_h4_rows(30) + [
        (110.0, 116.0, 109.0, 111.0, 1.0),  # idx30: sweep of the high, bullish -> OB candidate
        (110.0, 110.0, 105.0, 106.0, 1.0),  # idx31: i=0
        (106.0, 107.0, 102.0, 103.0, 1.0),  # idx32: i=1
        (103.0, 104.0, 97.0, 98.0, 1.0),  # idx33: i=2 - BOS fires here
    ]
    ctx = Context(INSTRUMENT)
    _push_daily_levels(ctx)
    _push_h4_rows(ctx, rows)
    signal = ICT().on_bar(ctx)
    assert signal is not None
    assert signal.direction == -1
    assert signal.tag == "ict_sweep_mss_ob"
    assert signal.entry == pytest.approx(110.5)  # OB midpoint (111 + 110) / 2
    assert signal.stop == pytest.approx(116.01)  # sweep high (116) + 1 tick
    assert signal.structure_target == pytest.approx(90.0)  # nearest Daily swing low


def test_fvg_entry_used_when_no_order_block_exists() -> None:
    """Every candle in the setup is bullish (no OB candidate), but a genuine gap forms in the
    displacement window - entry falls back to the FVG midpoint, tag `ict_sweep_mss_fvg`."""
    rows = _quiet_h4_rows(30) + [
        (98.0, 99.0, 88.0, 99.0, 1.0),  # idx30: sweep, bullish
        (99.0, 100.0, 98.0, 99.5, 1.0),  # idx31: i=0
        (99.5, 101.0, 99.0, 100.5, 1.0),  # idx32: i=1 (high=101, low of idx33=104 -> gap)
        (105.0, 110.0, 104.0, 109.0, 1.0),  # idx33: i=2 - BOS fires here
    ]
    ctx = Context(INSTRUMENT)
    _push_daily_levels(ctx)
    _push_h4_rows(ctx, rows)
    signal = ICT().on_bar(ctx)
    assert signal is not None
    assert signal.tag == "ict_sweep_mss_fvg"
    assert signal.entry == pytest.approx(102.0)  # FVG midpoint (100 + 104) / 2
    assert signal.stop == pytest.approx(87.99)  # sweep low (88) - 1 tick
    assert signal.structure_target == pytest.approx(115.0)


def test_bos_found_but_not_on_the_current_bar_returns_none() -> None:
    """If the array extends one bar past where the BOS actually completed, `find_bos` still
    finds that same (now-stale) index, but it no longer equals the current bar - a fresh
    strategy (no `_last_signalled_sweep` yet) must still not signal."""
    daily_bars, h4_bars = _load_fixture("bos_neckline_boundary")  # BOS completes at index 34
    ctx = Context(INSTRUMENT)
    for bar in daily_bars:
        ctx.push(bar)
    for bar in h4_bars:
        ctx.push(bar)
    extra = Bar(
        instrument=INSTRUMENT,
        tf="4h",
        ts_open=h4_bars[-1].ts_open + (h4_bars[-1].ts_open - h4_bars[-2].ts_open),
        open=106.5,
        high=107.0,
        low=106.0,
        close=106.7,
        volume=1.0,
    )
    ctx.push(extra)
    assert ICT().on_bar(ctx) is None


# --- minimum-history gates -----------------------------------------------------------------


def test_too_few_4h_bars_returns_none() -> None:
    ctx = Context(INSTRUMENT)
    daily_bars, h4_bars = _load_fixture("clean_sweep_mss_ob")
    for bar in daily_bars:
        ctx.push(bar)
    for bar in h4_bars[:10]:  # well under the 30-bar minimum
        ctx.push(bar)
    assert ICT().on_bar(ctx) is None


def test_too_few_daily_bars_returns_none() -> None:
    ctx = Context(INSTRUMENT)
    _daily_bars, h4_bars = _load_fixture("clean_sweep_mss_ob")
    # k=2 default needs 2*2+3=7 Daily bars; give it fewer.
    start = datetime(2026, 1, 1, tzinfo=UTC)
    for i in range(5):
        ctx.push(
            Bar(
                instrument=INSTRUMENT,
                tf="1d",
                ts_open=start + timedelta(days=i),
                open=100.0,
                high=101.0,
                low=99.0,
                close=100.0,
                volume=1.0,
            )
        )
    for bar in h4_bars:
        ctx.push(bar)
    assert ICT().on_bar(ctx) is None


# --- trimming-safe state (I3) ---------------------------------------------------------------


def _push_h4_row(ctx: Context, index: int, row: tuple[float, float, float, float, float]) -> None:
    o, h, low, c, v = row
    ctx.push(
        Bar(
            instrument=INSTRUMENT,
            tf="4h",
            ts_open=datetime(2026, 1, 12, tzinfo=UTC) + timedelta(hours=4 * index),
            open=o,
            high=h,
            low=low,
            close=c,
            volume=v,
        )
    )


def test_last_signalled_sweep_never_wrongly_suppresses_a_later_unrelated_sweep() -> None:
    """`_last_signalled_sweep` is stored as an absolute 4H `bar_index` and translated back to
    a row via `ctx.offset_of` before comparing - never compared as a raw row number, which
    would be reused by unrelated bars once trimming rotates the buffer.

    Engineered so a second, unrelated sweep's row - at the exact call where ITS OWN BOS
    confirms - numerically coincides with 30, the first sweep's raw row number (stored back
    when `max_bars=40` hadn't started trimming yet). A pre-fix implementation comparing raw
    row indices would read that coincidence as "the first sweep, already signalled" and
    wrongly suppress the second, entirely independent setup; the fix compares the second
    sweep's row against `ctx.offset_of(30)` (which by then points somewhere else entirely),
    so it correctly fires.
    """
    ctx = Context(INSTRUMENT, max_bars=40)
    _push_daily_levels(ctx)
    strategy = ICT(mss_within=9)

    first_sequence = _quiet_h4_rows(30) + [
        (101.0, 102.0, 88.0, 100.0, 1.0),  # idx30: sweep (bearish -> long), low=88
        (100.0, 100.5, 99.5, 100.2, 1.0),  # idx31: i=0
        (100.2, 100.8, 100.0, 100.5, 1.0),  # idx32: i=1
        (100.5, 108.0, 100.0, 106.0, 1.0),  # idx33: i=2 - BOS fires here
    ]
    signal = None
    for i, row in enumerate(first_sequence):
        _push_h4_row(ctx, i, row)
        signal = strategy.on_bar(ctx)
    assert signal is not None
    assert signal.direction == 1
    assert strategy._last_signalled_sweep == 30  # absolute bar_index; no trim has happened yet

    # 6 quiet bars bring the buffer to exactly max_bars (40) - still no trim.
    padding = _quiet_h4_rows(6)
    next_index = len(first_sequence)
    for row in padding:
        _push_h4_row(ctx, next_index, row)
        assert strategy.on_bar(ctx) is None  # the same old sweep, correctly never re-signalled
        next_index += 1

    # A brand-new, unrelated sweep, followed by 8 quiet bars and a BOS bar 9 bars later - by
    # then the buffer has rolled far enough that this second sweep's row lands on 30.
    second_sequence = (
        [(101.0, 102.0, 88.0, 100.0, 1.0)]
        + [(100.0, 100.5, 99.5, 100.0, 1.0) for _ in range(8)]
        + [(100.5, 110.0, 100.0, 108.0, 1.0)]
    )
    signal = None
    for row in second_sequence:
        _push_h4_row(ctx, next_index, row)
        signal = strategy.on_bar(ctx)
        next_index += 1

    assert ctx.offset_of(30) != 30  # row 30 now holds a different bar than it used to
    assert signal is not None  # the fix must not have suppressed this later, unrelated sweep
    assert signal.direction == 1
    assert strategy._last_signalled_sweep == 40  # the second sweep's own absolute bar_index


# --- the liquidity range is judged per candidate bar, not from the current close ------------


def _push_daily_rows(ctx: Context, rows: list[tuple[float, float, float, float, float]]) -> None:
    start = datetime(2025, 12, 20, tzinfo=UTC)
    for i, (o, h, low, c, v) in enumerate(rows):
        ctx.push(
            Bar(
                instrument=INSTRUMENT,
                tf="1d",
                ts_open=start + timedelta(days=i),
                open=o,
                high=h,
                low=low,
                close=c,
                volume=v,
            )
        )


_TWO_LOW_DAILY = [
    (100.0, 104.0, 92.0, 100.0, 1.0),
    (100.0, 103.0, 91.0, 100.0, 1.0),
    (100.0, 103.0, 90.0, 100.0, 1.0),  # swing low 90
    (100.0, 104.0, 91.0, 100.0, 1.0),
    (100.0, 115.0, 93.0, 100.0, 1.0),  # swing high 115
    (106.0, 110.0, 106.0, 108.0, 1.0),
    (108.0, 111.0, 106.0, 109.0, 1.0),
    (108.0, 110.0, 105.0, 109.0, 1.0),  # swing low 105 - the pivot the displacement crosses
    (109.0, 112.0, 107.0, 110.0, 1.0),
    (110.0, 112.0, 107.0, 110.0, 1.0),
    (110.0, 113.0, 108.0, 111.0, 1.0),
    (111.0, 113.0, 108.0, 112.0, 1.0),
]
"""Daily bars whose confirmed fractal pivots are exactly lows [90, 105] and highs [115]."""


def test_displacement_through_the_next_daily_pivot_keeps_its_own_sweep() -> None:
    """A sweep of 90 (range [90, 115]) followed by a displacement that closes above the next
    Daily swing low (105) and then breaks structure. At the BOS bar the *current* close sits
    in [105, 115], so a range read off the current close no longer contains the sweep bar and
    the setup is lost - the defect measured on real BTC/ETH bars (2026-09-07 run, where the
    accumulated Daily pivots are ~2.5% apart and this happens on ~45% of bars). Each
    candidate bar must be judged against the range around its own body: the sweep
    still counts and the setup fires on the BOS bar, exactly once.
    """
    rows = _quiet_h4_rows(30) + [
        (101.0, 102.0, 88.0, 100.0, 1.0),  # idx30: sweep of 90 (bearish -> OB), close in [90, 115]
        (100.0, 104.0, 99.5, 103.0, 1.0),  # idx31: i=0, still inside [90, 115]
        (103.0, 109.0, 102.5, 108.0, 1.0),  # idx32: i=1, closes above 105 -> current range moves
        (108.0, 112.0, 107.5, 111.0, 1.0),  # idx33: i=2, BOS (close > 109 neckline)
    ]
    ctx = Context(INSTRUMENT)
    _push_daily_rows(ctx, _TWO_LOW_DAILY)
    strategy = ICT()

    signals = []
    for i, row in enumerate(rows):
        _push_h4_row(ctx, i, row)
        signals.append(strategy.on_bar(ctx))

    assert signals[:-1] == [None] * (len(rows) - 1)
    signal = signals[-1]
    assert signal is not None
    assert signal.direction == 1
    assert signal.entry == pytest.approx(100.5)  # the sweep candle's own body midpoint
    assert signal.stop == pytest.approx(87.99)  # below the sweep low
    assert signal.structure_target == pytest.approx(115.0)
    assert signal.tag == "ict_sweep_mss_ob"


def test_find_sweep_against_judges_each_bar_by_the_range_its_body_traded_in() -> None:
    """`find_sweep_against(bars, range_at, ...)`: `range_at(body_low, body_high)` supplies the
    range for the bar being examined. A bar that lies outside the newest bar's range is still
    a sweep of its own range; a bar whose own range is undefined (None) is skipped."""

    def range_at(body_low: float, body_high: float) -> tuple[float, float] | None:
        if body_low < 90.0:
            return None
        return (90.0, 115.0) if body_high <= 115.0 else (115.0, 130.0)

    inside = [(102.0, 103.0, 101.0, 102.0, 1.0) for _ in range(10)]
    bars = _ohlcv(
        inside
        + [
            (101.0, 102.0, 88.0, 100.0, 1.0),  # sweep of 90, range [90, 115]
            (120.0, 124.0, 119.0, 123.0, 1.0),  # newest bar sits in [115, 130], no pierce
        ]
    )
    idx, direction = find_sweep_against(bars, range_at, lookback=12, min_pierce_pct=0.0003)
    assert idx == 10 and direction == 1

    # Newest-first: a later sweep, of whichever range, is returned before an earlier one.
    bars2 = _ohlcv(inside + [(101.0, 102.0, 88.0, 100.0, 1.0), (120.0, 131.0, 119.0, 123.0, 1.0)])
    idx2, direction2 = find_sweep_against(bars2, range_at, lookback=12, min_pierce_pct=0.0003)
    assert idx2 == 11 and direction2 == -1

    # An undefined range for a bar's body skips that bar rather than raising.
    bars3 = _ohlcv(inside + [(85.0, 86.0, 80.0, 85.0, 1.0)])
    assert find_sweep_against(bars3, range_at, lookback=12, min_pierce_pct=0.0003) == (None, None)


_THREE_LEVEL_DAILY = [
    (100.0, 105.0, 92.0, 100.0, 1.0),
    (100.0, 104.0, 91.0, 100.0, 1.0),
    (100.0, 103.0, 90.0, 100.0, 1.0),  # swing low 90
    (100.0, 104.0, 91.0, 100.0, 1.0),
    (100.0, 115.0, 92.0, 100.0, 1.0),  # swing high 115
    (100.0, 104.0, 93.0, 100.0, 1.0),
    (100.0, 103.0, 93.0, 100.0, 1.0),
    (100.0, 120.0, 93.0, 100.0, 1.0),
    (100.0, 130.0, 94.0, 100.0, 1.0),  # swing high 130
    (100.0, 120.0, 94.0, 100.0, 1.0),
    (100.0, 103.0, 94.0, 100.0, 1.0),
    (100.0, 104.0, 95.0, 100.0, 1.0),
]
"""Daily bars whose confirmed fractal pivots are exactly lows [90] and highs [115, 130]."""


_HIGH_BETWEEN_CLOSE_AND_OPEN_DAILY = [
    (100.0, 104.0, 92.0, 100.0, 1.0),
    (100.0, 103.0, 91.0, 100.0, 1.0),
    (100.0, 101.5, 90.0, 100.0, 1.0),  # swing low 90
    (100.0, 101.0, 91.0, 100.0, 1.0),
    (100.0, 102.0, 93.0, 100.0, 1.0),  # swing high 102 - sits inside the sweep candle's body
    (100.0, 101.0, 93.0, 100.0, 1.0),
    (100.0, 101.0, 93.0, 100.0, 1.0),
    (100.0, 110.0, 93.0, 100.0, 1.0),
    (100.0, 115.0, 94.0, 100.0, 1.0),  # swing high 115
    (100.0, 110.0, 94.0, 100.0, 1.0),
    (100.0, 103.0, 94.0, 100.0, 1.0),
    (100.0, 104.0, 95.0, 100.0, 1.0),
]
"""Daily bars whose confirmed fractal pivots are exactly lows [90] and highs [102, 115]."""


def test_range_is_anchored_on_the_sweep_candles_body_not_its_close() -> None:
    """The sweep candle opens at 103 and closes at 101 with a confirmed swing high at 102 in
    between. A range read around its *close* is [90, 102], which its open lies outside, so the
    open-inside guard would reject a genuine sweep of 90; the range around its *body* is
    [90, 115] and the sweep stands. This is the single decision that recovered most of the
    planted setups in the integration suite.
    """
    rows = _quiet_h4_rows(30) + [
        (103.0, 103.5, 88.0, 101.0, 1.0),  # idx30: sweep of 90, body [101, 103] straddles 102
        (101.0, 104.0, 100.5, 103.5, 1.0),  # idx31: i=0
        (103.5, 105.0, 103.0, 104.5, 1.0),  # idx32: i=1
        (104.5, 108.0, 104.0, 107.0, 1.0),  # idx33: i=2, BOS (close > 105 neckline)
    ]
    ctx = Context(INSTRUMENT)
    _push_daily_rows(ctx, _HIGH_BETWEEN_CLOSE_AND_OPEN_DAILY)
    strategy = ICT()

    signals = []
    for i, row in enumerate(rows):
        _push_h4_row(ctx, i, row)
        signals.append(strategy.on_bar(ctx))

    assert signals[:-1] == [None] * (len(rows) - 1)
    signal = signals[-1]
    assert signal is not None
    assert signal.direction == 1
    assert signal.entry == pytest.approx(102.0)  # idx30's own body midpoint
    assert signal.stop == pytest.approx(87.99)


def test_a_pullback_candle_that_opened_outside_the_range_is_not_a_sweep_in_on_bar() -> None:
    """The v0.3 pullback case, pinned on the strategy's own path. After a displacement out of
    [90, 115] a pullback candle opens at 118 (outside), wicks to 113 and closes back at 114; a
    bullish candle then supplies an order block and two falling bars break a short's neckline.
    Read against the range around the current close, that candle's high (118.5) would be a
    sweep of 115 and a short would fire on idx34. Its body [114, 118] anchors a range [90, 130]
    instead, which nothing pierces: no signal, in either direction, on any bar. (On this path
    the explicit open-inside check is inert by construction; the body anchor is the guard.)
    """
    rows = _quiet_h4_rows(30) + [
        (112.0, 119.0, 111.5, 118.0, 1.0),  # idx30: displacement closes above 115
        (118.0, 118.5, 113.0, 114.0, 1.0),  # idx31: pullback - opened outside, closed inside
        (112.5, 114.5, 112.0, 114.0, 1.0),  # idx32: i=0 for a would-be short; bullish -> an OB
        (114.0, 114.2, 111.0, 111.5, 1.0),  # idx33: i=1
        (111.5, 112.0, 109.0, 109.5, 1.0),  # idx34: i=2, would break a short's neckline
    ]
    ctx = Context(INSTRUMENT)
    _push_daily_rows(ctx, _THREE_LEVEL_DAILY)
    strategy = ICT()

    signals = []
    for i, row in enumerate(rows):
        _push_h4_row(ctx, i, row)
        signals.append(strategy.on_bar(ctx))

    assert signals == [None] * len(rows)


# --- performance (C1): the Daily swing-level cache must be incremental ----------------------


def _line_tracing_active() -> bool:
    """Whether coverage (or a debugger) is tracing every line, which slows this pure-Python
    loop roughly threefold: `sys.settrace` for the C tracer, `sys.monitoring` for sysmon."""
    return sys.gettrace() is not None or sys.monitoring.get_tool(sys.monitoring.COVERAGE_ID) is not None


def test_on_bar_stays_fast_over_a_multi_year_history() -> None:
    """Pushing eight years of bars (17,520 4H + 2,920 Daily) through `on_bar` must complete
    well under the 5s budget. The Daily series oscillates every ~6 days so several hundred
    pivots accumulate per side: a regression to a linear scan of the level caches for each
    of the up-to-twelve `range_at` lookups per bar measures ~11s here (bisection ~2s), and a
    regression to rebuilding the cache from scratch every bar takes minutes. Under a line
    tracer (`pytest --cov`) bisection itself measures ~5.3s, so the budget is tripled there;
    it still sits below what the linear scan costs when traced."""
    ctx = Context(INSTRUMENT)
    strategy = ICT()
    start = datetime(2020, 1, 1, tzinfo=UTC)
    daily_bars = [
        Bar(
            instrument=INSTRUMENT,
            tf="1d",
            ts_open=start + timedelta(days=i),
            open=100.0 + math.sin(i / 1.0) * 5.0,
            high=100.0 + math.sin(i / 1.0) * 5.0 + 2.0,
            low=100.0 + math.sin(i / 1.0) * 5.0 - 2.0,
            close=100.0 + math.sin(i / 1.0) * 5.0,
            volume=1.0,
        )
        for i in range(2_920)
    ]
    h4_bars = [
        Bar(
            instrument=INSTRUMENT,
            tf="4h",
            ts_open=start + timedelta(hours=4 * i),
            open=100.0 + math.sin(i / 40.0) * 5.0,
            high=100.0 + math.sin(i / 40.0) * 5.0 + 1.0,
            low=100.0 + math.sin(i / 40.0) * 5.0 - 1.0,
            close=100.0 + math.sin(i / 40.0) * 5.0,
            volume=1.0,
        )
        for i in range(17_520)
    ]

    daily_iter = iter(daily_bars)
    next_daily = next(daily_iter, None)
    started = time.perf_counter()
    for h4_bar in h4_bars:
        while next_daily is not None and next_daily.ts_open.date() < h4_bar.ts_open.date():
            ctx.push(next_daily)
            next_daily = next(daily_iter, None)
        ctx.push(h4_bar)
        strategy.on_bar(ctx)
    elapsed = time.perf_counter() - started

    budget = 15.0 if _line_tracing_active() else 5.0
    assert elapsed < budget, f"ICT.on_bar took {elapsed:.2f}s for 17,520 4H bars (budget: {budget:.0f}s)"
