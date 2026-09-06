"""ICT sweep -> market structure shift -> OB/FVG entry (design spec section 4).

Ported from the corrected SHURIKEN v0.3 `ICTAnalyzer` pipeline
(`docs/superpowers/reference/shuriken_ict_analyzer_v03.py`), with two substitutions: the
*liquidity range* is the nearest Daily swing high/low (`strategies/levels.py`) instead of the
Asia session range, and the execution timeframe is 4H instead of 5m. The three v0.3
regression fixes described in that file's module docstring are preserved verbatim in the
helpers below, since they are exactly what the fixtures in `tests/fixtures/ict_*.json` guard:

* `find_sweep` returns direction ``1`` (long) for a sweep of the range LOW and ``-1`` (short)
  for a sweep of the HIGH — the reverse of the original, pre-v0.3 mapping. It also requires
  the sweep candle to have *opened* inside the range, not just closed there, so a pullback
  candle after a displacement that already left the range is not misread as a fresh sweep in
  the opposite direction.
* `find_bos`'s neckline is the extreme of the candles strictly BEFORE the current one:
  `running_high`/`running_low` is updated *after* the comparison on each iteration, not
  before, so a bar cannot clear its own just-updated extreme by a hair.

One setup fires exactly once: `on_bar` requires the bar completing the break of structure to
be the current (most recently pushed) 4H bar, and remembers `_last_signalled_sweep` so a
sweep already signalled is never re-emitted even if a later bar's scan finds it again.
`_last_signalled_sweep` is kept as an absolute 4H `bar_index`, not a row offset into
`ctx.bars("4h")`: row offsets are reassigned to different bars once trimming rotates the
buffer, so a raw-row comparison could coincidentally match a later, unrelated sweep and
wrongly suppress it. `Context.offset_of` translates it back to a row for the comparison.

The Daily swing-level cache (`_swing_highs_cache`/`_swing_lows_cache`) is incremental: a
pivot at row `i` only becomes confirmable once row `i + k` has closed (see
`strategies/levels.py`), so when the Daily series grows from `old_n` to `new_n` bars, only
the window `[old_n - k, new_n - k)` newly becomes confirmable - the cache is extended by
scanning exactly that window, never rebuilt from the full history. It is keyed on the last
scanned Daily bar's `ts_open`, not on `ctx.bars("1d").shape[0]`, which saturates once the
Daily buffer is trimmed down to `max_bars` and would otherwise stop the cache from ever
advancing again.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

import numpy as np
from pydantic import ValidationError

from swingforge.core.context import Context
from swingforge.core.types import Bar, Signal, round_to_tick
from swingforge.strategies.levels import swing_highs, swing_lows

__all__ = [
    "ICT",
    "find_bos",
    "find_fvg",
    "find_order_block",
    "find_sweep",
]

_SWEEP_LOOKBACK = 12
_MIN_H4_BARS = 30
_BOS_MIN_INDEX = 2  # a BOS candle must be at least this far past the sweep (i >= 2)
_BOS_SWING_WINDOW = 10  # "the next up-to-10 bars" used for the swing_low/high side-value
_FVG_DISPLACEMENT_PAD = 5  # displacement window is [sweep_idx, bos_idx + 5)
_FVG_MIN_SIZE_PCT = 0.0001  # 0.01%


def find_sweep(
    bars4h: np.ndarray,
    level_low: float,
    level_high: float,
    lookback: int,
    min_pierce_pct: float,
) -> tuple[int | None, Literal[1, -1] | None]:
    """Scan the last `lookback` bars, newest-first, for a sweep of `level_low`/`level_high`.

    A bar whose open AND close both lie inside `[level_low, level_high]` and whose low pierces
    below `level_low - pierce` is a sweep of the low -> long (``1``). One whose high pierces
    above `level_high + pierce` is a sweep of the high -> short (``-1``). `pierce` is
    `min_pierce_pct` of the most recent bar's close (the "current price"). The open-inside
    requirement is the v0.3 fix: without it a pullback candle that opened outside the range
    reads as a sweep in the wrong direction.
    """
    n = bars4h.shape[0]
    if n == 0:
        return None, None
    price_ref = float(bars4h[-1, 3])
    min_pierce = price_ref * min_pierce_pct
    start = max(0, n - lookback)
    for i in range(n - 1, start - 1, -1):
        open_, high, low, close = (float(x) for x in bars4h[i, :4])
        inside = level_low <= close <= level_high and level_low <= open_ <= level_high
        if not inside:
            continue
        if low < level_low - min_pierce:
            return i, 1
        if high > level_high + min_pierce:
            return i, -1
    return None, None


def find_bos(
    bars4h: np.ndarray,
    sweep_idx: int,
    direction: Literal[1, -1],
    min_bos_pct: float,
    within: int,
) -> tuple[int | None, float, float]:
    """Find the break of structure after `sweep_idx`, at most `within` bars later.

    The neckline (`running_high` for a long, `running_low` for a short) starts at the sweep
    bar's own extreme and is updated with each post-sweep bar's extreme AFTER that bar has
    been checked against it — the v0.3 fix — so a candle is compared only to the bars
    strictly before it, never to an extreme that already includes its own high/low. `i >= 2`
    (the candle must be at least the third one after the sweep) mirrors the reference
    implementation's neckline confirmation delay.

    Returns `(bos_idx, neckline, swing_extreme)`, where `swing_extreme` is the low (long) or
    high (short) of the up-to-10 bars immediately after the sweep, clamped to what is
    available. `(None, 0.0, 0.0)` when no bar within the window qualifies.
    """
    n = bars4h.shape[0]
    post_start = sweep_idx + 1
    if post_start >= n:
        return None, 0.0, 0.0
    min_delta = float(bars4h[sweep_idx, 3]) * min_bos_pct
    limit = min(within, n - post_start)
    window_end = min(post_start + _BOS_SWING_WINDOW, n)

    if direction == 1:
        swing_low = float(bars4h[post_start:window_end, 2].min())
        running_high = float(bars4h[sweep_idx, 1])
        for i in range(limit):
            idx = post_start + i
            open_, high, low, close = (float(x) for x in bars4h[idx, :4])
            body = abs(close - open_)
            if close > running_high - min_delta and i >= _BOS_MIN_INDEX and body > 0:
                return idx, max(running_high, high), swing_low
            running_high = max(running_high, high)
        return None, 0.0, 0.0

    swing_high = float(bars4h[post_start:window_end, 1].max())
    running_low = float(bars4h[sweep_idx, 2])
    for i in range(limit):
        idx = post_start + i
        open_, high, low, close = (float(x) for x in bars4h[idx, :4])
        body = abs(close - open_)
        if close < running_low + min_delta and i >= _BOS_MIN_INDEX and body > 0:
            return idx, swing_high, min(running_low, low)
        running_low = min(running_low, low)
    return None, 0.0, 0.0


def find_fvg(
    bars4h: np.ndarray, start: int, end: int, direction: Literal[1, -1]
) -> tuple[float, float] | None:
    """The last (most recent) 3-bar fair-value gap in `[start, end)`, clamped to the array.

    A long gap needs `prev.high < next.low`; a short gap needs `prev.low > next.high`. Only
    gaps larger than `_FVG_MIN_SIZE_PCT` of the bounding price count. Returns `(low, high)`.
    """
    n = bars4h.shape[0]
    start = max(start, 0)
    end = min(end, n)
    best: tuple[float, float] | None = None
    for i in range(start + 1, end - 1):
        prev_high = float(bars4h[i - 1, 1])
        prev_low = float(bars4h[i - 1, 2])
        next_high = float(bars4h[i + 1, 1])
        next_low = float(bars4h[i + 1, 2])
        if direction == 1:
            if prev_high < next_low and (next_low - prev_high) > next_low * _FVG_MIN_SIZE_PCT:
                best = (prev_high, next_low)
        elif prev_low > next_high and (prev_low - next_high) > prev_low * _FVG_MIN_SIZE_PCT:
            best = (next_high, prev_low)
    return best


def _find_ts_row(history: tuple[Bar, ...], ts: datetime) -> int | None:
    """Row (scanning back from the newest) whose `ts_open` is `ts`, or None if it isn't (or
    is no longer, having been trimmed away) present."""
    for i in range(len(history) - 1, -1, -1):
        if history[i].ts_open == ts:
            return i
    return None


def _nearest_above(levels: list[float], price: float) -> float | None:
    above = [level for level in levels if level > price]
    return min(above) if above else None


def _nearest_below(levels: list[float], price: float) -> float | None:
    below = [level for level in levels if level < price]
    return max(below) if below else None


def find_order_block(
    bars4h: np.ndarray, sweep_idx: int, bos_idx: int, direction: Literal[1, -1]
) -> tuple[float, float] | None:
    """The last opposing-colour candle (bearish for a long) with a real body in
    `[sweep_idx, bos_idx]`. Returns `(body_high, body_low)`, or None if there is none."""
    pick: tuple[float, float] | None = None
    for i in range(sweep_idx, bos_idx + 1):
        open_, _high, _low, close = (float(x) for x in bars4h[i, :4])
        is_bearish = close < open_
        opposes = is_bearish if direction == 1 else not is_bearish
        body = abs(close - open_)
        if opposes and body > 0:
            pick = (max(open_, close), min(open_, close))
    return pick


class ICT:
    """Sweep of Daily liquidity -> 4H market structure shift -> OB/FVG limit entry."""

    name = "ict"

    def __init__(
        self,
        k: int = 2,
        mss_within: int = 6,
        expires_in_bars: int = 3,
        min_pierce_pct: float = 0.0003,
        min_bos_pct: float = 0.0002,
    ) -> None:
        self.k = k
        self.mss_within = mss_within
        self.expires_in_bars = expires_in_bars
        self.min_pierce_pct = min_pierce_pct
        self.min_bos_pct = min_bos_pct
        self._last_signalled_sweep: int | None = None
        self.rejected_signals = 0
        self._levels_last_daily_ts: datetime | None = None
        self._levels_marker_ts: datetime | None = None
        self._swing_highs_cache: list[float] = []
        self._swing_lows_cache: list[float] = []

    def reset(self) -> None:
        """Clear per-instance state: forgets the last sweep signalled and the Daily
        swing-level cache."""
        self._last_signalled_sweep = None
        self.rejected_signals = 0
        self._levels_last_daily_ts = None
        self._levels_marker_ts = None
        self._swing_highs_cache = []
        self._swing_lows_cache = []

    def _update_swing_levels(self, ctx: Context) -> None:
        """Extend `_swing_highs_cache`/`_swing_lows_cache` by whatever newly became
        confirmable since the last update - see the module docstring.

        The cache is append-only: a pivot stays available after its Daily bar has been
        trimmed out of the `Context` window, so under a small `max_bars` the liquidity
        range and `structure_target` may anchor on a level a from-scratch scan of the
        retained bars would no longer see. Unreachable at the default `max_bars`.
        """
        history = ctx.history("1d")
        if not history:
            return
        last_ts = history[-1].ts_open
        if last_ts == self._levels_last_daily_ts:
            return  # no new Daily bar since the last update

        daily = ctx.bars("1d")
        n = daily.shape[0]
        k = self.k
        end = n - k  # exclusive: rows [0, end) have k bars after them, i.e. are confirmable

        if self._levels_marker_ts is not None:
            marker_row = _find_ts_row(history, self._levels_marker_ts)
            if marker_row is None:
                # The last-checked candidate has been trimmed away entirely (only possible
                # if a whole `max_bars` window's worth of Daily bars arrived between two
                # calls) - bounded (by max_bars, not by total history) fallback: everything
                # before it is necessarily gone too, so rebuild from what's left.
                self._swing_highs_cache = [price for _, price in swing_highs(daily, k)]
                self._swing_lows_cache = [price for _, price in swing_lows(daily, k)]
                self._levels_marker_ts = history[end - 1].ts_open if end > 0 else None
                self._levels_last_daily_ts = last_ts
                return
            start = marker_row + 1
        else:
            start = k

        if end > start:
            highs_col = daily[:, 1]
            lows_col = daily[:, 2]
            for i in range(start, end):
                window_hi = highs_col[i - k : i + k + 1]
                pivot_hi = highs_col[i]
                if pivot_hi == window_hi.max() and np.count_nonzero(window_hi == pivot_hi) == 1:
                    self._swing_highs_cache.append(float(pivot_hi))
                window_lo = lows_col[i - k : i + k + 1]
                pivot_lo = lows_col[i]
                if pivot_lo == window_lo.min() and np.count_nonzero(window_lo == pivot_lo) == 1:
                    self._swing_lows_cache.append(float(pivot_lo))
            self._levels_marker_ts = history[end - 1].ts_open

        self._levels_last_daily_ts = last_ts

    def on_bar(self, ctx: Context) -> Signal | None:
        h4 = ctx.bars("4h")
        daily = ctx.bars("1d")
        if h4.shape[0] < _MIN_H4_BARS or daily.shape[0] < 2 * self.k + 3:
            return None

        close = float(h4[-1, 3])
        self._update_swing_levels(ctx)
        level_high = _nearest_above(self._swing_highs_cache, close)
        level_low = _nearest_below(self._swing_lows_cache, close)
        if level_high is None or level_low is None:
            return None

        sweep_idx, direction = find_sweep(h4, level_low, level_high, _SWEEP_LOOKBACK, self.min_pierce_pct)
        if sweep_idx is None or direction is None:
            return None
        # `_last_signalled_sweep` is an absolute 4H bar_index (trimming-safe); translate it
        # back to a row via `offset_of` rather than comparing raw rows, which trimming
        # reassigns to different bars over time - a raw-row comparison could coincidentally
        # match a fresh, unrelated sweep and wrongly suppress it.
        last_signalled_row = (
            ctx.offset_of(self._last_signalled_sweep) if self._last_signalled_sweep is not None else None
        )
        if sweep_idx == last_signalled_row:
            return None

        bos_idx, _neckline, _swing_extreme = find_bos(
            h4, sweep_idx, direction, self.min_bos_pct, self.mss_within
        )
        if bos_idx is None:
            return None
        current_idx = h4.shape[0] - 1
        if bos_idx != current_idx:
            return None

        # Confirmed setup: never re-emit for this sweep again, whatever happens below. Stored
        # as an absolute bar_index so it survives trimming (see the comment above).
        self._last_signalled_sweep = ctx.bar_index - (h4.shape[0] - 1 - sweep_idx)

        fvg_end = min(bos_idx + _FVG_DISPLACEMENT_PAD, h4.shape[0])
        ob = find_order_block(h4, sweep_idx, bos_idx, direction)
        fvg = find_fvg(h4, sweep_idx, fvg_end, direction)

        if ob is not None:
            entry = (ob[0] + ob[1]) / 2
            tag = "ict_sweep_mss_ob"
        elif fvg is not None:
            entry = (fvg[0] + fvg[1]) / 2
            tag = "ict_sweep_mss_fvg"
        else:
            return None

        tick = ctx.instrument.tick_size
        tick_f = float(tick)
        if direction == 1:
            stop = float(h4[sweep_idx, 2]) - tick_f
            structure_target = _nearest_above(self._swing_highs_cache, entry)
        else:
            stop = float(h4[sweep_idx, 1]) + tick_f
            structure_target = _nearest_below(self._swing_lows_cache, entry)

        target = round_to_tick(structure_target, tick) if structure_target is not None else None
        try:
            return Signal(
                direction=direction,
                entry=round_to_tick(entry, tick),
                stop=round_to_tick(stop, tick),
                structure_target=target,
                tag=tag,
                expires_in_bars=self.expires_in_bars,
            )
        except ValidationError:
            self.rejected_signals += 1
            return None
