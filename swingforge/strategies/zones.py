"""Supply/demand zone entries (design spec section 4).

A Daily *impulse* is a close-to-close move of at least `impulse_atr_mult` Daily ATR(14)s,
completed within at most `impulse_bars` Daily bars; the *zone* is the last opposing-colour
Daily candle at or before the impulse's start bar (its full `[low, high]` range, not just the
body). Zones are detected only from Daily bars that have already closed, and only once per
newly closed Daily bar — using that bar's own causal ATR(14) reading, computed locally below
so the impulse a bar's ATR judges is always the ATR *as of that bar*, not a later one.

The local Wilder ATR(14) series is incremental: `_atr_value` is seeded once (the simple mean
of the first 14 true ranges) and then carried forward one Wilder step per newly-scanned Daily
bar — it is never rebuilt from the full history, so detection stays cheap no matter how many
Daily bars have accumulated. Which Daily bars are "new" is tracked by `_daily_last_ts` (the
`ts_open` of the last Daily bar already scanned), not a row count: a row count saturates once
the Daily buffer has been trimmed down to `max_bars`, at which point it would never again
compare greater than the (now-constant) buffer length and detection would silently stop.

A zone stays `fresh` until the first 4H bar that opened at or after it was born whose range
overlaps it; that first touch always consumes freshness, whether or not it also produces a
signal (a touch without a close inside the zone still uses it up). Only the newest 20 zones
are kept, and a zone identical to one already held - the same opposing candle re-selected
because the next Daily bar extends the same impulse - is not appended again and does not
regain freshness while the original is still held (on real data almost half of all
detections were such copies, evicting genuinely fresh zones through the cap).

Freshness is a property of the bars, not of when the engine asked. `Engine` consults a
strategy only when it is flat, has no pending entry, the kill switch is off and the session
filter admits the bar (design spec section 4: the session filter gates *entries* only), so
`on_bar` first replays every 4H bar it has not yet seen (`_consume_unseen_touches`) exactly as
it would have processed them live - an overlap consumes freshness, and a close inside the zone
consumes it without a signal, that bar's entry being gone. A zone is born when its Daily bar
closes (`_Zone.born_ts`), and only a 4H bar that opened at or after that instant can touch it:
the opposing candle's own 4H bars overlap the zone by construction and must not burn it.
Measured on the 2026-09-07 Hyperliquid run before this: 38 (BTC) / 47 (SOL) fresh zones
overlapped by session-blocked bars under 'active', which then out-traded 'none'. The
tournament's 84-day warm-up is blocked for every session mode (`run_config` refuses bars
before `start`), so its 4H bars now consume freshness too and the 'none' arm shifts as well;
that is intended - a zone price visited during warm-up was never fresh at `start`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal

import numpy as np
from pydantic import ValidationError

from swingforge.core.context import Context
from swingforge.core.types import Bar, Instrument, Signal, round_to_tick
from swingforge.strategies.levels import (
    nearest_level_above,
    nearest_level_below,
    swing_highs,
    swing_lows,
)

__all__ = ["Zones"]

_MAX_ZONES = 20
_ATR_N = 14
_LEVELS_K = 2
_DAILY = timedelta(hours=24)
"""A Daily bar's span: a zone is born when the Daily bar that completed its impulse closes.

NOTE: duplicated from the canonical `_TF_SPAN["1d"]` in `swingforge/core/types.py` (as the
adapters do) rather than importing a private name; keep them in sync."""


@dataclass
class _Zone:
    direction: Literal[1, -1]  # 1 = demand (long), -1 = supply (short)
    low: float
    high: float
    born_ts: datetime  # the close of the Daily bar that completed the impulse
    source_ts: datetime  # `ts_open` of the opposing Daily candle the zone is cut from
    fresh: bool = True

    def same_as(self, other: _Zone) -> bool:
        """The same opposing candle, cut in the same direction. Identity is the candle, not
        its prices: two distinct candles with equal extremes (tick-rounded FX, dojis) are two
        zones, while the next Daily bar re-selecting this candle is not a new one."""
        return self.direction == other.direction and self.source_ts == other.source_ts

    def touched_by(self, ts_open: datetime, bar_low: float, bar_high: float) -> bool:
        """Whether a 4H bar opening at `ts_open` with this range overlaps the zone - only a
        bar that opened at or after the zone was born can (the first 4H bar the strategy
        sees after a Daily close opens exactly at `born_ts`, hence `>=`)."""
        return ts_open >= self.born_ts and bar_low <= self.high and bar_high >= self.low


def _true_range(ohlcv: np.ndarray) -> np.ndarray:
    high = ohlcv[1:, 1]
    low = ohlcv[1:, 2]
    prev_close = ohlcv[:-1, 3]
    return np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))


def _find_ts_row(history: tuple[Bar, ...], ts: datetime) -> int | None:
    """Row (scanning back from the newest) whose `ts_open` is `ts`, or None if it isn't (or
    is no longer, having been trimmed away) present."""
    for i in range(len(history) - 1, -1, -1):
        if history[i].ts_open == ts:
            return i
    return None


class Zones:
    """Entries on a fresh touch of a Daily supply/demand zone."""

    name = "zones"

    def __init__(
        self,
        impulse_atr_mult: float = 2.0,
        impulse_bars: int = 3,
        stop_atr_pad: float = 0.25,
        round_number_filter: bool = False,
        expires_in_bars: int = 3,
    ) -> None:
        self.impulse_atr_mult = impulse_atr_mult
        self.impulse_bars = impulse_bars
        self.stop_atr_pad = stop_atr_pad
        self.round_number_filter = round_number_filter
        self.expires_in_bars = expires_in_bars
        self._zones: list[_Zone] = []
        self._daily_last_ts: datetime | None = None
        self._h4_last_index: int = -1
        self._atr_value: float = math.nan
        self.rejected_signals = 0

    def reset(self) -> None:
        """Clear per-instance state: forgets every zone and rescans from scratch."""
        self._zones = []
        self._daily_last_ts = None
        self._h4_last_index = -1
        self._atr_value = math.nan
        self.rejected_signals = 0

    def on_bar(self, ctx: Context) -> Signal | None:
        self._detect_new_zones(ctx)

        current = ctx.last("4h")
        if current is None:
            return None
        if ctx.bar_index != self._h4_last_index + 1:
            self._consume_unseen_touches(ctx)
        self._h4_last_index = ctx.bar_index

        signal: Signal | None = None
        for zone in self._zones:
            if not zone.fresh or not zone.touched_by(current.ts_open, current.low, current.high):
                continue
            zone.fresh = False  # any touch consumes freshness, signal or not
            if signal is None and zone.low <= current.close <= zone.high:
                signal = self._build_signal(zone, ctx)
        return signal

    def _consume_unseen_touches(self, ctx: Context) -> None:
        """Apply every 4H bar between the last one `on_bar` saw and the current one to the
        zones, exactly as `on_bar` would have: an overlap consumes freshness, and a close
        inside the zone consumes it without a signal (see the module docstring).

        Only called across a gap in consults (`_h4_last_index` is an absolute 4H bar index,
        so the check is O(1) and the O(n) `ctx.history("4h")` copy is never paid on the every-bar
        path). The first call replays whatever 4H history the Context retains, so a zone
        price already visited never signals on the first consult either; if the last seen
        bar has been trimmed out of the window (`offset_of` is None: more than `max_bars`
        4H bars between two consults) the retained history is replayed. `_h4_last_index`
        is bookkeeping, not semantics - replaying already-seen bars is idempotent.
        """
        row = ctx.offset_of(self._h4_last_index) if self._h4_last_index >= 0 else None
        start = 0 if row is None else row + 1
        for bar in ctx.history("4h")[start:-1]:
            for zone in self._zones:
                if zone.fresh and zone.touched_by(bar.ts_open, bar.low, bar.high):
                    zone.fresh = False

    # --- zone detection ------------------------------------------------------

    def _detect_new_zones(self, ctx: Context) -> None:
        newest = ctx.last("1d")
        if newest is None:
            return
        last_ts = newest.ts_open
        if last_ts == self._daily_last_ts:
            return  # nothing new since the last scan - and no O(n) history copy paid

        history = ctx.history("1d")
        daily = ctx.bars("1d")
        n = daily.shape[0]
        closes = daily[:, 3]

        if self._daily_last_ts is None:
            start = 1
        else:
            row = _find_ts_row(history, self._daily_last_ts)
            if row is None:
                # The last-scanned bar has been trimmed away entirely (only possible if a
                # whole `max_bars` window's worth of Daily bars arrived between two calls) -
                # bounded (by max_bars, not by total history) fallback: reseed the local ATR
                # and rescan whatever is currently retained.
                start = 1
                self._atr_value = math.nan
            else:
                start = row + 1

        for end in range(start, n):
            atr = self._advance_atr(daily, end)
            if math.isnan(atr):
                continue
            threshold = self.impulse_atr_mult * atr
            impulse_start = self._find_impulse_start(closes, end, threshold)
            if impulse_start is None:
                continue
            direction: Literal[1, -1] = 1 if closes[end] > closes[impulse_start] else -1
            zone = self._find_opposing_candle(
                daily, history, impulse_start, direction, born_ts=history[end].ts_open + _DAILY
            )
            if zone is None or any(held.same_as(zone) for held in self._zones):
                continue
            self._zones.append(zone)
            if len(self._zones) > _MAX_ZONES:
                del self._zones[: len(self._zones) - _MAX_ZONES]

        self._daily_last_ts = last_ts

    def _advance_atr(self, daily: np.ndarray, end: int) -> float:
        """Wilder ATR(14) as of row `end`, carried forward from `self._atr_value` by exactly
        one Wilder step - seeded once (the simple mean of the first 14 true ranges) and never
        rebuilt from the full history afterwards."""
        if math.isnan(self._atr_value):
            if end < _ATR_N:
                return math.nan
            tr = _true_range(daily[: end + 1])
            self._atr_value = float(tr[:_ATR_N].mean())
            return self._atr_value
        high = float(daily[end, 1])
        low = float(daily[end, 2])
        prev_close = float(daily[end - 1, 3])
        true_range = max(high - low, abs(high - prev_close), abs(low - prev_close))
        self._atr_value = (self._atr_value * (_ATR_N - 1) + true_range) / _ATR_N
        return self._atr_value

    def _find_impulse_start(self, closes: np.ndarray, end: int, threshold: float) -> int | None:
        """Nearest start `i0` (within `impulse_bars` before `end`) whose close-to-close move
        into `end` already meets `threshold`; `None` if no window that short qualifies."""
        earliest = max(0, end - self.impulse_bars)
        for i0 in range(end - 1, earliest - 1, -1):
            if abs(closes[end] - closes[i0]) >= threshold:
                return i0
        return None

    def _find_opposing_candle(
        self,
        daily: np.ndarray,
        history: tuple[Bar, ...],
        impulse_start: int,
        direction: Literal[1, -1],
        *,
        born_ts: datetime,
    ) -> _Zone | None:
        """The nearest candle at or before `impulse_start` whose colour opposes `direction`,
        as a zone born at `born_ts` (the close of the Daily bar that completed the impulse).

        The impulse leg is the bars AFTER the start bar, so the start bar itself is
        legitimate zone material - it may be the opposing candle returned here.
        """
        for i in range(impulse_start, -1, -1):
            open_, high, low, close = (float(x) for x in daily[i, :4])
            is_bearish = close < open_
            opposes = is_bearish if direction == 1 else not is_bearish
            if opposes:
                return _Zone(
                    direction=direction, low=low, high=high, born_ts=born_ts, source_ts=history[i].ts_open
                )
        return None

    # --- signal construction ---------------------------------------------------

    def _build_signal(self, zone: _Zone, ctx: Context) -> Signal | None:
        atr = ctx.atr("1d", _ATR_N)
        if math.isnan(atr):
            return None
        mid = (zone.low + zone.high) / 2
        if self.round_number_filter and not self._passes_round_filter(mid, ctx.instrument):
            return None

        if zone.direction == 1:
            stop = zone.low - self.stop_atr_pad * atr
        else:
            stop = zone.high + self.stop_atr_pad * atr

        daily = ctx.bars("1d")
        if zone.direction == 1:
            structure_target = nearest_level_above(swing_highs(daily, _LEVELS_K), mid)
        else:
            structure_target = nearest_level_below(swing_lows(daily, _LEVELS_K), mid)

        tick = ctx.instrument.tick_size
        target = round_to_tick(structure_target, tick) if structure_target is not None else None
        tag = "zone_demand" if zone.direction == 1 else "zone_supply"
        try:
            return Signal(
                direction=zone.direction,
                entry=round_to_tick(mid, tick),
                stop=round_to_tick(stop, tick),
                structure_target=target,
                tag=tag,
                expires_in_bars=self.expires_in_bars,
            )
        except ValidationError:
            self.rejected_signals += 1
            return None

    @staticmethod
    def _passes_round_filter(mid: float, instrument: Instrument) -> bool:
        if instrument.session_profile == "fx":
            step = 0.0050
            tolerance = 0.0005
            if instrument.quote_ccy == "JPY":
                step *= 100
                tolerance *= 100
            nearest = round(mid / step) * step
            return abs(mid - nearest) <= tolerance
        # perp: round levels are multiples of `step = 10 ** (floor(log10(price)) - 1)` and
        # of `step / 2` (e.g. BTC ~64,200 -> step 1,000, rounding to the nearest 500; ETH
        # ~3,200 -> step 100, rounding to the nearest 50); tolerance is 0.1% of price.
        if mid <= 0:
            return False
        exponent = math.floor(math.log10(mid))
        step = 10.0 ** (exponent - 1)
        granularity = step / 2
        tolerance = mid * 0.001
        nearest = round(mid / granularity) * granularity
        return abs(mid - nearest) <= tolerance
