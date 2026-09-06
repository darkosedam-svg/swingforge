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

A zone stays `fresh` until the first 4H bar whose range overlaps it; that first touch always
consumes freshness, whether or not it also produces a signal (a touch without a close inside
the zone still uses it up). Only the newest 20 zones are kept.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
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


@dataclass
class _Zone:
    direction: Literal[1, -1]  # 1 = demand (long), -1 = supply (short)
    low: float
    high: float
    fresh: bool = True


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
        self._atr_value: float = math.nan
        self.rejected_signals = 0

    def reset(self) -> None:
        """Clear per-instance state: forgets every zone and rescans from scratch."""
        self._zones = []
        self._daily_last_ts = None
        self._atr_value = math.nan
        self.rejected_signals = 0

    def on_bar(self, ctx: Context) -> Signal | None:
        self._detect_new_zones(ctx)

        h4 = ctx.bars("4h")
        if h4.shape[0] == 0:
            return None
        _, bar_high, bar_low, bar_close, _ = (float(x) for x in h4[-1])

        signal: Signal | None = None
        for zone in self._zones:
            if not zone.fresh:
                continue
            if not (bar_low <= zone.high and bar_high >= zone.low):
                continue
            zone.fresh = False  # any touch consumes freshness, signal or not
            if signal is None and zone.low <= bar_close <= zone.high:
                signal = self._build_signal(zone, ctx)
        return signal

    # --- zone detection ------------------------------------------------------

    def _detect_new_zones(self, ctx: Context) -> None:
        history = ctx.history("1d")
        if not history:
            return
        last_ts = history[-1].ts_open
        if last_ts == self._daily_last_ts:
            return  # nothing new since the last scan

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
            zone = self._find_opposing_candle(daily, impulse_start, direction)
            if zone is None:
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
        self, daily: np.ndarray, impulse_start: int, direction: Literal[1, -1]
    ) -> _Zone | None:
        """The nearest candle at or before `impulse_start` whose colour opposes `direction`.

        The impulse leg is the bars AFTER the start bar, so the start bar itself is
        legitimate zone material - it may be the opposing candle returned here.
        """
        for i in range(impulse_start, -1, -1):
            open_, high, low, close = (float(x) for x in daily[i, :4])
            is_bearish = close < open_
            opposes = is_bearish if direction == 1 else not is_bearish
            if opposes:
                return _Zone(direction=direction, low=low, high=high)
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
