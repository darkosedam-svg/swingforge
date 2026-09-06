"""Per-instrument market state the engine feeds and strategies/exit rules read.

`Context` is the one deliberately mutable object in `core`: the engine pushes each closed
bar into it, and everything downstream reads. It holds one series per timeframe (1H
sub-bars, 4H execution, Daily bias), the current position, and Wilder's ATR and ADX.

Only the most recent `max_bars` bars per timeframe are retained; `bar_index` counts 4H bars
pushed since the run started and is unaffected by that trimming, so it stays a stable
reference for order expiry and time stops. Use `offset_of` to turn an absolute `bar_index`
back into a row of `bars("4h")`.

Reads are O(1) per bar: each timeframe owns a preallocated `(max_bars, 5)` buffer that
`push` writes one row of, and `atr`/`adx` keep incremental Wilder state per `(tf, n)`, so a
run that reads them on every bar costs one smoothing step per bar rather than a rebuild.
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from typing import Any

import numpy as np

from swingforge.core.types import TF, Bar, Instrument, Position

__all__ = ["Context"]

_TIMEFRAMES: tuple[TF, ...] = ("1h", "4h", "1d")
_EXECUTION_TF: TF = "4h"
_SNAPSHOT_CLOSES = 20
_EMPTY = np.empty((0, 5), dtype=np.float64)
_EMPTY.flags.writeable = False


def _true_range(ohlcv: np.ndarray) -> np.ndarray:
    """Wilder's true range for every bar after the first: shape (n - 1,)."""
    high = ohlcv[1:, 1]
    low = ohlcv[1:, 2]
    prev_close = ohlcv[:-1, 3]
    return np.maximum(
        high - low,
        np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)),
    )


def _wilder_rma(values: np.ndarray, n: int) -> np.ndarray:
    """Wilder's smoothing, seeded with the simple average of the first `n` values.

    Returns an array the same length as `values`, with NaN in the first `n - 1` slots where
    the average is not yet defined.
    """
    out = np.full(values.shape, np.nan, dtype=np.float64)
    if values.size < n:
        return out
    acc = float(values[:n].mean())
    out[n - 1] = acc
    for i in range(n, values.size):
        acc = (acc * (n - 1) + float(values[i])) / n
        out[i] = acc
    return out


def _rma_step(previous: float, value: float, n: int) -> float:
    """One Wilder smoothing step - the recurrence `_wilder_rma` applies per bar."""
    return (previous * (n - 1) + value) / n


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray, scale: float) -> np.ndarray:
    """`scale * numerator / denominator`, yielding 0.0 wherever the denominator is <= 0."""
    positive = denominator > 0
    safe = np.where(positive, denominator, 1.0)
    return np.where(positive, scale * numerator / safe, 0.0)


def _ratio(numerator: float, denominator: float, scale: float) -> float:
    """Scalar `_safe_ratio`: 0.0 when the denominator is <= 0."""
    return scale * numerator / denominator if denominator > 0 else 0.0


def _or_null(value: float) -> float | None:
    """NaN becomes None, so a snapshot is plain JSON with no `NaN` literal in it."""
    return None if math.isnan(value) else value


class Context:
    """Mutable market state for one instrument.

    The engine owns the mutation (`push`, `position`); strategies and exit rules only read.
    """

    def __init__(self, instrument: Instrument, max_bars: int = 50_000) -> None:
        if max_bars < 1:
            raise ValueError("max_bars must be >= 1")
        self.instrument = instrument
        self.max_bars = max_bars
        self.bar_index: int = -1
        self._position: Position | None = None
        self._series: dict[TF, list[Bar]] = {tf: [] for tf in _TIMEFRAMES}
        self._buffers: dict[TF, np.ndarray | None] = {tf: None for tf in _TIMEFRAMES}
        self._views: dict[TF, np.ndarray | None] = {tf: None for tf in _TIMEFRAMES}
        self._counts: dict[TF, int] = {tf: 0 for tf in _TIMEFRAMES}
        self._pushes: dict[TF, int] = {tf: 0 for tf in _TIMEFRAMES}
        self._atr_state: dict[tuple[TF, int], tuple[int, float]] = {}
        self._adx_state: dict[tuple[TF, int], tuple[int, float, float, float, float]] = {}

    # --- mutation (engine only) -------------------------------------------

    @property
    def position(self) -> Position | None:
        """Open exposure in this context's instrument, or None while flat."""
        return self._position

    @position.setter
    def position(self, value: Position | None) -> None:
        """Assign the current position; `None` clears it.

        A position in a different instrument is rejected: a context describes exactly one
        instrument, and silently holding a foreign position would corrupt every sizing and
        exit decision downstream.
        """
        if value is not None and value.instrument != self.instrument:
            raise ValueError(
                f"position in {value.instrument.venue}:{value.instrument.symbol} assigned to a "
                f"context for {self.instrument.venue}:{self.instrument.symbol}"
            )
        self._position = value

    def push(self, bar: Bar) -> None:
        """Append a closed bar to the series for its own timeframe.

        A 4H bar advances `bar_index`; 1H and Daily bars do not. Once a timeframe holds
        `max_bars` bars the oldest is dropped, which invalidates every row offset previously
        handed out for that timeframe.
        """
        if bar.instrument != self.instrument:
            raise ValueError(
                f"bar for {bar.instrument.venue}:{bar.instrument.symbol} pushed into a context "
                f"for {self.instrument.venue}:{self.instrument.symbol}"
            )
        tf = bar.tf
        series = self._series[tf]
        series.append(bar)
        if len(series) > self.max_bars:
            del series[: len(series) - self.max_bars]

        buffer = self._buffers[tf]
        if buffer is None:
            buffer = np.empty((self.max_bars, 5), dtype=np.float64)
            self._buffers[tf] = buffer
        count = self._counts[tf]
        if count == self.max_bars:
            buffer[:-1] = buffer[1:]
            row = count - 1
        else:
            row = count
            self._counts[tf] = count + 1
            self._views[tf] = None
        buffer[row] = (bar.open, bar.high, bar.low, bar.close, bar.volume)

        self._pushes[tf] += 1
        if tf == _EXECUTION_TF:
            self.bar_index += 1

    # --- reads -------------------------------------------------------------

    def bars(self, tf: TF) -> np.ndarray:
        """Retained bars as a read-only (n, 5) float64 array: open, high, low, close, volume.

        Oldest first; shape (0, 5) when the timeframe has no bars yet. The array is a
        read-only view onto the timeframe's buffer, reused across calls and only valid until
        the next `push` - copy it to keep or to modify it.

        Positional indexing by `bar_index` is invalid after trimming, since row 0 is then no
        longer bar 0: translate the index with `offset_of` first.
        """
        try:
            view = self._views[tf]
        except KeyError:
            raise ValueError(f"unknown timeframe {tf!r}; expected one of {_TIMEFRAMES}") from None
        if view is not None:
            return view
        buffer = self._buffers[tf]
        count = self._counts[tf]
        if buffer is None or count == 0:
            self._views[tf] = _EMPTY
            return _EMPTY
        view = buffer[:count]
        view.flags.writeable = False
        self._views[tf] = view
        return view

    def offset_of(self, bar_index: int) -> int | None:
        """Row offset into `bars("4h")` for an absolute 4H `bar_index`.

        None when that bar has already been trimmed away, or has not happened yet. Absolute
        bar indices survive trimming and row offsets do not, so orders and time stops carry
        the index and translate here when they need the row.
        """
        count = self._counts[_EXECUTION_TF]
        if count == 0 or bar_index > self.bar_index:
            return None
        offset = bar_index - (self.bar_index - count + 1)
        return offset if offset >= 0 else None

    def last(self, tf: TF) -> Bar | None:
        series = self._series_for(tf)
        return series[-1] if series else None

    def history(self, tf: TF) -> tuple[Bar, ...]:
        """The retained `Bar` objects for a timeframe, oldest first."""
        return tuple(self._series_for(tf))

    def ts(self) -> datetime | None:
        """`ts_open` of the most recent 4H bar, or None before any has been pushed."""
        last = self.last(_EXECUTION_TF)
        return last.ts_open if last is not None else None

    def snapshot(self) -> bytes:
        """This context's state as UTF-8 JSON - the producer of `Trade.context_snapshot`.

        Keys: `bar_index`, `ts` (ISO-8601, or null before the first 4H bar), `atr_4h`,
        `atr_1d`, `adx_4h`, `adx_1d`, and `closes_4h` (the last 20 4H closes, oldest first).
        An indicator that is not yet computable serialises as null rather than NaN, so the
        payload is plain JSON that any reader can parse.
        """
        ts = self.ts()
        closes = self.bars(_EXECUTION_TF)[-_SNAPSHOT_CLOSES:, 3]
        payload: dict[str, Any] = {
            "bar_index": self.bar_index,
            "ts": ts.isoformat() if ts is not None else None,
            "atr_4h": _or_null(self.atr("4h")),
            "atr_1d": _or_null(self.atr("1d")),
            "adx_4h": _or_null(self.adx("4h")),
            "adx_1d": _or_null(self.adx("1d")),
            "closes_4h": [float(close) for close in closes],
        }
        return json.dumps(payload).encode("utf-8")

    # --- indicators --------------------------------------------------------

    def atr(self, tf: TF, n: int = 14) -> float:
        """Wilder's ATR(n): the RMA of true range, seeded with the mean of the first n.

        NaN until there are at least `n + 1` bars, since the first true range needs a
        previous close. The smoothed value is memoised per `(tf, n)`: a repeat read on the
        same bar is free, and the next bar costs one smoothing step.
        """
        if n < 1:
            raise ValueError("n must be >= 1")
        ohlcv = self.bars(tf)
        if ohlcv.shape[0] < n + 1:
            return float("nan")

        key = (tf, n)
        pushes = self._pushes[tf]
        state = self._atr_state.get(key)
        if state is not None:
            seen, value = state
            if seen == pushes:
                return value
            if seen == pushes - 1:
                high = float(ohlcv[-1, 1])
                low = float(ohlcv[-1, 2])
                prev_close = float(ohlcv[-2, 3])
                true_range = max(high - low, abs(high - prev_close), abs(low - prev_close))
                value = _rma_step(value, true_range, n)
                self._atr_state[key] = (pushes, value)
                return value

        value = float(_wilder_rma(_true_range(ohlcv), n)[-1])
        self._atr_state[key] = (pushes, value)
        return value

    def adx(self, tf: TF, n: int = 14) -> float:
        """Wilder's ADX(n): RMA of DX, where DX comes from RMA-smoothed +DM/-DM over TR.

        NaN until there are at least `2n + 1` bars: n to seed the directional indicators and
        n more DX values to seed their average. A flat series has no directional movement,
        so both DI legs are zero and the ADX is 0.0, not NaN. The smoothed TR, +DM and -DM
        and the ADX itself are memoised per `(tf, n)`, as for `atr`.
        """
        if n < 1:
            raise ValueError("n must be >= 1")
        ohlcv = self.bars(tf)
        if ohlcv.shape[0] < 2 * n + 1:
            return float("nan")

        key = (tf, n)
        pushes = self._pushes[tf]
        state = self._adx_state.get(key)
        if state is not None:
            seen, tr_s, plus_s, minus_s, value = state
            if seen == pushes:
                return value
            if seen == pushes - 1:
                high = float(ohlcv[-1, 1])
                low = float(ohlcv[-1, 2])
                prev_high = float(ohlcv[-2, 1])
                prev_low = float(ohlcv[-2, 2])
                prev_close = float(ohlcv[-2, 3])
                up = high - prev_high
                down = prev_low - low
                true_range = max(high - low, abs(high - prev_close), abs(low - prev_close))
                tr_s = _rma_step(tr_s, true_range, n)
                plus_s = _rma_step(plus_s, up if (up > down and up > 0) else 0.0, n)
                minus_s = _rma_step(minus_s, down if (down > up and down > 0) else 0.0, n)
                plus_di = _ratio(plus_s, tr_s, 100.0)
                minus_di = _ratio(minus_s, tr_s, 100.0)
                dx = _ratio(abs(plus_di - minus_di), plus_di + minus_di, 100.0)
                value = _rma_step(value, dx, n)
                self._adx_state[key] = (pushes, tr_s, plus_s, minus_s, value)
                return value

        high_series = ohlcv[:, 1]
        low_series = ohlcv[:, 2]
        up_moves = high_series[1:] - high_series[:-1]
        down_moves = low_series[:-1] - low_series[1:]
        plus_dm = np.where((up_moves > down_moves) & (up_moves > 0), up_moves, 0.0)
        minus_dm = np.where((down_moves > up_moves) & (down_moves > 0), down_moves, 0.0)

        tr_rma = _wilder_rma(_true_range(ohlcv), n)
        plus_rma = _wilder_rma(plus_dm, n)
        minus_rma = _wilder_rma(minus_dm, n)

        seeded = slice(n - 1, None)
        tr_seeded = tr_rma[seeded]
        plus_di_series = _safe_ratio(plus_rma[seeded], tr_seeded, 100.0)
        minus_di_series = _safe_ratio(minus_rma[seeded], tr_seeded, 100.0)
        dx_series = _safe_ratio(
            np.abs(plus_di_series - minus_di_series), plus_di_series + minus_di_series, 100.0
        )
        value = float(_wilder_rma(dx_series, n)[-1])
        self._adx_state[key] = (
            pushes,
            float(tr_rma[-1]),
            float(plus_rma[-1]),
            float(minus_rma[-1]),
            value,
        )
        return value

    # --- internals ---------------------------------------------------------

    def _series_for(self, tf: TF) -> list[Bar]:
        try:
            return self._series[tf]
        except KeyError:
            raise ValueError(f"unknown timeframe {tf!r}; expected one of {_TIMEFRAMES}") from None
