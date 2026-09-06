"""Unit tests for `strategies/levels.py` — Daily swing-level fractal pivots."""

from __future__ import annotations

import numpy as np

from swingforge.strategies.levels import (
    nearest_level_above,
    nearest_level_below,
    swing_highs,
    swing_lows,
)


def _ohlcv(
    closes: list[float], highs: list[float] | None = None, lows: list[float] | None = None
) -> np.ndarray:
    """Build an (n, 5) array; highs/lows default to close +/- 0 (flat bars) unless given."""
    n = len(closes)
    highs = highs if highs is not None else list(closes)
    lows = lows if lows is not None else list(closes)
    out = np.zeros((n, 5), dtype=np.float64)
    for i in range(n):
        out[i] = (closes[i], highs[i], lows[i], closes[i], 1.0)
    return out


# A hand-built fixture: highs rise, peak at index 5, then fall; a clean single-bar pivot.
# index:   0    1    2    3    4    5    6    7    8    9   10
HIGHS = [100, 101, 102, 103, 104, 110, 104, 103, 102, 101, 100]
LOWS = [90, 91, 92, 93, 94, 95, 88, 87, 86, 85, 84]
CLOSES = [95, 96, 97, 98, 99, 100, 96, 95, 94, 93, 92]


def test_swing_highs_finds_the_hand_built_peak() -> None:
    daily = _ohlcv(CLOSES, highs=HIGHS, lows=LOWS)
    highs = swing_highs(daily, k=2)
    # Only bar 5 (110) is strictly greater than the 2 bars on each side (103,104 / 104,103).
    assert highs == [(5, 110.0)]


def test_swing_lows_finds_the_hand_built_trough() -> None:
    daily = _ohlcv(CLOSES, highs=HIGHS, lows=LOWS)
    lows = swing_lows(daily, k=2)
    # Bar 8 (86) is strictly less than the 2 bars each side (88,87 / 85,84)? 84 < 86 so bar 8
    # is NOT a low pivot (84 at bar 10 is lower but has no bars after it within k=2 -> bar 10
    # cannot be confirmed and bar 9 fails since bar 10 (84) < bar 9's low (85)). Only bar 5's
    # low (95) sits in a local rise-then-fall in isolation from the monotonic decline after
    # it, so with this monotonically-falling-after-peak series no low pivot exists except
    # where explicitly checked below with a dedicated up-down-up shape.
    assert lows == []


def test_swing_lows_hand_built_trough_up_down_up() -> None:
    # A clean V-shape: falls to a trough at index 5, then rises - a genuine low pivot.
    lows_arr = [100, 95, 90, 85, 80, 70, 80, 85, 90, 95, 100]
    daily = _ohlcv(lows_arr, highs=[v + 5 for v in lows_arr], lows=lows_arr)
    lows = swing_lows(daily, k=2)
    assert lows == [(5, 70.0)]


def test_no_lookahead_pivot_within_k_of_the_end_is_not_returned() -> None:
    # A peak placed at the very last bar can never be confirmed (no bars after it).
    highs = [100, 101, 102, 103, 104, 110]
    daily = _ohlcv(highs, highs=highs, lows=[h - 5 for h in highs])
    assert swing_highs(daily, k=2) == []
    # Moving the peak one bar earlier still isn't confirmable at k=2 (needs 2 bars after).
    highs2 = [100, 101, 102, 103, 110, 104]
    daily2 = _ohlcv(highs2, highs=highs2, lows=[h - 5 for h in highs2])
    assert swing_highs(daily2, k=2) == []
    # One more bar of history confirms it.
    highs3 = [100, 101, 102, 103, 110, 104, 103]
    daily3 = _ohlcv(highs3, highs=highs3, lows=[h - 5 for h in highs3])
    assert swing_highs(daily3, k=2) == [(4, 110.0)]


def test_swing_highs_too_short_series_returns_empty() -> None:
    daily = _ohlcv([100, 101, 102])
    assert swing_highs(daily, k=2) == []
    assert swing_lows(daily, k=2) == []


def test_nearest_level_above_and_below() -> None:
    levels = [(0, 100.0), (1, 105.0), (2, 110.0), (3, 95.0)]
    assert nearest_level_above(levels, 101.0) == 105.0
    assert nearest_level_below(levels, 101.0) == 100.0


def test_nearest_level_above_returns_none_when_all_at_or_below() -> None:
    levels = [(0, 100.0), (1, 95.0)]
    assert nearest_level_above(levels, 100.0) is None


def test_nearest_level_below_returns_none_when_all_at_or_above() -> None:
    levels = [(0, 100.0), (1, 105.0)]
    assert nearest_level_below(levels, 100.0) is None


def test_nearest_level_helpers_empty_levels() -> None:
    assert nearest_level_above([], 100.0) is None
    assert nearest_level_below([], 100.0) is None
