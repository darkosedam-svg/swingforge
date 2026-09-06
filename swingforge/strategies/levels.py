"""Daily swing-level helpers shared by the ICT and Zones entry strategies.

A swing high/low is a fractal pivot: bar ``i`` qualifies only when its high (low) is
strictly greater (less) than the highs (lows) of the ``k`` bars on each side. Because that
comparison needs the ``k`` bars *after* ``i`` to already exist, a pivot at ``i`` only becomes
knowable once bar ``i + k`` has closed — the loop bound below (``i`` ranges up to
``n - 1 - k``) enforces exactly that, so these functions never look ahead of the array
they are given.
"""

from __future__ import annotations

import numpy as np

__all__ = ["nearest_level_above", "nearest_level_below", "swing_highs", "swing_lows"]


def swing_highs(daily: np.ndarray, k: int = 2) -> list[tuple[int, float]]:
    """Fractal swing highs in a Daily ``(n, 5)`` OHLCV array, oldest-first.

    Bar ``i`` is a swing high when ``high[i]`` is strictly greater than every high in the
    ``k`` bars immediately before and after it. Only pivots with ``i + k <= n - 1`` are
    returned, so a level is never reported before the bars that confirm it have closed.
    """
    return _swing_pivots(daily[:, 1], k, find_high=True)


def swing_lows(daily: np.ndarray, k: int = 2) -> list[tuple[int, float]]:
    """Fractal swing lows in a Daily ``(n, 5)`` OHLCV array, oldest-first. See `swing_highs`."""
    return _swing_pivots(daily[:, 2], k, find_high=False)


def _swing_pivots(series: np.ndarray, k: int, *, find_high: bool) -> list[tuple[int, float]]:
    n = series.shape[0]
    out: list[tuple[int, float]] = []
    if k < 1 or n < 2 * k + 1:
        return out
    for i in range(k, n - k):
        pivot = series[i]
        window = series[i - k : i + k + 1]
        if find_high:
            if pivot == window.max() and np.count_nonzero(window == pivot) == 1:
                out.append((i, float(pivot)))
        else:
            if pivot == window.min() and np.count_nonzero(window == pivot) == 1:
                out.append((i, float(pivot)))
    return out


def nearest_level_above(levels: list[tuple[int, float]], price: float) -> float | None:
    """The closest level strictly above `price`, or None if every level is at or below it."""
    above = [level for _, level in levels if level > price]
    return min(above) if above else None


def nearest_level_below(levels: list[tuple[int, float]], price: float) -> float | None:
    """The closest level strictly below `price`, or None if every level is at or above it."""
    below = [level for _, level in levels if level < price]
    return max(below) if below else None
