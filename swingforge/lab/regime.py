"""Regime tag attached to every trade at entry: trend/range crossed with a vol tercile.

Diagnostic only - nothing here gates or sizes a trade. The exit question flips by regime
(fixed targets suit mean reversion, trails and partials suit trend continuation), so the
run report breaks every config down by these six labels and the tag is what makes that
breakdown possible.

Structure comes from Wilder's Daily ADX(14): above the threshold the market is trending,
at or below it - and when the ADX is not yet computable - it is ranging. Volatility is the
tercile of the current 20-day realized vol *within this instrument's own history*, so
"highvol" means high for this instrument rather than high against some cross-market
constant.
"""

from __future__ import annotations

from typing import Literal, get_args

import numpy as np

from swingforge.core.context import Context

__all__ = ["REGIMES", "Regime", "VolTercile", "realized_vol_series", "tag", "vol_tercile"]

VolTercile = Literal["lowvol", "midvol", "highvol"]
"""Where the current realized vol sits in this instrument's own history of it."""

Regime = Literal[
    "trend_lowvol",
    "trend_midvol",
    "trend_highvol",
    "range_lowvol",
    "range_midvol",
    "range_highvol",
]
"""The six tags of spec section 6: structure crossed with volatility tercile."""

REGIMES: tuple[Regime, ...] = get_args(Regime)
"""Every label `tag` can return, for report tables and exhaustiveness checks."""

_LOWER_TERCILE = 100.0 / 3.0
_UPPER_TERCILE = 200.0 / 3.0

_LABELS: dict[tuple[bool, VolTercile], Regime] = {
    (True, "lowvol"): "trend_lowvol",
    (True, "midvol"): "trend_midvol",
    (True, "highvol"): "trend_highvol",
    (False, "lowvol"): "range_lowvol",
    (False, "midvol"): "range_midvol",
    (False, "highvol"): "range_highvol",
}
"""Every (trending, tercile) pair spelled out, so the six labels are checked by the type."""


def realized_vol_series(closes: np.ndarray, window: int) -> np.ndarray:
    """Rolling standard deviation (ddof=1) of Daily log returns over `window` returns.

    One point per full window, oldest first, so `M` closes yield `M - window` points (the
    first close is consumed producing returns). Empty when there is not yet a full window.
    Log returns rather than simple ones so the measure is scale-free and additive.

    Closes must be finite and strictly positive: `log` of anything else is NaN or -inf,
    which would spread through the tercile until no comparison held and hand back a
    confident `midvol` for a series that is simply broken.
    """
    if window < 2:
        raise ValueError("window must be >= 2 for a ddof=1 standard deviation")
    prices = np.asarray(closes, dtype=np.float64).ravel()
    if not np.all(np.isfinite(prices)) or not np.all(prices > 0.0):
        raise ValueError("closes must be finite and strictly positive to take log returns")
    if prices.size < window + 1:
        return np.empty(0, dtype=np.float64)
    returns = np.diff(np.log(prices))
    windows = np.lib.stride_tricks.sliding_window_view(returns, window)
    return windows.std(axis=1, ddof=1)


def vol_tercile(series: np.ndarray) -> VolTercile:
    """Which third of its own history the last value of a rolling-vol series falls in.

    The comparison is against the expanding history of the same series, so the boundaries
    move with the instrument rather than being fixed constants. Strict comparisons, which
    makes a series with no dispersion at all (a constant, or a single point) `midvol`
    rather than arbitrarily low.
    """
    values = np.asarray(series, dtype=np.float64).ravel()
    if values.size == 0:
        raise ValueError("cannot take a tercile of an empty series")
    lower, upper = np.percentile(values, [_LOWER_TERCILE, _UPPER_TERCILE])
    last = float(values[-1])
    if last < float(lower):
        return "lowvol"
    if last > float(upper):
        return "highvol"
    return "midvol"


def tag(
    ctx: Context,
    *,
    adx_threshold: float = 25.0,
    vol_window: int = 20,
    min_history: int = 60,
) -> Regime:
    """The regime label for `ctx` right now - call it at entry and store it on the trade.

    Trending when Daily ADX(14) is strictly above `adx_threshold`. A NaN ADX (fewer than
    29 Daily bars, so Wilder's smoothing has not seeded) counts as ranging: absence of
    evidence for a trend is not evidence of one.

    The vol tercile needs at least `min_history` retained Daily bars before it means
    anything - a tercile of a handful of points is noise - and reports `midvol` below that,
    which is also the neutral answer when the instrument's vol never varied.
    """
    trending = ctx.adx("1d", 14) > adx_threshold  # NaN compares False, so an unseeded ADX ranges

    closes = ctx.bars("1d")[:, 3]
    if closes.size < min_history:
        vol: VolTercile = "midvol"
    else:
        series = realized_vol_series(closes, vol_window)
        vol = vol_tercile(series) if series.size else "midvol"

    return _LABELS[(trending, vol)]
