"""Synthetic stores for the integration suite, and the real strategy factories.

Two generators:

* `random_walk_store` is a thin wrapper over `tests.unit.synth_store.build_store` — a
  seeded geometric random walk on 1H bars, aggregated into consistent 4H and Daily bars.
  Nothing is planted in it, so no entry has an edge and the gate must fail every config.
* `planted_store` builds a series into which real **ICT setups** are injected on a fixed
  schedule, and (only when `edge=True`) a post-entry drift that makes those setups pay.

How a planted setup is built
----------------------------
The series is written at the 4H level and each 4H bar is then split into four 1H bars whose
union reproduces it exactly, so the 1H/4H/Daily views a `ReplaySource` serves are consistent
and every 4H bar carries its sub-bars.

The spine is a driftless log random walk (`_SPINE_SIGMA` per 4H bar). Every
`_SETUP_SPACING` bars the generator looks for a bar where the *actual* Daily swing levels
`strategies/levels.py` would report make a setup constructible, and then scripts the next
`_APPROACH_BARS + 5` bars:

1. an **approach** that walks price from wherever the spine left it to just above (below)
   the swept level;
2. a **sweep** candle that opens and closes inside the liquidity range but wicks
   `_STOP_OFFSET` (0.80% of price) past the level — far more than `ICT.min_pierce_pct`;
3. an **order block**: the last opposing-colour candle before the break of structure;
4. a **displacement leg** whose high stays under the sweep candle's, so the running neckline
   `ict.find_bos` carries is still the sweep candle's extreme;
5. a **break of structure** three bars after the sweep (`find_bos` needs `i >= 2`) closing
   clear of that neckline — this is the bar `ICT.on_bar` signals on;
6. a **retrace** window that trades down toward the order-block midpoint (the entry). On
   `1 - _UNFILLED_SHARE` of setups it trades *through* the entry and closes *exactly* on it,
   so the trade starts at breakeven with no built-in advantage; on the remaining
   `_UNFILLED_SHARE` it stops `_UNFILLED_GAP` short of the entry for the whole three-bar
   window `ICT`'s own `expires_in_bars=3` gives the order, so the limit is never touched and
   the setup produces a signal but no trade (see "The limit does not always fill" below).

Direction is drawn per setup rather than alternated (see `_planted_series`), and every setup
ends by pulling price back onto the level it swept, so the buy-and-hold curve stays a random
walk rather than picking up the planted drift — gate rule 5 compares against it.

**The limit does not always fill.** Scripting the retrace to close exactly on the entry, every
time, was originally load-bearing for reproducibility but it also means `expires_in_bars` is
never exercised adversely: every planted signal that fires becomes a trade, which is more
forgiving than a live limit order and makes `n_oos` optimistic relative to what the same edge
would show in practice. `_UNFILLED_SHARE` (drawn once per candidate slot from the same `rng`
the rest of the geometry uses, so it is identical for `edge=True` and `edge=False`) now stops
that share of setups `_UNFILLED_GAP` short of the entry instead, for the full three-bar fill
window — not just the retrace bar itself, since a lone short bar would leave the next
(unscripted) bar free to wander back onto the entry by chance and fill it anyway. `ICT` still
signals identically either way (the break-of-structure bar, and therefore the entry and stop
`ICT` computes, do not depend on what the retrace does), so
`test_ict_fires_on_the_planted_setups`'s ≥ 80% hit-rate floor is unaffected — it counts
signals, not fills.

The only difference between `edge=True` and `edge=False` is what happens *after* the entry:

* `edge=False` hands straight back to the spine — a symmetric random walk from the entry
  price, so the trade is a coin flip. Expectancy lands a little *below* zero rather than on
  it, because a bar that contains both the stop and the target resolves the adverse one first
  (see `_split`) — which is the honest, pessimistic reading and exactly what a control should
  do.
* `edge=True` scripts `_DRIFT_BARS` bars that run `m` R in the trade's favour (`m` uniform on
  `_DRIFT_R_RANGE`) and then come back to **exactly** the price the spine would have reached
  anyway. Both variants therefore leave the window at the same price, so the two stores share
  a spine and the same setup schedule; only the path inside the drift window differs.

`_MIN_LEVEL_GAP` is the load-bearing constraint. `ICT` re-reads the liquidity range on the
break-of-structure bar, so the swept level must still be the nearest Daily swing low below
that bar's close. Requiring the pre-setup price to already sit at least `_BOS_OFFSET` above
the level makes the window the break-of-structure adds — `(level, bos_close]` — a subset of
`(level, p)`, which holds no swing low by construction. A slot that cannot satisfy it is
skipped and the next bar is tried.

`tests/integration/test_planted_edge.py::test_ict_fires_on_the_planted_setups` is the
self-test: it replays `ICT()` over both stores and asserts a signal on the planted
break-of-structure bars. Re-run it after any change to `strategies/ict.py` — the geometry
below is tuned to that file's `find_sweep`/`find_bos`/`find_order_block`, and this module is
where to regenerate it from.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from functools import lru_cache
from typing import Literal, cast

import numpy as np

from swingforge.adapters.store import Store
from swingforge.core.types import Bar, Instrument, round_to_tick
from swingforge.lab.tournament import add_months
from swingforge.strategies.base import Strategy
from swingforge.strategies.baseline import Baseline
from swingforge.strategies.ict import ICT
from swingforge.strategies.session import SessionFilter, SessionMode, SessionProfile
from swingforge.strategies.zones import Zones
from tests.unit.synth_store import build_store

__all__ = [
    "PLANTED",
    "SYNTH_START",
    "PlantedSetup",
    "planted_setups",
    "planted_store",
    "random_walk_store",
    "real_entry_factory",
    "real_session_factory",
]

SYNTH_START = datetime(2020, 1, 1, tzinfo=UTC)
"""Midnight UTC, so the Daily bars line up with the 00:00 boundary the system assumes."""

PLANTED = Instrument(
    venue="hyperliquid",
    symbol="PLANT",
    tick_size=Decimal("0.01"),
    contract_multiplier=Decimal("1"),
    quote_ccy="USD",
    session_profile="perp",
)
"""The default planted-edge instrument: a 0.01 tick on a ~100 price, so rounding an entry
never moves it by a meaningful fraction of the 0.84%-of-price risk distance."""

_HOUR = timedelta(hours=1)
_SUBBARS_PER_4H = 4
_BARS_PER_DAY = 6

_INITIAL_PRICE = 100.0
_SPINE_SIGMA = 0.0022
"""Log standard deviation of one 4H spine step (0.22%).

Chosen against the planted risk distance of 0.84% of price: one R is about four spine steps,
so an unplanted trade takes several bars to resolve rather than being decided by noise on the
next bar, and `ctx.atr("1d")` — which sizes the `baseline` control's stop and sets how far
`trail_1_2` trails — comes out near 1R rather than several."""

_SETUP_SPACING = 22
_APPROACH_BARS = 3
_DRIFT_BARS = 14
_DRIFT_RISE_BARS = 8
_DRIFT_NOISE = 0.2
_DRIFT_R_RANGE = (-1.0, 4.2)
"""How far, in R, an `edge=True` setup runs before it comes back to the spine.

Drawn uniformly per setup, so roughly a fifth of the planted trades (1.0 of the 5.2-wide
range is negative) run *against* the entry and stop out, a further chunk stall near
breakeven, and the rest reach the 2R (and sometimes the 3R) target. That leaves a real but
ordinary edge rather than an implausible one, and the range straddling zero is load-bearing:
a config whose OOS R multiples are nearly all identical has an enormous per-observation
Sharpe, which inflates the tournament's `trial_sr_variance` and deflates *every* config's DSR
to zero. Gate rule 2 working exactly as designed — and, with a cleaner plant, the reason
nothing passes."""

_UNIT = 1e-4
_PATTERN: tuple[tuple[float, float, float, float], ...] = (
    (15.0, 18.0, -80.0, 6.0),  # sweep: opens/closes inside the range, wicks 80bp past it
    (6.0, 8.0, 1.0, 2.0),  # order block: the last opposing candle before the BOS
    (2.0, 14.0, 1.0, 12.0),  # displacement leg: high stays under the sweep candle's
    (12.0, 26.0, 11.0, 24.0),  # break of structure: closes clear of the neckline
    (24.0, 24.0, 2.0, 4.0),  # retrace: trades through the entry and closes on it
)
"""The five scripted bars of a long setup as `(open, high, low, close)` offsets from the
swept level, in units of `_UNIT` (one basis point). A short mirrors them (see `_pattern`)."""

_UNFILLED_SHARE = 0.2
"""Share of setups whose retrace stops short of the entry — see "The limit does not always
fill" in the module docstring. Lower to `0.1` if this ever drags the planted-edge gate's
`n_oos` below rule 1's floor of 60; re-run
`test_planted_edge.py::test_a_planted_ict_config_passes_every_gate_rule` after changing it."""
_UNFILLED_GAP = 1.0
"""How many basis points short of the entry the retrace stops, on an unfilled setup."""
_UNFILLED_PATTERN: tuple[tuple[float, float, float, float], ...] = (
    (24.0, 24.0, 5.0, 6.0),  # retrace: stops `_UNFILLED_GAP` above the entry, not through it
    (6.0, 7.0, 5.0, 6.0),  # hover: stays clear of the entry for the rest of the fill window
    (6.0, 7.0, 5.0, 6.0),  # hover: ...
)
"""The three bars of an unfilled setup's fill window (`ICT`'s `expires_in_bars=3`), replacing
`_PATTERN`'s single retrace bar. All three floor out at `_ENTRY_OFFSET + _UNFILLED_GAP` (5.0),
never at `_ENTRY_OFFSET` (4.0) itself, so the limit order at the entry is never touched. Fixed
offsets rather than steps off the spine: `_SPINE_SIGMA` is more than the whole 1bp gap this
has to respect, so a real step would blow through it more often than not."""

_ENTRY_OFFSET = 4.0  # (order-block open + close) / 2
_STOP_OFFSET = -80.0  # the sweep candle's wick
_BOS_OFFSET = 24.0  # the break-of-structure close
_SWEEP_BAR = 0
_BOS_BAR = 3
_RETRACE_BAR = 4
_PATTERN_BARS = len(_PATTERN)

_MIN_LEVEL_GAP = _BOS_OFFSET * _UNIT
"""The pre-setup price must sit at least this far past the level — see the module docstring."""
_MAX_LEVEL_GAP = 0.06
"""...and no further, so the approach is a plausible move rather than a cliff."""

_MIN_H4_BARS = 30  # ICT's own floor
_K = 2  # ICT's fractal half-width


@dataclass(frozen=True)
class PlantedSetup:
    """One injected ICT setup: where it is, which way it runs, and the levels it implies."""

    slot: int
    """4H index of the first scripted (approach) bar."""
    sweep_index: int
    """4H index of the sweep candle."""
    bos_index: int
    """4H index of the break-of-structure bar — the bar `ICT.on_bar` must signal on."""
    direction: Literal[1, -1]
    level: float
    """The Daily swing level the sweep runs through."""
    entry: float
    stop: float
    filled: bool = True
    """False for the `_UNFILLED_SHARE` of setups whose retrace stops short of `entry`: `ICT`
    still signals on `bos_index`, but no limit order ever fills and no trade results."""


# --- the random-walk store ----------------------------------------------------------


def random_walk_store(
    instruments: Sequence[Instrument], *, years: int = 4, seed: int, drift: float = 0.0
) -> Store:
    """A seeded geometric random walk with nothing planted in it, `years` long."""
    return build_store(instruments, start=SYNTH_START, months=years * 12, seed=seed, drift=drift)


# --- the real factories the CLI must mirror -----------------------------------------


def real_entry_factory(
    name: str, instrument: Instrument, *, baseline_rate: float | None, seed: int
) -> Strategy:
    """The tournament's `EntryFactory`, wired to the production strategies.

    ⚠ `swingforge.cli` must build its entries exactly this way: `baseline_rate` is trades per
    1,000 4H bars (`None` for everything but `baseline`, and `1.0` when ICT produced nothing)
    and `seed` is the per-config seed `run_config` derives — `Baseline` must take its
    randomness from that argument and nothing else.
    """
    if name == "ict":
        return ICT()
    if name == "zones":
        return Zones()
    if name == "baseline":
        return Baseline(target_trades_per_1000_bars=baseline_rate or 1.0, seed=seed)
    raise KeyError(f"unknown entry strategy {name!r}")


def real_session_factory(mode: str, profile: str) -> Callable[[Bar], bool]:
    """The tournament's `SessionFactory`, wired to the production `SessionFilter`."""
    return SessionFilter(cast(SessionMode, mode), cast(SessionProfile, profile))


# --- the planted store --------------------------------------------------------------


def planted_store(instrument: Instrument, *, years: int = 4, seed: int, edge: bool) -> Store:
    """An in-memory `Store` of 1H/4H/Daily bars with ICT setups planted in it.

    `edge=True` makes each setup run `_DRIFT_R_RANGE` R in its own direction after the entry;
    `edge=False` plants the identical setups and hands straight back to the random walk.
    """
    shapes, _setups = _planted_series(instrument, years, seed, edge)
    store = Store(":memory:")
    store.upsert_instruments([instrument])
    store.upsert_bars(_bars_from_shapes(instrument, shapes))
    return store


def planted_setups(
    instrument: Instrument, *, years: int = 4, seed: int, edge: bool
) -> tuple[PlantedSetup, ...]:
    """The setups `planted_store` injected for the same arguments — what the self-test checks."""
    return _planted_series(instrument, years, seed, edge)[1]


@lru_cache(maxsize=8)
def _planted_series(
    instrument: Instrument, years: int, seed: int, edge: bool
) -> tuple[np.ndarray, tuple[PlantedSetup, ...]]:
    """The `(n, 4)` OHLC array of 4H bars, and the setups planted into it.

    Memoised because `planted_store` and `planted_setups` are asked for the same series, and
    because the generator is closed-loop: it maintains the Daily fractal pivots as it goes so
    that each setup can be built against the levels `ICT` will actually read.
    """
    tick = instrument.tick_size
    hours = int((add_months(SYNTH_START, years * 12) - SYNTH_START).total_seconds() // 3600)
    hours -= hours % (_SUBBARS_PER_4H * _BARS_PER_DAY)
    n4h = hours // _SUBBARS_PER_4H

    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0, _SPINE_SIGMA, n4h)
    wick_high = np.abs(rng.normal(0.0, _SPINE_SIGMA / 3, n4h))
    wick_low = np.abs(rng.normal(0.0, _SPINE_SIGMA / 3, n4h))
    magnitudes = rng.uniform(*_DRIFT_R_RANGE, n4h)
    # Drawn per candidate slot, not alternated. A long setup's approach walks price *down*
    # onto the level it is about to sweep and a short's walks it *up*, so under strict
    # alternation the next setup's approach and sweep always ran in the previous trade's
    # favour -- a systematic tailwind for any trade still open 22 bars later, worth roughly
    # +0.25R on `edge=False` before this was randomised.
    directions = rng.integers(0, 2, n4h) * 2 - 1
    # Drawn per candidate slot too, and after `directions` so every array drawn before this
    # keeps the exact values it always had -- appending here changes nothing already
    # planted, only whether *new* setups get a fillable retrace. Shared across `edge` for
    # the same reason `directions` is: identical for `edge=True` and `edge=False`.
    unfilled_rolls = rng.random(n4h) < _UNFILLED_SHARE

    shapes = np.empty((n4h, 4), dtype=float)
    levels = _Levels()
    scripted: dict[int, tuple[float, float, float, float]] = {}
    setups: list[PlantedSetup] = []

    price = _INITIAL_PRICE
    next_attempt = _MIN_H4_BARS + _BARS_PER_DAY * (2 * _K + 3)
    window = _APPROACH_BARS + _PATTERN_BARS + (_DRIFT_BARS if edge else 0)

    for i in range(n4h):
        if i not in scripted and i >= next_attempt and i + window <= n4h:
            setup = _plan_setup(
                i,
                price,
                cast(Literal[1, -1], int(directions[i])),
                levels,
                tick,
                steps=steps,
                magnitude=float(magnitudes[i]),
                edge=edge,
                unfilled=bool(unfilled_rolls[i]),
                scripted=scripted,
            )
            if setup is not None:
                setups.append(setup)
                next_attempt = i + _SETUP_SPACING
        if i in scripted:
            ohlc = scripted.pop(i)
        else:
            close = price * math.exp(float(steps[i]))
            ohlc = _ohlc(
                price,
                close,
                high=max(price, close) * (1.0 + float(wick_high[i])),
                low=min(price, close) * (1.0 - float(wick_low[i])),
            )
        shapes[i] = ohlc
        price = ohlc[3]
        if (i + 1) % _BARS_PER_DAY == 0:
            day = shapes[i + 1 - _BARS_PER_DAY : i + 1]
            levels.push_day(float(day[:, 1].max()), float(day[:, 2].min()))

    shapes.flags.writeable = False  # the cache hands the same array to every caller
    return shapes, tuple(setups)


def _ohlc(open_: float, close: float, *, high: float, low: float) -> tuple[float, float, float, float]:
    """One bar, with `high`/`low` widened if a scripted body pokes through them."""
    return (open_, max(high, open_, close), min(low, open_, close), close)


def _plan_setup(
    slot: int,
    price: float,
    direction: Literal[1, -1],
    levels: _Levels,
    tick: Decimal,
    *,
    steps: np.ndarray,
    magnitude: float,
    edge: bool,
    unfilled: bool,
    scripted: dict[int, tuple[float, float, float, float]],
) -> PlantedSetup | None:
    """Script one setup into `scripted`, or return None when the levels do not allow it."""
    level = levels.nearest_low(price) if direction == 1 else levels.nearest_high(price)
    if level is None:
        return None
    gap = direction * (price / level - 1.0)
    if not _MIN_LEVEL_GAP <= gap <= _MAX_LEVEL_GAP:
        return None
    bos_close = _at(level, _BOS_OFFSET, direction)
    # The opposing level must still exist once the break of structure has moved price:
    # `ICT.on_bar` returns early unless it has both edges of the liquidity range.
    opposing = levels.nearest_high(bos_close) if direction == 1 else levels.nearest_low(bos_close)
    if opposing is None:
        return None

    sweep = slot + _APPROACH_BARS
    for offset, bar in enumerate(_approach(price, _at(level, _PATTERN[_SWEEP_BAR][0], direction))):
        scripted[slot + offset] = bar
    for offset, bar in enumerate(_pattern(level, direction)):
        scripted[sweep + offset] = bar

    entry = round_to_tick(_at(level, _ENTRY_OFFSET, direction), tick)
    stop = _at(level, _STOP_OFFSET, direction) - direction * float(tick)
    retrace = sweep + _RETRACE_BAR
    if unfilled:
        # Overwrites the retrace bar `_pattern` already scripted and the two bars after it
        # -- the whole three-bar window the entry order stays live for -- so the limit at
        # `entry` is never touched and no trade results. No drift either way: there is
        # nothing open to drift.
        for offset, bar in enumerate(_unfilled_retrace(level, direction)):
            scripted[retrace + offset] = bar
    elif edge:
        risk = direction * (entry - stop)
        for offset, bar in enumerate(
            _drift(entry, risk, direction, magnitude, steps[retrace + 1 : retrace + 1 + _DRIFT_BARS])
        ):
            scripted[retrace + 1 + offset] = bar
    return PlantedSetup(
        slot=slot,
        sweep_index=sweep,
        bos_index=sweep + _BOS_BAR,
        direction=direction,
        level=level,
        entry=entry,
        stop=stop,
        filled=not unfilled,
    )


def _at(level: float, offset: float, direction: int) -> float:
    """The price `offset` basis points past `level`, on the side `direction` implies."""
    return level * (1.0 + direction * offset * _UNIT)


def _approach(start: float, target: float) -> list[tuple[float, float, float, float]]:
    """`_APPROACH_BARS` log-linear bars carrying the spine's price onto the sweep candle's open."""
    path = np.exp(np.linspace(math.log(start), math.log(target), _APPROACH_BARS + 1))
    bars: list[tuple[float, float, float, float]] = []
    for previous, current in zip(path, path[1:], strict=False):
        open_, close = float(previous), float(current)
        bars.append(_ohlc(open_, close, high=max(open_, close) * 1.0002, low=min(open_, close) * 0.9998))
    return bars


def _mirrored_bars(
    level: float, direction: Literal[1, -1], offsets: Sequence[tuple[float, float, float, float]]
) -> list[tuple[float, float, float, float]]:
    """`offsets` (each an `(open, high, low, close)` offset from `level`, in `_UNIT`s) turned
    into bars, mirrored for a short (high and low swap roles)."""
    bars: list[tuple[float, float, float, float]] = []
    for open_, high, low, close in offsets:
        top, bottom = (high, low) if direction == 1 else (low, high)
        bars.append(
            _ohlc(
                _at(level, open_, direction),
                _at(level, close, direction),
                high=_at(level, top, direction),
                low=_at(level, bottom, direction),
            )
        )
    return bars


def _pattern(level: float, direction: Literal[1, -1]) -> list[tuple[float, float, float, float]]:
    """The five scripted bars of `_PATTERN`, mirrored for a short (high and low swap roles)."""
    return _mirrored_bars(level, direction, _PATTERN)


def _unfilled_retrace(level: float, direction: Literal[1, -1]) -> list[tuple[float, float, float, float]]:
    """The three scripted bars of `_UNFILLED_PATTERN`, replacing `_pattern`'s retrace bar for
    a setup whose limit is never touched — see the module docstring."""
    return _mirrored_bars(level, direction, _UNFILLED_PATTERN)


def _drift(
    entry: float, risk: float, direction: int, magnitude: float, steps: np.ndarray
) -> list[tuple[float, float, float, float]]:
    """The `edge=True` post-entry path: `magnitude` R in the trade's favour, then back.

    The last close is exactly `entry * exp(sum(steps))` — where the spine would have been had
    nothing been scripted — so an `edge=True` store and an `edge=False` store leave every
    setup window at the same price and go on sharing one random walk.
    """
    peak = entry + direction * magnitude * risk
    end = entry * math.exp(float(steps.sum()))
    rise = np.linspace(math.log(entry), math.log(peak), _DRIFT_RISE_BARS + 1)[1:]
    fall = np.linspace(math.log(peak), math.log(end), len(steps) - _DRIFT_RISE_BARS + 1)[1:]
    closes = np.exp(np.concatenate([rise, fall]) + _DRIFT_NOISE * steps)
    closes[-1] = end
    bars: list[tuple[float, float, float, float]] = []
    previous = entry
    for index, close in enumerate(closes):
        open_, current = previous, float(close)
        wick = abs(float(steps[index])) * 0.3
        bars.append(
            _ohlc(
                open_,
                current,
                high=max(open_, current) * (1.0 + wick),
                low=min(open_, current) * (1.0 - wick),
            )
        )
        previous = current
    return bars


class _Levels:
    """The Daily fractal pivots `strategies/levels.py` reports, maintained incrementally.

    A pivot at day `i` is only confirmable once day `i + k` has closed, so each completed day
    confirms exactly one new candidate — the same append-only cache `ICT` keeps, and therefore
    the same levels it will read back.
    """

    def __init__(self, k: int = _K) -> None:
        self._k = k
        self._highs: list[float] = []
        self._lows: list[float] = []
        self.highs: list[float] = []
        self.lows: list[float] = []

    def push_day(self, high: float, low: float) -> None:
        self._highs.append(high)
        self._lows.append(low)
        k = self._k
        pivot = len(self._highs) - 1 - k
        if pivot < k:
            return
        window_high = self._highs[pivot - k : pivot + k + 1]
        if max(window_high) == self._highs[pivot] and window_high.count(self._highs[pivot]) == 1:
            self.highs.append(self._highs[pivot])
        window_low = self._lows[pivot - k : pivot + k + 1]
        if min(window_low) == self._lows[pivot] and window_low.count(self._lows[pivot]) == 1:
            self.lows.append(self._lows[pivot])

    def nearest_low(self, price: float) -> float | None:
        below = [level for level in self.lows if level < price]
        return max(below) if below else None

    def nearest_high(self, price: float) -> float | None:
        above = [level for level in self.highs if level > price]
        return min(above) if above else None


# --- turning 4H shapes into a consistent 1H / 4H / Daily bar set --------------------


def _bars_from_shapes(instrument: Instrument, shapes: np.ndarray) -> list[Bar]:
    """Every 1H, 4H and Daily bar the shapes imply, exactly consistent with each other."""
    four_hour: list[Bar] = []
    one_hour: list[Bar] = []
    for index, (open_, high, low, close) in enumerate(shapes):
        ts_open = SYNTH_START + index * _SUBBARS_PER_4H * _HOUR
        four_hour.append(
            Bar(
                instrument=instrument,
                tf="4h",
                ts_open=ts_open,
                open=open_,
                high=high,
                low=low,
                close=close,
                volume=1000.0 + index,
            )
        )
        one_hour.extend(_split(instrument, four_hour[-1]))
    daily = [
        Bar(
            instrument=instrument,
            tf="1d",
            ts_open=chunk[0].ts_open,
            open=chunk[0].open,
            high=max(bar.high for bar in chunk),
            low=min(bar.low for bar in chunk),
            close=chunk[-1].close,
            volume=sum(bar.volume for bar in chunk),
        )
        for chunk in (
            four_hour[start : start + _BARS_PER_DAY]
            for start in range(0, len(four_hour) - _BARS_PER_DAY + 1, _BARS_PER_DAY)
        )
    ]
    return [*one_hour, *four_hour, *daily]


def _split(instrument: Instrument, bar: Bar) -> list[Bar]:
    """Four 1H bars whose open, close, high and low reproduce `bar` exactly.

    A bullish 4H bar is walked open -> low -> midpoint -> high -> close and a bearish one the
    other way round, so the sub-bar ordering the fill resolver reads is the conventional
    pessimistic one for a long (the adverse extreme is printed first) and its mirror for a
    short — symmetric, so neither direction is quietly favoured.
    """
    middle = (bar.high + bar.low) / 2
    legs = [bar.low, middle, bar.high] if bar.close >= bar.open else [bar.high, middle, bar.low]
    points = [bar.open, *legs, bar.close]
    return [
        Bar(
            instrument=instrument,
            tf="1h",
            ts_open=bar.ts_open + index * _HOUR,
            open=points[index],
            high=max(points[index], points[index + 1]),
            low=min(points[index], points[index + 1]),
            close=points[index + 1],
            volume=bar.volume / _SUBBARS_PER_4H,
        )
        for index in range(_SUBBARS_PER_4H)
    ]
