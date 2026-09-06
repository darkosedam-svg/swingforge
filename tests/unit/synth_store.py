"""A seeded synthetic `Store` and the toy strategies the lab tests replay through it.

Not a test module (pytest does not collect it): `test_tournament.py`,
`test_excursion.py` and `test_report.py` all import from here so the fixture is built one
way only.

The price series is a seeded geometric random walk on 1H bars. 4H bars are built from four
consecutive 1H bars and Daily bars from six consecutive 4H bars starting at 00:00 UTC, so
the three timeframes are exactly consistent with each other and `ReplaySource` can attach
every 1H sub-bar to its 4H parent.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime, timedelta

import numpy as np

from swingforge.adapters.store import Store
from swingforge.core.context import Context
from swingforge.core.types import Bar, Instrument, Signal
from swingforge.lab.tournament import add_months

__all__ = [
    "EntryFactoryStub",
    "EveryN",
    "Never",
    "Raising",
    "add_instrument",
    "build_store",
    "session_factory",
]

_HOUR = timedelta(hours=1)
_BARS_PER_4H = 4
_BARS_PER_DAY = 6  # 4H bars in a day


def _one_hour_bars(
    instrument: Instrument,
    start: datetime,
    hours: int,
    *,
    seed: int,
    drift: float,
    sigma: float = 0.004,
    initial: float = 100.0,
) -> list[Bar]:
    """A seeded geometric random walk as 1H bars; prices stay strictly positive."""
    rng = np.random.default_rng(seed)
    steps = rng.normal(drift, sigma, hours)
    wick_high = np.abs(rng.normal(0.0, sigma / 2, hours))
    wick_low = np.abs(rng.normal(0.0, sigma / 2, hours))
    bars: list[Bar] = []
    price = initial
    for index in range(hours):
        open_ = price
        close = open_ * math.exp(float(steps[index]))
        price = close
        bars.append(
            Bar(
                instrument=instrument,
                tf="1h",
                ts_open=start + index * _HOUR,
                open=open_,
                high=max(open_, close) * (1.0 + float(wick_high[index])),
                low=min(open_, close) * (1.0 - float(wick_low[index])),
                close=close,
                volume=1000.0 + index,
            )
        )
    return bars


def _aggregate(instrument: Instrument, bars: Sequence[Bar], group: int, tf: str) -> list[Bar]:
    """Roll `group` consecutive bars into one bar of `tf`; a trailing partial group is dropped."""
    out: list[Bar] = []
    for start in range(0, len(bars) - group + 1, group):
        chunk = bars[start : start + group]
        out.append(
            Bar(
                instrument=instrument,
                tf=tf,  # type: ignore[arg-type]
                ts_open=chunk[0].ts_open,
                open=chunk[0].open,
                high=max(bar.high for bar in chunk),
                low=min(bar.low for bar in chunk),
                close=chunk[-1].close,
                volume=sum(bar.volume for bar in chunk),
            )
        )
    return out


def build_store(
    instruments: Sequence[Instrument],
    *,
    start: datetime,
    months: int,
    seed: int,
    drift: float = 0.0,
) -> Store:
    """An in-memory `Store` holding `months` of consistent 1H/4H/Daily bars per instrument.

    `start` is floored to midnight UTC so the Daily bars line up with the 00:00 boundary
    the rest of the system assumes. Each instrument gets its own seed offset, so two
    symbols in one store are genuinely different series.
    """
    day_start = start.astimezone(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    store = Store(":memory:")
    for offset, instrument in enumerate(instruments):
        add_instrument(store, instrument, start=day_start, months=months, seed=seed + offset, drift=drift)
    return store


def add_instrument(
    store: Store,
    instrument: Instrument,
    *,
    start: datetime,
    months: int,
    seed: int,
    drift: float = 0.0,
) -> None:
    """Add one instrument's bars to an existing store — how a short-history symbol is made."""
    day_start = start.astimezone(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    hours = int((add_months(day_start, months) - day_start).total_seconds() // 3600)
    hours -= hours % (_BARS_PER_4H * _BARS_PER_DAY)
    one_hour = _one_hour_bars(instrument, day_start, hours, seed=seed, drift=drift)
    four_hour = _aggregate(instrument, one_hour, _BARS_PER_4H, "4h")
    daily = _aggregate(instrument, four_hour, _BARS_PER_DAY, "1d")
    store.upsert_instruments([instrument])
    store.upsert_bars([*one_hour, *four_hour, *daily])


# --- toy strategies ---------------------------------------------------------------


class EveryN:
    """Signals on every `n`-th 4H bar, at that bar's close, with a percentage stop."""

    def __init__(
        self,
        n: int,
        direction: int = 1,
        stop_pct: float = 0.01,
        *,
        expires_in_bars: int = 3,
        name: str = "every_n",
    ) -> None:
        self.n = n
        self.direction = direction
        self.stop_pct = stop_pct
        self.expires_in_bars = expires_in_bars
        self.name = name

    def on_bar(self, ctx: Context) -> Signal | None:
        bar = ctx.last("4h")
        if bar is None or ctx.bar_index % self.n != 0:
            return None
        entry = bar.close
        return Signal(
            direction=1 if self.direction == 1 else -1,
            entry=entry,
            stop=entry * (1.0 - self.direction * self.stop_pct),
            structure_target=None,
            tag=self.name,
            expires_in_bars=self.expires_in_bars,
        )


class Never:
    """Never signals: the "config produced no trades" case."""

    name = "never"

    def on_bar(self, ctx: Context) -> Signal | None:
        return None


class Raising:
    """Blows up on the first bar: the "one config errors, the run continues" case."""

    name = "raising"

    def on_bar(self, ctx: Context) -> Signal | None:
        raise RuntimeError("synthetic strategy failure")


class EntryFactoryStub:
    """An `EntryFactory` over named builders, recording the baseline rate and seed it is handed.

    `seeds` is every `(entry name, seed)` pair in call order, which is how the tests pin the
    per-config seed `run_config` derives; `baseline_rates` is the last rate seen per
    instrument, which only the `baseline` control is ever given.
    """

    def __init__(self, builders: Mapping[str, Callable[[Instrument, float | None], object]]) -> None:
        self.builders = dict(builders)
        self.baseline_rates: dict[str, float | None] = {}
        self.seeds: list[tuple[str, int]] = []

    def __call__(self, name: str, instrument: Instrument, *, baseline_rate: float | None, seed: int):  # type: ignore[no-untyped-def]
        self.seeds.append((name, seed))
        if name == "baseline":
            self.baseline_rates[f"{instrument.venue}:{instrument.symbol}"] = baseline_rate
        return self.builders[name](instrument, baseline_rate)


def session_factory(mode: str, profile: str) -> Callable[[Bar], bool]:
    """A toy session filter: `none` lets everything through, `london_ny` gates on the hour."""
    if mode == "none":
        return lambda bar: True
    if mode == "london_ny":
        return lambda bar: 7 <= bar.ts_open.hour < 21
    if mode == "active":
        if profile == "fx":
            return lambda bar: 7 <= bar.ts_open.hour < 21
        return lambda bar: bar.ts_open.hour >= 8 and bar.ts_open.weekday() < 5
    raise ValueError(f"unknown session mode {mode!r}")
