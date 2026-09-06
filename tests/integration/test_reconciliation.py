"""Paper and replay are the same machine: the bar transport must not change a fill.

`PaperBroker` and `Engine` are what backtest and paper trading share (design spec section 2);
only the bar source differs. Paper consumes `ReplaySource.stream`, an async iterator, one bar
at a time; a backtest consumes `ReplaySource.merged`, a list, in a `for` loop. If those two
paths ever diverged — a bar delivered twice, a Daily bar interleaved at the wrong point, state
carried across an `await` — the tournament's verdicts would not describe what paper does.

Two levels are pinned here:

* **Broker.** One scripted sequence of `submit`/`cancel`/`on_bar` calls, recorded off a real
  `Engine` run, is replayed against two fresh `PaperBroker`s: one driven by `stream("4h")`
  under `asyncio.run`, one by a `FakeStream` handing over the same bars synchronously. The
  `Fill`s must match one for one, in order.
* **Engine.** `Engine.run(source.merged(...))` against an `Engine` stepped bar by bar out of an
  async merge of `stream("4h")` and `stream("1d")` — the same close-time interleave `merged`
  does, rebuilt from two live streams rather than two lists. The closed `Trade`s must match.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterable, Iterator
from datetime import timedelta

import pytest

from swingforge.adapters.paper import PaperBroker
from swingforge.adapters.replay import ReplaySource
from swingforge.adapters.store import Store
from swingforge.core.costs import NullCostModel
from swingforge.core.engine import Engine
from swingforge.core.fills import FillResolver
from swingforge.core.portfolio import Portfolio
from swingforge.core.settings import Settings, StaticSettingsReader
from swingforge.core.types import Bar, Fill, Order, Trade
from swingforge.strategies.exits import FixedR
from swingforge.strategies.ict import ICT
from tests.integration.conftest import WALK, WALK_END
from tests.integration.synth import SYNTH_START
from tests.unit.synth_store import EveryN

_EVERY_N = 40
_FOUR_HOURS = timedelta(hours=4)
_ONE_DAY = timedelta(hours=24)

_Op = tuple[str, object]
"""One broker call: `("submit", Order)`, `("cancel", order_id)` or `("on_bar", bar_index)`."""


class _Recorder:
    """A `Broker` that records every call made to it and forwards to a real `PaperBroker`."""

    def __init__(self) -> None:
        self.broker = _fresh_broker()
        self.ops: list[_Op] = []

    def submit(self, order: Order) -> str:
        self.ops.append(("submit", order))
        return self.broker.submit(order)

    def cancel(self, order_id: str) -> None:
        self.ops.append(("cancel", order_id))
        self.broker.cancel(order_id)

    def on_bar(self, bar: Bar, bar_index: int) -> list[Fill]:
        self.ops.append(("on_bar", bar_index))
        return self.broker.on_bar(bar, bar_index)


class FakeStream:
    """The synchronous counterpart of `ReplaySource.stream`: the same bars, no event loop."""

    def __init__(self, bars: Iterable[Bar]) -> None:
        self._bars = list(bars)

    def __iter__(self) -> Iterator[Bar]:
        return iter(self._bars)


def _fresh_broker() -> PaperBroker:
    return PaperBroker(NullCostModel(), FillResolver())


def _script(bars: list[Bar]) -> list[_Op]:
    """Every broker call one `EveryN` + `FixedR(2)` engine run makes over `bars`.

    Recorded off the real engine rather than hand-written, so the replays below face the exact
    interleave the engine produces — including the second `on_bar` for the same `bar_index`
    that follows an entry fill, and the exit legs submitted between the two.
    """
    recorder = _Recorder()
    engine = Engine(
        WALK,
        EveryN(_EVERY_N),
        FixedR(2),
        recorder,
        Portfolio(10_000.0),
        StaticSettingsReader(Settings()),
    )
    engine.run(bars)
    return recorder.ops


def _apply(broker: PaperBroker, ops: list[_Op], cursor: int, bar: Bar, index: int) -> tuple[int, list[Fill]]:
    """Run the ops belonging to bar `index`, stopping before the first one for a later bar."""
    fills: list[Fill] = []
    while cursor < len(ops):
        kind, payload = ops[cursor]
        if kind == "on_bar" and payload != index:
            break
        cursor += 1
        if kind == "on_bar":
            fills.extend(broker.on_bar(bar, index))
        elif kind == "submit":
            broker.submit(payload)  # type: ignore[arg-type]
        else:
            broker.cancel(payload)  # type: ignore[arg-type]
    return cursor, fills


def _drive_sync(stream: Iterable[Bar], ops: list[_Op]) -> list[Fill]:
    broker = _fresh_broker()
    fills: list[Fill] = []
    cursor = 0
    for index, bar in enumerate(stream):
        cursor, produced = _apply(broker, ops, cursor, bar, index)
        fills.extend(produced)
    return fills


async def _drive_async(stream: AsyncIterator[Bar], ops: list[_Op]) -> list[Fill]:
    broker = _fresh_broker()
    fills: list[Fill] = []
    cursor = 0
    index = 0
    async for bar in stream:
        cursor, produced = _apply(broker, ops, cursor, bar, index)
        fills.extend(produced)
        index += 1
    return fills


async def _merged_stream(source: ReplaySource) -> AsyncIterator[Bar]:
    """`ReplaySource.merged`'s ordering, rebuilt live from the 4H and Daily streams.

    Bars are ordered by close time with the 4H bar first on a tie — the last 4H bar of a day
    closes at the same instant as that day's Daily bar, and the engine must see it first.
    """
    four_hour = source.stream(WALK, "4h")
    daily = source.stream(WALK, "1d")
    pending_4h = await anext(four_hour, None)
    pending_1d = await anext(daily, None)
    while pending_4h is not None or pending_1d is not None:
        take_4h = pending_1d is None or (
            pending_4h is not None
            and (pending_4h.ts_open + _FOUR_HOURS, 0) <= (pending_1d.ts_open + _ONE_DAY, 1)
        )
        if take_4h:
            assert pending_4h is not None
            yield pending_4h
            pending_4h = await anext(four_hour, None)
        else:
            assert pending_1d is not None
            yield pending_1d
            pending_1d = await anext(daily, None)


async def _stepwise_trades(source: ReplaySource) -> list[Trade]:
    engine = _ict_engine()
    closed: list[Trade] = []
    async for bar in _merged_stream(source):
        closed.extend(engine.step(bar))
    return closed


def _ict_engine() -> Engine:
    return Engine(
        WALK,
        ICT(),
        FixedR(2),
        _fresh_broker(),
        Portfolio(10_000.0),
        StaticSettingsReader(Settings()),
    )


@pytest.fixture(scope="module")
def four_hour_bars(walk_store: Store) -> list[Bar]:
    return ReplaySource(walk_store).history(WALK, "4h", SYNTH_START, WALK_END)


@pytest.fixture(scope="module")
def script(four_hour_bars: list[Bar]) -> list[_Op]:
    ops = _script(four_hour_bars)
    assert sum(1 for kind, _ in ops if kind == "submit") > 50, "the scripted run is too thin"
    return ops


def test_paper_and_replay_produce_the_same_fills(
    walk_store: Store, four_hour_bars: list[Bar], script: list[_Op]
) -> None:
    source = ReplaySource(walk_store)

    streamed = asyncio.run(_drive_async(source.stream(WALK, "4h"), script))
    faked = _drive_sync(FakeStream(four_hour_bars), script)

    assert streamed, "the scripted run produced no fills at all"
    assert streamed == faked


def test_the_async_stream_yields_exactly_the_replayed_bars(
    walk_store: Store, four_hour_bars: list[Bar]
) -> None:
    """The premise of the fill comparison above: the two transports carry the same bars."""

    async def _collect() -> list[Bar]:
        return [bar async for bar in ReplaySource(walk_store).stream(WALK, "4h")]

    assert asyncio.run(_collect()) == four_hour_bars


def test_engine_run_and_engine_step_close_the_same_trades(walk_store: Store) -> None:
    source = ReplaySource(walk_store)

    batch = _ict_engine().run(source.merged(WALK, SYNTH_START, WALK_END))
    stepwise = asyncio.run(_stepwise_trades(source))

    assert batch, "ICT closed no trades over the walk store: the comparison would be vacuous"
    assert batch == stepwise
