"""The adapter and strategy protocols are implementable as written.

Minimal conforming stubs, checked structurally. This is the executable form of the call
sequence documented in `swingforge.strategies.base`, and it fails loudly if a later work
unit changes a signature out from under the other agents.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime
from decimal import Decimal
from typing import Literal

from swingforge.adapters.base import BarSource, Broker, CostModel, ExitResolver
from swingforge.core.context import Context
from swingforge.core.types import (
    TF,
    Bar,
    CostBreakdown,
    ExitEvent,
    Fill,
    Instrument,
    Order,
    Position,
    Resolution,
    Signal,
    Trade,
)
from swingforge.strategies.base import ExitRule, Strategy

BTC = Instrument(
    venue="hyperliquid",
    symbol="BTC",
    tick_size=Decimal("0.5"),
    contract_multiplier=Decimal("1"),
    quote_ccy="USD",
    session_profile="perp",
)
TS = datetime(2026, 1, 2, 4, 0, tzinfo=UTC)
BAR = Bar(instrument=BTC, tf="4h", ts_open=TS, open=100.0, high=105.0, low=95.0, close=102.0, volume=1.0)
ENTRY_ORDER = Order(
    id="o1",
    instrument=BTC,
    direction=1,
    qty=1.0,
    kind="limit",
    price=100.0,
    expires_at_bar=3,
    leg="entry",
)
ENTRY_FILL = Fill(order_id="o1", ts=TS, price=100.0, qty=1.0, cost=CostBreakdown(), leg="entry")
TRADE = Trade(
    id="t-1",
    instrument=BTC,
    direction=1,
    entry_fill=ENTRY_FILL,
    stop=97.0,
    target=106.0,
    risk_r=300.0,
    opened_bar=0,
)


class StubBarSource:
    def history(self, instrument: Instrument, tf: TF, start: datetime, end: datetime) -> list[Bar]:
        return [BAR]

    def stream(self, instrument: Instrument, tf: TF) -> AsyncIterator[Bar]:
        async def _gen() -> AsyncIterator[Bar]:
            yield BAR

        return _gen()


class StubBroker:
    def __init__(self) -> None:
        self.submitted: list[Order] = []

    def submit(self, order: Order) -> str:
        self.submitted.append(order)
        return order.id

    def cancel(self, order_id: str) -> None:
        self.submitted = [o for o in self.submitted if o.id != order_id]

    def positions(self) -> list[Position]:
        return []

    def on_bar(self, bar: Bar, bar_index: int) -> list[Fill]:
        self.submitted = [
            o for o in self.submitted if o.expires_at_bar is None or bar_index <= o.expires_at_bar
        ]
        return [ENTRY_FILL] if self.submitted else []

    def fills(self) -> AsyncIterator[Fill]:
        async def _gen() -> AsyncIterator[Fill]:
            yield ENTRY_FILL

        return _gen()


class StubCostModel:
    def entry(self, order: Order, bar: Bar) -> CostBreakdown:
        return CostBreakdown(spread=0.1, slippage=0.25)

    def carry(self, position: Position, bar: Bar) -> float:
        return 0.0


class StubExitResolver:
    def resolve(
        self,
        bar: Bar,
        stop: float,
        targets: list[float],
        direction: Literal[1, -1],
        *,
        stop_after_partial: float | None = None,
    ) -> Resolution:
        return Resolution(events=(ExitEvent(leg="stop", price=stop),), mode="pessimistic")


class StubStrategy:
    name = "stub"

    def on_bar(self, ctx: Context) -> Signal | None:
        return Signal(
            direction=1, entry=100.0, stop=97.0, structure_target=None, tag="stub", expires_in_bars=3
        )


class StubExitRule:
    name = "stub_exit"

    def attach(self, trade: Trade, ctx: Context) -> list[Order]:
        return [
            Order(
                id=f"{trade.id}-stop",
                instrument=trade.instrument,
                direction=-trade.direction,
                qty=trade.entry_fill.qty,
                kind="stop",
                price=trade.stop,
                expires_at_bar=None,
                leg="stop",
                trade_id=trade.id,
            )
        ]

    def on_bar(self, trade: Trade, ctx: Context) -> list[Order]:
        return []


def test_stubs_satisfy_the_protocols() -> None:
    assert isinstance(StubBarSource(), BarSource)
    assert isinstance(StubBroker(), Broker)
    assert isinstance(StubCostModel(), CostModel)
    assert isinstance(StubExitResolver(), ExitResolver)
    assert isinstance(StubStrategy(), Strategy)
    assert isinstance(StubExitRule(), ExitRule)


def test_documented_call_sequence_runs_end_to_end() -> None:
    ctx = Context(BTC)
    ctx.push(BAR)

    strategy: Strategy = StubStrategy()
    exit_rule: ExitRule = StubExitRule()
    broker: Broker = StubBroker()

    signal = strategy.on_bar(ctx)
    assert signal is not None

    broker.submit(ENTRY_ORDER)
    fills = broker.on_bar(BAR, ctx.bar_index)
    assert [f.leg for f in fills] == ["entry"]

    legs = exit_rule.attach(TRADE, ctx)
    assert [(o.leg, o.trade_id) for o in legs] == [("stop", "t-1")]
    for leg in legs:
        broker.submit(leg)

    assert exit_rule.on_bar(TRADE, ctx) == []
    assert broker.positions() == []


def test_bar_source_and_cost_model_shapes() -> None:
    source: BarSource = StubBarSource()
    assert source.history(BTC, "4h", TS, TS) == [BAR]

    costs: CostModel = StubCostModel()
    assert costs.entry(ENTRY_ORDER, BAR).total == 0.35
    assert (
        costs.carry(
            Position(instrument=BTC, direction=1, qty=1.0, avg_price=100.0, stop=97.0, target=None), BAR
        )
        == 0.0
    )


def test_exit_resolver_shape() -> None:
    resolver: ExitResolver = StubExitResolver()
    res = resolver.resolve(BAR, 97.0, [106.0], 1, stop_after_partial=100.0)
    assert res.mode == "pessimistic"
    assert res.events[0].leg == "stop"
