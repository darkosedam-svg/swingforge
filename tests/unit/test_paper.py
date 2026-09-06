"""Tests for `swingforge.adapters.paper.PaperBroker`.

Uses a small fake `ExitResolver` (pessimistic: stop checked before each remaining target;
honours `stop_after_partial`; records the kwargs it was called with) instead of the real
`FillResolver`, which lives in another work unit's tree.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Literal

import pytest

from swingforge.adapters.base import Broker
from swingforge.adapters.paper import PaperBroker
from swingforge.core.types import (
    Bar,
    CostBreakdown,
    ExitEvent,
    Instrument,
    Order,
    Position,
    Resolution,
    round_to_tick,
)

BTC = Instrument(
    venue="hyperliquid",
    symbol="BTC",
    tick_size=Decimal("0.5"),
    contract_multiplier=Decimal("1"),
    quote_ccy="USD",
    session_profile="perp",
)
TS0 = datetime(2026, 1, 1, tzinfo=UTC)


def _bar(ts_open: datetime, *, open_: float = 100.0, high: float = 110.0, low: float = 90.0) -> Bar:
    """A 4H bar with `close == open` so range validity never depends on `close`."""
    return Bar(
        instrument=BTC, tf="4h", ts_open=ts_open, open=open_, high=high, low=low, close=open_, volume=1.0
    )


def _entry(
    id_: str,
    *,
    direction: Literal[1, -1] = 1,
    kind: Literal["market", "limit"] = "market",
    price: float | None = None,
    qty: float = 1.0,
    expires_at_bar: int | None = None,
    trade_id: str | None = None,
) -> Order:
    return Order(
        id=id_,
        instrument=BTC,
        direction=direction,
        qty=qty,
        kind=kind,
        price=price,
        expires_at_bar=expires_at_bar,
        leg="entry",
        trade_id=trade_id,
    )


def _leg(
    id_: str,
    trade_id: str,
    leg: Literal["stop", "target", "partial", "time"],
    *,
    direction: Literal[1, -1],
    kind: Literal["limit", "market", "stop"],
    price: float | None,
    qty: float = 1.0,
) -> Order:
    return Order(
        id=id_,
        instrument=BTC,
        direction=direction,
        qty=qty,
        kind=kind,
        price=price,
        expires_at_bar=None,
        leg=leg,
        trade_id=trade_id,
    )


@dataclass
class FakeCostModel:
    """Records every call; `carry` returns a fixed amount so accrual is easy to predict."""

    entry_cost: CostBreakdown = field(default_factory=CostBreakdown)
    carry_amount: float = 0.0
    entry_calls: list[tuple[Order, Bar]] = field(default_factory=list)
    carry_calls: list[tuple[Position, Bar]] = field(default_factory=list)

    def entry(self, order: Order, bar: Bar) -> CostBreakdown:
        self.entry_calls.append((order, bar))
        return self.entry_cost

    def carry(self, position: Position, bar: Bar) -> float:
        self.carry_calls.append((position, bar))
        return self.carry_amount


@dataclass
class ProportionalCostModel:
    """A cost model whose commission scales with the *order's* qty -- lets a test tell
    which qty the broker actually costed a fill against.
    """

    per_unit: float = 10.0
    entry_calls: list[tuple[Order, Bar]] = field(default_factory=list)

    def entry(self, order: Order, bar: Bar) -> CostBreakdown:
        self.entry_calls.append((order, bar))
        return CostBreakdown(commission=order.qty * self.per_unit)

    def carry(self, position: Position, bar: Bar) -> float:
        return 0.0


@dataclass
class FakeExitResolver:
    """Pessimistic: on each step, a touched stop wins over the next pending target.

    Honours `stop_after_partial`, switching the effective stop once `targets[0]` is taken,
    exactly like the real resolver is documented to. Records every call's kwargs.
    """

    calls: list[dict[str, object]] = field(default_factory=list)

    def resolve(
        self,
        bar: Bar,
        stop: float,
        targets: list[float],
        direction: Literal[1, -1],
        *,
        stop_after_partial: float | None = None,
    ) -> Resolution:
        self.calls.append(
            {
                "bar": bar,
                "stop": stop,
                "targets": list(targets),
                "direction": direction,
                "stop_after_partial": stop_after_partial,
            }
        )
        effective_stop = stop
        events: list[ExitEvent] = []
        idx = 0
        while True:
            stop_hit = bar.low <= effective_stop if direction == 1 else bar.high >= effective_stop
            if stop_hit:
                events.append(ExitEvent(leg="stop", price=effective_stop))
                break
            if idx >= len(targets):
                break
            target_price = targets[idx]
            target_hit = bar.high >= target_price if direction == 1 else bar.low <= target_price
            if not target_hit:
                break
            events.append(ExitEvent(leg="target", price=target_price, target_index=idx))
            if idx == 0 and stop_after_partial is not None:
                effective_stop = stop_after_partial
            idx += 1
        return Resolution(events=tuple(events), mode="pessimistic")


# -- protocol conformance -----------------------------------------------------


def test_paper_broker_satisfies_broker_protocol() -> None:
    broker = PaperBroker(FakeCostModel(), FakeExitResolver())
    assert isinstance(broker, Broker)


def test_submit_rejects_stop_entry_orders() -> None:
    broker = PaperBroker(FakeCostModel(), FakeExitResolver())
    stop_entry = Order(
        id="e1",
        instrument=BTC,
        direction=1,
        qty=1.0,
        kind="stop",
        price=100.0,
        expires_at_bar=None,
        leg="entry",
        trade_id=None,
    )
    with pytest.raises(NotImplementedError, match="stop-entry"):
        broker.submit(stop_entry)


# -- entry fills: market / limit -----------------------------------------


def test_market_entry_fills_at_bar_open_rounded_to_tick() -> None:
    broker = PaperBroker(FakeCostModel(), FakeExitResolver())
    order = _entry("e1", kind="market")
    broker.submit(order)
    bar = _bar(TS0, open_=101.3)

    fills = broker.on_bar(bar, 0)

    assert len(fills) == 1
    assert fills[0].price == round_to_tick(101.3, BTC.tick_size)
    assert fills[0].order_id == "e1"
    assert fills[0].leg == "entry"


def test_limit_buy_fills_at_better_of_open_and_price_when_touched() -> None:
    broker = PaperBroker(FakeCostModel(), FakeExitResolver())
    broker.submit(_entry("e1", kind="limit", price=100.0, direction=1))
    bar = _bar(TS0, open_=102.0, low=99.0, high=103.0)

    fills = broker.on_bar(bar, 0)

    assert len(fills) == 1
    assert fills[0].price == 100.0  # min(price=100, open=102)


def test_limit_buy_gapped_through_fills_at_open() -> None:
    broker = PaperBroker(FakeCostModel(), FakeExitResolver())
    broker.submit(_entry("e1", kind="limit", price=100.0, direction=1))
    bar = _bar(TS0, open_=98.0, low=97.0, high=99.0)

    fills = broker.on_bar(bar, 0)

    assert len(fills) == 1
    assert fills[0].price == 98.0  # min(price=100, open=98)


def test_limit_buy_not_touched_stays_pending() -> None:
    broker = PaperBroker(FakeCostModel(), FakeExitResolver())
    broker.submit(_entry("e1", kind="limit", price=100.0, direction=1))
    bar = _bar(TS0, open_=105.0, low=101.0, high=106.0)

    fills = broker.on_bar(bar, 0)

    assert fills == []
    assert "e1" in broker._pending


def test_limit_sell_fills_at_better_of_open_and_price_when_touched() -> None:
    broker = PaperBroker(FakeCostModel(), FakeExitResolver())
    broker.submit(_entry("e1", kind="limit", price=110.0, direction=-1))
    bar = _bar(TS0, open_=108.0, low=107.0, high=115.0)

    fills = broker.on_bar(bar, 0)

    assert len(fills) == 1
    assert fills[0].price == 110.0  # max(price=110, open=108)


def test_limit_sell_gapped_through_fills_at_open() -> None:
    broker = PaperBroker(FakeCostModel(), FakeExitResolver())
    broker.submit(_entry("e1", kind="limit", price=110.0, direction=-1))
    bar = _bar(TS0, open_=112.0, low=109.0, high=113.0)

    fills = broker.on_bar(bar, 0)

    assert len(fills) == 1
    assert fills[0].price == 112.0  # max(price=110, open=112)


# -- pending -> expired / cancelled --------------------------------------


def test_order_still_live_on_expiry_bar_but_dropped_after() -> None:
    broker = PaperBroker(FakeCostModel(), FakeExitResolver())
    broker.submit(_entry("e1", kind="limit", price=100.0, expires_at_bar=3))

    not_touched = _bar(TS0, open_=102.0, low=101.0, high=103.0)
    fills = broker.on_bar(not_touched, 3)  # bar_index == expires_at_bar: still live
    assert fills == []
    assert "e1" in broker._pending

    would_touch = _bar(TS0 + timedelta(hours=4), open_=95.0, low=90.0, high=100.0)
    fills2 = broker.on_bar(would_touch, 4)  # bar_index > expires_at_bar: dropped, no fill
    assert fills2 == []
    assert "e1" not in broker._pending


def test_cancel_removes_pending_order_and_unknown_id_is_noop() -> None:
    broker = PaperBroker(FakeCostModel(), FakeExitResolver())
    broker.submit(_entry("e1", kind="limit", price=100.0))

    broker.cancel("e1")
    broker.cancel("does-not-exist")  # no-op, must not raise

    bar = _bar(TS0, open_=100.0, low=90.0, high=110.0)  # would touch 100 if still pending
    fills = broker.on_bar(bar, 0)
    assert fills == []


# -- replacement / additive keys ------------------------------------------


def test_submit_replaces_pending_order_by_trade_id_and_leg() -> None:
    resolver = FakeExitResolver()
    broker = PaperBroker(FakeCostModel(), resolver)
    broker.submit(_entry("e1", kind="market", trade_id="t1"))
    broker.on_bar(_bar(TS0, open_=100.0), 0)  # entry fills, trade "t1" opens at 100

    ret_a = broker.submit(_leg("stop-a", "t1", "stop", direction=-1, kind="stop", price=95.0))
    ret_b = broker.submit(_leg("stop-b", "t1", "stop", direction=-1, kind="stop", price=90.0))
    assert ret_a == "stop-a"
    assert ret_b == "stop-b"
    assert "stop-a" not in broker._pending

    broker.cancel("stop-a")  # already replaced: no-op

    bar1 = _bar(TS0 + timedelta(hours=4), open_=100.0, low=89.0, high=101.0)
    fills = broker.on_bar(bar1, 1)

    assert len(fills) == 1
    assert fills[0].price == 90.0  # used stop-b's level, not the replaced stop-a
    assert resolver.calls[-1]["stop"] == 90.0


def test_entries_with_trade_id_none_are_additive() -> None:
    broker = PaperBroker(FakeCostModel(), FakeExitResolver())
    broker.submit(_entry("e1", kind="market"))
    broker.submit(_entry("e2", kind="market"))

    fills = broker.on_bar(_bar(TS0, open_=100.0), 0)

    assert {f.order_id for f in fills} == {"e1", "e2"}


# -- full flows: entry -> stop, entry -> partial -> breakeven --------------


def test_full_entry_to_stop_flow_within_same_bar() -> None:
    """Mirrors the engine's documented double `on_bar` call on an entry bar."""
    cost_model = FakeCostModel()
    resolver = FakeExitResolver()
    broker = PaperBroker(cost_model, resolver)
    broker.submit(_entry("e1", kind="market", trade_id="t1"))

    bar0 = _bar(TS0, open_=100.0, low=94.0, high=101.0)
    first = broker.on_bar(bar0, 0)
    assert [f.leg for f in first] == ["entry"]
    assert first[0].price == 100.0

    broker.submit(_leg("stop1", "t1", "stop", direction=-1, kind="stop", price=95.0))
    second = broker.on_bar(bar0, 0)  # same bar, same bar_index: resolves the new leg

    assert [f.leg for f in second] == ["stop"]
    assert second[0].price == 95.0
    assert second[0].qty == 1.0
    assert broker.positions() == []
    assert len(broker.log) == 2
    assert cost_model.carry_calls == []  # never open across a bar boundary: no carry


def test_entry_partial_then_breakeven_stop_in_same_bar() -> None:
    """A partial and the post-partial breakeven stop both trigger inside one bar.

    The resolver reports the *effective* stop level (`stop_after_partial`, i.e. breakeven)
    for the second event, and the broker must fill at that price -- not at the original,
    unmoved stop order's price -- since the stop truly executed at breakeven.
    """
    resolver = FakeExitResolver()
    broker = PaperBroker(FakeCostModel(), resolver)
    broker.submit(_entry("e1", kind="market", trade_id="t1"))
    broker.on_bar(_bar(TS0, open_=100.0), 0)  # entry_price = 100.0

    broker.submit(_leg("stop1", "t1", "stop", direction=-1, kind="stop", price=95.0))
    broker.submit(_leg("partial1", "t1", "partial", direction=-1, kind="limit", price=105.0, qty=0.5))
    broker.submit(_leg("target1", "t1", "target", direction=-1, kind="limit", price=110.0, qty=0.5))

    # low=99 is below breakeven(100) but above the original stop(95); high=106 touches the
    # partial(105) but not the runner(110).
    bar1 = _bar(TS0 + timedelta(hours=4), open_=100.0, low=99.0, high=106.0)
    fills = broker.on_bar(bar1, 1)

    assert resolver.calls[-1]["stop_after_partial"] == 100.0  # entry price

    assert [f.leg for f in fills] == ["partial", "stop"]
    assert fills[0].price == 105.0
    assert fills[0].qty == 0.5
    assert fills[1].price == 100.0  # breakeven, not the original stop level of 95.0
    assert fills[1].qty == 0.5
    assert broker.positions() == []


def test_exit_fill_cost_is_costed_against_the_remaining_qty_not_the_order_qty() -> None:
    """Same-bar partial-then-stop, with a qty-proportional cost model.

    The stop order's own `qty` is still the original entry size (1.0) -- pending orders
    are never re-submitted with a shrunk qty after a partial. The *fill* correctly reports
    qty=0.5 (the remaining size). Before the I1 fix, `_make_fill` priced every fill against
    the pending order's `qty` field, so the stop's cost would have been computed for 1.0
    units, not the 0.5 actually remaining -- silently doubling a qty-proportional cost.
    """
    cost_model = ProportionalCostModel(per_unit=10.0)
    resolver = FakeExitResolver()
    broker = PaperBroker(cost_model, resolver)
    broker.submit(_entry("e1", kind="market", trade_id="t1"))
    broker.on_bar(_bar(TS0, open_=100.0), 0)  # entry_price = 100.0

    broker.submit(_leg("stop1", "t1", "stop", direction=-1, kind="stop", price=95.0))
    broker.submit(_leg("partial1", "t1", "partial", direction=-1, kind="limit", price=105.0, qty=0.5))
    broker.submit(_leg("target1", "t1", "target", direction=-1, kind="limit", price=110.0, qty=0.5))

    bar1 = _bar(TS0 + timedelta(hours=4), open_=100.0, low=99.0, high=106.0)
    fills = broker.on_bar(bar1, 1)

    assert [f.leg for f in fills] == ["partial", "stop"]
    partial_fill, stop_fill = fills
    assert partial_fill.qty == 0.5
    assert partial_fill.cost.commission == pytest.approx(0.5 * 10.0)
    assert stop_fill.qty == 0.5  # the remaining size, not the stop order's original 1.0
    assert stop_fill.cost.commission == pytest.approx(0.5 * 10.0)  # costed for 0.5, not 1.0


def test_partial_then_exact_close_leaves_no_phantom_position() -> None:
    """qty 1.0, partial 0.7, remainder 0.3 -- a case where `1.0 - 0.7 != 0.3` in float64.

    Closing the trade must still leave exactly no position, regardless of how the leg
    quantities summed in floating point (WU-1E code-review pass, I2).
    """
    assert 1.0 - 0.7 != 0.3  # documents the float-dust case this test guards against
    resolver = FakeExitResolver()
    broker = PaperBroker(FakeCostModel(), resolver)
    broker.submit(_entry("e1", kind="market", trade_id="t1", qty=1.0))
    broker.on_bar(_bar(TS0, open_=100.0), 0)

    broker.submit(_leg("stop1", "t1", "stop", direction=-1, kind="stop", price=95.0))
    broker.submit(_leg("partial1", "t1", "partial", direction=-1, kind="limit", price=105.0, qty=0.7))
    broker.submit(_leg("target1", "t1", "target", direction=-1, kind="limit", price=110.0, qty=0.3))

    bar1 = _bar(TS0 + timedelta(hours=4), open_=103.0, low=101.0, high=106.0)  # touches partial only
    first = broker.on_bar(bar1, 1)
    assert [f.leg for f in first] == ["partial"]
    assert broker.positions()[0].qty == pytest.approx(0.3)

    bar2 = _bar(TS0 + timedelta(hours=8), open_=105.0, low=96.0, high=111.0)  # touches the runner target
    second = broker.on_bar(bar2, 2)

    assert [f.leg for f in second] == ["target"]
    assert second[0].qty == 0.3
    assert broker.positions() == []  # no phantom residue position
    assert "t1" not in broker._trades  # closed trades are dropped, not kept flat


def test_cancelled_remainder_still_reports_the_real_residual_position() -> None:
    """Cancelling the runner leg (instead of letting it fill) must not be mistaken for a
    close: the epsilon that absorbs float dust around zero must not also swallow a real,
    deliberately-left-open remainder.
    """
    resolver = FakeExitResolver()
    broker = PaperBroker(FakeCostModel(), resolver)
    broker.submit(_entry("e1", kind="market", trade_id="t1", qty=1.0))
    broker.on_bar(_bar(TS0, open_=100.0), 0)

    broker.submit(_leg("stop1", "t1", "stop", direction=-1, kind="stop", price=95.0))
    broker.submit(_leg("partial1", "t1", "partial", direction=-1, kind="limit", price=105.0, qty=0.7))
    broker.submit(_leg("target1", "t1", "target", direction=-1, kind="limit", price=110.0, qty=0.3))

    bar1 = _bar(TS0 + timedelta(hours=4), open_=103.0, low=101.0, high=106.0)  # touches partial only
    broker.on_bar(bar1, 1)
    assert broker.positions()[0].qty == pytest.approx(0.3)

    broker.cancel("target1")  # operator cancels the remainder instead of letting it fill

    assert "t1" in broker._trades  # cancelling a leg does not close the trade
    assert broker.positions()[0].qty == pytest.approx(0.3)  # still a real, open residual


def test_pending_target_without_a_stop_leg_raises() -> None:
    """Engine invariant: every exit rule attaches a stop before any target/partial. A
    trade with a pending target but no stop means that invariant was violated somewhere
    upstream -- this must fail loudly, not silently skip resolving the trade forever.
    """
    broker = PaperBroker(FakeCostModel(), FakeExitResolver())
    broker.submit(_entry("e1", kind="market", trade_id="t1"))
    broker.on_bar(_bar(TS0, open_=100.0), 0)

    broker.submit(_leg("target1", "t1", "target", direction=-1, kind="limit", price=110.0, qty=1.0))
    # deliberately no stop leg submitted

    bar1 = _bar(TS0 + timedelta(hours=4), open_=103.0, low=101.0, high=111.0)
    with pytest.raises(RuntimeError, match="no stop leg"):
        broker.on_bar(bar1, 1)


def test_carry_accrues_exactly_once_when_on_bar_is_called_twice_on_the_exit_bar() -> None:
    """Regression pin for I8: the same idempotency guard that protects the entry bar (see
    `test_full_entry_to_stop_flow_within_same_bar`) must also hold when the *second* call
    on the same `bar_index` is the one that would otherwise re-accrue carry right before an
    exit fill.
    """
    cost_model = FakeCostModel(carry_amount=3.0)
    broker = PaperBroker(cost_model, FakeExitResolver())
    broker.submit(_entry("e1", kind="market", trade_id="t1"))
    broker.on_bar(_bar(TS0, open_=100.0), 0)  # bar 0: entry, no carry (position just opened)

    broker.submit(_leg("stop1", "t1", "stop", direction=-1, kind="stop", price=95.0))
    exit_bar = _bar(TS0 + timedelta(hours=4), open_=100.0, low=90.0, high=101.0)

    first = broker.on_bar(exit_bar, 1)  # carry accrues once (+3), then the stop fills
    second = broker.on_bar(exit_bar, 1)  # same bar_index again: must be a no-op

    assert len(cost_model.carry_calls) == 1
    assert [f.leg for f in first] == ["stop"]
    assert first[0].cost.funding == 3.0
    assert second == []


def test_partial_then_runner_target_closes_the_trade() -> None:
    """The partial is taken on one bar; the runner's target closes the trade on a later one."""
    resolver = FakeExitResolver()
    broker = PaperBroker(FakeCostModel(), resolver)
    broker.submit(_entry("e1", kind="market", trade_id="t1"))
    broker.on_bar(_bar(TS0, open_=100.0), 0)

    broker.submit(_leg("stop1", "t1", "stop", direction=-1, kind="stop", price=95.0))
    broker.submit(_leg("partial1", "t1", "partial", direction=-1, kind="limit", price=105.0, qty=0.5))
    broker.submit(_leg("target1", "t1", "target", direction=-1, kind="limit", price=110.0, qty=0.5))

    # low=101 stays above breakeven(100), so the post-partial stop isn't (re-)touched here.
    bar1 = _bar(TS0 + timedelta(hours=4), open_=103.0, low=101.0, high=106.0)
    first = broker.on_bar(bar1, 1)
    assert [f.leg for f in first] == ["partial"]
    assert broker.positions()[0].qty == 0.5

    bar2 = _bar(TS0 + timedelta(hours=8), open_=105.0, low=96.0, high=111.0)  # stop(95) not touched
    second = broker.on_bar(bar2, 2)

    assert [f.leg for f in second] == ["target"]
    assert second[0].price == 110.0
    assert second[0].qty == 0.5
    assert broker.positions() == []


def test_time_leg_fills_at_open_and_bypasses_the_resolver() -> None:
    resolver = FakeExitResolver()
    broker = PaperBroker(FakeCostModel(), resolver)
    broker.submit(_entry("e1", kind="market", trade_id="t1"))
    broker.on_bar(_bar(TS0, open_=100.0), 0)

    broker.submit(_leg("stop1", "t1", "stop", direction=-1, kind="stop", price=95.0))
    broker.submit(_leg("time1", "t1", "time", direction=-1, kind="market", price=None))

    # low=90 would also trigger the stop, if the resolver were ever consulted.
    bar1 = _bar(TS0 + timedelta(hours=4), open_=103.0, low=90.0, high=104.0)
    fills = broker.on_bar(bar1, 1)

    assert len(fills) == 1
    assert fills[0].leg == "time"
    assert fills[0].price == 103.0  # bar.open, not any exit level
    assert fills[0].qty == 1.0
    assert broker.positions() == []
    assert resolver.calls == []


# -- costs and carry -----------------------------------------------------------


def test_fill_carries_cost_model_breakdown() -> None:
    custom_cost = CostBreakdown(spread=0.2, commission=0.1, slippage=0.05)
    cost_model = FakeCostModel(entry_cost=custom_cost)
    broker = PaperBroker(cost_model, FakeExitResolver())
    order = _entry("e1", kind="market")
    broker.submit(order)
    bar = _bar(TS0, open_=100.0)

    fills = broker.on_bar(bar, 0)

    assert fills[0].cost == custom_cost
    assert cost_model.entry_calls == [(order, bar)]


def test_carry_accrues_across_bars_into_the_exit_fills_funding() -> None:
    cost_model = FakeCostModel(carry_amount=3.0)
    broker = PaperBroker(cost_model, FakeExitResolver())
    broker.submit(_entry("e1", kind="market", trade_id="t1"))
    broker.on_bar(_bar(TS0, open_=100.0), 0)  # bar 0: entry, no carry (position just opened)

    holding = _bar(TS0 + timedelta(hours=4), open_=100.0, low=99.0, high=101.0)  # no stop touch
    assert broker.on_bar(holding, 1) == []  # bar 1: carry accrues (+3)

    broker.submit(_leg("stop1", "t1", "stop", direction=-1, kind="stop", price=95.0))
    exit_bar = _bar(TS0 + timedelta(hours=8), open_=100.0, low=90.0, high=101.0)
    fills = broker.on_bar(exit_bar, 2)  # bar 2: carry accrues again (+3), then the stop fills

    assert len(fills) == 1
    assert fills[0].cost.funding == 6.0
    assert len(cost_model.carry_calls) == 2


# -- log / idempotency / streaming ------------------------------------------


def test_log_records_fill_order_and_bar_open_ts() -> None:
    broker = PaperBroker(FakeCostModel(), FakeExitResolver())
    order = _entry("e1", kind="market")
    broker.submit(order)
    bar = _bar(TS0, open_=100.0)

    broker.on_bar(bar, 0)

    assert len(broker.log) == 1
    entry = broker.log[0]
    assert entry.fill.order_id == "e1"
    assert entry.order == order
    assert entry.bar_ts_open == TS0


def test_on_bar_is_idempotent_for_already_filled_orders() -> None:
    broker = PaperBroker(FakeCostModel(), FakeExitResolver())
    broker.submit(_entry("e1", kind="market"))
    bar = _bar(TS0, open_=100.0)

    first = broker.on_bar(bar, 0)
    second = broker.on_bar(bar, 0)

    assert len(first) == 1
    assert second == []
    assert len(broker.log) == 1


def test_fills_async_iterator_yields_produced_fills() -> None:
    broker = PaperBroker(FakeCostModel(), FakeExitResolver())
    broker.submit(_entry("e1", kind="market"))
    bar = _bar(TS0, open_=100.0)
    produced = broker.on_bar(bar, 0)

    async def _collect_one() -> object:
        agen = broker.fills()
        return await agen.__anext__()

    got = asyncio.run(_collect_one())
    assert got == produced[0]


def test_fills_awaits_readiness_event_instead_of_busy_waiting() -> None:
    """Regression pin: a consumer with an empty queue must suspend on an `asyncio.Event`,
    not spin-poll via `while True: await asyncio.sleep(0)`. We patch `asyncio.Event.wait`
    to record that it was actually awaited -- a busy-waiting implementation never calls
    `Event.wait` at all, so `wait_calls` is what actually distinguishes the two
    implementations (both leave the task merely "still pending" after a short sleep).
    """
    broker = PaperBroker(FakeCostModel(), FakeExitResolver())
    wait_calls = 0
    original_wait = asyncio.Event.wait

    async def _tracking_wait(self: asyncio.Event) -> bool:
        nonlocal wait_calls
        wait_calls += 1
        return await original_wait(self)

    async def _run() -> None:
        agen = broker.fills()
        task = asyncio.ensure_future(agen.__anext__())
        await asyncio.sleep(0.05)

        assert not task.done()  # nothing produced yet: consumer is still waiting
        assert wait_calls >= 1  # ... but by awaiting the event, not by spinning

        broker.submit(_entry("e1", kind="market"))
        broker.on_bar(_bar(TS0, open_=100.0), 0)

        fill = await task
        assert fill.order_id == "e1"
        await agen.aclose()

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(asyncio.Event, "wait", _tracking_wait)
        asyncio.run(_run())


def test_fills_delivers_fills_produced_after_the_consumer_started_in_order() -> None:
    """Fills produced by `on_bar` calls made *after* a consumer is already awaiting
    `fills()` on an empty queue must still be delivered, in the order they were produced.
    """
    broker = PaperBroker(FakeCostModel(), FakeExitResolver())

    async def _run() -> list[str]:
        agen = broker.fills()
        collected: list[str] = []

        async def _consume() -> None:
            try:
                async for fill in agen:
                    collected.append(fill.order_id)
                    if len(collected) == 2:
                        return
            finally:
                await agen.aclose()

        task = asyncio.ensure_future(_consume())
        await asyncio.sleep(0)  # let the consumer start awaiting on the empty queue

        broker.submit(_entry("e1", kind="market"))
        broker.on_bar(_bar(TS0, open_=100.0), 0)
        broker.submit(_entry("e2", kind="market"))
        broker.on_bar(_bar(TS0 + timedelta(hours=4), open_=100.0), 1)

        await task
        return collected

    collected = asyncio.run(_run())
    assert collected == ["e1", "e2"]


def test_close_drains_pending_fills_before_ending_the_stream() -> None:
    broker = PaperBroker(FakeCostModel(), FakeExitResolver())
    broker.submit(_entry("e1", kind="market"))
    broker.on_bar(_bar(TS0, open_=100.0), 0)  # one fill queued before the consumer starts

    broker.close()

    async def _run() -> list[str]:
        return [fill.order_id async for fill in broker.fills()]

    collected = asyncio.run(_run())
    assert collected == ["e1"]  # drained before the stream ended


def test_close_wakes_a_blocked_consumer_and_ends_the_stream() -> None:
    """The bug `close()` fixes: before it existed, a consumer blocked on an empty queue
    had no way to be told the stream is over, so `async for fill in broker.fills(): ...`
    would hang forever. `asyncio.wait_for` turns that hang into a test failure instead of
    an indefinite wait.
    """
    broker = PaperBroker(FakeCostModel(), FakeExitResolver())

    async def _run() -> list[str]:
        collected: list[str] = []

        async def _consume() -> None:
            async for fill in broker.fills():
                collected.append(fill.order_id)

        task = asyncio.ensure_future(_consume())
        await asyncio.sleep(0.05)
        assert not task.done()  # blocked: queue is empty

        broker.close()
        await asyncio.wait_for(task, timeout=1.0)
        return collected

    collected = asyncio.run(_run())
    assert collected == []
