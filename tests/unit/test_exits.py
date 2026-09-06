"""Unit tests for the 16 exit-rule tournament variants (`strategies/exits.py`).

Synthetic helpers only: `path` builds flat 4H bars from a list of closes, `four_h_bar` builds
one 4H bar with an explicit high/low (for the Trailing tests, which need real range), and
`daily_bars` builds a constant-range Daily series so `ctx.atr("1d")` comes out to an exact,
known value. `make_trade` builds a valid `Trade` from the handful of numbers each test cares
about, computing `risk_r` the way the engine does: `|entry - stop| * qty * multiplier`.
"""

from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Literal

import pytest

from swingforge.core.context import Context
from swingforge.core.types import Bar, CostBreakdown, Fill, Instrument, Signal, Trade
from swingforge.strategies.base import ExitRule
from swingforge.strategies.exits import (
    EXIT_GRID,
    ATRFixedR,
    FixedR,
    Partial,
    Structure,
    Trailing,
    remaining_qty,
    unrealized_r,
)

INSTRUMENT = Instrument(
    venue="hyperliquid",
    symbol="BTC",
    tick_size=Decimal("0.01"),
    contract_multiplier=Decimal("1"),
    quote_ccy="USD",
    session_profile="perp",
)
START = datetime(2026, 1, 1, tzinfo=UTC)


# --- synthetic helpers -------------------------------------------------------


def path(prices: list[float]) -> list[Bar]:
    """Flat 4H bars (open = high = low = close) from a list of closes, 4H apart."""
    bars = []
    for i, price in enumerate(prices):
        bars.append(
            Bar(
                instrument=INSTRUMENT,
                tf="4h",
                ts_open=START + timedelta(hours=4 * i),
                open=price,
                high=price,
                low=price,
                close=price,
                volume=1.0,
            )
        )
    return bars


def four_h_bar(index: int, high: float, low: float, close: float) -> Bar:
    """One 4H bar with a real range, for tests that need the high/low distinct from close."""
    open_ = close
    return Bar(
        instrument=INSTRUMENT,
        tf="4h",
        ts_open=START + timedelta(hours=4 * index),
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=1.0,
    )


def daily_bars(n: int, base: float = 100.0, rng: float = 4.0) -> list[Bar]:
    """A constant-range Daily series: every true range is exactly `rng`, so ATR == rng."""
    bars = []
    for i in range(n):
        bars.append(
            Bar(
                instrument=INSTRUMENT,
                tf="1d",
                ts_open=START + timedelta(hours=24 * i),
                open=base,
                high=base + rng / 2,
                low=base - rng / 2,
                close=base,
                volume=1.0,
            )
        )
    return bars


def make_trade(
    entry: float,
    stop: float,
    qty: float,
    direction: Literal[1, -1],
    structure_target: float | None = None,
    *,
    opened_bar: int = 0,
    legs: tuple[Fill, ...] = (),
    trade_id: str = "t1",
) -> Trade:
    mult = float(INSTRUMENT.contract_multiplier)
    risk_r = abs(entry - stop) * qty * mult
    entry_fill = Fill(
        order_id=f"{trade_id}-entry",
        ts=START,
        price=entry,
        qty=qty,
        cost=CostBreakdown(),
        leg="entry",
        trade_id=trade_id,
    )
    return Trade(
        id=trade_id,
        instrument=INSTRUMENT,
        direction=direction,
        entry_fill=entry_fill,
        legs=legs,
        stop=stop,
        target=structure_target,
        risk_r=risk_r,
        opened_bar=opened_bar,
    )


def atr_context(n: int = 15) -> Context:
    ctx = Context(INSTRUMENT)
    for bar in daily_bars(n):
        ctx.push(bar)
    return ctx


# --- helpers: remaining_qty / unrealized_r ----------------------------------


def test_remaining_qty_subtracts_filled_legs() -> None:
    entry_fill = Fill(order_id="e", ts=START, price=100.0, qty=2.0, cost=CostBreakdown(), leg="entry")
    leg = Fill(order_id="p", ts=START, price=105.0, qty=0.5, cost=CostBreakdown(), leg="partial")
    trade = Trade(
        id="t",
        instrument=INSTRUMENT,
        direction=1,
        entry_fill=entry_fill,
        legs=(leg,),
        stop=95.0,
        target=None,
        risk_r=10.0,
        opened_bar=0,
    )
    assert remaining_qty(trade) == pytest.approx(1.5)


def test_unrealized_r_matches_formula() -> None:
    trade = make_trade(entry=100.0, stop=95.0, qty=2.0, direction=1)
    expected = 1 * (105.0 - 100.0) * 2.0 * 1.0 / trade.risk_r
    assert unrealized_r(trade, 105.0) == pytest.approx(expected)


def test_unrealized_r_uses_remaining_qty_after_a_partial() -> None:
    entry_fill = Fill(order_id="e", ts=START, price=100.0, qty=2.0, cost=CostBreakdown(), leg="entry")
    leg = Fill(order_id="p", ts=START, price=105.0, qty=1.0, cost=CostBreakdown(), leg="partial")
    trade = Trade(
        id="t",
        instrument=INSTRUMENT,
        direction=1,
        entry_fill=entry_fill,
        legs=(leg,),
        stop=100.0,
        target=None,
        risk_r=10.0,
        opened_bar=0,
    )
    # Only 1.0 unit remains open, not the original 2.0.
    assert unrealized_r(trade, 110.0) == pytest.approx(1 * (110.0 - 100.0) * 1.0 * 1.0 / 10.0)


# --- FixedR ------------------------------------------------------------------


def test_fixed_r_initial_stop_is_the_signal_stop() -> None:
    rule = FixedR(2)
    signal = Signal(direction=1, entry=100.0, stop=95.0, structure_target=None, tag="t", expires_in_bars=3)
    ctx = Context(INSTRUMENT)
    assert rule.initial_stop(signal, ctx) == 95.0


def test_fixed_r_attach_builds_stop_and_target() -> None:
    rule = FixedR(2)
    trade = make_trade(entry=100.0, stop=95.0, qty=2.0, direction=1)
    ctx = Context(INSTRUMENT)
    orders = rule.attach(trade, ctx)
    assert len(orders) == 2
    stop_o, target_o = orders
    assert stop_o.leg == "stop"
    assert stop_o.kind == "stop"
    assert stop_o.price == pytest.approx(95.0)
    assert stop_o.qty == pytest.approx(2.0)
    assert stop_o.direction == -1
    assert stop_o.trade_id == trade.id
    assert target_o.leg == "target"
    assert target_o.kind == "limit"
    assert target_o.price == pytest.approx(110.0)
    assert target_o.qty == pytest.approx(2.0)
    assert target_o.direction == -1
    assert target_o.trade_id == trade.id


def test_fixed_r_attach_target_for_short() -> None:
    rule = FixedR(2)
    trade = make_trade(entry=100.0, stop=105.0, qty=1.0, direction=-1)
    ctx = Context(INSTRUMENT)
    orders = rule.attach(trade, ctx)
    target_o = orders[1]
    assert target_o.price == pytest.approx(90.0)


# --- ATRFixedR -----------------------------------------------------------------


def test_atr_fixed_r_initial_stop_overrides_signal_stop() -> None:
    ctx = atr_context()
    assert ctx.atr("1d") == pytest.approx(4.0)
    rule = ATRFixedR(rr=2, atr_mult=1.5)
    signal = Signal(direction=1, entry=100.0, stop=90.0, structure_target=None, tag="t", expires_in_bars=3)
    assert rule.initial_stop(signal, ctx) == pytest.approx(94.0)


def test_atr_fixed_r_attach_target_uses_atr_distance() -> None:
    ctx = atr_context()
    rule = ATRFixedR(rr=2, atr_mult=1.5)
    trade = make_trade(entry=100.0, stop=94.0, qty=1.0, direction=1)
    orders = rule.attach(trade, ctx)
    stop_o, target_o = orders
    assert stop_o.price == pytest.approx(94.0)
    assert target_o.price == pytest.approx(112.0)


def test_atr_fixed_r_initial_stop_falls_back_when_atr_is_nan() -> None:
    ctx = Context(INSTRUMENT)  # no Daily bars pushed
    rule = ATRFixedR(rr=2, atr_mult=1.5)
    signal = Signal(direction=1, entry=100.0, stop=90.0, structure_target=None, tag="t", expires_in_bars=3)
    assert rule.initial_stop(signal, ctx) == 90.0


# --- Structure -----------------------------------------------------------------


def test_structure_uses_structure_target_when_it_clears_1_5r() -> None:
    rule = Structure(fallback_rr=2)
    trade = make_trade(entry=100.0, stop=95.0, qty=1.0, direction=1, structure_target=100.0 + 1.6 * 5)
    ctx = Context(INSTRUMENT)
    orders = rule.attach(trade, ctx)
    assert orders[1].price == pytest.approx(100.0 + 1.6 * 5)


def test_structure_falls_back_when_target_is_too_close() -> None:
    rule = Structure(fallback_rr=2)
    trade = make_trade(entry=100.0, stop=95.0, qty=1.0, direction=1, structure_target=100.0 + 1.4 * 5)
    ctx = Context(INSTRUMENT)
    orders = rule.attach(trade, ctx)
    assert orders[1].price == pytest.approx(110.0)  # fallback: entry + 2 * r_dist(5)


def test_structure_falls_back_when_there_is_no_structure_target() -> None:
    rule = Structure(fallback_rr=2)
    trade = make_trade(entry=100.0, stop=95.0, qty=1.0, direction=1, structure_target=None)
    ctx = Context(INSTRUMENT)
    orders = rule.attach(trade, ctx)
    assert orders[1].price == pytest.approx(110.0)


def test_structure_initial_stop_is_the_signal_stop() -> None:
    rule = Structure(fallback_rr=2)
    signal = Signal(direction=1, entry=100.0, stop=95.0, structure_target=None, tag="t", expires_in_bars=3)
    ctx = Context(INSTRUMENT)
    assert rule.initial_stop(signal, ctx) == 95.0


# --- Trailing --------------------------------------------------------------


def test_trailing_initial_stop_is_the_signal_stop() -> None:
    rule = Trailing(activate_r=1, trail_atr=2)
    signal = Signal(direction=1, entry=100.0, stop=95.0, structure_target=None, tag="t", expires_in_bars=3)
    ctx = Context(INSTRUMENT)
    assert rule.initial_stop(signal, ctx) == 95.0


def test_trailing_attach_returns_only_a_stop() -> None:
    rule = Trailing(activate_r=1, trail_atr=2)
    trade = make_trade(entry=100.0, stop=95.0, qty=1.0, direction=1)
    ctx = Context(INSTRUMENT)
    orders = rule.attach(trade, ctx)
    assert [o.leg for o in orders] == ["stop"]


def test_trailing_does_not_move_before_activation() -> None:
    ctx = atr_context()
    rule = Trailing(activate_r=1, trail_atr=2)
    trade = make_trade(entry=100.0, stop=95.0, qty=1.0, direction=1, opened_bar=0)
    ctx.push(four_h_bar(0, high=101.0, low=99.0, close=100.0))  # threshold is 105
    assert rule.on_bar(trade, ctx) == []


def test_trailing_activates_and_moves_stop_up_for_a_long() -> None:
    ctx = atr_context()
    rule = Trailing(activate_r=1, trail_atr=2)
    trade = make_trade(entry=100.0, stop=95.0, qty=1.0, direction=1, opened_bar=0)
    ctx.push(four_h_bar(0, high=101.0, low=99.0, close=100.0))
    assert rule.on_bar(trade, ctx) == []

    ctx.push(four_h_bar(1, high=110.0, low=104.0, close=108.0))  # crosses the 105 threshold
    orders = rule.on_bar(trade, ctx)
    assert len(orders) == 1
    assert orders[0].leg == "stop"
    assert orders[0].price == pytest.approx(102.0)  # max(95, 110 - 2*4)
    trade = trade.model_copy(update={"stop": orders[0].price})

    # A lower high than the running max doesn't move the stop.
    ctx.push(four_h_bar(2, high=105.0, low=101.0, close=103.0))
    assert rule.on_bar(trade, ctx) == []

    # A new high moves it further up, never down.
    ctx.push(four_h_bar(3, high=130.0, low=120.0, close=125.0))
    orders = rule.on_bar(trade, ctx)
    assert len(orders) == 1
    assert orders[0].price == pytest.approx(122.0)  # max(102, 130 - 2*4)


def test_trailing_activates_and_moves_stop_down_for_a_short() -> None:
    ctx = atr_context()
    rule = Trailing(activate_r=1, trail_atr=2)
    trade = make_trade(entry=100.0, stop=105.0, qty=1.0, direction=-1, opened_bar=0)
    ctx.push(four_h_bar(0, high=101.0, low=99.0, close=100.0))  # threshold is 95
    assert rule.on_bar(trade, ctx) == []

    ctx.push(four_h_bar(1, high=96.0, low=90.0, close=92.0))  # crosses the 95 threshold
    orders = rule.on_bar(trade, ctx)
    assert len(orders) == 1
    assert orders[0].price == pytest.approx(98.0)  # min(105, 90 + 2*4)


def test_trailing_manage_returns_no_orders_when_opened_bar_offset_is_unresolvable() -> None:
    """If `trade.opened_bar` cannot be translated to a row offset in the pushed series —
    here because it is ahead of the bars pushed so far, which should not happen in the real
    engine but must not crash here — `_manage` returns `[]` rather than guessing a start
    point. The same path covers a trade whose entry bar has scrolled out of the context's
    retained window; the invariant that keeps that from mattering in practice is that a
    run's `max_bars` stays above its maximum holding period, so `opened_bar` is never
    trimmed away while the trade is still open."""
    ctx = atr_context()
    rule = Trailing(activate_r=1, trail_atr=2)
    trade = make_trade(entry=100.0, stop=95.0, qty=1.0, direction=1, opened_bar=5)
    ctx.push(four_h_bar(0, high=110.0, low=99.0, close=105.0))  # bar_index is 0, before opened_bar
    assert rule.on_bar(trade, ctx) == []


def test_trailing_skips_update_when_atr_is_nan() -> None:
    ctx = Context(INSTRUMENT)  # no Daily bars -> ATR NaN
    rule = Trailing(activate_r=1, trail_atr=2)
    trade = make_trade(entry=100.0, stop=95.0, qty=1.0, direction=1, opened_bar=0)
    ctx.push(four_h_bar(0, high=110.0, low=99.0, close=105.0))
    assert rule.on_bar(trade, ctx) == []


def test_trailing_never_moves_the_stop_backward_even_by_half_a_tick() -> None:
    """Round-then-compare (I1): the raw trail level (extreme minus trail_atr*ATR) can land
    a hair below an off-tick current stop. Rounding it first and only then comparing to
    `trade.stop` must reject that as a backward move, rather than rounding the old
    max()-then-round shape's result and emitting it because it doesn't float-equal the
    (unrounded) current stop.
    """
    ctx = atr_context()  # ATR("1d") == 4.0
    rule = Trailing(activate_r=0.8, trail_atr=1)
    entry_fill = Fill(order_id="e1", ts=START, price=95.0, qty=1.0, cost=CostBreakdown(), leg="entry")
    trade = Trade(
        id="t1",
        instrument=INSTRUMENT,
        direction=1,
        entry_fill=entry_fill,
        legs=(),
        stop=95.004,  # off-tick current stop
        target=None,
        risk_r=5.0,  # r_dist = 5.0, so threshold = 95 + 0.8*5 = 99.0
        opened_bar=0,
    )
    # extreme (99.001) clears the 99.0 threshold; raw trail level = 99.001 - 1*4 = 95.001,
    # which rounds to 95.00 -- below the current (off-tick) stop of 95.004.
    ctx.push(four_h_bar(0, high=99.001, low=98.0, close=99.0))
    assert rule.on_bar(trade, ctx) == []


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("direction", [1, -1])
def test_trailing_stop_is_monotone_and_eventually_activates_and_moves(
    seed: int, direction: Literal[1, -1]
) -> None:
    """Property, over random paths in both directions and several seeds: the stop never
    moves against the trade direction, never moves before the bar's extreme first reaches
    the activation threshold, and — over a long enough path drifting in the trade's favour —
    activates and moves the stop at least once (`ever_moved`)."""
    rng = random.Random(seed)
    ctx = atr_context()
    rule = Trailing(activate_r=1, trail_atr=1)
    entry = 100.0
    stop = 95.0 if direction == 1 else 105.0
    trade = make_trade(entry=entry, stop=stop, qty=1.0, direction=direction, opened_bar=0)
    r_dist = trade.risk_r / (trade.entry_fill.qty * float(trade.instrument.contract_multiplier))
    threshold = entry + direction * r_dist
    price = entry
    current_stop = trade.stop
    ever_activated = False
    ever_moved = False
    for i in range(80):
        # Biased so the path reliably drifts in the trade's favour within 80 bars.
        step = direction * rng.uniform(0, 4) + rng.uniform(-1, 1)
        close = price + step
        high = max(price, close) + rng.uniform(0, 2)
        low = min(price, close) - rng.uniform(0, 2)
        ctx.push(four_h_bar(i, high=high, low=low, close=close))
        price = close
        trade = trade.model_copy(update={"stop": current_stop})

        orders = rule.on_bar(trade, ctx)
        if orders and orders[0].leg == "time":
            break  # the time stop closed the trade; nothing left to trail
        extreme_reached = high >= threshold if direction == 1 else low <= threshold
        if not ever_activated and not extreme_reached:
            assert orders == [], "stop must not move before the activation threshold is reached"
        if orders:
            new_stop = orders[0].price
            if direction == 1:
                assert new_stop >= current_stop - 1e-9, "the long stop must never move down"
            else:
                assert new_stop <= current_stop + 1e-9, "the short stop must never move up"
            current_stop = new_stop
            ever_moved = True
        if extreme_reached:
            ever_activated = True

    assert ever_activated, "the path never crossed the activation threshold across 80 bars"
    assert ever_moved, "the stop never actually moved across 80 bars"


# --- Time stop (via FixedR, which never manages on its own) -----------------


def test_time_stop_fires_at_bar_10_when_unrealized_is_below_half_r() -> None:
    rule = FixedR(2)
    ctx = Context(INSTRUMENT)
    trade = make_trade(entry=100.0, stop=95.0, qty=1.0, direction=1, opened_bar=0)
    prices = [100.0] * 10 + [102.0]  # r_dist = 5, so 102 is +0.4R
    for bar in path(prices):
        ctx.push(bar)
    assert ctx.bar_index == 10
    orders = rule.on_bar(trade, ctx)
    assert len(orders) == 1
    order = orders[0]
    assert order.leg == "time"
    assert order.kind == "market"
    assert order.price is None
    assert order.qty == pytest.approx(1.0)
    assert order.direction == -1
    assert order.trade_id == trade.id


def test_time_stop_does_not_fire_when_unrealized_is_above_half_r() -> None:
    rule = FixedR(2)
    ctx = Context(INSTRUMENT)
    trade = make_trade(entry=100.0, stop=95.0, qty=1.0, direction=1, opened_bar=0)
    prices = [100.0] * 10 + [103.0]  # r_dist = 5, so 103 is +0.6R
    for bar in path(prices):
        ctx.push(bar)
    assert rule.on_bar(trade, ctx) == []


def test_time_stop_does_not_fire_before_bar_10() -> None:
    rule = FixedR(2)
    ctx = Context(INSTRUMENT)
    trade = make_trade(entry=100.0, stop=95.0, qty=1.0, direction=1, opened_bar=0)
    prices = [100.0] * 9  # bar_index 8, one short of the 10-bar threshold
    for bar in path(prices):
        ctx.push(bar)
    assert rule.on_bar(trade, ctx) == []


def test_time_stop_boundary_bar_9_is_quiet_bar_10_fires() -> None:
    """`bar_index - opened_bar == 9` must not fire; `== 10` must (the boundary the
    `< self.time_stop_bars` check draws)."""
    rule = FixedR(2)
    ctx = Context(INSTRUMENT)
    trade = make_trade(entry=100.0, stop=95.0, qty=1.0, direction=1, opened_bar=0)
    bars = path([100.0] * 11)
    for bar in bars[:10]:
        ctx.push(bar)
    assert ctx.bar_index - trade.opened_bar == 9
    assert rule.on_bar(trade, ctx) == []

    ctx.push(bars[10])
    assert ctx.bar_index - trade.opened_bar == 10
    orders = rule.on_bar(trade, ctx)
    assert len(orders) == 1
    assert orders[0].leg == "time"


def test_time_stop_fires_on_the_losing_side_just_inside_negative_half_r() -> None:
    rule = FixedR(2)
    ctx = Context(INSTRUMENT)
    trade = make_trade(entry=100.0, stop=95.0, qty=1.0, direction=1, opened_bar=0)
    prices = [100.0] * 10 + [98.0]  # r_dist = 5, so 98 is -0.4R
    for bar in path(prices):
        ctx.push(bar)
    orders = rule.on_bar(trade, ctx)
    assert len(orders) == 1
    assert orders[0].leg == "time"


def test_time_stop_does_not_fire_on_the_losing_side_past_negative_half_r() -> None:
    rule = FixedR(2)
    ctx = Context(INSTRUMENT)
    trade = make_trade(entry=100.0, stop=95.0, qty=1.0, direction=1, opened_bar=0)
    prices = [100.0] * 10 + [97.0]  # r_dist = 5, so 97 is -0.6R
    for bar in path(prices):
        ctx.push(bar)
    assert rule.on_bar(trade, ctx) == []


def test_time_stop_is_a_no_op_when_no_bar_has_been_pushed_yet() -> None:
    rule = FixedR(2)
    ctx = Context(INSTRUMENT)  # ctx.last("4h") is None
    trade = make_trade(entry=100.0, stop=95.0, qty=1.0, direction=1, opened_bar=-15)
    assert rule.on_bar(trade, ctx) == []


# --- Partial -----------------------------------------------------------------


class _StubTrailing(Trailing):
    """A `Trailing` subclass whose `initial_stop` is overridden, used only to prove
    `Partial.initial_stop` delegates rather than hard-coding `signal.stop` itself.
    Subclassing (rather than a free-standing stand-in) keeps `Partial.runner`'s
    `FixedR | Trailing` annotation honest (I4)."""

    def initial_stop(self, signal: Signal, ctx: Context) -> float:
        return 42.0


def test_partial_initial_stop_delegates_to_the_runner() -> None:
    rule = Partial(1, 0.5, _StubTrailing(1, 2))
    ctx = Context(INSTRUMENT)
    signal = Signal(direction=1, entry=100.0, stop=90.0, structure_target=None, tag="t", expires_in_bars=3)
    assert rule.initial_stop(signal, ctx) == 42.0


def test_partial_attach_with_fixed_r_runner() -> None:
    rule = Partial(1, 0.5, FixedR(3))
    trade = make_trade(entry=100.0, stop=95.0, qty=2.0, direction=1)
    ctx = Context(INSTRUMENT)
    orders = rule.attach(trade, ctx)
    assert len(orders) == 3
    stop_o, partial_o, target_o = orders
    assert stop_o.leg == "stop"
    assert stop_o.price == pytest.approx(95.0)
    assert stop_o.qty == pytest.approx(2.0)
    assert partial_o.leg == "partial"
    assert partial_o.kind == "limit"
    assert partial_o.price == pytest.approx(105.0)  # entry + 1 * r_dist(5)
    assert partial_o.qty == pytest.approx(1.0)  # 2.0 * 0.5
    assert target_o.leg == "target"
    assert target_o.price == pytest.approx(115.0)  # entry + 3 * r_dist(5)
    assert target_o.qty == pytest.approx(1.0)  # the remainder after the partial


def test_partial_attach_with_trailing_runner_has_no_target() -> None:
    rule = Partial(1, 0.5, Trailing(1, 2))
    trade = make_trade(entry=100.0, stop=95.0, qty=2.0, direction=1)
    ctx = Context(INSTRUMENT)
    orders = rule.attach(trade, ctx)
    assert [o.leg for o in orders] == ["stop", "partial"]


def test_partial_on_bar_does_nothing_before_a_partial_fill() -> None:
    rule = Partial(1, 0.5, FixedR(3))
    ctx = Context(INSTRUMENT)
    trade = make_trade(entry=100.0, stop=95.0, qty=2.0, direction=1, opened_bar=0)
    ctx.push(four_h_bar(0, high=101.0, low=99.0, close=100.0))
    assert rule.on_bar(trade, ctx) == []


def test_partial_on_bar_moves_to_breakeven_after_the_partial_fills() -> None:
    rule = Partial(1, 0.5, FixedR(3))
    ctx = Context(INSTRUMENT)
    entry_fill = Fill(
        order_id="e1", ts=START, price=100.0, qty=2.0, cost=CostBreakdown(), leg="entry", trade_id="t1"
    )
    partial_fill = Fill(
        order_id="p1",
        ts=START + timedelta(hours=4),
        price=105.0,
        qty=1.0,
        cost=CostBreakdown(),
        leg="partial",
        trade_id="t1",
    )
    trade = Trade(
        id="t1",
        instrument=INSTRUMENT,
        direction=1,
        entry_fill=entry_fill,
        legs=(partial_fill,),
        stop=95.0,
        target=None,
        risk_r=10.0,
        opened_bar=0,
    )
    ctx.push(four_h_bar(0, high=105.0, low=99.0, close=103.0))
    orders = rule.on_bar(trade, ctx)
    assert len(orders) == 1
    order = orders[0]
    assert order.leg == "stop"
    assert order.price == pytest.approx(100.0)
    assert order.qty == pytest.approx(1.0)  # 2.0 entered - 1.0 already scaled out


def test_partial_on_bar_is_quiet_once_breakeven_is_set_with_a_fixed_r_runner() -> None:
    rule = Partial(1, 0.5, FixedR(3))
    ctx = Context(INSTRUMENT)
    entry_fill = Fill(
        order_id="e1", ts=START, price=100.0, qty=2.0, cost=CostBreakdown(), leg="entry", trade_id="t1"
    )
    partial_fill = Fill(
        order_id="p1", ts=START, price=105.0, qty=1.0, cost=CostBreakdown(), leg="partial", trade_id="t1"
    )
    trade = Trade(
        id="t1",
        instrument=INSTRUMENT,
        direction=1,
        entry_fill=entry_fill,
        legs=(partial_fill,),
        stop=100.0,  # already at breakeven
        target=None,
        risk_r=10.0,
        opened_bar=0,
    )
    ctx.push(four_h_bar(0, high=110.0, low=99.0, close=108.0))
    assert rule.on_bar(trade, ctx) == []


def test_partial_on_bar_delegates_to_the_trailing_runner_after_breakeven() -> None:
    rule = Partial(1, 0.5, Trailing(1, 2))
    ctx = atr_context()
    entry_fill = Fill(
        order_id="e1", ts=START, price=100.0, qty=2.0, cost=CostBreakdown(), leg="entry", trade_id="t1"
    )
    partial_fill = Fill(
        order_id="p1", ts=START, price=105.0, qty=1.0, cost=CostBreakdown(), leg="partial", trade_id="t1"
    )
    trade = Trade(
        id="t1",
        instrument=INSTRUMENT,
        direction=1,
        entry_fill=entry_fill,
        legs=(partial_fill,),
        stop=100.0,  # already at breakeven -> the runner takes over
        target=None,
        risk_r=10.0,
        opened_bar=0,
    )
    ctx.push(four_h_bar(0, high=120.0, low=99.0, close=115.0))
    orders = rule.on_bar(trade, ctx)
    assert len(orders) == 1
    assert orders[0].leg == "stop"
    # r_dist = risk_r / (entry qty * mult) = 10 / 2 = 5; threshold = 100 + 5 = 105; high 120 activates.
    assert orders[0].price == pytest.approx(112.0)  # max(100, 120 - 2*4)


def test_partial_breakeven_guard_is_quiet_once_stop_matches_the_rounded_breakeven() -> None:
    """C1: an off-tick entry fill (100.003, tick 0.01) rounds to a breakeven of 100.00. Once
    a prior breakeven emission has actually set `trade.stop` to that rounded value, the next
    `on_bar` must not re-fire (the old `trade.stop != trade.entry_fill.price` check compared
    against the *unrounded* entry price, so it kept firing forever) and must fall through to
    the Trailing runner instead."""
    ctx = atr_context()  # ATR("1d") == 4.0
    rule = Partial(1, 0.5, Trailing(1, 2))
    entry_fill = Fill(
        order_id="e1", ts=START, price=100.003, qty=2.0, cost=CostBreakdown(), leg="entry", trade_id="t1"
    )
    partial_fill = Fill(
        order_id="p1", ts=START, price=105.0, qty=1.0, cost=CostBreakdown(), leg="partial", trade_id="t1"
    )
    trade = Trade(
        id="t1",
        instrument=INSTRUMENT,
        direction=1,
        entry_fill=entry_fill,
        legs=(partial_fill,),
        stop=95.0,
        target=None,
        risk_r=10.006,
        opened_bar=0,
    )
    ctx.push(four_h_bar(0, high=100.0, low=99.0, close=100.0))
    orders = rule.on_bar(trade, ctx)
    assert len(orders) == 1
    assert orders[0].leg == "stop"
    assert orders[0].price == pytest.approx(100.0)  # round_to_tick(100.003, 0.01)

    trade = trade.model_copy(update={"stop": orders[0].price})
    # Below the Trailing runner's activation threshold too, so the delegated call is quiet.
    ctx.push(four_h_bar(1, high=101.0, low=99.0, close=100.0))
    assert rule.on_bar(trade, ctx) == []


def test_partial_breakeven_guard_does_not_undo_a_pre_partial_trail() -> None:
    """C2 regression: if the Trailing runner already moved the stop above entry *before* the
    partial fills (a fast bar crossing the runner's own activation threshold), the later
    partial fill must not yank the stop back down to breakeven — the directional guard
    (`trade.direction * (breakeven - trade.stop) > 0`) must reject a "breakeven" that is
    actually behind the current stop."""
    ctx = atr_context()  # ATR("1d") == 4.0
    rule = Partial(1, 0.5, Trailing(1, 2))
    trade = make_trade(entry=100.0, stop=95.0, qty=2.0, direction=1, opened_bar=0)
    # r_dist = risk_r/(qty*mult) = 10/2 = 5; activation threshold = 105.
    ctx.push(four_h_bar(0, high=150.0, low=99.0, close=145.0))  # crosses 105, no partial fill yet
    orders = rule.on_bar(trade, ctx)
    assert len(orders) == 1
    assert orders[0].leg == "stop"
    assert orders[0].price == pytest.approx(142.0)  # max(95, 150 - 2*4), well above entry

    # Now the partial fills, with the trailed stop already in place.
    partial_fill = Fill(
        order_id="p1", ts=START, price=105.0, qty=1.0, cost=CostBreakdown(), leg="partial", trade_id="t1"
    )
    trade = trade.model_copy(update={"stop": orders[0].price, "legs": (partial_fill,)})

    # A quieter bar: no new high, so the Trailing runner itself has nothing to add either.
    ctx.push(four_h_bar(1, high=140.0, low=120.0, close=130.0))
    assert rule.on_bar(trade, ctx) == []  # must NOT reset the stop to the 100.0 breakeven


def test_partial_attach_skips_the_partial_leg_when_its_qty_rounds_to_zero() -> None:
    """I3: a tiny position where `partial_qty` rounds to 0.0 at 6dp — the partial leg is
    skipped and the trade is attached exactly as the runner alone would attach it."""
    rule = Partial(1, 0.5, FixedR(3))
    trade = make_trade(entry=100.0, stop=95.0, qty=1e-7, direction=1)
    ctx = Context(INSTRUMENT)
    assert rule.attach(trade, ctx) == FixedR(3).attach(trade, ctx)


def test_partial_attach_skips_the_partial_leg_when_the_runner_remainder_rounds_to_zero() -> None:
    """I3, the other branch: `scale_pct == 1.0` leaves nothing for the runner (`runner_qty`
    rounds to 0.0), which must also skip the partial leg rather than emit a zero-qty order."""
    rule = Partial(1, 1.0, Trailing(1, 2))
    trade = make_trade(entry=100.0, stop=95.0, qty=2.0, direction=1)
    ctx = Context(INSTRUMENT)
    assert rule.attach(trade, ctx) == Trailing(1, 2).attach(trade, ctx)


# --- EXIT_GRID ---------------------------------------------------------------


def test_exit_grid_has_16_unique_conforming_variants() -> None:
    assert len(EXIT_GRID) == 16
    names = [rule.name for rule in EXIT_GRID]
    assert len(names) == len(set(names))
    for rule in EXIT_GRID:
        assert isinstance(rule, ExitRule)


def test_exit_grid_names_match_the_contract() -> None:
    names = {rule.name for rule in EXIT_GRID}
    for expected in (
        "fixed_r_1.5",
        "fixed_r_2",
        "fixed_r_3",
        "atr_1.5_rr_2",
        "atr_1.5_rr_3",
        "atr_2_rr_2",
        "atr_2_rr_3",
        "atr_3_rr_2",
        "atr_3_rr_3",
        "structure_2",
        "trail_1_1",
        "trail_1_2",
        "trail_2_1",
        "trail_2_2",
        "partial_1_0.5_fixed_r_3",
        "partial_1_0.5_trail_1_2",
    ):
        assert expected in names
