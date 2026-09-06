"""Tests for `Engine`: the per-bar sequence documented in `swingforge.strategies.base`.

Fakes only — `core` may not import `strategies` or `adapters`, so `FakeBroker`,
`FakeStrategy` and the `FakeExitRule` variants below stand in for the real
implementations the other work units own. `FakeBroker` mirrors the contract's replacement
key `(trade_id, leg)`, expiry rule, and "stop resolved before target" pessimism.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import pytest

from swingforge.core.context import Context
from swingforge.core.engine import Engine, EngineError
from swingforge.core.portfolio import Portfolio
from swingforge.core.settings import InMemorySettingsReader, Settings, StaticSettingsReader
from swingforge.core.types import TF, Bar, CostBreakdown, Fill, Instrument, Order, Position, Signal, Trade

BTC = Instrument(
    venue="hyperliquid",
    symbol="BTC",
    tick_size=Decimal("0.5"),
    contract_multiplier=Decimal("1"),
    quote_ccy="USD",
    session_profile="perp",
)
START = datetime(2026, 1, 1, tzinfo=UTC)
_LEG_PRIORITY = {"entry": 0, "stop": 1, "partial": 2, "target": 3, "time": 4}


def make_bar(index: int, o: float, h: float, low: float, c: float, tf: TF = "4h") -> Bar:
    step = {"1h": timedelta(hours=1), "4h": timedelta(hours=4), "1d": timedelta(hours=24)}[tf]
    return Bar(
        instrument=BTC, tf=tf, ts_open=START + index * step, open=o, high=h, low=low, close=c, volume=1.0
    )


class FakeBroker:
    """Pending orders keyed by `(trade_id, leg)`, replacement on resubmission.

    Fills every pending `limit`/`market` order on the next `on_bar` whose bar range
    touches its price (market orders fill unconditionally, at `bar.open`); an order past
    its `expires_at_bar` is dropped with no fill. When a `stop` and a `target`/`partial`
    for the same trade both touch in one bar, only the `stop` fires — mirroring the
    contract's pessimistic same-bar resolution.
    """

    def __init__(self) -> None:
        self._pending: dict[tuple[str | None, str], Order] = {}
        self._untracked: dict[str, Order] = {}
        self.submitted_log: list[Order] = []
        self.cancelled_log: list[str] = []

    def submit(self, order: Order) -> str:
        self.submitted_log.append(order)
        if order.trade_id is None:
            self._untracked[order.id] = order
        else:
            self._pending[(order.trade_id, order.leg)] = order
        return order.id

    def cancel(self, order_id: str) -> None:
        self.cancelled_log.append(order_id)
        for key, order in list(self._pending.items()):
            if order.id == order_id:
                del self._pending[key]
        self._untracked.pop(order_id, None)

    def positions(self) -> list[Position]:
        return []

    def on_bar(self, bar: Bar, bar_index: int) -> list[Fill]:
        candidates: list[tuple[tuple[str | None, str], Order]] = list(self._pending.items())
        candidates += [((None, order.id), order) for order in self._untracked.values()]

        live: list[tuple[tuple[str | None, str], Order]] = []
        for key, order in candidates:
            if order.expires_at_bar is not None and bar_index > order.expires_at_bar:
                self._drop(key, order)
                continue
            live.append((key, order))

        touched: list[tuple[tuple[str | None, str], Order, float]] = []
        for key, order in live:
            price = self._touch_price(order, bar)
            if price is not None:
                touched.append((key, order, price))
        touched.sort(key=lambda item: _LEG_PRIORITY.get(item[1].leg, 99))

        stopped_trades = {order.trade_id for _, order, _ in touched if order.leg == "stop"}
        fills: list[Fill] = []
        for key, order, price in touched:
            if order.trade_id in stopped_trades and order.leg in ("target", "partial"):
                continue
            self._drop(key, order)
            fills.append(
                Fill(
                    order_id=order.id,
                    ts=bar.ts_open,
                    price=price,
                    qty=order.qty,
                    cost=CostBreakdown(),
                    leg=order.leg,
                    trade_id=order.trade_id,
                )
            )
        return fills

    async def fills(self) -> AsyncIterator[Fill]:  # pragma: no cover - not exercised
        return
        yield

    def _drop(self, key: tuple[str | None, str], order: Order) -> None:
        if order.trade_id is None:
            self._untracked.pop(order.id, None)
        else:
            self._pending.pop(key, None)

    @staticmethod
    def _touch_price(order: Order, bar: Bar) -> float | None:
        if order.kind == "market":
            return bar.open
        assert order.price is not None
        if bar.low <= order.price <= bar.high:
            return order.price
        return None


class _InjectedFillBroker(FakeBroker):
    """A `FakeBroker` that also hands back one fabricated fill on a chosen bar index.

    Used to exercise fill validation (F2/F3/F7): the engine's exit-fill handling must
    reject a fill the real broker could never produce (wrong trade_id, impossible qty, an
    entry fill with no pending entry, an exit fill with no open trade) rather than crash on
    a bare `assert` or silently corrupt state.
    """

    def __init__(self, injected_fill: Fill, on_bar_index: int) -> None:
        super().__init__()
        self._injected_fill = injected_fill
        self._on_bar_index = on_bar_index
        self._injected = False

    def on_bar(self, bar: Bar, bar_index: int) -> list[Fill]:
        fills = super().on_bar(bar, bar_index)
        if bar_index == self._on_bar_index and not self._injected:
            self._injected = True
            fills.append(self._injected_fill)
        return fills


class _StopPriceEntryBroker(FakeBroker):
    """A `FakeBroker` whose entry fill always lands at a forced price.

    Used to exercise M10: a real broker could fill an entry right on the stop level (e.g. a
    gap), which would otherwise make `risk_money` return 0 and crash `Trade` construction
    (`risk_r` is `PositiveFinite`).
    """

    def __init__(self, forced_price: float) -> None:
        super().__init__()
        self._forced_price = forced_price

    def on_bar(self, bar: Bar, bar_index: int) -> list[Fill]:
        fills = super().on_bar(bar, bar_index)
        return [f.model_copy(update={"price": self._forced_price}) if f.leg == "entry" else f for f in fills]


class FakeStrategy:
    """Emits a scripted `Signal` on chosen bar indices; `None` on every other bar."""

    name = "fake"

    def __init__(self, signals: dict[int, Signal], on_call: Callable[[], None] | None = None) -> None:
        self._signals = signals
        self._on_call = on_call
        self.calls: list[int] = []

    def on_bar(self, ctx: Context) -> Signal | None:
        self.calls.append(ctx.bar_index)
        if self._on_call is not None:
            self._on_call()
        return self._signals.get(ctx.bar_index)


def _exit_order(
    trade: Trade, *, leg: str, price: float, qty: float, kind: str = "limit", order_id: str | None = None
) -> Order:
    return Order(
        id=order_id or f"{trade.id}:{leg}",
        instrument=trade.instrument,
        direction=-trade.direction,
        qty=qty,
        kind=kind,
        price=price,
        expires_at_bar=None,
        leg=leg,
        trade_id=trade.id,
    )


class FakeExitRule:
    """One stop plus one target, at fixed R multiples of the initial stop distance."""

    name = "fake_exit"

    def __init__(self, target_r: float = 2.0) -> None:
        self._target_r = target_r

    def initial_stop(self, signal: Signal, ctx: Context) -> float:
        return signal.stop

    def attach(self, trade: Trade, ctx: Context) -> list[Order]:
        distance = abs(trade.entry_fill.price - trade.stop)
        target_price = trade.entry_fill.price + trade.direction * self._target_r * distance
        qty = trade.entry_fill.qty
        return [
            _exit_order(trade, leg="stop", price=trade.stop, qty=qty, kind="stop"),
            _exit_order(trade, leg="target", price=target_price, qty=qty, kind="limit"),
        ]

    def on_bar(self, trade: Trade, ctx: Context) -> list[Order]:
        return []


class FakePartialExitRule:
    """Scales out half at `partial_r`, runs the rest to `runner_r`, breakeven after."""

    name = "fake_partial_exit"

    def __init__(self, partial_r: float = 1.0, runner_r: float = 3.0, scale_pct: float = 0.5) -> None:
        self._partial_r = partial_r
        self._runner_r = runner_r
        self._scale_pct = scale_pct

    def initial_stop(self, signal: Signal, ctx: Context) -> float:
        return signal.stop

    def attach(self, trade: Trade, ctx: Context) -> list[Order]:
        distance = abs(trade.entry_fill.price - trade.stop)
        partial_price = trade.entry_fill.price + trade.direction * self._partial_r * distance
        runner_price = trade.entry_fill.price + trade.direction * self._runner_r * distance
        total_qty = trade.entry_fill.qty
        partial_qty = total_qty * self._scale_pct
        runner_qty = total_qty - partial_qty
        return [
            _exit_order(trade, leg="stop", price=trade.stop, qty=total_qty, kind="stop"),
            _exit_order(trade, leg="partial", price=partial_price, qty=partial_qty, kind="limit"),
            _exit_order(trade, leg="target", price=runner_price, qty=runner_qty, kind="limit"),
        ]

    def on_bar(self, trade: Trade, ctx: Context) -> list[Order]:
        already_scaled = any(leg.leg == "partial" for leg in trade.legs)
        at_breakeven = trade.stop == trade.entry_fill.price
        if already_scaled and not at_breakeven:
            remaining_qty = trade.entry_fill.qty - sum(leg.qty for leg in trade.legs)
            return [
                _exit_order(
                    trade,
                    leg="stop",
                    price=trade.entry_fill.price,
                    qty=remaining_qty,
                    kind="stop",
                    order_id=f"{trade.id}:stop:breakeven",
                )
            ]
        return []


class FakeTrailingStopExitRule:
    """One stop plus a far-away target; replaces the stop once on the second `on_bar` call."""

    name = "fake_trailing_exit"

    def __init__(self, new_stop: float) -> None:
        self._new_stop = new_stop
        self._trailed = False

    def initial_stop(self, signal: Signal, ctx: Context) -> float:
        return signal.stop

    def attach(self, trade: Trade, ctx: Context) -> list[Order]:
        distance = abs(trade.entry_fill.price - trade.stop)
        target_price = trade.entry_fill.price + trade.direction * 10.0 * distance
        qty = trade.entry_fill.qty
        return [
            _exit_order(trade, leg="stop", price=trade.stop, qty=qty, kind="stop"),
            _exit_order(trade, leg="target", price=target_price, qty=qty, kind="limit"),
        ]

    def on_bar(self, trade: Trade, ctx: Context) -> list[Order]:
        if not self._trailed:
            self._trailed = True
            return [
                _exit_order(
                    trade,
                    leg="stop",
                    price=self._new_stop,
                    qty=trade.entry_fill.qty,
                    kind="stop",
                    order_id=f"{trade.id}:stop:trail",
                )
            ]
        return []


class _MismatchedTradeIdExitRule:
    """Attaches a stop leg tagged with someone else's `trade_id` (exercises M12)."""

    name = "bad_exit"

    def initial_stop(self, signal: Signal, ctx: Context) -> float:
        return signal.stop

    def attach(self, trade: Trade, ctx: Context) -> list[Order]:
        order = _exit_order(trade, leg="stop", price=trade.stop, qty=trade.entry_fill.qty, kind="stop")
        return [order.model_copy(update={"trade_id": "someone-else"})]

    def on_bar(self, trade: Trade, ctx: Context) -> list[Order]:
        return []


def _signal(
    *, entry: float = 100.0, stop: float = 95.0, direction: int = 1, expires_in_bars: int = 5
) -> Signal:
    return Signal(
        direction=direction,
        entry=entry,
        stop=stop,
        structure_target=None,
        tag="fake",
        expires_in_bars=expires_in_bars,
    )


def _make_engine(
    *,
    strategy: FakeStrategy,
    exit_rule: Any,
    broker: FakeBroker,
    equity: float = 10_000.0,
    settings_reader: Any | None = None,
    session_allowed: Callable[[Bar], bool] | None = None,
    regime_tagger: Callable[[Context], str] | None = None,
) -> tuple[Engine, Portfolio]:
    portfolio = Portfolio(initial_equity=equity)
    reader = settings_reader if settings_reader is not None else StaticSettingsReader(Settings())
    engine = Engine(
        BTC,
        strategy,
        exit_rule,
        broker,
        portfolio,
        reader,
        session_allowed=session_allowed,
        regime_tagger=regime_tagger,
    )
    return engine, portfolio


# --- entry -> attach -> target hit -------------------------------------------------


def test_target_hit_realized_r_plus_two_and_equity_updated() -> None:
    strategy = FakeStrategy({1: _signal(entry=100.0, stop=95.0)})
    broker = FakeBroker()
    engine, portfolio = _make_engine(strategy=strategy, exit_rule=FakeExitRule(target_r=2.0), broker=broker)

    bars = [
        make_bar(0, 99.0, 100.0, 98.0, 99.5),  # bar_index 0: strategy called, no signal
        make_bar(1, 99.0, 100.0, 98.0, 99.5),  # bar_index 1: signal emitted, entry submitted
        make_bar(2, 101.0, 102.0, 99.0, 100.5),  # bar_index 2: entry fills at 100
        make_bar(3, 105.0, 111.0, 104.0, 110.0),  # bar_index 3: target (110) touched, stop (95) not
    ]
    closed: list[Trade] = []
    for bar in bars:
        closed.extend(engine.step(bar))

    assert len(closed) == 1
    trade = closed[0]
    assert trade.realized_r == pytest.approx(2.0)
    assert trade.risk_r == pytest.approx(100.0)
    assert len(trade.legs) == 1
    assert trade.legs[0].leg == "target"
    assert portfolio.equity == pytest.approx(10_200.0)
    assert engine.open_trade is None
    assert engine.trades == [trade]
    # The stop leg was still pending when the target closed the trade; it must be cancelled.
    assert broker.cancelled_log == [f"{trade.id}:stop"]


def test_stop_hit_realized_r_minus_one() -> None:
    strategy = FakeStrategy({0: _signal(entry=100.0, stop=95.0)})
    broker = FakeBroker()
    engine, portfolio = _make_engine(strategy=strategy, exit_rule=FakeExitRule(target_r=2.0), broker=broker)

    bars = [
        make_bar(0, 99.0, 100.0, 98.0, 99.5),  # signal emitted
        make_bar(1, 101.0, 102.0, 99.0, 100.5),  # entry fills at 100
        make_bar(2, 97.0, 98.0, 94.0, 96.0),  # stop (95) touched, target (110) not
    ]
    closed: list[Trade] = []
    for bar in bars:
        closed.extend(engine.step(bar))

    assert len(closed) == 1
    trade = closed[0]
    assert trade.realized_r == pytest.approx(-1.0)
    assert portfolio.equity == pytest.approx(9_900.0)


# --- kill switch --------------------------------------------------------------------


def test_kill_switch_blocks_new_entries_but_still_manages_open_trade() -> None:
    reader = InMemorySettingsReader(Settings(kill_switch=False))
    strategy = FakeStrategy({0: _signal(entry=100.0, stop=95.0), 3: _signal(entry=200.0, stop=190.0)})
    broker = FakeBroker()
    engine, portfolio = _make_engine(
        strategy=strategy, exit_rule=FakeExitRule(target_r=2.0), broker=broker, settings_reader=reader
    )

    engine.step(make_bar(0, 99.0, 100.0, 98.0, 99.5))  # signal emitted
    engine.step(make_bar(1, 101.0, 102.0, 99.0, 100.5))  # entry fills at 100
    assert engine.open_trade is not None

    reader.set(Settings(kill_switch=True))  # takes effect starting the next bar

    engine.step(make_bar(2, 100.0, 101.0, 99.0, 100.5))  # kill switch now active; trade still open
    assert engine.open_trade is not None  # stop/target not yet touched

    calls_before = len(strategy.calls)
    closed = engine.step(make_bar(3, 97.0, 98.0, 94.0, 96.0))  # stop touched -> trade closes
    assert len(closed) == 1
    assert closed[0].realized_r == pytest.approx(-1.0)
    # Bar 3 also scripts a fresh signal, but the kill switch must suppress the strategy call.
    assert len(strategy.calls) == calls_before
    assert engine.open_trade is None
    assert portfolio.trades == [closed[0]]


# --- settings applied at the next bar -------------------------------------------------


def test_settings_change_mid_bar_applies_starting_next_bar() -> None:
    reader = InMemorySettingsReader(Settings(risk_pct=0.01))

    def flip() -> None:
        reader.set(Settings(risk_pct=0.02))

    # An unreachable entry price plus a same-bar expiry frees the pending-entry slot on
    # the very next bar, without ever actually filling.
    strategy = FakeStrategy(
        {
            0: _signal(entry=1_000_000.0, stop=999_000.0, expires_in_bars=0),
            1: _signal(entry=1_000_000.0, stop=999_000.0, expires_in_bars=0),
        },
        on_call=flip,
    )
    broker = FakeBroker()
    engine, _ = _make_engine(
        strategy=strategy, exit_rule=FakeExitRule(), broker=broker, settings_reader=reader
    )

    engine.step(make_bar(0, 1.0, 2.0, 0.5, 1.5))
    first_entry = broker.submitted_log[-1]
    assert first_entry.qty == pytest.approx(100.0 / 1000.0)  # risk 1% of 10_000 / distance 1000

    engine.step(make_bar(1, 1.0, 2.0, 0.5, 1.5))
    second_entry = broker.submitted_log[-1]
    assert second_entry.qty == pytest.approx(200.0 / 1000.0)  # risk 2% of 10_000 / distance 1000


# --- MAE / MFE ------------------------------------------------------------------------


def test_mae_mfe_hand_path() -> None:
    strategy = FakeStrategy({0: _signal(entry=100.0, stop=95.0)})
    broker = FakeBroker()
    engine, _ = _make_engine(strategy=strategy, exit_rule=FakeExitRule(target_r=2.0), broker=broker)

    engine.step(make_bar(0, 99.0, 100.0, 98.0, 99.5))  # signal emitted
    engine.step(make_bar(1, 101.0, 102.0, 99.0, 100.5))  # fills at 100; bar itself: low 99, high 102
    trade = engine.open_trade
    assert trade is not None
    assert trade.mae_r == pytest.approx(1.0 / 5.0)  # entry(100) - low(99) = 1
    assert trade.mfe_r == pytest.approx(2.0 / 5.0)  # high(102) - entry(100) = 2

    engine.step(make_bar(2, 98.0, 101.0, 97.0, 100.0))  # low 97 -> deeper adverse; high 101 -> smaller
    trade = engine.open_trade
    assert trade is not None
    assert trade.mae_r == pytest.approx(3.0 / 5.0)  # entry(100) - low(97) = 3
    assert trade.mfe_r == pytest.approx(2.0 / 5.0)  # unchanged: 101 - 100 = 1 < previous 2

    engine.step(make_bar(3, 101.0, 108.0, 100.0, 107.0))  # high 108 -> new best; low 100 -> no new worst
    trade = engine.open_trade
    assert trade is not None
    assert trade.mae_r == pytest.approx(3.0 / 5.0)  # unchanged
    assert trade.mfe_r == pytest.approx(8.0 / 5.0)  # high(108) - entry(100) = 8


# --- partial + runner -----------------------------------------------------------------


def test_partial_then_runner_closes_with_two_legs_and_right_realized_r() -> None:
    strategy = FakeStrategy({0: _signal(entry=100.0, stop=95.0)})
    broker = FakeBroker()
    engine, portfolio = _make_engine(
        strategy=strategy,
        exit_rule=FakePartialExitRule(partial_r=1.0, runner_r=3.0, scale_pct=0.5),
        broker=broker,
    )

    engine.step(make_bar(0, 99.0, 100.0, 98.0, 99.5))  # signal emitted
    engine.step(make_bar(1, 99.0, 101.0, 98.0, 100.5))  # entry fills at 100
    assert engine.open_trade is not None
    assert engine.open_trade.stop == pytest.approx(95.0)

    engine.step(make_bar(2, 104.0, 106.0, 103.0, 105.5))  # partial (105) touched only
    trade = engine.open_trade
    assert trade is not None
    assert len(trade.legs) == 1
    assert trade.legs[0].leg == "partial"
    assert trade.stop == pytest.approx(100.0)  # breakeven applied after the scale-out

    closed = engine.step(make_bar(3, 113.0, 116.0, 112.0, 115.5))  # runner target (115) touched
    assert len(closed) == 1
    trade = closed[0]
    assert len(trade.legs) == 2
    assert [leg.leg for leg in trade.legs] == ["partial", "target"]
    assert trade.realized_r == pytest.approx(2.0)
    assert portfolio.equity == pytest.approx(10_200.0)
    # The replaced breakeven stop was still pending when the runner target closed the
    # trade; it must be cancelled (the original stop id was already replaced, not cancelled).
    assert broker.cancelled_log == [f"{trade.id}:stop:breakeven"]


# --- short-direction MAE/MFE and subbars -------------------------------------------------


def test_mae_mfe_short_direction() -> None:
    strategy = FakeStrategy({0: _signal(entry=100.0, stop=105.0, direction=-1)})
    broker = FakeBroker()
    engine, _ = _make_engine(strategy=strategy, exit_rule=FakeExitRule(target_r=2.0), broker=broker)

    engine.step(make_bar(0, 101.0, 102.0, 100.0, 101.5))  # signal emitted (short: stop above entry)
    engine.step(make_bar(1, 99.0, 101.0, 98.0, 100.0))  # entry (100) touched; bar low 98, high 101

    trade = engine.open_trade
    assert trade is not None
    assert trade.mae_r == pytest.approx(0.2)  # adverse: high(101) - entry(100) = 1; /5
    assert trade.mfe_r == pytest.approx(0.4)  # favourable: entry(100) - low(98) = 2; /5


def test_four_hour_bar_pushes_its_subbars_before_itself() -> None:
    strategy = FakeStrategy({})
    broker = FakeBroker()
    engine, _ = _make_engine(strategy=strategy, exit_rule=FakeExitRule(), broker=broker)

    sub1 = Bar(
        instrument=BTC, tf="1h", ts_open=START, open=100.0, high=100.5, low=99.5, close=100.2, volume=1.0
    )
    sub2 = Bar(
        instrument=BTC,
        tf="1h",
        ts_open=START + timedelta(hours=1),
        open=100.2,
        high=100.8,
        low=99.8,
        close=100.5,
        volume=1.0,
    )
    parent = Bar(
        instrument=BTC,
        tf="4h",
        ts_open=START,
        open=100.0,
        high=101.0,
        low=99.0,
        close=100.5,
        volume=4.0,
        subbars=(sub1, sub2),
    )

    engine.step(parent)

    assert engine.ctx.history("1h") == (sub1, sub2)
    assert engine.ctx.bar_index == 0


def test_zero_equity_sizes_to_zero_and_submits_no_entry() -> None:
    strategy = FakeStrategy({0: _signal(entry=100.0, stop=95.0)})
    broker = FakeBroker()
    engine, portfolio = _make_engine(strategy=strategy, exit_rule=FakeExitRule(), broker=broker, equity=0.0)

    engine.step(make_bar(0, 99.0, 100.0, 98.0, 99.5))

    assert strategy.calls == [0]
    assert broker.submitted_log == []
    assert engine.open_trade is None
    assert portfolio.equity == 0.0


# --- entry expiry -----------------------------------------------------------------------


def test_pending_entry_expires_unfilled() -> None:
    strategy = FakeStrategy({0: _signal(entry=100.0, stop=95.0, expires_in_bars=1)})
    broker = FakeBroker()
    engine, portfolio = _make_engine(strategy=strategy, exit_rule=FakeExitRule(), broker=broker)

    engine.step(make_bar(0, 200.0, 201.0, 199.0, 200.5))  # signal emitted; entry (100) not touched
    engine.step(make_bar(1, 200.0, 201.0, 199.0, 200.5))  # still live on its expiry bar; not touched
    engine.step(make_bar(2, 200.0, 201.0, 199.0, 200.5))  # bar_index 2 > expires_at_bar 1: dropped

    assert engine.open_trade is None
    assert engine.trades == []
    assert portfolio.trades == []
    assert len(broker.submitted_log) == 1  # never resubmitted


# --- 1d bars are pushed and ignored for trading ------------------------------------------


def test_daily_bars_are_pushed_but_never_traded() -> None:
    strategy = FakeStrategy({})
    broker = FakeBroker()
    engine, _ = _make_engine(strategy=strategy, exit_rule=FakeExitRule(), broker=broker)

    daily_bar = Bar(
        instrument=BTC, tf="1d", ts_open=START, open=100.0, high=101.0, low=99.0, close=100.5, volume=1.0
    )
    closed = engine.step(daily_bar)

    assert closed == []
    assert strategy.calls == []
    assert engine.ctx.bar_index == -1
    assert engine.ctx.history("1d") == (daily_bar,)


# --- session filter --------------------------------------------------------------------


def test_session_filter_suppresses_entries() -> None:
    strategy = FakeStrategy({0: _signal(entry=100.0, stop=95.0)})
    broker = FakeBroker()
    engine, _ = _make_engine(
        strategy=strategy, exit_rule=FakeExitRule(), broker=broker, session_allowed=lambda bar: False
    )

    engine.step(make_bar(0, 99.0, 100.0, 98.0, 99.5))

    assert strategy.calls == []
    assert broker.submitted_log == []
    assert engine.open_trade is None


# --- regime tag and context snapshot ----------------------------------------------------


def test_regime_tagger_and_context_snapshot_land_on_the_trade() -> None:
    strategy = FakeStrategy({0: _signal(entry=100.0, stop=95.0)})
    broker = FakeBroker()
    engine, _ = _make_engine(
        strategy=strategy,
        exit_rule=FakeExitRule(),
        broker=broker,
        regime_tagger=lambda ctx: "trend",
    )

    engine.step(make_bar(0, 99.0, 100.0, 98.0, 99.5))
    engine.step(make_bar(1, 101.0, 102.0, 99.0, 100.5))  # entry fills

    trade = engine.open_trade
    assert trade is not None
    assert trade.regime == "trend"
    assert trade.context_snapshot != b""


# --- run() ------------------------------------------------------------------------------


def test_run_feeds_every_bar_and_returns_all_closed_trades() -> None:
    strategy = FakeStrategy({0: _signal(entry=100.0, stop=95.0)})
    broker = FakeBroker()
    engine, portfolio = _make_engine(strategy=strategy, exit_rule=FakeExitRule(target_r=2.0), broker=broker)

    bars = [
        make_bar(0, 99.0, 100.0, 98.0, 99.5),
        make_bar(1, 101.0, 102.0, 99.0, 100.5),
        make_bar(2, 105.0, 111.0, 104.0, 110.0),
    ]
    closed = engine.run(bars)

    assert len(closed) == 1
    assert closed[0].realized_r == pytest.approx(2.0)
    assert portfolio.equity == pytest.approx(10_200.0)


# --- F1: excursion must include every live bar, entry and closing bars included ---------


def test_same_bar_entry_and_stop_updates_excursion_and_closes_at_minus_one_r() -> None:
    strategy = FakeStrategy({0: _signal(entry=100.0, stop=95.0)})
    broker = FakeBroker()
    engine, _ = _make_engine(strategy=strategy, exit_rule=FakeExitRule(target_r=2.0), broker=broker)

    engine.step(make_bar(0, 99.0, 100.0, 98.0, 99.5))  # signal emitted
    # bar 1: entry (100) and stop (95) both fall inside [90, 101] -> fills entry, then,
    # same bar, the stop; without F1 this bar's own range is never folded into MAE/MFE.
    closed = engine.step(make_bar(1, 100.0, 101.0, 90.0, 91.0))

    assert len(closed) == 1
    trade = closed[0]
    assert trade.realized_r == pytest.approx(-1.0)
    assert trade.mae_r >= 1.0  # entry(100) - low(90) = 10; / distance(5) = 2.0
    assert trade.mfe_r == pytest.approx(1.0 / 5.0)  # high(101) - entry(100) = 1


def test_target_hit_with_no_prior_favourable_move_updates_mfe_on_closing_bar() -> None:
    strategy = FakeStrategy({0: _signal(entry=100.0, stop=95.0)})
    broker = FakeBroker()
    engine, _ = _make_engine(strategy=strategy, exit_rule=FakeExitRule(target_r=2.0), broker=broker)

    engine.step(make_bar(0, 99.0, 100.0, 98.0, 99.5))  # signal emitted
    engine.step(make_bar(1, 100.0, 100.0, 99.0, 99.5))  # entry fills at 100; no favourable move yet
    trade = engine.open_trade
    assert trade is not None
    assert trade.mfe_r == pytest.approx(0.0)

    # Target (110) is touched directly on this bar, with no intervening favourable bar.
    # Without F1, the closing bar's own high is never folded in and mfe_r would stay 0 even
    # though the trade closed at +2R (captured/mfe would then be undefined, never <= 1).
    closed = engine.step(make_bar(2, 100.0, 110.0, 99.0, 109.0))
    assert len(closed) == 1
    trade = closed[0]
    assert trade.realized_r == pytest.approx(2.0)
    assert trade.mfe_r >= 2.0


# --- F2: an exit fill for another trade must be rejected --------------------------------


def test_exit_fill_for_another_trade_raises_engine_error() -> None:
    strategy = FakeStrategy({0: _signal(entry=100.0, stop=95.0)})
    alien_fill = Fill(
        order_id="ghost:stop",
        ts=START,
        price=95.0,
        qty=20.0,
        cost=CostBreakdown(),
        leg="stop",
        trade_id="ghost-trade",
    )
    broker = _InjectedFillBroker(alien_fill, on_bar_index=1)
    engine, _ = _make_engine(strategy=strategy, exit_rule=FakeExitRule(), broker=broker)

    engine.step(make_bar(0, 99.0, 100.0, 98.0, 99.5))  # signal emitted
    with pytest.raises(EngineError, match="ghost-trade"):
        engine.step(make_bar(1, 101.0, 102.0, 99.0, 100.5))  # entry fills; alien fill injected too


# --- F3: impossible exit quantities must be rejected --------------------------------------


def test_oversized_exit_fill_raises_engine_error() -> None:
    strategy = FakeStrategy({0: _signal(entry=100.0, stop=95.0)})
    # The entry order (and its trade_id) is assigned when it's submitted, at bar_index 0 —
    # the fill on bar 1 doesn't change it.
    trade_id = f"{BTC.venue}:{BTC.symbol}:0"
    bad_fill = Fill(
        order_id=f"{trade_id}:stop",
        ts=START,
        price=95.0,
        qty=999.0,
        cost=CostBreakdown(),
        leg="stop",
        trade_id=trade_id,
    )
    broker = _InjectedFillBroker(bad_fill, on_bar_index=2)
    engine, _ = _make_engine(strategy=strategy, exit_rule=FakeExitRule(), broker=broker)

    engine.step(make_bar(0, 99.0, 100.0, 98.0, 99.5))  # signal emitted
    engine.step(make_bar(1, 101.0, 102.0, 99.0, 100.5))  # entry fills; stop/target don't touch
    assert engine.open_trade is not None

    with pytest.raises(EngineError, match="qty"):
        engine.step(make_bar(2, 100.0, 101.0, 99.5, 100.5))  # injected oversized exit fill


def test_zero_qty_exit_fill_raises_engine_error() -> None:
    strategy = FakeStrategy({0: _signal(entry=100.0, stop=95.0)})
    trade_id = f"{BTC.venue}:{BTC.symbol}:0"
    bad_fill = Fill(
        order_id=f"{trade_id}:stop",
        ts=START,
        price=95.0,
        qty=0.0,
        cost=CostBreakdown(),
        leg="stop",
        trade_id=trade_id,
    )
    broker = _InjectedFillBroker(bad_fill, on_bar_index=2)
    engine, _ = _make_engine(strategy=strategy, exit_rule=FakeExitRule(), broker=broker)

    engine.step(make_bar(0, 99.0, 100.0, 98.0, 99.5))  # signal emitted
    engine.step(make_bar(1, 101.0, 102.0, 99.0, 100.5))  # entry fills; stop/target don't touch
    assert engine.open_trade is not None

    with pytest.raises(EngineError, match="qty"):
        engine.step(make_bar(2, 100.0, 101.0, 99.5, 100.5))  # injected zero-qty exit fill


# --- F4: ctx.position stays in sync with the open trade -----------------------------------


def test_stop_replacement_keeps_ctx_position_in_sync() -> None:
    strategy = FakeStrategy({0: _signal(entry=100.0, stop=95.0)})
    broker = FakeBroker()
    exit_rule = FakeTrailingStopExitRule(new_stop=98.0)
    engine, _ = _make_engine(strategy=strategy, exit_rule=exit_rule, broker=broker)

    engine.step(make_bar(0, 99.0, 100.0, 98.0, 99.5))  # signal emitted
    engine.step(make_bar(1, 101.0, 102.0, 99.0, 100.5))  # entry fills at 100
    assert engine.open_trade is not None

    engine.step(make_bar(2, 100.0, 101.0, 99.0, 100.5))  # exit_rule.on_bar replaces the stop
    trade = engine.open_trade
    assert trade is not None
    assert trade.stop == pytest.approx(98.0)
    assert engine.ctx.position is not None
    assert engine.ctx.position.stop == pytest.approx(trade.stop)


# --- F5: kill switch cancels a resting, unfilled entry -------------------------------------


def test_kill_switch_cancels_resting_pending_entry() -> None:
    reader = InMemorySettingsReader(Settings(kill_switch=False))
    strategy = FakeStrategy({0: _signal(entry=100.0, stop=95.0, expires_in_bars=5)})
    broker = FakeBroker()
    engine, _ = _make_engine(
        strategy=strategy, exit_rule=FakeExitRule(), broker=broker, settings_reader=reader
    )

    engine.step(make_bar(0, 200.0, 201.0, 199.0, 200.5))  # signal emitted; entry (100) unreachable
    entry_order = broker.submitted_log[-1]
    assert entry_order.leg == "entry"

    reader.set(Settings(kill_switch=True))  # takes effect starting the next bar

    # Entry (100) is inside this bar's range, so if the kill switch didn't cancel first, the
    # broker would fill it.
    engine.step(make_bar(1, 200.0, 201.0, 100.0, 200.5))

    assert engine.open_trade is None
    assert broker.cancelled_log == [entry_order.id]


# --- F6(b): entry order price is rounded to the instrument's tick -------------------------


def test_entry_order_price_rounds_to_tick_long() -> None:
    strategy = FakeStrategy({0: _signal(entry=100.3, stop=95.0)})
    broker = FakeBroker()
    engine, _ = _make_engine(strategy=strategy, exit_rule=FakeExitRule(), broker=broker)

    engine.step(make_bar(0, 99.0, 100.0, 98.0, 99.5))

    entry_order = broker.submitted_log[-1]
    assert entry_order.price == pytest.approx(100.5)


def test_entry_order_price_rounds_to_tick_short() -> None:
    strategy = FakeStrategy({0: _signal(entry=100.2, stop=105.0, direction=-1)})
    broker = FakeBroker()
    engine, _ = _make_engine(strategy=strategy, exit_rule=FakeExitRule(), broker=broker)

    engine.step(make_bar(0, 99.0, 100.0, 98.0, 99.5))

    entry_order = broker.submitted_log[-1]
    assert entry_order.price == pytest.approx(100.0)
    assert entry_order.direction == -1


# --- F6(c): entry order fields match the contract ------------------------------------------


def test_entry_order_fields_match_contract() -> None:
    strategy = FakeStrategy({3: _signal(entry=100.0, stop=95.0, expires_in_bars=4)})
    broker = FakeBroker()
    engine, _ = _make_engine(strategy=strategy, exit_rule=FakeExitRule(), broker=broker)

    for i in range(4):
        engine.step(make_bar(i, 200.0, 201.0, 199.0, 200.5))  # entry (100) unreachable

    entry_order = broker.submitted_log[-1]
    expected_trade_id = f"{BTC.venue}:{BTC.symbol}:3"
    assert entry_order.trade_id == expected_trade_id
    assert entry_order.id == f"{expected_trade_id}:entry"
    assert entry_order.kind == "limit"
    assert entry_order.leg == "entry"
    assert entry_order.expires_at_bar == 3 + 4


# --- F7: bare asserts on broker-supplied fills become explicit EngineErrors ---------------


def test_entry_fill_with_no_pending_entry_raises_engine_error() -> None:
    strategy = FakeStrategy({0: _signal(entry=100.0, stop=95.0)})
    trade_id = f"{BTC.venue}:{BTC.symbol}:1"
    ghost_fill = Fill(
        order_id="ghost:entry",
        ts=START,
        price=100.0,
        qty=1.0,
        cost=CostBreakdown(),
        leg="entry",
        trade_id=trade_id,
    )
    broker = _InjectedFillBroker(ghost_fill, on_bar_index=2)
    engine, _ = _make_engine(strategy=strategy, exit_rule=FakeExitRule(), broker=broker)

    engine.step(make_bar(0, 99.0, 100.0, 98.0, 99.5))  # signal emitted
    engine.step(make_bar(1, 101.0, 102.0, 99.0, 100.5))  # entry fills normally
    assert engine.open_trade is not None

    with pytest.raises(EngineError, match="pending entry"):
        engine.step(make_bar(2, 100.0, 101.0, 99.5, 100.5))  # ghost entry fill: no pending entry


def test_exit_fill_with_no_open_trade_raises_engine_error() -> None:
    strategy = FakeStrategy({})
    ghost_fill = Fill(
        order_id="ghost:stop",
        ts=START,
        price=95.0,
        qty=1.0,
        cost=CostBreakdown(),
        leg="stop",
        trade_id="hyperliquid:BTC:0",
    )
    broker = _InjectedFillBroker(ghost_fill, on_bar_index=0)
    engine, _ = _make_engine(strategy=strategy, exit_rule=FakeExitRule(), broker=broker)

    with pytest.raises(EngineError, match="no open trade"):
        engine.step(make_bar(0, 99.0, 100.0, 98.0, 99.5))


# --- M10: a fill landing exactly on the stop falls back to the planned risk ---------------


def test_entry_fill_at_stop_price_falls_back_to_planned_risk() -> None:
    strategy = FakeStrategy({0: _signal(entry=100.0, stop=95.0)})
    broker = _StopPriceEntryBroker(forced_price=95.0)  # the fill lands exactly on the stop
    engine, _ = _make_engine(strategy=strategy, exit_rule=FakeExitRule(), broker=broker)

    engine.step(make_bar(0, 99.0, 100.0, 98.0, 99.5))  # signal emitted
    engine.step(make_bar(1, 101.0, 102.0, 99.0, 100.5))  # entry order touches; fill forced to 95.0

    trade = engine.open_trade
    assert trade is not None
    assert trade.risk_r == pytest.approx(100.0)  # planned: |100 - 95| * qty(20) * mult(1)


# --- M12: an exit-rule order tagged for another trade must be rejected --------------------


def test_submit_legs_rejects_order_for_a_different_trade() -> None:
    strategy = FakeStrategy({0: _signal(entry=100.0, stop=95.0)})
    broker = FakeBroker()
    engine, _ = _make_engine(strategy=strategy, exit_rule=_MismatchedTradeIdExitRule(), broker=broker)

    engine.step(make_bar(0, 99.0, 100.0, 98.0, 99.5))  # signal emitted
    with pytest.raises(EngineError, match="trade_id"):
        engine.step(make_bar(1, 101.0, 102.0, 99.0, 100.5))  # entry fills; attach() returns a bad leg
