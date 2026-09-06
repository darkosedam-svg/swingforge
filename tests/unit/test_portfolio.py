"""Tests for `Portfolio`: sizing, risk money, realized pnl, and the equity ledger."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from swingforge.core.portfolio import QTY_EPS, QTY_STEP, Portfolio
from swingforge.core.types import CostBreakdown, Fill, Instrument, Trade

BTC = Instrument(
    venue="hyperliquid",
    symbol="BTC",
    tick_size=Decimal("0.5"),
    contract_multiplier=Decimal("1"),
    quote_ccy="USD",
    session_profile="perp",
)
BTC_10X = BTC.model_copy(update={"contract_multiplier": Decimal("10")})
TS = datetime(2026, 1, 2, 4, 0, tzinfo=UTC)


# --- size ---------------------------------------------------------------


def test_size_hand_numbers_multiplier_one() -> None:
    portfolio = Portfolio(initial_equity=10_000)
    qty = portfolio.size(entry=100.0, stop=95.0, instrument=BTC, risk_pct=0.01)
    assert qty == pytest.approx(20.0)


def test_size_hand_numbers_multiplier_ten() -> None:
    portfolio = Portfolio(initial_equity=10_000)
    qty = portfolio.size(entry=100.0, stop=95.0, instrument=BTC_10X, risk_pct=0.01)
    assert qty == pytest.approx(2.0)


def test_size_zero_stop_distance_returns_zero() -> None:
    portfolio = Portfolio(initial_equity=10_000)
    qty = portfolio.size(entry=100.0, stop=100.0, instrument=BTC, risk_pct=0.01)
    assert qty == 0.0


def test_size_floors_to_qty_step() -> None:
    portfolio = Portfolio(initial_equity=10_000)
    # risk = 100, distance = 3 -> raw qty = 33.3333...; floored to a multiple of QTY_STEP.
    qty = portfolio.size(entry=100.0, stop=97.0, instrument=BTC, risk_pct=0.01)
    assert qty <= 100.0 / 3.0
    assert qty == pytest.approx(33.333333, abs=1e-6)
    steps = qty / QTY_STEP
    assert steps == pytest.approx(round(steps), abs=1e-6)


def test_size_uses_portfolios_own_equity_not_a_parameter() -> None:
    """M11: `size` takes no `equity` argument; it must size against `self.equity`."""
    portfolio = Portfolio(initial_equity=10_000)
    portfolio.equity = 20_000
    qty = portfolio.size(entry=100.0, stop=95.0, instrument=BTC, risk_pct=0.01)
    assert qty == pytest.approx(40.0)  # risk = 20_000 * 0.01 = 200; / distance 5 = 40


def test_qty_eps_is_half_a_qty_step() -> None:
    assert QTY_EPS == QTY_STEP / 2


# --- risk_money -----------------------------------------------------------


def test_risk_money_matches_manual_calculation() -> None:
    portfolio = Portfolio(initial_equity=10_000)
    risk = portfolio.risk_money(entry=100.0, stop=95.0, qty=20.0, instrument=BTC)
    assert risk == pytest.approx(100.0)


def test_risk_money_scales_with_contract_multiplier() -> None:
    portfolio = Portfolio(initial_equity=10_000)
    risk = portfolio.risk_money(entry=100.0, stop=95.0, qty=2.0, instrument=BTC_10X)
    assert risk == pytest.approx(100.0)


# --- realized_pnl -----------------------------------------------------------


def test_realized_pnl_long_win_zero_costs() -> None:
    portfolio = Portfolio(initial_equity=10_000)
    entry_fill = Fill(order_id="e", ts=TS, price=100.0, qty=20.0, cost=CostBreakdown(), leg="entry")
    exit_fill = Fill(order_id="x", ts=TS, price=110.0, qty=20.0, cost=CostBreakdown(), leg="target")
    pnl = portfolio.realized_pnl(entry_fill, (exit_fill,), direction=1, instrument=BTC)
    assert pnl == pytest.approx(200.0)


def test_realized_pnl_short_loss_with_costs() -> None:
    portfolio = Portfolio(initial_equity=10_000)
    entry_fill = Fill(
        order_id="e", ts=TS, price=100.0, qty=10.0, cost=CostBreakdown(commission=1.0), leg="entry"
    )
    exit_fill = Fill(
        order_id="x", ts=TS, price=105.0, qty=10.0, cost=CostBreakdown(commission=1.0), leg="stop"
    )
    pnl = portfolio.realized_pnl(entry_fill, (exit_fill,), direction=-1, instrument=BTC)
    # short: -1 * (105 - 100) * 10 * 1 = -50; minus costs (1 + 1) = -52
    assert pnl == pytest.approx(-52.0)


def test_realized_pnl_sums_multiple_legs() -> None:
    portfolio = Portfolio(initial_equity=10_000)
    entry_fill = Fill(order_id="e", ts=TS, price=100.0, qty=20.0, cost=CostBreakdown(), leg="entry")
    partial = Fill(order_id="p", ts=TS, price=105.0, qty=10.0, cost=CostBreakdown(), leg="partial")
    runner = Fill(order_id="r", ts=TS, price=115.0, qty=10.0, cost=CostBreakdown(), leg="target")
    pnl = portfolio.realized_pnl(entry_fill, (partial, runner), direction=1, instrument=BTC)
    # (105 - 100) * 10 + (115 - 100) * 10 = 50 + 150 = 200
    assert pnl == pytest.approx(200.0)


# --- record ----------------------------------------------------------------


def _closed_trade(*, realized_r: float, risk_r: float, ts: datetime) -> Trade:
    entry_fill = Fill(order_id="e", ts=TS, price=100.0, qty=1.0, cost=CostBreakdown(), leg="entry")
    exit_fill = Fill(order_id="x", ts=ts, price=100.0, qty=1.0, cost=CostBreakdown(), leg="stop")
    return Trade(
        id="t",
        instrument=BTC,
        direction=1,
        entry_fill=entry_fill,
        legs=(exit_fill,),
        stop=95.0,
        target=None,
        risk_r=risk_r,
        realized_r=realized_r,
        opened_bar=0,
        closed_bar=1,
    )


def test_record_starts_with_empty_curve() -> None:
    portfolio = Portfolio(initial_equity=100.0)
    assert portfolio.equity_curve == []
    assert portfolio.trades == []


def test_record_three_trade_sequence_hand_numbers() -> None:
    portfolio = Portfolio(initial_equity=100.0)
    ts1 = datetime(2026, 1, 1, tzinfo=UTC)
    ts2 = datetime(2026, 1, 2, tzinfo=UTC)
    ts3 = datetime(2026, 1, 3, tzinfo=UTC)

    # M8: `record` takes the exact pnl rather than re-deriving it from realized_r * risk_r;
    # pass pnl = realized_r * risk_r explicitly so the hand numbers stay the same.
    trade1 = _closed_trade(realized_r=2.0, risk_r=portfolio.equity * 0.01, ts=ts1)
    portfolio.record(trade1, trade1.realized_r * trade1.risk_r)
    assert portfolio.equity == pytest.approx(102.0, abs=1e-9)

    trade2 = _closed_trade(realized_r=-1.0, risk_r=portfolio.equity * 0.01, ts=ts2)
    portfolio.record(trade2, trade2.realized_r * trade2.risk_r)
    assert portfolio.equity == pytest.approx(100.98, abs=1e-9)

    trade3 = _closed_trade(realized_r=0.5, risk_r=portfolio.equity * 0.01, ts=ts3)
    portfolio.record(trade3, trade3.realized_r * trade3.risk_r)
    assert portfolio.equity == pytest.approx(101.4849, abs=1e-9)

    assert len(portfolio.trades) == 3
    equity_values = [equity for _, equity in portfolio.equity_curve]
    assert equity_values == pytest.approx([102.0, 100.98, 101.4849], abs=1e-9)
    assert [ts for ts, _ in portfolio.equity_curve] == [ts1, ts2, ts3]


def test_record_uses_exact_pnl_not_realized_r_times_risk_r() -> None:
    """M8: equity moves by the pnl argument even if it doesn't equal realized_r * risk_r."""
    portfolio = Portfolio(initial_equity=100.0)
    ts = datetime(2026, 1, 1, tzinfo=UTC)
    trade = _closed_trade(realized_r=2.0, risk_r=1.0, ts=ts)  # realized_r * risk_r == 2.0
    portfolio.record(trade, pnl=5.0)  # exact pnl differs from realized_r * risk_r
    assert portfolio.equity == pytest.approx(105.0)
