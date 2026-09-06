"""Tests for `swingforge.lab.excursion`."""

from __future__ import annotations

import math
import zlib
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from swingforge.adapters.replay import ReplaySource
from swingforge.core.types import Bar, CostBreakdown, Fill, Instrument, Signal, Trade
from swingforge.lab import excursion
from swingforge.lab.excursion import HoldBars, excursion_summary, mae_mfe
from swingforge.lab.tournament import Config, add_months, run_config, warmup_start
from tests.unit.synth_store import EntryFactoryStub, EveryN, build_store, session_factory

BTC = Instrument(
    venue="hyperliquid",
    symbol="BTC",
    tick_size=Decimal("0.5"),
    contract_multiplier=Decimal("1"),
    quote_ccy="USD",
    session_profile="perp",
)
BASE = datetime(2021, 1, 1, tzinfo=UTC)


def _bar(index: int, low: float, high: float) -> Bar:
    return Bar(
        instrument=BTC,
        tf="4h",
        ts_open=BASE + timedelta(hours=4 * index),
        open=(low + high) / 2,
        high=high,
        low=low,
        close=(low + high) / 2,
        volume=1.0,
    )


def _trade(
    *,
    realized_r: float | None = 1.0,
    mae_r: float = 0.0,
    mfe_r: float = 0.0,
    direction: int = 1,
    entry_price: float = 100.0,
    qty: float = 1.0,
    risk_r: float = 10.0,
    opened_bar: int = 0,
    closed_bar: int | None = 2,
    closed: bool = True,
) -> Trade:
    entry = Fill(order_id="e", ts=BASE, price=entry_price, qty=qty, cost=CostBreakdown(), leg="entry")
    legs = (entry.model_copy(update={"leg": "stop", "ts": BASE + timedelta(hours=8)}),) if closed else ()
    return Trade(
        id=f"t{opened_bar}",
        instrument=BTC,
        direction=1 if direction == 1 else -1,
        entry_fill=entry,
        legs=legs,
        stop=entry_price - direction * risk_r / qty,
        target=None,
        risk_r=risk_r,
        realized_r=realized_r,
        mae_r=mae_r,
        mfe_r=mfe_r,
        opened_bar=opened_bar,
        closed_bar=closed_bar,
    )


def _open_trade(opened_bar: int) -> Trade:
    """A trade with nothing filled against it yet, so its full quantity is still open."""
    return _trade(opened_bar=opened_bar, closed_bar=None, realized_r=None, closed=False)


# --- mae_mfe ----------------------------------------------------------------------


def test_mae_mfe_measures_the_bars_a_trade_was_open_for() -> None:
    # risk_r 10 on qty 1 and multiplier 1 => 1R is 10 price units from the 100.0 entry.
    trade = _trade(opened_bar=1, closed_bar=3)
    bars = [
        _bar(0, 1.0, 999.0),  # before the entry: must be ignored
        _bar(1, 95.0, 105.0),  # -0.5R / +0.5R
        _bar(2, 85.0, 120.0),  # -1.5R / +2.0R
        _bar(3, 99.0, 101.0),
        _bar(4, 1.0, 999.0),  # after the exit: must be ignored
    ]
    assert mae_mfe(trade, bars) == pytest.approx((1.5, 2.0))


def test_mae_mfe_mirrors_for_a_short() -> None:
    trade = _trade(direction=-1, opened_bar=0, closed_bar=0)
    assert mae_mfe(trade, [_bar(0, 90.0, 130.0)]) == pytest.approx((3.0, 1.0))


def test_mae_mfe_runs_to_the_last_bar_for_an_open_trade() -> None:
    trade = _trade(opened_bar=0, closed_bar=None)
    assert mae_mfe(trade, [_bar(0, 99.0, 101.0), _bar(1, 70.0, 100.0)]) == pytest.approx((3.0, 0.1))


def test_mae_mfe_never_goes_negative() -> None:
    # a bar entirely above the entry has no adverse excursion at all
    trade = _trade(opened_bar=0, closed_bar=0)
    assert mae_mfe(trade, [_bar(0, 110.0, 120.0)]) == pytest.approx((0.0, 2.0))


def test_mae_mfe_rejects_a_bar_index_outside_the_list() -> None:
    with pytest.raises(ValueError, match="bar index"):
        mae_mfe(_trade(opened_bar=5, closed_bar=6), [_bar(0, 99.0, 101.0)])


def test_mae_mfe_of_a_zero_risk_distance_is_zero() -> None:
    # the engine's M10 case: the risk distance underflows, so there is no R denominator
    trade = _trade(opened_bar=0, closed_bar=0, qty=1e300).model_copy(update={"risk_r": 5e-324})
    assert mae_mfe(trade, [_bar(0, 90.0, 110.0)]) == (0.0, 0.0)


@pytest.mark.slow
def test_mae_mfe_agrees_with_the_engine_on_a_synthetic_run() -> None:
    end = add_months(BASE, 12)
    config = Config(entry="ict", exit="fixed_r_2", session="none", instrument=BTC)
    factory = EntryFactoryStub({"ict": lambda instrument, rate: EveryN(40, name="ict")})
    with build_store([BTC], start=BASE, months=12, seed=5) as store:
        source = ReplaySource(store)
        run = run_config(
            config,
            source,
            entry_factory=factory,
            session_factory=session_factory,
            start=BASE,
            end=end,
        )
        bars = [bar for bar in source.merged(BTC, warmup_start(BASE), end) if bar.tf == "4h"]

    assert run.trades
    for trade in run.trades:
        assert mae_mfe(trade, bars) == pytest.approx((trade.mae_r, trade.mfe_r))


# --- per-config excursion statistics ----------------------------------------------


def test_stopped_out_of_winner_rate_counts_losers_that_reached_one_r() -> None:
    trades = [
        _trade(realized_r=-1.0, mfe_r=1.4),  # loser that was up 1.4R
        _trade(realized_r=-1.0, mfe_r=0.3),  # loser that never got there
        _trade(realized_r=2.0, mfe_r=2.5),  # winner: not in the denominator
    ]
    assert excursion.stopped_out_of_winner_rate(trades) == pytest.approx(0.5)


def test_stopped_out_of_winner_rate_is_nan_without_losers() -> None:
    assert math.isnan(excursion.stopped_out_of_winner_rate([_trade(realized_r=1.0, mfe_r=1.0)]))
    assert math.isnan(excursion.stopped_out_of_winner_rate([]))


def test_median_exit_efficiency_is_the_median_captured_share_of_mfe() -> None:
    trades = [
        _trade(realized_r=1.0, mfe_r=2.0),  # 0.5
        _trade(realized_r=3.0, mfe_r=4.0),  # 0.75
        _trade(realized_r=-1.0, mfe_r=3.0),  # a loser: excluded
        _trade(realized_r=1.0, mfe_r=0.0),  # no MFE: excluded
    ]
    assert excursion.median_exit_efficiency(trades) == pytest.approx(0.625)


def test_median_exit_efficiency_clips_above_one() -> None:
    # MFE is measured on bar extremes, so a fill beyond the recorded high can exceed 1
    assert excursion.median_exit_efficiency([_trade(realized_r=3.0, mfe_r=2.0)]) == 1.0


def test_median_exit_efficiency_is_nan_without_winners() -> None:
    assert math.isnan(excursion.median_exit_efficiency([_trade(realized_r=-1.0, mfe_r=2.0)]))


def test_excursion_summary_reports_every_statistic() -> None:
    trades = [
        _trade(realized_r=-1.0, mae_r=1.0, mfe_r=1.5),
        _trade(realized_r=2.0, mae_r=0.2, mfe_r=3.0),
        _trade(realized_r=1.0, mae_r=0.6, mfe_r=2.0),
        _trade(realized_r=-1.0, mae_r=1.1, mfe_r=0.4),
    ]
    summary = excursion_summary(trades)
    assert summary.n == 4
    assert summary.stopped_out_of_winner_rate == pytest.approx(0.5)
    assert summary.median_exit_efficiency == pytest.approx((2.0 / 3.0 + 0.5) / 2)
    assert summary.median_mae_r == pytest.approx(0.8)
    assert summary.median_mfe_r == pytest.approx(1.75)
    assert summary.mfe_p75 == pytest.approx(2.25)


def test_excursion_summary_of_no_trades_is_all_nan() -> None:
    summary = excursion_summary([])
    assert summary.n == 0
    assert math.isnan(summary.median_mae_r)
    assert math.isnan(summary.mfe_p75)


# --- HoldBars and the raw-signal excursion ----------------------------------------


class _Ctx:
    """The three `Context` reads `HoldBars` makes."""

    def __init__(self, bar_index: int) -> None:
        self.bar_index = bar_index


def test_hold_bars_rejects_a_negative_holding_period() -> None:
    with pytest.raises(ValueError, match="bars must be"):
        HoldBars(-1)


def test_hold_bars_keeps_the_signal_stop() -> None:
    rule = HoldBars(20)
    signal = Signal(direction=1, entry=100.0, stop=98.0, structure_target=110.0, tag="t", expires_in_bars=3)
    assert rule.initial_stop(signal, _Ctx(0)) == 98.0  # type: ignore[arg-type]
    assert rule.name == "hold_20"


def test_hold_bars_attaches_a_stop_and_no_target() -> None:
    orders = HoldBars(20).attach(_open_trade(4), _Ctx(4))  # type: ignore[arg-type]
    assert [order.leg for order in orders] == ["stop"]
    assert orders[0].kind == "stop"
    assert orders[0].trade_id == "t4"


def test_hold_bars_closes_at_market_after_the_holding_period() -> None:
    rule = HoldBars(3)
    trade = _open_trade(10)
    assert rule.on_bar(trade, _Ctx(12)) == []  # type: ignore[arg-type]
    orders = rule.on_bar(trade, _Ctx(13))  # type: ignore[arg-type]
    assert [(order.leg, order.kind) for order in orders] == [("time", "market")]
    assert orders[0].price is None


def test_hold_bars_and_attach_ids_never_collide_on_one_bar() -> None:
    rule = HoldBars(0)
    trade = _open_trade(7)
    attached = rule.attach(trade, _Ctx(7))  # type: ignore[arg-type]
    managed = rule.on_bar(trade, _Ctx(7))  # type: ignore[arg-type]
    assert {order.id for order in attached}.isdisjoint({order.id for order in managed})


@pytest.mark.slow
def test_raw_signal_excursion_summarises_an_unmanaged_run() -> None:
    end = add_months(BASE, 12)
    config = Config(entry="ict", exit="fixed_r_2", session="none", instrument=BTC)
    factory = EntryFactoryStub({"ict": lambda instrument, rate: EveryN(40, name="ict")})
    with build_store([BTC], start=BASE, months=12, seed=5) as store:
        summary = excursion.raw_signal_excursion(
            config,
            ReplaySource(store),
            entry_factory=factory,
            session_factory=session_factory,
            start=BASE,
            end=end,
            hold_bars=12,
            seed=11,
        )
    assert summary.n > 0
    assert summary.median_mfe_r >= 0.0
    # the same per-config seed a graded run would hand this entry, so the two are comparable
    assert factory.seeds == [("ict", zlib.crc32(config.id.encode()) ^ 11)]
