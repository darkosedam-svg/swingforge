"""Contract tests for the frozen core models (design spec section 3)."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from swingforge.core import CONTRACT_VERSION
from swingforge.core.types import (
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
TS = datetime(2026, 1, 2, 4, 0, tzinfo=UTC)


def bar(**overrides: object) -> Bar:
    kwargs: dict[str, object] = {
        "instrument": BTC,
        "tf": "4h",
        "ts_open": TS,
        "open": 100.0,
        "high": 105.0,
        "low": 95.0,
        "close": 102.0,
        "volume": 10.0,
    }
    kwargs.update(overrides)
    return Bar(**kwargs)  # type: ignore[arg-type]


def order(**overrides: object) -> Order:
    kwargs: dict[str, object] = {
        "id": "o1",
        "instrument": BTC,
        "direction": 1,
        "qty": 1.0,
        "kind": "limit",
        "price": 100.0,
        "expires_at_bar": None,
        "leg": "entry",
    }
    kwargs.update(overrides)
    return Order(**kwargs)  # type: ignore[arg-type]


def fill(**overrides: object) -> Fill:
    kwargs: dict[str, object] = {
        "order_id": "o1",
        "ts": TS,
        "price": 100.0,
        "qty": 1.0,
        "cost": CostBreakdown(),
        "leg": "entry",
    }
    kwargs.update(overrides)
    return Fill(**kwargs)  # type: ignore[arg-type]


def test_contract_version_is_one() -> None:
    assert CONTRACT_VERSION == 1


# --- Signal ----------------------------------------------------------------


def test_signal_stop_wrong_side_rejected() -> None:
    with pytest.raises(ValidationError):
        Signal(direction=1, entry=100.0, stop=101.0, structure_target=None, tag="t", expires_in_bars=3)


def test_signal_stop_equal_entry_rejected() -> None:
    with pytest.raises(ValidationError):
        Signal(direction=1, entry=100.0, stop=100.0, structure_target=None, tag="t", expires_in_bars=3)


def test_short_signal_valid_and_wrong_side_rejected() -> None:
    short = Signal(direction=-1, entry=100.0, stop=103.0, structure_target=92.0, tag="ict", expires_in_bars=3)
    assert short.stop > short.entry

    with pytest.raises(ValidationError):
        Signal(direction=-1, entry=100.0, stop=97.0, structure_target=None, tag="t", expires_in_bars=3)


def test_long_signal_valid() -> None:
    long = Signal(direction=1, entry=100.0, stop=97.0, structure_target=None, tag="t", expires_in_bars=3)
    assert long.stop < long.entry


# --- Bar -------------------------------------------------------------------


def test_bar_nan_rejected() -> None:
    with pytest.raises(ValidationError):
        bar(close=float("nan"))


def test_bar_infinity_rejected() -> None:
    with pytest.raises(ValidationError):
        bar(high=float("inf"))


def test_bar_naive_datetime_rejected() -> None:
    with pytest.raises(ValidationError):
        bar(ts_open=datetime(2026, 1, 2, 4, 0))


def test_bar_non_utc_timezone_is_normalised_to_utc() -> None:
    plus_two = timezone(timedelta(hours=2))
    b = bar(ts_open=datetime(2026, 1, 2, 6, 0, tzinfo=plus_two))
    assert b.ts_open == TS
    assert b.ts_open.utcoffset() == timedelta(0)


def test_bar_range_must_contain_open_and_close() -> None:
    with pytest.raises(ValidationError):
        bar(low=101.0)
    with pytest.raises(ValidationError):
        bar(high=101.0)


def test_bar_negative_volume_rejected() -> None:
    with pytest.raises(ValidationError):
        bar(volume=-1.0)


def test_bar_accepts_subbars() -> None:
    sub = bar(tf="1h", high=103.0, close=101.0)
    parent = bar(subbars=(sub, sub))
    assert len(parent.subbars) == 2
    assert parent.subbars[0].tf == "1h"
    assert bar().subbars == ()


# --- Order -----------------------------------------------------------------


def test_order_zero_qty_rejected() -> None:
    with pytest.raises(ValidationError):
        order(qty=0.0)
    with pytest.raises(ValidationError):
        order(qty=-1.0)


def test_limit_order_requires_price() -> None:
    with pytest.raises(ValidationError):
        order(kind="limit", price=None)


def test_stop_order_requires_price() -> None:
    with pytest.raises(ValidationError):
        order(kind="stop", price=None, leg="stop")
    assert order(kind="stop", price=94.0, leg="stop").price == 94.0


def test_market_order_price_may_be_none() -> None:
    assert order(kind="market", price=None, leg="time").price is None


def test_order_carries_trade_id_and_leg() -> None:
    o = order(leg="target", trade_id="t-1")
    assert (o.leg, o.trade_id) == ("target", "t-1")
    assert order().trade_id is None


# --- CostBreakdown ---------------------------------------------------------


def test_cost_breakdown_total() -> None:
    cost = CostBreakdown(spread=0.5, commission=0.25, funding=0.125, slippage=0.0625)
    assert cost.total == pytest.approx(0.9375)
    assert CostBreakdown().total == 0.0


# --- Fill / Trade / Position ----------------------------------------------


def test_fill_requires_utc_timestamp() -> None:
    with pytest.raises(ValidationError):
        fill(ts=datetime(2026, 1, 2, 4, 0))


def test_fill_carries_leg_and_trade_id() -> None:
    f = fill(leg="partial", trade_id="t-1")
    assert (f.leg, f.trade_id) == ("partial", "t-1")
    assert fill().trade_id is None


def test_trade_risk_r_must_be_positive() -> None:
    kwargs: dict[str, object] = {
        "id": "t-1",
        "instrument": BTC,
        "direction": 1,
        "entry_fill": fill(),
        "stop": 97.0,
        "target": 106.0,
        "opened_bar": 12,
    }
    trade = Trade(risk_r=100.0, **kwargs)  # type: ignore[arg-type]
    assert trade.legs == ()
    assert trade.realized_r is None
    assert trade.mae_r == 0.0
    assert trade.mfe_r == 0.0
    assert trade.regime == ""
    assert trade.context_snapshot == b""
    assert trade.closed_bar is None

    with pytest.raises(ValidationError):
        Trade(risk_r=0.0, **kwargs)  # type: ignore[arg-type]


def test_position_qty_must_be_positive() -> None:
    pos = Position(instrument=BTC, direction=-1, qty=2.5, avg_price=100.0, stop=103.0, target=None)
    assert pos.qty == 2.5
    with pytest.raises(ValidationError):
        Position(instrument=BTC, direction=-1, qty=0.0, avg_price=100.0, stop=103.0, target=None)


# --- Instrument ------------------------------------------------------------


def test_instrument_tick_size_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        Instrument(
            venue="oanda",
            symbol="EUR_USD",
            tick_size=Decimal("0"),
            contract_multiplier=Decimal("1"),
            quote_ccy="USD",
            session_profile="fx",
        )


# --- Resolver result types -------------------------------------------------


def test_resolution_carries_events_and_mode() -> None:
    res = Resolution(
        events=(ExitEvent(leg="target", price=110.0, target_index=0), ExitEvent(leg="stop", price=100.0)),
        mode="subbars",
    )
    assert res.mode == "subbars"
    assert res.events[0].target_index == 0
    assert res.events[1].target_index is None


# --- Frozen-ness and tick rounding ----------------------------------------


def test_models_are_frozen() -> None:
    instances = [
        BTC,
        bar(),
        Signal(direction=1, entry=100.0, stop=97.0, structure_target=None, tag="t", expires_in_bars=3),
        order(),
        fill(),
        CostBreakdown(),
        Position(instrument=BTC, direction=1, qty=1.0, avg_price=100.0, stop=97.0, target=None),
        ExitEvent(leg="stop", price=97.0),
        Resolution(events=(), mode="pessimistic"),
    ]
    for instance in instances:
        field = next(iter(type(instance).model_fields))
        with pytest.raises(ValidationError):
            setattr(instance, field, getattr(instance, field))


def test_round_to_tick() -> None:
    assert round_to_tick(100.23, Decimal("0.5")) == 100.0
    assert round_to_tick(100.26, Decimal("0.5")) == 100.5
    assert round_to_tick(1.234567, Decimal("0.00001")) == pytest.approx(1.23457)
    # Half-even: both land on the even multiple of the tick.
    assert round_to_tick(100.25, Decimal("0.5")) == 100.0
    assert round_to_tick(100.75, Decimal("0.5")) == 101.0
    assert round_to_tick(-100.25, Decimal("0.5")) == -100.0
    assert isinstance(round_to_tick(100.23, Decimal("0.5")), float)


def test_round_to_tick_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        round_to_tick(float("nan"), Decimal("0.5"))
    with pytest.raises(ValueError):
        round_to_tick(100.0, Decimal("0"))


def test_every_price_field_rejects_nan() -> None:
    with pytest.raises(ValidationError):
        order(price=math.nan)
    with pytest.raises(ValidationError):
        fill(price=math.inf)
    with pytest.raises(ValidationError):
        CostBreakdown(funding=math.nan)


# --- Bar.subbars nesting rules ---------------------------------------------

EUR = Instrument(
    venue="oanda",
    symbol="EUR_USD",
    tick_size=Decimal("0.00001"),
    contract_multiplier=Decimal("1"),
    quote_ccy="USD",
    session_profile="fx",
)


def test_bar_accepts_a_valid_nested_bar() -> None:
    first = bar(tf="1h", high=103.0, close=101.0)
    second = bar(tf="1h", ts_open=TS + timedelta(hours=3), open=101.0, high=104.0, close=99.0)
    parent = bar(subbars=(first, second))
    assert [s.tf for s in parent.subbars] == ["1h", "1h"]

    daily = bar(tf="1d", subbars=(bar(tf="4h", ts_open=TS + timedelta(hours=20)),))
    assert daily.subbars[0].tf == "4h"


def test_bar_rejects_a_subbar_from_another_instrument() -> None:
    with pytest.raises(ValidationError):
        bar(subbars=(bar(tf="1h", instrument=EUR),))


def test_bar_rejects_a_subbar_that_is_not_finer() -> None:
    with pytest.raises(ValidationError):
        bar(subbars=(bar(tf="4h"),))
    with pytest.raises(ValidationError):
        bar(tf="4h", subbars=(bar(tf="1d"),))
    with pytest.raises(ValidationError):
        bar(tf="1h", subbars=(bar(tf="1h"),))


def test_bar_rejects_a_subbar_outside_the_parent_span() -> None:
    with pytest.raises(ValidationError):
        bar(subbars=(bar(tf="1h", ts_open=TS - timedelta(hours=1)),))
    with pytest.raises(ValidationError):
        bar(subbars=(bar(tf="1h", ts_open=TS + timedelta(hours=4)),))
    with pytest.raises(ValidationError):
        bar(tf="1d", subbars=(bar(tf="4h", ts_open=TS + timedelta(hours=24)),))
    assert bar(tf="1d", subbars=(bar(tf="4h", ts_open=TS + timedelta(hours=23)),)).subbars


def test_bar_rejects_a_subbar_outside_the_parent_range() -> None:
    with pytest.raises(ValidationError):
        bar(subbars=(bar(tf="1h", low=94.0),))
    with pytest.raises(ValidationError):
        bar(subbars=(bar(tf="1h", high=106.0),))


def test_bar_rejects_subbars_that_are_not_oldest_first() -> None:
    earlier = bar(tf="1h", ts_open=TS + timedelta(hours=1))
    later = bar(tf="1h", ts_open=TS + timedelta(hours=3))
    with pytest.raises(ValidationError):
        bar(subbars=(later, earlier))
    assert bar(subbars=(earlier, later)).subbars == (earlier, later)


# --- round_to_tick properties ----------------------------------------------

TICKS = [Decimal("1"), Decimal("0.5"), Decimal("0.25"), Decimal("0.01"), Decimal("0.00001")]


@given(
    price=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    tick=st.sampled_from(TICKS),
)
def test_round_to_tick_lands_on_a_tick_within_half_a_tick(price: float, tick: Decimal) -> None:
    result = round_to_tick(price, tick)

    steps = Decimal(str(result)) / tick
    assert abs(steps - steps.to_integral_value()) < Decimal("1e-9")

    assert abs(result - price) <= float(tick) / 2 + 1e-9 * max(1.0, abs(price))
    assert round_to_tick(result, tick) == result
