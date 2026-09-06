"""Tests for the cost helpers: the zero model, the stress scaling and the fill total."""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from swingforge.adapters.base import CostModel
from swingforge.core.costs import NullCostModel, StressedCostModel, stress, total_cost
from swingforge.core.types import Bar, CostBreakdown, Fill, Instrument, Order, Position

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
ORDER = Order(
    id="o1",
    instrument=BTC,
    direction=1,
    qty=2.0,
    kind="limit",
    price=100.0,
    expires_at_bar=None,
    leg="entry",
)
POSITION = Position(instrument=BTC, direction=1, qty=2.0, avg_price=100.0, stop=95.0, target=105.0)
RAW = CostBreakdown(spread=1.0, commission=2.0, funding=4.0, slippage=8.0)


class FixedCostModel:
    """An inner model with recognisable non-zero costs, so scaling is visible."""

    def entry(self, order: Order, bar: Bar) -> CostBreakdown:
        return RAW

    def carry(self, position: Position, bar: Bar) -> float:
        return 3.0


def test_null_cost_model_charges_nothing() -> None:
    model = NullCostModel()

    assert model.entry(ORDER, BAR) == CostBreakdown()
    assert model.entry(ORDER, BAR).total == 0.0
    assert model.carry(POSITION, BAR) == 0.0


def test_null_cost_model_satisfies_the_cost_model_protocol() -> None:
    assert isinstance(NullCostModel(), CostModel)


def test_stress_scales_spread_slippage_and_funding_but_never_commission() -> None:
    stressed = stress(RAW)

    assert stressed == CostBreakdown(spread=2.0, commission=2.0, funding=6.0, slippage=16.0)
    assert stressed.total == 26.0


def test_stress_multipliers_are_overridable() -> None:
    stressed = stress(RAW, spread_mult=3.0, slippage_mult=0.5, funding_mult=1.0)

    assert stressed == CostBreakdown(spread=3.0, commission=2.0, funding=4.0, slippage=4.0)


def test_stressed_cost_model_scales_the_inner_entry_and_carry() -> None:
    model = StressedCostModel(FixedCostModel())

    assert model.entry(ORDER, BAR) == CostBreakdown(spread=2.0, commission=2.0, funding=6.0, slippage=16.0)
    assert model.carry(POSITION, BAR) == pytest.approx(4.5)


def test_stressed_cost_model_multipliers_are_overridable() -> None:
    model = StressedCostModel(FixedCostModel(), spread_mult=1.0, slippage_mult=1.0, funding_mult=2.0)

    assert model.entry(ORDER, BAR) == CostBreakdown(spread=1.0, commission=2.0, funding=8.0, slippage=8.0)
    assert model.carry(POSITION, BAR) == 6.0


def test_stressed_cost_model_satisfies_the_cost_model_protocol() -> None:
    assert isinstance(StressedCostModel(NullCostModel()), CostModel)


def test_stressed_null_costs_stay_zero() -> None:
    model = StressedCostModel(NullCostModel())

    assert model.entry(ORDER, BAR) == CostBreakdown()
    assert model.carry(POSITION, BAR) == 0.0


def fill(cost: CostBreakdown) -> Fill:
    return Fill(order_id="o1", ts=TS, price=100.0, qty=1.0, cost=cost, leg="entry")


def test_total_cost_sums_every_component_of_every_fill() -> None:
    fills = [fill(RAW), fill(CostBreakdown(spread=0.5, commission=0.25))]

    assert total_cost(fills) == pytest.approx(15.75)


def test_total_cost_of_no_fills_is_zero() -> None:
    assert total_cost([]) == 0.0


def test_stressed_cost_model_is_frozen() -> None:
    """It carries only configuration, so a run cannot rescale costs halfway through."""
    model = StressedCostModel(NullCostModel())

    with pytest.raises(dataclasses.FrozenInstanceError):
        model.spread_mult = 5.0  # type: ignore[misc]
