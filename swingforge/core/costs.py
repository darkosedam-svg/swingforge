"""Cost helpers: a zero cost model, the gate's cost stress, and a fill total.

Real cost models are venue-specific and live in `swingforge.adapters` (Hyperliquid's taker
schedule and funding, OANDA's bid/ask spread and swap). What lives here is the arithmetic
that has no venue in it: charging nothing, scaling a recorded breakdown, and adding fills
up. Gate rule 6 (design spec section 6) re-runs rules 1-5 with spread and slippage doubled
and funding at 1.5x, which is exactly `stress`/`StressedCostModel` with their defaults.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import KW_ONLY, dataclass
from typing import Protocol

from swingforge.core.types import Bar, CostBreakdown, Fill, Order, Position

__all__ = ["NullCostModel", "StressedCostModel", "stress", "total_cost"]

SPREAD_MULT = 2.0
SLIPPAGE_MULT = 2.0
FUNDING_MULT = 1.5
"""Gate rule 6: spread and slippage x2, funding x1.5."""


class _CostSource(Protocol):
    """Structural mirror of `swingforge.adapters.base.CostModel`.

    `core` may not import `adapters` (the layering contract in `.importlinter`), so the
    wrapper below states the shape it needs instead of importing the protocol. Any
    `CostModel` satisfies this.
    """

    def entry(self, order: Order, bar: Bar) -> CostBreakdown: ...

    def carry(self, position: Position, bar: Bar) -> float: ...


class NullCostModel:
    """Charges nothing. The frictionless baseline a costed run is compared against."""

    def entry(self, order: Order, bar: Bar) -> CostBreakdown:
        return CostBreakdown()

    def carry(self, position: Position, bar: Bar) -> float:
        return 0.0


def stress(
    cost: CostBreakdown,
    *,
    spread_mult: float = SPREAD_MULT,
    slippage_mult: float = SLIPPAGE_MULT,
    funding_mult: float = FUNDING_MULT,
) -> CostBreakdown:
    """Scale a recorded breakdown for the gate's cost stress, leaving commission alone.

    Commission is contractual and does not widen when conditions do, so only the three
    market-driven components move. Post-hoc: it rescales what a run already recorded, so
    the gate can re-check a finished backtest without replaying it.
    """
    return CostBreakdown(
        spread=cost.spread * spread_mult,
        commission=cost.commission,
        funding=cost.funding * funding_mult,
        slippage=cost.slippage * slippage_mult,
    )


@dataclass(frozen=True)
class StressedCostModel:
    """Wraps a cost model so every cost it quotes comes out stressed.

    The in-run counterpart to `stress`: use this when a run should be *executed* under
    stressed costs rather than rescored afterwards. `carry` is funding, so it takes the
    funding multiplier.

    Frozen: the multipliers are the configuration of a run, so nothing can rescale costs
    halfway through one.
    """

    inner: _CostSource
    _: KW_ONLY
    spread_mult: float = SPREAD_MULT
    slippage_mult: float = SLIPPAGE_MULT
    funding_mult: float = FUNDING_MULT

    def entry(self, order: Order, bar: Bar) -> CostBreakdown:
        return stress(
            self.inner.entry(order, bar),
            spread_mult=self.spread_mult,
            slippage_mult=self.slippage_mult,
            funding_mult=self.funding_mult,
        )

    def carry(self, position: Position, bar: Bar) -> float:
        return self.inner.carry(position, bar) * self.funding_mult


def total_cost(fills: Iterable[Fill]) -> float:
    """Every cost component of every fill, in quote currency."""
    return sum((fill.cost.total for fill in fills), 0.0)
