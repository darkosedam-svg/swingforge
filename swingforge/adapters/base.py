"""Adapter protocols: where bars come from, where orders go, what they cost.

Backtest and paper trading run the same engine; only the `BarSource` differs (replay from
DuckDB vs a live stream). Everything below is structural typing — an implementation just
needs the methods, it does not inherit from these.

`swingforge.adapters` is the only package allowed to import venue SDKs, which is why these
protocols live here rather than in `core`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import datetime
from typing import Literal, Protocol, runtime_checkable

from swingforge.core.types import (
    TF,
    Bar,
    CostBreakdown,
    Fill,
    Instrument,
    Order,
    Position,
    Resolution,
)

__all__ = ["BarSource", "Broker", "CostModel", "ExitResolver"]


@runtime_checkable
class BarSource(Protocol):
    """A source of closed bars for one instrument and timeframe.

    Both methods yield *closed* bars only: a bar is emitted strictly after its close, never
    while it is still forming. This is the no-lookahead boundary of the whole system.
    """

    def history(self, instrument: Instrument, tf: TF, start: datetime, end: datetime) -> list[Bar]:
        """Closed bars with `start <= ts_open < end`, oldest first.

        `start` and `end` must be timezone-aware. A 4H bar may carry its 1H `subbars` so the
        exit resolver can order a stop and a target that both fall inside it.
        """
        ...

    def stream(self, instrument: Instrument, tf: TF) -> AsyncIterator[Bar]:
        """Yield each bar as it closes, indefinitely. Used by paper trading."""
        ...


@runtime_checkable
class Broker(Protocol):
    """Order routing and fill generation.

    Two modes share one implementation. In backtest and paper the engine drives the broker
    bar by bar with `on_bar`, which is synchronous and deterministic; `fills()` is the
    streaming view used when a venue pushes fills asynchronously.
    """

    def submit(self, order: Order) -> str:
        """Accept an order and return the id of the newly accepted order.

        The replacement key is `(trade_id, leg)`: an order whose pair matches a pending
        order *replaces* it, which is how an exit rule moves a trailing stop or a target
        without cancelling first. A `trade_id` of `None` never matches anything, so entry
        orders are always additive — across instruments too, since the key does not include
        the instrument. On replacement the replaced order's id is no longer pending, and
        `cancel` on it is a no-op.
        """
        ...

    def cancel(self, order_id: str) -> None:
        """Cancel a pending order. Cancelling an unknown or already-filled order is a no-op."""
        ...

    def positions(self) -> list[Position]:
        """Currently open positions."""
        ...

    def on_bar(self, bar: Bar, bar_index: int) -> list[Fill]:
        """Evaluate every pending order against one closed bar and return the fills.

        `bar_index` is the engine's `Context.bar_index` for this 4H bar. `on_bar` is called
        exactly once per closed execution-timeframe (4H) bar.

        Synchronous and ordered: the returned fills are in the order they occurred within
        the bar, as decided by the `ExitResolver` when more than one level was touched.

        An order is still live *on* its expiry bar and is dropped (no fill) when
        `bar_index > order.expires_at_bar`; an `expires_at_bar` of `None` never expires. The
        engine sets that field when it turns a `Signal` into an order:
        `expires_at_bar = ctx.bar_index + signal.expires_in_bars`, where `ctx.bar_index` is
        the bar the signal was emitted on.
        """
        ...

    def fills(self) -> AsyncIterator[Fill]:
        """Stream fills as the venue reports them. Paper/live streaming mode only."""
        ...


@runtime_checkable
class CostModel(Protocol):
    """Venue-specific trading costs, in quote currency."""

    def entry(self, order: Order, bar: Bar) -> CostBreakdown:
        """Cost of executing `order` against `bar`: spread, commission and slippage."""
        ...

    def carry(self, position: Position, bar: Bar) -> float:
        """Cost of holding `position` across `bar` — perp funding, or FX swap at rollover.

        Positive means the position paid; negative means it was paid. Returns 0.0 when the
        bar does not cross a funding or rollover boundary.
        """
        ...


@runtime_checkable
class ExitResolver(Protocol):
    """Decides which exit levels a bar hit, and in what order.

    Implemented by `swingforge.core.fills.FillResolver`; `PaperBroker` depends on it through
    this protocol so the resolution policy can be swapped in tests.
    """

    def resolve(
        self,
        bar: Bar,
        stop: float,
        targets: list[float],
        direction: Literal[1, -1],
        *,
        stop_after_partial: float | None = None,
    ) -> Resolution:
        """Resolve the exits triggered inside one closed bar.

        `targets` are ordered nearest-first; a hit reports its index in `ExitEvent`. When
        both a stop and a target lie inside the bar's range, the 1H `subbars` decide which
        came first and the result's `mode` is `"subbars"`; with no subbars the stop is
        assumed first and `mode` is `"pessimistic"`.

        `stop_after_partial` is the stop level that applies once the first target has been
        taken (breakeven, for a `Partial` exit rule); it is applied before the remaining
        levels are evaluated on the same bar. At most two targets are supported, and
        `stop_after_partial` applies after `targets[0]` is taken.

        The resolver reports price levels only; the caller maps `ExitEvent.target_index`
        back to the pending `Order` for that target and takes the `leg` label from that
        order — so a Partial's first target yields `Fill(leg="partial")` and the runner's
        target yields `Fill(leg="target")`. Time stops are submitted as `kind="market"`
        orders and never reach the resolver.

        On the bar a trade entered on, `PaperBroker` passes a clipped view of the bar - the
        candles before the entry dropped, the entry candle read pessimistically - so this
        method never has to know where inside the bar the entry happened (see
        `swingforge.adapters.paper._from_entry_onward`).
        """
        ...
