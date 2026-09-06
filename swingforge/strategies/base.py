"""Strategy and exit-rule protocols, and the call sequence that binds them.

One bar, start to finish::

    Engine
      -> Strategy.on_bar(ctx)              # flat and not killed: may return a Signal
      -> Signal
      -> Portfolio sizes it into Order(leg="entry")
      -> Broker.submit(order)
      -> Fill(leg="entry")                 # on a later bar, when the entry actually fills
      -> Engine builds the Trade
      -> ExitRule.attach(trade, ctx)       # once, right after the entry fill
      -> Broker.submit(...)                # the stop/target legs it returned, each carrying trade_id
      ... every closed bar while the trade is open:
      -> ExitRule.on_bar(trade, ctx)       # replacement or additional legs, or []
      -> Broker.submit(...)
      -> Broker.on_bar(bar, ctx.bar_index) -> [Fill]   # exit fills close or reduce the trade

`Broker.on_bar` is called exactly once per closed 4H bar, with that bar's
`Context.bar_index`. When the engine sizes a signal it converts the signal's relative expiry
into an absolute one: `expires_at_bar = ctx.bar_index + signal.expires_in_bars`, where
`ctx.bar_index` is the bar the signal was emitted on. An order is still live *on* its expiry
bar and is dropped when `bar_index > order.expires_at_bar`; `expires_at_bar is None` never
expires.

A replacement leg is expressed as a new `Order` with the same `trade_id` and the same `leg`
as the pending one; the broker replaces rather than adding, so a trailing stop that updates
every bar never accumulates duplicate legs. The replacement key is `(trade_id, leg)`, and a
`trade_id` of `None` never matches anything, so entry orders are always additive — across
instruments too. `submit` returns the id of the newly accepted order; on replacement the
replaced order's id is no longer pending, and `cancel` on it is a no-op. Returning `[]` from
`on_bar` leaves the existing legs untouched — it does not cancel them.

Both protocols are read-only with respect to `Context`: the engine owns every mutation.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from swingforge.core.context import Context
from swingforge.core.types import Order, Signal, Trade

__all__ = ["ExitRule", "Strategy"]


@runtime_checkable
class Strategy(Protocol):
    """An entry family: ICT structure, supply/demand zones, or the random baseline.

    Called on each closed 4H bar while flat. Daily bars are available through `ctx`.
    """

    name: str
    """Stable identifier used in config ids and result rows."""

    def on_bar(self, ctx: Context) -> Signal | None:
        """Return a `Signal` to enter on this bar, or None.

        The signal's `stop` is the invalidation level, never a distance, and the signal
        expires unfilled after `expires_in_bars` 4H bars. Sizing, session filtering and the
        kill switch are the engine's business, not the strategy's.
        """
        ...


@runtime_checkable
class ExitRule(Protocol):
    """How a trade is managed once it is open. One of the 16 tournament variants."""

    name: str
    """Stable identifier used in config ids and result rows."""

    def attach(self, trade: Trade, ctx: Context) -> list[Order]:
        """Build the initial exit legs, called once immediately after the entry fill.

        Returns the stop and target orders for the trade; every order carries the trade's
        `trade_id`. A rule that overrides the signal's stop (`ATRFixedR`) does so here.
        """
        ...

    def on_bar(self, trade: Trade, ctx: Context) -> list[Order]:
        """Adjust the trade on a closed bar; called on every bar while it is open.

        Returns new or replacement legs — a trailing stop update, a partial scale-out, a
        market order to close on the time stop — or `[]` to leave the trade as it stands.
        """
        ...
