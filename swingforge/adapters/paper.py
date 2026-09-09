"""`PaperBroker`: one `Broker` implementation for both venues (design spec section 5).

Same-bar entries: the exit legs the engine attaches after an entry fill are resolved only
against the part of that bar the trade lived through (`_from_entry_onward`) - the candles
before the entry candle are dropped and the entry candle is read pessimistically.

Wraps a `CostModel` for execution costs and carry, and an `ExitResolver` (typically
`swingforge.core.fills.FillResolver`, but any conforming implementation) for deciding which
exit leg a bar hit and in what order. State machine per pending order is
`pending -> filled | cancelled | expired`; every fill is logged with its triggering bar and
cost breakdown via `log`.

Replacement key: `(trade_id, leg)` when `trade_id is not None` (an order submitted with a key
matching a pending order replaces it — the replaced order's id is no longer pending, and
`cancel` on it becomes a no-op); a `trade_id` of `None` is always additive, keyed by `id`.
"""

from __future__ import annotations

import asyncio
import collections
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal

from pydantic import BaseModel, ConfigDict

from swingforge.adapters.base import CostModel, ExitResolver
from swingforge.core.types import Bar, Fill, Instrument, Order, Position, round_to_tick

__all__ = ["FillLog", "PaperBroker"]

# NOTE: duplicated from the canonical `_TF_SPAN` in `swingforge/core/types.py` (a contract
# file this work unit may not edit) -- keep the two in sync by hand if a timeframe's span
# ever changes.
_TF_SPAN: dict[str, timedelta] = {
    "1h": timedelta(hours=1),
    "4h": timedelta(hours=4),
    "1d": timedelta(hours=24),
}
_EXIT_LEGS = ("stop", "target", "partial", "time")
_MAX_QUEUED_FILLS = 10_000
_MAX_LOG_ENTRIES = 100_000
_QTY_EPSILON = 1e-9
"""Relative epsilon for "is this trade flat": `remaining <= entry_qty * _QTY_EPSILON` is
treated as closed. Guards `positions()`/carry-accrual against float dust (e.g. a partial
and a runner whose quantities should sum exactly to `entry_qty` but land a few ULPs off)
without masking a real, small-but-nonzero remainder (see the cancelled-remainder test)."""


class FillLog(BaseModel):
    """One produced fill, with the order and bar that triggered it."""

    model_config = ConfigDict(frozen=True)

    fill: Fill
    order: Order
    bar_ts_open: datetime


@dataclass
class _TradeState:
    """Minimal per-trade bookkeeping the broker needs, derived purely from fills it made.

    Tracks `filled_qty` (not `remaining_qty` directly) so `remaining_qty` is always
    *derived* as `entry_qty - filled_qty`: closing the trade sets `filled_qty = entry_qty`
    exactly, which guarantees an exact `0.0` remainder regardless of how the individual
    leg quantities summed in floating point (WU-1E code-review pass, I2).
    """

    instrument: Instrument
    direction: Literal[1, -1]
    entry_price: float
    entry_qty: float
    entry_bar_ts: datetime
    """`ts_open` of the bar the entry filled on: exits resolved on that same bar may only see
    the bar from the entry onward (see `_from_entry_onward`)."""
    entry_candle: int
    """Index into that bar's `subbars` (0 when it has none) of the candle the entry filled in."""
    filled_qty: float = 0.0
    accrued_carry: float = 0.0

    @property
    def remaining_qty(self) -> float:
        return self.entry_qty - self.filled_qty

    def is_flat(self) -> bool:
        """True once `remaining_qty` is zero, or close enough to be float dust."""
        return self.remaining_qty <= self.entry_qty * _QTY_EPSILON


def _bar_close(bar: Bar) -> datetime:
    return bar.ts_open + _TF_SPAN[bar.tf]


def _from_entry_onward(bar: Bar, state: _TradeState) -> Bar:
    """`bar` as the trade that entered on it experienced it.

    Candles that closed before the entry candle are dropped entirely. The entry candle
    itself cannot be split any finer, so it is read pessimistically from the trade's point
    of view: its `open` becomes the entry price (the first price available after the
    entry, which lies between the stop and the target, so the resolver's gap rule cannot
    credit a move that happened before the entry) and its favourable extreme is clipped to
    the entry price too (a target it reached may have been reached before the entry, so no
    target fills inside the entry candle), while its adverse extreme is kept (a stop it
    reached is assumed to have been reached after the entry). Later candles are untouched.
    Without subbars the whole bar is the entry candle. The parent bar's own OHLC is
    recomputed from the remaining candles so the copy is consistent for any resolver, not
    only one that reads `subbars` first; prices are clamped into the candle's range so the
    copy stays a valid bar after tick rounding.
    """
    candles = bar.subbars or (bar,)
    entry = candles[state.entry_candle]
    open_ = min(max(state.entry_price, entry.low), entry.high)
    if state.direction == 1:
        close = min(entry.close, open_)
        first = entry.model_copy(update={"open": open_, "high": max(open_, close), "close": close})
    else:
        close = max(entry.close, open_)
        first = entry.model_copy(update={"open": open_, "low": min(open_, close), "close": close})
    if not bar.subbars:
        return first
    remaining = (first, *candles[state.entry_candle + 1 :])
    return bar.model_copy(
        update={
            "subbars": remaining,
            "open": first.open,
            "high": max(candle.high for candle in remaining),
            "low": min(candle.low for candle in remaining),
            "close": remaining[-1].close,
        }
    )


class PaperBroker:
    """A `Broker` that fills orders against closed bars using a `CostModel` and `ExitResolver`."""

    def __init__(self, cost_model: CostModel, resolver: ExitResolver) -> None:
        self._cost_model = cost_model
        self._resolver = resolver
        self._pending: dict[str, Order] = {}
        self._keyed_index: dict[tuple[str, str], str] = {}
        self._trades: dict[str, _TradeState] = {}
        # A plain bounded deque, not `asyncio.Queue`: `on_bar` is synchronous and pushes
        # with a non-blocking append, so a bound has to mean "drop the oldest" rather than
        # raise/block -- `asyncio.Queue(maxsize=...)` would raise `QueueFull` out of
        # `put_nowait` once paper trading ran long enough to fill it. `maxlen` bounds
        # memory for a long-running paper session without that failure mode.
        self._fill_queue: collections.deque[Fill] = collections.deque(maxlen=_MAX_QUEUED_FILLS)
        self._last_carry_bar_index: int | None = None
        # Bounded the same way, and for the same reason, as `_fill_queue`: a long-running
        # paper session must not grow this without limit. `log` is a convenience for
        # introspection (tests, a REPL session) -- the durable audit trail is the store's
        # `fills` table, written by the CLI as each fill is produced -- so dropping the
        # oldest entries here (only once a session has produced over 100,000 fills) never
        # loses anything that isn't already persisted elsewhere.
        self.log: collections.deque[FillLog] = collections.deque(maxlen=_MAX_LOG_ENTRIES)
        # `_fill_ready` is created lazily (see `_ensure_fill_ready`) rather than here in
        # `__init__`, so a broker constructed outside a running event loop (e.g. at
        # module import time, or in a sync test fixture) stays constructible -- the
        # `asyncio.Event` only comes into being the first time `on_bar` or `fills` actually
        # needs one, which happens once a loop is running.
        self._fill_ready: asyncio.Event | None = None
        self._closed = False

    # -- Broker protocol ----------------------------------------------------

    def submit(self, order: Order) -> str:
        if order.leg == "entry" and order.kind == "stop":
            raise NotImplementedError("stop-entry orders are not supported by PaperBroker")
        if order.trade_id is not None:
            key = (order.trade_id, order.leg)
            old_id = self._keyed_index.get(key)
            if old_id is not None and old_id != order.id:
                self._pending.pop(old_id, None)
            self._keyed_index[key] = order.id
        self._pending[order.id] = order
        return order.id

    def cancel(self, order_id: str) -> None:
        self._remove_pending(order_id)

    def positions(self) -> list[Position]:
        return [
            self._position_for(trade_id, state)
            for trade_id, state in self._trades.items()
            if not state.is_flat()
        ]

    def on_bar(self, bar: Bar, bar_index: int) -> list[Fill]:
        """Idempotent for already-filled/expired orders: safe to call twice per `bar_index`.

        The engine does exactly that on an entry bar — once to learn the entry fill (so it
        can build the `Trade` and attach exit legs), and again, same `bar_index`, once those
        legs are pending, to let them resolve against the rest of that same bar - and only
        the rest: `_from_entry_onward` drops the candles that closed before the entry and
        reads the entry candle pessimistically (no target inside it, its stop still live).

        Engine invariant: a trade with pending target/partial legs must also carry a
        pending stop leg (every exit rule that attaches a target attaches a stop first) --
        `_fill_exits_for_trade` raises `RuntimeError` if that invariant is ever violated,
        rather than silently skipping resolution for that trade.
        """
        self._drop_expired(bar_index)
        self._accrue_carry_once(bar, bar_index)

        fills = [*self._fill_entries(bar), *self._fill_exits(bar)]
        for fill in fills:
            self._fill_queue.append(fill)
        if fills:
            self._ensure_fill_ready().set()
        return fills

    async def fills(self) -> AsyncIterator[Fill]:
        """Stream fills as they're produced by `on_bar`.

        Awaits an `asyncio.Event` instead of polling: a consumer with nothing to read
        suspends here rather than burning a core in a busy `while True: await
        asyncio.sleep(0)` spin. `on_bar` sets the event every time it appends at least one
        fill, so a suspended consumer wakes as soon as one is available.

        `close()` ends the stream (this generator returns, so a consumer's `async for fill
        in broker.fills(): ...` completes normally) once the queue has been fully drained
        -- a fill queued before `close()` is still delivered.

        The queue behind this stream is a bounded deque (see `_fill_queue`'s docstring):
        it drops the oldest fill only if a consumer falls more than 10,000 fills behind
        production, which the store's `fills` table (the durable audit trail, written by
        the CLI as each fill is produced) never does.
        """
        event = self._ensure_fill_ready()
        while True:
            while self._fill_queue:
                yield self._fill_queue.popleft()
            if self._closed:
                return
            event.clear()
            await event.wait()

    def close(self) -> None:
        """End the `fills()` stream once its queue has been drained.

        Safe to call whether or not a consumer is currently awaiting `fills()`, and
        whether or not the queue is currently empty.
        """
        self._closed = True
        self._ensure_fill_ready().set()

    def _ensure_fill_ready(self) -> asyncio.Event:
        if self._fill_ready is None:
            self._fill_ready = asyncio.Event()
        return self._fill_ready

    # -- pending-order bookkeeping -------------------------------------------

    def _remove_pending(self, order_id: str) -> Order | None:
        order = self._pending.pop(order_id, None)
        if order is not None and order.trade_id is not None:
            key = (order.trade_id, order.leg)
            if self._keyed_index.get(key) == order_id:
                del self._keyed_index[key]
        return order

    def _drop_expired(self, bar_index: int) -> None:
        expired = [
            order_id
            for order_id, order in self._pending.items()
            if order.expires_at_bar is not None and bar_index > order.expires_at_bar
        ]
        for order_id in expired:
            self._remove_pending(order_id)

    def _orders_for_trade(self, trade_id: str) -> list[Order]:
        # Iteration order here is `self._pending`'s insertion order (a plain dict), which is
        # also what drives the cross-trade fill order in `_fill_exits`: replacing a leg
        # (`_remove_pending` then re-`submit`) re-inserts it at the end, so a trade whose
        # leg was just replaced sorts after trades whose legs were not touched this bar.
        return [order for order in self._pending.values() if order.trade_id == trade_id]

    def _close_trade(self, trade_id: str) -> None:
        for order in self._orders_for_trade(trade_id):
            self._remove_pending(order.id)
        state = self._trades.pop(trade_id, None)
        if state is not None:
            # Set exactly (not incrementally decremented): whatever floating-point drift
            # the individual leg quantities accumulated, a closed trade's remainder is
            # exactly 0.0 -- this mutates the same object `_fill_exits_for_trade` still
            # holds as its local `state`, so its loop-top `state.is_flat()` check also sees
            # the exact close (WU-1E code-review pass, I2).
            state.filled_qty = state.entry_qty
            # Popped, not left flat in the dict: every pending leg for this trade was just
            # removed above, so nothing will look this trade_id up again -- keeps a
            # long-running paper session's state bounded to only currently-open trades.

    # -- positions / carry ----------------------------------------------------

    def _position_for(self, trade_id: str, state: _TradeState) -> Position:
        stop = state.entry_price
        targets: list[float] = []
        for order in self._orders_for_trade(trade_id):
            if order.leg == "stop" and order.price is not None:
                stop = order.price
            elif order.leg in ("target", "partial") and order.price is not None:
                targets.append(order.price)
        target = min(targets, key=lambda price: price * state.direction) if targets else None
        return Position(
            instrument=state.instrument,
            direction=state.direction,
            qty=state.remaining_qty,
            avg_price=state.entry_price,
            stop=stop,
            target=target,
        )

    def _accrue_carry_once(self, bar: Bar, bar_index: int) -> None:
        """Accrue carry for every open position exactly once per `bar_index`.

        Guarded so the engine's double-call-on-an-entry-bar pattern (see `on_bar`) never
        double-charges the same physical bar's funding/rollover.
        """
        if bar_index == self._last_carry_bar_index:
            return
        for trade_id, state in self._trades.items():
            if not state.is_flat():
                position = self._position_for(trade_id, state)
                state.accrued_carry += self._cost_model.carry(position, bar)
        self._last_carry_bar_index = bar_index

    # -- entry fills ----------------------------------------------------------

    def _fill_entries(self, bar: Bar) -> list[Fill]:
        """Fill every pending entry the bar touches, at the first candle that touches it.

        With 1H subbars the entry is placed in the subbar that first reached the limit (and
        a subbar that opened through the limit fills at its own open, the first price the
        order could get), so the same-bar exit pass knows which part of the bar the trade
        actually lived through. Without subbars the bar is the only candle. Subbars are
        assumed to cover the parent's range (a `Bar` only validates that they lie inside
        it); when they do not and only the parent touches the order, the parent fills it.
        """
        fills: list[Fill] = []
        candles = bar.subbars or (bar,)
        for order in [o for o in self._pending.values() if o.leg == "entry"]:
            hit = self._first_touch(order, candles)
            if hit is None and candles is not (bar,):
                # Subbars are validated to lie inside the parent but not to cover it: if
                # none touches the order while the bar as a whole does, fall back to the
                # whole bar (as if it had no subbars) rather than silently never filling.
                price = self._entry_fill_price(order, bar)
                hit = None if price is None else (0, price)
            if hit is None:
                continue
            entry_candle, price = hit
            fill = self._make_fill(order, bar, price, order.qty, "entry", carry_from=None)
            fills.append(fill)
            self._remove_pending(order.id)
            if order.trade_id is not None:
                self._trades[order.trade_id] = _TradeState(
                    instrument=order.instrument,
                    direction=order.direction,
                    entry_price=fill.price,
                    entry_qty=order.qty,
                    entry_bar_ts=bar.ts_open,
                    entry_candle=entry_candle,
                )
        return fills

    def _first_touch(self, order: Order, candles: tuple[Bar, ...]) -> tuple[int, float] | None:
        """The (candle index, price) `order` first fills at, or None if no candle touches it."""
        for index, candle in enumerate(candles):
            price = self._entry_fill_price(order, candle)
            if price is not None:
                return index, price
        return None

    @staticmethod
    def _entry_fill_price(order: Order, bar: Bar) -> float | None:
        """The price `order` fills at inside one candle (a subbar, or the whole bar), or None.

        A market order takes the candle's open (with subbars, the first subbar's open - which
        is the bar's open on consistent venue data). A limit touched by the candle fills at
        the better of the limit and the candle's open: a candle that opened through the limit
        fills at its open, the first price the resting order could get, so a limit gapped
        through by a LATER subbar fills better than the limit, never worse.
        """
        if order.kind == "market":
            return bar.open
        if order.kind == "limit":
            price = order.price
            if price is None:
                # `Order._price_matches_kind` requires a price for `kind="limit"`, so this
                # is unreachable through the public API; a raise (not a silent fallback)
                # keeps that invariant loud if it's ever violated some other way.
                raise RuntimeError(f"limit order {order.id!r} has no price despite kind='limit'")
            if order.direction == 1:
                return min(price, bar.open) if bar.low <= price else None
            return max(price, bar.open) if bar.high >= price else None
        return None

    # -- exit-leg fills ---------------------------------------------------------

    def _fill_exits(self, bar: Bar) -> list[Fill]:
        fills: list[Fill] = []
        # Trade processing order here follows `self._pending`'s insertion order (see
        # `_orders_for_trade`): a trade whose leg was replaced (removed + re-submitted)
        # this bar sorts after every trade whose legs were untouched.
        trade_ids = list(
            dict.fromkeys(
                order.trade_id
                for order in self._pending.values()
                if order.trade_id is not None and order.leg in _EXIT_LEGS
            )
        )
        for trade_id in trade_ids:
            fills.extend(self._fill_exits_for_trade(trade_id, bar))
        return fills

    def _fill_exits_for_trade(self, trade_id: str, bar: Bar) -> list[Fill]:
        state = self._trades.get(trade_id)
        if state is None or state.is_flat():
            return []

        orders = self._orders_for_trade(trade_id)
        time_order = next((o for o in orders if o.leg == "time"), None)
        if time_order is not None:
            # Fills at the bar's own open: a time stop is attached by the exit rule's manage
            # path bars after the entry, never on the entry bar, so this is never a pre-entry price.
            fill = self._make_fill(time_order, bar, bar.open, state.remaining_qty, "time", carry_from=state)
            self._close_trade(trade_id)
            return [fill]

        stop_order = next((o for o in orders if o.leg == "stop"), None)
        target_orders = sorted(
            (o for o in orders if o.leg in ("target", "partial") and o.price is not None),
            key=lambda o: o.price * state.direction,  # type: ignore[operator]
        )
        if stop_order is None or stop_order.price is None:
            if target_orders:
                # Engine invariant (documented on `on_bar`): every exit rule attaches a stop
                # before it ever attaches a target/partial, so a trade with pending targets
                # but no stop means the engine (or a test double) violated that invariant --
                # loud failure beats silently never resolving this trade's targets.
                raise RuntimeError(
                    f"trade {trade_id!r} has pending target/partial legs but no stop leg -- "
                    "every trade with pending exits must carry a stop order"
                )
            return []  # nothing resolvable yet: no stop, no targets pending either

        # `any(... "partial" ...)` re-scans `target_orders`, which is rebuilt fresh from
        # `self._pending` on every call (including the engine's documented same-bar-index
        # re-call after a leg fills): a partial that already filled this bar was removed
        # from `self._pending` by `_remove_pending` before this method could be re-entered,
        # so it can never still appear here as "pending" and trigger a second resolve.
        stop_after_partial = state.entry_price if any(o.leg == "partial" for o in target_orders) else None

        # On the entry bar, only the part of the bar the trade lived through (`_from_entry_onward`).
        resolution = self._resolver.resolve(
            _from_entry_onward(bar, state)
            if state.entry_bar_ts == bar.ts_open and state.instrument == bar.instrument
            else bar,
            stop_order.price,
            [o.price for o in target_orders if o.price is not None],
            state.direction,
            stop_after_partial=stop_after_partial,
        )

        fills: list[Fill] = []
        for event in resolution.events:
            if state.is_flat():
                break
            if event.leg == "stop":
                # `event.price` is the level the resolver actually executed at -- e.g. the
                # breakeven `stop_after_partial` level rather than `stop_order.price`, when a
                # partial and the post-partial stop both trigger within the same bar.
                qty = state.remaining_qty
                fills.append(self._make_fill(stop_order, bar, event.price, qty, "stop", state))
                self._close_trade(trade_id)
            else:
                if event.target_index is None:
                    # `ExitEvent.target_index` is `None` only for `leg="stop"` events (see
                    # its docstring); a raise here beats a confusing `IndexError` below.
                    raise RuntimeError(f"resolver returned a target event with no target_index: {event!r}")
                target_order = target_orders[event.target_index]
                fills.append(
                    self._make_fill(target_order, bar, event.price, target_order.qty, target_order.leg, state)
                )
                if target_order.leg == "partial":
                    self._remove_pending(target_order.id)
                    state.filled_qty += target_order.qty
                else:  # the runner's target: at most two targets, so this is the final one
                    self._close_trade(trade_id)
        return fills

    # -- shared fill construction -------------------------------------------------

    def _make_fill(
        self,
        order: Order,
        bar: Bar,
        price: float,
        qty: float,
        leg: Literal["entry", "stop", "target", "partial", "time"],
        carry_from: _TradeState | None,
    ) -> Fill:
        rounded = round_to_tick(price, order.instrument.tick_size)
        # Cost against the *actual filled quantity*, not the pending order's original qty --
        # they diverge for a stop/time exit on a trade that was already partially closed
        # (`qty` here is `state.remaining_qty`, smaller than `order.qty`); a qty-proportional
        # cost model must see that smaller amount (WU-1E code-review pass, I1).
        cost_order = order if qty == order.qty else order.model_copy(update={"qty": qty})
        cost = self._cost_model.entry(cost_order, bar)
        if carry_from is not None:
            cost = cost.model_copy(update={"funding": cost.funding + carry_from.accrued_carry})
            carry_from.accrued_carry = 0.0
        fill = Fill(
            order_id=order.id,
            # Bar close, even for an entry fill (priced off the touching candle's open) -- a
            # `Fill.ts` always marks when the *bar* closed, not the intrabar instant a
            # market/limit order would have actually touched, since replay only ever sees
            # bars after they've closed (see `ReplaySource`/`BarSource`, no lookahead).
            ts=_bar_close(bar),
            price=rounded,
            qty=qty,
            cost=cost,
            leg=leg,
            trade_id=order.trade_id,
        )
        self.log.append(FillLog(fill=fill, order=order, bar_ts_open=bar.ts_open))
        return fill
