"""The engine: drives `Strategy`, `ExitRule`, `Broker` and `Portfolio` bar by bar.

This is the orchestrator for the call sequence documented in `swingforge.strategies.base`.
Backtest and paper trading share this one engine (design spec section 2); only the bar
source differs, and that lives outside `core` entirely — `step`/`run` just take bars.

`core` may import only stdlib, pydantic and numpy, so `Strategy`, `ExitRule` and `Broker`
are used here only as duck-typed arguments: nothing is imported from `strategies` or
`adapters`, and nothing in this module inherits from their protocols.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Literal, Protocol

from swingforge.core.context import Context
from swingforge.core.portfolio import QTY_EPS, Portfolio
from swingforge.core.settings import SettingsReader
from swingforge.core.types import Bar, Fill, Instrument, Order, Position, Signal, Trade, round_to_tick

__all__ = ["Engine", "EngineError"]


class EngineError(RuntimeError):
    """Raised when broker-supplied data (a fill, or an exit-rule order) is impossible.

    These are the only exceptions the engine raises deliberately: bad external input, not
    an internal invariant. A caller integrating a real broker should treat this as a signal
    that the broker and the engine have disagreed about state and stop the run.
    """


class _StrategyLike(Protocol):
    def on_bar(self, ctx: Context) -> Signal | None: ...


class _ExitRuleLike(Protocol):
    def initial_stop(self, signal: Signal, ctx: Context) -> float: ...

    def attach(self, trade: Trade, ctx: Context) -> list[Order]: ...

    def on_bar(self, trade: Trade, ctx: Context) -> list[Order]: ...


class _BrokerLike(Protocol):
    def submit(self, order: Order) -> str: ...

    def cancel(self, order_id: str) -> None: ...

    def on_bar(self, bar: Bar, bar_index: int) -> list[Fill]: ...


@dataclass
class _PendingEntry:
    """An entry order the engine submitted while flat, not yet filled or expired."""

    signal: Signal
    stop: float
    trade_id: str
    expires_at_bar: int
    order_id: str


@dataclass
class _OpenTrade:
    """Mutable bookkeeping for the one trade the engine may have open at a time.

    `Trade` itself is frozen (design spec section 3), so the engine keeps this mutable
    shadow and builds a fresh `Trade` snapshot from it whenever `ExitRule` or the public
    `open_trade` property needs one.
    """

    trade_id: str
    instrument: Instrument
    direction: Literal[1, -1]
    entry_fill: Fill
    stop: float
    target: float | None
    risk_r: float
    risk_distance: float
    opened_bar: int
    regime: str
    context_snapshot: bytes
    remaining_qty: float
    legs: list[Fill] = field(default_factory=list)
    mae_r: float = 0.0
    mfe_r: float = 0.0
    pending_leg_ids: dict[str, str] = field(default_factory=dict)

    def to_trade(self) -> Trade:
        return Trade(
            id=self.trade_id,
            instrument=self.instrument,
            direction=self.direction,
            entry_fill=self.entry_fill,
            legs=tuple(self.legs),
            stop=self.stop,
            target=self.target,
            risk_r=self.risk_r,
            mae_r=self.mae_r,
            mfe_r=self.mfe_r,
            regime=self.regime,
            context_snapshot=self.context_snapshot,
            opened_bar=self.opened_bar,
        )


class Engine:
    """Drives one instrument's `Context` from a chronological stream of closed bars.

    Bars arrive as a mixed stream of `1d` and `4h` bars (a `4h` bar may carry `1h`
    `subbars`); `1h` bars pushed directly are also accepted defensively. Only `4h` bars
    advance `Context.bar_index` and are eligible for trading — see `step` for the exact
    per-bar sequence.
    """

    def __init__(
        self,
        instrument: Instrument,
        strategy: _StrategyLike,
        exit_rule: _ExitRuleLike,
        broker: _BrokerLike,
        portfolio: Portfolio,
        settings_reader: SettingsReader,
        *,
        session_allowed: Callable[[Bar], bool] | None = None,
        regime_tagger: Callable[[Context], str] | None = None,
        context: Context | None = None,
    ) -> None:
        self.instrument = instrument
        self.strategy = strategy
        self.exit_rule = exit_rule
        self.broker = broker
        self.portfolio = portfolio

        self._settings_reader = settings_reader
        self._session_allowed = session_allowed
        self._regime_tagger = regime_tagger
        self.ctx = context if context is not None else Context(instrument)

        # Adopt whatever settings the reader holds at construction time, so the very
        # first bar already has a defined `self.settings` to read `risk_pct` etc. from.
        self._settings_version, self.settings = self._settings_reader.current()

        self._open: _OpenTrade | None = None
        self._pending_entry: _PendingEntry | None = None

    @property
    def open_trade(self) -> Trade | None:
        """A frozen snapshot of the currently open trade, or None while flat."""
        return self._open.to_trade() if self._open is not None else None

    @property
    def trades(self) -> list[Trade]:
        """Every trade closed so far this run.

        `Portfolio.trades` is the single owner of this list (M13): the engine appends a
        closed trade there via `Portfolio.record` and reads it back through this read-only
        property, rather than keeping its own separate copy that could drift.
        """
        return self.portfolio.trades

    def run(self, bars: Iterable[Bar]) -> list[Trade]:
        """Feed every bar to `step` in order; return every trade closed across the run."""
        closed: list[Trade] = []
        for bar in bars:
            closed.extend(self.step(bar))
        return closed

    def step(self, bar: Bar) -> list[Trade]:
        """Advance the engine by one closed bar; return trades closed on this bar.

        1. Refresh settings (a version bump made *during* the previous bar's processing
           is picked up now, at this bar's close — never mid-bar). If the kill switch is now
           on and an entry order is still resting unfilled, cancel it immediately, before the
           broker ever sees this bar.
        2. Push the bar (and, for a 4H bar, its 1H subbars first) into `Context`.
        3. Evaluate the broker against this bar and process the fills in order; an entry
           fill opens the trade, attaches its exit legs, and re-evaluates the same bar
           once more (same-bar exits are resolved pessimistically by the broker).
        4. If a trade is still open, update MAE/MFE and let the exit rule adjust it.
        5. If flat with no pending entry, not killed, and in session: ask the strategy
           for a signal and size/submit it.
        """
        version, settings = self._settings_reader.current()
        if version != self._settings_version:
            self._settings_version = version
            self.settings = settings

        if self.settings.kill_switch and self._pending_entry is not None:
            self.broker.cancel(self._pending_entry.order_id)
            self._pending_entry = None

        if bar.tf in ("1d", "1h"):
            self.ctx.push(bar)
            return []

        for sub in bar.subbars:
            self.ctx.push(sub)
        self.ctx.push(bar)
        bar_index = self.ctx.bar_index

        closed = self._handle_fills(self.broker.on_bar(bar, bar_index), bar, bar_index)

        if self._pending_entry is not None and bar_index > self._pending_entry.expires_at_bar:
            self._pending_entry = None

        if self._open is not None:
            self._update_excursion(bar)
            self._run_exit_rule(bar_index)

        if (
            self._open is None
            and self._pending_entry is None
            and not self.settings.kill_switch
            and (self._session_allowed is None or self._session_allowed(bar))
        ):
            self._maybe_enter(bar_index)

        return closed

    # --- fills ---------------------------------------------------------------------

    def _handle_fills(self, fills: list[Fill], bar: Bar, bar_index: int) -> list[Trade]:
        closed: list[Trade] = []
        for fill in fills:
            if fill.leg == "entry":
                self._open_trade(fill, bar, bar_index)
                # Same-bar re-evaluation: the just-attached stop/target may also fall
                # inside this same bar's range. The broker resolves that pessimistically.
                closed.extend(self._handle_fills(self.broker.on_bar(bar, bar_index), bar, bar_index))
            else:
                trade = self._apply_exit_fill(fill, bar, bar_index)
                if trade is not None:
                    closed.append(trade)
        return closed

    def _open_trade(self, fill: Fill, bar: Bar, bar_index: int) -> None:
        pending = self._pending_entry
        if pending is None:
            raise EngineError(
                f"broker delivered an entry fill (order_id={fill.order_id!r}) with no pending entry on record"
            )
        self._pending_entry = None

        stop = pending.stop
        target = pending.signal.structure_target
        risk_r = self.portfolio.risk_money(fill.price, stop, fill.qty, self.instrument)
        if risk_r <= 0:
            # M10: the fill landed exactly on the stop (e.g. a gap), so the *actual* stop
            # distance is 0 and risk_money returns 0 — which would fail Trade construction
            # (risk_r is PositiveFinite). Fall back to the *planned* risk from the signal's
            # entry, so the trade still carries a meaningful R denominator.
            risk_r = self.portfolio.risk_money(pending.signal.entry, stop, fill.qty, self.instrument)
        risk_distance = abs(fill.price - stop)
        regime = self._regime_tagger(self.ctx) if self._regime_tagger is not None else ""

        open_trade = _OpenTrade(
            trade_id=pending.trade_id,
            instrument=self.instrument,
            direction=pending.signal.direction,
            entry_fill=fill,
            stop=stop,
            target=target,
            risk_r=risk_r,
            risk_distance=risk_distance,
            opened_bar=bar_index,
            regime=regime,
            context_snapshot=self.ctx.snapshot(),
            remaining_qty=fill.qty,
        )
        self._open = open_trade
        self._sync_position()

        legs = self.exit_rule.attach(open_trade.to_trade(), self.ctx)
        self._submit_legs(open_trade, legs)

        # F1: the entry bar itself counts as a live bar for MAE/MFE (its range may extend
        # past the entry price in both directions). `max()` in `_update_excursion` makes
        # this idempotent alongside the per-bar update in `step`.
        self._update_excursion(bar)

    def _apply_exit_fill(self, fill: Fill, bar: Bar, bar_index: int) -> Trade | None:
        open_trade = self._open
        if open_trade is None:
            raise EngineError(
                f"broker delivered an exit fill (order_id={fill.order_id!r}) with no open trade on record"
            )
        if fill.trade_id is not None and fill.trade_id != open_trade.trade_id:
            raise EngineError(
                f"exit fill (order_id={fill.order_id!r}) trade_id {fill.trade_id!r} does not "
                f"match the open trade {open_trade.trade_id!r}"
            )
        if fill.qty <= 0 or fill.qty > open_trade.remaining_qty + QTY_EPS:
            raise EngineError(
                f"exit fill (order_id={fill.order_id!r}) qty {fill.qty!r} is impossible for "
                f"trade {open_trade.trade_id!r} with remaining_qty={open_trade.remaining_qty!r}"
            )

        # F1: fold this bar's own range into MAE/MFE before the frozen `Trade` snapshot is
        # built below, so a trade that closes on this very bar still gets it counted.
        self._update_excursion(bar)

        open_trade.legs.append(fill)
        open_trade.remaining_qty -= fill.qty
        open_trade.pending_leg_ids.pop(fill.leg, None)

        if open_trade.remaining_qty > QTY_EPS:
            self._sync_position()
            return None

        for order_id in open_trade.pending_leg_ids.values():
            self.broker.cancel(order_id)

        pnl = self.portfolio.realized_pnl(
            open_trade.entry_fill, tuple(open_trade.legs), open_trade.direction, self.instrument
        )
        trade = open_trade.to_trade().model_copy(
            update={"realized_r": pnl / open_trade.risk_r, "closed_bar": bar_index}
        )
        self.portfolio.record(trade, pnl)
        self._open = None
        self.ctx.position = None
        return trade

    # --- per-bar management ----------------------------------------------------------

    def _update_excursion(self, bar: Bar) -> None:
        """Fold one bar's high/low range into the open trade's MAE/MFE.

        Called from three places: the end of `_open_trade` (so the entry bar counts), the
        top of `_apply_exit_fill` (so the closing bar counts, before its frozen snapshot is
        built), and once per bar in `step` while a trade stays open in between. `max()`
        makes repeated calls for the same bar idempotent, so calling this from all three
        sites — which can overlap on the bar a trade opens or closes on — never double
        counts anything.

        Using the bar's full high/low, rather than clipping to the entry/exit price,
        slightly overstates MFE past the exit and MAE before the entry: on the entry bar the
        excursion should really start at the fill price, and on the exit bar it should stop
        there, but this uses the bar's extremes regardless. This is the conservative
        direction — it can only inflate MAE/MFE, never understate them — so trade efficiency
        (captured R / MFE) never exceeds 1. A follow-up could use the 1H subbars inside a 4H
        bar to bound the excursion within its own entry/exit bar precisely.
        """
        open_trade = self._open
        assert open_trade is not None
        entry_price = open_trade.entry_fill.price
        if open_trade.direction == 1:
            adverse = entry_price - bar.low
            favourable = bar.high - entry_price
        else:
            adverse = bar.high - entry_price
            favourable = entry_price - bar.low
        distance = open_trade.risk_distance
        if distance <= 0:
            return
        open_trade.mae_r = max(open_trade.mae_r, max(0.0, adverse) / distance)
        open_trade.mfe_r = max(open_trade.mfe_r, max(0.0, favourable) / distance)

    def _run_exit_rule(self, bar_index: int) -> None:
        open_trade = self._open
        assert open_trade is not None
        orders = self.exit_rule.on_bar(open_trade.to_trade(), self.ctx)
        self._submit_legs(open_trade, orders)

    def _submit_legs(self, open_trade: _OpenTrade, orders: list[Order]) -> None:
        """Submit each order and fold stop/target replacements into the open trade.

        Two orders in the same `orders` batch carrying the same `leg` is not rejected — the
        later one simply wins, since it is submitted (and so recorded in
        `pending_leg_ids`/`stop`/`target`) after the earlier one. An `ExitRule` should not
        emit that, but nothing here depends on it not happening.
        """
        changed_stop_or_target = False
        for order in orders:
            if order.trade_id is not None and order.trade_id != open_trade.trade_id:
                raise EngineError(
                    f"exit-rule order (leg={order.leg!r}) trade_id {order.trade_id!r} does not "
                    f"match the open trade {open_trade.trade_id!r}"
                )
            order_id = self.broker.submit(order)
            open_trade.pending_leg_ids[order.leg] = order_id
            if order.leg == "stop" and order.price is not None:
                open_trade.stop = order.price
                changed_stop_or_target = True
            elif order.leg == "target":
                open_trade.target = order.price
                changed_stop_or_target = True

        # F4: keep ctx.position's stop/target aligned with the open trade whenever an
        # ExitRule replaces one (e.g. a trailing stop, or the breakeven move after a scale-out).
        if changed_stop_or_target and self._open is not None:
            self._sync_position()

    def _sync_position(self) -> None:
        """Rebuild `ctx.position` from the open trade's current stop/target/remaining qty."""
        open_trade = self._open
        assert open_trade is not None
        self.ctx.position = Position(
            instrument=self.instrument,
            direction=open_trade.direction,
            qty=open_trade.remaining_qty,
            avg_price=open_trade.entry_fill.price,
            stop=open_trade.stop,
            target=open_trade.target,
        )

    # --- entries -----------------------------------------------------------------

    def _maybe_enter(self, bar_index: int) -> None:
        signal = self.strategy.on_bar(self.ctx)
        if signal is None:
            return
        stop = self.exit_rule.initial_stop(signal, self.ctx)
        qty = self.portfolio.size(signal.entry, stop, self.instrument, self.settings.risk_pct)
        if qty <= 0:
            return
        trade_id = f"{self.instrument.venue}:{self.instrument.symbol}:{bar_index}"
        expires_at_bar = bar_index + signal.expires_in_bars
        order = Order(
            id=f"{trade_id}:entry",
            instrument=self.instrument,
            direction=signal.direction,
            qty=qty,
            kind="limit",
            price=round_to_tick(signal.entry, self.instrument.tick_size),
            expires_at_bar=expires_at_bar,
            leg="entry",
            trade_id=trade_id,
        )
        order_id = self.broker.submit(order)
        self._pending_entry = _PendingEntry(
            signal=signal, stop=stop, trade_id=trade_id, expires_at_bar=expires_at_bar, order_id=order_id
        )
