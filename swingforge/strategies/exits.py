"""Exit rules: the stop/target legs crossed with entry strategies in the tournament.

Every rule implements the `ExitRule` protocol (`strategies/base.py`): `initial_stop` before
sizing, `attach` once right after the entry fill, `on_bar` on every closed 4H bar while the
trade is open. The engine builds `Order`s from what these methods return and submits them to
the broker, which replaces a pending leg on the key `(trade_id, leg)`.

Shared vocabulary, per the orchestrator:
- ``r_dist`` is the price distance per 1R for the trade's *original* size:
  ``risk_r / (entry_fill.qty * contract_multiplier)``. It stays valid even after the stop
  moves, so every R-based level (targets, activation thresholds) is computed from it rather
  than re-derived from the current stop.
- ``unrealized_r`` and ``remaining_qty`` are exported helpers any caller can use to judge an
  open trade; every rule here uses them internally too. In particular the time stop judges
  unrealized R on the *remaining* quantity, not the trade's original size — intentionally:
  once a partial has scaled out, the question the time stop asks is whether the runner
  itself is working, not whether the trade as a whole has been profitable.
- The time stop (close at market after `time_stop_bars` 4H bars if unrealized profit is
  under `time_stop_min_r` R) is identical across all 16 variants, so it lives once in
  `_ExitRuleBase.on_bar`, which every rule inherits; a rule's own per-bar logic goes in
  `_manage`, called only when the time stop does not fire. `on_bar`'s time-stop check runs
  first and, when it fires, returns *only* the market order for that bar — never alongside
  other legs. Invariant this depends on: a run's `max_bars` (its maximum simulated holding
  period) must exceed `time_stop_bars`, so a trade is never force-closed by the run's
  horizon before the time stop gets a chance to fire.
- ATR everywhere means `ctx.atr("1d", 14)`, the Daily ATR. Below `n + 1` Daily bars it is
  NaN: `ATRFixedR.initial_stop` then falls back to `signal.stop`, and `Trailing` skips its
  trailing update for that bar (but the time stop still applies).
- Order ids carry a phase suffix (`"a"` for orders built in `attach`, `"m"` for orders built
  in `on_bar`/`_manage`) so a stop rebuilt in `on_bar` on the trade's entry-fill bar never
  collides with the one `attach` already built for that same bar — `attach` and `on_bar` can
  both fire on that bar (see `strategies/base.py`).
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np

from swingforge.core.context import Context
from swingforge.core.types import Order, Signal, Trade, round_to_tick
from swingforge.strategies.base import ExitRule

__all__ = [
    "EXIT_GRID",
    "ATRFixedR",
    "FixedR",
    "Partial",
    "Structure",
    "Trailing",
    "remaining_qty",
    "unrealized_r",
]


def remaining_qty(trade: Trade) -> float:
    """Quantity still open: the entry fill's qty minus every exit leg filled so far."""
    return trade.entry_fill.qty - sum(leg.qty for leg in trade.legs)


def unrealized_r(trade: Trade, price: float) -> float:
    """Open profit or loss at `price`, in R, on the quantity still open."""
    mult = float(trade.instrument.contract_multiplier)
    return trade.direction * (price - trade.entry_fill.price) * remaining_qty(trade) * mult / trade.risk_r


def _r_dist(trade: Trade) -> float:
    """Price distance per 1R for the trade's original size — stable once the stop moves."""
    return trade.risk_r / (trade.entry_fill.qty * float(trade.instrument.contract_multiplier))


def _fmt(value: float) -> str:
    """Render a grid parameter for a rule name: integers bare, everything else as-is."""
    return str(int(value)) if float(value).is_integer() else str(value)


def _order_id(trade: Trade, leg: str, ctx: Context, phase: Literal["a", "m"]) -> str:
    """`phase` is "a" for an order built in `attach`, "m" for one built in `on_bar`/
    `_manage` — see the module docstring for why that suffix is needed."""
    return f"{trade.id}:{leg}:{ctx.bar_index}:{phase}"


def _stop_order(
    trade: Trade,
    ctx: Context,
    phase: Literal["a", "m"],
    *,
    price: float | None = None,
    qty: float | None = None,
) -> Order:
    resolved_price = trade.stop if price is None else price
    resolved_qty = remaining_qty(trade) if qty is None else qty
    return Order(
        id=_order_id(trade, "stop", ctx, phase),
        instrument=trade.instrument,
        direction=-trade.direction,
        qty=resolved_qty,
        kind="stop",
        price=round_to_tick(resolved_price, trade.instrument.tick_size),
        expires_at_bar=None,
        leg="stop",
        trade_id=trade.id,
    )


def _limit_order(
    trade: Trade,
    ctx: Context,
    price: float,
    leg: Literal["target", "partial"],
    qty: float,
    phase: Literal["a", "m"],
) -> Order:
    return Order(
        id=_order_id(trade, leg, ctx, phase),
        instrument=trade.instrument,
        direction=-trade.direction,
        qty=qty,
        kind="limit",
        price=round_to_tick(price, trade.instrument.tick_size),
        expires_at_bar=None,
        leg=leg,
        trade_id=trade.id,
    )


def _time_stop_market_order(trade: Trade, ctx: Context, qty: float) -> Order:
    return Order(
        id=_order_id(trade, "time", ctx, "m"),
        instrument=trade.instrument,
        direction=-trade.direction,
        qty=qty,
        kind="market",
        price=None,
        expires_at_bar=None,
        leg="time",
        trade_id=trade.id,
    )


def _stop_and_target(trade: Trade, ctx: Context, target: float) -> list[Order]:
    """Stop at `trade.stop` plus a limit target at `target`, both sized to the remaining
    qty. Shared by `_fixed_rr_orders` (FixedR/ATRFixedR) and `Structure.attach` — both are
    only ever called from `attach`, hence the hard-coded "a" phase.
    """
    qty = remaining_qty(trade)
    return [_stop_order(trade, ctx, "a"), _limit_order(trade, ctx, target, "target", qty, "a")]


def _fixed_rr_orders(trade: Trade, ctx: Context, rr: float) -> list[Order]:
    """Stop at `trade.stop` plus a target `rr` R away — shared by FixedR and ATRFixedR."""
    r_dist = _r_dist(trade)
    target = trade.entry_fill.price + trade.direction * rr * r_dist
    return _stop_and_target(trade, ctx, target)


class _ExitRuleBase:
    """The time stop every rule shares: implemented once, called from every `on_bar`.

    Concrete rules call `super().__init__(time_stop_bars=..., time_stop_min_r=...)` and
    implement `_manage` for their own per-bar logic; `on_bar` itself is not overridden.
    """

    name: str
    time_stop_bars: int
    time_stop_min_r: float

    def __init__(self, *, time_stop_bars: int = 10, time_stop_min_r: float = 0.5) -> None:
        self.time_stop_bars = time_stop_bars
        self.time_stop_min_r = time_stop_min_r

    def on_bar(self, trade: Trade, ctx: Context) -> list[Order]:
        time_order = self._time_stop_order(trade, ctx)
        if time_order is not None:
            return [time_order]
        return self._manage(trade, ctx)

    def _time_stop_order(self, trade: Trade, ctx: Context) -> Order | None:
        if ctx.bar_index - trade.opened_bar < self.time_stop_bars:
            return None
        bar = ctx.last("4h")
        if bar is None:
            return None
        if abs(unrealized_r(trade, bar.close)) >= self.time_stop_min_r:
            return None
        return _time_stop_market_order(trade, ctx, remaining_qty(trade))

    def _manage(self, trade: Trade, ctx: Context) -> list[Order]:
        """Rule-specific per-bar management; the default is to do nothing further."""
        return []


class FixedR(_ExitRuleBase):
    """Stop where the signal says, target a fixed R multiple away. Grid: rr in {1.5, 2, 3}."""

    def __init__(self, rr: float, *, time_stop_bars: int = 10, time_stop_min_r: float = 0.5) -> None:
        super().__init__(time_stop_bars=time_stop_bars, time_stop_min_r=time_stop_min_r)
        self.rr = rr
        self.name = f"fixed_r_{_fmt(rr)}"

    def initial_stop(self, signal: Signal, ctx: Context) -> float:
        return signal.stop

    def attach(self, trade: Trade, ctx: Context) -> list[Order]:
        return _fixed_rr_orders(trade, ctx, self.rr)


class ATRFixedR(_ExitRuleBase):
    """Stop at `atr_mult` Daily ATRs from entry (overrides the signal's stop), target `rr` R
    away using that same distance. Grid: atr_mult in {1.5, 2, 3} x rr in {2, 3}.
    """

    def __init__(
        self, rr: float, atr_mult: float, *, time_stop_bars: int = 10, time_stop_min_r: float = 0.5
    ) -> None:
        super().__init__(time_stop_bars=time_stop_bars, time_stop_min_r=time_stop_min_r)
        self.rr = rr
        self.atr_mult = atr_mult
        self.name = f"atr_{_fmt(atr_mult)}_rr_{_fmt(rr)}"

    def initial_stop(self, signal: Signal, ctx: Context) -> float:
        atr = ctx.atr("1d")
        if math.isnan(atr):
            return signal.stop
        return signal.entry - signal.direction * self.atr_mult * atr

    def attach(self, trade: Trade, ctx: Context) -> list[Order]:
        return _fixed_rr_orders(trade, ctx, self.rr)


class Structure(_ExitRuleBase):
    """Target the signal's structural level if it clears 1.5R, else a fixed-R fallback."""

    def __init__(
        self, fallback_rr: float = 2, *, time_stop_bars: int = 10, time_stop_min_r: float = 0.5
    ) -> None:
        super().__init__(time_stop_bars=time_stop_bars, time_stop_min_r=time_stop_min_r)
        self.fallback_rr = fallback_rr
        self.name = f"structure_{_fmt(fallback_rr)}"

    def initial_stop(self, signal: Signal, ctx: Context) -> float:
        return signal.stop

    def attach(self, trade: Trade, ctx: Context) -> list[Order]:
        r_dist = _r_dist(trade)
        candidate = trade.target
        if candidate is not None and trade.direction * (candidate - trade.entry_fill.price) >= 1.5 * r_dist:
            target = candidate
        else:
            target = trade.entry_fill.price + trade.direction * self.fallback_rr * r_dist
        return _stop_and_target(trade, ctx, target)


class Trailing(_ExitRuleBase):
    """No target: once price has moved `activate_r` R in favour, trail the stop `trail_atr`
    Daily ATRs behind the best price reached since entry. Grid: activate_r in {1, 2} x
    trail_atr in {1, 2}.
    """

    def __init__(
        self, activate_r: float, trail_atr: float, *, time_stop_bars: int = 10, time_stop_min_r: float = 0.5
    ) -> None:
        super().__init__(time_stop_bars=time_stop_bars, time_stop_min_r=time_stop_min_r)
        self.activate_r = activate_r
        self.trail_atr = trail_atr
        self.name = f"trail_{_fmt(activate_r)}_{_fmt(trail_atr)}"

    def initial_stop(self, signal: Signal, ctx: Context) -> float:
        return signal.stop

    def attach(self, trade: Trade, ctx: Context) -> list[Order]:
        return [_stop_order(trade, ctx, "a")]

    def _manage(self, trade: Trade, ctx: Context) -> list[Order]:
        atr = ctx.atr("1d")
        if math.isnan(atr):
            return []
        r_dist = _r_dist(trade)
        threshold = trade.entry_fill.price + trade.direction * self.activate_r * r_dist
        ohlcv = ctx.bars("4h")
        # `opened_bar` unresolvable (trimmed out of the retained window, or not yet
        # happened): do nothing rather than guess a start point. In practice this never
        # fires for a trade still open, because `max_bars` is kept above any run's holding
        # period (see the module docstring).
        start = ctx.offset_of(trade.opened_bar)
        if start is None:
            return []
        # A non-None `start` is always a valid row index into `ohlcv` (see `offset_of`), so
        # `highs`/`lows` are never actually empty here; the `size == 0` guard is defensive
        # only, folded into the same short-circuit as the threshold check so it does not
        # need its own test to reach full branch coverage (np.max/np.min is still computed
        # only once per call, via the walrus assignment).
        if trade.direction == 1:
            highs = ohlcv[start:, 1]
            if highs.size == 0 or (extreme := float(np.max(highs))) < threshold:
                return []
            rounded = round_to_tick(extreme - self.trail_atr * atr, trade.instrument.tick_size)
            if rounded <= trade.stop:
                return []
        else:
            lows = ohlcv[start:, 2]
            if lows.size == 0 or (extreme := float(np.min(lows))) > threshold:
                return []
            rounded = round_to_tick(extreme + self.trail_atr * atr, trade.instrument.tick_size)
            if rounded >= trade.stop:
                return []
        return [_stop_order(trade, ctx, "m", price=rounded)]


class Partial(_ExitRuleBase):
    """Scale out `scale_pct` of the position at `scale_r` R, move the rest to breakeven once
    that partial fills, then hand the remainder to `runner` (FixedR or Trailing). Grid: the
    two runners named in the tournament, `Partial(1, 0.5, FixedR(3))` and
    `Partial(1, 0.5, Trailing(1, 2))`.

    If the position is too small for the split — the partial leg or the runner's remainder
    rounds to <= 0 at 6dp — the partial leg is skipped entirely and the trade is attached and
    managed exactly as `runner` alone would (no partial ever fills, so `_manage` naturally
    delegates the same way it does after a real breakeven).
    """

    def __init__(
        self,
        scale_r: float,
        scale_pct: float,
        runner: FixedR | Trailing,
        *,
        time_stop_bars: int = 10,
        time_stop_min_r: float = 0.5,
    ) -> None:
        super().__init__(time_stop_bars=time_stop_bars, time_stop_min_r=time_stop_min_r)
        self.scale_r = scale_r
        self.scale_pct = scale_pct
        self.runner = runner
        self.name = f"partial_{_fmt(scale_r)}_{_fmt(scale_pct)}_{runner.name}"

    def initial_stop(self, signal: Signal, ctx: Context) -> float:
        return self.runner.initial_stop(signal, ctx)

    def attach(self, trade: Trade, ctx: Context) -> list[Order]:
        r_dist = _r_dist(trade)
        full_qty = trade.entry_fill.qty
        partial_qty = round(full_qty * self.scale_pct, 6)
        runner_qty = round(full_qty - partial_qty, 6)
        if partial_qty <= 0 or runner_qty <= 0:
            return self.runner.attach(trade, ctx)
        partial_price = trade.entry_fill.price + trade.direction * self.scale_r * r_dist
        orders = [
            _stop_order(trade, ctx, "a", qty=full_qty),
            _limit_order(trade, ctx, partial_price, "partial", partial_qty, "a"),
        ]
        if isinstance(self.runner, FixedR):
            target = trade.entry_fill.price + trade.direction * self.runner.rr * r_dist
            orders.append(_limit_order(trade, ctx, target, "target", runner_qty, "a"))
        return orders

    def _manage(self, trade: Trade, ctx: Context) -> list[Order]:
        has_partial_fill = any(leg.leg == "partial" for leg in trade.legs)
        if has_partial_fill:
            breakeven = round_to_tick(trade.entry_fill.price, trade.instrument.tick_size)
            if trade.direction * (breakeven - trade.stop) > 0:
                return [_stop_order(trade, ctx, "m", price=breakeven, qty=remaining_qty(trade))]
        if isinstance(self.runner, Trailing):
            return self.runner._manage(trade, ctx)
        return []


EXIT_GRID: tuple[ExitRule, ...] = (
    FixedR(1.5),
    FixedR(2),
    FixedR(3),
    *(ATRFixedR(rr, mult) for mult in (1.5, 2, 3) for rr in (2, 3)),
    Structure(2),
    *(Trailing(activate, trail) for activate in (1, 2) for trail in (1, 2)),
    Partial(1, 0.5, FixedR(3)),
    Partial(1, 0.5, Trailing(1, 2)),
)
"""The 16 tournament exit variants, in grid order. Every `name` is unique and stable."""
