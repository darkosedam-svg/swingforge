"""Equity accounting and position sizing (design spec section 4 last paragraph: risk 1%).

`Portfolio` owns the run's equity and closed-trade ledger. It does not touch orders or
fills directly — the engine hands it a stop distance to size against and, later, a
finished `Trade` to record. Sizing and pnl math live here so the engine stays a pure
orchestrator of the call sequence documented in `swingforge.strategies.base`.
"""

from __future__ import annotations

from datetime import datetime
from decimal import ROUND_FLOOR, Decimal
from typing import Literal

from swingforge.core.types import Fill, Instrument, Trade

__all__ = ["QTY_EPS", "QTY_STEP", "Portfolio"]

QTY_STEP = 1e-6
"""Quantity floor/step used by `Portfolio.size`.

`Instrument` (contract v2) carries no per-symbol lot-size field, so there is no exchange
lot step to floor against. `QTY_STEP` is a fixed, conservative stand-in — small enough that
it never binds for the instruments this system trades. Revisit if a later work unit adds a
real lot-size field to `Instrument`.
"""

QTY_EPS = QTY_STEP / 2
"""Epsilon for float-safe quantity comparisons against a sized/remaining qty.

Half of `QTY_STEP` — small enough that it never masks a real off-by-one-step error, but
enough to absorb float round-trip noise when comparing a fill's `qty` against, e.g., a
trade's `remaining_qty`.
"""


class Portfolio:
    """Equity, the sizing formula, and the closed-trade ledger for one engine run."""

    def __init__(self, initial_equity: float) -> None:
        self.equity: float = initial_equity
        self.trades: list[Trade] = []
        self.equity_curve: list[tuple[datetime, float]] = []

    def size(
        self,
        entry: float,
        stop: float,
        instrument: Instrument,
        risk_pct: float,
    ) -> float:
        """Position size in instrument units, floored to a multiple of `QTY_STEP`.

        `risk = self.equity * risk_pct`; `qty = risk / |entry - stop| / contract_multiplier`.
        Sizes against the portfolio's own current equity — there is no separate `equity`
        argument, so a caller cannot size against a stale or foreign value. Returns `0.0`
        when the stop distance is zero (nothing to size against) rather than raising or
        dividing by zero — signals with a degenerate stop simply size to zero and are
        dropped by the caller.
        """
        distance = abs(entry - stop)
        if distance == 0:
            return 0.0
        risk = self.equity * risk_pct
        raw_qty = risk / distance / float(instrument.contract_multiplier)
        if raw_qty <= 0:
            return 0.0
        step = Decimal(str(QTY_STEP))
        steps = (Decimal(str(raw_qty)) / step).to_integral_value(rounding=ROUND_FLOOR)
        return float(steps * step)

    def risk_money(self, entry: float, stop: float, qty: float, instrument: Instrument) -> float:
        """Money at risk for one trade: `|entry - stop| * qty * contract_multiplier`.

        This is the value stored as `Trade.risk_r` — every R figure on the trade is
        `pnl / risk_r`.
        """
        return abs(entry - stop) * qty * float(instrument.contract_multiplier)

    def realized_pnl(
        self,
        entry_fill: Fill,
        legs: tuple[Fill, ...],
        direction: Literal[1, -1],
        instrument: Instrument,
    ) -> float:
        """Money pnl for a closed trade: gross move on every exit leg, minus all costs.

        `realized_r = pnl / risk_r` is the caller's job (the engine has `risk_r` on hand
        as `Trade.risk_r` already); this method only ever deals in money.
        """
        multiplier = float(instrument.contract_multiplier)
        gross = sum(direction * (leg.price - entry_fill.price) * leg.qty * multiplier for leg in legs)
        costs = entry_fill.cost.total + sum(leg.cost.total for leg in legs)
        return gross - costs

    def record(self, trade: Trade, pnl: float) -> None:
        """Apply a closed trade's exact money pnl to equity and append it to the ledger.

        `equity += pnl`. `pnl` is the caller's exact money result (typically straight from
        `realized_pnl`) rather than `trade.realized_r * trade.risk_r`: `realized_r` is
        itself `pnl / risk_r`, so re-multiplying it back out round-trips through that
        division and can lose precision. `realized_r` stays on `Trade` purely as the
        reporting field. One point is appended to `equity_curve` at the trade's closing
        fill timestamp (the last leg's `ts` — a recorded trade always has at least one exit
        leg).
        """
        assert trade.realized_r is not None, "record() requires a closed trade with realized_r set"
        self.equity += pnl
        self.trades.append(trade)
        closing_ts = trade.legs[-1].ts if trade.legs else trade.entry_fill.ts
        self.equity_curve.append((closing_ts, self.equity))
