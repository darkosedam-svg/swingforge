"""MAE/MFE excursion analysis: is the exit leaving money on the table, or cutting winners?

Design spec section 6 asks for MAE/MFE in R per trade, exit efficiency (captured / MFE),
a stopped-out-of-winner rate and a median exit efficiency per config — and for the same
measurement on *raw signals with no exit*, which is what `HoldBars` and
`raw_signal_excursion` are for. The research note's test protocol is the reason: the
winners' MAE distribution sets the minimum non-noise stop and the MFE distribution sets a
realistic target, and both questions are downstream of an entry edge that has to be shown
against a baseline first.

The engine already records `Trade.mae_r`/`mfe_r` as it runs. `mae_mfe` recomputes them
from the 4H bars so a run can be cross-checked, and so a trade list read back out of the
store can be re-measured without a replay.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from swingforge.adapters.replay import ReplaySource
from swingforge.core.context import Context
from swingforge.core.types import Bar, Order, Signal, Trade, round_to_tick
from swingforge.lab.regime import tag
from swingforge.lab.tournament import (
    Config,
    EntryFactory,
    SessionFactory,
    run_config,
)
from swingforge.strategies.exits import remaining_qty

__all__ = [
    "ExcursionSummary",
    "HoldBars",
    "excursion_summary",
    "mae_mfe",
    "median_exit_efficiency",
    "raw_signal_excursion",
    "stopped_out_of_winner_rate",
]

_WINNER_MFE_R = 1.0
"""How far in profit a loser must have travelled to count as "stopped out of a winner"."""


def _risk_distance(trade: Trade) -> float:
    """Price distance per 1R, exactly as the engine fixed it at entry.

    The engine holds `risk_distance = |entry price - initial stop|` and never updates it,
    while `Trade.stop` moves whenever the exit rule replaces the stop leg — so the closed
    trade cannot be read for it directly. `risk_r` is `|entry - initial stop| * qty * mult`,
    so dividing it back out recovers the same number for every trade the engine sized
    normally.
    """
    return trade.risk_r / (trade.entry_fill.qty * float(trade.instrument.contract_multiplier))


def mae_mfe(trade: Trade, bars: Sequence[Bar]) -> tuple[float, float]:
    """Recompute `(mae_r, mfe_r)` for `trade` from the 4H bars it was open across.

    `bars` is the run's 4H bar stream in order, indexed by `Context.bar_index` — so
    `bars[trade.opened_bar]` is the bar the entry filled on and `bars[trade.closed_bar]`
    the bar it closed on (the last bar, for a trade still open). Both ends are included,
    matching the engine, which folds the entry bar in when it opens the trade and the
    closing bar in before it freezes the `Trade`.

    Like the engine this uses each bar's full high/low rather than clipping to the fill
    price, so it can only overstate the excursion, never understate it — which keeps exit
    efficiency at or below 1. A trade whose risk distance is zero (a fill exactly on its
    stop) reports `(0.0, 0.0)`, again as the engine does.
    """
    last = len(bars) - 1 if trade.closed_bar is None else trade.closed_bar
    if not bars or trade.opened_bar < 0 or last >= len(bars) or last < trade.opened_bar:
        raise ValueError(
            f"trade {trade.id!r} spans bar index {trade.opened_bar}..{last}, outside the "
            f"{len(bars)} bars given"
        )
    distance = _risk_distance(trade)
    if distance <= 0.0 or not math.isfinite(distance):
        return 0.0, 0.0

    entry_price = trade.entry_fill.price
    mae = 0.0
    mfe = 0.0
    for bar in bars[trade.opened_bar : last + 1]:
        if trade.direction == 1:
            adverse, favourable = entry_price - bar.low, bar.high - entry_price
        else:
            adverse, favourable = bar.high - entry_price, entry_price - bar.low
        mae = max(mae, max(0.0, adverse) / distance)
        mfe = max(mfe, max(0.0, favourable) / distance)
    return mae, mfe


def stopped_out_of_winner_rate(trades: Sequence[Trade]) -> float:
    """Share of losing trades that had already been at least +1R in the open.

    The research note's threshold for widening a stop: a losing trade that reached +1R was
    a winner the exit gave back, not a bad entry. `nan` when there were no losers at all,
    because the question then has no denominator rather than an answer of zero.
    """
    losers = [t for t in trades if t.realized_r is not None and t.realized_r < 0.0]
    if not losers:
        return math.nan
    return sum(1 for t in losers if t.mfe_r >= _WINNER_MFE_R) / len(losers)


def median_exit_efficiency(trades: Sequence[Trade]) -> float:
    """Median `realized_r / mfe_r` across winning trades, clipped to `[0, 1]`.

    Winners only: a loser's captured share of a favourable excursion is negative and says
    nothing about how well the exit converted an available move. Clipped because MFE comes
    from bar extremes, which a fill can occasionally beat. `nan` when nothing qualifies.
    """
    ratios = [
        min(1.0, max(0.0, t.realized_r / t.mfe_r))
        for t in trades
        if t.realized_r is not None and t.realized_r > 0.0 and t.mfe_r > 0.0
    ]
    return float(np.median(ratios)) if ratios else math.nan


@dataclass(frozen=True)
class ExcursionSummary:
    """One config's excursion profile, as the report prints it."""

    n: int
    stopped_out_of_winner_rate: float
    median_exit_efficiency: float
    median_mae_r: float
    median_mfe_r: float
    mfe_p75: float


def excursion_summary(trades: Sequence[Trade]) -> ExcursionSummary:
    """Summarise a trade list's excursions; every statistic is `nan` for an empty list."""
    mae = [t.mae_r for t in trades]
    mfe = [t.mfe_r for t in trades]
    return ExcursionSummary(
        n=len(trades),
        stopped_out_of_winner_rate=stopped_out_of_winner_rate(trades),
        median_exit_efficiency=median_exit_efficiency(trades),
        median_mae_r=float(np.median(mae)) if mae else math.nan,
        median_mfe_r=float(np.median(mfe)) if mfe else math.nan,
        mfe_p75=float(np.percentile(mfe, 75)) if mfe else math.nan,
    )


# --- the raw-signal excursion ------------------------------------------------------


class HoldBars:
    """An `ExitRule` that manages nothing: the signal's stop, then out after `bars` bars.

    This is the control the spec's "also run on raw signals with no exit" calls for. It
    keeps the invalidation stop (without one the trade has no R denominator and the broker
    has no leg to resolve) and otherwise lets the trade run its full holding period, so the
    MAE/MFE it produces describe the *entry*, not the exit that was layered on top of it.

    Deliberately not part of `EXIT_GRID`: it is a measuring instrument, not a candidate.
    Unlike every grid rule it has no time-stop dead band — it always closes on schedule,
    so the holding period is the same for every trade and the excursions are comparable.
    """

    def __init__(self, bars: int = 20) -> None:
        if bars < 0:
            raise ValueError("bars must be >= 0")
        self.bars = bars
        self.name = f"hold_{bars}"

    def initial_stop(self, signal: Signal, ctx: Context) -> float:
        return signal.stop

    def attach(self, trade: Trade, ctx: Context) -> list[Order]:
        return [self._stop_order(trade, ctx)]

    def on_bar(self, trade: Trade, ctx: Context) -> list[Order]:
        if ctx.bar_index - trade.opened_bar < self.bars:
            return []
        return [self._market_order(trade, ctx)]

    def _stop_order(self, trade: Trade, ctx: Context) -> Order:
        return Order(
            id=f"{trade.id}:stop:{ctx.bar_index}:a",
            instrument=trade.instrument,
            direction=-trade.direction,
            qty=remaining_qty(trade),
            kind="stop",
            price=round_to_tick(trade.stop, trade.instrument.tick_size),
            expires_at_bar=None,
            leg="stop",
            trade_id=trade.id,
        )

    def _market_order(self, trade: Trade, ctx: Context) -> Order:
        return Order(
            id=f"{trade.id}:time:{ctx.bar_index}:m",
            instrument=trade.instrument,
            direction=-trade.direction,
            qty=remaining_qty(trade),
            kind="market",
            price=None,
            expires_at_bar=None,
            leg="time",
            trade_id=trade.id,
        )


def raw_signal_excursion(
    config: Config,
    source: ReplaySource,
    *,
    entry_factory: EntryFactory,
    session_factory: SessionFactory,
    start: datetime,
    end: datetime,
    hold_bars: int = 20,
    seed: int = 0,
    regime_tagger: Callable[[Context], str] = tag,
) -> ExcursionSummary:
    """Replay `config`'s *entry* with `HoldBars` and summarise the excursions it produced.

    `config.exit` is ignored: the point is to see whether the entry has any excursion edge
    at all before crediting one of the 16 exits with it. Everything else — instrument,
    session, window, warmup, the `start` entry gate — is exactly what `run_config` does for
    a graded config, so the two are comparable trade for trade. That includes `seed`, which
    must be the tournament's own: `run_config` derives the entry's seed from it and the
    config id, so passing the same one puts a random entry on exactly the same signals it
    took in the graded run. Costs stay at `run_config`'s frictionless default: this measures
    how far price travelled, not what the round trip would have netted.
    """
    run = run_config(
        config,
        source,
        entry_factory=entry_factory,
        session_factory=session_factory,
        regime_tagger=regime_tagger,
        start=start,
        end=end,
        seed=seed,
        exit_override=HoldBars(hold_bars),
    )
    return excursion_summary(run.trades)
