"""The random control entry (design spec section 4).

Fires with a fixed per-bar probability; direction is drawn from the same seeded generator.
Stop distance is deterministic — `stop_atr_mult` Daily ATRs from entry, on the side the
drawn direction implies — never itself randomly drawn. Both the fire roll and the direction
roll are drawn unconditionally on every `on_bar` call, whether or not the bar ends up firing,
so the RNG stream position is a pure function of `(seed, bars presented)` and never of
outcomes; `on_bar` only ever runs while flat (the `Strategy` protocol's contract), so "bars
presented" means every closed 4H bar seen while flat. The same replay always produces the
same trades.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from pydantic import ValidationError

from swingforge.core.context import Context
from swingforge.core.types import Signal, round_to_tick

__all__ = ["Baseline"]


class Baseline:
    """Random 4H entries, frequency-matched via `target_trades_per_1000_bars`. The control."""

    name = "baseline"

    def __init__(
        self,
        target_trades_per_1000_bars: float,
        seed: int,
        stop_atr_mult: float = 1.5,
        expires_in_bars: int = 3,
    ) -> None:
        self.target_trades_per_1000_bars = target_trades_per_1000_bars
        self.seed = seed
        self.stop_atr_mult = stop_atr_mult
        self.expires_in_bars = expires_in_bars
        self._rng = np.random.default_rng(seed)
        self.rejected_signals = 0

    def reset(self) -> None:
        """Clear per-instance state: re-seeds the generator from scratch."""
        self._rng = np.random.default_rng(self.seed)
        self.rejected_signals = 0

    def on_bar(self, ctx: Context) -> Signal | None:
        bar = ctx.last("4h")
        if bar is None:
            return None
        # Both draws happen every bar, fire or not, so the RNG stream position never
        # depends on outcomes - see the module docstring.
        fire_roll = self._rng.random()
        direction_roll = self._rng.random()
        probability = self.target_trades_per_1000_bars / 1000.0
        if fire_roll >= probability:
            return None
        direction: Literal[1, -1] = 1 if direction_roll < 0.5 else -1

        atr = ctx.atr("1d")
        if math.isnan(atr):
            atr = ctx.atr("4h")
        if math.isnan(atr):
            return None

        entry = bar.close
        stop = entry - direction * self.stop_atr_mult * atr
        tick = ctx.instrument.tick_size
        try:
            return Signal(
                direction=direction,
                entry=round_to_tick(entry, tick),
                stop=round_to_tick(stop, tick),
                structure_target=None,
                tag="baseline",
                expires_in_bars=self.expires_in_bars,
            )
        except ValidationError:
            self.rejected_signals += 1
            return None
