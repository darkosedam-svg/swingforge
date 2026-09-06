"""Unit tests for `strategies/baseline.py` — the seeded random control entry."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from swingforge.core.context import Context
from swingforge.core.types import Bar, Instrument, Signal
from swingforge.strategies.baseline import Baseline

INSTRUMENT = Instrument(
    venue="hyperliquid",
    symbol="BTC",
    tick_size=Decimal("0.01"),
    contract_multiplier=Decimal("1"),
    quote_ccy="USD",
    session_profile="perp",
)
START = datetime(2026, 1, 1, tzinfo=UTC)


def _daily_bar(index: int, close: float = 100.0, rng: float = 4.0) -> Bar:
    return Bar(
        instrument=INSTRUMENT,
        tf="1d",
        ts_open=START + timedelta(hours=24 * index),
        open=close,
        high=close + rng / 2,
        low=close - rng / 2,
        close=close,
        volume=1.0,
    )


def _h4_bar(index: int, close: float) -> Bar:
    return Bar(
        instrument=INSTRUMENT,
        tf="4h",
        ts_open=START + timedelta(hours=4 * index),
        open=close,
        high=close,
        low=close,
        close=close,
        volume=1.0,
    )


def _seeded_context(n_daily: int = 20) -> Context:
    ctx = Context(INSTRUMENT)
    for i in range(n_daily):
        ctx.push(_daily_bar(i))
    return ctx


def _run(seed: int, target: float, n_bars: int) -> list[Signal | None]:
    ctx = _seeded_context()
    strategy = Baseline(target_trades_per_1000_bars=target, seed=seed)
    signals: list[Signal | None] = []
    for i in range(n_bars):
        # Small drift so a close-based entry price varies bar to bar.
        ctx.push(_h4_bar(i, 100.0 + math.sin(i / 17.0) * 3.0))
        signals.append(strategy.on_bar(ctx))
    return signals


def test_no_bars_yet_returns_none() -> None:
    ctx = Context(INSTRUMENT)
    strategy = Baseline(target_trades_per_1000_bars=50, seed=1)
    assert strategy.on_bar(ctx) is None


def test_missing_atr_returns_none_even_if_fire_would_trigger() -> None:
    # Only 1 daily bar and 1 four-hour bar: both ATR(14) reads are NaN, so no signal is
    # possible no matter what the RNG draws - a very high target guarantees a "fire" draw.
    ctx = Context(INSTRUMENT)
    ctx.push(_daily_bar(0))
    ctx.push(_h4_bar(0, 100.0))
    strategy = Baseline(target_trades_per_1000_bars=1000.0, seed=1)
    assert strategy.on_bar(ctx) is None


def test_signal_shape_when_it_fires() -> None:
    ctx = _seeded_context()
    strategy = Baseline(target_trades_per_1000_bars=1000.0, seed=7)  # always fires
    ctx.push(_h4_bar(0, 100.0))
    signal = strategy.on_bar(ctx)
    assert signal is not None
    assert signal.tag == "baseline"
    assert signal.structure_target is None
    assert signal.expires_in_bars == 3
    assert signal.direction in (1, -1)
    # stop is 1.5 * ATR(14, 1d) away from entry, on the correct side.
    atr = ctx.atr("1d")
    expected_stop = signal.entry - signal.direction * 1.5 * atr
    assert signal.stop == pytest.approx(expected_stop)


def test_frequency_matches_target_within_10_percent_over_10k_bars() -> None:
    target = 50.0  # 5% per-bar fire probability
    n_bars = 10_000
    signals = _run(seed=123, target=target, n_bars=n_bars)
    fired = sum(1 for s in signals if s is not None)
    expected = target / 1000.0 * n_bars
    assert abs(fired - expected) <= expected * 0.10


def test_long_share_is_roughly_half() -> None:
    target = 200.0  # plenty of fires for a stable ratio
    n_bars = 10_000
    signals = _run(seed=321, target=target, n_bars=n_bars)
    fired = [s for s in signals if s is not None]
    assert len(fired) > 100
    longs = sum(1 for s in fired if s.direction == 1)
    long_share = longs / len(fired)
    assert 0.45 <= long_share <= 0.55


def test_same_seed_same_bars_gives_identical_sequence() -> None:
    a = _run(seed=99, target=80.0, n_bars=500)
    b = _run(seed=99, target=80.0, n_bars=500)
    assert a == b


def test_different_seed_gives_a_different_sequence() -> None:
    a = _run(seed=1, target=200.0, n_bars=500)
    b = _run(seed=2, target=200.0, n_bars=500)
    assert a != b


def test_reset_reproduces_the_original_sequence() -> None:
    ctx = _seeded_context()
    strategy = Baseline(target_trades_per_1000_bars=200.0, seed=55)
    first: list[Signal | None] = []
    for i in range(300):
        ctx.push(_h4_bar(i, 100.0 + math.sin(i / 13.0) * 2.0))
        first.append(strategy.on_bar(ctx))

    strategy.reset()
    ctx2 = _seeded_context()
    second: list[Signal | None] = []
    for i in range(300):
        ctx2.push(_h4_bar(i, 100.0 + math.sin(i / 13.0) * 2.0))
        second.append(strategy.on_bar(ctx2))

    assert first == second


def test_direction_roll_is_independent_of_the_fire_outcome() -> None:
    """Both the fire roll and the direction roll are drawn every bar, whether or not the
    bar fires - the RNG stream position is a pure function of `(seed, bars presented)`, not
    of outcomes. Two strategies sharing a seed but firing at very different rates must
    therefore draw the identical direction for any bar where they both happen to fire."""
    n_bars = 4000

    def _run_directions(target: float) -> dict[int, int]:
        ctx = _seeded_context()
        strategy = Baseline(target_trades_per_1000_bars=target, seed=42)
        directions: dict[int, int] = {}
        for i in range(n_bars):
            ctx.push(_h4_bar(i, 100.0 + math.sin(i / 11.0) * 3.0))
            signal = strategy.on_bar(ctx)
            if signal is not None:
                directions[i] = signal.direction
        return directions

    always_fires = _run_directions(1000.0)
    rarely_fires = _run_directions(20.0)  # ~2% per bar -> ~80 expected fires
    assert len(always_fires) == n_bars
    assert 0 < len(rarely_fires) < n_bars
    for bar_index, direction in rarely_fires.items():
        assert direction == always_fires[bar_index]


def test_rejected_signals_counter_increments_when_the_stop_collapses_onto_entry() -> None:
    """A flat Daily series (ATR == 0, not NaN) makes stop == entry, which the Signal
    validator rejects since the stop is no longer strictly on the wrong side."""
    ctx = Context(INSTRUMENT)
    for i in range(20):
        ctx.push(_daily_bar(i, close=100.0, rng=0.0))
    strategy = Baseline(target_trades_per_1000_bars=1000.0, seed=7)  # always fires
    ctx.push(_h4_bar(0, 100.0))
    assert strategy.rejected_signals == 0
    assert strategy.on_bar(ctx) is None
    assert strategy.rejected_signals == 1

    strategy.reset()
    assert strategy.rejected_signals == 0
