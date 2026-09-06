"""Regime tagger: Daily ADX trend/range crossed with the realized-volatility tercile.

Fixtures are built as Daily bar sequences and pushed through a real `Context`, so the ADX
these tests see is the same Wilder ADX the engine will read - the tagger is never handed a
pre-cooked indicator value.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import numpy as np
import pytest

from swingforge.core.context import Context
from swingforge.core.types import Bar, Instrument
from swingforge.lab.regime import REGIMES, realized_vol_series, tag, vol_tercile

EUR = Instrument(
    venue="oanda",
    symbol="EUR_USD",
    tick_size=Decimal("0.00001"),
    contract_multiplier=Decimal("1"),
    quote_ccy="USD",
    session_profile="fx",
)
START = datetime(2026, 1, 1, tzinfo=UTC)


def daily_context(closes: list[float]) -> Context:
    """A context holding `closes` as Daily bars, each spanning from the previous close."""
    ctx = Context(EUR)
    previous = closes[0]
    for index, close in enumerate(closes):
        ctx.push(
            Bar(
                instrument=EUR,
                tf="1d",
                ts_open=START + timedelta(days=index),
                open=previous,
                high=max(previous, close),
                low=min(previous, close),
                close=close,
                volume=1.0,
            )
        )
        previous = close
    return ctx


def uptrend(n: int = 80, step: float = 0.01) -> list[float]:
    """A clean monotone advance: every bar makes a new high and a new low above the last."""
    return [1.0 * (1.0 + step) ** i for i in range(n)]


def oscillation(n: int = 80, seed: int = 1) -> list[float]:
    """A noisy six-day cycle around a fixed level: +DM and -DM balance, so ADX stays low.

    A pure two-level zigzag will not do - its highs and lows are constant after the first
    bar, so Wilder sees one directional move and then nothing, and the ADX pins at 100.
    """
    rng = np.random.default_rng(seed)
    return [1.0 + 0.01 * math.sin(2.0 * math.pi * i / 6.0) + float(rng.normal(0.0, 0.002)) for i in range(n)]


def walk(sigmas: list[float], seed: int) -> list[float]:
    """A log-normal walk opening at 1.0 whose per-bar volatility follows `sigmas`."""
    rng = np.random.default_rng(seed)
    closes = [1.0]
    for sigma in sigmas:
        closes.append(closes[-1] * math.exp(float(rng.normal(0.0, sigma))))
    return closes


def expected_tercile(closes: list[float], window: int) -> str:
    """The tercile written out longhand from its definition, touching no `lab` code."""
    logs = [math.log(b / a) for a, b in zip(closes[:-1], closes[1:], strict=True)]
    vols = [float(np.std(logs[i : i + window], ddof=1)) for i in range(len(logs) - window + 1)]
    lower, upper = np.percentile(vols, [100.0 / 3.0, 200.0 / 3.0])
    if vols[-1] < lower:
        return "lowvol"
    return "highvol" if vols[-1] > upper else "midvol"


# --- realized_vol_series ------------------------------------------------------


def test_realized_vol_series_is_the_rolling_std_of_daily_log_returns() -> None:
    closes = np.array([1.0, 1.02, 1.01, 1.05, 1.03, 1.06, 1.10])
    series = realized_vol_series(closes, 3)
    returns = np.diff(np.log(closes))
    expected = [float(np.std(returns[i : i + 3], ddof=1)) for i in range(len(returns) - 2)]
    assert series == pytest.approx(expected, rel=1e-12)


def test_realized_vol_series_has_one_point_per_full_window() -> None:
    # M closes give M-1 returns, which give M-window rolling windows.
    for m in (25, 61, 100):
        assert realized_vol_series(np.array(uptrend(m)), 20).size == m - 20


def test_realized_vol_series_is_empty_when_there_are_too_few_closes() -> None:
    assert realized_vol_series(np.array([1.0, 1.1, 1.2]), 20).size == 0


def test_realized_vol_of_a_constant_growth_rate_is_zero() -> None:
    series = realized_vol_series(np.array(uptrend(40)), 20)
    assert series == pytest.approx(np.zeros(20), abs=1e-12)


def test_realized_vol_series_rejects_a_degenerate_window() -> None:
    for window in (0, 1, -3):
        with pytest.raises(ValueError):
            realized_vol_series(np.array(uptrend(40)), window)


def test_realized_vol_series_rejects_a_non_positive_close() -> None:
    # log(0) is -inf and log(-1) is NaN, either of which would poison the whole tercile
    # into never comparing True and so mislabel the regime `midvol` without a word.
    closes = uptrend(40)
    for bad in (0.0, -1.0):
        broken = [*closes[:-1], bad]
        with pytest.raises(ValueError):
            realized_vol_series(np.array(broken), 20)


# --- vol_tercile --------------------------------------------------------------


def test_vol_tercile_judges_the_last_value_against_the_whole_series() -> None:
    rising = np.arange(30, dtype=float)
    assert vol_tercile(rising) == "highvol"
    assert vol_tercile(rising[::-1].copy()) == "lowvol"
    assert vol_tercile(np.array([0.0, 10.0, 5.0])) == "midvol"


def test_vol_tercile_uses_the_thirds_of_the_expanding_history() -> None:
    series = np.arange(90, dtype=float)
    lower, upper = np.percentile(series, [100.0 / 3.0, 200.0 / 3.0])
    assert vol_tercile(np.append(series, lower - 1.0)) == "lowvol"
    assert vol_tercile(np.append(series, upper + 1.0)) == "highvol"
    assert vol_tercile(np.append(series, (lower + upper) / 2.0)) == "midvol"


def test_vol_tercile_of_a_constant_series_is_midvol() -> None:
    assert vol_tercile(np.full(30, 0.004)) == "midvol"
    assert vol_tercile(np.array([1.0])) == "midvol"


def test_vol_tercile_rejects_an_empty_series() -> None:
    with pytest.raises(ValueError):
        vol_tercile(np.array([]))


# --- tag ----------------------------------------------------------------------


def test_a_clean_daily_uptrend_is_tagged_trend() -> None:
    ctx = daily_context(uptrend(80))
    assert ctx.adx("1d", 14) > 25.0
    assert tag(ctx).startswith("trend_")


def test_an_oscillating_range_is_tagged_range() -> None:
    ctx = daily_context(oscillation(80))
    assert ctx.adx("1d", 14) <= 25.0
    assert tag(ctx).startswith("range_")


def test_a_flat_market_is_tagged_range_midvol() -> None:
    assert tag(daily_context([1.0] * 80)) == "range_midvol"


def test_an_undefined_adx_is_treated_as_range() -> None:
    ctx = daily_context(uptrend(10))  # below 2n+1 bars, so ADX is NaN
    assert math.isnan(ctx.adx("1d", 14))
    assert tag(ctx).startswith("range_")


def test_the_adx_threshold_is_configurable() -> None:
    ctx = daily_context(uptrend(80))
    assert tag(ctx, adx_threshold=1000.0).startswith("range_")
    assert tag(ctx, adx_threshold=0.0).startswith("trend_")


def test_the_trend_comparison_is_strict() -> None:
    # A flat market has no directional movement at all, so its ADX is exactly 0.0. At a
    # threshold of 0.0 it must still range: the rule is `>`, not `>=`.
    flat = daily_context([1.0] * 80)
    assert flat.adx("1d", 14) == 0.0
    assert tag(flat, adx_threshold=0.0).startswith("range_")


def test_a_calm_tail_after_a_wild_history_is_lowvol() -> None:
    rng = np.random.default_rng(2)
    wild = [1.0]
    for shock in rng.normal(0.0, 0.05, 60):
        wild.append(wild[-1] * math.exp(float(shock)))
    calm = [wild[-1] * (1.0 + 0.0001 * i) for i in range(1, 31)]
    ctx = daily_context(wild + calm)
    series = realized_vol_series(ctx.bars("1d")[:, 3], 20)
    assert series[-1] < np.percentile(series, 100.0 / 3.0)
    assert tag(ctx).endswith("_lowvol")


def test_a_wild_tail_after_a_calm_history_is_highvol() -> None:
    rng = np.random.default_rng(3)
    calm = [1.0 * (1.0 + 0.0001 * i) for i in range(70)]
    wild = [calm[-1]]
    for shock in rng.normal(0.0, 0.08, 20):
        wild.append(wild[-1] * math.exp(float(shock)))
    ctx = daily_context(calm + wild[1:])
    series = realized_vol_series(ctx.bars("1d")[:, 3], 20)
    assert series[-1] > np.percentile(series, 200.0 / 3.0)
    assert tag(ctx).endswith("_highvol")


def test_too_little_daily_history_falls_back_to_midvol() -> None:
    for bars in (1, 21, 59):
        assert tag(daily_context(uptrend(bars))).endswith("_midvol")


def test_the_history_floor_is_inclusive_and_configurable() -> None:
    # A violent history with a calm last window, so the honest answer is `lowvol` and a
    # neutral `midvol` can only mean the floor fired.
    closes = walk([0.03] * 39 + [0.0005] * 20, seed=7)
    assert len(closes) == 60
    assert expected_tercile(closes, 20) == "lowvol"
    # Exactly `min_history` bars is enough: the floor rejects *fewer* than min_history.
    assert tag(daily_context(closes), min_history=60).endswith("_lowvol")
    assert tag(daily_context(closes[:-1]), min_history=60).endswith("_midvol")
    # Raising the floor above the history available forces the neutral answer back.
    assert tag(daily_context(closes), min_history=100).endswith("_midvol")


def test_the_vol_window_is_configurable() -> None:
    # Calm, then violent, then dead calm for five bars: a 5-day window sees only the calm
    # tail while a 40-day window still straddles the violence, so the two disagree.
    closes = walk([0.002] * 49 + [0.05] * 25 + [1e-6] * 5, seed=11)
    ctx = daily_context(closes)
    assert expected_tercile(closes, 5) == "lowvol"
    assert expected_tercile(closes, 40) == "highvol"
    assert tag(ctx, vol_window=5).endswith("_lowvol")
    assert tag(ctx, vol_window=40).endswith("_highvol")


def test_every_random_walk_gets_one_of_the_six_labels() -> None:
    seen = set()
    for seed in range(40):
        rng = np.random.default_rng(seed)
        drift = float(rng.uniform(-0.01, 0.01))
        vol = float(rng.uniform(0.001, 0.05))
        closes = [1.0]
        for shock in rng.normal(drift, vol, 120):
            closes.append(closes[-1] * math.exp(float(shock)))
        label = tag(daily_context(closes))
        assert label in REGIMES
        seen.add(label)
    assert len(seen) > 1  # the tagger actually discriminates rather than answering one label


def test_regimes_lists_the_six_labels_the_spec_names() -> None:
    assert set(REGIMES) == {
        f"{structure}_{vol}" for structure in ("trend", "range") for vol in ("lowvol", "midvol", "highvol")
    }
