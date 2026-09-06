"""Behaviour tests for the exit fill resolver (design spec section 5).

Every case fixes the exact touch, gap-through and sequencing semantics the paper broker
depends on, so a change in resolver policy has to change a test here first.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Literal

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from swingforge.adapters.base import ExitResolver
from swingforge.core.fills import NO_EXITS, FillResolver
from swingforge.core.types import Bar, Instrument, Resolution

BTC = Instrument(
    venue="hyperliquid",
    symbol="BTC",
    tick_size=Decimal("0.5"),
    contract_multiplier=Decimal("1"),
    quote_ccy="USD",
    session_profile="perp",
)
TS = datetime(2026, 1, 2, 4, 0, tzinfo=UTC)


def bar4h(open_: float, high: float, low: float, close: float, subbars: tuple[Bar, ...] = ()) -> Bar:
    """A 4H execution bar, optionally carrying its 1H subbars."""
    return Bar(
        instrument=BTC,
        tf="4h",
        ts_open=TS,
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=1.0,
        subbars=subbars,
    )


def sub1h(hour: int, open_: float, high: float, low: float, close: float) -> Bar:
    """The 1H bar starting ``hour`` hours into the 4H bar above."""
    return Bar(
        instrument=BTC,
        tf="1h",
        ts_open=TS + timedelta(hours=hour),
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=1.0,
    )


def summarise(resolution: Resolution) -> list[tuple[str, int | None, float]]:
    """``(leg, target_index, price)`` per event, in occurrence order."""
    return [(e.leg, e.target_index, e.price) for e in resolution.events]


# A long trade whose stop (95) and target (105) both sit inside the 4H range.
LONG_STOP = 95.0
LONG_TARGETS = [105.0]
LONG_BAR_ARGS = (100.0, 106.0, 94.0, 101.0)

# The short mirror: stop 105, target 95, both inside the same range.
SHORT_STOP = 105.0
SHORT_TARGETS = [95.0]
SHORT_BAR_ARGS = (100.0, 106.0, 94.0, 99.0)


def test_subbars_order_a_target_before_the_stop() -> None:
    bar = bar4h(
        *LONG_BAR_ARGS,
        subbars=(
            sub1h(0, 100.0, 105.5, 99.5, 105.0),  # tags 105, never reaches 95
            sub1h(1, 105.0, 105.5, 94.0, 95.0),  # would have stopped out, but the trade is closed
        ),
    )

    result = FillResolver().resolve(bar, LONG_STOP, LONG_TARGETS, 1)

    assert summarise(result) == [("target", 0, 105.0)]
    assert result.mode == "subbars"


def test_subbars_order_the_stop_before_a_target() -> None:
    bar = bar4h(
        *LONG_BAR_ARGS,
        subbars=(
            sub1h(0, 100.0, 100.5, 94.0, 95.0),  # tags 95 first
            sub1h(1, 95.0, 105.5, 95.0, 105.0),
        ),
    )

    result = FillResolver().resolve(bar, LONG_STOP, LONG_TARGETS, 1)

    assert summarise(result) == [("stop", None, 95.0)]
    assert result.mode == "subbars"


def test_without_subbars_the_stop_is_assumed_first() -> None:
    bar = bar4h(*LONG_BAR_ARGS)

    result = FillResolver().resolve(bar, LONG_STOP, LONG_TARGETS, 1)

    assert summarise(result) == [("stop", None, 95.0)]
    assert result.mode == "pessimistic"


def test_short_subbars_order_a_target_before_the_stop() -> None:
    bar = bar4h(
        *SHORT_BAR_ARGS,
        subbars=(
            sub1h(0, 100.0, 100.5, 94.5, 95.0),  # tags 95, never reaches 105
            sub1h(1, 95.0, 105.5, 95.0, 105.0),
        ),
    )

    result = FillResolver().resolve(bar, SHORT_STOP, SHORT_TARGETS, -1)

    assert summarise(result) == [("target", 0, 95.0)]
    assert result.mode == "subbars"


def test_short_subbars_order_the_stop_before_a_target() -> None:
    bar = bar4h(
        *SHORT_BAR_ARGS,
        subbars=(
            sub1h(0, 100.0, 105.5, 99.5, 105.0),  # tags 105 first
            sub1h(1, 105.0, 105.5, 94.0, 95.0),
        ),
    )

    result = FillResolver().resolve(bar, SHORT_STOP, SHORT_TARGETS, -1)

    assert summarise(result) == [("stop", None, 105.0)]
    assert result.mode == "subbars"


def test_short_without_subbars_the_stop_is_assumed_first() -> None:
    bar = bar4h(*SHORT_BAR_ARGS)

    result = FillResolver().resolve(bar, SHORT_STOP, SHORT_TARGETS, -1)

    assert summarise(result) == [("stop", None, 105.0)]
    assert result.mode == "pessimistic"


@pytest.mark.parametrize(
    ("direction", "stop", "targets", "bar_args", "expected"),
    [
        # A long gapping down through its stop fills at the open, not at the stop.
        (1, 95.0, [105.0], (93.0, 96.0, 92.0, 94.0), ("stop", None, 93.0)),
        # A long gapping up through its target fills at the open.
        (1, 95.0, [105.0], (107.0, 108.0, 106.0, 107.0), ("target", 0, 107.0)),
        # The short mirror: gapping up through the stop, down through the target.
        (-1, 105.0, [95.0], (107.0, 108.0, 106.0, 107.0), ("stop", None, 107.0)),
        (-1, 105.0, [95.0], (93.0, 96.0, 92.0, 94.0), ("target", 0, 93.0)),
    ],
)
def test_a_gap_through_a_level_fills_at_the_open(
    direction: Literal[1, -1],
    stop: float,
    targets: list[float],
    bar_args: tuple[float, float, float, float],
    expected: tuple[str, int | None, float],
) -> None:
    result = FillResolver().resolve(bar4h(*bar_args), stop, targets, direction)

    assert summarise(result) == [expected]


@pytest.mark.parametrize(
    ("direction", "stop", "targets", "bar_args", "expected"),
    [
        # Touching is inclusive: the extreme sitting exactly on the level fills it.
        (1, 95.0, [105.0], (100.0, 104.0, 95.0, 96.0), ("stop", None, 95.0)),
        (1, 95.0, [105.0], (100.0, 105.0, 96.0, 104.0), ("target", 0, 105.0)),
        (-1, 105.0, [95.0], (100.0, 105.0, 96.0, 104.0), ("stop", None, 105.0)),
        (-1, 105.0, [95.0], (100.0, 104.0, 95.0, 96.0), ("target", 0, 95.0)),
    ],
)
def test_an_extreme_exactly_on_a_level_fills_it(
    direction: Literal[1, -1],
    stop: float,
    targets: list[float],
    bar_args: tuple[float, float, float, float],
    expected: tuple[str, int | None, float],
) -> None:
    result = FillResolver().resolve(bar4h(*bar_args), stop, targets, direction)

    assert summarise(result) == [expected]


@pytest.mark.parametrize("with_subbars", [False, True])
def test_nothing_touched_yields_no_events(with_subbars: bool) -> None:
    args = (100.0, 104.0, 96.0, 101.0)
    bar = bar4h(*args, subbars=(sub1h(0, *args),) if with_subbars else ())

    result = FillResolver().resolve(bar, LONG_STOP, LONG_TARGETS, 1)

    assert result.events == ()
    assert result.mode == ("subbars" if with_subbars else "pessimistic")


@pytest.mark.parametrize("with_subbars", [False, True])
def test_only_the_target_touched_yields_one_target_event(with_subbars: bool) -> None:
    args = (100.0, 106.0, 96.0, 105.0)
    bar = bar4h(*args, subbars=(sub1h(0, *args),) if with_subbars else ())

    result = FillResolver().resolve(bar, LONG_STOP, LONG_TARGETS, 1)

    assert summarise(result) == [("target", 0, 105.0)]
    assert result.mode == ("subbars" if with_subbars else "pessimistic")


@pytest.mark.parametrize("with_subbars", [False, True])
def test_only_the_stop_touched_yields_one_stop_event(with_subbars: bool) -> None:
    args = (100.0, 104.0, 94.0, 96.0)
    bar = bar4h(*args, subbars=(sub1h(0, *args),) if with_subbars else ())

    result = FillResolver().resolve(bar, LONG_STOP, LONG_TARGETS, 1)

    assert summarise(result) == [("stop", None, 95.0)]
    assert result.mode == ("subbars" if with_subbars else "pessimistic")


@pytest.mark.parametrize(
    ("bar_args", "expected"),
    [
        ((100.0, 104.0, 96.0, 101.0), []),
        ((100.0, 104.0, 94.0, 96.0), [("stop", None, 95.0)]),
    ],
)
def test_a_trade_with_no_targets_can_only_stop_out(
    bar_args: tuple[float, float, float, float], expected: list[tuple[str, int | None, float]]
) -> None:
    result = FillResolver().resolve(bar4h(*bar_args), LONG_STOP, [], 1)

    assert summarise(result) == expected


# Partial: take targets[0] at 101, move the stop to breakeven (100), run the rest to 103.
PARTIAL_TARGETS = [101.0, 103.0]
BREAKEVEN = 100.0


def test_partial_then_breakeven_stop_across_subbars() -> None:
    bar = bar4h(
        100.2,
        103.5,
        99.0,
        102.0,
        subbars=(
            sub1h(0, 100.2, 101.5, 100.1, 101.2),  # takes 101, never reaches the old stop
            sub1h(1, 101.2, 101.6, 99.5, 99.6),  # dips through the new breakeven stop
        ),
    )

    result = FillResolver().resolve(bar, LONG_STOP, PARTIAL_TARGETS, 1, stop_after_partial=BREAKEVEN)

    assert summarise(result) == [("target", 0, 101.0), ("stop", None, 100.0)]
    assert result.mode == "subbars"


def test_partial_then_runner_target_across_subbars() -> None:
    bar = bar4h(
        100.2,
        103.5,
        99.0,
        103.0,
        subbars=(
            sub1h(0, 100.2, 101.5, 100.1, 101.2),  # takes 101
            sub1h(1, 101.2, 103.2, 101.0, 103.0),  # runs to 103 without revisiting 100
        ),
    )

    result = FillResolver().resolve(bar, LONG_STOP, PARTIAL_TARGETS, 1, stop_after_partial=BREAKEVEN)

    assert summarise(result) == [("target", 0, 101.0), ("target", 1, 103.0)]
    assert result.mode == "subbars"


def test_partial_then_breakeven_stop_without_subbars() -> None:
    result = FillResolver().resolve(
        bar4h(100.5, 102.5, 99.5, 102.0), LONG_STOP, PARTIAL_TARGETS, 1, stop_after_partial=BREAKEVEN
    )

    assert summarise(result) == [("target", 0, 101.0), ("stop", None, 100.0)]
    assert result.mode == "pessimistic"


def test_without_subbars_the_breakeven_stop_still_beats_the_runner_target() -> None:
    """Both the new stop and targets[1] lie in the range, so the stop is assumed first."""
    result = FillResolver().resolve(
        bar4h(100.5, 103.5, 99.5, 102.0), LONG_STOP, PARTIAL_TARGETS, 1, stop_after_partial=BREAKEVEN
    )

    assert summarise(result) == [("target", 0, 101.0), ("stop", None, 100.0)]


def test_without_subbars_both_targets_are_taken_when_the_new_stop_holds() -> None:
    result = FillResolver().resolve(
        bar4h(100.5, 103.5, 100.4, 103.0), LONG_STOP, PARTIAL_TARGETS, 1, stop_after_partial=BREAKEVEN
    )

    assert summarise(result) == [("target", 0, 101.0), ("target", 1, 103.0)]


def test_without_a_stop_after_partial_the_original_stop_stays_in_force() -> None:
    result = FillResolver().resolve(bar4h(100.5, 103.5, 99.5, 102.0), LONG_STOP, PARTIAL_TARGETS, 1)

    assert summarise(result) == [("target", 0, 101.0), ("target", 1, 103.0)]


# F1: the gap rule belongs to a stop that was already live when the candle opened. A stop
# swapped in mid-candle was reached the long way round — price ran to targets[0] and back —
# so it fills at the level, never at the candle's open.
@pytest.mark.parametrize("with_subbars", [False, True])
@pytest.mark.parametrize(
    ("direction", "stop", "targets", "stop_after_partial", "bar_args", "expected"),
    [
        (
            1,
            95.0,
            [101.0, 103.0],
            100.0,
            (99.0, 103.5, 98.5, 102.0),
            [("target", 0, 101.0), ("stop", None, 100.0)],
        ),
        (
            -1,
            105.0,
            [99.0, 97.0],
            100.0,
            (101.0, 101.5, 96.5, 97.0),
            [("target", 0, 99.0), ("stop", None, 100.0)],
        ),
    ],
)
def test_a_stop_swapped_in_mid_candle_fills_at_the_level_not_the_open(
    with_subbars: bool,
    direction: Literal[1, -1],
    stop: float,
    targets: list[float],
    stop_after_partial: float,
    bar_args: tuple[float, float, float, float],
    expected: list[tuple[str, int | None, float]],
) -> None:
    bar = bar4h(*bar_args, subbars=(sub1h(0, *bar_args),) if with_subbars else ())

    result = FillResolver().resolve(bar, stop, targets, direction, stop_after_partial=stop_after_partial)

    assert summarise(result) == expected


def test_a_stop_swapped_in_an_earlier_subbar_is_live_at_the_next_subbars_open() -> None:
    """The swapped stop survives into subbar 1, which opens below it: a genuine gap."""
    bar = bar4h(
        100.2,
        101.5,
        98.5,
        98.8,
        subbars=(
            sub1h(0, 100.2, 101.5, 100.1, 101.2),  # takes 101 and swaps the stop to 100
            sub1h(1, 99.0, 99.5, 98.5, 98.8),  # opens below the stop it inherited
        ),
    )

    result = FillResolver().resolve(bar, LONG_STOP, PARTIAL_TARGETS, 1, stop_after_partial=BREAKEVEN)

    assert summarise(result) == [("target", 0, 101.0), ("stop", None, 99.0)]


# F2: a target the stop has already passed is unreachable, so it is dropped rather than
# rejected. Indices stay the caller's, so dropping targets[0] leaves the runner at index 1.
@pytest.mark.parametrize(
    ("direction", "stop", "targets", "bar_args", "expected"),
    [
        (1, 102.0, [101.0, 105.0], (103.0, 106.0, 102.5, 105.5), [("target", 1, 105.0)]),
        (-1, 98.0, [99.0, 95.0], (97.0, 97.5, 94.0, 94.5), [("target", 1, 95.0)]),
    ],
)
def test_a_target_the_stop_has_passed_is_dropped_not_rejected(
    direction: Literal[1, -1],
    stop: float,
    targets: list[float],
    bar_args: tuple[float, float, float, float],
    expected: list[tuple[str, int | None, float]],
) -> None:
    result = FillResolver().resolve(bar4h(*bar_args), stop, targets, direction)

    assert summarise(result) == expected


@pytest.mark.parametrize(
    ("direction", "stop", "targets"),
    [
        (1, 95.0, [95.0]),  # a target level *at* the stop is not strictly beyond it
        (-1, 105.0, [105.0]),
    ],
)
def test_a_target_sitting_on_the_stop_is_dropped(
    direction: Literal[1, -1], stop: float, targets: list[float]
) -> None:
    """Without the drop the wrong-side extreme would report a nonsensical target fill."""
    result = FillResolver().resolve(bar4h(100.0, 104.0, 96.0, 101.0), stop, targets, direction)

    assert result.events == ()


def test_a_runner_target_the_new_stop_has_passed_is_dropped_after_the_partial() -> None:
    """``stop_after_partial`` sits on ``targets[1]``, so only the stop can close the runner."""
    result = FillResolver().resolve(
        bar4h(103.6, 105.0, 103.5, 104.5), LONG_STOP, PARTIAL_TARGETS, 1, stop_after_partial=103.0
    )

    # The bar opened above targets[0], so that leg gapped through and filled at the open.
    assert summarise(result) == [("target", 0, 103.6)]


@pytest.mark.parametrize(
    ("stop", "targets", "direction", "stop_after_partial"),
    [
        (95.0, [105.0], 0, None),  # direction must be 1 or -1
        (95.0, [101.0, 103.0, 105.0], 1, None),  # at most two targets
    ],
)
def test_nonsensical_inputs_raise(
    stop: float, targets: list[float], direction: int, stop_after_partial: float | None
) -> None:
    with pytest.raises(ValueError):
        FillResolver().resolve(
            bar4h(*LONG_BAR_ARGS),
            stop,
            targets,
            direction,  # type: ignore[arg-type]
            stop_after_partial=stop_after_partial,
        )


def test_stop_after_partial_is_irrelevant_when_the_partial_closes_the_trade() -> None:
    """With a single target nothing is left after the partial, so any level is accepted."""
    result = FillResolver().resolve(bar4h(*LONG_BAR_ARGS), LONG_STOP, [105.0], 1, stop_after_partial=999.0)

    assert summarise(result) == [("stop", None, 95.0)]


# F3: a bar that touches nothing is the common case, so the two empty verdicts are shared.
@pytest.mark.parametrize("with_subbars", [False, True])
def test_an_empty_verdict_reuses_the_module_singleton(with_subbars: bool) -> None:
    args = (100.0, 104.0, 96.0, 101.0)
    bar = bar4h(*args, subbars=(sub1h(0, *args),) if with_subbars else ())

    result = FillResolver().resolve(bar, LONG_STOP, LONG_TARGETS, 1)

    assert result is NO_EXITS["subbars" if with_subbars else "pessimistic"]


def test_the_empty_verdict_singletons_carry_no_events() -> None:
    assert NO_EXITS["subbars"] == Resolution(events=(), mode="subbars")
    assert NO_EXITS["pessimistic"] == Resolution(events=(), mode="pessimistic")


def test_fill_resolver_satisfies_the_exit_resolver_protocol() -> None:
    assert isinstance(FillResolver(), ExitResolver)


@st.composite
def partial_cases(draw: st.DrawFn) -> tuple[Bar, float, list[float], Literal[1, -1], float | None]:
    """A bar whose range holds the stop, one or two targets, and any breakeven stop.

    Levels are drawn as fractions of the range and mapped by direction, so a long counts up
    from its stop and a short counts down from its. When subbars are generated the last one
    spans the whole parent range, so some subbar always touches something and the
    "exactly one event" invariant has teeth.
    """
    direction: Literal[1, -1] = draw(st.sampled_from((1, -1)))
    low = draw(st.floats(min_value=1.0, max_value=10_000.0))
    span = draw(st.floats(min_value=1.0, max_value=1_000.0))
    high = low + span

    def at(fraction: float) -> float:
        """The price that far into the range from its losing end, in the profit direction."""
        return low + fraction * span if direction == 1 else high - fraction * span

    stop = at(draw(st.floats(min_value=0.02, max_value=0.20)))
    breakeven = at(draw(st.floats(min_value=0.25, max_value=0.40)))
    targets = [at(draw(st.floats(min_value=0.50, max_value=0.70)))]
    if draw(st.booleans()):
        targets.append(at(draw(st.floats(min_value=0.75, max_value=0.95))))
    stop_after_partial = breakeven if draw(st.booleans()) else None

    bar_open = low + draw(st.floats(min_value=0.0, max_value=1.0)) * span
    bar_close = low + draw(st.floats(min_value=0.0, max_value=1.0)) * span

    subbars: tuple[Bar, ...] = ()
    if draw(st.booleans()):
        pieces: list[Bar] = []
        for hour in range(draw(st.integers(min_value=0, max_value=3))):
            edges = sorted(low + draw(st.floats(min_value=0.0, max_value=1.0)) * span for _ in range(2))
            width = edges[1] - edges[0]
            inner = [edges[0] + draw(st.floats(min_value=0.0, max_value=1.0)) * width for _ in range(2)]
            pieces.append(sub1h(hour, inner[0], edges[1], edges[0], inner[1]))
        pieces.append(sub1h(len(pieces), bar_open, high, low, bar_close))
        subbars = tuple(pieces)

    bar = bar4h(bar_open, high, low, bar_close, subbars=subbars)
    return bar, stop, targets, direction, stop_after_partial


@settings(deadline=None, max_examples=250)
@given(partial_cases())
def test_events_are_an_ordered_prefix_of_the_targets_closed_by_at_most_one_stop(
    case: tuple[Bar, float, list[float], Literal[1, -1], float | None],
) -> None:
    bar, stop, targets, direction, stop_after_partial = case

    result = FillResolver().resolve(bar, stop, targets, direction, stop_after_partial=stop_after_partial)

    legs = [event.leg for event in result.events]
    assert result.events, "every level sits inside the range, so the bar must resolve something"
    assert legs.count("stop") <= 1, "one stop closes the position; there is nothing left to stop"
    assert "stop" not in legs[:-1], "nothing is reported once the stop has closed the position"
    taken = [event.target_index for event in result.events if event.leg == "target"]
    assert taken == list(range(len(taken))), "targets fill at most once each, nearest first"
    assert len(taken) <= len(targets)
    for event in result.events:
        assert bar.low <= event.price <= bar.high
    assert result.mode == ("subbars" if bar.subbars else "pessimistic")
    if len(targets) == 1:
        assert len(result.events) == 1, "both levels are reachable, so exactly one of them must fill"
