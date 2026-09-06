"""Which exit levels a closed bar hit, and in what order (design spec section 5).

The resolver is the whole answer to "the stop and the target were both inside that 4H bar
— which one filled?". It is pure: one bar in, a :class:`~swingforge.core.types.Resolution`
out, no state between calls, so backtest and paper trading get identical verdicts.

Semantics, in full, because the paper broker is written against them:

*Touch.* A long (``direction=1``) is stopped when ``bar.low <= stop`` and takes a target
``t`` when ``bar.high >= t``; a short (``-1``) is stopped when ``bar.high >= stop`` and
takes a target when ``bar.low <= t``. Touching is inclusive — price *at* the level fills.

*Price.* A touched level fills at the level itself, except when the candle opened at or
beyond it in the adverse direction (a gap through the level): then it fills at ``open``,
because that is the first price actually available. The gap rule belongs to levels that
were already live when the candle opened, so a stop that ``stop_after_partial`` swapped in
part-way through a candle fills at the level, never at that candle's open — price
demonstrably passed through it on the way back from ``targets[0]``. The swapped stop is
live again at the next subbar's open, where a gap through it counts once more. Prices are
passed through untouched; tick rounding belongs to the broker boundary
(:func:`~swingforge.core.types.round_to_tick`).

*Order.* With 1H ``subbars`` the resolver walks them oldest-first and the first subbar to
touch anything decides; ``mode`` is ``"subbars"``. Without subbars the whole bar is judged
at once and the stop is assumed to have come first; ``mode`` is ``"pessimistic"``. Either
way a single price range is read pessimistically: the stop in force is tested before the
next target, so a range holding both yields the stop.

*Sequencing.* ``targets`` are nearest-first and trusted in that order (index 0 is the
Partial's first leg, index 1 the runner). Taking ``targets[0]`` swaps the stop for
``stop_after_partial`` when one was given, and everything evaluated afterwards — the
remainder of the same candle, and every later subbar — sees the new stop. A stop event ends
the bar (the position is closed) and so does the last target; anything the bar did
afterwards is not reported.

*Reachability.* A target that is not strictly beyond the stop in force, in the profit
direction, is unreachable: a trailing stop has already ratcheted past it, so the stop would
close the position before price could ever get there. Such a target is dropped from this
call's evaluation rather than rejected — the stop governs. The rule is re-applied to
``targets[1]`` once ``stop_after_partial`` swaps in. Dropping never renumbers anything:
``ExitEvent.target_index`` is always the caller's own index into ``targets``.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Literal

from swingforge.core.types import Bar, ExitEvent, Resolution

__all__ = ["NO_EXITS", "FillResolver"]

MAX_TARGETS = 2
"""A trade carries at most two exit legs: the Partial's first target and the runner's."""

NO_EXITS: Mapping[Literal["subbars", "pessimistic"], Resolution] = MappingProxyType(
    {
        "subbars": Resolution(events=(), mode="subbars"),
        "pessimistic": Resolution(events=(), mode="pessimistic"),
    }
)
"""The two empty verdicts, shared rather than rebuilt.

Most bars touch nothing, and :class:`~swingforge.core.types.Resolution` is frozen, so one
instance per ``mode`` can serve every such call.
"""


def _stop_touched(candle: Bar, stop: float, direction: Literal[1, -1]) -> bool:
    """True when ``candle`` traded at or through ``stop`` on the losing side."""
    extreme = candle.low if direction == 1 else candle.high
    return (extreme - stop) * direction <= 0


def _target_touched(candle: Bar, target: float, direction: Literal[1, -1]) -> bool:
    """True when ``candle`` traded at or through ``target`` on the winning side."""
    extreme = candle.high if direction == 1 else candle.low
    return (extreme - target) * direction >= 0


def _stop_price(candle: Bar, stop: float, direction: Literal[1, -1], *, live_at_open: bool) -> float:
    """``stop``, or the open when the candle gapped through a stop that was live at it.

    A stop swapped in part-way through this candle (``live_at_open=False``) was not there to
    be gapped through: price ran out to ``targets[0]`` and came back, so it passed the new
    level on the way and fills there.
    """
    if live_at_open and (candle.open - stop) * direction <= 0:
        return candle.open
    return stop


def _target_price(candle: Bar, target: float, direction: Literal[1, -1]) -> float:
    """``target``, or the open when the candle gapped through it."""
    return candle.open if (candle.open - target) * direction >= 0 else target


def _reachable(target: float, stop: float, direction: Literal[1, -1]) -> bool:
    """True while ``target`` still sits strictly beyond ``stop`` in the profit direction."""
    return (target - stop) * direction > 0


def _validate(targets: list[float], direction: int) -> None:
    """Reject inputs no trade could have produced. Deliberately shallow.

    Levels are not cross-checked: a stop that has passed a target is a legitimate trailing
    state, handled by dropping that target rather than by raising.
    """
    if direction not in (1, -1):
        raise ValueError(f"direction must be 1 or -1, got {direction!r}")
    if len(targets) > MAX_TARGETS:
        raise ValueError(f"at most {MAX_TARGETS} targets are supported, got {len(targets)}")


class FillResolver:
    """The production :class:`~swingforge.adapters.base.ExitResolver`.

    Stateless and cheap to construct; one instance can serve every instrument in a run.
    """

    def resolve(
        self,
        bar: Bar,
        stop: float,
        targets: list[float],
        direction: Literal[1, -1],
        *,
        stop_after_partial: float | None = None,
    ) -> Resolution:
        """Resolve the exits triggered inside one closed bar, in occurrence order.

        ``targets`` are trusted to be nearest-first and at most two long. An empty
        ``events`` tuple means the bar touched nothing. See the module docstring for the
        touch, gap-through and sequencing rules this implements.

        A target the stop has already passed is unreachable; the stop governs, so such a
        target is dropped for this call and never reported. ``ExitEvent.target_index``
        still indexes the caller's own ``targets`` list.

        Raises ``ValueError`` only when ``direction`` is not ``1``/``-1`` or when more than
        two targets are given.
        """
        _validate(targets, direction)

        # One price range per subbar when we have them, otherwise the bar itself.
        candles: tuple[Bar, ...] = bar.subbars or (bar,)
        mode: Literal["subbars", "pessimistic"] = "subbars" if bar.subbars else "pessimistic"
        events: list[ExitEvent] = []
        current_stop = stop
        next_target = 0

        for candle in candles:
            # Whatever stop we carry into a candle was live when that candle opened, so the
            # gap rule applies to it. Taking targets[0] below clears this for the rest of
            # that candle only: the swapped stop is live again at the next candle's open.
            stop_live_at_open = True
            while True:
                if _stop_touched(candle, current_stop, direction):
                    price = _stop_price(candle, current_stop, direction, live_at_open=stop_live_at_open)
                    events.append(ExitEvent(leg="stop", price=price))
                    return Resolution(events=tuple(events), mode=mode)
                while next_target < len(targets) and not _reachable(
                    targets[next_target], current_stop, direction
                ):
                    next_target += 1  # the stop has ratcheted past this leg: it can never fill
                if next_target >= len(targets):
                    # Nothing is left to aim at — the caller gave no targets, or every one
                    # still pending was just dropped. Taking the last reachable target
                    # returns below instead, so from here only a stop can happen, on this
                    # candle or a later one.
                    break
                level = targets[next_target]
                if not _target_touched(candle, level, direction):
                    break
                events.append(
                    ExitEvent(
                        leg="target",
                        price=_target_price(candle, level, direction),
                        target_index=next_target,
                    )
                )
                if next_target == 0 and stop_after_partial is not None:
                    current_stop = stop_after_partial
                    stop_live_at_open = False
                next_target += 1
                if next_target >= len(targets):
                    return Resolution(events=tuple(events), mode=mode)

        if not events:
            return NO_EXITS[mode]
        return Resolution(events=tuple(events), mode=mode)
