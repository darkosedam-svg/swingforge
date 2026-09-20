"""Wall-clock budgets that survive a busy machine.

The performance tests guard against algorithmic regressions (a linear scan where a bisection
was, a cache rebuilt every bar), and they do it with a stopwatch. On a developer's desktop the
stopwatch also measures the browser, the editor and whatever else holds the cores: the same test
that takes 3s alone took 17s beside a loaded browser (2026-09-20), and a budget sized for a quiet
machine then fails with nothing wrong.

So a budget is scaled by how slow the machine is *right now*: a fixed pure-Python loop is timed
just before and just after the measured section, and the budget grows by the worse of the two
readings over the loop's time on a quiet machine. Contention slows the loop and the code under
test alike, so a busy machine stretches both and a real regression still overshoots; a machine
faster than the reference one is simply held to the unscaled budget.
"""

from __future__ import annotations

import time

__all__ = ["load_factor", "scaled_budget"]

_NOMINAL_LOOP_S = 0.045
"""The calibration loop on the reference machine with nothing else running (best of nine)."""

_LOOP_ITERATIONS = 400_000
_READINGS = 3


def _calibration_loop() -> float:
    started = time.perf_counter()
    total = 0.0
    for i in range(_LOOP_ITERATIONS):
        total += i * 0.5
    return time.perf_counter() - started


def load_factor() -> float:
    """How many times slower than the quiet reference this process is running now; never below 1."""
    readings = sorted(_calibration_loop() for _ in range(_READINGS))
    return max(1.0, readings[_READINGS // 2] / _NOMINAL_LOOP_S)


def scaled_budget(budget_s: float, before: float) -> float:
    """`budget_s` stretched by the worse of `before` (a `load_factor()` taken just ahead of the
    timed section) and the load as it is now, just after it."""
    return budget_s * max(before, load_factor())
