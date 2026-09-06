"""Entry-time session gating (design spec section 4).

Three modes, checked against a bar's `ts_open` (already UTC — `Bar.ts_open` normalises any
tz-aware offset at construction):

- `"none"` — always allowed.
- `"london_ny"` — bars opening in `[07:00, 21:00)` UTC.
- `"active"` — for `fx` instruments, identical to `london_ny`; for `perp` instruments,
  excludes bars opening in `[00:00, 07:00)` UTC and any bar opening on a Saturday or Sunday.
"""

from __future__ import annotations

from typing import Literal

from swingforge.core.types import Bar

__all__ = ["SessionFilter", "allowed"]

SessionProfile = Literal["fx", "perp"]
SessionMode = Literal["none", "london_ny", "active"]

_LONDON_NY_START_H = 7
_LONDON_NY_END_H = 21
_PERP_QUIET_END_H = 7


def allowed(bar: Bar, profile: SessionProfile, mode: SessionMode) -> bool:
    """Whether `bar` may be used for a new entry under `mode`/`profile`."""
    if mode == "none":
        return True
    hour = bar.ts_open.hour
    if mode == "london_ny":
        return _LONDON_NY_START_H <= hour < _LONDON_NY_END_H
    # mode == "active"
    if profile == "fx":
        return _LONDON_NY_START_H <= hour < _LONDON_NY_END_H
    if 0 <= hour < _PERP_QUIET_END_H:
        return False
    return bar.ts_open.weekday() < 5


class SessionFilter:
    """Callable `Bar -> bool` gate for the engine's `session_allowed`."""

    def __init__(self, mode: SessionMode, profile: SessionProfile) -> None:
        self.mode = mode
        self.profile = profile

    def __call__(self, bar: Bar) -> bool:
        return allowed(bar, self.profile, self.mode)

    def reset(self) -> None:
        """No-op: `SessionFilter` carries no per-instance state to clear.

        Exists so a runner can call `reset()` uniformly across every strategy and gate
        component without special-casing this one.
        """
        return None
