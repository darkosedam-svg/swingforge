"""Operator-editable run settings and the protocol for reading them.

`Settings` is the only thing a human may change while a run is in flight (via the
dashboard's write route). Nothing in `core` or `strategies` is editable: strategy
parameters belong to the tournament, not to an operator.

The engine reads settings at bar close only, through a `SettingsReader`. The reader returns
a version alongside the values so the engine can tell "unchanged" from "changed" without
comparing models, and so a change that lands mid-bar is picked up on the *next* bar rather
than halfway through the current one.
"""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["InMemorySettingsReader", "Settings", "SettingsReader", "StaticSettingsReader"]


class Settings(BaseModel):
    """Run settings, capped so an operator cannot set anything dangerous.

    `risk_pct` is hard-capped at 2% of equity per trade by the model itself, so the cap
    holds no matter which code path writes it. `enabled_instruments` maps a venue name to
    the symbols enabled on it; `paper_configs` holds the config ids currently running
    paper.

    Frozen, so a `Settings` handed to the engine cannot be re-pointed underneath it. Note
    that freezing prevents field reassignment, not in-place mutation of the
    `enabled_instruments` dict: treat it as read-only and build a new `Settings` to change
    it.
    """

    model_config = ConfigDict(frozen=True)

    risk_pct: float = Field(0.01, gt=0, le=0.02)
    enabled_instruments: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    paper_configs: tuple[str, ...] = ()
    session: Literal["none", "london_ny", "active"] = "none"
    time_stop_bars: int = Field(10, ge=1)
    kill_switch: bool = False


@runtime_checkable
class SettingsReader(Protocol):
    """Source of the current settings for a run.

    `current()` returns `(version, settings)`. The version increments on every change, so
    the engine can refresh cheaply: read at each bar close, and only re-apply when the
    version moved.
    """

    def current(self) -> tuple[int, Settings]: ...


class StaticSettingsReader:
    """A reader whose settings never change: version is always 0.

    Used by backtests, where settings are fixed for the whole run, and by unit tests. Truly
    static: the settings are held privately and `settings` is read-only, so a caller cannot
    swap them in behind the engine and leave the version claiming nothing moved.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings if settings is not None else Settings()

    @property
    def settings(self) -> Settings:
        """The fixed settings this reader was built with."""
        return self._settings

    def current(self) -> tuple[int, Settings]:
        return 0, self._settings


class InMemorySettingsReader:
    """A reader whose settings can be replaced, bumping the version each time.

    This is what a test drives when it needs a settings change to land between bars: `set`
    increments the version by 1 on every call, including when the new values equal the old
    ones, so the engine always sees a change it must re-apply at the next bar close.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._version = 0
        self._settings = settings if settings is not None else Settings()

    def current(self) -> tuple[int, Settings]:
        return self._version, self._settings

    def set(self, settings: Settings) -> None:
        """Replace the settings and bump the version by 1."""
        self._settings = settings
        self._version += 1
