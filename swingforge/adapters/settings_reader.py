"""`TransientSettingsReader`: a `SettingsReader` that never holds the store open.

Paper trading's concurrency rule (WU-3A handoff): a running `paper` process must never hold
its venue `Store` open across bars, so the dashboard's write route (or `backfill`/
`tournament` run against the same file) can always get in between paper's own brief
open-write-close cycles. `current()` pays a fresh-connection cost on every call in exchange
for that -- acceptable at the CLI's bar-by-bar cadence (4H at the fastest), never at
per-tick frequency.
"""

from __future__ import annotations

from pathlib import Path

from swingforge.adapters.store import Store
from swingforge.core.settings import Settings

__all__ = ["TransientSettingsReader"]


class TransientSettingsReader:
    """Opens `path` read-only, reads `current_settings()`, and closes -- every call.

    A write via `Store.write_settings` from another process (the dashboard, or a human
    editing the settings table directly) is visible on the very next `current()` call,
    since nothing here caches across calls.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = path

    def current(self) -> tuple[int, Settings]:
        store = Store(self._path, read_only=True)
        try:
            return store.current_settings()
        finally:
            store.close()
