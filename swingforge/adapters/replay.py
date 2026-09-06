"""`ReplaySource`: the only data path into the engine during backtests.

Reads closed bars back out of a `Store` (DuckDB), attaching 1H `subbars` to each 4H bar so
the fill resolver can order a stop and a target that both fall inside it, and interleaving
4H and Daily bars by close time for `merged()`, the stream the engine actually consumes.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

from swingforge.adapters.store import Store
from swingforge.core.types import TF, Bar, Instrument

__all__ = ["ReplaySource"]

# NOTE: duplicated from the canonical `_TF_SPAN` in `swingforge/core/types.py` (a contract
# file this work unit may not edit) -- keep the two in sync by hand if a timeframe's span
# ever changes.
_TF_SPAN: dict[str, timedelta] = {
    "1h": timedelta(hours=1),
    "4h": timedelta(hours=4),
    "1d": timedelta(hours=24),
}
_SUBBARS_PER_4H = 4
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


def _floor_to_4h(ts: datetime) -> datetime:
    """Floor `ts` to the start of its containing 4H bucket (00:00, 04:00, ... UTC).

    Every 1H bar fetched for subbar-attach gets bucketed by this key so it lines up with
    the 4H bar whose span contains it, in one pass over the whole batch rather than one
    query per 4H bar (see `history`).
    """
    return _EPOCH + (ts - _EPOCH) // _TF_SPAN["4h"] * _TF_SPAN["4h"]


class ReplaySource:
    """A `BarSource` that replays historical bars out of a `Store`.

    Every bar it yields is already closed (the store only ever holds closed bars), so
    there is no separate "wait for close" step: replay simply reads history in order.
    """

    def __init__(self, store: Store) -> None:
        self._store = store

    def history(self, instrument: Instrument, tf: TF, start: datetime, end: datetime) -> list[Bar]:
        """Closed bars in `[start, end)`, oldest first.

        4H bars have their 1H `subbars` attached, but only when all four exist for that
        bar's span — partial coverage yields `subbars=()`, so the fill resolver falls back
        to its pessimistic (stop-first) assumption rather than ordering on incomplete data.

        Fetches every 1H bar the whole batch could need in one `Store.bars` call --
        `[start, end + 4h)`, since the last 4H bar in range can span up to 4h past `end` --
        then buckets them by their containing 4H boundary (`_floor_to_4h`) in one pass.
        That is exactly two `Store.bars` calls for any `tf == "4h"` request (one for the 4H
        bars, one for their 1H subbars), independent of how many 4H bars are returned; the
        previous version queried once per 4H bar (`test_history_query_count_is_two`).
        """
        bars = self._store.bars(instrument, tf, start, end)
        if tf != "4h" or not bars:
            return bars
        subbars = self._store.bars(instrument, "1h", start, end + _TF_SPAN["4h"])
        buckets: dict[datetime, list[Bar]] = {}
        for sub in subbars:
            buckets.setdefault(_floor_to_4h(sub.ts_open), []).append(sub)
        result: list[Bar] = []
        for bar in bars:
            bucket = buckets.get(bar.ts_open, [])
            if len(bucket) != _SUBBARS_PER_4H:
                result.append(bar)
                continue
            # `model_validate` (not `model_copy`, which skips validators) re-runs `Bar`'s
            # subbar checks -- a genuinely malformed subbar (e.g. one whose high/low falls
            # outside its parent's range) raises `pydantic.ValidationError` here instead of
            # silently producing an invalid `Bar` for the resolver to trip over later.
            result.append(Bar.model_validate({**bar.model_dump(), "subbars": tuple(bucket)}))
        return result

    async def stream(self, instrument: Instrument, tf: TF) -> AsyncIterator[Bar]:
        """Yield the store's full history for `(instrument, tf)`, oldest first."""
        bar_range = self._store.bar_range(instrument, tf)
        if bar_range is None:
            return
        start, last_open, _ = bar_range
        end = last_open + _TF_SPAN[tf]
        for bar in self.history(instrument, tf, start, end):
            yield bar

    def merged(self, instrument: Instrument, start: datetime, end: datetime) -> list[Bar]:
        """4H bars (with subbars) and Daily bars, interleaved by close time.

        This is the stream the engine actually consumes: Daily gives strategies their bias,
        4H is the execution timeframe. Ties (a Daily bar closing at the same instant as a 4H
        bar, i.e. the last 4H bar of the day) put the 4H bar first.
        """
        four_hour = self.history(instrument, "4h", start, end)
        daily = self.history(instrument, "1d", start, end)
        tagged = [(bar.ts_open + _TF_SPAN["4h"], 0, bar) for bar in four_hour] + [
            (bar.ts_open + _TF_SPAN["1d"], 1, bar) for bar in daily
        ]
        tagged.sort(key=lambda item: (item[0], item[1]))
        return [bar for _, _, bar in tagged]
