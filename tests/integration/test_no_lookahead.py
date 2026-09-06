"""Nothing a replay decides before T may depend on a bar after T.

Two stores are built from one price series: store A as generated, store B identical up to a
cutoff `T` and with every bar opening at or after `T` shifted 5% higher. `T` is a midnight UTC
boundary, so the split falls between whole bars on all three timeframes and the 1H/4H/Daily
views of B stay exactly as consistent with each other as A's.

Replay the same config over both. Everything the engine committed to at or before `T` — which
trades were entered, at what fill, with what risk, in what regime, against what `Context`
snapshot, and every exit leg that filled by `T` — must be identical. A strategy that peeked at
even one bar past `T` would move one of those.

The mirror assertion is what gives the test teeth: after `T` the two runs must *disagree*, or
the mutation was not visible to the strategy at all and the first assertion proved nothing.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timedelta

import pytest

from swingforge.adapters.replay import ReplaySource
from swingforge.adapters.store import Store
from swingforge.core.types import Bar, Instrument, Trade
from swingforge.lab.tournament import Config, add_months, run_config
from tests.integration.conftest import LOOKAHEAD_YEARS
from tests.integration.synth import PLANTED, SYNTH_START, real_entry_factory, real_session_factory

CUTOFF = add_months(SYNTH_START, 12)
"""Midnight UTC, so no bar of any timeframe straddles it."""

END = add_months(SYNTH_START, LOOKAHEAD_YEARS * 12)
SHIFT = 1.05
_SPANS: dict[str, timedelta] = {
    "1h": timedelta(hours=1),
    "4h": timedelta(hours=4),
    "1d": timedelta(hours=24),
}
_PRICES = ("open", "high", "low", "close")

ENTRIES = ("ict", "zones")
_MIN_TRADES = {"ict": 40, "zones": 5}
"""Trades each entry must place on each side of the cutoff for the comparison to say anything.

`Zones` needs a Daily impulse of two ATRs and a fresh opposing zone behind it, which is a rare
shape even in planted data -- it takes about sixteen trades over these two years where `ICT`
takes nearly two hundred. The floor is per entry so that its sparseness is recorded here
rather than quietly weakening the `ict` case as well."""


def _shifted_store(source: Store, instrument: Instrument, cutoff: datetime) -> Store:
    """A copy of `source` with every bar opening at or after `cutoff` scaled by `SHIFT`."""
    store = Store(":memory:")
    store.upsert_instruments([instrument])
    bars: list[Bar] = []
    for tf in ("1h", "4h", "1d"):
        window = source.bar_range(instrument, tf)
        assert window is not None, f"the source store holds no {tf} bars"
        first, last, _ = window
        for bar in source.bars(instrument, tf, first, last + _SPANS[tf]):
            bars.append(
                bar
                if bar.ts_open < cutoff
                else bar.model_copy(update={name: getattr(bar, name) * SHIFT for name in _PRICES})
            )
    store.upsert_bars(bars)
    return store


def _replay(store: Store, entry: str) -> tuple[Trade, ...]:
    config = Config(entry=entry, exit="fixed_r_2", session="none", instrument=PLANTED)
    return run_config(
        config,
        ReplaySource(store),
        entry_factory=real_entry_factory,
        session_factory=real_session_factory,
        start=SYNTH_START,
        end=END,
    ).trades


def _committed(trade: Trade, cutoff: datetime) -> tuple[object, ...]:
    """Everything about `trade` that was already decided by `cutoff`.

    `stop`, `target`, `mae_r`, `mfe_r`, `realized_r` and `closed_bar` are deliberately left out:
    they are the *current* state of a trade still open at the cutoff, and a trade that outlives
    `T` is entitled to finish differently in the shifted world.
    """
    return (
        trade.id,
        trade.direction,
        trade.entry_fill,
        trade.risk_r,
        trade.opened_bar,
        trade.regime,
        trade.context_snapshot,
        tuple(leg for leg in trade.legs if leg.ts <= cutoff),
    )


@pytest.fixture(scope="module")
def shifted_store(lookahead_store: Store) -> Iterator[Store]:
    store = _shifted_store(lookahead_store, PLANTED, CUTOFF)
    yield store
    store.close()


@pytest.fixture(scope="module", params=ENTRIES)
def runs(
    request: pytest.FixtureRequest, lookahead_store: Store, shifted_store: Store
) -> tuple[str, tuple[Trade, ...], tuple[Trade, ...]]:
    """One `(entry, original trades, shifted trades)` triple per entry family, replayed once."""
    entry = str(request.param)
    return entry, _replay(lookahead_store, entry), _replay(shifted_store, entry)


def test_the_shifted_store_only_differs_after_the_cutoff(
    lookahead_store: Store, shifted_store: Store
) -> None:
    """The mutation is exactly what it claims to be, on every timeframe."""
    for tf in ("1h", "4h", "1d"):
        window = lookahead_store.bar_range(PLANTED, tf)
        assert window is not None
        first, last, _ = window
        original = lookahead_store.bars(PLANTED, tf, first, last + _SPANS[tf])
        shifted = shifted_store.bars(PLANTED, tf, first, last + _SPANS[tf])
        pairs = list(zip(original, shifted, strict=True))
        before = [(a, b) for a, b in pairs if a.ts_open < CUTOFF]
        after = [(a, b) for a, b in pairs if a.ts_open >= CUTOFF]
        assert before and after
        assert all(a == b for a, b in before)
        for a, b in after:
            for name in _PRICES:
                assert getattr(b, name) == pytest.approx(getattr(a, name) * SHIFT)


def test_trades_entered_by_the_cutoff_are_unchanged(
    runs: tuple[str, tuple[Trade, ...], tuple[Trade, ...]],
) -> None:
    entry, original, shifted = runs
    before_original = [_committed(t, CUTOFF) for t in original if t.entry_fill.ts <= CUTOFF]
    before_shifted = [_committed(t, CUTOFF) for t in shifted if t.entry_fill.ts <= CUTOFF]

    assert len(before_original) >= _MIN_TRADES[entry], (
        f"{entry} entered only {len(before_original)} trades before the cutoff"
    )
    assert before_original == before_shifted


def test_the_mutation_does_change_what_happens_after_the_cutoff(
    runs: tuple[str, tuple[Trade, ...], tuple[Trade, ...]],
) -> None:
    """Without this the test above would pass on a strategy that ignores prices entirely."""
    entry, original, shifted = runs
    after_original = [t for t in original if t.entry_fill.ts > CUTOFF]
    after_shifted = [t for t in shifted if t.entry_fill.ts > CUTOFF]

    assert len(after_original) >= _MIN_TRADES[entry], (
        f"{entry} entered only {len(after_original)} trades after the cutoff"
    )
    assert after_original != after_shifted
