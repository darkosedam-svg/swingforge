"""Shared fixtures: the synthetic stores, built once per session.

Every generator here is seeded, so a store is a pure function of its arguments and building
it once and sharing it across tests is safe — nothing in this suite writes bars back.
`tests/integration/test_determinism.py` is the test that proves the replays over them are
reproducible in the first place, so the sharing is not an assumption, it is a result.
"""

from __future__ import annotations

from collections.abc import Iterator
from decimal import Decimal

import pytest

from swingforge.adapters.store import Store
from swingforge.core.types import Instrument
from swingforge.lab.tournament import add_months
from tests.integration.synth import PLANTED, SYNTH_START, planted_store, random_walk_store

WALK = Instrument(
    venue="hyperliquid",
    symbol="WALK",
    tick_size=Decimal("0.01"),
    contract_multiplier=Decimal("1"),
    quote_ccy="USD",
    session_profile="perp",
)
"""The instrument behind `walk_store` — a fine tick on a ~100 price, so rounding never
swallows a level."""

WALK_YEARS = 2
WALK_SEED = 11
WALK_END = add_months(SYNTH_START, WALK_YEARS * 12)
"""One past the last 4H bar's open plus its span, so `[SYNTH_START, WALK_END)` is the store."""

PLANTED_YEARS = 4
PLANTED_SEED = 0

LOOKAHEAD_YEARS = 2
LOOKAHEAD_SEED = 3


@pytest.fixture(scope="session")
def walk_store() -> Iterator[Store]:
    """Two years of a seeded random walk with nothing planted in it."""
    store = random_walk_store([WALK], years=WALK_YEARS, seed=WALK_SEED)
    yield store
    store.close()


@pytest.fixture(scope="session")
def lookahead_store() -> Iterator[Store]:
    """Two years of planted setups — the base series for the no-lookahead mutation.

    Planted rather than a plain random walk purely for *density*. Over four years of an
    unplanted walk `ICT` takes 26 trades and `Zones` 7, so a mutation test built on one would
    be comparing a handful of trades and proving very little; the same two years of planted
    setups give `ICT` roughly 190 and `Zones` 16, split either side of the cutoff. Whether the
    setups pay is irrelevant here, so `edge=True` is chosen only because its post-entry drift
    is the displacement `Zones` looks for.
    """
    store = planted_store(PLANTED, years=LOOKAHEAD_YEARS, seed=LOOKAHEAD_SEED, edge=True)
    yield store
    store.close()


@pytest.fixture(scope="session")
def planted_edge_store() -> Iterator[Store]:
    """Four years of planted ICT setups that pay: the tournament must find the edge."""
    store = planted_store(PLANTED, years=PLANTED_YEARS, seed=PLANTED_SEED, edge=True)
    yield store
    store.close()


@pytest.fixture(scope="session")
def planted_flat_store() -> Iterator[Store]:
    """The same setups at the same bars, with no drift after the entry: the control."""
    store = planted_store(PLANTED, years=PLANTED_YEARS, seed=PLANTED_SEED, edge=False)
    yield store
    store.close()
