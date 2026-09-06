"""Contract test: Hyperliquid `history()` schema, UTC bar alignment, no gaps.

Manual / cassette-replay only (see `tests/contract/README.md`). Constructs the real
`hyperliquid.info.Info` client; HL's candle endpoints are public, so no credentials are
needed even when recording.
"""

from __future__ import annotations

import itertools
from datetime import UTC, datetime, timedelta

import pytest

from swingforge.adapters.hyperliquid.bars import DEFAULT_PERPS, HyperliquidBars, hl_instrument

from .conftest import skip_if_no_cassette

CASSETTE = "hyperliquid_btc_4h_30d.yaml"


@pytest.mark.contract
def test_hyperliquid_history_shape_and_alignment(vcr) -> None:
    skip_if_no_cassette(CASSETTE)

    with vcr.use_cassette(CASSETTE):
        from hyperliquid.info import Info

        info = Info(base_url=None, skip_ws=True)
        src = HyperliquidBars(info=info)
        instrument = hl_instrument("BTC", DEFAULT_PERPS["BTC"])
        # Fixed historical window so a replayed cassette's recorded request always matches.
        end = datetime(2024, 6, 1, tzinfo=UTC)
        start = end - timedelta(days=30)

        bars = src.history(instrument, "4h", start, end)

    assert bars, "expected at least one bar from the cassette"
    for bar in bars:
        assert bar.ts_open.tzinfo is UTC
        assert bar.ts_open.hour in {0, 4, 8, 12, 16, 20}
        assert bar.ts_open.minute == 0
        assert bar.ts_open.second == 0
    for prev, curr in itertools.pairwise(bars):
        assert curr.ts_open - prev.ts_open == timedelta(hours=4), "gap between consecutive 4H bars"
