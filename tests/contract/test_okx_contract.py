"""Contract test: OKX `history()` schema, UTC bar alignment, no gaps - 4H and Daily.

Manual / cassette-replay only (see `tests/contract/README.md`). Drives the real `OkxClient`
over `urllib`; the endpoints are public, so no credentials are needed even when recording.
The Daily case is the one a venue quirk can break silently: OKX's plain `1D` candle opens at
16:00 UTC, and only `1Dutc` lines up with the rest of the system.
"""

from __future__ import annotations

import itertools
from datetime import UTC, datetime, timedelta

import pytest

from swingforge.adapters.okx import OkxBars, OkxClient, okx_instrument

from .conftest import skip_if_no_cassette

CASSETTE = "okx_btc_4h_1d_30d.yaml"


@pytest.mark.contract
def test_okx_history_shape_and_alignment(vcr) -> None:
    skip_if_no_cassette(CASSETTE)

    with vcr.use_cassette(CASSETTE):
        client = OkxClient(min_interval_s=0.0)
        instrument = okx_instrument(client, "BTC")
        src = OkxBars(client)
        # Fixed historical window so a replayed cassette's recorded requests always match.
        end = datetime(2024, 6, 1, tzinfo=UTC)
        start = end - timedelta(days=30)
        four_hour = src.history(instrument, "4h", start, end)
        daily = src.history(instrument, "1d", start, end)

    assert instrument.venue == "okx" and instrument.quote_ccy == "USDT"
    assert instrument.tick_size > 0

    assert len(four_hour) == 30 * 6
    for bar in four_hour:
        assert bar.ts_open.tzinfo is UTC
        assert bar.ts_open.hour in {0, 4, 8, 12, 16, 20}
        assert (bar.ts_open.minute, bar.ts_open.second) == (0, 0)
        assert bar.low <= min(bar.open, bar.close) <= max(bar.open, bar.close) <= bar.high
    for prev, curr in itertools.pairwise(four_hour):
        assert curr.ts_open - prev.ts_open == timedelta(hours=4), "gap between consecutive 4H bars"

    assert len(daily) == 30
    assert all((bar.ts_open.hour, bar.ts_open.minute) == (0, 0) for bar in daily), (
        "Daily must open at 00:00 UTC"
    )
    # a Daily bar is the six 4H bars of its day: same open, same close, same extremes
    first_day = [bar for bar in four_hour if bar.ts_open.date() == daily[0].ts_open.date()]
    assert daily[0].open == first_day[0].open and daily[0].close == first_day[-1].close
    assert daily[0].high == max(bar.high for bar in first_day)
    assert daily[0].low == min(bar.low for bar in first_day)
