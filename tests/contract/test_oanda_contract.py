"""Contract test: OANDA `history()` schema, UTC timestamps, and the Daily re-cut.

Manual / cassette-replay only (see `tests/contract/README.md`). Constructs the real
`oandapyV20.API` client from `OANDA_TOKEN`/`OANDA_ACCOUNT_ID` env vars, falling back to
dummy values when replaying — the Authorization header is filtered out of the cassette and
is not part of vcrpy's request matching, so a dummy token still replays correctly.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta

import pytest

from swingforge.adapters.oanda.bars import OandaBars, oanda_instrument, recut_daily

from .conftest import skip_if_no_cassette

CASSETTE = "oanda_eur_usd_1d_10d.yaml"


@pytest.mark.contract
def test_oanda_daily_recut_matches_1h_highs_and_utc_timestamps(vcr) -> None:
    skip_if_no_cassette(CASSETTE)
    token = os.environ.get("OANDA_TOKEN", "dummy-token")

    with vcr.use_cassette(CASSETTE, allow_playback_repeats=True):
        import oandapyV20

        api = oandapyV20.API(access_token=token, environment="practice")
        src = OandaBars(api=api)
        instrument = oanda_instrument("EUR_USD")
        # Fixed historical window so a replayed cassette's recorded request always matches.
        end = datetime(2024, 6, 1, tzinfo=UTC)
        start = end - timedelta(days=10)

        h1_bars = src.history(instrument, "1h", start, end)
        daily_bars = src.history(instrument, "1d", start, end)

    assert h1_bars, "expected at least one H1 bar from the cassette"
    assert daily_bars, "expected at least one re-cut daily bar from the cassette"

    expected_by_day = {bar.ts_open: bar for bar in recut_daily(h1_bars)}
    for daily in daily_bars:
        assert daily.ts_open.tzinfo is UTC
        assert daily.ts_open.hour == 0
        assert daily.ts_open.minute == 0

        match = expected_by_day[daily.ts_open]
        assert daily.high == match.high
        assert daily.low == match.low

        day_h1 = [b for b in h1_bars if b.ts_open.date() == daily.ts_open.date()]
        assert day_h1, "a re-cut day must be backed by at least one 1H bar"
        assert daily.high == max(b.high for b in day_h1)
        assert daily.low == min(b.low for b in day_h1)
