"""Unit tests for the OKX research adapter (`swingforge.adapters.okx`): no network.

The facts these tests encode were read off the live public API on 2026-09-20 (see the WU-OKX
handoff): `history-candles` pages newest-first, up to 300 rows, with an *exclusive* `after`
cursor; the still-forming candle is served with `confirm == "0"`; Daily bars open at 16:00 UTC
unless the `1Dutc` bar is asked for; funding settles every 8 hours and only the last three
months of it are served.
"""

from __future__ import annotations

import asyncio
import json
import urllib.error
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import pytest

from swingforge.adapters.okx import (
    DEFAULT_FUNDING_RATE,
    DEFAULT_SWAPS,
    OkxBars,
    OkxClient,
    OkxCosts,
    OkxError,
    inst_id,
    load_funding,
    okx_instrument,
    okx_instruments,
)
from swingforge.core.types import Bar, Instrument, Order, Position

BTC = Instrument(
    venue="okx",
    symbol="BTC",
    tick_size=Decimal("0.1"),
    contract_multiplier=Decimal("1"),
    quote_ccy="USDT",
    session_profile="perp",
)
T0 = datetime(2024, 1, 1, tzinfo=UTC)
H4 = timedelta(hours=4)


def _ms(moment: datetime) -> int:
    return int(moment.timestamp() * 1000)


def _row(ts_open: datetime, close: float = 100.0, *, confirm: str = "1") -> list[str]:
    ohlc = [str(close - 1.0), str(close + 1.0), str(close - 2.0), str(close)]
    return [str(_ms(ts_open)), *ohlc, "1200", "12.5", "1250.0", confirm]


class FakeOkx:
    """Serves candles the way the venue does: newest first, older than `after`, `limit` at most."""

    def __init__(
        self, candles: list[list[str]], *, funding: list[dict[str, str]] | None = None, cap: int = 300
    ) -> None:
        self.cap = cap
        self.candles = sorted(candles, key=lambda row: -int(row[0]))
        self.funding = sorted(funding or [], key=lambda row: -int(row["fundingTime"]))
        self.candle_calls: list[tuple[str, str, int | None, int]] = []
        self.funding_calls: list[int | None] = []

    def history_candles(
        self, inst: str, bar: str, *, after_ms: int | None = None, limit: int = 300
    ) -> list[list[str]]:
        self.candle_calls.append((inst, bar, after_ms, limit))
        older = [row for row in self.candles if after_ms is None or int(row[0]) < after_ms]
        return older[: min(limit, self.cap)]

    def funding_rate_history(
        self, inst: str, *, after_ms: int | None = None, limit: int = 100
    ) -> list[dict[str, str]]:
        self.funding_calls.append(after_ms)
        older = [row for row in self.funding if after_ms is None or int(row["fundingTime"]) < after_ms]
        return older[:limit]

    def instrument(self, inst: str) -> dict[str, str]:
        return {"instId": inst, "tickSz": "0.1", "ctVal": "0.01", "settleCcy": "USDT", "state": "live"}


# --- the client --------------------------------------------------------------------------------


def _client(responses: list[Any], **kwargs: Any) -> tuple[OkxClient, list[str], list[float]]:
    urls: list[str] = []
    sleeps: list[float] = []

    def fetch(url: str) -> bytes:
        urls.append(url)
        response = responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return json.dumps(response).encode()

    client = OkxClient(fetch=fetch, sleep=sleeps.append, min_interval_s=0.0, **kwargs)
    return client, urls, sleeps


def test_client_builds_the_history_candles_request() -> None:
    client, urls, _ = _client([{"code": "0", "msg": "", "data": [_row(T0)]}])
    rows = client.history_candles("BTC-USDT-SWAP", "4H", after_ms=1_700_000_000_000, limit=300)
    assert rows == [_row(T0)]
    assert urls == [
        "https://www.okx.com/api/v5/market/history-candles?instId=BTC-USDT-SWAP&bar=4H&limit=300&after=1700000000000"
    ]


def test_client_raises_the_venues_own_error_code() -> None:
    client, _, _ = _client([{"code": "51001", "msg": "Instrument ID does not exist", "data": []}])
    with pytest.raises(OkxError, match="51001.*does not exist"):
        client.instrument("NOPE-USDT-SWAP")


def test_client_backs_off_on_a_rate_limit_and_then_succeeds() -> None:
    """The venue answers a burst with HTTP 429, or with its own code 50011 in a 200."""
    too_many = urllib.error.HTTPError("https://www.okx.com", 429, "Too Many Requests", None, None)  # type: ignore[arg-type]
    client, urls, sleeps = _client(
        [too_many, {"code": "50011", "msg": "Too Many Requests", "data": []}, {"code": "0", "data": []}]
    )
    assert client.history_candles("BTC-USDT-SWAP", "4H") == []
    assert len(urls) == 3
    assert sleeps == [1.0, 2.0]


def test_client_gives_up_on_a_rate_limit_that_never_lifts() -> None:
    limited = {"code": "50011", "msg": "Too Many Requests", "data": []}
    client, urls, sleeps = _client([limited] * 10, max_retries=3)
    with pytest.raises(OkxError, match="50011"):
        client.history_candles("BTC-USDT-SWAP", "4H")
    assert len(urls) == 4 and sleeps == [1.0, 2.0, 4.0]


def test_client_retries_what_a_cdn_does_to_a_long_backfill() -> None:
    """A 5xx, a connection that did not hold, a timeout, an HTML interstitial: over eight hundred
    requests these are likely, and losing the run to one of them at request 700 helps nobody."""
    unavailable = urllib.error.HTTPError("https://www.okx.com", 503, "Service Unavailable", None, None)  # type: ignore[arg-type]
    responses: list[Any] = [unavailable, urllib.error.URLError("connection reset"), TimeoutError("timed out")]
    client, urls, sleeps = _client([*responses, {"code": "0", "data": [_row(T0)]}])
    assert client.history_candles("BTC-USDT-SWAP", "4H") == [_row(T0)]
    assert len(urls) == 4 and sleeps == [1.0, 2.0, 4.0]


def test_client_lets_the_callers_own_mistake_through_at_once() -> None:
    bad_request = urllib.error.HTTPError("https://www.okx.com", 400, "Bad Request", None, None)  # type: ignore[arg-type]
    client, urls, sleeps = _client([bad_request])
    with pytest.raises(urllib.error.HTTPError):
        client.instrument("BTC-USDT-SWAP")
    assert len(urls) == 1 and sleeps == []


def test_client_names_a_body_that_is_not_json() -> None:
    """The CDN's error page, not the venue's envelope: say so, instead of `Expecting value: line 1`."""
    urls: list[str] = []

    def fetch(url: str) -> bytes:
        urls.append(url)
        return b"<html><body>502 Bad Gateway</body></html>"

    client = OkxClient(fetch=fetch, sleep=lambda _s: None, min_interval_s=0.0, max_retries=2)
    with pytest.raises(OkxError, match="non-JSON body.*502 Bad Gateway"):
        client.history_candles("BTC-USDT-SWAP", "4H")
    assert len(urls) == 3  # retried like any transient failure, then reported


def test_client_reports_an_envelope_without_a_code() -> None:
    client, _, _ = _client([{"data": []}])
    with pytest.raises(OkxError, match="missing"):
        client.history_candles("BTC-USDT-SWAP", "4H")


def test_client_builds_the_funding_request() -> None:
    row = {"instId": "BTC-USDT-SWAP", "fundingTime": "1700000000000", "fundingRate": "0.0001"}
    client, urls, _ = _client([{"code": "0", "data": [row]}])
    assert client.funding_rate_history("BTC-USDT-SWAP", after_ms=1_700_000_000_001) == [row]
    assert urls == [
        "https://www.okx.com/api/v5/public/funding-rate-history?instId=BTC-USDT-SWAP&limit=100&after=1700000000001"
    ]


def test_client_instrument_returns_the_venues_row_and_refuses_an_empty_answer() -> None:
    row = {"instId": "BTC-USDT-SWAP", "tickSz": "0.1", "state": "live"}
    client, urls, _ = _client([{"code": "0", "data": [row]}, {"code": "0", "data": []}])
    assert client.instrument("BTC-USDT-SWAP") == row
    assert urls == ["https://www.okx.com/api/v5/public/instruments?instType=SWAP&instId=BTC-USDT-SWAP"]
    with pytest.raises(LookupError, match="NOPE-USDT-SWAP"):
        client.instrument("NOPE-USDT-SWAP")


def test_client_spaces_its_requests() -> None:
    """20 requests per 2 seconds is the venue's budget for `history-candles`; a backfill is
    several hundred pages, so the client keeps a minimum interval rather than finding the limit."""
    now = [100.0]
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        now[0] += seconds

    client = OkxClient(
        fetch=lambda url: json.dumps({"code": "0", "data": []}).encode(),
        sleep=sleep,
        clock=lambda: now[0],
        min_interval_s=0.25,
    )
    client.history_candles("BTC-USDT-SWAP", "4H")
    now[0] += 0.1
    client.history_candles("BTC-USDT-SWAP", "4H")
    assert sleeps == [pytest.approx(0.15)]


# --- bars --------------------------------------------------------------------------------------


def test_history_pages_backwards_and_returns_oldest_first() -> None:
    candles = [_row(T0 + i * H4, close=100.0 + i) for i in range(700)]
    fake = FakeOkx(candles)
    source = OkxBars(fake, now=lambda: T0 + timedelta(days=400))

    bars = source.history(BTC, "4h", T0, T0 + 700 * H4)

    assert [bar.ts_open for bar in bars] == [T0 + i * H4 for i in range(700)]
    assert bars[0].close == 100.0 and bars[-1].close == 799.0
    assert bars[0].volume == 12.5  # base-coin volume (`volCcy`), not the contract count
    assert all(bar.instrument is BTC and bar.tf == "4h" for bar in bars)
    # three pages: 300 + 300 + 100, each cursor the oldest candle of the page before
    assert [call[2] for call in fake.candle_calls] == [
        _ms(T0 + 700 * H4),
        _ms(T0 + 400 * H4),
        _ms(T0 + 100 * H4),
    ]
    assert {call[:2] for call in fake.candle_calls} == {("BTC-USDT-SWAP", "4H")}


def test_history_is_complete_whatever_page_size_the_venue_serves() -> None:
    """The venue documents 100 rows a page and serves 300. Reading a short page as the end of
    history would, the day it enforces its own documentation, hand back the newest hundred bars
    of every window with no error - and this venue exists to supply the history."""
    candles = [_row(T0 + i * H4) for i in range(700)]
    for cap, pages in ((300, 3), (100, 7), (37, 19)):
        fake = FakeOkx(candles, cap=cap)
        bars = OkxBars(fake, now=lambda: T0 + timedelta(days=400)).history(BTC, "4h", T0, T0 + 700 * H4)
        assert len(bars) == 700, cap
        assert len(fake.candle_calls) == pages, cap


def test_history_asks_once_more_when_it_runs_out_of_history_before_the_start() -> None:
    """Only an empty page says there is nothing older - e.g. a window opening before the listing."""
    fake = FakeOkx([_row(T0 + i * H4) for i in range(50)])
    bars = OkxBars(fake, now=lambda: T0 + timedelta(days=400)).history(BTC, "4h", T0 - 500 * H4, T0 + 50 * H4)
    assert len(bars) == 50
    assert [call[2] for call in fake.candle_calls] == [_ms(T0 + 50 * H4), _ms(T0)]


def test_history_refuses_a_series_with_a_gap() -> None:
    """The venue has not omitted a candle in four years of five swaps, so a hole is an anomaly
    to fail on, not data to store: nothing downstream checks contiguity."""
    candles = [_row(T0 + i * H4) for i in range(10) if i != 6]
    source = OkxBars(FakeOkx(candles), now=lambda: T0 + timedelta(days=400))
    with pytest.raises(RuntimeError, match=r"gap between 2024-01-01T20:00.*2024-01-02T04:00"):
        source.history(BTC, "4h", T0, T0 + 10 * H4)


def test_history_refuses_a_malformed_row() -> None:
    source = OkxBars(FakeOkx([_row(T0)[:5]]), now=lambda: T0 + timedelta(days=400))
    with pytest.raises(ValueError, match="malformed 4H candle for BTC-USDT-SWAP"):
        source.history(BTC, "4h", T0, T0 + H4)


def test_history_stops_paging_once_it_is_past_the_start() -> None:
    candles = [_row(T0 + i * H4) for i in range(1_000)]
    fake = FakeOkx(candles)
    source = OkxBars(fake, now=lambda: T0 + timedelta(days=400))

    bars = source.history(BTC, "4h", T0 + 900 * H4, T0 + 1_000 * H4)

    assert len(bars) == 100 and bars[0].ts_open == T0 + 900 * H4
    assert len(fake.candle_calls) == 1  # 300 rows reached back past `start`: no second page


def test_history_window_is_half_open_and_needs_no_data_outside_it() -> None:
    candles = [_row(T0 + i * H4) for i in range(10)]
    source = OkxBars(FakeOkx(candles), now=lambda: T0 + timedelta(days=400))
    bars = source.history(BTC, "4h", T0 + 2 * H4, T0 + 5 * H4)
    assert [bar.ts_open for bar in bars] == [T0 + 2 * H4, T0 + 3 * H4, T0 + 4 * H4]
    assert source.history(BTC, "4h", T0 - 30 * H4, T0 - 20 * H4) == []


def test_history_drops_the_candle_that_is_still_forming() -> None:
    """Served with `confirm == "0"`; and whatever the flag says, a candle whose close lies in
    the future has not closed."""
    candles = [_row(T0), _row(T0 + H4), _row(T0 + 2 * H4, confirm="0")]
    source = OkxBars(FakeOkx(candles), now=lambda: T0 + 2 * H4 + timedelta(minutes=5))
    assert [bar.ts_open for bar in source.history(BTC, "4h", T0, T0 + 10 * H4)] == [T0, T0 + H4]

    lying = [_row(T0), _row(T0 + H4, confirm="1")]
    early = OkxBars(FakeOkx(lying), now=lambda: T0 + H4 + timedelta(minutes=5))
    assert [bar.ts_open for bar in early.history(BTC, "4h", T0, T0 + 10 * H4)] == [T0]


@pytest.mark.parametrize(("tf", "bar"), [("1h", "1H"), ("4h", "4H"), ("1d", "1Dutc")])
def test_history_asks_for_the_utc_aligned_bar(tf: str, bar: str) -> None:
    """`1D` opens at 16:00 UTC on this venue; the rest of the system assumes 00:00."""
    fake = FakeOkx([])
    OkxBars(fake, now=lambda: T0).history(BTC, tf, T0 - timedelta(days=5), T0)  # type: ignore[arg-type]
    assert fake.candle_calls[0][1] == bar


def test_history_refuses_a_daily_bar_that_does_not_open_at_midnight_utc() -> None:
    wrong = [_row(T0 + timedelta(hours=16))]
    source = OkxBars(FakeOkx(wrong), now=lambda: T0 + timedelta(days=30))
    with pytest.raises(ValueError, match="UTC"):
        source.history(BTC, "1d", T0, T0 + timedelta(days=5))


def test_history_does_not_loop_on_a_venue_that_ignores_the_cursor() -> None:
    class Stuck(FakeOkx):
        def history_candles(self, inst: str, bar: str, *, after_ms: int | None = None, limit: int = 300):  # type: ignore[no-untyped-def]
            self.candle_calls.append((inst, bar, after_ms, limit))
            return self.candles[:limit]

    fake = Stuck([_row(T0 + i * H4) for i in range(400)])
    bars = OkxBars(fake, now=lambda: T0 + timedelta(days=400)).history(BTC, "4h", T0 - 50 * H4, T0 + 400 * H4)
    assert len(fake.candle_calls) == 2  # the second page made no progress
    assert len(bars) == 300


def test_stream_is_refused_because_the_venue_is_research_only() -> None:
    source = OkxBars(FakeOkx([]))

    async def first() -> None:
        async for _ in source.stream(BTC, "4h"):
            pass

    with pytest.raises(NotImplementedError, match="research"):
        asyncio.run(first())


# --- instruments -------------------------------------------------------------------------------


def test_inst_id_is_the_usdt_margined_swap() -> None:
    assert inst_id(BTC) == "BTC-USDT-SWAP"


def test_okx_instrument_takes_the_venues_tick_and_sizes_in_coins() -> None:
    instrument = okx_instrument(FakeOkx([]), "BTC")
    assert instrument == BTC
    # the venue trades 0.01-BTC contracts; a price-path backtest sizes in coins, as on Hyperliquid
    assert instrument.contract_multiplier == Decimal("1")


def test_okx_instruments_default_to_the_five_the_hyperliquid_run_graded() -> None:
    assert DEFAULT_SWAPS == ("BTC", "ETH", "SOL", "ARB", "HYPE")
    assert [i.symbol for i in okx_instruments(FakeOkx([]))] == list(DEFAULT_SWAPS)
    assert [i.symbol for i in okx_instruments(FakeOkx([]), ["ETH"])] == ["ETH"]


def test_okx_instrument_refuses_a_swap_that_is_not_live() -> None:
    class Suspended(FakeOkx):
        def instrument(self, inst: str) -> dict[str, str]:
            return {**super().instrument(inst), "state": "suspend"}

    with pytest.raises(ValueError, match="suspend"):
        okx_instrument(Suspended([]), "BTC")


# --- costs -------------------------------------------------------------------------------------


def _bar(ts_open: datetime, tf: str = "4h", close: float = 100.0) -> Bar:
    return Bar(
        instrument=BTC, tf=tf, ts_open=ts_open, open=close, high=close, low=close, close=close, volume=1.0
    )  # type: ignore[arg-type]


def _position(direction: int, qty: float = 2.0) -> Position:
    return Position(instrument=BTC, direction=direction, qty=qty, avg_price=100.0, stop=90.0, target=None)  # type: ignore[arg-type]


def test_entry_cost_is_taker_commission_plus_tick_slippage() -> None:
    order = Order(
        id="o1",
        instrument=BTC,
        direction=1,
        kind="limit",
        qty=2.0,
        price=50_000.0,
        expires_at_bar=None,
        leg="entry",
        trade_id="t1",
    )
    cost = OkxCosts().entry(order, _bar(T0))
    assert cost.commission == pytest.approx(0.0005 * 2.0 * 50_000.0)
    assert cost.slippage == pytest.approx(0.5 * 0.1 * 2.0)
    assert cost.spread == 0.0 and cost.funding == 0.0


def test_carry_charges_each_eight_hour_settlement_a_bar_crosses() -> None:
    costs = OkxCosts(funding=[(T0, 0.0002), (T0 + timedelta(hours=8), -0.0001)])
    long, short = _position(1), _position(-1)
    # 04:00-08:00 crosses the 08:00 settlement; 00:00-04:00 crosses none (its own open does not count)
    assert costs.carry(long, _bar(T0)) == 0.0
    assert costs.carry(long, _bar(T0 + H4)) == pytest.approx(-0.0001 * 2.0 * 100.0)
    assert costs.carry(short, _bar(T0 + H4)) == pytest.approx(0.0001 * 2.0 * 100.0)
    # a Daily bar crosses three: 08:00, 16:00 and the next 00:00 - the last two at the latest known rate
    assert costs.carry(long, _bar(T0, "1d")) == pytest.approx(3 * -0.0001 * 2.0 * 100.0)


def test_carry_falls_back_to_the_baseline_rate_before_the_recorded_history() -> None:
    """The venue serves three months of funding; a four-year backtest is charged the venue's
    baseline (0.01% per 8 hours, longs pay) for everything older - a stated assumption, not data."""
    assert DEFAULT_FUNDING_RATE == 0.0001
    recorded_from = T0 + timedelta(days=30)
    costs = OkxCosts(funding=[(recorded_from, 0.0005)])
    assert costs.carry(_position(1), _bar(T0 + H4)) == pytest.approx(0.0001 * 2.0 * 100.0)
    assert costs.carry(_position(1), _bar(recorded_from + H4)) == pytest.approx(0.0005 * 2.0 * 100.0)
    assert OkxCosts().carry(_position(-1), _bar(T0 + H4)) == pytest.approx(-0.0001 * 2.0 * 100.0)
    assert OkxCosts(default_funding_rate=0.0).carry(_position(1), _bar(T0 + H4)) == 0.0


def _funding_row(ts: datetime, rate: str, realized: str = "") -> dict[str, str]:
    return {
        "instId": "BTC-USDT-SWAP",
        "fundingTime": str(_ms(ts)),
        "fundingRate": rate,
        "realizedRate": realized,
    }


def test_load_funding_pages_backwards_and_prefers_the_realized_rate() -> None:
    eight = timedelta(hours=8)
    rows = [_funding_row(T0 + i * eight, "0.0001", realized="0.00012" if i == 0 else "") for i in range(250)]
    fake = FakeOkx([], funding=rows)

    funding = load_funding(fake, "BTC-USDT-SWAP", T0, T0 + 250 * eight)

    assert len(funding) == 250
    assert funding[0] == (T0, 0.00012) and funding[1] == (T0 + eight, 0.0001)
    assert [ts for ts, _ in funding] == sorted(ts for ts, _ in funding)
    assert fake.funding_calls == [_ms(T0 + 250 * eight) + 1, _ms(T0 + 150 * eight), _ms(T0 + 50 * eight)]


def test_load_funding_does_not_read_a_short_page_as_the_end() -> None:
    class SmallPages(FakeOkx):
        def funding_rate_history(self, inst: str, *, after_ms: int | None = None, limit: int = 100):  # type: ignore[no-untyped-def]
            return super().funding_rate_history(inst, after_ms=after_ms, limit=min(limit, 40))

    eight = timedelta(hours=8)
    fake = SmallPages([], funding=[_funding_row(T0 + i * eight, "0.0001") for i in range(100)])
    assert len(load_funding(fake, "BTC-USDT-SWAP", T0, T0 + 100 * eight)) == 100


def test_load_funding_keeps_a_settled_rate_of_exactly_zero() -> None:
    """`realizedRate or fundingRate` must not fall through on a zero: the values are strings, and
    "0" is truthy - which is what this pins, should they ever be parsed earlier."""
    rows = [
        _funding_row(T0, "0.0003", realized="0"),
        _funding_row(T0 + timedelta(hours=8), "0.0003", realized="0.0"),
    ]
    funding = load_funding(FakeOkx([], funding=rows), "BTC-USDT-SWAP", T0, T0 + timedelta(hours=8))
    assert [rate for _, rate in funding] == [0.0, 0.0]


def test_load_funding_keeps_to_the_window_it_was_asked_for() -> None:
    eight = timedelta(hours=8)
    fake = FakeOkx([], funding=[_funding_row(T0 + i * eight, "0.0001") for i in range(30)])
    funding = load_funding(fake, "BTC-USDT-SWAP", T0 + 10 * eight, T0 + 20 * eight)
    assert [ts for ts, _ in funding] == [T0 + i * eight for i in range(10, 21)]
    assert load_funding(FakeOkx([]), "BTC-USDT-SWAP", T0, T0 + eight) == []
