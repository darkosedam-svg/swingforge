"""Tests for `swingforge.cli`: the composition root.

Every venue-facing boundary is a fake (a fake `BarSource`, `Info`, entry factory); nothing
here touches a network. `typer.testing.CliRunner` drives the commands end to end; a handful
of module-level helpers are also exercised directly for speed and precision.
"""

from __future__ import annotations

import asyncio
import collections
import sys
import types
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import duckdb
import pytest
from typer.testing import CliRunner

from swingforge import cli
from swingforge.adapters.hyperliquid.bars import DEFAULT_PERPS, HyperliquidBars
from swingforge.adapters.oanda.bars import OandaBars
from swingforge.adapters.oanda.costs import OandaCosts
from swingforge.adapters.paper import PaperBroker
from swingforge.adapters.settings_reader import TransientSettingsReader
from swingforge.adapters.store import Store
from swingforge.cli import PerInstrumentCostModel, app
from swingforge.core.costs import NullCostModel
from swingforge.core.engine import Engine
from swingforge.core.fills import FillResolver
from swingforge.core.portfolio import Portfolio
from swingforge.core.settings import StaticSettingsReader
from swingforge.core.types import (
    Bar,
    CostBreakdown,
    Fill,
    Instrument,
    Order,
    Signal,
    Trade,
)
from swingforge.strategies.baseline import Baseline
from swingforge.strategies.exits import EXIT_GRID
from swingforge.strategies.ict import ICT
from swingforge.strategies.zones import Zones
from tests.unit.synth_store import add_instrument

runner = CliRunner()

BTC = Instrument(
    venue="hyperliquid",
    symbol="BTC",
    tick_size=Decimal("0.5"),
    contract_multiplier=Decimal("1"),
    quote_ccy="USD",
    session_profile="perp",
)
EUR_USD = Instrument(
    venue="oanda",
    symbol="EUR_USD",
    tick_size=Decimal("0.00001"),
    contract_multiplier=Decimal("1"),
    quote_ccy="USD",
    session_profile="fx",
)
TS0 = datetime(2026, 1, 1, tzinfo=UTC)


def _bar(
    ts_open: datetime,
    *,
    tf: str = "4h",
    open_: float = 100.0,
    high: float = 105.0,
    low: float = 95.0,
    close: float = 101.0,
    instrument: Instrument = BTC,
) -> Bar:
    return Bar(
        instrument=instrument,
        tf=tf,
        ts_open=ts_open,
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=10.0,
    )


# --- small pure helpers --------------------------------------------------------------


def test_split_trims_and_drops_empty_items() -> None:
    assert cli._split(" a, b ,,c ") == ["a", "b", "c"]


def test_parse_dt_naive_becomes_utc() -> None:
    parsed = cli._parse_dt("2026-01-01T00:00:00")
    assert parsed.tzinfo is UTC


def test_parse_dt_keeps_an_explicit_offset() -> None:
    parsed = cli._parse_dt("2026-01-01T00:00:00+02:00")
    assert parsed.utcoffset() == timedelta(hours=2)


def test_windows_splits_into_bounded_chunks() -> None:
    start = datetime(2020, 1, 1, tzinfo=UTC)
    end = datetime(2022, 6, 1, tzinfo=UTC)
    windows = cli._windows(start, end, timedelta(days=365))
    assert windows[0][0] == start
    assert windows[-1][1] == end
    assert all(a < b for a, b in windows)
    assert all(b - a <= timedelta(days=365) for a, b in windows)


def test_filesystem_safe_replaces_colons_only() -> None:
    assert (
        cli._filesystem_safe("tournament:hyperliquid:20260101T0000") == "tournament-hyperliquid-20260101T0000"
    )


def test_parse_config_id_splits_four_parts() -> None:
    assert cli._parse_config_id("ict|fixed_r_2|none|hyperliquid:BTC") == (
        "ict",
        "fixed_r_2",
        "none",
        "hyperliquid:BTC",
    )


@pytest.mark.parametrize("bad", ["ict|fixed_r_2|none", "ict|fixed_r_2|none|hyperliquid-BTC"])
def test_parse_config_id_rejects_malformed_ids(bad: str) -> None:
    with pytest.raises(ValueError, match="invalid config id"):
        cli._parse_config_id(bad)


def test_find_instrument_matches_on_symbol() -> None:
    store = Store(":memory:")
    store.upsert_instruments([BTC])
    try:
        found = cli._find_instrument(store, "hyperliquid", "hyperliquid:BTC")
        assert found.symbol == "BTC"
        with pytest.raises(ValueError, match="not found"):
            cli._find_instrument(store, "hyperliquid", "hyperliquid:ETH")
        with pytest.raises(ValueError, match="does not match"):
            cli._find_instrument(store, "oanda", "hyperliquid:BTC")
    finally:
        store.close()


def test_chronological_orders_by_close_time() -> None:
    four_hour = [_bar(TS0), _bar(TS0 + timedelta(hours=4))]
    daily = [_bar(TS0 - timedelta(hours=24), tf="1d", high=110.0, low=90.0)]
    merged = cli._chronological(four_hour, daily)
    assert [bar.tf for bar in merged] == ["1d", "4h", "4h"]


class _NeverSignals:
    name = "none"

    def on_bar(self, ctx):  # type: ignore[no-untyped-def]
        return None


class _WarmSource:
    """A `BarSource` fake returning fixed history regardless of the requested window."""

    def __init__(self, daily: list[Bar], four_hour: list[Bar]) -> None:
        self._by_tf = {"1d": daily, "4h": four_hour}

    def history(self, instrument: Instrument, tf: str, start: datetime, end: datetime) -> list[Bar]:
        return self._by_tf.get(tf, [])


def test_warm_up_pushes_bars_directly_into_context_bypassing_broker_and_strategy() -> None:
    daily = [_bar(TS0 - timedelta(hours=24), tf="1d", high=110.0, low=90.0)]
    four_hour = [_bar(TS0), _bar(TS0 + timedelta(hours=4))]
    source = _WarmSource(daily, four_hour)
    engine = Engine(
        BTC,
        _NeverSignals(),
        EXIT_GRID[0],
        PaperBroker(NullCostModel(), FillResolver()),
        Portfolio(10_000.0),
        StaticSettingsReader(),
    )

    async def _run() -> None:
        await cli._warm_up(engine, source, BTC)

    asyncio.run(_run())

    assert engine.ctx.bar_index == 1  # two 4H bars pushed directly, zero-indexed
    assert len(engine.ctx.history("1d")) == 1
    assert engine.portfolio.trades == []  # no broker/strategy involvement: no phantom trades


def test_warm_up_ignores_a_daily_bars_own_subbars() -> None:
    """Minor fix: only a 4H bar's own subbars are pushed here. A Daily bar may also legally
    carry subbars (`Bar` allows "4h" as a Daily subbar tf) -- but those would be the very 4H
    bars already pushed separately via `four_hour`, so pushing them again would double-count
    them and inflate `ctx.bar_index`."""
    daily_ts = TS0 - timedelta(hours=24)
    embedded_4h = _bar(daily_ts, high=105.0, low=95.0)  # a legal, but unwanted, Daily subbar
    daily = [
        Bar(
            instrument=BTC,
            tf="1d",
            ts_open=daily_ts,
            open=100.0,
            high=110.0,
            low=90.0,
            close=101.0,
            volume=10.0,
            subbars=(embedded_4h,),
        )
    ]
    four_hour = [_bar(TS0), _bar(TS0 + timedelta(hours=4))]
    source = _WarmSource(daily, four_hour)
    engine = Engine(
        BTC,
        _NeverSignals(),
        EXIT_GRID[0],
        PaperBroker(NullCostModel(), FillResolver()),
        Portfolio(10_000.0),
        StaticSettingsReader(),
    )

    async def _run() -> None:
        await cli._warm_up(engine, source, BTC)

    asyncio.run(_run())

    # Only the two real 4H bars ever advance `bar_index` -- the Daily bar's embedded "4h"
    # subbar must not have been pushed a second time.
    assert engine.ctx.bar_index == 1


def test_drain_fill_log_returns_only_new_entries() -> None:
    broker = PaperBroker(NullCostModel(), FillResolver())
    order = Order(
        id="e1",
        instrument=BTC,
        direction=1,
        qty=1.0,
        kind="market",
        price=None,
        expires_at_bar=None,
        leg="entry",
        trade_id="t1",
    )
    broker.submit(order)
    broker.on_bar(_bar(TS0), 0)

    fills, cursor = cli._drain_fill_log(broker, None)
    assert len(fills) == 1
    assert cursor is broker.log[-1]

    fills_again, cursor_again = cli._drain_fill_log(broker, cursor)
    assert fills_again == []
    assert cursor_again == cursor


def test_drain_fill_log_survives_the_broker_log_evicting_past_entries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """I1: `broker.log`'s bound (`_MAX_LOG_ENTRIES`) drops the oldest entries once exceeded.

    A plain integer/length cursor breaks against that (the deque's length freezes once
    full, so `log[cursor:]` would silently stop returning anything new). Draining once per
    bar, as `paper` does, keeps up with a small `maxlen` -- every fill produced is still
    echoed and returned before it is ever evicted.
    """
    broker = PaperBroker(NullCostModel(), FillResolver())
    monkeypatch.setattr(broker, "log", collections.deque(maxlen=3))

    cursor = None
    drained: list[Fill] = []
    for i in range(5):
        order = Order(
            id=f"e{i}",
            instrument=BTC,
            direction=1,
            qty=1.0,
            kind="market",
            price=None,
            expires_at_bar=None,
            leg="entry",
            trade_id=f"t{i}",
        )
        broker.submit(order)
        broker.on_bar(_bar(TS0 + timedelta(hours=4 * i)), i)
        fills, cursor = cli._drain_fill_log(broker, cursor)
        drained.extend(fills)

    assert len(drained) == 5
    assert [fill.trade_id for fill in drained] == ["t0", "t1", "t2", "t3", "t4"]


def test_persist_bar_writes_fills_open_trade_and_equity(tmp_path) -> None:
    store_path = tmp_path / "v.duckdb"
    store = Store(store_path)
    store.upsert_instruments([BTC])
    store.close()

    fill = Fill(order_id="o1", ts=TS0, price=100.0, qty=1.0, cost=CostBreakdown(), leg="entry", trade_id="t1")
    open_trade = Trade(
        id="t1",
        instrument=BTC,
        direction=1,
        entry_fill=fill,
        stop=95.0,
        target=None,
        risk_r=5.0,
        opened_bar=0,
    )

    cli._persist_bar(store_path, "run1", [fill], open_trade, [], (TS0, 10_000.0))

    store = Store(store_path, read_only=True)
    trades = store.trades("run1")
    equity = store.equity("run1")
    store.close()

    assert len(trades) == 1
    assert trades[0].closed_bar is None
    assert trades[0].realized_r is None
    assert equity == [(TS0, 10_000.0)]


def test_persist_bar_rolls_back_the_whole_bar_on_a_failed_write(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """C3: `_persist_bar`'s fills/trades/equity writes for one bar run inside a single
    transaction, so a kill (or any failure) between writes can never leave a trade row
    without its matching equity row -- simulated here by a `write_equity` that raises once.
    The failed bar's writes all roll back together; a retry of the same bar lands them all.
    """
    store_path = tmp_path / "v.duckdb"
    store = Store(store_path)
    store.upsert_instruments([BTC])
    store.close()

    fill = Fill(order_id="o1", ts=TS0, price=100.0, qty=1.0, cost=CostBreakdown(), leg="entry", trade_id="t1")
    open_trade = Trade(
        id="t1",
        instrument=BTC,
        direction=1,
        entry_fill=fill,
        stop=95.0,
        target=None,
        risk_r=5.0,
        opened_bar=0,
    )

    real_write_equity = Store.write_equity
    calls = {"n": 0}

    def flaky_write_equity(self, run_id, rows):  # type: ignore[no-untyped-def]
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("simulated failure between writes")
        return real_write_equity(self, run_id, rows)

    monkeypatch.setattr(Store, "write_equity", flaky_write_equity)

    with pytest.raises(RuntimeError, match="simulated failure between writes"):
        cli._persist_bar(store_path, "run1", [fill], open_trade, [], (TS0, 10_000.0))

    store = Store(store_path, read_only=True)
    trades_after_failure = store.trades("run1")
    equity_after_failure = store.equity("run1")
    store.close()
    assert trades_after_failure == []  # rolled back together with the failed equity write
    assert equity_after_failure == []

    cli._persist_bar(store_path, "run1", [fill], open_trade, [], (TS0, 10_000.0))

    store = Store(store_path, read_only=True)
    trades_after_retry = store.trades("run1")
    equity_after_retry = store.equity("run1")
    store.close()
    assert len(trades_after_retry) == 1
    assert equity_after_retry == [(TS0, 10_000.0)]


# --- venue wiring: _bar_source / _instruments / _funding / _build_cost_model --------


def test_bar_source_builds_the_right_class(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OANDA_TOKEN", raising=False)
    assert isinstance(cli._bar_source("hyperliquid"), HyperliquidBars)
    assert isinstance(cli._bar_source("oanda"), OandaBars)
    with pytest.raises(ValueError, match="unknown venue"):
        cli._bar_source("bogus")


def test_instruments_oanda_defaults_to_the_six_static_symbols() -> None:
    result = cli._instruments("oanda", OandaBars(api=object()), None)
    assert {inst.symbol for inst in result} == set(cli._OANDA_SIX)
    assert all(inst.venue == "oanda" for inst in result)


def test_instruments_oanda_override_uses_the_given_symbols() -> None:
    result = cli._instruments("oanda", OandaBars(api=object()), "EUR_USD,GBP_USD")
    assert [inst.symbol for inst in result] == ["EUR_USD", "GBP_USD"]


class _FakeHLInfo:
    def candles_snapshot(self, name, interval, startTime, endTime):  # noqa: N803
        return []

    def funding_history(self, name, startTime, endTime=None):  # noqa: N803
        return [{"time": 1_700_000_000_000, "fundingRate": "0.0001"}]

    def meta_and_asset_ctxs(self):
        universe = [
            {"name": "BTC", "szDecimals": 5},
            {"name": "ETH", "szDecimals": 4},
            {"name": "SOL", "szDecimals": 3},
            {"name": "DOGE", "szDecimals": 1},
        ]
        ctxs = [
            {"dayNtlVlm": "1000", "markPx": "60000"},
            {"dayNtlVlm": "900", "markPx": "3000"},
            {"dayNtlVlm": "800", "markPx": "150"},
            {"dayNtlVlm": "1200", "markPx": "0.2"},
        ]
        return {"universe": universe}, ctxs


def test_instruments_hyperliquid_default_ranks_via_hl_instruments() -> None:
    source = HyperliquidBars(info=_FakeHLInfo())
    result = cli._instruments("hyperliquid", source, None)
    assert {"BTC", "ETH", "SOL"} <= {inst.symbol for inst in result}


def test_instruments_hyperliquid_override_falls_back_to_a_conservative_tick() -> None:
    source = HyperliquidBars(info=_FakeHLInfo())
    result = cli._instruments("hyperliquid", source, "BTC,DOGE")
    by_symbol = {inst.symbol: inst for inst in result}
    assert by_symbol["BTC"].tick_size == DEFAULT_PERPS["BTC"]
    assert by_symbol["DOGE"].tick_size == cli._HL_FALLBACK_TICK


def test_funding_is_always_empty_for_oanda() -> None:
    assert cli._funding("oanda", OandaBars(api=object()), EUR_USD, TS0, TS0) == []


def test_funding_hyperliquid_loads_from_info() -> None:
    source = HyperliquidBars(info=_FakeHLInfo())
    rows = cli._funding("hyperliquid", source, BTC, TS0, TS0 + timedelta(days=1))
    assert rows == [(datetime.fromtimestamp(1_700_000_000, tz=UTC), 0.0001)]


def test_build_cost_model_hyperliquid_is_per_instrument_dispatcher() -> None:
    store = Store(":memory:")
    try:
        store.upsert_instruments([BTC])
        model = cli._build_cost_model("hyperliquid", store, [BTC])
        assert isinstance(model, PerInstrumentCostModel)
        bar = _bar(TS0)
        order = Order(
            id="e1",
            instrument=BTC,
            direction=1,
            qty=1.0,
            kind="market",
            price=None,
            expires_at_bar=None,
            leg="entry",
            trade_id=None,
        )
        cost = model.entry(order, bar)
        assert cost.commission > 0  # HyperliquidCosts charges a taker fee
    finally:
        store.close()


def test_per_instrument_cost_model_raises_for_an_unconfigured_instrument() -> None:
    model = PerInstrumentCostModel({"hyperliquid:BTC": NullCostModel()})
    order = Order(
        id="e1",
        instrument=EUR_USD,
        direction=1,
        qty=1.0,
        kind="market",
        price=None,
        expires_at_bar=None,
        leg="entry",
        trade_id=None,
    )
    with pytest.raises(LookupError, match="no cost model configured") as excinfo:
        model.entry(order, _bar(TS0, instrument=EUR_USD))
    assert not isinstance(excinfo.value, KeyError)  # a clean message, not KeyError's double-quoted repr


def test_build_cost_model_oanda_without_credentials_has_zero_swap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OANDA_ACCOUNT_ID", raising=False)
    monkeypatch.delenv("OANDA_TOKEN", raising=False)
    store = Store(":memory:")
    try:
        model = cli._build_cost_model("oanda", store, [EUR_USD])
        assert isinstance(model, OandaCosts)
        assert model.financing == {}
    finally:
        store.close()


def test_build_cost_model_oanda_with_credentials_loads_financing(monkeypatch: pytest.MonkeyPatch) -> None:
    from swingforge.adapters.oanda.costs import Financing

    monkeypatch.setenv("OANDA_ACCOUNT_ID", "acc123")
    monkeypatch.setenv("OANDA_TOKEN", "tok123")
    monkeypatch.setattr(cli, "_oanda_api", lambda token, environment: object())
    monkeypatch.setattr(
        cli, "load_financing", lambda api, account_id, symbol: Financing(long_rate=0.01, short_rate=-0.02)
    )
    store = Store(":memory:")
    try:
        model = cli._build_cost_model("oanda", store, [EUR_USD])
        assert isinstance(model, OandaCosts)
        assert model.financing["EUR_USD"].long_rate == 0.01
    finally:
        store.close()


def test_entry_factory_builds_the_three_known_entries() -> None:
    assert isinstance(cli._entry_factory("ict", BTC, baseline_rate=None, seed=0), ICT)
    assert isinstance(cli._entry_factory("zones", BTC, baseline_rate=None, seed=0), Zones)
    default_rate = cli._entry_factory("baseline", BTC, baseline_rate=None, seed=5)
    assert isinstance(default_rate, Baseline)
    assert default_rate.target_trades_per_1000_bars == 1.0
    custom_rate = cli._entry_factory("baseline", BTC, baseline_rate=2.5, seed=5)
    assert custom_rate.target_trades_per_1000_bars == 2.5


def test_entry_factory_rejects_an_unknown_name() -> None:
    with pytest.raises(ValueError, match="unknown entry"):
        cli._entry_factory("bogus", BTC, baseline_rate=None, seed=0)


def test_session_factory_builds_a_working_predicate() -> None:
    predicate = cli._session_factory("none", "perp")
    assert predicate(_bar(TS0)) is True


# --- backfill --------------------------------------------------------------------


class _FakeBackfillSource:
    """A `BarSource` fake returning canned bars for `backfill`'s tests.

    Ignores `start`/`end` and always hands back its whole canned set: the bars are dated
    relative to a fixed, arbitrary synthetic clock, not to the real `datetime.now(UTC)`
    `backfill` computes its window from, so filtering by the requested window would starve
    every call. `--years 1` keeps `backfill` to a single request window per timeframe
    anyway, so nothing here is asked for the same bars twice.
    """

    def __init__(self, bars_by_tf: dict[str, list[Bar]]) -> None:
        self._bars_by_tf = bars_by_tf

    def history(self, instrument: Instrument, tf: str, start: datetime, end: datetime) -> list[Bar]:
        return list(self._bars_by_tf.get(tf, []))


def _synthetic_bars_by_tf(
    instrument: Instrument, *, start: datetime, months: int, seed: int
) -> dict[str, list[Bar]]:
    """Consistent 1h/4h/1d bars for `instrument`, built via the shared synth-store helpers."""
    store = Store(":memory:")
    add_instrument(store, instrument, start=start, months=months, seed=seed)
    far_past = datetime(2000, 1, 1, tzinfo=UTC)
    far_future = datetime(2035, 1, 1, tzinfo=UTC)
    bars_by_tf = {tf: store.bars(instrument, tf, far_past, far_future) for tf in ("1h", "4h", "1d")}
    store.close()
    return bars_by_tf


def test_backfill_is_idempotent_and_reports_months(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    bars_by_tf = _synthetic_bars_by_tf(BTC, start=datetime(2024, 1, 1, tzinfo=UTC), months=3, seed=1)
    source = _FakeBackfillSource(bars_by_tf)
    monkeypatch.setattr(cli, "_bar_source", lambda venue, **kw: source)
    monkeypatch.setattr(cli, "_instruments", lambda venue, src, override: [BTC])
    monkeypatch.setattr(cli, "_funding", lambda venue, src, instrument, start, end: [])

    data_dir = tmp_path / "data"
    args = ["backfill", "--venue", "hyperliquid", "--years", "1", "--data-dir", str(data_dir)]

    first = runner.invoke(app, args)
    assert first.exit_code == 0, first.output
    assert "months=" in first.output

    store_path = data_dir / "hyperliquid.duckdb"
    store = Store(store_path, read_only=True)
    count_1h_first = store.bar_range(BTC, "1h")[2]
    count_4h_first = store.bar_range(BTC, "4h")[2]
    store.close()
    assert count_1h_first > 0

    second = runner.invoke(app, args)
    assert second.exit_code == 0, second.output

    store = Store(store_path, read_only=True)
    count_1h_second = store.bar_range(BTC, "1h")[2]
    count_4h_second = store.bar_range(BTC, "4h")[2]
    store.close()

    assert count_1h_second == count_1h_first
    assert count_4h_second == count_4h_first


def test_backfill_instruments_override_is_forwarded(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    bars_by_tf = _synthetic_bars_by_tf(BTC, start=datetime(2024, 1, 1, tzinfo=UTC), months=1, seed=2)
    source = _FakeBackfillSource(bars_by_tf)
    captured: dict[str, str | None] = {}

    def fake_instruments(venue: str, src: object, override: str | None) -> list[Instrument]:
        captured["override"] = override
        return [BTC]

    monkeypatch.setattr(cli, "_bar_source", lambda venue, **kw: source)
    monkeypatch.setattr(cli, "_instruments", fake_instruments)
    monkeypatch.setattr(cli, "_funding", lambda venue, src, instrument, start, end: [])

    result = runner.invoke(
        app,
        [
            "backfill",
            "--venue",
            "hyperliquid",
            "--years",
            "1",
            "--instruments",
            "BTC",
            "--data-dir",
            str(tmp_path / "data"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured["override"] == "BTC"


def test_backfill_rejects_an_unknown_venue() -> None:
    result = runner.invoke(app, ["backfill", "--venue", "bogus", "--years", "1"])
    assert result.exit_code != 0


def test_backfill_reports_a_clean_error_with_no_instruments(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setattr(cli, "_bar_source", lambda venue, **kw: _FakeBackfillSource({}))
    monkeypatch.setattr(cli, "_instruments", lambda venue, src, override: [])
    result = runner.invoke(
        app, ["backfill", "--venue", "hyperliquid", "--years", "1", "--data-dir", str(tmp_path / "data")]
    )
    assert result.exit_code == 1
    assert "error:" in result.output


# --- tournament --------------------------------------------------------------------

SYNTH_START = datetime(2021, 1, 1, tzinfo=UTC)
SYNTH_MONTHS = 20


def _seeded_store(tmp_path, *, months: int = SYNTH_MONTHS):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    store_path = data_dir / "hyperliquid.duckdb"
    store = Store(store_path)
    add_instrument(store, BTC, start=SYNTH_START, months=months, seed=7)
    store.close()
    return data_dir, store_path


def test_tournament_command_runs_and_writes_a_report(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    data_dir, store_path = _seeded_store(tmp_path)
    monkeypatch.setattr(cli, "_tournament_run_id", lambda venue: "tournament:hyperliquid:fixed")
    out_dir = tmp_path / "reports"

    result = runner.invoke(
        app,
        [
            "tournament",
            "--venue",
            "hyperliquid",
            "--out",
            str(out_dir),
            "--entries",
            "baseline",
            "--exits",
            "fixed_r_2",
            "--sessions",
            "none",
            "--data-dir",
            str(data_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "pooled rows:" in result.output

    reports = list(out_dir.glob("*.md"))
    assert len(reports) == 1
    assert reports[0].name == "tournament-hyperliquid-fixed.md"

    store = Store(store_path, read_only=True)
    rows = store.results("tournament:hyperliquid:fixed")
    store.close()
    assert any(row["split"] == "pooled" for row in rows)


def test_tournament_resume_replays_nothing(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    from swingforge.lab import tournament as tournament_module

    data_dir, _store_path = _seeded_store(tmp_path)
    args = [
        "tournament",
        "--venue",
        "hyperliquid",
        "--out",
        str(tmp_path / "reports"),
        "--entries",
        "baseline",
        "--exits",
        "fixed_r_2",
        "--sessions",
        "none",
        "--run-id",
        "tournament:hyperliquid:fixed",
        "--data-dir",
        str(data_dir),
    ]

    first = runner.invoke(app, args)
    assert first.exit_code == 0, first.output

    replayed: list[str] = []
    original_run_config = tournament_module.run_config

    def counting(config, *a, **kw):  # type: ignore[no-untyped-def]
        replayed.append(config.id)
        return original_run_config(config, *a, **kw)

    monkeypatch.setattr(tournament_module, "run_config", counting)

    second = runner.invoke(app, [*args, "--resume"])
    assert second.exit_code == 0, second.output
    assert replayed == []


def test_tournament_resume_without_run_id_is_a_clean_error(tmp_path) -> None:
    data_dir, _store_path = _seeded_store(tmp_path)
    result = runner.invoke(
        app,
        [
            "tournament",
            "--venue",
            "hyperliquid",
            "--resume",
            "--data-dir",
            str(data_dir),
        ],
    )
    assert result.exit_code == 2, result.output
    assert "--resume requires --run-id" in result.output


def test_tournament_run_id_option_is_used_verbatim(tmp_path) -> None:
    data_dir, store_path = _seeded_store(tmp_path)
    result = runner.invoke(
        app,
        [
            "tournament",
            "--venue",
            "hyperliquid",
            "--out",
            str(tmp_path / "reports"),
            "--entries",
            "baseline",
            "--exits",
            "fixed_r_2",
            "--sessions",
            "none",
            "--run-id",
            "tournament:hyperliquid:pinned",
            "--data-dir",
            str(data_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "run id: tournament:hyperliquid:pinned" in result.output

    store = Store(store_path, read_only=True)
    rows = store.results("tournament:hyperliquid:pinned")
    store.close()
    assert any(row["split"] == "pooled" for row in rows)


def test_tournament_prints_run_id_and_passing_configs(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    data_dir, _store_path = _seeded_store(tmp_path)
    monkeypatch.setattr(cli, "_tournament_run_id", lambda venue: "tournament:hyperliquid:printed")
    result = runner.invoke(
        app,
        [
            "tournament",
            "--venue",
            "hyperliquid",
            "--out",
            str(tmp_path / "reports"),
            "--entries",
            "baseline",
            "--exits",
            "fixed_r_2",
            "--sessions",
            "none",
            "--data-dir",
            str(data_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()
    assert "run id: tournament:hyperliquid:printed" in lines
    assert "passing configs:" in lines


def test_tournament_instruments_narrows_to_the_given_symbols(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    store_path = data_dir / "hyperliquid.duckdb"
    store = Store(store_path)
    add_instrument(store, BTC, start=SYNTH_START, months=SYNTH_MONTHS, seed=7)
    eth = Instrument(
        venue="hyperliquid",
        symbol="ETH",
        tick_size=Decimal("0.05"),
        contract_multiplier=Decimal("1"),
        quote_ccy="USD",
        session_profile="perp",
    )
    add_instrument(store, eth, start=SYNTH_START, months=SYNTH_MONTHS, seed=8)
    store.close()

    result = runner.invoke(
        app,
        [
            "tournament",
            "--venue",
            "hyperliquid",
            "--out",
            str(tmp_path / "reports"),
            "--entries",
            "baseline",
            "--exits",
            "fixed_r_2",
            "--sessions",
            "none",
            "--instruments",
            "BTC",
            "--data-dir",
            str(data_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "hyperliquid:ETH" not in result.output


def test_tournament_instruments_rejects_an_unknown_symbol(tmp_path) -> None:
    data_dir, _store_path = _seeded_store(tmp_path)
    result = runner.invoke(
        app,
        [
            "tournament",
            "--venue",
            "hyperliquid",
            "--instruments",
            "BTC,NOPE",
            "--data-dir",
            str(data_dir),
        ],
    )
    assert result.exit_code == 2, result.output
    assert "unknown instrument" in result.output
    assert "NOPE" in result.output


def test_tournament_reports_no_instruments_error(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    store = Store(data_dir / "hyperliquid.duckdb")
    store.close()
    result = runner.invoke(app, ["tournament", "--venue", "hyperliquid", "--data-dir", str(data_dir)])
    assert result.exit_code == 1
    assert "error:" in result.output


def test_tournament_requires_an_existing_store(tmp_path) -> None:
    result = runner.invoke(
        app, ["tournament", "--venue", "hyperliquid", "--data-dir", str(tmp_path / "nope")]
    )
    assert result.exit_code == 1
    assert "run backfill first" in result.output


# --- paper --------------------------------------------------------------------


class _FakePaperSource:
    """A `BarSource` fake: a finite `stream()` plus empty `history()` (warm-up is a no-op)."""

    def __init__(self, stream_bars: list[Bar], on_before_yield=None) -> None:
        self._stream_bars = stream_bars
        self._on_before_yield = on_before_yield

    def history(self, instrument: Instrument, tf: str, start: datetime, end: datetime) -> list[Bar]:
        return []

    async def stream(self, instrument: Instrument, tf: str):
        for bar in self._stream_bars:
            if self._on_before_yield is not None:
                self._on_before_yield()
            yield bar


class _SignalsOnce:
    """Signals long on the first `on_bar` call while flat, never again."""

    name = "signals_once"

    def __init__(self) -> None:
        self.calls = 0

    def on_bar(self, ctx):  # type: ignore[no-untyped-def]
        self.calls += 1
        if self.calls > 1:
            return None
        bar = ctx.last("4h")
        return Signal(
            direction=1,
            entry=bar.close,
            stop=bar.close - 2.0,
            structure_target=None,
            tag="t",
            expires_in_bars=3,
        )


def _paper_stream_bars() -> list[Bar]:
    """3 bars, tight enough that once the trade opens (stop 98.0 / target 104.0 -- see
    `_SignalsOnce` and `fixed_r_2`), neither level is touched, so the trade is still open
    when the stream ends."""
    return [
        _bar(TS0, open_=100.0, high=101.0, low=99.0, close=100.0),  # hour == 0: exercises the daily fetch
        _bar(TS0 + timedelta(hours=4), open_=100.0, high=101.0, low=99.0, close=100.5),
        _bar(TS0 + timedelta(hours=8), open_=100.5, high=101.5, low=99.5, close=101.0),
    ]


def _seeded_instrument_store(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    store_path = data_dir / "hyperliquid.duckdb"
    store = Store(store_path)
    store.upsert_instruments([BTC])
    store.close()
    return data_dir, store_path


def test_paper_writes_fills_open_trade_and_equity(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    data_dir, store_path = _seeded_instrument_store(tmp_path)
    source = _FakePaperSource(_paper_stream_bars())
    monkeypatch.setattr(cli, "_bar_source", lambda venue, **kw: source)
    monkeypatch.setattr(
        cli, "_entry_factory", lambda name, instrument, *, baseline_rate, seed: _SignalsOnce()
    )

    config_id = "signals_once|fixed_r_2|none|hyperliquid:BTC"
    result = runner.invoke(
        app, ["paper", "--venue", "hyperliquid", "--config", config_id, "--data-dir", str(data_dir)]
    )
    assert result.exit_code == 0, result.output
    assert "fill entry" in result.output
    assert f"run id: paper:hyperliquid:{config_id}" in result.output  # I4

    run_id = f"paper:hyperliquid:{config_id}"
    store = Store(store_path, read_only=True)
    trades = store.trades(run_id)
    equity = store.equity(run_id)
    store.close()

    assert len(trades) == 1
    assert trades[0].closed_bar is None
    assert trades[0].realized_r is None
    assert len(equity) == 3  # one point per 4H bar processed


def test_paper_refuses_a_baseline_config(tmp_path) -> None:
    data_dir, _store_path = _seeded_instrument_store(tmp_path)
    result = runner.invoke(
        app,
        [
            "paper",
            "--venue",
            "hyperliquid",
            "--config",
            "baseline|fixed_r_2|none|hyperliquid:BTC",
            "--data-dir",
            str(data_dir),
        ],
    )
    assert result.exit_code == 2, result.output
    assert "baseline is the control and never trades paper" in result.output


def test_paper_trade_ids_stay_monotonic_across_restarts(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """C1: `Engine` builds a `trade_id` from `ctx.bar_index`, so a second `paper` process for
    the same run id must never restart that clock from scratch -- it would reuse a
    `trade_id` the first process already used. The `equity` table (one row per live bar,
    written nowhere else) is that clock's persistent record across a restart."""
    data_dir, store_path = _seeded_instrument_store(tmp_path)
    monkeypatch.setattr(cli, "_bar_source", lambda venue, **kw: _FakePaperSource(_paper_stream_bars()))
    monkeypatch.setattr(
        cli, "_entry_factory", lambda name, instrument, *, baseline_rate, seed: _SignalsOnce()
    )

    config_id = "signals_once|fixed_r_2|none|hyperliquid:BTC"
    run_id = f"paper:hyperliquid:{config_id}"
    args = ["paper", "--venue", "hyperliquid", "--config", config_id, "--data-dir", str(data_dir)]

    first = runner.invoke(app, args)
    assert first.exit_code == 0, first.output
    store = Store(store_path, read_only=True)
    trades_after_first = store.trades(run_id)
    store.close()
    assert len(trades_after_first) == 1
    first_index = int(trades_after_first[0].id.rsplit(":", 1)[-1])
    bars_processed = len(_paper_stream_bars())

    # The trade opened by the first run is still open when its stream ends (see
    # `_paper_stream_bars`); abandon it (C2) so the second run is free to start.
    second = runner.invoke(app, [*args, "--abandon-open-trade"])
    assert second.exit_code == 0, second.output

    store = Store(store_path, read_only=True)
    trades_after_second = store.trades(run_id)
    store.close()

    assert len(trades_after_second) == 2  # two distinct trade rows accumulate
    ids = {trade.id for trade in trades_after_second}
    assert len(ids) == 2  # ids differ
    second_index = max(int(trade.id.rsplit(":", 1)[-1]) for trade in trades_after_second)
    assert second_index == first_index + bars_processed


def test_paper_refuses_to_start_with_an_orphaned_open_trade(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    data_dir, store_path = _seeded_instrument_store(tmp_path)
    monkeypatch.setattr(cli, "_bar_source", lambda venue, **kw: _FakePaperSource(_paper_stream_bars()))
    monkeypatch.setattr(
        cli, "_entry_factory", lambda name, instrument, *, baseline_rate, seed: _SignalsOnce()
    )
    config_id = "signals_once|fixed_r_2|none|hyperliquid:BTC"
    run_id = f"paper:hyperliquid:{config_id}"
    args = ["paper", "--venue", "hyperliquid", "--config", config_id, "--data-dir", str(data_dir)]

    first = runner.invoke(app, args)
    assert first.exit_code == 0, first.output
    store = Store(store_path, read_only=True)
    open_trade = next(t for t in store.trades(run_id) if t.closed_bar is None)
    store.close()

    second = runner.invoke(app, args)
    assert second.exit_code == 3, second.output
    assert open_trade.id in second.output
    assert open_trade.entry_fill.ts.isoformat() in second.output


def test_paper_abandon_open_trade_closes_it_at_0r_and_starts(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    data_dir, store_path = _seeded_instrument_store(tmp_path)
    monkeypatch.setattr(cli, "_bar_source", lambda venue, **kw: _FakePaperSource(_paper_stream_bars()))
    monkeypatch.setattr(
        cli, "_entry_factory", lambda name, instrument, *, baseline_rate, seed: _SignalsOnce()
    )
    config_id = "signals_once|fixed_r_2|none|hyperliquid:BTC"
    run_id = f"paper:hyperliquid:{config_id}"
    args = ["paper", "--venue", "hyperliquid", "--config", config_id, "--data-dir", str(data_dir)]

    first = runner.invoke(app, args)
    assert first.exit_code == 0, first.output
    store = Store(store_path, read_only=True)
    open_trade = next(t for t in store.trades(run_id) if t.closed_bar is None)
    store.close()

    second = runner.invoke(app, [*args, "--abandon-open-trade"])
    assert second.exit_code == 0, second.output
    assert f"abandoned open trade {open_trade.id} at 0R" in second.output

    store = Store(store_path, read_only=True)
    trades = store.trades(run_id)
    store.close()
    closed = next(t for t in trades if t.id == open_trade.id)
    assert closed.closed_bar == open_trade.opened_bar
    assert closed.realized_r == 0.0


def test_paper_rejects_an_unknown_exit_name(tmp_path) -> None:
    data_dir, _ = _seeded_instrument_store(tmp_path)
    result = runner.invoke(
        app,
        [
            "paper",
            "--venue",
            "hyperliquid",
            "--config",
            "ict|not_a_real_exit|none|hyperliquid:BTC",
            "--data-dir",
            str(data_dir),
        ],
    )
    assert result.exit_code == 1
    assert "error:" in result.output


def test_paper_never_holds_the_store_open_between_bars(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    data_dir, _store_path = _seeded_instrument_store(tmp_path)

    counters = {"opens": 0, "closes": 0}
    real_init = Store.__init__
    real_close = Store.close

    def counting_init(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        counters["opens"] += 1
        real_init(self, *args, **kwargs)

    def counting_close(self):  # type: ignore[no-untyped-def]
        counters["closes"] += 1
        real_close(self)

    monkeypatch.setattr(Store, "__init__", counting_init)
    monkeypatch.setattr(Store, "close", counting_close)

    balanced_checks: list[bool] = []

    def _check_balanced() -> None:
        balanced_checks.append(counters["opens"] == counters["closes"])

    source = _FakePaperSource(_paper_stream_bars(), on_before_yield=_check_balanced)
    monkeypatch.setattr(cli, "_bar_source", lambda venue, **kw: source)
    monkeypatch.setattr(
        cli, "_entry_factory", lambda name, instrument, *, baseline_rate, seed: _SignalsOnce()
    )

    config_id = "signals_once|fixed_r_2|none|hyperliquid:BTC"
    result = runner.invoke(
        app, ["paper", "--venue", "hyperliquid", "--config", config_id, "--data-dir", str(data_dir)]
    )
    assert result.exit_code == 0, result.output
    assert balanced_checks  # the fake stream actually ran
    assert all(balanced_checks)
    assert counters["opens"] == counters["closes"]  # balanced at the very end too


def test_paper_no_phantom_equity_row_on_a_zero_bar_run(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """C4: when the stream raises before yielding a single bar, the `finally`-block flush
    must not write an equity row -- the `equity` table is `Engine`'s persistent bar clock
    (C1), so a phantom row here would silently advance it even though nothing happened.
    Restarting against the same run id (still zero bars processed) must leave that clock
    exactly where it was."""
    data_dir, store_path = _seeded_instrument_store(tmp_path)

    class _RaisesBeforeAnyBarSource:
        def history(self, instrument, tf, start, end):  # type: ignore[no-untyped-def]
            return []

        async def stream(self, instrument, tf):  # type: ignore[no-untyped-def]
            raise RuntimeError("venue exploded before first bar")
            yield  # pragma: no cover -- unreachable; makes this an async generator

    monkeypatch.setattr(cli, "_bar_source", lambda venue, **kw: _RaisesBeforeAnyBarSource())
    monkeypatch.setattr(
        cli, "_entry_factory", lambda name, instrument, *, baseline_rate, seed: _NeverSignals()
    )

    config_id = "none_signal|fixed_r_2|none|hyperliquid:BTC"
    run_id = f"paper:hyperliquid:{config_id}"
    args = ["paper", "--venue", "hyperliquid", "--config", config_id, "--data-dir", str(data_dir)]

    first = runner.invoke(app, args)
    assert first.exit_code == 1
    assert "error: venue exploded before first bar" in first.output

    store = Store(store_path, read_only=True)
    equity_after_first = store.equity(run_id)
    store.close()
    assert equity_after_first == []

    second = runner.invoke(app, args)  # a restart against the same (still bar-less) run id
    assert second.exit_code == 1

    store = Store(store_path, read_only=True)
    equity_after_second = store.equity(run_id)
    store.close()
    assert equity_after_second == []
    assert len(equity_after_second) == len(equity_after_first)  # bar clock unchanged


def test_transient_settings_reader_used_by_paper_sees_a_live_kill_switch(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """The engine reads settings via `TransientSettingsReader`; a version bump from another
    `Store` instance must be visible without paper ever holding the store open (WU-3A's
    concurrency rule) -- exercised here directly rather than through a live paper run."""
    store_path = tmp_path / "v.duckdb"
    store = Store(store_path)
    store.close()

    reader = TransientSettingsReader(store_path)
    version_before, settings_before = reader.current()
    assert settings_before.kill_switch is False

    from swingforge.core.settings import Settings

    writer = Store(store_path)
    writer.write_settings(Settings(kill_switch=True), actor="test")
    writer.close()

    version_after, settings_after = reader.current()
    assert version_after > version_before
    assert settings_after.kill_switch is True


def test_paper_final_flush_failure_does_not_mask_the_original_exception(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """I5: a failure in the final `finally`-block flush must never replace whatever
    exception was already propagating out of the streaming loop (e.g. the venue's own
    error) -- it is echoed alongside, not swapped in for it."""
    data_dir, _store_path = _seeded_instrument_store(tmp_path)

    class _FailingAfterOneBarSource:
        def history(self, instrument, tf, start, end):  # type: ignore[no-untyped-def]
            return []

        async def stream(self, instrument, tf):  # type: ignore[no-untyped-def]
            yield _bar(TS0)
            raise RuntimeError("venue exploded")

    monkeypatch.setattr(cli, "_bar_source", lambda venue, **kw: _FailingAfterOneBarSource())
    monkeypatch.setattr(
        cli, "_entry_factory", lambda name, instrument, *, baseline_rate, seed: _NeverSignals()
    )

    calls = {"n": 0}
    real_persist_bar = cli._persist_bar

    def flaky_persist_bar(*a, **kw):  # type: ignore[no-untyped-def]
        calls["n"] += 1
        if calls["n"] == 2:  # the final flush in `finally`
            raise RuntimeError("store write failed")
        return real_persist_bar(*a, **kw)

    monkeypatch.setattr(cli, "_persist_bar", flaky_persist_bar)

    result = runner.invoke(
        app,
        [
            "paper",
            "--venue",
            "hyperliquid",
            "--config",
            "none_signal|fixed_r_2|none|hyperliquid:BTC",
            "--data-dir",
            str(data_dir),
        ],
    )
    assert result.exit_code == 1
    assert "error: venue exploded" in result.output
    assert "error during final flush: store write failed" in result.output


def test_persist_bar_retries_a_locked_store_then_succeeds(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    store_path = tmp_path / "v.duckdb"
    store = Store(store_path)
    store.close()

    real_store_init = Store.__init__
    attempts = {"n": 0}

    def flaky_init(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        attempts["n"] += 1
        if attempts["n"] <= 2:
            raise duckdb.IOException("locked")
        real_store_init(self, *args, **kwargs)

    monkeypatch.setattr(Store, "__init__", flaky_init)
    slept: list[float] = []

    cli._persist_bar(store_path, "run1", [], None, [], (TS0, 10_000.0), sleep=slept.append)

    assert slept == [cli.PERSIST_BACKOFF_S, cli.PERSIST_BACKOFF_S]

    monkeypatch.setattr(Store, "__init__", real_store_init)
    store = Store(store_path, read_only=True)
    equity = store.equity("run1")
    store.close()
    assert equity == [(TS0, 10_000.0)]


def test_persist_bar_gives_up_after_persist_retries_attempts(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store_path = tmp_path / "v.duckdb"
    store = Store(store_path)
    store.close()

    def always_locked(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        raise duckdb.IOException("locked")

    monkeypatch.setattr(Store, "__init__", always_locked)
    slept: list[float] = []

    with pytest.raises(duckdb.IOException):
        cli._persist_bar(store_path, "run1", [], None, [], (TS0, 10_000.0), sleep=slept.append)

    assert len(slept) == cli.PERSIST_RETRIES


def test_paper_steps_the_daily_bar_fetched_at_midnight(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """I7: the live loop fetches the previous UTC day's Daily bar whenever a 4H bar opens at
    `hour==0` and steps it through `engine.step` -- pinned here by identity on `ctx.last`."""
    data_dir, _store_path = _seeded_instrument_store(tmp_path)
    daily_bar = _bar(TS0 - timedelta(hours=24), tf="1d", high=110.0, low=90.0)

    class _DailySource:
        def history(self, instrument, tf, start, end):  # type: ignore[no-untyped-def]
            # Only the live loop's exact 24h midnight-fetch window sees `daily_bar`; warm-up
            # (a much wider window) must not, so the only way it can reach `ctx` is via the
            # live loop's `engine.step(daily_bar)` call -- not warm-up's direct `ctx.push`.
            if tf == "1d" and end - start == timedelta(hours=24):
                return [daily_bar]
            return []

        async def stream(self, instrument, tf):  # type: ignore[no-untyped-def]
            yield _paper_stream_bars()[0]

    monkeypatch.setattr(cli, "_bar_source", lambda venue, **kw: _DailySource())
    monkeypatch.setattr(
        cli, "_entry_factory", lambda name, instrument, *, baseline_rate, seed: _NeverSignals()
    )

    engines: list[Engine] = []
    real_engine = cli.Engine

    def capturing_engine(*a, **kw):  # type: ignore[no-untyped-def]
        engine = real_engine(*a, **kw)
        engines.append(engine)
        return engine

    monkeypatch.setattr(cli, "Engine", capturing_engine)

    result = runner.invoke(
        app,
        [
            "paper",
            "--venue",
            "hyperliquid",
            "--config",
            "none_signal|fixed_r_2|none|hyperliquid:BTC",
            "--data-dir",
            str(data_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    assert len(engines) == 1
    assert engines[0].ctx.last("1d") is daily_bar


def test_paper_keyboard_interrupt_stops_cleanly_after_a_final_flush(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """I7: a `KeyboardInterrupt` from the stream is a graceful stop -- `broker.close()` runs,
    one more flush happens, and the process exits 0."""
    data_dir, store_path = _seeded_instrument_store(tmp_path)

    class _InterruptingSource:
        def history(self, instrument, tf, start, end):  # type: ignore[no-untyped-def]
            return []

        async def stream(self, instrument, tf):  # type: ignore[no-untyped-def]
            yield _paper_stream_bars()[0]
            raise KeyboardInterrupt()

    monkeypatch.setattr(cli, "_bar_source", lambda venue, **kw: _InterruptingSource())
    monkeypatch.setattr(
        cli, "_entry_factory", lambda name, instrument, *, baseline_rate, seed: _NeverSignals()
    )

    close_calls = {"n": 0}
    real_close = PaperBroker.close

    def counting_close(self):  # type: ignore[no-untyped-def]
        close_calls["n"] += 1
        return real_close(self)

    monkeypatch.setattr(PaperBroker, "close", counting_close)

    persist_calls = {"n": 0}
    real_persist_bar = cli._persist_bar

    def counting_persist_bar(*a, **kw):  # type: ignore[no-untyped-def]
        persist_calls["n"] += 1
        return real_persist_bar(*a, **kw)

    monkeypatch.setattr(cli, "_persist_bar", counting_persist_bar)

    config_id = "none_signal|fixed_r_2|none|hyperliquid:BTC"
    result = runner.invoke(
        app, ["paper", "--venue", "hyperliquid", "--config", config_id, "--data-dir", str(data_dir)]
    )
    assert result.exit_code == 0, result.output
    assert "paper trading stopped" in result.output

    # Pins the docstring's "final flush + close()" claim: `broker.close()` runs exactly
    # once, and `_persist_bar` runs exactly twice -- once for the one bar processed in the
    # loop, once more for the `finally`-block flush.
    assert close_calls["n"] == 1
    assert persist_calls["n"] == 2

    run_id = f"paper:hyperliquid:{config_id}"
    store = Store(store_path, read_only=True)
    equity = store.equity(run_id)
    store.close()

    # One bar processed before the interrupt, then the `finally` flush runs once more
    # (idempotently, over the same equity point) -- the process still exits cleanly.
    assert len(equity) == 1


# --- web --------------------------------------------------------------------


def test_web_refuses_bind_all_without_a_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SWINGFORGE_TOKEN", raising=False)
    result = runner.invoke(app, ["web", "--host", "0.0.0.0"])
    assert result.exit_code == 2


def test_web_refuses_any_non_loopback_host_without_a_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """Minor fix: the refusal generalises past the literal `0.0.0.0` to any host that isn't
    one of the recognised loopback addresses."""
    monkeypatch.delenv("SWINGFORGE_TOKEN", raising=False)
    result = runner.invoke(app, ["web", "--host", "10.0.0.5"])
    assert result.exit_code == 2
    assert "requires SWINGFORGE_TOKEN" in result.output


def test_web_allows_loopback_hosts_without_a_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SWINGFORGE_TOKEN", raising=False)
    fake_module = types.ModuleType("swingforge.web.app")
    fake_module.create_app = lambda data_dir=None, token=None: "sentinel-app"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "swingforge.web.app", fake_module)
    monkeypatch.setattr(cli.uvicorn, "run", lambda app_obj, *, host, port: None)

    for host in ("127.0.0.1", "localhost", "::1"):
        result = runner.invoke(app, ["web", "--host", host])
        assert result.exit_code == 0, result.output


def test_web_builds_the_app_and_calls_uvicorn_run(monkeypatch: pytest.MonkeyPatch) -> None:
    create_app_calls: dict[str, object] = {}

    def fake_create_app(data_dir=None, token=None):  # type: ignore[no-untyped-def]
        create_app_calls["data_dir"] = data_dir
        create_app_calls["token"] = token
        return "sentinel-app"

    fake_module = types.ModuleType("swingforge.web.app")
    fake_module.create_app = fake_create_app  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "swingforge.web.app", fake_module)

    run_calls: dict[str, object] = {}

    def fake_run(app_obj, *, host, port):  # type: ignore[no-untyped-def]
        run_calls["app"] = app_obj
        run_calls["host"] = host
        run_calls["port"] = port

    monkeypatch.setattr(cli.uvicorn, "run", fake_run)
    monkeypatch.setenv("SWINGFORGE_TOKEN", "secret")

    result = runner.invoke(app, ["web", "--host", "127.0.0.1", "--port", "9999", "--data-dir", "somedir"])
    assert result.exit_code == 0, result.output
    assert create_app_calls == {"data_dir": "somedir", "token": "secret"}
    assert run_calls == {"app": "sentinel-app", "host": "127.0.0.1", "port": 9999}


def test_web_reports_a_clean_error_when_the_dashboard_module_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "swingforge.web.app", None)  # forces an ImportError on import
    monkeypatch.setenv("SWINGFORGE_TOKEN", "secret")
    result = runner.invoke(app, ["web"])
    assert result.exit_code == 1
    assert "dashboard module not available" in result.output


# --- report --------------------------------------------------------------------


def _results_row(
    run_id: str, config_id: str, *, ts: datetime, n_oos: int, exp_oos: float, passed: bool | None
) -> dict:
    entry, exit_name, session, _inst_key = config_id.split("|")
    return {
        "run_id": run_id,
        "ts": ts,
        "config_id": config_id,
        "split": "pooled",
        "entry": entry,
        "exit": exit_name,
        "session": session,
        "venue": "hyperliquid",
        "symbol": "BTC",
        "n_oos": n_oos,
        "exp_oos": exp_oos,
        "passed": passed,
    }


def test_report_prints_the_passing_config(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    store_path = data_dir / "hyperliquid.duckdb"
    store = Store(store_path)
    run_id = "tournament:hyperliquid:20260101T0000"
    store.write_results(
        [
            _results_row(
                run_id, "ict|fixed_r_2|none|hyperliquid:BTC", ts=TS0, n_oos=80, exp_oos=0.25, passed=True
            ),
            _results_row(
                run_id,
                "baseline|fixed_r_2|none|hyperliquid:BTC",
                ts=TS0,
                n_oos=80,
                exp_oos=-0.05,
                passed=False,
            ),
        ]
    )
    store.close()

    result = runner.invoke(
        app, ["report", "--venue", "hyperliquid", "--run-id", run_id, "--data-dir", str(data_dir)]
    )
    assert result.exit_code == 0, result.output
    assert "ict|fixed_r_2|none|hyperliquid:BTC" in result.output
    lines = result.output.splitlines()
    passing_line = next(line for line in lines if "ict|fixed_r_2|none|hyperliquid:BTC" in line)
    assert "yes" in passing_line


def test_report_latest_picks_the_most_recent_run(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    store_path = data_dir / "hyperliquid.duckdb"
    store = Store(store_path)
    old_run = "tournament:hyperliquid:20250101T0000"
    new_run = "tournament:hyperliquid:20260101T0000"
    store.write_results(
        [
            _results_row(
                old_run,
                "ict|fixed_r_2|none|hyperliquid:BTC",
                ts=datetime(2025, 1, 1, tzinfo=UTC),
                n_oos=10,
                exp_oos=0.1,
                passed=False,
            ),
            _results_row(
                new_run,
                "zones|fixed_r_2|none|hyperliquid:BTC",
                ts=datetime(2026, 1, 1, tzinfo=UTC),
                n_oos=20,
                exp_oos=0.2,
                passed=True,
            ),
        ]
    )
    store.close()

    result = runner.invoke(app, ["report", "--venue", "hyperliquid", "--data-dir", str(data_dir)])
    assert result.exit_code == 0, result.output
    assert "zones|fixed_r_2|none|hyperliquid:BTC" in result.output
    assert "ict|fixed_r_2|none|hyperliquid:BTC" not in result.output


def test_report_no_runs_is_a_clean_error(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    store = Store(data_dir / "hyperliquid.duckdb")
    store.close()
    result = runner.invoke(app, ["report", "--venue", "hyperliquid", "--data-dir", str(data_dir)])
    assert result.exit_code == 1
    assert "error:" in result.output


def test_report_explicit_run_id_with_no_rows_is_a_clean_error(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    store = Store(data_dir / "hyperliquid.duckdb")
    store.close()
    result = runner.invoke(
        app,
        [
            "report",
            "--venue",
            "hyperliquid",
            "--run-id",
            "tournament:hyperliquid:nope",
            "--data-dir",
            str(data_dir),
        ],
    )
    assert result.exit_code == 1
    assert "no results found" in result.output


def test_report_requires_an_existing_store(tmp_path) -> None:
    result = runner.invoke(app, ["report", "--venue", "hyperliquid", "--data-dir", str(tmp_path / "nope")])
    assert result.exit_code == 1
    assert "run backfill first" in result.output
