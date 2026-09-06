"""Tests for `swingforge.adapters.store.Store` and `DuckDBSettingsReader`."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import duckdb
import pytest

from swingforge.adapters.store import DuckDBSettingsReader, Store
from swingforge.core.settings import Settings
from swingforge.core.types import Bar, CostBreakdown, Fill, Instrument, Order, Trade

BTC = Instrument(
    venue="hyperliquid",
    symbol="BTC",
    tick_size=Decimal("0.5"),
    contract_multiplier=Decimal("1"),
    quote_ccy="USD",
    session_profile="perp",
)
ETH = Instrument(
    venue="hyperliquid",
    symbol="ETH",
    tick_size=Decimal("0.01"),
    contract_multiplier=Decimal("1"),
    quote_ccy="USD",
    session_profile="perp",
)


def _bar(ts_open: datetime, tf: str = "4h", instrument: Instrument = BTC) -> Bar:
    return Bar(
        instrument=instrument,
        tf=tf,
        ts_open=ts_open,
        open=100.0,
        high=105.0,
        low=95.0,
        close=102.0,
        volume=10.0,
        bid_close=101.5,
        ask_close=102.5,
    )


@pytest.fixture
def store() -> Store:
    with Store(":memory:") as s:
        yield s


def test_create_schema_is_idempotent(store: Store) -> None:
    store.create_schema()
    store.create_schema()  # must not raise


def test_context_manager_closes() -> None:
    with Store(":memory:") as s:
        s.upsert_instruments([BTC])
    with pytest.raises(duckdb.Error):
        s._conn.execute("SELECT 1")


# -- instruments --------------------------------------------------------------


def test_upsert_and_read_instruments_round_trip(store: Store) -> None:
    store.upsert_instruments([BTC, ETH])
    got = store.instruments("hyperliquid")
    assert got == [BTC, ETH]  # sorted by symbol: BTC, ETH


def test_upsert_instruments_is_idempotent_on_venue_symbol(store: Store) -> None:
    store.upsert_instruments([BTC])
    updated = BTC.model_copy(update={"tick_size": Decimal("1.0")})
    store.upsert_instruments([updated])
    got = store.instruments("hyperliquid")
    assert len(got) == 1
    assert got[0].tick_size == Decimal("1.0")


def test_instrument_decimal_fields_round_trip_exactly(store: Store) -> None:
    odd = Instrument(
        venue="oanda",
        symbol="EURUSD",
        tick_size=Decimal("0.00010"),
        contract_multiplier=Decimal("100000"),
        quote_ccy="USD",
        session_profile="fx",
    )
    store.upsert_instruments([odd])
    got = store.instruments("oanda")[0]
    assert got.tick_size == odd.tick_size
    assert str(got.tick_size) == str(odd.tick_size)
    assert got.contract_multiplier == odd.contract_multiplier


# -- bars -----------------------------------------------------------------


def test_upsert_bars_is_idempotent_row_count(store: Store) -> None:
    ts = datetime(2026, 1, 1, tzinfo=UTC)
    bars = [_bar(ts + timedelta(hours=4 * i)) for i in range(5)]
    store.upsert_bars(bars)
    store.upsert_bars(bars)
    count = store._conn.execute("SELECT COUNT(*) FROM bars").fetchone()[0]
    assert count == 5


def test_upsert_bars_does_not_store_subbars(store: Store) -> None:
    ts = datetime(2026, 1, 1, tzinfo=UTC)
    sub = Bar(instrument=BTC, tf="1h", ts_open=ts, open=100, high=101, low=99, close=100.5, volume=1.0)
    parent = Bar(
        instrument=BTC,
        tf="4h",
        ts_open=ts,
        open=100,
        high=105,
        low=95,
        close=102,
        volume=10.0,
        subbars=(sub,),
    )
    store.upsert_bars([parent])
    count = store._conn.execute("SELECT COUNT(*) FROM bars").fetchone()[0]
    assert count == 1  # subbar not stored alongside its parent
    got = store.bars(BTC, "4h", ts, ts + timedelta(hours=4))
    assert got[0].subbars == ()


def test_bars_query_window_and_ordering(store: Store) -> None:
    ts = datetime(2026, 1, 1, tzinfo=UTC)
    bars = [_bar(ts + timedelta(hours=4 * i)) for i in range(6)]
    store.upsert_bars(bars)
    got = store.bars(BTC, "4h", ts + timedelta(hours=4), ts + timedelta(hours=16))
    assert [b.ts_open for b in got] == [ts + timedelta(hours=4 * i) for i in (1, 2, 3)]
    assert all(b.ts_open.tzinfo is not None for b in got)
    for b in got:
        assert b.ts_open.utcoffset() == timedelta(0)


def test_bar_range(store: Store) -> None:
    assert store.bar_range(BTC, "4h") is None
    ts = datetime(2026, 1, 1, tzinfo=UTC)
    bars = [_bar(ts + timedelta(hours=4 * i)) for i in range(3)]
    store.upsert_bars(bars)
    lo, hi, count = store.bar_range(BTC, "4h")
    assert lo == ts
    assert hi == ts + timedelta(hours=8)
    assert count == 3


# -- funding ----------------------------------------------------------------


def test_upsert_and_read_funding(store: Store) -> None:
    ts = datetime(2026, 1, 1, tzinfo=UTC)
    rows = [(ts, 0.0001), (ts + timedelta(hours=8), -0.0002)]
    store.upsert_funding(BTC, rows)
    got = store.funding(BTC, ts, ts + timedelta(hours=16))
    assert got == rows
    # idempotent upsert on (venue, symbol, ts)
    store.upsert_funding(BTC, rows)
    got2 = store.funding(BTC, ts, ts + timedelta(hours=16))
    assert len(got2) == 2


# -- orders / fills (write-only log tables) ----------------------------------


def test_write_orders_persists_rows(store: Store) -> None:
    order = Order(
        id="o1",
        instrument=BTC,
        direction=1,
        qty=1.0,
        kind="limit",
        price=100.0,
        expires_at_bar=3,
        leg="entry",
        trade_id="t-1",
    )
    store.write_orders("run-1", [order])
    row = store._conn.execute(
        "SELECT id, venue, symbol, trade_id FROM orders WHERE run_id = ?", ["run-1"]
    ).fetchone()
    assert row == ("o1", "hyperliquid", "BTC", "t-1")


def test_write_orders_is_idempotent_on_run_id_and_id(store: Store) -> None:
    order = Order(
        id="o1",
        instrument=BTC,
        direction=1,
        qty=1.0,
        kind="market",
        price=None,
        expires_at_bar=None,
        leg="entry",
        trade_id="t-1",
    )
    store.write_orders("run-1", [order])
    store.write_orders("run-1", [order])
    count = store._conn.execute("SELECT COUNT(*) FROM orders WHERE run_id = ?", ["run-1"]).fetchone()[0]
    assert count == 1


def test_write_fills_persists_cost_breakdown(store: Store) -> None:
    ts = datetime(2026, 1, 1, tzinfo=UTC)
    fill = Fill(
        order_id="o1",
        ts=ts,
        price=100.0,
        qty=1.0,
        cost=CostBreakdown(spread=0.1, commission=0.2, funding=0.3, slippage=0.05),
        leg="entry",
        trade_id="t-1",
    )
    store.write_fills("run-1", [fill])
    row = store._conn.execute(
        "SELECT order_id, spread, commission, funding, slippage FROM fills WHERE run_id = ?", ["run-1"]
    ).fetchone()
    assert row == ("o1", 0.1, 0.2, 0.3, 0.05)


def test_write_fills_is_idempotent_on_run_order_leg_ts(store: Store) -> None:
    ts = datetime(2026, 1, 1, tzinfo=UTC)
    fill = Fill(order_id="o1", ts=ts, price=100.0, qty=1.0, cost=CostBreakdown(), leg="entry", trade_id="t-1")
    store.write_fills("run-1", [fill])
    store.write_fills("run-1", [fill])
    count = store._conn.execute("SELECT COUNT(*) FROM fills WHERE run_id = ?", ["run-1"]).fetchone()[0]
    assert count == 1


# -- trades -----------------------------------------------------------------


def _trade(ts: datetime) -> Trade:
    entry_fill = Fill(
        order_id="o1",
        ts=ts,
        price=100.0,
        qty=1.0,
        cost=CostBreakdown(spread=0.1, commission=0.2),
        leg="entry",
        trade_id="t-1",
    )
    stop_fill = Fill(
        order_id="o1-stop",
        ts=ts + timedelta(hours=4),
        price=97.0,
        qty=1.0,
        cost=CostBreakdown(funding=1.5),
        leg="stop",
        trade_id="t-1",
    )
    return Trade(
        id="t-1",
        instrument=BTC,
        direction=1,
        entry_fill=entry_fill,
        legs=(stop_fill,),
        stop=97.0,
        target=106.0,
        risk_r=300.0,
        realized_r=-1.0,
        mae_r=-1.2,
        mfe_r=0.4,
        regime="range",
        context_snapshot=b'{"foo": 1}',
        opened_bar=0,
        closed_bar=1,
    )


def test_write_trades_requires_instrument_upserted_first(store: Store) -> None:
    trade = _trade(datetime(2026, 1, 1, tzinfo=UTC))
    with pytest.raises(ValueError, match="instrument"):
        store.write_trades("run-1", [trade])


def test_trades_round_trip_exactly(store: Store) -> None:
    store.upsert_instruments([BTC])
    trade = _trade(datetime(2026, 1, 1, tzinfo=UTC))
    store.write_trades("run-1", [trade])
    got = store.trades("run-1")
    assert got == [trade]


def test_trades_scoped_by_run_id(store: Store) -> None:
    store.upsert_instruments([BTC])
    trade = _trade(datetime(2026, 1, 1, tzinfo=UTC))
    store.write_trades("run-1", [trade])
    store.write_trades("run-2", [trade])
    assert len(store.trades("run-1")) == 1
    assert len(store.trades("run-2")) == 1
    assert store.trades("no-such-run") == []


def test_write_trades_is_idempotent_row_count(store: Store) -> None:
    store.upsert_instruments([BTC])
    trade = _trade(datetime(2026, 1, 1, tzinfo=UTC))
    store.write_trades("run-1", [trade])
    store.write_trades("run-1", [trade])
    count = store._conn.execute("SELECT COUNT(*) FROM trades WHERE run_id = ?", ["run-1"]).fetchone()[0]
    assert count == 1


def test_trades_resolves_instrument_once_per_distinct_venue_symbol(
    store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Three trades, two distinct instruments (BTC used twice) -> exactly 2 lookups."""
    store.upsert_instruments([BTC, ETH])
    t1 = _trade(datetime(2026, 1, 1, tzinfo=UTC)).model_copy(update={"id": "t-1"})
    t2 = _trade(datetime(2026, 1, 1, tzinfo=UTC)).model_copy(update={"id": "t-2"})
    t3 = _trade(datetime(2026, 1, 1, tzinfo=UTC)).model_copy(update={"id": "t-3", "instrument": ETH})
    store.write_trades("run-1", [t1, t2, t3])

    calls = []
    original = store._instrument_by_venue_symbol

    def _counting(venue: str, symbol: str) -> Instrument | None:
        calls.append((venue, symbol))
        return original(venue, symbol)

    monkeypatch.setattr(store, "_instrument_by_venue_symbol", _counting)
    got = store.trades("run-1")

    assert len(got) == 3
    assert len(calls) == 2  # one for (hyperliquid, BTC), one for (hyperliquid, ETH)


def test_write_trades_resolves_instrument_once_per_distinct_venue_symbol(
    store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Three trades, two distinct instruments (BTC used twice) -> exactly 2 lookups.

    `write_trades` validates every trade's instrument is already upserted (one lookup per
    trade today, ~8ms each); it must cache per distinct (venue, symbol) within the call, as
    `trades()` already does.
    """
    store.upsert_instruments([BTC, ETH])
    t1 = _trade(datetime(2026, 1, 1, tzinfo=UTC)).model_copy(update={"id": "t-1"})
    t2 = _trade(datetime(2026, 1, 1, tzinfo=UTC)).model_copy(update={"id": "t-2"})
    t3 = _trade(datetime(2026, 1, 1, tzinfo=UTC)).model_copy(update={"id": "t-3", "instrument": ETH})

    calls = []
    original = store._instrument_by_venue_symbol

    def _counting(venue: str, symbol: str) -> Instrument | None:
        calls.append((venue, symbol))
        return original(venue, symbol)

    monkeypatch.setattr(store, "_instrument_by_venue_symbol", _counting)
    store.write_trades("run-1", [t1, t2, t3])

    assert len(calls) == 2  # one for (hyperliquid, BTC), one for (hyperliquid, ETH)
    assert len(store.trades("run-1")) == 3


def test_write_trades_still_rejects_a_missing_instrument_with_cache(store: Store) -> None:
    # The per-call cache must not paper over a genuinely-missing instrument: nothing has
    # been upserted, so the very first (and only) lookup must still raise.
    trade = _trade(datetime(2026, 1, 1, tzinfo=UTC))
    with pytest.raises(ValueError, match="instrument"):
        store.write_trades("run-1", [trade])


def test_context_snapshot_round_trips_non_empty_and_empty(store: Store) -> None:
    store.upsert_instruments([BTC])
    with_snapshot = _trade(datetime(2026, 1, 1, tzinfo=UTC)).model_copy(
        update={"id": "t-1", "context_snapshot": b'{"bar_index": 5}'}
    )
    without_snapshot = _trade(datetime(2026, 1, 1, tzinfo=UTC)).model_copy(
        update={"id": "t-2", "context_snapshot": b""}
    )
    store.write_trades("run-1", [with_snapshot, without_snapshot])
    got = {t.id: t for t in store.trades("run-1")}
    assert got["t-1"].context_snapshot == b'{"bar_index": 5}'
    assert got["t-2"].context_snapshot == b""


def test_context_snapshot_round_trips_nul_and_high_bytes(store: Store) -> None:
    store.upsert_instruments([BTC])
    tricky = bytes(range(256))  # every byte value, including NUL (0x00) and high bytes (>=0x80)
    trade = _trade(datetime(2026, 1, 1, tzinfo=UTC)).model_copy(update={"context_snapshot": tricky})
    store.write_trades("run-1", [trade])
    got = store.trades("run-1")
    assert len(got) == 1
    assert got[0].context_snapshot == tricky


# -- write_trades performance (C2) --------------------------------------------


@pytest.mark.slow
def test_write_trades_2000_under_3s(store: Store) -> None:
    import time

    store.upsert_instruments([BTC])
    ts0 = datetime(2020, 1, 1, tzinfo=UTC)
    trades = [_trade(ts0 + timedelta(hours=4 * i)).model_copy(update={"id": f"t-{i}"}) for i in range(2_000)]

    # DuckDB lazily imports pandas (now that it's a project dependency; see the module-level
    # note in `store.py`) on a process's *first* parameterized `execute(sql, [...])` call --
    # a one-time cost (~1-2.5s here) unrelated to `write_trades` itself. A long-running
    # process pays it once, on whatever query happens to run first; pay it here, before the
    # timer starts, rather than let test order nondeterministically fold it into the budget.
    store._instrument_by_venue_symbol(BTC.venue, BTC.symbol)

    start = time.perf_counter()
    store.write_trades("run-1", trades)
    elapsed = time.perf_counter() - start
    assert elapsed < 3.0, f"write_trades(2,000 trades) took {elapsed:.2f}s, expected < 3s"

    count = store._conn.execute("SELECT COUNT(*) FROM trades WHERE run_id = ?", ["run-1"]).fetchone()[0]
    assert count == 2_000


# -- equity -----------------------------------------------------------------


def test_write_and_read_equity(store: Store) -> None:
    ts = datetime(2026, 1, 1, tzinfo=UTC)
    rows = [(ts, 10_000.0), (ts + timedelta(hours=4), 10_050.0)]
    store.write_equity("run-1", rows)
    assert store.equity("run-1") == rows


def test_write_equity_is_idempotent_on_run_id_and_ts(store: Store) -> None:
    ts = datetime(2026, 1, 1, tzinfo=UTC)
    store.write_equity("run-1", [(ts, 10_000.0)])
    store.write_equity("run-1", [(ts, 10_050.0)])  # same (run_id, ts): overwrites, not duplicates
    got = store.equity("run-1")
    assert got == [(ts, 10_050.0)]


# -- results ------------------------------------------------------------------


def test_write_and_read_results(store: Store) -> None:
    ts = datetime(2026, 1, 1, tzinfo=UTC)
    row = {
        "run_id": "run-1",
        "ts": ts,
        "config_id": "ict-fixedr-2",
        "entry": "ict",
        "exit": "fixedr",
        "session": "none",
        "venue": "hyperliquid",
        "symbol": "BTC",
        "split": "oos",
        "n_is": 100,
        "n_oos": 30,
        "exp_is": 0.2,
        "exp_oos": 0.15,
        "dsr_prob": 0.97,
        "boot_p5": 0.01,
        "diff_p5": 0.02,
        "mar": 1.5,
        "mar_bh": 0.8,
        "g1": True,
        "g2": True,
        "g3": True,
        "g4": True,
        "g5": True,
        "g6": True,
        "passed": True,
        "resolution_mode": "subbars",
        "excluded_reason": None,
    }
    store.write_results([row])
    got = store.results("run-1")
    assert len(got) == 1
    assert got[0] == row


def test_results_nullable_gate_flags(store: Store) -> None:
    # config_id/split are the (non-ts) primary-key columns and so must be non-null; every
    # other column (the gate flags checked below among them) is optional.
    row = {"run_id": "run-1", "ts": datetime(2026, 1, 1, tzinfo=UTC), "config_id": "c1", "split": "is"}
    store.write_results([row])
    got = store.results("run-1")[0]
    assert got["g1"] is None
    assert got["passed"] is None


def test_write_results_rejects_unknown_column(store: Store) -> None:
    row = {
        "run_id": "run-1",
        "ts": datetime(2026, 1, 1, tzinfo=UTC),
        "config_id": "c1",
        "split": "is",
        "bogus": 1,
    }
    with pytest.raises(ValueError, match="bogus"):
        store.write_results([row])


def test_write_results_is_idempotent_on_run_config_split(store: Store) -> None:
    row = {
        "run_id": "run-1",
        "ts": datetime(2026, 1, 1, tzinfo=UTC),
        "config_id": "c1",
        "split": "is",
        "mar": 1.0,
    }
    store.write_results([row])
    store.write_results([{**row, "mar": 2.0}])
    got = store.results("run-1")
    assert len(got) == 1
    assert got[0]["mar"] == 2.0


# -- settings -----------------------------------------------------------------


def test_current_settings_defaults_to_version_zero(store: Store) -> None:
    version, settings = store.current_settings()
    assert version == 0
    assert settings == Settings()


def test_write_settings_bumps_version_and_logs(store: Store) -> None:
    v1 = store.write_settings(Settings(risk_pct=0.015), actor="alice")
    v2 = store.write_settings(Settings(risk_pct=0.01), actor="bob")
    v3 = store.write_settings(Settings(kill_switch=True), actor="alice")
    assert (v1, v2, v3) == (1, 2, 3)

    version, current = store.current_settings()
    assert version == 3
    assert current.kill_switch is True

    log = store.settings_log()
    assert [entry["version"] for entry in log] == [1, 2, 3]
    assert [entry["actor"] for entry in log] == ["alice", "bob", "alice"]


def test_duckdb_settings_reader_sees_new_version(store: Store) -> None:
    reader = DuckDBSettingsReader(store)
    assert reader.current() == (0, Settings())
    store.write_settings(Settings(risk_pct=0.02), actor="alice")
    version, settings = reader.current()
    assert version == 1
    assert settings.risk_pct == 0.02


def test_write_settings_rolls_back_on_mid_write_failure(
    store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    store.write_settings(Settings(risk_pct=0.015), actor="alice")

    # `DuckDBPyConnection.execute` is a C-extension attribute: read-only per instance, so
    # the fake has to replace it on the class (monkeypatch restores it after the test).
    conn_cls = type(store._conn)
    real_execute = conn_cls.execute
    calls = {"n": 0}

    def _flaky_execute(self: object, sql: str, *args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        calls["n"] += 1
        # call 1 = current_settings()'s SELECT, call 2 = DELETE FROM settings (this really
        # runs, emptying the table), call 3 = INSERT INTO settings -- fail here so there is
        # a real deleted-but-not-yet-reinserted state for ROLLBACK to have to undo.
        if calls["n"] == 3:
            raise RuntimeError("simulated mid-write failure")
        return real_execute(self, sql, *args, **kwargs)

    monkeypatch.setattr(conn_cls, "execute", _flaky_execute)
    with pytest.raises(RuntimeError, match="simulated mid-write failure"):
        store.write_settings(Settings(risk_pct=0.02), actor="bob")

    monkeypatch.setattr(conn_cls, "execute", real_execute)
    version, settings = store.current_settings()
    assert version == 1
    assert settings.risk_pct == 0.015  # bob's write never landed; alice's is still current
    assert [entry["actor"] for entry in store.settings_log()] == ["alice"]  # no orphaned log entry


# -- Store construction -------------------------------------------------------


def test_store_read_only_passthrough(tmp_path: object) -> None:
    path = str(tmp_path) + "/store.duckdb"  # type: ignore[operator]
    with Store(path) as writer:
        writer.upsert_instruments([BTC])

    with Store(path, read_only=True) as reader:
        assert reader.instruments("hyperliquid") == [BTC]
        with pytest.raises(duckdb.Error):
            reader.upsert_instruments([ETH])


# -- bulk-write performance (C1) -----------------------------------------------


@pytest.mark.slow
def test_upsert_bars_10k_under_5s_and_idempotent(store: Store) -> None:
    import time

    ts0 = datetime(2020, 1, 1, tzinfo=UTC)
    bars = [_bar(ts0 + timedelta(hours=4 * i)) for i in range(10_000)]

    start = time.perf_counter()
    store.upsert_bars(bars)
    elapsed = time.perf_counter() - start
    assert elapsed < 5.0, f"upsert_bars(10,000 bars) took {elapsed:.2f}s, expected < 5s"

    count = store._conn.execute("SELECT COUNT(*) FROM bars").fetchone()[0]
    assert count == 10_000

    store.upsert_bars(bars)  # idempotent: re-upserting the same bars must not duplicate rows
    count2 = store._conn.execute("SELECT COUNT(*) FROM bars").fetchone()[0]
    assert count2 == 10_000
