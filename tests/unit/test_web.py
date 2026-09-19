"""Tests for the swingforge.web dashboard: FastAPI routes over real DuckDB stores."""

from __future__ import annotations

import warnings
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import duckdb
import pytest

# The installed starlette (1.6.0, per uv.lock) emits a one-time `StarletteDeprecationWarning`
# (a `UserWarning` subclass, so pyproject.toml's `filterwarnings = ["error", ...]` does not
# already downgrade it the way it does for `DeprecationWarning`) on first import of
# `starlette.testclient`, nudging towards an `httpx2` package this project does not (and, per
# this WU's file ownership, must not) depend on. Suppressed locally, at the one import site,
# rather than editing pyproject.toml's warning policy for the whole suite.
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from fastapi.testclient import TestClient

from swingforge.adapters.store import Store
from swingforge.core.settings import Settings
from swingforge.core.types import Bar, CostBreakdown, Fill, Instrument, Trade
from swingforge.web import app as app_module
from swingforge.web import queries
from swingforge.web.app import create_app

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
SOL = Instrument(
    venue="hyperliquid",
    symbol="SOL",
    tick_size=Decimal("0.01"),
    contract_multiplier=Decimal("1"),
    quote_ccy="USD",
    session_profile="perp",
)
EURUSD = Instrument(
    venue="oanda",
    symbol="EURUSD",
    tick_size=Decimal("0.00010"),
    contract_multiplier=Decimal("100000"),
    quote_ccy="USD",
    session_profile="fx",
)


def _bar(instrument: Instrument, ts_open: datetime, *, close: float = 100.0, tf: str = "4h") -> Bar:
    return Bar(
        instrument=instrument,
        tf=tf,
        ts_open=ts_open,
        open=close - 1,
        high=close + 2,
        low=close - 3,
        close=close,
        volume=10.0,
    )


def _open_trade(
    trade_id: str,
    instrument: Instrument,
    *,
    entry_ts: datetime,
    entry_price: float = 100.0,
    entry_qty: float = 2.0,
    direction: int = 1,
    risk_r: float = 50.0,
    stop: float = 90.0,
    target: float | None = None,
) -> Trade:
    entry_fill = Fill(
        order_id=f"{trade_id}-entry",
        ts=entry_ts,
        price=entry_price,
        qty=entry_qty,
        cost=CostBreakdown(),
        leg="entry",
        trade_id=trade_id,
    )
    return Trade(
        id=trade_id,
        instrument=instrument,
        direction=direction,
        entry_fill=entry_fill,
        legs=(),
        stop=stop,
        target=target,
        risk_r=risk_r,
        realized_r=None,
        mae_r=-0.1,
        mfe_r=0.3,
        regime="range",
        opened_bar=0,
        closed_bar=None,
    )


def _closed_trade(
    trade_id: str,
    instrument: Instrument,
    *,
    entry_ts: datetime,
    exit_ts: datetime,
    realized_r: float,
    regime: str = "trend",
    entry_price: float = 100.0,
    stop: float = 90.0,
    target: float | None = 120.0,
    risk_r: float = 50.0,
) -> Trade:
    entry_fill = Fill(
        order_id=f"{trade_id}-entry",
        ts=entry_ts,
        price=entry_price,
        qty=2.0,
        cost=CostBreakdown(),
        leg="entry",
        trade_id=trade_id,
    )
    exit_fill = Fill(
        order_id=f"{trade_id}-target",
        ts=exit_ts,
        price=target if target is not None else entry_price + 10,
        qty=2.0,
        cost=CostBreakdown(),
        leg="target",
        trade_id=trade_id,
    )
    return Trade(
        id=trade_id,
        instrument=instrument,
        direction=1,
        entry_fill=entry_fill,
        legs=(exit_fill,),
        stop=stop,
        target=target,
        risk_r=risk_r,
        realized_r=realized_r,
        mae_r=-0.2,
        mfe_r=1.5,
        regime=regime,
        opened_bar=0,
        closed_bar=2,
    )


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    return tmp_path


def _hl_store(data_dir: Path) -> Store:
    return Store(data_dir / "hyperliquid.duckdb")


def _oanda_store(data_dir: Path) -> Store:
    return Store(data_dir / "oanda.duckdb")


# -- auth ---------------------------------------------------------------------


def test_root_serves_html_without_token(data_dir: Path) -> None:
    app = create_app(data_dir=data_dir, token=None)
    client = TestClient(app)
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]


def test_unauthenticated_401_when_token_configured(data_dir: Path) -> None:
    app = create_app(data_dir=data_dir, token="secret")
    client = TestClient(app)
    resp = client.get("/api/overview")
    assert resp.status_code == 401
    assert resp.json() == {"detail": "unauthorized"}


def test_wrong_bearer_token_401(data_dir: Path) -> None:
    app = create_app(data_dir=data_dir, token="secret")
    client = TestClient(app)
    resp = client.get("/api/overview", headers={"Authorization": "Bearer wrong"})
    assert resp.status_code == 401


def test_correct_bearer_token_200(data_dir: Path) -> None:
    app = create_app(data_dir=data_dir, token="secret")
    client = TestClient(app)
    resp = client.get("/api/overview", headers={"Authorization": "Bearer secret"})
    assert resp.status_code == 200


def test_no_token_configured_is_open(data_dir: Path) -> None:
    app = create_app(data_dir=data_dir, token=None)
    client = TestClient(app)
    resp = client.get("/api/overview")
    assert resp.status_code == 200


def test_root_open_even_when_token_configured(data_dir: Path) -> None:
    """C4: `/` (the static HTML shell) is never behind the bearer check -- only `/api/*` is."""
    app = create_app(data_dir=data_dir, token="secret")
    client = TestClient(app)
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]


def test_put_settings_401_without_token_when_configured(data_dir: Path) -> None:
    """C4/M4: the one write route is under `/api/*` too, so it must be guarded the same way."""
    client = TestClient(create_app(data_dir=data_dir, token="secret"))
    resp = client.put("/api/settings", json={"venue": "hyperliquid", "settings": {}})
    assert resp.status_code == 401


def test_docs_disabled_when_token_configured(data_dir: Path) -> None:
    """C4: the auto-generated docs would otherwise leak the whole API shape without auth."""
    client = TestClient(create_app(data_dir=data_dir, token="secret"))
    assert client.get("/docs").status_code == 404
    assert client.get("/redoc").status_code == 404
    assert client.get("/openapi.json").status_code == 404


def test_docs_available_when_no_token_configured(data_dir: Path) -> None:
    client = TestClient(create_app(data_dir=data_dir, token=None))
    assert client.get("/docs").status_code == 200


def test_non_ascii_bearer_header_401_not_500(data_dir: Path) -> None:
    """C3: a header that doesn't even decode cleanly as UTF-8 must compare as a mismatch, not
    crash -- raw bytes (httpx refuses a non-ASCII `str` header value outright) carrying both a
    valid UTF-8 sequence and a byte that is not valid UTF-8 on its own."""
    client = TestClient(create_app(data_dir=data_dir, token="secret"))
    resp = client.get("/api/overview", headers={"Authorization": "Bearer café☃".encode()})
    assert resp.status_code == 401
    resp2 = client.get("/api/overview", headers={"Authorization": b"Bearer \xff\xfe"})
    assert resp2.status_code == 401


def test_missing_authorization_header_401(data_dir: Path) -> None:
    """C3: no header at all must be a plain 401, not an attribute error on `None`."""
    client = TestClient(create_app(data_dir=data_dir, token="secret"))
    resp = client.get("/api/overview")
    assert resp.status_code == 401
    assert resp.json() == {"detail": "unauthorized"}


# -- store-open retry / 503 (C1, C2) -------------------------------------------


def test_overview_503_when_store_stays_locked(data_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """C1: every open attempt raising `duckdb.IOException` must retry-then-503, not 500."""
    with _hl_store(data_dir):
        pass  # create the venue file with the real __init__ before patching it

    def always_locked(self: Store, path: object, read_only: bool = False) -> None:
        raise duckdb.IOException("database is locked")

    monkeypatch.setattr(Store, "__init__", always_locked)
    monkeypatch.setattr(app_module.time, "sleep", lambda _seconds: None)

    client = TestClient(create_app(data_dir=data_dir))
    resp = client.get("/api/overview")
    assert resp.status_code == 503
    assert resp.json() == {"detail": "venue busy"}


def test_overview_503_body_on_locked_settings_put(data_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """C1: the one write route (PUT /api/settings) goes through the same retry-then-503 path."""

    def always_locked(self: Store, path: object, read_only: bool = False) -> None:
        raise duckdb.IOException("database is locked")

    monkeypatch.setattr(Store, "__init__", always_locked)
    monkeypatch.setattr(app_module.time, "sleep", lambda _seconds: None)

    client = TestClient(create_app(data_dir=data_dir))
    resp = client.put("/api/settings", json={"venue": "hyperliquid", "settings": {}})
    assert resp.status_code == 503
    assert resp.json() == {"detail": "venue busy"}


def test_overview_retries_transient_lock_then_succeeds(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """C1: a lock that clears within `OPEN_RETRIES` attempts must not surface as an error at all."""
    with _hl_store(data_dir):
        pass

    original_init = Store.__init__
    calls = {"n": 0}

    def flaky_init(self: Store, path: object, read_only: bool = False) -> None:
        calls["n"] += 1
        if calls["n"] <= 2:
            raise duckdb.IOException("database is locked")
        original_init(self, path, read_only)

    monkeypatch.setattr(Store, "__init__", flaky_init)
    monkeypatch.setattr(app_module.time, "sleep", lambda _seconds: None)

    client = TestClient(create_app(data_dir=data_dir))
    resp = client.get("/api/overview")
    assert resp.status_code == 200
    assert calls["n"] == 3


def test_open_present_stores_closes_already_opened_store_on_later_failure(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """C2: opening venues incrementally must not leak the first venue's handle when a later
    venue's open fails -- the first store's `close()` must still be called."""
    with _hl_store(data_dir):
        pass
    with _oanda_store(data_dir):
        pass

    class FakeStore:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    opened: list[FakeStore] = []

    def fake_open(path: Path, *, read_only: bool) -> FakeStore:
        if "oanda" in str(path):
            raise RuntimeError("boom opening oanda")
        store = FakeStore()
        opened.append(store)
        return store

    monkeypatch.setattr(app_module, "_open_store_with_retry", fake_open)

    with pytest.raises(RuntimeError):
        app_module._open_present_stores(data_dir)

    assert len(opened) == 1
    assert opened[0].closed is True


# -- overview: missing venue, equity, positions, fills -------------------------


def test_missing_venue_file_is_skipped(data_dir: Path) -> None:
    client = TestClient(create_app(data_dir=data_dir))
    resp = client.get("/api/overview")
    assert resp.status_code == 200
    assert resp.json()["venues"] == {}


def test_overview_computes_unrealized_r_for_open_paper_trade(data_dir: Path) -> None:
    ts0 = datetime(2026, 1, 1, tzinfo=UTC)
    with _hl_store(data_dir) as store:
        store.upsert_instruments([BTC])
        store.upsert_bars([_bar(BTC, ts0, close=108.0)])
        trade = _open_trade(
            "t-1", BTC, entry_ts=ts0, entry_price=100.0, entry_qty=2.0, direction=1, risk_r=50.0
        )
        # a second open BTC position -- exercises the latest-4H-bar cache being *reused*
        # for a second row of the same symbol, not just populated once.
        trade2 = _open_trade(
            "t-2", BTC, entry_ts=ts0, entry_price=104.0, entry_qty=1.0, direction=-1, risk_r=20.0
        )
        store.write_trades("paper:hyperliquid:cfgA", [trade, trade2])

    client = TestClient(create_app(data_dir=data_dir))
    resp = client.get("/api/overview")
    assert resp.status_code == 200
    positions = resp.json()["venues"]["hyperliquid"]["positions"]
    assert len(positions) == 2
    by_id = {p["trade_id"]: p for p in positions}
    # direction(-1) * (108 - 104) * qty(1) * multiplier(1) / risk_r(20) = -4 / 20 = -0.2
    assert by_id["t-2"]["unrealized_r"] == pytest.approx(-0.2)
    pos = by_id["t-1"]
    assert pos["run_id"] == "paper:hyperliquid:cfgA"
    assert pos["trade_id"] == "t-1"
    assert pos["symbol"] == "BTC"
    # direction(1) * (108 - 100) * qty(2) * multiplier(1) / risk_r(50) = 16 / 50 = 0.32
    assert pos["unrealized_r"] == pytest.approx(0.32)


def test_overview_unrealized_r_uses_contract_multiplier_for_fx(data_dir: Path) -> None:
    """M4: EURUSD's `contract_multiplier=100000` must be applied -- a lot-sized FX qty of 1.0
    without the multiplier would produce a `unrealized_r` about 100,000x too small, so this
    fails immediately if `* multiplier` is ever dropped from the formula."""
    ts0 = datetime(2026, 1, 1, tzinfo=UTC)
    with _oanda_store(data_dir) as store:
        store.upsert_instruments([EURUSD])
        store.upsert_bars([_bar(EURUSD, ts0, close=1.1010)])
        trade = _open_trade(
            "t-fx", EURUSD, entry_ts=ts0, entry_price=1.1000, entry_qty=1.0, direction=1, risk_r=100.0
        )
        store.write_trades("paper:oanda:cfgA", [trade])

    client = TestClient(create_app(data_dir=data_dir))
    positions = client.get("/api/overview").json()["venues"]["oanda"]["positions"]
    assert len(positions) == 1
    # direction(1) * (1.1010 - 1.1000) * qty(1.0) * multiplier(100000) / risk_r(100) = 1.0
    assert positions[0]["unrealized_r"] == pytest.approx(1.0)


def test_overview_unrealized_r_none_without_a_4h_bar(data_dir: Path) -> None:
    ts0 = datetime(2026, 1, 1, tzinfo=UTC)
    with _hl_store(data_dir) as store:
        store.upsert_instruments([BTC])
        trade = _open_trade("t-1", BTC, entry_ts=ts0)
        store.write_trades("paper:hyperliquid:cfgA", [trade])

    client = TestClient(create_app(data_dir=data_dir))
    positions = client.get("/api/overview").json()["venues"]["hyperliquid"]["positions"]
    assert positions[0]["unrealized_r"] is None


def test_overview_unrealized_r_none_when_instrument_not_in_venues_table(data_dir: Path) -> None:
    """A trades row whose symbol has no matching row in `instruments` for this venue (a data
    inconsistency the public Store API can't itself produce, since write_trades requires the
    instrument first) must not crash the overview -- unrealized_r degrades to None."""
    ts0 = datetime(2026, 1, 1, tzinfo=UTC)
    with _hl_store(data_dir) as store:
        store._conn.execute(
            """
            INSERT INTO trades (run_id, id, venue, symbol, direction, entry_price, entry_ts,
                entry_qty, stop, target, risk_r, realized_r, mae_r, mfe_r, regime, opened_bar,
                closed_bar, context_snapshot, legs_json)
            VALUES ('paper:hyperliquid:cfgA', 't-1', 'hyperliquid', 'GHOST', 1, 100.0, ?, 2.0,
                90.0, NULL, 50.0, NULL, 0.0, 0.0, 'range', 0, NULL, NULL, '{}')
            """,
            [ts0.replace(tzinfo=None)],
        )

    client = TestClient(create_app(data_dir=data_dir))
    positions = client.get("/api/overview").json()["venues"]["hyperliquid"]["positions"]
    assert len(positions) == 1
    assert positions[0]["symbol"] == "GHOST"
    assert positions[0]["unrealized_r"] is None


def test_overview_equity_grouped_by_run_id(data_dir: Path) -> None:
    ts0 = datetime(2026, 1, 1, tzinfo=UTC)
    with _hl_store(data_dir) as store:
        store.write_equity("paper:hyperliquid:cfgA", [(ts0, 10_000.0), (ts0 + timedelta(hours=4), 10_050.0)])
        store.write_equity("paper:hyperliquid:cfgB", [(ts0, 5_000.0)])
        store.write_equity("some-other-run", [(ts0, 999.0)])  # not a paper run for this venue

    client = TestClient(create_app(data_dir=data_dir))
    equity = client.get("/api/overview").json()["venues"]["hyperliquid"]["equity"]
    assert set(equity) == {"paper:hyperliquid:cfgA", "paper:hyperliquid:cfgB"}
    assert len(equity["paper:hyperliquid:cfgA"]) == 2
    assert equity["paper:hyperliquid:cfgA"][0][1] == 10_000.0
    assert equity["paper:hyperliquid:cfgB"][0][1] == 5_000.0


def test_overview_fills_last_20_across_paper_runs(data_dir: Path) -> None:
    ts0 = datetime(2026, 1, 1, tzinfo=UTC)
    with _hl_store(data_dir) as store:
        for i in range(25):
            fill = Fill(
                order_id=f"o{i}",
                ts=ts0 + timedelta(hours=4 * i),
                price=100.0 + i,
                qty=1.0,
                cost=CostBreakdown(spread=0.1, commission=0.2, funding=0.05, slippage=0.05),
                leg="entry",
                trade_id=f"t{i}",
            )
            store.write_fills("paper:hyperliquid:cfgA", [fill])

    client = TestClient(create_app(data_dir=data_dir))
    fills = client.get("/api/overview").json()["venues"]["hyperliquid"]["fills"]
    assert len(fills) == 20
    # newest first
    assert fills[0]["order_id"] == "o24"
    assert fills[0]["cost_total"] == pytest.approx(0.4)


# -- staleness ------------------------------------------------------------------


def test_staleness_fresh_and_stale(data_dir: Path) -> None:
    now = datetime.now(UTC)
    fresh_ts = now - timedelta(hours=4)  # bar closed "now" -> 0 bars behind
    stale_ts = now - timedelta(hours=24)  # bar closed 20h ago -> 5 bars behind (> 2)
    with _hl_store(data_dir) as store:
        store.upsert_instruments([BTC, ETH])
        store.upsert_bars([_bar(BTC, fresh_ts), _bar(ETH, stale_ts)])

    client = TestClient(create_app(data_dir=data_dir))
    overview = client.get("/api/overview").json()
    staleness = {row["symbol"]: row for row in overview["venues"]["hyperliquid"]["staleness"]}
    assert staleness["BTC"]["bars_behind"] == 0
    assert staleness["BTC"]["stale"] is False
    assert staleness["ETH"]["bars_behind"] == 5
    assert staleness["ETH"]["stale"] is True


def test_staleness_skips_instrument_with_no_bars(data_dir: Path) -> None:
    with _hl_store(data_dir) as store:
        store.upsert_instruments([BTC])
    client = TestClient(create_app(data_dir=data_dir))
    staleness = client.get("/api/overview").json()["venues"]["hyperliquid"]["staleness"]
    assert staleness == []


# -- settings -------------------------------------------------------------------


def test_put_settings_risk_over_cap_422(data_dir: Path) -> None:
    client = TestClient(create_app(data_dir=data_dir))
    resp = client.put(
        "/api/settings",
        json={"venue": "hyperliquid", "settings": {"risk_pct": 0.03}},
    )
    assert resp.status_code == 422


def test_put_settings_bumps_version_and_logs_exactly_once_per_call(data_dir: Path) -> None:
    client = TestClient(create_app(data_dir=data_dir))
    resp1 = client.put("/api/settings", json={"venue": "hyperliquid", "settings": {"risk_pct": 0.015}})
    assert resp1.status_code == 200
    assert resp1.json() == {"venue": "hyperliquid", "version": 1}

    resp2 = client.put("/api/settings", json={"venue": "hyperliquid", "settings": {"kill_switch": True}})
    assert resp2.status_code == 200
    assert resp2.json() == {"venue": "hyperliquid", "version": 2}

    with Store(data_dir / "hyperliquid.duckdb", read_only=True) as store:
        version, settings = store.current_settings()
        assert version == 2
        assert settings.kill_switch is True
        log = store.settings_log()
        assert [entry["version"] for entry in log] == [1, 2]
        assert all(entry["actor"] == "dashboard" for entry in log)


def test_put_settings_creates_venue_file_if_missing(data_dir: Path) -> None:
    assert not (data_dir / "oanda.duckdb").exists()
    client = TestClient(create_app(data_dir=data_dir))
    resp = client.put("/api/settings", json={"venue": "oanda", "settings": {}})
    assert resp.status_code == 200
    assert (data_dir / "oanda.duckdb").exists()


def test_get_settings_per_venue(data_dir: Path) -> None:
    with _hl_store(data_dir) as store:
        store.write_settings(Settings(risk_pct=0.015), actor="alice")

    client = TestClient(create_app(data_dir=data_dir))
    resp = client.get("/api/settings")
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) == {"hyperliquid"}
    assert body["hyperliquid"]["version"] == 1
    assert body["hyperliquid"]["settings"]["risk_pct"] == 0.015


def test_get_settings_empty_when_no_venue_files(data_dir: Path) -> None:
    client = TestClient(create_app(data_dir=data_dir))
    assert client.get("/api/settings").json() == {}


# -- tournament -------------------------------------------------------------------


def _result_row(
    run_id: str,
    ts: datetime,
    config_id: str,
    *,
    exp_oos: float | None,
    passed: bool | None,
    exit: str | None = "fixedr",
    venue: str | None = "hyperliquid",
    symbol: str | None = "BTC",
    resolution_mode: str | None = "subbars",
    excluded_reason: str | None = None,
    g6: bool | None = True,
    split: str = "oos",
    mar: float = 1.5,
) -> dict:
    return {
        "run_id": run_id,
        "ts": ts,
        "config_id": config_id,
        "entry": "ict",
        "exit": exit,
        "session": "none",
        "venue": venue,
        "symbol": symbol,
        "split": split,
        "n_is": 100,
        "n_oos": 30,
        "exp_is": 0.18,
        "exp_oos": exp_oos,
        "dsr_prob": 0.97,
        "boot_p5": 0.01,
        "diff_p5": 0.02,
        "mar": mar,
        "mar_bh": 0.8,
        "g1": True,
        "g2": True,
        "g3": True,
        "g4": True,
        "g5": True,
        "g6": g6,
        "passed": passed,
        "resolution_mode": resolution_mode,
        "excluded_reason": excluded_reason,
    }


def test_tournament_latest_404_when_empty(data_dir: Path) -> None:
    client = TestClient(create_app(data_dir=data_dir))
    resp = client.get("/api/tournament/latest")
    assert resp.status_code == 404
    assert resp.json() == {"detail": "no tournament results"}


def test_tournament_latest_skips_a_present_store_with_no_tournament_rows(data_dir: Path) -> None:
    """A present venue whose `results` table has rows, but none of them a tournament run
    (only paper-run rows, which `latest_tournament_run` must ignore), must not stop the
    search -- the other venue's real tournament run should still be found."""
    ts = datetime(2026, 1, 1, tzinfo=UTC)
    with _oanda_store(data_dir) as store:
        store.write_results([_result_row("paper:oanda:cfgX", ts, "cfgX", exp_oos=0.1, passed=True)])
    with _hl_store(data_dir) as store:
        store.write_results([_result_row("run-1", ts, "cfg1", exp_oos=0.2, passed=True)])

    client = TestClient(create_app(data_dir=data_dir))
    resp = client.get("/api/tournament/latest")
    assert resp.status_code == 200
    assert resp.json()["run_id"] == "run-1"


def test_tournament_latest_picks_latest_run_and_shapes_response(data_dir: Path) -> None:
    old_ts = datetime(2026, 1, 1, tzinfo=UTC)
    older_ts = datetime(2025, 1, 1, tzinfo=UTC)
    new_ts = datetime(2026, 6, 1, tzinfo=UTC)

    # a second venue with only an *older* tournament run -- exercises pooling the "latest
    # run" search across multiple stores (the cross-store comparison, not just cross-run
    # within one store's own single best-row query).
    with _oanda_store(data_dir) as store:
        store.upsert_instruments([EURUSD])
        store.write_results([_result_row("run-fx-old", older_ts, "cfgFX", exp_oos=0.10, passed=True)])

    with _hl_store(data_dir) as store:
        store.upsert_instruments([BTC, ETH, SOL])
        # an older tournament run: must lose to the newer one even though its exp_oos is huge.
        store.write_results([_result_row("run-old", old_ts, "cfgZ", exp_oos=0.99, passed=True)])

        store.write_results(
            [
                _result_row(
                    "run-new",
                    new_ts,
                    "cfgA",
                    exp_oos=0.20,
                    passed=True,
                    venue="hyperliquid",
                    symbol="BTC",
                    mar=float("inf"),  # zero-drawdown MAR -- must come back as JSON null, not Infinity
                ),
                _result_row(
                    "run-new",
                    new_ts,
                    "cfgB",
                    exp_oos=0.05,
                    passed=False,
                    exit="IS_SELECTED",
                    venue="hyperliquid",
                    symbol="ETH",
                    resolution_mode="pessimistic",
                ),
                _result_row(
                    "run-new",
                    new_ts,
                    "cfgC",
                    exp_oos=None,
                    passed=None,
                    venue="hyperliquid",
                    symbol="SOL",
                    excluded_reason="insufficient_history",
                    resolution_mode=None,
                    g6=None,
                ),
                # fails the gate but has a HIGHER exp_oos than cfgA -- distinguishes "top"
                # (sorted purely by exp_oos) from "gate" (passed desc, then exp_oos desc).
                _result_row(
                    "run-new", new_ts, "cfgD", exp_oos=0.50, passed=False, venue="hyperliquid", symbol="BTC"
                ),
            ]
        )
        # trades for the run's children, for the regime breakdown
        store.write_trades(
            "run-new:cfgA",
            [
                _closed_trade(
                    "ta-1",
                    BTC,
                    entry_ts=old_ts,
                    exit_ts=old_ts + timedelta(hours=8),
                    realized_r=1.0,
                    regime="trend",
                ),
                _closed_trade(
                    "ta-2",
                    BTC,
                    entry_ts=old_ts,
                    exit_ts=old_ts + timedelta(hours=8),
                    realized_r=-0.5,
                    regime="trend",
                ),
                # still open (realized_r is None) -- must be excluded from the regime breakdown.
                _open_trade("ta-3", BTC, entry_ts=old_ts),
            ],
        )
        store.write_trades(
            "run-new:cfgB",
            [
                _closed_trade(
                    "tb-1",
                    ETH,
                    entry_ts=old_ts,
                    exit_ts=old_ts + timedelta(hours=8),
                    realized_r=2.0,
                    regime="range",
                ),
            ],
        )

    client = TestClient(create_app(data_dir=data_dir))
    resp = client.get("/api/tournament/latest")
    assert resp.status_code == 200
    body = resp.json()
    assert body["run_id"] == "run-new"

    gate_ids = [row["config_id"] for row in body["gate"]]
    assert gate_ids[0] == "cfgA"  # only passed=True row goes first regardless of exp_oos
    assert set(gate_ids) == {"cfgA", "cfgB", "cfgC", "cfgD"}
    gate_by_id = {row["config_id"]: row for row in body["gate"]}
    assert gate_by_id["cfgA"]["mar"] is None  # inf MAR sanitized to JSON null, never "Infinity"

    top_ids = [row["config_id"] for row in body["top"]]
    assert top_ids[0] == "cfgD"  # top ignores `passed`, sorts purely by exp_oos desc
    assert top_ids[1] == "cfgA"

    selected_ids = [row["config_id"] for row in body["selected"]]
    assert selected_ids == ["cfgB"]

    excluded_ids = [row["config_id"] for row in body["excluded"]]
    assert excluded_ids == ["cfgC"]

    cost_stress = {row["config_id"]: row for row in body["cost_stress"]}
    assert cost_stress["cfgC"]["g6"] is None
    assert cost_stress["cfgA"]["exp_oos"] == pytest.approx(0.20)

    resolution_modes = {
        (row["venue"], row["symbol"]): row["resolution_mode"] for row in body["resolution_modes"]
    }
    assert resolution_modes[("hyperliquid", "BTC")] == "subbars"
    assert resolution_modes[("hyperliquid", "ETH")] == "pessimistic"
    assert resolution_modes[("hyperliquid", "SOL")] is None

    regime = {row["regime"]: row for row in body["regime"]}
    assert regime["trend"]["n"] == 2
    assert regime["trend"]["expectancy"] == pytest.approx(0.25)
    assert regime["range"]["n"] == 1
    assert regime["range"]["expectancy"] == pytest.approx(2.0)


def test_tournament_resolution_modes_skips_rows_with_no_venue_or_symbol(data_dir: Path) -> None:
    ts = datetime(2026, 1, 1, tzinfo=UTC)
    with _hl_store(data_dir) as store:
        store.write_results(
            [
                {
                    "run_id": "run-x",
                    "ts": ts,
                    "config_id": "c1",
                    "split": "oos",
                }
            ]
        )
    client = TestClient(create_app(data_dir=data_dir))
    body = client.get("/api/tournament/latest").json()
    assert body["resolution_modes"] == []
    assert body["gate"][0]["passed"] is None


def test_tournament_payload_serves_universe_rows_apart(data_dir: Path) -> None:
    """A `venue:*` row pools instruments: it is not an instrument with a fill mode of its own,
    and unmarked in `gate`/`top`/`cost_stress` it would crowd the instruments' own rows out."""
    ts = datetime(2026, 1, 1, tzinfo=UTC)
    shared = {
        "run_id": "run-u",
        "ts": ts,
        "split": "pooled",
        "venue": "hyperliquid",
        "entry": "ict",
        "session": "none",
    }
    btc = "ict|fixed_r_2|none|hyperliquid:BTC"
    thin = "ict|fixed_r_3|none|hyperliquid:*"
    graded = "ict|fixed_r_2|none|hyperliquid:*"
    view = "ict|IS_SELECTED|none|hyperliquid:*"
    with _hl_store(data_dir) as store:
        store.write_results(
            [
                {
                    **shared,
                    "config_id": btc,
                    "exit": "fixed_r_2",
                    "symbol": "BTC",
                    "exp_oos": 0.1,
                    "resolution_mode": "subbars",
                },
                {
                    **shared,
                    "config_id": thin,
                    "exit": "fixed_r_3",
                    "symbol": "*",
                    "exp_oos": 9.0,
                    "g1": False,
                    "passed": False,
                    "resolution_mode": "mixed",
                },
                {
                    **shared,
                    "config_id": graded,
                    "exit": "fixed_r_2",
                    "symbol": "*",
                    "exp_oos": 0.2,
                    "g1": True,
                    "passed": False,
                    "resolution_mode": "mixed",
                },
                {
                    **shared,
                    "config_id": view,
                    "exit": "IS_SELECTED",
                    "symbol": "*",
                    "exp_oos": 0.3,
                    "g1": False,
                    "passed": False,
                    "resolution_mode": "mixed",
                },
            ]
        )
    client = TestClient(create_app(data_dir=data_dir))
    body = client.get("/api/tournament/latest").json()
    assert body["resolution_modes"] == [
        {"venue": "hyperliquid", "symbol": "BTC", "resolution_mode": "subbars"}
    ]
    for key in ("gate", "top", "cost_stress"):
        assert [row["config_id"] for row in body[key]] == [btc], key
    assert body["selected"] == []
    # the row that cleared rule 1 leads, whatever a thinner pool's expectancy says
    assert [row["config_id"] for row in body["universe"]] == [graded, thin, view]
    assert body["universe_omitted"] == 0


def test_the_universe_cap_keeps_the_rows_that_cleared_rule_1(data_dir: Path) -> None:
    """A full sweep holds ~150 universe rows, most of them thin pools whose few trades flatter
    their expectancy. Capped on `exp_oos` alone they would push out the handful the gate could
    grade - the only rows the table exists to show."""
    ts = datetime(2026, 1, 1, tzinfo=UTC)
    shared = {"run_id": "run-cap", "ts": ts, "split": "pooled", "venue": "hyperliquid", "symbol": "*"}
    thin = [
        {
            **shared,
            "config_id": f"zones|thin_{i:02d}|none|hyperliquid:*",
            "entry": "zones",
            "exit": f"thin_{i:02d}",
            "session": "none",
            "exp_oos": 1.0 + i / 100,
            "g1": False,
            "passed": False,
        }
        for i in range(60)
    ]
    graded = [
        {
            **shared,
            "config_id": f"ict|graded_{i}|none|hyperliquid:*",
            "entry": "ict",
            "exit": f"graded_{i}",
            "session": "none",
            "exp_oos": 0.12,
            "g1": True,
            "passed": False,
        }
        for i in range(6)
    ]
    with _hl_store(data_dir) as store:
        store.write_results([*thin, *graded])
    client = TestClient(create_app(data_dir=data_dir))
    body = client.get("/api/tournament/latest").json()
    served = [row["config_id"] for row in body["universe"]]
    assert len(served) == 50 and body["universe_omitted"] == 16
    # equal expectancies: the config id breaks the tie, so the order never rests on scan order
    assert served[:6] == [f"ict|graded_{i}|none|hyperliquid:*" for i in range(6)]
    assert body["gate"] == [] and body["top"] == []


def test_tournament_regime_breakdown_weighted_across_stores(data_dir: Path) -> None:
    """I1: the cross-store merge must be n-weighted, not a plain average of each store's own
    average -- this distinguishes the correct (1+2+3+10)/4=4.0 from the naive (2.0+10.0)/2=6.0."""
    ts = datetime(2026, 1, 1, tzinfo=UTC)
    with _hl_store(data_dir) as store:
        store.upsert_instruments([BTC])
        store.write_results([_result_row("run-multi", ts, "cfgH", exp_oos=0.1, passed=True)])
        store.write_trades(
            "run-multi:cfgH",
            [
                _closed_trade("h1", BTC, entry_ts=ts, exit_ts=ts + timedelta(hours=8), realized_r=1.0),
                _closed_trade("h2", BTC, entry_ts=ts, exit_ts=ts + timedelta(hours=8), realized_r=2.0),
                _closed_trade("h3", BTC, entry_ts=ts, exit_ts=ts + timedelta(hours=8), realized_r=3.0),
            ],
        )
    with _oanda_store(data_dir) as store:
        store.upsert_instruments([EURUSD])
        store.write_results(
            [_result_row("run-multi", ts, "cfgO", exp_oos=0.05, passed=True, venue="oanda", symbol="EURUSD")]
        )
        store.write_trades(
            "run-multi:cfgO",
            [_closed_trade("o1", EURUSD, entry_ts=ts, exit_ts=ts + timedelta(hours=8), realized_r=10.0)],
        )

    client = TestClient(create_app(data_dir=data_dir))
    body = client.get("/api/tournament/latest").json()
    assert body["run_id"] == "run-multi"
    regime = {row["regime"]: row for row in body["regime"]}
    assert regime["trend"]["n"] == 4
    assert regime["trend"]["expectancy"] == pytest.approx(4.0)


def test_tournament_gate_and_cost_stress_are_capped_and_deterministic(data_dir: Path) -> None:
    """I4: `gate`/`cost_stress` are capped to every passing config + top 50 by exp_oos + every
    IS_SELECTED row, deduplicated, plus an `omitted` count -- not the unbounded full row set."""
    ts = datetime(2026, 1, 1, tzinfo=UTC)
    rows = [
        _result_row("run-cap", ts, f"cfg-plain-{i}", exp_oos=float(60 - i), passed=False, exit="fixedr")
        for i in range(60)
    ]
    # two passing configs with exp_oos far below the top-50 cutoff -- kept on `passed` alone.
    rows.append(_result_row("run-cap", ts, "cfg-pass-1", exp_oos=-1.0, passed=True))
    rows.append(_result_row("run-cap", ts, "cfg-pass-2", exp_oos=-2.0, passed=True))
    # one IS_SELECTED config, also below the cutoff -- kept regardless of `passed`.
    rows.append(_result_row("run-cap", ts, "cfg-selected", exp_oos=-3.0, passed=False, exit="IS_SELECTED"))

    with _hl_store(data_dir) as store:
        store.write_results(rows)

    client = TestClient(create_app(data_dir=data_dir))
    resp1 = client.get("/api/tournament/latest")
    resp2 = client.get("/api/tournament/latest")
    assert resp1.status_code == 200
    body1, body2 = resp1.json(), resp2.json()
    assert body1 == body2  # deterministic across repeated calls

    expected_ids = {f"cfg-plain-{i}" for i in range(50)} | {"cfg-pass-1", "cfg-pass-2", "cfg-selected"}
    assert {row["config_id"] for row in body1["gate"]} == expected_ids
    assert {row["config_id"] for row in body1["cost_stress"]} == expected_ids
    assert len(body1["gate"]) == 53
    assert len(body1["cost_stress"]) == 53
    assert body1["gate_omitted"] == len(rows) - 53
    assert body1["cost_stress_omitted"] == len(rows) - 53


# -- trade detail -----------------------------------------------------------------


def test_trade_detail_returns_window_and_fields(data_dir: Path) -> None:
    entry_ts = datetime(2026, 3, 10, tzinfo=UTC)
    exit_ts = entry_ts + timedelta(hours=8)
    start = entry_ts - timedelta(hours=4 * 30)  # 2026-03-05T00:00
    end = exit_ts + timedelta(hours=4 * 10) + timedelta(hours=4)  # 2026-03-12T04:00

    with _hl_store(data_dir) as store:
        store.upsert_instruments([BTC])
        trade = _closed_trade("t-1", BTC, entry_ts=entry_ts, exit_ts=exit_ts, realized_r=1.2, regime="trend")
        store.write_trades("run-1:cfgA", [trade])
        store.upsert_bars(
            [
                _bar(BTC, start - timedelta(hours=4), close=90.0),  # before window: excluded
                _bar(BTC, start, close=91.0),  # at start: included
                _bar(BTC, entry_ts, close=100.0),  # mid: included
                _bar(BTC, end - timedelta(hours=4), close=110.0),  # just before end: included
                _bar(BTC, end, close=111.0),  # at end: excluded (end is exclusive)
                _bar(BTC, end + timedelta(hours=4), close=112.0),  # after window: excluded
            ]
        )

    client = TestClient(create_app(data_dir=data_dir))
    resp = client.get("/api/trades/run-1:cfgA/t-1")
    assert resp.status_code == 200
    body = resp.json()
    assert body["trade_id"] == "t-1"
    assert body["venue"] == "hyperliquid"
    assert body["symbol"] == "BTC"
    assert body["stop"] == trade.stop
    assert body["target"] == trade.target
    assert body["mae_r"] == pytest.approx(trade.mae_r)
    assert body["mfe_r"] == pytest.approx(trade.mfe_r)
    assert len(body["legs"]) == 1

    bar_closes = [bar["close"] for bar in body["bars"]]
    assert bar_closes == [91.0, 100.0, 110.0]


def test_trade_detail_open_trade_uses_entry_ts_for_window_when_no_legs(data_dir: Path) -> None:
    entry_ts = datetime(2026, 3, 10, tzinfo=UTC)
    with _hl_store(data_dir) as store:
        store.upsert_instruments([BTC])
        trade = _open_trade("t-1", BTC, entry_ts=entry_ts)
        store.write_trades("paper:hyperliquid:cfgA", [trade])
        store.upsert_bars([_bar(BTC, entry_ts, close=100.0)])

    client = TestClient(create_app(data_dir=data_dir))
    resp = client.get("/api/trades/paper:hyperliquid:cfgA/t-1")
    assert resp.status_code == 200
    assert resp.json()["legs"] == []
    assert len(resp.json()["bars"]) == 1


def test_trade_detail_404_when_run_missing(data_dir: Path) -> None:
    client = TestClient(create_app(data_dir=data_dir))
    resp = client.get("/api/trades/no-such-run/t-1")
    assert resp.status_code == 404


def test_trade_detail_404_when_trade_id_missing(data_dir: Path) -> None:
    entry_ts = datetime(2026, 3, 10, tzinfo=UTC)
    with _hl_store(data_dir) as store:
        store.upsert_instruments([BTC])
        trade = _open_trade("t-1", BTC, entry_ts=entry_ts)
        store.write_trades("paper:hyperliquid:cfgA", [trade])

    client = TestClient(create_app(data_dir=data_dir))
    resp = client.get("/api/trades/paper:hyperliquid:cfgA/does-not-exist")
    assert resp.status_code == 404


def test_trade_detail_raises_when_instrument_missing_from_instruments_table(data_dir: Path) -> None:
    """I2: mirrors `Store.trades`' own defensive guard -- a `trades` row whose instrument is
    missing from `instruments` (a data inconsistency the public `Store` API can't itself
    produce, only reachable here via a raw `store._conn` insert) must raise rather than
    silently reconstruct a trade with a wrong/missing instrument."""
    entry_ts = datetime(2026, 3, 10, tzinfo=UTC)
    with _hl_store(data_dir) as store:
        store._conn.execute(
            """
            INSERT INTO trades (run_id, id, venue, symbol, direction, entry_price, entry_ts,
                entry_qty, stop, target, risk_r, realized_r, mae_r, mfe_r, regime, opened_bar,
                closed_bar, context_snapshot, legs_json)
            VALUES ('run-1', 't-1', 'hyperliquid', 'GHOST', 1, 100.0, ?, 2.0,
                90.0, NULL, 50.0, NULL, 0.0, 0.0, 'range', 0, NULL, NULL, '{}')
            """,
            [entry_ts.replace(tzinfo=None)],
        )
        with pytest.raises(ValueError, match="GHOST"):
            queries._find_trade({"hyperliquid": store}, "run-1", "t-1")


def test_trade_detail_selects_correct_trade_among_several_in_same_run(data_dir: Path) -> None:
    """I2: the single-row lookup must filter by `id` too, not just return the run's first row."""
    entry_ts = datetime(2026, 3, 10, tzinfo=UTC)
    with _hl_store(data_dir) as store:
        store.upsert_instruments([BTC])
        trade_a = _closed_trade(
            "t-a", BTC, entry_ts=entry_ts, exit_ts=entry_ts + timedelta(hours=8), realized_r=1.0
        )
        trade_b = _closed_trade(
            "t-b", BTC, entry_ts=entry_ts, exit_ts=entry_ts + timedelta(hours=8), realized_r=-2.0
        )
        store.write_trades("run-1:cfgA", [trade_a, trade_b])

    client = TestClient(create_app(data_dir=data_dir))
    resp = client.get("/api/trades/run-1:cfgA/t-b")
    assert resp.status_code == 200
    body = resp.json()
    assert body["trade_id"] == "t-b"
    assert body["realized_r"] == pytest.approx(-2.0)


# -- static HTML shell (M2, M3, M5) --------------------------------------------------


def test_static_html_removed_dead_settings_fields_constant(data_dir: Path) -> None:
    """M2: `SETTINGS_FIELDS` was declared but never referenced anywhere -- dead code, deleted."""
    html = TestClient(create_app(data_dir=data_dir, token=None)).get("/").text
    assert "SETTINGS_FIELDS" not in html


def test_static_html_overview_polling_checks_visibility_and_active_view(data_dir: Path) -> None:
    """M3: the 10s overview poll must not run unconditionally in the background forever -- it
    has to check both which tab is active and `document.hidden`."""
    html = TestClient(create_app(data_dir=data_dir, token=None)).get("/").text
    assert "document.hidden" in html
    assert "activeView" in html


def test_static_html_token_change_reloads_active_view(data_dir: Path) -> None:
    """M5: changing the token must reload the current view (a stale render was left showing
    data fetched under the old/no token otherwise), not just silently persist it."""
    html = TestClient(create_app(data_dir=data_dir, token=None)).get("/").text
    assert "reloadActiveView" in html


# -- module-level app -----------------------------------------------------------------


def test_module_level_app_is_a_fastapi_instance() -> None:
    from swingforge.web.app import app

    assert app is not None
    assert TestClient(app).get("/").status_code == 200
