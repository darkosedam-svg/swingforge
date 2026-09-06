"""DuckDB-backed persistence: one file per venue (design spec section 5).

Every table lives in a single DuckDB database (one file per venue, or ``":memory:"`` for
tests). Timestamps are stored as naive ``TIMESTAMP`` columns holding a UTC instant rather
than DuckDB's native ``TIMESTAMPTZ`` — see the module-level note below for why — and are
always handed back to callers as timezone-aware UTC ``datetime`` objects.

``Decimal`` fields on :class:`~swingforge.core.types.Instrument` (``tick_size``,
``contract_multiplier``) are stored as ``VARCHAR`` of ``str(Decimal)``, which round-trips
exactly through the ``Decimal`` constructor.

Gotcha: the installed ``duckdb`` raises ``ModuleNotFoundError: No module named 'pytz'``
when fetching a native ``TIMESTAMPTZ`` column, because it lazily imports ``pytz`` to build
the tz-aware Python object on the way out. ``pytz`` is not a project dependency and this
package may not touch ``pyproject.toml``, so every timestamp column here is a plain
``TIMESTAMP`` (no tz) storing the UTC instant; tz-awareness is reattached in Python with
``.replace(tzinfo=UTC)`` on read and stripped with ``.replace(tzinfo=None)`` on write (safe
because every datetime accepted by :mod:`swingforge.core.types` is already normalised to
UTC by its validators).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import duckdb
import numpy as np

from swingforge.core.settings import Settings
from swingforge.core.types import TF, Bar, Fill, Instrument, Order, Trade

__all__ = ["DuckDBSettingsReader", "Store"]

_RESULTS_COLUMNS = (
    "run_id",
    "ts",
    "config_id",
    "entry",
    "exit",
    "session",
    "venue",
    "symbol",
    "split",
    "n_is",
    "n_oos",
    "exp_is",
    "exp_oos",
    "dsr_prob",
    "boot_p5",
    "diff_p5",
    "mar",
    "mar_bh",
    "g1",
    "g2",
    "g3",
    "g4",
    "g5",
    "g6",
    "passed",
    "resolution_mode",
    "excluded_reason",
)

_TRADES_COLUMNS = (
    "run_id",
    "id",
    "venue",
    "symbol",
    "direction",
    "entry_price",
    "entry_ts",
    "entry_qty",
    "stop",
    "target",
    "risk_r",
    "realized_r",
    "mae_r",
    "mfe_r",
    "regime",
    "opened_bar",
    "closed_bar",
    "context_snapshot",
    "legs_json",
)


def _to_naive_utc(value: datetime) -> datetime:
    """Normalise a tz-aware datetime to UTC and strip tzinfo for storage."""
    return value.astimezone(UTC).replace(tzinfo=None)


def _from_naive_utc(value: datetime) -> datetime:
    """Reattach UTC tzinfo to a naive datetime read back from storage."""
    return value.replace(tzinfo=UTC)


class Store:
    """One DuckDB file (or ``":memory:"``) holding every table for one venue."""

    def __init__(self, path: str | Path, read_only: bool = False) -> None:
        self._conn = duckdb.connect(str(path), read_only=read_only)
        if not read_only:
            self.create_schema()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        self._conn.close()

    # -- schema ---------------------------------------------------------------

    def create_schema(self) -> None:
        """Create every table if it does not already exist. Safe to call repeatedly.

        Uses ``CREATE TABLE IF NOT EXISTS`` throughout, so this stays idempotent even as
        columns/constraints evolve -- but that also means an existing on-disk database
        created before a primary key was added to a table (see the WU-1E code-review pass:
        ``orders``, ``fills``, ``equity`` and ``results`` all gained PKs) keeps its old,
        constraint-less schema; ``CREATE TABLE IF NOT EXISTS`` does not retrofit
        constraints onto a table that already exists. Any such pre-existing database file
        must be deleted and rebuilt from source data to pick up the new PKs and the
        idempotent-upsert behaviour they enable. Acceptable at this stage (no run has
        shipped real data yet); a migration path is out of scope here.
        """
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS instruments (
                venue VARCHAR NOT NULL,
                symbol VARCHAR NOT NULL,
                tick_size VARCHAR NOT NULL,
                contract_multiplier VARCHAR NOT NULL,
                quote_ccy VARCHAR NOT NULL,
                session_profile VARCHAR NOT NULL,
                PRIMARY KEY (venue, symbol)
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS bars (
                venue VARCHAR NOT NULL,
                symbol VARCHAR NOT NULL,
                tf VARCHAR NOT NULL,
                ts_open TIMESTAMP NOT NULL,
                open DOUBLE NOT NULL,
                high DOUBLE NOT NULL,
                low DOUBLE NOT NULL,
                close DOUBLE NOT NULL,
                volume DOUBLE NOT NULL,
                bid_close DOUBLE,
                ask_close DOUBLE,
                PRIMARY KEY (venue, symbol, tf, ts_open)
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS funding (
                venue VARCHAR NOT NULL,
                symbol VARCHAR NOT NULL,
                ts TIMESTAMP NOT NULL,
                rate DOUBLE NOT NULL,
                PRIMARY KEY (venue, symbol, ts)
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS orders (
                run_id VARCHAR NOT NULL,
                id VARCHAR NOT NULL,
                venue VARCHAR NOT NULL,
                symbol VARCHAR NOT NULL,
                direction INTEGER NOT NULL,
                qty DOUBLE NOT NULL,
                kind VARCHAR NOT NULL,
                price DOUBLE,
                expires_at_bar INTEGER,
                leg VARCHAR NOT NULL,
                trade_id VARCHAR,
                PRIMARY KEY (run_id, id)
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS fills (
                run_id VARCHAR NOT NULL,
                order_id VARCHAR NOT NULL,
                ts TIMESTAMP NOT NULL,
                price DOUBLE NOT NULL,
                qty DOUBLE NOT NULL,
                spread DOUBLE NOT NULL,
                commission DOUBLE NOT NULL,
                funding DOUBLE NOT NULL,
                slippage DOUBLE NOT NULL,
                leg VARCHAR NOT NULL,
                trade_id VARCHAR,
                PRIMARY KEY (run_id, order_id, leg, ts)
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS trades (
                run_id VARCHAR NOT NULL,
                id VARCHAR NOT NULL,
                venue VARCHAR NOT NULL,
                symbol VARCHAR NOT NULL,
                direction INTEGER NOT NULL,
                entry_price DOUBLE NOT NULL,
                entry_ts TIMESTAMP NOT NULL,
                entry_qty DOUBLE NOT NULL,
                stop DOUBLE NOT NULL,
                target DOUBLE,
                risk_r DOUBLE NOT NULL,
                realized_r DOUBLE,
                mae_r DOUBLE NOT NULL,
                mfe_r DOUBLE NOT NULL,
                regime VARCHAR NOT NULL,
                opened_bar INTEGER NOT NULL,
                closed_bar INTEGER,
                context_snapshot BLOB,
                legs_json VARCHAR NOT NULL,
                PRIMARY KEY (run_id, id)
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS equity (
                run_id VARCHAR NOT NULL,
                ts TIMESTAMP NOT NULL,
                equity DOUBLE NOT NULL,
                PRIMARY KEY (run_id, ts)
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS results (
                run_id VARCHAR NOT NULL,
                ts TIMESTAMP NOT NULL,
                config_id VARCHAR NOT NULL,
                entry VARCHAR,
                exit VARCHAR,
                session VARCHAR,
                venue VARCHAR,
                symbol VARCHAR,
                split VARCHAR NOT NULL,
                n_is INTEGER,
                n_oos INTEGER,
                exp_is DOUBLE,
                exp_oos DOUBLE,
                dsr_prob DOUBLE,
                boot_p5 DOUBLE,
                diff_p5 DOUBLE,
                mar DOUBLE,
                mar_bh DOUBLE,
                g1 BOOLEAN,
                g2 BOOLEAN,
                g3 BOOLEAN,
                g4 BOOLEAN,
                g5 BOOLEAN,
                g6 BOOLEAN,
                passed BOOLEAN,
                resolution_mode VARCHAR,
                excluded_reason VARCHAR,
                PRIMARY KEY (run_id, config_id, split)
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS settings (
                version INTEGER NOT NULL,
                payload VARCHAR NOT NULL,
                updated_at TIMESTAMP NOT NULL
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS settings_log (
                version INTEGER NOT NULL,
                payload VARCHAR NOT NULL,
                ts TIMESTAMP NOT NULL,
                actor VARCHAR NOT NULL
            )
            """
        )

    # -- bulk-write helper ------------------------------------------------------

    def _bulk_upsert(
        self,
        table: str,
        columns: Sequence[str],
        rows: Sequence[tuple[Any, ...]],
        conflict_cols: Sequence[str],
        *,
        str_cols: frozenset[str] = frozenset(),
        float_cols: frozenset[str] = frozenset(),
        int_cols: frozenset[str] = frozenset(),
        datetime_cols: frozenset[str] = frozenset(),
        nullable_str_cols: frozenset[str] = frozenset(),
        nullable_float_cols: frozenset[str] = frozenset(),
        nullable_int_cols: frozenset[str] = frozenset(),
        nullable_bool_cols: frozenset[str] = frozenset(),
    ) -> None:
        """Upsert ``rows`` into ``table`` with one statement instead of one per row.

        Every bulk writer used to build its INSERT with ``con.executemany`` -- one
        prepared-statement round trip (with an ``ON CONFLICT`` constraint check) *per
        row*, which measured at roughly 20 rows/s for `upsert_bars` (a 10,000-row upsert
        would take on the order of 8 minutes). Instead: build one numpy array per column,
        register the whole dict as a relation, and issue a single ``INSERT ... SELECT ...
        FROM <relation> ON CONFLICT (...) DO UPDATE`` -- one columnar transfer and one
        statement, however many rows.

        Every column must be declared into exactly one of the typed buckets below --
        deliberately, there is no untyped/object-array fallback for anything perf
        -sensitive. A `numpy.ndarray` of ``dtype=object`` measured at ~17s to *register* a
        single column of 10,000 identical values in this environment (no `pandas`
        installed; DuckDB's numpy-object path apparently falls back to a per-element
        Python scan), against ~0.001-0.04s for a native-dtype array of the same size --
        three orders of magnitude, and it hit every nullable/string column indiscriminately.
        A nullable column instead gets two arrays: a native-dtype "value" array with an
        arbitrary placeholder (``0``/``0.0``/``""``/``False``) standing in for `None`, and a
        parallel boolean "is `None`" mask; the SELECT reconstructs true SQL ``NULL`` with
        ``CASE WHEN mask THEN NULL ELSE value END``. (The one exception is a `bytes`/BLOB
        column such as `trades.context_snapshot`: numpy has no safe variable-length byte
        dtype, so that one column is built as a plain Python-object array via
        `np.array(values, dtype=object)` outside every bucket below -- acceptable since
        run-artifact volume for `trades` is orders of magnitude below the 10k-row
        `upsert_bars` benchmark this method exists for.)

        Measured at >200,000 rows/s for a 10,000-bar `upsert_bars` (register + insert
        combined), against ~20 rows/s before -- see `test_upsert_bars_10k_under_5s`.

        ``columns`` must list every column of the insert (and match `rows`' tuple order,
        and every column must appear in exactly one of the `*_cols` buckets); every column
        not in `conflict_cols` gets `DO UPDATE SET col = excluded.col`.
        """
        if not rows:
            return
        arrays: dict[str, np.ndarray] = {}
        select_exprs: list[str] = []
        for i, col in enumerate(columns):
            values = [row[i] for row in rows]
            quoted = f'"{col}"'
            if col in datetime_cols:
                arrays[col] = np.array(values, dtype="datetime64[us]")
                select_exprs.append(quoted)
            elif col in float_cols:
                arrays[col] = np.asarray(values, dtype="float64")
                select_exprs.append(quoted)
            elif col in int_cols:
                arrays[col] = np.asarray(values, dtype="int64")
                select_exprs.append(quoted)
            elif col in str_cols:
                arrays[col] = np.array(values)  # native fixed-width unicode, not dtype=object
                select_exprs.append(quoted)
            elif col in nullable_float_cols:
                mask = f"_null_{col}"
                arrays[col] = np.array([0.0 if v is None else v for v in values], dtype="float64")
                arrays[mask] = np.array([v is None for v in values], dtype="bool")
                select_exprs.append(f'CASE WHEN "{mask}" THEN NULL ELSE {quoted} END')
            elif col in nullable_int_cols:
                mask = f"_null_{col}"
                arrays[col] = np.array([0 if v is None else v for v in values], dtype="int64")
                arrays[mask] = np.array([v is None for v in values], dtype="bool")
                select_exprs.append(f'CASE WHEN "{mask}" THEN NULL ELSE {quoted} END')
            elif col in nullable_str_cols:
                mask = f"_null_{col}"
                arrays[col] = np.array(["" if v is None else v for v in values])
                arrays[mask] = np.array([v is None for v in values], dtype="bool")
                select_exprs.append(f'CASE WHEN "{mask}" THEN NULL ELSE {quoted} END')
            elif col in nullable_bool_cols:
                mask = f"_null_{col}"
                arrays[col] = np.array([False if v is None else v for v in values], dtype="bool")
                arrays[mask] = np.array([v is None for v in values], dtype="bool")
                select_exprs.append(f'CASE WHEN "{mask}" THEN NULL ELSE {quoted} END')
            else:
                arrays[col] = np.array(values, dtype=object)  # BLOB/bytes: see docstring
                select_exprs.append(quoted)
        update_cols = [c for c in columns if c not in conflict_cols]
        col_list = ", ".join(f'"{c}"' for c in columns)
        select_list = ", ".join(select_exprs)
        conflict_list = ", ".join(f'"{c}"' for c in conflict_cols)
        set_clause = ", ".join(f'"{c}" = excluded."{c}"' for c in update_cols)
        self._conn.register("_bulk_stage", arrays)
        try:
            self._conn.execute(
                f"INSERT INTO {table} ({col_list}) SELECT {select_list} FROM _bulk_stage "
                f"ON CONFLICT ({conflict_list}) DO UPDATE SET {set_clause}"
            )
        finally:
            self._conn.unregister("_bulk_stage")

    # -- instruments ------------------------------------------------------------

    def upsert_instruments(self, instruments: Sequence[Instrument]) -> None:
        columns = ("venue", "symbol", "tick_size", "contract_multiplier", "quote_ccy", "session_profile")
        rows = [
            (
                inst.venue,
                inst.symbol,
                str(inst.tick_size),
                str(inst.contract_multiplier),
                inst.quote_ccy,
                inst.session_profile,
            )
            for inst in instruments
        ]
        self._bulk_upsert(
            "instruments", columns, rows, conflict_cols=("venue", "symbol"), str_cols=frozenset(columns)
        )

    def instruments(self, venue: str) -> list[Instrument]:
        rows = self._conn.execute(
            """
            SELECT venue, symbol, tick_size, contract_multiplier, quote_ccy, session_profile
            FROM instruments WHERE venue = ? ORDER BY symbol
            """,
            [venue],
        ).fetchall()
        return [self._row_to_instrument(row) for row in rows]

    def _instrument_by_venue_symbol(self, venue: str, symbol: str) -> Instrument | None:
        row = self._conn.execute(
            """
            SELECT venue, symbol, tick_size, contract_multiplier, quote_ccy, session_profile
            FROM instruments WHERE venue = ? AND symbol = ?
            """,
            [venue, symbol],
        ).fetchone()
        return self._row_to_instrument(row) if row is not None else None

    @staticmethod
    def _row_to_instrument(row: tuple[Any, ...]) -> Instrument:
        venue, symbol, tick_size, contract_multiplier, quote_ccy, session_profile = row
        return Instrument(
            venue=venue,
            symbol=symbol,
            tick_size=Decimal(tick_size),
            contract_multiplier=Decimal(contract_multiplier),
            quote_ccy=quote_ccy,
            session_profile=session_profile,
        )

    # -- bars ---------------------------------------------------------------

    def upsert_bars(self, bars: Sequence[Bar]) -> int:
        """Upsert bars on ``(venue, symbol, tf, ts_open)``. Subbars are NOT stored.

        Each bar of a coarser timeframe carries its own row only; if the finer bars
        (e.g. 1H inside a 4H bar) need to be queryable too, upsert them separately as
        their own list of bars. Idempotent: upserting the same bars twice leaves the
        table's row count unchanged. Returns the number of bars *submitted* (``len(bars)``),
        not the number of rows that were newly inserted vs. updated -- upserting an already
        -present bar still counts towards the return value.
        """
        columns = (
            "venue",
            "symbol",
            "tf",
            "ts_open",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "bid_close",
            "ask_close",
        )
        rows = [
            (
                bar.instrument.venue,
                bar.instrument.symbol,
                bar.tf,
                _to_naive_utc(bar.ts_open),
                bar.open,
                bar.high,
                bar.low,
                bar.close,
                bar.volume,
                bar.bid_close,
                bar.ask_close,
            )
            for bar in bars
        ]
        self._bulk_upsert(
            "bars",
            columns,
            rows,
            conflict_cols=("venue", "symbol", "tf", "ts_open"),
            str_cols=frozenset({"venue", "symbol", "tf"}),
            float_cols=frozenset({"open", "high", "low", "close", "volume"}),
            datetime_cols=frozenset({"ts_open"}),
            nullable_float_cols=frozenset({"bid_close", "ask_close"}),
        )
        return len(rows)

    def bars(self, instrument: Instrument, tf: TF, start: datetime, end: datetime) -> list[Bar]:
        rows = self._conn.execute(
            """
            SELECT ts_open, open, high, low, close, volume, bid_close, ask_close
            FROM bars
            WHERE venue = ? AND symbol = ? AND tf = ? AND ts_open >= ? AND ts_open < ?
            ORDER BY ts_open
            """,
            [instrument.venue, instrument.symbol, tf, _to_naive_utc(start), _to_naive_utc(end)],
        ).fetchall()
        return [
            Bar(
                instrument=instrument,
                tf=tf,
                ts_open=_from_naive_utc(ts_open),
                open=open_,
                high=high,
                low=low,
                close=close,
                volume=volume,
                bid_close=bid_close,
                ask_close=ask_close,
            )
            for ts_open, open_, high, low, close, volume, bid_close, ask_close in rows
        ]

    def bar_range(self, instrument: Instrument, tf: TF) -> tuple[datetime, datetime, int] | None:
        row = self._conn.execute(
            """
            SELECT MIN(ts_open), MAX(ts_open), COUNT(*) FROM bars WHERE venue = ? AND symbol = ? AND tf = ?
            """,
            [instrument.venue, instrument.symbol, tf],
        ).fetchone()
        if row is None or row[2] == 0:
            return None
        lo, hi, count = row
        return _from_naive_utc(lo), _from_naive_utc(hi), count

    # -- funding ---------------------------------------------------------------

    def upsert_funding(self, instrument: Instrument, rows: Sequence[tuple[datetime, float]]) -> None:
        values = [(instrument.venue, instrument.symbol, _to_naive_utc(ts), rate) for ts, rate in rows]
        self._bulk_upsert(
            "funding",
            ("venue", "symbol", "ts", "rate"),
            values,
            conflict_cols=("venue", "symbol", "ts"),
            str_cols=frozenset({"venue", "symbol"}),
            float_cols=frozenset({"rate"}),
            datetime_cols=frozenset({"ts"}),
        )

    def funding(self, instrument: Instrument, start: datetime, end: datetime) -> list[tuple[datetime, float]]:
        rows = self._conn.execute(
            """
            SELECT ts, rate FROM funding
            WHERE venue = ? AND symbol = ? AND ts >= ? AND ts < ?
            ORDER BY ts
            """,
            [instrument.venue, instrument.symbol, _to_naive_utc(start), _to_naive_utc(end)],
        ).fetchall()
        return [(_from_naive_utc(ts), rate) for ts, rate in rows]

    # -- run artifacts: orders, fills, trades, equity, results -----------------

    def write_orders(self, run_id: str, orders: Sequence[Order]) -> None:
        """Log ``orders`` for ``run_id``. Idempotent on ``(run_id, id)``: re-writing the same
        order updates its row in place rather than duplicating it (see the WU-1E
        code-review pass: `orders` gained that primary key so a re-run of a partially
        -written batch does not double-count).
        """
        columns = (
            "run_id",
            "id",
            "venue",
            "symbol",
            "direction",
            "qty",
            "kind",
            "price",
            "expires_at_bar",
            "leg",
            "trade_id",
        )
        rows = [
            (
                run_id,
                order.id,
                order.instrument.venue,
                order.instrument.symbol,
                order.direction,
                order.qty,
                order.kind,
                order.price,
                order.expires_at_bar,
                order.leg,
                order.trade_id,
            )
            for order in orders
        ]
        self._bulk_upsert(
            "orders",
            columns,
            rows,
            conflict_cols=("run_id", "id"),
            str_cols=frozenset({"run_id", "id", "venue", "symbol", "kind", "leg"}),
            int_cols=frozenset({"direction"}),
            float_cols=frozenset({"qty"}),
            nullable_float_cols=frozenset({"price"}),
            nullable_int_cols=frozenset({"expires_at_bar"}),
            nullable_str_cols=frozenset({"trade_id"}),
        )

    def write_fills(self, run_id: str, fills: Sequence[Fill]) -> None:
        """Log ``fills`` for ``run_id``. Idempotent on ``(run_id, order_id, leg, ts)``."""
        columns = (
            "run_id",
            "order_id",
            "ts",
            "price",
            "qty",
            "spread",
            "commission",
            "funding",
            "slippage",
            "leg",
            "trade_id",
        )
        rows = [
            (
                run_id,
                fill.order_id,
                _to_naive_utc(fill.ts),
                fill.price,
                fill.qty,
                fill.cost.spread,
                fill.cost.commission,
                fill.cost.funding,
                fill.cost.slippage,
                fill.leg,
                fill.trade_id,
            )
            for fill in fills
        ]
        self._bulk_upsert(
            "fills",
            columns,
            rows,
            conflict_cols=("run_id", "order_id", "leg", "ts"),
            str_cols=frozenset({"run_id", "order_id", "leg"}),
            float_cols=frozenset({"price", "qty", "spread", "commission", "funding", "slippage"}),
            datetime_cols=frozenset({"ts"}),
            nullable_str_cols=frozenset({"trade_id"}),
        )

    def write_trades(self, run_id: str, trades: Sequence[Trade]) -> None:
        rows = []
        for trade in trades:
            venue, symbol = trade.instrument.venue, trade.instrument.symbol
            if self._instrument_by_venue_symbol(venue, symbol) is None:
                raise ValueError(
                    f"cannot write trade {trade.id!r}: instrument {venue}/{symbol} has not been "
                    "upserted into the instruments table (call upsert_instruments first)"
                )
            legs_json = json.dumps(
                {
                    "entry_fill": trade.entry_fill.model_dump(mode="json"),
                    "legs": [leg.model_dump(mode="json") for leg in trade.legs],
                }
            )
            rows.append(
                (
                    run_id,
                    trade.id,
                    venue,
                    symbol,
                    trade.direction,
                    trade.entry_fill.price,
                    _to_naive_utc(trade.entry_fill.ts),
                    trade.entry_fill.qty,
                    trade.stop,
                    trade.target,
                    trade.risk_r,
                    trade.realized_r,
                    trade.mae_r,
                    trade.mfe_r,
                    trade.regime,
                    trade.opened_bar,
                    trade.closed_bar,
                    trade.context_snapshot,
                    legs_json,
                )
            )
        self._bulk_upsert(
            "trades",
            _TRADES_COLUMNS,
            rows,
            conflict_cols=("run_id", "id"),
            str_cols=frozenset({"run_id", "id", "venue", "symbol", "regime", "legs_json"}),
            int_cols=frozenset({"direction", "opened_bar"}),
            float_cols=frozenset({"entry_price", "entry_qty", "stop", "risk_r", "mae_r", "mfe_r"}),
            datetime_cols=frozenset({"entry_ts"}),
            nullable_float_cols=frozenset({"target", "realized_r"}),
            nullable_int_cols=frozenset({"closed_bar"}),
            # context_snapshot (BLOB) is left undeclared -> falls to the object-array bucket;
            # see `_bulk_upsert`'s docstring for why that one column is the accepted exception.
        )

    def trades(self, run_id: str) -> list[Trade]:
        rows = self._conn.execute(
            """
            SELECT id, venue, symbol, direction, stop, target, risk_r, realized_r, mae_r, mfe_r,
                   regime, opened_bar, closed_bar, context_snapshot, legs_json
            FROM trades WHERE run_id = ? ORDER BY opened_bar, id
            """,
            [run_id],
        ).fetchall()
        result = []
        instrument_cache: dict[tuple[str, str], Instrument] = {}
        for (
            trade_id,
            venue,
            symbol,
            direction,
            stop,
            target,
            risk_r,
            realized_r,
            mae_r,
            mfe_r,
            regime,
            opened_bar,
            closed_bar,
            context_snapshot,
            legs_json,
        ) in rows:
            key = (venue, symbol)
            instrument = instrument_cache.get(key)
            if instrument is None:
                instrument = self._instrument_by_venue_symbol(venue, symbol)
                if instrument is None:
                    raise ValueError(
                        f"cannot reconstruct trade {trade_id!r}: instrument {venue}/{symbol} is "
                        "missing from the instruments table"
                    )
                instrument_cache[key] = instrument
            payload = json.loads(legs_json)
            entry_fill = Fill.model_validate(payload["entry_fill"])
            legs = tuple(Fill.model_validate(leg) for leg in payload["legs"])
            result.append(
                Trade(
                    id=trade_id,
                    instrument=instrument,
                    direction=direction,
                    entry_fill=entry_fill,
                    legs=legs,
                    stop=stop,
                    target=target,
                    risk_r=risk_r,
                    realized_r=realized_r,
                    mae_r=mae_r,
                    mfe_r=mfe_r,
                    regime=regime,
                    context_snapshot=context_snapshot or b"",
                    opened_bar=opened_bar,
                    closed_bar=closed_bar,
                )
            )
        return result

    def write_equity(self, run_id: str, rows: Sequence[tuple[datetime, float]]) -> None:
        """Idempotent on ``(run_id, ts)``: re-writing the same timestamp overwrites its equity."""
        values = [(run_id, _to_naive_utc(ts), equity) for ts, equity in rows]
        self._bulk_upsert(
            "equity",
            ("run_id", "ts", "equity"),
            values,
            conflict_cols=("run_id", "ts"),
            str_cols=frozenset({"run_id"}),
            float_cols=frozenset({"equity"}),
            datetime_cols=frozenset({"ts"}),
        )

    def equity(self, run_id: str) -> list[tuple[datetime, float]]:
        rows = self._conn.execute(
            "SELECT ts, equity FROM equity WHERE run_id = ? ORDER BY ts", [run_id]
        ).fetchall()
        return [(_from_naive_utc(ts), equity) for ts, equity in rows]

    def write_results(self, results: Sequence[dict[str, Any]]) -> None:
        """Idempotent on ``(run_id, config_id, split)``: re-writing the same key overwrites it.

        Raises ``ValueError`` on any key not in `_RESULTS_COLUMNS` -- a typo'd or renamed
        result field used to be silently dropped by the old ``result.get(col)`` scan; this
        surfaces it immediately instead.
        """
        ts_index = _RESULTS_COLUMNS.index("ts")  # hoisted: same for every row, not per-row work
        rows = []
        for result in results:
            unknown = set(result) - set(_RESULTS_COLUMNS)
            if unknown:
                raise ValueError(f"unknown results column(s): {sorted(unknown)!r}")
            row = [result.get(col) for col in _RESULTS_COLUMNS]
            ts = row[ts_index]
            if ts is not None:
                row[ts_index] = _to_naive_utc(ts)
            rows.append(tuple(row))
        self._bulk_upsert(
            "results",
            _RESULTS_COLUMNS,
            rows,
            conflict_cols=("run_id", "config_id", "split"),
            str_cols=frozenset({"run_id", "config_id", "split"}),
            datetime_cols=frozenset({"ts"}),
            nullable_str_cols=frozenset(
                {"entry", "exit", "session", "venue", "symbol", "resolution_mode", "excluded_reason"}
            ),
            nullable_int_cols=frozenset({"n_is", "n_oos"}),
            nullable_float_cols=frozenset(
                {"exp_is", "exp_oos", "dsr_prob", "boot_p5", "diff_p5", "mar", "mar_bh"}
            ),
            nullable_bool_cols=frozenset({"g1", "g2", "g3", "g4", "g5", "g6", "passed"}),
        )

    def results(self, run_id: str) -> list[dict[str, Any]]:
        columns = ", ".join(f'"{col}"' for col in _RESULTS_COLUMNS)
        rows = self._conn.execute(
            f'SELECT {columns} FROM results WHERE run_id = ? ORDER BY "ts"', [run_id]
        ).fetchall()
        ts_index = _RESULTS_COLUMNS.index("ts")
        out = []
        for row in rows:
            values = list(row)
            if values[ts_index] is not None:
                values[ts_index] = _from_naive_utc(values[ts_index])
            out.append(dict(zip(_RESULTS_COLUMNS, values, strict=True)))
        return out

    # -- settings ---------------------------------------------------------------

    def current_settings(self) -> tuple[int, Settings]:
        # `settings` is a single-row "current value" table (write_settings DELETEs before
        # INSERTing), so ORDER BY is defensive rather than load-bearing today -- it keeps
        # this method correct even if that single-row invariant is ever relaxed.
        row = self._conn.execute(
            "SELECT version, payload FROM settings ORDER BY version DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return 0, Settings()
        version, payload = row
        return version, Settings.model_validate_json(payload)

    def write_settings(self, settings: Settings, actor: str) -> int:
        """Atomically bump the settings version and append to the log.

        The three statements (clear `settings`, insert the new current row, append to
        `settings_log`) run inside one transaction: if anything raises partway through,
        `ROLLBACK` restores the previous settings row exactly, so a reader never observes
        a version bump without a matching log entry (or vice versa).
        """
        prev_version, _ = self.current_settings()
        new_version = prev_version + 1
        payload = settings.model_dump_json()
        now = datetime.now(UTC)
        self._conn.begin()
        try:
            self._conn.execute("DELETE FROM settings")
            self._conn.execute(
                "INSERT INTO settings (version, payload, updated_at) VALUES (?, ?, ?)",
                [new_version, payload, _to_naive_utc(now)],
            )
            self._conn.execute(
                "INSERT INTO settings_log (version, payload, ts, actor) VALUES (?, ?, ?, ?)",
                [new_version, payload, _to_naive_utc(now), actor],
            )
        except Exception:
            self._conn.rollback()
            raise
        else:
            self._conn.commit()
        return new_version

    def settings_log(self) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT version, payload, ts, actor FROM settings_log ORDER BY version"
        ).fetchall()
        return [
            {"version": version, "payload": json.loads(payload), "ts": _from_naive_utc(ts), "actor": actor}
            for version, payload, ts, actor in rows
        ]


class DuckDBSettingsReader:
    """A `SettingsReader` backed by a `Store`'s `settings` table.

    `current()` re-reads the store on every call, so a write via `Store.write_settings`
    (e.g. from the dashboard's write route) is visible on the very next read.
    """

    def __init__(self, store: Store) -> None:
        self._store = store

    def current(self) -> tuple[int, Settings]:
        return self._store.current_settings()
