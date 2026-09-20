"""SQL and row-shaping helpers backing the dashboard's read routes.

Every function here takes an already-open :class:`~swingforge.adapters.store.Store` (opened
read-only, one per request -- see ``app.py``) and returns plain JSON-safe dicts/lists.
``Store`` publishes no reader for ``fills`` at all, and no way to discover which run ids
exist for a pattern like ``paper:{venue}:%`` -- so those two things go through
``store._conn`` directly rather than opening a second ``duckdb.connect`` to a file a
``Store`` already has open (a second read-only handle to the same file is unnecessary and,
on Windows, risks a file-lock conflict for no benefit within one request).

Conventions this module assumes (see the handoff for the authoritative statement the CLI
must follow):

- Paper run ids are ``paper:{venue}:{config_id}``.
- A tournament's parent run id owns its ``results`` rows; each config's ``trades``/``equity``
  rows live under the child run id ``f"{run_id}:{config_id}"``.
- An open paper position is a ``trades`` row with ``closed_bar IS NULL``.
"""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from swingforge.adapters.store import Store
from swingforge.core.types import Bar, Fill, Instrument, Trade

__all__ = [
    "VENUES",
    "latest_4h_bar",
    "latest_tournament_run",
    "overview_for_venue",
    "present_venues",
    "regime_breakdown",
    "settings_payload",
    "to_json_float",
    "tournament_payload",
    "trade_detail",
    "venue_db_path",
]

VENUES: tuple[str, ...] = ("hyperliquid", "oanda")
"""Every venue the dashboard knows how to look for. Order is the response iteration order.

Paper-trading venues only, on purpose: `okx` is research-only (WU-OKX handoff), and listing it
here would let one of its runs displace the traded venue's as "latest tournament"."""

_BAR_SPAN_4H = timedelta(hours=4)

_FLOAT_RESULT_COLS = ("exp_is", "exp_oos", "dsr_prob", "boot_p5", "diff_p5", "mar", "mar_bh")


def to_json_float(value: float | None) -> float | None:
    """Map a non-finite float (``inf``/``-inf``/``nan``) to ``None``; pass everything else through.

    Python's stdlib ``json`` encoder happily emits the literal tokens ``Infinity``/``NaN``,
    which are not valid RFC-8259 JSON and choke a browser's ``JSON.parse`` -- the gate
    handoff's note about ``MAR`` at zero drawdown. Every float this module hands back that
    could plausibly be infinite or NaN goes through here first.
    """
    if value is None:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def venue_db_path(data_dir: str | Path, venue: str) -> Path:
    """The DuckDB file for ``venue`` under ``data_dir`` (may or may not exist)."""
    return Path(data_dir) / f"{venue}.duckdb"


def present_venues(data_dir: str | Path) -> list[str]:
    """Venues whose DuckDB file exists under ``data_dir``, in :data:`VENUES` order."""
    return [venue for venue in VENUES if venue_db_path(data_dir, venue).exists()]


def _attach_utc(value: datetime) -> datetime:
    """Reattach UTC tzinfo to a naive datetime fetched straight off ``store._conn``."""
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


# -- bars / staleness ---------------------------------------------------------


def latest_4h_bar(store: Store, instrument: Instrument) -> Bar | None:
    """The most recent closed 4H bar for ``instrument``, or ``None`` with no 4H history."""
    bar_range = store.bar_range(instrument, "4h")
    if bar_range is None:
        return None
    _, hi, _ = bar_range
    bars = store.bars(instrument, "4h", hi, hi + timedelta(seconds=1))
    return bars[0] if bars else None


def staleness_for_venue(store: Store, venue: str, now: datetime) -> list[dict[str, Any]]:
    """Per-instrument 4H staleness for every instrument on ``venue`` that has 4H history.

    ``bars_behind`` is how many whole 4H spans have elapsed since the latest bar *closed*
    (``ts_open + 4h``), floored and never negative; ``stale`` is ``bars_behind > 2``.
    An instrument with no 4H bars at all is omitted (nothing to report yet).
    """
    out: list[dict[str, Any]] = []
    for instrument in store.instruments(venue):
        bar = latest_4h_bar(store, instrument)
        if bar is None:
            continue
        bar_close = bar.ts_open + _BAR_SPAN_4H
        behind = max(0, math.floor((now - bar_close).total_seconds() / _BAR_SPAN_4H.total_seconds()))
        out.append(
            {
                "symbol": instrument.symbol,
                "last_bar_ts": bar.ts_open,
                "bars_behind": behind,
                "stale": behind > 2,
            }
        )
    return out


# -- paper equity / positions / fills -----------------------------------------


def paper_equity_for_venue(store: Store, venue: str) -> dict[str, list[list[Any]]]:
    """``{run_id: [[ts, equity], ...]}`` for every paper run id under ``venue``."""
    pattern = f"paper:{venue}:%"
    rows = store._conn.execute(
        "SELECT run_id, ts, equity FROM equity WHERE run_id LIKE ? ORDER BY run_id, ts", [pattern]
    ).fetchall()
    out: dict[str, list[list[Any]]] = {}
    for run_id, ts, equity in rows:
        out.setdefault(run_id, []).append([_attach_utc(ts), equity])
    return out


def paper_positions_for_venue(store: Store, venue: str) -> list[dict[str, Any]]:
    """Open paper positions (``closed_bar IS NULL``) across every paper run id under ``venue``.

    Unrealized R is ``direction * (last_4h_close - entry_price) * entry_qty *
    contract_multiplier / risk_r``, using the latest 4H bar for the trade's instrument; it
    is ``None`` when the instrument is unknown or has no 4H bars yet (rather than raising, so
    one stale instrument doesn't take down the whole overview).
    """
    pattern = f"paper:{venue}:%"
    rows = store._conn.execute(
        """
        SELECT run_id, id, symbol, direction, entry_price, entry_ts, entry_qty, stop, target, risk_r
        FROM trades
        WHERE run_id LIKE ? AND closed_bar IS NULL
        ORDER BY entry_ts
        """,
        [pattern],
    ).fetchall()
    if not rows:
        return []
    instruments = {inst.symbol: inst for inst in store.instruments(venue)}
    bar_cache: dict[str, Bar | None] = {}
    positions: list[dict[str, Any]] = []
    for run_id, trade_id, symbol, direction, entry_price, entry_ts, entry_qty, stop, target, risk_r in rows:
        instrument = instruments.get(symbol)
        unrealized_r = None
        if instrument is not None:
            if symbol not in bar_cache:
                bar_cache[symbol] = latest_4h_bar(store, instrument)
            bar = bar_cache[symbol]
            if bar is not None:
                multiplier = float(instrument.contract_multiplier)
                unrealized_r = direction * (bar.close - entry_price) * entry_qty * multiplier / risk_r
        positions.append(
            {
                "run_id": run_id,
                "trade_id": trade_id,
                "symbol": symbol,
                "direction": direction,
                "entry_price": entry_price,
                "qty": entry_qty,
                "stop": stop,
                "target": target,
                "unrealized_r": to_json_float(unrealized_r),
                "opened_at": _attach_utc(entry_ts),
            }
        )
    return positions


def paper_fills_for_venue(store: Store, venue: str, limit: int = 20) -> list[dict[str, Any]]:
    """The most recent ``limit`` fills (any leg) across every paper run id under ``venue``."""
    pattern = f"paper:{venue}:%"
    rows = store._conn.execute(
        """
        SELECT run_id, order_id, ts, price, qty, spread, commission, funding, slippage, leg, trade_id
        FROM fills
        WHERE run_id LIKE ?
        ORDER BY ts DESC
        LIMIT ?
        """,
        [pattern, limit],
    ).fetchall()
    return [
        {
            "run_id": run_id,
            "order_id": order_id,
            "ts": _attach_utc(ts),
            "price": price,
            "qty": qty,
            "leg": leg,
            "trade_id": trade_id,
            "cost_total": spread + commission + funding + slippage,
        }
        for run_id, order_id, ts, price, qty, spread, commission, funding, slippage, leg, trade_id in rows
    ]


def overview_for_venue(store: Store, venue: str, now: datetime) -> dict[str, Any]:
    """The full ``/api/overview`` payload for one venue."""
    return {
        "equity": paper_equity_for_venue(store, venue),
        "positions": paper_positions_for_venue(store, venue),
        "fills": paper_fills_for_venue(store, venue),
        "staleness": staleness_for_venue(store, venue, now),
    }


# -- tournament ----------------------------------------------------------------


def latest_tournament_run(stores: dict[str, Store]) -> tuple[str, datetime] | None:
    """The ``(run_id, ts)`` of the greatest-``ts`` non-paper ``results`` row across ``stores``.

    "Latest tournament" = the ``results`` row set with the greatest ``ts`` whose ``run_id``
    does not start with ``paper:``. Pooled across every store passed in, since one tournament
    run's configs can span both venues and each config's row lives in its own venue's file.
    """
    best: tuple[str, datetime] | None = None
    for store in stores.values():
        row = store._conn.execute(
            """
            SELECT run_id, MAX(ts) FROM results
            WHERE run_id NOT LIKE 'paper:%'
            GROUP BY run_id
            ORDER BY MAX(ts) DESC
            LIMIT 1
            """
        ).fetchone()
        if row is None or row[0] is None:
            continue
        run_id, ts = row
        ts = _attach_utc(ts)
        if best is None or ts > best[1]:
            best = (run_id, ts)
    return best


def _sanitize_result_row(row: dict[str, Any]) -> dict[str, Any]:
    row = dict(row)
    for col in _FLOAT_RESULT_COLS:
        row[col] = to_json_float(row.get(col))
    return row


def _exp_oos_sort_key(row: dict[str, Any]) -> float:
    exp_oos = row.get("exp_oos")
    return exp_oos if exp_oos is not None else float("-inf")


def regime_breakdown(stores: dict[str, Store], run_id: str) -> list[dict[str, Any]]:
    """Expectancy per regime tag, pooled across every child run's *closed* trades.

    Every tournament config's trades live under a child run id ``f"{run_id}:{config_id}"``
    (see module docstring); rather than looking up each config's trades individually and
    reconstructing full `Trade` objects just to read `regime`/`realized_r` off them, this
    aggregates directly in SQL with one ``GROUP BY regime`` query per store, matching every
    child run at once via ``run_id LIKE '{run_id}:%'``. Each store contributes its own
    ``(n, avg)`` per regime; those are combined into a single running ``(n, sum)`` per regime
    so the final expectancy is a proper n-weighted average across stores, not a plain average
    of each store's own average.
    """
    pattern = f"{run_id}:%"
    totals: dict[str, tuple[int, float]] = {}  # regime -> (n, sum of realized_r)
    for store in stores.values():
        rows = store._conn.execute(
            """
            SELECT regime, COUNT(*), AVG(realized_r)
            FROM trades
            WHERE run_id LIKE ? AND realized_r IS NOT NULL
            GROUP BY regime
            """,
            [pattern],
        ).fetchall()
        for regime, n, avg in rows:
            prev_n, prev_sum = totals.get(regime, (0, 0.0))
            totals[regime] = (prev_n + n, prev_sum + avg * n)
    return [
        {"regime": regime, "n": n, "expectancy": to_json_float(total / n)}
        for regime, (n, total) in sorted(totals.items())
    ]


_UNIVERSE_SYMBOL = "*"
"""`swingforge.lab.tournament.UNIVERSE_SYMBOL`, restated: that module pulls in the strategies
and the paper broker, which the `web-is-read-only` import contract keeps out of this package."""


def _resolution_modes(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: dict[tuple[Any, Any], Any] = {}
    for row in rows:
        key = (row.get("venue"), row.get("symbol"))
        if key == (None, None) or row.get("symbol") == _UNIVERSE_SYMBOL:
            continue  # a universe row pools instruments; it is not one
        seen.setdefault(key, row.get("resolution_mode"))
    return [
        {"venue": venue, "symbol": symbol, "resolution_mode": mode} for (venue, symbol), mode in seen.items()
    ]


_CAPPED_TOP_N = 50
"""How many rows `_capped_rows` keeps purely on `exp_oos` rank, independent of `passed`/`exit`."""


def _capped_rows(
    rows: list[dict[str, Any]], *, rule_1_first: bool = False
) -> tuple[list[dict[str, Any]], int]:
    """Every passing config, the top `_CAPPED_TOP_N` by ``exp_oos``, and every ``IS_SELECTED``
    row -- deduplicated, in ``rows``' original relative order -- plus how many rows that left
    out.

    A tournament's row count scales with the search space (a large sweep can be thousands of
    configs); `gate` and `cost_stress` used to ship every single one, so this bounds both to a
    deterministic subset instead of growing the payload unboundedly with tournament size.
    Deterministic because `sorted` is stable: ties in `exp_oos` keep ``rows``' original order,
    which is itself fixed by the caller (one ``ORDER BY ts`` query per store, in store-iteration
    order) -- not left to per-call dict/set iteration.

    `rule_1_first` ranks the rows that cleared rule 1 ahead of the rest, for the universe
    rows: a full sweep holds ~150 of them, most of them thin pools whose few trades flatter
    their expectancy, and ranked on `exp_oos` alone they push the handful the gate could
    actually grade out of the cap.
    """

    def rank(pair: tuple[int, dict[str, Any]]) -> tuple[bool, float]:
        return (rule_1_first and pair[1].get("g1") is True, _exp_oos_sort_key(pair[1]))

    top_indices = {i for i, _ in sorted(enumerate(rows), key=rank, reverse=True)[:_CAPPED_TOP_N]}
    keep_indices = {
        i
        for i, row in enumerate(rows)
        if row.get("passed") is True or i in top_indices or row.get("exit") == "IS_SELECTED"
    }
    kept = [row for i, row in enumerate(rows) if i in keep_indices]
    return kept, len(rows) - len(kept)


def tournament_payload(stores: dict[str, Store], run_id: str) -> dict[str, Any]:
    """The full ``/api/tournament/latest`` payload for the given (already-chosen) ``run_id``.

    Universe rows (``symbol == "*"``: one entry, exit and session graded across a venue's
    instruments) are served under ``universe`` and kept out of ``gate``, ``top``,
    ``selected`` and ``cost_stress``: they are the only rows that reach rule 1's sample on a
    short history, and unmarked they would crowd every instrument's own row out of those
    lists. Rows that cleared rule 1 lead ``universe``, as in the markdown report.
    """
    every_row: list[dict[str, Any]] = []
    for store in stores.values():
        every_row.extend(_sanitize_result_row(row) for row in store.results(run_id))
    rows = [row for row in every_row if row.get("symbol") != _UNIVERSE_SYMBOL]
    universe_rows, universe_omitted = _capped_rows(
        [row for row in every_row if row.get("symbol") == _UNIVERSE_SYMBOL], rule_1_first=True
    )

    def gate_key(row: dict[str, Any]) -> tuple[bool, float]:
        return (row.get("passed") is True, _exp_oos_sort_key(row))

    capped_rows, omitted = _capped_rows(rows)
    gate = sorted(capped_rows, key=gate_key, reverse=True)
    top = sorted(rows, key=_exp_oos_sort_key, reverse=True)[:20]
    selected = [row for row in rows if row.get("exit") == "IS_SELECTED"]
    excluded = [row for row in every_row if row.get("excluded_reason") is not None]
    universe = sorted(
        universe_rows,
        key=lambda row: (
            row.get("passed") is not True,
            row.get("g1") is not True,
            -_exp_oos_sort_key(row),
            str(row.get("config_id")),
        ),
    )
    cost_stress = [
        {"config_id": row.get("config_id"), "exp_oos": row.get("exp_oos"), "g6": row.get("g6")}
        for row in capped_rows
    ]
    ts = max((row["ts"] for row in every_row if row.get("ts") is not None), default=None)
    return {
        "run_id": run_id,
        "ts": ts,
        "gate": gate,
        "gate_omitted": omitted,
        "universe": universe,
        "universe_omitted": universe_omitted,
        "top": top,
        "selected": selected,
        "regime": regime_breakdown(stores, run_id),
        "cost_stress": cost_stress,
        "cost_stress_omitted": omitted,
        "excluded": excluded,
        "resolution_modes": _resolution_modes(rows),  # `rows` holds no universe row
    }


# -- trade detail ---------------------------------------------------------------

_TRADE_ROW_COLUMNS = (
    "id",
    "venue",
    "symbol",
    "direction",
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


def _row_to_trade(store: Store, row: tuple[Any, ...]) -> Trade:
    """Reconstruct a `Trade` from one `trades` row (columns: `_TRADE_ROW_COLUMNS`).

    This duplicates the row-shaping half of `Store.trades` (instrument lookup + legs-JSON
    parsing) rather than calling it, since that method always fetches every row for a
    ``run_id`` -- there is no single-row equivalent on the public `Store` API to reuse, and
    `Store` may not be touched by this WU.
    """
    (
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
    ) = row
    instrument = store._instrument_by_venue_symbol(venue, symbol)
    if instrument is None:
        raise ValueError(
            f"cannot reconstruct trade {trade_id!r}: instrument {venue}/{symbol} is missing "
            "from the instruments table"
        )
    payload = json.loads(legs_json)
    entry_fill = Fill.model_validate(payload["entry_fill"])
    legs = tuple(Fill.model_validate(leg) for leg in payload["legs"])
    return Trade(
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


def _find_trade(stores: dict[str, Store], run_id: str, trade_id: str) -> tuple[Trade, Store] | None:
    """The ``(run_id, trade_id)`` trade and the store owning it, or ``None`` if no store has it.

    Fetches exactly one row per store (``WHERE run_id = ? AND id = ?``, both parameterised)
    instead of pulling every trade for ``run_id`` and scanning for a matching id in Python --
    a run_id/config can carry many trades, only one of which is ever needed here.
    """
    columns = ", ".join(_TRADE_ROW_COLUMNS)
    for store in stores.values():
        row = store._conn.execute(
            f"SELECT {columns} FROM trades WHERE run_id = ? AND id = ?", [run_id, trade_id]
        ).fetchone()
        if row is not None:
            return _row_to_trade(store, row), store
    return None


def trade_detail(stores: dict[str, Store], run_id: str, trade_id: str) -> dict[str, Any] | None:
    """The ``/api/trades/{run_id}/{trade_id}`` payload, or ``None`` if no store has it.

    The 4H bars window is ``[entry_ts - 30 bars, exit_ts + 10 bars]`` -- the spec's "opened_bar
    - 30 to closed_bar + 10" translated from bar-index arithmetic (trades carry indices into a
    particular replay run, not a ts) to a timestamp window anchored on the entry fill and the
    last exit leg's fill (or the entry again, for a still-open trade).
    """
    found = _find_trade(stores, run_id, trade_id)
    if found is None:
        return None
    trade, owning_store = found

    entry_ts = trade.entry_fill.ts
    exit_ts = trade.legs[-1].ts if trade.legs else entry_ts
    start = entry_ts - 30 * _BAR_SPAN_4H
    end = exit_ts + 10 * _BAR_SPAN_4H + _BAR_SPAN_4H  # +1 span: store.bars' end bound is exclusive
    bars = owning_store.bars(trade.instrument, "4h", start, end)

    return {
        "run_id": run_id,
        "trade_id": trade.id,
        "venue": trade.instrument.venue,
        "symbol": trade.instrument.symbol,
        "direction": trade.direction,
        "entry_price": trade.entry_fill.price,
        "entry_ts": entry_ts,
        "stop": trade.stop,
        "target": trade.target,
        "risk_r": trade.risk_r,
        "realized_r": to_json_float(trade.realized_r),
        "mae_r": trade.mae_r,
        "mfe_r": trade.mfe_r,
        "regime": trade.regime,
        "opened_bar": trade.opened_bar,
        "closed_bar": trade.closed_bar,
        "legs": [leg.model_dump(mode="json") for leg in trade.legs],
        "bars": [
            {
                "ts": bar.ts_open,
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
            }
            for bar in bars
        ],
    }


# -- settings ---------------------------------------------------------------


def settings_payload(stores: dict[str, Store]) -> dict[str, Any]:
    """``{venue: {version, settings}}`` for every venue in ``stores``."""
    out: dict[str, Any] = {}
    for venue, store in stores.items():
        version, settings = store.current_settings()
        out[venue] = {"version": version, "settings": settings.model_dump(mode="json")}
    return out
