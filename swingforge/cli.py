"""swingforge CLI: the composition root.

Wires the real venue adapters, strategies, exit rules, tournament, paper engine and
dashboard together into five commands: `backfill`, `tournament`, `paper`, `web`, `report`.
`swingforge.cli` sits outside the package's import layers (`.importlinter`) and may import
anything -- everything else in the codebase only ever sees this module wire it together.

`.env` is loaded once at import time via `python-dotenv`; nothing here ever prints a secret
(a credential lands in `os.environ`, never in a log line). Every command prints concise
progress to stdout and exits non-zero with a one-line message on failure -- see `_fail`.

Store path convention (shared with the dashboard, WU-3C): `<data_dir>/<venue>.duckdb`.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any, NoReturn, cast

import duckdb
import typer
import uvicorn
from dotenv import load_dotenv

from swingforge.adapters.hyperliquid.bars import HyperliquidBars, hl_instruments, hl_instruments_for
from swingforge.adapters.hyperliquid.costs import HyperliquidCosts, load_funding
from swingforge.adapters.oanda.bars import ApiLike, OandaBars, oanda_instrument
from swingforge.adapters.oanda.costs import OandaCosts, load_financing
from swingforge.adapters.paper import FillLog, PaperBroker
from swingforge.adapters.settings_reader import TransientSettingsReader
from swingforge.adapters.store import Store
from swingforge.core.engine import Engine
from swingforge.core.fills import FillResolver
from swingforge.core.portfolio import Portfolio
from swingforge.core.types import TF, Bar, CostBreakdown, Fill, Instrument, Order, Position, Trade
from swingforge.lab.regime import tag
from swingforge.lab.report import write_report
from swingforge.lab.tournament import (
    ENTRIES,
    EXITS,
    SESSIONS,
    UNIVERSE_SYMBOL,
    exit_rule,
    instrument_key,
    months_between,
    run_tournament,
)
from swingforge.strategies.base import Strategy
from swingforge.strategies.baseline import Baseline
from swingforge.strategies.ict import ICT
from swingforge.strategies.session import SessionFilter, SessionMode, SessionProfile
from swingforge.strategies.zones import Zones

__all__ = ["app"]

load_dotenv()

app = typer.Typer(add_completion=False, no_args_is_help=True)


class Venue(StrEnum):
    """CLI choice for `--venue`; converted to a plain `str` before touching any logic below."""

    HYPERLIQUID = "hyperliquid"
    OANDA = "oanda"


_VENUES: tuple[str, ...] = tuple(v.value for v in Venue)
_OANDA_SIX: tuple[str, ...] = ("EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "GBP_JPY", "XAU_USD")

_YEAR = timedelta(days=365)
_ONE_DAY = timedelta(hours=24)
_FOUR_HOURS = timedelta(hours=4)
_TFS: tuple[TF, ...] = ("1h", "4h", "1d")
_WARMUP_DAILY_BARS = 60
_WARMUP_4H_BARS = 200
_DEFAULT_INITIAL_EQUITY = 10_000.0
_LOOPBACK_HOSTS: frozenset[str] = frozenset({"127.0.0.1", "localhost", "::1"})

PERSIST_RETRIES = 30
PERSIST_BACKOFF_S = 10.0
"""`_persist_bar`'s retry budget against a locked store file (~5 minutes total): the nightly
incremental backfill (WU-1A/1B) holds a venue's store open for a few minutes, and paper
trading is meant to wait that out rather than die to it -- see `_persist_bar`."""


def _fail(message: str, *, code: int = 1) -> NoReturn:
    typer.echo(f"error: {message}", err=True)
    raise typer.Exit(code=code)


def _split(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_dt(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def _store_path(data_dir: str, venue: str) -> Path:
    return Path(data_dir) / f"{venue}.duckdb"


def _existing_store_path(data_dir: str, venue: str) -> Path:
    path = _store_path(data_dir, venue)
    if not path.exists():
        raise ValueError(f"no store at {path}; run backfill first")
    return path


# --- venue wiring: bar sources, instrument lists, cost models -----------------------


def _bar_source(venue: str, *, poll_delay_s: float = 5.0) -> HyperliquidBars | OandaBars:
    """The real `BarSource` for `venue`. A small factory so tests can monkeypatch it."""
    if venue == "hyperliquid":
        return HyperliquidBars(poll_delay_s=poll_delay_s)
    if venue == "oanda":
        return OandaBars(
            account_id=os.environ.get("OANDA_ACCOUNT_ID"),
            token=os.environ.get("OANDA_TOKEN"),
            environment=os.environ.get("OANDA_ENV", "practice"),
            poll_delay_s=poll_delay_s,
        )
    raise ValueError(f"unknown venue {venue!r}: expected one of {_VENUES}")


def _instruments(venue: str, source: HyperliquidBars | OandaBars, override: str | None) -> list[Instrument]:
    """The instruments to backfill/trade: `--instruments` overrides the venue's own list."""
    symbols = _split(override) if override else None
    if venue == "hyperliquid":
        assert isinstance(source, HyperliquidBars)
        if symbols is not None:
            # Same tick rule as the default path (one `meta_and_asset_ctxs` call): a fixed
            # fallback tick here would overwrite the store's derived tick on every re-run.
            return hl_instruments_for(source._get_info(), symbols)
        return hl_instruments(source._get_info(), extra=5)
    if venue == "oanda":
        names = symbols if symbols is not None else list(_OANDA_SIX)
        return [oanda_instrument(name) for name in names]
    raise ValueError(f"unknown venue {venue!r}: expected one of {_VENUES}")


class PerInstrumentCostModel:
    """Dispatches `entry`/`carry` to a per-instrument `CostModel`, keyed by `venue:symbol`.

    `run_config`/`run_tournament`/`PaperBroker` all take a single `CostModel`, but
    Hyperliquid's funding history is per-instrument -- this is the thin dispatcher the
    WU-3A design calls for so one `HyperliquidCosts` (loaded with that instrument's own
    funding rows) serves each traded instrument.
    """

    def __init__(self, models: Mapping[str, Any]) -> None:
        self._models = dict(models)

    def _for(self, instrument: Instrument) -> Any:
        key = instrument_key(instrument)
        try:
            return self._models[key]
        except KeyError:
            # A plain `LookupError`, not a re-raised `KeyError`: `KeyError.__str__` wraps
            # its argument in an extra layer of quoting (`KeyError: "no cost model ... "`),
            # which reads badly in a traceback -- `LookupError` prints the message as given.
            raise LookupError(f"no cost model configured for instrument {key!r}") from None

    def entry(self, order: Order, bar: Bar) -> CostBreakdown:
        return self._for(order.instrument).entry(order, bar)

    def carry(self, position: Position, bar: Bar) -> float:
        return self._for(position.instrument).carry(position, bar)


def _oanda_api(token: str | None, environment: str) -> ApiLike:
    import oandapyV20  # type: ignore[import-untyped]

    return oandapyV20.API(access_token=token, environment=environment)


def _build_cost_model(venue: str, store: Store, instruments: Sequence[Instrument]) -> Any:
    """The real cost model for `venue`: per-instrument Hyperliquid funding, or OANDA swap.

    Hyperliquid: one `HyperliquidCosts` per instrument, loaded with that instrument's own
    funding history from the store, dispatched by `PerInstrumentCostModel`.

    OANDA: `OandaCosts(financing={})` (swap always 0) unless `OANDA_ACCOUNT_ID`/`OANDA_TOKEN`
    are both set in the environment, in which case each instrument's financing is fetched
    live via `load_financing`.
    """
    if venue == "hyperliquid":
        models: dict[str, Any] = {}
        for instrument in instruments:
            bar_range = store.bar_range(instrument, "4h")
            funding_rows: list[tuple[datetime, float]] = []
            if bar_range is not None:
                first, last, _count = bar_range
                funding_rows = store.funding(instrument, first, last + _FOUR_HOURS)
            models[instrument_key(instrument)] = HyperliquidCosts(funding=funding_rows)
        return PerInstrumentCostModel(models)
    if venue == "oanda":
        account_id = os.environ.get("OANDA_ACCOUNT_ID")
        token = os.environ.get("OANDA_TOKEN")
        if account_id and token:
            api = _oanda_api(token, os.environ.get("OANDA_ENV", "practice"))
            financing = {
                instrument.symbol: load_financing(api, account_id, instrument.symbol)
                for instrument in instruments
            }
            return OandaCosts(financing=financing)
        return OandaCosts(financing={})
    raise ValueError(f"unknown venue {venue!r}: expected one of {_VENUES}")


# --- strategy/session wiring (⚠ binding orchestrator decisions, see the handoff) ----


def _entry_factory(name: str, instrument: Instrument, *, baseline_rate: float | None, seed: int) -> Strategy:
    if name == "ict":
        return ICT()
    if name == "zones":
        return Zones()
    if name == "baseline":
        return Baseline(target_trades_per_1000_bars=baseline_rate or 1.0, seed=seed)
    raise ValueError(f"unknown entry {name!r}: expected one of {ENTRIES}")


def _session_factory(mode: str, profile: str) -> Callable[[Bar], bool]:
    return SessionFilter(cast(SessionMode, mode), cast(SessionProfile, profile))


# --- backfill ------------------------------------------------------------------------


def _windows(start: datetime, end: datetime, span: timedelta) -> list[tuple[datetime, datetime]]:
    out: list[tuple[datetime, datetime]] = []
    cursor = start
    while cursor < end:
        nxt = min(cursor + span, end)
        out.append((cursor, nxt))
        cursor = nxt
    return out


def _funding(
    venue: str,
    source: HyperliquidBars | OandaBars,
    instrument: Instrument,
    start: datetime,
    end: datetime,
) -> list[tuple[datetime, float]]:
    """Funding-rate history for `instrument` over `[start, end)`; always empty for OANDA.

    A small factory, like `_bar_source`/`_instruments`, so tests can monkeypatch it instead
    of needing a real `HyperliquidBars` behind the fake source `backfill`'s test drives.
    """
    if venue != "hyperliquid":
        return []
    assert isinstance(source, HyperliquidBars)
    return load_funding(source._get_info(), instrument.symbol, start, end)


def _backfill(venue: str, years: int, instruments_opt: str | None, data_dir: str) -> None:
    source = _bar_source(venue)
    instruments = _instruments(venue, source, instruments_opt)
    if not instruments:
        raise ValueError("no instruments to backfill")
    store_path = _store_path(data_dir, venue)
    store_path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(UTC)
    start = now - years * _YEAR
    typer.echo(f"backfilling {len(instruments)} instrument(s) on {venue}: {start.date()} -> {now.date()}")

    # ⚠ The store is opened once for the whole backfill; no other process should write to
    # the same file while this runs (DuckDB allows only one read-write connection at a time).
    with Store(store_path) as store:
        store.upsert_instruments(instruments)
        for instrument in instruments:
            counts: dict[str, int] = {}
            for tf in _TFS:
                total = 0
                for window_start, window_end in _windows(start, now, _YEAR):
                    bars = source.history(instrument, tf, window_start, window_end)
                    total += store.upsert_bars(bars)
                counts[tf] = total
            # Incremental: funding is append-only per hour, so resume from the last stored
            # settlement (re-fetching that one row is an idempotent upsert) instead of paging
            # the whole span again on every nightly run.
            stored_funding = store.funding(instrument, start, now)
            funding_start = stored_funding[-1][0] if stored_funding else start
            store.upsert_funding(instrument, _funding(venue, source, instrument, funding_start, now))
            bar_range = store.bar_range(instrument, "4h")
            months = 0.0 if bar_range is None else months_between(bar_range[0], bar_range[1] + _FOUR_HOURS)
            typer.echo(
                f"  {instrument_key(instrument):<20} 1h={counts['1h']:<8} 4h={counts['4h']:<8} "
                f"1d={counts['1d']:<8} months={months:.1f}"
            )
    typer.echo("backfill complete")


@app.command()
def backfill(
    venue: Venue = typer.Option(..., "--venue", help="hyperliquid or oanda"),  # noqa: B008
    years: int = typer.Option(..., "--years", min=1),  # noqa: B008
    instruments: str | None = typer.Option(None, "--instruments", help="comma-separated symbols"),  # noqa: B008
    data_dir: str = typer.Option("data", "--data-dir", envvar="SWINGFORGE_DATA_DIR"),  # noqa: B008
) -> None:
    """Backfill 1h/4h/1d bars (and Hyperliquid funding) for every traded instrument."""
    try:
        _backfill(venue.value, years, instruments, data_dir)
    except typer.Exit:
        raise
    except Exception as exc:  # the CLI boundary: one line, non-zero exit, never a traceback
        _fail(str(exc))


# --- tournament ------------------------------------------------------------------------


def _filesystem_safe(run_id: str) -> str:
    """`run_id` with `:` replaced by `-`, so the report filename is valid on every OS.

    `run_id` itself (the store's key for this run's trades/equity/results) keeps its
    colons; only the filename derived from it is sanitised here.
    """
    return run_id.replace(":", "-")


def _tournament_run_id(venue: str) -> str:
    """`tournament:{venue}:{YYYYmmddTHHMM}` -- a small factory so tests can pin it down.

    Resuming an interrupted sweep (`--resume`) only finds anything to resume when the two
    invocations share a run id; pinning this in a test is how `--resume` is exercised
    deterministically instead of racing the clock-minute boundary.
    """
    return f"tournament:{venue}:{datetime.now(UTC):%Y%m%dT%H%M}"


def _tournament(
    venue: str,
    out: str,
    seed: int,
    resume: bool,
    run_id_opt: str | None,
    entries: str | None,
    exits: str | None,
    sessions: str | None,
    instruments_opt: str | None,
    start: str | None,
    end: str | None,
    data_dir: str,
) -> None:
    # I2: `--resume` only means something against a *specific* prior run id -- the old
    # minute-granularity auto id made resuming a race against the clock. Failing fast here
    # (before touching the store) beats resuming the wrong (or no) run silently.
    if resume and run_id_opt is None:
        _fail("--resume requires --run-id (the run id to resume)", code=2)
    run_id = run_id_opt if run_id_opt is not None else _tournament_run_id(venue)

    store_path = _existing_store_path(data_dir, venue)
    start_dt = _parse_dt(start) if start else None
    end_dt = _parse_dt(end) if end else None
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)

    with Store(store_path) as store:
        all_instruments = store.instruments(venue)
        if not all_instruments:
            raise ValueError(f"no instruments in {store_path}; run backfill first")
        if instruments_opt:
            # Mismatch fix: `--instruments` narrows the store's own list; a symbol the
            # store has never backfilled is a clean, early error rather than a silent
            # no-op or a KeyError three layers into `run_tournament`.
            wanted = _split(instruments_opt)
            by_symbol = {instrument.symbol: instrument for instrument in all_instruments}
            unknown = [symbol for symbol in wanted if symbol not in by_symbol]
            if unknown:
                _fail(
                    f"unknown instrument(s) {unknown!r} for venue {venue!r}; run backfill first",
                    code=2,
                )
            instruments = [by_symbol[symbol] for symbol in wanted]
        else:
            instruments = all_instruments
        cost_model = _build_cost_model(venue, store, instruments)
        result = run_tournament(
            store,
            instruments,
            entry_factory=_entry_factory,
            session_factory=_session_factory,
            cost_model=cost_model,
            resolver=FillResolver(),
            regime_tagger=tag,
            entries=_split(entries) if entries else list(ENTRIES),
            exits=_split(exits) if exits else list(EXITS),
            sessions=_split(sessions) if sessions else list(SESSIONS),
            start=start_dt,
            end=end_dt,
            seed=seed,
            resume=resume,
            run_id=run_id,
            progress=lambda config_id: typer.echo(f"  {config_id}"),
        )
        report_path = write_report(result, out_dir / f"{_filesystem_safe(run_id)}.md")

    pooled = [row for row in result.rows if row["split"] == "pooled"]
    # I3: every passing config id, `IS_SELECTED` views and `venue:*` universe trials included
    # -- they are pooled rows like any other and are gated the same way, so no separate list
    # is needed for them (`_universe_hint` says what a passing universe id is for).
    passed = sorted((row for row in pooled if row["passed"] is True), key=lambda row: str(row["config_id"]))
    typer.echo(f"run id: {run_id}")
    typer.echo(f"report written to {report_path}")
    typer.echo(f"pooled rows: {len(pooled)}; passed: {len(passed)}")
    typer.echo(f"universe trials: {len(result.universe)}")
    # Rule 2's deflation inputs, so a blanket "nothing passes" verdict can be audited: a
    # handful of tiny-n configs with huge per-trade Sharpes inflate V for the whole run.
    typer.echo(f"n_trials: {result.n_trials}; trial_sr_variance: {result.trial_sr_variance}")
    typer.echo("passing configs:")
    for row in passed:
        typer.echo(f"  {row['config_id']}")
    hint = _universe_hint([str(row["config_id"]) for row in passed])
    if hint is not None:
        typer.echo(hint)
    for instrument, reason in result.excluded:
        typer.echo(f"excluded {instrument_key(instrument)}: {reason}")


def _universe_hint(passed_ids: Sequence[str]) -> str | None:
    """What to do with a passing `venue:*` config, or `None` when there is none."""
    if not any(config_id.endswith(f":{UNIVERSE_SYMBOL}") for config_id in passed_ids):
        return None
    return (
        f"a `venue:{UNIVERSE_SYMBOL}` config passed on its instruments traded together: enable it per "
        "instrument (the report lists how many it pools; see deploy/README.md)"
    )


@app.command()
def tournament(
    venue: Venue = typer.Option(..., "--venue"),  # noqa: B008
    out: str = typer.Option("reports", "--out"),  # noqa: B008
    seed: int = typer.Option(0, "--seed"),  # noqa: B008
    resume: bool = typer.Option(False, "--resume"),  # noqa: B008
    run_id: str | None = typer.Option(None, "--run-id", help="pin the run id; required with --resume"),  # noqa: B008
    entries: str | None = typer.Option(None, "--entries", help="comma-separated, default all"),  # noqa: B008
    exits: str | None = typer.Option(None, "--exits", help="comma-separated, default all"),  # noqa: B008
    sessions: str | None = typer.Option(None, "--sessions", help="comma-separated, default all"),  # noqa: B008
    instruments: str | None = typer.Option(  # noqa: B008
        None, "--instruments", help="comma-separated symbols, default every instrument in the store"
    ),
    start: str | None = typer.Option(None, "--start", help="ISO 8601, default the store's own range"),  # noqa: B008
    end: str | None = typer.Option(None, "--end", help="ISO 8601, default the store's own range"),  # noqa: B008
    data_dir: str = typer.Option("data", "--data-dir", envvar="SWINGFORGE_DATA_DIR"),  # noqa: B008
) -> None:
    """Run the tournament matrix, gate every config, and write a report + `results` rows."""
    try:
        _tournament(
            venue.value,
            out,
            seed,
            resume,
            run_id,
            entries,
            exits,
            sessions,
            instruments,
            start,
            end,
            data_dir,
        )
    except typer.Exit:
        raise
    except Exception as exc:
        _fail(str(exc))


# --- paper ------------------------------------------------------------------------


def _parse_config_id(config_id: str) -> tuple[str, str, str, str]:
    parts = config_id.split("|")
    if len(parts) != 4 or ":" not in parts[3]:
        raise ValueError(f"invalid config id {config_id!r}; expected entry|exit|session|venue:symbol")
    entry, exit_name, session, inst_key = parts
    if inst_key.endswith(f":{UNIVERSE_SYMBOL}"):
        raise ValueError(
            f"config id {config_id!r} pools every instrument of the venue; run one `paper` per instrument, "
            f"e.g. {entry}|{exit_name}|{session}|{inst_key.removesuffix(UNIVERSE_SYMBOL)}<SYMBOL>"
        )
    return entry, exit_name, session, inst_key


def _find_instrument(store: Store, venue: str, inst_key: str) -> Instrument:
    inst_venue, _, symbol = inst_key.partition(":")
    if inst_venue != venue:
        raise ValueError(f"config instrument venue {inst_venue!r} does not match --venue {venue!r}")
    for instrument in store.instruments(venue):
        if instrument.symbol == symbol:
            return instrument
    raise ValueError(f"instrument {inst_key!r} not found in store; run backfill first")


def _chronological(four_hour: Sequence[Bar], daily: Sequence[Bar]) -> list[Bar]:
    """4H and Daily bars interleaved by close time, 4H first on a tie.

    Mirrors `ReplaySource.merged`'s tagging (a contract-adjacent module this work unit does
    not own) so paper's warm-up sees the same chronological order a backtest replay would.
    """
    tagged = [(bar.ts_open + _FOUR_HOURS, 0, bar) for bar in four_hour] + [
        (bar.ts_open + _ONE_DAY, 1, bar) for bar in daily
    ]
    tagged.sort(key=lambda item: (item[0], item[1]))
    return [bar for _, _, bar in tagged]


async def _warm_up(engine: Engine, source: HyperliquidBars | OandaBars, instrument: Instrument) -> None:
    """Seed `engine.ctx` with recent history so ATR/ADX/regime are ready on the first live bar.

    ⚠ Bars are pushed directly into `engine.ctx` -- never through `engine.step` -- so
    warm-up never touches the strategy, the broker or the portfolio and produces no
    phantom fills or trades; see the WU-3A handoff.
    """
    now = datetime.now(UTC)
    daily = source.history(instrument, "1d", now - timedelta(days=95), now)[-_WARMUP_DAILY_BARS:]
    four_hour = source.history(instrument, "4h", now - timedelta(days=40), now)[-_WARMUP_4H_BARS:]
    for bar in _chronological(four_hour, daily):
        if bar.tf == "4h":
            # Only a 4H bar's own subbars belong here (its 1H bars, pushed so ATR/ADX see
            # them). A Daily bar may also legally carry subbars (`Bar._check_subbars` allows
            # "4h" as a Daily subbar tf) -- but those would be the very 4H bars already
            # pushed separately via `four_hour` above; pushing them again here would
            # double-count them and inflate `ctx.bar_index` (every "4h" push advances it),
            # corrupting the monotonic bar clock set below.
            for sub in bar.subbars:
                engine.ctx.push(sub)
        engine.ctx.push(bar)


def _drain_fill_log(broker: PaperBroker, cursor: FillLog | None) -> tuple[list[Fill], FillLog | None]:
    """Every `Fill` logged since `cursor`, echoed to stdout, and the new cursor position.

    I1: `broker.log` is a bounded deque (`_MAX_LOG_ENTRIES`): once more fills have been
    produced than it holds, the oldest are silently evicted from its front. A plain
    integer/length cursor breaks against that -- once the deque is full, its length stops
    changing, so `log[cursor:]` freezes at `[]` forever and every later fill is dropped
    without ever being echoed or persisted. Tracking the last-drained *entry itself*
    survives eviction instead: if it is still present in the current snapshot, everything
    after it is new; if it has since been evicted, every entry now in the deque was
    produced after it, so the whole snapshot is new.
    """
    log = list(broker.log)
    if cursor is None:
        new_entries = log
    else:
        idx = next((i for i, entry in enumerate(log) if entry is cursor), None)
        new_entries = log if idx is None else log[idx + 1 :]
    for entry in new_entries:
        fill = entry.fill
        typer.echo(f"fill {fill.leg} {fill.qty:g}@{fill.price:g} ts={fill.ts.isoformat()}")
    new_cursor = log[-1] if log else cursor
    return [entry.fill for entry in new_entries], new_cursor


def _persist_bar(
    store_path: Path,
    run_id: str,
    fills: Sequence[Fill],
    open_trade: Trade | None,
    closed_trades: Sequence[Trade],
    equity_point: tuple[datetime, float] | None,
    *,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Open the store once, write this bar's fills/trades/equity in one transaction, and close it.

    ⚠ The open trade (if any) is written every bar as a provisional row -- `closed_bar` and
    `realized_r` both `None` -- and is upserted in place with its final values the bar it
    closes on (the dashboard's own convention: same `id`, same primary key).

    `equity_point` is `None` for the zero-bar case (the `finally`-block flush after a stream
    that never yielded a bar): the `equity` table is `Engine`'s persistent monotonic bar
    clock (C1), so writing a point here with nothing to back it would silently advance that
    clock. With every argument empty/`None` (nothing to write at all) this returns without
    even opening the store.

    C3: the three writes run inside one DuckDB transaction (`BEGIN`/`COMMIT`, `ROLLBACK` on
    any failure) so a kill (or any exception) between them can never leave a trade row
    without its matching equity row -- that window could otherwise reuse a trade id after
    `--abandon-open-trade`.

    I6: the nightly incremental backfill (WU-1A/1B) holds a venue's store file open for a
    few minutes; landing in that window raises `duckdb.IOException` (file locked) rather
    than blocking. The whole transaction above is retried up to `PERSIST_RETRIES` times,
    `PERSIST_BACKOFF_S` seconds apart (~5 minutes total) before giving up -- paper trading is
    meant to wait that lock out, not die to it. `sleep` is injectable so a test can pin this
    down without actually waiting.
    """
    if not fills and open_trade is None and not closed_trades and equity_point is None:
        return
    attempt = 0
    while True:
        try:
            with Store(store_path) as store:
                # `Store` has no public transaction helper (only `write_settings` reaches
                # for `begin`/`commit`/`rollback` internally) -- `store._conn` directly, as
                # `swingforge/web/queries.py` already does for read-only access.
                store._conn.begin()
                try:
                    if fills:
                        store.write_fills(run_id, fills)
                    trades = [*closed_trades]
                    if open_trade is not None:
                        trades.append(open_trade)
                    if trades:
                        store.write_trades(run_id, trades)
                    if equity_point is not None:
                        store.write_equity(run_id, [equity_point])
                except Exception:
                    store._conn.rollback()
                    raise
                else:
                    store._conn.commit()
            return
        except duckdb.IOException:
            attempt += 1
            if attempt > PERSIST_RETRIES:
                raise
            sleep(PERSIST_BACKOFF_S)


@dataclass
class _PaperRuntime:
    """Everything `_paper`'s streaming loop needs, assembled once at startup.

    Split out of `_paper` (WU-3A code review) so the bar-by-bar loop stays short and every
    startup concern -- config parsing, the baseline/open-trade refusals, warm-up, the
    monotonic bar clock -- can be exercised without driving a fake bar stream.
    """

    run_id: str
    store_path: Path
    instrument: Instrument
    source: HyperliquidBars | OandaBars
    broker: PaperBroker
    portfolio: Portfolio
    engine: Engine


async def _build_paper_runtime(
    venue: str, config_id: str, poll_delay: float, data_dir: str, abandon_open_trade: bool
) -> _PaperRuntime:
    """Parse `config_id`, wire up the engine, warm it up, and set its monotonic bar clock.

    C2: refuses to start (exit 3) if the store still holds a trade for this run id with
    `closed_bar IS NULL` -- an open position from a previous paper process. Paper does not
    restore positions across restarts (a fresh `Engine`/`Portfolio` starts flat every time),
    so a crash or a kill between bars leaves that trade's real-world fate genuinely unknown;
    `--abandon-open-trade` closes it in the store at 0R (`closed_bar = opened_bar`,
    `realized_r = 0.0`, `regime` unchanged) and echoes which trade it closed, rather than
    silently guessing either way.
    """
    entry, exit_name, session, inst_key = _parse_config_id(config_id)
    if entry == "baseline":
        _fail("baseline is the control and never trades paper", code=2)
    exit_rule_obj = exit_rule(exit_name)  # fail fast on a typo, before anything else runs
    store_path = _existing_store_path(data_dir, venue)
    run_id = f"paper:{venue}:{config_id}"

    with Store(store_path) as store:
        instrument = _find_instrument(store, venue, inst_key)
        cost_model = _build_cost_model(venue, store, [instrument])
        equity_rows = store.equity(run_id)
        open_trades = [trade for trade in store.trades(run_id) if trade.closed_bar is None]
        if open_trades:
            if not abandon_open_trade:
                trade = open_trades[0]
                _fail(
                    f"open trade {trade.id} (entry ts={trade.entry_fill.ts.isoformat()}) from a "
                    "previous paper run; paper does not restore positions across restarts -- "
                    "resolve it out of band, then pass --abandon-open-trade to close it at 0R "
                    "and start",
                    code=3,
                )
            for trade in open_trades:
                closed = trade.model_copy(update={"closed_bar": trade.opened_bar, "realized_r": 0.0})
                store.write_trades(run_id, [closed])
                typer.echo(f"abandoned open trade {trade.id} at 0R")
    initial_equity = equity_rows[-1][1] if equity_rows else _DEFAULT_INITIAL_EQUITY

    source = _bar_source(venue, poll_delay_s=poll_delay)
    broker = PaperBroker(cost_model, FillResolver())
    portfolio = Portfolio(initial_equity)
    strategy = _entry_factory(entry, instrument, baseline_rate=None, seed=0)
    engine = Engine(
        instrument,
        strategy,
        exit_rule_obj,
        broker,
        portfolio,
        TransientSettingsReader(store_path),
        session_allowed=_session_factory(session, instrument.session_profile),
        regime_tagger=tag,
    )

    await _warm_up(engine, source, instrument)
    # C1: force the bar clock past every bar this run id has ever persisted, so a trade
    # opened on the first live bar after a restart still gets a fresh, never-before-used
    # `trade_id` (`Engine._maybe_enter` builds it as f"{venue}:{symbol}:{ctx.bar_index}").
    # The `equity` table is the run's own bar clock -- exactly one row per live bar,
    # written nowhere else -- so its row count survives a restart even though `Context`
    # itself does not.
    engine.ctx.bar_index = _WARMUP_4H_BARS - 1 + len(equity_rows)

    typer.echo(f"run id: {run_id}")
    typer.echo(f"paper trading {config_id} on {venue}; initial equity {initial_equity:.2f}")
    return _PaperRuntime(run_id, store_path, instrument, source, broker, portfolio, engine)


async def _paper(
    venue: str, config_id: str, poll_delay: float, data_dir: str, abandon_open_trade: bool
) -> None:
    runtime = await _build_paper_runtime(venue, config_id, poll_delay, data_dir, abandon_open_trade)
    log_cursor: FillLog | None = None
    last_bar_close = datetime.now(UTC)
    bars_seen = 0
    try:
        async for bar in runtime.source.stream(runtime.instrument, "4h"):
            if bar.ts_open.hour == 0:
                day_start = bar.ts_open - _ONE_DAY
                for daily_bar in runtime.source.history(runtime.instrument, "1d", day_start, bar.ts_open):
                    runtime.engine.step(daily_bar)
            killed_before = runtime.engine.settings.kill_switch
            closed = runtime.engine.step(bar)
            if runtime.engine.settings.kill_switch != killed_before:
                state = "engaged" if runtime.engine.settings.kill_switch else "released"
                typer.echo(f"kill switch {state}")
            fills, log_cursor = _drain_fill_log(runtime.broker, log_cursor)
            last_bar_close = bar.ts_open + _FOUR_HOURS
            bars_seen += 1
            equity_point = (last_bar_close, runtime.portfolio.equity)
            _persist_bar(
                runtime.store_path, runtime.run_id, fills, runtime.engine.open_trade, closed, equity_point
            )
    finally:
        # Graceful stop (KeyboardInterrupt or the stream ending): stop taking new orders,
        # then one more flush so a fill/trade produced between the last persist and the
        # stop is never lost. Idempotent (every write upserts), so this never double-counts.
        #
        # I5: a failure *in this flush* must never replace whatever exception is already
        # propagating out of the `try` above (e.g. the venue error that ended the stream) --
        # letting it do so would silently swap the real cause for an unrelated store error.
        # So the flush is wrapped here too: its own failure is only ever raised when there
        # was nothing else in flight to report; otherwise it is echoed and the original
        # exception is left to keep propagating undisturbed.
        original_exc_active = sys.exc_info()[0] is not None
        try:
            runtime.broker.close()
            fills, log_cursor = _drain_fill_log(runtime.broker, log_cursor)
            # C4: no bar was ever processed (the stream ended/raised before yielding one) --
            # `equity` is `Engine`'s persistent bar clock (C1), so this flush must not write
            # a point for it; `_persist_bar` then no-ops entirely when there is nothing else
            # to write either.
            final_equity_point = (last_bar_close, runtime.portfolio.equity) if bars_seen else None
            _persist_bar(
                runtime.store_path,
                runtime.run_id,
                fills,
                runtime.engine.open_trade,
                [],
                final_equity_point,
            )
        except Exception as flush_exc:
            typer.echo(f"error during final flush: {flush_exc}", err=True)
            if not original_exc_active:
                raise


@app.command()
def paper(
    venue: Venue = typer.Option(..., "--venue"),  # noqa: B008
    config: str = typer.Option(..., "--config", help="entry|exit|session|venue:symbol"),  # noqa: B008
    poll_delay: float = typer.Option(5.0, "--poll-delay"),  # noqa: B008
    abandon_open_trade: bool = typer.Option(  # noqa: B008
        False,
        "--abandon-open-trade",
        help="close a previous run's still-open trade at 0R and start",
    ),
    data_dir: str = typer.Option("data", "--data-dir", envvar="SWINGFORGE_DATA_DIR"),  # noqa: B008
) -> None:
    """Run one config live: `Engine` on the venue's real bar stream, through `PaperBroker`.

    Not built: no live orders (paper only), and no retry on a venue error -- an exception
    from the bar source ends this process; a supervisor (systemd) is expected to restart it.
    """
    try:
        asyncio.run(_paper(venue.value, config, poll_delay, data_dir, abandon_open_trade))
    except typer.Exit:
        raise
    except KeyboardInterrupt:
        typer.echo("paper trading stopped")
    except Exception as exc:
        _fail(str(exc))


# --- web ------------------------------------------------------------------------


@app.command()
def web(
    host: str = typer.Option("127.0.0.1", "--host"),  # noqa: B008
    port: int = typer.Option(8787, "--port"),  # noqa: B008
    data_dir: str = typer.Option("data", "--data-dir", envvar="SWINGFORGE_DATA_DIR"),  # noqa: B008
) -> None:
    """Serve the read/write dashboard over every present venue store under `data_dir`."""
    token = os.environ.get("SWINGFORGE_TOKEN") or None
    if host not in _LOOPBACK_HOSTS and not token:
        _fail(f"--host {host} requires SWINGFORGE_TOKEN to be set", code=2)
    try:
        # Lazy: swingforge.web.app is another work unit's file; importing it here (not at
        # module load) keeps the rest of this CLI usable if it's ever absent, and lets a
        # test substitute a fake module via sys.modules.
        from swingforge.web.app import create_app
    except ImportError as exc:
        _fail(f"dashboard module not available: {exc}")
    fastapi_app = create_app(data_dir=data_dir, token=token)
    typer.echo(f"serving dashboard on {host}:{port} (data_dir={data_dir})")
    uvicorn.run(fastapi_app, host=host, port=port)


# --- report ------------------------------------------------------------------------


def _latest_run_id(store: Store, venue: str) -> str | None:
    """The most recently started tournament run for `venue`, from `results.run_id`.

    `Store` exposes no "list run ids" query and this package may not extend it (it owns
    only `cli.py`) -- every row of one run shares that run's own `ts` (WU-2C handoff), so
    the most recent row under this venue's `tournament:` prefix names the latest run.
    """
    row = store._conn.execute(
        "SELECT run_id FROM results WHERE run_id LIKE ? ORDER BY ts DESC LIMIT 1",
        [f"tournament:{venue}:%"],
    ).fetchone()
    return None if row is None else str(row[0])


def _render_gate_summary(run_id: str, rows: list[dict[str, Any]]) -> str:
    pooled = sorted(
        (row for row in rows if row["split"] == "pooled"),
        key=lambda row: (row["passed"] is not True, str(row["config_id"])),
    )
    lines = [
        f"# swingforge tournament {run_id}",
        "",
        "| config | n_oos | exp_oos | passed |",
        "|---|---|---|---|",
    ]
    for row in pooled:
        n_oos = "-" if row["n_oos"] is None else str(row["n_oos"])
        exp_oos = "-" if row["exp_oos"] is None else f"{row['exp_oos']:.3f}"
        passed = "-" if row["passed"] is None else ("yes" if row["passed"] else "no")
        lines.append(f"| {row['config_id']} | {n_oos} | {exp_oos} | {passed} |")
    return "\n".join(lines)


def _report(venue: str, run_id: str, data_dir: str) -> None:
    store_path = _existing_store_path(data_dir, venue)
    store = Store(store_path, read_only=True)
    try:
        resolved = _latest_run_id(store, venue) if run_id == "latest" else run_id
        if resolved is None:
            raise ValueError(f"no tournament runs found for venue {venue!r}")
        rows = store.results(resolved)
    finally:
        store.close()
    if not rows:
        raise ValueError(f"no results found for run {resolved!r}")
    typer.echo(_render_gate_summary(resolved, rows))


@app.command()
def report(
    venue: Venue = typer.Option(..., "--venue"),  # noqa: B008
    run_id: str = typer.Option("latest", "--run-id"),  # noqa: B008
    data_dir: str = typer.Option("data", "--data-dir", envvar="SWINGFORGE_DATA_DIR"),  # noqa: B008
) -> None:
    """Print a markdown gate summary from the `results` table of an existing run -- no re-run."""
    try:
        _report(venue.value, run_id, data_dir)
    except typer.Exit:
        raise
    except Exception as exc:
        _fail(str(exc))


if __name__ == "__main__":
    app()
