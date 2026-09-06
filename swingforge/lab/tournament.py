"""The tournament: anchored walk-forward over the config matrix, into the statistical gate.

Matrix (design spec section 6): entry {3} x exit {16} x session {3} x instrument {14} =
2,016 configs. Every config is replayed through the same `Engine` the paper broker drives,
its trades are split into in-sample and out-of-sample windows by an *anchored* walk-forward
(IS 12 months -> OOS 3 months, step 3 months), and its pooled OOS trades are handed to
`swingforge.lab.gate.evaluate`.

**One replay per config, not one per split.** The walk-forward here is anchored (every
split's IS starts at the run's `start`) and *nothing* is fitted on IS: the exit-grid
parameters are enumerated, not optimised, so a config's behaviour on a given bar does not
depend on which split that bar happens to fall in. Replaying a config once over
`[start - warmup, end)` therefore produces exactly the trade list every split would have
produced, and a trade belongs to split *k*'s IS iff `entry_fill.ts < is_end_k` and to its
OOS iff `oos_start_k <= entry_fill.ts < oos_end_k`. That turns 2,016 x 12 replays into
2,016. The one thing this equivalence does *not* cover is the IS-*selected* view (see
`run_tournament`), which picks a different exit per split — but that selection is made
across already-computed runs, not by replaying anything.

**Two passes.** Gate rule 2 needs `trial_sr_variance`, the variance of the per-observation
Sharpe ratio *across the trials that entered the search* (WU-1D: the default
`1/(T-1)` is an estimator variance and under-deflates). That is only knowable once every
config has run, so `run_tournament` replays every config first (pass 1), then computes V
and evaluates every gate (pass 2).

**What pass 1 keeps.** Only what pass 2 reads: per config a `_Replayed` (trades, 4H bar
count, resolution mode), and per *instrument* one close series, which every config of that
instrument shares. Keeping a whole `ConfigRun` per config instead — each with its own copy
of the instrument's ~8,700 closes — is what made a 2,016-config sweep unaffordable, so the
full runs are returned only on request (`run_tournament(..., keep_runs=True)`).

**Warmup.** A config is replayed from `warmup_start(start)`, far enough back that Daily
ATR/ADX and the regime tagger's vol history are seeded by the time `start` arrives. No
entry is taken before `start`: the session filter handed to the engine is wrapped so it
refuses every bar opening before `start`, which keeps the warmup out of both the trade list
and the equity curve.
"""

from __future__ import annotations

import calendar
import logging
import math
import zlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Literal, Protocol

import numpy as np
from pydantic import BaseModel, ConfigDict

from swingforge.adapters.base import CostModel, ExitResolver
from swingforge.adapters.paper import PaperBroker
from swingforge.adapters.replay import ReplaySource
from swingforge.adapters.store import _RESULTS_COLUMNS, Store
from swingforge.core.context import Context
from swingforge.core.costs import NullCostModel, stress
from swingforge.core.engine import Engine
from swingforge.core.fills import FillResolver
from swingforge.core.portfolio import Portfolio
from swingforge.core.settings import Settings, StaticSettingsReader
from swingforge.core.types import Bar, Instrument, Trade
from swingforge.lab import gate
from swingforge.lab.gate import GateResult
from swingforge.lab.regime import tag
from swingforge.strategies.base import ExitRule, Strategy
from swingforge.strategies.exits import EXIT_GRID

__all__ = [
    "ENTRIES",
    "EXITS",
    "EXIT_RULES",
    "MAX_CONSECUTIVE_ERRORS",
    "SESSIONS",
    "WARMUP_DAILY_BARS",
    "Config",
    "ConfigRun",
    "EntryFactory",
    "SessionFactory",
    "Split",
    "TournamentResult",
    "add_months",
    "buy_and_hold_curve",
    "configs",
    "exit_rule",
    "expectancy",
    "instrument_key",
    "months_between",
    "oos_years",
    "realized_rs",
    "run_config",
    "run_tournament",
    "split_trades",
    "stressed_r",
    "walk_forward_splits",
    "warmup_start",
]

ENTRIES: tuple[str, ...] = ("ict", "zones", "baseline")
"""The three entry families (design spec section 4). `baseline` is the frequency-matched control."""

SESSIONS: tuple[str, ...] = ("none", "london_ny", "active")
"""The three session filters swept as a tournament dimension."""

EXIT_RULES: Mapping[str, ExitRule] = {rule.name: rule for rule in EXIT_GRID}
"""The 16 exit variants by name. Every rule is stateless, so one instance serves every config."""

EXITS: tuple[str, ...] = tuple(EXIT_RULES)
"""Exit-rule names in `EXIT_GRID` order — the tie-break order for IS selection."""

WARMUP_DAILY_BARS = 60
"""Daily bars replayed before `start` so ATR(14)/ADX(14) and the regime tagger are seeded.

60 is the regime tagger's own `min_history`, which is the longest warmup any component
needs (ADX(14) seeds at 29 Daily bars). `warmup_start` converts it to a calendar span.
"""

_INF_SENTINEL = 1e9
"""What an infinite MAR is written as in the `results` table.

`GateResult` serialises a zero-drawdown MAR as a bare `Infinity` (WU-1D's closing gotcha),
which DuckDB's DOUBLE column would take but no JSON reader downstream would. `1e9` is far
above any finite MAR a real config produces, so it sorts and compares like "infinite"
without poisoning the column; the report renders it back as an unbounded ratio.
"""

_DEFAULT_SETTINGS = Settings()
_NULL_COST_MODEL = NullCostModel()
_FILL_RESOLVER = FillResolver()
"""Module-level singletons: all three are frozen or stateless, and using them as defaults
keeps the signatures free of call-in-default-argument (ruff B008)."""

_MIN_IS_TRADES_FOR_SELECTION = 10
"""IS trades a split must hold before its exit choice means anything (see `run_tournament`)."""

MAX_CONSECUTIVE_ERRORS = 20
"""Per-config replay failures in a row before `run_tournament` gives up on the whole run.

One config blowing up is a row (a bad symbol, a strategy that cannot be built). Twenty in
a row is a broken store, a broken factory or a broken machine, and grinding through the
remaining 1,996 to produce 2,016 identical `error:` rows helps nobody. The counter resets
on the first config that runs, so scattered failures never trip it.
"""

_SECONDS_PER_YEAR = 365.25 * 24 * 3600
_FOUR_HOURS = timedelta(hours=4)

logger = logging.getLogger(__name__)


# --- the config matrix -------------------------------------------------------------


class Config(BaseModel):
    """One point of the tournament matrix: an entry family, an exit rule, a session, a symbol."""

    model_config = ConfigDict(frozen=True)

    entry: str
    exit: str
    session: str
    instrument: Instrument

    @property
    def id(self) -> str:
        """`entry|exit|session|venue:symbol` — the stable key used in every `results` row."""
        return f"{self.entry}|{self.exit}|{self.session}|{instrument_key(self.instrument)}"


def instrument_key(instrument: Instrument) -> str:
    """`venue:symbol`, the key instruments are reported under."""
    return f"{instrument.venue}:{instrument.symbol}"


def configs(
    instruments: Sequence[Instrument],
    *,
    entries: Sequence[str] = ENTRIES,
    exits: Sequence[str] = EXITS,
    sessions: Sequence[str] = SESSIONS,
) -> list[Config]:
    """Every `Config` in the matrix, grouped by instrument then entry then exit then session."""
    return [
        Config(entry=entry, exit=exit_name, session=session, instrument=instrument)
        for instrument in instruments
        for entry in entries
        for exit_name in exits
        for session in sessions
    ]


def exit_rule(name: str) -> ExitRule:
    """The `EXIT_GRID` rule called `name`; raises `KeyError` for anything else."""
    try:
        return EXIT_RULES[name]
    except KeyError:
        raise KeyError(f"unknown exit rule {name!r}; expected one of {EXITS}") from None


# --- calendar arithmetic and the anchored walk-forward -----------------------------


def add_months(moment: datetime, months: int) -> datetime:
    """`moment` shifted by whole calendar months, clamping the day to the target month's last.

    Walk-forward windows are stated in months, not days, so 12 months from 31 January is
    31 January — and 1 month from it is 28/29 February rather than a 31-day slide.
    """
    total = moment.month - 1 + months
    year = moment.year + total // 12
    month = total % 12 + 1
    return moment.replace(year=year, month=month, day=min(moment.day, calendar.monthrange(year, month)[1]))


def months_between(start: datetime, end: datetime) -> int:
    """Whole calendar months from `start` to `end`; 0 when `end` is not after `start`."""
    if end <= start:
        return 0
    count = (end.year - start.year) * 12 + (end.month - start.month) + 1
    while count > 0 and add_months(start, count) > end:
        count -= 1
    return count


@dataclass(frozen=True)
class Split:
    """One anchored walk-forward split. `is_start` is the run's `start` for every split."""

    is_start: datetime
    is_end: datetime
    oos_start: datetime
    oos_end: datetime

    @property
    def label(self) -> str:
        """`YYYY-MM` of the OOS window's first month — the `split` column of a `results` row."""
        return f"{self.oos_start:%Y-%m}"


def walk_forward_splits(
    start: datetime,
    end: datetime,
    *,
    is_months: int = 12,
    oos_months: int = 3,
    step_months: int = 3,
) -> list[Split]:
    """Anchored splits over `[start, end)`: IS always begins at `start`, OOS steps forward.

    Split *k* trains on `[start, start + is_months + k*step_months)` and tests on the
    `oos_months` that follow it. A split whose OOS window would run past `end` is dropped,
    so every returned window is complete; consecutive OOS windows are contiguous and
    non-overlapping when `step_months == oos_months`.
    """
    if is_months < 1 or oos_months < 1 or step_months < 1:
        raise ValueError("is_months, oos_months and step_months must all be >= 1")
    splits: list[Split] = []
    step = 0
    while True:
        oos_start = add_months(start, is_months + step * step_months)
        oos_end = add_months(oos_start, oos_months)
        if oos_end > end:
            return splits
        splits.append(Split(is_start=start, is_end=oos_start, oos_start=oos_start, oos_end=oos_end))
        step += 1


def oos_years(splits: Sequence[Split]) -> float:
    """Total out-of-sample span in calendar years — the `years` both MAR figures share."""
    seconds = sum((split.oos_end - split.oos_start).total_seconds() for split in splits)
    return seconds / _SECONDS_PER_YEAR


def warmup_start(start: datetime, *, daily_bars: int = WARMUP_DAILY_BARS) -> datetime:
    """How far before `start` a replay must begin to have `daily_bars` Daily bars in hand.

    Padded by 7/5 so a five-session FX week still yields `daily_bars` Daily bars; a perp
    trading seven days a week simply gets more warmup than it needs, which costs a little
    replay time and nothing else.
    """
    return start - timedelta(days=math.ceil(daily_bars * 7 / 5))


# --- running one config ------------------------------------------------------------


class EntryFactory(Protocol):
    """Builds the entry strategy for a config.

    The concrete strategies live in `swingforge.strategies` and are wired in by the CLI;
    the tournament only knows this signature. `baseline_rate` is the trade rate per 1,000
    4H bars the `baseline` control must match (design spec section 4) and is `None` for
    every other entry family.

    `seed` is the config's own seed, derived by `run_config` from the config id and the
    run's seed. Any entry that draws random numbers — the `baseline` control above all —
    must seed itself from it and from nothing else, so that a config's trades depend on
    which config it is and not on where it fell in the matrix or how often it has been run.
    """

    def __call__(
        self, name: str, instrument: Instrument, *, baseline_rate: float | None, seed: int
    ) -> Strategy: ...


class SessionFactory(Protocol):
    """Builds the session filter for a config: `(mode, instrument.session_profile) -> predicate`."""

    def __call__(self, mode: str, profile: str) -> Callable[[Bar], bool]: ...


class _WarmContext(Context):
    """A `Context` that keeps its ATR/ADX memo warm by reading them once per push.

    `Context.atr`/`adx` memoise per `(tf, n)` but only step forward in O(1) when they are
    read on *consecutive* pushes; read once every few bars — which is exactly what a
    strategy that signals rarely does, since `Context.snapshot` and the regime tagger only
    run at an entry — every read re-evaluates Wilder's recurrence over the whole retained
    series in Python. Over a 4-year replay that is the single largest cost in the
    tournament.

    Touching the four indicators this system reads (ATR and ADX on 4H and Daily, always
    `n=14`) after every push keeps every later read on the O(1) path. The values are
    identical either way: the incremental step continues the same recurrence the full
    recompute evaluates, which `test_warm_context_does_not_change_the_run` pins down. The
    real fix belongs in `Context` itself, which owns the memo.
    """

    def push(self, bar: Bar) -> None:
        super().push(bar)
        if bar.tf != "1h":
            self.atr(bar.tf)
            self.adx(bar.tf)


class _MemoisedSource(ReplaySource):
    """A `ReplaySource` that keeps the last merged bar stream it built.

    Every config of one instrument replays exactly the same `[warmup, end)` stream, and the
    tournament runs an instrument's configs back to back — so without this each of an
    instrument's 144 configs would re-query DuckDB and re-validate every `Bar`. Bars are
    frozen and the engine only reads them, so one list can serve every config. A single
    slot is enough: the key changes once per instrument.
    """

    def __init__(self, store: Store) -> None:
        super().__init__(store)
        self._key: tuple[str, datetime, datetime] | None = None
        self._bars: list[Bar] = []

    def merged(self, instrument: Instrument, start: datetime, end: datetime) -> list[Bar]:
        key = (instrument_key(instrument), start, end)
        if key != self._key:
            self._key = key
            self._bars = super().merged(instrument, start, end)
        return self._bars


@dataclass(frozen=True)
class ConfigRun:
    """Everything one config's single replay produced.

    `trades` are chronological. The engine holds at most one open trade at a time, so the
    order trades close in is also the order they opened in, and `Portfolio.trades` is
    already chronological by entry.
    """

    config: Config
    trades: tuple[Trade, ...]
    equity_curve: tuple[tuple[datetime, float], ...]
    n_bars: int
    resolution_mode: Literal["subbars", "pessimistic", "mixed"]

    @property
    def trades_per_1000_bars(self) -> float:
        """Trade frequency, the quantity the `baseline` entry is matched against."""
        return 0.0 if self.n_bars == 0 else len(self.trades) * 1000.0 / self.n_bars


@dataclass(frozen=True)
class _Replayed:
    """The slice of a `ConfigRun` pass 2 reads, which is what `run_tournament` retains.

    Dropping the config, the equity curve and (above all) the per-config copy of the
    instrument's close series is what makes 2,016 configs fit in memory; see this module's
    docstring.
    """

    trades: tuple[Trade, ...]
    n_bars: int
    resolution_mode: Literal["subbars", "pessimistic", "mixed"]


def _resolution_mode(bars: Sequence[Bar]) -> Literal["subbars", "pessimistic", "mixed"]:
    """How the fill resolver could order exits over these 4H bars.

    `"subbars"` when every bar carries its 1H sub-bars, `"pessimistic"` when none does (the
    resolver then assumes the stop came first), `"mixed"` when the coverage is partial. A
    range with no 4H bars at all reports `"pessimistic"`: nothing was resolvable on sub-bars.
    """
    with_subbars = sum(1 for bar in bars if bar.subbars)
    if with_subbars == 0:
        return "pessimistic"
    return "subbars" if with_subbars == len(bars) else "mixed"


def _in_range_bars(source: ReplaySource, instrument: Instrument, start: datetime, end: datetime) -> list[Bar]:
    """The 4H bars of `[start, end)`, out of the same merged stream a replay reads.

    Against a `_MemoisedSource` this is a list comprehension over bars already in memory,
    which is what makes both the per-instrument close series and a resumed config's
    `n_bars`/`resolution_mode` free to recompute.
    """
    return [
        bar
        for bar in source.merged(instrument, warmup_start(start), end)
        if bar.tf == "4h" and start <= bar.ts_open < end
    ]


def run_config(
    config: Config,
    source: ReplaySource,
    *,
    entry_factory: EntryFactory,
    session_factory: SessionFactory,
    cost_model: CostModel = _NULL_COST_MODEL,
    resolver: ExitResolver = _FILL_RESOLVER,
    regime_tagger: Callable[[Context], str] = tag,
    start: datetime,
    end: datetime,
    initial_equity: float = 10_000.0,
    settings: Settings = _DEFAULT_SETTINGS,
    baseline_rate: float | None = None,
    seed: int = 0,
    warm_context: bool = True,
    exit_override: ExitRule | None = None,
) -> ConfigRun:
    """Replay one config once over `[warmup_start(start), end)` and return what it produced.

    Entries are gated to bars opening at or after `start`: the session predicate the engine
    receives is `session_factory(...)` wrapped with that check, so the warmup seeds the
    indicators without contributing trades or equity moves.

    The entry factory is handed `crc32(config.id) ^ seed`, not `seed` itself: every config
    of a run gets its own stream, derived from *which* config it is rather than from its
    position in the matrix, so narrowing the sweep or reordering it leaves each config's
    random entries exactly where they were.

    `exit_override` replaces `config.exit`'s grid rule without touching the config id, which
    is how `swingforge.lab.excursion` measures an entry under `HoldBars`. `warm_context`
    only changes how fast the replay runs (see `_WarmContext`), never what it produces.
    """
    strategy = entry_factory(
        config.entry,
        config.instrument,
        baseline_rate=baseline_rate,
        seed=zlib.crc32(config.id.encode()) ^ seed,
    )
    in_session = session_factory(config.session, config.instrument.session_profile)
    portfolio = Portfolio(initial_equity)
    engine = Engine(
        config.instrument,
        strategy,
        exit_rule(config.exit) if exit_override is None else exit_override,
        PaperBroker(cost_model, resolver),
        portfolio,
        StaticSettingsReader(settings),
        session_allowed=lambda bar: bar.ts_open >= start and in_session(bar),
        regime_tagger=regime_tagger,
        context=_WarmContext(config.instrument) if warm_context else Context(config.instrument),
    )

    bars = source.merged(config.instrument, warmup_start(start), end)
    engine.run(bars)

    in_range = [bar for bar in bars if bar.tf == "4h" and start <= bar.ts_open < end]
    return ConfigRun(
        config=config,
        trades=tuple(portfolio.trades),
        equity_curve=tuple(portfolio.equity_curve),
        n_bars=len(in_range),
        resolution_mode=_resolution_mode(in_range),
    )


# --- slicing, stressing and benchmarking a trade list -------------------------------


def split_trades(trades: Sequence[Trade], split: Split) -> tuple[list[Trade], list[Trade]]:
    """Partition `trades` into `(in_sample, out_of_sample)` by entry-fill timestamp."""
    in_sample = [t for t in trades if split.is_start <= t.entry_fill.ts < split.is_end]
    out_of_sample = [t for t in trades if split.oos_start <= t.entry_fill.ts < split.oos_end]
    return in_sample, out_of_sample


def realized_rs(trades: Sequence[Trade]) -> list[float]:
    """The realized R of every closed trade, chronological — the gate's input series."""
    return [t.realized_r for t in trades if t.realized_r is not None]


def expectancy(trades: Sequence[Trade]) -> float | None:
    """Mean realized R, or `None` for an empty list (which is not an expectancy of zero)."""
    rs = realized_rs(trades)
    return sum(rs) / len(rs) if rs else None


def stressed_r(
    trade: Trade,
    *,
    spread_mult: float = 2.0,
    slippage_mult: float = 2.0,
    funding_mult: float = 1.5,
) -> float:
    """`trade`'s realized R re-priced under gate rule 6's cost stress, post-hoc.

    Only costs are scaled, and a cost that is a fraction of a tick never moves a *fill*:
    the same bars touch the same levels at the same prices, so the trade's gross result is
    unchanged and only the cost total moves. That makes the stress exact arithmetic on a
    finished trade rather than a second replay — the extra cost across the entry fill and
    every exit leg, divided by the trade's own risk.
    """
    if trade.realized_r is None:
        raise ValueError(f"trade {trade.id!r} is not closed; stressed_r needs a realized R")
    extra = 0.0
    for fill in (trade.entry_fill, *trade.legs):
        stressed = stress(
            fill.cost, spread_mult=spread_mult, slippage_mult=slippage_mult, funding_mult=funding_mult
        )
        extra += stressed.total - fill.cost.total
    return trade.realized_r - extra / trade.risk_r


def buy_and_hold_curve(
    closes: Sequence[tuple[datetime, float]],
    windows: Sequence[tuple[datetime, datetime]],
) -> list[float]:
    """Buy-and-hold equity over the OOS windows only, chained so the gaps cost nothing.

    Rule 5 compares the config against buy-and-hold *on the same span*, and the config only
    trades out of sample — so the benchmark is held only during the OOS windows too. Each
    window's closes are compounded bar to bar and the next window resumes from where the
    last left off, which is what "chained multiplicatively" buys: the price move across the
    IS gap between two OOS windows is neither credited nor charged to the benchmark.

    The curve always starts at 1.0 and is therefore never empty, so `gate.mar` can always
    read it.
    """
    curve = [1.0]
    level = 1.0
    for window_start, window_end in windows:
        segment = [close for ts, close in closes if window_start <= ts < window_end]
        for previous, current in zip(segment, segment[1:], strict=False):
            if previous <= 0.0:
                raise ValueError(f"close {previous} is not positive; a buy-and-hold curve needs prices")
            level *= current / previous
            curve.append(level)
    return curve


# --- the tournament ----------------------------------------------------------------


@dataclass(frozen=True)
class TournamentResult:
    """Everything one tournament run produced, in memory and already written to the store.

    `rows` are exactly the dicts handed to `Store.write_results`. `gates` and `oos_trades`
    are keyed by config id, including the derived `IS_SELECTED` ids. `selection` maps an
    `IS_SELECTED` config id to `{split label: chosen exit}`. `n_trials` and
    `trial_sr_variance` are the two run-wide figures every gate was evaluated against.

    `runs` is empty unless the run asked for `keep_runs=True`: nothing downstream needs it
    (the report reads `rows`, `gates` and `oos_trades`) and holding 2,016 of them is what
    the sweep cannot afford.
    """

    run_id: str
    rows: tuple[dict[str, Any], ...]
    excluded: tuple[tuple[Instrument, str], ...]
    gates: dict[str, GateResult]
    selection: dict[str, dict[str, str]]
    resolution_modes: dict[str, str]
    runs: dict[str, ConfigRun] = field(default_factory=dict)
    oos_trades: dict[str, tuple[Trade, ...]] = field(default_factory=dict)
    n_trials: int = 0
    trial_sr_variance: float | None = None


def _finite_or_none(value: float | None) -> float | None:
    """Map a gate figure onto what the `results` table can hold: `inf` -> 1e9, `nan` -> None."""
    if value is None:
        return None
    if math.isnan(value):
        return None
    if math.isinf(value):
        return math.copysign(_INF_SENTINEL, value)
    return value


def _row(run_id: str, config_id: str, split: str, *, ts: datetime, **values: Any) -> dict[str, Any]:
    """A `results` row with every column of `Store._RESULTS_COLUMNS` present.

    `ts` is the run's own start, passed in rather than read per row: 2,016 configs write
    ~26,000 rows, and one run stamped with 26,000 different instants is not a run anyone
    can group by, order by or reason about.
    """
    row: dict[str, Any] = dict.fromkeys(_RESULTS_COLUMNS)
    row["run_id"] = run_id
    row["config_id"] = config_id
    row["split"] = split
    row["ts"] = ts
    row.update(values)
    unknown = set(row) - set(_RESULTS_COLUMNS)
    if unknown:  # pragma: no cover - a typo in this module, caught the moment it is made
        raise ValueError(f"unknown results column(s): {sorted(unknown)!r}")
    return row


def _identity_columns(entry: str, exit_name: str, session: str, inst: Instrument) -> dict[str, Any]:
    """The five columns that name a config in a `results` row, whatever the split."""
    return {
        "entry": entry,
        "exit": exit_name,
        "session": session,
        "venue": inst.venue,
        "symbol": inst.symbol,
    }


def _gate_columns(result: GateResult) -> dict[str, Any]:
    """The gate half of a pooled `results` row, with `inf`/`nan` mapped for storage."""
    return {
        "dsr_prob": _finite_or_none(result.dsr.prob if result.dsr is not None else None),
        "boot_p5": _finite_or_none(result.boot_p5),
        "diff_p5": _finite_or_none(result.diff_p5),
        "mar": _finite_or_none(result.mar_config),
        "mar_bh": _finite_or_none(result.mar_bh),
        "g1": result.rule1,
        "g2": result.rule2,
        "g3": result.rule3,
        "g4": result.rule4,
        "g5": result.rule5,
        "g6": result.rule6,
        "passed": result.passed,
    }


def _ordered_entries(entries: Sequence[str]) -> list[str]:
    """`ict` first (it sets the baseline's trade rate), `baseline` last, the rest in between."""
    return (
        [e for e in entries if e == "ict"]
        + [e for e in entries if e not in ("ict", "baseline")]
        + [e for e in entries if e == "baseline"]
    )


def _pooled_oos(trades: Sequence[Trade], splits: Sequence[Split]) -> list[Trade]:
    """Every OOS trade across the splits, chronological and each counted once."""
    pooled: list[Trade] = []
    for split in splits:
        pooled.extend(split_trades(trades, split)[1])
    return pooled


def _pooled_is(trades: Sequence[Trade], splits: Sequence[Split]) -> list[Trade]:
    """The widest in-sample window: the last split's, which contains every earlier one.

    The walk-forward is anchored, so split k's IS window is a prefix of split k+1's. Adding
    them up would count the early trades once per split; taking the last one counts each
    trade exactly once and is the largest honest IS sample.
    """
    return [] if not splits else split_trades(trades, splits[-1])[0]


def run_tournament(
    store: Store,
    instruments: Sequence[Instrument],
    *,
    entry_factory: EntryFactory,
    session_factory: SessionFactory,
    cost_model: CostModel = _NULL_COST_MODEL,
    resolver: ExitResolver = _FILL_RESOLVER,
    regime_tagger: Callable[[Context], str] = tag,
    entries: Sequence[str] = ENTRIES,
    exits: Sequence[str] = EXITS,
    sessions: Sequence[str] = SESSIONS,
    start: datetime | None = None,
    end: datetime | None = None,
    min_months: int = 18,
    seed: int = 0,
    resume: bool = False,
    keep_runs: bool = False,
    run_id: str | None = None,
    progress: Callable[[str], None] | None = None,
) -> TournamentResult:
    """Run the whole matrix, gate every config, and write `results`, `trades` and `equity`.

    Per instrument the run window is the intersection of `start`/`end` (each defaulting to
    the store's own 4H bar range) with the bars actually held; an instrument with fewer
    than `min_months` of history there is excluded from the gate and listed with the reason
    `insufficient_history:<months>`.

    Configs run instrument by instrument, `ict` first so the `baseline` control can be
    frequency-matched to the mean ICT trade rate per 1,000 4H bars on that instrument
    (design spec section 4); with no ICT trades at all the target rate falls back to 1.0.
    Every `run_config` and every `gate.evaluate` is wrapped: a failure is logged and becomes
    one `results` row carrying `excluded_reason="error:<type>: <message>"` rather than
    losing the run — but `MAX_CONSECUTIVE_ERRORS` replay failures in a row abort it.

    Beside the 2,016 per-config verdicts, an **IS-selected** view is written per
    `(entry, session, instrument)`: for each split the exit with the highest IS expectancy
    is chosen (ties go to `EXIT_GRID` order; a split with fewer than 10 IS trades is
    skipped), that exit's OOS trades are taken for that split, and the result is pooled
    across splits under the config id `entry|IS_SELECTED|session|venue:symbol`. Its `n_is`
    pools each split's chosen exit over that split's *incremental* IS window, so no trade is
    counted twice.

    ⚠ `n_trials` — what every deflated Sharpe is deflated by — is `len(matrix) + len(views)`:
    the selection views are results of the same search and were themselves chosen by looking
    at 16 exits per split, so counting only the enumerated matrix would under-deflate the
    one view a reader is most likely to trade.

    `seed` seeds both the gate's bootstraps and, per config, the entry factory (see
    `run_config`). Per-config trades and equity go to the store under
    `f"{run_id}:{config.id}"`, keeping every config's ledger separable under the
    `(run_id, ...)` primary keys.

    With `resume=True` a config whose child run id already holds trades is not replayed:
    its trades are read back and its bar count and resolution mode recomputed from the
    source. That makes a re-run of an interrupted sweep cheap, at the cost of trusting what
    is in the store — see the WU-2C handoff for when it is *not* safe. `keep_runs=True`
    additionally returns every `ConfigRun`; it is off by default because holding them is
    what a 2,016-config sweep cannot afford.
    """
    started = datetime.now(UTC)
    run_id = run_id or f"tournament-{started:%Y%m%dT%H%M%S}"
    for name in exits:
        exit_rule(name)  # fail fast on a typo rather than 2,000 configs later
    source = _MemoisedSource(store)
    store.upsert_instruments(instruments)  # `write_trades` resolves instruments through this table

    windows, excluded = _instrument_windows(store, instruments, start=start, end=end, min_months=min_months)
    ordered = _ordered_entries(entries)

    # --- pass 1: replay (or resume) every config -------------------------------
    matrix = _run_matrix(
        store,
        source,
        windows,
        run_id=run_id,
        entry_factory=entry_factory,
        session_factory=session_factory,
        cost_model=cost_model,
        resolver=resolver,
        regime_tagger=regime_tagger,
        ordered_entries=ordered,
        exits=exits,
        sessions=sessions,
        seed=seed,
        resume=resume,
        keep_runs=keep_runs,
        progress=progress,
    )

    # --- pass 2: V across every trial, then every gate ---------------------------
    splits_by_instrument = {
        instrument_key(instrument): walk_forward_splits(window_start, window_end)
        for instrument, (window_start, window_end) in windows.items()
    }
    pooled: dict[str, tuple[Trade, ...]] = {
        config.id: tuple(
            _pooled_oos(
                matrix.replayed[config.id].trades, splits_by_instrument[instrument_key(config.instrument)]
            )
        )
        for config in matrix.configs
        if config.id in matrix.replayed
    }
    trial_sharpes = [gate.sharpe(realized_rs(t)) for t in pooled.values() if len(t) >= 2]
    trial_sr_variance = float(np.var(trial_sharpes, ddof=1)) if len(trial_sharpes) >= 2 else None

    selection, views = _is_selected_views(
        matrix.replayed, splits_by_instrument, ordered, exits, sessions, windows
    )
    # Every view has to be in `pooled` before any row is built: a view's own rule-4 baseline
    # is `baseline|IS_SELECTED|session|instrument`, which is another view, and neither the
    # matrix rows nor the view rows can be trusted to reach it first.
    pooled.update({config_id: tuple(view.oos_trades) for config_id, view in views.items()})

    context = _GateContext(
        run_id=run_id,
        ts=started,
        splits=splits_by_instrument,
        closes=matrix.closes,
        pooled=pooled,
        n_trials=len(matrix.configs) + len(views),
        seed=seed,
        trial_sr_variance=trial_sr_variance,
        gates={},
    )
    rows = _config_rows(matrix, context)
    rows.extend(_is_selected_rows(views, windows, matrix.resolution_modes, context))

    store.write_results(rows)
    return TournamentResult(
        run_id=run_id,
        rows=tuple(rows),
        excluded=tuple(excluded),
        gates=context.gates,
        selection=selection,
        resolution_modes=matrix.resolution_modes,
        runs=matrix.runs,
        oos_trades=pooled,
        n_trials=context.n_trials,
        trial_sr_variance=trial_sr_variance,
    )


def _instrument_windows(
    store: Store,
    instruments: Sequence[Instrument],
    *,
    start: datetime | None,
    end: datetime | None,
    min_months: int,
) -> tuple[dict[Instrument, tuple[datetime, datetime]], list[tuple[Instrument, str]]]:
    """The `[start, end)` each instrument is run over, and the ones with too little history.

    The window is the *intersection* of what was asked for and what the store holds. A
    four-year request against six months of bars is six months of history, not four years
    of it: measuring the request instead would wave a brand-new listing straight past
    `min_months` and hand it a walk-forward whose early splits contain no bars at all.
    """
    windows: dict[Instrument, tuple[datetime, datetime]] = {}
    excluded: list[tuple[Instrument, str]] = []
    for instrument in instruments:
        bar_range = store.bar_range(instrument, "4h")
        if bar_range is None:
            excluded.append((instrument, "insufficient_history:0"))
            continue
        first, last, _ = bar_range
        window_start = first if start is None else max(start, first)
        window_end = (last + _FOUR_HOURS) if end is None else min(end, last + _FOUR_HOURS)
        months = months_between(window_start, window_end)
        if months < min_months:
            excluded.append((instrument, f"insufficient_history:{months}"))
            continue
        windows[instrument] = (window_start, window_end)
    return windows, excluded


# --- pass 1 ------------------------------------------------------------------------


@dataclass
class _Matrix:
    """What pass 1 attempted and what came back, keyed by config id.

    `closes` is keyed by instrument instead: every config of an instrument buys and holds
    the same series, and one copy per config is the memory C1 was about.
    """

    configs: list[Config] = field(default_factory=list)
    replayed: dict[str, _Replayed] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)
    persist_errors: dict[str, str] = field(default_factory=dict)
    resolution_modes: dict[str, str] = field(default_factory=dict)
    closes: dict[str, tuple[tuple[datetime, float], ...]] = field(default_factory=dict)
    runs: dict[str, ConfigRun] = field(default_factory=dict)


def _run_matrix(
    store: Store,
    source: ReplaySource,
    windows: Mapping[Instrument, tuple[datetime, datetime]],
    *,
    run_id: str,
    entry_factory: EntryFactory,
    session_factory: SessionFactory,
    cost_model: CostModel,
    resolver: ExitResolver,
    regime_tagger: Callable[[Context], str],
    ordered_entries: Sequence[str],
    exits: Sequence[str],
    sessions: Sequence[str],
    seed: int,
    resume: bool,
    keep_runs: bool,
    progress: Callable[[str], None] | None,
) -> _Matrix:
    """Replay every config of every qualifying instrument; see `run_tournament`.

    Two failures are told apart. A config that will not *run* is logged, recorded as an
    `error:` reason and dropped from the gate — and `MAX_CONSECUTIVE_ERRORS` of those in a
    row abort the run. A config that ran but will not *persist* is logged and recorded as a
    `persist_error:` note on its pooled row, and otherwise keeps everything: its trades, its
    place in the trial variance, and its gate verdict. Losing the write is not a reason to
    also lose the answer.
    """
    matrix = _Matrix()
    consecutive = 0
    first_error: Exception | None = None
    for instrument, (window_start, window_end) in windows.items():
        key = instrument_key(instrument)
        in_range = _in_range_bars(source, instrument, window_start, window_end)
        matrix.closes[key] = tuple((bar.ts_open, bar.close) for bar in in_range)
        ict_rates: list[float] = []
        baseline_rate: float | None = None
        for entry in ordered_entries:
            if entry == "baseline":
                mean_rate = sum(ict_rates) / len(ict_rates) if ict_rates else 0.0
                baseline_rate = mean_rate or 1.0
            for exit_name in exits:
                for session in sessions:
                    config = Config(entry=entry, exit=exit_name, session=session, instrument=instrument)
                    matrix.configs.append(config)
                    if progress is not None:
                        progress(config.id)
                    child_id = f"{run_id}:{config.id}"
                    stored: list[Trade] = []
                    try:
                        stored = store.trades(child_id) if resume else []
                        run = (
                            _resumed_run(
                                config,
                                stored,
                                source,
                                start=window_start,
                                end=window_end,
                                equity=tuple(store.equity(child_id)) if keep_runs else (),
                            )
                            if stored
                            else run_config(
                                config,
                                source,
                                entry_factory=entry_factory,
                                session_factory=session_factory,
                                cost_model=cost_model,
                                resolver=resolver,
                                regime_tagger=regime_tagger,
                                start=window_start,
                                end=window_end,
                                baseline_rate=baseline_rate if entry == "baseline" else None,
                                seed=seed,
                            )
                        )
                    except Exception as exc:  # one bad config is a row, never a lost run
                        logger.exception("tournament config %s could not be run", config.id)
                        matrix.errors[config.id] = f"error:{type(exc).__name__}: {exc}"[:200]
                        first_error = first_error or exc
                        consecutive += 1
                        if consecutive >= MAX_CONSECUTIVE_ERRORS:
                            raise RuntimeError(
                                f"aborting run {run_id!r}: {consecutive} consecutive config failures, "
                                f"the last of them {config.id!r}"
                            ) from first_error
                        continue
                    consecutive, first_error = 0, None
                    matrix.replayed[config.id] = _Replayed(
                        trades=run.trades, n_bars=run.n_bars, resolution_mode=run.resolution_mode
                    )
                    matrix.resolution_modes.setdefault(key, run.resolution_mode)
                    if keep_runs:
                        matrix.runs[config.id] = run
                    if entry == "ict":
                        ict_rates.append(run.trades_per_1000_bars)
                    if not stored:
                        _persist(store, child_id, run, matrix)
    return matrix


def _persist(store: Store, child_id: str, run: ConfigRun, matrix: _Matrix) -> None:
    """Write one config's ledger, recording a failure against the config without losing it."""
    try:
        store.write_trades(child_id, run.trades)
        store.write_equity(child_id, run.equity_curve)
    except Exception as exc:  # the config still ran: keep its gate, flag the write
        logger.exception("tournament config %s ran but could not be stored", run.config.id)
        matrix.persist_errors[run.config.id] = f"persist_error:{type(exc).__name__}: {exc}"[:200]


def _resumed_run(
    config: Config,
    trades: Sequence[Trade],
    source: ReplaySource,
    *,
    start: datetime,
    end: datetime,
    equity: tuple[tuple[datetime, float], ...],
) -> ConfigRun:
    """Rebuild a `ConfigRun` from trades already in the store, replaying nothing.

    `n_bars` and `resolution_mode` are recomputed from the instrument's bars, which the
    memoised source is already holding, so this costs one query rather than a replay.
    `equity` is read back by the caller only when the run asked to keep its `ConfigRun`s:
    neither of the two passes looks at an equity curve.
    """
    in_range = _in_range_bars(source, config.instrument, start, end)
    return ConfigRun(
        config=config,
        trades=tuple(trades),
        equity_curve=equity,
        n_bars=len(in_range),
        resolution_mode=_resolution_mode(in_range),
    )


# --- pass 2 ------------------------------------------------------------------------


@dataclass(frozen=True)
class _GateContext:
    """What every pooled row of one run shares: the run's stamp and the gate's inputs.

    `gates` is filled in as the rows are built — it is the same dict `TournamentResult`
    hands back, so a row and its `GateResult` can never disagree.
    """

    run_id: str
    ts: datetime
    splits: Mapping[str, Sequence[Split]]
    closes: Mapping[str, tuple[tuple[datetime, float], ...]]
    pooled: Mapping[str, tuple[Trade, ...]]
    n_trials: int
    seed: int
    trial_sr_variance: float | None
    gates: dict[str, GateResult]


def _config_rows(matrix: _Matrix, context: _GateContext) -> list[dict[str, Any]]:
    """Every `results` row of the enumerated matrix: one per split, then the pooled gate row."""
    rows: list[dict[str, Any]] = []
    for config in matrix.configs:
        identity = _identity_columns(config.entry, config.exit, config.session, config.instrument)
        if config.id in matrix.errors:
            rows.append(
                _row(
                    context.run_id,
                    config.id,
                    "pooled",
                    ts=context.ts,
                    **identity,
                    excluded_reason=matrix.errors[config.id],
                )
            )
            continue
        replayed = matrix.replayed[config.id]
        key = instrument_key(config.instrument)
        splits = context.splits[key]
        for split in splits:
            is_trades, oos = split_trades(replayed.trades, split)
            rows.append(
                _row(
                    context.run_id,
                    config.id,
                    split.label,
                    ts=context.ts,
                    **identity,
                    n_is=len(is_trades),
                    n_oos=len(oos),
                    exp_is=expectancy(is_trades),
                    exp_oos=expectancy(oos),
                    resolution_mode=replayed.resolution_mode,
                )
            )
        rows.append(
            _pooled_row(
                run_id=context.run_id,
                ts=context.ts,
                config_id=config.id,
                identity=identity,
                is_trades=_pooled_is(replayed.trades, splits),
                oos=context.pooled[config.id],
                baseline=context.pooled.get(f"baseline|{config.exit}|{config.session}|{key}", ()),
                closes=context.closes[key],
                splits=splits,
                resolution_mode=replayed.resolution_mode,
                n_trials=context.n_trials,
                seed=context.seed,
                trial_sr_variance=context.trial_sr_variance,
                gates=context.gates,
                note=matrix.persist_errors.get(config.id),
            )
        )
    return rows


def _is_selected_rows(
    views: Mapping[str, _ISSelectedView],
    windows: Mapping[Instrument, tuple[datetime, datetime]],
    resolution_modes: Mapping[str, str],
    context: _GateContext,
) -> list[dict[str, Any]]:
    """The pooled `results` row for each IS-selected view; see `_is_selected_views`."""
    by_key = {instrument_key(instrument): instrument for instrument in windows}
    rows: list[dict[str, Any]] = []
    for config_id, view in views.items():
        entry, _, session, key = config_id.split("|")
        rows.append(
            _pooled_row(
                run_id=context.run_id,
                ts=context.ts,
                config_id=config_id,
                identity=_identity_columns(entry, "IS_SELECTED", session, by_key[key]),
                is_trades=view.is_trades,
                oos=view.oos_trades,
                baseline=context.pooled.get(f"baseline|IS_SELECTED|{session}|{key}", ()),
                closes=context.closes.get(key, ()),
                splits=context.splits[key],
                resolution_mode=resolution_modes.get(key),
                n_trials=context.n_trials,
                seed=context.seed,
                trial_sr_variance=context.trial_sr_variance,
                gates=context.gates,
            )
        )
    return rows


def _pooled_row(
    *,
    run_id: str,
    ts: datetime,
    config_id: str,
    identity: dict[str, Any],
    is_trades: Sequence[Trade],
    oos: Sequence[Trade],
    baseline: Sequence[Trade],
    closes: Sequence[tuple[datetime, float]],
    splits: Sequence[Split],
    resolution_mode: str | None,
    n_trials: int,
    seed: int,
    trial_sr_variance: float | None,
    gates: dict[str, GateResult],
    note: str | None = None,
) -> dict[str, Any]:
    """Build the `split="pooled"` row for one config, running the gate on its pooled OOS.

    A config whose gate cannot be evaluated at all — `equity_curve_from_r` refusing a
    ruinous R multiple, an empty buy-and-hold window, a zero-length OOS span — still gets
    its row, carrying the reason in `excluded_reason` and no gate columns.

    `note` is something already known to be wrong with the config that did *not* stop it
    being graded (today: its ledger could not be stored). It is joined onto whatever the
    gate has to say, so `excluded_reason` remains the one column to read.
    """
    base = {
        **identity,
        "n_is": len(is_trades),
        "n_oos": len(oos),
        "exp_is": expectancy(is_trades),
        "exp_oos": expectancy(oos),
        "resolution_mode": resolution_mode,
    }
    try:
        result = gate.evaluate(
            realized_rs(oos),
            realized_rs(baseline),
            buy_and_hold_curve(closes, [(s.oos_start, s.oos_end) for s in splits]),
            years=oos_years(splits),
            n_trials=n_trials,
            stressed_r=[stressed_r(t) for t in oos],
            seed=seed,
            trial_sr_variance=trial_sr_variance,
        )
    except Exception as exc:  # a blown-up config is a failed row, not a lost run
        logger.exception("tournament config %s could not be graded", config_id)
        reason = f"error:{type(exc).__name__}: {exc}"[:200]
        return _row(run_id, config_id, "pooled", ts=ts, **base, excluded_reason=_join(note, reason))
    gates[config_id] = result
    return _row(run_id, config_id, "pooled", ts=ts, **base, **_gate_columns(result), excluded_reason=note)


def _join(note: str | None, reason: str) -> str:
    """One `excluded_reason` for a row that has more than one thing to say about itself."""
    return (f"{note}; {reason}" if note else reason)[:200]


@dataclass(frozen=True)
class _ISSelectedView:
    """One `(entry, session, instrument)`'s per-split exit choice and the trades it implies."""

    chosen: dict[str, str]
    is_trades: list[Trade]
    oos_trades: list[Trade]


def _is_selected_views(
    replayed: Mapping[str, _Replayed],
    splits_by_instrument: Mapping[str, Sequence[Split]],
    entries: Sequence[str],
    exits: Sequence[str],
    sessions: Sequence[str],
    windows: Mapping[Instrument, tuple[datetime, datetime]],
) -> tuple[dict[str, dict[str, str]], dict[str, _ISSelectedView]]:
    """Pick the best-IS exit per split and pool that exit's OOS trades across splits.

    The IS series pools each split's chosen exit over that split's *incremental* IS window
    — `[previous split's is_end, this split's is_end)`, and `[is_start, is_end)` for the
    first — so it counts each trade once, always under the exit that was selected for the
    split that trade informed.

    ⚠ Each view is a trial of the same search: choosing per split between 16 enumerated
    exits is a search over those exits, and the resulting view is a candidate a reader may
    well trade. `run_tournament` therefore deflates by `len(matrix) + len(views)`, not by
    the matrix alone.
    """
    selection: dict[str, dict[str, str]] = {}
    views: dict[str, _ISSelectedView] = {}
    for instrument in windows:
        key = instrument_key(instrument)
        splits = splits_by_instrument[key]
        if not splits:
            continue
        for entry in entries:
            for session in sessions:
                view = _select_exits(replayed, splits, entry, session, key, exits)
                if not view.chosen:
                    continue
                config_id = f"{entry}|IS_SELECTED|{session}|{key}"
                selection[config_id] = view.chosen
                views[config_id] = view
    return selection, views


def _select_exits(
    replayed: Mapping[str, _Replayed],
    splits: Sequence[Split],
    entry: str,
    session: str,
    key: str,
    exits: Sequence[str],
) -> _ISSelectedView:
    """The IS-selected view for one `(entry, session, instrument)`; see `_is_selected_views`."""
    chosen: dict[str, str] = {}
    is_trades: list[Trade] = []
    oos_trades: list[Trade] = []
    previous_is_end = splits[0].is_start
    for split in splits:
        best_name: str | None = None
        best_expectancy = -math.inf
        for exit_name in exits:  # EXIT_GRID order, so a tie goes to the earlier rule
            run = replayed.get(f"{entry}|{exit_name}|{session}|{key}")
            if run is None:
                continue
            candidate = split_trades(run.trades, split)[0]
            if len(candidate) < _MIN_IS_TRADES_FOR_SELECTION:
                continue
            value = expectancy(candidate)
            if value is not None and value > best_expectancy:
                best_name, best_expectancy = exit_name, value
        if best_name is None:
            continue
        chosen[split.label] = best_name
        window_is, window_oos = split_trades(replayed[f"{entry}|{best_name}|{session}|{key}"].trades, split)
        is_trades.extend(t for t in window_is if t.entry_fill.ts >= previous_is_end)
        oos_trades.extend(window_oos)
        previous_is_end = split.is_end
    return _ISSelectedView(chosen=chosen, is_trades=is_trades, oos_trades=oos_trades)
