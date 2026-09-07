"""Tests for `swingforge.lab.tournament`."""

from __future__ import annotations

import zlib
from collections.abc import Sequence
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import numpy as np
import pytest

from swingforge.adapters.replay import ReplaySource
from swingforge.adapters.store import Store
from swingforge.core.types import CostBreakdown, Fill, Instrument, Trade
from swingforge.lab import gate, tournament
from swingforge.lab.tournament import Config, Split, add_months, configs, instrument_key, months_between
from swingforge.strategies.exits import EXIT_GRID
from tests.unit.synth_store import (
    EntryFactoryStub,
    EveryN,
    Never,
    Raising,
    add_instrument,
    build_store,
    session_factory,
)

BTC = Instrument(
    venue="hyperliquid",
    symbol="BTC",
    tick_size=Decimal("0.5"),
    contract_multiplier=Decimal("1"),
    quote_ccy="USD",
    session_profile="perp",
)
ETH = BTC.model_copy(update={"symbol": "ETH"})


# --- the config matrix -----------------------------------------------------------


def test_configs_is_the_full_matrix() -> None:
    got = configs([BTC, ETH])
    assert len(got) == 3 * 16 * 3 * 2
    assert len({config.id for config in got}) == len(got)


def test_config_id_is_entry_exit_session_instrument() -> None:
    config = Config(entry="ict", exit="fixed_r_2", session="london_ny", instrument=BTC)
    assert config.id == "ict|fixed_r_2|london_ny|hyperliquid:BTC"


def test_config_is_frozen() -> None:
    config = Config(entry="ict", exit="fixed_r_2", session="none", instrument=BTC)
    with pytest.raises(ValueError):
        config.entry = "zones"  # type: ignore[misc]


def test_configs_honours_narrowed_dimensions() -> None:
    got = configs([BTC], entries=("ict",), exits=("fixed_r_2", "fixed_r_3"), sessions=("none",))
    assert [config.id for config in got] == [
        "ict|fixed_r_2|none|hyperliquid:BTC",
        "ict|fixed_r_3|none|hyperliquid:BTC",
    ]


def test_exit_names_default_to_the_whole_grid() -> None:
    assert tuple(rule.name for rule in EXIT_GRID) == tournament.EXITS
    assert set(tournament.EXIT_RULES) == set(tournament.EXITS)


def test_exit_rule_rejects_an_unknown_name() -> None:
    with pytest.raises(KeyError):
        tournament.exit_rule("no_such_rule")


# --- calendar month arithmetic ---------------------------------------------------


def test_add_months_walks_calendar_months() -> None:
    start = datetime(2021, 1, 31, tzinfo=UTC)
    assert add_months(start, 1) == datetime(2021, 2, 28, tzinfo=UTC)
    assert add_months(start, 12) == datetime(2022, 1, 31, tzinfo=UTC)
    assert add_months(start, 0) == start


def test_add_months_crosses_the_year_boundary() -> None:
    assert add_months(datetime(2021, 11, 15, tzinfo=UTC), 3) == datetime(2022, 2, 15, tzinfo=UTC)


def test_months_between_counts_whole_months() -> None:
    a = datetime(2021, 1, 1, tzinfo=UTC)
    assert months_between(a, datetime(2021, 1, 1, tzinfo=UTC)) == 0
    assert months_between(a, datetime(2021, 6, 30, tzinfo=UTC)) == 5
    assert months_between(a, datetime(2021, 7, 1, tzinfo=UTC)) == 6
    assert months_between(a, datetime(2020, 1, 1, tzinfo=UTC)) == 0


# --- anchored walk-forward -------------------------------------------------------


def test_walk_forward_splits_over_four_years() -> None:
    start = datetime(2020, 1, 1, tzinfo=UTC)
    end = datetime(2024, 1, 1, tzinfo=UTC)
    splits = tournament.walk_forward_splits(start, end)

    assert len(splits) == 12
    assert all(split.is_start == start for split in splits)
    assert all(split.is_end == split.oos_start for split in splits)
    assert splits[0].oos_start == datetime(2021, 1, 1, tzinfo=UTC)
    assert splits[0].oos_end == datetime(2021, 4, 1, tzinfo=UTC)
    assert splits[-1].oos_end == end
    for earlier, later in zip(splits, splits[1:], strict=False):
        assert earlier.oos_end == later.oos_start  # contiguous and non-overlapping


def test_walk_forward_splits_drops_a_truncated_final_window() -> None:
    start = datetime(2020, 1, 1, tzinfo=UTC)
    splits = tournament.walk_forward_splits(start, datetime(2021, 5, 1, tzinfo=UTC))
    assert len(splits) == 1
    assert splits[0].oos_end == datetime(2021, 4, 1, tzinfo=UTC)


def test_walk_forward_splits_is_empty_when_history_is_too_short() -> None:
    start = datetime(2020, 1, 1, tzinfo=UTC)
    assert tournament.walk_forward_splits(start, datetime(2020, 12, 1, tzinfo=UTC)) == []


def test_split_is_frozen() -> None:
    split = tournament.walk_forward_splits(
        datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 4, 1, tzinfo=UTC)
    )[0]
    assert isinstance(split, Split)
    with pytest.raises(FrozenInstanceError):
        split.is_start = datetime(2020, 2, 1, tzinfo=UTC)  # type: ignore[misc]


def test_oos_years_sums_the_out_of_sample_span() -> None:
    splits = tournament.walk_forward_splits(
        datetime(2020, 1, 1, tzinfo=UTC), datetime(2024, 1, 1, tzinfo=UTC)
    )
    assert tournament.oos_years(splits) == pytest.approx(3.0, abs=0.01)
    assert tournament.oos_years([]) == 0.0


# --- splitting a trade list ------------------------------------------------------


def _trade(ts: datetime, realized_r: float, *, costs: CostBreakdown | None = None) -> Trade:
    fill = Fill(
        order_id="o",
        ts=ts,
        price=100.0,
        qty=1.0,
        cost=costs if costs is not None else CostBreakdown(),
        leg="entry",
    )
    exit_fill = fill.model_copy(update={"leg": "stop", "ts": ts + timedelta(hours=4)})
    return Trade(
        id=f"t{ts.isoformat()}",
        instrument=BTC,
        direction=1,
        entry_fill=fill,
        legs=(exit_fill,),
        stop=99.0,
        target=None,
        risk_r=100.0,
        realized_r=realized_r,
        opened_bar=0,
        closed_bar=1,
    )


def test_split_trades_partitions_by_entry_timestamp() -> None:
    split = Split(
        is_start=datetime(2020, 1, 1, tzinfo=UTC),
        is_end=datetime(2021, 1, 1, tzinfo=UTC),
        oos_start=datetime(2021, 1, 1, tzinfo=UTC),
        oos_end=datetime(2021, 4, 1, tzinfo=UTC),
    )
    trades = [
        _trade(datetime(2020, 6, 1, tzinfo=UTC), 1.0),
        _trade(datetime(2021, 2, 1, tzinfo=UTC), 2.0),
        _trade(datetime(2021, 5, 1, tzinfo=UTC), 3.0),
    ]
    is_trades, oos_trades = tournament.split_trades(trades, split)
    assert [t.realized_r for t in is_trades] == [1.0]
    assert [t.realized_r for t in oos_trades] == [2.0]


# --- post-hoc cost stress --------------------------------------------------------


def test_stressed_r_subtracts_the_extra_cost_in_r() -> None:
    costs = CostBreakdown(spread=1.0, commission=2.0, funding=4.0, slippage=3.0)
    trade = _trade(datetime(2021, 1, 1, tzinfo=UTC), 2.0, costs=costs)
    # entry + one leg, each: spread 1->2 (+1), funding 4->6 (+2), slippage 3->6 (+3);
    # commission untouched. 6 extra per fill, 12 across both, over risk_r=100 -> 0.12.
    assert tournament.stressed_r(trade) == pytest.approx(2.0 - 0.12)


def test_stressed_r_is_unchanged_by_a_costless_trade() -> None:
    trade = _trade(datetime(2021, 1, 1, tzinfo=UTC), 1.5)
    assert tournament.stressed_r(trade) == pytest.approx(1.5)


def test_stressed_r_rejects_an_open_trade() -> None:
    trade = _trade(datetime(2021, 1, 1, tzinfo=UTC), 1.0).model_copy(update={"realized_r": None})
    with pytest.raises(ValueError, match="closed"):
        tournament.stressed_r(trade)


# --- buy and hold ----------------------------------------------------------------


def test_buy_and_hold_curve_chains_segments_without_gap_jumps() -> None:
    base = datetime(2021, 1, 1, tzinfo=UTC)
    closes = [
        (base, 100.0),
        (base + timedelta(days=1), 110.0),
        (base + timedelta(days=40), 500.0),  # in the gap: must not be counted
        (base + timedelta(days=100), 200.0),
        (base + timedelta(days=101), 220.0),
    ]
    windows = [
        (base, base + timedelta(days=2)),
        (base + timedelta(days=100), base + timedelta(days=102)),
    ]
    curve = tournament.buy_and_hold_curve(closes, windows)
    assert curve == pytest.approx([1.0, 1.1, 1.21])


def test_buy_and_hold_curve_of_no_closes_is_flat() -> None:
    base = datetime(2021, 1, 1, tzinfo=UTC)
    assert tournament.buy_and_hold_curve([], [(base, base)]) == [1.0]


# --- end to end on a synthetic store ---------------------------------------------


SYNTH_START = datetime(2021, 1, 1, tzinfo=UTC)
SYNTH_MONTHS = 20


@pytest.fixture(scope="module")
def store() -> Store:
    """One 20-month BTC store shared by the `run_config` tests, which only read from it."""
    built = build_store([BTC], start=SYNTH_START, months=SYNTH_MONTHS, seed=7)
    yield built
    built.close()


# Signal spacing in 4H bars. Deliberately sparse: `Store.write_trades` costs about 30 ms
# per trade in this environment (see the WU-2C handoff), so a busier synthetic entry makes
# this file slow without testing anything more.
ICT_EVERY = 60
ZONES_EVERY = 80


def _factory(**overrides):
    builders = {
        "ict": lambda instrument, rate: EveryN(ICT_EVERY, name="ict"),
        "zones": lambda instrument, rate: EveryN(ZONES_EVERY, name="zones"),
        # `rate` is trades per 1,000 4H bars, so one signal every 1000/rate bars matches it.
        "baseline": lambda instrument, rate: EveryN(
            max(1, round(1000 / max(rate or 1.0, 1e-9))), name="baseline"
        ),
    }
    builders.update(overrides)
    return EntryFactoryStub(builders)


def test_run_config_replays_once_and_reports_the_window(store) -> None:
    end = tournament.add_months(SYNTH_START, SYNTH_MONTHS)
    run = tournament.run_config(
        Config(entry="ict", exit="fixed_r_2", session="none", instrument=BTC),
        ReplaySource(store),
        entry_factory=_factory(),
        session_factory=session_factory,
        start=SYNTH_START,
        end=end,
    )

    assert run.n_bars > 0
    assert run.trades, "the synthetic entry should produce trades"
    assert all(trade.entry_fill.ts >= SYNTH_START for trade in run.trades)
    # every 4H bar of the synthetic store has all four 1H sub-bars
    assert run.resolution_mode == "subbars"
    assert run.equity_curve and len(run.equity_curve) == len(run.trades)


def test_run_config_warmup_never_contributes_a_trade(store) -> None:
    end = tournament.add_months(SYNTH_START, SYNTH_MONTHS)
    later = tournament.add_months(SYNTH_START, 6)
    run = tournament.run_config(
        Config(entry="ict", exit="fixed_r_2", session="none", instrument=BTC),
        ReplaySource(store),
        entry_factory=_factory(),
        session_factory=session_factory,
        start=later,
        end=end,
    )
    assert run.trades
    assert min(trade.entry_fill.ts for trade in run.trades) >= later


def test_run_config_without_signals_returns_no_trades(store) -> None:
    run = tournament.run_config(
        Config(entry="ict", exit="fixed_r_2", session="none", instrument=BTC),
        ReplaySource(store),
        entry_factory=_factory(ict=lambda instrument, rate: Never()),
        session_factory=session_factory,
        start=SYNTH_START,
        end=tournament.add_months(SYNTH_START, SYNTH_MONTHS),
    )
    assert run.trades == ()
    assert run.equity_curve == ()
    assert run.n_bars > 0


def test_run_config_reports_pessimistic_without_subbars() -> None:
    # its own store: dropping the 1H bars is the point, so it must not be shared
    with build_store([BTC], start=SYNTH_START, months=6, seed=7) as bare:
        bare._conn.execute("DELETE FROM bars WHERE tf = '1h'")
        run = tournament.run_config(
            Config(entry="ict", exit="fixed_r_2", session="none", instrument=BTC),
            ReplaySource(bare),
            entry_factory=_factory(),
            session_factory=session_factory,
            start=SYNTH_START,
            end=tournament.add_months(SYNTH_START, 6),
        )
    assert run.resolution_mode == "pessimistic"


def test_run_config_oos_trades_start_at_the_oos_boundary(store) -> None:
    end = tournament.add_months(SYNTH_START, SYNTH_MONTHS)
    run = tournament.run_config(
        Config(entry="ict", exit="fixed_r_2", session="none", instrument=BTC),
        ReplaySource(store),
        entry_factory=_factory(),
        session_factory=session_factory,
        start=SYNTH_START,
        end=end,
    )
    split = tournament.walk_forward_splits(SYNTH_START, end)[0]
    _, oos = tournament.split_trades(run.trades, split)
    assert oos
    assert all(split.oos_start <= t.entry_fill.ts < split.oos_end for t in oos)


def test_run_config_propagates_a_failing_strategy(store) -> None:
    # run_tournament turns this into an `error:` row; run_config itself does not swallow it
    with pytest.raises(RuntimeError, match="synthetic strategy failure"):
        tournament.run_config(
            Config(entry="ict", exit="fixed_r_2", session="none", instrument=BTC),
            ReplaySource(store),
            entry_factory=_factory(ict=lambda instrument, rate: Raising()),
            session_factory=session_factory,
            start=SYNTH_START,
            end=tournament.add_months(SYNTH_START, SYNTH_MONTHS),
        )


def test_run_config_reports_mixed_when_only_some_bars_have_subbars() -> None:
    with build_store([BTC], start=SYNTH_START, months=6, seed=7) as partial:
        cutoff = tournament.add_months(SYNTH_START, 3).replace(tzinfo=None)
        partial._conn.execute("DELETE FROM bars WHERE tf = '1h' AND ts_open >= ?", [cutoff])
        run = tournament.run_config(
            Config(entry="ict", exit="fixed_r_2", session="none", instrument=BTC),
            ReplaySource(partial),
            entry_factory=_factory(),
            session_factory=session_factory,
            start=SYNTH_START,
            end=tournament.add_months(SYNTH_START, 6),
        )
    assert run.resolution_mode == "mixed"


def test_run_config_hands_the_factory_a_per_config_derived_seed(store) -> None:
    end = tournament.add_months(SYNTH_START, SYNTH_MONTHS)
    first = Config(entry="ict", exit="fixed_r_2", session="none", instrument=BTC)
    second = Config(entry="ict", exit="fixed_r_3", session="none", instrument=BTC)
    factory = _factory(ict=lambda instrument, rate: Never())
    for config in (first, second):
        tournament.run_config(
            config,
            ReplaySource(store),
            entry_factory=factory,
            session_factory=session_factory,
            start=SYNTH_START,
            end=end,
            seed=5,
        )
    # derived from the config id, so it does not move when the matrix is reordered
    assert factory.seeds == [
        ("ict", zlib.crc32(first.id.encode()) ^ 5),
        ("ict", zlib.crc32(second.id.encode()) ^ 5),
    ]
    assert factory.seeds[0][1] != factory.seeds[1][1]


def test_run_config_defaults_the_seed_to_the_config_alone(store) -> None:
    config = Config(entry="ict", exit="fixed_r_2", session="none", instrument=BTC)
    factory = _factory(ict=lambda instrument, rate: Never())
    tournament.run_config(
        config,
        ReplaySource(store),
        entry_factory=factory,
        session_factory=session_factory,
        start=SYNTH_START,
        end=tournament.add_months(SYNTH_START, SYNTH_MONTHS),
    )
    assert factory.seeds == [("ict", zlib.crc32(config.id.encode()))]


# --- the whole tournament --------------------------------------------------------


SOL = BTC.model_copy(update={"symbol": "SOL"})
TEST_EXITS = ("fixed_r_2", "fixed_r_3")
TEST_SESSIONS = ("none",)
RUN_ID = "test-run"


def _refuse(instrument: Instrument):
    raise LookupError(f"no zones strategy for {instrument.symbol}")


@pytest.fixture(scope="module")
def tournament_run():
    """One tournament over BTC + ETH (20 months) and a SOL too short to qualify.

    Module-scoped: the sweep is the expensive part of this file, so every assertion below
    reads the same run rather than repeating it.
    """
    store = build_store([BTC, ETH], start=SYNTH_START, months=SYNTH_MONTHS, seed=7)
    add_instrument(store, SOL, start=SYNTH_START, months=6, seed=99)
    factory = _factory(
        # the entry factory refuses ETH: those configs error, the rest of the run goes on.
        zones=lambda instrument, rate: (
            _refuse(instrument) if instrument.symbol == "ETH" else EveryN(ZONES_EVERY, name="zones")
        )
    )
    seen: list[str] = []
    result = tournament.run_tournament(
        store,
        [BTC, ETH, SOL],
        entry_factory=factory,
        session_factory=session_factory,
        exits=TEST_EXITS,
        sessions=TEST_SESSIONS,
        run_id=RUN_ID,
        progress=seen.append,
    )
    yield store, result, factory, seen
    store.close()


@pytest.mark.slow
def test_run_tournament_visits_every_config_once(tournament_run) -> None:
    _, _, _, seen = tournament_run
    # 3 entries x 2 exits x 1 session x 2 qualifying instruments
    assert len(seen) == 3 * len(TEST_EXITS) * len(TEST_SESSIONS) * 2
    assert len(set(seen)) == len(seen)
    assert "ict|fixed_r_2|none|hyperliquid:BTC" in seen


@pytest.mark.slow
def test_run_tournament_excludes_the_short_instrument(tournament_run) -> None:
    _, result, _, _ = tournament_run
    assert result.excluded == ((SOL, "insufficient_history:6"),)
    assert not any(row["symbol"] == "SOL" for row in result.rows)


@pytest.mark.slow
def test_run_tournament_row_count_is_splits_plus_pooled_plus_is_selected(tournament_run) -> None:
    _, result, _, seen = tournament_run
    end = tournament.add_months(SYNTH_START, SYNTH_MONTHS)
    splits = len(tournament.walk_forward_splits(SYNTH_START, end))
    assert splits == 2
    errored = [row for row in result.rows if row["excluded_reason"] is not None]
    assert len(errored) == 2  # ETH x zones x 2 exits
    assert all(row["excluded_reason"].startswith("error:LookupError") for row in errored)

    ok = len(seen) - len(errored)
    is_selected = [row for row in result.rows if row["exit"] == "IS_SELECTED"]
    assert len(result.rows) == ok * (splits + 1) + len(errored) + len(is_selected)


@pytest.mark.slow
def test_run_tournament_baseline_rate_matches_the_ict_rate(tournament_run) -> None:
    store, _, factory, _ = tournament_run
    for instrument in (BTC, ETH):
        key = instrument_key(instrument)
        # start/end default to the instrument's own range, so every stored 4H bar is in window
        n_bars = store.bar_range(instrument, "4h")[2]
        rates = [
            len(store.trades(f"{RUN_ID}:ict|{exit_name}|none|{key}")) * 1000.0 / n_bars
            for exit_name in TEST_EXITS
        ]
        assert factory.baseline_rates[key] == pytest.approx(sum(rates) / len(rates))
        assert factory.baseline_rates[key] > 0.0


@pytest.mark.slow
def test_run_tournament_keeps_no_runs_unless_asked(tournament_run) -> None:
    # C1: holding 2,016 `ConfigRun`s (each with the instrument's whole close series) is what
    # made the sweep unaffordable; the report needs `rows`/`gates`/`oos_trades` only.
    _, result, _, _ = tournament_run
    assert result.runs == {}
    assert result.oos_trades


@pytest.mark.slow
def test_run_tournament_gates_every_config_that_ran(tournament_run) -> None:
    _, result, _, seen = tournament_run
    errored = {row["config_id"] for row in result.rows if row["excluded_reason"] is not None}
    for config_id in seen:
        assert (config_id in result.gates) is (config_id not in errored), config_id
    pooled = [row for row in result.rows if row["split"] == "pooled" and row["excluded_reason"] is None]
    assert len(pooled) == len(result.gates)
    for row in pooled:
        gate_result = result.gates[row["config_id"]]
        assert row["g1"] is gate_result.rule1
        assert row["passed"] is gate_result.passed
        assert row["n_oos"] == gate_result.n


@pytest.mark.slow
def test_run_tournament_counts_the_selection_views_as_trials(tournament_run) -> None:
    _, result, _, seen = tournament_run
    views = {row["config_id"] for row in result.rows if row["exit"] == "IS_SELECTED"}
    assert result.n_trials == len(seen) + len(views)
    for row in result.rows:
        if row["split"] == "pooled" and row["config_id"] in result.gates:
            assert result.gates[row["config_id"]].dsr is None or (
                result.gates[row["config_id"]].dsr.n_trials == result.n_trials
            )


@pytest.mark.slow
def test_run_tournament_trial_variance_is_taken_over_the_selectable_trials(tournament_run) -> None:
    _, result, _, _ = tournament_run
    sharpes = [
        gate.sharpe(tournament.realized_rs(trades))
        for config_id, trades in result.oos_trades.items()
        if "|IS_SELECTED|" not in config_id and len(trades) >= tournament.V_MIN_TRADES
    ]
    if len(sharpes) >= 2:
        assert result.trial_sr_variance == pytest.approx(float(np.var(sharpes, ddof=1)))
    else:
        assert result.trial_sr_variance is None


def test_trial_variance_ignores_trials_below_the_rule_1_floor() -> None:
    t0 = datetime(2024, 1, 1, tzinfo=UTC)
    long_a = [
        _trade(t0 + timedelta(hours=4 * i), r) for i, r in enumerate([2.0, -1.0, 1.5, -1.0, 0.5, 2.0] * 10)
    ]
    long_b = [
        _trade(t0 + timedelta(hours=4 * i), r) for i, r in enumerate([-1.0, 2.0, -1.0, 3.0, -1.0, 0.0] * 10)
    ]
    # Four near-identical stop-outs: |Sharpe| in the hundreds, pure estimator noise.
    tiny = [_trade(t0 + timedelta(hours=4 * i), r) for i, r in enumerate([-1.0, -1.01, -1.0, -1.01])]

    expected = float(
        np.var(
            [gate.sharpe(tournament.realized_rs(long_a)), gate.sharpe(tournament.realized_rs(long_b))], ddof=1
        )
    )
    assert tournament.trial_variance([long_a, long_b, tiny]) == pytest.approx(expected)
    assert tournament.trial_variance([long_a, long_b]) == pytest.approx(expected)
    # Below two qualifying trials there is no cross-trial variance to speak of.
    assert tournament.trial_variance([long_a, tiny]) is None
    assert tournament.trial_variance([tiny, tiny]) is None
    # The floor is configurable, and rule 1's default is what the tournament uses.
    assert tournament.V_MIN_TRADES == 60
    assert tournament.trial_variance([tiny, tiny], min_trades=2) is not None


@pytest.mark.slow
def test_run_tournament_is_selected_rows_carry_a_selection(tournament_run) -> None:
    _, result, _, _ = tournament_run
    assert result.selection
    for config_id, chosen in result.selection.items():
        assert "|IS_SELECTED|" in config_id
        assert set(chosen) == {"2022-01", "2022-04"}
        assert set(chosen.values()) <= set(TEST_EXITS)
    assert "zones|IS_SELECTED|none|hyperliquid:ETH" not in result.selection


@pytest.mark.slow
def test_run_tournament_records_the_resolution_mode(tournament_run) -> None:
    _, result, _, _ = tournament_run
    assert result.resolution_modes == {"hyperliquid:BTC": "subbars", "hyperliquid:ETH": "subbars"}


@pytest.mark.slow
def test_run_tournament_persists_rows_trades_and_equity(tournament_run) -> None:
    store, result, _, _ = tournament_run
    stored = store.results(RUN_ID)
    assert len(stored) == len(result.rows)
    config_id = "ict|fixed_r_2|none|hyperliquid:BTC"
    assert store.trades(f"{RUN_ID}:{config_id}")
    assert store.equity(f"{RUN_ID}:{config_id}")
    assert len({row["ts"] for row in result.rows}) == 1  # one stamp for the whole run


@pytest.mark.slow
def test_run_tournament_split_rows_leave_the_gate_columns_empty(tournament_run) -> None:
    _, result, _, _ = tournament_run
    per_split = [row for row in result.rows if row["split"] != "pooled"]
    assert per_split
    for row in per_split:
        assert row["dsr_prob"] is None and row["passed"] is None
        assert row["n_is"] is not None and row["n_oos"] is not None


def test_run_tournament_rejects_an_unknown_exit_name() -> None:
    with build_store([BTC], start=SYNTH_START, months=6, seed=1) as store, pytest.raises(KeyError):
        tournament.run_tournament(
            store,
            [BTC],
            entry_factory=_factory(),
            session_factory=session_factory,
            exits=("nope",),
        )


def test_run_tournament_excludes_an_instrument_with_no_bars() -> None:
    with build_store([BTC], start=SYNTH_START, months=6, seed=1) as store:
        result = tournament.run_tournament(
            store,
            [SOL],
            entry_factory=_factory(),
            session_factory=session_factory,
            exits=("fixed_r_2",),
            sessions=("none",),
        )
    assert result.excluded == ((SOL, "insufficient_history:0"),)
    assert result.rows == ()
    assert result.n_trials == 0
    assert result.trial_sr_variance is None


def test_run_tournament_intersects_the_window_with_the_actual_history() -> None:
    # a four-year request against six months of bars measures six months, not four years
    with build_store([SOL], start=SYNTH_START, months=6, seed=1) as store:
        result = tournament.run_tournament(
            store,
            [SOL],
            entry_factory=_factory(),
            session_factory=session_factory,
            exits=("fixed_r_2",),
            sessions=("none",),
            start=datetime(2019, 1, 1, tzinfo=UTC),
            end=datetime(2023, 1, 1, tzinfo=UTC),
        )
    assert result.excluded == ((SOL, "insufficient_history:6"),)
    assert result.rows == ()


def test_run_tournament_aborts_after_too_many_consecutive_errors() -> None:
    with (
        build_store([BTC], start=SYNTH_START, months=6, seed=1) as store,
        pytest.raises(RuntimeError, match="consecutive") as caught,
    ):
        tournament.run_tournament(
            store,
            [BTC],
            entry_factory=_factory(ict=lambda instrument, rate: Raising()),
            session_factory=session_factory,
            entries=("ict",),
            sessions=("none", "london_ny"),  # 16 exits x 2 sessions, well past the limit
            min_months=1,
        )
    assert isinstance(caught.value.__cause__, RuntimeError)
    assert "synthetic strategy failure" in str(caught.value.__cause__)


def test_warm_context_does_not_change_the_run() -> None:
    end = tournament.add_months(SYNTH_START, 12)
    config = Config(entry="ict", exit="trail_1_2", session="none", instrument=BTC)
    with build_store([BTC], start=SYNTH_START, months=12, seed=3) as store:
        source = ReplaySource(store)
        kwargs: dict = {
            "entry_factory": _factory(),
            "session_factory": session_factory,
            "start": SYNTH_START,
            "end": end,
        }
        warm = tournament.run_config(config, source, warm_context=True, **kwargs)
        cold = tournament.run_config(config, source, warm_context=False, **kwargs)
    assert warm.trades
    assert warm == cold  # trades, equity curve, bar count and resolution mode alike


# --- resuming a run, and a store that refuses to take it -------------------------

RESUME_MONTHS = 16
RESUME_MIN_MONTHS = 15  # 16 months is one 12/3 split, which is all these tests need
RESUME_CONFIGS = tuple(f"{entry}|fixed_r_2|none|hyperliquid:BTC" for entry in tournament.ENTRIES)


def _without_ts(rows: Sequence[dict]) -> list[dict]:
    """Rows minus the wall-clock stamp, which is the only thing two runs may disagree on."""
    return [{key: value for key, value in row.items() if key != "ts"} for row in rows]


def _small_run_kwargs(factory) -> dict:
    return {
        "entry_factory": factory,
        "session_factory": session_factory,
        "exits": ("fixed_r_2",),
        "sessions": ("none",),
        "min_months": RESUME_MIN_MONTHS,
    }


@pytest.mark.slow
def test_resume_reuses_stored_trades_instead_of_replaying(monkeypatch) -> None:
    kwargs = _small_run_kwargs(_factory()) | {"run_id": "resume-run"}
    with build_store([BTC], start=SYNTH_START, months=RESUME_MONTHS, seed=7) as store:
        first = tournament.run_tournament(store, [BTC], **kwargs)
        assert all(store.trades(f"resume-run:{config_id}") for config_id in RESUME_CONFIGS)

        replayed: list[str] = []
        original = tournament.run_config

        def counting(config, *args, **inner):
            replayed.append(config.id)
            return original(config, *args, **inner)

        monkeypatch.setattr(tournament, "run_config", counting)
        second = tournament.run_tournament(store, [BTC], resume=True, keep_runs=True, **kwargs)

    assert replayed == []
    assert _without_ts(second.rows) == _without_ts(first.rows)
    assert set(second.runs) == set(RESUME_CONFIGS)
    assert all(run.trades and run.equity_curve for run in second.runs.values())


@pytest.mark.slow
def test_a_failed_store_write_annotates_the_row_but_keeps_the_gate(monkeypatch) -> None:
    config_id = f"ict|fixed_r_2|none|{instrument_key(BTC)}"

    def refuse(run_id: str, trades) -> None:
        raise OSError("disk on fire")

    with build_store([BTC], start=SYNTH_START, months=RESUME_MONTHS, seed=7) as store:
        monkeypatch.setattr(store, "write_trades", refuse)
        result = tournament.run_tournament(
            store,
            [BTC],
            entries=("ict",),
            run_id="persist-run",
            **_small_run_kwargs(_factory()),
        )
        assert store.equity(f"persist-run:{config_id}") == []  # the write never got that far

    row = next(r for r in result.rows if r["config_id"] == config_id and r["split"] == "pooled")
    assert row["excluded_reason"].startswith("persist_error:OSError")
    assert row["passed"] is not None  # the config still ran, and was still graded
    assert config_id in result.gates
    assert result.oos_trades[config_id]


def test_finite_or_none_maps_inf_and_nan() -> None:
    assert tournament._finite_or_none(None) is None
    assert tournament._finite_or_none(float("nan")) is None
    assert tournament._finite_or_none(float("inf")) == 1e9
    assert tournament._finite_or_none(float("-inf")) == -1e9
    assert tournament._finite_or_none(1.25) == 1.25


# --- the pooled row and its gate, without an engine in the way ---------------------

_TS = datetime(2024, 5, 4, 3, 2, 1, tzinfo=UTC)


def _series(rs: Sequence[float], first: datetime) -> list[Trade]:
    return [_trade(first + timedelta(hours=4 * index), r) for index, r in enumerate(rs)]


def _oos_fixture() -> tuple[list[Split], list[tuple[datetime, float]]]:
    splits = tournament.walk_forward_splits(SYNTH_START, tournament.add_months(SYNTH_START, 20))
    closes = [(splits[0].oos_start + timedelta(hours=4 * i), 100.0 * (1.0 + 0.001 * i)) for i in range(200)]
    return splits, closes


def test_pooled_row_carries_every_gate_column() -> None:
    splits, closes = _oos_fixture()
    rng = np.random.default_rng(1)
    oos = _series(list(rng.normal(0.35, 1.0, 90)), splits[0].oos_start)
    baseline = _series(list(rng.normal(0.0, 1.0, 90)), splits[0].oos_start)
    gates: dict[str, object] = {}
    row = tournament._pooled_row(
        run_id="r",
        ts=_TS,
        config_id="c",
        identity={
            "entry": "ict",
            "exit": "fixed_r_2",
            "session": "none",
            "venue": "hyperliquid",
            "symbol": "BTC",
        },
        is_trades=oos[:10],
        oos=oos,
        baseline=baseline,
        closes=closes,
        splits=splits,
        resolution_mode="subbars",
        n_trials=12,
        seed=0,
        trial_sr_variance=0.05,
        gates=gates,  # type: ignore[arg-type]
    )
    assert row["split"] == "pooled"
    assert row["ts"] == _TS  # the run's own stamp, not one per row
    assert row["excluded_reason"] is None
    assert row["n_oos"] == 90 and row["exp_is"] is not None
    assert gates["c"].rule1 is True  # type: ignore[attr-defined]
    for column in ("dsr_prob", "boot_p5", "diff_p5", "mar", "mar_bh", "g1", "g2", "g3", "g4", "g5", "g6"):
        assert row[column] is not None, column
    assert isinstance(row["passed"], bool)


def test_pooled_row_maps_an_infinite_mar_to_the_sentinel() -> None:
    splits, closes = _oos_fixture()
    # every trade a winner: the compounded curve never draws down, so MAR is infinite.
    oos = _series([1.0] * 70, splits[0].oos_start)
    gates: dict[str, object] = {}
    row = tournament._pooled_row(
        run_id="r",
        ts=_TS,
        config_id="c",
        identity={
            "entry": "ict",
            "exit": "fixed_r_2",
            "session": "none",
            "venue": "hyperliquid",
            "symbol": "BTC",
        },
        is_trades=[],
        oos=oos,
        baseline=oos,
        closes=closes,
        splits=splits,
        resolution_mode="subbars",
        n_trials=12,
        seed=0,
        trial_sr_variance=0.05,
        gates=gates,  # type: ignore[arg-type]
    )
    assert row["mar"] == 1e9
    assert row["exp_is"] is None and row["n_is"] == 0


def test_pooled_row_turns_a_ruinous_config_into_an_excluded_row() -> None:
    splits, closes = _oos_fixture()
    oos = _series([1.0] * 60 + [-200.0], splits[0].oos_start)
    gates: dict[str, object] = {}
    row = tournament._pooled_row(
        run_id="r",
        ts=_TS,
        config_id="c",
        identity={
            "entry": "ict",
            "exit": "fixed_r_2",
            "session": "none",
            "venue": "hyperliquid",
            "symbol": "BTC",
        },
        is_trades=[],
        oos=oos,
        baseline=[],
        closes=closes,
        splits=splits,
        resolution_mode="subbars",
        n_trials=12,
        seed=0,
        trial_sr_variance=None,
        gates=gates,  # type: ignore[arg-type]
    )
    assert row["excluded_reason"].startswith("error:ValueError")
    assert row["passed"] is None
    assert gates == {}


# --- guards and internals ---------------------------------------------------------


def test_walk_forward_splits_rejects_a_non_positive_window() -> None:
    with pytest.raises(ValueError, match=">= 1"):
        tournament.walk_forward_splits(
            datetime(2020, 1, 1, tzinfo=UTC), datetime(2024, 1, 1, tzinfo=UTC), is_months=0
        )


def test_buy_and_hold_curve_rejects_a_non_positive_close() -> None:
    base = datetime(2021, 1, 1, tzinfo=UTC)
    closes = [(base, 0.0), (base + timedelta(hours=4), 100.0)]
    with pytest.raises(ValueError, match="not positive"):
        tournament.buy_and_hold_curve(closes, [(base, base + timedelta(days=1))])


def test_select_exits_skips_a_split_without_enough_is_trades() -> None:
    splits = tournament.walk_forward_splits(
        datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 4, 1, tzinfo=UTC)
    )
    config = Config(entry="ict", exit="fixed_r_2", session="none", instrument=BTC)
    thin = tournament._Replayed(
        trades=(_trade(datetime(2020, 6, 1, tzinfo=UTC), 1.0),),  # one IS trade, floor is ten
        n_bars=10,
        resolution_mode="subbars",
    )
    view = tournament._select_exits(
        {config.id: thin}, splits, "ict", "none", "hyperliquid:BTC", ("fixed_r_2",)
    )
    assert view.chosen == {}
    assert view.is_trades == [] and view.oos_trades == []


def test_is_selected_views_skips_an_instrument_with_no_splits() -> None:
    selection, views = tournament._is_selected_views(
        {},
        {"hyperliquid:BTC": []},
        ("ict",),
        ("fixed_r_2",),
        ("none",),
        {BTC: (SYNTH_START, SYNTH_START)},
    )
    assert selection == {} and views == {}
