"""The end-to-end claim: the tournament finds a planted edge, and only a planted edge.

Two stores hold the *same* ICT setups at the same bars (`tests/integration/synth.py`). On one
the trade drifts in its own direction after the entry; on the other it hands straight back to
the random walk. Nothing else differs — the self-test below pins that `ICT` emits on the same
bars in both, so the two runs are the same search over the same entry frequency, and only the
outcomes are different.

What must then hold:

* on the planted store at least one `ict` config clears all six gate rules, on enough
  out-of-sample trades for rule 1 to be more than a formality, and **no** `baseline` config
  clears them — the frequency-matched random control must not be able to fake the result;
* on the control store **nothing** clears them, `ict` included, even though rule 1 is still
  satisfied — so the failure is the statistics talking, not an empty sample;
* on a **plain random walk with nothing planted at all** (spec section 8), nothing clears
  them either — for `zones` and the frequency-matched `baseline` as well as `ict`, not just
  the pair the planted comparison above already covers;
* on **five identical copies of that walk** nothing clears them either, the universe trials
  included — pooling instruments that move as one hands the gate the same trades five times,
  and that must buy no significance.

The gate figures behind both verdicts are printed (visible under `pytest -s`, and repeated in
every failure message) so a human can read what the run actually decided.

Slow: four full tournaments, about two minutes end to end. Run with
`uv run pytest -q -m slow tests/integration`.
"""

from __future__ import annotations

import math
from datetime import timedelta

import numpy as np
import pytest

from swingforge.adapters.replay import ReplaySource
from swingforge.adapters.store import Store
from swingforge.core.context import Context
from swingforge.core.costs import NullCostModel
from swingforge.core.types import Signal
from swingforge.lab.gate import deflated_sharpe
from swingforge.lab.tournament import TournamentResult, entry_days, realized_rs, run_tournament
from swingforge.strategies.ict import ICT
from swingforge.strategies.levels import swing_highs, swing_lows
from tests.integration.conftest import PLANTED_SEED, PLANTED_YEARS, WALK, WALK_SEED
from tests.integration.synth import (
    PLANTED,
    SYNTH_START,
    planted_setups,
    random_walk_store,
    real_entry_factory,
    real_session_factory,
)
from tests.unit.synth_store import add_instrument

pytestmark = pytest.mark.slow

ENTRIES = ("ict", "baseline")
EXITS = ("fixed_r_2", "fixed_r_3", "trail_1_2")
SESSIONS = ("none",)
SEED = 0

MIN_OOS_TRADES = 60
"""`gate.evaluate`'s own rule-1 floor: below it rules 2-6 are `None` and the run says nothing."""
MIN_HIT_RATE = 0.8
"""Share of planted setups `ICT` must actually signal on for the plant to mean anything."""

_FOUR_HOURS = timedelta(hours=4)


# --- running the two tournaments -----------------------------------------------------


def _tournament(store: Store) -> TournamentResult:
    """`ict` and its frequency-matched `baseline` over three exits, costs switched off.

    `NullCostModel` keeps the question to "is there an edge in the price path" rather than
    "does it survive this venue's spread"; it also makes gate rule 6 (the cost stress) a
    restatement of rules 1-5, since stressing a zero cost leaves every R multiple where it was.
    """
    return run_tournament(
        store,
        [PLANTED],
        entry_factory=real_entry_factory,
        session_factory=real_session_factory,
        cost_model=NullCostModel(),
        entries=ENTRIES,
        exits=EXITS,
        sessions=SESSIONS,
        seed=SEED,
    )


@pytest.fixture(scope="module")
def edge_result(planted_edge_store: Store) -> TournamentResult:
    return _tournament(planted_edge_store)


@pytest.fixture(scope="module")
def flat_result(planted_flat_store: Store) -> TournamentResult:
    return _tournament(planted_flat_store)


def _pooled(result: TournamentResult, entry: str | None = None) -> list[dict]:
    return [
        row for row in result.rows if row["split"] == "pooled" and (entry is None or row["entry"] == entry)
    ]


def _report(result: TournamentResult, label: str) -> str:
    """The gate, config by config, in one readable block."""
    lines = [
        f"{label}: n_trials={result.n_trials} trial_sr_variance={result.trial_sr_variance}",
        f"{'config':<46} {'n_oos':>5} {'exp_oos':>8} {'dsr':>7} {'boot_p5':>8} "
        f"{'diff_p5':>8} {'mar':>10} {'mar_bh':>8}  rules",
    ]
    for row in sorted(_pooled(result), key=lambda item: item["config_id"]):
        gate = result.gates.get(row["config_id"])
        dsr = None if gate is None or gate.dsr is None else round(gate.dsr.prob, 4)
        rules = "-" if gate is None else "".join(_flag(value) for value in gate.rules().values())
        lines.append(
            f"{row['config_id']:<46} {row['n_oos']:>5} {_num(row['exp_oos']):>8} {_num(dsr):>7} "
            f"{_num(row['boot_p5']):>8} {_num(row['diff_p5']):>8} {_num(row['mar']):>10} "
            f"{_num(row['mar_bh']):>8}  {rules} passed={row['passed']}"
        )
    return "\n".join(lines)


def _flag(value: bool | None) -> str:
    return "." if value is None else ("Y" if value else "n")


def _num(value: float | None) -> str:
    return "-" if value is None else f"{value:.4g}"


# --- the self-test: are the setups actually there? -----------------------------------


def _ict_signals(store: Store) -> dict[int, Signal]:
    """Every `Signal` a fresh `ICT` emits over the store, keyed by 4H bar index.

    Driven straight through `Context` rather than the engine, so a signal is counted whether or
    not a trade happened to be open at the time — this measures the entry *frequency*, which is
    what has to match between the two stores.
    """
    window = store.bar_range(PLANTED, "4h")
    assert window is not None
    first, last, _ = window
    bars = ReplaySource(store).merged(PLANTED, first, last + _FOUR_HOURS)

    context = Context(PLANTED)
    strategy = ICT()
    signals: dict[int, Signal] = {}
    for bar in bars:
        for sub in bar.subbars:
            context.push(sub)
        context.push(bar)
        if bar.tf != "4h":
            continue
        signal = strategy.on_bar(context)
        if signal is not None:
            signals[context.bar_index] = signal
    return signals


@pytest.fixture(scope="module")
def edge_signals(planted_edge_store: Store) -> dict[int, Signal]:
    return _ict_signals(planted_edge_store)


@pytest.fixture(scope="module")
def flat_signals(planted_flat_store: Store) -> dict[int, Signal]:
    return _ict_signals(planted_flat_store)


def _daily_ohlcv(store: Store) -> np.ndarray:
    """Every Daily bar of `PLANTED`, as the `(n, 5)` array `swingforge.strategies.levels`
    reads: open, high, low, close, volume, oldest first."""
    window = store.bar_range(PLANTED, "1d")
    assert window is not None
    first, last, _ = window
    context = Context(PLANTED)
    for bar in ReplaySource(store).history(PLANTED, "1d", first, last + timedelta(hours=24)):
        context.push(bar)
    return context.bars("1d").copy()


@pytest.mark.parametrize(("fixture", "edge"), [("edge_signals", True), ("flat_signals", False)])
def test_ict_fires_on_the_planted_setups(request: pytest.FixtureRequest, fixture: str, edge: bool) -> None:
    """The plant is real: `ICT` signals on the break-of-structure bar, the right way, at the
    order-block midpoint. If this fails, `tests/integration/synth.py` needs regenerating
    against whatever `strategies/ict.py` now does."""
    signals: dict[int, Signal] = request.getfixturevalue(fixture)
    setups = planted_setups(PLANTED, years=PLANTED_YEARS, seed=PLANTED_SEED, edge=edge)

    assert setups
    hits = [setup for setup in setups if setup.bos_index in signals]
    assert len(hits) >= MIN_HIT_RATE * len(setups), (
        f"ICT signalled on {len(hits)} of {len(setups)} planted setups (edge={edge})"
    )
    for setup in hits:
        signal = signals[setup.bos_index]
        assert signal.direction == setup.direction
        assert signal.entry == pytest.approx(setup.entry)
        assert signal.stop == pytest.approx(setup.stop, abs=float(PLANTED.tick_size))

    # `_Levels` (synth.py's own incremental fractal cache) is meant to track exactly what
    # `strategies/levels.py` would report, not merely something ICT is willing to trade off
    # -- cross-check a handful of setups' swept levels against the production functions
    # directly, over the same store's Daily bars.
    store: Store = request.getfixturevalue("planted_edge_store" if edge else "planted_flat_store")
    daily = _daily_ohlcv(store)
    low_levels = [level for _, level in swing_lows(daily, k=2)]
    high_levels = [level for _, level in swing_highs(daily, k=2)]
    for setup in setups:
        official = low_levels if setup.direction == 1 else high_levels
        assert any(math.isclose(setup.level, level, rel_tol=1e-9) for level in official), (
            f"setup at slot {setup.slot} swept {setup.level}, which is not a swing "
            f"{'low' if setup.direction == 1 else 'high'} per strategies/levels.py"
        )


def test_both_stores_offer_ict_the_same_entries(
    edge_signals: dict[int, Signal], flat_signals: dict[int, Signal]
) -> None:
    """The two runs differ in outcome, not in opportunity.

    Both stores share one spine — every setup window ends at exactly the price the random walk
    would have reached — so the drift only ever perturbs the Daily extremes inside its own
    window. That moves a handful of fractal pivots and so a handful of setups, which is why
    this is a tolerance rather than an equality.
    """
    assert len(edge_signals) == pytest.approx(len(flat_signals), rel=0.05)


# --- the gate ------------------------------------------------------------------------


def test_a_planted_ict_config_passes_every_gate_rule(edge_result: TournamentResult) -> None:
    report = _report(edge_result, "planted edge=True")
    print(report)

    passing = [row for row in _pooled(edge_result, "ict") if row["passed"] is True]
    assert passing, f"no ict config cleared the gate on the planted store\n{report}"

    best = max(passing, key=lambda row: row["exp_oos"])
    gate = edge_result.gates[best["config_id"]]
    assert all(value is True for value in gate.rules().values()), f"{gate.rules()}\n{report}"
    assert best["n_oos"] >= MIN_OOS_TRADES, (
        f"{best['config_id']} passed on only {best['n_oos']} OOS trades\n{report}"
    )


def test_no_baseline_config_passes_on_the_planted_store(edge_result: TournamentResult) -> None:
    """The random control has the same trade frequency and the same exits; it must still fail."""
    passing = [row["config_id"] for row in _pooled(edge_result, "baseline") if row["passed"] is True]
    assert passing == [], _report(edge_result, "planted edge=True")


def test_nothing_passes_without_the_planted_edge(flat_result: TournamentResult) -> None:
    report = _report(flat_result, "planted edge=False")
    print(report)

    passing = [row["config_id"] for row in _pooled(flat_result) if row["passed"] is True]
    assert passing == [], report


def test_the_control_run_fails_on_statistics_not_on_sample_size(flat_result: TournamentResult) -> None:
    """Rule 1 short-circuits the rest of the gate, so a thin control would prove nothing."""
    report = _report(flat_result, "planted edge=False")
    ict_rows = _pooled(flat_result, "ict")
    assert ict_rows, report
    for row in ict_rows:
        assert row["n_oos"] >= MIN_OOS_TRADES, report
        assert row["g1"] is True, report


# --- the pure random-walk tournament (spec section 8) ---------------------------------

RANDOM_WALK_YEARS = 4
"""Same span as the planted stores, so this control costs one more tournament, not a longer
one. Nothing is planted at all here -- not even the flat control's ICT geometry -- so this
is the strongest possible negative: no structure of any kind for `ict`/`zones` to key off."""


@pytest.fixture(scope="module")
def random_walk_result() -> TournamentResult:
    store = random_walk_store([WALK], years=RANDOM_WALK_YEARS, seed=WALK_SEED)
    try:
        return run_tournament(
            store,
            [WALK],
            entry_factory=real_entry_factory,
            session_factory=real_session_factory,
            cost_model=NullCostModel(),
            entries=("ict", "zones", "baseline"),
            exits=EXITS,
            sessions=SESSIONS,
            seed=SEED,
        )
    finally:
        store.close()


def test_nothing_passes_on_a_pure_random_walk(random_walk_result: TournamentResult) -> None:
    """No plant at all: nothing should clear the gate, for `ict` and `zones` as well as the
    frequency-matched `baseline` -- the planted-store tests above already show the search
    can't be fooled by a matched control; this shows it can't be fooled by chance alone.

    An unplanted walk gives `ict` (and `baseline`, matched to its rate) too few OOS trades
    over four years to ever clear rule 1's n>=60 floor -- observed counts are in the report
    below. That floor doing its job is not itself proof the statistics would have rejected
    the walk; the second assertion shows they do, by re-running the deflated-Sharpe rule
    (rule 2) on the very same pooled OOS trades with the floor lifted.
    """
    report = _report(random_walk_result, "pure random walk")
    print(report)

    passing = [row["config_id"] for row in _pooled(random_walk_result) if row["passed"] is True]
    assert passing == [], report

    candidates = [row for row in _pooled(random_walk_result) if row["entry"] in ("ict", "baseline")]
    assert candidates, report
    counts = {row["config_id"]: row["n_oos"] for row in candidates}
    if max(counts.values()) >= MIN_OOS_TRADES:
        return  # the floor was cleared by at least one config; "passing == []" already proved it

    for row in candidates:
        oos = random_walk_result.oos_trades[row["config_id"]]
        if not oos:
            continue
        dsr = deflated_sharpe(
            realized_rs(oos),
            random_walk_result.n_trials,
            trial_sr_variance=random_walk_result.trial_sr_variance,
        )
        assert dsr.prob < 0.95, (
            f"{row['config_id']} would have cleared rule 2 on its {len(oos)} OOS trades once "
            f"the n>={MIN_OOS_TRADES} floor was lifted (dsr.prob={dsr.prob})\n"
            f"observed n_oos per config: {counts}\n{report}"
        )


# --- the cloned random walk: pooling must not manufacture significance ----------------

CLONES = 5
"""Copies of one walk pooled into each universe trial. Identical series are the worst case of
instruments that move together: every `ict` trade exists `CLONES` times, on the same bars."""

CLONE_EXITS = EXITS[:2]


@pytest.fixture(scope="module")
def cloned_walk_result() -> TournamentResult:
    clones = [WALK.model_copy(update={"symbol": f"WALK{index}"}) for index in range(CLONES)]
    store = Store(":memory:")
    try:
        for clone in clones:  # the same seed for every clone: the same bars under five symbols
            add_instrument(store, clone, start=SYNTH_START, months=RANDOM_WALK_YEARS * 12, seed=WALK_SEED)
        return run_tournament(
            store,
            clones,
            entry_factory=real_entry_factory,
            session_factory=real_session_factory,
            cost_model=NullCostModel(),
            entries=ENTRIES,
            exits=CLONE_EXITS,
            sessions=SESSIONS,
            seed=SEED,
        )
    finally:
        store.close()


def test_pooling_clones_of_a_random_walk_passes_nothing(cloned_walk_result: TournamentResult) -> None:
    report = _report(cloned_walk_result, f"{CLONES} clones of a pure random walk")
    print(report)

    passing = [row["config_id"] for row in _pooled(cloned_walk_result) if row["passed"] is True]
    assert passing == [], report


def test_a_pool_of_clones_is_read_at_one_clones_worth_of_evidence(
    cloned_walk_result: TournamentResult,
) -> None:
    """The `ict` universe trials hold every trade `CLONES` times - enough to clear rule 1, which
    one walk alone never does - and rule 2 must read them at the days they were entered on, not
    at the row count: the deflated Sharpe comes out where one clone's own trades put it, and
    reading the rows as independent trades would have put it somewhere else."""
    result = cloned_walk_result
    report = _report(result, f"{CLONES} clones of a pure random walk")
    universe = [row for row in _pooled(result, "ict") if row["symbol"] == "*"]
    assert len(universe) >= len(CLONE_EXITS), report  # one per exit, plus a view if one was selected
    graded = [row for row in universe if row["g1"] is True]
    assert graded, f"no ict universe trial cleared rule 1, so this control proves nothing\n{report}"
    # no single walk reaches rule 1's floor, so V is never measured and the gate falls back to
    # the analytical 1/(n - 1) - at the effective size, for a universe trial
    assert result.trial_sr_variance is None, report

    for row in graded:
        trades = result.oos_trades[row["config_id"]]
        one_clone = result.oos_trades[row["config_id"].replace(":*", ":WALK0")]
        gate = result.gates[row["config_id"]]
        assert gate.dsr is not None, report
        assert len(trades) == CLONES * len(one_clone), report
        assert gate.dsr.effective_n == entry_days(trades) == entry_days(one_clone), report

        alone = deflated_sharpe(realized_rs(one_clone), result.n_trials, effective_n=entry_days(one_clone))
        naive = deflated_sharpe(realized_rs(trades), result.n_trials)
        # equal up to the Sharpe's ddof, which a cloned sample shifts by about 3%
        assert gate.dsr.prob == pytest.approx(alone.prob, abs=0.05), report
        assert gate.dsr.sr_star == pytest.approx(alone.sr_star), report
        assert abs(naive.prob - alone.prob) > abs(gate.dsr.prob - alone.prob), report
        assert gate.rule2 is False, report
