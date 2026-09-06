"""A replay is reproducible: same store, same config, same trades — down to the stored bytes.

This is the property everything else in the suite leans on. `run_config` builds a fresh
strategy, broker and portfolio per call, so two runs may only agree if nothing in the path
carries hidden state: the memoised `Context` indicator cache, the `_MemoisedSource` bar cache,
the seeded entry factory (`crc32(config.id) ^ seed`), and the broker's insertion-ordered
pending book all have to land in the same place twice.
"""

from __future__ import annotations

import pytest

from swingforge.adapters.replay import ReplaySource
from swingforge.adapters.store import Store
from swingforge.lab.tournament import Config, ConfigRun, run_config
from tests.integration.conftest import WALK, WALK_END
from tests.integration.synth import SYNTH_START, real_entry_factory, real_session_factory
from tests.unit.synth_store import EntryFactoryStub, EveryN, session_factory

_EVERY_N = 40
_RUN_A = "determinism:a"
_RUN_B = "determinism:b"

CONFIGS = {
    "every_n": Config(entry="every_n", exit="fixed_r_2", session="none", instrument=WALK),
    "ict": Config(entry="ict", exit="fixed_r_2", session="none", instrument=WALK),
}


def _replay(store: Store, name: str) -> ConfigRun:
    """One `run_config` of `CONFIGS[name]`, through a `ReplaySource` built fresh each call."""
    if name == "every_n":
        factory = EntryFactoryStub({"every_n": lambda instrument, rate: EveryN(_EVERY_N)})
        return run_config(
            CONFIGS[name],
            ReplaySource(store),
            entry_factory=factory,
            session_factory=session_factory,
            start=SYNTH_START,
            end=WALK_END,
        )
    return run_config(
        CONFIGS[name],
        ReplaySource(store),
        entry_factory=real_entry_factory,
        session_factory=real_session_factory,
        start=SYNTH_START,
        end=WALK_END,
    )


@pytest.mark.parametrize("name", list(CONFIGS))
def test_two_replays_of_one_config_are_identical(walk_store: Store, name: str) -> None:
    first = _replay(walk_store, name)
    second = _replay(walk_store, name)

    assert first.trades, f"{name} produced no trades: the comparison would be vacuous"
    assert first == second


@pytest.mark.parametrize("name", list(CONFIGS))
def test_written_trades_are_identical_under_two_run_ids(walk_store: Store, name: str) -> None:
    """The rows two replays persist differ only in `run_id`, in both directions.

    `EXCEPT ALL` is a multiset difference, so an empty result each way says the two runs
    wrote the same rows the same number of times each -- plain `EXCEPT` dedupes first and
    would call two runs identical even if one wrote a row once and the other wrote it twice,
    which is exactly the kind of non-determinism this test exists to catch. The row counts
    are kept alongside it as a cheap, independent pin on the multiplicities. Every column is
    compared, `context_snapshot` and `legs_json` included.
    """
    first = _replay(walk_store, name)
    second = _replay(walk_store, name)

    with Store(":memory:") as store:
        store.upsert_instruments([WALK])
        store.write_trades(_RUN_A, first.trades)
        store.write_trades(_RUN_B, second.trades)

        counts = store._conn.execute(
            "SELECT run_id, count(*) FROM trades GROUP BY run_id ORDER BY run_id"
        ).fetchall()
        assert counts == [(_RUN_A, len(first.trades)), (_RUN_B, len(second.trades))]

        for left, right in ((_RUN_A, _RUN_B), (_RUN_B, _RUN_A)):
            missing = store._conn.execute(
                "SELECT * EXCLUDE (run_id) FROM trades WHERE run_id = ? "
                "EXCEPT ALL SELECT * EXCLUDE (run_id) FROM trades WHERE run_id = ?",
                [left, right],
            ).fetchall()
            assert missing == [], f"{left} holds rows {right} does not, at the same multiplicity"
