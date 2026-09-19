"""Tests for `swingforge.lab.report`."""

from __future__ import annotations

import math
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from swingforge.core.types import CostBreakdown, Fill, Instrument, Trade
from swingforge.lab import report
from swingforge.lab.excursion import ExcursionSummary
from swingforge.lab.gate import DSRResult, GateResult
from swingforge.lab.tournament import Config, ConfigRun, TournamentResult

SNAPSHOT = Path(__file__).parent / "snapshots" / "report_small.md"

BTC = Instrument(
    venue="hyperliquid",
    symbol="BTC",
    tick_size=Decimal("0.5"),
    contract_multiplier=Decimal("1"),
    quote_ccy="USD",
    session_profile="perp",
)
SOL = BTC.model_copy(update={"symbol": "SOL"})
BASE = datetime(2022, 1, 1, tzinfo=UTC)

WINNER = "ict|fixed_r_2|none|hyperliquid:BTC"
LOSER = "baseline|fixed_r_2|none|hyperliquid:BTC"
BROKEN = "zones|fixed_r_2|none|hyperliquid:BTC"
RUINED = "zones|trail_1_2|none|hyperliquid:BTC"
SELECTED = "ict|IS_SELECTED|none|hyperliquid:BTC"
UNIVERSE = "ict|fixed_r_2|none|hyperliquid:*"
UNIVERSE_THIN = "zones|fixed_r_2|none|hyperliquid:*"
UNIVERSE_VIEW = "ict|IS_SELECTED|none|hyperliquid:*"
UNIVERSE_HEADING = "## Gate, pooled across instruments"


def _trade(index: int, realized_r: float, regime: str, *, spread: float = 0.5) -> Trade:
    entry = Fill(
        order_id=f"o{index}",
        ts=BASE + timedelta(hours=4 * index),
        price=100.0,
        qty=1.0,
        cost=CostBreakdown(spread=spread, slippage=0.25, funding=0.2),
        leg="entry",
    )
    return Trade(
        id=f"t{index}",
        instrument=BTC,
        direction=1,
        entry_fill=entry,
        legs=(entry.model_copy(update={"leg": "stop", "ts": entry.ts + timedelta(hours=8)}),),
        stop=90.0,
        target=None,
        risk_r=100.0,
        realized_r=realized_r,
        mae_r=0.4 + 0.1 * (index % 3),
        mfe_r=1.2 + 0.2 * (index % 4),
        regime=regime,
        opened_bar=index,
        closed_bar=index + 2,
    )


def _gate(*, passed: bool, mar: float) -> GateResult:
    return GateResult(
        n=8,
        baseline_n=8,
        rule1=True,
        rule2=passed,
        rule3=True,
        rule4=passed,
        rule5=True,
        rule6=passed,
        dsr=DSRResult(
            sr=0.31,
            sr_star=0.12,
            prob=0.97 if passed else 0.42,
            n=8,
            n_trials=12,
            skew=0.1,
            kurt=3.2,
        ),
        boot_p5=0.08 if passed else -0.04,
        diff_p5=0.05 if passed else -0.11,
        mar_config=mar,
        mar_bh=0.75,
    )


def _row(config_id: str, split: str, **values: object) -> dict:
    entry, exit_name, session, key = config_id.split("|")
    venue, symbol = key.split(":")
    row: dict = dict.fromkeys(
        (
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
    )
    row.update(
        run_id="small",
        ts=BASE,
        config_id=config_id,
        entry=entry,
        exit=exit_name,
        session=session,
        venue=venue,
        symbol=symbol,
        split=split,
        resolution_mode="subbars",
    )
    row.update(values)
    return row


def _result() -> TournamentResult:
    winner_trades = tuple(
        _trade(i, r, regime)
        for i, (r, regime) in enumerate(
            [
                (2.0, "trend_highvol"),
                (-1.0, "trend_highvol"),
                (1.5, "range_midvol"),
                (-1.0, "range_midvol"),
                (3.0, "trend_lowvol"),
                (-1.0, "range_midvol"),
                (0.5, "trend_highvol"),
                (-1.0, "range_lowvol"),
            ]
        )
    )
    loser_trades = tuple(_trade(i, -0.2, "range_midvol") for i in range(8))
    selected_trades = winner_trades[:4]

    rows = [
        _row(WINNER, "2023-01", n_is=20, n_oos=8, exp_is=0.41, exp_oos=0.5),
        _row(
            WINNER,
            "pooled",
            n_is=20,
            n_oos=8,
            exp_is=0.41,
            exp_oos=0.5,
            dsr_prob=0.97,
            boot_p5=0.08,
            diff_p5=0.05,
            mar=1e9,
            mar_bh=0.75,
            g1=True,
            g2=True,
            g3=True,
            g4=True,
            g5=True,
            g6=True,
            passed=True,
        ),
        _row(LOSER, "2023-01", n_is=18, n_oos=8, exp_is=-0.15, exp_oos=-0.2),
        _row(
            LOSER,
            "pooled",
            n_is=18,
            n_oos=8,
            exp_is=-0.15,
            exp_oos=-0.2,
            dsr_prob=0.42,
            boot_p5=-0.04,
            diff_p5=-0.11,
            mar=0.2,
            mar_bh=0.75,
            g1=True,
            g2=False,
            g3=True,
            g4=False,
            g5=True,
            g6=False,
            passed=False,
        ),
        _row(BROKEN, "pooled", excluded_reason="error:RuntimeError: synthetic strategy failure"),
        # ran, but the gate blew up on it: it has expectancies and no gate columns at all
        _row(
            RUINED,
            "pooled",
            n_is=14,
            n_oos=61,
            exp_is=0.2,
            exp_oos=0.9,
            excluded_reason="error:ValueError: a -200 R trade ruins the equity curve",
        ),
        _row(
            SELECTED,
            "pooled",
            n_is=12,
            n_oos=4,
            exp_is=0.30,
            exp_oos=0.375,
            dsr_prob=0.61,
            boot_p5=0.01,
            diff_p5=0.02,
            mar=1.4,
            mar_bh=0.75,
            g1=True,
            g2=False,
            g3=True,
            g4=True,
            g5=True,
            g6=None,
            passed=False,
        ),
    ]
    config = Config(entry="ict", exit="fixed_r_2", session="none", instrument=BTC)
    return TournamentResult(
        run_id="small",
        rows=tuple(rows),
        excluded=((SOL, "insufficient_history:6"),),
        gates={
            WINNER: _gate(passed=True, mar=math.inf),
            LOSER: _gate(passed=False, mar=0.2),
            SELECTED: _gate(passed=False, mar=1.4).model_copy(update={"rule6": None}),
        },
        selection={SELECTED: {"2023-01": "fixed_r_2", "2023-04": "trail_1_2"}},
        resolution_modes={"hyperliquid:BTC": "subbars"},
        runs={
            WINNER: ConfigRun(
                config=config,
                trades=winner_trades,
                equity_curve=(),
                n_bars=100,
                resolution_mode="subbars",
            )
        },
        n_trials=14,
        trial_sr_variance=0.05,
        oos_trades={WINNER: winner_trades, LOSER: loser_trades, SELECTED: selected_trades},
    )


# --- snapshot --------------------------------------------------------------------


def test_render_matches_the_snapshot() -> None:
    assert report.render(_result(), top_n=3) == SNAPSHOT.read_text(encoding="utf-8")


def test_render_is_deterministic() -> None:
    assert report.render(_result()) == report.render(_result())


# --- structure -------------------------------------------------------------------


@pytest.mark.parametrize(
    "heading",
    [
        "# swingforge tournament small",
        "## Gate",
        UNIVERSE_HEADING,
        "## Top 20 configs, IS vs OOS",
        "## IS-selected exit per split",
        "## Regime breakdown",
        "## Excursion",
        "## Cost stress",
        "## Excluded",
        "## Fill resolution mode",
    ],
)
def test_render_has_every_section(heading: str) -> None:
    assert heading in report.render(_result()).splitlines()


def test_sections_appear_in_the_documented_order() -> None:
    text = report.render(_result())
    order = [
        "## Gate",
        UNIVERSE_HEADING,
        "## Top 20 configs",
        "## IS-selected exit per split",
        "## Regime breakdown",
        "## Excursion",
        "## Cost stress",
        "## Excluded",
        "## Fill resolution mode",
    ]
    positions = [text.index(heading) for heading in order]
    assert positions == sorted(positions)


def _escaped(config_id: str) -> str:
    return config_id.replace("|", "\\|")


def _section(text: str, heading: str) -> list[str]:
    """Every line of one `## ` section, up to the next heading."""
    lines = text.splitlines()
    start = lines.index(heading)
    end = next(
        (i for i, line in enumerate(lines[start + 1 :], start + 1) if line.startswith("## ")),
        len(lines),
    )
    return lines[start:end]


def _table_body(text: str, heading: str) -> list[str]:
    """The data rows of that section's pipe table: header and separator dropped."""
    table = [line for line in _section(text, heading) if line.startswith("|")]
    return table[2:]


def test_gate_table_has_one_row_per_shown_config() -> None:
    text = report.render(_result())
    table = [line for line in _section(text, "## Gate") if line.startswith("|")]
    assert table[0].startswith("| config |")
    # winner, ruined, IS-selected, baseline: the un-run config never reaches the table
    assert len(_table_body(text, "## Gate")) == 4
    assert _escaped(BROKEN) not in text.split("## Excluded")[0]


def test_config_ids_are_escaped_so_the_pipe_table_survives_them() -> None:
    text = report.render(_result())
    table = [line for line in _section(text, "## Gate") if line.startswith("|")]
    header, separator = table[0], table[1]
    row = next(line for line in table if line.startswith(f"| {_escaped(WINNER)} |"))
    # unescaped pipes are the column delimiters; every row must have as many as the header
    assert row.count("|") - row.count(r"\|") == header.count("|") == separator.count("|")


def test_the_large_tables_keep_the_top_configs_and_count_the_rest() -> None:
    # the best by exp_oos, the one that passed and the IS-selected view; the rest counted
    text = report.render(_result(), top_n=1)
    for heading in ("## Gate", "## Excursion", "## Cost stress"):
        assert any("2 rows omitted" in line for line in _section(text, heading)), heading
        assert len(_table_body(text, heading)) == 3, heading
    assert _escaped(LOSER) not in text


def test_a_dominant_view_does_not_steal_a_top_n_slot() -> None:
    # WU-2C reviewer finding: `_shown_rows` used to rank the top_n including `IS_SELECTED`
    # rows, while `_top_configs` ranks them out and appends the views afterwards. So a view
    # ranked above every graded config could occupy the one top_n=1 slot and knock the best
    # non-view config out of the gate/excursion/cost-stress tables, even though that config
    # still won its rightful spot in the "Top N" table.
    result = _result()
    rows = tuple(
        row if not (row["config_id"] == SELECTED and row["split"] == "pooled") else {**row, "exp_oos": 5.0}
        for row in result.rows
    )
    dominant_view = replace(result, rows=rows)
    text = report.render(dominant_view, top_n=1)
    for heading in ("## Top 1 configs, IS vs OOS", "## Gate", "## Excursion", "## Cost stress"):
        assert _escaped(RUINED) in "\n".join(_section(text, heading)), heading


def test_a_passing_config_is_shown_however_it_ranks() -> None:
    # top_n=0 leaves only the two rules that are never subject to the cutoff
    body = _table_body(report.render(_result(), top_n=0), "## Gate")
    assert [line.split(" | ")[0] for line in body] == [f"| {_escaped(WINNER)}", f"| {_escaped(SELECTED)}"]


def test_only_the_ungraded_config_is_hidden_at_a_high_cutoff() -> None:
    text = report.render(_result(), top_n=99)
    assert any("1 row omitted" in line for line in _section(text, "## Gate"))


def test_gate_table_sorts_passing_configs_first() -> None:
    body = _table_body(report.render(_result()), "## Gate")
    assert body[0].startswith(f"| {_escaped(WINNER)} |")


def test_infinite_mar_renders_as_the_infinity_sign() -> None:
    text = report.render(_result())
    assert "∞" in text
    assert "1000000000" not in text


def test_missing_figures_render_as_an_en_dash() -> None:
    # a config that ran but whose gate blew up keeps its expectancies and loses the rest
    line = next(
        line
        for line in _table_body(report.render(_result()), "## Gate")
        if line.startswith(f"| {_escaped(RUINED)} |")
    )
    assert line.count("–") >= 12
    assert "| 0.900 |" in line


def test_top_n_limits_the_graded_rows_but_keeps_is_selected() -> None:
    body = _table_body(report.render(_result(), top_n=1), "## Top 1 configs, IS vs OOS")
    assert len(body) == 2  # one graded config, plus the IS-selected row
    assert _escaped(SELECTED) in body[-1]


def test_regime_breakdown_pools_over_passing_configs() -> None:
    text = report.render(_result())
    section = _section(text, "## Regime breakdown")
    assert "Pooled over the configs that passed the gate." in section
    assert len(_table_body(text, "## Regime breakdown")) == 4  # four distinct regime tags


def test_regime_breakdown_falls_back_to_every_config() -> None:
    result = _result()
    without_passes = TournamentResult(
        run_id=result.run_id,
        rows=result.rows,
        excluded=result.excluded,
        gates={key: gate.model_copy(update={"rule2": False}) for key, gate in result.gates.items()},
        selection=result.selection,
        resolution_modes=result.resolution_modes,
        runs=result.runs,
        oos_trades=result.oos_trades,
    )
    section = _section(report.render(without_passes), "## Regime breakdown")
    assert any("No config passed the gate" in line for line in section)


# --- universe trials: their own section, never the per-instrument tables ----------


def _result_with_universe() -> TournamentResult:
    """`_result()` plus three universe rows over BTC and SOL: one that clears rule 1, a thin one
    with a flattering expectancy, and an IS-selected view."""
    result = _result()
    sol_trades = tuple(
        # entered on the same bars as BTC's trades 0..3: four new trades, no new entry days
        _trade(i, -0.5, "range_midvol").model_copy(update={"id": f"s{i}", "instrument": SOL})
        for i in range(4)
    )
    merged = tuple(sorted((*result.oos_trades[WINNER], *sol_trades), key=lambda t: (t.entry_fill.ts, t.id)))
    gate_columns = {
        "dsr_prob": 0.42,
        "boot_p5": -0.04,
        "diff_p5": -0.11,
        "mar": 0.2,
        "mar_bh": 0.75,
        "g1": True,
        "g2": False,
        "g3": True,
        "g4": False,
        "g5": True,
        "g6": False,
        "passed": False,
    }
    rows = (
        *result.rows,
        _row(UNIVERSE, "pooled", n_is=38, n_oos=12, exp_is=0.2, exp_oos=0.1, **gate_columns),
        _row(UNIVERSE_THIN, "pooled", n_is=9, n_oos=3, exp_is=0.1, exp_oos=9.0, g1=False, passed=False),
        _row(UNIVERSE_VIEW, "pooled", n_is=30, n_oos=8, exp_is=0.3, exp_oos=0.05, **gate_columns),
    )
    members = ("hyperliquid:BTC", "hyperliquid:SOL")
    return replace(
        result,
        rows=rows,
        gates={
            **result.gates,
            UNIVERSE: _gate(passed=False, mar=0.2),
            UNIVERSE_THIN: GateResult(n=3, rule1=False),
            UNIVERSE_VIEW: _gate(passed=False, mar=0.2),
        },
        oos_trades={
            **result.oos_trades,
            UNIVERSE: merged,
            UNIVERSE_THIN: sol_trades[:3],
            UNIVERSE_VIEW: merged[:8],
        },
        universe={UNIVERSE: members, UNIVERSE_THIN: members, UNIVERSE_VIEW: members},
    )


def test_universe_rows_get_their_own_gate_table() -> None:
    text = report.render(_result_with_universe())
    table = [line for line in _section(text, UNIVERSE_HEADING) if line.startswith("|")]
    assert table[0].startswith("| config | instruments | entry days | n_oos |")
    body = _table_body(text, UNIVERSE_HEADING)
    assert len(body) == 3
    # the three-trade pool has by far the best expectancy and still reads last: it never cleared rule 1
    assert body[-1].startswith(f"| {_escaped(UNIVERSE_THIN)} |")
    line = next(line for line in body if line.startswith(f"| {_escaped(UNIVERSE)} |"))
    # two members; twelve trades entered on the two days BTC's eight already covered
    # the members are named: they are what a pass would be enabled on
    assert line.startswith(f"| {_escaped(UNIVERSE)} | 2 (BTC, SOL) | 2 | 12 |")
    header, separator = table[0], table[1]
    assert line.count("|") - line.count(r"\|") == header.count("|") == separator.count("|")


def test_the_universe_table_ranks_the_rows_that_clear_rule_1_first() -> None:
    """A three-trade pool with a 9R expectancy is noise; the row a reader needs is the one the
    gate could actually grade, so that is what survives a tight cutoff - beside every view."""
    text = report.render(_result_with_universe(), top_n=1)
    body = _table_body(text, UNIVERSE_HEADING)
    assert [line.split(" | ")[0] for line in body] == [
        f"| {_escaped(UNIVERSE)}",
        f"| {_escaped(UNIVERSE_VIEW)}",
    ]
    assert any("1 row omitted" in line for line in _section(text, UNIVERSE_HEADING))


def test_universe_rows_stay_out_of_the_per_instrument_gate_and_ranking() -> None:
    with_universe = report.render(_result_with_universe(), top_n=1)
    plain = report.render(_result(), top_n=1)
    for heading in ("## Gate", "## Top 1 configs, IS vs OOS"):
        assert _section(with_universe, heading) == _section(plain, heading), heading


def test_the_universe_rows_shown_get_their_excursion_and_cost_stress_too() -> None:
    """Rule 6 was graded on a universe row like any other, and it is the row a reader would act
    on: its stress delta belongs in the report beside the instruments'."""
    text = report.render(_result_with_universe(), top_n=1)
    for heading in ("## Excursion", "## Cost stress"):
        body = _table_body(text, heading)
        configs = {line.split(" | ")[0] for line in body}
        assert {f"| {_escaped(UNIVERSE)}", f"| {_escaped(UNIVERSE_VIEW)}"} <= configs, heading
        assert f"| {_escaped(UNIVERSE_THIN)}" not in configs, heading
        assert len(body) == 3 + 2, (
            heading
        )  # the three instrument rows and the two universe rows shown, in config-id order
        assert any("3 rows omitted" in line for line in _section(text, heading)), heading


def _regime_total(text: str) -> int:
    return sum(int(line.split(" | ")[1]) for line in _table_body(text, "## Regime breakdown"))


def test_the_regime_fallback_counts_a_universe_trade_once() -> None:
    """With nothing passing, the breakdown pools every config that ran - and a universe trial's
    trades are its members' trades, already counted under the same entry, exit and session."""
    result = _result_with_universe()
    failing = {key: gate.model_copy(update={"rule2": False}) for key, gate in result.gates.items()}
    # winner 8 + loser 8 + view 4, then what only the universe rows hold: SOL's 4 under the
    # winner's config, 3 under `zones`, and 4 under the view. Counted naively it would be 43.
    assert _regime_total(report.render(replace(result, gates=failing))) == 8 + 8 + 4 + 4 + 3 + 4


def test_a_passing_universe_row_does_not_recount_a_passing_members_trades() -> None:
    result = _result_with_universe()
    gates = {**result.gates, UNIVERSE: _gate(passed=True, mar=1.0)}
    text = report.render(replace(result, gates=gates))
    assert "Pooled over the configs that passed the gate." in _section(text, "## Regime breakdown")
    assert _regime_total(text) == 8 + 4  # BTC's eight once, SOL's four once - not 8 + 12


def test_a_universe_row_that_could_not_be_graded_is_listed_as_excluded() -> None:
    result = _result_with_universe()
    rows = tuple(
        {**row, "excluded_reason": "error:ValueError: close 0.0 is not positive", "passed": None}
        if row["config_id"] == UNIVERSE
        else row
        for row in result.rows
    )
    text = report.render(replace(result, rows=rows))
    assert any(_escaped(UNIVERSE) in line for line in _table_body(text, "## Excluded"))


def test_raw_signal_rows_are_appended_to_the_excursion_table() -> None:
    summary = ExcursionSummary(
        n=42,
        stopped_out_of_winner_rate=0.25,
        median_exit_efficiency=0.6,
        median_mae_r=0.7,
        median_mfe_r=1.9,
        mfe_p75=2.4,
    )
    text = report.render(_result(), raw_signal={WINNER: summary})
    assert f"| {_escaped(WINNER)} (raw signal) | 42 |" in text


def test_empty_result_still_renders_every_section() -> None:
    empty = TournamentResult(
        run_id="empty",
        rows=(),
        excluded=(),
        gates={},
        selection={},
        resolution_modes={},
        runs={},
    )
    text = report.render(empty)
    assert "## Gate" in text
    assert text.count("_none_") >= 7
    assert "rows omitted" not in text  # nothing to hide, so no note


def test_the_header_sends_dashboards_to_the_results_table() -> None:
    assert "`results` table" in report.render(_result()).splitlines()[2]


def test_the_report_needs_no_config_runs() -> None:
    # C1: `run_tournament(keep_runs=False)` returns `runs={}`, and the report is unchanged
    result = _result()
    without_runs = TournamentResult(
        run_id=result.run_id,
        rows=result.rows,
        excluded=result.excluded,
        gates=result.gates,
        selection=result.selection,
        resolution_modes=result.resolution_modes,
        oos_trades=result.oos_trades,
    )
    assert without_runs.runs == {}
    assert report.render(without_runs, top_n=3) == SNAPSHOT.read_text(encoding="utf-8")


def test_write_report_writes_utf8_with_lf(tmp_path: Path) -> None:
    destination = report.write_report(_result(), tmp_path / "nested" / "run.md")
    assert destination.exists()
    raw = destination.read_bytes()
    assert b"\r\n" not in raw
    assert raw.decode("utf-8") == report.render(_result())


def test_number_renders_a_stray_bool_as_a_flag() -> None:
    # defensive: bool is an int, so without the guard `True` would print as 1.000
    assert report._number(True) == "\u2713"
    assert report._number(False) == "\u2717"
