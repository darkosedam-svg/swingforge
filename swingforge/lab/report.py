"""The markdown run report: the human-readable half of a tournament's output.

Design spec section 6 asks for a report per run alongside the `results` table — gate table,
top configs IS vs OOS, regime breakdown, excursion, cost-stress deltas — plus the excluded
instruments and the fill-resolution mode per instrument, which section 5 requires every
report to state.

Everything here is derived from a `TournamentResult` and nothing is recomputed from the
store, so the report always describes exactly the run in hand. Output is deterministic:
every table has an explicit sort key ending in the config id, so two renders of the same
result are byte-identical.

**Truncation.** A full matrix is 2,016 configs, and 2,016 rows of anything is not something
a person reads. The three per-config tables (gate, excursion, cost stress) therefore show
every config that *passed*, the `top_n` best by out-of-sample expectancy and every
`IS_SELECTED` view, and count the rest in a line under the table. The `results` table is
the complete record and the report says so in its own first line.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from swingforge.core.types import Trade
from swingforge.lab.excursion import ExcursionSummary, excursion_summary
from swingforge.lab.gate import GateResult
from swingforge.lab.tournament import TournamentResult, expectancy, stressed_r

__all__ = ["render", "write_report"]

_HEADER_NOTE = (
    "Written to be read. The complete record of this run is the `results` table — anything "
    "downstream (dashboards, screens, further analysis) should read that, not this file, "
    "which shows only the configs worth a human's attention."
)

_MISSING = "–"
"""What an unmeasured figure prints as: an en dash, never a bare `None` or `nan`."""

_INFINITY = "∞"
_YES = "✓"
_NO = "✗"
_DECIMALS = 3

_IS_SELECTED = "IS_SELECTED"


def _number(value: float | None) -> str:
    """A figure at 3 decimals; `None`/`nan` become `–` and an infinite ratio becomes `∞`."""
    if value is None:
        return _MISSING
    if isinstance(value, bool):  # guard: bool is an int, and would print as 1.000
        return _flag(value)
    if math.isnan(value):
        return _MISSING
    if math.isinf(value):
        return _INFINITY if value > 0 else f"-{_INFINITY}"
    return f"{value:.{_DECIMALS}f}"


def _flag(value: bool | None) -> str:
    """A gate rule: `✓` held, `✗` failed, `–` never evaluated."""
    return _MISSING if value is None else (_YES if value else _NO)


def _cell(text: str) -> str:
    """Escape a cell for a pipe table.

    Config ids are `entry|exit|session|venue:symbol`, so almost every cell in this report
    contains the character that delimits a markdown table column. Without this the gate
    table renders as sixteen ragged columns of nonsense.
    """
    return text.replace("|", r"\|")


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    """A markdown pipe table, or a one-line note when there is nothing to show."""
    if not rows:
        return ["_none_", ""]
    lines = [
        "| " + " | ".join(_cell(header) for header in headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines.extend("| " + " | ".join(_cell(value) for value in row) + " |" for row in rows)
    lines.append("")
    return lines


def _pooled_rows(result: TournamentResult) -> list[dict[str, Any]]:
    """The `split="pooled"` rows, in a stable order: passed first, then expectancy, then id."""

    def key(row: dict[str, Any]) -> tuple[bool, float, str]:
        exp = row["exp_oos"]
        return (row["passed"] is not True, -(exp if exp is not None else -math.inf), row["config_id"])

    return sorted((row for row in result.rows if row["split"] == "pooled"), key=key)


def _shown_rows(pooled: Sequence[dict[str, Any]], top_n: int) -> tuple[list[dict[str, Any]], int]:
    """The pooled rows the per-config tables show, and how many that leaves out.

    Every config that passed the gate, the `top_n` best by out-of-sample expectancy, and
    every `IS_SELECTED` view — which is the walk-forward answer and belongs in the report
    whatever it ranks. A config that never produced an expectancy (its replay failed) is
    never shown: its row would be a line of en dashes, and the `Excluded` section names it
    with its reason. `pooled`'s own order is kept, so the choice is deterministic.
    """
    graded = sorted(
        (row for row in pooled if row["exp_oos"] is not None),
        key=lambda row: (-row["exp_oos"], row["config_id"]),
    )
    keep = {row["config_id"] for row in graded[:top_n]}
    keep |= {row["config_id"] for row in pooled if row["passed"] is True or row["exit"] == _IS_SELECTED}
    shown = [row for row in pooled if row["config_id"] in keep]
    return shown, len(pooled) - len(shown)


def _omitted_note(omitted: int) -> list[str]:
    """The line a truncated table carries, or nothing at all when nothing was hidden."""
    return [] if omitted == 0 else [f"{omitted} rows omitted; the full set is in the `results` table.", ""]


def _stressed_expectancy(trades: Sequence[Trade]) -> float | None:
    """Mean R across the same trades re-priced under rule 6's cost stress; `None` for none."""
    stressed = [stressed_r(trade) for trade in trades if trade.realized_r is not None]
    return sum(stressed) / len(stressed) if stressed else None


def _gate_of(result: TournamentResult, config_id: str) -> GateResult | None:
    return result.gates.get(config_id)


def _heading(text: str, level: int = 2) -> list[str]:
    return ["#" * level + " " + text, ""]


def _gate_table(
    result: TournamentResult, pooled: Sequence[dict[str, Any]], shown: Sequence[dict[str, Any]], omitted: int
) -> list[str]:
    headers = [
        "config",
        "n_oos",
        "exp_is",
        "exp_oos",
        "dsr prob",
        "boot p5",
        "diff p5",
        "MAR",
        "B&H MAR",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "passed",
    ]
    rows: list[list[str]] = []
    for row in shown:
        gate = _gate_of(result, row["config_id"])
        rules = (
            gate.rules()
            if gate is not None
            else dict.fromkeys(("rule1", "rule2", "rule3", "rule4", "rule5", "rule6"))
        )
        rows.append(
            [
                row["config_id"],
                _MISSING if row["n_oos"] is None else str(row["n_oos"]),
                _number(row["exp_is"]),
                _number(row["exp_oos"]),
                _number(gate.dsr.prob if gate is not None and gate.dsr is not None else None),
                _number(gate.boot_p5 if gate is not None else None),
                _number(gate.diff_p5 if gate is not None else None),
                _number(gate.mar_config if gate is not None else None),
                _number(gate.mar_bh if gate is not None else None),
                *(_flag(rules[name]) for name in ("rule1", "rule2", "rule3", "rule4", "rule5", "rule6")),
                _flag(None if gate is None else gate.passed),
            ]
        )
    lines = _heading("Gate")
    # `passed is None`, not `excluded_reason is not None`: a config whose ledger failed to
    # store carries a reason and a verdict, and is not one of the ones that could not run.
    failures = [row for row in pooled if row["passed"] is None]
    if failures:
        lines.append(f"{len(failures)} config(s) could not be graded; their reason is in the last section.")
        lines.append("")
    lines.extend(_omitted_note(omitted))
    lines.extend(_table(headers, rows))
    return lines


def _top_configs(pooled: Sequence[dict[str, Any]], top_n: int) -> list[str]:
    graded = [row for row in pooled if row["exit"] != _IS_SELECTED and row["exp_oos"] is not None]
    graded.sort(key=lambda row: (-row["exp_oos"], row["config_id"]))
    selected = sorted(
        (row for row in pooled if row["exit"] == _IS_SELECTED), key=lambda row: row["config_id"]
    )
    rows = [
        [row["config_id"], _number(row["exp_is"]), _number(row["exp_oos"]), str(row["n_oos"] or 0)]
        for row in [*graded[:top_n], *selected]
    ]
    lines = _heading(f"Top {top_n} configs, IS vs OOS")
    lines.append("Every `IS_SELECTED` row follows, whatever its rank: it is the walk-forward view.")
    lines.append("")
    lines.extend(_table(["config", "exp_is", "exp_oos", "n_oos"], rows))
    return lines


def _selection_table(result: TournamentResult) -> list[str]:
    splits = sorted({label for chosen in result.selection.values() for label in chosen})
    rows = [
        [config_id.replace(f"|{_IS_SELECTED}|", "|"), *(chosen.get(label, _MISSING) for label in splits)]
        for config_id, chosen in sorted(result.selection.items())
    ]
    lines = _heading("IS-selected exit per split")
    lines.append("The exit with the highest in-sample expectancy for that split, applied out of sample.")
    lines.append("")
    lines.extend(_table(["entry|session|instrument", *splits], rows))
    return lines


def _regime_table(result: TournamentResult) -> list[str]:
    passed = sorted(config_id for config_id, gate in result.gates.items() if gate.passed)
    scope = passed or sorted(result.oos_trades)
    note = (
        "Pooled over the configs that passed the gate."
        if passed
        else "No config passed the gate, so this is pooled over every config that ran."
    )
    buckets: dict[str, list[float]] = {}
    for config_id in scope:
        for trade in result.oos_trades.get(config_id, ()):
            if trade.realized_r is not None:
                buckets.setdefault(trade.regime or _MISSING, []).append(trade.realized_r)
    rows = [[regime, str(len(rs)), _number(sum(rs) / len(rs))] for regime, rs in sorted(buckets.items())]
    lines = _heading("Regime breakdown")
    lines.append(note)
    lines.append(
        "Diagnostic only: a trade appears once per config it belongs to, so counts are not "
        "independent across exits sharing an entry."
    )
    lines.append("")
    lines.extend(_table(["regime", "n", "expectancy (R)"], rows))
    return lines


def _excursion_table(
    shown: Sequence[dict[str, Any]],
    omitted: int,
    excursions: Mapping[str, ExcursionSummary],
    raw_signal: Mapping[str, ExcursionSummary] | None,
) -> list[str]:
    headers = [
        "config",
        "n",
        "stopped out of winner",
        "median exit eff.",
        "median MAE (R)",
        "median MFE (R)",
        "MFE p75 (R)",
    ]
    rows: list[list[str]] = [
        [row["config_id"], *_excursion_cells(excursions[row["config_id"]])]
        for row in sorted(shown, key=lambda item: item["config_id"])
    ]
    for config_id, summary in sorted((raw_signal or {}).items()):
        rows.append([f"{config_id} (raw signal)", *_excursion_cells(summary)])
    lines = _heading("Excursion")
    lines.append("Pooled out-of-sample trades per config. A `(raw signal)` row is the same entry")
    lines.append("held with no exit management, so it measures the entry rather than the exit.")
    lines.append("")
    lines.extend(_omitted_note(omitted))
    lines.extend(_table(headers, rows))
    return lines


def _excursion_cells(summary: ExcursionSummary) -> list[str]:
    return [
        str(summary.n),
        _number(summary.stopped_out_of_winner_rate),
        _number(summary.median_exit_efficiency),
        _number(summary.median_mae_r),
        _number(summary.median_mfe_r),
        _number(summary.mfe_p75),
    ]


def _cost_stress_table(
    result: TournamentResult,
    shown: Sequence[dict[str, Any]],
    omitted: int,
    stressed: Mapping[str, float | None],
) -> list[str]:
    rows: list[list[str]] = []
    for row in sorted(shown, key=lambda item: item["config_id"]):
        base = expectancy(result.oos_trades.get(row["config_id"], ()))
        stressed_exp = stressed[row["config_id"]]
        delta = None if base is None or stressed_exp is None else stressed_exp - base
        gate = _gate_of(result, row["config_id"])
        rows.append(
            [
                row["config_id"],
                _number(base),
                _number(stressed_exp),
                _number(delta),
                _flag(None if gate is None else gate.rule6),
            ]
        )
    lines = _heading("Cost stress")
    lines.append("Spread and slippage x2, funding x1.5, applied post hoc to the same fills (rule 6).")
    lines.append("")
    lines.extend(_omitted_note(omitted))
    lines.extend(_table(["config", "exp_oos", "stressed exp_oos", "delta", "rule 6"], rows))
    return lines


def _excluded_table(result: TournamentResult, pooled: Sequence[dict[str, Any]]) -> list[str]:
    rows = [
        [f"{instrument.venue}:{instrument.symbol}", reason]
        for instrument, reason in sorted(result.excluded, key=lambda item: (item[0].venue, item[0].symbol))
    ]
    rows.extend(
        [row["config_id"], row["excluded_reason"]]
        for row in sorted(pooled, key=lambda item: item["config_id"])
        if row["excluded_reason"] is not None
    )
    lines = _heading("Excluded")
    lines.append(
        "Instruments dropped for want of history, and configs whose run, gate or ledger write failed."
    )
    lines.append("")
    lines.extend(_table(["instrument / config", "reason"], rows))
    return lines


def _resolution_table(result: TournamentResult) -> list[str]:
    rows = [[key, mode] for key, mode in sorted(result.resolution_modes.items())]
    lines = _heading("Fill resolution mode")
    lines.append(
        "`subbars` means every 4H bar carried its 1H bars, so a stop and a target inside one "
        "bar were ordered on real data; `pessimistic` means the stop was assumed first."
    )
    lines.append("")
    lines.extend(_table(["instrument", "mode"], rows))
    return lines


def render(
    result: TournamentResult,
    *,
    top_n: int = 20,
    raw_signal: Mapping[str, ExcursionSummary] | None = None,
) -> str:
    """The full markdown report for one tournament run.

    `top_n` bounds the three per-config tables as well as the "top configs" one; see this
    module's docstring. `raw_signal` is an optional `{config id: ExcursionSummary}` from
    `swingforge.lab.excursion.raw_signal_excursion`; when given, those rows are appended to
    the excursion table so an entry's own excursion sits beside the managed version of it.
    """
    lines = [f"# swingforge tournament {result.run_id}", "", _HEADER_NOTE, ""]
    pooled = _pooled_rows(result)
    shown, omitted = _shown_rows(pooled, top_n)
    # once per config, not once per table: both walk the same pooled OOS trade lists
    excursions = {
        row["config_id"]: excursion_summary(result.oos_trades.get(row["config_id"], ())) for row in shown
    }
    stressed = {
        row["config_id"]: _stressed_expectancy(result.oos_trades.get(row["config_id"], ())) for row in shown
    }
    lines.extend(_gate_table(result, pooled, shown, omitted))
    lines.extend(_top_configs(pooled, top_n))
    lines.extend(_selection_table(result))
    lines.extend(_regime_table(result))
    lines.extend(_excursion_table(shown, omitted, excursions, raw_signal))
    lines.extend(_cost_stress_table(result, shown, omitted, stressed))
    lines.extend(_excluded_table(result, pooled))
    lines.extend(_resolution_table(result))
    return "\n".join(lines).rstrip("\n") + "\n"


def write_report(
    result: TournamentResult,
    path: str | Path,
    *,
    top_n: int = 20,
    raw_signal: Mapping[str, ExcursionSummary] | None = None,
) -> Path:
    """Render the report and write it to `path` as UTF-8 with LF endings; returns the path."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(render(result, top_n=top_n, raw_signal=raw_signal), encoding="utf-8", newline="\n")
    return destination
