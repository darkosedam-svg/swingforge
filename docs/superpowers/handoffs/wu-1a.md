# WU-1A — FillResolver + cost math (branch `wu/1a`)

Built: `core/fills.py` (`FillResolver`, satisfies `ExitResolver`) and `core/costs.py` (`NullCostModel`, `stress`, `StressedCostModel`, `total_cost`), plus 57 tests in `tests/unit/test_fills.py` and `test_costs.py` — 100% line and branch coverage on both files.

Resolver semantics the paper-broker author must assume:
- Touch is inclusive: a long is stopped at `low <= stop` and takes a target at `high >= t`; a short mirrors it (`high >= stop`, `low <= t`).
- A touched level fills at the level, unless the candle being evaluated opened at or beyond it adversely (gap-through) — then at that candle's `open`. The gap rule covers only a level that was **live when that candle opened**: a stop `stop_after_partial` swapped in mid-candle fills at the level itself (price passed through it on the way back from `targets[0]`), and is live again — gap rule back on — at the next subbar's open. Prices pass through raw: tick rounding stays at the broker boundary.
- With `subbars` the resolver walks them oldest-first and `mode="subbars"`; with none it judges the whole bar once and `mode="pessimistic"`. `mode` reflects the data available, not the outcome — a bar with subbars that touches nothing is still `"subbars"`.
- One candle can yield more than one event and is read pessimistically: the stop in force is tested before the next target, so a range holding both yields the stop. Taking `targets[0]` swaps in `stop_after_partial` for everything evaluated afterwards — the remainder of that same candle in either mode, and every later subbar.
- A target not strictly beyond the stop in force (a trailing stop that ratcheted past a still-pending leg) is **unreachable and dropped** for that call rather than rejected — the stop governs; the same check is re-applied to `targets[1]` once `stop_after_partial` swaps in. Dropping never renumbers: `target_index` is always the caller's own index. `resolve` raises only for `direction ∉ {1,-1}` and more than two targets.
- A stop event, or the last target, ends the bar; nothing after it is reported. Empty `events` means nothing was touched — the two empty verdicts are the shared `fills.NO_EXITS` singletons — and a stop event always carries `target_index=None`.

Decisions: gap-through uses the evaluated candle's own `open`; `MAX_TARGETS = 2` is a module constant if the broker wants it; `StressedCostModel` is a frozen dataclass with the same constructor signature.

Not built: no `resolve_exits()` free function (the class is the contract), no tick rounding, no venue cost models (adapters own Hyperliquid/OANDA), and no partial *quantity* logic — the resolver reports levels only, and the caller maps `target_index` back to the pending order to get the `Fill.leg`.

Gotcha: `core` may not import `adapters`, so `StressedCostModel(inner)` types `inner` against a private `_CostSource` protocol that mirrors `CostModel`. Keep the two in step if that protocol ever changes.
