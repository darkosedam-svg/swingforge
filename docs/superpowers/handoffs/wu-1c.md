# WU-1C — exit rules (`swingforge/strategies/exits.py`, based on `contract-v2`)

16 variants in `EXIT_GRID` (now a `tuple`): `fixed_r_{1.5,2,3}`; `atr_{1.5,2,3}_rr_{2,3}` (mult×rr, 6); `structure_2`; `trail_{1,2}×{1,2}` (4); `partial_1_0.5_fixed_r_3`; `partial_1_0.5_trail_1_2`. Every level derives from `r_dist = risk_r/(entry_qty*mult)`, fixed at entry.

ATR fallback: `ATRFixedR.initial_stop` → `signal.stop` when `ctx.atr("1d")` is NaN; `Trailing._manage` returns `[]` for that bar (time stop still applies).

Time stop lives once in `_ExitRuleBase.on_bar` (renamed from `_TimeStopMixin`; base `__init__(time_stop_bars, time_stop_min_r)`, subclasses `super().__init__(...)`). It measures `unrealized_r` on the *remaining* qty, deliberately: after a partial, it judges whether the runner is working, not the original trade size.

Post-review fixes (code review, 2026-09-06): **C1/C2** `Partial._manage`'s breakeven guard is now directional and tick-normalised (`trade.direction * (round_to_tick(entry, tick) - trade.stop) > 0`), so it neither re-fires forever against an off-tick entry nor yanks a stop a pre-partial trail already moved past breakeven back down. **I1** `Trailing._manage` now rounds the candidate stop *before* comparing (`rounded > stop` long / `< stop` short) instead of round-then-equality, so it never moves backward by a sub-tick amount. **I3** `Partial.attach` skips the partial leg entirely (delegates to `runner.attach`) when the partial or runner-remainder qty rounds to ≤0 at 6dp.

Minor: extracted `_stop_and_target(trade, ctx, target)` (attach-only, shared by `_fixed_rr_orders`/`Structure.attach`); order ids carry an `:a`/`:m` phase suffix so `attach` and `on_bar` on the same bar never collide; `Trailing._manage` returns `[]` when `ctx.offset_of(trade.opened_bar)` is `None` (invariant: a run's `max_bars` must exceed its max holding period, or `opened_bar` could be trimmed mid-trade); `Partial.runner` is now typed `FixedR | Trailing`.

100% branch coverage on `exits.py` (51 tests, `tests/unit/test_exits.py`, up from 34); full suite (138 tests), `lint-imports`, `ruff check`/`format`, and `mypy swingforge` all pass clean. Coverage is per branch, not per state — some defensive branches (e.g. an empty extreme array, which `offset_of` makes unreachable) are folded into an existing short-circuit rather than given their own test.
