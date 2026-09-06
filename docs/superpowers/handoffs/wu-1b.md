# WU-1B — Portfolio + Engine

Built: `core/portfolio.py` (sizing, risk money, realized pnl, equity ledger), `core/engine.py` (bar-by-bar orchestrator). 129 unit tests total; both files 100% branch coverage — the once-"unreachable" `_update_excursion` guard is now covered too (see M10 below).

Per-bar sequence in `Engine.step`: (1) refresh settings; if the kill switch just turned on and an entry order is still resting unfilled, `broker.cancel` it before the broker ever sees this bar (F5); (2) push the bar; (3) `broker.on_bar` and process fills — an entry fill opens the trade and re-evaluates the same bar recursively; an exit fill is validated first (`EngineError` on a mismatched `trade_id` or an impossible `qty`, F2/F3) then folds this bar's range into `mae_r`/`mfe_r` *before* the closing snapshot is built (F1); (4) if still open, fold the bar into excursion again (idempotent via `max()`) and run the exit rule, syncing `ctx.position` via `_sync_position()` whenever it replaces stop/target (F4); (5) if flat/not killed/in session, size and submit an entry order.

`EngineError(RuntimeError)` (engine.py) covers all bad-broker-input paths (F2/F3/F7, plus M12 for an exit-rule order tagged for another trade); purely internal invariants stay as `assert`.

`Portfolio.size` drops its redundant `equity` arg (M11); `Portfolio.record(trade, pnl)` takes the exact pnl instead of re-deriving `realized_r * risk_r` (M8); `QTY_EPS = QTY_STEP / 2` (M9, exported) is the shared qty-comparison epsilon; `Engine.trades` is now a read-only property over `Portfolio.trades` (M13).

M10: a fill landing exactly on the stop (risk distance 0) falls back to the *planned* risk (`|signal.entry - stop| * qty * mult`) so `Trade.risk_r` stays positive.
