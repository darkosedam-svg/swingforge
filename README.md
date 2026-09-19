# swingforge

A venue-agnostic swing-trading research and paper-trading system: Daily bias, 4H execution, crypto perps (Hyperliquid) and FX majors (OANDA practice). It answers one question honestly: does any of three entry families (ICT structure, supply/demand zones, or a frequency-matched random baseline) show an out-of-sample edge, and which exit rule captures it best. Backtest and paper trading share one engine, and no result exists without passing a statistical gate that accounts for the full search.

Scope: research and paper trading only. Live order placement is a separate spec, gated on tournament and paper results.

## Status

All work units of the implementation plan are merged and verified:

- 899 tests: 898 pass and one contract test skips until the OANDA cassette is recorded (the Hyperliquid one is). 27 of them are marked `slow` (the planted-edge integration controls and the performance budgets). Coverage is 99% on `core` and `lab`, 98% overall.
- Import layering (`core` ← `strategies`/`adapters` ← `lab` ← `web`) is enforced by `import-linter` inside the test suite.
- The planted-edge integration test shows the real ICT entry clearing all six gate rules while the random baseline never does, and nothing passes on edge-free or pure-random-walk data.

The Hyperliquid half of the runbook has been executed locally (2026-09-07, re-run 2026-09-09). The contract cassette is recorded, the store holds 8 instruments (27 months of 4H bars for BTC/ETH/SOL/ARB, the venue's retention limit; funding complete), and the tournament graded 744 trials: **none pass**. Every config fails rule 1. An adversarial investigation of the first run's low trade frequency cleared the engine and the port and found three strategy defects, all fixed and re-run: ICT re-anchored its liquidity range to the current close every bar (now the range around each candidate bar's own body); zones consumed freshness only on bars the engine consulted (now every bar); consecutive Daily impulses appended duplicate zones. The re-run also exposed and fixed a broker defect that predates everything: same-bar exits were resolved against sub-bars that traded before the entry, crediting phantom fills (ICT averaged +3R per trade on a pure random walk; now near zero). After the fixes ICT with no session filter produces about 19 in-sample trades per year and 12.5 out-of-sample per 15 months per config, the best out-of-sample count anywhere is 15 against the 60 required, and at the observed per-trade Sharpe rule 2 would still need 60 to 100 trades. No paper config is enabled. Per-run tables are in `docs/superpowers/handoffs/wu-2a.md`; the open strategy decision (walking every sweep candidate instead of newest-first) is recorded there with its measurements.

Still human-only: OANDA credentials (cassette, backfill, tournament) and the VPS itself. See `deploy/README.md`.

## Layout

```
swingforge/
  core/         types, context (Wilder ATR/ADX), fill resolver, costs, portfolio, engine, settings
  strategies/   ict, zones, baseline, session filter, levels, the 16-variant exit grid
  adapters/     DuckDB store, replay source, paper broker, hyperliquid/, oanda/, transient settings reader
  lab/          gate (deflated Sharpe, stationary bootstrap, MAR), tournament, excursion, regime, report
  web/          FastAPI dashboard (read-mostly; one settings write route)
  cli.py        backfill, tournament, paper, web, report
tests/          unit/, integration/ (slow), contract/ (vcrpy cassettes)
deploy/         systemd units, install.sh, VPS runbook
docs/superpowers/
  specs/        design spec and the research note on fixed exits
  plans/        the multi-agent implementation plan
  handoffs/     one note per work unit: what was built, deviations, gotchas
```

## Quickstart

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pytest -q                      # everything, slow tests included (~3 min)
uv run pytest -q -m "not slow"        # the fast loop
uv run pytest -q -m slow              # only the integration controls and performance budgets (~1.5 min)
uv run pytest -q --cov=swingforge     # with coverage (~7 min; line tracing slows the replay loops)
uv run lint-imports && uv run ruff check . && uv run mypy swingforge
```

Copy `.env.example` to `.env` and fill in `OANDA_TOKEN`, `OANDA_ACCOUNT_ID` (practice) and `SWINGFORGE_TOKEN` (dashboard bearer token). Hyperliquid candles are public and need no key.

## Workflow

```bash
swingforge backfill --venue hyperliquid --years 4 --data-dir data
swingforge backfill --venue oanda --years 4 --data-dir data
swingforge tournament --venue hyperliquid --out reports --data-dir data
swingforge report --venue hyperliquid --run-id <printed run id> --data-dir data
swingforge paper --venue hyperliquid --config "ict|fixed_r_2|london_ny|hyperliquid:BTC" --data-dir data
swingforge web --host 127.0.0.1 --port 8787 --data-dir data
```

One DuckDB file per venue lives under `--data-dir` (env `SWINGFORGE_DATA_DIR`). The tournament prints its run id and the passing config ids; `--run-id` fixes the id (required with `--resume`), and `--instruments`, `--entries`, `--exits`, `--sessions`, `--start`, `--end` narrow a run. Paper never trades a `baseline` config.

## How the tournament works

- Matrix: 3 entries × 16 exits × 3 session filters × instruments. Each config is replayed once over its full history; anchored walk-forward splits (12 months in-sample, 3 months out-of-sample, stepping 3 months) are applied by entry timestamp, which is exact because nothing is fitted on the in-sample window.
- Only pooled out-of-sample trades reach the gate. All six rules must hold: at least 60 trades; deflated Sharpe above 0 at 95% with the trial count set to the whole matrix; stationary-bootstrap 5th-percentile expectancy above 0; beats the frequency-matched baseline under the same exit and session; beats buy-and-hold on MAR; and all of that again with spread and slippage doubled and funding scaled by 1.5.
- An IS-selected view picks the best exit per split on in-sample expectancy and reports that choice's out-of-sample result beside the fixed-exit configs.
- A pooled view grades each entry, exit and session (and each IS-selected view) once more on the out-of-sample trades of every instrument of the venue together, under `entry|exit|session|venue:*`: against the pooled baseline and an equal-weight buy-and-hold of the same instruments, with every pooled trial added to the trial count. Rule 1 counts the pooled trades; rule 2 reads them at the number of distinct days they were entered on, because instruments of one venue move together. The two can straddle 60: a pooled row may clear rule 1 on fewer than 60 entry days, and the report prints both.
- Output: a `results` table per venue and a markdown report per run (gate table, top configs IS vs OOS, regime breakdown, MAE/MFE excursion, cost-stress deltas, excluded instruments, fill-resolution mode).

## Design decisions that differ from the spec

Each is recorded in the relevant handoff note under `docs/superpowers/handoffs/`.

- Hyperliquid funding accrues hourly (the venue settles hourly), not at 8h boundaries.
- `hyperliquid-execution-toolkit` does not exist on PyPI; the official `hyperliquid-python-sdk` is used, and the "extra five" perps are chosen by daily notional volume rather than a 90-day median.
- The engine records MAE/MFE on every bar a trade was live, including the entry and closing bars, so exit efficiency never exceeds 1.
- DuckDB's file lock is exclusive across processes even for read-only opens. The paper process therefore opens its store only briefly at each bar close and retries a locked file for up to five minutes; the dashboard retries briefly and returns 503; the nightly backfill runs under a lock and does not restart paper.
- Paper does not restore an open position across restarts. It refuses to start over an orphaned open trade unless `--abandon-open-trade` closes it at 0R. Equity rows double as the run's bar clock so trade ids stay unique across restarts.
- The dashboard may import `adapters.store` (for the atomic settings write); it still cannot import strategies, venue adapters, replay or paper code.
- Rule 2's cross-trial Sharpe variance is taken over trials with at least 60 pooled out-of-sample trades (rule 1's floor). Shorter trials have unbounded per-trade Sharpe estimates, and on the first real Hyperliquid sweep four-trade trials set the variance to 2,721, which no strategy could clear. The trial count still includes every trial.
- The spec grades one instrument at a time, and one instrument does not reach rule 1's 60 out-of-sample trades in the history a venue serves. The tournament therefore also grades pooled `venue:*` trials (above). A pooled pass is a verdict on the instruments traded together; `swingforge paper` refuses a `venue:*` id and the config is enabled one instrument at a time.
- Hyperliquid serves only its newest 5,000 candles per interval, so a backfill holds about 27 months of 4H bars and 7 months of 1H bars however many years are requested; funding history is paged and complete.

## Deployment

`deploy/README.md` is the runbook: provisioning, credentials, backfill, tournament, reading the report, enabling gate-passed configs as `swingforge-paper@<venue>-<slug>` instances, the 60-day / 30-trade check, paper-vs-backtest reconciliation, cassette recording, kill switch, logs and upgrades.
