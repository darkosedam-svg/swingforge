# swingforge

A venue-agnostic swing-trading research and paper-trading system: Daily bias, 4H execution, crypto perps (Hyperliquid) and FX majors (OANDA practice). It answers one question honestly: does any of three entry families (ICT structure, supply/demand zones, or a frequency-matched random baseline) show an out-of-sample edge, and which exit rule captures it best. Backtest and paper trading share one engine, and no result exists without passing a statistical gate that accounts for the full search.

Scope: research and paper trading only. Live order placement is a separate spec, gated on tournament and paper results.

## Status

All work units of the implementation plan are merged and verified:

- 950 tests: 949 pass and one contract test skips until the OANDA cassette is recorded (the Hyperliquid one is). 32 of them are marked `slow` (the planted-edge integration controls and the performance budgets). Coverage was 99% on `core` and `lab`, 98% overall, when last measured (2026-09-18).
- Import layering (`core` ← `strategies`/`adapters` ← `lab` ← `web`) is enforced by `import-linter` inside the test suite.
- The planted-edge integration test shows the real ICT entry clearing all six gate rules while the random baseline never does, and nothing passes on edge-free or pure-random-walk data - including five identical random walks pooled into one trial, which clear rule 1 and are read at one walk's worth of evidence.

The Hyperliquid half of the runbook has been executed locally (2026-09-07, re-run 2026-09-09). The contract cassette is recorded, the store holds 8 instruments (27 months of 4H bars for BTC/ETH/SOL/ARB, the venue's retention limit; funding complete), and the tournament graded 744 trials: **none pass**. Every config fails rule 1. An adversarial investigation of the first run's low trade frequency cleared the engine and the port and found three strategy defects, all fixed and re-run: ICT re-anchored its liquidity range to the current close every bar (now the range around each candidate bar's own body); zones consumed freshness only on bars the engine consulted (now every bar); consecutive Daily impulses appended duplicate zones. The re-run also exposed and fixed a broker defect that predates everything: same-bar exits were resolved against sub-bars that traded before the entry, crediting phantom fills (ICT averaged +3R per trade on a pure random walk; now near zero). After the fixes ICT with no session filter produces about 19 in-sample trades per year and 12.5 out-of-sample per 15 months per config, the best out-of-sample count anywhere is 15 against the 60 required, and at the observed per-trade Sharpe rule 2 would still need 60 to 100 trades. No paper config is enabled. Per-run tables are in `docs/superpowers/handoffs/wu-2a.md`; the strategy decision that was open at the time (walking every sweep candidate instead of newest-first) was adopted on 2026-09-19 and is recorded there with its measurements.

Pooling the instruments (2026-09-19, run `tournament:hyperliquid:20260919`, 893 trials) gets past rule 1 for one family: ICT with no session filter holds 62 to 63 pooled out-of-sample trades across ARB, BTC, ETH, HYPE and SOL, entered on 54 to 55 distinct days, and all 16 of its exits clear the 60-trade floor. **Still none pass**, and no longer for want of trades: the best pooled expectancy is +0.10R (deflated-Sharpe probability 0.002 against 0.95, bootstrap 5th percentile -0.22R, below the pooled baseline), and the average over the exits is -0.13R. The session-filtered ICT arms read +0.13R and +0.19R on 37 and 18 pooled trades, and zones is negative everywhere; those remain sample-size verdicts. The fresh replay reproduced all 744 per-instrument rows of the 2026-09-09 run exactly. Tables, the dependence guard (rule 2 reads a pooled row at its entry days) and the open decision on rule 1 are in `docs/superpowers/handoffs/wu-2c.md`.

With ICT's candidate walk adopted (2026-09-20, run `tournament:hyperliquid:20260920`, 901 trials) ICT trades three to four times as often: with no session filter 46 out-of-sample trades per config and instrument instead of 12.5 (best 65; twelve single-instrument rows clear the 60-trade floor on their own), and pooled 204 to 258 trades with no filter, 152 to 181 under London/NY and 106 to 119 under active hours, on 88 to 176 distinct entry days. **Still none pass, and for no ICT arm is sample size the reason any more**: of 306 ICT rows 64 clear rule 1 and none clears rule 2 or rule 3. Pooled expectancy is -0.19R with no filter, -0.07R under London/NY and +0.07R under active hours; the friendlier small-sample readings above (+0.13R and +0.19R on 37 and 18 trades) did not survive the larger sample. The best row, ICT under active hours with the trailing exit that activates at 2R and trails by one ATR (`trail_2_1`), holds +0.26R on 110 trades (+0.27R in sample), beats the pooled baseline and buy-and-hold, and still has a bootstrap 5th percentile of -0.05R and a deflated-Sharpe probability of about zero against 901 trials. Zones' rows are identical to the previous run. Tables are in `docs/superpowers/handoffs/wu-2a.md`.

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
uv run pytest -q                      # everything, slow tests included (~5 min)
uv run pytest -q -m "not slow"        # the fast loop
uv run pytest -q -m slow              # only the integration controls and performance budgets (~3.5 min)
uv run pytest -q --cov=swingforge     # with coverage (slower still; line tracing slows the replay loops)
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
- ICT's liquidity range is the nearest Daily swing pivots around each candidate bar's own body, and the entry fires for the newest sweep whose break of structure is the current bar. The reference analyzer stops at the newest sweep, which suits its one fixed Asia range per day; with pivot ranges a newer sweep with no break yet would shadow an older setup on the bar that confirms it. The walk multiplies the setups found where nothing was planted (2.7x on the synthetic planted stores; 16 to 246 out-of-sample trades on a pure random walk, where they earn nothing), which is what lets the null controls reject on statistics rather than on the trade-count floor.
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
