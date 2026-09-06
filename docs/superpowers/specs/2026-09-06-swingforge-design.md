# swingforge — Design Spec

Date: 2026-09-06
Status: approved in brainstorming (sections 1–6), pending written review
Scope: research + paper trading. Live execution is a separate spec, gated on tournament and paper results.

## 1. Purpose

A standalone, venue-agnostic swing trading system (Daily bias / 4H execution) that answers one question honestly: does any of three entry families — ICT structure, supply/demand zones, or a frequency-matched random baseline — show an out-of-sample edge on crypto perps and FX majors, and which exit rule captures it best. The system is built so that backtest, paper trading and (later) live execution share one code path, and so that no result can be produced without passing a statistical gate that accounts for the full search.

Non-goals for this spec: real order placement, portfolio-level allocation across configs, strategy parameters editable outside the tournament, any dashboard write path into `core/` or `strategies/`.

## 2. Architecture and repo layout

Python 3.12, `uv`, Pydantic v2 (frozen models at every boundary), numpy, DuckDB for storage, `typer` CLI, FastAPI for the dashboard only.

```
swingforge/
  core/         types.py engine.py portfolio.py costs.py fills.py context.py settings.py
  strategies/   base.py ict.py zones.py baseline.py exits.py
  adapters/     base.py replay.py paper.py hyperliquid/{bars.py,costs.py} oanda/{bars.py,costs.py}
  lab/          tournament.py gate.py excursion.py report.py regime.py
  web/          app.py static/index.html
  cli.py
tests/          unit/ integration/ contract/
docs/superpowers/{specs,plans}/
```

Dependency rules (enforced by an import-linter test):
- `core` imports only stdlib, pydantic, numpy.
- `strategies` imports `core` only.
- `adapters` is the only package that imports venue SDKs; imports `core`.
- `lab` imports `core`, `strategies`, `adapters.replay`, `adapters.paper`.
- `web` imports `lab` and DuckDB only. Nothing imports `web`.

Data flow: `BarSource` yields closed bars → `Engine` updates `Context` → `Strategy.on_bar(ctx)` → `Signal | None` → `Portfolio.size(signal)` → `Broker.submit(Order)` → `Fill` → `Portfolio` opens `Trade`, `ExitRule.attach(trade, ctx)` sets stop/target legs → subsequent bars: `FillResolver` decides exits → `Trade` closed and persisted with a context snapshot.

Backtest and paper differ only in the `BarSource` (replay vs stream) and nothing else. `Broker` has exactly one implementation in this spec: `PaperBroker`.

## 3. Core types (`core/types.py`)

```
Instrument(venue: Literal["hyperliquid","oanda"], symbol, tick_size: Decimal,
           contract_multiplier: Decimal, quote_ccy, session_profile: Literal["fx","perp"])
Bar(instrument, tf: Literal["1h","4h","1d"], ts_open: datetime(UTC), open, high, low, close,
    volume, bid_close: float|None, ask_close: float|None, subbars: tuple[Bar,...] = ())
Signal(direction: Literal[1,-1], entry: float, stop: float, structure_target: float|None,
       tag: str, expires_in_bars: int)      # validator: stop strictly on the wrong side of entry
Order(id, instrument, direction, qty, kind: Literal["limit","market"], price: float|None,
      expires_at_bar: int|None, leg: Literal["entry","stop","target","partial","time"])
Fill(order_id, ts, price, qty, cost: CostBreakdown)
CostBreakdown(spread: float, commission: float, funding: float, slippage: float)
Trade(id, instrument, direction, entry_fill, legs: list[Fill], stop: float, target: float|None,
      risk_r: float, realized_r: float|None, mae_r, mfe_r, regime: str, context_snapshot: bytes)
Position(instrument, direction, qty, avg_price, stop, target)
```

All prices are floats internally with `tick_size` rounding at the broker boundary. NaN, non-positive size, and stop on the wrong side raise at construction.

## 4. Strategies (`strategies/`)

All run on 4H bars with Daily bars available in `Context`. All emit a stop with the signal; the stop is the invalidation level, never a distance.

**`ict.py`** — sweep of a Daily swing high/low on 4H → market structure shift against the sweep within `N=6` bars → limit at the resulting OB/FVG midpoint → stop one tick beyond the sweep extreme. `structure_target` = nearest opposing Daily liquidity level. Logic ported from the corrected SHURIKEN v0.3 ICTAnalyzer (sweep direction, pullback-vs-sweep, BOS neckline fixes are regression-tested). Entry expires after `M=3` bars unfilled.

**`zones.py`** — zone = base of a Daily impulse (≥ 2×ATR(14,D) move within ≤ 3 bars); bounds = last opposing candle before the impulse. Fresh until touched once. Entry: 4H close inside a fresh zone → limit at zone midpoint, stop beyond far edge + 0.25×ATR. Optional round-number confluence filter (boolean, default off, tournament dimension).

**`baseline.py`** — random 4H bars, frequency-matched to the ICT strategy's trade count per instrument from the most recent tournament run, random direction, stop at 1.5×ATR. Seeded. This is the control.

**`exits.py`** — `ExitRule` protocol: `attach(trade, ctx) -> list[Order]`, `on_bar(trade, ctx) -> list[Order]`.
- `FixedR(rr)` — grid {1.5, 2, 3}
- `ATRFixedR(rr, atr_mult)` — overrides stop to `atr_mult×ATR(14,D)`; grid mult {1.5,2,3} × rr {2,3}
- `Structure(fallback_rr=2)` — target at `structure_target` if ≥ 1.5R away, else FixedR fallback
- `Trailing(activate_r, trail_atr)` — grid activate {1,2} × trail {1,2}
- `Partial(scale_r=1, scale_pct=0.5, runner)` — runner ∈ {FixedR(3), Trailing(1,2)}; stop to breakeven after scale
- Time stop on every rule: close at market after 10 4H bars if |unrealized| < 0.5R. Not swept.
Total exit variants: 16.

**Session filter** (entry gating only, tournament dimension on all instruments): `none`, `london_ny` (bars opening 07:00–21:00 UTC), `active` (fx: same as london_ny; perp: exclude 00:00–07:00 UTC bars and Sat/Sun bars).

Risk per trade fixed at 1% of equity for the tournament. Sizing not swept in v1. No session-switching or regime-switching inside strategies.

## 5. Adapters (`adapters/`)

Protocols (`base.py`):
```
BarSource: history(instrument, tf, start, end) -> list[Bar]; stream(instrument, tf) -> AsyncIterator[Bar]
Broker:    submit(order) -> str; cancel(order_id); positions() -> list[Position]; fills() -> AsyncIterator[Fill]
CostModel: entry(order, bar) -> CostBreakdown; carry(position, bar) -> float
```

**Hyperliquid**: bars via `hyperliquid-execution-toolkit` candle client (thin wrapper). Instruments: BTC, ETH, SOL + 5 by 90-day median volume at backfill. Costs: toolkit taker schedule, funding accrued at 8h boundaries from funding history, slippage 0.5 tick.

**OANDA**: v20 REST, practice account. Instruments: EUR/USD, GBP/USD, USD/JPY, AUD/USD, GBP/JPY, XAU/USD. Bid/ask candles; mid for signals, bid/ask for fills. Costs: spread from the fill bar's bid/ask, no commission, swap at 21:00 UTC rollover from the financing endpoint. OANDA daily candles (21:00 UTC boundary) are re-cut to 00:00 UTC from 1H bars so Daily aligns with perps.

**Replay** (`replay.py`): reads DuckDB, yields bars strictly after close, attaches 1H `subbars` to each 4H bar for the fill resolver. The only data path into the engine during backtests.

**PaperBroker** (`paper.py`): one class for both venues; wraps a `BarSource` for prices and a `CostModel`. Limit fills when a subsequent closed bar's range touches price. State machine `pending → filled | cancelled | expired`. Every fill logged with triggering bar and cost breakdown.

**FillResolver** (`core/fills.py`): when stop and target both lie inside one 4H bar, walk `subbars` (1H) to decide order; if no subbars, stop is assumed first. Handles up to two exit legs per trade (Partial). Every report states which resolution mode was used per instrument.

**Persistence**: one DuckDB file per venue; tables `bars`, `funding`, `orders`, `fills`, `trades`, `equity`, `results`, `settings`, `settings_log`. Backfill idempotent (upsert on `instrument, tf, ts_open`).

## 6. Tournament and gate (`lab/`)

Matrix: entry {3} × exit {16} × session {3} × instrument {14} = 2,016 configs. Serial loop, one `results` row per config.

Walk-forward: anchored; IS 12 months → OOS 3 months, step 3 months. Backfill target ≥ 4 years FX, maximum available for perps; instruments with < 18 months excluded from the gate and listed as excluded. Exit-grid parameters selected on IS per split, applied on OOS; only OOS trades feed the gate; IS/OOS shown side by side.

Gate — all must hold on pooled OOS trades:
1. `n_trades ≥ 60`
2. Deflated Sharpe (Bailey & López de Prado) > 0 at 95%, trials = 2,016
3. Stationary bootstrap (1,000 resamples), 5th-percentile expectancy > 0 R
4. Beats the `baseline` entry under the same exit + session: bootstrap difference, 5th percentile > 0
5. Beats buy-and-hold on the instrument on MAR
6. Rules 1–5 hold again with spread/slippage ×2 and funding ×1.5

Regime tag per trade at entry: trend if Daily ADX(14) > 25 else range; realized-vol tercile. Diagnostic only.

Excursion: per trade MAE/MFE in R, exit efficiency = captured/MFE; per config: stopped-out-of-winner rate, median exit efficiency; also run on raw signals with no exit.

Outputs: markdown report per run (gate table, top configs IS vs OOS, regime breakdown, excursion, cost-stress deltas) + `results` table.

Paper promotion: gate pass → paper for max(30 trades, 60 days) → paper/backtest reconciliation on the overlapping window must pass before a live spec is written.

## 7. Dashboard (`web/`)

FastAPI serving one self-contained HTML (vanilla JS, Recharts via CDN). `127.0.0.1:8787` default, `--host 0.0.0.0` + bearer token from `.env` for VPS. Polling 10s.

Read routes: `/` (paper equity per venue, open positions with unrealized R, last 20 fills, adapter staleness per instrument in bars), `/tournament` (gate table, IS/OOS top configs, regime, cost-stress), `/trades/{id}` (bars around entry, SL/TP, MAE/MFE path).

Write route: `PUT /api/settings` → `settings` table. Editable: enabled instruments per venue, gate-passed configs running paper, risk % (Pydantic cap 2%), session filter, time-stop bars, paper kill switch. Every change appended to `settings_log`. Engine reads settings at bar close only. Nothing in `core/` or `strategies/` is editable.

## 8. Testing

Unit (every commit): types validation; fill resolver hand cases + hypothesis property (exactly one of SL/TP); each exit rule on synthetic paths; strategy fixtures incl. v0.3 regressions; cost accrual timing; gate math on worked examples; baseline frequency match.
Integration (PR): replay determinism (byte-identical `trades`); no-lookahead mutation test; paper/replay reconciliation (identical fills); planted-edge synthetic tournament (ict passes, baseline fails) and pure random walk (all fail).
Contract (manual, `vcrpy`): OANDA and Hyperliquid `history()` schema, UTC, bar alignment, OANDA daily re-cut.
Coverage: 85% on `core` and `lab`.

## 9. Open decisions resolved

- No partial scale-outs → reversed: `Partial` is in (section 4).
- Session filter FX-only → reversed: dimension on all instruments (section 4).
- Dashboard out of scope → reversed: read-mostly dashboard with a settings table (section 7).
- IBKR/MT5 rejected in favour of OANDA v20 practice (headless, bid/ask candles, 4y+ history).
- Rolling walk-forward rejected for anchored (swing trade sparsity).
