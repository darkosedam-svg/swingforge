# swingforge — Multi-Agent Implementation & Deployment Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Each work unit (WU) is dispatched to a fresh subagent with: this file, the spec (`docs/superpowers/specs/2026-09-06-swingforge-design.md`), and `swingforge/core/types.py` + `swingforge/adapters/base.py` from Wave 0. Every WU applies TDD per step: write failing test → run → minimal implementation → run → commit.

**Goal:** Build swingforge to the point where a full 2,016-config tournament runs on real backfilled data, produces a gate report, and any gate-passing config runs paper on a VPS with the dashboard reachable over a URL.

**Architecture:** Single event-driven engine with two clocks (replay / stream); strategies and exit rules as independent protocols crossed by a tournament; statistical gate deflated for the full search; one PaperBroker for both venues; read-mostly FastAPI dashboard over DuckDB.

**Tech Stack:** Python 3.12, uv, pydantic v2, numpy, duckdb, typer, fastapi, uvicorn, hypothesis, pytest, vcrpy, import-linter, `hyperliquid-execution-toolkit`, `oandapyV20`, systemd on Ubuntu 24 VPS.

---

## 0. Orchestration model

**Why waves, not a free DAG.** The spec's import rules (`core` ← `strategies`/`adapters` ← `lab` ← `web`) are the dependency graph. Anything that shares a wave shares no files and imports only frozen artifacts from earlier waves, so agents cannot conflict. The orchestrator (you, or a lead agent) does not write code after Wave 0 — it dispatches, reviews, and merges.

**Mechanics.**
- One git repo, `main` protected. Each WU runs in its own worktree/branch `wu/<id>` (superpowers:using-git-worktrees). Merge order within a wave is irrelevant by construction; merge only after the two-stage review.
- Two-stage review per WU (from subagent-driven-development): (1) spec-compliance reviewer checks the WU against the spec section and the interface contract; (2) code-quality reviewer runs the WU's tests + `uv run pytest tests/unit -q` + `uv run lint-imports`. Both must pass. Reviewer agents are fresh, not the implementer.
- Interface contract = Wave 0 files. A WU that needs a contract change stops and reports; the orchestrator changes the contract, bumps `CONTRACT_VERSION` in `core/__init__.py`, and re-dispatches affected WUs. This is the only cross-wave communication channel.
- Every WU ends with: tests green, coverage on its own files reported, one squash commit `feat(<pkg>): <WU title>`, and a 10-line handoff note appended to `docs/superpowers/handoffs/<wu-id>.md` (what was built, what was deliberately not built, gotchas).
- Model routing: Wave 0, gate math (WU-2C), fills (WU-1A) and all reviewers on Opus-class; the rest on Sonnet-class. The expensive parts are where subtle bugs cost the most.

**Timeline (single machine, one human reviewing):** Wave 0 ½ day · Wave 1 1 day · Wave 2 1 day · Wave 3 1 day · Wave 4 ½ day + backfill time + 60-day paper window. Wall-clock ~4 days to first tournament report.

---

## 1. File structure (locked in Wave 0)

| Path | Responsibility | Owner WU |
|---|---|---|
| `pyproject.toml`, `.pre-commit-config.yaml`, `.importlinter`, `.env.example` | toolchain, lint, contract enforcement | WU-0 |
| `swingforge/core/types.py` | all frozen Pydantic models | WU-0 |
| `swingforge/core/context.py` | `Context`: bar arrays per tf, position, instrument meta, ATR/ADX helpers | WU-0 |
| `swingforge/adapters/base.py` | `BarSource`, `Broker`, `CostModel` protocols | WU-0 |
| `swingforge/strategies/base.py` | `Strategy`, `ExitRule` protocols | WU-0 |
| `swingforge/core/fills.py` | `FillResolver` | WU-1A |
| `swingforge/core/costs.py` | `CostBreakdown` math, `NullCostModel` | WU-1A |
| `swingforge/core/portfolio.py` | sizing (1% risk ÷ stop distance), R accounting, equity | WU-1B |
| `swingforge/core/engine.py` | bar loop, order routing, trade lifecycle | WU-1B |
| `swingforge/core/settings.py` | `Settings` model (risk cap 2%), DuckDB read at bar close | WU-1B |
| `swingforge/strategies/exits.py` | FixedR, ATRFixedR, Structure, Trailing, Partial, time stop | WU-1C |
| `swingforge/lab/gate.py` | deflated Sharpe, stationary bootstrap, MAR, gate rules 1–6 | WU-1D |
| `swingforge/lab/regime.py` | ADX(14,D) trend/range + vol tercile tagger | WU-1D |
| `swingforge/adapters/store.py` | DuckDB schema, upsert, readers | WU-1E |
| `swingforge/adapters/replay.py` | replay BarSource with subbar attachment | WU-1E |
| `swingforge/adapters/paper.py` | PaperBroker + order state machine | WU-1E |
| `swingforge/strategies/ict.py` | sweep → MSS → OB/FVG | WU-2A |
| `swingforge/strategies/zones.py` | supply/demand AOI | WU-2A |
| `swingforge/strategies/baseline.py` | frequency-matched random | WU-2A |
| `swingforge/strategies/session.py` | session filter | WU-2A |
| `swingforge/adapters/hyperliquid/{bars,costs}.py` | HL bars wrapper, funding/taker costs | WU-2B |
| `swingforge/adapters/oanda/{bars,costs}.py` | OANDA v20 bars, spread/swap costs, daily re-cut | WU-2B |
| `swingforge/lab/tournament.py` | config matrix, anchored walk-forward, results writer | WU-2C |
| `swingforge/lab/excursion.py` | MAE/MFE, exit efficiency | WU-2C |
| `swingforge/lab/report.py` | markdown report | WU-2C |
| `swingforge/cli.py` | `backfill`, `tournament`, `paper`, `report`, `web` | WU-3A |
| `tests/integration/*` | determinism, no-lookahead, reconciliation, planted-edge | WU-3B |
| `swingforge/web/app.py`, `swingforge/web/static/index.html` | dashboard | WU-3C |
| `deploy/*.service`, `deploy/backfill.timer`, `deploy/README.md` | systemd units, VPS runbook | WU-4A |

---

## 2. Wave 0 — Contract (one agent, Opus, sequential)

### WU-0: Toolchain + frozen interfaces
**Files:** create everything in the WU-0 rows above plus `tests/unit/test_types.py`, `tests/unit/test_contract_imports.py`.

- [ ] `uv init --python 3.12`, add deps: pydantic>=2.7, numpy, duckdb, typer, fastapi, uvicorn, hypothesis, pytest, pytest-cov, vcrpy, import-linter, ruff. Commit `chore: toolchain`.
- [ ] `.importlinter` layers contract exactly as spec §2. `tests/unit/test_contract_imports.py` shells out to `lint-imports` and asserts exit 0.
- [ ] `core/types.py` — models verbatim from spec §3. Validators: `Signal` raises if `(entry - stop) * direction <= 0`; `Order.qty > 0`; any price `math.isfinite`. `CONTRACT_VERSION = 1` in `core/__init__.py`.
- [ ] `tests/unit/test_types.py`:
  ```python
  def test_signal_stop_wrong_side_rejected():
      with pytest.raises(ValidationError):
          Signal(direction=1, entry=100.0, stop=101.0, structure_target=None, tag="t", expires_in_bars=3)
  def test_signal_stop_equal_entry_rejected(): ...   # stop == entry
  def test_bar_nan_rejected(): ...
  def test_order_zero_qty_rejected(): ...
  ```
- [ ] `core/context.py` — `Context` with `bars(tf) -> np.ndarray` (OHLCV columns), `atr(tf, n=14)`, `adx(tf, n=14)`, `position: Position | None`, `instrument`, `bar_index`. Tests: ATR/ADX against hand-computed 20-bar fixture, tolerance 1e-9.
- [ ] `adapters/base.py`, `strategies/base.py` — protocols verbatim from spec §4/§5. `ExitRule` has `attach(trade, ctx) -> list[Order]` and `on_bar(trade, ctx) -> list[Order]`.
- [ ] Commit `feat(core): frozen contract v1`. Tag `contract-v1`. **Gate: orchestrator reviews the contract personally before Wave 1 dispatch.**

---

## 3. Wave 1 — Five parallel agents, no shared files

### WU-1A: FillResolver + cost math (Opus)
**Files:** `core/fills.py`, `core/costs.py`, `tests/unit/test_fills.py`, `tests/unit/test_costs.py`
- [ ] Test: bar with SL and TP both inside; subbars show TP touched first → resolver returns `("target", price)`; subbars show SL first → `("stop", price)`.
- [ ] Test: no subbars → always `("stop", ...)` and `resolution_mode == "pessimistic"`.
- [ ] Hypothesis property: for any bar with both levels inside `[low, high]`, result is exactly one of stop/target, never both/neither.
- [ ] Test: two exit legs (Partial): first leg resolved before second on same bar; stop-to-breakeven applied before evaluating the second leg.
- [ ] Implement `FillResolver.resolve(bar, stop, targets: list[float], direction) -> Resolution`. Resolution carries `mode`.
- [ ] `costs.py`: `CostBreakdown.total`, `NullCostModel` (zeros) for unit tests elsewhere. Test totals.
- [ ] Commit `feat(core): fill resolver and cost math`.

### WU-1B: Portfolio + Engine + Settings (Sonnet)
**Files:** `core/portfolio.py`, `core/engine.py`, `core/settings.py`, tests for each.
- [ ] `Settings` model: `risk_pct: float = Field(0.01, le=0.02)`, `enabled_instruments`, `paper_configs`, `session`, `time_stop_bars: int = 10`, `kill_switch: bool`. Test: `risk_pct=0.03` raises.
- [ ] `Portfolio.size(signal, equity, instrument) -> qty`: `risk = equity * risk_pct; qty = risk / abs(entry - stop) / contract_multiplier`, rounded down to tick. Test with hand numbers.
- [ ] `Portfolio.record(trade)` updates equity and `realized_r`. Test R accounting over a 3-trade sequence (+2R, −1R, +0.5R → equity path).
- [ ] `Engine.run(bars: Iterable[Bar], strategy, exit_rule, broker, portfolio, settings_reader)`: per bar: refresh settings (if reader returns new version), update ctx, call `exit_rule.on_bar` for open trade, then `strategy.on_bar` if flat and not killed, size, submit. Uses a `FakeBroker` in tests (fills every limit immediately at price). Test: kill switch true → no new entries, open trade still managed.
- [ ] Test: settings change applied at next bar, not mid-bar (reader returns new value while a bar is being processed; assert the bar that was in flight used the old value).
- [ ] Commit `feat(core): portfolio, engine, settings`.

### WU-1C: Exit rules (Sonnet)
**Files:** `strategies/exits.py`, `tests/unit/test_exits.py`
- [ ] Synthetic path helper: `path(prices: list[float]) -> list[Bar]` with `subbars=()`.
- [ ] `FixedR(2)`: entry 100, stop 95 → target 110. Test attach returns one stop + one target order at those prices.
- [ ] `ATRFixedR(rr=2, atr_mult=1.5)`: with ATR 4 → stop 94, target 112, overriding the signal's stop. Test.
- [ ] `Structure(fallback_rr=2)`: `structure_target` 1.4R away → fallback used; 1.6R away → structure used. Two tests.
- [ ] `Trailing(activate_r=1, trail_atr=2)`: stop unchanged until +1R; after that, on each bar `stop = max(stop, high - 2*ATR)` for longs. Property: stop is monotone non-decreasing for longs.
- [ ] `Partial(scale_r=1, scale_pct=0.5, runner=FixedR(3))`: at +1R emits partial close 50% and a stop-modify to entry; remainder targets 3R. Test order sequence.
- [ ] Time stop: bar 10 with |unrealized| 0.4R → market close order; 0.6R → none. Two tests.
- [ ] `EXIT_GRID` list constant with the 16 variants from spec §4; test `len == 16`.
- [ ] Commit `feat(strategies): exit rules`.

### WU-1D: Gate math + regime (Opus)
**Files:** `lab/gate.py`, `lab/regime.py`, tests.
- [ ] `deflated_sharpe(returns, n_trials, skew, kurt) -> (dsr, p)` per Bailey & López de Prado 2014. Test against the paper's worked example (SR 2.5 annualised, 1,000 trials → known DSR); tolerance 1e-3.
- [ ] `stationary_bootstrap(x, n=1000, p=0.1, seed)` → 5th-percentile mean. Test: constant series returns the constant; seed reproducibility.
- [ ] `mar(equity_curve)`. Test on a hand curve.
- [ ] `GateResult` with per-rule booleans + values. `evaluate(config_trades, baseline_trades, bh_curve, n_trials=2016, cost_stress: bool)`. Tests: 59 trades fails rule 1 only-other-rules-unevaluated; a planted +0.3R mean series with n=200 passes rules 2–3; identical trades vs baseline fails rule 4.
- [ ] `regime.py`: `tag(ctx) -> str` in `{"trend_lowvol", ..., "range_highvol"}`. Test on fixtures where ADX is known.
- [ ] Commit `feat(lab): gate and regime`.

### WU-1E: Store + Replay + PaperBroker (Sonnet)
**Files:** `adapters/store.py`, `adapters/replay.py`, `adapters/paper.py`, tests.
- [ ] `store.py`: `Store(path)`, `create_schema()` with all tables from spec §5, `upsert_bars(list[Bar])` idempotent (test: insert twice → count unchanged), readers `bars(instrument, tf, start, end)`.
- [ ] `replay.py`: `ReplaySource(store)`; `stream(instrument,"4h")` yields 4H bars in ts order, each with `subbars` = the 4 matching 1H bars if present. Test: subbar timestamps all within `[ts_open, ts_open+4h)`; test: a 4H bar is never yielded before all its 1H bars exist.
- [ ] `paper.py`: `PaperBroker(source, cost_model)`; `submit` limit → pending; `on_bar(bar)` fills pending limits if `low <= price <= high` at limit price (+ slippage from cost model), expires by `expires_at_bar`, market orders fill at `close`. Tests for each transition; test fill carries `CostBreakdown`.
- [ ] Commit `feat(adapters): store, replay, paper broker`.

**Wave 1 merge gate:** all five branches reviewed and merged; `uv run pytest tests/unit -q` green on `main`; `lint-imports` green.

---

## 4. Wave 2 — Three parallel agents

### WU-2A: Entry strategies + session filter (Sonnet, with ICT fixtures reviewed by Opus)
**Files:** `strategies/ict.py`, `strategies/zones.py`, `strategies/baseline.py`, `strategies/session.py`, `tests/unit/test_ict.py`, `test_zones.py`, `test_baseline.py`, `test_session.py`, `tests/fixtures/ict_*.json`.
- [ ] Fixtures (hand-built 4H+Daily bar sequences, JSON): `clean_sweep_mss_ob`, `pullback_not_sweep` (v0.3 regression), `sweep_no_mss_within_6`, `bos_neckline_boundary` (off-by-one regression).
- [ ] `ict.py`: port corrected v0.3 ICTAnalyzer logic; `on_bar` returns `Signal` only on `clean_sweep_mss_ob` fixture, `None` on the other three. Stop = sweep extreme ± tick. `structure_target` = nearest opposing Daily swing.
- [ ] `zones.py`: fixture with a ≥2 ATR Daily impulse; first 4H close inside zone → signal at midpoint, stop beyond far edge + 0.25 ATR; second touch → `None`. Round-number filter test: zone midpoint 1.2003 with filter on → `None` unless within 5 pips of 1.2000.
- [ ] `baseline.py`: `Baseline(target_trades_per_1000_bars, seed)`; over 10k bars count within ±10%, direction share within 45–55%.
- [ ] `session.py`: `allowed(bar, profile, mode) -> bool`; table-driven tests for `none/london_ny/active` × `fx/perp` incl. Saturday bar on perp `active` → False.
- [ ] Commit `feat(strategies): ict, zones, baseline, session`.

### WU-2B: Venue adapters (Sonnet)
**Files:** `adapters/hyperliquid/bars.py`, `adapters/hyperliquid/costs.py`, `adapters/oanda/bars.py`, `adapters/oanda/costs.py`, `tests/contract/*` (vcrpy cassettes recorded once by the human with real credentials; CI runs replay only).
- [ ] HL `bars.py`: wraps toolkit candle client → `Bar` (UTC, `bid/ask None`). `history` paginates; `stream` polls `history` each bar close (HL candle WS optional later). Contract test: 4H bars open at 00/04/08/…; no gaps in a 30-day cassette.
- [ ] HL `costs.py`: taker fee from toolkit schedule; `carry` = funding rate × notional at 8h boundaries. Test: no funding between boundaries.
- [ ] OANDA `bars.py`: `oandapyV20` `InstrumentsCandles` with `price="BA"`; mid OHLC + `bid_close/ask_close`. Daily re-cut: fetch 1H, aggregate to 00:00-UTC days. Contract test: re-cut daily high == max of its 24 1H highs.
- [ ] OANDA `costs.py`: spread = `(ask_close - bid_close)/2` × qty at fill; swap via `AccountInstruments` financing at 21:00 UTC. Test timing.
- [ ] Commit `feat(adapters): hyperliquid and oanda`.

### WU-2C: Tournament + excursion + report (Opus)
**Files:** `lab/tournament.py`, `lab/excursion.py`, `lab/report.py`, tests using `NullCostModel` and a synthetic `Store`.
- [ ] `configs()` → list of `Config(entry, exit, session, instrument)`; test `len == 3*16*3*len(instruments)`.
- [ ] `walk_forward_splits(start, end)`: anchored IS 12m / OOS 3m step 3m; test on a 4-year range yields the expected split count and no OOS overlap.
- [ ] `run_config(config, store, split)`: builds Engine + ReplaySource + PaperBroker, returns trades tagged IS/OOS + regime. Test: trades in OOS have `ts >= oos_start`.
- [ ] `run_tournament(store, instruments)`: loops, writes `results` rows (config, split, n_is, n_oos, exp_is, exp_oos, dsr, boot_p5, gate booleans, resolution_mode). Test on a 2-instrument synthetic store that rows == configs × splits. Excludes instruments with < 18 months and records them in `excluded`.
- [ ] `excursion.py`: `mae_mfe(trade, bars)`, `stopped_out_of_winner_rate(trades)`, `median_exit_efficiency(trades)`; test on hand trades.
- [ ] `report.py`: renders markdown sections from spec §6 "Outputs"; snapshot test.
- [ ] Commit `feat(lab): tournament, excursion, report`.

**Wave 2 merge gate:** as Wave 1.

---

## 5. Wave 3 — Three parallel agents

### WU-3A: CLI + backfill (Sonnet)
**Files:** `swingforge/cli.py`, `tests/unit/test_cli.py`
- [ ] `swingforge backfill --venue oanda|hyperliquid --years 4` → idempotent, prints per-instrument bar counts and months available.
- [ ] `swingforge tournament --venue ... --out reports/` → runs, writes report + results.
- [ ] `swingforge paper --venue ... --config <id>` → Engine on `stream()` with PaperBroker, settings from store, logs to `equity`.
- [ ] `swingforge web --host --port`. Tests via `typer.testing.CliRunner` with monkeypatched adapters.
- [ ] Commit `feat(cli)`.

### WU-3B: Integration tests (Opus)
**Files:** `tests/integration/test_determinism.py`, `test_no_lookahead.py`, `test_reconciliation.py`, `test_planted_edge.py`, `tests/integration/synth.py`
- [ ] `synth.py`: random-walk generator; `planted(edge=True)` injects +N-bar momentum after synthetic Daily-sweep patterns so `ict` has a real edge.
- [ ] Determinism: two `run_config` runs on the same store → `trades` tables equal (`duckdb` `EXCEPT` both ways empty).
- [ ] No-lookahead: mutate all bars after T (shift prices +5%); trades with entry ≤ T identical.
- [ ] Reconciliation: `PaperBroker` fed by `ReplaySource.stream` vs a `FakeStream` yielding the same bars → identical fills.
- [ ] Planted edge: tournament on `planted(True)` → at least one `ict` config passes all six gate rules and no `baseline` config passes; on `planted(False)` → zero passes. Mark `@pytest.mark.slow`.
- [ ] Commit `test(integration)`.

### WU-3C: Dashboard (Sonnet)
**Files:** `web/app.py`, `web/static/index.html`, `tests/unit/test_web.py`
- [ ] Routes from spec §7. Bearer token middleware when `SWINGFORGE_TOKEN` set. `GET /api/overview`, `/api/tournament/latest`, `/api/trades/{id}`, `/api/settings`, `PUT /api/settings`.
- [ ] Tests: `PUT` risk 0.03 → 422; each `PUT` → exactly one `settings_log` row; instrument with last bar > 2 bar-lengths old → `stale: true`; unauthenticated with token set → 401.
- [ ] `index.html`: three views, Recharts equity curve, polling 10s, settings form. Manual check only.
- [ ] Commit `feat(web): dashboard`.

**Wave 3 merge gate:** full suite incl. `slow` green; coverage ≥ 85% on `core`, `lab`.

---

## 6. Wave 4 — Deployment (one agent + human)

### WU-4A: VPS deployment (Sonnet; human executes the credential steps)
**Files:** `deploy/swingforge-paper@.service` (templated per venue), `deploy/swingforge-web.service`, `deploy/swingforge-backfill.service` + `.timer` (daily 00:10 UTC), `deploy/README.md`, `deploy/install.sh`.
- [ ] `install.sh`: creates `swingforge` user, clones repo to `/opt/swingforge`, `uv sync`, copies units, `systemctl enable --now`.
- [ ] Units: `Restart=on-failure`, `EnvironmentFile=/etc/swingforge.env` (OANDA practice token, HL read-only, `SWINGFORGE_TOKEN`), `WorkingDirectory=/opt/swingforge`, `ExecStart=/opt/swingforge/.venv/bin/swingforge paper --venue %i`.
- [ ] Web unit binds `0.0.0.0:8787`; README documents `ufw allow from <your-ip> to any port 8787` — no reverse proxy/TLS in v1 since the token + IP allowlist is enough for a single-user paper dashboard. Note the upgrade path (Caddy + TLS) for when live is specced.
- [ ] README runbook: backfill → tournament → read report → `PUT /api/settings` to enable gate-passed configs → paper runs → 60-day/30-trade check → reconciliation command.
- [ ] Human step: record vcrpy cassettes with real credentials once; commit cassettes (no secrets — vcrpy filter headers).
- [ ] Commit `chore(deploy)`.

**Go-live of paper (not live trading):** backfill both venues → `tournament` → report reviewed by human → configs enabled in dashboard → paper services started. Anything that changes the gate or engine after this point re-runs the planted-edge test before deploy.

---

## 7. Self-review against the spec

- Spec §2 layout → WU-0/1/2/3 file table covers every path. ✔
- §3 types → WU-0. §4 strategies + exits + session → WU-1C, WU-2A. §5 adapters/replay/paper/fills → WU-1A, WU-1E, WU-2B. §6 tournament/gate/regime/excursion/report/promotion → WU-1D, WU-2C, WU-4A runbook. §7 dashboard → WU-3C. §8 testing → WU-3B + per-WU tests; contract tests in WU-2B. ✔
- Placeholder scan: no TBD/TODO. Concrete grids, thresholds, and assertions are given; each WU's subagent writes the code from these tests, which is the intended TDD split.
- Type consistency: `Signal.structure_target`, `ExitRule.attach/on_bar`, `FillResolver.resolve(...) -> Resolution(mode)`, `Store.upsert_bars`, `ReplaySource.stream`, `PaperBroker.on_bar` used consistently across WUs. Any rename is a contract bump (§0).
- Scope: one repo, one spec, one plan — paper only. Live execution is explicitly out.
