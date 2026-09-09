# WU-2A — entry strategies + session filter

Built: `strategies/levels.py`, `ict.py`, `zones.py`, `baseline.py`, `session.py`; 4 hand-built fixtures in `tests/fixtures/ict_*.json` with a JSON-driven loader in `test_ict.py`. Plus one code-review fix round (below). 81 tests across the 5 strategy files (318 total, full suite green); coverage: levels/session/baseline 100%, ict/zones 95%.

`ICT(k=2, mss_within=6, expires_in_bars=3, min_pierce_pct=0.0003, min_bos_pct=0.0002)`, `Zones(impulse_atr_mult=2.0, impulse_bars=3, stop_atr_pad=0.25, round_number_filter=False, expires_in_bars=3)`, `Baseline(target_trades_per_1000_bars, seed, stop_atr_mult=1.5, expires_in_bars=3)`, `SessionFilter(mode, profile)`. `name` = `"ict"`/`"zones"`/`"baseline"`; all four expose `reset()` (`SessionFilter.reset()` is a documented no-op — it carries no state).

ICT (SHURIKEN v0.3 corrected): liquidity range = nearest Daily swing high/low, served from an incremental cache (`_swing_highs_cache`/`_swing_lows_cache`, keyed on the last Daily bar's `ts_open`) instead of rescanning `levels.py`'s fractal pivots from scratch every 4H bar; sweep = a candle opening AND closing inside the range but wicking past one edge (low-sweep=long, high-sweep=short); BOS = a later candle (`i>=2`, within `mss_within` bars) closing above/below the extreme of the bars strictly BEFORE it (neckline updates after the check). Entry = OB midpoint else FVG midpoint else no signal; fires only when the BOS bar is the current bar; `_last_signalled_sweep` (an absolute `bar_index`, translated via `ctx.offset_of` — trimming-safe) blocks re-emission for the same sweep.

Zones: impulse = `|close[end]-close[i0]| >= impulse_atr_mult * ATR(14,D)[end]` for the nearest `i0` within `impulse_bars`, using a local causal Wilder ATR carried forward one step per new Daily bar (never rebuilt), tracked by `_daily_last_ts` rather than a row count so it keeps advancing after the buffer saturates under trimming; zone = nearest opposing-colour candle at/before `i0`, full `[low,high]`. `fresh` flips False the first time ANY 4H bar that opened at or after the zone was born (its Daily bar's close, `_Zone.born_ts`) overlaps it, signal or not - whether or not the engine consulted the strategy on that bar: `on_bar` replays every unseen 4H bar across a consult gap first (post-plan fix 2026-09-09, below); newest 20 kept, and a re-detected zone (same opposing candle, `_Zone.source_ts`) is neither appended again nor refreshed. Perp round-number filter: `step = 10**(floor(log10(price))-1)`, levels at multiples of `step` and `step/2`, tolerance 0.1% of price (replaces the old fixed {1,2,5}x10^n table).

Baseline: `numpy.random.default_rng(seed)` draws a fire roll AND a direction roll every bar unconditionally (whether or not the bar fires), so RNG stream position is a pure function of `(seed, bars presented)` only; stop distance is deterministic (`stop_atr_mult` Daily ATRs), never itself drawn. `reset()` re-seeds identically. Session: `none`->True; `london_ny`/`active`+fx -> `[07:00,21:00)` UTC; `active`+perp -> excludes `[00:00,07:00)` UTC and Sat/Sun.

Fix round (code review): ICT's swing-level cache and Zones' local ATR are now incremental — pushing 8,760 4H + 1,460 Daily bars through `on_bar` (was O(n^2)): ICT 65.5s -> 0.91s, Zones 1.24s -> 0.49s (measured this round via `git stash`, not the reviewer's original numbers). Each strategy now has a `rejected_signals` counter (reset by `reset()`). The BOS-neckline regression test was rewritten to a shape that actually discriminates the fix from a `max(prior, own_high)` bug (confirmed both ways in an uncommitted scratch script); the fixture's own note is corrected around idx34.

Gotchas: ICT needs >=30 4H bars and >=2k+3 Daily bars; Zones/Baseline return None (never a degenerate signal) when `ctx.atr("1d")` is NaN. Small decisions: zone round-number filter scales BOTH the level step and tolerance x100 for JPY quotes; a zone's freshness is consumed by `on_bar` before the round-number filter is checked, so a filtered-out touch still can't be reused. Not built: tournament wiring / engine integration (out of scope; `Strategy` protocol conformance is already covered by WU-0's `test_protocols.py`).

**Post-plan fix (2026-09-09, zones).** Two defects confirmed on the first real Hyperliquid tournament (2026-09-07): (1) freshness was consumed only inside `Zones.on_bar`, but the engine consults a strategy only when flat, with no pending entry, kill switch off and in session, so session-blocked bars silently preserved zones price had already visited (38 BTC / 47 SOL unconsumed touches under `active`, which then out-traded `none`); (2) consecutive Daily impulses re-selecting the same opposing candle appended exact duplicate zones (26 of BTC's 54), crowding the 20-zone cap. Now: `on_bar` replays every 4H bar since the last consult (`_consume_unseen_touches`, keyed by absolute `bar_index` so the every-bar path pays nothing; the first consult replays the retained window, so the tournament's warm-up bars consume freshness too and the `none` arm shifts as well - intended), a zone is born when its Daily bar closes and only bars opening at or after that can touch it, and zone identity is the source candle's `ts_open`. `core/engine.py` and `strategies/base.py` are untouched. Before/after zone trade counts per session are recorded in the re-run section at the end of this note.

**Post-plan fix (2026-09-09, ICT).** Confirmed on the first real Hyperliquid tournament (2026-09-07): the liquidity range was read off the current bar's close and every lookback bar had to open and close inside it, so once the displacement after a sweep closed past the next Daily pivot (they sit ~2.5% apart on real data) the range moved away from the sweep bar and the setup was lost - ~65% of spec-valid sweep->BOS setups. `find_sweep_against(bars, range_at, ...)` now judges each candidate against the range around its own body (nearest swing low below the body, nearest swing high above it, from the pivots confirmed as of the current bar); anchoring on the close instead rejected genuine sweeps whose open sat above the pivot nearest their close (planted-store hit rate 34%/18% vs 100% body-anchored). Level caches are kept sorted and bisected. Everything else - N=6, the 12-bar lookback, newest-first, the v0.3 guards, fire-once - is unchanged. **Open decision, not part of this fix:** with pivot ranges the newest-first scan lets a newer sweep with no BOS yet shadow an older setup on the very bar it confirms (the investigation's second-largest loss). Walking every candidate and firing for the newest one whose BOS is the current bar recovers it, and on the synthetic stores it raises ICT's pure-random-walk sample from ~16 to 246 OOS trades (so the null control rejects on statistics rather than on rule 1) - but it also multiplies unplanted setups ~2.7x (planted store: n_oos 220 -> 250, exp_oos 0.94 -> 0.77, edge/flat signal parity 1.2% -> 6.1%), which is a strategy-definition change the spec's owner should make deliberately. Before/after ICT trade counts are recorded in the re-run section at the end of this note.

## Re-run after the 2026-09-09 fixes

Three runs on the same store (5 Hyperliquid instruments, 27 months of 4H bars, seed 0): `tournament:hyperliquid:20260907` (original; carries the same-bar fill artefact, see the WU-1E handoff), `tournament:hyperliquid:20260909-broker` (old strategies, corrected broker - the honest "before"), `tournament:hyperliquid:20260909` (corrected broker plus the ICT and zones fixes above). `tournament:hyperliquid:20260909-fills` also exists in the store; it used an intermediate broker and is superseded. Nothing passes the gate in any of them: the largest pooled OOS count went from 13 to 15 against rule 1's 60, so the verdict is still a sample-size verdict (see `README.md`).

ICT, per config, averaged over the 16 exits (before -> after; `n_is` is 12 months, `n_oos` 15 months pooled):

| session | avg n_is | avg n_oos | avg exp_oos (R) |
|---|---|---|---|
| none | 13.1 -> 19.2 | 7.1 -> 12.5 | -0.18 -> -0.14 |
| london_ny | 8.8 -> 12.8 | 4.1 -> 7.4 | -0.10 -> +0.10 |
| active | 5.1 -> 7.5 | 2.1 -> 3.6 | -0.20 -> +0.23 |

| instrument (session none) | avg n_is | avg n_oos | max n_oos | avg exp_oos (R) |
|---|---|---|---|---|
| ARB | 13.0 -> 19.0 | 8.0 -> 14.0 | 8 -> 14 | +0.56 -> +0.16 |
| BTC | 14.3 -> 28.1 | 6.0 -> 14.0 | 6 -> 14 | -0.43 -> -0.38 |
| ETH | 11.4 -> 13.0 | 11.7 -> 14.6 | 13 -> 15 | -0.22 -> +0.05 |
| HYPE | 12.0 -> 13.6 | 4.9 -> 8.9 | 5 -> 9 | -0.39 -> -0.16 |
| SOL | 14.6 -> 22.0 | 5.0 -> 11.0 | 5 -> 11 | -0.43 -> -0.38 |

Zones, per config, averaged over the 16 exits (total trades over the 27 months, pooled OOS, expectancy):

| session | avg trades | avg n_oos | avg exp_oos (R) |
|---|---|---|---|
| none | 9.8 -> 10.8 | 3.4 -> 4.0 | -0.34 -> -0.35 |
| london_ny | 9.8 -> 7.6 | 3.2 -> 2.8 | -0.44 -> -0.54 |
| active | 10.3 -> 6.5 | 3.2 -> 2.6 | -0.45 -> -0.53 |

The session-gated arms lose the trades that only existed because a session-blocked bar had left a zone fresh; `active` no longer out-trades `none`. The frequency-matched baseline follows ICT up (avg n_oos 3.6 -> 5.4). The best rows after the fixes are ICT on ETH under `london_ny` (n_oos 10, exp_oos +0.6 to +1.0R across four exits) and ICT on ARB with no filter (n_oos 14, +0.5 to +0.7R) - suggestive, and an order of magnitude short of the sample the gate needs.

