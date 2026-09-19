# WU-2A — entry strategies + session filter

Built: `strategies/levels.py`, `ict.py`, `zones.py`, `baseline.py`, `session.py`; 4 hand-built fixtures in `tests/fixtures/ict_*.json` with a JSON-driven loader in `test_ict.py`. Plus one code-review fix round (below). 81 tests across the 5 strategy files (318 total, full suite green); coverage: levels/session/baseline 100%, ict/zones 95%.

`ICT(k=2, mss_within=6, expires_in_bars=3, min_pierce_pct=0.0003, min_bos_pct=0.0002)`, `Zones(impulse_atr_mult=2.0, impulse_bars=3, stop_atr_pad=0.25, round_number_filter=False, expires_in_bars=3)`, `Baseline(target_trades_per_1000_bars, seed, stop_atr_mult=1.5, expires_in_bars=3)`, `SessionFilter(mode, profile)`. `name` = `"ict"`/`"zones"`/`"baseline"`; all four expose `reset()` (`SessionFilter.reset()` is a documented no-op — it carries no state).

ICT (SHURIKEN v0.3 corrected): liquidity range = nearest Daily swing high/low, served from an incremental cache (`_swing_highs_cache`/`_swing_lows_cache`, keyed on the last Daily bar's `ts_open`) instead of rescanning `levels.py`'s fractal pivots from scratch every 4H bar; sweep = a candle opening AND closing inside the range but wicking past one edge (low-sweep=long, high-sweep=short); BOS = a later candle (`i>=2`, within `mss_within` bars) closing above/below the extreme of the bars strictly BEFORE it (neckline updates after the check). Entry = OB midpoint else FVG midpoint else no signal; fires only when the BOS bar is the current bar; `_last_signalled_sweep` (an absolute `bar_index`, translated via `ctx.offset_of` — trimming-safe) blocks re-emission for the same sweep.

Zones: impulse = `|close[end]-close[i0]| >= impulse_atr_mult * ATR(14,D)[end]` for the nearest `i0` within `impulse_bars`, using a local causal Wilder ATR carried forward one step per new Daily bar (never rebuilt), tracked by `_daily_last_ts` rather than a row count so it keeps advancing after the buffer saturates under trimming; zone = nearest opposing-colour candle at/before `i0`, full `[low,high]`. `fresh` flips False the first time ANY 4H bar that opened at or after the zone was born (its Daily bar's close, `_Zone.born_ts`) overlaps it, signal or not - whether or not the engine consulted the strategy on that bar: `on_bar` replays every unseen 4H bar across a consult gap first (post-plan fix 2026-09-09, below); newest 20 kept, and a re-detected zone (same opposing candle, `_Zone.source_ts`) is neither appended again nor refreshed. Perp round-number filter: `step = 10**(floor(log10(price))-1)`, levels at multiples of `step` and `step/2`, tolerance 0.1% of price (replaces the old fixed {1,2,5}x10^n table).

Baseline: `numpy.random.default_rng(seed)` draws a fire roll AND a direction roll every bar unconditionally (whether or not the bar fires), so RNG stream position is a pure function of `(seed, bars presented)` only; stop distance is deterministic (`stop_atr_mult` Daily ATRs), never itself drawn. `reset()` re-seeds identically. Session: `none`->True; `london_ny`/`active`+fx -> `[07:00,21:00)` UTC; `active`+perp -> excludes `[00:00,07:00)` UTC and Sat/Sun.

Fix round (code review): ICT's swing-level cache and Zones' local ATR are now incremental — pushing 8,760 4H + 1,460 Daily bars through `on_bar` (was O(n^2)): ICT 65.5s -> 0.91s, Zones 1.24s -> 0.49s (measured this round via `git stash`, not the reviewer's original numbers). Each strategy now has a `rejected_signals` counter (reset by `reset()`). The BOS-neckline regression test was rewritten to a shape that actually discriminates the fix from a `max(prior, own_high)` bug (confirmed both ways in an uncommitted scratch script); the fixture's own note is corrected around idx34.

Gotchas: ICT needs >=30 4H bars and >=2k+3 Daily bars; Zones/Baseline return None (never a degenerate signal) when `ctx.atr("1d")` is NaN. Small decisions: zone round-number filter scales BOTH the level step and tolerance x100 for JPY quotes; a zone's freshness is consumed by `on_bar` before the round-number filter is checked, so a filtered-out touch still can't be reused. Not built: tournament wiring / engine integration (out of scope; `Strategy` protocol conformance is already covered by WU-0's `test_protocols.py`).

**Post-plan fix (2026-09-09, zones).** Two defects confirmed on the first real Hyperliquid tournament (2026-09-07): (1) freshness was consumed only inside `Zones.on_bar`, but the engine consults a strategy only when flat, with no pending entry, kill switch off and in session, so session-blocked bars silently preserved zones price had already visited (38 BTC / 47 SOL unconsumed touches under `active`, which then out-traded `none`); (2) consecutive Daily impulses re-selecting the same opposing candle appended exact duplicate zones (26 of BTC's 54), crowding the 20-zone cap. Now: `on_bar` replays every 4H bar since the last consult (`_consume_unseen_touches`, keyed by absolute `bar_index` so the every-bar path pays nothing; the first consult replays the retained window, so the tournament's warm-up bars consume freshness too and the `none` arm shifts as well - intended), a zone is born when its Daily bar closes and only bars opening at or after that can touch it, and zone identity is the source candle's `ts_open`. `core/engine.py` and `strategies/base.py` are untouched. Before/after zone trade counts per session are recorded in the re-run section at the end of this note.

**Post-plan fix (2026-09-09, ICT).** Confirmed on the first real Hyperliquid tournament (2026-09-07): the liquidity range was read off the current bar's close and every lookback bar had to open and close inside it, so once the displacement after a sweep closed past the next Daily pivot (they sit ~2.5% apart on real data) the range moved away from the sweep bar and the setup was lost - ~65% of spec-valid sweep->BOS setups. `find_sweep_against(bars, range_at, ...)` now judges each candidate against the range around its own body (nearest swing low below the body, nearest swing high above it, from the pivots confirmed as of the current bar); anchoring on the close instead rejected genuine sweeps whose open sat above the pivot nearest their close (planted-store hit rate 34%/18% vs 100% body-anchored). Level caches are kept sorted and bisected. Everything else - N=6, the 12-bar lookback, newest-first, the v0.3 guards, fire-once - is unchanged. **Open decision, not part of this fix (adopted on 2026-09-19, see the end of this note):** with pivot ranges the newest-first scan lets a newer sweep with no BOS yet shadow an older setup on the very bar it confirms (the investigation's second-largest loss). Walking every candidate and firing for the newest one whose BOS is the current bar recovers it, and on the synthetic stores it raises ICT's pure-random-walk sample from ~16 to 246 OOS trades (so the null control rejects on statistics rather than on rule 1) - but it also multiplies unplanted setups ~2.7x (planted store: n_oos 220 -> 250, exp_oos 0.94 -> 0.76, edge/flat signal parity 1.2% -> 6.1%), which is a strategy-definition change the spec's owner should make deliberately. Before/after ICT trade counts are recorded in the re-run section at the end of this note.

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

## Candidate walk adopted (2026-09-19)

The owner's decision on the open item above: `ICT.on_bar` now walks its sweep candidates newest-first and fires for the newest one **whose break of structure is the current bar**, instead of stopping at the newest sweep and giving up when it has no break yet (the reference's shape, right for one fixed Asia range per day, wrong for pivot ranges where a fifth of all 4H bars are sweeps). `iter_sweeps_against` yields every sweep (`find_sweep_against` is its first result, `find_sweep` the fixed-range form, both unchanged in behaviour). `find_bos` can only place a break `_BOS_MIN_INDEX + 1` = 3 to `mss_within` = 6 bars after its sweep, so the walk (`find_setup_breaking_now`) looks at exactly those four bars (`skip_newest=3`, lookback `min(_SWEEP_LOOKBACK, mss_within + 1)`, so never further back than the old scan): equivalent to walking all twelve, which a unit test and the spec review's differential run (840,000 strategy-bars, `mss_within` 2 to 20, no divergence) both check. **Cost.** On data with no sweeps it is faster than the scan it replaces (1.4s against 2.0s on the eight-year sine series, which times the scan and nothing else). On sweep-rich data the bare walk is slower, because it has nearly two candidates a bar to ask `find_bos` about; `_closes_through_neckline` - does the newest bar close through the running extreme since the sweep, one slice - is the necessary condition for the newest bar to be a sweep's break, and with it `find_bos` only confirms that no earlier bar broke first. Measured on an eight-year random walk with its own Daily bars: 3.2s and 1,831 `find_bos` calls with the precheck, 5.2s and 26,527 calls without, 3.2s for the newest-first scan - and 1,472 signals against the old scan's 156; a second performance test now covers that regime. **Tie-break.** Several sweeps break on the same bar on about a fifth of the bars that fire; the newest wins (the last grab of liquidity before the break, and the reference's own preference), which also sets the stop - the newer sweep's wick is the nearer one more often than not (median risk 0.89x, review measurement), so the position is correspondingly larger at a fixed `risk_pct`. **Fire-once.** For a fixed direction a sweep's break is the *first* bar that qualifies, which later bars never change - but the direction is re-read every bar against pivots that keep being confirmed, and the review saw it flip on 3 of ~10,800 sweep bars, so the guarantee does not rest on that: every sweep signalled within the lookback is remembered (`_signalled_sweeps`, absolute indices, pruned past the lookback; `_last_signalled_sweep` is the most recent) and passed over, and a bar that has produced a signal produces no second one on a repeat call. Everything else - the body-anchored range, N=6, the v0.3 guards, OB before FVG, stop beyond the sweep wick - is unchanged, and the three v0.3 regression fixtures and the pullback guard stay green (the fixtures still discriminate under the walk - re-injecting the reversed mapping or the eager neckline fails 14 and 13 tests; the fixture-level pullback test is vacuous at the `on_bar` level, as before, because the body-anchored range makes that branch unreachable, and the direct `find_sweep` test is what guards it).

What it costs, measured on the synthetic stores (newest-first -> walk), and accepted with the decision:

| store (4 years, `fixed_r_2`, no session filter) | ICT signals | n_oos | exp_oos (R) | gate |
|---|---|---|---|---|
| planted edge | 636 -> 1,100 (347 planted, all hit, both ways) | 220 -> 250 | +0.94 -> +0.76 | passes (dsr prob 0.9999) -> passes (0.993) |
| planted control (no drift) | 644 -> 1,171 (363 planted, all hit) | 227 -> 265 | -0.06 -> +0.005 | fails -> fails |
| pure random walk | not counted | 16 -> 246 | +0.19 -> -0.02 | fails on rule 1 -> fails on rules 2, 3, 4 and 6 |

The setups nobody planted multiply about 2.7x (289 -> 753 and 281 -> 808), which dilutes the planted store's expectancy and moves the edge/flat signal parity from 1.2% to 6.1% (the plant itself differs by 4.4%; `test_both_stores_offer_ict_the_same_entries` now allows 10%, and separately asserts that every planted setup is taken in the store it was planted in - which is not a substitute for the looser parity bound, only a guarantee that neither plant is under-sampled). In exchange the null controls have teeth: on a pure random walk ICT holds ~240 OOS trades at about zero expectancy and is rejected by the statistics, where before it never reached the trade-count floor. The real-data before/after is in the re-run section that follows.

## Re-run with the candidate walk (2026-09-20)

`tournament:hyperliquid:20260920`, the same store replayed on `main` at `381d6d7` (seed 0), against `tournament:hyperliquid:20260919` (newest-first scan, same gate, same pooling): 901 trials (751 per instrument and view, 150 universe), `trial_sr_variance` 0.0151, **0 passed**. Zones was not touched and its 289 rows are identical in both runs; the frequency-matched baseline follows ICT's new rate.

Per instrument and config, averaged over the 16 exits (before -> after):

| entry | session | avg n_is | avg n_oos | max n_oos | avg exp_oos (R) | rows clearing rule 1 |
|---|---|---|---|---|---|---|
| ict | none | 19.1 -> 64.9 | 12.5 -> 46.2 | 15 -> 65 | -0.14 -> -0.18 | 0 -> 12 of 80 |
| ict | london_ny | 12.8 -> 47.9 | 7.4 -> 33.0 | 10 -> 46 | +0.10 -> -0.06 | 0 -> 0 |
| ict | active | 7.5 -> 32.8 | 3.6 -> 22.5 | 6 -> 32 | +0.23 -> +0.09 | 0 -> 0 |
| baseline | none | 11.9 -> 41.6 | 7.3 -> 25.5 | 14 -> 41 | +0.00 -> +0.00 | 0 -> 0 |

| instrument (ict, no filter) | avg n_is | avg n_oos | max n_oos | avg exp_oos (R) |
|---|---|---|---|---|
| ARB | 19.0 -> 53.9 | 14.0 -> 37.8 | 14 -> 40 | +0.16 -> +0.05 |
| BTC | 28.1 -> 67.0 | 14.0 -> 59.2 | 14 -> 65 | -0.38 -> -0.11 |
| ETH | 13.0 -> 61.2 | 14.6 -> 51.0 | 15 -> 57 | +0.05 -> -0.33 |
| HYPE | 13.6 -> 55.2 | 8.9 -> 29.4 | 9 -> 34 | -0.16 -> -0.24 |
| SOL | 22.0 -> 87.1 | 11.0 -> 53.8 | 11 -> 62 | -0.38 -> -0.29 |

Pooled across ARB, BTC, ETH, HYPE and SOL (the universe trials), averaged over the 16 exits:

| entry | session | avg n_oos | range | avg exp_oos (R) | rows clearing rule 1 |
|---|---|---|---|---|---|
| ict | none | 62.6 -> 231.2 | 204-258 | -0.13 -> -0.19 | 16 -> 16 |
| ict | london_ny | 37.0 -> 165.2 | 152-181 | +0.13 -> -0.07 | 0 -> 16 |
| ict | active | 18.0 -> 112.4 | 106-119 | +0.19 -> +0.07 | 0 -> 16 |
| baseline | none | 36.4 -> 127.5 | 106-141 | +0.01 -> +0.00 | 0 -> 16 |

The walk did on real data what it did on the synthetic stores: ICT trades three to four times as often (the 2026-09-07 investigation had put the shadowing loss second only to the range defect). Sample size is no longer what stops any ICT arm - twelve single-instrument rows (BTC and SOL) clear rule 1 on their own, every pooled ICT row clears it on 106 to 258 trades, and on entry days too (88 to 176 distinct days on the rows the report shows, so the open decision on rule 1 in the WU-2C handoff no longer changes any ICT verdict). What stops them is the statistics: of the 306 ICT rows (per instrument, views and pooled) 64 clear rule 1, **none** clears rule 2 or rule 3, 2 clear rule 4 and 31 rule 5. The best row is `ict|trail_2_1|active|hyperliquid:*`: 110 trades on 91 entry days, +0.26R out of sample against +0.27R in sample, ahead of the pooled baseline (diff p5 +0.05R) and of buy-and-hold (MAR 1.70 against -0.30) - with a bootstrap 5th-percentile expectancy of -0.05R and a deflated-Sharpe probability that rounds to zero against 901 trials. With no session filter the best pooled expectancy is -0.005R on 204 trades.

The friendlier readings of the previous run did not survive a larger sample: the London/NY arm went from +0.13R on 37 pooled trades to -0.07R on 165, the active-hours arm from +0.19R on 18 to +0.07R on 112. That is what the gate's trade-count floor exists to say, now said by the data. The one pattern left is that the four trailing exits under the active-hours filter are positive both in sample (+0.10R to +0.27R) and out of sample (+0.18R to +0.26R, on 106-111 trades); at 901 trials that is well inside what a search finds by chance, and it is the first thing a longer history should be asked about.

