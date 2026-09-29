# 908 trials, nothing passes

*A case study in telling a real trading edge from a search artefact.*

On 2026-09-20 I ran a strategy tournament over crypto perpetual-swap history on five instruments: 47 months for the three majors (36 of them out of sample), 41 for ARB, 18 for HYPE. It graded 908 trials, and **zero** passed the statistical gate. Pooled across the five swaps and averaged over the 16 exits, the ICT entry family's out-of-sample expectancy was **-0.001R** with no session filter, **-0.001R** under a London/New York filter and **-0.012R** under active hours, on 185 to 418 pooled trades a row. The random baseline it was matched against averaged **-0.000R**.

That null is the correct answer. Getting it took a gate I trust, a synthetic control showing the gate can say yes, and a bug hunt that removed a fake +3R per trade on a synthetic random walk.

The system is **swingforge**, a research and paper-trading engine I built: Daily bias, 4H execution, three entry families (ICT market structure, supply/demand zones, a frequency-matched random baseline), 16 exit rules, three session filters, one engine for backtest and paper, 994 tests.

## The gate: six rules, all must hold

Only pooled out-of-sample trades from anchored walk-forward splits (12 months in, 3 out, stepping 3) reach the gate.

1. **At least 60 out-of-sample trades.** Below that, rules 2-6 are reported as missing, not failed.
2. **Deflated Sharpe probability of at least 0.95**, with the trial count set to the whole search.
3. **Stationary-bootstrap 5th-percentile expectancy above zero.**
4. **Beats the frequency-matched random baseline** under the same exit and session.
5. **Beats buy-and-hold on MAR.**
6. **All of the above again with spread and slippage doubled and funding scaled by 1.5.**

Two details matter:

- **The trial count includes everything a reader might pick from**: the enumerated matrix, the in-sample best-exit views and the pooled cross-instrument trials.
- **Pooled trades are not independent.** Rule 2 reads a pooled row at its number of *distinct entry days*, not its trade count. That removes simultaneous entries only, so it is an upper bound on the independent sample, not a safe estimate. A review measured it 15% too high for five instruments correlated at 0.8 that never share an entry bar, and 61% too high when half the entries coincide. Entries either side of midnight UTC also count twice.

On the first real sweep, four-trade trials with near-identical stop-outs set the cross-trial Sharpe variance to **2,721** (over 740 trials), a bar nothing could clear. The variance is now measured only over trials that reach the 60-trade floor. That makes the gate easier to pass, so every run prints the variance it used.

## The planted-edge test: can the gate say yes?

A gate that rejects everything proves nothing, so two synthetic stores are planted the same way on one shared random-walk spine. On one, price drifts in the trade's direction after entry. On the other, it hands straight back to the random walk. The plants differ by 4.4%, and only 95 of their break-of-structure bars are common to both. ICT fires on **347 of 347** and **363 of 363** planted setups, and the two stores' total signal counts (**1,100 vs 1,171**) sit within 10%.

The test asserts:

1. **Planted store:** at least one ICT config passes the gate on 60+ out-of-sample trades, and **no** random-baseline config does.
2. **No-edge store:** nothing passes, although ICT still clears rule 1.
3. **Plain random walk:** nothing passes for ICT, zones or baseline, although ICT takes about 240 out-of-sample trades there.
4. **Five identical copies of that walk, pooled:** nothing passes. Every trade appears five times and each copy clears rule 1 alone, but rule 2 reads the pool at one copy's entry days, so the deflated Sharpe lands where one copy puts it.

All 10 tests pass on a re-run for this write-up.

This control is much easier than the real search. It grades two entries and three exits with costs switched off, so the trial count is **8**, not 908, and rule 6 just repeats rules 1-5. The plant is large: the passing row made **+0.756R on 250 trades** (deflated-Sharpe probability 0.993), and the other two ICT exits failed rule 2. It shows the gate can say yes, not that it would pass a realistic edge at 908 trials.

## The phantom-fill bug: +3R per trade on a random walk

The most valuable finding in the project was a bug. No test failed on it; a re-run after three strategy fixes exposed it.

The paper broker filled an entry from the whole 4H bar. Then, on the same-bar exit check, it handed the fill resolver the whole bar again, **including the 1H sub-bars that traded before the entry**. If price had already hit the target earlier in that bar, the trade was credited with a move it was never in.

On the synthetic random walk, **ICT averaged +3R per trade**, with one **+28R** trade under a fixed 2R exit. The random-walk null test still passed, but only through rule 1's floor.

Now same-bar exits see only the bar from the entry onward. Inside the entry candle, the open becomes the entry price, the favourable extreme is clipped to the entry and the adverse extreme is kept. A trade can lose on its entry candle but never win on it. Afterwards ICT's mean R on the same walk was near zero, with no fill beyond the 2R target, and the planted-edge controls still passed. Every number produced before the fix carries the artefact, so the 2026-09-07 Hyperliquid tournament was re-graded with the corrected broker.

If your backtest makes money on a random walk, check the fills before the strategy.

## The four-year verdict

Hyperliquid serves only its newest 5,000 candles per interval: **27 months of 4H, 15 of them out of sample**. There, nothing passed, but one pattern stayed positive: trailing exits under active hours, at **+0.18R to +0.26R on 106-111 pooled trades**. Fifteen months cannot separate that from what a 901-trial search finds by chance.

So I added OKX as a research-only venue. It serves 4H history back to within weeks of listing, and a four-year backfill stored **47 months** for BTC, ETH and SOL. The strategies, exits, gate and pooling are the same.

**`tournament:okx:20260920`: 908 trials, 0 passed.**

| entry | session | pooled n_oos | exp_oos, avg of 16 exits (R) |
|---|---|---|---|
| ICT | none | 325-418 | -0.001 |
| ICT | London/NY | 258-301 | -0.001 |
| ICT | active hours | 185-209 | -0.012 |
| random baseline | none | 215-251 | -0.000 |
| zones | none | 36-37 | -0.29 |

- **The Hyperliquid pattern did not survive.** Over 36 months the same four trailing-exit rows read **+0.08R, +0.12R, -0.13R and -0.13R** on 185-202 trades. Signs that flip with the window are what noise looks like.
- **The best row is a lead, not a result.** ICT on BTC, London/NY, `structure_2` exit: **+0.85R on 66 trades** (+0.92R in sample), with a bootstrap 5th percentile of **+0.005R** and a deflated-Sharpe probability of **0.024 against 0.95**. One such row is what 908 trials produce by chance. Its 5th-percentile margin over zero is far inside the ~0.1R funding uncertainty below.
- **Period, not venue.** I re-ran OKX on exactly Hyperliquid's 27 months (901 trials, 0 passed), and it reproduced Hyperliquid. ICT read **-0.178R / -0.074R / +0.060R** against **-0.193R / -0.073R / +0.071R**. Every fixed-stop ICT exit agreed within **0.04R**, pooled over the four shared instruments, and the same trailing pattern appeared. The pattern belongs to those 15 months.
- **The comparison priced a data gap.** Hyperliquid serves only about 7 months of 1H candles, so most of its history resolves a stop and a target inside one 4H bar stop-first. OKX replays sub-bars throughout. That is worth **+0.08R to +0.17R** on trailing and partial exits. With the fairer fills, those exits still did not recover.

## What I did not claim

- **OKX funding is mostly assumed.** The API serves about three months. Older settlements are charged the 0.01%-per-8h baseline, and rule 6's stress cannot widen that to a real spike. The written rule: **an OKX result holding by less than ~0.1R per trade is unresolved.**
- **The venues are not like for like** on fills, fees or funding cadence; the report prints each instrument's resolution mode.
- **Scope:** crypto perps, 4H, five instruments, up to four years of history. FX is not yet run.

## What this means for your desk

A search that ends in "no" still has a useful output: a verdict you can defend. Trials counted, correlation deflated, the random control's score, honest fills, and the assumptions that could still move the answer.

That is what I build and review: research infrastructure that can say **"your edge is not real"** with the evidence attached, and that can say "real" when a large synthetic edge is planted in a small search.

*- Darko*

---

*Darko Vlahovic · jessuskrist84@gmail.com — trading infrastructure and backtest-rigor engineering. The engine behind this study is this repository.*
