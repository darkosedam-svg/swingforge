# swingforge tournament small

Written to be read. The complete record of this run is the `results` table — anything downstream (dashboards, screens, further analysis) should read that, not this file, which shows only the configs worth a human's attention.

## Gate

2 config(s) could not be graded; their reason is in the last section.

1 row omitted; the full set is in the `results` table.

| config | n_oos | exp_is | exp_oos | dsr prob | boot p5 | diff p5 | MAR | B&H MAR | 1 | 2 | 3 | 4 | 5 | 6 | passed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ict\|fixed_r_2\|none\|hyperliquid:BTC | 8 | 0.410 | 0.500 | 0.970 | 0.080 | 0.050 | ∞ | 0.750 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| zones\|trail_1_2\|none\|hyperliquid:BTC | 61 | 0.200 | 0.900 | – | – | – | – | – | – | – | – | – | – | – | – |
| ict\|IS_SELECTED\|none\|hyperliquid:BTC | 4 | 0.300 | 0.375 | 0.420 | -0.040 | -0.110 | 1.400 | 0.750 | ✓ | ✗ | ✓ | ✗ | ✓ | – | ✗ |
| baseline\|fixed_r_2\|none\|hyperliquid:BTC | 8 | -0.150 | -0.200 | 0.420 | -0.040 | -0.110 | 0.200 | 0.750 | ✓ | ✗ | ✓ | ✗ | ✓ | ✗ | ✗ |

## Top 3 configs, IS vs OOS

Every `IS_SELECTED` row follows, whatever its rank: it is the walk-forward view.

| config | exp_is | exp_oos | n_oos |
|---|---|---|---|
| zones\|trail_1_2\|none\|hyperliquid:BTC | 0.200 | 0.900 | 61 |
| ict\|fixed_r_2\|none\|hyperliquid:BTC | 0.410 | 0.500 | 8 |
| baseline\|fixed_r_2\|none\|hyperliquid:BTC | -0.150 | -0.200 | 8 |
| ict\|IS_SELECTED\|none\|hyperliquid:BTC | 0.300 | 0.375 | 4 |

## IS-selected exit per split

The exit with the highest in-sample expectancy for that split, applied out of sample.

| entry\|session\|instrument | 2023-01 | 2023-04 |
|---|---|---|
| ict\|none\|hyperliquid:BTC | fixed_r_2 | trail_1_2 |

## Regime breakdown

Pooled over the configs that passed the gate.
Diagnostic only: a trade appears once per config it belongs to, so counts are not independent across exits sharing an entry.

| regime | n | expectancy (R) |
|---|---|---|
| range_lowvol | 1 | -1.000 |
| range_midvol | 3 | -0.167 |
| trend_highvol | 3 | 0.500 |
| trend_lowvol | 1 | 3.000 |

## Excursion

Pooled out-of-sample trades per config. A `(raw signal)` row is the same entry
held with no exit management, so it measures the entry rather than the exit.

1 row omitted; the full set is in the `results` table.

| config | n | stopped out of winner | median exit eff. | median MAE (R) | median MFE (R) | MFE p75 (R) |
|---|---|---|---|---|---|---|
| baseline\|fixed_r_2\|none\|hyperliquid:BTC | 8 | 1.000 | – | 0.500 | 1.500 | 1.650 |
| ict\|IS_SELECTED\|none\|hyperliquid:BTC | 4 | 1.000 | 0.969 | 0.450 | 1.500 | 1.650 |
| ict\|fixed_r_2\|none\|hyperliquid:BTC | 8 | 1.000 | 0.969 | 0.500 | 1.500 | 1.650 |
| zones\|trail_1_2\|none\|hyperliquid:BTC | 0 | – | – | – | – | – |

## Cost stress

Spread and slippage x2, funding x1.5, applied post hoc to the same fills (rule 6).

1 row omitted; the full set is in the `results` table.

| config | exp_oos | stressed exp_oos | delta | rule 6 |
|---|---|---|---|---|
| baseline\|fixed_r_2\|none\|hyperliquid:BTC | -0.200 | -0.217 | -0.017 | ✗ |
| ict\|IS_SELECTED\|none\|hyperliquid:BTC | 0.375 | 0.358 | -0.017 | – |
| ict\|fixed_r_2\|none\|hyperliquid:BTC | 0.375 | 0.358 | -0.017 | ✓ |
| zones\|trail_1_2\|none\|hyperliquid:BTC | – | – | – | – |

## Excluded

Instruments dropped for want of history, and configs whose run, gate or ledger write failed.

| instrument / config | reason |
|---|---|
| hyperliquid:SOL | insufficient_history:6 |
| zones\|fixed_r_2\|none\|hyperliquid:BTC | error:RuntimeError: synthetic strategy failure |
| zones\|trail_1_2\|none\|hyperliquid:BTC | error:ValueError: a -200 R trade ruins the equity curve |

## Fill resolution mode

`subbars` means every 4H bar carried its 1H bars, so a stop and a target inside one bar were ordered on real data; `pessimistic` means the stop was assumed first.

| instrument | mode |
|---|---|
| hyperliquid:BTC | subbars |
