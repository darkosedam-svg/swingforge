# swingforge VPS runbook

Paper trading only. Nothing here places a live order or moves real money - `paper`
fills against replayed/streamed bars into a local DuckDB file, and OANDA is always used
via its **practice** environment (`OANDA_ENV=practice`).

Everything below is written to be run top to bottom by someone who has never seen this
repo. Commands are Ubuntu 24.04 / systemd; adjust package names if you use something else.

## 0. Concepts you need before you start

- **Layout on the VPS:** repo + venv at `/opt/swingforge`, DuckDB files and reports at
  `/var/lib/swingforge` (`SWINGFORGE_DATA_DIR`), shared secrets at `/etc/swingforge.env`
  (mode `0640`, owner `root:swingforge`), per-instance paper configs at
  `/etc/swingforge/paper/*.conf`.
- **Instance naming.** A CLI config id looks like `entry|exit|session|venue:symbol`
  (e.g. `ict|fixed_r_2|london_ny|hyperliquid:BTC`) - it contains `|` and `:`, which
  systemd instance names cannot. So a `swingforge-paper@<venue>-<slug>` unit (e.g.
  `swingforge-paper@hyperliquid-btc-ict`) reads its real config id from
  `/etc/swingforge/paper/<venue>-<slug>.conf` instead of from the instance name itself;
  `<venue>` is always the text before the *first* hyphen. See step 6.
- **Concurrency rule (paper vs. backfill/tournament).** `swingforge paper` only holds a
  venue's DuckDB file briefly, at each 4H bar close - and if it finds the file locked, it
  retries for up to ~5 minutes (`PERSIST_RETRIES=30 x 10s`) before giving up. The nightly
  `swingforge-backfill.timer` run is incremental (it only fetches bars since the last run)
  and normally holds a venue's file for a few minutes - comfortably inside paper's retry
  budget, so **the nightly backfill no longer stops or restarts any `swingforge-paper@`
  instance** (see `swingforge-backfill.service`'s own comment). A dashboard request against
  that venue during this window just sees `503 {"detail": "venue busy"}` and succeeds on
  retry (next-but-one bullet).

  A **manual long run** - a full re-backfill (`--years 4` run by hand, e.g. to rebuild a
  venue's file from scratch) or `tournament` - is different: it holds the file for *hours*,
  well past paper's retry budget. **Stop the relevant paper instance(s) yourself first** for
  those, and take the same lock the systemd unit takes (next bullet):
  ```bash
  sudo systemctl stop 'swingforge-paper@*'
  sudo -u swingforge flock /var/lib/swingforge/.venue.lock \
    /opt/swingforge/.venv/bin/swingforge backfill --venue hyperliquid --years 4 \
      --data-dir /var/lib/swingforge   # or tournament ...; --data-dir is required outside the units
  sudo systemctl start 'swingforge-paper@*'   # or just the ones you actually enabled
  ```
  **After restarting an instance that had an open trade when it was stopped**, `swingforge
  paper` refuses to start (exit code 3, naming the orphaned trade in its output/journal)
  rather than silently resume risk it can no longer verify. Either wait for that trade to
  close before stopping paper for the long run (preferred), or restart it with
  `--abandon-open-trade`, which closes the trade at `0R` in the `trades` table - a
  paper-only concession, never applicable to a live account. See sections 7 and 12.
- **Concurrency rule (backfill/tournament vs. each other).** `swingforge-backfill.service`'s
  `ExecStart` wraps both venue backfills in a single non-blocking `flock -n
  /var/lib/swingforge/.venue.lock` - if a manual `backfill`/`tournament` run already holds
  that lock when the nightly timer fires, the service fails fast (visible in
  `journalctl -u swingforge-backfill`) instead of queuing behind it. Manual runs must take
  the same lock, *without* `-n` (block and wait, as in the snippet above), so two manual
  runs (or a manual run and the timer) never write to the same venue's DuckDB file at once.
  Before starting a long manual run you don't want the nightly timer interrupting:
  ```bash
  sudo systemctl mask swingforge-backfill.timer     # before the long manual run
  # ... run backfill / tournament ...
  sudo systemctl unmask swingforge-backfill.timer    # after
  ```
- **The dashboard does not need to be stopped for any of this.** `swingforge-web.service`
  opens each venue's DuckDB file read-only, briefly, per request, retrying a locked file a
  few times before giving up - so while `backfill`/`tournament` (or a manual run) holds a
  venue's file, dashboard requests against that venue (including a settings `PUT`, which
  opens the file read-write) return `503 {"detail": "venue busy"}` instead of hanging or
  erroring; they succeed again as soon as the lock is released. Just retry after a moment.

## 1. Provision the VPS

Ubuntu 24.04, any provider. As root:

```bash
apt-get update && apt-get install -y ufw git curl sudo
ufw allow OpenSSH
# Replace <your-ip> with the single IP (or CIDR) you'll view the dashboard from.
ufw allow from <your-ip> to any port 8787 proto tcp
ufw enable
```

There is deliberately **no reverse proxy and no TLS in v1**: the dashboard is read-mostly,
single-user, paper-only, and protected by both a bearer token (`SWINGFORGE_TOKEN`) and the
`ufw` IP allowlist above. `swingforge web --host 0.0.0.0 ...` itself refuses to bind
`0.0.0.0` at all if `SWINGFORGE_TOKEN` is unset. **When live trading is ever specced**,
put this behind Caddy (or nginx) with automatic TLS before exposing it beyond your own IP.

## 2. Install

```bash
git clone https://github.com/darkosedam-svg/swingforge.git /tmp/swingforge-bootstrap
cd /tmp/swingforge-bootstrap
sudo bash deploy/install.sh
```

`install.sh` is idempotent (safe to re-run) and:
- creates the `swingforge` system user, installs a pinned `uv` release if missing,
- clones (or fast-forwards, as the `swingforge` user, to sidestep git's dubious-ownership
  check) the real checkout to `/opt/swingforge` and runs `uv sync --frozen
  --compile-bytecode`,
- creates `/var/lib/swingforge{,/reports}` and `/etc/swingforge/paper/`,
- seeds `/etc/swingforge.env` from `deploy/swingforge.env.example` **only if that file
  does not already exist** (so a second run never clobbers real credentials),
- installs the four systemd units, then enables `swingforge-web.service` (without
  starting it - it will refuse to bind until `/etc/swingforge.env` has a real
  `SWINGFORGE_TOKEN`; step 3 below starts it) and enables + starts
  `swingforge-backfill.timer`. **It does not enable or start any `swingforge-paper@`
  instance** - that only happens once a config has passed the gate (step 8).

Set `REPO_URL=<fork-url>` and/or `BRANCH=<branch>` before calling `install.sh` if you are
installing from somewhere other than the default GitHub origin.

## 3. Credentials

Edit `/etc/swingforge.env` (root only):

```bash
sudo nano /etc/swingforge.env
```

- `OANDA_TOKEN` / `OANDA_ACCOUNT_ID`: from an OANDA **practice** account (not live).
  `OANDA_ENV` should stay `practice`.
- `SWINGFORGE_TOKEN`: generate a real one, don't leave it blank:
  ```bash
  openssl rand -hex 32
  ```
- `HL_ACCOUNT_ADDRESS`: a read-only Hyperliquid wallet address (reserved; `paper` does
  not use it today).

Then apply it:

```bash
sudo systemctl restart swingforge-web.service
```

## 4. Backfill

First time (both venues, 4 years each - the venue's DuckDB file lives at
`/var/lib/swingforge/<venue>.duckdb`):

```bash
sudo systemctl start swingforge-backfill.service
sudo journalctl -u swingforge-backfill.service -f    # watch it run
```

This is exactly what `swingforge-backfill.timer` fires nightly at `00:10 UTC` (plus up to
`RandomizedDelaySec=300`; `Persistent=true`, so a missed night due to downtime still runs
once the box is back). Once the initial 4-year backfill above has completed, every
subsequent nightly run through this same unit is incremental and normally holds a venue's
file for only a few minutes - well inside `swingforge paper`'s own retry budget, so the
unit does **not** stop or restart any `swingforge-paper@*` instance around it (see the
concurrency rule in section 0 and `swingforge-backfill.service`'s own comment). Only a
manual **full** re-backfill (this same `--years 4` command run by hand, e.g. to rebuild a
venue's file from scratch) needs the manual stop/start dance from section 0 - that run,
unlike the incremental nightly one, holds the file for hours.

The service now fails loudly if either venue's backfill fails (it no longer swallows the
exit status) - **check `systemctl status swingforge-backfill.service` and
`journalctl -u swingforge-backfill` at least weekly** so a failure doesn't sit unnoticed
until you go looking for missing bars. If you want to be alerted immediately instead, add
an `OnFailure=<your-alerting-unit>.service` line to `swingforge-backfill.service`'s
`[Unit]` section (not shipped here - there is no alerting unit in this repo to point it
at).

## 5. Tournament

Stop paper for both venues first, and take the same `.venue.lock` the nightly backfill
takes (section 0) so it can't start mid-run:

```bash
sudo systemctl stop 'swingforge-paper@*'
sudo -u swingforge flock /var/lib/swingforge/.venue.lock \
  /opt/swingforge/.venv/bin/swingforge tournament --venue hyperliquid --out /var/lib/swingforge/reports \
  --data-dir /var/lib/swingforge
sudo -u swingforge flock /var/lib/swingforge/.venue.lock \
  /opt/swingforge/.venv/bin/swingforge tournament --venue oanda --out /var/lib/swingforge/reports \
  --data-dir /var/lib/swingforge
sudo systemctl start 'swingforge-paper@*'   # or just the ones you actually enabled
```

Add `--seed N` to pin the RNG. Each run prints `run id: tournament:{venue}:{timestamp}` at
the end, together with the list of config ids that passed the gate - copy both down (the
run id feeds step 6's `report --run-id`; a passing config id feeds step 7). `--resume`
continues an interrupted run and now *requires* `--run-id <id>`
(the id the interrupted attempt printed, or that you passed it originally) so it knows
which run to continue (see the tournament's own resume caveats in its handoff before
relying on this for anything but "the process died, restart it"). Both venues' DuckDB
files must not have an active paper instance running against them while the tournament
runs (section 0) - the `flock` above only serialises against another manual/scheduled
backfill or tournament, it does not stop paper instances for you.

## 6. Read the report

Print the gate summary for the run id that step 5 printed:

```bash
/opt/swingforge/.venv/bin/swingforge report --venue hyperliquid --run-id tournament:hyperliquid:<timestamp from step 5>
```

It shows, for that run id:
- the gate table - which configs passed all 6 rules on pooled OOS trades,
- IS vs OOS for the top configs,
- the regime and excursion breakdowns,
- the cost-stress (2x spread/slippage, 1.5x funding) columns.

(A full markdown copy of the same report also lands under `/var/lib/swingforge/reports/` -
`tournament` prints its exact path too, if you want a file to keep or share instead.)

Only a config that **passes the gate** goes on to paper (step 8). The `results` table in
the venue's DuckDB file is the source of truth both `report` and the markdown file were
generated from, if you want to query it directly - see step 8 for the read-only python
heredoc pattern used for every direct query in this runbook (there is no `duckdb` CLI
installed in this deploy, and every query here opens the file read-only so it never fights
a concurrent writer for the lock).

## 7. Enable a gate-passed config for paper

`swingforge paper` only accepts `ict`/`zones` entry configs - a `baseline` config id (it
exists only to prove the gate rejects noise) is refused outright and never runs as paper,
even if it somehow appears in a passing list. Copy an `ict`/`zones` config id from the list
`swingforge tournament` printed at the end of step 5 (or re-check it with `report
--run-id` from step 6) - say `ict|fixed_r_2|london_ny|hyperliquid:BTC` passed. Pick an
instance name `<venue>-<slug>` (`<venue>` must be the text before the first hyphen - see
section 0):

```bash
sudo install -d -m 0750 -o root -g swingforge /etc/swingforge/paper
echo 'ict|fixed_r_2|london_ny|hyperliquid:BTC' | \
  sudo tee /etc/swingforge/paper/hyperliquid-btc-ict.conf >/dev/null
sudo chmod 0640 /etc/swingforge/paper/hyperliquid-btc-ict.conf
sudo chown root:swingforge /etc/swingforge/paper/hyperliquid-btc-ict.conf

sudo systemctl enable --now swingforge-paper@hyperliquid-btc-ict
sudo systemctl status swingforge-paper@hyperliquid-btc-ict
```

If this instance was previously stopped (e.g. for a manual long run per section 0) while it
had an open trade, this start will fail with exit code 3 naming the orphaned trade - see
section 0's note and section 12 for how to resolve it before it will start.

Then tell the dashboard about it (sourcing the real token from `/etc/swingforge.env`
rather than pasting it literally, and adjusting the body to the venues/symbols/configs you
actually enabled - this replaces the whole settings object, so include everything you want
enabled, not just the new addition):

```bash
SWINGFORGE_TOKEN="$(sudo sed -n 's/^SWINGFORGE_TOKEN=//p' /etc/swingforge.env)"   # the env file is root:swingforge 0640
curl -sS -X PUT http://<vps-ip>:8787/api/settings \
  -H "Authorization: Bearer $SWINGFORGE_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
        "enabled_instruments": {"hyperliquid": ["BTC"]},
        "paper_configs": ["ict|fixed_r_2|london_ny|hyperliquid:BTC"],
        "risk_pct": 0.01,
        "session": "london_ny",
        "time_stop_bars": 10,
        "kill_switch": false
      }'
```

(A `PUT` while backfill/tournament holds that venue's file returns
`503 {"detail": "venue busy"}` - see section 0 - just retry.)

Every `PUT` is appended to the `settings` table's `settings_log`, so the dashboard's
history shows exactly who changed what and when. The engine only re-reads settings at bar
close, so a change lands on the *next* 4H bar, not mid-bar.

## 8. The 60-day / 30-trade check

Per the design (spec section 6): a config stays in paper until it has run for
**at least 60 days AND produced at least 30 trades** - whichever takes longer. Check
progress from the dashboard (open positions, last fills) or query the venue's DuckDB file
directly. Every query in this runbook runs read-only, straight through the venv's own
`python` in a heredoc, rather than a `duckdb` CLI (there isn't one installed in this
deploy) - so it's copy-pasteable, and it never opens a file read-write while backfill,
tournament or another paper instance might hold it:

```bash
/opt/swingforge/.venv/bin/python - <<'PY'
import datetime as dt

import duckdb

# `swingforge paper` prints `run id: paper:{venue}:{config_id}` at startup (check
# `journalctl -u swingforge-paper@<instance>` if you didn't watch it start), and the
# dashboard's /trades/{id} view shows it too. Use that printed id verbatim below. Do NOT
# filter by the systemd instance name ("hyperliquid-btc-ict") - that name is local to this
# VPS's units and never reaches the CLI or gets written to `trades`.
RUN_ID = "paper:hyperliquid:ict|fixed_r_2|london_ny|hyperliquid:BTC"

con = duckdb.connect("/var/lib/swingforge/hyperliquid.duckdb", read_only=True)
n_trades, first_trade = con.execute(
    "SELECT count(*), min(entry_ts) FROM trades WHERE run_id = ?", [RUN_ID]
).fetchone()
print(f"n_trades={n_trades} (need >= 30), first_trade={first_trade}")
if first_trade is not None:
    days_running = (dt.datetime.now(dt.UTC) - first_trade.replace(tzinfo=dt.UTC)).days
    print(f"days_running={days_running} (need >= 60)")
PY
```

## 9. Paper-vs-backtest reconciliation

Before writing any live spec, the paper run's trades over its own window must match what
a fresh backtest replay produces over that *same* historical window, for the *same*
config. This is the check the design calls "paper/backtest reconciliation" (spec section
6) and it must pass, not just "look close":

There is one DuckDB file per venue (`/var/lib/swingforge/<venue>.duckdb`), holding every
run's `trades` distinguished by `run_id` - the live paper run and a reconciliation replay
both land in the *same* file, under different `run_id`s. Say the paper config is
`ict|fixed_r_2|london_ny|hyperliquid:BTC` (entry `ict`, exit `fixed_r_2`, session
`london_ny`) on `hyperliquid`:

```bash
# 1. The paper run's run_id is paper:{venue}:{config_id} (see step 8 - never the systemd
#    instance name). Get its first trade timestamp from step 8's query (its last is
#    max(entry_ts) for the same run_id) - call it <paper_start_iso> below.

# 2. Re-run a replay narrowed to exactly that config, instrument and window (not a full
#    tournament sweep), so its trades land under a fresh, predictable run_id in the same
#    DuckDB file. --instruments narrows the replay to the paper instrument; --run-id pins
#    an explicit id instead of the usual auto-generated tournament:{venue}:{timestamp} one,
#    so you don't have to go find it afterwards. Stop paper for this venue and take the
#    .venue.lock first (section 0/5):
sudo systemctl stop 'swingforge-paper@hyperliquid-btc-ict'
sudo -u swingforge flock /var/lib/swingforge/.venue.lock \
  /opt/swingforge/.venv/bin/swingforge tournament \
  --venue hyperliquid --out /var/lib/swingforge/reports/reconcile --data-dir /var/lib/swingforge \
  --entries ict --exits fixed_r_2 --sessions london_ny --instruments BTC \
  --run-id reconcile-<date> \
  --start <paper_start_iso> --end "$(date -u +%FT%TZ)"
sudo systemctl start 'swingforge-paper@hyperliquid-btc-ict'
# Use the exact --run-id you passed above ("reconcile-<date>") as <reconcile_run_id> below.

# 3. Compare row-by-row: same entry_ts, direction, entry_price and realized_r for every
#    trade the paper run produced. Any mismatch means the live paper fills diverged from
#    the backtest for reasons that need explaining (a data gap, a fill-resolution
#    difference, a clock/timezone bug) before trusting a live spec built on this config.
#    The FULL OUTER JOIN's ON (rather than USING) plus explicit coalesce() surfaces a trade
#    missing from *either* side as its own row instead of silently dropping it:
/opt/swingforge/.venv/bin/python - <<'PY'
import duckdb

PAPER_RUN_ID = "paper:hyperliquid:ict|fixed_r_2|london_ny|hyperliquid:BTC"
RECONCILE_RUN_ID = "reconcile-<date>"  # the exact --run-id passed in step 2 above

con = duckdb.connect("/var/lib/swingforge/hyperliquid.duckdb", read_only=True)
rows = con.execute(
    """
    SELECT
        coalesce(p.entry_ts, b.entry_ts)     AS entry_ts,
        coalesce(p.direction, b.direction)   AS direction,
        p.entry_price AS paper_entry, b.entry_price AS bt_entry,
        p.realized_r  AS paper_r,     b.realized_r  AS bt_r
    FROM (SELECT * FROM trades WHERE run_id = ?) p
    FULL OUTER JOIN (SELECT * FROM trades WHERE run_id = ?) b
      ON p.entry_ts = b.entry_ts AND p.direction = b.direction
    WHERE p.entry_price IS DISTINCT FROM b.entry_price
       OR p.realized_r  IS DISTINCT FROM b.realized_r
       OR p.entry_price IS NULL OR b.entry_price IS NULL
    ORDER BY entry_ts
    """,
    [PAPER_RUN_ID, RECONCILE_RUN_ID],
).fetchall()
for row in rows:
    print(row)
print("PASS: no mismatches" if not rows else f"FAIL: {len(rows)} mismatched/missing trades")
PY
# An empty result set (PASS) is a pass.
```

## 10. Record vcrpy cassettes (manual, once, human only)

The contract tests (`tests/contract/test_hyperliquid_contract.py`,
`test_oanda_contract.py`) replay pre-recorded HTTP cassettes so CI never needs live
credentials. Record them once, from a machine that has real OANDA practice credentials
(see `tests/contract/README.md` for the full explanation):

```bash
SWINGFORGE_RECORD=1 uv run pytest -m contract tests/contract/test_hyperliquid_contract.py -v
OANDA_TOKEN=<your-practice-token> SWINGFORGE_RECORD=1 \
  uv run pytest -m contract tests/contract/test_oanda_contract.py -v
```

Before committing the new/updated cassette files, confirm no secret leaked into them
(`conftest.py` already strips the `Authorization` header, but verify anyway):

```bash
grep -ri authorization tests/contract/cassettes    # must print nothing
```

Only commit the cassette files if that grep is empty.

## 11. Kill switch

The dashboard's settings form has a `kill_switch` toggle (`PUT /api/settings` with
`"kill_switch": true`, or the same body as step 7 with that one field flipped). The
engine checks settings at every bar close, so flipping it stops new entries within one
bar close, without restarting any systemd unit. Flip it back to resume.

## 12. Reading logs

```bash
sudo journalctl -u swingforge-paper@hyperliquid-btc-ict -f
sudo journalctl -u swingforge-web.service -f
sudo journalctl -u swingforge-backfill.service -f
sudo journalctl -u swingforge-backfill.timer --since "7 days ago"
```

If `systemctl status swingforge-paper@<instance>` shows it failed with **exit code 3**,
`journalctl` for that unit names an orphaned open trade left over from before its last stop
(see section 0's note and section 7) - restarting it as-is (or waiting for `Restart=on-failure`
to retry) will not help until you either let that trade close naturally or run it manually
once with `--abandon-open-trade`.

## 13. Upgrading

```bash
sudo -u swingforge git -C /opt/swingforge pull --ff-only origin main
sudo -u swingforge "$(command -v uv || echo /usr/local/bin/uv)" sync --frozen --compile-bytecode --directory /opt/swingforge
sudo bash /opt/swingforge/deploy/install.sh   # idempotent - refreshes systemd units for any changes
sudo systemctl restart swingforge-web.service
sudo systemctl restart 'swingforge-paper@*'
```

**Anything that changes the gate or the engine must re-run
`uv run pytest -m slow tests/integration` (green) before this upgrade step** - that suite
covers the planted-edge synthetic tournament (`ict` passes, `baseline` fails) and the
pure-random-walk case (everything fails), which is what stands between "the gate still
means something" and silently paper-trading noise.
