# WU-4A — VPS deployment (branch `wu/4a`)

Review fixes applied (round 1): `install.sh` sudo/idempotency/pinned-uv fixes; `swingforge-backfill.service`'s dual-venue `ExecStart` now `flock -n`-wrapped and non-`-`-prefixed (fails loudly); paper/web env-var/config-guard hardening. See git history for round-1 detail.

Round 2 (post-review): `swingforge-backfill.service` no longer stops/restarts `swingforge-paper@*` (`ExecStartPre`/`ExecStopPost` removed) — the nightly run is incremental (holds a venue file minutes, not hours) and `swingforge paper` rides it out via its own ~5min locked-store retry (`PERSIST_RETRIES=30×10s`); only a manual full re-backfill/`tournament` (hours) still needs paper stopped first (README §0/5/7, backfill unit comment).

New concession documented in §0/§7/§12 and the paper unit comment: restarting an instance with an orphaned open trade makes `swingforge paper` refuse to start (exit 3, names the trade); operator waits for it to close or restarts with `--abandon-open-trade` (closes at 0R, paper-only). §7 also notes `paper` refuses `baseline` configs (only `ict`/`zones` run).

README §5–9 rewritten around new CLI surface: `tournament --run-id` (required with `--resume`) and printed `run id:`+passing-config list; `tournament --instruments SYM` to narrow a reconciliation replay; `paper` prints `run id: paper:{venue}:{config_id}`; `report --venue --run-id` replaces manual report-reading in §6; §9's replay now pins `--run-id reconcile-<date>` instead of relying on the auto timestamp id.

`tests/unit/test_deploy.py`: 17 tests (was 18) — dropped the now-obsolete `ExecStopPost` absolute-path test, updated the backfill root/env test to assert `ExecStartPre`/`ExecStopPost` are gone entirely. Full unit suite 776 passed; `bash -n`, `ruff check`, `ruff format` all clean.
