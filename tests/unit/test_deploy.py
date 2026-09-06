"""Deploy artifacts (WU-4A): systemd units, install.sh.

Nothing here starts a real systemd service (this repo is authored on Windows; no systemd
is available anywhere in CI for it either). Instead:
  - every unit file is parsed as INI (systemd unit syntax is INI-compatible) and checked
    for the keys the design binds as required;
  - `deploy/install.sh` is checked for bash syntax only (`bash -n`), plus targeted greps
    for the idempotency/security properties the design calls out;
  - the trickiest pieces of logic - deriving a venue ("hyperliquid") from a systemd
    instance name ("hyperliquid-btc-ict") in `swingforge-paper@.service`, its non-empty
    config-file guard, and `swingforge-backfill.service`'s "run both venues, fail if
    either did" exit-code tracking - are extracted from the *actual* unit file text,
    unescaped exactly as systemd itself would (see the comments in those files), and
    executed for real via `bash -c` (with the real `swingforge`/`sudo` invocations swapped
    for inert stubs, since neither exists in this test environment).
"""

from __future__ import annotations

import configparser
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DEPLOY_DIR = REPO_ROOT / "deploy"


def _bash_executable() -> str:
    found = shutil.which("bash")
    if found is None:
        pytest.fail("bash is not on PATH; this test needs a real bash to check deploy/ scripts")
    return found


def _unit_config(path: Path) -> configparser.RawConfigParser:
    """Parse a systemd unit file as INI.

    `RawConfigParser` (rather than `ConfigParser`) does no `%`-interpolation of its own, so
    it does not choke on a value like `%i` or `%%%%` that is meaningful to systemd but not
    to configparser. `strict=False` additionally tolerates a section repeating an option
    (irrelevant now that `swingforge-backfill.service` has only one `ExecStart=`, but
    harmless to keep). `optionxform = str` keeps keys case-sensitive, since systemd's are
    (`WorkingDirectory`, not `workingdirectory`).
    """
    assert path.is_file(), f"missing unit file: {path}"
    parser = configparser.RawConfigParser(strict=False)
    parser.optionxform = str  # type: ignore[assignment,method-assign]
    parser.read(path, encoding="utf-8")
    return parser


def _systemd_unescape(value: str) -> str:
    """Undo systemd's Exec*= escaping (systemd.service(5), "Command lines"): a literal `$`
    is written `$$` and a literal `%` is written `%%`, because systemd itself expands
    `$VAR`/`${VAR}` and `%`-specifiers in Exec* line values before the string is ever
    handed to the shell. Every bash snippet in these units is authored by doubling every
    real `$`/`%`, so this is the exact inverse.
    """
    return value.replace("$$", "$").replace("%%", "%")


_HARDENING_KEYS = {
    "NoNewPrivileges": "yes",
    "ProtectSystem": "strict",
    "ProtectHome": "yes",
    "ReadWritePaths": "/var/lib/swingforge",
    "PrivateTmp": "yes",
}


# --------------------------------------------------------------------------------------
# install.sh: syntax, plus targeted greps for the idempotency/security properties (C1/I4).
# --------------------------------------------------------------------------------------


def test_install_sh_is_syntactically_valid_bash() -> None:
    script = DEPLOY_DIR / "install.sh"
    assert script.is_file()
    proc = subprocess.run(
        [_bash_executable(), "-n", str(script)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"bash -n failed:\n{proc.stderr}"


def test_install_sh_never_starts_a_paper_instance() -> None:
    """Starting paper instances is the runbook's job (deploy/README.md), not install.sh's -
    the install must be safe to re-run without silently (re)launching live paper trading.
    Mentioning the unit name in a comment is fine; actually enabling/starting one is not.
    """
    text = (DEPLOY_DIR / "install.sh").read_text(encoding="utf-8")
    assert not re.search(r"systemctl\s+(enable|start)[^\n]*swingforge-paper@", text)


def test_install_sh_checks_for_sudo() -> None:
    """C1: the script now runs several steps as the service user via `sudo -u`; fail with a
    clear message up front if `sudo` itself is missing rather than a confusing error
    partway through step 3/4."""
    text = (DEPLOY_DIR / "install.sh").read_text(encoding="utf-8")
    assert re.search(r"command -v sudo", text), "install.sh must check that 'sudo' is available"


def test_install_sh_git_update_runs_as_service_user() -> None:
    """C1: root running `git pull`/`fetch`/`checkout` on a tree owned by the service user
    trips git's "dubious ownership" safety check (root is not exempt from it). Running the
    update as the service user itself (whose uid matches the directory's owner from the
    previous run's `chown`) avoids the check entirely, keeping re-runs idempotent."""
    text = (DEPLOY_DIR / "install.sh").read_text(encoding="utf-8")
    update_lines = [
        line.strip()
        for line in text.splitlines()
        if re.search(r'git -C "\$INSTALL_DIR"\s+(fetch|checkout|pull)\b', line)
    ]
    assert len(update_lines) == 3, f"expected fetch/checkout/pull update lines, found: {update_lines}"
    for line in update_lines:
        assert line.startswith('sudo -u "$SERVICE_USER" git'), (
            f"git update commands must run as the service user, not root: {line!r}"
        )
    # The initial clone (fresh install, no pre-existing ownership to conflict with) may
    # still legitimately run as root.
    assert re.search(r"^\s*git clone --branch", text, re.MULTILINE)


def test_install_sh_uses_pinned_uv_version() -> None:
    """I4: install a pinned uv release rather than astral.sh/uv/install.sh's rolling
    "latest", so a re-run months from now can't silently pick up an untested version."""
    text = (DEPLOY_DIR / "install.sh").read_text(encoding="utf-8")
    assert not re.search(r"https://astral\.sh/uv/install\.sh", text), (
        "must not use uv's rolling 'latest' installer URL"
    )
    assert re.search(r'UV_VERSION="[\d.]+"', text), 'expected a pinned UV_VERSION="x.y.z"'
    assert 'curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh"' in text
    # The trust assumption (fetched over HTTPS from astral.sh, run as root) must be documented.
    assert "trust" in text.lower()


def test_install_sh_uv_sync_uses_frozen_and_compile_bytecode() -> None:
    text = (DEPLOY_DIR / "install.sh").read_text(encoding="utf-8")
    assert re.search(r"sync --frozen --compile-bytecode", text)


def test_install_sh_enables_web_without_starting_it() -> None:
    """C1: enable (but do not start) the web unit -- deploy/README.md's credentials step is
    what actually starts/restarts it, once /etc/swingforge.env has a real SWINGFORGE_TOKEN."""
    text = (DEPLOY_DIR / "install.sh").read_text(encoding="utf-8")
    assert re.search(r"^systemctl enable swingforge-web\.service\s*$", text, re.MULTILINE)
    assert not re.search(r"systemctl enable --now swingforge-web\.service", text)
    assert re.search(r"^systemctl enable --now swingforge-backfill\.timer\s*$", text, re.MULTILINE)


# --------------------------------------------------------------------------------------
# Unit files: required keys per the design (spec section 6/7, plan WU-4A).
# --------------------------------------------------------------------------------------


def test_paper_unit_has_the_required_service_keys() -> None:
    cfg = _unit_config(DEPLOY_DIR / "swingforge-paper@.service")
    service = cfg["Service"]
    assert service["Restart"] == "on-failure"
    assert service["EnvironmentFile"] == "/etc/swingforge.env"
    assert service["WorkingDirectory"] == "/opt/swingforge"
    assert service["User"] == "swingforge"
    assert cfg["Install"]["WantedBy"] == "multi-user.target"
    for key, expected in _HARDENING_KEYS.items():
        assert service[key] == expected, f"missing/incorrect hardening key {key}"

    text = (DEPLOY_DIR / "swingforge-paper@.service").read_text(encoding="utf-8")
    assert "Environment=PYTHONDONTWRITEBYTECODE=1" in text
    assert "Environment=HOME=/var/lib/swingforge" in text
    # Minors: every Environment= line must precede EnvironmentFile=.
    env_file_idx = text.index("EnvironmentFile=")
    for marker in (
        "Environment=PYTHONDONTWRITEBYTECODE=1",
        "Environment=HOME=/var/lib/swingforge",
        "Environment=INSTANCE=%i",
    ):
        assert text.index(marker) < env_file_idx, f"{marker!r} must appear before EnvironmentFile="


def test_web_unit_has_the_required_service_keys_and_binds_all_interfaces() -> None:
    cfg = _unit_config(DEPLOY_DIR / "swingforge-web.service")
    service = cfg["Service"]
    assert service["Restart"] == "on-failure"
    assert service["EnvironmentFile"] == "/etc/swingforge.env"
    assert service["WorkingDirectory"] == "/opt/swingforge"
    assert service["User"] == "swingforge"
    exec_start = service["ExecStart"]
    assert "--host 0.0.0.0" in exec_start
    assert "--port 8787" in exec_start
    for key, expected in _HARDENING_KEYS.items():
        assert service[key] == expected, f"missing/incorrect hardening key {key}"

    text = (DEPLOY_DIR / "swingforge-web.service").read_text(encoding="utf-8")
    assert "Environment=PYTHONDONTWRITEBYTECODE=1" in text
    assert "Environment=HOME=/var/lib/swingforge" in text
    env_file_idx = text.index("EnvironmentFile=")
    for marker in ("Environment=PYTHONDONTWRITEBYTECODE=1", "Environment=HOME=/var/lib/swingforge"):
        assert text.index(marker) < env_file_idx, f"{marker!r} must appear before EnvironmentFile="


def test_backfill_service_runs_as_root_and_reads_the_shared_env_file() -> None:
    """User=root (not swingforge) is deliberate here: it lets the backfill command's own
    `sudo -u swingforge` (dropping to the unprivileged service user for each venue) work
    without a dedicated sudoers rule -- root's default `ALL=(ALL) ALL` entry already covers
    it. This unit no longer stops/restarts swingforge-paper@ instances around the run (that
    was the old ExecStartPre/ExecStopPost pair -- removed): the incremental nightly backfill
    only holds a venue's file for a few minutes, and `swingforge paper` already retries a
    locked store for up to ~5 minutes before giving up (see the paper unit's own comment),
    so paper rides out this window on its own."""
    cfg = _unit_config(DEPLOY_DIR / "swingforge-backfill.service")
    service = cfg["Service"]
    assert service["Type"] == "oneshot"
    assert service["User"] == "root"
    assert service["EnvironmentFile"] == "/etc/swingforge.env"
    assert service["WorkingDirectory"] == "/opt/swingforge"
    assert service["TimeoutStartSec"] == "infinity"
    assert "ExecStartPre" not in service, "must not stop paper instances around the run anymore"
    assert "ExecStopPost" not in service, "must not restart paper instances around the run anymore"

    text = (DEPLOY_DIR / "swingforge-backfill.service").read_text(encoding="utf-8")
    assert not re.search(r"^ExecStartPre=", text, re.MULTILINE), (
        "must not stop paper instances around the run anymore (directive removed, not just emptied)"
    )
    assert not re.search(r"^ExecStopPost=", text, re.MULTILINE), (
        "must not restart paper instances around the run anymore (directive removed, not just emptied)"
    )

    exec_lines = [line for line in text.splitlines() if line.startswith("ExecStart=")]
    assert len(exec_lines) == 1, "both venue backfills must run inside a single ExecStart= line (I3)"
    assert not exec_lines[0].startswith("ExecStart=-"), (
        "the backfill ExecStart must NOT be '-'-prefixed: a failure has to fail the unit "
        "(I3) instead of being silently swallowed like the old two-line '-' form was"
    )


def test_backfill_service_has_no_install_section() -> None:
    """I8: swingforge-backfill.service is only ever started via its timer or a manual
    `systemctl start` -- it must not be `enable`-able on its own."""
    cfg = _unit_config(DEPLOY_DIR / "swingforge-backfill.service")
    assert "Install" not in cfg


def test_backfill_timer_is_nightly_and_catches_up_missed_runs() -> None:
    cfg = _unit_config(DEPLOY_DIR / "swingforge-backfill.timer")
    timer = cfg["Timer"]
    assert timer["OnCalendar"] == "*-*-* 00:10:00 UTC"
    assert timer["Persistent"] == "true"
    assert timer["Unit"] == "swingforge-backfill.service"
    assert timer["RandomizedDelaySec"] == "300"


def test_env_example_declares_the_six_expected_variables() -> None:
    text = (DEPLOY_DIR / "swingforge.env.example").read_text(encoding="utf-8")
    for var in (
        "SWINGFORGE_DATA_DIR",
        "SWINGFORGE_TOKEN",
        "OANDA_TOKEN",
        "OANDA_ACCOUNT_ID",
        "OANDA_ENV",
        "HL_ACCOUNT_ADDRESS",
    ):
        assert re.search(rf"^{var}=", text, re.MULTILINE), f"missing {var}= in env.example"
    assert "SWINGFORGE_DATA_DIR=/var/lib/swingforge" in text


def test_env_example_secrets_are_empty() -> None:
    """Minors: the committed example must never carry a real-looking secret value -- a
    second `install.sh` run never overwrites a real /etc/swingforge.env, but the example
    itself must stay a template."""
    text = (DEPLOY_DIR / "swingforge.env.example").read_text(encoding="utf-8")
    for var in ("SWINGFORGE_TOKEN", "OANDA_TOKEN", "OANDA_ACCOUNT_ID", "HL_ACCOUNT_ADDRESS"):
        match = re.search(rf"^{var}=(.*)$", text, re.MULTILINE)
        assert match is not None, f"missing {var}= in env.example"
        assert match.group(1) == "", (
            f"{var} must be left empty in the committed example, got {match.group(1)!r}"
        )


# --------------------------------------------------------------------------------------
# The real logic in the unit files, extracted and executed via a real bash.
# --------------------------------------------------------------------------------------


def test_paper_unit_execstart_derives_venue_from_instance_name() -> None:
    text = (DEPLOY_DIR / "swingforge-paper@.service").read_text(encoding="utf-8")
    match = re.search(r"^ExecStart=(.+)$", text, re.MULTILINE)
    assert match is not None, "no ExecStart= line in swingforge-paper@.service"
    raw_execstart = match.group(1)

    # Escaping sanity check: the authored line really doubles every literal '$' and '%'
    # (systemd would otherwise try to expand them itself before bash ever runs).
    assert "$${INSTANCE%%%%-*}" in raw_execstart, (
        "ExecStart should write the bash expression '${INSTANCE%%-*}' with every $ and % "
        "doubled for systemd - see the unit file's own comment"
    )

    unescaped = _systemd_unescape(raw_execstart)
    assert "${INSTANCE%%-*}" in unescaped

    script_match = re.search(r"/bin/bash -c '(.*)'\s*$", unescaped)
    assert script_match is not None, "ExecStart is not a /bin/bash -c '...' invocation"
    script = script_match.group(1)

    statements = [s.strip() for s in script.split(";")]
    venue_statements = [s for s in statements if s.startswith("VENUE=")]
    assert len(venue_statements) == 1
    venue_stmt = venue_statements[0]
    assert venue_stmt == 'VENUE="${INSTANCE%%-*}"'

    bash = _bash_executable()
    cases = [
        ("hyperliquid-btc-ict", "hyperliquid"),
        ("oanda-eur-usd-ict", "oanda"),
        ("hyperliquid-btc-fixed_r_2-none", "hyperliquid"),
        ("hyperliquid", "hyperliquid"),  # degenerate: no hyphen at all
    ]
    for instance, expected_venue in cases:
        proc = subprocess.run(
            [bash, "-c", f"INSTANCE={instance!r}; {venue_stmt}; printf '%s' \"$VENUE\""],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout == expected_venue, (
            f"instance {instance!r}: expected venue {expected_venue!r}, got {proc.stdout!r}"
        )


def test_paper_unit_execstart_requires_nonempty_config() -> None:
    """Minors: an empty/missing .conf file must fail fast and distinctly (exit 78, sysexits
    EX_CONFIG) rather than silently invoking `swingforge paper --config ""`. `cat` is faked
    here via a bash function so the test needs neither a real
    /etc/swingforge/paper/*.conf file nor Windows<->POSIX path translation through Git Bash."""
    text = (DEPLOY_DIR / "swingforge-paper@.service").read_text(encoding="utf-8")
    match = re.search(r"^ExecStart=(.+)$", text, re.MULTILINE)
    assert match is not None
    unescaped = _systemd_unescape(match.group(1))
    script_match = re.search(r"/bin/bash -c '(.*)'\s*$", unescaped)
    assert script_match is not None
    script = script_match.group(1)

    assert 'CONF="$(cat "/etc/swingforge/paper/$INSTANCE.conf")"' in script
    assert '[[ -n "$CONF" ]] || { echo "empty config for instance $INSTANCE" >&2; exit 78; }' in script
    assert 'paper --venue "$VENUE" --config "$CONF"' in script

    prefix, marker, _rest = script.partition("exec /opt/swingforge")
    assert marker, "expected the script to end in `exec /opt/swingforge/...`"

    bash = _bash_executable()
    for conf_content, expect_rc in [("ict|fixed_r_2|london_ny|hyperliquid:BTC", 0), ("", 78)]:
        wrapped = (
            "cat() { printf '%s' \"$FAKE_CONF\"; }; "
            f"INSTANCE=hyperliquid-btc-ict; FAKE_CONF={conf_content!r}; "
            f'{prefix} printf "VENUE=%s CONF=%s" "$VENUE" "$CONF"'
        )
        proc = subprocess.run([bash, "-c", wrapped], capture_output=True, text=True)
        assert proc.returncode == expect_rc, proc.stderr
        if expect_rc == 0:
            assert proc.stdout == f"VENUE=hyperliquid CONF={conf_content}"
        else:
            assert "empty config for instance hyperliquid-btc-ict" in proc.stderr


def test_backfill_execstart_is_flock_wrapped_and_runs_both_venues_reporting_either_failure() -> None:
    """I3 + I9: both venue backfills run inside one ExecStart, wrapped in a non-blocking
    `flock -n` against /var/lib/swingforge/.venue.lock (fail fast rather than queue behind
    a manual run already holding it). Both venues must always run - a failure in the first
    must not skip the second - and the unit's own exit code must reflect either failing."""
    text = (DEPLOY_DIR / "swingforge-backfill.service").read_text(encoding="utf-8")
    exec_lines = [line for line in text.splitlines() if line.startswith("ExecStart=")]
    assert len(exec_lines) == 1
    raw = exec_lines[0][len("ExecStart=") :]
    assert raw.startswith("/usr/bin/flock -n /var/lib/swingforge/.venue.lock "), (
        "backfill ExecStart must be wrapped in a non-blocking flock on the shared venue lock"
    )

    script_match = re.search(r"/bin/bash -c '(.*)'\s*$", raw)
    assert script_match is not None
    script = _systemd_unescape(script_match.group(1))
    hl_idx = script.index("backfill --venue hyperliquid --years 4")
    oanda_idx = script.index("backfill --venue oanda --years 4")
    assert hl_idx < oanda_idx, "hyperliquid backfill must run before oanda's"

    bash = _bash_executable()
    for hl_ok, oanda_ok in [(True, True), (False, True), (True, False), (False, False)]:
        stub = re.sub(
            r"/usr/bin/sudo\b[^;]*backfill --venue hyperliquid --years 4",
            f"(echo HL_RAN; exit {0 if hl_ok else 1})",
            script,
        )
        stub = re.sub(
            r"/usr/bin/sudo\b[^;]*backfill --venue oanda --years 4",
            f"(echo OANDA_RAN; exit {0 if oanda_ok else 1})",
            stub,
        )
        proc = subprocess.run([bash, "-c", stub], capture_output=True, text=True)
        assert "HL_RAN" in proc.stdout, f"hyperliquid backfill must always run ({hl_ok=} {oanda_ok=})"
        assert "OANDA_RAN" in proc.stdout, (
            f"oanda backfill must always run even if hyperliquid failed ({hl_ok=} {oanda_ok=})"
        )
        if hl_ok and oanda_ok:
            assert proc.returncode == 0, proc.stderr
        else:
            assert proc.returncode != 0, "unit must fail if either venue's backfill failed"
