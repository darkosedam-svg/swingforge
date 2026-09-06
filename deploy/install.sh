#!/usr/bin/env bash
# deploy/install.sh -- idempotent VPS bootstrap for swingforge (PAPER TRADING ONLY).
#
# Usage (as root, e.g. via sudo): REPO_URL=<url> bash deploy/install.sh
#   REPO_URL defaults to the project's GitHub origin (see below) and is only needed if you
#   are installing from a fork or mirror.
#
# What this script does, in order, and nothing more:
#   1. creates the `swingforge` system user (no login shell) if it does not exist yet
#   2. installs `uv` to /usr/local/bin if it is not already on PATH
#   3. clones the repo to /opt/swingforge, or fast-forwards an existing checkout there
#   4. `uv sync --frozen --compile-bytecode` to build /opt/swingforge/.venv from the
#      committed lockfile
#   5. creates /var/lib/swingforge{,/reports} and /etc/swingforge/paper/
#   6. seeds /etc/swingforge.env from deploy/swingforge.env.example, but only if that path
#      does not already exist -- re-running this script never overwrites real credentials
#   7. installs the systemd units, `daemon-reload`s, enables (but does not start) the web
#      unit, and enables + starts the nightly backfill timer
#
# It deliberately does NOT start any `swingforge-paper@` instance and does NOT fill in
# /etc/swingforge.env -- see deploy/README.md for both of those (human) steps.

set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/darkosedam-svg/swingforge.git}"
BRANCH="${BRANCH:-main}"
INSTALL_DIR="/opt/swingforge"
DATA_DIR="/var/lib/swingforge"
ENV_FILE="/etc/swingforge.env"
PAPER_CONF_DIR="/etc/swingforge/paper"
SERVICE_USER="swingforge"
UNIT_DIR="/etc/systemd/system"

if [[ "$(id -u)" -ne 0 ]]; then
  echo "install.sh must be run as root (e.g. sudo bash deploy/install.sh)." >&2
  exit 1
fi

# This script drops to "$SERVICE_USER" via `sudo -u` at several points below (the git
# update and `uv sync`, plus every backfill/tournament/paper invocation the systemd units
# and README runbook make later, all run as the unprivileged service user, never as root)
# -- fail with a clear message up front rather than a confusing "sudo: command not found"
# partway through.
if ! command -v sudo >/dev/null 2>&1; then
  echo "install.sh requires 'sudo' (used to run git/uv/swingforge as the unprivileged" >&2
  echo "'$SERVICE_USER' user). Install it first, e.g.: apt-get install -y sudo" >&2
  exit 1
fi

# --- 1. system user ----------------------------------------------------------
if id -u "$SERVICE_USER" >/dev/null 2>&1; then
  echo "[1/7] user '$SERVICE_USER' already exists, skipping"
else
  echo "[1/7] creating system user '$SERVICE_USER'"
  # --no-create-home: this is a service account, not an interactive login; its systemd
  # units set WorkingDirectory explicitly so a real home directory is not needed. Passing
  # --home-dir still records /opt/swingforge as metadata (useful for e.g. `ssh -i ...` tools
  # that inspect /etc/passwd) without touching the filesystem, which matters here because
  # step 3 below is what actually creates that directory via `git clone`.
  useradd --system --no-create-home --home-dir "$INSTALL_DIR" --shell /usr/sbin/nologin "$SERVICE_USER"
fi

# --- 2. uv ---------------------------------------------------------------------
# Pinned to a specific release (rather than astral.sh/uv/install.sh's rolling "latest") so a
# re-run of this script months from now can't silently pick up an untested uv version. Bump
# this deliberately, not as a side effect of some other change. Trust assumption: this
# installer script is fetched over HTTPS from astral.sh and executed as root, same as the
# `curl | sh` pattern most VPS bootstrap docs already ask you to trust for other tools.
UV_VERSION="0.11.7"
if command -v uv >/dev/null 2>&1; then
  UV_BIN="$(command -v uv)"
  echo "[2/7] uv already installed at $UV_BIN, skipping"
elif [[ -x /usr/local/bin/uv ]]; then
  UV_BIN="/usr/local/bin/uv"
  echo "[2/7] uv already installed at $UV_BIN, skipping"
else
  echo "[2/7] installing uv $UV_VERSION to /usr/local/bin"
  curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" | UV_INSTALL_DIR=/usr/local/bin sh
  UV_BIN="/usr/local/bin/uv"
fi

# --- 3. clone or update the repo -----------------------------------------------
if [[ -d "$INSTALL_DIR/.git" ]]; then
  echo "[3/7] updating existing checkout at $INSTALL_DIR"
  # Run as "$SERVICE_USER" (not root): a previous run's chown below already made it the
  # owner of $INSTALL_DIR, and git's "dubious ownership" safety check trips whenever the
  # invoking uid doesn't match the repo directory's owning uid -- root is not exempt from
  # this. Matching the invoking user to the owning user keeps this idempotent without a
  # `git config --global --add safe.directory` workaround.
  sudo -u "$SERVICE_USER" git -C "$INSTALL_DIR" fetch --tags origin
  sudo -u "$SERVICE_USER" git -C "$INSTALL_DIR" checkout "$BRANCH"
  sudo -u "$SERVICE_USER" git -C "$INSTALL_DIR" pull --ff-only origin "$BRANCH"
elif [[ -e "$INSTALL_DIR" ]]; then
  echo "install.sh: $INSTALL_DIR exists and is not a git checkout; refusing to overwrite it." >&2
  echo "Move it aside (or remove it) and re-run." >&2
  exit 1
else
  echo "[3/7] cloning $REPO_URL ($BRANCH) into $INSTALL_DIR"
  git clone --branch "$BRANCH" "$REPO_URL" "$INSTALL_DIR"
fi
chown -R "$SERVICE_USER:$SERVICE_USER" "$INSTALL_DIR"

# --- 4. uv sync ------------------------------------------------------------------
echo "[4/7] uv sync --frozen --compile-bytecode"
(cd "$INSTALL_DIR" && sudo -u "$SERVICE_USER" "$UV_BIN" sync --frozen --compile-bytecode)

# --- 5. data dirs -------------------------------------------------------------------
echo "[5/7] creating $DATA_DIR, $DATA_DIR/reports, $PAPER_CONF_DIR"
install -d -o "$SERVICE_USER" -g "$SERVICE_USER" -m 0750 "$DATA_DIR" "$DATA_DIR/reports"
# Pre-create the venue lock so its ownership does not depend on who runs flock first.
install -o "$SERVICE_USER" -g "$SERVICE_USER" -m 0644 /dev/null "$DATA_DIR/.venue.lock"
# Owned by root, group-readable by swingforge: a human creates the per-instance .conf
# files here (as root, via the runbook) and the paper@ units only ever need to read them.
install -d -o root -g "$SERVICE_USER" -m 0750 "$PAPER_CONF_DIR"

# --- 6. env file -------------------------------------------------------------------
if [[ -f "$ENV_FILE" ]]; then
  echo "[6/7] $ENV_FILE already exists, leaving it untouched"
else
  echo "[6/7] seeding $ENV_FILE from deploy/swingforge.env.example"
  install -o root -g "$SERVICE_USER" -m 0640 \
    "$INSTALL_DIR/deploy/swingforge.env.example" "$ENV_FILE"
  echo "    -> fill in real credentials in $ENV_FILE before starting web/backfill units"
fi

# --- 7. systemd units --------------------------------------------------------------
echo "[7/7] installing units, daemon-reload, enabling web (not started) + backfill timer"
install -m 0644 "$INSTALL_DIR/deploy/swingforge-paper@.service" "$UNIT_DIR/swingforge-paper@.service"
install -m 0644 "$INSTALL_DIR/deploy/swingforge-web.service" "$UNIT_DIR/swingforge-web.service"
install -m 0644 "$INSTALL_DIR/deploy/swingforge-backfill.service" "$UNIT_DIR/swingforge-backfill.service"
install -m 0644 "$INSTALL_DIR/deploy/swingforge-backfill.timer" "$UNIT_DIR/swingforge-backfill.timer"

systemctl daemon-reload
# Enabled but NOT started: swingforge-web.service will fail to bind 0.0.0.0 until
# SWINGFORGE_TOKEN is set (see deploy/README.md's credentials step, which is what actually
# starts/restarts it once real values are in place).
systemctl enable swingforge-web.service
systemctl enable --now swingforge-backfill.timer

cat <<'EOF'

install.sh done.

Not started (see deploy/README.md):
  - /etc/swingforge.env may still need real credentials
  - swingforge-web.service is enabled but not started -- start it after filling in
    /etc/swingforge.env (the credentials step does this via `systemctl restart`)
  - no swingforge-paper@<instance> unit has been enabled or started
EOF
