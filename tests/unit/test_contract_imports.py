"""The dependency rules of spec section 2 are executable: `lint-imports` must pass.

This is the enforcement point for the layering contract. Every work unit codes against
it, so it runs on every commit rather than behind a marker.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = REPO_ROOT / ".importlinter"


def _lint_imports_executable() -> str:
    """Locate the `lint-imports` console script (``.exe`` on Windows)."""
    found = shutil.which("lint-imports")
    if found is not None:
        return found
    scripts_dir = Path(sys.executable).parent
    for candidate in (scripts_dir / "lint-imports.exe", scripts_dir / "lint-imports"):
        if candidate.exists():
            return str(candidate)
    pytest.fail("lint-imports is not installed; run `uv sync` first")


def test_import_contracts_hold() -> None:
    assert CONFIG.is_file(), f"missing import-linter config at {CONFIG}"
    proc = subprocess.run(
        [_lint_imports_executable(), "--config", str(CONFIG)],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert proc.returncode == 0, (
        f"lint-imports failed (exit {proc.returncode})\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )
