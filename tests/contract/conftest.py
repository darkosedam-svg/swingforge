"""Shared vcrpy configuration for contract tests (design spec section 8).

Contract tests hit real venue schemas but never a live network call in normal CI: they
replay a pre-recorded cassette under `tests/contract/cassettes/`. When a test's cassette
file is missing, it is skipped rather than attempting a live call — the fast unit suite
must stay green with no cassettes present. See `tests/contract/README.md` to record one.

Set `SWINGFORGE_RECORD=1` to switch every `vcr` fixture in this session to
`record_mode="once"` (record if the cassette is missing, otherwise replay) for the human
recording step; it is unset (`record_mode="none"`, replay-only) in ordinary runs.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import vcr as vcrpy

CASSETTE_DIR = Path(__file__).parent / "cassettes"

_RECORD_MODE = "once" if os.environ.get("SWINGFORGE_RECORD") else "none"


_DROPPED_RESPONSE_HEADERS = {"set-cookie", "cf-ray", "x-brokerid", "report-to", "nel"}


def _without_cookies(response: dict) -> dict:
    """Keep a venue's cookies and request ids out of a cassette. Anonymous bot-management
    cookies on the public venues; on a credentialed one a session cookie would be a secret."""
    response["headers"] = {
        name: value
        for name, value in response["headers"].items()
        if name.lower() not in _DROPPED_RESPONSE_HEADERS
    }
    return response


@pytest.fixture
def vcr() -> vcrpy.VCR:
    """A `VCR` instance configured per the design spec: no secrets ever hit a cassette."""
    return vcrpy.VCR(
        record_mode=_RECORD_MODE,
        filter_headers=["Authorization", "Cookie"],
        before_record_response=_without_cookies,
        cassette_library_dir=str(CASSETTE_DIR),
    )


def skip_if_no_cassette(cassette_name: str) -> None:
    """Skip the current test when its cassette is missing and we are not recording."""
    if _RECORD_MODE == "once":
        return
    path = CASSETTE_DIR / cassette_name
    if not path.exists():
        pytest.skip(
            f"no cassette recorded at {path}; see tests/contract/README.md to record one "
            "(SWINGFORGE_RECORD=1 with real credentials)"
        )
