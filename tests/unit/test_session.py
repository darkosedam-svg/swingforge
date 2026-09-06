"""Unit tests for `strategies/session.py` — the entry-time session gate."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from swingforge.core.types import Bar, Instrument
from swingforge.strategies.session import SessionFilter, allowed

FX_INSTRUMENT = Instrument(
    venue="oanda",
    symbol="EUR_USD",
    tick_size=Decimal("0.0001"),
    contract_multiplier=Decimal("1"),
    quote_ccy="USD",
    session_profile="fx",
)
PERP_INSTRUMENT = Instrument(
    venue="hyperliquid",
    symbol="BTC",
    tick_size=Decimal("0.01"),
    contract_multiplier=Decimal("1"),
    quote_ccy="USD",
    session_profile="perp",
)


def _bar(ts: datetime, instrument: Instrument = FX_INSTRUMENT) -> Bar:
    return Bar(
        instrument=instrument,
        tf="4h",
        ts_open=ts,
        open=1.0,
        high=1.0,
        low=1.0,
        close=1.0,
        volume=1.0,
    )


# Monday 2026-01-05 .. Sunday 2026-01-11 (Python weekday(): Mon=0 .. Sun=6)
MONDAY = datetime(2026, 1, 5, tzinfo=UTC)
SATURDAY = datetime(2026, 1, 10, tzinfo=UTC)
SUNDAY = datetime(2026, 1, 11, tzinfo=UTC)


@pytest.mark.parametrize(
    "ts,profile,mode,expected",
    [
        # mode "none" - always True regardless of hour/profile.
        (MONDAY.replace(hour=0), "fx", "none", True),
        (MONDAY.replace(hour=23), "perp", "none", True),
        (SATURDAY.replace(hour=12), "perp", "none", True),
        # mode "london_ny" - [07:00, 21:00) UTC, same for both profiles.
        (MONDAY.replace(hour=6, minute=59), "fx", "london_ny", False),
        (MONDAY.replace(hour=7, minute=0), "fx", "london_ny", True),
        (MONDAY.replace(hour=20, minute=59), "fx", "london_ny", True),
        (MONDAY.replace(hour=21, minute=0), "fx", "london_ny", False),
        (MONDAY.replace(hour=7, minute=0), "perp", "london_ny", True),
        # mode "active", fx profile - identical to london_ny.
        (MONDAY.replace(hour=6, minute=59), "fx", "active", False),
        (MONDAY.replace(hour=7, minute=0), "fx", "active", True),
        (MONDAY.replace(hour=20, minute=59), "fx", "active", True),
        (MONDAY.replace(hour=21, minute=0), "fx", "active", False),
        # A Saturday should NOT matter for fx active (only perp excludes weekends).
        (SATURDAY.replace(hour=12), "fx", "active", True),
        # mode "active", perp profile - excludes [00:00, 07:00) and Sat/Sun.
        (MONDAY.replace(hour=6, minute=59), "perp", "active", False),
        (MONDAY.replace(hour=7, minute=0), "perp", "active", True),
        (MONDAY.replace(hour=0, minute=0), "perp", "active", False),
        (MONDAY.replace(hour=23, minute=59), "perp", "active", True),
        (SATURDAY.replace(hour=12), "perp", "active", False),
        (SUNDAY.replace(hour=12), "perp", "active", False),
    ],
)
def test_allowed_table(ts: datetime, profile: str, mode: str, expected: bool) -> None:
    instrument = FX_INSTRUMENT if profile == "fx" else PERP_INSTRUMENT
    assert allowed(_bar(ts, instrument), profile, mode) is expected  # type: ignore[arg-type]


def test_session_filter_is_callable_and_matches_allowed() -> None:
    gate = SessionFilter(mode="active", profile="perp")
    assert gate(_bar(MONDAY.replace(hour=7), PERP_INSTRUMENT)) is True
    assert gate(_bar(SATURDAY.replace(hour=7), PERP_INSTRUMENT)) is False


def test_session_filter_none_mode_always_true() -> None:
    gate = SessionFilter(mode="none", profile="fx")
    assert gate(_bar(SATURDAY.replace(hour=3), FX_INSTRUMENT)) is True


def test_session_filter_reset_is_a_no_op() -> None:
    """`reset()` exists (so a runner can reset every component uniformly) and is a no-op:
    `SessionFilter` carries no per-instance state, so behaviour is unchanged after calling it."""
    gate = SessionFilter(mode="active", profile="perp")
    assert gate.reset() is None
    assert gate(_bar(MONDAY.replace(hour=7), PERP_INSTRUMENT)) is True
    assert gate(_bar(SATURDAY.replace(hour=7), PERP_INSTRUMENT)) is False
