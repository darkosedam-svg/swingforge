"""OANDA cost model: spread from the fill bar's bid/ask, and swap at the 21:00 UTC rollover.

No commission (OANDA's practice/retail pricing is spread-only) and no per-fill slippage —
those are zero by construction, matching the design spec's OANDA cost paragraph.
"""

from __future__ import annotations

import types
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, time, timedelta

from swingforge.adapters.oanda.bars import ApiLike
from swingforge.core.types import Bar, CostBreakdown, Order, Position

__all__ = ["Financing", "OandaCosts", "load_financing"]

_TF_DURATION: dict[str, timedelta] = {
    "1h": timedelta(hours=1),
    "4h": timedelta(hours=4),
    "1d": timedelta(hours=24),
}

_DAY_NAME_TO_WEEKDAY: dict[str, int] = {
    "MONDAY": 0,
    "TUESDAY": 1,
    "WEDNESDAY": 2,
    "THURSDAY": 3,
    "FRIDAY": 4,
    "SATURDAY": 5,
    "SUNDAY": 6,
}
"""Python `date.weekday()` convention: Monday=0 ... Sunday=6."""


@dataclass(frozen=True)
class Financing:
    """One instrument's OANDA financing rates, from `AccountInstruments`.

    `long_rate`/`short_rate` are annualised fractions (negative = you pay). `days_charged`
    maps a `date.weekday()` int to the number of days charged for a rollover landing on
    that weekday — OANDA typically charges 3 days on Wednesday to cover the weekend.

    `days_charged` is stored as a `types.MappingProxyType` (N6): a frozen dataclass only
    blocks *reassigning* an attribute, not mutating a mutable value held by one, so a plain
    `dict` here would let `financing.days_charged[k] = v` silently succeed on a supposedly
    immutable value. Wrapping it makes that raise `TypeError`, and the accepted input can
    still be any `Mapping` (a plain dict included) — it is copied into the proxy in
    `__post_init__`. `__hash__` is overridden explicitly because the default
    dataclass-generated one would hash the raw `days_charged` field, and a mappingproxy is
    itself unhashable (it just forwards to its underlying mapping); hashing
    `tuple(sorted(days_charged.items()))` instead gives a real, order-independent hash.
    """

    long_rate: float
    short_rate: float
    days_charged: Mapping[int, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "days_charged", types.MappingProxyType(dict(self.days_charged)))

    def __hash__(self) -> int:
        return hash((self.long_rate, self.short_rate, tuple(sorted(self.days_charged.items()))))


def _rollover_boundary(ts_open: datetime, duration: timedelta, hour: int) -> datetime | None:
    """The 21:00-UTC-style rollover instant in `(ts_open, ts_open + duration]`, if any."""
    end = ts_open + duration
    day = ts_open.date()
    for offset in (-1, 0, 1):
        candidate = datetime.combine(day + timedelta(days=offset), time(hour=hour), tzinfo=UTC)
        if ts_open < candidate <= end:
            return candidate
    return None


class OandaCosts:
    """Spread-only entry cost, and 21:00-UTC-rollover swap carry."""

    def __init__(self, financing: Mapping[str, Financing] | None = None, rollover_hour_utc: int = 21) -> None:
        self.financing: dict[str, Financing] = dict(financing) if financing else {}
        self.rollover_hour_utc = rollover_hour_utc
        self._warned_missing_quote: set[str] = set()

    def entry(self, order: Order, bar: Bar) -> CostBreakdown:
        if bar.bid_close is None or bar.ask_close is None:
            symbol = order.instrument.symbol
            if symbol not in self._warned_missing_quote:
                self._warned_missing_quote.add(symbol)
                warnings.warn(
                    f"{symbol!r} bar has no bid/ask close; treating spread as zero for this "
                    "and any further fill on this instrument until one is seen with a quote "
                    "(only warns once per instrument)",
                    stacklevel=2,
                )
            return CostBreakdown()
        spread = (bar.ask_close - bar.bid_close) / 2 * order.qty
        return CostBreakdown(spread=spread, commission=0.0, funding=0.0, slippage=0.0)

    def carry(self, position: Position, bar: Bar) -> float:
        duration = _TF_DURATION[bar.tf]
        rollover = _rollover_boundary(bar.ts_open, duration, self.rollover_hour_utc)
        if rollover is None:
            return 0.0
        financing = self.financing.get(position.instrument.symbol)
        if financing is None:
            return 0.0
        rate = financing.long_rate if position.direction == 1 else financing.short_rate
        days_charged = financing.days_charged.get(rollover.weekday(), 1)
        notional = position.qty * bar.close * float(position.instrument.contract_multiplier)
        return -rate * notional * days_charged / 365


def load_financing(api: ApiLike, account_id: str, name: str) -> Financing:
    """Fetch and shape `AccountInstruments` financing data for one instrument.

    Raises `ValueError` (I6) if `name` is not among the account's instruments, rather than
    letting a bare `next()` over an empty match raise an opaque `StopIteration`.
    """
    from oandapyV20.endpoints.accounts import AccountInstruments  # type: ignore[import-untyped]

    endpoint = AccountInstruments(accountID=account_id, params={"instruments": name})
    response = api.request(endpoint)
    entry = next((e for e in response["instruments"] if e["name"] == name), None)
    if entry is None:
        raise ValueError(f"{name!r} not in AccountInstruments for account {account_id}")
    financing = entry["financing"]
    days_charged = {
        _DAY_NAME_TO_WEEKDAY[day["dayOfWeek"].upper()]: int(day["daysCharged"])
        for day in financing["financingDaysOfWeek"]
    }
    return Financing(
        long_rate=float(financing["longRate"]),
        short_rate=float(financing["shortRate"]),
        days_charged=days_charged,
    )
