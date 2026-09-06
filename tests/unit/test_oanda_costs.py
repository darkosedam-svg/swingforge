"""Unit tests for `swingforge.adapters.oanda.costs`."""

from __future__ import annotations

import warnings
from datetime import UTC, datetime

import pytest

from swingforge.adapters.oanda.bars import oanda_instrument
from swingforge.adapters.oanda.costs import Financing, OandaCosts, load_financing
from swingforge.core.types import Bar, Order, Position

EUR_USD = oanda_instrument("EUR_USD")


def _bar(ts_open: datetime, tf: str = "4h", bid: float | None = 1.0999, ask: float | None = 1.1001) -> Bar:
    return Bar(
        instrument=EUR_USD,
        tf=tf,
        ts_open=ts_open,
        open=1.1,
        high=1.101,
        low=1.099,
        close=1.1,
        volume=100.0,
        bid_close=bid,
        ask_close=ask,
    )


def _order(qty: float = 10_000.0, direction: int = 1) -> Order:
    return Order(
        id="o1",
        instrument=EUR_USD,
        direction=direction,
        qty=qty,
        kind="market",
        price=None,
        expires_at_bar=None,
        leg="entry",
    )


def test_entry_spread_arithmetic() -> None:
    costs = OandaCosts()
    bar = _bar(datetime(2024, 1, 2, 1, tzinfo=UTC), bid=1.0998, ask=1.1002)
    order = _order(qty=10_000.0)

    breakdown = costs.entry(order, bar)

    assert breakdown.spread == pytest.approx((1.1002 - 1.0998) / 2 * 10_000.0)
    assert breakdown.commission == 0.0
    assert breakdown.slippage == 0.0
    assert breakdown.funding == 0.0


def test_entry_spread_zero_when_bid_ask_missing() -> None:
    """Minor: a bar with no bid/ask must warn (once) rather than silently return a zero
    spread with no signal that data is missing."""
    costs = OandaCosts()
    bar = _bar(datetime(2024, 1, 2, 1, tzinfo=UTC), bid=None, ask=None)
    order = _order()

    with pytest.warns(UserWarning, match="bid/ask"):
        breakdown = costs.entry(order, bar)

    assert breakdown.total == 0.0


def test_entry_missing_bid_ask_warns_only_once_per_instrument() -> None:
    """Minor: repeated fills on the same instrument with no bid/ask must warn once, not
    spam a warning on every fill."""
    costs = OandaCosts()
    bar = _bar(datetime(2024, 1, 2, 1, tzinfo=UTC), bid=None, ask=None)
    order = _order()

    with pytest.warns(UserWarning, match="bid/ask"):
        costs.entry(order, bar)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        breakdown = costs.entry(order, bar)  # second call on the same instrument: silent

    assert breakdown.total == 0.0


def test_carry_zero_when_bar_does_not_span_rollover() -> None:
    financing = {"EUR_USD": Financing(long_rate=-0.02, short_rate=0.01, days_charged={2: 3})}
    costs = OandaCosts(financing=financing)
    # 4H bar 14:00-18:00 UTC does not reach 21:00.
    bar = _bar(datetime(2024, 1, 3, 14, tzinfo=UTC))
    position = Position(instrument=EUR_USD, direction=1, qty=10_000.0, avg_price=1.1, stop=1.09, target=None)

    assert costs.carry(position, bar) == 0.0


def test_carry_charges_once_on_the_wednesday_rollover_with_its_days_charged() -> None:
    # Wednesday 2024-01-03, 21:00 UTC rollover typically charges 3 days (covers the weekend).
    financing = {"EUR_USD": Financing(long_rate=-0.02, short_rate=0.01, days_charged={2: 3})}
    costs = OandaCosts(financing=financing)
    bar = _bar(datetime(2024, 1, 3, 20, tzinfo=UTC))  # 20:00-24:00 spans 21:00
    long_position = Position(
        instrument=EUR_USD, direction=1, qty=10_000.0, avg_price=1.1, stop=1.09, target=None
    )

    funding = costs.carry(long_position, bar)

    long_rate = -0.02
    expected_notional = 10_000.0 * bar.close
    assert funding == pytest.approx(-long_rate * expected_notional * 3 / 365)
    assert funding > 0  # negative long_rate means the long pays


def test_carry_sign_for_short_with_positive_rate() -> None:
    financing = {"EUR_USD": Financing(long_rate=-0.02, short_rate=0.01, days_charged={2: 3})}
    costs = OandaCosts(financing=financing)
    bar = _bar(datetime(2024, 1, 3, 20, tzinfo=UTC))
    short_position = Position(
        instrument=EUR_USD, direction=-1, qty=10_000.0, avg_price=1.1, stop=1.11, target=None
    )

    funding = costs.carry(short_position, bar)

    expected_notional = 10_000.0 * bar.close
    assert funding == pytest.approx(-0.01 * expected_notional * 3 / 365)
    assert funding < 0  # positive short_rate means the short is paid


def test_carry_uses_default_one_day_charge_when_weekday_missing() -> None:
    financing = {"EUR_USD": Financing(long_rate=-0.02, short_rate=0.01, days_charged={})}
    costs = OandaCosts(financing=financing)
    bar = _bar(datetime(2024, 1, 3, 20, tzinfo=UTC))
    position = Position(instrument=EUR_USD, direction=1, qty=10_000.0, avg_price=1.1, stop=1.09, target=None)

    funding = costs.carry(position, bar)

    long_rate = -0.02
    expected_notional = 10_000.0 * bar.close
    assert funding == pytest.approx(-long_rate * expected_notional * 1 / 365)


def test_financing_is_hashable() -> None:
    """N6: `Financing` is `frozen=True` and must be genuinely hashable — `days_charged`
    can no longer be a plain (unhashable) dict.
    """
    financing = Financing(long_rate=-0.02, short_rate=0.01, days_charged={2: 3})
    other = Financing(long_rate=-0.02, short_rate=0.01, days_charged={2: 3})

    assert hash(financing) == hash(other)
    assert financing == other


def test_financing_days_charged_is_immutable() -> None:
    """N6: `days_charged` must reject in-place mutation, not just attribute reassignment
    (the dataclass being `frozen=True` already blocks reassignment, but a plain dict value
    could still be mutated in place)."""
    financing = Financing(long_rate=-0.02, short_rate=0.01, days_charged={2: 3})

    with pytest.raises(TypeError):
        financing.days_charged[5] = 1  # type: ignore[index]


def test_carry_zero_when_instrument_not_in_financing_table() -> None:
    costs = OandaCosts(financing={})
    bar = _bar(datetime(2024, 1, 3, 20, tzinfo=UTC))
    position = Position(instrument=EUR_USD, direction=1, qty=10_000.0, avg_price=1.1, stop=1.09, target=None)

    assert costs.carry(position, bar) == 0.0


def test_rollover_boundary_at_a_bar_that_opens_exactly_on_it() -> None:
    # ts_open < rollover <= ts_open + duration: a bar opening exactly at 21:00 does NOT
    # itself span a rollover that already happened at its open.
    financing = {"EUR_USD": Financing(long_rate=-0.02, short_rate=0.01, days_charged={2: 3})}
    costs = OandaCosts(financing=financing)
    bar = _bar(datetime(2024, 1, 3, 21, tzinfo=UTC))
    position = Position(instrument=EUR_USD, direction=1, qty=10_000.0, avg_price=1.1, stop=1.09, target=None)

    # The next rollover is the following day at 21:00, outside this 4H bar's span.
    assert costs.carry(position, bar) == 0.0


class FakeAccountApi:
    def __init__(self, entries: list[dict]) -> None:
        self.entries = entries

    def request(self, endpoint) -> dict:
        return {"instruments": self.entries}


def test_load_financing_shapes_rates_and_days_charged() -> None:
    entries = [
        {
            "name": "EUR_USD",
            "financing": {
                "longRate": "-0.0187",
                "shortRate": "0.0091",
                "financingDaysOfWeek": [
                    {"dayOfWeek": "MONDAY", "daysCharged": 1},
                    {"dayOfWeek": "TUESDAY", "daysCharged": 1},
                    {"dayOfWeek": "WEDNESDAY", "daysCharged": 3},
                    {"dayOfWeek": "THURSDAY", "daysCharged": 1},
                    {"dayOfWeek": "FRIDAY", "daysCharged": 1},
                    {"dayOfWeek": "SATURDAY", "daysCharged": 0},
                    {"dayOfWeek": "SUNDAY", "daysCharged": 0},
                ],
            },
        }
    ]
    financing = load_financing(FakeAccountApi(entries), "acct-1", "EUR_USD")

    assert financing.long_rate == pytest.approx(-0.0187)
    assert financing.short_rate == pytest.approx(0.0091)
    assert financing.days_charged[2] == 3  # Wednesday
    assert financing.days_charged[5] == 0  # Saturday


def test_load_financing_raises_on_unknown_instrument_name() -> None:
    """I6: an instrument name absent from `AccountInstruments` must raise a clear
    `ValueError`, not let a bare `next()` over an empty match raise `StopIteration`."""
    entries = [
        {
            "name": "EUR_USD",
            "financing": {
                "longRate": "-0.0187",
                "shortRate": "0.0091",
                "financingDaysOfWeek": [{"dayOfWeek": "WEDNESDAY", "daysCharged": 3}],
            },
        }
    ]
    with pytest.raises(ValueError, match="NOT_LISTED"):
        load_financing(FakeAccountApi(entries), "acct-1", "NOT_LISTED")
