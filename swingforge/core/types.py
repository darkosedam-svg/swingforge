"""Frozen domain models shared by every layer (design spec section 3).

Every model is immutable (``frozen=True``): once a bar, order, fill or trade exists it
describes something that happened and cannot be edited in place. Prices and sizes are
floats internally; ``tick_size`` rounding happens at the broker boundary via
:func:`round_to_tick`. NaN and infinity are rejected at construction, as are non-positive
order/position sizes and a stop on the wrong side of the entry.

This module holds models, validators and the tick helper only. No engine logic lives here.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from decimal import ROUND_HALF_EVEN, Decimal
from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, field_validator, model_validator

__all__ = [
    "TF",
    "Bar",
    "CostBreakdown",
    "ExitEvent",
    "Fill",
    "Instrument",
    "Order",
    "Position",
    "Resolution",
    "Signal",
    "Trade",
    "round_to_tick",
]

TF = Literal["1h", "4h", "1d"]
"""Timeframes the system trades: 1H sub-bars, 4H execution, Daily bias."""

_TF_SPAN: dict[str, timedelta] = {
    "1h": timedelta(hours=1),
    "4h": timedelta(hours=4),
    "1d": timedelta(hours=24),
}
"""How long one bar of each timeframe lasts."""

_FINER_TFS: dict[str, tuple[str, ...]] = {"1h": (), "4h": ("1h",), "1d": ("1h", "4h")}
"""Which timeframes may appear inside a bar of each timeframe."""


def _finite(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError("must be a finite number (NaN and infinity are rejected)")
    return value


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise ValueError("must be timezone-aware; naive datetimes are rejected")
    return value.astimezone(UTC)


Finite = Annotated[float, AfterValidator(_finite)]
"""A float that is neither NaN nor infinite."""

PositiveFinite = Annotated[float, AfterValidator(_finite), Field(gt=0)]
"""A finite float strictly greater than zero."""

UtcDatetime = Annotated[datetime, AfterValidator(_utc)]
"""A timezone-aware datetime, normalised to UTC."""


class Instrument(BaseModel):
    """A tradable symbol on one venue, with the metadata sizing and rounding need."""

    model_config = ConfigDict(frozen=True)

    venue: Literal["hyperliquid", "oanda", "okx"]
    symbol: str
    tick_size: Decimal
    contract_multiplier: Decimal
    quote_ccy: str
    session_profile: Literal["fx", "perp"]

    @field_validator("tick_size")
    @classmethod
    def _tick_size_positive(cls, value: Decimal) -> Decimal:
        if value <= 0:
            raise ValueError("tick_size must be > 0")
        return value


class Bar(BaseModel):
    """One closed OHLCV bar.

    ``ts_open`` is the bar's opening timestamp and must be timezone-aware; any offset is
    normalised to UTC. ``subbars`` carries the finer-grained bars inside this one (1H bars
    inside a 4H bar) so the fill resolver can order a stop and a target that both sit
    inside the same bar; it is empty when the source has no sub-bar data.

    A subbar must genuinely be inside this bar: same ``instrument``, a finer timeframe (1H
    inside 4H; 1H or 4H inside Daily), ``ts_open`` within ``[ts_open, ts_open + span)`` for
    this bar's span, and a range contained by this one (``low >= parent.low`` and
    ``high <= parent.high``). Subbars are ordered oldest-first.
    """

    model_config = ConfigDict(frozen=True)

    instrument: Instrument
    tf: TF
    ts_open: UtcDatetime
    open: Finite
    high: Finite
    low: Finite
    close: Finite
    volume: Finite
    bid_close: Finite | None = None
    ask_close: Finite | None = None
    subbars: tuple[Bar, ...] = ()

    @model_validator(mode="after")
    def _check_range(self) -> Bar:
        if self.volume < 0:
            raise ValueError("volume must be >= 0")
        if self.low > min(self.open, self.close):
            raise ValueError("low must be <= min(open, close)")
        if self.high < max(self.open, self.close):
            raise ValueError("high must be >= max(open, close)")
        return self

    @model_validator(mode="after")
    def _check_subbars(self) -> Bar:
        if not self.subbars:
            return self
        finer = _FINER_TFS[self.tf]
        end = self.ts_open + _TF_SPAN[self.tf]
        previous: datetime | None = None
        for sub in self.subbars:
            if sub.instrument != self.instrument:
                raise ValueError("every subbar must belong to the same instrument as its parent")
            if sub.tf not in finer:
                raise ValueError(f"a {self.tf} bar may only contain {finer or 'no'} subbars, got {sub.tf!r}")
            if not self.ts_open <= sub.ts_open < end:
                raise ValueError(
                    f"subbar ts_open {sub.ts_open.isoformat()} is outside the parent span "
                    f"[{self.ts_open.isoformat()}, {end.isoformat()})"
                )
            if sub.low < self.low:
                raise ValueError("subbar low must be >= the parent low")
            if sub.high > self.high:
                raise ValueError("subbar high must be <= the parent high")
            if previous is not None and sub.ts_open < previous:
                raise ValueError("subbars must be ordered oldest-first")
            previous = sub.ts_open
        return self


Bar.model_rebuild()


class Signal(BaseModel):
    """A strategy's intent to enter.

    ``stop`` is the invalidation level, never a distance, and must lie strictly on the
    wrong side of ``entry`` for the signal's direction. ``structure_target`` is the nearest
    opposing structural level when the strategy knows one, else ``None``.

    ``expires_in_bars`` is a count of 4H bars, relative to the bar the signal was emitted
    on. The engine converts it to the absolute bar index the entry order carries:
    ``expires_at_bar = ctx.bar_index + signal.expires_in_bars``, where ``ctx.bar_index`` is
    the bar the signal was emitted on. The order is still live *on* that bar and is dropped
    once the broker sees ``bar_index > expires_at_bar``.
    """

    model_config = ConfigDict(frozen=True)

    direction: Literal[1, -1]
    entry: Finite
    stop: Finite
    structure_target: Finite | None
    tag: str
    expires_in_bars: int

    @model_validator(mode="after")
    def _stop_on_wrong_side(self) -> Signal:
        if (self.entry - self.stop) * self.direction <= 0:
            raise ValueError(
                "stop must be strictly on the wrong side of entry "
                f"(direction={self.direction}, entry={self.entry}, stop={self.stop})"
            )
        return self


class CostBreakdown(BaseModel):
    """The four cost components attributed to a fill, in quote currency."""

    model_config = ConfigDict(frozen=True)

    spread: Finite = 0.0
    commission: Finite = 0.0
    funding: Finite = 0.0
    slippage: Finite = 0.0

    @property
    def total(self) -> float:
        return self.spread + self.commission + self.funding + self.slippage


class Order(BaseModel):
    """An order sent to a broker.

    ``leg`` says what the order is for: the entry, or one of the exit legs an
    :class:`~swingforge.strategies.base.ExitRule` attaches to a trade. ``trade_id`` groups
    the exit legs of one trade so a broker can replace a pending leg (same ``trade_id`` and
    ``leg``) when a rule updates it. The engine assigns the trade id when it submits the
    entry order, so an entry carries ``trade_id`` too (its ``(trade_id, "entry")`` key is
    unique, hence never a replacement) and the broker can attribute the entry fill to the
    trade — it needs the entry price to apply the breakeven stop after a partial fill.
    Only orders created outside the engine may leave ``trade_id`` as ``None``.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    instrument: Instrument
    direction: Literal[1, -1]
    qty: PositiveFinite
    kind: Literal["limit", "market", "stop"]
    price: Finite | None
    expires_at_bar: int | None
    leg: Literal["entry", "stop", "target", "partial", "time"]
    trade_id: str | None = None

    @model_validator(mode="after")
    def _price_matches_kind(self) -> Order:
        if self.kind in ("limit", "stop") and self.price is None:
            raise ValueError(f"a {self.kind} order requires a price")
        return self


class Fill(BaseModel):
    """An executed order, whole or partial, with the costs it incurred."""

    model_config = ConfigDict(frozen=True)

    order_id: str
    ts: UtcDatetime
    price: Finite
    qty: Finite
    cost: CostBreakdown
    leg: Literal["entry", "stop", "target", "partial", "time"]
    trade_id: str | None = None


class Trade(BaseModel):
    """One round trip: an entry fill plus the exit legs that closed it.

    ``risk_r`` is the money at risk per 1R, so every R figure on the trade is
    ``pnl / risk_r``. ``context_snapshot`` holds the serialised
    :class:`~swingforge.core.context.Context` state at entry, for post-hoc analysis:
    :meth:`~swingforge.core.context.Context.snapshot` is its producer, and the payload is
    the UTF-8 JSON document that method returns.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    instrument: Instrument
    direction: Literal[1, -1]
    entry_fill: Fill
    legs: tuple[Fill, ...] = ()
    stop: Finite
    target: Finite | None
    risk_r: PositiveFinite
    realized_r: Finite | None = None
    mae_r: Finite = 0.0
    mfe_r: Finite = 0.0
    regime: str = ""
    context_snapshot: bytes = b""
    opened_bar: int
    closed_bar: int | None = None


class Position(BaseModel):
    """A broker's view of open exposure in one instrument."""

    model_config = ConfigDict(frozen=True)

    instrument: Instrument
    direction: Literal[1, -1]
    qty: PositiveFinite
    avg_price: Finite
    stop: Finite
    target: Finite | None


class ExitEvent(BaseModel):
    """One exit the resolver decided happened inside a bar.

    ``target_index`` is the position of the hit level in the ``targets`` list handed to
    :meth:`~swingforge.adapters.base.ExitResolver.resolve`, and is ``None`` for a stop.

    The resolver reports price levels only; the caller maps ``target_index`` back to the
    pending :class:`Order` for that target and takes the ``leg`` label from that order — so
    a Partial's first target yields ``Fill(leg="partial")`` and the runner's target yields
    ``Fill(leg="target")``. Time stops are submitted as ``kind="market"`` orders and never
    reach the resolver.
    """

    model_config = ConfigDict(frozen=True)

    leg: Literal["stop", "target"]
    price: Finite
    target_index: int | None = None


class Resolution(BaseModel):
    """The resolver's verdict for one bar, in occurrence order.

    ``mode`` records how the verdict was reached: ``"subbars"`` when 1H bars ordered the
    levels, ``"pessimistic"`` when there were none and the stop was assumed first. Reports
    state the mode per instrument, so it is part of the result rather than a log line.
    """

    model_config = ConfigDict(frozen=True)

    events: tuple[ExitEvent, ...]
    mode: Literal["subbars", "pessimistic"]


def round_to_tick(price: float, tick_size: Decimal) -> float:
    """Round ``price`` to the nearest multiple of ``tick_size``, ties to even.

    Half-even keeps repeated rounding unbiased, which matters because every backtest price
    passes through here. Rounding is done in :class:`~decimal.Decimal` so a tick like
    ``0.01`` behaves exactly.
    """
    if not math.isfinite(price):
        raise ValueError("price must be a finite number")
    if tick_size <= 0:
        raise ValueError("tick_size must be > 0")
    steps = (Decimal(str(price)) / tick_size).quantize(Decimal(1), rounding=ROUND_HALF_EVEN)
    return float(steps * tick_size)
