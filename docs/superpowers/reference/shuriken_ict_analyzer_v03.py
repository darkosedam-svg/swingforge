"""
SHURIKEN ICT Signal Analyser — ported from v4 `ict_analyzer.py` (April 2026) into the lab, unchanged except:
  * `_find_sweep` direction corrected: sweep of Asia LOW -> "long", sweep of Asia HIGH -> "short"
    (the original returned the opposite, contradicting its own docstring/AsiaRange helpers; with that inversion
    `_find_bos` searched the wrong way and the pipeline returned None on the canonical Seek & Destroy scenario).
  * `_find_sweep` also requires the sweep candle to OPEN inside the range; otherwise the first pullback candle after
    a displacement that left the range is misread as an opposite-direction sweep.
  * `_find_bos` compared each candle's close against a running extreme that already INCLUDED that candle's own
    high/low, so BOS could only fire on candles closing within 0.02% of their own extreme. The neck line is now the
    extreme of the candles BEFORE the current one, as the docstring always said.
  * logging stripped, Candle takes floats directly. Everything else is the production logic.

Pipeline: Asia range (00:00-08:00 UTC, 1h) -> sweep of range extreme (exec TF) -> BOS -> FVG in displacement
-> Order Block (last opposing candle before BOS) -> OTE 61.8-79% retrace -> entry when price is in OTE/FVG/OB.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


@dataclass(frozen=True)
class Candle:
    t: int; o: float; h: float; l: float; c: float; v: float = 0.0
    @property
    def is_bullish(self): return self.c >= self.o
    @property
    def is_bearish(self): return self.c < self.o
    @property
    def body_high(self): return max(self.o, self.c)
    @property
    def body_low(self): return min(self.o, self.c)
    @property
    def body_size(self): return abs(self.c - self.o)
    @property
    def range_size(self): return self.h - self.l


def candles_from_raw(raw: list[dict]) -> list[Candle]:
    out = []
    for r in raw:
        try:
            out.append(Candle(int(r["t"]), float(r["o"]), float(r["h"]), float(r["l"]), float(r["c"]), float(r.get("v", 0))))
        except (KeyError, ValueError, TypeError):
            continue
    return sorted(out, key=lambda x: x.t)


@dataclass
class AsiaRange:
    high: float; low: float; mid: float = field(init=False)
    def __post_init__(self): self.mid = (self.high + self.low) / 2
    @property
    def width(self): return self.high - self.low
    def is_sweep_high(self, candle: Candle) -> bool: return candle.h > self.high and candle.c <= self.high
    def is_sweep_low(self, candle: Candle) -> bool: return candle.l < self.low and candle.c >= self.low


@dataclass
class FVG:
    direction: str; top: float; bottom: float; formed_at_idx: int
    @property
    def mid(self): return (self.top + self.bottom) / 2
    @property
    def size(self): return self.top - self.bottom
    def price_inside(self, p): return self.bottom <= p <= self.top


@dataclass
class OrderBlock:
    direction: str; high: float; low: float; formed_at_idx: int
    @property
    def mid(self): return (self.high + self.low) / 2
    def price_inside(self, p): return self.low <= p <= self.high


@dataclass
class OTEZone:
    direction: str; entry_top: float; entry_bottom: float; stop_loss: float; swing_high: float; swing_low: float
    def price_inside(self, p): return self.entry_bottom <= p <= self.entry_top


@dataclass
class ICTSetup:
    coin: str; direction: str; session: str; asia_range: AsiaRange; sweep_idx: int; bos_idx: int
    fvg: Optional[FVG]; order_block: Optional[OrderBlock]; ote_zone: OTEZone; current_price: float
    entry_triggered: bool; confidence: float; notes: list[str] = field(default_factory=list)


class ICTAnalyzer:
    ASIA_START_H = 0; ASIA_END_H = 8
    MIN_SWEEP_PIPS_PCT = 0.0003
    MIN_BOS_PCT = 0.0002
    MIN_DISPLACEMENT_ATR_MULT = 0.6
    OTE_RETRACE_LOW = 0.618; OTE_RETRACE_HIGH = 0.790
    W_FVG = 0.30; W_OB = 0.20; W_ATR = 0.25; W_SWEEP = 0.25
    SWEEP_LOOKBACK = 96

    def analyse(self, coin: str, candles_1h: list[Candle], candles_5m: list[Candle], session: str = "ny") -> Optional[ICTSetup]:
        if len(candles_1h) < 12 or len(candles_5m) < 30:
            return None
        current_price = candles_5m[-1].c
        asia = self._asia_range(candles_1h)
        if asia is None or asia.width < current_price * 0.001:
            return None
        sweep_idx, direction = self._find_sweep(candles_5m, asia, current_price)
        if sweep_idx is None:
            return None
        bos_idx, swing_high, swing_low = self._find_bos(candles_5m, sweep_idx, direction)
        if bos_idx is None:
            return None
        disp = candles_5m[sweep_idx: bos_idx + 5]
        fvg = self._find_fvg(disp, direction)
        ob = self._find_order_block(candles_5m, sweep_idx, bos_idx, direction)
        ote = self._build_ote(direction, swing_high, swing_low, current_price)
        triggered = self._entry_triggered(current_price, ote, fvg, ob, direction)
        atr = self._atr(candles_5m[-20:])
        conf = self._confidence(fvg, ob, disp, atr, candles_5m[sweep_idx], asia)
        notes = ([f"FVG {fvg.direction} sz={fvg.size:.5f}"] if fvg else []) + ([f"OB {ob.direction} mid={ob.mid:.4f}"] if ob else []) + [f"ATR={atr:.5f} conf={conf:.2f}"]
        return ICTSetup(coin, direction, session, asia, sweep_idx, bos_idx, fvg, ob, ote, current_price, triggered, conf, notes)

    def _asia_range(self, candles_1h):
        asia = []
        for c in reversed(candles_1h):
            h = datetime.fromtimestamp(c.t / 1000, tz=timezone.utc).hour
            if self.ASIA_START_H <= h < self.ASIA_END_H:
                asia.append(c)
            elif asia:
                break
        if len(asia) < 3:
            return None
        return AsiaRange(max(c.h for c in asia), min(c.l for c in asia))

    def _find_sweep(self, candles, asia, current_price):
        """Sweep of Asia LOW (stops under the range run, close back inside) -> LONG. Sweep of Asia HIGH -> SHORT.
        The candle must OPEN inside the range as well: without that, the first pullback candle after a displacement
        that already left the range reads as a fresh sweep in the opposite direction (second correction vs v4)."""
        min_pierce = current_price * self.MIN_SWEEP_PIPS_PCT
        lookback = min(len(candles) - 1, self.SWEEP_LOOKBACK)
        start = len(candles) - lookback
        for i in range(len(candles) - 1, start - 1, -1):
            c = candles[i]
            inside = asia.low <= c.c <= asia.high and asia.low <= c.o <= asia.high
            if c.l < asia.low - min_pierce and inside:
                return i, "long"
            if c.h > asia.high + min_pierce and inside:
                return i, "short"
        return None, None

    def _find_bos(self, candles, sweep_idx, direction):
        post = candles[sweep_idx + 1:]
        if len(post) < 3:
            return None, 0.0, 0.0
        min_delta = candles[sweep_idx].c * self.MIN_BOS_PCT
        if direction == "long":
            swing_low = min(c.l for c in post[:10]); running_high = candles[sweep_idx].h
            for i, c in enumerate(post):
                if c.c > running_high - min_delta and i >= 2 and c.body_size > 0:   # neck line = highs BEFORE this candle
                    return sweep_idx + 1 + i, max(running_high, c.h), swing_low
                running_high = max(running_high, c.h)
        else:
            swing_high = max(c.h for c in post[:10]); running_low = candles[sweep_idx].l
            for i, c in enumerate(post):
                if c.c < running_low + min_delta and i >= 2 and c.body_size > 0:
                    return sweep_idx + 1 + i, swing_high, min(running_low, c.l)
                running_low = min(running_low, c.l)
        return None, 0.0, 0.0

    def _find_fvg(self, candles, direction):
        if len(candles) < 3:
            return None
        best = None
        for i in range(1, len(candles) - 1):
            prev, nxt = candles[i - 1], candles[i + 1]
            if direction == "long" and prev.h < nxt.l:
                g = FVG("bullish", nxt.l, prev.h, i)
                if g.size > nxt.l * 0.0001: best = g
            elif direction == "short" and prev.l > nxt.h:
                g = FVG("bearish", prev.l, nxt.h, i)
                if g.size > prev.l * 0.0001: best = g
        return best

    def _find_order_block(self, candles, sweep_idx, bos_idx, direction):
        seg = candles[sweep_idx: bos_idx + 1]
        pick, idx = None, 0
        for i, c in enumerate(seg):
            if (c.is_bearish if direction == "long" else c.is_bullish) and c.body_size > 0:
                pick, idx = c, i
        if pick is None:
            return None
        return OrderBlock("bullish" if direction == "long" else "bearish", pick.body_high, pick.body_low, sweep_idx + idx)

    def _build_ote(self, direction, swing_high, swing_low, current_price):
        leg = swing_high - swing_low; tick = current_price * 0.0001
        if direction == "long":
            top = swing_high - self.OTE_RETRACE_LOW * leg; bot = swing_high - self.OTE_RETRACE_HIGH * leg; sl = swing_low - 2 * tick
        else:
            bot = swing_low + self.OTE_RETRACE_LOW * leg; top = swing_low + self.OTE_RETRACE_HIGH * leg; sl = swing_high + 2 * tick
        return OTEZone(direction, round(max(top, bot), 6), round(min(top, bot), 6), round(sl, 6), swing_high, swing_low)

    def _entry_triggered(self, price, ote, fvg, ob, direction):
        if ote.price_inside(price):
            return True
        guard = price >= ote.stop_loss if direction == "long" else price <= ote.stop_loss
        if fvg and fvg.price_inside(price) and guard:
            return True
        if ob and ob.price_inside(price) and guard:
            return True
        return False

    def _confidence(self, fvg, ob, disp, atr, sweep_candle, asia):
        fvg_s = 1.0 if fvg else 0.0; ob_s = 1.0 if ob else 0.0
        if disp and atr > 0:
            disp_s = min(1.0, (sum(c.body_size for c in disp) / len(disp)) / (atr * self.MIN_DISPLACEMENT_ATR_MULT))
        else:
            disp_s = 0.5
        if asia.width > 0:
            wick = max(sweep_candle.h - asia.high, asia.low - sweep_candle.l, 0)
            sweep_s = min(1.0, wick / (asia.width * 0.15))
        else:
            sweep_s = 0.5
        return round(self.W_FVG * fvg_s + self.W_OB * ob_s + self.W_ATR * disp_s + self.W_SWEEP * sweep_s, 4)

    @staticmethod
    def _atr(candles, period=14):
        if len(candles) < 2:
            return 0.0
        trs = [max(c.h - c.l, abs(c.h - candles[i - 1].c), abs(c.l - candles[i - 1].c)) for i, c in enumerate(candles) if i >= 1][:period]
        return sum(trs) / len(trs) if trs else 0.0
