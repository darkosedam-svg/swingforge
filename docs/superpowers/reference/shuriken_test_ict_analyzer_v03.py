"""Ports the April 2026 ict_analyzer unit checks, plus the direction regression."""
from shuriken_lab.ict_analyzer import AsiaRange, Candle, FVG, ICTAnalyzer, OrderBlock


def c(t, o, h, l, cc, v=100.0):
    return Candle(t=t, o=o, h=h, l=l, c=cc, v=v)


def test_asia_range_sweeps():
    ar = AsiaRange(high=105.0, low=100.0)
    assert ar.is_sweep_high(c(0, 102, 106, 101, 104)) and ar.is_sweep_low(c(0, 102, 103, 99, 101))
    assert not ar.is_sweep_high(c(0, 102, 104, 101, 103))


def test_fvg_and_order_block():
    az = ICTAnalyzer()
    f = az._find_fvg([c(0, 100, 101, 99, 100), c(1, 100, 105, 100, 104), c(2, 104, 108, 102, 107)], "long")
    assert f and f.direction == "bullish" and f.bottom == 101 and f.top == 102
    assert az._find_fvg([c(0, 100, 103, 99, 102), c(1, 102, 106, 101, 105), c(2, 105, 107, 102, 106)], "long") is None
    ob = az._find_order_block([c(0, 105, 106, 104, 104), c(1, 104, 105, 103, 103), c(2, 103, 108, 103, 107)], 0, 2, "long")
    assert ob and ob.direction == "bullish" and ob.high == 104 and ob.low == 103


def test_ote_levels():
    az = ICTAnalyzer()
    o = az._build_ote("long", 110.0, 100.0, 105.0)
    assert abs(o.entry_top - 103.82) < 0.01 and abs(o.entry_bottom - 102.10) < 0.01 and o.stop_loss < o.entry_bottom


def test_sweep_of_asia_low_is_a_long_setup():
    """Regression: the original _find_sweep returned 'short' on a low sweep, contradicting its own docstring,
    AsiaRange helpers and the smoke-test scenario, so _find_bos then searched the wrong way and returned None."""
    az = ICTAnalyzer(); ar = AsiaRange(high=105.0, low=100.0)
    cands = [c(i, 102, 103, 101, 102) for i in range(10)] + [c(10, 102, 103, 99.5, 101)]   # wick below 100, close inside
    idx, direction = az._find_sweep(cands, ar, 101.0)
    assert idx == 10 and direction == "long"
    cands2 = [c(i, 102, 103, 101, 102) for i in range(10)] + [c(10, 102, 105.5, 101, 104)]
    assert az._find_sweep(cands2, ar, 104.0)[1] == "short"


def test_pullback_candle_is_not_a_sweep():
    az = ICTAnalyzer(); ar = AsiaRange(high=105.0, low=100.0)
    cands = [c(i, 102, 103, 101, 102) for i in range(10)] + [c(10, 108, 108.5, 103, 104)]   # opened above range, pulled back in
    assert az._find_sweep(cands, ar, 104.0) == (None, None)


def test_full_pipeline_fires_on_sweep_low_then_bos():
    def t_ms(h, m=0): return int((h * 60 + m) * 60 * 1000)
    c1h = [Candle(t=t_ms(h), o=69200 + h * 20, h=69600 - h * 5, l=69100 + h * 10, c=69400 + h * 15, v=100) for h in range(8)]
    c1h += [Candle(t=t_ms(8), o=69300, h=69350, l=69040, c=69150, v=200), Candle(t=t_ms(9), o=69150, h=69820, l=69140, c=69780, v=400),
            Candle(t=t_ms(10), o=69780, h=69900, l=69750, c=69850, v=250), Candle(t=t_ms(11), o=69850, h=69920, l=69790, c=69880, v=180)]
    c5 = [Candle(t=t_ms(0) + i * 300_000, o=69100 + (i % 10) * 30, h=69140 + (i % 10) * 30, l=69070 + (i % 10) * 30, c=69110 + (i % 10) * 30, v=50) for i in range(96)]
    c5 += [Candle(t=t_ms(8, 0), o=69200, h=69250, l=69040, c=69150, v=200),
           Candle(t=t_ms(8, 5), o=69150, h=69350, l=69140, c=69320, v=280),
           Candle(t=t_ms(8, 10), o=69320, h=69700, l=69310, c=69660, v=450),
           Candle(t=t_ms(8, 15), o=69660, h=69820, l=69640, c=69800, v=350),
           Candle(t=t_ms(8, 20), o=69800, h=69850, l=69780, c=69830, v=200),
           Candle(t=t_ms(8, 25), o=69830, h=69840, l=69720, c=69740, v=150)]
    s = ICTAnalyzer().analyse("BTC", c1h, c5, "london")
    assert s is not None and s.direction == "long" and s.entry_triggered and s.confidence > 0.45
