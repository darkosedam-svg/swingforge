"""OKX USDT-margined perpetual swaps: a research-only venue (deep history, no paper trading)."""

from swingforge.adapters.okx.bars import DEFAULT_SWAPS, OkxBars, inst_id, okx_instrument, okx_instruments
from swingforge.adapters.okx.client import OkxClient, OkxError, OkxLike
from swingforge.adapters.okx.costs import DEFAULT_FUNDING_RATE, FUNDING_INTERVAL, OkxCosts, load_funding

__all__ = [
    "DEFAULT_FUNDING_RATE",
    "DEFAULT_SWAPS",
    "FUNDING_INTERVAL",
    "OkxBars",
    "OkxClient",
    "OkxCosts",
    "OkxError",
    "OkxLike",
    "inst_id",
    "load_funding",
    "okx_instrument",
    "okx_instruments",
]
