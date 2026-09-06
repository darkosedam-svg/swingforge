# Contract tests

These tests hit a real venue's HTTP schema, but only against a **pre-recorded cassette** —
they never make a live call in ordinary `pytest` runs. With no cassette present under
`tests/contract/cassettes/`, both tests `pytest.skip` (see `conftest.py`); the fast unit
suite always stays green. `@pytest.mark.contract` marks both tests so they can also be
excluded explicitly (`pytest -m "not contract"`).

## Recording a cassette (manual, once)

Run with real credentials and `SWINGFORGE_RECORD=1`, which switches the `vcr` fixture from
`record_mode="none"` (replay-only) to `record_mode="once"` (record if the cassette file is
missing, otherwise replay it unchanged):

```bash
# Hyperliquid: public endpoints, no credentials needed.
SWINGFORGE_RECORD=1 uv run pytest -m contract tests/contract/test_hyperliquid_contract.py -v

# OANDA: needs a practice-account token. OANDA_ACCOUNT_ID isn't read by the test itself
# (history() doesn't need an account id), but export it too if you extend the test to also
# exercise instrument_from_account / load_financing.
OANDA_TOKEN=<your-practice-token> SWINGFORGE_RECORD=1 \
  uv run pytest -m contract tests/contract/test_oanda_contract.py -v
```

This writes `tests/contract/cassettes/hyperliquid_btc_4h_30d.yaml` and
`.../oanda_eur_usd_1d_10d.yaml`. Delete the file and re-run to re-record after a schema
change upstream.

TODO: the current Hyperliquid cassette's 30-day/4H window never approaches the venue's
5000-candle-per-call cap, so it cannot exercise the multi-page/resume paging logic in
`HyperliquidBars.history` (C1/C2) against the real API — that's covered by unit tests
against a fake `Info` only (see `test_hl_bars.py`). Recording a second cassette for a 1H
window long enough to span multiple pages (5000+ hours, i.e. roughly 7+ months) against
the live endpoint would additionally confirm the real venue's cap and pagination shape
still match what the fake asserts.

## No secrets in cassettes

`conftest.py`'s `vcr` fixture sets `filter_headers=["Authorization"]`, so the bearer token
sent with every OANDA request is stripped from the recorded interaction before it ever
touches disk. Hyperliquid's endpoints carry no credentials at all. Still, **inspect a newly
recorded cassette before committing it** — confirm no `Authorization` value, account id, or
other secret leaked into a URL, query string, or response body.

## Determinism

Both tests use a fixed historical UTC window (`2024-06-01` minus 30/10 days), not
`datetime.now()`, so a replayed cassette's request always matches what was recorded —
vcrpy's default matcher compares method/host/path/query, which for these endpoints
includes the millisecond start/end timestamps.
