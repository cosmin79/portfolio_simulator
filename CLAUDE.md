# Development Guide for Portfolio Simulator

## Project layout

| File | Purpose |
|---|---|
| `portfolio_sim.py` | Core engine: data fetching, simulation, metrics |
| `portfolio_app.py` | Streamlit GUI (calls `portfolio_sim`) |
| `run_portfolio.py` | CLI script (calls `portfolio_sim`) |
| `tests/test_unit.py` | Unit tests — fast, no network |
| `tests/test_integration.py` | Integration tests — require Yahoo Finance access |

---

## Running tests

### Without Gradle (recommended during development)

```bash
# Activate the virtual environment first
source .venv/bin/activate

# Unit tests only (fast, no network) — run these always
pytest tests/ -v -m "not integration"

# Integration tests (require internet, ~30 s)
pytest tests/ -v -m integration

# Full suite
pytest tests/ -v
```

### With Gradle (`brew install gradle` required)

```bash
gradle unitTest          # fast, no network
gradle integrationTest   # requires Yahoo Finance access
gradle test              # full suite
gradle run               # CLI simulation
gradle app               # Streamlit UI
```

---

## Test requirement — mandatory rule

**All tests must pass before merging or shipping any change.**

- Run `pytest tests/ -v -m "not integration"` at minimum before every commit.
- Run the full suite (including integration tests) before any release or PR.
- When adding new functionality:
  1. Write unit tests that exercise the new code paths.
  2. If the change touches data fetching or a new external integration, add an
     integration test that validates the expected format and plausibility of
     the returned data.
  3. Do not disable or delete existing tests to make new code pass. Fix the
     code instead.

---

## What the integration tests protect

The integration tests in `tests/test_integration.py` explicitly guard against:

1. **Yahoo Finance column-name changes** — any rename of `"Close"` in the
   raw yfinance response will cause `TestYFinanceRawFormat` to fail immediately.
2. **Multi-ticker MultiIndex format** — the test asserts that a multi-ticker
   download returns a MultiIndex DataFrame with `"Close"` at the top level.
3. **Risk-free rate unit error** — `^IRX` data is in percent; we divide by 100.
   A test asserts that all returned values are `< 1.0`.
4. **FX rate plausibility** — sanity-range checks on GBP/USD and EUR/USD.
5. **Full pipeline smoke test** — end-to-end run on H1 2020 SPY (contains the
   COVID crash) verifying no exceptions and plausible drawdown.

---

## Key design decisions to preserve

- `simulate_portfolio` returns `portfolio_values` in **USD** and `total_invested`
  in the **input currency**. Callers must convert `portfolio_values` to local
  currency *before* calling `returns_from_simulation` when FX is involved.
- `auto_adjust=True` is used for equity prices (dividends are reinvested).
  `auto_adjust=False` is used for `^IRX` and FX tickers (they have no dividends).
- DDCA reserve grows at the risk-free rate daily. The maximum deployment in any
  single month is 1.5× the normal per-ticker contribution.
- Weight normalisation (`w /= w.sum()`) in `simulate_portfolio` silently adjusts
  weights that don't sum to exactly 1.0. `validate_weights` in `run_portfolio.py`
  is the guard against user error; the engine itself is tolerant.
