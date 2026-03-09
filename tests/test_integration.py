"""
Integration tests for portfolio_sim.py — require live Yahoo Finance access.

These tests detect:
  1. yfinance API / column-name format changes
  2. Structural changes in returned DataFrames / Series
  3. Plausibility regressions in fetched data (e.g. prices always > 0)

Run with:
    pytest tests/test_integration.py -v -m integration

Skip during offline / CI runs:
    pytest -m "not integration"
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest
import yfinance as yf

from portfolio_sim import (
    fetch_prices,
    fetch_risk_free_rate,
    fetch_fx_rate,
    simulate_portfolio,
    returns_from_simulation,
    calculate_metrics,
)

pytestmark = pytest.mark.integration

# Use a short, well-established historical window for speed and reproducibility
TEST_START_YEAR = 2020
TEST_END_YEAR = 2020
TEST_END_MONTH = 6   # Jan–Jun 2020: ~125 trading days, contains COVID crash + recovery


# ---------------------------------------------------------------------------
# fetch_prices — format guard
# ---------------------------------------------------------------------------

class TestFetchPricesFormat:

    def test_returns_dataframe(self):
        prices = fetch_prices(["SPY"], TEST_START_YEAR, end_year=TEST_END_YEAR, end_month=TEST_END_MONTH)
        assert isinstance(prices, pd.DataFrame)

    def test_has_datetime_index(self):
        prices = fetch_prices(["SPY"], TEST_START_YEAR, end_year=TEST_END_YEAR, end_month=TEST_END_MONTH)
        assert isinstance(prices.index, pd.DatetimeIndex)

    def test_index_sorted_ascending(self):
        prices = fetch_prices(["SPY"], TEST_START_YEAR, end_year=TEST_END_YEAR, end_month=TEST_END_MONTH)
        assert prices.index.is_monotonic_increasing

    def test_single_ticker_column_name(self):
        """Single ticker: column must be the ticker symbol, not 'Close' or 'Price'."""
        prices = fetch_prices(["SPY"], TEST_START_YEAR, end_year=TEST_END_YEAR, end_month=TEST_END_MONTH)
        assert "SPY" in prices.columns, (
            f"Expected column 'SPY', got {list(prices.columns)}. "
            "yfinance may have changed its single-ticker return format."
        )

    def test_multi_ticker_all_columns_present(self):
        tickers = ["SPY", "AGG"]
        prices = fetch_prices(tickers, TEST_START_YEAR, end_year=TEST_END_YEAR, end_month=TEST_END_MONTH)
        for t in tickers:
            assert t in prices.columns, (
                f"Expected column '{t}', got {list(prices.columns)}. "
                "yfinance multi-ticker column structure may have changed."
            )

    def test_no_nan_after_fetch(self):
        """fetch_prices drops rows with any NaN — result must be complete."""
        prices = fetch_prices(["SPY"], TEST_START_YEAR, end_year=TEST_END_YEAR, end_month=TEST_END_MONTH)
        assert not prices.isnull().any().any(), "fetch_prices returned NaN values"

    def test_all_prices_positive(self):
        prices = fetch_prices(["SPY"], TEST_START_YEAR, end_year=TEST_END_YEAR, end_month=TEST_END_MONTH)
        assert (prices > 0).all().all(), "fetch_prices returned non-positive prices"

    def test_row_count_plausible(self):
        """Jan–Jun 2020 should have roughly 120–130 trading days."""
        prices = fetch_prices(["SPY"], TEST_START_YEAR, end_year=TEST_END_YEAR, end_month=TEST_END_MONTH)
        assert 115 <= len(prices) <= 135, (
            f"Expected ~125 trading days for H1 2020, got {len(prices)}. "
            "Data availability may have changed."
        )

    def test_invalid_ticker_raises_value_error(self):
        # yfinance may return empty data (→ "No overlapping trading days") or a
        # columns mismatch (→ "Could not retrieve data") depending on the ticker.
        with pytest.raises(ValueError):
            fetch_prices(["INVALID_TICKER_XYZ_404"], TEST_START_YEAR, end_year=TEST_END_YEAR, end_month=TEST_END_MONTH)

    def test_prices_within_plausible_spy_range(self):
        """SPY was roughly 225–340 during H1 2020."""
        prices = fetch_prices(["SPY"], TEST_START_YEAR, end_year=TEST_END_YEAR, end_month=TEST_END_MONTH)
        spy = prices["SPY"]
        assert spy.min() > 200, f"SPY min {spy.min():.2f} unexpectedly low"
        assert spy.max() < 400, f"SPY max {spy.max():.2f} unexpectedly high"


# ---------------------------------------------------------------------------
# fetch_prices — yfinance raw format guard (internal)
# ---------------------------------------------------------------------------

class TestYFinanceRawFormat:
    """
    Directly call yfinance to assert the raw DataFrame structure our code depends on.
    These tests will fail immediately if Yahoo Finance changes its API response format,
    alerting us before any production run.
    """

    def test_single_ticker_raw_has_close_column(self):
        """
        portfolio_sim.fetch_prices relies on raw["Close"] for single-ticker responses.
        """
        raw = yf.download("SPY", start="2020-01-02", end="2020-01-31", auto_adjust=True, progress=False)
        assert not raw.empty, "yfinance returned empty DataFrame for SPY"
        assert "Close" in raw.columns or (
            isinstance(raw.columns, pd.MultiIndex) and "Close" in raw.columns.get_level_values(0)
        ), (
            f"Expected 'Close' in raw yfinance columns, got {list(raw.columns)}. "
            "yfinance has changed its column names — update fetch_prices accordingly."
        )

    def test_multi_ticker_raw_has_multiindex_with_close(self):
        """
        For multiple tickers, yfinance should return a MultiIndex DataFrame
        with 'Close' as the top-level field.
        """
        raw = yf.download(["SPY", "AGG"], start="2020-01-02", end="2020-01-31", auto_adjust=True, progress=False)
        assert not raw.empty
        assert isinstance(raw.columns, pd.MultiIndex), (
            "Expected MultiIndex columns for multi-ticker download. "
            f"Got flat columns: {list(raw.columns)}"
        )
        top_level = raw.columns.get_level_values(0).unique().tolist()
        assert "Close" in top_level, (
            f"'Close' not in top-level MultiIndex fields: {top_level}. "
            "yfinance column structure has changed."
        )

    def test_irx_raw_has_close_column(self):
        """
        fetch_risk_free_rate relies on raw["Close"] for ^IRX.
        ^IRX uses auto_adjust=False.
        """
        raw = yf.download("^IRX", start="2020-01-02", end="2020-06-30", auto_adjust=False, progress=False)
        assert not raw.empty, "^IRX returned empty — risk-free rate fetch will fail"
        assert "Close" in raw.columns or (
            isinstance(raw.columns, pd.MultiIndex) and "Close" in raw.columns.get_level_values(0)
        ), (
            f"Expected 'Close' for ^IRX, got {list(raw.columns)}. "
            "yfinance may have changed its format for indices."
        )

    def test_fx_raw_has_close_column(self):
        """
        fetch_fx_rate relies on raw["Close"] for GBPUSD=X.
        """
        raw = yf.download("GBPUSD=X", start="2020-01-02", end="2020-06-30", auto_adjust=False, progress=False)
        assert not raw.empty, "GBPUSD=X returned empty"
        assert "Close" in raw.columns or (
            isinstance(raw.columns, pd.MultiIndex) and "Close" in raw.columns.get_level_values(0)
        ), (
            f"Expected 'Close' for GBPUSD=X, got {list(raw.columns)}. "
            "FX rate fetch will fail."
        )


# ---------------------------------------------------------------------------
# fetch_risk_free_rate — format and plausibility
# ---------------------------------------------------------------------------

class TestFetchRiskFreeRate:

    def test_returns_series(self):
        rf = fetch_risk_free_rate(TEST_START_YEAR, end_year=TEST_END_YEAR, end_month=TEST_END_MONTH)
        assert isinstance(rf, pd.Series)

    def test_has_datetime_index(self):
        rf = fetch_risk_free_rate(TEST_START_YEAR, end_year=TEST_END_YEAR, end_month=TEST_END_MONTH)
        assert isinstance(rf.index, pd.DatetimeIndex)

    def test_values_are_decimal_not_percent(self):
        """
        ^IRX raw data is in percent (e.g. 1.5 for 1.5%).
        fetch_risk_free_rate must divide by 100 → values < 1.0.
        If this fails, the risk-free rate is being used as 100× too large.
        """
        rf = fetch_risk_free_rate(TEST_START_YEAR, end_year=TEST_END_YEAR, end_month=TEST_END_MONTH)
        assert rf.max() < 1.0, (
            f"Risk-free rate max = {rf.max():.4f}. Expected < 1.0 (decimal form). "
            "Possible regression: /100 division may have been removed."
        )

    def test_values_non_negative(self):
        """
        T-bill rates are effectively non-negative, but ^IRX can quote fractionally
        negative values (e.g. -0.001) in near-zero-rate environments.
        Allow a small tolerance (-0.005 = -0.5 bps annualised).
        """
        rf = fetch_risk_free_rate(TEST_START_YEAR, end_year=TEST_END_YEAR, end_month=TEST_END_MONTH)
        assert rf.min() >= -0.005, f"Risk-free rate implausibly negative: {rf.min():.4f}"

    def test_values_plausible_range(self):
        """T-bill rates should be between 0% and 20% annually."""
        rf = fetch_risk_free_rate(TEST_START_YEAR, end_year=TEST_END_YEAR, end_month=TEST_END_MONTH)
        assert rf.max() <= 0.20, f"Risk-free rate unexpectedly high: {rf.max():.4f}"

    def test_non_empty(self):
        rf = fetch_risk_free_rate(TEST_START_YEAR, end_year=TEST_END_YEAR, end_month=TEST_END_MONTH)
        assert len(rf) > 0

    def test_series_name(self):
        rf = fetch_risk_free_rate(TEST_START_YEAR, end_year=TEST_END_YEAR, end_month=TEST_END_MONTH)
        assert rf.name == "risk_free_rate"

    def test_h1_2020_rates_were_low(self):
        """
        In H1 2020 the Fed cut to near-zero in March.
        The mean rate should be between 0% and 2%.
        """
        rf = fetch_risk_free_rate(TEST_START_YEAR, end_year=TEST_END_YEAR, end_month=TEST_END_MONTH)
        assert 0.0 <= rf.mean() <= 0.02, (
            f"Average H1-2020 T-bill rate = {rf.mean():.4f}, expected 0–2%. "
            "This may indicate a unit error in the conversion."
        )


# ---------------------------------------------------------------------------
# fetch_fx_rate — format and plausibility
# ---------------------------------------------------------------------------

class TestFetchFXRate:

    def test_usd_returns_none(self):
        result = fetch_fx_rate("USD", TEST_START_YEAR)
        assert result is None

    def test_gbp_returns_series(self):
        result = fetch_fx_rate("GBP", TEST_START_YEAR, end_year=TEST_END_YEAR, end_month=TEST_END_MONTH)
        assert isinstance(result, pd.Series)

    def test_gbp_has_datetime_index(self):
        result = fetch_fx_rate("GBP", TEST_START_YEAR, end_year=TEST_END_YEAR, end_month=TEST_END_MONTH)
        assert isinstance(result.index, pd.DatetimeIndex)

    def test_gbp_series_name(self):
        result = fetch_fx_rate("GBP", TEST_START_YEAR, end_year=TEST_END_YEAR, end_month=TEST_END_MONTH)
        assert result.name == "GBPUSD"

    def test_gbp_plausible_range(self):
        """GBP/USD is historically between 1.0 and 2.5."""
        result = fetch_fx_rate("GBP", TEST_START_YEAR, end_year=TEST_END_YEAR, end_month=TEST_END_MONTH)
        assert result.min() > 0.8, f"GBPUSD too low: {result.min():.4f}"
        assert result.max() < 2.5, f"GBPUSD too high: {result.max():.4f}"

    def test_eur_plausible_range(self):
        """EUR/USD is historically between 0.9 and 1.6."""
        result = fetch_fx_rate("EUR", TEST_START_YEAR, end_year=TEST_END_YEAR, end_month=TEST_END_MONTH)
        assert result.min() > 0.8, f"EURUSD too low: {result.min():.4f}"
        assert result.max() < 1.6, f"EURUSD too high: {result.max():.4f}"

    def test_lowercase_currency_accepted(self):
        """Currency code should be case-insensitive."""
        result_upper = fetch_fx_rate("GBP", TEST_START_YEAR, end_year=TEST_END_YEAR, end_month=TEST_END_MONTH)
        result_lower = fetch_fx_rate("gbp", TEST_START_YEAR, end_year=TEST_END_YEAR, end_month=TEST_END_MONTH)
        pd.testing.assert_series_equal(result_upper, result_lower, check_names=False)


# ---------------------------------------------------------------------------
# Full end-to-end pipeline
# ---------------------------------------------------------------------------

class TestFullPipeline:

    @pytest.fixture(scope="class")
    def spy_prices(self):
        return fetch_prices(["SPY"], TEST_START_YEAR, end_year=TEST_END_YEAR, end_month=TEST_END_MONTH)

    @pytest.fixture(scope="class")
    def rf(self):
        return fetch_risk_free_rate(TEST_START_YEAR, end_year=TEST_END_YEAR, end_month=TEST_END_MONTH)

    def test_simulation_runs_without_error(self, spy_prices, rf):
        vals, invested, reserve = simulate_portfolio(
            spy_prices, {"SPY": 1.0},
            initial_investment=10_000,
            monthly_contribution=500,
            risk_free_rate=rf,
        )
        assert len(vals) == len(spy_prices)
        assert len(invested) == len(spy_prices)
        assert len(reserve) == len(spy_prices)

    def test_simulation_values_positive(self, spy_prices, rf):
        vals, _, _ = simulate_portfolio(
            spy_prices, {"SPY": 1.0},
            initial_investment=10_000,
            monthly_contribution=0,
            risk_free_rate=rf,
        )
        assert (vals > 0).all()

    def test_returns_no_inf(self, spy_prices, rf):
        vals, invested, _ = simulate_portfolio(
            spy_prices, {"SPY": 1.0},
            initial_investment=10_000,
            monthly_contribution=500,
            risk_free_rate=rf,
        )
        rets = returns_from_simulation(vals, invested)
        assert not np.isinf(rets.dropna()).any()

    def test_metrics_all_keys_present(self, spy_prices, rf):
        vals, invested, _ = simulate_portfolio(
            spy_prices, {"SPY": 1.0},
            initial_investment=10_000,
            monthly_contribution=500,
            risk_free_rate=rf,
        )
        rets = returns_from_simulation(vals, invested)
        m = calculate_metrics(vals, invested, rets, risk_free_rate=rf)
        required_keys = {
            "CAGR", "Sharpe Ratio", "Max Drawdown", "Win Rate (daily)",
            "Sortino Ratio", "Calmar Ratio", "Final Value ($)", "Total Invested ($)",
        }
        assert required_keys.issubset(set(m.keys()))

    def test_h1_2020_spy_had_drawdown(self, spy_prices, rf):
        """
        COVID crash hit SPY ~34% in Feb-Mar 2020.
        Max drawdown over H1 2020 must be materially negative.
        """
        vals, invested, _ = simulate_portfolio(
            spy_prices, {"SPY": 1.0},
            initial_investment=10_000,
            monthly_contribution=0,
            risk_free_rate=rf,
        )
        rets = returns_from_simulation(vals, invested)
        m = calculate_metrics(vals, invested, rets, risk_free_rate=rf)
        assert m["Max Drawdown"] < -0.10, (
            f"Max drawdown = {m['Max Drawdown']:.2%}. "
            "Expected worse than -10% given the COVID crash in H1 2020."
        )

    def test_ddca_simulation_runs_with_real_data(self, spy_prices, rf):
        """DDCA path should not raise with real SPY data."""
        vals, invested, reserve = simulate_portfolio(
            spy_prices, {"SPY": 1.0},
            initial_investment=10_000,
            monthly_contribution=500,
            ddca_thresholds={"SPY": 0.10},
            risk_free_rate=rf,
        )
        assert len(vals) == len(spy_prices)
        # Reserve should have been built up and potentially triggered (COVID crash)
        assert reserve.max() >= 0

    def test_multi_portfolio_pipeline(self, spy_prices, rf):
        """Two-portfolio pipeline (60/40) should run and produce distinct results."""
        tickers = ["SPY", "AGG"]
        prices = fetch_prices(tickers, TEST_START_YEAR, end_year=TEST_END_YEAR, end_month=TEST_END_MONTH)

        results = []
        for weights in [{"SPY": 1.0}, {"SPY": 0.6, "AGG": 0.4}]:
            v, i, _ = simulate_portfolio(prices, weights, 10_000, 500, risk_free_rate=rf)
            r = returns_from_simulation(v, i)
            m = calculate_metrics(v, i, r, risk_free_rate=rf)
            results.append(m)

        # 100% SPY and 60/40 should produce different metrics
        assert results[0]["Max Drawdown"] != pytest.approx(results[1]["Max Drawdown"], abs=0.01)
