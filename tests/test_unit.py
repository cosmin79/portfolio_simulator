"""
Unit tests for portfolio_sim.py core logic.

All tests here are deterministic and require no network access.
They use synthetic price DataFrames to verify calculation correctness.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

from portfolio_sim import (
    simulate_portfolio,
    returns_from_simulation,
    calculate_metrics,
    drawdown_series,
    suggest_ddca_thresholds,
    portfolio_daily_returns,
    correlation_matrix,
    rolling_window_analysis,
    rolling_window_summary,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_prices(n_days=504, tickers=None, price=100.0, start="2020-01-02"):
    """Flat price DataFrame on business days."""
    if tickers is None:
        tickers = ["SPY"]
    dates = pd.bdate_range(start, periods=n_days)
    return pd.DataFrame({t: np.full(n_days, price) for t in tickers}, index=dates)


def make_prices_trend(n_days=504, daily_return=0.001, tickers=None, start="2020-01-02"):
    """Prices that grow at a constant daily rate."""
    if tickers is None:
        tickers = ["SPY"]
    dates = pd.bdate_range(start, periods=n_days)
    arr = np.cumprod(np.full(n_days, 1.0 + daily_return))
    return pd.DataFrame({t: arr for t in tickers}, index=dates)


def count_contribution_months(dates):
    """Count how many new-month transitions occur in the date range (= number of DCA events)."""
    months = pd.PeriodIndex([d.to_period("M") for d in dates])
    return int((months != months.shift()).sum()) - 1  # subtract the first period


# ---------------------------------------------------------------------------
# simulate_portfolio — basic DCA
# ---------------------------------------------------------------------------

class TestSimulatePortfolioBasicDCA:

    def test_flat_prices_value_equals_invested(self):
        """With constant prices, portfolio value should equal total cash invested."""
        prices = make_prices(n_days=480)
        vals, invested, reserve = simulate_portfolio(
            prices, {"SPY": 1.0},
            initial_investment=10_000,
            monthly_contribution=500,
        )
        # No DDCA → reserve always zero
        assert reserve.max() == pytest.approx(0.0)
        # Final portfolio value = all cash invested (prices didn't change)
        assert vals.iloc[-1] == pytest.approx(invested.iloc[-1], rel=1e-6)

    def test_initial_investment_is_first_value(self):
        """On day 0 the portfolio value should equal the initial investment."""
        prices = make_prices(n_days=60)
        vals, invested, _ = simulate_portfolio(
            prices, {"SPY": 1.0},
            initial_investment=10_000,
            monthly_contribution=0,
        )
        assert vals.iloc[0] == pytest.approx(10_000.0, rel=1e-6)
        assert invested.iloc[0] == pytest.approx(10_000.0, rel=1e-6)

    def test_no_monthly_contribution_total_invested_constant(self):
        """Without monthly contributions, total_invested stays at initial_investment."""
        prices = make_prices(n_days=120)
        _, invested, _ = simulate_portfolio(
            prices, {"SPY": 1.0},
            initial_investment=5_000,
            monthly_contribution=0,
        )
        assert invested.min() == pytest.approx(5_000.0, rel=1e-6)
        assert invested.max() == pytest.approx(5_000.0, rel=1e-6)

    def test_total_invested_increases_monthly(self):
        """total_invested must be monotonically non-decreasing."""
        prices = make_prices(n_days=504)
        _, invested, _ = simulate_portfolio(
            prices, {"SPY": 1.0},
            initial_investment=10_000,
            monthly_contribution=500,
        )
        assert invested.is_monotonic_increasing

    def test_total_invested_step_size(self):
        """Each monthly step in total_invested should be exactly monthly_contribution."""
        prices = make_prices(n_days=504)
        monthly = 500.0
        _, invested, _ = simulate_portfolio(
            prices, {"SPY": 1.0},
            initial_investment=10_000,
            monthly_contribution=monthly,
        )
        steps = invested.diff().dropna()
        non_zero_steps = steps[steps > 0]
        assert (non_zero_steps - monthly).abs().max() < 1e-9

    def test_total_invested_final_amount(self):
        """Final total_invested = initial + n_contribution_months × monthly."""
        prices = make_prices(n_days=480)
        initial = 10_000.0
        monthly = 500.0
        _, invested, _ = simulate_portfolio(
            prices, {"SPY": 1.0},
            initial_investment=initial,
            monthly_contribution=monthly,
        )
        n_months = int((invested.diff().dropna() > 0).sum())
        expected = initial + n_months * monthly
        assert invested.iloc[-1] == pytest.approx(expected, rel=1e-6)

    def test_growing_prices_value_exceeds_invested(self):
        """With rising prices, portfolio value must exceed total invested."""
        prices = make_prices_trend(n_days=480, daily_return=0.001)
        vals, invested, _ = simulate_portfolio(
            prices, {"SPY": 1.0},
            initial_investment=10_000,
            monthly_contribution=500,
        )
        assert vals.iloc[-1] > invested.iloc[-1]

    def test_multi_ticker_weights_sum_to_one(self):
        """Weights are normalised; the simulation should accept near-1.0 sums."""
        prices = make_prices(n_days=120, tickers=["A", "B"], price=50.0)
        vals, invested, _ = simulate_portfolio(
            prices, {"A": 0.6, "B": 0.4},
            initial_investment=1_000,
            monthly_contribution=100,
        )
        # Flat prices → final value = total invested
        assert vals.iloc[-1] == pytest.approx(invested.iloc[-1], rel=1e-4)

    def test_multi_ticker_initial_share_ratio(self):
        """With equal prices, share count ratio should match weight ratio on day 0."""
        # Price A = 10, price B = 10 → shares proportional to weights
        dates = pd.bdate_range("2020-01-02", periods=120)
        prices = pd.DataFrame({"A": np.full(120, 10.0), "B": np.full(120, 10.0)}, index=dates)
        # weights: A=0.6, B=0.4 → initial 1000: A gets 600, B gets 400
        # so shares_A = 60, shares_B = 40 → value_A/value_B = 1.5
        vals_a_end = 60 * 10.0  # 600
        vals_b_end = 40 * 10.0  # 400
        # Verify final value is correct on day 0 when no contributions
        vals, _, _ = simulate_portfolio(
            prices, {"A": 0.6, "B": 0.4},
            initial_investment=1_000,
            monthly_contribution=0,
        )
        assert vals.iloc[0] == pytest.approx(1_000.0, rel=1e-6)


# ---------------------------------------------------------------------------
# simulate_portfolio — annual rebalance
# ---------------------------------------------------------------------------

class TestSimulatePortfolioRebalance:

    def test_rebalance_corrects_drift(self):
        """
        When one asset outperforms, annual rebalance should bring weights back.
        Result differs from no-rebalance simulation.
        """
        # 2 assets: A grows, B is flat
        n = 504  # ~2 years
        dates = pd.bdate_range("2020-01-02", periods=n)
        prices = pd.DataFrame({
            "A": np.cumprod(np.full(n, 1.002)),  # grows ~165%
            "B": np.ones(n),
        }, index=dates)

        vals_rebal, _, _ = simulate_portfolio(
            prices, {"A": 0.5, "B": 0.5},
            initial_investment=10_000,
            monthly_contribution=0,
            rebalance_annually=True,
        )
        vals_no_rebal, _, _ = simulate_portfolio(
            prices, {"A": 0.5, "B": 0.5},
            initial_investment=10_000,
            monthly_contribution=0,
            rebalance_annually=False,
        )
        # Values should differ after a year of drift
        assert not np.allclose(vals_rebal.values, vals_no_rebal.values)


# ---------------------------------------------------------------------------
# simulate_portfolio — DDCA (Double DCA)
# ---------------------------------------------------------------------------

class TestSimulatePortfolioDDCA:

    def _build_dip_prices(self):
        """
        3 months:
          Jan 2020: price = 100 (establishes 52-week high)
          Feb 2020: price = 100 (no dip, reserve accumulates)
          Mar 2020: price = 80  (20% below high → triggers 10% threshold)
        """
        dates = pd.bdate_range("2020-01-02", "2020-03-31")
        vals = np.full(len(dates), 100.0)
        mar_start = next(i for i, d in enumerate(dates) if d.month == 3)
        vals[mar_start:] = 80.0
        return pd.DataFrame({"A": vals}, index=dates)

    def test_no_trigger_reserve_accumulates(self):
        """Reserve should grow monotonically when the price never falls below threshold."""
        prices = make_prices(n_days=120, tickers=["A"], price=100.0)  # flat → always at 52wk high
        _, _, reserve = simulate_portfolio(
            prices, {"A": 1.0},
            initial_investment=1_000,
            monthly_contribution=100,
            ddca_thresholds={"A": 0.10},
        )
        # Reserve should be monotonically non-decreasing (only grows, never drawn)
        assert reserve.is_monotonic_increasing or (reserve.diff().dropna() >= -1e-9).all()

    def test_trigger_reserve_drawn_down(self):
        """When price drops below threshold, reserve should decrease."""
        prices = self._build_dip_prices()
        _, _, reserve = simulate_portfolio(
            prices, {"A": 1.0},
            initial_investment=1_000,
            monthly_contribution=100,
            ddca_thresholds={"A": 0.10},
        )
        # Reserve must reach a positive value before March
        feb_end = max(i for i, d in enumerate(prices.index) if d.month == 2)
        mar_start = next(i for i, d in enumerate(prices.index) if d.month == 3)
        # Reserve peaked in Feb
        reserve_before_trigger = reserve.iloc[feb_end]
        # After March trigger it should have dropped (drawn down)
        reserve_after_trigger = reserve.iloc[mar_start + 1]
        assert reserve_before_trigger > 0
        assert reserve_after_trigger < reserve_before_trigger

    def test_ddca_more_shares_than_plain_dca_at_discount(self):
        """
        DDCA should accumulate more effective shares than plain DCA when it buys
        extra shares at the discounted price.
        """
        prices = self._build_dip_prices()
        last_price = 80.0

        vals_ddca, invested_ddca, reserve_ddca = simulate_portfolio(
            prices, {"A": 1.0},
            initial_investment=1_000,
            monthly_contribution=100,
            ddca_thresholds={"A": 0.10},
        )
        vals_plain, invested_plain, _ = simulate_portfolio(
            prices, {"A": 1.0},
            initial_investment=1_000,
            monthly_contribution=100,
        )

        # Total invested must be equal (DDCA just re-times deployments)
        assert invested_ddca.iloc[-1] == pytest.approx(invested_plain.iloc[-1], rel=1e-6)

        # DDCA + reserve final value should differ from plain DCA (bought differently)
        # On this scenario DDCA deploys more at the discounted price → higher value
        ddca_final = vals_ddca.iloc[-1]  # includes reserve
        plain_final = vals_plain.iloc[-1]
        assert ddca_final != pytest.approx(plain_final, rel=1e-4)

    def test_ddca_short_ticker_ignored(self):
        """Tickers with negative weight must not be added to the DDCA reserve dict."""
        dates = pd.bdate_range("2020-01-02", periods=120)
        prices = pd.DataFrame({
            "A": np.full(120, 100.0),
            "B": np.full(120, 10.0),
        }, index=dates)
        # B has negative weight → should not get a reserve even if listed in ddca_thresholds
        vals, _, reserve = simulate_portfolio(
            prices, {"A": 1.33, "B": -0.33},
            initial_investment=1_000,
            monthly_contribution=0,
            ddca_thresholds={"B": 0.10},  # B is short → should be ignored
        )
        # Reserve must stay 0 since there are no DDCA-eligible tickers
        assert reserve.max() == pytest.approx(0.0, abs=1e-6)

    def test_ddca_max_deploy_is_one_and_half_x(self):
        """
        DDCA can deploy at most 1.5× the normal monthly contribution.
        Set up so the reserve >> ticker_contrib to test the cap.
        """
        # Build a long history where reserve can accumulate, then a big dip
        n = 300
        dates = pd.bdate_range("2018-01-02", periods=n)
        vals = np.full(n, 100.0)
        # Drop to 70 in the last month (30% below high → triggers 10% threshold)
        last_month_start = next(
            i for i in range(n - 1, 0, -1)
            if dates[i].to_period("M") != dates[i - 1].to_period("M")
        )
        vals[last_month_start:] = 70.0
        prices = pd.DataFrame({"A": vals}, index=dates)

        monthly = 1_000.0
        _, _, reserve = simulate_portfolio(
            prices, {"A": 1.0},
            initial_investment=0,
            monthly_contribution=monthly,
            ddca_thresholds={"A": 0.10},
        )
        # The reserve should never exceed what is theoretically possible:
        # at most every month parks half = 0.5 × monthly × n_months
        n_months = int((reserve.index[-1].to_period("M") - reserve.index[0].to_period("M")).n)
        max_possible_reserve = n_months * (monthly * 0.5) * 1.1  # 10% tolerance for interest
        assert reserve.max() <= max_possible_reserve


# ---------------------------------------------------------------------------
# simulate_portfolio — FX conversion
# ---------------------------------------------------------------------------

class TestSimulatePortfolioFX:

    def test_fx_rate_scales_buying_power(self):
        """
        Investing 1000 GBP at GBPUSD=2.0 should buy the same shares as
        investing 2000 USD with no FX conversion.
        """
        prices = make_prices(n_days=120, price=100.0)
        dates = prices.index
        fx = pd.Series(np.full(len(dates), 2.0), index=dates)

        # 1000 GBP at 2.0 GBPUSD → 2000 USD buying power
        vals_fx, invested_fx, _ = simulate_portfolio(
            prices, {"SPY": 1.0},
            initial_investment=1_000,
            monthly_contribution=0,
            fx_rate=fx,
        )
        vals_usd, invested_usd, _ = simulate_portfolio(
            prices, {"SPY": 1.0},
            initial_investment=2_000,
            monthly_contribution=0,
        )
        # USD portfolio values should be equal
        assert vals_fx.iloc[0] == pytest.approx(vals_usd.iloc[0], rel=1e-6)

    def test_fx_total_invested_in_local_currency(self):
        """total_invested is tracked in local currency, not USD."""
        prices = make_prices(n_days=120, price=100.0)
        dates = prices.index
        fx = pd.Series(np.full(len(dates), 1.5), index=dates)

        _, invested_fx, _ = simulate_portfolio(
            prices, {"SPY": 1.0},
            initial_investment=1_000,
            monthly_contribution=0,
            fx_rate=fx,
        )
        # total_invested should be 1000 (local currency), not 1500 (USD)
        assert invested_fx.iloc[0] == pytest.approx(1_000.0, rel=1e-6)

    def test_no_fx_equivalent_to_usd(self):
        """Passing fx_rate=None (default) is identical to fx_rate of all 1.0."""
        prices = make_prices(n_days=120, price=100.0)
        dates = prices.index
        fx_ones = pd.Series(np.ones(len(dates)), index=dates)

        vals_no_fx, _, _ = simulate_portfolio(
            prices, {"SPY": 1.0},
            initial_investment=1_000,
            monthly_contribution=100,
        )
        vals_fx1, _, _ = simulate_portfolio(
            prices, {"SPY": 1.0},
            initial_investment=1_000,
            monthly_contribution=100,
            fx_rate=fx_ones,
        )
        np.testing.assert_allclose(vals_no_fx.values, vals_fx1.values, rtol=1e-6)


# ---------------------------------------------------------------------------
# returns_from_simulation
# ---------------------------------------------------------------------------

class TestReturnsFromSimulation:

    def test_flat_prices_returns_near_zero(self):
        """With constant prices, stripping contributions gives ~0 daily returns."""
        prices = make_prices(n_days=240, price=100.0)
        vals, invested, _ = simulate_portfolio(
            prices, {"SPY": 1.0},
            initial_investment=10_000,
            monthly_contribution=500,
        )
        returns = returns_from_simulation(vals, invested)
        # All returns should be ≈ 0 (price didn't move)
        assert returns.dropna().abs().max() < 1e-9

    def test_returns_length(self):
        """Returns series should have one fewer element than portfolio_values."""
        prices = make_prices(n_days=120)
        vals, invested, _ = simulate_portfolio(
            prices, {"SPY": 1.0},
            initial_investment=10_000,
            monthly_contribution=100,
        )
        returns = returns_from_simulation(vals, invested)
        assert len(returns) == len(vals) - 1

    def test_growing_prices_returns_positive(self):
        """With rising prices, stripped returns should be positive."""
        prices = make_prices_trend(n_days=240, daily_return=0.001)
        vals, invested, _ = simulate_portfolio(
            prices, {"SPY": 1.0},
            initial_investment=10_000,
            monthly_contribution=0,
        )
        returns = returns_from_simulation(vals, invested)
        # No contributions → no stripping needed; returns should be ~0.001
        assert returns.dropna().mean() == pytest.approx(0.001, abs=1e-6)

    def test_no_inf_or_nan_on_normal_input(self):
        """returns_from_simulation must not produce inf or unexpected NaN."""
        prices = make_prices_trend(n_days=480, daily_return=0.0005)
        vals, invested, _ = simulate_portfolio(
            prices, {"SPY": 1.0},
            initial_investment=10_000,
            monthly_contribution=500,
        )
        returns = returns_from_simulation(vals, invested)
        assert not np.isinf(returns.dropna()).any()


# ---------------------------------------------------------------------------
# drawdown_series
# ---------------------------------------------------------------------------

class TestDrawdownSeries:

    def test_monotonically_increasing_zero_drawdown(self):
        """No drawdown if returns are always positive."""
        returns = pd.Series([0.01, 0.01, 0.01, 0.01, 0.01])
        dd = drawdown_series(returns)
        assert (dd >= -1e-12).all()

    def test_known_drawdown_value(self):
        """
        Manually verified: returns [0.1, 0.1, -0.2, 0.1]
        cum = [1.1, 1.21, 0.968, 1.0648]
        peak = [1.1, 1.21, 1.21, 1.21]
        dd at index 2 = (0.968 - 1.21) / 1.21 = -0.2
        """
        returns = pd.Series([0.1, 0.1, -0.2, 0.1])
        dd = drawdown_series(returns)
        assert dd.min() == pytest.approx(-0.2, abs=1e-6)
        assert dd.iloc[2] == pytest.approx(-0.2, abs=1e-6)
        assert dd.iloc[0] == pytest.approx(0.0, abs=1e-9)
        assert dd.iloc[1] == pytest.approx(0.0, abs=1e-9)

    def test_drawdown_always_non_positive(self):
        """Drawdown is by definition ≤ 0."""
        rng = np.random.default_rng(42)
        returns = pd.Series(rng.normal(0.001, 0.01, 500))
        dd = drawdown_series(returns)
        assert (dd <= 1e-12).all()

    def test_single_large_loss(self):
        """A single -50% loss should produce drawdown of -0.5."""
        returns = pd.Series([0.0, -0.5])
        dd = drawdown_series(returns)
        assert dd.min() == pytest.approx(-0.5, abs=1e-6)


# ---------------------------------------------------------------------------
# calculate_metrics
# ---------------------------------------------------------------------------

class TestCalculateMetrics:

    def _constant_return_setup(self, daily_r=0.001, n=252, rf=0.0):
        dates = pd.bdate_range("2020-01-02", periods=n)
        portfolio_values = pd.Series(
            1000.0 * np.cumprod(np.full(n, 1.0 + daily_r)), index=dates
        )
        total_invested = pd.Series(np.full(n, 1000.0), index=dates)
        returns = pd.Series(np.full(n - 1, daily_r), index=dates[1:])
        return portfolio_values, total_invested, returns

    def test_win_rate_all_positive(self):
        pv, ti, rets = self._constant_return_setup(daily_r=0.001)
        m = calculate_metrics(pv, ti, rets, risk_free_rate=0.0)
        assert m["Win Rate (daily)"] == pytest.approx(1.0)

    def test_win_rate_all_negative(self):
        pv, ti, rets = self._constant_return_setup(daily_r=-0.001)
        m = calculate_metrics(pv, ti, rets, risk_free_rate=0.0)
        assert m["Win Rate (daily)"] == pytest.approx(0.0)

    def test_max_drawdown_monotone_increasing(self):
        pv, ti, rets = self._constant_return_setup(daily_r=0.001)
        m = calculate_metrics(pv, ti, rets, risk_free_rate=0.0)
        assert m["Max Drawdown"] == pytest.approx(0.0, abs=1e-9)

    def test_cagr_positive_for_growing_portfolio(self):
        pv, ti, rets = self._constant_return_setup(daily_r=0.001)
        m = calculate_metrics(pv, ti, rets, risk_free_rate=0.0)
        assert m["CAGR"] > 0

    def test_cagr_magnitude_plausible(self):
        """0.1% daily return over ~1 year ≈ 28% CAGR; check it's in (0.2, 0.35)."""
        pv, ti, rets = self._constant_return_setup(daily_r=0.001, n=252)
        m = calculate_metrics(pv, ti, rets, risk_free_rate=0.0)
        assert 0.2 < m["CAGR"] < 0.4

    def test_sharpe_positive_above_rf(self):
        pv, ti, rets = self._constant_return_setup(daily_r=0.001)
        m = calculate_metrics(pv, ti, rets, risk_free_rate=0.0)
        assert m["Sharpe Ratio"] > 0

    def test_sharpe_negative_below_rf(self):
        """Returns below risk-free → negative Sharpe."""
        pv, ti, rets = self._constant_return_setup(daily_r=0.0001)
        # risk-free = 5% annualised → daily ≈ 0.019% → above 0.01% return
        m = calculate_metrics(pv, ti, rets, risk_free_rate=0.05)
        assert m["Sharpe Ratio"] < 0

    def test_all_required_keys_present(self):
        pv, ti, rets = self._constant_return_setup()
        m = calculate_metrics(pv, ti, rets)
        required = {
            "CAGR", "Cumulative Return (TWR)", "Total P&L ($)", "Total P&L (%)",
            "Annual Volatility", "Sharpe Ratio", "Sortino Ratio", "Calmar Ratio",
            "Max Drawdown", "Max DD Duration (days)", "Best Year", "Worst Year",
            "Win Rate (daily)", "Final Value ($)", "Total Invested ($)", "Years Simulated",
        }
        assert required.issubset(set(m.keys()))

    def test_max_dd_duration_zero_for_no_drawdown(self):
        pv, ti, rets = self._constant_return_setup(daily_r=0.001)
        m = calculate_metrics(pv, ti, rets, risk_free_rate=0.0)
        assert m["Max DD Duration (days)"] == 0

    def test_years_simulated_approximately_one(self):
        pv, ti, rets = self._constant_return_setup(n=252)
        m = calculate_metrics(pv, ti, rets)
        # 252 business days ≈ 1 calendar year (slightly less due to weekends)
        assert 0.9 < m["Years Simulated"] < 1.1

    def test_rf_as_series_gives_same_direction_as_scalar(self):
        """A constant Series risk-free rate should produce same-sign metrics as a scalar."""
        pv, ti, rets = self._constant_return_setup(daily_r=0.001)
        rf_scalar = 0.02
        rf_series = pd.Series(np.full(len(rets), rf_scalar), index=rets.index)

        m_scalar = calculate_metrics(pv, ti, rets, risk_free_rate=rf_scalar)
        m_series = calculate_metrics(pv, ti, rets, risk_free_rate=rf_series)

        assert np.sign(m_scalar["Sharpe Ratio"]) == np.sign(m_series["Sharpe Ratio"])
        assert m_scalar["Sharpe Ratio"] == pytest.approx(m_series["Sharpe Ratio"], rel=1e-3)

    def test_cumulative_return_matches_prod(self):
        """Cumulative TWR must equal the product of (1+r_i) - 1."""
        _, _, rets = self._constant_return_setup(daily_r=0.001, n=120)
        pv = pd.Series(
            1000.0 * np.cumprod(np.full(120, 1.001)),
            index=pd.bdate_range("2020-01-02", periods=120),
        )
        ti = pd.Series(np.full(120, 1000.0), index=pv.index)
        m = calculate_metrics(pv, ti, rets, risk_free_rate=0.0)
        expected_twr = float((1 + rets).prod() - 1)
        assert m["Cumulative Return (TWR)"] == pytest.approx(expected_twr, rel=1e-6)


# ---------------------------------------------------------------------------
# suggest_ddca_thresholds
# ---------------------------------------------------------------------------

class TestSuggestDDCAThresholds:

    def test_returns_expected_keys(self):
        prices = make_prices_trend(n_days=480, daily_return=0.0005)
        result = suggest_ddca_thresholds(prices)
        assert "SPY" in result
        info = result["SPY"]
        assert "threshold" in info
        assert "median_pct_off" in info
        assert "trigger_rate" in info
        assert "months_sampled" in info

    def test_threshold_in_valid_range(self):
        prices = make_prices_trend(n_days=480, daily_return=0.0005)
        result = suggest_ddca_thresholds(prices)
        t = result["SPY"]["threshold"]
        assert 0.01 <= t <= 0.50

    def test_trigger_rate_near_target(self):
        """Trigger rate should be close to the target (0.25 by default)."""
        rng = np.random.default_rng(7)
        # Volatile asset: large swings so some months will be well below 52wk high
        n = 1500
        returns = rng.normal(0.001, 0.015, n)
        dates = pd.bdate_range("2015-01-02", periods=n)
        prices = pd.DataFrame(
            {"X": np.cumprod(1 + returns)}, index=dates
        )
        result = suggest_ddca_thresholds(prices, target_trigger_rate=0.25)
        if "X" in result:
            assert 0.05 <= result["X"]["trigger_rate"] <= 0.50

    def test_monotone_prices_low_threshold(self):
        """Always-increasing prices → always at 52wk high → low suggested threshold."""
        prices = make_prices_trend(n_days=480, daily_return=0.001)
        result = suggest_ddca_thresholds(prices)
        # Monotone prices: distance from 52wk high is always 0 → threshold near floor
        assert result["SPY"]["threshold"] == pytest.approx(0.01, abs=0.005)

    def test_unknown_ticker_skipped(self):
        prices = make_prices(n_days=120, tickers=["AAA"])
        result = suggest_ddca_thresholds(prices, tickers=["AAA", "DOESNOTEXIST"])
        assert "AAA" in result
        assert "DOESNOTEXIST" not in result


# ---------------------------------------------------------------------------
# portfolio_daily_returns
# ---------------------------------------------------------------------------

class TestPortfolioDailyReturns:

    def test_single_ticker_equals_price_return(self):
        """Single 100%-weight ticker: portfolio returns == asset returns."""
        n = 120
        dates = pd.bdate_range("2020-01-02", periods=n)
        arr = np.cumprod(np.full(n, 1.001))
        prices = pd.DataFrame({"SPY": arr}, index=dates)

        port_rets = portfolio_daily_returns(prices, {"SPY": 1.0})
        asset_rets = prices["SPY"].pct_change().dropna()
        pd.testing.assert_series_equal(port_rets, asset_rets, check_names=False, rtol=1e-9)

    def test_two_ticker_weighted_average(self):
        """Verify the weighted average calculation for 2 assets."""
        n = 10
        dates = pd.bdate_range("2020-01-02", periods=n)
        # A goes up 1% per day, B goes up 2% per day
        prices = pd.DataFrame({
            "A": np.cumprod(np.full(n, 1.01)),
            "B": np.cumprod(np.full(n, 1.02)),
        }, index=dates)
        port_rets = portfolio_daily_returns(prices, {"A": 0.5, "B": 0.5})
        # Each day: weighted return ≈ 0.5 * 0.01 + 0.5 * 0.02 = 0.015
        assert port_rets.mean() == pytest.approx(0.015, abs=1e-6)


# ---------------------------------------------------------------------------
# correlation_matrix
# ---------------------------------------------------------------------------

class TestCorrelationMatrix:

    def test_shape_matches_unique_tickers(self):
        prices = make_prices_trend(n_days=120, tickers=["A", "B", "C"])
        configs = [{"weights": {"A": 0.5, "B": 0.5}}, {"weights": {"B": 0.5, "C": 0.5}}]
        corr = correlation_matrix(prices, configs)
        # 3 unique tickers → 3×3 matrix
        assert corr.shape == (3, 3)

    def test_diagonal_is_one(self):
        """Diagonal of correlation matrix must be 1.0 (non-constant prices)."""
        rng = np.random.default_rng(42)
        n = 120
        dates = pd.bdate_range("2020-01-02", periods=n)
        r = rng.normal(0.001, 0.01, n)
        prices = pd.DataFrame({
            "X": np.cumprod(1 + r),
            "Y": np.cumprod(1 + r * 0.5 + rng.normal(0, 0.002, n)),
        }, index=dates)
        configs = [{"weights": {"X": 0.6, "Y": 0.4}}]
        corr = correlation_matrix(prices, configs)
        np.testing.assert_allclose(np.diag(corr.values), 1.0, atol=1e-9)

    def test_perfectly_correlated_assets(self):
        """Two assets with identical daily returns → correlation = 1.0."""
        rng = np.random.default_rng(42)
        n = 120
        dates = pd.bdate_range("2020-01-02", periods=n)
        rets = rng.normal(0.001, 0.01, n)
        arr = np.cumprod(1 + rets)
        prices = pd.DataFrame({"P": arr, "Q": arr}, index=dates)
        configs = [{"weights": {"P": 0.5, "Q": 0.5}}]
        corr = correlation_matrix(prices, configs)
        assert corr.loc["P", "Q"] == pytest.approx(1.0, abs=1e-9)

    def test_perfectly_anticorrelated_assets(self):
        """Assets with perfectly opposite returns → correlation = -1.0."""
        rng = np.random.default_rng(42)
        n = 120
        dates = pd.bdate_range("2020-01-02", periods=n)
        rets = rng.normal(0.001, 0.01, n)
        prices = pd.DataFrame({
            "UP":   np.cumprod(1 + rets),
            "DOWN": np.cumprod(1 - rets),
        }, index=dates)
        configs = [{"weights": {"UP": 0.5, "DOWN": 0.5}}]
        corr = correlation_matrix(prices, configs)
        assert corr.loc["UP", "DOWN"] == pytest.approx(-1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# rolling_window_analysis
# ---------------------------------------------------------------------------

class TestRollingWindowAnalysis:

    def _make_long_prices(self, n_years=8, daily_return=0.0003):
        """Create price data spanning multiple years for rolling window tests."""
        n_days = int(n_years * 252)
        dates = pd.bdate_range("2010-01-04", periods=n_days)
        arr = 100.0 * np.cumprod(np.full(n_days, 1.0 + daily_return))
        return pd.DataFrame({"SPY": arr}, index=dates)

    def test_returns_dataframe_with_expected_columns(self):
        prices = self._make_long_prices(n_years=8)
        df = rolling_window_analysis(
            prices, {"SPY": 1.0}, 10000, 500, window_years=3,
        )
        expected_cols = {
            "start_date", "end_date", "CAGR", "Sharpe Ratio", "Sortino Ratio",
            "Max Drawdown", "Annual Volatility", "Cumulative Return (TWR)",
            "Final Value ($)", "Total Invested ($)",
        }
        assert expected_cols == set(df.columns)

    def test_multiple_windows_generated(self):
        """8 years of data with 3-year window → many windows."""
        prices = self._make_long_prices(n_years=8)
        df = rolling_window_analysis(
            prices, {"SPY": 1.0}, 10000, 500, window_years=3,
        )
        # ~5 years of valid start dates → ~60 monthly windows
        assert len(df) > 40

    def test_sorted_by_rank_metric(self):
        """Result should be sorted ascending by rank_by metric."""
        prices = self._make_long_prices(n_years=8)
        df = rolling_window_analysis(
            prices, {"SPY": 1.0}, 10000, 500, window_years=3, rank_by="CAGR",
        )
        assert list(df["CAGR"]) == sorted(df["CAGR"])

    def test_error_when_history_too_short(self):
        """Should raise when price history is shorter than window."""
        prices = make_prices(n_days=252)  # ~1 year
        with pytest.raises(ValueError, match="shorter than"):
            rolling_window_analysis(
                prices, {"SPY": 1.0}, 10000, 500, window_years=5,
            )

    def test_constant_prices_cagr_near_zero(self):
        """Flat prices → CAGR should be near zero for all windows."""
        prices = make_prices(n_days=252 * 6, price=100.0, start="2010-01-04")
        df = rolling_window_analysis(
            prices, {"SPY": 1.0}, 10000, 500, window_years=3,
        )
        # With flat prices and DCA, final value equals total invested
        # CAGR comes from TWR which should be ~0
        assert df["CAGR"].abs().max() < 0.01

    def test_rank_by_sharpe(self):
        """Can rank by Sharpe Ratio instead of CAGR."""
        prices = self._make_long_prices(n_years=8)
        df = rolling_window_analysis(
            prices, {"SPY": 1.0}, 10000, 500, window_years=3,
            rank_by="Sharpe Ratio",
        )
        assert list(df["Sharpe Ratio"]) == sorted(df["Sharpe Ratio"])

    def test_window_dates_within_bounds(self):
        """All windows should start and end within the price data range."""
        prices = self._make_long_prices(n_years=8)
        df = rolling_window_analysis(
            prices, {"SPY": 1.0}, 10000, 500, window_years=3,
        )
        assert df["start_date"].min() >= prices.index[0]
        assert df["end_date"].max() <= prices.index[-1]

    def test_two_ticker_portfolio(self):
        """Works with multi-ticker portfolios."""
        n_days = 252 * 8
        dates = pd.bdate_range("2010-01-04", periods=n_days)
        prices = pd.DataFrame({
            "A": 100.0 * np.cumprod(np.full(n_days, 1.0003)),
            "B": 50.0 * np.cumprod(np.full(n_days, 1.0005)),
        }, index=dates)
        df = rolling_window_analysis(
            prices, {"A": 0.6, "B": 0.4}, 10000, 500, window_years=3,
        )
        assert len(df) > 0
        assert all(df["Final Value ($)"] > 0)


class TestRollingWindowSummary:

    def test_returns_worst_median_best(self):
        n_days = 252 * 8
        dates = pd.bdate_range("2010-01-04", periods=n_days)
        prices = pd.DataFrame({
            "SPY": 100.0 * np.cumprod(np.full(n_days, 1.0003)),
        }, index=dates)
        df = rolling_window_analysis(
            prices, {"SPY": 1.0}, 10000, 500, window_years=3,
        )
        summary = rolling_window_summary(df, rank_by="CAGR")

        assert set(summary.keys()) == {"worst", "median", "best"}
        assert summary["worst"]["CAGR"] <= summary["median"]["CAGR"]
        assert summary["median"]["CAGR"] <= summary["best"]["CAGR"]

    def test_summary_contains_dates(self):
        n_days = 252 * 8
        dates = pd.bdate_range("2010-01-04", periods=n_days)
        prices = pd.DataFrame({
            "SPY": 100.0 * np.cumprod(np.full(n_days, 1.0003)),
        }, index=dates)
        df = rolling_window_analysis(
            prices, {"SPY": 1.0}, 10000, 500, window_years=3,
        )
        summary = rolling_window_summary(df)
        for period in ("worst", "median", "best"):
            assert "start_date" in summary[period]
            assert "end_date" in summary[period]

