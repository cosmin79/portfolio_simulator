"""
portfolio_sim.py - Core simulation engine

Handles data fetching, portfolio simulation with DCA, and performance metrics.
"""

import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

RateInput = float | pd.Series  # type alias used throughout


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def fetch_prices(tickers: list[str], start_year: int, start_month: int = 1, end_year: int | None = None, end_month: int = 12) -> pd.DataFrame:
    """
    Download adjusted close prices for a list of tickers.

    Returns a DataFrame indexed by date, one column per ticker.
    Rows where ANY ticker has missing data are dropped, so the returned
    DataFrame represents the period during which ALL tickers were trading.
    """
    start = f"{start_year}-{start_month:02d}-01"
    if end_year:
        # Last day of the specified end month
        import calendar
        last_day = calendar.monthrange(end_year, end_month)[1]
        end = f"{end_year}-{end_month:02d}-{last_day}"
    else:
        end = datetime.today().strftime("%Y-%m-%d")

    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)

    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        # Single ticker returns flat columns
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})

    # Keep only the requested tickers (yfinance may return extras)
    prices = prices[[t for t in tickers if t in prices.columns]]

    missing = [t for t in tickers if t not in prices.columns]
    if missing:
        raise ValueError(f"Could not retrieve data for: {missing}")

    # Warn about any ticker that limits the shared date window
    first_valid = {t: prices[t].first_valid_index() for t in tickers}
    last_valid  = {t: prices[t].last_valid_index()  for t in tickers}
    for t in tickers:
        if first_valid[t] is not None and first_valid[t] > prices.index[0]:
            warnings.warn(
                f"{t} only has data from {first_valid[t].date()} — "
                f"this ticker is limiting the simulation start date.",
                stacklevel=2,
            )
        if last_valid[t] is not None and last_valid[t] < prices.index[-1]:
            warnings.warn(
                f"{t} data ends at {last_valid[t].date()} — "
                f"this ticker is limiting the simulation end date.",
                stacklevel=2,
            )

    prices = prices.dropna()

    if prices.empty:
        raise ValueError(
            f"No overlapping trading days found for {tickers} from {start_year}."
        )

    return prices


def fetch_fx_rate(
    currency: str,
    start_year: int,
    start_month: int = 1,
    end_year: int | None = None,
    end_month: int = 12,
) -> pd.Series | None:
    """
    Download the USD-per-unit exchange rate for `currency` (e.g. "GBP" → GBPUSD=X).

    Returns None when currency is "USD" (no conversion needed).
    Raises ValueError if the data cannot be fetched.
    """
    if currency.upper() == "USD":
        return None

    import calendar
    ticker = f"{currency.upper()}USD=X"
    start = f"{start_year}-{start_month:02d}-01"
    if end_year:
        last_day = calendar.monthrange(end_year, end_month)[1]
        end = f"{end_year}-{end_month:02d}-{last_day}"
    else:
        end = datetime.today().strftime("%Y-%m-%d")

    raw = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    if raw.empty:
        raise ValueError(
            f"Could not fetch FX rate {ticker} for {start} → {end}. "
            f"Check that the currency code is valid."
        )
    series = raw["Close"].squeeze()
    series.name = f"{currency.upper()}USD"
    return series


def fetch_risk_free_rate(
    start_year: int,
    start_month: int = 1,
    end_year: int | None = None,
    end_month: int = 12,
) -> pd.Series:
    """
    Download the 13-week T-bill annualised yield (^IRX) as a decimal Series
    indexed by trading date.

    Returns a Series of *annualised* rates (e.g. 0.05 = 5%).
    Raises ValueError if the data cannot be fetched.
    """
    import calendar
    start = f"{start_year}-{start_month:02d}-01"
    if end_year:
        last_day = calendar.monthrange(end_year, end_month)[1]
        end = f"{end_year}-{end_month:02d}-{last_day}"
    else:
        end = datetime.today().strftime("%Y-%m-%d")
    raw = yf.download("^IRX", start=start, end=end, auto_adjust=False, progress=False)
    if raw.empty:
        raise ValueError(f"Could not fetch ^IRX (3-month T-bill) data from {start} to {end}.")
    series = raw["Close"].squeeze() / 100.0
    series.name = "risk_free_rate"
    return series


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def simulate_portfolio(
    prices: pd.DataFrame,
    weights: dict[str, float],
    initial_investment: float,
    monthly_contribution: float,
    rebalance_annually: bool = False,
    ddca_thresholds: dict[str, float] | None = None,
    risk_free_rate: RateInput = 0.04,
    fx_rate: pd.Series | None = None,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Simulate a portfolio with an initial lump-sum investment and fixed monthly DCA.

    Each month's contribution is deployed on the first trading day of that month,
    split proportionally according to `weights`.

    If `rebalance_annually` is True, the portfolio is rebalanced back to target
    weights on the first trading day of each new calendar year, before that
    month's contribution is deployed.

    Double DCA mode (ddca_thresholds)
    ----------------------------------
    Pass a dict mapping ticker → drawdown threshold (e.g. {"SPY": 0.10}).
    For each DDCA-enabled ticker, every month:
      1. Half the normal contribution is always invested immediately.
      2. The other half is added to a per-ticker cash reserve.
      3. If the current price is ≥ (1 − threshold) × 52-week high → no extra draw.
      4. If the current price is  < (1 − threshold) × 52-week high →
         draw min(full_contrib, reserve) extra from the reserve and invest it.
         Maximum total deployed that month = 1.5 × normal contribution.
    The reserve earns `risk_free_rate` daily while parked.
    Tickers without a threshold entry use regular DCA.
    Negative-weight (short) tickers always use regular DCA regardless.

    Currency conversion (`fx_rate`)
    --------------------------------
    Pass a pd.Series of USD-per-local-currency rates (e.g. GBPUSD) aligned to
    trading dates.  When provided, `initial_investment` and `monthly_contribution`
    are treated as local-currency amounts and converted to USD at each day's rate
    before buying shares.  `total_invested` is returned in local currency.
    `portfolio_values` and `reserve_values` are returned in USD — divide by the
    fx_rate series after the call to convert to local currency.

    Returns
    -------
    portfolio_values : pd.Series  — daily market value of shares + reserve (USD)
    total_invested   : pd.Series  — cumulative cash committed in input currency
    reserve_values   : pd.Series  — daily total value sitting in the cash reserve (USD)
    """
    tickers = list(weights.keys())
    w = np.array([weights[t] for t in tickers], dtype=float)
    w /= w.sum()  # normalise just in case

    prices_sub = prices[tickers]
    prices_arr = prices_sub.values
    dates = prices_sub.index

    # Pre-compute FX rate array aligned to price dates (USD per local currency unit)
    if fx_rate is not None:
        fx_arr = fx_rate.reindex(dates, method="ffill").bfill().values
    else:
        fx_arr = np.ones(len(dates))

    # Pre-compute a per-day risk-free rate array aligned to the price dates
    if isinstance(risk_free_rate, pd.Series):
        rf_aligned = risk_free_rate.reindex(dates, method="ffill").bfill()
        daily_rf_arr = (1 + rf_aligned.values) ** (1 / 252) - 1
    else:
        daily_rf_arr = np.full(len(dates), (1 + risk_free_rate) ** (1 / 252) - 1)

    # Per-ticker cash reserves (only for DDCA tickers with positive weight)
    ddca = ddca_thresholds or {}
    reserves: dict[str, float] = {
        t: 0.0 for t, wt in zip(tickers, w) if t in ddca and wt > 0
    }

    # Buy initial allocation at first-day open (approximated by first close)
    # Convert local currency to USD using the first day's FX rate
    initial_usd = initial_investment * fx_arr[0]
    shares = initial_usd * w / prices_arr[0]
    total_invested_now = initial_investment  # tracked in local currency
    prev_month = dates[0].to_period("M")
    prev_year  = dates[0].year

    values:   list[float] = []
    invested: list[float] = []
    res_vals: list[float] = []

    for i in range(len(dates)):
        # Grow reserves by risk-free rate each day
        for t in reserves:
            reserves[t] *= (1 + daily_rf_arr[i])

        if i > 0:
            curr_month = dates[i].to_period("M")
            curr_year  = dates[i].year
            if curr_month != prev_month:
                # Annual rebalance on first trading day of a new year
                # (shares only; reserves are managed independently)
                if rebalance_annually and curr_year != prev_year:
                    port_val = float(np.dot(shares, prices_arr[i]))
                    shares = port_val * w / prices_arr[i]

                # Monthly contributions per ticker (convert local → USD)
                monthly_contrib_usd = monthly_contribution * fx_arr[i]
                for j, t in enumerate(tickers):
                    ticker_contrib = monthly_contrib_usd * w[j]

                    if t in reserves:  # DDCA-enabled long ticker
                        half = ticker_contrib / 2.0

                        # Park half in reserve
                        reserves[t] += half

                        # Check 52-week high (up to 252 trading days back)
                        lookback_start = max(0, i - 251)
                        high_52w = prices_arr[lookback_start : i + 1, j].max()
                        current_price = prices_arr[i, j]
                        threshold = ddca[t]

                        if current_price < (1.0 - threshold) * high_52w:
                            # Below threshold → double-down: draw extra from reserve
                            from_reserve = min(ticker_contrib, reserves[t])
                            reserves[t] -= from_reserve
                            invest_amount = half + from_reserve
                        else:
                            invest_amount = half

                        shares[j] += invest_amount / prices_arr[i, j]

                    else:  # regular DCA (including short/leveraged tickers)
                        shares[j] += ticker_contrib / prices_arr[i, j]

                total_invested_now += monthly_contribution
                prev_month = curr_month
                prev_year  = curr_year

        total_reserve = sum(reserves.values())
        values.append(float(np.dot(shares, prices_arr[i])) + total_reserve)
        invested.append(total_invested_now)
        res_vals.append(total_reserve)

    return (
        pd.Series(values,   index=dates, name="Portfolio Value"),
        pd.Series(invested, index=dates, name="Total Invested"),
        pd.Series(res_vals, index=dates, name="Reserve"),
    )


# ---------------------------------------------------------------------------
# DDCA threshold suggestion
# ---------------------------------------------------------------------------

def suggest_ddca_thresholds(
    prices: pd.DataFrame,
    tickers: list[str] | None = None,
    target_trigger_rate: float = 0.25,
) -> dict[str, dict]:
    """
    For each ticker, suggest a DDCA threshold and report key stats.

    The threshold is the value T such that the double-down would have fired
    in approximately `target_trigger_rate` fraction of months historically
    (e.g. 0.25 → fires ~3 months/year on average, giving the reserve time
    to accumulate between events).

    Computed as the (1 - target_trigger_rate) quantile of the monthly
    "% below 52-week high" distribution.

    Returns a dict:
        {
            ticker: {
                "threshold":       float,   # suggested threshold (e.g. 0.18)
                "median_pct_off":  float,   # median monthly distance from 52wk high
                "trigger_rate":    float,   # actual trigger rate at suggested threshold
                "months_sampled":  int,
            }
        }
    """
    if tickers is None:
        tickers = list(prices.columns)

    result = {}
    for ticker in tickers:
        if ticker not in prices.columns:
            continue

        px    = prices[ticker].values
        dates = prices.index

        monthly_pct_off: list[float] = []
        prev_month = dates[0].to_period("M")

        for i in range(1, len(dates)):
            curr_month = dates[i].to_period("M")
            if curr_month != prev_month:
                lookback_start = max(0, i - 251)
                high_52w = px[lookback_start : i + 1].max()
                pct_off  = (high_52w - px[i]) / high_52w
                monthly_pct_off.append(float(pct_off))
                prev_month = curr_month

        if not monthly_pct_off:
            continue

        arr = np.array(monthly_pct_off)
        # Threshold = (1 - rate) quantile so that `rate` fraction of months exceed it
        raw = float(np.quantile(arr, 1.0 - target_trigger_rate))
        # Round to nearest 0.5 % and clip to [1 %, 50 %]
        threshold = float(np.clip(round(raw * 200) / 200, 0.01, 0.50))
        actual_rate = float((arr > threshold).mean())

        result[ticker] = {
            "threshold":      threshold,
            "median_pct_off": float(np.median(arr)),
            "trigger_rate":   actual_rate,
            "months_sampled": len(arr),
        }

    return result


# ---------------------------------------------------------------------------
# Returns helpers
# ---------------------------------------------------------------------------

def portfolio_daily_returns(prices: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    """
    Constant-weight time-weighted daily returns.

    Useful for correlation heatmaps and per-asset analysis, but does NOT
    reflect weight drift (no-rebalance) or rebalancing events.  Use
    `returns_from_simulation` for accurate per-simulation metrics.
    """
    tickers = list(weights.keys())
    w = np.array([weights[t] for t in tickers], dtype=float)
    w /= w.sum()

    asset_rets = prices[tickers].pct_change().dropna()
    return asset_rets.dot(w)


def returns_from_simulation(
    portfolio_values: pd.Series,
    total_invested: pd.Series,
) -> pd.Series:
    """
    Derive daily market returns from the simulation output, stripping out
    the effect of cash injections.

    On a contribution day the portfolio value includes new cash; without
    adjusting for this, the "return" would be artificially inflated.
    Stripped formula:
        r_i = (value_i - contribution_i) / value_{i-1} - 1

    This captures actual weight drift (no rebalance) and rebalancing events,
    so metrics computed from this series correctly differ between the two modes.
    """
    contributions = total_invested.diff().fillna(0)
    # value before new cash / previous end-of-day value
    market_values = portfolio_values - contributions
    prev_values   = portfolio_values.iloc[:-1].values
    # Guard against division by zero when the portfolio hasn't been funded yet
    # (initial_investment=0 means portfolio_value=0 until the first contribution)
    raw = np.where(prev_values > 0, market_values.iloc[1:].values / prev_values - 1, np.nan)
    returns = pd.Series(raw, index=portfolio_values.index[1:])
    # Drop any stray inf/nan so they don't poison cumulative products
    return returns.replace([np.inf, -np.inf], np.nan)


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Drawdown at each date (negative values, e.g. -0.15 means -15%)."""
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    return (cum - peak) / peak


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def calculate_metrics(
    portfolio_values: pd.Series,
    total_invested: pd.Series,
    returns: pd.Series,
    risk_free_rate: RateInput = 0.04,
) -> dict:
    """
    Compute a comprehensive set of portfolio performance metrics.

    Parameters
    ----------
    portfolio_values : daily market value of the portfolio
    total_invested   : cumulative cash deployed
    returns          : time-weighted daily returns (from portfolio_daily_returns)
    risk_free_rate   : annual risk-free rate — either a scalar float or a
                       pd.Series of annualised rates indexed by date (e.g.
                       from fetch_risk_free_rate).  When a Series is passed
                       it is aligned and interpolated to match the returns index.
    """
    if isinstance(risk_free_rate, pd.Series):
        rf_aligned = risk_free_rate.reindex(returns.index, method="ffill").bfill()
        daily_rf = (1 + rf_aligned) ** (1 / 252) - 1
    else:
        daily_rf = (1 + risk_free_rate) ** (1 / 252) - 1
    n_years = (portfolio_values.index[-1] - portfolio_values.index[0]).days / 365.25

    final_value = portfolio_values.iloc[-1]
    total_inv = total_invested.iloc[-1]

    # ---- Return metrics ----
    cumulative_twr = float((1 + returns).prod() - 1)
    cagr = float((1 + cumulative_twr) ** (1 / n_years) - 1) if n_years > 0 else float("nan")

    total_pnl = final_value - total_inv
    total_pnl_pct = total_pnl / total_inv

    # ---- Risk metrics ----
    annual_vol = float(returns.std() * np.sqrt(252))

    excess = returns - daily_rf
    sharpe = float(excess.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else float("nan")

    below_rf = returns < daily_rf
    down = (returns - daily_rf)[below_rf]
    if len(down) > 0:
        down_std_daily = float(np.sqrt((down ** 2).mean()))
        mean_excess = float(excess.mean())
        sortino = float(mean_excess / down_std_daily * np.sqrt(252)) if down_std_daily > 0 else float("nan")
    else:
        sortino = float("nan")

    dd = drawdown_series(returns)
    max_dd = float(dd.min())

    # Max drawdown duration (consecutive days in drawdown)
    in_dd = dd < 0
    groups = (in_dd != in_dd.shift()).cumsum()
    dd_lengths = in_dd.groupby(groups).sum()
    max_dd_days = int(dd_lengths.max()) if in_dd.any() else 0

    calmar = cagr / abs(max_dd) if max_dd != 0 else float("nan")

    # ---- Per-year breakdown ----
    annual_rets = returns.resample("YE").apply(lambda x: float((1 + x).prod() - 1))
    best_year = float(annual_rets.max())
    worst_year = float(annual_rets.min())

    # Win rate
    win_rate = float((returns > 0).mean())

    return {
        "CAGR": cagr,
        "Cumulative Return (TWR)": cumulative_twr,
        "Total P&L ($)": total_pnl,
        "Total P&L (%)": total_pnl_pct,
        "Annual Volatility": annual_vol,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Calmar Ratio": calmar,
        "Max Drawdown": max_dd,
        "Max DD Duration (days)": max_dd_days,
        "Best Year": best_year,
        "Worst Year": worst_year,
        "Win Rate (daily)": win_rate,
        "Final Value ($)": final_value,
        "Total Invested ($)": total_inv,
        "Years Simulated": n_years,
    }


# ---------------------------------------------------------------------------
# Correlation
# ---------------------------------------------------------------------------

def correlation_matrix(prices: pd.DataFrame, portfolio_configs: list[dict]) -> pd.DataFrame:
    """
    Return the correlation matrix of daily returns for all unique tickers
    across the supplied portfolio configs.

    Each portfolio config is a dict with at least a 'weights' key:
        {'name': '...', 'weights': {'SPY': 0.6, 'BND': 0.4}}
    """
    all_tickers = []
    for cfg in portfolio_configs:
        for t in cfg["weights"]:
            if t not in all_tickers:
                all_tickers.append(t)

    rets = prices[all_tickers].pct_change().dropna()
    return rets.corr()


# ---------------------------------------------------------------------------
# Annual returns table
# ---------------------------------------------------------------------------

def annual_returns_table(returns: pd.Series) -> pd.Series:
    """Yearly total returns from a daily return series."""
    return returns.resample("YE").apply(lambda x: float((1 + x).prod() - 1))


# ---------------------------------------------------------------------------
# Rolling window analysis — worst / median / best period finder
# ---------------------------------------------------------------------------

def rolling_window_analysis(
    prices: pd.DataFrame,
    weights: dict[str, float],
    initial_investment: float,
    monthly_contribution: float,
    window_years: int = 5,
    rank_by: str = "CAGR",
    rebalance_annually: bool = False,
    ddca_thresholds: dict[str, float] | None = None,
    risk_free_rate: RateInput = 0.04,
    fx_rate: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Slide a fixed-length window across history and run the full DCA simulation
    for each starting month.  Returns a DataFrame with one row per window,
    sorted by `rank_by` ascending (worst first).

    Parameters
    ----------
    prices              : price DataFrame covering the full available history
    weights             : portfolio weights
    initial_investment  : lump-sum at start of each window (in local currency)
    monthly_contribution: monthly DCA amount (in local currency)
    window_years        : length of each window in years (default 5)
    rank_by             : metric key to sort by — any key from calculate_metrics
                          (default "CAGR")
    rebalance_annually  : passed to simulate_portfolio
    ddca_thresholds     : passed to simulate_portfolio
    risk_free_rate      : scalar or Series; if Series, sliced per window
    fx_rate             : optional FX Series; sliced per window

    Returns
    -------
    DataFrame with columns:
        start_date, end_date, CAGR, Sharpe Ratio, Sortino Ratio,
        Max Drawdown, Annual Volatility, Cumulative Return (TWR),
        Final Value ($), Total Invested ($)
    Sorted by `rank_by` ascending.
    """
    tickers = list(weights.keys())
    available = prices[tickers].dropna()
    all_dates = available.index

    if len(all_dates) < 2:
        raise ValueError("Not enough price data for rolling window analysis.")

    window_days = int(window_years * 365.25)
    first_date = all_dates[0]
    last_possible_start = all_dates[-1] - pd.Timedelta(days=window_days)

    if last_possible_start < first_date:
        raise ValueError(
            f"Price history ({(all_dates[-1] - first_date).days / 365.25:.1f} yrs) "
            f"is shorter than the requested {window_years}-year window."
        )

    # Generate monthly start dates
    start_dates = []
    seen_months: set[tuple[int, int]] = set()
    for d in all_dates:
        if d > last_possible_start:
            break
        key = (d.year, d.month)
        if key not in seen_months:
            seen_months.add(key)
            start_dates.append(d)

    rows: list[dict] = []
    for start in start_dates:
        end = start + pd.Timedelta(days=window_days)
        window_prices = available.loc[start:end]

        if len(window_prices) < 20:  # skip tiny windows
            continue

        # Slice risk-free rate and FX for this window
        if isinstance(risk_free_rate, pd.Series):
            rf_window = risk_free_rate.loc[
                (risk_free_rate.index >= start) & (risk_free_rate.index <= end)
            ]
            rf_arg: RateInput = rf_window if not rf_window.empty else 0.04
        else:
            rf_arg = risk_free_rate

        if fx_rate is not None:
            fx_window = fx_rate.loc[
                (fx_rate.index >= start) & (fx_rate.index <= end)
            ]
            fx_arg: pd.Series | None = fx_window if not fx_window.empty else None
        else:
            fx_arg = None

        vals_usd, invested, _ = simulate_portfolio(
            window_prices,
            weights,
            initial_investment,
            monthly_contribution,
            rebalance_annually=rebalance_annually,
            ddca_thresholds=ddca_thresholds,
            risk_free_rate=rf_arg,
            fx_rate=fx_arg,
        )

        # Convert to local currency if FX is involved
        if fx_arg is not None:
            fx_al = fx_rate.reindex(vals_usd.index, method="ffill").bfill()
            vals = vals_usd / fx_al
        else:
            vals = vals_usd

        rets = returns_from_simulation(vals, invested)
        metrics = calculate_metrics(vals, invested, rets, rf_arg)

        rows.append({
            "start_date": window_prices.index[0],
            "end_date": window_prices.index[-1],
            "CAGR": metrics["CAGR"],
            "Sharpe Ratio": metrics["Sharpe Ratio"],
            "Sortino Ratio": metrics["Sortino Ratio"],
            "Max Drawdown": metrics["Max Drawdown"],
            "Annual Volatility": metrics["Annual Volatility"],
            "Cumulative Return (TWR)": metrics["Cumulative Return (TWR)"],
            "Final Value ($)": metrics["Final Value ($)"],
            "Total Invested ($)": metrics["Total Invested ($)"],
        })

    if not rows:
        raise ValueError("No valid windows could be constructed from the data.")

    df = pd.DataFrame(rows)
    df = df.sort_values(rank_by, ascending=True).reset_index(drop=True)
    return df


def rolling_window_summary(
    analysis: pd.DataFrame,
    rank_by: str = "CAGR",
) -> dict[str, dict]:
    """
    Extract worst, median, and best periods from a rolling_window_analysis result.

    Returns a dict with keys "worst", "median", "best", each containing the
    full row as a dict.
    """
    df = analysis.sort_values(rank_by, ascending=True).reset_index(drop=True)

    worst = df.iloc[0].to_dict()
    best = df.iloc[-1].to_dict()
    median_idx = len(df) // 2
    median = df.iloc[median_idx].to_dict()

    return {
        "worst": worst,
        "median": median,
        "best": best,
    }
