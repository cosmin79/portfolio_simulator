# Portfolio Simulator

Compare up to three investment portfolios with DCA, leverage, and advanced Double-DCA logic.
Data is sourced from Yahoo Finance via `yfinance`.

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Usage

There are two interfaces that share the same core engine (`portfolio_sim.py`).

### CLI — `run_portfolio.py`

Edit the `CONFIGURATION` block at the top of the file, then run:

```bash
.venv/bin/python run_portfolio.py
```

Four separate figure windows open (Overview, Risk, Assets, Metrics table),
and four PNGs are saved to the current directory.

**Configuration options:**

```python
PORTFOLIOS = [
    {
        "name": "My Portfolio",
        "weights": {
            "VT":   0.60,
            "BND":  0.30,
            "GLD":  0.10,
        },
        # Optional — omit for plain DCA on all tickers
        "ddca_thresholds": {
            "VT":  0.05,   # trigger double-down when >5% below 52w high
            "GLD": 0.10,
        },
    },
    # Add up to 3 portfolios
]

START_YEAR           = 2015    # data availability may push this later
INITIAL_INVESTMENT   = 10_000  # USD lump-sum on day one
MONTHLY_CONTRIBUTION = 500     # USD added every month
RISK_FREE_RATE       = 0.04   # annual, used for Sharpe/Sortino and reserve growth
REBALANCE_ANNUALLY   = False   # rebalance to target weights each January
```

**Suggested DDCA thresholds** are printed automatically at startup — use these
as a starting point before editing the config.

### GUI — `portfolio_app.py`

```bash
.venv/bin/streamlit run portfolio_app.py
```

Opens a web UI at `http://localhost:8501`.

- **Sidebar** — configure up to 3 portfolios via editable tables (ticker, weight, DDCA %)
- **Suggest DDCA Thresholds** button — fetches price history and recommends per-ticker thresholds
- **Run Simulation** button — runs the full simulation and displays all charts

---

## Data notes

- All prices are fetched with `auto_adjust=True`, which uses the dividend- and split-adjusted close price. This means **dividends are automatically reinvested** — all return and CAGR figures reflect total return, not price return.
- The risk-free rate is fetched live from Yahoo Finance (`^IRX`, 13-week T-bill yield) for the exact simulation period, so Sharpe and Sortino ratios correctly reflect the rate environment of the time rather than a fixed assumption.

---

## Features

### Portfolios
- Up to 3 portfolios compared side-by-side
- Weights must sum to 1.0 (net); negative weights model short/leveraged positions
  (e.g. `"BIL": -0.33` to represent a 33% cash borrowing cost)
- Long exposure and borrowed % shown separately for leveraged portfolios

### Investment modes
| Mode | Behaviour |
|---|---|
| **Plain DCA** | Full monthly contribution deployed on the first trading day of each month |
| **Annual rebalance** | Portfolio reset to target weights each January before that month's contribution |
| **Double DCA (DDCA)** | Half the monthly contribution is always invested; the other half parks in a per-ticker cash reserve. When the price drops more than `threshold`% below its 52-week high, an extra draw from the reserve (up to 1× the full allocation) is deployed — maximum 1.5× the normal contribution that month |

### Double DCA detail
- Thresholds are **per-ticker** because volatility differs across assets
- The reserve earns the risk-free rate while parked
- Run **Suggest DDCA Thresholds** to calibrate: the helper finds the threshold
  that would have triggered ~25% of months historically, giving the reserve time
  to build between events
- DDCA trades a small amount of raw CAGR for lower drawdown and better
  risk-adjusted returns (Sharpe/Sortino) — it is a volatility-dampening tool,
  not a return-enhancer

### Metrics reported
| Metric | Description |
|---|---|
| CAGR | Compound annual growth rate (time-weighted) |
| Cumulative Return | Total time-weighted return over the period |
| Total P&L | Final value minus total cash invested |
| Sharpe Ratio | Annualised excess return / volatility |
| Sortino Ratio | Annualised excess return / downside deviation |
| Calmar Ratio | CAGR / absolute max drawdown |
| Max Drawdown | Largest peak-to-trough decline |
| Max DD Duration | Longest number of days spent below a prior peak |
| Best / Worst Year | Best and worst single calendar-year return |
| Win Rate | Fraction of trading days with a positive return |
| Long Exposure / Borrowed | Gross long and short weights (leveraged portfolios) |

### Charts (Streamlit tabs)
| Tab | Contents |
|---|---|
| Overview | Portfolio value over time, KPI cards, annual return bars |
| Risk & Drawdown | Drawdown fill chart, return distribution, rolling 12-month Sharpe |
| Assets & Correlation | Correlation heatmap, normalised asset prices, weight bar charts |
| Full Metrics | Styled comparison table with best-value highlighting, CSV download |

---

## Ticker symbols

Use Yahoo Finance symbols. Useful tickers by asset class:

### Equities
| Ticker | Name | Data from |
|---|---|---|
| `VOO` | S&P 500 (Vanguard) | 2010 |
| `SPY` | S&P 500 (SPDR) | 1993 |
| `VTI` | US Total Stock Market | 2001 |
| `VT` | World Total Stock Market | 2008 |
| `QQQ` | Nasdaq-100 | 1999 |
| `VXUS` | Total International (ex-US) | 2011 |
| `EWJ` | Japan | 1996 |

### Bonds
| Ticker | Name | Data from |
|---|---|---|
| `AGG` | US Aggregate Bond Market | 2003 |
| `BND` | US Aggregate Bond Market (Vanguard) | 2007 |
| `LQD` | Investment Grade Corporate Bonds | 2002 |
| `TLT` | 20+ Year US Treasuries | 2002 |
| `IEF` | 7–10 Year US Treasuries | 2002 |

### Cash / T-bills
| Ticker | Name | Data from |
|---|---|---|
| `BIL` | 1–3 Month T-bills | 2007 |
| `SHY` | 1–3 Year Treasuries | 2002 |

Use a **negative weight** on a cash/T-bill ticker to model borrowing cost (e.g. `"BIL": -0.33` for 33% leverage).

### Commodities
| Ticker | Name | Data from |
|---|---|---|
| `GLD` | Gold | 2004 |
| `GCC` | Diversified Commodity Basket (equal-weight) | 2008 |
| `DBC` | Diversified Commodity Basket (energy-heavy) | 2006 |
| `DJP` | Bloomberg Commodity Index ETN | 2006 |

### Alternatives / Managed Futures (CTAs)
| Ticker | Name | Data from |
|---|---|---|
| `DBMF` | iMGP DBi Managed Futures (replicates top CTAs) | 2019 |
| `AMFAX` | AQR Managed Futures (mutual fund) | 2010 |
| `WTMF` | WisdomTree Managed Futures | 2011 |

### Individual stocks & crypto
| Ticker | Examples |
|---|---|
| Stocks | `AAPL`, `TSLA`, `PLTR` |
| Crypto | `BTC-USD`, `ETH-USD` |

---

The simulation starts from whichever date **all** tickers in **all** portfolios
have data — so adding a recently-listed ticker will shorten the backtest window.
