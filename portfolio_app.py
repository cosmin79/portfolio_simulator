"""
portfolio_app.py - Streamlit GUI

Run with:
    streamlit run portfolio_app.py
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from portfolio_sim import (
    fetch_prices,
    fetch_fx_rate,
    fetch_risk_free_rate,
    simulate_portfolio,
    portfolio_daily_returns,
    returns_from_simulation,
    calculate_metrics,
    correlation_matrix,
    drawdown_series,
    annual_returns_table,
    suggest_ddca_thresholds,
    rolling_window_analysis,
    rolling_window_summary,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Portfolio Simulator",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Portfolio Simulator")
st.caption("Compare two portfolios with DCA · Sharpe · Sortino · Drawdown · Correlation")

# ---------------------------------------------------------------------------
# Sidebar helpers
# ---------------------------------------------------------------------------

COLORS = ["#2196F3", "#FF5722", "#4CAF50"]   # blue, deep-orange, green

PRESETS: dict[str, list[tuple[str, float]]] = {
    # --- Custom / leveraged ---
    "All Weather CTA": [
        ("SPY",   0.603),
        ("AGG",   0.402),
        ("LQD",  -0.335),
        ("AMFAX", 0.330),
    ],
    "All Weather Commodity": [
        ("SPY",  0.603),
        ("AGG",  0.402),
        ("LQD", -0.335),
        ("GLD",  0.165),
        ("GCC",  0.165),
    ],
    # --- Classic allocations (portfoliocharts.com) ---
    "Total Stock Market": [
        ("VTI", 1.0),
    ],
    "Classic 60-40 (Bogle)": [
        ("VTI", 0.60),
        ("AGG", 0.40),
    ],
    "Three-Fund (Bogleheads)": [
        ("VTI", 0.64),
        ("EFA", 0.16),
        ("AGG", 0.20),
    ],
    "No-Brainer (Bernstein)": [
        ("VTI", 0.25),
        ("VB",  0.25),
        ("EFA", 0.25),
        ("SHY", 0.25),
    ],
    "Core Four (Ferri)": [
        ("VTI", 0.48),
        ("EFA", 0.24),
        ("AGG", 0.20),
        ("VNQ", 0.08),
    ],
    "Permanent (Harry Browne)": [
        ("VTI", 0.25),
        ("TLT", 0.25),
        ("BIL", 0.25),
        ("GLD", 0.25),
    ],
    "Larry (Swedroe)": [
        ("VBR", 0.15),
        ("DLS", 0.075),
        ("VWO", 0.075),
        ("AGG", 0.70),
    ],
    "Golden Butterfly": [
        ("VTI", 0.20),
        ("VBR", 0.20),
        ("TLT", 0.20),
        ("SHY", 0.20),
        ("GLD", 0.20),
    ],
    "All Seasons (Dalio)": [
        ("VTI", 0.30),
        ("TLT", 0.40),
        ("AGG", 0.15),
        ("GSG", 0.075),
        ("GLD", 0.075),
    ],
    "Ivy Portfolio (Faber)": [
        ("VTI", 0.20),
        ("EFA", 0.20),
        ("AGG", 0.20),
        ("GSG", 0.20),
        ("VNQ", 0.20),
    ],
    "Swensen (Yale)": [
        ("VTI", 0.30),
        ("EFA", 0.15),
        ("VWO", 0.05),
        ("AGG", 0.30),
        ("VNQ", 0.20),
    ],
    "Weird Portfolio": [
        ("VBR", 0.20),
        ("SCZ", 0.20),
        ("TLT", 0.20),
        ("VNQ", 0.20),
        ("GLD", 0.20),
    ],
}


def portfolio_input_block(label: str, default_tickers: list[tuple[str, float]]) -> dict:
    """
    Renders a portfolio input section in the sidebar.
    Returns a dict {ticker: weight}.

    Negative weights represent short / borrowed positions (e.g. short BIL to
    model a cash borrowing cost in a leveraged strategy).  Weights must sum
    to 1.0 (net NAV = 100 %).  Non-leveraged portfolios can be auto-normalised;
    leveraged ones cannot, because rescaling would alter the leverage ratio.
    """
    st.sidebar.subheader(label)

    preset_options = ["Custom"] + list(PRESETS.keys())
    selected_preset = st.sidebar.selectbox(
        "Load preset", preset_options, index=0, key=f"preset_{label}"
    )
    active_tickers = PRESETS[selected_preset] if selected_preset != "Custom" else default_tickers

    default_df = pd.DataFrame(
        [(t, w, None) for t, w in active_tickers],
        columns=["Ticker", "Weight", "DDCA %"],
    )
    edited = st.sidebar.data_editor(
        default_df,
        num_rows="dynamic",
        use_container_width=True,
        key=f"editor_{label}_{selected_preset}",
        column_config={
            "Ticker": st.column_config.TextColumn("Ticker", width="small"),
            "Weight": st.column_config.NumberColumn(
                "Weight",
                min_value=-10.0,
                max_value=10.0,
                step=0.001,
                format="%.3f",
                help="Negative = short/borrowed position",
            ),
            "DDCA %": st.column_config.NumberColumn(
                "DDCA %",
                min_value=0.0,
                max_value=100.0,
                step=1.0,
                format="%.0f",
                help="Double-DCA threshold: % below 52-week high that triggers double-down. Leave blank for regular DCA.",
            ),
        },
    )

    weights_raw: dict[str, float] = {}
    ddca_raw: dict[str, float] = {}
    for _, row in edited.iterrows():
        ticker = str(row["Ticker"]).strip().upper()
        if not ticker or ticker == "NAN":
            continue
        weight = float(row["Weight"]) if not pd.isna(row["Weight"]) else 0.0
        if weight != 0.0:
            weights_raw[ticker] = weight
        ddca_val = row.get("DDCA %")
        if ddca_val is not None and not pd.isna(ddca_val) and float(ddca_val) > 0:
            ddca_raw[ticker] = float(ddca_val) / 100.0

    if not weights_raw:
        st.sidebar.error(f"{label}: no valid weights entered.")
        return weights_raw, {}, selected_preset

    net = sum(weights_raw.values())
    is_leveraged = any(w < 0 for w in weights_raw.values())

    if abs(net - 1.0) > 1e-3:
        if is_leveraged:
            st.sidebar.error(
                f"{label}: net weights sum to **{net:.4f}** (must be 1.0). "
                f"Adjust manually — auto-normalisation is disabled for leveraged portfolios."
            )
        else:
            st.sidebar.warning(f"{label}: weights sum to {net:.3f} — normalising to 1.0.")
            weights_raw = {t: w / net for t, w in weights_raw.items()}

    if is_leveraged:
        long_exp = sum(w for w in weights_raw.values() if w > 0)
        borrowed = abs(sum(w for w in weights_raw.values() if w < 0))
        st.sidebar.info(f"{label}: long **{long_exp:.1%}** | borrowed **{borrowed:.1%}**")

    if ddca_raw:
        ddca_summary = ", ".join(f"{t} {v:.0%}" for t, v in ddca_raw.items())
        st.sidebar.info(f"{label}: DDCA active — {ddca_summary}")

    return weights_raw, ddca_raw, selected_preset


# ---------------------------------------------------------------------------
# Sidebar — portfolio inputs (up to 3)
# ---------------------------------------------------------------------------
st.sidebar.header("Portfolios")

_DEFAULTS = [
    [("SPY", 0.60), ("BND", 0.30), ("GLD", 0.10)],
    [("QQQ", 0.50), ("VTI", 0.30), ("VXUS", 0.20)],
    [],   # third portfolio starts empty; add tickers to enable it
]
_LABELS = ["Portfolio A", "Portfolio B", "Portfolio C"]

portfolio_inputs = []
for i, (label, defaults) in enumerate(zip(_LABELS, _DEFAULTS)):
    if i > 0:
        st.sidebar.divider()
    weights, ddca, preset_name = portfolio_input_block(label, defaults)
    default_name = preset_name if preset_name != "Custom" else label
    name = st.sidebar.text_input(f"Name for {label}", value=default_name, key=f"name_{i}_{preset_name}")
    portfolio_inputs.append({"name": name, "weights": weights, "ddca_thresholds": ddca})

# ---------------------------------------------------------------------------
# Sidebar — investment settings
# ---------------------------------------------------------------------------
st.sidebar.divider()
st.sidebar.header("Investment Settings")

from datetime import datetime as _dt
_now = _dt.today()

start_year = st.sidebar.slider("Start Year", 1970, _now.year, 2015)
start_month = st.sidebar.slider("Start Month", 1, 12, 1)
use_custom_end = st.sidebar.checkbox("Custom end date", value=False)
if use_custom_end:
    end_year  = st.sidebar.slider("End Year",  1970, _now.year, _now.year)
    end_month = st.sidebar.slider("End Month", 1, 12, _now.month)
else:
    end_year  = None
    end_month = 12
CURRENCY_SYMBOLS = {"USD": "$", "GBP": "£", "EUR": "€", "CHF": "CHF "}
currency = st.sidebar.selectbox("Investment Currency", list(CURRENCY_SYMBOLS.keys()), index=0)
ccy_sym = CURRENCY_SYMBOLS[currency]

initial_inv = st.sidebar.number_input(
    f"Initial Investment ({currency})", min_value=0, max_value=10_000_000, value=10_000, step=1_000
)
monthly_contrib = st.sidebar.number_input(
    f"Monthly Contribution ({currency})", min_value=0, max_value=100_000, value=500, step=100
)
rebalance_annually = st.sidebar.checkbox(
    "Rebalance annually (each January)",
    value=False,
    help="On the first trading day of each new year, sell/buy assets to restore target weights.",
)

st.sidebar.divider()
st.sidebar.header("Rolling Period Analysis")
rolling_window_years = st.sidebar.slider(
    "Window length (years)", 1, 20, 5,
    help="Slide a fixed-length window across history to find the worst, median, and best starting periods.",
)
rolling_rank_by = st.sidebar.selectbox(
    "Rank periods by",
    ["CAGR", "Sharpe Ratio", "Max Drawdown", "Sortino Ratio", "Annual Volatility"],
    index=0,
    help="Which metric to use for determining worst/median/best.",
)

run_btn      = st.sidebar.button("Run Simulation", type="primary", use_container_width=True)
compare_btn  = st.sidebar.button("Compare All Presets", use_container_width=True,
                                  help="Run every preset portfolio with the current investment settings and rank them side by side.")
suggest_btn  = st.sidebar.button("Suggest DDCA Thresholds", use_container_width=True,
                                  help="Fetch price history and recommend per-ticker thresholds based on historical drawdown distribution.")

# Persist compare mode across reruns so that interacting with widgets inside
# the compare block doesn't reset the page.
if compare_btn:
    st.session_state["compare_active"] = True
if run_btn or suggest_btn:
    st.session_state["compare_active"] = False
compare_active = st.session_state.get("compare_active", False)

# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def pct(v):
    return f"{v * 100:.2f}%"

def dollar(v):
    return f"{ccy_sym}{v:,.0f}"

def fmt(key, v):
    if isinstance(v, float) and np.isnan(v):
        return "N/A"
    if key in ("CAGR", "Cumulative Return (TWR)", "Total P&L (%)",
               "Annual Volatility", "Max Drawdown", "Best Year", "Worst Year",
               "Win Rate (daily)"):
        return pct(v)
    if key in ("Total P&L ($)", "Final Value ($)", "Total Invested ($)"):
        return dollar(v)
    if key in ("Sharpe Ratio", "Sortino Ratio", "Calmar Ratio"):
        return f"{v:.3f}"
    if key in ("Long Exposure", "Borrowed"):
        return pct(v)
    if key == "Max DD Duration (days)":
        return f"{int(v):,} days"
    if key == "Years Simulated":
        return f"{v:.1f} yrs"
    return str(v)


# ---------------------------------------------------------------------------
# DDCA threshold suggestion
# ---------------------------------------------------------------------------

if suggest_btn:
    all_cfgs    = [p for p in portfolio_inputs if p["weights"]]
    long_tickers = list(dict.fromkeys(
        t for p in all_cfgs for t, w in p["weights"].items() if w > 0
    ))
    if not long_tickers:
        st.warning("Add at least one ticker first.")
    else:
        with st.spinner("Fetching price history for threshold analysis…"):
            try:
                import warnings as _warnings
                with _warnings.catch_warnings(record=True) as _caught:
                    _warnings.simplefilter("always")
                    px = fetch_prices(long_tickers, start_year, start_month, end_year, end_month)
                for w in _caught:
                    st.warning(str(w.message))
            except ValueError as e:
                st.error(str(e))
                px = None
        if px is not None:
            suggestions = suggest_ddca_thresholds(px, tickers=long_tickers)
            st.subheader("Suggested DDCA Thresholds")
            st.caption(
                "Threshold = % below 52-week high that triggers the double-down. "
                "Calibrated so the trigger fires ~25 % of months (≈3×/year), "
                "giving the reserve time to build between events. "
                "Increase the threshold for volatile stocks to avoid firing every month."
            )
            rows = []
            for t, s in suggestions.items():
                rows.append({
                    "Ticker":           t,
                    "Suggested threshold": f"{s['threshold']:.1%}",
                    "Median off 52w high": f"{s['median_pct_off']:.1%}",
                    "Actual trigger rate": f"{s['trigger_rate']:.1%}",
                    "Months sampled":      s["months_sampled"],
                })
            st.dataframe(pd.DataFrame(rows).set_index("Ticker"), use_container_width=True)
            st.info(
                "💡 If the suggested threshold is very high (>25 %), the stock is frequently "
                "far below its 52-week high and DDCA will fire almost every month — "
                "consider whether DDCA is appropriate for it, or use a threshold closer to "
                "its median drawdown so you get occasional large deployments instead of "
                "constant small ones."
            )


# ---------------------------------------------------------------------------
# Compare all presets
# ---------------------------------------------------------------------------

if compare_active:
    # ------------------------------------------------------------------
    # Step 1: Fetch data & run all simulations ONCE, cache in session_state
    # ------------------------------------------------------------------
    if compare_btn or "compare_results" not in st.session_state:
        preset_cfgs = [
            (name, tickers)
            for name, tickers in PRESETS.items()
            if all(w >= 0 for _, w in tickers)
        ]

        all_preset_tickers = list(dict.fromkeys(
            t for _, tickers in preset_cfgs for t, _ in tickers
        ))

        with st.spinner(f"Downloading data for {len(all_preset_tickers)} tickers across {len(preset_cfgs)} presets…"):
            try:
                import warnings as _warnings
                with _warnings.catch_warnings(record=True) as _caught:
                    _warnings.simplefilter("always")
                    prices = fetch_prices(all_preset_tickers, start_year, start_month, end_year, end_month)
                for w in _caught:
                    st.warning(str(w.message))
            except ValueError as e:
                st.error(str(e))
                st.stop()

        try:
            rf_series = fetch_risk_free_rate(start_year, start_month, end_year, end_month)
        except ValueError as e:
            st.error(str(e))
            st.stop()

        try:
            fx_series = fetch_fx_rate(currency, start_year, start_month, end_year, end_month)
        except ValueError as e:
            st.error(str(e))
            st.stop()

        # Run single-period simulation + rolling window analysis for each preset
        single_period_rows = []
        rolling_results = {}   # name → rolling_window_analysis DataFrame
        value_series = {}
        skipped = []
        progress = st.progress(0, text="Simulating presets…")

        for idx, (name, ticker_weights) in enumerate(preset_cfgs):
            weights = {t: w for t, w in ticker_weights}
            missing = [t for t in weights if t not in prices.columns]
            if missing:
                skipped.append(f"{name} (missing {', '.join(missing)})")
                progress.progress((idx + 1) / len(preset_cfgs))
                continue

            # Single period simulation
            try:
                vals_usd, invested_local, _ = simulate_portfolio(
                    prices, weights, initial_inv, monthly_contrib,
                    rebalance_annually=rebalance_annually,
                    risk_free_rate=rf_series,
                    fx_rate=fx_series,
                )
                if fx_series is not None:
                    fx_al = fx_series.reindex(vals_usd.index, method="ffill").bfill()
                    vals = vals_usd / fx_al
                else:
                    vals = vals_usd
                invested = invested_local
                rets = returns_from_simulation(vals, invested)
                metrics = calculate_metrics(vals, invested, rets, rf_series)
                single_period_rows.append({"Portfolio": name, **metrics})
                value_series[name] = vals
            except Exception:
                skipped.append(name)
                progress.progress((idx + 1) / len(preset_cfgs))
                continue

            # Rolling window analysis
            try:
                rwa = rolling_window_analysis(
                    prices, weights, initial_inv, monthly_contrib,
                    window_years=rolling_window_years,
                    rank_by="CAGR",  # rank_by doesn't matter for storage; we re-sort later
                    rebalance_annually=rebalance_annually,
                    risk_free_rate=rf_series,
                    fx_rate=fx_series,
                )
                rolling_results[name] = rwa
            except ValueError:
                pass  # not enough history for this preset — rolling will be unavailable

            progress.progress((idx + 1) / len(preset_cfgs), text=f"Simulated {idx + 1}/{len(preset_cfgs)}…")
        progress.empty()

        st.session_state["compare_results"] = {
            "single_period_rows": single_period_rows,
            "rolling_results": rolling_results,
            "value_series": value_series,
            "skipped": skipped,
            "actual_start": prices.index[0].date(),
            "actual_end": prices.index[-1].date(),
            "invested": invested,
            "n_trading_days": len(prices),
        }

    # ------------------------------------------------------------------
    # Step 2: Render from cached results (survives widget interactions)
    # ------------------------------------------------------------------
    cached = st.session_state["compare_results"]
    actual_start = cached["actual_start"]
    actual_end = cached["actual_end"]

    st.info(
        f"Data: **{actual_start}** → **{actual_end}** "
        f"({cached['n_trading_days']:,} trading days) · "
        f"Initial: **{dollar(initial_inv)}** + **{dollar(monthly_contrib)}/mo**"
    )

    compare_mode = st.radio(
        "Comparison mode",
        ["Single period", "Worst period", "Median period", "Best period"],
        horizontal=True,
        help=(
            "**Single period**: simulate from the sidebar start→end dates. "
            "**Worst/Median/Best period**: run rolling window analysis and rank each "
            "preset by its worst, median, or best N-year window."
        ),
    )
    use_rolling = compare_mode != "Single period"

    rank_metric = st.selectbox(
        "Rank by", ["CAGR", "Sharpe Ratio", "Sortino Ratio", "Max Drawdown", "Annual Volatility"],
        index=0, key="compare_rank_metric",
    )

    if cached["skipped"]:
        st.warning(f"Skipped: {', '.join(cached['skipped'])}")

    # Build leaderboard from cached data
    if use_rolling:
        period_key = {"Worst period": "worst", "Median period": "median", "Best period": "best"}[compare_mode]
        leaderboard_rows = []
        for name, rwa in cached["rolling_results"].items():
            summary = rolling_window_summary(rwa, rank_by=rank_metric)
            s = summary[period_key]
            start_str = s["start_date"].strftime("%Y-%m-%d") if hasattr(s["start_date"], "strftime") else str(s["start_date"])
            end_str = s["end_date"].strftime("%Y-%m-%d") if hasattr(s["end_date"], "strftime") else str(s["end_date"])
            leaderboard_rows.append({
                "Portfolio": name,
                "Period": f"{start_str} → {end_str}",
                "Windows": len(rwa),
                "CAGR": s["CAGR"],
                "Sharpe Ratio": s["Sharpe Ratio"],
                "Sortino Ratio": s["Sortino Ratio"],
                "Max Drawdown": s["Max Drawdown"],
                "Annual Volatility": s["Annual Volatility"],
                "Cumulative Return (TWR)": s["Cumulative Return (TWR)"],
                "Final Value ($)": s["Final Value ($)"],
                "Total Invested ($)": s["Total Invested ($)"],
            })
        if not leaderboard_rows:
            st.error(f"No presets have enough history for {rolling_window_years}-year rolling windows.")
            st.stop()
    else:
        leaderboard_rows = cached["single_period_rows"]

    if not leaderboard_rows:
        st.error("No presets could be simulated with the available data range.")
        st.stop()

    df_lb = pd.DataFrame(leaderboard_rows)

    # Sort: higher is better for most metrics, except Max Drawdown / Volatility
    if rank_metric == "Max Drawdown":
        df_lb = df_lb.sort_values(rank_metric, ascending=False).reset_index(drop=True)
    elif rank_metric == "Annual Volatility":
        df_lb = df_lb.sort_values(rank_metric, ascending=True).reset_index(drop=True)
    else:
        df_lb = df_lb.sort_values(rank_metric, ascending=False).reset_index(drop=True)
    df_lb.index = df_lb.index + 1
    df_lb.index.name = "Rank"

    # Display columns
    if use_rolling:
        show_cols = [
            "Portfolio", "Period", "CAGR", "Sharpe Ratio", "Sortino Ratio",
            "Annual Volatility", "Max Drawdown",
            "Final Value ($)", "Total Invested ($)", "Windows",
        ]
        mode_label = f"{compare_mode} ({rolling_window_years}yr windows)"
    else:
        show_cols = [
            "Portfolio", "CAGR", "Sharpe Ratio", "Sortino Ratio", "Calmar Ratio",
            "Annual Volatility", "Max Drawdown", "Max DD Duration (days)",
            "Final Value ($)", "Total Invested ($)", "Best Year", "Worst Year",
        ]
        mode_label = f"Single period ({actual_start} → {actual_end})"

    display = df_lb[[c for c in show_cols if c in df_lb.columns]].copy()

    # Format for display
    for c in ["CAGR", "Annual Volatility", "Max Drawdown", "Best Year", "Worst Year", "Cumulative Return (TWR)"]:
        if c in display.columns:
            display[c] = df_lb[c].map(lambda v: pct(v) if not (isinstance(v, float) and np.isnan(v)) else "N/A")
    for c in ["Sharpe Ratio", "Sortino Ratio", "Calmar Ratio"]:
        if c in display.columns:
            display[c] = df_lb[c].map(lambda v: f"{v:.3f}" if not (isinstance(v, float) and np.isnan(v)) else "N/A")
    for c in ["Final Value ($)", "Total Invested ($)"]:
        if c in display.columns:
            display[c] = df_lb[c].map(lambda v: dollar(v))
    if "Max DD Duration (days)" in display.columns:
        display["Max DD Duration (days)"] = df_lb["Max DD Duration (days)"].map(lambda v: f"{int(v):,}")

    st.subheader(f"Preset Leaderboard — {mode_label} — ranked by {rank_metric}")
    st.dataframe(display, use_container_width=True, height=min(800, 40 + 35 * len(display)))

    # Portfolio value chart (single-period mode only)
    if not use_rolling and cached["value_series"]:
        st.subheader("Portfolio Value — All Presets")
        fig_all = go.Figure()
        ranked_names = df_lb["Portfolio"].tolist()
        for name in ranked_names:
            if name in cached["value_series"]:
                v = cached["value_series"][name]
                fig_all.add_trace(go.Scatter(
                    x=v.index, y=v, name=name, mode="lines", line=dict(width=1.5),
                ))
        fig_all.add_trace(go.Scatter(
            x=cached["invested"].index, y=cached["invested"],
            name="Total Invested", line=dict(color="gray", width=1.5, dash="dash"), opacity=0.6,
        ))
        fig_all.update_layout(
            xaxis_title="Date",
            yaxis_title=f"Value ({currency})",
            yaxis_tickprefix=ccy_sym,
            yaxis_tickformat=",.0f",
            hovermode="x unified",
            height=500,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_all, use_container_width=True)

    # CSV download
    csv_lb = df_lb[[c for c in show_cols if c in df_lb.columns]].to_csv()
    st.download_button(
        label="Download leaderboard as CSV",
        data=csv_lb,
        file_name="preset_leaderboard.csv",
        mime="text/csv",
    )


# ---------------------------------------------------------------------------
# Main — simulation & charts
# ---------------------------------------------------------------------------

elif run_btn:
    portfolio_cfgs = [p for p in portfolio_inputs if p["weights"]]
    if len(portfolio_cfgs) < 1:
        st.error("Add at least one portfolio with valid tickers and weights.")
        st.stop()
    all_tickers = list(dict.fromkeys(t for p in portfolio_cfgs for t in p["weights"]))

    with st.spinner(f"Downloading data for {all_tickers} …"):
        try:
            import warnings as _warnings
            with _warnings.catch_warnings(record=True) as _caught:
                _warnings.simplefilter("always")
                prices = fetch_prices(all_tickers, start_year, start_month, end_year, end_month)
            for w in _caught:
                st.warning(str(w.message))
        except ValueError as e:
            st.error(str(e))
            st.stop()

    actual_start = prices.index[0].date()
    actual_end   = prices.index[-1].date()

    try:
        rf_series = fetch_risk_free_rate(start_year, start_month, end_year, end_month)
    except ValueError as e:
        st.error(str(e))
        st.stop()
    rf_mean = rf_series.reindex(prices.index, method="ffill").bfill().mean()

    # FX rate (USD per local currency unit); None when currency is USD
    try:
        fx_series = fetch_fx_rate(currency, start_year, start_month, end_year, end_month)
    except ValueError as e:
        st.error(str(e))
        st.stop()

    fx_info = ""
    if fx_series is not None:
        fx_aligned = fx_series.reindex(prices.index, method="ffill").bfill()
        fx_info = f" · Avg {currency}/USD: **{fx_aligned.mean():.4f}**"

    st.info(
        f"Data: **{actual_start}** → **{actual_end}** "
        f"({len(prices):,} trading days, {len(all_tickers)} tickers) · "
        f"Avg risk-free rate: **{rf_mean:.2%}** (^IRX){fx_info}"
    )

    # ---- simulate ----
    results = []
    for cfg, color in zip(portfolio_cfgs, COLORS):
        vals_usd, invested_local, reserve_usd = simulate_portfolio(
            prices, cfg["weights"], initial_inv, monthly_contrib,
            rebalance_annually=rebalance_annually,
            ddca_thresholds=cfg.get("ddca_thresholds") or None,
            risk_free_rate=rf_series,
            fx_rate=fx_series,
        )
        # Convert USD portfolio values to local currency
        if fx_series is not None:
            fx_al = fx_series.reindex(vals_usd.index, method="ffill").bfill()
            vals    = vals_usd    / fx_al
            reserve = reserve_usd / fx_al
        else:
            vals    = vals_usd
            reserve = reserve_usd
        invested = invested_local
        rets = returns_from_simulation(vals, invested)
        metrics = calculate_metrics(vals, invested, rets, rf_series)
        metrics["Long Exposure"] = sum(w for w in cfg["weights"].values() if w > 0)
        metrics["Borrowed"]      = abs(sum(w for w in cfg["weights"].values() if w < 0))
        ann = annual_returns_table(rets)
        dd  = drawdown_series(rets)
        results.append({
            "name":     cfg["name"],
            "weights":  cfg["weights"],
            "ddca":     cfg.get("ddca_thresholds") or {},
            "color":    color,
            "values":   vals,
            "invested": invested,
            "reserve":  reserve,
            "returns":  rets,
            "metrics":  metrics,
            "annual":   ann,
            "drawdown": dd,
        })

    corr = correlation_matrix(prices, portfolio_cfgs)

    # ================================================================ Tabs
    tab_overview, tab_risk, tab_assets, tab_metrics, tab_periods = st.tabs(
        ["Overview", "Risk & Drawdown", "Assets & Correlation", "Full Metrics", "Period Analysis"]
    )

    # ---------------------------------------------------------------- Overview
    with tab_overview:
        # KPI cards — one row per portfolio
        kpi_keys = ["CAGR", "Sharpe Ratio", "Max Drawdown", "Final Value ($)"]
        for r in results:
            st.markdown(f"**{r['name']}**")
            kpi_cols = st.columns(len(kpi_keys))
            for col, k in zip(kpi_cols, kpi_keys):
                col.metric(label=k, value=fmt(k, r["metrics"][k]))

        st.divider()

        # Portfolio value chart
        fig_val = go.Figure()
        for r in results:
            fig_val.add_trace(go.Scatter(
                x=r["values"].index, y=r["values"],
                name=r["name"], line=dict(color=r["color"], width=2),
            ))
            if r["reserve"].max() > 0:
                fig_val.add_trace(go.Scatter(
                    x=r["reserve"].index, y=r["reserve"],
                    name=f"{r['name']} — DDCA reserve",
                    line=dict(color=r["color"], width=1, dash="dot"),
                    opacity=0.6,
                ))
        # Total invested (same for all)
        fig_val.add_trace(go.Scatter(
            x=results[0]["invested"].index, y=results[0]["invested"],
            name="Total Invested", line=dict(color="gray", width=1.5, dash="dash"),
            opacity=0.7,
        ))
        fig_val.update_layout(
            title=dict(
                text=(
                    f"Portfolio Value  |  Initial: {dollar(initial_inv)}"
                    f" + {dollar(monthly_contrib)}/mo"
                ),
                font=dict(size=15),
            ),
            xaxis_title="Date",
            yaxis_title=f"Value ({currency})",
            yaxis_tickprefix=ccy_sym,
            yaxis_tickformat=",.0f",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=430,
        )
        st.plotly_chart(fig_val, use_container_width=True)

        # Annual returns — one subplot per portfolio
        st.subheader("Annual Returns")
        n_p = len(results)
        fig_ann = make_subplots(rows=1, cols=n_p, subplot_titles=[r["name"] for r in results])
        for col_idx, r in enumerate(results, 1):
            ann = r["annual"] * 100
            bar_colors = [r["color"] if v >= 0 else "#E53935" for v in ann]
            fig_ann.add_trace(
                go.Bar(
                    x=ann.index.year.astype(str),
                    y=ann.values,
                    marker_color=bar_colors,
                    name=r["name"],
                    showlegend=False,
                    text=[f"{v:.1f}%" for v in ann.values],
                    textposition="outside",
                ),
                row=1, col=col_idx,
            )
            fig_ann.update_yaxes(ticksuffix="%", row=1, col=col_idx)
        fig_ann.update_layout(height=350)
        st.plotly_chart(fig_ann, use_container_width=True)

    # ---------------------------------------------------------------- Risk
    with tab_risk:
        # Drawdown chart
        fig_dd = go.Figure()
        for r in results:
            dd_pct = r["drawdown"] * 100
            fig_dd.add_trace(go.Scatter(
                x=dd_pct.index, y=dd_pct,
                name=r["name"],
                fill="tozeroy",
                line=dict(color=r["color"], width=1.5),
                opacity=0.6,
            ))
        fig_dd.update_layout(
            title="Drawdown from Peak",
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            yaxis_ticksuffix="%",
            hovermode="x unified",
            height=380,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_dd, use_container_width=True)

        # Return distribution
        st.subheader("Daily Return Distribution")
        fig_dist = go.Figure()
        for r in results:
            fig_dist.add_trace(go.Histogram(
                x=r["returns"] * 100,
                name=r["name"],
                opacity=0.65,
                nbinsx=80,
                marker_color=r["color"],
            ))
        fig_dist.update_layout(
            barmode="overlay",
            xaxis_title="Daily Return (%)",
            yaxis_title="Count",
            xaxis_ticksuffix="%",
            height=330,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_dist, use_container_width=True)

        # Rolling 12-month Sharpe
        st.subheader("Rolling 12-Month Sharpe Ratio")
        fig_sharpe = go.Figure()
        daily_rf_s = (1 + rf_series.reindex(results[0]["returns"].index, method="ffill").bfill()) ** (1 / 252) - 1
        for r in results:
            excess = r["returns"] - daily_rf_s
            rolling_sharpe = (
                excess.rolling(252).mean()
                .div(r["returns"].rolling(252).std())
                * np.sqrt(252)
            )
            fig_sharpe.add_trace(go.Scatter(
                x=rolling_sharpe.index, y=rolling_sharpe,
                name=r["name"], line=dict(color=r["color"], width=2),
            ))
        fig_sharpe.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig_sharpe.update_layout(
            xaxis_title="Date", yaxis_title="Sharpe Ratio",
            hovermode="x unified", height=330,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_sharpe, use_container_width=True)

    # ---------------------------------------------------------------- Assets
    with tab_assets:
        # Correlation heatmap
        st.subheader("Asset Correlation Matrix (daily returns)")
        fig_corr = px.imshow(
            corr,
            color_continuous_scale="RdYlGn",
            zmin=-1, zmax=1,
            text_auto=".2f",
            aspect="auto",
        )
        fig_corr.update_layout(height=400 + len(corr) * 20)
        st.plotly_chart(fig_corr, use_container_width=True)

        # Individual asset performance
        st.subheader("Individual Asset Returns (normalised to 100 at start)")
        fig_assets = go.Figure()
        for ticker in corr.columns:
            norm = prices[ticker] / prices[ticker].iloc[0] * 100
            fig_assets.add_trace(go.Scatter(
                x=norm.index, y=norm,
                name=ticker, mode="lines", line=dict(width=1.5),
            ))
        fig_assets.add_hline(y=100, line_dash="dot", line_color="gray", opacity=0.4)
        fig_assets.update_layout(
            xaxis_title="Date",
            yaxis_title="Normalised Value",
            hovermode="x unified",
            height=380,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_assets, use_container_width=True)

        # Portfolio weight charts
        # Use a bar chart so negative (short) weights display correctly.
        st.subheader("Portfolio Weights")
        for col, r in zip(st.columns(len(results)), results):
            with col:
                tickers = list(r["weights"].keys())
                wvals   = list(r["weights"].values())
                bar_colors = [r["color"] if w >= 0 else "#E53935" for w in wvals]
                fig_w = go.Figure(go.Bar(
                    x=tickers,
                    y=wvals,
                    marker_color=bar_colors,
                    text=[f"{w:.1%}" for w in wvals],
                    textposition="outside",
                ))
                fig_w.add_hline(y=0, line_color="gray", line_width=0.8)
                fig_w.update_layout(
                    title=r["name"],
                    yaxis_tickformat=".0%",
                    height=300,
                    margin=dict(t=40, b=10, l=10, r=10),
                )
                st.plotly_chart(fig_w, use_container_width=True)

    # ---------------------------------------------------------------- Metrics table
    with tab_metrics:
        metric_order = [
            "Long Exposure",
            "Borrowed",
            "CAGR",
            "Cumulative Return (TWR)",
            "Total P&L ($)",
            "Total P&L (%)",
            "Final Value ($)",
            "Total Invested ($)",
            "Annual Volatility",
            "Sharpe Ratio",
            "Sortino Ratio",
            "Calmar Ratio",
            "Max Drawdown",
            "Max DD Duration (days)",
            "Best Year",
            "Worst Year",
            "Win Rate (daily)",
            "Years Simulated",
        ]

        # Rename ($) labels to reflect the actual display currency
        label_map = {
            "Total P&L ($)":    f"Total P&L ({currency})",
            "Final Value ($)":  f"Final Value ({currency})",
            "Total Invested ($)": f"Total Invested ({currency})",
        }

        table_rows = {}
        for key in metric_order:
            display_key = label_map.get(key, key)
            table_rows[display_key] = {r["name"]: fmt(key, r["metrics"].get(key, float("nan"))) for r in results}

        df_metrics = pd.DataFrame(table_rows).T
        df_metrics.index.name = "Metric"

        # Highlight better value (green for higher is better, red for lower)
        higher_is_better = {
            "CAGR", "Cumulative Return (TWR)",
            f"Total P&L ({currency})", "Total P&L (%)",
            f"Final Value ({currency})", "Sharpe Ratio", "Sortino Ratio", "Calmar Ratio",
            "Best Year", "Win Rate (daily)",
        }
        lower_is_better = {"Annual Volatility", "Max Drawdown", "Max DD Duration (days)", "Worst Year"}

        def highlight(row):
            if len(results) < 2:
                return [""] * len(row)
            name_list = [r["name"] for r in results]
            key = row.name
            raw = [r["metrics"].get(key, float("nan")) for r in results]

            if any(np.isnan(v) if isinstance(v, float) else False for v in raw):
                return [""] * len(row)

            styles = [""] * len(row)
            if key in higher_is_better:
                best_idx = int(np.argmax(raw))
                styles[best_idx] = "background-color: #C8E6C9; font-weight: bold"
            elif key in lower_is_better:
                best_idx = int(np.argmin(raw))
                styles[best_idx] = "background-color: #C8E6C9; font-weight: bold"
            return styles

        styled = df_metrics.style.apply(highlight, axis=1)
        st.dataframe(styled, use_container_width=True, height=620)

        # Download button
        csv = df_metrics.to_csv()
        st.download_button(
            label="Download metrics as CSV",
            data=csv,
            file_name="portfolio_metrics.csv",
            mime="text/csv",
        )

    # ---------------------------------------------------------------- Period Analysis
    with tab_periods:
        st.subheader(f"Rolling {rolling_window_years}-Year Period Analysis")
        st.caption(
            f"Each portfolio is simulated across every possible {rolling_window_years}-year "
            f"window (monthly start dates) using the same DCA settings. "
            f"Periods are ranked by **{rolling_rank_by}** to identify the worst, median, and best starting points."
        )

        for r in results:
            st.markdown(f"#### {r['name']}")
            try:
                with st.spinner(f"Analysing {r['name']}…"):
                    rwa = rolling_window_analysis(
                        prices, r["weights"], initial_inv, monthly_contrib,
                        window_years=rolling_window_years,
                        rank_by=rolling_rank_by,
                        rebalance_annually=rebalance_annually,
                        ddca_thresholds=r.get("ddca") or None,
                        risk_free_rate=rf_series,
                        fx_rate=fx_series,
                    )
            except ValueError as e:
                st.warning(str(e))
                continue

            summary = rolling_window_summary(rwa, rank_by=rolling_rank_by)

            # KPI cards for worst / median / best
            period_cols = st.columns(3)
            for col, (period_key, period_label, period_color) in zip(
                period_cols,
                [("worst", "Worst", "#E53935"), ("median", "Median", "#FFC107"), ("best", "Best", "#4CAF50")],
            ):
                s = summary[period_key]
                start_str = s["start_date"].strftime("%Y-%m") if hasattr(s["start_date"], "strftime") else str(s["start_date"])
                end_str = s["end_date"].strftime("%Y-%m") if hasattr(s["end_date"], "strftime") else str(s["end_date"])
                with col:
                    st.markdown(
                        f'<div style="border-left: 4px solid {period_color}; padding-left: 12px;">'
                        f"<strong>{period_label}</strong> &nbsp; {start_str} → {end_str}</div>",
                        unsafe_allow_html=True,
                    )
                    sub_cols = st.columns(3)
                    sub_cols[0].metric("CAGR", pct(s["CAGR"]))
                    sub_cols[1].metric("Sharpe", f"{s['Sharpe Ratio']:.3f}")
                    sub_cols[2].metric("Max DD", pct(s["Max Drawdown"]))

            # Distribution histogram
            is_pct_metric = rolling_rank_by in (
                "CAGR", "Max Drawdown", "Annual Volatility", "Cumulative Return (TWR)",
            )
            metric_vals = rwa[rolling_rank_by]
            plot_vals = metric_vals * 100 if is_pct_metric else metric_vals
            suffix = "%" if is_pct_metric else ""

            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(
                x=plot_vals, nbinsx=30,
                marker_color=r["color"], opacity=0.75,
                name=rolling_rank_by,
            ))
            for period_key, period_label, line_color, dash in [
                ("worst",  "Worst",  "#E53935", "dash"),
                ("median", "Median", "#FFC107", "solid"),
                ("best",   "Best",   "#4CAF50", "dash"),
            ]:
                val = summary[period_key][rolling_rank_by]
                plot_v = val * 100 if is_pct_metric else val
                fig_hist.add_vline(
                    x=plot_v, line_color=line_color, line_dash=dash, line_width=2,
                    annotation_text=f"{period_label}: {plot_v:.2f}{suffix}",
                    annotation_position="top",
                )
            fig_hist.update_layout(
                title=f"{rolling_rank_by} across {len(rwa)} rolling {rolling_window_years}-year windows",
                xaxis_title=f"{rolling_rank_by} ({suffix.strip()})" if suffix else rolling_rank_by,
                yaxis_title="Number of windows",
                height=350,
                showlegend=False,
            )
            st.plotly_chart(fig_hist, use_container_width=True)

            # Full results table (collapsed)
            with st.expander(f"All {len(rwa)} windows — sorted by {rolling_rank_by}"):
                display_df = rwa.copy()
                display_df["start_date"] = display_df["start_date"].dt.strftime("%Y-%m-%d")
                display_df["end_date"] = display_df["end_date"].dt.strftime("%Y-%m-%d")
                for c in ["CAGR", "Max Drawdown", "Annual Volatility", "Cumulative Return (TWR)"]:
                    if c in display_df.columns:
                        display_df[c] = display_df[c].map(lambda v: f"{v:.2%}")
                for c in ["Sharpe Ratio", "Sortino Ratio"]:
                    if c in display_df.columns:
                        display_df[c] = display_df[c].map(lambda v: f"{v:.3f}")
                for c in ["Final Value ($)", "Total Invested ($)"]:
                    if c in display_df.columns:
                        display_df[c] = display_df[c].map(lambda v: f"{ccy_sym}{v:,.0f}")
                st.dataframe(display_df, use_container_width=True, height=400)

                csv_periods = rwa.to_csv(index=False)
                st.download_button(
                    label=f"Download {r['name']} period data as CSV",
                    data=csv_periods,
                    file_name=f"rolling_periods_{r['name'].replace(' ', '_')}.csv",
                    mime="text/csv",
                    key=f"csv_periods_{r['name']}",
                )

            st.divider()

else:
    st.info(
        "Configure your two portfolios in the sidebar, then click **Run Simulation**.\n\n"
        "- Use the editable tables to add/remove tickers and set weights (they will be normalised to 1.0).\n"
        "- Tickers are Yahoo Finance symbols (e.g. `SPY`, `BTC-USD`, `AAPL`)."
    )
