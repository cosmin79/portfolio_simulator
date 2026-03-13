"""
run_portfolio.py - CLI / script interface

Edit the CONFIGURATION section below, then run:
    python run_portfolio.py

Opens four separate figure windows (use your window manager / alt-tab to switch):
  Figure 1 — Portfolio value over time + annual returns
  Figure 2 — Drawdown + return distribution + rolling Sharpe
  Figure 3 — Correlation heatmap + risk/return bar + asset weights
  Figure 4 — Full metrics table
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import seaborn as sns
import pandas as pd
import numpy as np

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
    rolling_window_analysis,
    rolling_window_summary,
)

# ===========================================================================
# CONFIGURATION — edit this section
# ===========================================================================

# Pick two portfolios from the PRESET_PORTFOLIOS dict below (or define your own).
# Each entry needs "name" and "weights". Optional: "ddca_thresholds".
#
# Example with DDCA:
# {
#     "name": "All weather portfolio",
#     "weights": {"VT": 0.603, "BND": 0.402, "BIL": -0.335, "GCC": 0.165, "GLD": 0.165},
#     "ddca_thresholds": {"VT": 0.10, "GCC": 0.10, "GLD": 0.10},
# }

PRESET_PORTFOLIOS = {
    # --- Custom / leveraged ---
    "All Weather CTA":       {"VT": 0.603, "AGG": 0.402, "LQD": -0.335, "AMFAX": 0.330},
    "All Weather Commodity":  {"VT": 0.603, "AGG": 0.402, "LQD": -0.335, "GLD": 0.165, "GCC": 0.165},
    # --- Classic allocations (source: portfoliocharts.com) ---
    "Total Stock Market":     {"VTI": 1.0},
    "Classic 60-40":          {"VTI": 0.60, "AGG": 0.40},
    "Three-Fund":             {"VTI": 0.64, "EFA": 0.16, "AGG": 0.20},
    "No-Brainer":             {"VTI": 0.25, "VB": 0.25, "EFA": 0.25, "SHY": 0.25},
    "Core Four":              {"VTI": 0.48, "EFA": 0.24, "AGG": 0.20, "VNQ": 0.08},
    "Permanent":              {"VTI": 0.25, "TLT": 0.25, "BIL": 0.25, "GLD": 0.25},
    "Larry":                  {"VBR": 0.15, "DLS": 0.075, "VWO": 0.075, "AGG": 0.70},
    "Golden Butterfly":       {"VTI": 0.20, "VBR": 0.20, "TLT": 0.20, "SHY": 0.20, "GLD": 0.20},
    "All Seasons":            {"VTI": 0.30, "TLT": 0.40, "AGG": 0.15, "GSG": 0.075, "GLD": 0.075},
    "Ivy Portfolio":          {"VTI": 0.20, "EFA": 0.20, "AGG": 0.20, "GSG": 0.20, "VNQ": 0.20},
    "Swensen":                {"VTI": 0.30, "EFA": 0.15, "VWO": 0.05, "AGG": 0.30, "VNQ": 0.20},
    "Weird Portfolio":        {"VBR": 0.20, "SCZ": 0.20, "TLT": 0.20, "VNQ": 0.20, "GLD": 0.20},
}

PORTFOLIOS = [
    {"name": "Golden Butterfly", "weights": PRESET_PORTFOLIOS["Golden Butterfly"]},
    {"name": "All Seasons",      "weights": PRESET_PORTFOLIOS["All Seasons"]},
]

START_YEAR           = 2021     # simulation start (data availability may push this later)
CURRENCY             = "USD"    # investment currency: "USD", "GBP", "EUR", or "CHF"
INITIAL_INVESTMENT   = 100_000  # lump-sum at start (in CURRENCY)
MONTHLY_CONTRIBUTION = 0        # added every month (in CURRENCY)
REBALANCE_ANNUALLY   = False    # rebalance to target weights each January
ROLLING_WINDOW_YEARS = 5        # rolling window length for worst/median/best period analysis (0 = skip)
ROLLING_RANK_BY      = "CAGR"   # metric to rank periods: "CAGR", "Sharpe Ratio", "Max Drawdown", etc.

# ===========================================================================
# — nothing below here needs changing —
# ===========================================================================

COLORS = ["#2196F3", "#FF5722"]   # blue, deep-orange


def fmt_pct(v: float) -> str:
    return f"{v * 100:.2f}%"


def fmt_dollar(v: float) -> str:
    return f"${v:,.0f}"


def fmt_metric(key: str, v) -> str:
    if isinstance(v, float) and np.isnan(v):
        return "N/A"
    if key in ("CAGR", "Cumulative Return (TWR)", "Total P&L (%)",
               "Annual Volatility", "Max Drawdown",
               "Best Year", "Worst Year", "Win Rate (daily)"):
        return fmt_pct(v)
    if key in ("Total P&L ($)", "Final Value ($)", "Total Invested ($)"):
        return fmt_dollar(v)
    if key in ("Sharpe Ratio", "Sortino Ratio", "Calmar Ratio"):
        return f"{v:.3f}"
    if key in ("Long Exposure", "Borrowed"):
        return fmt_pct(v)
    if key == "Max DD Duration (days)":
        return f"{int(v):,} days"
    if key == "Years Simulated":
        return f"{v:.1f} yrs"
    return str(v)


def validate_weights(portfolios: list[dict]) -> None:
    for p in portfolios:
        total = sum(p["weights"].values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Portfolio '{p['name']}' weights sum to {total:.4f}, must be 1.0"
            )


def run() -> None:
    validate_weights(PORTFOLIOS)

    # ------------------------------------------------------------------ data
    all_tickers = list(
        dict.fromkeys(t for p in PORTFOLIOS for t in p["weights"])
    )
    print(f"Fetching data for: {all_tickers}  (from {START_YEAR})")
    import warnings as _warnings
    with _warnings.catch_warnings(record=True) as _caught:
        _warnings.simplefilter("always")
        prices = fetch_prices(all_tickers, START_YEAR)
    for w in _caught:
        print(f"WARNING: {w.message}")
    actual_start = prices.index[0].date()
    actual_end   = prices.index[-1].date()
    print(f"Data range: {actual_start} → {actual_end}  ({len(prices)} trading days)")

    rf_series = fetch_risk_free_rate(START_YEAR)
    rf_mean = rf_series.reindex(prices.index, method="ffill").bfill().mean()
    print(f"Risk-free rate: {rf_mean:.2%} avg (^IRX 3-month T-bill)")

    fx_series = fetch_fx_rate(CURRENCY, START_YEAR)
    if fx_series is not None:
        fx_mean = fx_series.reindex(prices.index, method="ffill").bfill().mean()
        print(f"Currency: {CURRENCY}  (avg {CURRENCY}/USD: {fx_mean:.4f})")

    # ------------------------------------------------------------ simulation
    results = []
    for p in PORTFOLIOS:
        vals_usd, invested_local, reserve_usd = simulate_portfolio(
            prices, p["weights"], INITIAL_INVESTMENT, MONTHLY_CONTRIBUTION,
            rebalance_annually=REBALANCE_ANNUALLY,
            ddca_thresholds=p.get("ddca_thresholds"),
            risk_free_rate=rf_series,
            fx_rate=fx_series,
        )
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
        metrics["Long Exposure"] = sum(w for w in p["weights"].values() if w > 0)
        metrics["Borrowed"]      = abs(sum(w for w in p["weights"].values() if w < 0))
        annual_rets = annual_returns_table(rets)
        dd = drawdown_series(rets)
        results.append({
            "name":        p["name"],
            "weights":     p["weights"],
            "ddca":        p.get("ddca_thresholds") or {},
            "values":      vals,
            "invested":    invested,
            "reserve":     reserve,
            "returns":     rets,
            "metrics":     metrics,
            "annual_rets": annual_rets,
            "drawdown":    dd,
        })

    corr = correlation_matrix(prices, PORTFOLIOS)

    sns.set_theme(style="darkgrid", font_scale=0.95)
    rebal_str = "annual rebalance" if REBALANCE_ANNUALLY else "no rebalance"
    subtitle = (
        f"Initial: {fmt_dollar(INITIAL_INVESTMENT)} + {fmt_dollar(MONTHLY_CONTRIBUTION)}/mo  |  "
        f"{rebal_str}  |  {actual_start} → {actual_end}"
    )

    # ================================================================ Fig 1
    # Portfolio value + annual returns
    # ================================================================
    fig1, axes1 = plt.subplots(
        3, 1,
        figsize=(14, 13),
        gridspec_kw={"height_ratios": [3, 1.6, 1.6]},
        constrained_layout=True,
    )
    fig1.suptitle(f"Portfolio Comparison — Overview\n{subtitle}", fontsize=12, fontweight="bold")

    ax_val, ax_ann0, ax_ann1 = axes1

    # Portfolio value
    for r, color in zip(results, COLORS):
        ax_val.plot(r["values"].index, r["values"], label=r["name"], color=color, lw=2)
        if r["reserve"].max() > 0:
            ax_val.fill_between(
                r["reserve"].index, 0, r["reserve"],
                alpha=0.12, color=color, label=f"{r['name']} — reserve",
            )
    ax_val.plot(
        results[0]["invested"].index, results[0]["invested"],
        label="Total Invested", color="gray", lw=1.3, linestyle="--", alpha=0.75,
    )
    ax_val.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax_val.set_ylabel("Value (USD)")
    ax_val.set_title("Portfolio Value Over Time  (shaded band = DDCA cash reserve)")
    ax_val.legend()

    # Annual returns per portfolio
    for ax_ann, r, color in zip([ax_ann0, ax_ann1], results, COLORS):
        ann = r["annual_rets"] * 100
        bar_colors = [color if v >= 0 else "#E53935" for v in ann]
        bars = ax_ann.bar(ann.index.year.astype(str), ann.values, color=bar_colors, alpha=0.85)
        ax_ann.axhline(0, color="black", lw=0.7)
        ax_ann.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
        ax_ann.set_title(f"{r['name']} — Annual Returns")
        ax_ann.set_ylabel("Return (%)")
        for bar, val in zip(bars, ann.values):
            ax_ann.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (0.4 if val >= 0 else -1.5),
                f"{val:.1f}%",
                ha="center", va="bottom", fontsize=7,
            )

    # ================================================================ Fig 2
    # Drawdown + return distribution + rolling Sharpe
    # ================================================================
    fig2, (ax_dd, ax_dist, ax_rs) = plt.subplots(
        3, 1,
        figsize=(14, 13),
        constrained_layout=True,
    )
    fig2.suptitle(f"Portfolio Comparison — Risk\n{subtitle}", fontsize=12, fontweight="bold")

    # Drawdown
    for r, color in zip(results, COLORS):
        dd_pct = r["drawdown"] * 100
        ax_dd.fill_between(dd_pct.index, dd_pct, 0, alpha=0.35, color=color, label=r["name"])
        ax_dd.plot(dd_pct.index, dd_pct, color=color, lw=0.9)
    ax_dd.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax_dd.set_title("Drawdown from Peak")
    ax_dd.set_ylabel("Drawdown (%)")
    ax_dd.legend()

    # Daily return distribution
    for r, color in zip(results, COLORS):
        ax_dist.hist(r["returns"] * 100, bins=80, alpha=0.55, color=color, label=r["name"])
    ax_dist.set_title("Daily Return Distribution")
    ax_dist.set_xlabel("Daily Return (%)")
    ax_dist.set_ylabel("Count")
    ax_dist.legend()

    # Rolling 12-month Sharpe
    ref_returns = results[0]["returns"]
    daily_rf_s = (1 + rf_series.reindex(ref_returns.index, method="ffill").bfill()) ** (1 / 252) - 1
    for r, color in zip(results, COLORS):
        excess = r["returns"] - daily_rf_s
        rs = (
            excess.rolling(252).mean()
            .div(r["returns"].rolling(252).std())
            * np.sqrt(252)
        )
        ax_rs.plot(rs.index, rs, label=r["name"], color=color, lw=1.5)
    ax_rs.axhline(0, color="gray", lw=0.8, linestyle="--")
    ax_rs.set_title("Rolling 12-Month Sharpe Ratio")
    ax_rs.set_ylabel("Sharpe Ratio")
    ax_rs.legend()

    # ================================================================ Fig 3
    # Correlation heatmap + risk/return bars + weight pies
    # ================================================================
    fig3 = plt.figure(figsize=(14, 11), constrained_layout=True)
    fig3.suptitle(f"Portfolio Comparison — Assets\n{subtitle}", fontsize=12, fontweight="bold")
    gs3 = gridspec.GridSpec(2, 2, figure=fig3)
    ax_corr  = fig3.add_subplot(gs3[0, :])   # full width
    ax_bar   = fig3.add_subplot(gs3[1, 0])
    ax_pies  = fig3.add_subplot(gs3[1, 1])
    ax_pies.axis("off")

    # Correlation heatmap
    sns.heatmap(
        corr,
        ax=ax_corr,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        vmin=-1, vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.6},
    )
    ax_corr.set_title("Asset Correlation (daily returns)")

    # Key metrics bar chart
    metric_keys = ["CAGR", "Annual Volatility", "Max Drawdown", "Sharpe Ratio", "Sortino Ratio"]
    x = np.arange(len(metric_keys))
    bar_width = 0.35
    for i, (r, color) in enumerate(zip(results, COLORS)):
        vals_list = [abs(r["metrics"][k]) for k in metric_keys]
        bars = ax_bar.bar(x + i * bar_width, [v * 100 if "Ratio" not in k else v for v, k in zip(vals_list, metric_keys)],
                          bar_width, label=r["name"], color=color, alpha=0.85)
        for bar in bars:
            ax_bar.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.2,
                f"{bar.get_height():.2f}",
                ha="center", va="bottom", fontsize=7,
            )
    ax_bar.set_xticks(x + bar_width / 2)
    ax_bar.set_xticklabels(["CAGR %", "Volatility %", "|MaxDD| %", "Sharpe", "Sortino"], fontsize=8)
    ax_bar.set_title("Risk / Return Metrics")
    ax_bar.legend(fontsize=8)

    # Weight bar charts (bars handle negative weights; pies cannot)
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    w_ax_list = [
        inset_axes(ax_pies, width="45%", height="85%", loc="center left",
                   bbox_to_anchor=ax_pies.get_position(), bbox_transform=fig3.transFigure),
        inset_axes(ax_pies, width="45%", height="85%", loc="center right",
                   bbox_to_anchor=ax_pies.get_position(), bbox_transform=fig3.transFigure),
    ]
    for w_ax, r, color in zip(w_ax_list, results, COLORS):
        tickers = list(r["weights"].keys())
        wvals   = list(r["weights"].values())
        bar_colors = [color if w >= 0 else "#E53935" for w in wvals]
        w_ax.bar(tickers, wvals, color=bar_colors, alpha=0.85)
        w_ax.axhline(0, color="black", lw=0.7)
        w_ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0%}"))
        w_ax.set_title(r["name"], fontsize=9)
        w_ax.tick_params(axis="x", labelsize=7)

    # ================================================================ Fig 4
    # Full metrics table — given its own figure so it can be maximised
    # ================================================================
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
    col_labels = ["Metric"] + [r["name"] for r in results]
    table_data = []
    for key in metric_order:
        row = [key]
        for r in results:
            row.append(fmt_metric(key, r["metrics"].get(key, float("nan"))))
        table_data.append(row)

    n_rows = len(table_data)
    fig4_h = max(6, 0.42 * (n_rows + 1))
    fig4, ax_tbl = plt.subplots(figsize=(11, fig4_h), constrained_layout=True)
    fig4.suptitle("Portfolio Comparison — Full Metrics", fontsize=12, fontweight="bold")
    ax_tbl.axis("off")

    tbl = ax_tbl.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.auto_set_column_width(list(range(len(col_labels))))

    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#37474F")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    for i in range(1, n_rows + 1):
        bg = "#FAFAFA" if i % 2 == 0 else "#ECEFF1"
        for j in range(len(col_labels)):
            tbl[i, j].set_facecolor(bg)

    # ================================================================ Fig 5
    # Rolling window analysis — worst / median / best periods
    # ================================================================
    figs_to_save = [
        (fig1, "portfolio_overview.png"),
        (fig2, "portfolio_risk.png"),
        (fig3, "portfolio_assets.png"),
        (fig4, "portfolio_metrics.png"),
    ]

    if ROLLING_WINDOW_YEARS > 0:
        print(f"\nRolling {ROLLING_WINDOW_YEARS}-year window analysis (rank by {ROLLING_RANK_BY})…")
        n_portfolios = len(results)
        fig5, axes5 = plt.subplots(
            n_portfolios, 2,
            figsize=(16, 5 * n_portfolios),
            constrained_layout=True,
            squeeze=False,
        )
        fig5.suptitle(
            f"Rolling {ROLLING_WINDOW_YEARS}-Year Period Analysis — "
            f"Ranked by {ROLLING_RANK_BY}\n{subtitle}",
            fontsize=12, fontweight="bold",
        )

        for row_idx, (r, color) in enumerate(zip(results, COLORS)):
            try:
                rwa = rolling_window_analysis(
                    prices, r["weights"], INITIAL_INVESTMENT, MONTHLY_CONTRIBUTION,
                    window_years=ROLLING_WINDOW_YEARS,
                    rank_by=ROLLING_RANK_BY,
                    rebalance_annually=REBALANCE_ANNUALLY,
                    ddca_thresholds=r.get("ddca") or None,
                    risk_free_rate=rf_series,
                    fx_rate=fx_series,
                )
            except ValueError as e:
                axes5[row_idx, 0].text(0.5, 0.5, str(e), ha="center", va="center", fontsize=10)
                axes5[row_idx, 0].set_title(r["name"])
                axes5[row_idx, 1].axis("off")
                continue

            summary = rolling_window_summary(rwa, rank_by=ROLLING_RANK_BY)

            # Left panel: histogram of the rank metric across all windows
            ax_hist = axes5[row_idx, 0]
            metric_vals = rwa[ROLLING_RANK_BY]
            is_pct_metric = ROLLING_RANK_BY in (
                "CAGR", "Max Drawdown", "Annual Volatility", "Cumulative Return (TWR)",
            )
            plot_vals = metric_vals * 100 if is_pct_metric else metric_vals
            ax_hist.hist(plot_vals, bins=25, color=color, alpha=0.75, edgecolor="white")
            suffix = "%" if is_pct_metric else ""

            # Mark worst, median, best
            for label, style, key in [
                ("Worst",  {"color": "#E53935", "ls": "--", "lw": 2}, "worst"),
                ("Median", {"color": "#FFC107", "ls": "-",  "lw": 2}, "median"),
                ("Best",   {"color": "#4CAF50", "ls": "--", "lw": 2}, "best"),
            ]:
                val = summary[key][ROLLING_RANK_BY]
                plot_v = val * 100 if is_pct_metric else val
                ax_hist.axvline(plot_v, **style, label=f"{label}: {plot_v:.2f}{suffix}")

            ax_hist.set_title(f"{r['name']} — {ROLLING_RANK_BY} Distribution ({len(rwa)} windows)")
            ax_hist.set_xlabel(f"{ROLLING_RANK_BY} ({suffix.strip()})" if suffix else ROLLING_RANK_BY)
            ax_hist.set_ylabel("Count")
            ax_hist.legend(fontsize=8)

            # Right panel: summary table
            ax_tbl = axes5[row_idx, 1]
            ax_tbl.axis("off")
            summary_keys = [
                "CAGR", "Sharpe Ratio", "Max Drawdown",
                "Annual Volatility", "Cumulative Return (TWR)",
                "Final Value ($)", "Total Invested ($)",
            ]
            col_labels = ["Metric", "Worst", "Median", "Best"]
            tbl_data = []
            for mk in summary_keys:
                row_data = [mk]
                for period in ("worst", "median", "best"):
                    row_data.append(fmt_metric(mk, summary[period].get(mk, float("nan"))))
                tbl_data.append(row_data)
            # Add start/end dates
            for date_key in ("start_date", "end_date"):
                label = "Start" if "start" in date_key else "End"
                row_data = [label]
                for period in ("worst", "median", "best"):
                    d = summary[period][date_key]
                    row_data.append(d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d))
                tbl_data.append(row_data)

            tbl = ax_tbl.table(
                cellText=tbl_data,
                colLabels=col_labels,
                cellLoc="center",
                loc="center",
                bbox=[0, 0, 1, 1],
            )
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(9)
            tbl.auto_set_column_width(list(range(len(col_labels))))
            for j in range(len(col_labels)):
                tbl[0, j].set_facecolor("#37474F")
                tbl[0, j].set_text_props(color="white", fontweight="bold")
            for i in range(1, len(tbl_data) + 1):
                bg = "#FAFAFA" if i % 2 == 0 else "#ECEFF1"
                for j in range(len(col_labels)):
                    tbl[i, j].set_facecolor(bg)
            ax_tbl.set_title(f"{r['name']} — Worst / Median / Best {ROLLING_WINDOW_YEARS}yr Period")

            # Print summary to console
            print(f"\n  {r['name']}:")
            for period in ("worst", "median", "best"):
                s = summary[period]
                start_str = s["start_date"].strftime("%Y-%m-%d") if hasattr(s["start_date"], "strftime") else s["start_date"]
                end_str = s["end_date"].strftime("%Y-%m-%d") if hasattr(s["end_date"], "strftime") else s["end_date"]
                print(f"    {period.capitalize():>6}: {start_str} → {end_str}  "
                      f"CAGR={s['CAGR']:.2%}  Sharpe={s['Sharpe Ratio']:.3f}  MaxDD={s['Max Drawdown']:.2%}")

        figs_to_save.append((fig5, "portfolio_periods.png"))

    # ================================================================ Save & show
    for fig, fname in figs_to_save:
        fig.savefig(fname, dpi=150, bbox_inches="tight")
    saved = ", ".join(f for _, f in figs_to_save)
    print(f"Saved → {saved}")
    plt.show()


if __name__ == "__main__":
    run()
