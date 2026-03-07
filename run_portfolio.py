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
    simulate_portfolio,
    portfolio_daily_returns,
    returns_from_simulation,
    calculate_metrics,
    correlation_matrix,
    drawdown_series,
    annual_returns_table,
)

# ===========================================================================
# CONFIGURATION — edit this section
# ===========================================================================

PORTFOLIOS = [
    {
        "name": "All weather portfolio",
        "weights": {
            "VT":  0.603,
            "BND": 0.402,
            "BIL": -0.335,
            "GCC":  0.165,
            "GLD":  0.165,
        },
        # Optional: per-ticker Double DCA thresholds.
        # Tickers listed here use DDCA; others use regular DCA.
        # Value = how far below 52-week high triggers the double-down (e.g. 0.10 = 10%).
        "ddca_thresholds": {
            "VT":  0.10,
            "GCC": 0.15,
            "GLD": 0.10,
        },
    },
    {
        "name": "All world",
        "weights": {
            "VT":  1.0,
        },
        # No ddca_thresholds key → plain DCA for all tickers
    },
]

START_YEAR           = 2007     # simulation start (data availability may push this later)
INITIAL_INVESTMENT   = 100_000  # USD lump-sum at start
MONTHLY_CONTRIBUTION = 2000     # USD added every month
RISK_FREE_RATE       = 0.04    # annual, e.g. 0.04 = 4 %
REBALANCE_ANNUALLY   = False    # rebalance to target weights each January

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
    prices = fetch_prices(all_tickers, START_YEAR)
    actual_start = prices.index[0].date()
    actual_end   = prices.index[-1].date()
    print(f"Data range: {actual_start} → {actual_end}  ({len(prices)} trading days)")

    # ------------------------------------------------------------ simulation
    results = []
    for p in PORTFOLIOS:
        vals, invested, reserve = simulate_portfolio(
            prices, p["weights"], INITIAL_INVESTMENT, MONTHLY_CONTRIBUTION,
            rebalance_annually=REBALANCE_ANNUALLY,
            ddca_thresholds=p.get("ddca_thresholds"),
            risk_free_rate=RISK_FREE_RATE,
        )
        rets = returns_from_simulation(vals, invested)
        metrics = calculate_metrics(vals, invested, rets, RISK_FREE_RATE)
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
    daily_rf = (1 + RISK_FREE_RATE) ** (1 / 252) - 1
    for r, color in zip(results, COLORS):
        rs = (
            (r["returns"] - daily_rf)
            .rolling(252).mean()
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

    # ================================================================ Save & show
    fig1.savefig("portfolio_overview.png",  dpi=150, bbox_inches="tight")
    fig2.savefig("portfolio_risk.png",      dpi=150, bbox_inches="tight")
    fig3.savefig("portfolio_assets.png",    dpi=150, bbox_inches="tight")
    fig4.savefig("portfolio_metrics.png",   dpi=150, bbox_inches="tight")
    print("Saved → portfolio_overview.png, portfolio_risk.png, portfolio_assets.png, portfolio_metrics.png")
    plt.show()


if __name__ == "__main__":
    run()
