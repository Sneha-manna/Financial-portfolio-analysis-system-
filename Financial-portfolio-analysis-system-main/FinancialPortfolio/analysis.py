"""
analysis.py
-----------
Responsible for portfolio allocation analytics, historical performance curves,
basic risk metrics (volatility, diversification, concentration), and Plotly
interactive financial visualizations.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

THEME_BG = "rgba(0,0,0,0)"
THEME_PAPER = "rgba(0,0,0,0)"
THEME_GRID = "rgba(148, 163, 184, 0.14)"
THEME_TEXT = "#E2E8F0"
THEME_MUTED = "#94A3B8"
EMERALD = "#10B981"
EMERALD_LIGHT = "#34D399"
CORAL = "#EF4444"
AMBER = "#F59E0B"
BLUE = "#38BDF8"

PALETTE = [
    "#10B981",
    "#38BDF8",
    "#34D399",
    "#A78BFA",
    "#F59E0B",
    "#2DD4BF",
    "#60A5FA",
    "#FB7185",
    "#4ADE80",
    "#FBBF24",
]


def _apply_dark_layout(fig: go.Figure, title: str = "", height: int = 400) -> go.Figure:
    """Apply consistent dark-navy and emerald financial theme styling to Plotly figures."""
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color=THEME_TEXT), x=0.01),
        paper_bgcolor=THEME_PAPER,
        plot_bgcolor=THEME_BG,
        font=dict(color=THEME_TEXT, family="Inter, system-ui, sans-serif"),
        height=height,
        margin=dict(l=16, r=16, t=48 if title else 24, b=16),
        legend=dict(
            bgcolor="rgba(15, 26, 46, 0.7)",
            bordercolor="rgba(148, 163, 184, 0.2)",
            borderwidth=1,
            font=dict(size=12, color=THEME_TEXT),
        ),
        hoverlabel=dict(
            bgcolor="#0F172A",
            bordercolor="#10B981",
            font=dict(color="#F8FAFC", size=12),
        ),
    )
    fig.update_xaxes(gridcolor=THEME_GRID, zerolinecolor=THEME_GRID, tickfont=dict(color=THEME_MUTED))
    fig.update_yaxes(gridcolor=THEME_GRID, zerolinecolor=THEME_GRID, tickfont=dict(color=THEME_MUTED))
    return fig


def compute_allocation_tables(holdings_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute company-wise and sector-wise allocation DataFrames from holdings.
    """
    if holdings_df is None or holdings_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    total_invested = float(holdings_df["invested_amount"].sum())
    company_df = holdings_df[
        ["company", "symbol", "sector", "invested_amount", "current_value"]
    ].copy()
    company_df["investment_pct"] = (
        (company_df["invested_amount"] / total_invested) * 100.0 if total_invested > 0 else 0.0
    )

    valid_cv = company_df["current_value"].dropna()
    total_cv = float(valid_cv.sum()) if not valid_cv.empty else 0.0
    company_df["current_value_pct"] = company_df["current_value"].apply(
        lambda v: (float(v) / total_cv) * 100.0 if pd.notna(v) and total_cv > 0 else None
    )
    company_df = company_df.sort_values("invested_amount", ascending=False).reset_index(drop=True)

    sector_df = (
        company_df.groupby("sector", as_index=False)
        .agg(
            invested_amount=("invested_amount", "sum"),
            stocks_count=("symbol", "count"),
        )
        .sort_values("invested_amount", ascending=False)
        .reset_index(drop=True)
    )
    sector_df["investment_pct"] = (
        (sector_df["invested_amount"] / total_invested) * 100.0 if total_invested > 0 else 0.0
    )

    return company_df, sector_df


def compute_portfolio_historical_series(
    holdings_df: pd.DataFrame,
    histories_by_symbol: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Compute historical daily portfolio market value using current holding quantities
    and real historical closing prices for each held stock.
    """
    if holdings_df is None or holdings_df.empty:
        return pd.DataFrame()

    price_frames = []
    qty_map = {}
    for _, row in holdings_df.iterrows():
        sym = row["symbol"]
        qty = float(row["quantity"])
        hist = histories_by_symbol.get(sym)
        if hist is not None and not hist.empty and "Close" in hist.columns:
            s = hist["Close"].astype(float).rename(sym)
            price_frames.append(s)
            qty_map[sym] = qty

    if not price_frames:
        return pd.DataFrame()

    combined = pd.concat(price_frames, axis=1).sort_index().ffill().dropna(how="all")
    if combined.empty:
        return pd.DataFrame()

    # For days where an individual stock has not yet started trading in the window, fill with 0 value or drop leading NaNs
    combined = combined.dropna()
    if combined.empty:
        return pd.DataFrame()

    portfolio_val = pd.Series(0.0, index=combined.index)
    for sym, qty in qty_map.items():
        if sym in combined.columns:
            portfolio_val += combined[sym] * qty

    total_invested = float(
        holdings_df[holdings_df["symbol"].isin(qty_map.keys())]["invested_amount"].sum()
    )

    out = pd.DataFrame(
        {
            "Date": combined.index,
            "Portfolio Value": portfolio_val.values,
            "Invested Cost": total_invested,
        }
    )
    return out


def compute_normalized_performance(histories_by_symbol: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Normalize each stock's Close price to cumulative percentage return (%)
    starting at 0% on the first common/available date in the selected range.
    """
    series_list = []
    for sym, hist in histories_by_symbol.items():
        if hist is not None and not hist.empty and "Close" in hist.columns:
            close = hist["Close"].astype(float).dropna()
            if len(close) >= 2 and close.iloc[0] > 0:
                norm = ((close / close.iloc[0]) - 1.0) * 100.0
                series_list.append(norm.rename(sym))

    if not series_list:
        return pd.DataFrame()

    return pd.concat(series_list, axis=1).sort_index()


def compute_risk_metrics(
    holdings_df: pd.DataFrame,
    histories_by_symbol: Dict[str, pd.DataFrame],
) -> Tuple[pd.DataFrame, dict]:
    """
    Compute beginner-friendly historical risk & diversification indicators:
      - Stock-wise annualized volatility (%), annualized return (%), max drawdown (%)
      - Portfolio diversification (HHI, Effective Number of Stocks)
      - Investment concentration (largest holding %, top-3 %)
      - Portfolio historical volatility (%)
    """
    if holdings_df is None or holdings_df.empty:
        return pd.DataFrame(), {}

    total_invested = float(holdings_df["invested_amount"].sum())
    stock_rows = []
    weighted_returns_frames = []

    for _, row in holdings_df.iterrows():
        sym = row["symbol"]
        company = row["company"]
        weight = (float(row["invested_amount"]) / total_invested) if total_invested > 0 else 0.0
        hist = histories_by_symbol.get(sym)

        if hist is None or hist.empty or "Close" not in hist.columns or len(hist) < 5:
            stock_rows.append(
                {
                    "Company": company,
                    "Symbol": sym,
                    "Weight (%)": weight * 100.0,
                    "Annualized Volatility (%)": None,
                    "Historical Annual Return (%)": None,
                    "Max Drawdown (%)": None,
                    "Risk Level": "Data Unavailable",
                }
            )
            continue

        close = hist["Close"].astype(float).dropna()
        daily_ret = close.pct_change().dropna()
        if len(daily_ret) < 3:
            continue

        ann_vol = float(np.std(daily_ret, ddof=1) * np.sqrt(252) * 100.0)
        mean_daily = float(np.mean(daily_ret))
        ann_ret = float(((1.0 + mean_daily) ** 252 - 1.0) * 100.0)

        rolling_peak = close.cummax()
        drawdowns = (close / rolling_peak) - 1.0
        max_dd = float(drawdowns.min() * 100.0)

        if ann_vol < 18.0:
            risk_label = "Low"
        elif ann_vol <= 30.0:
            risk_label = "Moderate"
        else:
            risk_label = "High"

        stock_rows.append(
            {
                "Company": company,
                "Symbol": sym,
                "Weight (%)": weight * 100.0,
                "Annualized Volatility (%)": ann_vol,
                "Historical Annual Return (%)": ann_ret,
                "Max Drawdown (%)": max_dd,
                "Risk Level": risk_label,
            }
        )
        weighted_returns_frames.append((daily_ret * weight).rename(sym))

    risk_df = pd.DataFrame(stock_rows)

    # Concentration & Diversification based on invested capital weights
    weights = (holdings_df["invested_amount"] / total_invested).values if total_invested > 0 else np.array([])
    if len(weights) > 0:
        sorted_idx = np.argsort(-weights)
        largest_weight_pct = float(weights[sorted_idx[0]] * 100.0)
        largest_stock = f"{holdings_df.iloc[sorted_idx[0]]['company']} ({holdings_df.iloc[sorted_idx[0]]['symbol']})"
        top3_weight_pct = float(np.sum(weights[sorted_idx[:3]]) * 100.0)
        hhi = float(np.sum(weights ** 2))
        effective_stocks = float(1.0 / hhi) if hhi > 0 else 0.0
    else:
        largest_weight_pct = 0.0
        largest_stock = "N/A"
        top3_weight_pct = 0.0
        hhi = 0.0
        effective_stocks = 0.0

    n_stocks = len(holdings_df)
    if n_stocks <= 1 or effective_stocks < 2.0:
        div_label = "Low Diversification"
    elif effective_stocks < 4.5:
        div_label = "Moderate Diversification"
    else:
        div_label = "Well Diversified"

    if largest_weight_pct > 40.0:
        conc_label = "High Concentration"
    elif largest_weight_pct > 25.0:
        conc_label = "Moderate Concentration"
    else:
        conc_label = "Balanced Concentration"

    # Historical portfolio volatility
    port_vol_pct = None
    if weighted_returns_frames:
        port_ret_df = pd.concat(weighted_returns_frames, axis=1).dropna()
        if len(port_ret_df) >= 5:
            port_daily = port_ret_df.sum(axis=1)
            port_vol_pct = float(np.std(port_daily, ddof=1) * np.sqrt(252) * 100.0)

    portfolio_risk = {
        "num_stocks": n_stocks,
        "effective_num_stocks": effective_stocks,
        "hhi_score": hhi * 10_000.0,
        "diversification_label": div_label,
        "largest_weight_pct": largest_weight_pct,
        "largest_stock": largest_stock,
        "top3_weight_pct": top3_weight_pct,
        "concentration_label": conc_label,
        "portfolio_volatility_pct": port_vol_pct,
    }
    return risk_df, portfolio_risk


# ---------------------------------------------------------------------------
# Plotly Chart Builders
# ---------------------------------------------------------------------------

def create_allocation_pie_chart(company_df: pd.DataFrame, value_col: str = "invested_amount") -> go.Figure:
    """Create interactive donut chart of company-wise portfolio allocation."""
    label_title = "Invested Capital Distribution" if value_col == "invested_amount" else "Current Value Distribution"
    fig = px.pie(
        company_df,
        names="company",
        values=value_col,
        hole=0.48,
        color_discrete_sequence=PALETTE,
        custom_data=["symbol"],
    )
    fig.update_traces(
        textposition="inside",
        textinfo="percent+label",
        hovertemplate="<b>%{label}</b> (%{customdata[0]})<br>Amount: ₹%{value:,.2f}<br>Share: %{percent}<extra></extra>",
        marker=dict(line=dict(color="#070D19", width=2)),
    )
    return _apply_dark_layout(fig, title=label_title, height=390)


def create_sector_allocation_chart(sector_df: pd.DataFrame) -> go.Figure:
    """Create interactive bar/pie chart of sector-wise portfolio distribution."""
    fig = px.pie(
        sector_df,
        names="sector",
        values="invested_amount",
        hole=0.45,
        color_discrete_sequence=[BLUE, EMERALD, AMBER, "#A78BFA", "#2DD4BF", "#FB7185"],
    )
    fig.update_traces(
        textposition="inside",
        textinfo="percent+label",
        hovertemplate="<b>%{label}</b><br>Invested: ₹%{value:,.2f}<br>Share: %{percent}<extra></extra>",
        marker=dict(line=dict(color="#070D19", width=2)),
    )
    return _apply_dark_layout(fig, title="Sector-Wise Allocation", height=390)


def create_stock_price_chart(
    history_df: pd.DataFrame,
    symbol: str,
    company_name: str = "",
) -> go.Figure:
    """Create interactive price chart with 20-DMA and 50-DMA overlays."""
    df = history_df.copy()
    df["DMA_20"] = df["Close"].rolling(window=20).mean()
    df["DMA_50"] = df["Close"].rolling(window=50).mean()

    title_name = f"{company_name} ({symbol})" if company_name else symbol
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Close"],
            mode="lines",
            name="Close Price (₹)",
            line=dict(color=EMERALD, width=2.4),
            fill="tozeroy",
            fillcolor="rgba(16, 185, 129, 0.10)",
            hovertemplate="Date: %{x|%Y-%m-%d}<br>Close: ₹%{y:,.2f}<extra></extra>",
        )
    )
    if df["DMA_20"].notna().sum() > 1:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["DMA_20"],
                mode="lines",
                name="20-Day Avg",
                line=dict(color=BLUE, width=1.4, dash="dot"),
                hovertemplate="Date: %{x|%Y-%m-%d}<br>20-DMA: ₹%{y:,.2f}<extra></extra>",
            )
        )
    if df["DMA_50"].notna().sum() > 1:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["DMA_50"],
                mode="lines",
                name="50-Day Avg",
                line=dict(color=AMBER, width=1.4, dash="dash"),
                hovertemplate="Date: %{x|%Y-%m-%d}<br>50-DMA: ₹%{y:,.2f}<extra></extra>",
            )
        )
    fig.update_yaxes(title_text="Price (₹)", tickprefix="₹")
    return _apply_dark_layout(fig, title=f"Historical Stock Price — {title_name}", height=420)


def create_portfolio_value_chart(portfolio_series_df: pd.DataFrame) -> go.Figure:
    """Plot historical market value of current holdings vs total invested cost."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=portfolio_series_df["Date"],
            y=portfolio_series_df["Portfolio Value"],
            mode="lines",
            name="Portfolio Market Value",
            line=dict(color=EMERALD, width=2.5),
            fill="tozeroy",
            fillcolor="rgba(16, 185, 129, 0.12)",
            hovertemplate="Date: %{x|%Y-%m-%d}<br>Value: ₹%{y:,.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=portfolio_series_df["Date"],
            y=portfolio_series_df["Invested Cost"],
            mode="lines",
            name="Total Invested Cost",
            line=dict(color=AMBER, width=1.8, dash="dash"),
            hovertemplate="Date: %{x|%Y-%m-%d}<br>Invested: ₹%{y:,.2f}<extra></extra>",
        )
    )
    fig.update_yaxes(title_text="Value (₹)", tickprefix="₹")
    return _apply_dark_layout(fig, title="Historical Portfolio Value vs Invested Cost", height=400)


def create_normalized_comparison_chart(norm_df: pd.DataFrame) -> go.Figure:
    """Plot normalized cumulative % return comparison across portfolio stocks."""
    fig = go.Figure()
    for idx, col in enumerate(norm_df.columns):
        fig.add_trace(
            go.Scatter(
                x=norm_df.index,
                y=norm_df[col],
                mode="lines",
                name=col,
                line=dict(color=PALETTE[idx % len(PALETTE)], width=2.0),
                hovertemplate=f"<b>{col}</b><br>Date: %{{x|%Y-%m-%d}}<br>Return: %{{y:+.2f}}%<extra></extra>",
            )
        )
    fig.add_hline(y=0, line_dash="dot", line_color="rgba(148, 163, 184, 0.4)")
    fig.update_yaxes(title_text="Cumulative Return (%)", ticksuffix="%")
    return _apply_dark_layout(fig, title="Individual Stock Historical Performance Comparison (%)", height=400)


def create_pnl_comparison_chart(holdings_df: pd.DataFrame) -> go.Figure:
    """Plot stock-wise Profit/Loss (₹) and Invested vs Current Value bars."""
    valid_df = holdings_df.dropna(subset=["profit_loss"]).copy()
    colors = [EMERALD if val >= 0 else CORAL for val in valid_df["profit_loss"]]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=valid_df["symbol"],
            y=valid_df["profit_loss"],
            marker_color=colors,
            name="Profit / Loss (₹)",
            text=[f"₹{v:+,.0f}" for v in valid_df["profit_loss"]],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Profit/Loss: ₹%{y:+,.2f}<extra></extra>",
        )
    )
    fig.add_hline(y=0, line_color="rgba(148, 163, 184, 0.5)")
    fig.update_yaxes(title_text="Profit / Loss (₹)", tickprefix="₹")
    return _apply_dark_layout(fig, title="Stock-Wise Profit / Loss Comparison", height=390)


def create_risk_return_scatter(risk_df: pd.DataFrame) -> go.Figure:
    """Plot Annualized Volatility (%) vs Historical Annual Return (%) for each stock."""
    valid = risk_df.dropna(subset=["Annualized Volatility (%)", "Historical Annual Return (%)"]).copy()
    fig = px.scatter(
        valid,
        x="Annualized Volatility (%)",
        y="Historical Annual Return (%)",
        size="Weight (%)",
        color="Risk Level",
        text="Symbol",
        color_discrete_map={"Low": EMERALD, "Moderate": AMBER, "High": CORAL},
        hover_data={"Company": True, "Weight (%)": ":.1f", "Max Drawdown (%)": ":.2f"},
    )
    fig.update_traces(textposition="top center")
    fig.add_hline(y=0, line_dash="dot", line_color="rgba(148, 163, 184, 0.4)")
    return _apply_dark_layout(fig, title="Historical Risk vs Return Profile", height=400)
