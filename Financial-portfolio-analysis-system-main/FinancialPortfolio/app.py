"""
app.py
------
Main Streamlit application entry point for the Financial Portfolio Analysis System.
Provides an interactive dark-navy & emerald-green dashboard for tracking a virtual
Indian stock investment portfolio with real YFinance market data and CSV persistence.
"""

from __future__ import annotations

from datetime import date
from typing import Dict

import pandas as pd
import streamlit as st

from analysis import (
    compute_allocation_tables,
    compute_normalized_performance,
    compute_portfolio_historical_series,
    compute_risk_metrics,
    create_allocation_pie_chart,
    create_normalized_comparison_chart,
    create_pnl_comparison_chart,
    create_portfolio_value_chart,
    create_risk_return_scatter,
    create_sector_allocation_chart,
    create_stock_price_chart,
)
from portfolio import (
    DEFAULT_CSV_PATH,
    DEFAULT_VIRTUAL_CAPITAL,
    SETTINGS_CSV_PATH,
    VIRTUAL_CAPITAL_OPTIONS,
    add_transaction,
    build_holdings_summary,
    clear_portfolio,
    format_inr,
    load_transactions,
    load_virtual_capital,
    remove_holding_by_symbol,
    remove_transaction_by_id,
    save_virtual_capital,
)
from stock_data import (
    POPULAR_INDIAN_STOCKS,
    VALID_TIME_RANGES,
    current_ist_timestamp,
    fetch_stock_history,
    fetch_stock_sector,
    normalize_symbol,
)

# ---------------------------------------------------------------------------
# Page Configuration & Custom Dark Navy / Emerald Financial Theme CSS
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Financial Portfolio Analysis System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .stApp {
        background: radial-gradient(circle at top right, #0D1B33 0%, #070D19 55%, #050912 100%);
        color: #F1F5F9;
    }
    section[data-testid="stSidebar"] {
        background-color: #0B1324;
        border-right: 1px solid rgba(148, 163, 184, 0.14);
    }
    .hero-banner {
        background: linear-gradient(135deg, #0F1D36 0%, #0B2528 100%);
        border: 1px solid rgba(16, 185, 129, 0.28);
        border-radius: 14px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1.25rem;
    }
    .hero-title {
        font-size: 1.65rem;
        font-weight: 700;
        color: #F8FAFC;
        margin: 0 0 0.25rem 0;
        letter-spacing: -0.02em;
    }
    .hero-subtitle {
        font-size: 0.92rem;
        color: #CBD5E1;
        margin: 0;
    }
    .virtual-badge {
        display: inline-block;
        background: rgba(16, 185, 129, 0.16);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.4);
        border-radius: 999px;
        padding: 0.2rem 0.7rem;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.03em;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(180deg, rgba(17, 28, 51, 0.92) 0%, rgba(12, 20, 38, 0.96) 100%);
        border: 1px solid rgba(148, 163, 184, 0.16);
        border-radius: 12px;
        padding: 1rem 1.15rem;
        margin-bottom: 0.75rem;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.22);
    }
    .metric-label {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94A3B8;
        margin-bottom: 0.35rem;
        font-weight: 600;
    }
    .metric-value {
        font-size: 1.48rem;
        font-weight: 700;
        color: #F8FAFC;
        font-variant-numeric: tabular-nums;
        line-height: 1.2;
    }
    .metric-sub {
        font-size: 0.8rem;
        margin-top: 0.35rem;
        color: #94A3B8;
        font-variant-numeric: tabular-nums;
    }
    .pos { color: #10B981 !important; font-weight: 600; }
    .neg { color: #EF4444 !important; font-weight: 600; }
    .muted-note {
        font-size: 0.82rem;
        color: #94A3B8;
    }
    .info-box {
        background: rgba(15, 26, 46, 0.85);
        border-left: 3px solid #10B981;
        border-radius: 8px;
        padding: 0.85rem 1rem;
        margin: 0.5rem 0 1rem 0;
        font-size: 0.9rem;
        color: #CBD5E1;
    }
    div[data-testid="stDataFrame"] {
        border: 1px solid rgba(148, 163, 184, 0.16);
        border-radius: 10px;
        overflow: hidden;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached Market Data Wrappers
# ---------------------------------------------------------------------------
@st.cache_data(ttl=300, show_spinner=False)
def cached_stock_history(symbol: str, period: str = "1y") -> dict:
    return fetch_stock_history(symbol=symbol, period=period)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_stock_sector(symbol: str) -> str:
    return fetch_stock_sector(symbol=symbol)


def render_metric_card(label: str, value: str, subtext: str = "", value_class: str = "") -> None:
    cls_attr = f" {value_class}" if value_class else ""
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value{cls_attr}">{value}</div>
            <div class="metric-sub">{subtext}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def format_pct(val: float | None, signed: bool = True) -> str:
    if val is None or pd.isna(val):
        return "N/A"
    return f"{val:+.2f}%" if signed else f"{val:.2f}%"


def init_session_state() -> None:
    if "virtual_capital" not in st.session_state:
        st.session_state.virtual_capital = load_virtual_capital(SETTINGS_CSV_PATH)
    if "nav_page" not in st.session_state:
        st.session_state.nav_page = "Dashboard Overview"
    if "add_purchase_price" not in st.session_state:
        st.session_state.add_purchase_price = 100.0
    if "last_preset" not in st.session_state:
        st.session_state.last_preset = None


init_session_state()

# ---------------------------------------------------------------------------
# Load Portfolio Data & Fetch Live Market Snapshots
# ---------------------------------------------------------------------------
transactions_df = load_transactions(DEFAULT_CSV_PATH)
held_symbols = sorted(transactions_df["symbol"].unique().tolist()) if not transactions_df.empty else []

snapshots_1y: Dict[str, dict] = {}
sectors_map: Dict[str, str] = {}
if held_symbols:
    with st.spinner("Retrieving market data from Yahoo Finance..."):
        for sym in held_symbols:
            snapshots_1y[sym] = cached_stock_history(sym, period="1y")
            sectors_map[sym] = cached_stock_sector(sym)

virtual_capital = st.session_state.virtual_capital
holdings_df, portfolio_summary = build_holdings_summary(
    transactions_df=transactions_df,
    price_snapshots=snapshots_1y,
    virtual_capital=virtual_capital,
    sector_map=sectors_map,
)

# Determine latest retrieval timestamp across snapshots
retrieval_timestamps = [
    snap["retrieved_at"] for snap in snapshots_1y.values() if snap.get("retrieved_at")
]
latest_retrieval_ts = retrieval_timestamps[0] if retrieval_timestamps else current_ist_timestamp()


# ---------------------------------------------------------------------------
# Sidebar Navigation & Virtual Capital Configuration
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📈 Portfolio Navigator")
    nav_options = [
        "Dashboard Overview",
        "Add Stocks & Holdings",
        "Stock Performance & Market Data",
        "Portfolio Allocation",
        "Risk Analysis",
        "Transaction History & Export",
    ]
    selected_page = st.radio(
        "Navigate",
        options=nav_options,
        index=nav_options.index(st.session_state.nav_page),
        label_visibility="collapsed",
    )
    st.session_state.nav_page = selected_page

    st.markdown("---")
    st.markdown("### 💰 Virtual Investment Capital")
    st.caption("Virtual money for educational simulation only — not real funds.")

    capital_labels = {amt: format_inr(amt, decimals=0) for amt in VIRTUAL_CAPITAL_OPTIONS}
    current_idx = (
        VIRTUAL_CAPITAL_OPTIONS.index(virtual_capital)
        if virtual_capital in VIRTUAL_CAPITAL_OPTIONS
        else VIRTUAL_CAPITAL_OPTIONS.index(DEFAULT_VIRTUAL_CAPITAL)
    )
    chosen_capital = st.selectbox(
        "Select Virtual Capital",
        options=VIRTUAL_CAPITAL_OPTIONS,
        index=current_idx,
        format_func=lambda x: capital_labels[x],
    )

    if chosen_capital != st.session_state.virtual_capital:
        if chosen_capital < portfolio_summary["total_investment"]:
            st.error(
                f"Cannot reduce capital to {format_inr(chosen_capital, 0)} because "
                f"your current investments total {format_inr(portfolio_summary['total_investment'])}. "
                "Remove holdings first."
            )
        else:
            st.session_state.virtual_capital = chosen_capital
            save_virtual_capital(chosen_capital, SETTINGS_CSV_PATH)
            st.rerun()

    st.markdown("---")
    st.markdown("**Quick Virtual Account Summary**")
    st.write(f"• Capital: **{format_inr(st.session_state.virtual_capital, 0)}**")
    st.write(f"• Invested: **{format_inr(portfolio_summary['total_investment'])}**")
    st.write(f"• Virtual Cash: **{format_inr(portfolio_summary['available_cash'])}**")
    st.write(f"• Stocks Held: **{portfolio_summary['num_stocks_held']}**")

    st.markdown("---")
    if st.button("🔄 Refresh Live Market Data", width="stretch"):
        st.cache_data.clear()
        st.rerun()
    st.caption(f"Last check: {latest_retrieval_ts}")


# ---------------------------------------------------------------------------
# Hero Header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-banner">
        <div class="virtual-badge">Virtual Portfolio · Educational Simulation Only</div>
        <div class="hero-title">Financial Portfolio Analysis System</div>
        <p class="hero-subtitle">
            Track Indian equity holdings, evaluate profit &amp; loss with live Yahoo Finance prices,
            inspect diversification &amp; risk indicators, and export CSV investment reports.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Global alert if any held symbol failed to fetch live data
if portfolio_summary["unavailable_symbols"]:
    failed_list = ", ".join(portfolio_summary["unavailable_symbols"])
    st.error(
        f"Live market data could not be retrieved for: **{failed_list}**. "
        "No placeholder or sample prices are shown. Please verify ticker symbols "
        "(e.g., RELIANCE.NS) and internet connectivity, then click 'Refresh Live Market Data'."
    )


# ---------------------------------------------------------------------------
# Helper: Formatted Holdings Table
# ---------------------------------------------------------------------------
def build_display_holdings_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    display = pd.DataFrame(
        {
            "Company Name": df["company"],
            "Stock Symbol": df["symbol"],
            "Quantity": df["quantity"].apply(lambda q: f"{q:g}"),
            "Purchase Price": df["purchase_price"].apply(format_inr),
            "Latest Price": df["latest_price"].apply(format_inr),
            "Invested Amount": df["invested_amount"].apply(format_inr),
            "Current Value": df["current_value"].apply(format_inr),
            "Profit / Loss": df["profit_loss"].apply(
                lambda x: f"+{format_inr(x)}" if pd.notna(x) and x > 0 else format_inr(x)
            ),
            "Return (%)": df["return_pct"].apply(format_pct),
            "Data Date": df["latest_date"].fillna("Unavailable"),
        }
    )
    return display


# ===========================================================================
# PAGE 1: DASHBOARD OVERVIEW
# ===========================================================================
if selected_page == "Dashboard Overview":
    c1, c2, c3 = st.columns(3)
    with c1:
        render_metric_card(
            "Total Investment",
            format_inr(portfolio_summary["total_investment"]),
            subtext="Total cost basis across holdings",
        )
    with c2:
        curr_val = portfolio_summary["current_portfolio_value"]
        val_str = format_inr(curr_val) if curr_val is not None else "Unavailable"
        sub = (
            f"Market data retrieved: {latest_retrieval_ts}"
            if portfolio_summary["all_prices_available"]
            else "Partial / unavailable live market prices"
        )
        render_metric_card("Current Portfolio Value", val_str, subtext=sub)
    with c3:
        render_metric_card(
            "Available Virtual Cash",
            format_inr(portfolio_summary["available_cash"]),
            subtext=f"Virtual balance out of {format_inr(virtual_capital, 0)} (Educational)",
        )

    c4, c5, c6 = st.columns(3)
    pnl_val = portfolio_summary["total_profit_loss"]
    ret_val = portfolio_summary["total_return_pct"]
    pnl_cls = "pos" if (pnl_val is not None and pnl_val > 0) else ("neg" if (pnl_val is not None and pnl_val < 0) else "")

    with c4:
        pnl_str = (
            (f"+{format_inr(pnl_val)}" if pnl_val > 0 else format_inr(pnl_val))
            if pnl_val is not None
            else "Unavailable"
        )
        render_metric_card(
            "Total Profit / Loss",
            pnl_str,
            subtext="Current Value − Invested Amount",
            value_class=pnl_cls,
        )
    with c5:
        ret_str = format_pct(ret_val) if ret_val is not None else "Unavailable"
        render_metric_card(
            "Return Percentage",
            ret_str,
            subtext="Portfolio return on invested capital",
            value_class=pnl_cls,
        )
    with c6:
        render_metric_card(
            "Stocks Held",
            str(portfolio_summary["num_stocks_held"]),
            subtext=f"{len(transactions_df)} recorded transaction(s)",
        )

    if holdings_df.empty:
        st.markdown(
            """
            <div class="info-box">
                <b>Welcome to your Virtual Portfolio!</b> Your portfolio currently has no stocks.
                Click <b>Add Stocks &amp; Holdings</b> in the sidebar (or the button below) to buy
                your first Indian stock using virtual capital.
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("➕ Add Your First Stock", type="primary"):
            st.session_state.nav_page = "Add Stocks & Holdings"
            st.rerun()
    else:
        st.markdown("### Portfolio Snapshot")
        col_left, col_right = st.columns(2)
        company_alloc_df, _ = compute_allocation_tables(holdings_df)

        with col_left:
            if not company_alloc_df.empty:
                st.plotly_chart(
                    create_allocation_pie_chart(company_alloc_df, value_col="invested_amount"),
                    width="stretch",
                )
        with col_right:
            if not holdings_df.dropna(subset=["profit_loss"]).empty:
                st.plotly_chart(
                    create_pnl_comparison_chart(holdings_df),
                    width="stretch",
                )
            else:
                st.info("Profit/Loss comparison chart requires valid live stock prices.")

        histories_1y = {
            sym: snap["history"]
            for sym, snap in snapshots_1y.items()
            if snap.get("success") and not snap["history"].empty
        }
        port_hist_df = compute_portfolio_historical_series(holdings_df, histories_1y)
        if not port_hist_df.empty:
            st.plotly_chart(create_portfolio_value_chart(port_hist_df), width="stretch")

        st.markdown("### Current Holdings Summary")
        st.dataframe(build_display_holdings_df(holdings_df), width="stretch", hide_index=True)


# ===========================================================================
# PAGE 2: ADD STOCKS & HOLDINGS
# ===========================================================================
elif selected_page == "Add Stocks & Holdings":
    st.markdown("### ➕ Add Indian Stock to Virtual Portfolio")
    st.caption(
        f"Available Virtual Cash: **{format_inr(portfolio_summary['available_cash'])}** "
        "(Virtual money for educational simulation)"
    )

    stock_choices = list(POPULAR_INDIAN_STOCKS.keys()) + ["Custom Stock (Enter Manually)"]
    selected_preset = st.selectbox("Choose a Popular Indian Company or Custom", options=stock_choices)

    if selected_preset == "Custom Stock (Enter Manually)":
        default_company = ""
        default_symbol = ""
    else:
        default_company = selected_preset
        default_symbol = POPULAR_INDIAN_STOCKS[selected_preset]

    if st.session_state.last_preset != selected_preset:
        st.session_state.last_preset = selected_preset
        st.session_state.symbol_preview_input = default_symbol
        st.session_state.add_company_input = default_company
        st.session_state.add_symbol_input = default_symbol

    # Live price preview helper for the chosen symbol
    preview_symbol = st.text_input(
        "Stock Symbol to Preview / Use (e.g., RELIANCE.NS, TCS.NS, INFY.NS)",
        key="symbol_preview_input",
    ).strip().upper()

    col_p1, col_p2 = st.columns([3, 1])
    live_snap = None
    if preview_symbol:
        live_snap = cached_stock_history(preview_symbol, period="1mo")
        with col_p1:
            if live_snap["success"]:
                chg = live_snap["daily_change"]
                chg_pct = live_snap["daily_change_pct"]
                chg_text = (
                    f" ({chg:+.2f}, {chg_pct:+.2f}%)"
                    if chg is not None and chg_pct is not None
                    else ""
                )
                st.info(
                    f"Latest available price for **{preview_symbol}**: "
                    f"**{format_inr(live_snap['latest_price'])}**{chg_text} · "
                    f"Market Date: **{live_snap['latest_date']}** · "
                    f"Retrieved: {live_snap['retrieved_at']}"
                )
            else:
                st.warning(live_snap["error"])
        with col_p2:
            if live_snap["success"] and st.button("Use Latest Price", width="stretch"):
                st.session_state.add_purchase_price = round(float(live_snap["latest_price"]), 2)
                st.session_state.add_symbol_input = preview_symbol
                st.rerun()

    with st.form("add_stock_form"):
        f_col1, f_col2 = st.columns(2)
        with f_col1:
            company_input = st.text_input(
                "Company Name",
                key="add_company_input",
                placeholder="e.g., Reliance Industries",
            )
            quantity_input = st.number_input(
                "Quantity (Shares)",
                min_value=1,
                step=1,
                value=10,
            )
        with f_col2:
            symbol_input = st.text_input(
                "Stock Symbol (NSE ends with .NS, BSE with .BO)",
                key="add_symbol_input",
                placeholder="e.g., RELIANCE.NS",
            )
            purchase_price_input = st.number_input(
                "Purchase Price per Share (₹)",
                min_value=0.01,
                step=1.0,
                key="add_purchase_price",
                format="%.2f",
            )

        purchase_date_input = st.date_input(
            "Purchase Date",
            value=date.today(),
            max_value=date.today(),
        )

        submitted = st.form_submit_button("Add Stock to Portfolio", type="primary", width="stretch")

        if submitted:
            ok, msg, _ = add_transaction(
                company=company_input,
                symbol=symbol_input,
                quantity=quantity_input,
                purchase_price=purchase_price_input,
                purchase_date=purchase_date_input,
                available_cash=portfolio_summary["available_cash"],
                csv_path=DEFAULT_CSV_PATH,
            )
            if ok:
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)

    st.markdown("---")
    st.markdown("### 📋 Portfolio Holdings Table")
    if holdings_df.empty:
        st.info("No holdings yet. Add a stock above to populate your portfolio.")
    else:
        st.dataframe(build_display_holdings_df(holdings_df), width="stretch", hide_index=True)

        st.markdown("#### 🗑️ Remove a Holding")
        rem_col1, rem_col2 = st.columns([3, 1])
        symbol_options = holdings_df["symbol"].tolist()
        label_map = {
            row["symbol"]: f"{row['company']} ({row['symbol']}) — {row['quantity']:g} shares"
            for _, row in holdings_df.iterrows()
        }
        with rem_col1:
            sym_to_remove = st.selectbox(
                "Select holding to remove",
                options=symbol_options,
                format_func=lambda s: label_map.get(s, s),
            )
        with rem_col2:
            st.write("")
            st.write("")
            if st.button("Remove Holding", type="secondary", width="stretch"):
                remove_holding_by_symbol(sym_to_remove, DEFAULT_CSV_PATH)
                st.success(f"Removed {sym_to_remove} from your portfolio.")
                st.rerun()


# ===========================================================================
# PAGE 3: STOCK PERFORMANCE & MARKET DATA
# ===========================================================================
elif selected_page == "Stock Performance & Market Data":
    st.markdown("### 📊 Stock Price Data & Performance Analysis")

    available_stocks = {}
    for _, row in holdings_df.iterrows():
        available_stocks[row["symbol"]] = row["company"]
    for comp, sym in POPULAR_INDIAN_STOCKS.items():
        if sym not in available_stocks:
            available_stocks[sym] = comp

    sel_col1, sel_col2, sel_col3 = st.columns([2, 1, 1])
    with sel_col1:
        chosen_sym = st.selectbox(
            "Select Stock",
            options=list(available_stocks.keys()),
            format_func=lambda s: f"{available_stocks[s]} ({s})",
        )
    with sel_col2:
        custom_sym = st.text_input("Or Enter Custom Symbol", value="", placeholder="e.g., LT.NS")
    with sel_col3:
        range_label = st.selectbox("Historical Time Range", options=list(VALID_TIME_RANGES.keys()), index=3)

    target_symbol = normalize_symbol(custom_sym) if custom_sym.strip() else chosen_sym
    target_company = available_stocks.get(target_symbol, target_symbol)
    period_code = VALID_TIME_RANGES[range_label]

    with st.spinner(f"Fetching {range_label} historical prices for {target_symbol}..."):
        # Always fetch 1y snapshot for 1M/6M/1Y stats, and range-specific history for chart
        snap_1y = cached_stock_history(target_symbol, period="1y")
        snap_range = (
            snap_1y
            if period_code == "1y"
            else cached_stock_history(target_symbol, period=period_code)
        )

    if not snap_range["success"]:
        st.error(snap_range["error"])
    else:
        st.caption(
            f"Market Data Date: **{snap_range['latest_date']}** · "
            f"Retrieved Timestamp: **{snap_range['retrieved_at']}**"
        )

        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            render_metric_card("Latest Price", format_inr(snap_range["latest_price"]), subtext=target_symbol)
        with m2:
            dc = snap_range["daily_change"]
            dcp = snap_range["daily_change_pct"]
            dc_cls = "pos" if (dc is not None and dc > 0) else ("neg" if (dc is not None and dc < 0) else "")
            dc_str = f"{dc:+.2f}" if dc is not None else "N/A"
            render_metric_card(
                "Daily Change",
                f"₹{dc_str}" if dc is not None else "N/A",
                subtext=format_pct(dcp),
                value_class=dc_cls,
            )
        with m3:
            p1m = snap_1y.get("perf_1m") if snap_1y["success"] else None
            p1m_cls = "pos" if (p1m is not None and p1m > 0) else ("neg" if (p1m is not None and p1m < 0) else "")
            render_metric_card("1-Month Perf", format_pct(p1m), subtext="Trailing 1M return", value_class=p1m_cls)
        with m4:
            p6m = snap_1y.get("perf_6m") if snap_1y["success"] else None
            p6m_cls = "pos" if (p6m is not None and p6m > 0) else ("neg" if (p6m is not None and p6m < 0) else "")
            render_metric_card("6-Month Perf", format_pct(p6m), subtext="Trailing 6M return", value_class=p6m_cls)
        with m5:
            p1y = snap_1y.get("perf_1y") if snap_1y["success"] else None
            p1y_cls = "pos" if (p1y is not None and p1y > 0) else ("neg" if (p1y is not None and p1y < 0) else "")
            render_metric_card("1-Year Perf", format_pct(p1y), subtext="Trailing 1Y return", value_class=p1y_cls)

        st.plotly_chart(
            create_stock_price_chart(snap_range["history"], target_symbol, target_company),
            width="stretch",
        )

        with st.expander("View Recent Historical Prices Table"):
            hist_display = snap_range["history"].tail(30).sort_index(ascending=False).copy()
            hist_display.index = hist_display.index.strftime("%Y-%m-%d")
            hist_display.index.name = "Date"
            cols_to_show = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in hist_display.columns]
            st.dataframe(hist_display[cols_to_show].round(2), width="stretch")

    st.markdown("---")
    st.markdown("### 📈 Portfolio-Wide Performance Comparisons")
    if holdings_df.empty:
        st.info("Add stocks to your portfolio to compare portfolio value, normalized returns, and profit/loss.")
    else:
        range_histories = {}
        for sym in holdings_df["symbol"].tolist():
            res = cached_stock_history(sym, period=period_code)
            if res["success"] and not res["history"].empty:
                range_histories[sym] = res["history"]

        col_a, col_b = st.columns(2)
        with col_a:
            port_series = compute_portfolio_historical_series(holdings_df, range_histories)
            if not port_series.empty:
                st.plotly_chart(create_portfolio_value_chart(port_series), width="stretch")
            else:
                st.info("Insufficient historical series to chart portfolio value.")

        with col_b:
            norm_df = compute_normalized_performance(range_histories)
            if not norm_df.empty:
                st.plotly_chart(create_normalized_comparison_chart(norm_df), width="stretch")
            else:
                st.info("Insufficient historical series to chart normalized performance.")

        if not holdings_df.dropna(subset=["profit_loss"]).empty:
            st.plotly_chart(create_pnl_comparison_chart(holdings_df), width="stretch")


# ===========================================================================
# PAGE 4: PORTFOLIO ALLOCATION
# ===========================================================================
elif selected_page == "Portfolio Allocation":
    st.markdown("### 🥧 Portfolio Allocation & Sector Distribution")

    if holdings_df.empty:
        st.info("Your portfolio is empty. Add stocks to view company-wise and sector-wise allocation.")
    else:
        company_alloc_df, sector_alloc_df = compute_allocation_tables(holdings_df)
        basis_choice = st.radio(
            "Allocation Chart Basis",
            options=["Invested Amount", "Current Value"],
            horizontal=True,
        )
        val_col = "invested_amount" if basis_choice == "Invested Amount" else "current_value"

        col1, col2 = st.columns(2)
        with col1:
            plot_df = company_alloc_df.dropna(subset=[val_col])
            if not plot_df.empty:
                st.plotly_chart(
                    create_allocation_pie_chart(plot_df, value_col=val_col),
                    width="stretch",
                )
            else:
                st.warning("Current value data is unavailable to plot this chart.")
        with col2:
            if not sector_alloc_df.empty:
                st.plotly_chart(create_sector_allocation_chart(sector_alloc_df), width="stretch")

        st.markdown("#### Company-Wise Allocation")
        comp_table = pd.DataFrame(
            {
                "Company": company_alloc_df["company"],
                "Symbol": company_alloc_df["symbol"],
                "Sector": company_alloc_df["sector"],
                "Invested Amount": company_alloc_df["invested_amount"].apply(format_inr),
                "Investment (%)": company_alloc_df["investment_pct"].apply(lambda x: f"{x:.2f}%"),
                "Current Value": company_alloc_df["current_value"].apply(format_inr),
                "Current Share (%)": company_alloc_df["current_value_pct"].apply(
                    lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A"
                ),
            }
        )
        st.dataframe(comp_table, width="stretch", hide_index=True)

        st.markdown("#### Sector-Wise Allocation")
        sec_table = pd.DataFrame(
            {
                "Sector": sector_alloc_df["sector"],
                "Stocks Count": sector_alloc_df["stocks_count"],
                "Invested Amount": sector_alloc_df["invested_amount"].apply(format_inr),
                "Investment (%)": sector_alloc_df["investment_pct"].apply(lambda x: f"{x:.2f}%"),
            }
        )
        st.dataframe(sec_table, width="stretch", hide_index=True)
        if "Sector Unavailable" in sector_alloc_df["sector"].values:
            st.caption(
                "Note: Sector metadata is sourced directly from Yahoo Finance. "
                "Entries marked 'Sector Unavailable' indicate Yahoo Finance did not return sector info."
            )


# ===========================================================================
# PAGE 5: RISK ANALYSIS
# ===========================================================================
elif selected_page == "Risk Analysis":
    st.markdown("### 🛡️ Basic Portfolio Risk Analysis")

    if holdings_df.empty:
        st.info("Add stocks to your portfolio to evaluate volatility, diversification, and concentration.")
    else:
        histories_1y = {
            sym: snap["history"]
            for sym, snap in snapshots_1y.items()
            if snap.get("success") and not snap["history"].empty
        }
        risk_df, port_risk = compute_risk_metrics(holdings_df, histories_1y)

        r1, r2, r3, r4 = st.columns(4)
        with r1:
            pv = port_risk.get("portfolio_volatility_pct")
            render_metric_card(
                "Portfolio Volatility (1Y)",
                f"{pv:.2f}%" if pv is not None else "N/A",
                subtext="Annualized daily return fluctuation",
            )
        with r2:
            render_metric_card(
                "Diversification",
                port_risk.get("diversification_label", "N/A"),
                subtext=f"Effective stocks: {port_risk.get('effective_num_stocks', 0):.2f} / {port_risk.get('num_stocks', 0)}",
            )
        with r3:
            render_metric_card(
                "Top Stock Weight",
                f"{port_risk.get('largest_weight_pct', 0):.1f}%",
                subtext=port_risk.get("largest_stock", ""),
            )
        with r4:
            render_metric_card(
                "Concentration Risk",
                port_risk.get("concentration_label", "N/A"),
                subtext=f"Top 3 holdings: {port_risk.get('top3_weight_pct', 0):.1f}%",
            )

        if not risk_df.empty and not risk_df.dropna(subset=["Annualized Volatility (%)"]).empty:
            st.plotly_chart(create_risk_return_scatter(risk_df), width="stretch")

            display_risk = risk_df.copy()
            display_risk["Weight (%)"] = display_risk["Weight (%)"].apply(lambda x: f"{x:.2f}%")
            display_risk["Annualized Volatility (%)"] = display_risk["Annualized Volatility (%)"].apply(
                lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A"
            )
            display_risk["Historical Annual Return (%)"] = display_risk["Historical Annual Return (%)"].apply(
                lambda x: f"{x:+.2f}%" if pd.notna(x) else "N/A"
            )
            display_risk["Max Drawdown (%)"] = display_risk["Max Drawdown (%)"].apply(
                lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A"
            )
            st.markdown("#### Stock-Wise Historical Risk Comparison")
            st.dataframe(display_risk, width="stretch", hide_index=True)

        st.markdown("#### 📖 Risk Indicators Explained Simply")
        st.markdown(
            """
            <div class="info-box">
            <ul style="margin: 0; padding-left: 1.2rem;">
                <li><b>Historical Volatility:</b> Measures how sharply a stock's daily price moves up and down over the past year. Lower volatility (&lt;18%) generally means steadier price movement; higher volatility (&gt;30%) means wider price swings.</li>
                <li><b>Portfolio Diversification:</b> Spreading money across multiple independent companies helps ensure one stock's drop does not overly hurt your entire portfolio.</li>
                <li><b>Investment Concentration:</b> Shows how much of your capital sits in your single largest stock. Keeping any single stock under ~25–30% reduces company-specific risk.</li>
                <li><b>Max Drawdown:</b> The largest peak-to-trough percentage drop observed in the past 1 year.</li>
            </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.warning(
            "Disclaimer: Risk metrics and returns are calculated strictly from past 1-year historical market prices "
            "for educational purposes. Historical performance does not guarantee future returns."
        )


# ===========================================================================
# PAGE 6: TRANSACTION HISTORY & EXPORT
# ===========================================================================
elif selected_page == "Transaction History & Export":
    st.markdown("### 🧾 Transaction History & CSV Reports")

    if transactions_df.empty:
        st.info("No transactions recorded yet.")
    else:
        tx_display = pd.DataFrame(
            {
                "Entry ID": transactions_df["id"],
                "Date": transactions_df["date"],
                "Company": transactions_df["company"],
                "Symbol": transactions_df["symbol"],
                "Quantity": transactions_df["quantity"].apply(lambda q: f"{q:g}"),
                "Purchase Price": transactions_df["purchase_price"].apply(format_inr),
                "Total Investment": transactions_df["total_investment"].apply(format_inr),
            }
        )
        st.dataframe(tx_display, width="stretch", hide_index=True)

        st.markdown("#### 📥 Export CSV Reports")
        exp_col1, exp_col2 = st.columns(2)
        with exp_col1:
            tx_csv_bytes = transactions_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Transaction History (CSV)",
                data=tx_csv_bytes,
                file_name="portfolio_transactions.csv",
                mime="text/csv",
                width="stretch",
            )
        with exp_col2:
            report_df = holdings_df.drop(columns=["error"], errors="ignore")
            report_csv_bytes = report_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Portfolio Valuation & P&L Report (CSV)",
                data=report_csv_bytes,
                file_name="portfolio_analysis_report.csv",
                mime="text/csv",
                width="stretch",
            )

        st.markdown("---")
        st.markdown("#### 🗑️ Manage Entries")
        del_col1, del_col2 = st.columns([3, 1])
        tx_ids = transactions_df["id"].tolist()
        tx_labels = {
            row["id"]: f"{row['id']} | {row['date']} | {row['company']} ({row['symbol']}) × {row['quantity']:g}"
            for _, row in transactions_df.iterrows()
        }
        with del_col1:
            selected_tx = st.selectbox(
                "Select a transaction entry to delete",
                options=tx_ids,
                format_func=lambda x: tx_labels.get(x, x),
            )
        with del_col2:
            st.write("")
            st.write("")
            if st.button("Delete Transaction", width="stretch"):
                remove_transaction_by_id(selected_tx, DEFAULT_CSV_PATH)
                st.success(f"Deleted transaction {selected_tx}.")
                st.rerun()

        with st.expander("Danger Zone: Clear Entire Portfolio"):
            confirm_clear = st.checkbox("I confirm I want to delete all saved transactions in portfolio.csv")
            if st.button("Clear All Portfolio Data", disabled=not confirm_clear):
                clear_portfolio(DEFAULT_CSV_PATH)
                st.success("All portfolio transactions have been cleared.")
                st.rerun()
