"""
stock_data.py
-------------
Responsible for retrieving real historical and latest available stock market data
from Yahoo Finance using YFinance for Indian (NSE/BSE) and other listed equities.
Never fabricates prices on network or API failure.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

IST = ZoneInfo("Asia/Kolkata")

POPULAR_INDIAN_STOCKS: Dict[str, str] = {
    "Reliance Industries": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "Infosys": "INFY.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "Tata Motors": "TATAMOTORS.NS",
    "ITC": "ITC.NS",
    "SBI": "SBIN.NS",
    "Wipro": "WIPRO.NS",
    "Bharti Airtel": "BHARTIARTL.NS",
}

VALID_TIME_RANGES: Dict[str, str] = {
    "1 Month": "1mo",
    "3 Months": "3mo",
    "6 Months": "6mo",
    "1 Year": "1y",
    "2 Years": "2y",
    "5 Years": "5y",
}


def current_ist_timestamp() -> str:
    """Return the current timestamp formatted in IST."""
    return datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST")


def normalize_symbol(symbol: str) -> str:
    """Clean and uppercase a stock ticker symbol."""
    return symbol.strip().upper()


def _compute_period_return(close_series: pd.Series, months: int) -> Optional[float]:
    """
    Calculate percentage return over `months` months from a clean DatetimeIndex Close series.
    Returns None if insufficient historical data exists.
    """
    if close_series is None or len(close_series) < 2:
        return None

    latest_dt = close_series.index[-1]
    latest_price = float(close_series.iloc[-1])
    target_dt = latest_dt - pd.DateOffset(months=months)

    prior = close_series[close_series.index <= target_dt]
    if not prior.empty:
        base_price = float(prior.iloc[-1])
    else:
        # Handle exact-window fetches where the earliest bar is within 10 calendar days of target_dt
        earliest_dt = close_series.index[0]
        if (earliest_dt - target_dt).days <= 10:
            base_price = float(close_series.iloc[0])
        else:
            return None

    if base_price <= 0:
        return None
    return ((latest_price - base_price) / base_price) * 100.0


def fetch_stock_history(symbol: str, period: str = "1y") -> dict:
    """
    Fetch historical OHLCV stock data and compute latest price, daily change,
    and 1M / 6M / 1Y performance metrics via YFinance.

    Returns a dictionary containing `success` (bool), `history` (DataFrame),
    metrics, timestamps, and `error` (str | None).
    """
    clean_symbol = normalize_symbol(symbol)
    retrieved_at = current_ist_timestamp()

    if not clean_symbol:
        return {
            "success": False,
            "symbol": clean_symbol,
            "history": pd.DataFrame(),
            "latest_price": None,
            "previous_close": None,
            "daily_change": None,
            "daily_change_pct": None,
            "latest_date": None,
            "retrieved_at": retrieved_at,
            "perf_1m": None,
            "perf_6m": None,
            "perf_1y": None,
            "error": "Stock symbol cannot be empty.",
        }

    try:
        ticker = yf.Ticker(clean_symbol)
        hist = ticker.history(period=period, auto_adjust=False)

        if hist is None or hist.empty or "Close" not in hist.columns:
            return {
                "success": False,
                "symbol": clean_symbol,
                "history": pd.DataFrame(),
                "latest_price": None,
                "previous_close": None,
                "daily_change": None,
                "daily_change_pct": None,
                "latest_date": None,
                "retrieved_at": retrieved_at,
                "perf_1m": None,
                "perf_6m": None,
                "perf_1y": None,
                "error": (
                    f"No historical market data returned for '{clean_symbol}'. "
                    "Verify the symbol (e.g., RELIANCE.NS for NSE or TCS.BO for BSE) "
                    "and check your internet connection."
                ),
            }

        df = hist.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df.index = df.index.normalize()
        df = df[~df.index.duplicated(keep="last")].sort_index()
        df = df.dropna(subset=["Close"])

        if df.empty:
            return {
                "success": False,
                "symbol": clean_symbol,
                "history": pd.DataFrame(),
                "latest_price": None,
                "previous_close": None,
                "daily_change": None,
                "daily_change_pct": None,
                "latest_date": None,
                "retrieved_at": retrieved_at,
                "perf_1m": None,
                "perf_6m": None,
                "perf_1y": None,
                "error": f"Retrieved dataset for '{clean_symbol}' has no valid closing prices.",
            }

        close_series = df["Close"].astype(float)
        latest_price = float(close_series.iloc[-1])
        latest_date = df.index[-1].strftime("%Y-%m-%d")

        previous_close: Optional[float] = None
        daily_change: Optional[float] = None
        daily_change_pct: Optional[float] = None
        if len(close_series) >= 2:
            previous_close = float(close_series.iloc[-2])
            daily_change = latest_price - previous_close
            if previous_close > 0:
                daily_change_pct = (daily_change / previous_close) * 100.0

        return {
            "success": True,
            "symbol": clean_symbol,
            "history": df,
            "latest_price": latest_price,
            "previous_close": previous_close,
            "daily_change": daily_change,
            "daily_change_pct": daily_change_pct,
            "latest_date": latest_date,
            "retrieved_at": retrieved_at,
            "perf_1m": _compute_period_return(close_series, 1),
            "perf_6m": _compute_period_return(close_series, 6),
            "perf_1y": _compute_period_return(close_series, 12),
            "error": None,
        }
    except Exception as exc:
        return {
            "success": False,
            "symbol": clean_symbol,
            "history": pd.DataFrame(),
            "latest_price": None,
            "previous_close": None,
            "daily_change": None,
            "daily_change_pct": None,
            "latest_date": None,
            "retrieved_at": retrieved_at,
            "perf_1m": None,
            "perf_6m": None,
            "perf_1y": None,
            "error": f"Failed to retrieve market data for '{clean_symbol}': {exc}",
        }


def fetch_stock_sector(symbol: str) -> str:
    """
    Attempt to retrieve sector metadata from YFinance `.info`.
    Returns 'Sector Unavailable' if unavailable or blocked.
    """
    clean_symbol = normalize_symbol(symbol)
    if not clean_symbol:
        return "Sector Unavailable"
    try:
        info = yf.Ticker(clean_symbol).info
        if isinstance(info, dict):
            sector = info.get("sector")
            if sector and isinstance(sector, str) and sector.strip():
                return sector.strip()
    except Exception:
        pass
    return "Sector Unavailable"


def fetch_portfolio_market_data(symbols: List[str], period: str = "1y") -> dict:
    """
    Fetch market snapshots and sector metadata for a list of unique stock symbols.
    Returns:
      {
        "snapshots": {symbol: dict},
        "sectors": {symbol: str},
        "retrieved_at": str
      }
    """
    unique_symbols = sorted({normalize_symbol(s) for s in symbols if s and s.strip()})
    snapshots: Dict[str, dict] = {}
    sectors: Dict[str, str] = {}

    for sym in unique_symbols:
        snapshots[sym] = fetch_stock_history(sym, period=period)
        sectors[sym] = fetch_stock_sector(sym)

    return {
        "snapshots": snapshots,
        "sectors": sectors,
        "retrieved_at": current_ist_timestamp(),
    }
