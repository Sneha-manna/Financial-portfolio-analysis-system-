"""
portfolio.py
------------
Responsible for CSV persistence of stock transactions, virtual capital management,
input validation, Indian Rupee (INR) formatting, and core portfolio P&L calculations.
"""

from __future__ import annotations

import math
import uuid
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CSV_PATH = BASE_DIR / "portfolio.csv"
SETTINGS_CSV_PATH = BASE_DIR / "portfolio_settings.csv"

VIRTUAL_CAPITAL_OPTIONS = [100_000, 200_000, 500_000]
DEFAULT_VIRTUAL_CAPITAL = 500_000

CSV_COLUMNS = [
    "id",
    "date",
    "company",
    "symbol",
    "quantity",
    "purchase_price",
    "total_investment",
]


def format_inr(amount: Optional[float], decimals: int = 2) -> str:
    """
    Format a numeric amount in the Indian numbering system with the ₹ symbol.
    Examples:
        100000 -> ₹1,00,000.00
        500000 -> ₹5,00,000.00
        -1234.5 -> -₹1,234.50
    """
    if amount is None or (isinstance(amount, float) and (math.isnan(amount) or math.isinf(amount))):
        return "N/A"

    sign = "-" if amount < 0 else ""
    abs_val = abs(float(amount))
    formatted = f"{abs_val:.{decimals}f}"

    if "." in formatted:
        integer_part, decimal_part = formatted.split(".")
        dec_str = f".{decimal_part}"
    else:
        integer_part = formatted
        dec_str = ""

    if len(integer_part) <= 3:
        grouped = integer_part
    else:
        last_three = integer_part[-3:]
        remaining = integer_part[:-3]
        groups = []
        while len(remaining) > 2:
            groups.append(remaining[-2:])
            remaining = remaining[:-2]
        if remaining:
            groups.append(remaining)
        groups.reverse()
        grouped = ",".join(groups) + "," + last_three

    return f"{sign}₹{grouped}{dec_str}"


def ensure_portfolio_csv(csv_path: Path | str = DEFAULT_CSV_PATH) -> Path:
    """Ensure portfolio.csv exists with the expected header columns."""
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.stat().st_size == 0:
        pd.DataFrame(columns=CSV_COLUMNS).to_csv(path, index=False)
    return path


def load_transactions(csv_path: Path | str = DEFAULT_CSV_PATH) -> pd.DataFrame:
    """Load persisted stock transactions from CSV into a cleanly typed DataFrame."""
    path = ensure_portfolio_csv(csv_path)
    try:
        df = pd.read_csv(path, dtype={"id": str, "date": str, "company": str, "symbol": str})
    except Exception:
        return pd.DataFrame(columns=CSV_COLUMNS)

    for col in CSV_COLUMNS:
        if col not in df.columns:
            df[col] = pd.Series(dtype="object")

    df = df[CSV_COLUMNS].copy()
    if df.empty:
        return df

    df["id"] = df["id"].fillna("").astype(str)
    df["date"] = df["date"].fillna("").astype(str)
    df["company"] = df["company"].fillna("").astype(str).str.strip()
    df["symbol"] = df["symbol"].fillna("").astype(str).str.strip().str.upper()
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(0.0)
    df["purchase_price"] = pd.to_numeric(df["purchase_price"], errors="coerce").fillna(0.0)
    df["total_investment"] = pd.to_numeric(df["total_investment"], errors="coerce")
    df["total_investment"] = df["total_investment"].fillna(df["quantity"] * df["purchase_price"])
    return df


def save_transactions(df: pd.DataFrame, csv_path: Path | str = DEFAULT_CSV_PATH) -> None:
    """Persist transactions DataFrame to CSV."""
    path = ensure_portfolio_csv(csv_path)
    out = df.copy()
    for col in CSV_COLUMNS:
        if col not in out.columns:
            out[col] = ""
    out[CSV_COLUMNS].to_csv(path, index=False)


def load_virtual_capital(settings_path: Path | str = SETTINGS_CSV_PATH) -> int:
    """Load persisted virtual capital choice if saved, else return DEFAULT_VIRTUAL_CAPITAL."""
    path = Path(settings_path)
    if path.exists() and path.stat().st_size > 0:
        try:
            df = pd.read_csv(path)
            if "virtual_capital" in df.columns and not df.empty:
                val = int(df["virtual_capital"].iloc[0])
                if val in VIRTUAL_CAPITAL_OPTIONS:
                    return val
        except Exception:
            pass
    return DEFAULT_VIRTUAL_CAPITAL


def save_virtual_capital(amount: int, settings_path: Path | str = SETTINGS_CSV_PATH) -> None:
    """Persist selected virtual capital to lightweight CSV settings."""
    if amount not in VIRTUAL_CAPITAL_OPTIONS:
        raise ValueError(f"Virtual capital must be one of {VIRTUAL_CAPITAL_OPTIONS}.")
    path = Path(settings_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"virtual_capital": int(amount)}]).to_csv(path, index=False)


def validate_stock_entry(
    company: str,
    symbol: str,
    quantity: float,
    purchase_price: float,
    purchase_date: date | str,
    available_cash: float,
) -> Tuple[bool, str]:
    """Validate user inputs before recording a virtual stock purchase."""
    if not company or not company.strip():
        return False, "Please enter a valid company name."
    if not symbol or not symbol.strip():
        return False, "Please enter a valid stock symbol (e.g., RELIANCE.NS)."
    if quantity is None or float(quantity) <= 0:
        return False, "Quantity must be greater than zero."
    if purchase_price is None or float(purchase_price) <= 0:
        return False, "Purchase price must be greater than zero."

    try:
        parsed_date = (
            purchase_date
            if isinstance(purchase_date, date) and not isinstance(purchase_date, datetime)
            else pd.to_datetime(purchase_date).date()
        )
    except Exception:
        return False, "Please provide a valid purchase date."

    if parsed_date > date.today():
        return False, "Purchase date cannot be in the future."

    total_cost = float(quantity) * float(purchase_price)
    if total_cost > float(available_cash) + 1e-6:
        return (
            False,
            f"Insufficient virtual cash! Required {format_inr(total_cost)}, "
            f"but only {format_inr(available_cash)} is available. "
            "Reduce quantity or increase virtual capital in the sidebar.",
        )

    return True, ""


def add_transaction(
    company: str,
    symbol: str,
    quantity: float,
    purchase_price: float,
    purchase_date: date | str,
    available_cash: float,
    csv_path: Path | str = DEFAULT_CSV_PATH,
) -> Tuple[bool, str, pd.DataFrame]:
    """Validate and append a new stock investment entry to portfolio.csv."""
    valid, error_msg = validate_stock_entry(
        company=company,
        symbol=symbol,
        quantity=quantity,
        purchase_price=purchase_price,
        purchase_date=purchase_date,
        available_cash=available_cash,
    )
    df = load_transactions(csv_path)
    if not valid:
        return False, error_msg, df

    parsed_date_str = (
        purchase_date.strftime("%Y-%m-%d")
        if isinstance(purchase_date, (date, datetime))
        else pd.to_datetime(purchase_date).strftime("%Y-%m-%d")
    )
    qty_val = float(quantity)
    price_val = float(purchase_price)
    invested_val = round(qty_val * price_val, 2)

    new_row = {
        "id": uuid.uuid4().hex[:8].upper(),
        "date": parsed_date_str,
        "company": company.strip(),
        "symbol": symbol.strip().upper(),
        "quantity": qty_val,
        "purchase_price": price_val,
        "total_investment": invested_val,
    }

    if df.empty:
        updated_df = pd.DataFrame([new_row], columns=CSV_COLUMNS)
    else:
        updated_df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    save_transactions(updated_df, csv_path)
    return True, f"Added {qty_val:g} shares of {new_row['company']} ({new_row['symbol']}).", updated_df


def remove_holding_by_symbol(symbol: str, csv_path: Path | str = DEFAULT_CSV_PATH) -> pd.DataFrame:
    """Remove all transactions matching a stock symbol and persist changes."""
    df = load_transactions(csv_path)
    if df.empty:
        return df
    clean_symbol = symbol.strip().upper()
    updated_df = df[df["symbol"] != clean_symbol].reset_index(drop=True)
    save_transactions(updated_df, csv_path)
    return updated_df


def remove_transaction_by_id(tx_id: str, csv_path: Path | str = DEFAULT_CSV_PATH) -> pd.DataFrame:
    """Remove a single transaction entry by its ID and persist changes."""
    df = load_transactions(csv_path)
    if df.empty:
        return df
    updated_df = df[df["id"] != str(tx_id)].reset_index(drop=True)
    save_transactions(updated_df, csv_path)
    return updated_df


def clear_portfolio(csv_path: Path | str = DEFAULT_CSV_PATH) -> pd.DataFrame:
    """Clear all transactions from portfolio.csv."""
    empty_df = pd.DataFrame(columns=CSV_COLUMNS)
    save_transactions(empty_df, csv_path)
    return empty_df


def compute_holding_metrics(
    quantity: float,
    purchase_price: float,
    latest_price: Optional[float],
) -> dict:
    """
    Core financial formulas:
        Invested Amount   = Quantity * Purchase Price
        Current Value     = Quantity * Latest Available Price
        Profit/Loss       = Current Value - Invested Amount
        Return Percentage = (Profit/Loss / Invested Amount) * 100
    """
    invested_amount = float(quantity) * float(purchase_price)
    if latest_price is None or (isinstance(latest_price, float) and math.isnan(latest_price)):
        return {
            "invested_amount": invested_amount,
            "current_value": None,
            "profit_loss": None,
            "return_pct": None,
        }

    current_value = float(quantity) * float(latest_price)
    profit_loss = current_value - invested_amount
    return_pct = (profit_loss / invested_amount) * 100.0 if invested_amount > 0 else 0.0
    return {
        "invested_amount": invested_amount,
        "current_value": current_value,
        "profit_loss": profit_loss,
        "return_pct": return_pct,
    }


def build_holdings_summary(
    transactions_df: pd.DataFrame,
    price_snapshots: Dict[str, dict],
    virtual_capital: float = DEFAULT_VIRTUAL_CAPITAL,
    sector_map: Optional[Dict[str, str]] = None,
) -> Tuple[pd.DataFrame, dict]:
    """
    Aggregate transaction history into stock-wise holdings and compute
    portfolio-wide valuation and P&L metrics.
    """
    sector_map = sector_map or {}

    if transactions_df is None or transactions_df.empty:
        empty_holdings = pd.DataFrame(
            columns=[
                "company",
                "symbol",
                "quantity",
                "purchase_price",
                "latest_price",
                "invested_amount",
                "current_value",
                "profit_loss",
                "return_pct",
                "daily_change_pct",
                "sector",
                "latest_date",
                "data_status",
                "error",
            ]
        )
        summary = {
            "virtual_capital": float(virtual_capital),
            "total_investment": 0.0,
            "current_portfolio_value": 0.0,
            "available_cash": float(virtual_capital),
            "total_profit_loss": 0.0,
            "total_return_pct": 0.0,
            "num_stocks_held": 0,
            "all_prices_available": True,
            "unavailable_symbols": [],
        }
        return empty_holdings, summary

    rows = []
    unavailable_symbols = []
    total_investment_all = 0.0
    total_invested_with_prices = 0.0
    total_current_value_with_prices = 0.0

    grouped = transactions_df.groupby("symbol", sort=False)
    for symbol, grp in grouped:
        total_qty = float(grp["quantity"].sum())
        if total_qty <= 0:
            continue

        total_invested = float((grp["quantity"] * grp["purchase_price"]).sum())
        avg_purchase_price = total_invested / total_qty
        company_name = str(grp["company"].iloc[-1])

        snap = price_snapshots.get(symbol, {})
        latest_price = snap.get("latest_price") if snap.get("success") else None

        metrics = compute_holding_metrics(total_qty, avg_purchase_price, latest_price)
        total_investment_all += metrics["invested_amount"]

        if metrics["current_value"] is not None:
            total_invested_with_prices += metrics["invested_amount"]
            total_current_value_with_prices += metrics["current_value"]
            data_status = "Live"
        else:
            unavailable_symbols.append(symbol)
            data_status = "Unavailable"

        rows.append(
            {
                "company": company_name,
                "symbol": symbol,
                "quantity": total_qty,
                "purchase_price": avg_purchase_price,
                "latest_price": latest_price,
                "invested_amount": metrics["invested_amount"],
                "current_value": metrics["current_value"],
                "profit_loss": metrics["profit_loss"],
                "return_pct": metrics["return_pct"],
                "daily_change_pct": snap.get("daily_change_pct") if snap.get("success") else None,
                "sector": sector_map.get(symbol, "Sector Unavailable"),
                "latest_date": snap.get("latest_date") if snap.get("success") else None,
                "data_status": data_status,
                "error": snap.get("error") if not snap.get("success") else None,
            }
        )

    holdings_df = pd.DataFrame(rows)
    available_cash = float(virtual_capital) - total_investment_all
    all_prices_available = len(unavailable_symbols) == 0

    if not rows:
        current_portfolio_value = 0.0
        total_profit_loss = 0.0
        total_return_pct = 0.0
    elif len(unavailable_symbols) == len(rows):
        # None of the held stocks have live market data available
        current_portfolio_value = None
        total_profit_loss = None
        total_return_pct = None
    else:
        current_portfolio_value = total_current_value_with_prices
        total_profit_loss = total_current_value_with_prices - total_invested_with_prices
        total_return_pct = (
            (total_profit_loss / total_invested_with_prices) * 100.0
            if total_invested_with_prices > 0
            else 0.0
        )

    summary = {
        "virtual_capital": float(virtual_capital),
        "total_investment": total_investment_all,
        "current_portfolio_value": current_portfolio_value,
        "available_cash": available_cash,
        "total_profit_loss": total_profit_loss,
        "total_return_pct": total_return_pct,
        "num_stocks_held": len(rows),
        "all_prices_available": all_prices_available,
        "unavailable_symbols": unavailable_symbols,
    }
    return holdings_df, summary
