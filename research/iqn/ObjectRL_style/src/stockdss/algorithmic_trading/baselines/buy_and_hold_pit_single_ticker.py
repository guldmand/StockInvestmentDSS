from __future__ import annotations

import pandas as pd


def run_buy_and_hold(
    trade_data: pd.DataFrame,
    ticker: str,
    initial_amount: float = 1_000_000.0,
    price_column: str | None = None,
) -> pd.DataFrame:
    """
    Run a simple point-in-time compatible buy-and-hold baseline for one ticker.

    The strategy:
    - Uses only the supplied trade period.
    - Buys at the first available price in the trade data.
    - Holds until the final available date.
    - Normalizes portfolio value to initial_amount at the first trade date.

    Parameters
    ----------
    trade_data:
        PIT trade dataframe containing at least date, tic/ticker, and price columns.
    ticker:
        Single ticker to evaluate, e.g. AAPL.
    initial_amount:
        Initial portfolio value.
    price_column:
        Optional explicit price column. If None, the function prefers adj_close, then close.

    Returns
    -------
    pd.DataFrame
        Columns:
        - date
        - ticker
        - price
        - shares
        - cash
        - account_value
        - strategy
    """
    if trade_data.empty:
        raise ValueError("trade_data is empty.")

    df = trade_data.copy()

    ticker_col = _detect_ticker_column(df)
    date_col = _detect_date_column(df)
    price_col = price_column or _detect_price_column(df)

    df = df[df[ticker_col].astype(str).str.upper() == ticker.upper()].copy()

    if df.empty:
        raise ValueError(f"No rows found for ticker: {ticker}")

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    df = df.dropna(subset=[price_col])

    if df.empty:
        raise ValueError(f"No valid prices found for ticker: {ticker}")

    first_price = float(df[price_col].iloc[0])

    if first_price <= 0:
        raise ValueError(f"First price must be positive. Got: {first_price}")

    shares = initial_amount / first_price
    account_values = shares * df[price_col].astype(float)

    result = pd.DataFrame(
        {
            "date": df[date_col],
            "ticker": ticker.upper(),
            "price": df[price_col].astype(float),
            "shares": shares,
            "cash": 0.0,
            "account_value": account_values,
            "strategy": f"{ticker.upper()}_buy_and_hold",
        }
    )

    return result


def _detect_ticker_column(df: pd.DataFrame) -> str:
    for candidate in ["tic", "ticker", "symbol"]:
        if candidate in df.columns:
            return candidate

    raise ValueError(
        "Could not detect ticker column. Expected one of: tic, ticker, symbol."
    )


def _detect_date_column(df: pd.DataFrame) -> str:
    for candidate in ["date", "datetime", "timestamp"]:
        if candidate in df.columns:
            return candidate

    raise ValueError(
        "Could not detect date column. Expected one of: date, datetime, timestamp."
    )


def _detect_price_column(df: pd.DataFrame) -> str:
    for candidate in ["adj_close", "close", "Close", "Adj Close"]:
        if candidate in df.columns:
            return candidate

    raise ValueError(
        "Could not detect price column. Expected one of: adj_close, close, Close, Adj Close."
    )
