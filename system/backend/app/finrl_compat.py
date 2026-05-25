"""FinRL-compatible dataframe helpers.

This module does not start training. It only converts stored OHLCV data into the
column layout commonly used by FinRL/yfinance examples: date, open, high, low,
close, volume, tic, day.
"""

from __future__ import annotations

import pandas as pd


FINRL_PRICE_COLUMNS = [
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "tic",
    "day",
]


def to_finrl_price_frame(prices: pd.DataFrame) -> pd.DataFrame:
    """Convert internal market_prices_daily rows to a FinRL-style DataFrame."""

    if prices.empty:
        return pd.DataFrame(columns=FINRL_PRICE_COLUMNS)

    frame = prices.copy()
    frame["date"] = pd.to_datetime(frame["price_date"]).dt.strftime("%Y-%m-%d")
    frame["tic"] = frame["ticker"].astype(str).str.upper()
    frame["day"] = pd.to_datetime(frame["price_date"]).dt.dayofweek

    finrl_frame = frame[
        [
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "tic",
            "day",
        ]
    ].copy()

    return finrl_frame.sort_values(["date", "tic"]).reset_index(drop=True)
