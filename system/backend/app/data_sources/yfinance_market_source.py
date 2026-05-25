"""Local yfinance-compatible market data source.

This source is used both as a direct fallback and as the normalization reference
for SDU_DataScienceTool-based ingestion.

Canonical adapter output:
date, ticker, open, high, low, close, adj_close, volume
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import pandas as pd
import yfinance as yf


@dataclass(frozen=True)
class MarketDataRequest:
    """Simple request object for daily OHLCV market data."""

    tickers: list[str]
    start_date: str
    end_date: str


class YFinanceMarketSource:
    """Fetch daily OHLCV data using yfinance and return a normalized DataFrame."""

    source_name = "yfinance"

    async def fetch_prices(
        self,
        tickers: Iterable[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []

        for ticker in tickers:
            normalized_ticker = ticker.strip().upper()
            if not normalized_ticker:
                continue

            frame = yf.download(
                normalized_ticker,
                start=start_date,
                end=end_date,
                interval="1d",
                auto_adjust=False,
                progress=False,
                group_by="column",
                threads=False,
            )

            if frame is None or frame.empty:
                continue

            frame = frame.reset_index()
            frame["ticker"] = normalized_ticker
            frames.append(frame)

        if not frames:
            return pd.DataFrame(
                columns=[
                    "date",
                    "ticker",
                    "open",
                    "high",
                    "low",
                    "close",
                    "adj_close",
                    "volume",
                ]
            )

        return normalize_price_frame(pd.concat(frames, ignore_index=True))


def _flatten_column_name(column: Any) -> str:
    """Convert plain or MultiIndex-style column names into lowercase snake_case."""

    if isinstance(column, tuple):
        parts = [
            str(part).strip().lower().replace(" ", "_")
            for part in column
            if str(part).strip() and str(part).strip().lower() != "none"
        ]
        return "_".join(parts)

    return str(column).strip().lower().replace(" ", "_")


def _find_first_column(
    columns: list[str],
    ticker: str | None,
    canonical_name: str,
    aliases: list[str] | None = None,
) -> str | None:
    """Find canonical, ticker-suffixed or ticker-prefixed column variants."""

    aliases = aliases or []
    ticker_lower = ticker.lower() if ticker else None

    candidates = [
        canonical_name,
        *aliases,
    ]

    if ticker_lower:
        candidates.extend(
            [
                f"{canonical_name}_{ticker_lower}",
                f"{ticker_lower}_{canonical_name}",
                *[f"{alias}_{ticker_lower}" for alias in aliases],
                *[f"{ticker_lower}_{alias}" for alias in aliases],
            ]
        )

    for candidate in candidates:
        if candidate in columns:
            return candidate

    return None


def normalize_price_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize yfinance/SDU_DataScienceTool OHLCV output.

    Handles examples such as:
    - open, high, low, close, volume
    - open_aapl, high_aapl, low_aapl, close_aapl, volume_aapl
    - aapl_open, aapl_high, aapl_low, aapl_close, aapl_volume
    - adj_close, adj_close_aapl, adjusted_close
    """

    if frame is None or frame.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "ticker",
                "open",
                "high",
                "low",
                "close",
                "adj_close",
                "volume",
            ]
        )

    renamed = frame.copy()

    # If date is in the index, preserve it.
    if renamed.index.name is not None:
        renamed = renamed.reset_index()

    renamed.columns = [_flatten_column_name(column) for column in renamed.columns]
    renamed = renamed.loc[:, ~renamed.columns.duplicated()]

    columns = list(renamed.columns)

    date_column = next(
        (
            candidate
            for candidate in ["date", "datetime", "timestamp", "price_date", "index"]
            if candidate in columns
        ),
        None,
    )

    if date_column is None:
        raise ValueError(
            "Could not find a date column in market data frame. "
            f"Available columns after adapter normalization: {columns}"
        )

    ticker_column = next(
        (
            candidate
            for candidate in ["ticker", "tic", "symbol"]
            if candidate in columns
        ),
        None,
    )

    inferred_ticker: str | None = None

    if ticker_column is not None:
        ticker_values = (
            renamed[ticker_column].dropna().astype(str).str.upper().unique().tolist()
        )

        if ticker_values:
            inferred_ticker = ticker_values[0]

    # Fallback: infer ticker from columns like open_aapl / close_msft.
    if inferred_ticker is None:
        for column in columns:
            for prefix in ["open_", "high_", "low_", "close_", "volume_", "adj_close_"]:
                if column.startswith(prefix) and len(column) > len(prefix):
                    inferred_ticker = column.replace(prefix, "").upper()
                    break
            if inferred_ticker:
                break

    if inferred_ticker is None:
        raise ValueError(
            "Could not find or infer ticker in market data frame. "
            f"Available columns after adapter normalization: {columns}"
        )

    ticker_lower = inferred_ticker.lower()

    open_column = _find_first_column(columns, ticker_lower, "open")
    high_column = _find_first_column(columns, ticker_lower, "high")
    low_column = _find_first_column(columns, ticker_lower, "low")
    close_column = _find_first_column(columns, ticker_lower, "close")
    volume_column = _find_first_column(columns, ticker_lower, "volume")
    adj_close_column = _find_first_column(
        columns,
        ticker_lower,
        "adj_close",
        aliases=["adjusted_close", "adjclose"],
    )

    required_map = {
        "open": open_column,
        "high": high_column,
        "low": low_column,
        "close": close_column,
        "volume": volume_column,
    }

    missing = [
        canonical_name
        for canonical_name, source_column in required_map.items()
        if source_column is None
    ]

    if missing:
        raise ValueError(
            f"Missing required OHLCV columns: {missing}. "
            f"Available columns after adapter normalization: {columns}"
        )

    if adj_close_column is None:
        adj_close_column = close_column

    normalized = pd.DataFrame(
        {
            "date": pd.to_datetime(renamed[date_column], errors="coerce").dt.date,
            "ticker": inferred_ticker,
            "open": pd.to_numeric(renamed[open_column], errors="coerce"),
            "high": pd.to_numeric(renamed[high_column], errors="coerce"),
            "low": pd.to_numeric(renamed[low_column], errors="coerce"),
            "close": pd.to_numeric(renamed[close_column], errors="coerce"),
            "adj_close": pd.to_numeric(renamed[adj_close_column], errors="coerce"),
            "volume": pd.to_numeric(renamed[volume_column], errors="coerce"),
        }
    )

    normalized = normalized.dropna(
        subset=[
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
    )

    if normalized.empty:
        raise ValueError(
            "Market data was returned, but no valid OHLCV rows remained after "
            f"adapter normalization. Available columns after adapter normalization: {columns}"
        )

    normalized["ticker"] = normalized["ticker"].astype(str).str.upper()
    normalized["volume"] = normalized["volume"].fillna(0).astype("int64")

    normalized = normalized.drop_duplicates(subset=["ticker", "date"])
    normalized = normalized.sort_values(["ticker", "date"]).reset_index(drop=True)

    return normalized[
        [
            "date",
            "ticker",
            "open",
            "high",
            "low",
            "close",
            "adj_close",
            "volume",
        ]
    ]
