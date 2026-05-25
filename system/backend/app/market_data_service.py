"""Market data service for ingestion, DuckDB persistence and FinRL exports."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any
import uuid

import pandas as pd
import yfinance as yf

from app.data_sources.sdu_datascience_adapter import (
    SDUDataScienceMarketAdapter,
    SDUDataScienceToolUnavailable,
)
from app.data_sources.yfinance_market_source import YFinanceMarketSource
from app.finrl_compat import to_finrl_price_frame
from app.market_data_repository import MarketDataRepository

REQUIRED_OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


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
    ticker: str,
    canonical_name: str,
    aliases: list[str],
) -> str | None:
    """
    Find the first matching source column for a canonical OHLCV field.

    Handles:
    - open
    - open_aapl
    - aapl_open
    - Open / AAPL from flattened yfinance MultiIndex
    """

    ticker_lower = ticker.lower()

    candidates = [
        canonical_name,
        f"{canonical_name}_{ticker_lower}",
        f"{ticker_lower}_{canonical_name}",
        *aliases,
        *[f"{alias}_{ticker_lower}" for alias in aliases],
        *[f"{ticker_lower}_{alias}" for alias in aliases],
    ]

    for candidate in candidates:
        if candidate in columns:
            return candidate

    return None


def normalize_market_data_frame(
    df: pd.DataFrame,
    ticker: str,
    source: str = "sdu-datascience-tool",
) -> pd.DataFrame:
    """
    Normalize SDU_DataScienceTool / yfinance-style market data into the
    canonical StockInvestmentDSS daily OHLCV schema.

    Canonical output columns:
    date, ticker, open, high, low, close, adjusted_close, volume, source, ingested_at
    """

    if df is None or df.empty:
        raise ValueError(f"No market data returned for ticker: {ticker}")

    ticker_upper = ticker.strip().upper()
    ticker_lower = ticker_upper.lower()

    if not ticker_upper:
        raise ValueError("Ticker is required")

    normalized = df.copy()

    # Always reset index if date might be stored there.
    normalized = normalized.reset_index()

    # Flatten any yfinance MultiIndex columns.
    normalized.columns = [_flatten_column_name(column) for column in normalized.columns]

    # Drop duplicate columns created by reset_index or yfinance variants.
    normalized = normalized.loc[:, ~normalized.columns.duplicated()]

    columns = list(normalized.columns)

    # Date can appear as date, datetime, index, or price_date variants.
    date_column = None
    for candidate in ["date", "datetime", "price_date", "index"]:
        if candidate in columns:
            date_column = candidate
            break

    if date_column is None:
        raise ValueError(
            "Missing required date column. "
            f"Available columns after normalization: {columns}"
        )

    field_columns = {
        "open": _find_first_column(columns, ticker_lower, "open", []),
        "high": _find_first_column(columns, ticker_lower, "high", []),
        "low": _find_first_column(columns, ticker_lower, "low", []),
        "close": _find_first_column(columns, ticker_lower, "close", []),
        "volume": _find_first_column(columns, ticker_lower, "volume", []),
        "adjusted_close": _find_first_column(
            columns,
            ticker_lower,
            "adjusted_close",
            ["adj_close", "adjclose"],
        ),
    }

    missing = [
        canonical_name
        for canonical_name in REQUIRED_OHLCV_COLUMNS
        if field_columns[canonical_name] is None
    ]

    if missing:
        raise ValueError(
            f"Missing required OHLCV columns: {missing}. "
            f"Available columns after normalization: {columns}"
        )

    output = pd.DataFrame(
        {
            "date": pd.to_datetime(normalized[date_column], errors="coerce").dt.date,
            "ticker": ticker_upper,
            "open": pd.to_numeric(normalized[field_columns["open"]], errors="coerce"),
            "high": pd.to_numeric(normalized[field_columns["high"]], errors="coerce"),
            "low": pd.to_numeric(normalized[field_columns["low"]], errors="coerce"),
            "close": pd.to_numeric(normalized[field_columns["close"]], errors="coerce"),
            "adjusted_close": pd.to_numeric(
                normalized[field_columns["adjusted_close"] or field_columns["close"]],
                errors="coerce",
            ),
            "volume": pd.to_numeric(
                normalized[field_columns["volume"]], errors="coerce"
            ),
            "source": source,
            "ingested_at": datetime.now(timezone.utc).isoformat(),
        }
    )

    output = output.dropna(subset=["date", "open", "high", "low", "close", "volume"])

    if output.empty:
        raise ValueError(
            "Market data was returned, but no valid OHLCV rows remained after "
            f"normalization. Available columns after normalization: {columns}"
        )

    output["volume"] = output["volume"].astype("int64")

    return output[
        [
            "date",
            "ticker",
            "open",
            "high",
            "low",
            "close",
            "adjusted_close",
            "volume",
            "source",
            "ingested_at",
        ]
    ]


class MarketDataService:
    """Application service used by FastAPI routes."""

    def __init__(self) -> None:
        self.repository = MarketDataRepository()

    async def ingest_ticker(
        self,
        ticker: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict[str, Any]:
        normalized_ticker = ticker.strip().upper()

        if not normalized_ticker:
            raise ValueError("Ticker is required")

        if not end_date:
            end_date = date.today().isoformat()

        if not start_date:
            start_date = (date.today() - timedelta(days=365 * 2)).isoformat()

        ingestion_id = str(uuid.uuid4())
        source_name = "sdu-datascience-tool:yahoo"
        fallback_message: str | None = None

        try:
            adapter = SDUDataScienceMarketAdapter()
            prices = await adapter.fetch_prices(
                [normalized_ticker],
                start_date,
                end_date,
            )
        except SDUDataScienceToolUnavailable as exc:
            source_name = "yfinance:fallback"
            adapter = YFinanceMarketSource()
            prices = await adapter.fetch_prices(
                [normalized_ticker],
                start_date,
                end_date,
            )
            fallback_message = str(exc)
        except Exception as exc:
            source_name = "yfinance:fallback-after-sdu-error"
            adapter = YFinanceMarketSource()
            prices = await adapter.fetch_prices(
                [normalized_ticker],
                start_date,
                end_date,
            )
            fallback_message = f"SDU_DataScienceTool call failed, fallback used: {exc}"

        if prices is None or prices.empty:
            self.repository.log_ingestion(
                ingestion_id=ingestion_id,
                source=source_name,
                ticker=normalized_ticker,
                start_date=start_date,
                end_date=end_date,
                rows_written=0,
                status="empty",
                message="No price rows returned",
            )

            return {
                "status": "empty",
                "ticker": normalized_ticker,
                "source": source_name,
                "rows_written": 0,
                "start_date": start_date,
                "end_date": end_date,
                "message": "No price rows returned",
            }

        canonical_prices = normalize_market_data_frame(
            df=prices,
            ticker=normalized_ticker,
            source=source_name,
        )

        metadata = self._fetch_symbol_metadata(normalized_ticker, source_name)
        self.repository.upsert_symbol(metadata)

        rows_written = self.repository.replace_prices_for_ticker(
            ticker=normalized_ticker,
            prices=canonical_prices,
            source=source_name,
        )

        self.repository.log_ingestion(
            ingestion_id=ingestion_id,
            source=source_name,
            ticker=normalized_ticker,
            start_date=start_date,
            end_date=end_date,
            rows_written=rows_written,
            status="ok",
            message=fallback_message,
        )

        return {
            "status": "ok",
            "ticker": normalized_ticker,
            "source": source_name,
            "rows_written": rows_written,
            "start_date": start_date,
            "end_date": end_date,
            "message": fallback_message,
        }

    def get_stock(self, ticker: str) -> dict[str, Any] | None:
        return self.repository.get_symbol(ticker)

    def get_prices(self, ticker: str, limit: int = 250) -> pd.DataFrame:
        return self.repository.get_prices(ticker, limit=limit)

    def search(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        return self.repository.search_symbols(query, limit=limit)

    def get_finrl_prices(self, ticker: str, limit: int = 250) -> pd.DataFrame:
        prices = self.get_prices(ticker, limit=limit)
        return to_finrl_price_frame(prices)

    def _fetch_symbol_metadata(self, ticker: str, source_name: str) -> dict[str, Any]:
        """
        Fetch lightweight ticker metadata.

        This is deliberately small; fundamentals/valuation are separate future tasks.
        """

        try:
            info = yf.Ticker(ticker).get_info()
        except Exception:
            info = {}

        return {
            "ticker": ticker,
            "name": info.get("shortName") or info.get("longName") or ticker,
            "exchange": info.get("exchange") or info.get("fullExchangeName"),
            "currency": info.get("currency"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "source": source_name,
        }
