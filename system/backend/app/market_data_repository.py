"""DuckDB repository for market data foundation."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from app.config import settings


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def get_runtime_data_path() -> Path:
    runtime_path = Path(settings.runtime_data_path)
    runtime_path.mkdir(parents=True, exist_ok=True)
    return runtime_path


def get_duckdb_path() -> Path:
    duckdb_path = Path(settings.duckdb_path)
    duckdb_path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb_path


def get_connection() -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(get_duckdb_path()))


def ensure_market_data_schema() -> None:
    """Create market data tables if they do not already exist."""

    get_runtime_data_path()
    connection = get_connection()

    try:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS market_symbols (
                ticker VARCHAR PRIMARY KEY,
                name VARCHAR,
                exchange VARCHAR,
                currency VARCHAR,
                sector VARCHAR,
                industry VARCHAR,
                source VARCHAR NOT NULL,
                first_seen_at TIMESTAMP NOT NULL,
                updated_at TIMESTAMP NOT NULL
            );
            """)

        connection.execute("""
            CREATE TABLE IF NOT EXISTS market_prices_daily (
                ticker VARCHAR NOT NULL,
                price_date DATE NOT NULL,
                open DOUBLE NOT NULL,
                high DOUBLE NOT NULL,
                low DOUBLE NOT NULL,
                close DOUBLE NOT NULL,
                adj_close DOUBLE,
                volume BIGINT NOT NULL,
                source VARCHAR NOT NULL,
                ingested_at TIMESTAMP NOT NULL,
                PRIMARY KEY (ticker, price_date)
            );
            """)

        connection.execute("""
            CREATE TABLE IF NOT EXISTS data_ingestion_log (
                ingestion_id VARCHAR PRIMARY KEY,
                source VARCHAR NOT NULL,
                ticker VARCHAR NOT NULL,
                start_date DATE NOT NULL,
                end_date DATE NOT NULL,
                rows_written INTEGER NOT NULL,
                status VARCHAR NOT NULL,
                message VARCHAR,
                created_at TIMESTAMP NOT NULL
            );
            """)
    finally:
        connection.close()


def _prepare_prices_frame(
    ticker: str,
    prices: pd.DataFrame,
    source: str,
) -> pd.DataFrame:
    """Normalize service-level price data into repository/DuckDB schema."""

    if prices is None or prices.empty:
        return pd.DataFrame()

    normalized_ticker = ticker.strip().upper()
    frame = prices.copy()

    frame.columns = [
        str(column).strip().lower().replace(" ", "_") for column in frame.columns
    ]

    # Accept both service naming and repository naming.
    rename_map = {
        "date": "price_date",
        "datetime": "price_date",
        "adjusted_close": "adj_close",
        "adjclose": "adj_close",
    }

    frame = frame.rename(
        columns={
            source_column: target_column
            for source_column, target_column in rename_map.items()
            if source_column in frame.columns
        }
    )

    required_columns = [
        "ticker",
        "price_date",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    missing = [column for column in required_columns if column not in frame.columns]

    if missing:
        raise ValueError(
            f"Missing required repository price columns: {missing}. "
            f"Available columns: {list(frame.columns)}"
        )

    if "adj_close" not in frame.columns:
        frame["adj_close"] = frame["close"]

    frame["ticker"] = frame["ticker"].astype(str).str.upper()
    frame = frame[frame["ticker"] == normalized_ticker].copy()

    if frame.empty:
        return frame

    frame["price_date"] = pd.to_datetime(
        frame["price_date"],
        errors="raise",
    ).dt.date

    for column in ["open", "high", "low", "close", "adj_close"]:
        frame[column] = pd.to_numeric(frame[column], errors="raise")

    frame["volume"] = pd.to_numeric(frame["volume"], errors="raise").astype("int64")
    frame["source"] = source
    frame["ingested_at"] = now_utc()

    frame = frame.drop_duplicates(subset=["ticker", "price_date"])
    frame = frame.sort_values(["ticker", "price_date"]).reset_index(drop=True)

    return frame[
        [
            "ticker",
            "price_date",
            "open",
            "high",
            "low",
            "close",
            "adj_close",
            "volume",
            "source",
            "ingested_at",
        ]
    ]


class MarketDataRepository:
    """Repository for symbols, daily prices and ingestion logs."""

    def __init__(self) -> None:
        ensure_market_data_schema()

    def upsert_symbol(self, metadata: dict[str, Any]) -> None:
        ticker = str(metadata["ticker"]).upper()
        timestamp = now_utc()

        connection = get_connection()
        try:
            existing = connection.execute(
                "SELECT ticker FROM market_symbols WHERE ticker = ?",
                [ticker],
            ).fetchone()

            if existing:
                connection.execute(
                    """
                    UPDATE market_symbols
                    SET
                        name = ?,
                        exchange = ?,
                        currency = ?,
                        sector = ?,
                        industry = ?,
                        source = ?,
                        updated_at = ?
                    WHERE ticker = ?
                    """,
                    [
                        metadata.get("name"),
                        metadata.get("exchange"),
                        metadata.get("currency"),
                        metadata.get("sector"),
                        metadata.get("industry"),
                        metadata.get("source", "unknown"),
                        timestamp,
                        ticker,
                    ],
                )
            else:
                connection.execute(
                    """
                    INSERT INTO market_symbols (
                        ticker,
                        name,
                        exchange,
                        currency,
                        sector,
                        industry,
                        source,
                        first_seen_at,
                        updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        ticker,
                        metadata.get("name"),
                        metadata.get("exchange"),
                        metadata.get("currency"),
                        metadata.get("sector"),
                        metadata.get("industry"),
                        metadata.get("source", "unknown"),
                        timestamp,
                        timestamp,
                    ],
                )
        finally:
            connection.close()

    def replace_prices_for_ticker(
        self,
        ticker: str,
        prices: pd.DataFrame,
        source: str,
    ) -> int:
        normalized_ticker = ticker.strip().upper()
        frame = _prepare_prices_frame(
            ticker=normalized_ticker,
            prices=prices,
            source=source,
        )

        if frame.empty:
            return 0

        connection = get_connection()
        registered = False

        try:
            min_date = frame["price_date"].min()
            max_date = frame["price_date"].max()

            connection.execute("BEGIN TRANSACTION")

            connection.execute(
                """
                DELETE FROM market_prices_daily
                WHERE ticker = ?
                  AND price_date BETWEEN ? AND ?
                """,
                [normalized_ticker, min_date, max_date],
            )

            connection.register("prices_df", frame)
            registered = True

            connection.execute("""
                INSERT INTO market_prices_daily (
                    ticker,
                    price_date,
                    open,
                    high,
                    low,
                    close,
                    adj_close,
                    volume,
                    source,
                    ingested_at
                )
                SELECT
                    ticker,
                    price_date,
                    open,
                    high,
                    low,
                    close,
                    adj_close,
                    volume,
                    source,
                    ingested_at
                FROM prices_df
                """)

            connection.execute("COMMIT")
            return int(len(frame))

        except Exception:
            # DuckDB can already have aborted the transaction after a failed
            # statement. In that case ROLLBACK itself raises
            # "no transaction is active", so suppress rollback errors and
            # re-raise the real original exception.
            try:
                connection.execute("ROLLBACK")
            except Exception:
                pass

            raise

        finally:
            if registered:
                try:
                    connection.unregister("prices_df")
                except Exception:
                    pass

            connection.close()

    def log_ingestion(
        self,
        ingestion_id: str,
        source: str,
        ticker: str,
        start_date: str,
        end_date: str,
        rows_written: int,
        status: str,
        message: str | None = None,
    ) -> None:
        connection = get_connection()
        try:
            connection.execute(
                """
                INSERT INTO data_ingestion_log (
                    ingestion_id,
                    source,
                    ticker,
                    start_date,
                    end_date,
                    rows_written,
                    status,
                    message,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    ingestion_id,
                    source,
                    ticker.upper(),
                    start_date,
                    end_date,
                    rows_written,
                    status,
                    message,
                    now_utc(),
                ],
            )
        finally:
            connection.close()

    def get_symbol(self, ticker: str) -> dict[str, Any] | None:
        connection = get_connection()
        try:
            row = connection.execute(
                """
                SELECT
                    ticker,
                    name,
                    exchange,
                    currency,
                    sector,
                    industry,
                    source,
                    first_seen_at,
                    updated_at
                FROM market_symbols
                WHERE ticker = ?
                LIMIT 1
                """,
                [ticker.upper()],
            ).fetchone()
        finally:
            connection.close()

        if not row:
            return None

        return {
            "ticker": row[0],
            "name": row[1],
            "exchange": row[2],
            "currency": row[3],
            "sector": row[4],
            "industry": row[5],
            "source": row[6],
            "first_seen_at": row[7],
            "updated_at": row[8],
        }

    def search_symbols(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        normalized_query = f"%{query.strip().upper()}%"

        connection = get_connection()
        try:
            rows = connection.execute(
                """
                SELECT
                    ticker,
                    name,
                    exchange,
                    currency,
                    sector,
                    industry,
                    source
                FROM market_symbols
                WHERE ticker ILIKE ?
                   OR COALESCE(name, '') ILIKE ?
                ORDER BY ticker
                LIMIT ?
                """,
                [normalized_query, normalized_query, limit],
            ).fetchall()
        finally:
            connection.close()

        return [
            {
                "ticker": row[0],
                "name": row[1],
                "exchange": row[2],
                "currency": row[3],
                "sector": row[4],
                "industry": row[5],
                "source": row[6],
            }
            for row in rows
        ]

    def get_prices(self, ticker: str, limit: int = 250) -> pd.DataFrame:
        connection = get_connection()
        try:
            frame = connection.execute(
                """
                SELECT
                    ticker,
                    price_date,
                    open,
                    high,
                    low,
                    close,
                    adj_close,
                    volume,
                    source,
                    ingested_at
                FROM market_prices_daily
                WHERE ticker = ?
                ORDER BY price_date DESC
                LIMIT ?
                """,
                [ticker.upper(), limit],
            ).df()
        finally:
            connection.close()

        if frame.empty:
            return frame

        return frame.sort_values("price_date").reset_index(drop=True)
