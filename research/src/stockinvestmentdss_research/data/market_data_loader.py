"""Market data loader for research experiments.

This module is intentionally independent from the running FastAPI app.

It supports the #166 research smoke test:

SDU_DataScienceTool/yfinance-compatible data
-> CSV
-> Parquet
-> DuckDB
-> reload from DuckDB
-> FinRL-compatible dataframe

The output schema for FinRL-compatible prices is:

date, open, high, low, close, volume, tic, day

No RL training is started here.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import duckdb
import pandas as pd


OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


@dataclass(frozen=True)
class MarketDataSmokeConfig:
    """Configuration for the research market-data smoke test."""

    tickers: tuple[str, ...] = ("AAPL", "MSFT", "SPY", "NVDA")
    start_date: str = "2025-01-01"
    end_date: str = "2026-01-01"
    artifact_dir: Path = Path("research/experiments/artifacts/market_data_smoke")
    duckdb_path: Path = Path("system/runtime-data/market_research.duckdb")


def flatten_column_name(column: Any) -> str:
    """Convert yfinance/SDU multi-index columns to lowercase snake_case."""

    if isinstance(column, tuple):
        parts = [
            str(part).strip().lower().replace(" ", "_")
            for part in column
            if str(part).strip() and str(part).strip().lower() != "none"
        ]
        return "_".join(parts)

    return str(column).strip().lower().replace(" ", "_")


def find_column(
    columns: list[str],
    ticker: str,
    canonical_name: str,
    aliases: list[str] | None = None,
) -> str | None:
    """Find a canonical, ticker-prefixed or ticker-suffixed column."""

    aliases = aliases or []
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


def infer_ticker(columns: list[str], fallback_ticker: str) -> str:
    """Infer ticker from columns such as open_aapl or aapl_open."""

    fallback = fallback_ticker.strip().upper()
    fallback_lower = fallback.lower()

    # If the requested ticker is visible in a column, trust it.
    for column in columns:
        if column.endswith(f"_{fallback_lower}") or column.startswith(f"{fallback_lower}_"):
            return fallback

    # Generic inference from either price_ticker or ticker_price.
    price_names = ["open", "high", "low", "close", "volume", "adj_close", "adjusted_close"]
    for column in columns:
        for price_name in price_names:
            prefix = f"{price_name}_"
            suffix = f"_{price_name}"

            if column.startswith(prefix) and len(column) > len(prefix):
                return column.replace(prefix, "").upper()

            if column.endswith(suffix) and len(column) > len(suffix):
                return column.replace(suffix, "").upper()

    return fallback


def normalize_price_frame(frame: pd.DataFrame, ticker: str, source: str) -> pd.DataFrame:
    """Normalize one ticker's raw market data to the shared research schema."""

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
                "source",
                "ingested_at",
            ]
        )

    normalized = frame.copy()

    # Preserve index dates when needed.
    if normalized.index.name is not None:
        normalized = normalized.reset_index()

    normalized.columns = [flatten_column_name(column) for column in normalized.columns]
    normalized = normalized.loc[:, ~normalized.columns.duplicated()]

    columns = list(normalized.columns)
    inferred_ticker = infer_ticker(columns, ticker)

    date_column = next(
        (
            candidate
            for candidate in ["date", "datetime", "timestamp", "price_date", "index"]
            if candidate in columns
        ),
        None,
    )

    if date_column is None:
        raise ValueError(f"Missing date column. Available columns: {columns}")

    column_map = {
        "open": find_column(columns, inferred_ticker, "open"),
        "high": find_column(columns, inferred_ticker, "high"),
        "low": find_column(columns, inferred_ticker, "low"),
        "close": find_column(columns, inferred_ticker, "close"),
        "volume": find_column(columns, inferred_ticker, "volume"),
        "adj_close": find_column(
            columns,
            inferred_ticker,
            "adj_close",
            aliases=["adjusted_close", "adjclose"],
        ),
    }

    missing = [column for column in OHLCV_COLUMNS if column_map[column] is None]
    if missing:
        raise ValueError(
            f"Missing OHLCV columns for {inferred_ticker}: {missing}. "
            f"Available columns: {columns}"
        )

    if column_map["adj_close"] is None:
        column_map["adj_close"] = column_map["close"]

    output = pd.DataFrame(
        {
            "date": pd.to_datetime(normalized[date_column], errors="coerce").dt.date,
            "ticker": inferred_ticker,
            "open": pd.to_numeric(normalized[column_map["open"]], errors="coerce"),
            "high": pd.to_numeric(normalized[column_map["high"]], errors="coerce"),
            "low": pd.to_numeric(normalized[column_map["low"]], errors="coerce"),
            "close": pd.to_numeric(normalized[column_map["close"]], errors="coerce"),
            "adj_close": pd.to_numeric(normalized[column_map["adj_close"]], errors="coerce"),
            "volume": pd.to_numeric(normalized[column_map["volume"]], errors="coerce"),
            "source": source,
            "ingested_at": datetime.now(timezone.utc).isoformat(),
        }
    )

    output = output.dropna(subset=["date", "open", "high", "low", "close", "volume"])
    output["ticker"] = output["ticker"].astype(str).str.upper()
    output["volume"] = output["volume"].fillna(0).astype("int64")

    return output.sort_values(["ticker", "date"]).reset_index(drop=True)


def fetch_with_sdu_datascience_tool(
    tickers: Iterable[str],
    start_date: str,
    end_date: str,
) -> tuple[pd.DataFrame, str]:
    """Fetch market data through SDU_DataScienceTool when available.

    The package has evolved across course projects, so this function tries the
    public package first and falls back cleanly to yfinance if the local package
    interface differs.
    """

    frames: list[pd.DataFrame] = []

    try:
        # This import path matches the package intent. If the package interface
        # changes, the exception below will trigger the yfinance fallback.
        from sdu_datascience_tool.data_sources.yahoo_finance_source import YahooFinanceSource  # type: ignore

        source = YahooFinanceSource()

        for ticker in tickers:
            raw = source.fetch_prices(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
            )
            frames.append(
                normalize_price_frame(
                    frame=raw,
                    ticker=ticker,
                    source="sdu-datascience-tool:yahoo",
                )
            )

        if frames:
            return pd.concat(frames, ignore_index=True), "sdu-datascience-tool:yahoo"

    except Exception:
        # Keep this smoke test robust across machines and package versions.
        pass

    return fetch_with_yfinance(tickers, start_date, end_date)


def fetch_with_yfinance(
    tickers: Iterable[str],
    start_date: str,
    end_date: str,
) -> tuple[pd.DataFrame, str]:
    """Fetch daily OHLCV with yfinance fallback."""

    import yfinance as yf

    frames: list[pd.DataFrame] = []

    for ticker in tickers:
        normalized_ticker = ticker.strip().upper()

        raw = yf.download(
            normalized_ticker,
            start=start_date,
            end=end_date,
            interval="1d",
            auto_adjust=False,
            progress=False,
            group_by="column",
            threads=False,
        )

        if raw is None or raw.empty:
            continue

        raw = raw.reset_index()
        raw["ticker"] = normalized_ticker

        frames.append(
            normalize_price_frame(
                frame=raw,
                ticker=normalized_ticker,
                source="yfinance:fallback",
            )
        )

    if not frames:
        return pd.DataFrame(), "yfinance:fallback"

    return pd.concat(frames, ignore_index=True), "yfinance:fallback"


def to_finrl_price_frame(prices: pd.DataFrame) -> pd.DataFrame:
    """Convert shared market data to a FinRL-compatible price dataframe."""

    if prices.empty:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume", "tic", "day"])

    frame = prices.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["tic"] = frame["ticker"].astype(str).str.upper()
    frame["day"] = frame["date"].dt.dayofweek
    frame["date"] = frame["date"].dt.strftime("%Y-%m-%d")

    return frame[
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
    ].sort_values(["date", "tic"]).reset_index(drop=True)


def write_market_data_artifacts(
    prices: pd.DataFrame,
    artifact_dir: Path,
) -> dict[str, Path]:
    """Write CSV and Parquet artifacts."""

    artifact_dir.mkdir(parents=True, exist_ok=True)

    csv_path = artifact_dir / "market_prices_daily.csv"
    parquet_path = artifact_dir / "market_prices_daily.parquet"
    finrl_csv_path = artifact_dir / "finrl_prices.csv"

    prices.to_csv(csv_path, index=False)
    prices.to_parquet(parquet_path, index=False)
    to_finrl_price_frame(prices).to_csv(finrl_csv_path, index=False)

    return {
        "csv": csv_path,
        "parquet": parquet_path,
        "finrl_csv": finrl_csv_path,
    }


def write_market_data_to_duckdb(prices: pd.DataFrame, duckdb_path: Path) -> int:
    """Persist smoke-test market data to DuckDB."""

    duckdb_path.parent.mkdir(parents=True, exist_ok=True)

    frame = prices.copy()
    frame["price_date"] = pd.to_datetime(frame["date"]).dt.date

    insert_frame = frame[
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

    with duckdb.connect(str(duckdb_path)) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS research_market_prices_daily (
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
            """
        )

        connection.register("research_prices_df", insert_frame)

        connection.execute(
            """
            DELETE FROM research_market_prices_daily
            WHERE ticker IN (SELECT DISTINCT ticker FROM research_prices_df)
            """
        )

        connection.execute(
            """
            INSERT INTO research_market_prices_daily
            SELECT * FROM research_prices_df
            """
        )

    return int(len(insert_frame))


def read_market_data_from_duckdb(duckdb_path: Path, tickers: Iterable[str]) -> pd.DataFrame:
    """Read smoke-test prices back from DuckDB."""

    ticker_list = [ticker.strip().upper() for ticker in tickers]

    with duckdb.connect(str(duckdb_path)) as connection:
        return connection.execute(
            """
            SELECT
                ticker,
                price_date AS date,
                open,
                high,
                low,
                close,
                adj_close,
                volume,
                source,
                ingested_at
            FROM research_market_prices_daily
            WHERE ticker IN (SELECT * FROM UNNEST(?))
            ORDER BY price_date, ticker
            """,
            [ticker_list],
        ).df()
