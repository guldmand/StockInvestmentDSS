"""SDU_DataScienceTool adapter for StockInvestmentDSS market data.

The PoC uses SDU_DataScienceTool as the primary adapter boundary for API/data
source access. If the package import or call fails in local development, the
service can fall back to a plain yfinance-compatible source without changing
the backend API contract.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from app.data_sources.yfinance_market_source import normalize_price_frame


class SDUDataScienceToolUnavailable(RuntimeError):
    """Raised when SDU_DataScienceTool is not available in the runtime."""


class SDUDataScienceMarketAdapter:
    """Adapter around SDU_DataScienceTool's YahooFinanceSource."""

    source_name = "sdu-datascience-tool:yahoo"

    def __init__(self) -> None:
        try:
            from sdu_dst.sources.yahoo import YahooFinanceSource
        except Exception as exc:  # pragma: no cover - depends on external package
            raise SDUDataScienceToolUnavailable(
                "Could not import sdu_dst.sources.yahoo.YahooFinanceSource. "
                "Install SDU_DataScienceTool or use the service fallback."
            ) from exc

        self._source = YahooFinanceSource()

    async def fetch_prices(
        self,
        tickers: Iterable[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Fetch daily prices through SDU_DataScienceTool and normalize output."""

        normalized_tickers = [
            ticker.strip().upper()
            for ticker in tickers
            if ticker and ticker.strip()
        ]

        if not normalized_tickers:
            return pd.DataFrame()

        frame = await self._source.fetch_prices(
            normalized_tickers,
            start_date,
            end_date,
        )

        return normalize_price_frame(frame)
