"""FastAPI market data endpoints."""

from __future__ import annotations

from datetime import date
from typing import Any

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

from app.market_data_service import MarketDataService

router = APIRouter(prefix="/market", tags=["Market Data"])


class IngestResponse(BaseModel):
    status: str
    ticker: str
    source: str
    rows_written: int
    start_date: str | None = None
    end_date: str | None = None
    message: str | None = None


class StockResponse(BaseModel):
    ticker: str
    name: str | None = None
    exchange: str | None = None
    currency: str | None = None
    sector: str | None = None
    industry: str | None = None
    source: str
    first_seen_at: Any | None = None
    updated_at: Any | None = None


class PriceRow(BaseModel):
    ticker: str
    price_date: Any
    open: float
    high: float
    low: float
    close: float
    adj_close: float | None = None
    volume: int
    source: str
    ingested_at: Any


class PricesResponse(BaseModel):
    ticker: str
    count: int
    rows: list[PriceRow]


class FinRLPricesResponse(BaseModel):
    ticker: str
    count: int
    columns: list[str]
    rows: list[dict[str, Any]]


@router.post("/ingest/{ticker}", response_model=IngestResponse)
async def ingest_ticker(
    ticker: str,
    start_date: str | None = Query(default=None, description="YYYY-MM-DD"),
    end_date: str | None = Query(default=None, description="YYYY-MM-DD"),
) -> IngestResponse:
    """Ingest daily OHLCV prices through SDU_DataScienceTool/yfinance into DuckDB."""

    service = MarketDataService()

    try:
        result = await service.ingest_ticker(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Market ingestion failed: {exc}",
        ) from exc

    return IngestResponse(**result)


@router.get("/search")
def search_symbols(
    q: str = Query(min_length=1, max_length=32),
    limit: int = Query(default=20, ge=1, le=50),
) -> dict[str, Any]:
    """Search symbols already known to the local DuckDB market data store."""

    service = MarketDataService()
    rows = service.search(q, limit=limit)

    return {
        "query": q,
        "count": len(rows),
        "rows": rows,
    }


@router.get("/stocks/{ticker}", response_model=StockResponse)
def get_stock(ticker: str) -> StockResponse:
    """Return stored metadata for one stock symbol."""

    service = MarketDataService()
    stock = service.get_stock(ticker)

    if not stock:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"{ticker.upper()} not found. Run POST /market/ingest/{ticker.upper()} first.",
        )

    return StockResponse(**stock)


@router.get("/prices/{ticker}", response_model=PricesResponse)
def get_prices(
    ticker: str,
    limit: int = Query(default=250, ge=1, le=5000),
) -> PricesResponse:
    """Return stored daily OHLCV prices for one ticker."""

    service = MarketDataService()
    frame = service.get_prices(ticker, limit=limit)

    rows = frame.to_dict(orient="records") if not frame.empty else []

    return PricesResponse(
        ticker=ticker.upper(),
        count=len(rows),
        rows=rows,
    )


@router.get("/finrl/{ticker}", response_model=FinRLPricesResponse)
def get_finrl_prices(
    ticker: str,
    limit: int = Query(default=250, ge=1, le=5000),
) -> FinRLPricesResponse:
    """Return stored prices in a FinRL-compatible dataframe layout.

    This endpoint is a smoke-test/export surface only. It does not train RL agents.
    """

    service = MarketDataService()
    frame = service.get_finrl_prices(ticker, limit=limit)

    return FinRLPricesResponse(
        ticker=ticker.upper(),
        count=len(frame),
        columns=list(frame.columns),
        rows=frame.to_dict(orient="records"),
    )
