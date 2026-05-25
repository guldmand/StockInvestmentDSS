# src/stock_investment_dss/data/ticker_universes.py

from __future__ import annotations

import os

DEMO_2_TICKERS = ("AAPL", "MSFT")
DEMO_5_TICKERS = ("AAPL", "MSFT", "NVDA", "AMZN", "GOOGL")
DEMO_10_TICKERS = (
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOGL",
    "META",
    "TSLA",
    "JPM",
    "UNH",
    "XOM",
)
DEMO_30_TICKERS = (
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOGL",
    "META",
    "TSLA",
    "AVGO",
    "BRK-B",
    "LLY",
    "JPM",
    "V",
    "UNH",
    "XOM",
    "MA",
    "COST",
    "HD",
    "PG",
    "JNJ",
    "NFLX",
    "ABBV",
    "BAC",
    "KO",
    "CRM",
    "AMD",
    "PEP",
    "WMT",
    "ADBE",
    "CSCO",
    "TMO",
)


PREDEFINED_UNIVERSES: dict[str, tuple[str, ...]] = {
    "demo_2": DEMO_2_TICKERS,
    "demo_5": DEMO_5_TICKERS,
    "demo_10": DEMO_10_TICKERS,
    "demo_30": DEMO_30_TICKERS,
}


def _parse_ticker_list(value: str | None) -> tuple[str, ...]:
    if value is None or value.strip() == "":
        return tuple()

    tickers: list[str] = []
    seen: set[str] = set()

    for raw_ticker in value.replace(";", ",").split(","):
        ticker = raw_ticker.strip().upper()
        if not ticker or ticker in seen:
            continue
        tickers.append(ticker)
        seen.add(ticker)

    return tuple(tickers)


def get_ticker_universe(universe_id: str) -> tuple[str, ...]:
    """Return tickers for a named or explicitly supplied universe.

    Explicit environment ticker lists have priority so experiment runners can
    define ad-hoc universes without editing source code:

    - STOCK_INVESTMENT_DSS_FINRL_TICKERS
    - STOCK_INVESTMENT_DSS_DAILY_DATA_TICKERS

    Named presets remain available for reproducible runs:
    demo_2, demo_5, demo_10, demo_30.
    """

    explicit_tickers = _parse_ticker_list(
        os.getenv("STOCK_INVESTMENT_DSS_FINRL_TICKERS")
        or os.getenv("STOCK_INVESTMENT_DSS_DAILY_DATA_TICKERS")
    )
    if explicit_tickers:
        return explicit_tickers

    normalized_universe_id = universe_id.strip().lower()

    if normalized_universe_id == "custom":
        raise ValueError(
            "universe_id='custom' requires STOCK_INVESTMENT_DSS_FINRL_TICKERS "
            "or STOCK_INVESTMENT_DSS_DAILY_DATA_TICKERS."
        )

    try:
        return PREDEFINED_UNIVERSES[normalized_universe_id]
    except KeyError as exc:
        available = ", ".join(sorted(PREDEFINED_UNIVERSES.keys()))
        raise ValueError(
            f"Unknown universe_id='{universe_id}'. Available: {available}. "
            "For an ad-hoc universe, set STOCK_INVESTMENT_DSS_FINRL_TICKERS."
        ) from exc
