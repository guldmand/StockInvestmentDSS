"""#166 research smoke test.

Goal:
- fetch daily stock data through SDU_DataScienceTool where possible
- fall back to yfinance if the local SDU package interface differs
- write CSV
- write Parquet
- write DuckDB
- reload from DuckDB
- convert to FinRL-compatible dataframe

No RL training is started by this script.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running from repo root without installing the local research package.
REPO_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_SRC = REPO_ROOT / "research" / "src"

if str(RESEARCH_SRC) not in sys.path:
    sys.path.insert(0, str(RESEARCH_SRC))

from stockinvestmentdss_research.data.market_data_loader import (  # noqa: E402
    MarketDataSmokeConfig,
    fetch_with_sdu_datascience_tool,
    read_market_data_from_duckdb,
    to_finrl_price_frame,
    write_market_data_artifacts,
    write_market_data_to_duckdb,
)


def main() -> None:
    config = MarketDataSmokeConfig()

    print("#166 research smoke test")
    print(f"tickers:     {', '.join(config.tickers)}")
    print(f"start_date:  {config.start_date}")
    print(f"end_date:    {config.end_date}")
    print(f"duckdb_path: {config.duckdb_path}")
    print()

    prices, source = fetch_with_sdu_datascience_tool(
        tickers=config.tickers,
        start_date=config.start_date,
        end_date=config.end_date,
    )

    if prices.empty:
        raise SystemExit("No market data was returned.")

    print(f"source used: {source}")
    print(f"rows fetched: {len(prices)}")
    print(f"columns: {list(prices.columns)}")
    print()

    artifact_paths = write_market_data_artifacts(
        prices=prices,
        artifact_dir=config.artifact_dir,
    )

    rows_written = write_market_data_to_duckdb(
        prices=prices,
        duckdb_path=config.duckdb_path,
    )

    reloaded = read_market_data_from_duckdb(
        duckdb_path=config.duckdb_path,
        tickers=config.tickers,
    )

    finrl_frame = to_finrl_price_frame(reloaded)

    required_finrl_columns = {"date", "open", "high", "low", "close", "volume", "tic", "day"}
    missing = required_finrl_columns.difference(finrl_frame.columns)

    if missing:
        raise SystemExit(f"FinRL-compatible frame is missing columns: {sorted(missing)}")

    print("Artifacts written:")
    for name, path in artifact_paths.items():
        print(f"- {name}: {path}")

    print()
    print(f"DuckDB rows written: {rows_written}")
    print(f"DuckDB rows reloaded: {len(reloaded)}")
    print(f"FinRL-compatible rows: {len(finrl_frame)}")
    print(f"FinRL-compatible columns: {list(finrl_frame.columns)}")
    print()
    print("FinRL-compatible sample:")
    print(finrl_frame.head(8).to_string(index=False))
    print()
    print("OK: research smoke test passed.")


if __name__ == "__main__":
    main()
