# src/stock_investment_dss/data/market_data_paths.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from stock_investment_dss.utilities.paths import DATA_DIRECTORY

MARKET_DATA_DIRECTORY = DATA_DIRECTORY / "market"

DAILY_DIRECTORY = MARKET_DATA_DIRECTORY / "daily"
DAILY_RAW_DIRECTORY = DAILY_DIRECTORY / "raw"
DAILY_PROCESSED_DIRECTORY = DAILY_DIRECTORY / "processed"
DAILY_METADATA_DIRECTORY = DAILY_DIRECTORY / "metadata"


@dataclass(frozen=True)
class DailyMarketDataPaths:
    dataset_id: str
    raw_file: Path
    processed_file: Path
    metadata_file: Path


def ensure_market_data_directories() -> None:
    for directory in [
        MARKET_DATA_DIRECTORY,
        DAILY_DIRECTORY,
        DAILY_RAW_DIRECTORY,
        DAILY_PROCESSED_DIRECTORY,
        DAILY_METADATA_DIRECTORY,
    ]:
        directory.mkdir(parents=True, exist_ok=True)


def get_daily_market_data_paths(dataset_id: str) -> DailyMarketDataPaths:
    ensure_market_data_directories()

    safe_dataset_id = dataset_id.strip().lower()

    return DailyMarketDataPaths(
        dataset_id=safe_dataset_id,
        raw_file=DAILY_RAW_DIRECTORY / f"market_data_{safe_dataset_id}_1d_raw.csv",
        processed_file=DAILY_PROCESSED_DIRECTORY
        / f"market_data_{safe_dataset_id}_1d_finrl.csv",
        metadata_file=DAILY_METADATA_DIRECTORY
        / f"market_data_{safe_dataset_id}_1d_metadata.json",
    )
