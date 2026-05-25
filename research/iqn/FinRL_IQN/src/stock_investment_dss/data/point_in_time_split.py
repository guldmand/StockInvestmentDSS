# src/stock_investment_dss/data/point_in_time_split.py

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import pandas as pd

REQUIRED_POINT_IN_TIME_COLUMNS = {"date", "tic", "close"}


@dataclass(frozen=True)
class PointInTimeSplitResult:
    split_id: str
    point_in_time: str
    train_start_date: str | None
    train_end_date: str
    trade_start_date: str
    trade_end_date: str | None
    train_data: pd.DataFrame
    trade_data: pd.DataFrame
    metadata: dict[str, Any]


def _normalize_market_data(data: pd.DataFrame) -> pd.DataFrame:
    frame = data.copy()

    missing_columns = REQUIRED_POINT_IN_TIME_COLUMNS - set(frame.columns)
    if missing_columns:
        raise ValueError(
            "Point-in-time split data is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["tic"] = frame["tic"].astype(str).str.upper().str.strip()

    frame = frame.dropna(subset=["date", "tic"])
    frame = frame.sort_values(["date", "tic"]).reset_index(drop=True)

    if frame.empty:
        raise ValueError("Point-in-time split input data is empty after normalization.")

    return frame


def _format_date(value: pd.Timestamp | None) -> str | None:
    if value is None:
        return None

    return value.strftime("%Y-%m-%d")


def validate_point_in_time_split(
    train_data: pd.DataFrame,
    trade_data: pd.DataFrame,
    point_in_time: str,
    expected_tickers: tuple[str, ...] | None = None,
    min_tickers_per_date: int | None = None,
) -> None:
    if train_data.empty:
        raise ValueError("Point-in-time train split is empty.")

    if trade_data.empty:
        raise ValueError("Point-in-time trade/simulation split is empty.")

    pit_timestamp = pd.to_datetime(point_in_time)

    max_train_date = train_data["date"].max()
    min_trade_date = trade_data["date"].min()

    if max_train_date >= pit_timestamp:
        raise ValueError(
            "Point-in-time leakage detected: train data contains rows on or after "
            f"point_in_time={point_in_time}. max_train_date={max_train_date}"
        )

    if min_trade_date < pit_timestamp:
        raise ValueError(
            "Point-in-time split error: trade data contains rows before "
            f"point_in_time={point_in_time}. min_trade_date={min_trade_date}"
        )

    if expected_tickers is not None:
        expected = {ticker.upper().strip() for ticker in expected_tickers}
        train_tickers = set(train_data["tic"].unique().tolist())
        trade_tickers = set(trade_data["tic"].unique().tolist())

        missing_train = expected - train_tickers
        missing_trade = expected - trade_tickers

        if missing_train:
            raise ValueError(
                f"Train split is missing expected tickers: {sorted(missing_train)}"
            )

        if missing_trade:
            raise ValueError(
                f"Trade split is missing expected tickers: {sorted(missing_trade)}"
            )

    if min_tickers_per_date is not None and min_tickers_per_date > 0:
        train_counts = train_data.groupby("date")["tic"].nunique()
        trade_counts = trade_data.groupby("date")["tic"].nunique()

        bad_train_dates = train_counts[train_counts < min_tickers_per_date]
        bad_trade_dates = trade_counts[trade_counts < min_tickers_per_date]

        if not bad_train_dates.empty:
            raise ValueError(
                "Train split has dates with too few tickers. "
                f"Required={min_tickers_per_date}. "
                f"Bad dates={bad_train_dates.to_dict()}"
            )

        if not bad_trade_dates.empty:
            raise ValueError(
                "Trade split has dates with too few tickers. "
                f"Required={min_tickers_per_date}. "
                f"Bad dates={bad_trade_dates.to_dict()}"
            )


def create_point_in_time_split(
    data: pd.DataFrame,
    split_id: str,
    point_in_time: str,
    trade_end_date: str | None = None,
    expected_tickers: tuple[str, ...] | None = None,
    min_tickers_per_date: int | None = None,
    source_metadata: dict[str, Any] | None = None,
) -> PointInTimeSplitResult:
    """
    Create a simple point-in-time split.

    Semantics:
        train_data = rows where date < point_in_time
        trade_data = rows where date >= point_in_time

    If trade_end_date is provided:
        trade_data = rows where point_in_time <= date < trade_end_date

    This mirrors the V1 logic but works directly on the processed FinRL dataframe
    returned by the current data pipeline.
    """
    frame = _normalize_market_data(data)

    pit_timestamp = pd.to_datetime(point_in_time)

    if trade_end_date is not None:
        trade_end_timestamp = pd.to_datetime(trade_end_date)
    else:
        trade_end_timestamp = None

    train_data = frame[frame["date"] < pit_timestamp].copy()

    if trade_end_timestamp is None:
        trade_data = frame[frame["date"] >= pit_timestamp].copy()
    else:
        trade_data = frame[
            (frame["date"] >= pit_timestamp) & (frame["date"] < trade_end_timestamp)
        ].copy()

    validate_point_in_time_split(
        train_data=train_data,
        trade_data=trade_data,
        point_in_time=point_in_time,
        expected_tickers=expected_tickers,
        min_tickers_per_date=min_tickers_per_date,
    )

    train_start = train_data["date"].min()
    train_end = train_data["date"].max()
    trade_start = trade_data["date"].min()
    trade_end = trade_data["date"].max()

    train_tickers = sorted(train_data["tic"].unique().tolist())
    trade_tickers = sorted(trade_data["tic"].unique().tolist())

    metadata = {
        "split_id": split_id,
        "split_type": "point_in_time",
        "point_in_time": point_in_time,
        "requested_trade_end_date": trade_end_date,
        "train_start_date": _format_date(train_start),
        "train_end_date": _format_date(train_end),
        "trade_start_date": _format_date(trade_start),
        "trade_end_date": _format_date(trade_end),
        "train_row_count": int(len(train_data)),
        "trade_row_count": int(len(trade_data)),
        "train_column_count": int(len(train_data.columns)),
        "trade_column_count": int(len(trade_data.columns)),
        "train_tickers": train_tickers,
        "trade_tickers": trade_tickers,
        "expected_tickers": list(expected_tickers) if expected_tickers else None,
        "min_tickers_per_date": min_tickers_per_date,
        "no_train_leakage": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_metadata": source_metadata or {},
    }

    train_data["date"] = train_data["date"].dt.strftime("%Y-%m-%d")
    trade_data["date"] = trade_data["date"].dt.strftime("%Y-%m-%d")

    return PointInTimeSplitResult(
        split_id=split_id,
        point_in_time=point_in_time,
        train_start_date=metadata["train_start_date"],
        train_end_date=metadata["train_end_date"],
        trade_start_date=metadata["trade_start_date"],
        trade_end_date=metadata["trade_end_date"],
        train_data=train_data,
        trade_data=trade_data,
        metadata=metadata,
    )
