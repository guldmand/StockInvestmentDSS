"""
Point-in-time split utility for D-IQN-DSS.

Purpose:
- Use one master market data file as the source of truth.
- Split it into train/trade datasets based on a chosen point-in-time date.
- Save metadata so the experiment can document exactly what was known when.

Example:

PowerShell:
    $env:PYTHONPATH="src"
    python -m stockdss.data.point_in_time_split `
        --input data/market_data_full_500.csv `
        --point-in-time 2026-01-01 `
        --dataset-tag pit_500_20260101

Output:
    data/train_data_pit_500_20260101.csv
    data/trade_data_pit_500_20260101.csv
    outputs/pit/pit_metadata_pit_500_20260101.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

REQUIRED_COLUMNS = {"date", "tic", "close"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create point-in-time train/trade split from master market data."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to master market data CSV, e.g. data/market_data_full_500.csv",
    )

    parser.add_argument(
        "--point-in-time",
        required=True,
        help="Point-in-time split date, e.g. 2026-01-01. Rows before this date become train data.",
    )

    parser.add_argument(
        "--dataset-tag",
        required=True,
        help="Tag used in output filenames, e.g. pit_500_20260101",
    )

    parser.add_argument(
        "--output-data-dir",
        default="data",
        help="Directory where train/trade CSV files are saved. Default: data",
    )

    parser.add_argument(
        "--metadata-dir",
        default="outputs/pit",
        help="Directory where PIT metadata JSON is saved. Default: outputs/pit",
    )

    parser.add_argument(
        "--trade-end-date",
        default=None,
        help="Optional final date for trade data, e.g. 2026-03-20.",
    )

    parser.add_argument(
        "--min-tickers-per-date",
        type=int,
        default=1,
        help="Minimum number of tickers required per date after split. Default: 1",
    )

    return parser.parse_args()


def validate_required_columns(df: pd.DataFrame, input_path: Path) -> None:
    missing_columns = REQUIRED_COLUMNS - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Input file {input_path} is missing required columns: "
            f"{sorted(missing_columns)}"
        )


def load_master_data(input_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    validate_required_columns(df, input_path)

    df["date"] = pd.to_datetime(df["date"])
    df["tic"] = df["tic"].astype(str)

    df = df.sort_values(["date", "tic"]).reset_index(drop=True)

    return df


def create_point_in_time_split(
    df: pd.DataFrame,
    point_in_time: pd.Timestamp,
    trade_end_date: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = df[df["date"] < point_in_time].copy()
    trade_df = df[df["date"] >= point_in_time].copy()

    if trade_end_date is not None:
        trade_df = trade_df[trade_df["date"] <= trade_end_date].copy()

    train_df = train_df.sort_values(["date", "tic"]).reset_index(drop=True)
    trade_df = trade_df.sort_values(["date", "tic"]).reset_index(drop=True)

    return train_df, trade_df


def validate_split(
    train_df: pd.DataFrame,
    trade_df: pd.DataFrame,
    point_in_time: pd.Timestamp,
    min_tickers_per_date: int,
) -> None:
    if train_df.empty:
        raise ValueError(
            f"Train split is empty. Point-in-time may be too early: {point_in_time.date()}"
        )

    if trade_df.empty:
        raise ValueError(
            f"Trade split is empty. Point-in-time may be too late: {point_in_time.date()}"
        )

    train_max_date = train_df["date"].max()
    trade_min_date = trade_df["date"].min()

    if train_max_date >= point_in_time:
        raise ValueError("Point-in-time violation: train data contains future rows.")

    if trade_min_date < point_in_time:
        raise ValueError("Point-in-time violation: trade data contains pre-PIT rows.")

    train_counts = train_df.groupby("date")["tic"].nunique()
    trade_counts = trade_df.groupby("date")["tic"].nunique()

    if train_counts.min() < min_tickers_per_date:
        raise ValueError(
            "Train split has dates with too few tickers. "
            f"Minimum found: {train_counts.min()}, required: {min_tickers_per_date}"
        )

    if trade_counts.min() < min_tickers_per_date:
        raise ValueError(
            "Trade split has dates with too few tickers. "
            f"Minimum found: {trade_counts.min()}, required: {min_tickers_per_date}"
        )


def save_csv_for_finrl(df: pd.DataFrame, output_path: Path) -> None:
    """
    Save CSV in a FinRL-compatible style.

    We keep the dataframe index in the CSV because many FinRL tutorials load
    files using:
        df = pd.read_csv(path)
        df = df.set_index(df.columns[0])
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df_to_save = df.copy()
    df_to_save["date"] = df_to_save["date"].dt.strftime("%Y-%m-%d")
    df_to_save.to_csv(output_path)


def build_metadata(
    input_path: Path,
    train_path: Path,
    trade_path: Path,
    dataset_tag: str,
    point_in_time: pd.Timestamp,
    trade_end_date: pd.Timestamp | None,
    master_df: pd.DataFrame,
    train_df: pd.DataFrame,
    trade_df: pd.DataFrame,
) -> dict[str, Any]:
    train_tickers = sorted(train_df["tic"].unique().tolist())
    trade_tickers = sorted(trade_df["tic"].unique().tolist())
    shared_tickers = sorted(set(train_tickers).intersection(set(trade_tickers)))

    metadata: dict[str, Any] = {
        "dataset_tag": dataset_tag,
        "input_path": str(input_path),
        "train_path": str(train_path),
        "trade_path": str(trade_path),
        "point_in_time": point_in_time.strftime("%Y-%m-%d"),
        "trade_end_date": (
            trade_end_date.strftime("%Y-%m-%d") if trade_end_date is not None else None
        ),
        "split_rule": {
            "train": "date < point_in_time",
            "trade": "date >= point_in_time",
            "trade_end_filter": (
                "date <= trade_end_date" if trade_end_date is not None else None
            ),
        },
        "master": {
            "rows": int(len(master_df)),
            "start_date": master_df["date"].min().strftime("%Y-%m-%d"),
            "end_date": master_df["date"].max().strftime("%Y-%m-%d"),
            "ticker_count": int(master_df["tic"].nunique()),
        },
        "train": {
            "rows": int(len(train_df)),
            "start_date": train_df["date"].min().strftime("%Y-%m-%d"),
            "end_date": train_df["date"].max().strftime("%Y-%m-%d"),
            "ticker_count": int(train_df["tic"].nunique()),
        },
        "trade": {
            "rows": int(len(trade_df)),
            "start_date": trade_df["date"].min().strftime("%Y-%m-%d"),
            "end_date": trade_df["date"].max().strftime("%Y-%m-%d"),
            "ticker_count": int(trade_df["tic"].nunique()),
        },
        "shared_ticker_count": int(len(shared_tickers)),
        "train_only_ticker_count": int(len(set(train_tickers) - set(trade_tickers))),
        "trade_only_ticker_count": int(len(set(trade_tickers) - set(train_tickers))),
        "columns": list(master_df.columns),
    }

    return metadata


def save_metadata(metadata: dict[str, Any], metadata_path: Path) -> None:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)


def print_summary(metadata: dict[str, Any]) -> None:
    separator = "=" * 100

    print(separator)
    print("Point-in-time split completed")
    print(separator)
    print(f"Dataset tag:       {metadata['dataset_tag']}")
    print(f"Input:             {metadata['input_path']}")
    print(f"Point-in-time:     {metadata['point_in_time']}")
    print(f"Trade end date:    {metadata['trade_end_date']}")
    print()
    print(f"Train output:      {metadata['train_path']}")
    print(f"Trade output:      {metadata['trade_path']}")
    print()
    print(
        f"Master: rows={metadata['master']['rows']:,}, "
        f"tickers={metadata['master']['ticker_count']}, "
        f"date_range={metadata['master']['start_date']} → {metadata['master']['end_date']}"
    )
    print(
        f"Train:  rows={metadata['train']['rows']:,}, "
        f"tickers={metadata['train']['ticker_count']}, "
        f"date_range={metadata['train']['start_date']} → {metadata['train']['end_date']}"
    )
    print(
        f"Trade:  rows={metadata['trade']['rows']:,}, "
        f"tickers={metadata['trade']['ticker_count']}, "
        f"date_range={metadata['trade']['start_date']} → {metadata['trade']['end_date']}"
    )
    print()
    print(f"Shared tickers:    {metadata['shared_ticker_count']}")
    print(f"Train-only tickers:{metadata['train_only_ticker_count']}")
    print(f"Trade-only tickers:{metadata['trade_only_ticker_count']}")
    print(separator)


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    output_data_dir = Path(args.output_data_dir)
    metadata_dir = Path(args.metadata_dir)

    point_in_time = pd.to_datetime(args.point_in_time)
    trade_end_date = (
        pd.to_datetime(args.trade_end_date) if args.trade_end_date is not None else None
    )

    master_df = load_master_data(input_path)

    train_df, trade_df = create_point_in_time_split(
        df=master_df,
        point_in_time=point_in_time,
        trade_end_date=trade_end_date,
    )

    validate_split(
        train_df=train_df,
        trade_df=trade_df,
        point_in_time=point_in_time,
        min_tickers_per_date=args.min_tickers_per_date,
    )

    train_path = output_data_dir / f"train_data_{args.dataset_tag}.csv"
    trade_path = output_data_dir / f"trade_data_{args.dataset_tag}.csv"
    metadata_path = metadata_dir / f"pit_metadata_{args.dataset_tag}.json"

    save_csv_for_finrl(train_df, train_path)
    save_csv_for_finrl(trade_df, trade_path)

    metadata = build_metadata(
        input_path=input_path,
        train_path=train_path,
        trade_path=trade_path,
        dataset_tag=args.dataset_tag,
        point_in_time=point_in_time,
        trade_end_date=trade_end_date,
        master_df=master_df,
        train_df=train_df,
        trade_df=trade_df,
    )

    save_metadata(metadata, metadata_path)

    print_summary(metadata)
    print(f"Metadata saved:    {metadata_path}")


if __name__ == "__main__":
    main()
