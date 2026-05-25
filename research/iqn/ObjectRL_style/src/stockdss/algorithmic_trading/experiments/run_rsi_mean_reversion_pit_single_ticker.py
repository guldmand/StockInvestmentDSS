from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from stockdss.algorithmic_trading.baselines.rsi_mean_reversion_pit_single_ticker import (
    run_rsi_mean_reversion,
)


def print_banner(title: str) -> None:
    print("=" * 100)
    print(title)
    print("=" * 100)


def print_saved_outputs(outputs: dict) -> None:
    print("Saved outputs:")
    for key, path in outputs.items():
        print(f"- {key}: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PIT single-ticker algorithmic trading baseline: RSI mean-reversion."
    )
    parser.add_argument("--trade-data", required=True)
    parser.add_argument("--dataset-tag", required=True)
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--run-root", default=None)
    parser.add_argument("--initial-amount", type=float, default=1_000_000.0)
    parser.add_argument("--rsi-window", type=int, default=14)
    parser.add_argument("--oversold", type=float, default=30.0)
    parser.add_argument("--overbought", type=float, default=70.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print_banner("StockDSS - Algorithmic Trading Baseline: RSI mean-reversion")
    print(f"Trade data:       {args.trade_data}")
    print(f"Dataset tag:      {args.dataset_tag}")
    print(f"Ticker:           {args.ticker.upper()}")
    print(f"Run name:         {args.run_name}")
    print(f"Run root:         {args.run_root}")
    print(f"Initial amount:   {args.initial_amount:,.2f}")
    print("=" * 100)

    outputs = run_rsi_mean_reversion(
        trade_data=args.trade_data,
        dataset_tag=args.dataset_tag,
        ticker=args.ticker,
        run_name=args.run_name,
        run_root=args.run_root,
        initial_amount=args.initial_amount,
        rsi_window=args.rsi_window,
        oversold=args.oversold,
        overbought=args.overbought,
    )

    print()
    print_banner("RSI mean-reversion finished")

    metrics_path = outputs.get("metrics")
    if metrics_path:
        metrics = pd.read_csv(metrics_path)
        print(metrics.to_string(index=False))
        print()

    print_saved_outputs(outputs)


if __name__ == "__main__":
    main()
