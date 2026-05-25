from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from stockdss.algorithmic_trading.baselines.macd_signal_pit_single_ticker import (
    run_macd_signal,
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
        description="Run PIT single-ticker algorithmic trading baseline: MACD signal."
    )
    parser.add_argument("--trade-data", required=True)
    parser.add_argument("--dataset-tag", required=True)
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--run-root", default=None)
    parser.add_argument("--initial-amount", type=float, default=1_000_000.0)
    parser.add_argument("--fast-window", type=int, default=12)
    parser.add_argument("--slow-window", type=int, default=26)
    parser.add_argument("--signal-window", type=int, default=9)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print_banner("StockDSS - Algorithmic Trading Baseline: MACD signal")
    print(f"Trade data:       {args.trade_data}")
    print(f"Dataset tag:      {args.dataset_tag}")
    print(f"Ticker:           {args.ticker.upper()}")
    print(f"Run name:         {args.run_name}")
    print(f"Run root:         {args.run_root}")
    print(f"Initial amount:   {args.initial_amount:,.2f}")
    print("=" * 100)

    outputs = run_macd_signal(
        trade_data=args.trade_data,
        dataset_tag=args.dataset_tag,
        ticker=args.ticker,
        run_name=args.run_name,
        run_root=args.run_root,
        initial_amount=args.initial_amount,
        fast_window=args.fast_window,
        slow_window=args.slow_window,
        signal_window=args.signal_window,
    )

    print()
    print_banner("MACD signal finished")

    metrics_path = outputs.get("metrics")
    if metrics_path:
        metrics = pd.read_csv(metrics_path)
        print(metrics.to_string(index=False))
        print()

    print_saved_outputs(outputs)


if __name__ == "__main__":
    main()
