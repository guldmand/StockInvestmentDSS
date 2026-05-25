from __future__ import annotations

import argparse

import pandas as pd

from stockdss.algorithmic_trading.baselines.equal_weight_buy_and_hold_pit_portfolio import (
    run_equal_weight_buy_and_hold_portfolio,
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
        description="Run PIT portfolio algorithmic trading baseline: equal-weight buy-and-hold."
    )
    parser.add_argument("--trade-data", required=True)
    parser.add_argument("--dataset-tag", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--run-root", default=None)
    parser.add_argument("--initial-amount", type=float, default=1_000_000.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print_banner(
        "StockDSS - Algorithmic Trading Baseline: Equal-weight Buy-and-Hold Portfolio"
    )
    print(f"Trade data:       {args.trade_data}")
    print(f"Dataset tag:      {args.dataset_tag}")
    print(f"Run name:         {args.run_name}")
    print(f"Run root:         {args.run_root}")
    print(f"Initial amount:   {args.initial_amount:,.2f}")
    print("=" * 100)

    outputs = run_equal_weight_buy_and_hold_portfolio(
        trade_data=args.trade_data,
        dataset_tag=args.dataset_tag,
        run_name=args.run_name,
        run_root=args.run_root,
        initial_amount=args.initial_amount,
    )

    print()
    print_banner("Equal-weight buy-and-hold portfolio finished")

    metrics_path = outputs.get("metrics")
    if metrics_path:
        metrics = pd.read_csv(metrics_path)
        print(metrics.to_string(index=False))
        print()

    print_saved_outputs(outputs)


if __name__ == "__main__":
    main()
