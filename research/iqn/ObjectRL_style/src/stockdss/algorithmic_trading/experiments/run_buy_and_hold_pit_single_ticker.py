from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from stockdss.algorithmic_trading.baselines.buy_and_hold_pit_single_ticker import (
    run_buy_and_hold,
)
from stockdss.algorithmic_trading.metrics.trading_metrics import (
    calculate_account_metrics,
    make_single_ticker_output_dirs,
    save_account_value_plot,
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
        description="Run PIT-compatible buy-and-hold baseline for a single ticker."
    )
    parser.add_argument("--trade-data", required=True, help="Path to PIT trade CSV.")
    parser.add_argument("--dataset-tag", required=True, help="Dataset tag.")
    parser.add_argument("--ticker", required=True, help="Ticker, e.g. AAPL.")
    parser.add_argument("--run-name", default="buy_and_hold")
    parser.add_argument("--run-root", default=None)
    parser.add_argument("--initial-amount", type=float, default=1_000_000.0)
    parser.add_argument("--price-column", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ticker = args.ticker.upper()

    results_dir, plots_dir = make_single_ticker_output_dirs(
        dataset_tag=args.dataset_tag,
        run_name=args.run_name,
        run_root=args.run_root,
        strategy_folder="buy_and_hold",
    )

    print_banner("StockDSS - Algorithmic Trading Baseline: Buy-and-Hold")
    print(f"Trade data:       {args.trade_data}")
    print(f"Dataset tag:      {args.dataset_tag}")
    print(f"Ticker:           {ticker}")
    print(f"Run name:         {args.run_name}")
    print(f"Run root:         {args.run_root}")
    print(f"Initial amount:   {args.initial_amount:,.2f}")
    print(f"Results dir:      {results_dir}")
    print(f"Plots dir:        {plots_dir}")
    print("=" * 100)

    trade_df = pd.read_csv(args.trade_data)
    result = run_buy_and_hold(
        trade_data=trade_df,
        ticker=ticker,
        initial_amount=args.initial_amount,
        price_column=args.price_column,
    )

    strategy_name = f"{ticker}_buy_and_hold"
    metrics = calculate_account_metrics(
        result,
        strategy=strategy_name,
        source="Algorithmic Trading / non-RL",
        initial_amount=args.initial_amount,
    )

    result_path = results_dir / f"{ticker.lower()}_buy_and_hold_account_values.csv"
    metrics_path = results_dir / f"{ticker.lower()}_buy_and_hold_metrics.csv"
    config_path = results_dir / f"{ticker.lower()}_buy_and_hold_config.json"
    plot_path = plots_dir / f"{ticker.lower()}_buy_and_hold_account_value.png"

    result.to_csv(result_path, index=False)
    metrics.to_csv(metrics_path, index=False)

    config = {
        "strategy": strategy_name,
        "source": "Algorithmic Trading / non-RL",
        "trade_data": args.trade_data,
        "dataset_tag": args.dataset_tag,
        "ticker": ticker,
        "run_name": args.run_name,
        "run_root": args.run_root,
        "initial_amount": float(args.initial_amount),
        "price_column": args.price_column,
        "output_files": {
            "account": str(result_path),
            "metrics": str(metrics_path),
            "config": str(config_path),
            "plot": str(plot_path),
        },
    }

    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    save_account_value_plot(
        result,
        output_path=plot_path,
        title=f"{strategy_name} account value",
    )

    outputs = {
        "account": result_path,
        "metrics": metrics_path,
        "config": config_path,
        "plot": plot_path,
    }

    print()
    print_banner("Buy-and-hold finished")
    print(metrics.to_string(index=False))
    print()
    print_saved_outputs(outputs)


if __name__ == "__main__":
    main()
