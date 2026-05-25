from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def print_banner(title: str) -> None:
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def run_command(title: str, command: list[str], continue_on_error: bool) -> None:
    print_banner(title)
    print(" ".join(command))
    print()

    result = subprocess.run(command)

    if result.returncode != 0:
        message = f"Command failed with exit code {result.returncode}: {title}"

        if continue_on_error:
            print()
            print(f"WARNING: {message}")
            return

        raise RuntimeError(message)


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_pair_grid(value: str) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        left, right = item.split(":")
        result.append((int(left), int(right)))
    return result


def parse_rsi_grid(value: str) -> list[tuple[int, float, float]]:
    result: list[tuple[int, float, float]] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        window, oversold, overbought = item.split(":")
        result.append((int(window), float(oversold), float(overbought)))
    return result


def parse_macd_grid(value: str) -> list[tuple[int, int, int]]:
    result: list[tuple[int, int, int]] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        fast, slow, signal = item.split(":")
        result.append((int(fast), int(slow), int(signal)))
    return result


def parse_bollinger_grid(value: str) -> list[tuple[int, float]]:
    result: list[tuple[int, float]] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        window, num_std = item.split(":")
        result.append((int(window), float(num_std)))
    return result


def parse_volatility_grid(value: str) -> list[tuple[int, int, float]]:
    result: list[tuple[int, int, float]] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        momentum_window, volatility_window, max_volatility = item.split(":")
        result.append(
            (int(momentum_window), int(volatility_window), float(max_volatility))
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run all classical non-RL algorithmic trading baselines and compare results."
    )

    parser.add_argument("--trade-data", required=True)
    parser.add_argument("--dataset-tag", required=True)
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--initial-amount", type=float, default=1_000_000.0)
    parser.add_argument("--continue-on-error", action="store_true")

    # Backward-compatible single-run defaults.
    parser.add_argument("--sma-fast-window", type=int, default=50)
    parser.add_argument("--sma-slow-window", type=int, default=200)
    parser.add_argument("--ema-fast-window", type=int, default=12)
    parser.add_argument("--ema-slow-window", type=int, default=26)
    parser.add_argument("--momentum-lookback-window", type=int, default=60)
    parser.add_argument("--rsi-window", type=int, default=14)
    parser.add_argument("--oversold", type=float, default=30)
    parser.add_argument("--overbought", type=float, default=70)
    parser.add_argument("--macd-fast-window", type=int, default=12)
    parser.add_argument("--macd-slow-window", type=int, default=26)
    parser.add_argument("--macd-signal-window", type=int, default=9)
    parser.add_argument("--bollinger-window", type=int, default=20)
    parser.add_argument("--bollinger-num-std", type=float, default=2.0)
    parser.add_argument("--breakout-lookback-window", type=int, default=20)
    parser.add_argument("--volatility-momentum-window", type=int, default=20)
    parser.add_argument("--volatility-window", type=int, default=20)
    parser.add_argument("--max-annualized-volatility", type=float, default=0.4)

    # Optional parameter grids. Use comma-separated values.
    # Examples:
    # --sma-grid "20:50,50:200,100:300"
    # --momentum-windows "20,60,120"
    # --volatility-grid "20:20:0.4,60:20:0.4"
    parser.add_argument("--sma-grid", default=None)
    parser.add_argument("--ema-grid", default=None)
    parser.add_argument("--momentum-windows", default=None)
    parser.add_argument("--rsi-grid", default=None)
    parser.add_argument("--macd-grid", default=None)
    parser.add_argument("--bollinger-grid", default=None)
    parser.add_argument("--breakout-windows", default=None)
    parser.add_argument("--volatility-grid", default=None)
    parser.add_argument("--skip-equal-weight", action="store_true")

    args = parser.parse_args()

    ticker = args.ticker.upper()
    run_root = Path(args.run_root)
    run_root.mkdir(parents=True, exist_ok=True)

    base_args = [
        "--trade-data",
        args.trade_data,
        "--dataset-tag",
        args.dataset_tag,
        "--ticker",
        ticker,
        "--run-root",
        str(run_root),
        "--initial-amount",
        str(args.initial_amount),
    ]

    experiments: list[tuple[str, list[str]]] = []

    experiments.append(
        (
            "C1. Buy-and-hold single ticker",
            [
                sys.executable,
                "-m",
                "stockdss.algorithmic_trading.experiments.run_buy_and_hold_pit_single_ticker",
                *base_args,
                "--run-name",
                f"test_{ticker.lower()}_buy_and_hold",
            ],
        )
    )

    sma_grid = (
        parse_pair_grid(args.sma_grid)
        if args.sma_grid
        else [(args.sma_fast_window, args.sma_slow_window)]
    )
    for fast, slow in sma_grid:
        experiments.append(
            (
                f"C2. SMA crossover single ticker {fast}/{slow}",
                [
                    sys.executable,
                    "-m",
                    "stockdss.algorithmic_trading.experiments.run_sma_crossover_pit_single_ticker",
                    *base_args,
                    "--run-name",
                    f"test_{ticker.lower()}_sma_{fast}_{slow}",
                    "--fast-window",
                    str(fast),
                    "--slow-window",
                    str(slow),
                ],
            )
        )

    ema_grid = (
        parse_pair_grid(args.ema_grid)
        if args.ema_grid
        else [(args.ema_fast_window, args.ema_slow_window)]
    )
    for fast, slow in ema_grid:
        experiments.append(
            (
                f"C3. EMA crossover single ticker {fast}/{slow}",
                [
                    sys.executable,
                    "-m",
                    "stockdss.algorithmic_trading.experiments.run_ema_crossover_pit_single_ticker",
                    *base_args,
                    "--run-name",
                    f"test_{ticker.lower()}_ema_{fast}_{slow}",
                    "--fast-window",
                    str(fast),
                    "--slow-window",
                    str(slow),
                ],
            )
        )

    momentum_windows = (
        parse_int_list(args.momentum_windows)
        if args.momentum_windows
        else [args.momentum_lookback_window]
    )
    for window in momentum_windows:
        experiments.append(
            (
                f"C4. Momentum single ticker {window}",
                [
                    sys.executable,
                    "-m",
                    "stockdss.algorithmic_trading.experiments.run_momentum_pit_single_ticker",
                    *base_args,
                    "--run-name",
                    f"test_{ticker.lower()}_momentum_{window}",
                    "--lookback-window",
                    str(window),
                ],
            )
        )

    rsi_grid = (
        parse_rsi_grid(args.rsi_grid)
        if args.rsi_grid
        else [(args.rsi_window, args.oversold, args.overbought)]
    )
    for window, oversold, overbought in rsi_grid:
        experiments.append(
            (
                f"C5. RSI mean-reversion single ticker {window}/{oversold:g}/{overbought:g}",
                [
                    sys.executable,
                    "-m",
                    "stockdss.algorithmic_trading.experiments.run_rsi_mean_reversion_pit_single_ticker",
                    *base_args,
                    "--run-name",
                    f"test_{ticker.lower()}_rsi_{window}_{oversold:g}_{overbought:g}",
                    "--rsi-window",
                    str(window),
                    "--oversold",
                    str(oversold),
                    "--overbought",
                    str(overbought),
                ],
            )
        )

    macd_grid = (
        parse_macd_grid(args.macd_grid)
        if args.macd_grid
        else [(args.macd_fast_window, args.macd_slow_window, args.macd_signal_window)]
    )
    for fast, slow, signal in macd_grid:
        experiments.append(
            (
                f"C6. MACD signal single ticker {fast}/{slow}/{signal}",
                [
                    sys.executable,
                    "-m",
                    "stockdss.algorithmic_trading.experiments.run_macd_signal_pit_single_ticker",
                    *base_args,
                    "--run-name",
                    f"test_{ticker.lower()}_macd_{fast}_{slow}_{signal}",
                    "--fast-window",
                    str(fast),
                    "--slow-window",
                    str(slow),
                    "--signal-window",
                    str(signal),
                ],
            )
        )

    bollinger_grid = (
        parse_bollinger_grid(args.bollinger_grid)
        if args.bollinger_grid
        else [(args.bollinger_window, args.bollinger_num_std)]
    )
    for window, num_std in bollinger_grid:
        experiments.append(
            (
                f"C7. Bollinger mean-reversion single ticker {window}/{num_std:g}",
                [
                    sys.executable,
                    "-m",
                    "stockdss.algorithmic_trading.experiments.run_bollinger_mean_reversion_pit_single_ticker",
                    *base_args,
                    "--run-name",
                    f"test_{ticker.lower()}_bollinger_{window}_{num_std:g}",
                    "--window",
                    str(window),
                    "--num-std",
                    str(num_std),
                ],
            )
        )

    breakout_windows = (
        parse_int_list(args.breakout_windows)
        if args.breakout_windows
        else [args.breakout_lookback_window]
    )
    for window in breakout_windows:
        experiments.append(
            (
                f"C8. Breakout single ticker {window}",
                [
                    sys.executable,
                    "-m",
                    "stockdss.algorithmic_trading.experiments.run_breakout_pit_single_ticker",
                    *base_args,
                    "--run-name",
                    f"test_{ticker.lower()}_breakout_{window}",
                    "--lookback-window",
                    str(window),
                ],
            )
        )

    volatility_grid = (
        parse_volatility_grid(args.volatility_grid)
        if args.volatility_grid
        else [
            (
                args.volatility_momentum_window,
                args.volatility_window,
                args.max_annualized_volatility,
            )
        ]
    )
    for momentum_window, volatility_window, max_volatility in volatility_grid:
        experiments.append(
            (
                f"C9. Volatility filter single ticker {momentum_window}/{volatility_window}/{max_volatility:g}",
                [
                    sys.executable,
                    "-m",
                    "stockdss.algorithmic_trading.experiments.run_volatility_filter_pit_single_ticker",
                    *base_args,
                    "--run-name",
                    f"test_{ticker.lower()}_vol_{momentum_window}_{volatility_window}_{str(max_volatility).replace('.', '')}",
                    "--momentum-window",
                    str(momentum_window),
                    "--volatility-window",
                    str(volatility_window),
                    "--max-annualized-volatility",
                    str(max_volatility),
                ],
            )
        )

    if not args.skip_equal_weight:
        experiments.append(
            (
                "C10. Equal-weight buy-and-hold portfolio",
                [
                    sys.executable,
                    "-m",
                    "stockdss.algorithmic_trading.experiments.run_equal_weight_buy_and_hold_pit_portfolio",
                    "--trade-data",
                    args.trade_data,
                    "--dataset-tag",
                    args.dataset_tag,
                    "--run-name",
                    "test_equal_weight_buy_and_hold",
                    "--run-root",
                    str(run_root),
                    "--initial-amount",
                    str(args.initial_amount),
                ],
            )
        )

    print_banner("StockDSS - Run all algorithmic trading baselines")
    print(f"Trade data:       {args.trade_data}")
    print(f"Dataset tag:      {args.dataset_tag}")
    print(f"Ticker:           {ticker}")
    print(f"Run root:         {run_root}")
    print(f"Initial amount:   {args.initial_amount:,.2f}")
    print(f"Continue errors:  {args.continue_on_error}")
    print(f"Experiments:      {len(experiments)}")

    for title, command in experiments:
        run_command(title, command, args.continue_on_error)

    compare_command = [
        sys.executable,
        "-m",
        "stockdss.algorithmic_trading.experiments.compare_algorithmic_results",
        "--run-root",
        str(run_root),
        "--show",
    ]

    run_command("C11. Compare algorithmic trading results", compare_command, False)

    print_banner("All algorithmic trading experiments finished")
    print(f"Run root: {run_root}")
    print("Summary:")
    print(
        f"- {run_root / 'algorithmic_trading' / 'summary' / 'algorithmic_trading_summary.csv'}"
    )
    print(
        f"- {run_root / 'algorithmic_trading' / 'summary' / 'algorithmic_trading_summary.md'}"
    )
    print(
        f"- {run_root / 'algorithmic_trading' / 'summary' / 'algorithmic_trading_summary_returns.png'}"
    )


if __name__ == "__main__":
    main()
