from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

from stockdss.algorithmic_trading.metrics.trading_metrics import (
    calculate_account_metrics,
    make_portfolio_output_dirs,
    save_account_value_plot,
)


def run_equal_weight_buy_and_hold_portfolio(
    *,
    trade_data: str | Path,
    dataset_tag: str,
    run_name: str,
    run_root: str | Path | None = None,
    initial_amount: float = 1_000_000.0,
) -> Dict[str, Path]:
    path = Path(trade_data)
    if not path.exists():
        raise FileNotFoundError(f"Trade data not found: {path}")

    df = pd.read_csv(path)
    required = {"date", "tic", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Trade data is missing required columns: {sorted(missing)}")

    df["date"] = pd.to_datetime(df["date"])
    df["tic"] = df["tic"].astype(str).str.upper()
    df["close"] = df["close"].astype(float)
    df = df.sort_values(["date", "tic"]).reset_index(drop=True)

    price_wide = df.pivot_table(index="date", columns="tic", values="close", aggfunc="last").sort_index()
    price_wide = price_wide.dropna(axis=1, how="any")

    if price_wide.empty:
        raise ValueError("No complete ticker price matrix could be built for equal-weight buy-and-hold.")

    tickers = price_wide.columns.tolist()
    first_prices = price_wide.iloc[0]
    capital_per_asset = float(initial_amount) / len(tickers)
    shares = capital_per_asset / first_prices

    portfolio_values = price_wide.mul(shares, axis=1).sum(axis=1)

    account = pd.DataFrame(
        {
            "date": portfolio_values.index.strftime("%Y-%m-%d"),
            "account_value": portfolio_values.values,
        }
    )

    weights = pd.DataFrame(
        {
            "ticker": tickers,
            "initial_price": first_prices.values,
            "initial_capital": capital_per_asset,
            "shares": shares.values,
            "initial_weight": 1.0 / len(tickers),
        }
    )

    files_dir, plots_dir = make_portfolio_output_dirs(
        dataset_tag=dataset_tag,
        run_name=run_name,
        run_root=run_root,
        strategy_folder="equal_weight_buy_and_hold",
    )

    strategy = f"equal_weight_buy_and_hold_{len(tickers)}"
    metrics = calculate_account_metrics(
        account,
        strategy=strategy,
        source="Algorithmic Trading / non-RL",
        initial_amount=initial_amount,
    )

    config = {
        "strategy": strategy,
        "dataset_tag": dataset_tag,
        "trade_data": str(trade_data),
        "run_name": run_name,
        "run_root": str(run_root) if run_root else None,
        "initial_amount": float(initial_amount),
        "stock_dimension": int(len(tickers)),
        "signal_rule": "buy equal-dollar allocation on first trade date and hold until final trade date",
    }

    account_path = files_dir / "equal_weight_buy_and_hold_account_values.csv"
    metrics_path = files_dir / "equal_weight_buy_and_hold_metrics.csv"
    weights_path = files_dir / "equal_weight_buy_and_hold_weights.csv"
    config_path = files_dir / "equal_weight_buy_and_hold_config.json"
    plot_path = plots_dir / "equal_weight_buy_and_hold_account_value.png"

    # Create output folders before writing files.
    for output_path in [account_path, metrics_path, weights_path, config_path, plot_path]:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    account.to_csv(account_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    weights.to_csv(weights_path, index=False)
    pd.Series(config).to_json(config_path, indent=2)
    save_account_value_plot(account, output_path=plot_path, title=f"{strategy} account value")

    return {
        "account": account_path,
        "metrics": metrics_path,
        "weights": weights_path,
        "config": config_path,
        "plot": plot_path,
    }
