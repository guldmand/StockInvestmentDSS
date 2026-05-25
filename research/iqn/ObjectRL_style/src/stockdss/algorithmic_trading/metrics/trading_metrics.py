from __future__ import annotations

from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {"date", "tic", "close"}


def _normalise_run_root(run_root: str | Path | None) -> Path | None:
    if run_root is None:
        return None
    run_root_path = Path(run_root)
    if str(run_root_path).strip() == "":
        return None
    return run_root_path


def make_single_ticker_output_dirs(
    *,
    dataset_tag: str,
    run_name: str,
    run_root: str | Path | None,
    strategy_folder: str,
) -> tuple[Path, Path]:
    """
    Create output directories for one single-ticker algorithmic trading run.

    With a central run root, outputs are written to:

        <run_root>/algorithmic_trading/results/<strategy_folder>/<run_name>/
        <run_root>/algorithmic_trading/plots/<strategy_folder>/<run_name>/

    Without a run root, outputs are written to:

        outputs/algorithmic_trading/<dataset_tag>/<run_name>/results/<strategy_folder>/
        outputs/algorithmic_trading/<dataset_tag>/<run_name>/plots/<strategy_folder>/
    """
    root = _normalise_run_root(run_root)

    if root is not None:
        base_dir = root / "algorithmic_trading"
        results_dir = base_dir / "results" / strategy_folder / run_name
        plots_dir = base_dir / "plots" / strategy_folder / run_name
    else:
        base_dir = Path("outputs") / "algorithmic_trading" / dataset_tag / run_name
        results_dir = base_dir / "results" / strategy_folder
        plots_dir = base_dir / "plots" / strategy_folder

    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    return results_dir, plots_dir


def make_portfolio_output_dirs(
    *,
    dataset_tag: str,
    run_name: str,
    run_root: str | Path | None,
    strategy_folder: str,
) -> tuple[Path, Path]:
    """
    Create output directories for a portfolio-level algorithmic trading run.
    Same convention as make_single_ticker_output_dirs.
    """
    return make_single_ticker_output_dirs(
        dataset_tag=dataset_tag,
        run_name=run_name,
        run_root=run_root,
        strategy_folder=strategy_folder,
    )


def load_trade_data_single_ticker(trade_data: str | Path, ticker: str) -> pd.DataFrame:
    path = Path(trade_data)
    if not path.exists():
        raise FileNotFoundError(f"Trade data not found: {path}")

    df = pd.read_csv(path)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Trade data is missing required columns: {sorted(missing)}")

    result = df.copy()
    result["date"] = pd.to_datetime(result["date"])
    result["tic"] = result["tic"].astype(str).str.upper()
    result["close"] = pd.to_numeric(result["close"], errors="coerce")

    ticker_upper = ticker.upper()
    result = result[result["tic"] == ticker_upper].copy()
    result = result.dropna(subset=["date", "close"])
    result = result.sort_values("date").reset_index(drop=True)

    if result.empty:
        raise ValueError(f"No rows found for ticker: {ticker_upper}")

    if (result["close"] <= 0).any():
        bad_dates = result.loc[result["close"] <= 0, "date"].head(5).tolist()
        raise ValueError(
            f"Ticker {ticker_upper} has non-positive close prices. "
            f"First bad dates: {bad_dates}"
        )

    return result


def calculate_account_metrics(
    account_values: pd.DataFrame,
    *,
    strategy: str,
    source: str,
    initial_amount: float,
) -> pd.DataFrame:
    if "account_value" not in account_values.columns:
        raise ValueError("account_values must contain an 'account_value' column.")

    values = pd.to_numeric(account_values["account_value"], errors="coerce").dropna()
    if values.empty:
        raise ValueError("account_values contains no valid numeric account values.")

    start_value = float(values.iloc[0])
    end_value = float(values.iloc[-1])
    profit_loss = end_value - float(initial_amount)
    total_return_pct = (profit_loss / float(initial_amount)) * 100.0

    running_max = values.cummax()
    drawdown = (values / running_max) - 1.0
    max_drawdown_pct = float(drawdown.min() * 100.0)

    daily_returns = values.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    return_std = daily_returns.std(ddof=1)

    if daily_returns.empty or pd.isna(return_std) or return_std == 0:
        annualized_sharpe = float("nan")
    else:
        annualized_sharpe = float((daily_returns.mean() / return_std) * np.sqrt(252.0))

    return pd.DataFrame(
        [
            {
                "strategy": strategy,
                "source": source,
                "start_value": start_value,
                "end_value": end_value,
                "profit_loss": profit_loss,
                "total_return_pct": total_return_pct,
                "max_drawdown_pct": max_drawdown_pct,
                "annualized_sharpe": annualized_sharpe,
                "days": int(len(values)),
                "ended_above_initial": bool(end_value > float(initial_amount)),
            }
        ]
    )


def save_account_value_plot(
    account_values: pd.DataFrame,
    *,
    output_path: str | Path,
    title: str,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if "date" not in account_values.columns or "account_value" not in account_values.columns:
        raise ValueError("account_values must contain 'date' and 'account_value' columns.")

    plot_df = account_values.copy()
    plot_df["date"] = pd.to_datetime(plot_df["date"])

    plt.figure(figsize=(12, 6))
    plt.plot(plot_df["date"], plot_df["account_value"])
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Account value")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
