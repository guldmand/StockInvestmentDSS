"""
Bollinger Band mean-reversion point-in-time baseline for a single ticker.

Signal logic:
    - Compute a rolling mean and rolling standard deviation of close prices
      to derive Bollinger Bands (middle, upper, lower).
    - If today's close is below the lower band, enter a long position (invested).
    - If today's close is above the upper band, exit to cash.
    - Between the bands, hold the previous position.
    - Position is shifted by one trading day to avoid same-day look-ahead.

Pre-computed column optimisation:
    If the input DataFrame contains both ``boll_ub`` and ``boll_lb`` columns,
    those pre-computed bands are used directly and the ``boll_middle`` series is
    derived from a rolling mean of close prices.  This behaviour is data-aware
    and is preserved from the V1 implementation.  If either column is absent,
    all three bands are computed from close prices using rolling statistics.

    When ``force_recompute_bands=True``, pre-computed band columns are ignored
    even if present and all three bands are derived from close prices.  This
    ensures that the ``num_std`` parameter is respected, which is required for
    multi-configuration grid runs that sweep over ``num_std`` values.

Point-in-time safety:
    rolling().mean() and rolling().std() use min_periods=window so that band
    values are NaN until a full window of data is available.  The stateful loop
    does not update the position when band values are NaN.  The position shift(1)
    prevents any day-t signal from affecting the day-t position.

Output contract:
    Writes four files to:
        outputs/run_registry/algorithmic_baselines/<dataset_tag>/bollinger_mean_reversion/<ticker>/

    - <ticker>_bollinger_mr_account_values.csv   — time series with bands and signals
    - <ticker>_bollinger_mr_metrics.csv          — performance metrics (Etape 2 canonical)
    - <ticker>_bollinger_mr_config.json          — run parameters
    - <ticker>_bollinger_mr_account_value.png    — portfolio value plot

    Performance metrics are computed via
    stock_investment_dss.metrics.trading_metrics.calculate_account_metrics,
    which wraps the canonical V2 evaluation module.  No metric formulas are
    duplicated in this file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from stock_investment_dss.metrics.trading_metrics import (
    calculate_account_metrics,
    load_trade_data_single_ticker,
    save_account_value_plot,
)
from stock_investment_dss.utilities.paths import create_run_paths, RunPaths


def run_bollinger_mean_reversion(
    *,
    trade_data: str | Path,
    dataset_tag: str,
    ticker: str,
    initial_amount: float = 1_000_000.0,
    window: int = 20,
    num_std: float = 2.0,
    force_recompute_bands: bool = False,
    transaction_cost_pct: float = 0.001,
    strategy_folder: str | None = None,
    run_paths: Optional[RunPaths] = None,
    output_subpath: Optional[str] = None,
) -> Dict[str, Path]:
    """
    Run Bollinger Band mean-reversion for one ticker and write outputs to the
    V2 canonical run directory.

    Parameters
    ----------
    trade_data:
        Path to a CSV file with columns ``date``, ``tic``, ``close``.
        Pre-computed ``boll_ub`` and ``boll_lb`` columns are used automatically
        when present, unless ``force_recompute_bands`` is ``True``.
    dataset_tag:
        Short identifier for the dataset, used in the output directory path.
    ticker:
        Ticker symbol (case-insensitive).
    initial_amount:
        Initial portfolio capital.
    window:
        Rolling window for computing the middle band and standard deviation.
        Always applied when ``force_recompute_bands=True``.  Only applied as a
        fallback when pre-computed band columns are absent and
        ``force_recompute_bands=False``.
    num_std:
        Number of standard deviations for the upper and lower bands.
        Always applied when ``force_recompute_bands=True``.  Only applied as a
        fallback when pre-computed band columns are absent and
        ``force_recompute_bands=False``.
    force_recompute_bands:
        If ``False`` (default), pre-computed ``boll_ub`` / ``boll_lb`` columns
        are used when available.  This preserves backward compatibility with
        Etape 3a–3d default-config outputs.
        If ``True``, pre-computed band columns are ignored even if present and
        all three bands are derived from close prices using ``window`` and
        ``num_std``.  Use ``True`` when running multi-configuration grid sweeps
        that vary ``num_std``.
    strategy_folder:
        Optional output subfolder override.  If ``None`` or empty, the default
        folder ``bollinger_mean_reversion`` is used.

    Returns
    -------
    dict[str, Path]
        Keys: ``account``, ``metrics``, ``config``, ``plot``.
    """
    if window <= 0:
        raise ValueError("window must be positive.")
    if num_std <= 0:
        raise ValueError("num_std must be positive.")
    if transaction_cost_pct < 0:
        raise ValueError("transaction_cost_pct must be non-negative.")

    ticker = ticker.upper()
    df = load_trade_data_single_ticker(trade_data, ticker)

    if (not force_recompute_bands) and {"boll_ub", "boll_lb"}.issubset(df.columns):
        df["boll_upper"] = df["boll_ub"].astype(float)
        df["boll_lower"] = df["boll_lb"].astype(float)
        df["boll_middle"] = df["close"].rolling(window, min_periods=window).mean()
    else:
        df["boll_middle"] = df["close"].rolling(window, min_periods=window).mean()
        rolling_std = df["close"].rolling(window, min_periods=window).std(ddof=0)
        df["boll_upper"] = df["boll_middle"] + num_std * rolling_std
        df["boll_lower"] = df["boll_middle"] - num_std * rolling_std

    signal = []
    current_position = 0
    for close, lower, upper in zip(df["close"], df["boll_lower"], df["boll_upper"]):
        if pd.notna(lower) and close < lower:
            current_position = 1
        elif pd.notna(upper) and close > upper:
            current_position = 0
        signal.append(current_position)

    df["signal"] = signal
    df["position"] = df["signal"].shift(1).fillna(0.0)

    returns = df["close"].pct_change().fillna(0.0)
    position_change = df["position"].diff().abs().fillna(df["position"].abs())
    strategy_return = (df["position"] * returns) - (
        position_change * float(transaction_cost_pct)
    )
    account_value = float(initial_amount) * (1.0 + strategy_return).cumprod()

    account = pd.DataFrame(
        {
            "date": df["date"].dt.strftime("%Y-%m-%d"),
            "ticker": ticker,
            "close": df["close"],
            "boll_lower": df["boll_lower"],
            "boll_middle": df["boll_middle"],
            "boll_upper": df["boll_upper"],
            "signal": df["signal"],
            "position": df["position"],
            "account_value": account_value,
        }
    )

    if run_paths is None:
        _folder = (
            strategy_folder
            if (strategy_folder and strategy_folder.strip())
            else "bollinger_mean_reversion"
        )
        run_name = (
            f"d_iqn_dss_algorithmic_baseline_{_folder}_{dataset_tag}_{ticker.lower()}"
        )
        run_paths = create_run_paths(run_name)
    _sub = Path(output_subpath) if output_subpath else Path("")
    data_dir = run_paths.data_directory / _sub
    metrics_dir = run_paths.metrics_directory / _sub
    config_dir = run_paths.config_directory / _sub
    plots_dir = run_paths.plots_directory / _sub
    for d in [data_dir, metrics_dir, config_dir, plots_dir]:
        d.mkdir(parents=True, exist_ok=True)

    strategy = f"{ticker}_bollinger_mr_{window}_{num_std:g}"
    metrics = calculate_account_metrics(
        account,
        strategy=strategy,
        source="Algorithmic Trading / non-RL",
        initial_amount=initial_amount,
    )

    config = {
        "strategy": strategy,
        "dataset_tag": dataset_tag,
        "ticker": ticker,
        "trade_data": str(trade_data),
        "initial_amount": float(initial_amount),
        "window": int(window),
        "num_std": float(num_std),
        "force_recompute_bands": bool(force_recompute_bands),
        "transaction_cost_pct": float(transaction_cost_pct),
        "signal_rule": (
            "buy/invested if close < lower band; sell/cash if close > upper band;"
            " otherwise hold previous position; signal shifted by 1 day"
        ),
    }

    account_path = data_dir / f"{ticker.lower()}_bollinger_mr_account_values.csv"
    metrics_path = metrics_dir / f"{ticker.lower()}_bollinger_mr_metrics.csv"
    config_path = config_dir / f"{ticker.lower()}_bollinger_mr_config.json"
    plot_path = plots_dir / f"{ticker.lower()}_bollinger_mr_account_value.png"

    account.to_csv(account_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    with open(config_path, "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2)
    save_account_value_plot(
        account, output_path=plot_path, title=f"{strategy} — account value"
    )

    return {
        "account": account_path,
        "metrics": metrics_path,
        "config": config_path,
        "plot": plot_path,
    }
