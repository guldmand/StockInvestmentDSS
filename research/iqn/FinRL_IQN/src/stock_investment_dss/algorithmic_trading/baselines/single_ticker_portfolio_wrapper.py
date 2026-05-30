"""Single-ticker strategy → equal-weight portfolio aggregation wrapper.

For each single-ticker strategy (buy_and_hold, sma_crossover, etc.), this
wrapper:

1. Splits ``initial_amount`` equally across all universe tickers (budget = N).
2. Runs the strategy independently on each ticker with its budget.
3. Aligns per-ticker account value series on the date index, sums them to
   produce a portfolio-level account value series.
4. Computes portfolio-level metrics via ``calculate_account_metrics``.

This enables fair head-to-head comparison of single-ticker strategies
(run as equal-weight portfolios) against portfolio-native strategies
(FinRL, IQN, naive-1/N) on the same capital base and PIT eval window.

Usage as module::

    from stock_investment_dss.algorithmic_trading.baselines.single_ticker_portfolio_wrapper import (
        SingleTickerPortfolioWrapper,
    )

Usage as script (for subprocess calls from demo runners)::

    python -m stock_investment_dss.algorithmic_trading.baselines.single_ticker_portfolio_wrapper \\
        --strategy-name buy_and_hold \\
        --market-data data/market/daily/imports/market_data_full_500.csv \\
        --tickers ALL \\
        --pit-start 2024-01-01 \\
        --pit-end 2026-05-26 \\
        --initial-amount 1000000 \\
        --dataset-tag sp500 \\
        --output-dir outputs/runs/<run_dir>
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from stock_investment_dss.metrics.trading_metrics import (
    calculate_account_metrics,
    save_account_value_plot,
)
from stock_investment_dss.utilities.paths import create_run_paths, RunPaths

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Strategy registry
# ---------------------------------------------------------------------------

STRATEGY_REGISTRY: Dict[str, Dict[str, Any]] = {
    "buy_and_hold": {
        "module": "stock_investment_dss.algorithmic_trading.baselines.buy_and_hold_pit_single_ticker",
        "function": "run_buy_and_hold",
        "params": {},
    },
    "sma_crossover": {
        "module": "stock_investment_dss.algorithmic_trading.baselines.sma_crossover_pit_single_ticker",
        "function": "run_sma_crossover",
        "params": {"fast_window": 10, "slow_window": 30},
    },
    "ema_crossover": {
        "module": "stock_investment_dss.algorithmic_trading.baselines.ema_crossover_pit_single_ticker",
        "function": "run_ema_crossover",
        "params": {"fast_window": 12, "slow_window": 26},
    },
    "macd_signal": {
        "module": "stock_investment_dss.algorithmic_trading.baselines.macd_signal_pit_single_ticker",
        "function": "run_macd_signal",
        "params": {"fast_window": 12, "slow_window": 26, "signal_window": 9},
    },
    "rsi_mean_reversion": {
        "module": "stock_investment_dss.algorithmic_trading.baselines.rsi_mean_reversion_pit_single_ticker",
        "function": "run_rsi_mean_reversion",
        "params": {"rsi_window": 14, "oversold": 30.0, "overbought": 70.0},
    },
    "bollinger_mean_reversion": {
        "module": "stock_investment_dss.algorithmic_trading.baselines.bollinger_mean_reversion_pit_single_ticker",
        "function": "run_bollinger_mean_reversion",
        "params": {"window": 20, "num_std": 2.0},
    },
    "momentum": {
        "module": "stock_investment_dss.algorithmic_trading.baselines.momentum_pit_single_ticker",
        "function": "run_momentum",
        "params": {"lookback_window": 20},
    },
    "breakout": {
        "module": "stock_investment_dss.algorithmic_trading.baselines.breakout_pit_single_ticker",
        "function": "run_breakout",
        "params": {"lookback_window": 20},
    },
    "volatility_filter": {
        "module": "stock_investment_dss.algorithmic_trading.baselines.volatility_filter_pit_single_ticker",
        "function": "run_volatility_filter",
        "params": {
            "momentum_window": 10,
            "volatility_window": 20,
            "max_annualized_volatility": 0.4,
        },
    },
}


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------


class SingleTickerPortfolioWrapper:
    """Aggregates per-ticker strategy results into a portfolio-level account series.

    Args:
        strategy_name: Key in ``STRATEGY_REGISTRY`` (e.g. ``"buy_and_hold"``).
        universe_tickers: Tickers to include in the equal-weight portfolio.
        market_df: Full market DataFrame (all tickers, all dates).
        pit_start_date: PIT eval window start (inclusive), e.g. ``"2024-01-01"``.
        pit_end_date: PIT eval window end (exclusive), e.g. ``"2026-05-26"``.
        initial_amount: Total portfolio capital; split equally across tickers.
        transaction_cost_pct: Fraction per trade applied to each ticker's bucket.
        dataset_tag: Dataset identifier used in output paths.
        run_paths: Pre-created ``RunPaths`` instance. Created fresh if ``None``.
        output_subpath: Subdirectory within run_paths directories for outputs.
    """

    def __init__(
        self,
        strategy_name: str,
        universe_tickers: List[str],
        market_df: pd.DataFrame,
        pit_start_date: str,
        pit_end_date: str,
        initial_amount: float = 1_000_000.0,
        transaction_cost_pct: float = 0.001,
        dataset_tag: str = "sp500",
        run_paths: Optional[RunPaths] = None,
        output_subpath: Optional[str] = None,
    ) -> None:
        if strategy_name not in STRATEGY_REGISTRY:
            raise ValueError(
                f"Unknown strategy '{strategy_name}'. "
                f"Available: {sorted(STRATEGY_REGISTRY.keys())}"
            )
        self.strategy_name = strategy_name
        self.universe_tickers = list(universe_tickers)
        self.market_df = market_df
        self.pit_start_date = pit_start_date
        self.pit_end_date = pit_end_date
        self.initial_amount = float(initial_amount)
        self.transaction_cost_pct = float(transaction_cost_pct)
        self.dataset_tag = dataset_tag
        if run_paths is None:
            run_paths = create_run_paths(
                f"d_iqn_dss_single_ticker_portfolio_{strategy_name}_{dataset_tag}"
            )
        self.run_paths = run_paths
        self.output_subpath = (
            output_subpath or f"single_ticker_portfolio/{strategy_name}"
        )

    def run(self) -> Dict[str, Path]:
        """Run strategy on all tickers, aggregate, compute metrics, write outputs.

        Returns:
            Dict with keys ``account``, ``metrics``, ``config``, ``plot`` pointing
            to the written output files.
        """
        registry_entry = STRATEGY_REGISTRY[self.strategy_name]
        mod = importlib.import_module(registry_entry["module"])
        strategy_fn = getattr(mod, registry_entry["function"])
        extra_params: Dict[str, Any] = dict(registry_entry["params"])

        n_tickers = len(self.universe_tickers)
        if n_tickers == 0:
            raise ValueError("universe_tickers is empty.")

        bucket = self.initial_amount / n_tickers

        # Option A: materialise the in-memory DataFrame to a temp CSV once so
        # that path-only strategy functions (sma_crossover, ema_crossover, etc.)
        # can call pd.read_csv without any changes to strategy files.  buy_and_hold
        # also accepts a path, so this is safe for all registered strategies.
        _tmp = tempfile.NamedTemporaryFile(
            suffix=".csv", prefix="stpw_market_", delete=False
        )
        _tmp.close()  # close before writing so pandas can open it on all OSes
        _market_data_path = Path(_tmp.name)
        self.market_df.to_csv(_market_data_path, index=False)
        log.debug(
            "Wrote temp market CSV: %s (%d rows)",
            _market_data_path,
            len(self.market_df),
        )

        per_ticker_series: List[pd.Series] = []
        failed_tickers: List[str] = []

        for ticker in self.universe_tickers:
            try:
                result = strategy_fn(
                    trade_data=_market_data_path,  # path-safe: all strategies accept str|Path
                    ticker=ticker,
                    dataset_tag=self.dataset_tag,
                    initial_amount=bucket,
                    transaction_cost_pct=self.transaction_cost_pct,
                    pit_start_date=self.pit_start_date,
                    pit_end_date=self.pit_end_date,
                    strategy_folder=self.strategy_name,
                    run_paths=self.run_paths,
                    output_subpath=(
                        f"{self.output_subpath}/per_ticker/{ticker.lower()}"
                    ),
                    **extra_params,
                )
                account_path = result["account"]
                df_ticker = pd.read_csv(account_path)
                if (
                    "date" not in df_ticker.columns
                    or "account_value" not in df_ticker.columns
                ):
                    raise ValueError(
                        f"Account CSV missing 'date' or 'account_value' columns: "
                        f"{account_path}"
                    )
                df_ticker["date"] = pd.to_datetime(df_ticker["date"])
                series = df_ticker.set_index("date")["account_value"].rename(ticker)
                per_ticker_series.append(series)
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "Strategy %s failed for ticker %s: %s",
                    self.strategy_name,
                    ticker,
                    exc,
                )
                failed_tickers.append(ticker)

        if not per_ticker_series:
            raise RuntimeError(
                f"All {n_tickers} tickers failed for strategy '{self.strategy_name}'. "
                "Cannot produce portfolio account values."
            )

        # Align on date intersection and sum across tickers
        combined = pd.concat(per_ticker_series, axis=1)
        combined = combined.ffill().dropna(axis=0, how="any")
        portfolio_series = combined.sum(axis=1)  # indexed by date

        n_succeeded = n_tickers - len(failed_tickers)
        portfolio_label = f"{self.strategy_name}_portfolio_{n_succeeded}tickers"

        account_df = pd.DataFrame(
            {
                "date": portfolio_series.index.strftime("%Y-%m-%d"),
                "account_value": portfolio_series.values,
                "strategy": portfolio_label,
            }
        )

        metrics = calculate_account_metrics(
            account_df,
            strategy=portfolio_label,
            source=f"single_ticker_portfolio/{self.strategy_name}",
            initial_amount=self.initial_amount,
        )

        # Write outputs
        out_root = self.run_paths.data_directory / self.output_subpath
        metrics_dir = self.run_paths.metrics_directory / self.output_subpath
        config_dir = self.run_paths.config_directory / self.output_subpath
        plots_dir = self.run_paths.plots_directory / self.output_subpath
        for d in (out_root, metrics_dir, config_dir, plots_dir):
            d.mkdir(parents=True, exist_ok=True)

        account_path = out_root / "portfolio_account_value.csv"
        metrics_path = metrics_dir / "portfolio_metrics.csv"
        config_path = config_dir / "portfolio_config.json"
        plot_path = plots_dir / "portfolio_account_value_plot.png"

        account_df.to_csv(account_path, index=False)
        metrics.to_csv(metrics_path, index=False)

        config = {
            "strategy_name": self.strategy_name,
            "strategy_params": extra_params,
            "n_tickers_requested": n_tickers,
            "n_tickers_succeeded": n_succeeded,
            "failed_tickers": failed_tickers,
            "initial_amount": self.initial_amount,
            "bucket_per_ticker": bucket,
            "transaction_cost_pct": self.transaction_cost_pct,
            "pit_start_date": self.pit_start_date,
            "pit_end_date": self.pit_end_date,
            "dataset_tag": self.dataset_tag,
        }
        with open(config_path, "w", encoding="utf-8") as fh:
            json.dump(config, fh, indent=2)

        try:
            save_account_value_plot(
                account_df,
                output_path=plot_path,
                title=f"{portfolio_label} — equal-weight portfolio account value",
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("Plot generation failed: %s", exc)

        log.info(
            "Portfolio %s: n=%d/%d succeeded, account=%s, metrics=%s",
            self.strategy_name,
            n_succeeded,
            n_tickers,
            account_path,
            metrics_path,
        )

        # Clean up temp market CSV written at the start of run().
        try:
            os.unlink(_market_data_path)
        except OSError:
            pass

        return {
            "account": account_path,
            "metrics": metrics_path,
            "config": config_path,
            "plot": plot_path,
        }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run a single-ticker strategy as an equal-weight portfolio.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--strategy-name",
        required=True,
        choices=sorted(STRATEGY_REGISTRY.keys()),
        help="Strategy to aggregate into a portfolio.",
    )
    p.add_argument(
        "--market-data",
        required=True,
        help="Path to market data CSV (all tickers, all dates).",
    )
    p.add_argument(
        "--tickers",
        default="ALL",
        help="Comma-separated ticker list, or 'ALL' to use all tickers in market data.",
    )
    p.add_argument(
        "--pit-start",
        required=True,
        help="PIT eval window start date (inclusive), e.g. 2024-01-01.",
    )
    p.add_argument(
        "--pit-end",
        required=True,
        help="PIT eval window end date (exclusive), e.g. 2026-05-26.",
    )
    p.add_argument(
        "--initial-amount",
        type=float,
        default=1_000_000.0,
        help="Total portfolio capital.",
    )
    p.add_argument(
        "--transaction-cost-pct",
        type=float,
        default=0.001,
        help="Transaction cost fraction per trade.",
    )
    p.add_argument(
        "--dataset-tag",
        default="sp500",
        help="Dataset identifier used in output paths.",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Override run directory. If not set, a timestamped run directory is created.",
    )
    return p


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    args = _build_parser().parse_args()

    log.info("Loading market data from %s", args.market_data)
    market_df = pd.read_csv(args.market_data)

    if args.tickers.upper() == "ALL":
        ticker_col = next(
            (c for c in ("tic", "ticker", "symbol") if c in market_df.columns),
            None,
        )
        if ticker_col is None:
            log.error(
                "Could not find ticker column (tic/ticker/symbol) in market data."
            )
            return 1
        tickers = sorted(
            market_df[ticker_col].astype(str).str.upper().unique().tolist()
        )
        log.info("Using ALL %d tickers from market data.", len(tickers))
    else:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
        log.info("Using %d tickers from --tickers argument.", len(tickers))

    run_paths: Optional[RunPaths] = None
    if args.output_dir is not None:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        run_paths = create_run_paths(
            f"d_iqn_dss_single_ticker_portfolio_{args.strategy_name}_{args.dataset_tag}"
        )

    wrapper = SingleTickerPortfolioWrapper(
        strategy_name=args.strategy_name,
        universe_tickers=tickers,
        market_df=market_df,
        pit_start_date=args.pit_start,
        pit_end_date=args.pit_end,
        initial_amount=args.initial_amount,
        transaction_cost_pct=args.transaction_cost_pct,
        dataset_tag=args.dataset_tag,
        run_paths=run_paths,
    )

    result = wrapper.run()
    log.info("Done. Outputs:")
    for key, path in result.items():
        log.info("  %s: %s", key, path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
