# src/stock_investment_dss/runner/run_finrl_baseline_suite_smoke_test.py

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from stock_investment_dss.baselines.finrl_baseline_suite import (
    FinRLBaselineSuiteConfig,
    normalize_agents,
    run_finrl_baseline_suite,
)
from stock_investment_dss.data.finrl_data_pipeline import (
    load_or_create_finrl_daily_dataset,
)
from stock_investment_dss.data.point_in_time_split import (
    create_point_in_time_split,
)
from stock_investment_dss.evaluation.portfolio_metrics import write_json
from stock_investment_dss.utilities.config import (
    get_boolean_environment_variable,
    get_environment_variable,
)
from stock_investment_dss.utilities.logging import (
    setup_run_logger,
    setup_system_logger,
)
from stock_investment_dss.utilities.paths import PROJECT_ROOT, create_run_paths
from stock_investment_dss.utilities.seed import set_global_seed


def get_int_environment_variable(name: str, default: int) -> int:
    value = get_environment_variable(name, default=str(default))
    return int(value or default)


def get_float_environment_variable(name: str, default: float) -> float:
    value = get_environment_variable(name, default=str(default))
    return float(value or default)


# ============================================================================
# RICH OUTPUT HELPERS (writes to plots/, metrics/, config/, plus W&B)
# ============================================================================


def _write_config_snapshot(
    run_paths: Any,
    summary: dict[str, Any],
    system_logger: Any,
) -> Path | None:
    """Write a full configuration snapshot to config/finrl_baseline_suite_config.json.

    Mirrors the algorithmic-baseline pattern where every run gets a config
    snapshot. Captures both the suite parameters and all
    STOCK_INVESTMENT_DSS_* environment variables for full reproducibility.
    """
    try:
        run_paths.config_directory.mkdir(parents=True, exist_ok=True)

        env_vars_snapshot = {
            key: value
            for key, value in os.environ.items()
            if key.startswith("STOCK_INVESTMENT_DSS_")
            and "API_KEY" not in key
            and "SECRET" not in key
        }

        config_payload = {
            "run_id": summary.get("run_id"),
            "run_directory": summary.get("run_directory"),
            "random_seed": summary.get("random_seed"),
            "baseline_suite": summary.get("baseline_suite", {}),
            "source_dataset": summary.get("source_dataset", {}),
            "point_in_time_split": summary.get("point_in_time_split", {}),
            "environment_variables": env_vars_snapshot,
        }

        config_path = run_paths.config_directory / "finrl_baseline_suite_config.json"
        config_path.write_text(
            json.dumps(config_payload, indent=2, default=str),
            encoding="utf-8",
        )
        system_logger.info("Wrote config snapshot: %s", config_path)
        return config_path
    except Exception as exc:
        system_logger.warning("Failed to write config snapshot: %s", exc)
        return None


def _write_per_agent_metrics(
    run_paths: Any,
    comparison_records: list[dict[str, Any]],
    system_logger: Any,
) -> list[Path]:
    """Write separate metrics JSON per agent to metrics/finrl_baseline_suite/{agent}/.

    The comparison_snapshot.csv contains one row per agent with all metrics.
    This helper splits it into per-agent JSONs for easier downstream consumption
    and to match the algorithmic-baselines output pattern.
    """
    written_paths: list[Path] = []
    metric_keys = (
        "rank",
        "agent_name",
        "status",
        "initial_value",
        "final_value",
        "profit_loss",
        "total_return_pct",
        "max_drawdown_pct",
        "annualized_volatility_pct",
        "annualized_sharpe",
        "cvar_pct",
        "total_transaction_cost",
        "total_trades",
        "turnover_estimate_pct",
        "finrl_trades",
        "finrl_cost",
        "trading_status",
        "action_mean_abs",
        "action_max_abs",
        "non_zero_action_steps",
    )

    suite_metrics_root = run_paths.metrics_directory / "finrl_baseline_suite"

    for row in comparison_records:
        try:
            agent_name = str(row.get("agent_name") or "unknown").lower()
            agent_dir = suite_metrics_root / agent_name
            agent_dir.mkdir(parents=True, exist_ok=True)

            metrics_payload = {key: row.get(key) for key in metric_keys if key in row}

            metrics_path = agent_dir / f"{agent_name}_metrics.json"
            metrics_path.write_text(
                json.dumps(metrics_payload, indent=2, default=str),
                encoding="utf-8",
            )
            written_paths.append(metrics_path)
        except Exception as exc:
            system_logger.warning(
                "Failed to write metrics for agent %s: %s",
                row.get("agent_name"),
                exc,
            )

    if written_paths:
        system_logger.info(
            "Wrote %d per-agent metrics JSONs to %s",
            len(written_paths),
            suite_metrics_root,
        )
    return written_paths


def _render_per_agent_plots(
    run_paths: Any,
    comparison_records: list[dict[str, Any]],
    system_logger: Any,
) -> list[Path]:
    """Render portfolio value + action plots per agent to plots/finrl_baseline_suite/{agent}/.

    Reads asset_memory.csv and action_memory.csv (already written by the suite)
    and produces matplotlib plots for visual inspection.
    """
    plots_written: list[Path] = []
    suite_plots_root = run_paths.plots_directory / "finrl_baseline_suite"

    for row in comparison_records:
        try:
            agent_name = str(row.get("agent_name") or "unknown").lower()
            agent_plot_dir = suite_plots_root / agent_name
            agent_plot_dir.mkdir(parents=True, exist_ok=True)

            # ---- Portfolio value plot from asset_memory ----
            asset_memory_path = row.get("asset_memory_path")
            if asset_memory_path and Path(asset_memory_path).exists():
                try:
                    asset_df = pd.read_csv(asset_memory_path)

                    fig, ax = plt.subplots(figsize=(10, 5))
                    # The asset_memory CSV typically has a single numeric column
                    # plus an index/date column. Plot the largest numeric column.
                    numeric_cols = asset_df.select_dtypes(include="number").columns
                    if len(numeric_cols) > 0:
                        # Pick the column whose values vary most (i.e. account value)
                        value_col = max(
                            numeric_cols,
                            key=lambda c: (
                                asset_df[c].std() if asset_df[c].std() > 0 else 0
                            ),
                        )
                        ax.plot(
                            asset_df.index,
                            asset_df[value_col],
                            color="#1f77b4",
                            linewidth=1.5,
                        )
                        ax.set_xlabel("Trade step")
                        ax.set_ylabel("Account value ($)")
                        ax.set_title(
                            f"{agent_name.upper()} portfolio value over trade window"
                        )
                        ax.grid(axis="y", alpha=0.3)
                        fig.tight_layout()

                        plot_path = agent_plot_dir / f"{agent_name}_portfolio_value.png"
                        fig.savefig(plot_path, dpi=140, bbox_inches="tight")
                        plt.close(fig)
                        plots_written.append(plot_path)
                except Exception as exc:
                    system_logger.warning(
                        "Failed to render portfolio value plot for %s: %s",
                        agent_name,
                        exc,
                    )

            # ---- Action over time plot from action_memory (RL agents only) ----
            action_memory_path = row.get("action_memory_path")
            if action_memory_path and Path(action_memory_path).exists():
                try:
                    action_df = pd.read_csv(action_memory_path)
                    numeric_action_cols = [
                        c
                        for c in action_df.columns
                        if pd.api.types.is_numeric_dtype(action_df[c])
                    ]

                    if numeric_action_cols:
                        # Plot 1: action over time (line per ticker)
                        fig, ax = plt.subplots(figsize=(12, 5))
                        for col in numeric_action_cols:
                            ax.plot(
                                action_df.index,
                                action_df[col],
                                label=col,
                                linewidth=0.8,
                                alpha=0.7,
                            )
                        ax.set_xlabel("Trade step")
                        ax.set_ylabel("Action value")
                        ax.set_title(
                            f"{agent_name.upper()} actions over time per ticker"
                        )
                        ax.axhline(0.0, linestyle="--", linewidth=0.8, color="gray")
                        ax.legend(
                            loc="upper right",
                            fontsize=7,
                            ncol=2,
                            framealpha=0.85,
                        )
                        ax.grid(axis="y", alpha=0.3)
                        fig.tight_layout()
                        plot_path = (
                            agent_plot_dir / f"{agent_name}_actions_over_time.png"
                        )
                        fig.savefig(plot_path, dpi=140, bbox_inches="tight")
                        plt.close(fig)
                        plots_written.append(plot_path)

                        # Plot 2: action distribution histogram (overlay all tickers)
                        fig, ax = plt.subplots(figsize=(10, 5))
                        all_actions = action_df[numeric_action_cols].values.flatten()
                        all_actions = all_actions[~np.isnan(all_actions)]
                        if len(all_actions) > 0:
                            ax.hist(
                                all_actions,
                                bins=50,
                                color="#1f77b4",
                                edgecolor="black",
                                alpha=0.8,
                            )
                            ax.set_xlabel("Action value")
                            ax.set_ylabel("Frequency")
                            ax.set_title(
                                f"{agent_name.upper()} action value distribution "
                                f"(all tickers, all steps)"
                            )
                            ax.axvline(0.0, linestyle="--", linewidth=0.8, color="red")
                            ax.grid(axis="y", alpha=0.3)
                            fig.tight_layout()
                            plot_path = (
                                agent_plot_dir / f"{agent_name}_action_distribution.png"
                            )
                            fig.savefig(plot_path, dpi=140, bbox_inches="tight")
                            plt.close(fig)
                            plots_written.append(plot_path)
                except Exception as exc:
                    system_logger.warning(
                        "Failed to render action plots for %s: %s",
                        agent_name,
                        exc,
                    )

        except Exception as exc:
            system_logger.warning(
                "Failed to render plots for agent %s: %s",
                row.get("agent_name"),
                exc,
            )

    if plots_written:
        system_logger.info(
            "Wrote %d per-agent plots to %s",
            len(plots_written),
            suite_plots_root,
        )
    return plots_written


def _log_outputs_to_wandb(
    run_paths: Any,
    summary: dict[str, Any],
    comparison_records: list[dict[str, Any]],
    system_logger: Any,
) -> None:
    """Log FinRL baseline suite outputs to W&B if enabled.

    Mirrors the pattern used by run_iqn_decision_audit_report.py.
    Gated by STOCK_INVESTMENT_DSS_WANDB_ENABLED env var.
    """
    if not get_boolean_environment_variable(
        "STOCK_INVESTMENT_DSS_WANDB_ENABLED", default=False
    ):
        return

    try:
        from stock_investment_dss.experiment_tracking.wandb_tracking import (
            init_wandb_run,
        )
    except Exception as exc:
        system_logger.warning("W&B not available for FinRL logging: %s", exc)
        return

    try:
        wandb_group = (
            get_environment_variable(
                "STOCK_INVESTMENT_DSS_WANDB_GROUP",
                default="finrl-baseline-suite",
            )
            or "finrl-baseline-suite"
        )

        wandb_run = init_wandb_run(
            run_name=str(summary.get("run_id") or run_paths.run_id),
            job_type="finrl_baseline_suite",
            group=wandb_group,
            tags=["finrl", "baseline", "sb3", "parametric-rl", "stockdss"],
            run_directory=str(run_paths.run_directory),
            config={
                "dataset_id": summary.get("source_dataset", {}).get("dataset_id"),
                "universe_id": summary.get("source_dataset", {}).get("universe_id"),
                "tickers": summary.get("source_dataset", {}).get("tickers"),
                "point_in_time": summary.get("point_in_time_split", {}).get(
                    "point_in_time"
                ),
                "trade_end_date": summary.get("point_in_time_split", {}).get(
                    "trade_end_date"
                ),
                "agents": summary.get("baseline_suite", {}).get("agents"),
                "total_timesteps": summary.get("baseline_suite", {}).get(
                    "total_timesteps"
                ),
                "include_mvo": summary.get("baseline_suite", {}).get("include_mvo"),
                "initial_amount": summary.get("baseline_suite", {}).get(
                    "initial_amount"
                ),
                "hmax": summary.get("baseline_suite", {}).get("hmax"),
                "buy_cost_pct": summary.get("baseline_suite", {}).get("buy_cost_pct"),
                "sell_cost_pct": summary.get("baseline_suite", {}).get("sell_cost_pct"),
                "reward_scaling": summary.get("baseline_suite", {}).get(
                    "reward_scaling"
                ),
                "seed": summary.get("random_seed"),
            },
        )
        if wandb_run is None:
            return

        import wandb

        # Per-agent scalar metrics
        for row in comparison_records:
            agent_name = row.get("agent_name") or "unknown"
            for metric_key in (
                "final_value",
                "total_return_pct",
                "max_drawdown_pct",
                "annualized_volatility_pct",
                "annualized_sharpe",
                "cvar_pct",
                "total_transaction_cost",
                "total_trades",
                "turnover_estimate_pct",
            ):
                value = row.get(metric_key)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    wandb.log({f"{agent_name}/{metric_key}": value})

        # Top-level seed (helpful for filtering across the W&B project)
        seed_value = summary.get("random_seed")
        if isinstance(seed_value, int):
            wandb.log({"meta/seed": seed_value})

        # Log key per-agent plots as W&B images for the dashboard
        plots_root = Path(run_paths.plots_directory) / "finrl_baseline_suite"
        if plots_root.exists():
            for png_path in plots_root.rglob("*.png"):
                try:
                    rel_path = png_path.relative_to(plots_root)
                    key = "/".join(rel_path.with_suffix("").parts)
                    wandb.log({f"plot/{key}": wandb.Image(str(png_path))})
                except Exception as exc:
                    system_logger.warning(
                        "Failed to log plot %s to W&B: %s", png_path, exc
                    )

        # Upload run artifacts (data/, summary/, models/, plots/, metrics/, config/)
        artifact_name = f"finrl_baseline_suite_{summary.get('run_id', 'unknown')}"
        artifact = wandb.Artifact(
            name=artifact_name,
            type="finrl-baseline-suite",
            description=(
                "FinRL baseline suite outputs: per-agent metrics, asset memory, "
                "action memory, trained SB3 models, plots, and config snapshot."
            ),
        )
        for folder_name in ("data", "summary", "models", "plots", "metrics", "config"):
            folder_path = Path(run_paths.run_directory) / folder_name
            if folder_path.exists() and any(folder_path.rglob("*")):
                artifact.add_dir(str(folder_path), name=folder_name)
        wandb_run.log_artifact(artifact)

        wandb_run.finish()
        system_logger.info("W&B logging completed for FinRL baseline suite run.")

    except Exception as exc:
        system_logger.warning(
            "W&B logging failed but local FinRL run completed: %s", exc
        )


# ============================================================================
# MAIN
# ============================================================================


def main() -> int:
    log_level = (
        get_environment_variable(
            "STOCK_INVESTMENT_DSS_LOG_LEVEL",
            default="INFO",
        )
        or "INFO"
    )

    system_logger = setup_system_logger(log_level=log_level)
    system_logger.info("Starting StockInvestmentDSS FinRL baseline suite smoke test.")
    system_logger.info("Project root: %s", PROJECT_ROOT)

    run_paths = None

    try:
        seed_value = int(
            get_environment_variable("STOCK_INVESTMENT_DSS_RANDOM_SEED", default=None)
            or get_environment_variable("STOCK_INVESTMENT_DSS_FINRL_SEED", default=None)
            or get_environment_variable("STOCK_INVESTMENT_DSS_SB3_SEED", default=None)
            or "42"
        )
        set_global_seed(seed_value)
        system_logger.info("Random seed: %d", seed_value)

        universe_id = (
            get_environment_variable(
                "STOCK_INVESTMENT_DSS_DAILY_DATA_UNIVERSE",
                default="demo_2",
            )
            or "demo_2"
        )

        dataset_id = (
            get_environment_variable(
                "STOCK_INVESTMENT_DSS_DAILY_DATASET_ID",
                default=universe_id,
            )
            or universe_id
        )

        start_date = (
            get_environment_variable(
                "STOCK_INVESTMENT_DSS_DAILY_DATA_START",
                default="2023-10-01",
            )
            or "2023-10-01"
        )

        end_date = (
            get_environment_variable(
                "STOCK_INVESTMENT_DSS_DAILY_DATA_END",
                default="2024-02-01",
            )
            or "2024-02-01"
        )

        use_cache = get_boolean_environment_variable(
            "STOCK_INVESTMENT_DSS_DAILY_DATA_USE_CACHE",
            default=True,
        )

        allow_download = get_boolean_environment_variable(
            "STOCK_INVESTMENT_DSS_DAILY_DATA_ALLOW_DOWNLOAD",
            default=True,
        )

        force_download = get_boolean_environment_variable(
            "STOCK_INVESTMENT_DSS_FORCE_FINRL_DOWNLOAD",
            default=False,
        )

        import_file = get_environment_variable(
            "STOCK_INVESTMENT_DSS_DAILY_DATA_IMPORT_FILE",
            default=None,
        )

        chunk_size = get_int_environment_variable(
            "STOCK_INVESTMENT_DSS_DAILY_DATA_CHUNK_SIZE",
            default=1,
        )

        sleep_seconds = get_float_environment_variable(
            "STOCK_INVESTMENT_DSS_DAILY_DATA_SLEEP_SECONDS",
            default=5.0,
        )

        use_technical_indicators = get_boolean_environment_variable(
            "STOCK_INVESTMENT_DSS_DAILY_DATA_USE_TECHNICAL_INDICATORS",
            default=True,
        )

        use_vix = get_boolean_environment_variable(
            "STOCK_INVESTMENT_DSS_DAILY_DATA_USE_VIX",
            default=False,
        )

        use_turbulence = get_boolean_environment_variable(
            "STOCK_INVESTMENT_DSS_DAILY_DATA_USE_TURBULENCE",
            default=False,
        )

        yfinance_impersonate = (
            get_environment_variable(
                "STOCK_INVESTMENT_DSS_YFINANCE_IMPERSONATE",
                default="firefox135",
            )
            or "firefox135"
        )

        yfinance_timeout_seconds = get_int_environment_variable(
            "STOCK_INVESTMENT_DSS_YFINANCE_TIMEOUT_SECONDS",
            default=30,
        )

        split_id = (
            get_environment_variable(
                "STOCK_INVESTMENT_DSS_PIT_SPLIT_ID",
                default=f"{dataset_id}_pit",
            )
            or f"{dataset_id}_pit"
        )

        point_in_time = (
            get_environment_variable(
                "STOCK_INVESTMENT_DSS_PIT_POINT_IN_TIME",
                default="2024-01-16",
            )
            or "2024-01-16"
        )

        trade_end_date = get_environment_variable(
            "STOCK_INVESTMENT_DSS_PIT_TRADE_END_DATE",
            default=end_date,
        )

        min_tickers_per_date = get_int_environment_variable(
            "STOCK_INVESTMENT_DSS_PIT_MIN_TICKERS_PER_DATE",
            default=0,
        )

        raw_agents = (
            get_environment_variable(
                "STOCK_INVESTMENT_DSS_FINRL_BASELINE_AGENTS",
                default="a2c,ddpg,td3,ppo,sac,mvo",
            )
            or "a2c,ddpg,td3,ppo,sac,mvo"
        )

        include_mvo = get_boolean_environment_variable(
            "STOCK_INVESTMENT_DSS_FINRL_BASELINE_INCLUDE_MVO",
            default=True,
        )

        agents = normalize_agents(
            raw_agents=raw_agents,
            include_mvo=include_mvo,
        )

        total_timesteps = get_int_environment_variable(
            "STOCK_INVESTMENT_DSS_FINRL_BASELINE_TOTAL_TIMESTEPS",
            default=500,
        )

        initial_amount = get_float_environment_variable(
            "STOCK_INVESTMENT_DSS_FINRL_ENV_INITIAL_AMOUNT",
            default=1_000_000.0,
        )

        hmax = get_int_environment_variable(
            "STOCK_INVESTMENT_DSS_FINRL_ENV_HMAX",
            default=10000,
        )

        buy_cost_pct = get_float_environment_variable(
            "STOCK_INVESTMENT_DSS_FINRL_ENV_BUY_COST_PCT",
            default=0.001,
        )

        sell_cost_pct = get_float_environment_variable(
            "STOCK_INVESTMENT_DSS_FINRL_ENV_SELL_COST_PCT",
            default=0.001,
        )

        reward_scaling = get_float_environment_variable(
            "STOCK_INVESTMENT_DSS_FINRL_ENV_REWARD_SCALING",
            default=0.0001,
        )

        device = (
            get_environment_variable(
                "STOCK_INVESTMENT_DSS_FINRL_BASELINE_DEVICE",
                default="auto",
            )
            or "auto"
        )

        deterministic_backtest = get_boolean_environment_variable(
            "STOCK_INVESTMENT_DSS_FINRL_BASELINE_DETERMINISTIC_BACKTEST",
            default=True,
        )

        run_paths = create_run_paths("d_iqn_dss_finrl_baseline_suite_smoke_test")
        run_logger = setup_run_logger(run_paths, log_level=log_level)

        run_logger.info("Created run directory: %s", run_paths.run_directory)
        run_logger.info("Run id: %s", run_paths.run_id)
        run_logger.info("Dataset id: %s", dataset_id)
        run_logger.info("Universe id: %s", universe_id)
        run_logger.info("Date range: %s -> %s", start_date, end_date)
        run_logger.info("Point in time: %s", point_in_time)
        run_logger.info("Trade end date: %s", trade_end_date)
        run_logger.info("Baseline agents: %s", agents)
        run_logger.info("Total timesteps per RL agent: %s", total_timesteps)
        run_logger.info("Initial amount: %s", initial_amount)
        run_logger.info("hmax: %s", hmax)
        run_logger.info("Device: %s", device)
        run_logger.info("Include MVO: %s", include_mvo)

        daily_data_result = load_or_create_finrl_daily_dataset(
            universe_id=universe_id,
            dataset_id=dataset_id,
            start_date=start_date,
            end_date=end_date,
            use_cache=use_cache,
            allow_download=allow_download,
            force_download=force_download,
            import_file=import_file,
            chunk_size=chunk_size,
            sleep_seconds=sleep_seconds,
            use_technical_indicators=use_technical_indicators,
            use_vix=use_vix,
            use_turbulence=use_turbulence,
            yfinance_impersonate=yfinance_impersonate,
            yfinance_timeout_seconds=yfinance_timeout_seconds,
        )

        split_result = create_point_in_time_split(
            data=daily_data_result.processed_data,
            split_id=split_id,
            point_in_time=point_in_time,
            trade_end_date=trade_end_date,
            expected_tickers=daily_data_result.tickers,
            min_tickers_per_date=(
                min_tickers_per_date if min_tickers_per_date > 0 else None
            ),
            source_metadata=daily_data_result.metadata,
        )

        suite_output_directory = run_paths.data_directory / "finrl_baseline_suite"
        suite_model_directory = run_paths.models_directory / "finrl_baseline_suite"

        suite_config = FinRLBaselineSuiteConfig(
            agents=agents,
            total_timesteps=total_timesteps,
            initial_amount=initial_amount,
            hmax=hmax,
            buy_cost_pct=buy_cost_pct,
            sell_cost_pct=sell_cost_pct,
            reward_scaling=reward_scaling,
            device=device,
            deterministic_backtest=deterministic_backtest,
            include_mvo=include_mvo,
        )

        suite_result = run_finrl_baseline_suite(
            train_data=split_result.train_data,
            trade_data=split_result.trade_data,
            tickers=list(daily_data_result.tickers),
            output_directory=suite_output_directory,
            model_directory=suite_model_directory,
            config=suite_config,
        )

        comparison = suite_result["comparison"]
        comparison_path = suite_result["comparison_path"]

        summary = {
            "status": "ok",
            "project_name": "StockInvestmentDSS",
            "prototype_name": "D-IQN-DSS",
            "run_id": run_paths.run_id,
            "project_root": str(PROJECT_ROOT),
            "run_directory": str(run_paths.run_directory),
            "random_seed": seed_value,
            "source_dataset": {
                "dataset_id": daily_data_result.dataset_id,
                "universe_id": daily_data_result.universe_id,
                "tickers": list(daily_data_result.tickers),
                "actual_data_method": daily_data_result.metadata.get(
                    "actual_data_method"
                ),
                "cache_origin_data_method": daily_data_result.metadata.get(
                    "cache_origin_data_method"
                ),
                "fallback_used": daily_data_result.metadata.get("fallback_used"),
            },
            "point_in_time_split": {
                "split_id": split_result.split_id,
                "point_in_time": split_result.point_in_time,
                "trade_end_date": trade_end_date,
                "train_row_count": int(len(split_result.train_data)),
                "trade_row_count": int(len(split_result.trade_data)),
            },
            "baseline_suite": {
                "agents": agents,
                "rl_agents": [agent for agent in agents if agent != "mvo"],
                "include_mvo": include_mvo,
                "total_timesteps": total_timesteps,
                "initial_amount": initial_amount,
                "hmax": hmax,
                "buy_cost_pct": buy_cost_pct,
                "sell_cost_pct": sell_cost_pct,
                "reward_scaling": reward_scaling,
                "device": device,
                "deterministic_backtest": deterministic_backtest,
                "suite_output_directory": str(suite_output_directory),
                "suite_model_directory": str(suite_model_directory),
                "comparison_path": str(comparison_path),
            },
            "results": suite_result["summary"].get("results", []),
            "next_step": (
                "Use the FinRL baseline suite comparison as the blue baseline layer "
                "against discrete DSS, IQN, and later IQN+EDL variants."
            ),
        }

        summary_path = (
            run_paths.summary_directory / "finrl_baseline_suite_smoke_summary.json"
        )

        write_json(summary_path, summary)

        comparison_snapshot_path = (
            run_paths.summary_directory / "finrl_baseline_suite_comparison_snapshot.csv"
        )
        comparison.to_csv(comparison_snapshot_path, index=False)

        run_logger.info("FinRL baseline suite smoke test completed.")
        run_logger.info("Agents: %s", agents)
        run_logger.info("Comparison path: %s", comparison_path)
        run_logger.info("Comparison snapshot: %s", comparison_snapshot_path)
        run_logger.info("Summary path: %s", summary_path)

        comparison_records = (
            comparison.to_dict(orient="records") if not comparison.empty else []
        )

        if not comparison.empty:
            run_logger.info("Comparison table:")
            for row in comparison_records:
                run_logger.info(
                    "rank=%s agent=%s final_value=%s total_return_pct=%s max_drawdown_pct=%s sharpe=%s",
                    row.get("rank"),
                    row.get("agent_name"),
                    row.get("final_value"),
                    row.get("total_return_pct"),
                    row.get("max_drawdown_pct"),
                    row.get("annualized_sharpe"),
                )

        # ====================================================================
        # RICH OUTPUT WRITES (config/, metrics/, plots/, plus W&B)
        # ====================================================================
        _write_config_snapshot(run_paths, summary, system_logger)
        _write_per_agent_metrics(run_paths, comparison_records, system_logger)
        _render_per_agent_plots(run_paths, comparison_records, system_logger)
        _log_outputs_to_wandb(
            run_paths=run_paths,
            summary=summary,
            comparison_records=comparison_records,
            system_logger=system_logger,
        )

        system_logger.info(
            "StockInvestmentDSS FinRL baseline suite smoke test completed successfully."
        )

        return 0

    except Exception:
        system_logger.exception(
            "StockInvestmentDSS FinRL baseline suite smoke test failed."
        )

        if run_paths is not None:
            try:
                run_logger = setup_run_logger(run_paths, log_level=log_level)
                run_logger.exception("Run failed.")
            except Exception:
                pass

        return 1


if __name__ == "__main__":
    raise SystemExit(main())
