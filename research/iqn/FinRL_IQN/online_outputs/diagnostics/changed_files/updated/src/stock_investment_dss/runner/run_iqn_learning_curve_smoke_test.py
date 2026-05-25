# src/stock_investment_dss/runner/run_iqn_learning_curve_smoke_test.py

from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from stock_investment_dss.data.finrl_data_pipeline import (
    load_or_create_finrl_daily_dataset,
)
from stock_investment_dss.data.point_in_time_split import create_point_in_time_split
from stock_investment_dss.decision.action_mask import DSSActionMaskGenerator
from stock_investment_dss.decision.decision_actions import action_to_label
from stock_investment_dss.decision.investor_risk_profile import InvestorRiskProfile
from stock_investment_dss.decision.risk_aware_action_resolver import (
    RiskAwareActionResolver,
)
from stock_investment_dss.environments.discrete_finrl_decision_env import (
    DiscreteFinRLDecisionEnv,
)
from stock_investment_dss.environments.finrl_env_factory import (
    FinRLStockTradingEnvConfig,
    create_finrl_stock_trading_env,
    unpack_reset_result,
)
from stock_investment_dss.evaluation.portfolio_metrics import (
    compute_portfolio_metrics,
    write_json,
)
from stock_investment_dss.rl.agents.iqn_agent import IQNAgent
from stock_investment_dss.rl.config.iqn_config import IQNConfig, build_iqn_config
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

ACTION_LABELS = {
    0: "HOLD",
    1: "BUY",
    2: "SELL",
    3: "REBALANCE",
    4: "CHANGE_STRATEGY",
}


def get_int_environment_variable(name: str, default: int) -> int:
    value = get_environment_variable(name, default=str(default))
    return int(value or default)


def get_float_environment_variable(name: str, default: float) -> float:
    value = get_environment_variable(name, default=str(default))
    return float(value or default)


def config_to_dict(config: IQNConfig) -> dict[str, Any]:
    data: dict[str, Any] = {}

    for key, value in vars(config).items():
        if isinstance(value, torch.device):
            data[key] = str(value)
        else:
            data[key] = value

    return data


def json_safe_value(value: Any) -> Any:
    """Convert common non-JSON values to JSON-safe structures."""

    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {str(key): json_safe_value(item) for key, item in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [json_safe_value(item) for item in value]

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    return str(value)


def get_metadata_value(
    metadata: dict[str, Any], *keys: str, default: Any = None
) -> Any:
    """Read the first available metadata key from a potentially sparse dict."""

    if not isinstance(metadata, dict):
        return default

    for key in keys:
        if key in metadata:
            return metadata[key]

    return default


def date_minus_one_day(date_text: str | None) -> str | None:
    if not date_text:
        return None

    try:
        return (pd.Timestamp(date_text) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    except Exception:
        return None


def parse_missing_expected_tickers_from_error(error_message: str) -> list[str]:
    """Parse messages like: Train split is missing expected tickers: ['AAPL']."""

    marker = "missing expected tickers:"
    if marker not in error_message:
        return []

    try:
        raw = error_message.split(marker, 1)[1].strip()
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, (list, tuple, set)):
            return sorted(str(item).upper() for item in parsed)
    except Exception:
        pass

    return []


def build_data_provenance_summary(
    *,
    dataset_id: str | None,
    universe_id: str | None,
    tickers: list[str] | tuple[str, ...] | None,
    start_date: str | None,
    end_date: str | None,
    use_cache: bool | None,
    allow_download: bool | None,
    force_download: bool | None,
    import_file: str | None,
    chunk_size: int | None,
    sleep_seconds: float | None,
    use_technical_indicators: bool | None,
    use_vix: bool | None,
    use_turbulence: bool | None,
    yfinance_impersonate: str | None,
    yfinance_timeout_seconds: int | None,
    daily_data_result: Any | None = None,
    failure: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a transparent data acquisition report for successful and failed runs."""

    metadata = (
        getattr(daily_data_result, "metadata", {})
        if daily_data_result is not None
        else {}
    )
    if not isinstance(metadata, dict):
        metadata = {}

    result_tickers = list(getattr(daily_data_result, "tickers", []) or [])
    requested_tickers = list(
        tickers or result_tickers or metadata.get("tickers", []) or []
    )

    failure_message = str((failure or {}).get("message") or "")
    missing_expected_tickers = parse_missing_expected_tickers_from_error(
        failure_message
    )

    metadata_failed_tickers = (
        get_metadata_value(
            metadata,
            "failed_tickers",
            "failed_downloads",
            "download_failed_tickers",
            default=[],
        )
        or []
    )
    metadata_downloaded_tickers = (
        get_metadata_value(
            metadata,
            "downloaded_tickers",
            "successful_tickers",
            "successful_downloads",
            default=[],
        )
        or []
    )

    failed_tickers = sorted(
        set(str(item).upper() for item in metadata_failed_tickers)
        | set(missing_expected_tickers)
    )
    successful_tickers = sorted(
        str(item).upper() for item in metadata_downloaded_tickers
    )

    actual_data_method = get_metadata_value(
        metadata,
        "actual_data_method",
        "data_source",
        "method",
        "source",
        default=None,
    )

    cache_used = bool(get_metadata_value(metadata, "cache_used", default=False))
    fallback_used = bool(get_metadata_value(metadata, "fallback_used", default=False))
    metadata_import_file = get_metadata_value(metadata, "import_file", default=None)
    import_file_used = bool(metadata_import_file) or (
        actual_data_method is not None and "import" in str(actual_data_method).lower()
    )

    download_attempted = bool(
        get_metadata_value(
            metadata,
            "download_attempted",
            default=bool(allow_download and (force_download or not cache_used)),
        )
    )
    download_success = None
    if download_attempted:
        download_success = bool(not failed_tickers and not fallback_used)

    return {
        "dataset_id": dataset_id or get_metadata_value(metadata, "dataset_id"),
        "universe_id": universe_id or get_metadata_value(metadata, "universe_id"),
        "requested_tickers": requested_tickers,
        "result_tickers": result_tickers,
        "start_date": start_date or get_metadata_value(metadata, "start_date"),
        "end_date": end_date or get_metadata_value(metadata, "end_date"),
        "requested_source": "FinRL/YahooDownloader/yfinance",
        "actual_data_method": actual_data_method,
        "final_source_used": (
            "canonical_cache"
            if cache_used
            else (
                "import_file"
                if import_file_used
                else (
                    "fresh_download"
                    if download_success
                    else (
                        "partial_or_failed_download"
                        if download_attempted
                        else "unknown"
                    )
                )
            )
        ),
        "use_cache_requested": use_cache,
        "cache_used": cache_used,
        "allow_download": allow_download,
        "force_download": force_download,
        "download_attempted": download_attempted,
        "download_success": download_success,
        "download_method_attempted": get_metadata_value(
            metadata, "download_method_attempted"
        ),
        "downloaded_tickers": successful_tickers,
        "failed_tickers": failed_tickers,
        "fallback_used": fallback_used,
        "fallback_reason": get_metadata_value(metadata, "fallback_reason"),
        "import_file_requested": import_file,
        "import_file_used": import_file_used,
        "import_file": metadata_import_file,
        "chunk_size": chunk_size,
        "sleep_seconds": sleep_seconds,
        "use_technical_indicators": use_technical_indicators,
        "use_vix": use_vix,
        "use_turbulence": use_turbulence,
        "yfinance_impersonate": yfinance_impersonate,
        "yfinance_timeout_seconds": yfinance_timeout_seconds,
        "raw_row_count": get_metadata_value(metadata, "raw_row_count"),
        "processed_row_count": get_metadata_value(metadata, "processed_row_count"),
        "metadata": json_safe_value(metadata),
        "failure": json_safe_value(failure),
    }


def build_experiment_context_summary(
    *,
    run_id: str | None,
    run_directory: str | None,
    status: str,
    dataset_id: str | None,
    universe_id: str | None,
    tickers: list[str] | tuple[str, ...] | None,
    start_date: str | None,
    end_date: str | None,
    split_id: str | None,
    point_in_time: str | None,
    trade_end_date: str | None,
    initial_amount: float | None,
    total_steps: int | None,
    learning_starts: int | None,
    eval_interval: int | None,
    random_seed: int | None,
    split_result: Any | None = None,
    final_eval: dict[str, Any] | None = None,
    failure: dict[str, Any] | None = None,
) -> dict[str, Any]:
    final_eval = final_eval or {}

    train_row_count = None
    trade_row_count = None
    resolved_split_id = split_id

    if split_result is not None:
        try:
            train_row_count = int(len(split_result.train_data))
            trade_row_count = int(len(split_result.trade_data))
            resolved_split_id = getattr(split_result, "split_id", split_id)
        except Exception:
            pass

    final_value = final_eval.get("final_value")
    profit_loss = final_eval.get("profit_loss")
    total_return_pct = final_eval.get("total_return_pct")

    made_money = None
    if profit_loss is not None:
        try:
            made_money = float(profit_loss) > 0.0
        except Exception:
            made_money = None

    return {
        "status": status,
        "run_id": run_id,
        "run_directory": run_directory,
        "dataset_id": dataset_id,
        "universe_id": universe_id,
        "tickers": list(tickers or []),
        "market_data_start": start_date,
        "market_data_end": end_date,
        "pit_cutoff": point_in_time,
        "split_id": resolved_split_id,
        "train_window_start": start_date,
        "train_window_end": date_minus_one_day(point_in_time),
        "eval_window_start": point_in_time,
        "eval_window_end": trade_end_date,
        "train_row_count": train_row_count,
        "trade_row_count": trade_row_count,
        "initial_capital": initial_amount,
        "final_value": final_value,
        "profit_loss": profit_loss,
        "total_return_pct": total_return_pct,
        "made_money": made_money,
        "annualized_sharpe": final_eval.get("annualized_sharpe"),
        "max_drawdown_pct": final_eval.get("max_drawdown_pct"),
        "cvar_pct": final_eval.get("cvar_pct"),
        "total_trades": final_eval.get("total_trades"),
        "total_transaction_cost": final_eval.get("total_transaction_cost"),
        "random_seed": random_seed,
        "total_steps": total_steps,
        "learning_starts": learning_starts,
        "eval_interval": eval_interval,
        "failure": json_safe_value(failure),
    }


def format_currency(value: Any) -> str:
    try:
        if value is None:
            return "unknown"
        return f"{float(value):,.2f}"
    except Exception:
        return str(value)


def format_pct(value: Any) -> str:
    try:
        if value is None:
            return "unknown"
        return f"{float(value):+.2f}%"
    except Exception:
        return str(value)


def render_experiment_context_markdown(
    context: dict[str, Any],
    provenance: dict[str, Any],
) -> str:
    made_money = context.get("made_money")
    if made_money is True:
        made_money_text = "Yes"
    elif made_money is False:
        made_money_text = "No"
    else:
        made_money_text = "unknown"

    failed_tickers = provenance.get("failed_tickers") or []
    downloaded_tickers = provenance.get("downloaded_tickers") or []

    lines = [
        "# Experiment Context Summary",
        "",
        "## Experiment Window",
        "",
        f"- Dataset ID: `{context.get('dataset_id')}`",
        f"- Universe: `{context.get('universe_id')}`",
        f"- Tickers: {', '.join(context.get('tickers') or [])}",
        "",
        f"- Market data start: `{context.get('market_data_start')}`",
        f"- Market data end: `{context.get('market_data_end')}`",
        f"- PIT cutoff: `{context.get('pit_cutoff')}`",
        f"- Train window: `{context.get('train_window_start')}` → `{context.get('train_window_end')}`",
        f"- Eval/trade window: `{context.get('eval_window_start')}` → `{context.get('eval_window_end')}`",
        f"- Train rows: `{context.get('train_row_count')}`",
        f"- Trade rows: `{context.get('trade_row_count')}`",
        "",
        "## Result",
        "",
        f"- Initial capital: `{format_currency(context.get('initial_capital'))}`",
        f"- Final value: `{format_currency(context.get('final_value'))}`",
        f"- Profit/Loss: `{format_currency(context.get('profit_loss'))}`",
        f"- Total return: `{format_pct(context.get('total_return_pct'))}`",
        f"- Made money: **{made_money_text}**",
        f"- Sharpe: `{context.get('annualized_sharpe')}`",
        f"- Max drawdown: `{format_pct(context.get('max_drawdown_pct'))}`",
        f"- CVaR: `{format_pct(context.get('cvar_pct'))}`",
        f"- Trades: `{context.get('total_trades')}`",
        "",
        "## Data Provenance",
        "",
        f"- Requested source: `{provenance.get('requested_source')}`",
        f"- Final source used: `{provenance.get('final_source_used')}`",
        f"- Actual data method: `{provenance.get('actual_data_method')}`",
        f"- Download attempted: `{provenance.get('download_attempted')}`",
        f"- Download success: `{provenance.get('download_success')}`",
        f"- Cache used: `{provenance.get('cache_used')}`",
        f"- Fallback used: `{provenance.get('fallback_used')}`",
        f"- Import file used: `{provenance.get('import_file_used')}`",
        f"- Import file: `{provenance.get('import_file')}`",
        f"- Downloaded tickers: {', '.join(downloaded_tickers) if downloaded_tickers else 'none / unknown'}",
        f"- Failed tickers: {', '.join(failed_tickers) if failed_tickers else 'none'}",
        f"- Raw rows: `{provenance.get('raw_row_count')}`",
        f"- Processed rows: `{provenance.get('processed_row_count')}`",
    ]

    if context.get("failure"):
        failure = context.get("failure") or {}
        lines.extend(
            [
                "",
                "## Failure",
                "",
                f"- Failure type: `{failure.get('type')}`",
                f"- Failure stage: `{failure.get('stage')}`",
                f"- Failure message: `{failure.get('message')}`",
            ]
        )

    return "\n".join(lines) + "\n"


def write_experiment_context_outputs(
    *,
    run_paths: Any,
    context: dict[str, Any],
    provenance: dict[str, Any],
) -> dict[str, str]:
    context_json_path = run_paths.summary_directory / "experiment_context_summary.json"
    context_md_path = run_paths.summary_directory / "experiment_context_summary.md"
    provenance_json_path = run_paths.data_directory / "data_provenance_summary.json"

    write_json(context_json_path, context)
    write_json(provenance_json_path, provenance)
    context_md_path.write_text(
        render_experiment_context_markdown(context, provenance),
        encoding="utf-8",
    )

    return {
        "experiment_context_summary_json_path": str(context_json_path),
        "experiment_context_summary_md_path": str(context_md_path),
        "data_provenance_summary_path": str(provenance_json_path),
    }


def log_experiment_context_to_terminal(
    logger: Any,
    context: dict[str, Any],
    provenance: dict[str, Any],
) -> None:
    logger.info("Experiment Window")
    logger.info("-----------------")
    logger.info("Dataset ID: %s", context.get("dataset_id"))
    logger.info("Universe: %s", context.get("universe_id"))
    logger.info("Tickers: %s", ", ".join(context.get("tickers") or []))
    logger.info("Market data start: %s", context.get("market_data_start"))
    logger.info("Market data end: %s", context.get("market_data_end"))
    logger.info("PIT cutoff: %s", context.get("pit_cutoff"))
    logger.info(
        "Train window: %s -> %s",
        context.get("train_window_start"),
        context.get("train_window_end"),
    )
    logger.info(
        "Eval/trade window: %s -> %s",
        context.get("eval_window_start"),
        context.get("eval_window_end"),
    )
    logger.info("Initial capital: %s", format_currency(context.get("initial_capital")))
    logger.info("Final value: %s", format_currency(context.get("final_value")))
    logger.info("Profit/Loss: %s", format_currency(context.get("profit_loss")))
    logger.info("Total return: %s", format_pct(context.get("total_return_pct")))
    logger.info("Made money: %s", context.get("made_money"))
    logger.info("Data source used: %s", provenance.get("final_source_used"))
    logger.info("Download attempted: %s", provenance.get("download_attempted"))
    logger.info("Download success: %s", provenance.get("download_success"))
    logger.info("Cache used: %s", provenance.get("cache_used"))
    logger.info("Import file used: %s", provenance.get("import_file_used"))
    logger.info("Failed tickers: %s", provenance.get("failed_tickers"))


def write_failure_provenance_outputs(
    *,
    run_paths: Any,
    local_values: dict[str, Any],
    exc: BaseException,
    logger: Any,
) -> None:
    """Write audit files even when the run fails before training/evaluation."""

    failure = {
        "type": type(exc).__name__,
        "message": str(exc),
        "stage": (
            "point_in_time_split_validation"
            if "split" in str(exc).lower() or "expected tickers" in str(exc).lower()
            else "unknown"
        ),
    }

    daily_data_result = local_values.get("daily_data_result")
    split_result = local_values.get("split_result")
    final_eval = local_values.get("final_eval") or {}

    result_tickers = list(getattr(daily_data_result, "tickers", []) or [])
    explicit_tickers_text = local_values.get("explicit_tickers_text")
    if explicit_tickers_text:
        requested_tickers = [
            item.strip().upper()
            for item in str(explicit_tickers_text).replace(";", ",").split(",")
            if item.strip()
        ]
    else:
        requested_tickers = result_tickers

    context = build_experiment_context_summary(
        run_id=getattr(run_paths, "run_id", None),
        run_directory=str(getattr(run_paths, "run_directory", "")),
        status="failed_before_training",
        dataset_id=local_values.get("dataset_id"),
        universe_id=local_values.get("universe_id"),
        tickers=requested_tickers or result_tickers,
        start_date=local_values.get("start_date"),
        end_date=local_values.get("end_date"),
        split_id=local_values.get("split_id"),
        point_in_time=local_values.get("point_in_time"),
        trade_end_date=local_values.get("trade_end_date"),
        initial_amount=local_values.get("initial_amount"),
        total_steps=local_values.get("total_steps"),
        learning_starts=local_values.get("learning_starts"),
        eval_interval=local_values.get("eval_interval"),
        random_seed=local_values.get("random_seed"),
        split_result=split_result,
        final_eval=final_eval,
        failure=failure,
    )

    provenance = build_data_provenance_summary(
        dataset_id=local_values.get("dataset_id"),
        universe_id=local_values.get("universe_id"),
        tickers=requested_tickers or result_tickers,
        start_date=local_values.get("start_date"),
        end_date=local_values.get("end_date"),
        use_cache=local_values.get("use_cache"),
        allow_download=local_values.get("allow_download"),
        force_download=local_values.get("force_download"),
        import_file=local_values.get("import_file"),
        chunk_size=local_values.get("chunk_size"),
        sleep_seconds=local_values.get("sleep_seconds"),
        use_technical_indicators=local_values.get("use_technical_indicators"),
        use_vix=local_values.get("use_vix"),
        use_turbulence=local_values.get("use_turbulence"),
        yfinance_impersonate=local_values.get("yfinance_impersonate"),
        yfinance_timeout_seconds=local_values.get("yfinance_timeout_seconds"),
        daily_data_result=daily_data_result,
        failure=failure,
    )

    output_paths = write_experiment_context_outputs(
        run_paths=run_paths,
        context=context,
        provenance=provenance,
    )

    failure_summary = {
        "status": "failed_before_training",
        "failure": failure,
        "experiment_context": context,
        "data_provenance": provenance,
        "outputs": output_paths,
    }
    failure_json_path = run_paths.summary_directory / "experiment_failure_summary.json"
    failure_md_path = run_paths.summary_directory / "experiment_failure_summary.md"
    write_json(failure_json_path, failure_summary)
    failure_md_path.write_text(
        render_experiment_context_markdown(context, provenance),
        encoding="utf-8",
    )

    log_experiment_context_to_terminal(logger, context, provenance)
    logger.info("Wrote failure provenance summary: %s", failure_json_path)


def make_wandb_key(value: str) -> str:
    """Return a W&B-safe metric/media key without path separators."""

    return (
        str(value)
        .replace("\\", "_")
        .replace("/", "_")
        .replace(" ", "_")
        .replace(":", "_")
    )


def make_wandb_table_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Convert nested/object-heavy values to W&B Table-safe scalar strings."""

    if dataframe.empty:
        return dataframe.copy()

    safe_df = dataframe.copy()

    for column in safe_df.columns:
        if safe_df[column].dtype == "object":
            safe_df[column] = safe_df[column].map(
                lambda value: (
                    json.dumps(value, ensure_ascii=False, sort_keys=True)
                    if isinstance(value, (dict, list, tuple, set))
                    else value
                )
            )

    return safe_df.replace({np.nan: None})


def create_wandb_runtime_config(
    *,
    run_id: str,
    random_seed: int,
    dataset_id: str,
    universe_id: str,
    tickers: list[str],
    start_date: str,
    end_date: str,
    split_id: str,
    point_in_time: str,
    trade_end_date: str,
    initial_amount: float,
    hmax: int,
    buy_cost_pct: float,
    sell_cost_pct: float,
    reward_scaling: float,
    total_steps: int,
    learning_starts: int,
    eval_interval: int,
    max_eval_steps: int,
    score_mode: str,
    risk_lambda: float,
    disable_change_strategy: bool,
    iqn_config: IQNConfig,
    risk_profile: InvestorRiskProfile,
) -> dict[str, Any]:
    """Build the W&B config payload for this IQN learning-curve run."""

    return {
        "project_name": "StockInvestmentDSS",
        "prototype_name": "D-IQN-DSS",
        "runner": "run_iqn_learning_curve_smoke_test",
        "run_id": run_id,
        "random_seed": random_seed,
        "dataset": {
            "dataset_id": dataset_id,
            "universe_id": universe_id,
            "tickers": tickers,
            "start_date": start_date,
            "end_date": end_date,
        },
        "point_in_time_split": {
            "split_id": split_id,
            "point_in_time": point_in_time,
            "trade_end_date": trade_end_date,
        },
        "environment": {
            "initial_amount": initial_amount,
            "hmax": hmax,
            "buy_cost_pct": buy_cost_pct,
            "sell_cost_pct": sell_cost_pct,
            "reward_scaling": reward_scaling,
        },
        "iqn_experiment": {
            "total_steps": total_steps,
            "learning_starts": learning_starts,
            "eval_interval": eval_interval,
            "max_eval_steps": max_eval_steps,
            "score_mode": score_mode,
            "risk_lambda": risk_lambda,
            "disable_change_strategy": disable_change_strategy,
        },
        "iqn_config": config_to_dict(iqn_config),
        "risk_profile": vars(risk_profile),
    }


def init_optional_wandb_run(
    *,
    run_paths: Any,
    config: dict[str, Any],
    logger: Any,
) -> Any | None:
    """Initialize W&B when enabled. Keep the experiment runnable without W&B."""

    if not get_boolean_environment_variable(
        "STOCK_INVESTMENT_DSS_WANDB_ENABLED",
        default=False,
    ):
        return None

    try:
        import wandb

        project = (
            get_environment_variable(
                "STOCK_INVESTMENT_DSS_WANDB_PROJECT",
                default="StockInvestmentDSS",
            )
            or "StockInvestmentDSS"
        )
        entity = get_environment_variable(
            "STOCK_INVESTMENT_DSS_WANDB_ENTITY",
            default=None,
        )
        group = (
            get_environment_variable(
                "STOCK_INVESTMENT_DSS_WANDB_GROUP",
                default="iqn-learning-curves",
            )
            or "iqn-learning-curves"
        )

        run = wandb.init(
            project=project,
            entity=entity or None,
            name=run_paths.run_id,
            group=group,
            job_type="iqn_learning_curve_smoke_test",
            config=config,
            dir=str(run_paths.run_directory),
            tags=[
                "iqn",
                "learning-curve",
                "distributional-rl",
                "stockdss",
            ],
        )
        logger.info("W&B run initialized: %s", getattr(run, "url", None))
        return run
    except Exception:
        logger.exception("W&B initialization failed. Continuing without W&B logging.")
        return None


def wandb_log_metrics(
    wandb_run: Any | None,
    metrics: dict[str, Any],
    *,
    step: int | None = None,
) -> None:
    """Safely log scalar metrics to W&B."""

    if wandb_run is None:
        return

    safe_metrics: dict[str, Any] = {}
    for key, value in metrics.items():
        if value is None:
            continue
        if isinstance(value, (str, bool, int, float)):
            safe_metrics[key] = value
        elif isinstance(value, np.generic):
            safe_metrics[key] = value.item()

    if not safe_metrics:
        return

    try:
        wandb_run.log(safe_metrics, step=step)
    except Exception:
        pass


def wandb_log_plot_images(
    wandb_run: Any | None,
    plot_paths: dict[str, str],
    logger: Any,
) -> None:
    """Log generated plots to W&B robustly.

    The primary guarantee is artifact logging. W&B image previews have shown
    intermittent Windows path-copy failures in this project, so previews are
    optional and disabled by default.

    Enable previews only when needed with:
        STOCK_INVESTMENT_DSS_WANDB_LOG_IMAGE_PREVIEWS=true
    """

    if wandb_run is None:
        return

    try:
        import wandb
    except Exception:
        logger.exception("Failed to import wandb for plot logging.")
        return

    existing_plot_paths: dict[str, Path] = {}
    for key, path_string in plot_paths.items():
        path = Path(path_string)
        if not path.exists() or path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        existing_plot_paths[make_wandb_key(key)] = path

    if not existing_plot_paths:
        return

    try:
        artifact = wandb.Artifact(
            name=f"{wandb_run.name}_plots",
            type="plots",
            description="IQN learning curve plots generated by StockInvestmentDSS.",
        )
        for safe_key, path in existing_plot_paths.items():
            artifact.add_file(str(path), name=f"{safe_key}{path.suffix.lower()}")
        wandb_run.log_artifact(artifact)
    except Exception:
        logger.exception("Failed to log W&B plot artifact.")

    log_image_previews = get_boolean_environment_variable(
        "STOCK_INVESTMENT_DSS_WANDB_LOG_IMAGE_PREVIEWS",
        default=False,
    )
    if not log_image_previews:
        return

    for safe_key, path in existing_plot_paths.items():
        try:
            wandb_run.log({f"plot_image_{safe_key}": wandb.Image(str(path))})
        except Exception as exc:
            logger.warning("Failed to log W&B plot image preview for %s: %s", path, exc)


def wandb_log_tables(
    wandb_run: Any | None,
    *,
    training_df: pd.DataFrame,
    episode_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    eval_step_df: pd.DataFrame | None = None,
    logger: Any,
) -> None:
    """Log compact diagnostic tables to W&B.

    W&B Tables infer a fixed schema. Nested dictionaries with different keys
    across rows, such as action_counts, therefore need to be serialized before
    table creation.
    """

    if wandb_run is None:
        return

    try:
        import wandb
    except Exception:
        logger.exception("Failed to import wandb for table logging.")
        return

    max_rows = get_int_environment_variable(
        "STOCK_INVESTMENT_DSS_WANDB_MAX_TABLE_ROWS",
        default=5000,
    )

    table_payload = {}

    try:
        if not eval_df.empty:
            safe_eval_df = make_wandb_table_dataframe(eval_df)
            table_payload["table_eval_records"] = wandb.Table(dataframe=safe_eval_df)
    except Exception:
        logger.exception("Failed to build W&B eval_records table.")

    try:
        if eval_step_df is not None and not eval_step_df.empty:
            compact_eval_step_df = eval_step_df.copy()
            if len(compact_eval_step_df) > max_rows:
                compact_eval_step_df = compact_eval_step_df.tail(max_rows)
            safe_eval_step_df = make_wandb_table_dataframe(compact_eval_step_df)
            table_payload["table_eval_step_records_tail"] = wandb.Table(
                dataframe=safe_eval_step_df
            )
    except Exception:
        logger.exception("Failed to build W&B eval_step_records table.")

    try:
        if not episode_df.empty:
            safe_episode_df = make_wandb_table_dataframe(episode_df)
            table_payload["table_episode_records"] = wandb.Table(
                dataframe=safe_episode_df
            )
    except Exception:
        logger.exception("Failed to build W&B episode_records table.")

    try:
        if not training_df.empty:
            compact_columns = [
                column
                for column in [
                    "step",
                    "episode",
                    "action_label",
                    "reward",
                    "loss",
                    "epsilon",
                    "effective_action",
                    "portfolio_value_after",
                    "cash_after",
                    "finrl_cost",
                    "finrl_trades",
                ]
                if column in training_df.columns
            ]
            compact_training_df = training_df[compact_columns].copy()
            if len(compact_training_df) > max_rows:
                compact_training_df = compact_training_df.tail(max_rows)
            safe_training_df = make_wandb_table_dataframe(compact_training_df)
            table_payload["table_training_records_tail"] = wandb.Table(
                dataframe=safe_training_df
            )
    except Exception:
        logger.exception("Failed to build W&B training_records_tail table.")

    if not table_payload:
        return

    try:
        wandb_run.log(table_payload)
    except Exception:
        logger.exception("Failed to log W&B tables.")


def wandb_log_artifacts(
    wandb_run: Any | None,
    *,
    run_id: str,
    output_paths: dict[str, str],
    model_path: Path,
    summary_path: Path,
    logger: Any,
) -> None:
    """Log saved files and model checkpoint as W&B artifacts."""

    if wandb_run is None:
        return

    try:
        import wandb

        safe_run_id = run_id.replace("/", "_").replace("\\", "_")

        outputs_artifact = wandb.Artifact(
            name=f"{safe_run_id}-outputs",
            type="iqn-learning-curve-outputs",
            description="Saved outputs from the IQN learning-curve diagnostic run.",
        )

        for path_string in output_paths.values():
            path = Path(path_string)
            if path.exists() and path.is_file():
                outputs_artifact.add_file(str(path), name=path.name)

        if summary_path.exists():
            outputs_artifact.add_file(str(summary_path), name=summary_path.name)

        wandb_run.log_artifact(outputs_artifact)

        if model_path.exists():
            model_artifact = wandb.Artifact(
                name=f"{safe_run_id}-model",
                type="model",
                description="IQN model checkpoint from the learning-curve diagnostic run.",
            )
            model_artifact.add_file(str(model_path), name=model_path.name)
            wandb_run.log_artifact(model_artifact)
    except Exception:
        logger.exception("Failed to log W&B artifacts.")


def wandb_finish(wandb_run: Any | None) -> None:
    if wandb_run is None:
        return

    try:
        wandb_run.finish()
    except Exception:
        pass


def create_risk_profile_from_environment() -> InvestorRiskProfile:
    preset = (
        (
            get_environment_variable(
                "STOCK_INVESTMENT_DSS_INVESTOR_RISK_PROFILE",
                default="balanced",
            )
            or "balanced"
        )
        .strip()
        .lower()
    )

    if preset == "defensive":
        return InvestorRiskProfile.defensive()

    if preset == "aggressive":
        return InvestorRiskProfile.aggressive()

    return InvestorRiskProfile.balanced()


def create_iqn_config_from_environment(
    total_steps: int,
    learning_starts: int,
    seed: int,
) -> IQNConfig:
    requested_device = (
        get_environment_variable(
            "STOCK_INVESTMENT_DSS_IQN_DEVICE",
            default="cuda" if torch.cuda.is_available() else "cpu",
        )
        or "cpu"
    )

    if requested_device == "cuda" and not torch.cuda.is_available():
        requested_device = "cpu"

    candidate_values = {
        "env_name": "D-IQN-DSS-FinRL-DiscreteDecisionEnv",
        "seed": seed,
        "total_steps": total_steps,
        "learning_starts": learning_starts,
        "device": torch.device(requested_device),
        "hidden_dim": get_int_environment_variable(
            "STOCK_INVESTMENT_DSS_IQN_HIDDEN_DIM",
            default=128,
        ),
        "cosine_embedding_dim": get_int_environment_variable(
            "STOCK_INVESTMENT_DSS_IQN_COSINE_EMBEDDING_DIM",
            default=64,
        ),
        "lr": get_float_environment_variable(
            "STOCK_INVESTMENT_DSS_IQN_LEARNING_RATE",
            default=1e-4,
        ),
        "gamma": get_float_environment_variable(
            "STOCK_INVESTMENT_DSS_IQN_GAMMA",
            default=0.99,
        ),
        "kappa": get_float_environment_variable(
            "STOCK_INVESTMENT_DSS_IQN_KAPPA",
            default=1.0,
        ),
        "batch_size": get_int_environment_variable(
            "STOCK_INVESTMENT_DSS_IQN_BATCH_SIZE",
            default=64,
        ),
        "replay_capacity": get_int_environment_variable(
            "STOCK_INVESTMENT_DSS_IQN_REPLAY_CAPACITY",
            default=100_000,
        ),
        "num_tau_samples": get_int_environment_variable(
            "STOCK_INVESTMENT_DSS_IQN_NUM_TAU_SAMPLES",
            default=32,
        ),
        "num_tau_prime_samples": get_int_environment_variable(
            "STOCK_INVESTMENT_DSS_IQN_NUM_TAU_PRIME_SAMPLES",
            default=32,
        ),
        "num_action_quantiles": get_int_environment_variable(
            "STOCK_INVESTMENT_DSS_IQN_NUM_ACTION_QUANTILES",
            default=32,
        ),
        "epsilon_start": get_float_environment_variable(
            "STOCK_INVESTMENT_DSS_IQN_EPSILON_START",
            default=1.0,
        ),
        "epsilon_final": get_float_environment_variable(
            "STOCK_INVESTMENT_DSS_IQN_EPSILON_FINAL",
            default=0.05,
        ),
        "epsilon_decay_steps": get_int_environment_variable(
            "STOCK_INVESTMENT_DSS_IQN_EPSILON_DECAY_STEPS",
            default=max(1, total_steps // 2),
        ),
        "target_update_interval": get_int_environment_variable(
            "STOCK_INVESTMENT_DSS_IQN_TARGET_UPDATE_INTERVAL",
            default=500,
        ),
    }

    config = build_iqn_config()

    # Runner-specific values are applied after the base config has been built.
    # This preserves StockDSS defaults, optional JSON/env presets, and still
    # lets this learning-curve runner define its concrete experiment settings.
    for key, value in candidate_values.items():
        if hasattr(config, key):
            setattr(config, key, value)

    return config


def get_buffer_length(agent: IQNAgent) -> int:
    try:
        return len(agent.replay_buffer)
    except TypeError:
        pass

    for attr in ["size", "length", "count"]:
        if hasattr(agent.replay_buffer, attr):
            value = getattr(agent.replay_buffer, attr)
            return int(value() if callable(value) else value)

    if hasattr(agent.replay_buffer, "buffer"):
        return len(agent.replay_buffer.buffer)

    if hasattr(agent.replay_buffer, "memory"):
        return len(agent.replay_buffer.memory)

    return 0


def add_transition_to_replay_buffer(
    agent: IQNAgent,
    state: np.ndarray,
    action: int,
    reward: float,
    next_state: np.ndarray,
    done: bool,
) -> None:
    transition = (state, action, reward, next_state, done)

    for method_name in ["add", "push", "store", "append"]:
        method = getattr(agent.replay_buffer, method_name, None)

        if method is None:
            continue

        for args in [
            (state, action, reward, next_state, done),
            transition,
        ]:
            try:
                if isinstance(args, tuple) and len(args) == 5:
                    method(*args)
                else:
                    method(args)
                return
            except TypeError:
                continue

    raise AttributeError(
        "Could not add transition to replay buffer. "
        "Expected replay buffer to expose add/push/store/append."
    )


def create_discrete_finrl_env(
    tickers: list[str],
    market_data: pd.DataFrame,
    initial_amount: float,
    hmax: int,
    buy_cost_pct: float,
    sell_cost_pct: float,
    reward_scaling: float,
    risk_profile: InvestorRiskProfile,
):
    finrl_env, prepared_data, finrl_env_metadata = create_finrl_stock_trading_env(
        market_data=market_data,
        tickers=tickers,
        config=FinRLStockTradingEnvConfig(
            initial_amount=initial_amount,
            hmax=hmax,
            buy_cost_pct=buy_cost_pct,
            sell_cost_pct=sell_cost_pct,
            reward_scaling=reward_scaling,
            print_verbosity=10_000,
        ),
        technical_indicators=None,
    )

    resolver = RiskAwareActionResolver(
        tickers=tickers,
        hmax=hmax,
        risk_profile=risk_profile,
    )

    action_mask_generator = DSSActionMaskGenerator(
        tickers=tickers,
        risk_profile=risk_profile,
        allow_change_strategy_without_signal=True,
    )

    env = DiscreteFinRLDecisionEnv(
        finrl_env=finrl_env,
        tickers=tickers,
        resolver=resolver,
        action_mask_generator=action_mask_generator,
        enforce_action_mask=True,
    )

    return env, prepared_data, finrl_env_metadata


def maybe_disable_change_strategy(
    action_mask: list[int] | np.ndarray | None,
    disable_change_strategy: bool,
) -> list[int] | np.ndarray | None:
    if action_mask is None:
        return action_mask

    if not disable_change_strategy:
        return action_mask

    if len(action_mask) >= 5:
        action_mask = list(action_mask)
        action_mask[4] = 0

    return action_mask


def score_action_distribution(
    values: dict[str, Any],
    score_mode: str,
    risk_lambda: float,
) -> float:
    """Score an IQN action return distribution for evaluation-time action choice.

    This function is the DSS/evaluation-time action-selection policy on top of
    the IQN distribution estimates. It is not the IQN training loss.

    Supported ablation modes:
    - mean
    - q50 / median
    - q25, q75, q90
    - cvar10
    - q50_minus_cvar_penalty
    - mean_minus_cvar_penalty
    - mean_plus_cvar10
    """

    normalized_mode = str(score_mode or "q50_minus_cvar_penalty").strip().lower()

    mean = float(values.get("mean", 0.0))
    q25 = float(values.get("q25", 0.0))
    q50 = float(values.get("q50", 0.0))
    q75 = float(values.get("q75", 0.0))
    q90 = float(values.get("q90", 0.0))
    cvar10 = float(values.get("cvar10", 0.0))

    if normalized_mode == "mean":
        return mean

    if normalized_mode in {"q50", "median"}:
        return q50

    if normalized_mode == "q25":
        return q25

    if normalized_mode == "q75":
        return q75

    if normalized_mode == "q90":
        return q90

    if normalized_mode == "cvar10":
        return cvar10

    if normalized_mode == "q50_minus_cvar_penalty":
        return q50 - risk_lambda * abs(cvar10)

    if normalized_mode == "mean_minus_cvar_penalty":
        return mean - risk_lambda * abs(cvar10)

    if normalized_mode == "mean_plus_cvar10":
        return mean + cvar10

    raise ValueError(
        f"Unsupported IQN score mode: {score_mode}. "
        "Use mean, q50, median, q25, q75, q90, cvar10, "
        "q50_minus_cvar_penalty, mean_minus_cvar_penalty, or mean_plus_cvar10."
    )


def choose_action_from_distribution(
    distribution_output: dict[str, Any],
    score_mode: str,
    risk_lambda: float,
) -> tuple[int, str, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []

    for action_label, values in distribution_output.get("distributions", {}).items():
        allowed = bool(values.get("allowed"))
        score = score_action_distribution(values, score_mode, risk_lambda)

        rows.append(
            {
                "action": action_label,
                "action_index": int(values.get("action_index")),
                "allowed": allowed,
                "score_mode": score_mode,
                "score": score,
                "mean": values.get("mean"),
                "q10": values.get("q10"),
                "q25": values.get("q25"),
                "q50": values.get("q50"),
                "q75": values.get("q75"),
                "q90": values.get("q90"),
                "cvar10": values.get("cvar10"),
            }
        )

    allowed_rows = [row for row in rows if row["allowed"]]

    if not allowed_rows:
        return 0, "HOLD", rows

    selected = sorted(allowed_rows, key=lambda row: row["score"], reverse=True)[0]
    return int(selected["action_index"]), str(selected["action"]), rows


def evaluate_iqn_agent(
    agent: IQNAgent,
    tickers: list[str],
    trade_data: pd.DataFrame,
    step: int,
    initial_amount: float,
    hmax: int,
    buy_cost_pct: float,
    sell_cost_pct: float,
    reward_scaling: float,
    risk_profile: InvestorRiskProfile,
    score_mode: str,
    risk_lambda: float,
    num_quantiles: int,
    max_eval_steps: int,
    disable_change_strategy: bool,
    state_norm_scale: float = 1.0,
) -> tuple[dict[str, Any], list[dict[str, Any]], pd.DataFrame, list[dict[str, Any]]]:
    env, _prepared_trade_data, _metadata = create_discrete_finrl_env(
        tickers=tickers,
        market_data=trade_data,
        initial_amount=initial_amount,
        hmax=hmax,
        buy_cost_pct=buy_cost_pct,
        sell_cost_pct=sell_cost_pct,
        reward_scaling=reward_scaling,
        risk_profile=risk_profile,
    )

    reset_result = env.reset()
    observation, _reset_info = unpack_reset_result(reset_result)
    state = np.asarray(observation, dtype=np.float32).reshape(-1)
    if state_norm_scale != 1.0:
        state = state / state_norm_scale

    was_training = agent.online_net.training
    agent.online_net.eval()
    agent.target_net.eval()

    decision_rows: list[dict[str, Any]] = []
    distribution_rows: list[dict[str, Any]] = []

    done = False
    eval_step = 0

    while not done and eval_step < max_eval_steps:
        action_mask_info = env.get_action_mask()
        action_mask = action_mask_info.get("mask_vector")
        action_mask = maybe_disable_change_strategy(
            action_mask,
            disable_change_strategy=disable_change_strategy,
        )

        distribution_output = agent.estimate_action_distributions(
            state=state,
            num_quantiles=num_quantiles,
            action_mask=action_mask,
        )

        action, action_label, estimate_rows = choose_action_from_distribution(
            distribution_output=distribution_output,
            score_mode=score_mode,
            risk_lambda=risk_lambda,
        )

        next_observation, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)

        decision_record = info.get("decision_record", {})

        for row in estimate_rows:
            distribution_rows.append(
                {
                    "train_step": step,
                    "eval_step": eval_step,
                    "chosen_action": action_label,
                    **row,
                }
            )

        state_after = decision_record.get("state_after") or {}
        state_before = decision_record.get("state_before") or {}
        resolved_action = decision_record.get("resolved_action") or {}
        execution_delta = decision_record.get("execution_delta") or {}

        holdings_before = state_before.get("holdings") or {}
        holdings_after = state_after.get("holdings") or {}
        executed_holdings_delta = execution_delta.get("executed_holdings_delta") or {}

        selected_ticker = resolved_action.get("selected_ticker")
        decision_rows.append(
            {
                "decision_id": f"iqn_eval_{int(step)}_{int(eval_step)}",
                "train_step": int(step),
                "eval_step": int(eval_step),
                "chosen_action_index": int(action),
                "chosen_action_label": action_label,
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "done": done,
                "effective_action": decision_record.get(
                    "effective_decision_action_label"
                ),
                "action_was_masked": decision_record.get("action_was_masked"),
                "selected_ticker": selected_ticker,
                "requested_shares": resolved_action.get("requested_shares"),
                "submitted_shares_estimate": resolved_action.get(
                    "submitted_shares_estimate"
                ),
                "hmax_limited": resolved_action.get("hmax_limited"),
                "requested_cash_value": resolved_action.get("requested_cash_value"),
                "submitted_cash_value_estimate": resolved_action.get(
                    "submitted_cash_value_estimate"
                ),
                "resolved_reason": resolved_action.get("reason"),
                "cash_before": state_before.get("cash"),
                "cash_after": state_after.get("cash"),
                "portfolio_value_before": state_before.get("portfolio_value"),
                "portfolio_value_after": state_after.get("portfolio_value"),
                "cash_delta": execution_delta.get("cash_delta"),
                "portfolio_value_delta": execution_delta.get("portfolio_value_delta"),
                "cost_delta": execution_delta.get("cost_delta"),
                "trades_delta": execution_delta.get("trades_delta"),
                "executed_shares_delta": execution_delta.get("executed_shares_delta"),
                "selected_ticker_holdings_before": (
                    holdings_before.get(selected_ticker) if selected_ticker else None
                ),
                "selected_ticker_holdings_after": (
                    holdings_after.get(selected_ticker) if selected_ticker else None
                ),
                "selected_ticker_holdings_delta": (
                    executed_holdings_delta.get(selected_ticker)
                    if selected_ticker
                    else None
                ),
                "holdings_before_json": json.dumps(holdings_before, sort_keys=True),
                "holdings_after_json": json.dumps(holdings_after, sort_keys=True),
                "holdings_delta_json": json.dumps(
                    executed_holdings_delta, sort_keys=True
                ),
            }
        )

        state = np.asarray(next_observation, dtype=np.float32).reshape(-1)
        if state_norm_scale != 1.0:
            state = state / state_norm_scale
        eval_step += 1

    asset_memory = env.save_asset_memory()

    metrics_result = compute_portfolio_metrics(
        asset_memory=asset_memory,
        decision_memory={"decisions": env.save_decision_memory()},
        step_table=pd.DataFrame(decision_rows),
    )

    action_counts = (
        pd.DataFrame(decision_rows)["chosen_action_label"].value_counts().to_dict()
        if decision_rows
        else {}
    )

    summary = {
        "train_step": int(step),
        "score_mode": score_mode,
        "risk_lambda": risk_lambda,
        "executed_eval_steps": int(eval_step),
        "action_counts": action_counts,
        **metrics_result.summary,
    }

    if was_training:
        agent.online_net.train()

    return summary, distribution_rows, asset_memory, decision_rows


def save_line_plot(
    data: pd.DataFrame,
    x_column: str,
    y_column: str,
    output_path: Path,
    title: str,
    x_label: str,
    y_label: str,
    group_column: str | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 5))

    if group_column and group_column in data.columns:
        for group_name, group_data in data.groupby(group_column):
            group_data = group_data.sort_values(x_column)
            plt.plot(group_data[x_column], group_data[y_column], label=str(group_name))
        plt.legend()
    else:
        data = data.sort_values(x_column)
        plt.plot(data[x_column], data[y_column])

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_marker_line_plot(
    data: pd.DataFrame,
    x_column: str,
    y_column: str,
    output_path: Path,
    title: str,
    x_label: str,
    y_label: str,
    zero_reference: bool = False,
) -> None:
    """Save a readable RL-style learning curve with explicit checkpoint markers."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    plot_data = data.dropna(subset=[x_column, y_column]).copy()
    if plot_data.empty:
        return

    plot_data = plot_data.sort_values(x_column)

    plt.figure(figsize=(12, 5))
    plt.plot(plot_data[x_column], plot_data[y_column], marker="o")

    if zero_reference:
        plt.axhline(0.0, linestyle="--", linewidth=1)

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_iqn_classic_eval_learning_curve(
    eval_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Save a classic RL-style evaluation learning curve.

    This is intentionally focused on periodic out-of-sample evaluation return,
    not training loss. It is the closest analogue to common RL learning curves
    where x = training timesteps and y = evaluation performance.
    """

    if "train_step" not in eval_df.columns or "total_return_pct" not in eval_df.columns:
        return

    save_marker_line_plot(
        data=eval_df,
        x_column="train_step",
        y_column="total_return_pct",
        output_path=output_path,
        title="IQN Classic Evaluation Learning Curve: Total Return",
        x_label="Training step",
        y_label="Evaluation total return (%)",
        zero_reference=True,
    )


def save_iqn_smoothed_episode_reward_curve(
    episode_df: pd.DataFrame,
    output_path: Path,
    window: int = 10,
) -> None:
    """Save a more classic RL-style episode reward curve with moving average."""

    if (
        "episode" not in episode_df.columns
        or "episode_reward" not in episode_df.columns
    ):
        return

    plot_df = episode_df.dropna(subset=["episode", "episode_reward"]).copy()
    if plot_df.empty:
        return

    plot_df = plot_df.sort_values("episode")
    rolling_window = min(window, max(1, len(plot_df)))
    plot_df["episode_reward_moving_average"] = (
        plot_df["episode_reward"]
        .rolling(
            window=rolling_window,
            min_periods=1,
        )
        .mean()
    )

    save_marker_line_plot(
        data=plot_df,
        x_column="episode",
        y_column="episode_reward_moving_average",
        output_path=output_path,
        title=f"IQN Classic Learning Curve: Episode Reward (MA-{rolling_window})",
        x_label="Episode",
        y_label="Episode reward moving average",
        zero_reference=True,
    )


def main() -> int:
    log_level = (
        get_environment_variable("STOCK_INVESTMENT_DSS_LOG_LEVEL", default="INFO")
        or "INFO"
    )

    system_logger = setup_system_logger(log_level=log_level)
    system_logger.info("Starting StockInvestmentDSS IQN learning curve smoke test.")
    system_logger.info("Project root: %s", PROJECT_ROOT)

    run_paths = None
    wandb_run = None

    try:
        random_seed = get_int_environment_variable(
            "STOCK_INVESTMENT_DSS_RANDOM_SEED",
            default=get_int_environment_variable(
                "STOCK_INVESTMENT_DSS_IQN_SEED",
                default=42,
            ),
        )
        set_global_seed(random_seed)

        explicit_tickers_text = get_environment_variable(
            "STOCK_INVESTMENT_DSS_FINRL_TICKERS", default=None
        ) or get_environment_variable(
            "STOCK_INVESTMENT_DSS_DAILY_DATA_TICKERS", default=None
        )
        universe_id = (
            get_environment_variable("STOCK_INVESTMENT_DSS_UNIVERSE_ID", default=None)
            or get_environment_variable(
                "STOCK_INVESTMENT_DSS_DAILY_DATA_UNIVERSE",
                default=("custom" if explicit_tickers_text else "demo_2"),
            )
            or ("custom" if explicit_tickers_text else "demo_2")
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
        split_id = (
            get_environment_variable(
                "STOCK_INVESTMENT_DSS_PIT_SPLIT_ID",
                default=f"{dataset_id}_pit",
            )
            or f"{dataset_id}_pit"
        )

        use_cache = get_boolean_environment_variable(
            "STOCK_INVESTMENT_DSS_DAILY_DATA_USE_CACHE", default=True
        )
        allow_download = get_boolean_environment_variable(
            "STOCK_INVESTMENT_DSS_DAILY_DATA_ALLOW_DOWNLOAD", default=True
        )
        force_download = get_boolean_environment_variable(
            "STOCK_INVESTMENT_DSS_FORCE_FINRL_DOWNLOAD", default=False
        )
        import_file = get_environment_variable(
            "STOCK_INVESTMENT_DSS_DAILY_DATA_IMPORT_FILE", default=None
        )
        chunk_size = get_int_environment_variable(
            "STOCK_INVESTMENT_DSS_DAILY_DATA_CHUNK_SIZE", default=1
        )
        sleep_seconds = get_float_environment_variable(
            "STOCK_INVESTMENT_DSS_DAILY_DATA_SLEEP_SECONDS", default=5.0
        )
        use_technical_indicators = get_boolean_environment_variable(
            "STOCK_INVESTMENT_DSS_DAILY_DATA_USE_TECHNICAL_INDICATORS",
            default=True,
        )
        use_vix = get_boolean_environment_variable(
            "STOCK_INVESTMENT_DSS_DAILY_DATA_USE_VIX", default=False
        )
        use_turbulence = get_boolean_environment_variable(
            "STOCK_INVESTMENT_DSS_DAILY_DATA_USE_TURBULENCE", default=False
        )
        yfinance_impersonate = (
            get_environment_variable(
                "STOCK_INVESTMENT_DSS_YFINANCE_IMPERSONATE",
                default="firefox135",
            )
            or "firefox135"
        )
        yfinance_timeout_seconds = get_int_environment_variable(
            "STOCK_INVESTMENT_DSS_YFINANCE_TIMEOUT_SECONDS", default=30
        )
        min_tickers_per_date = get_int_environment_variable(
            "STOCK_INVESTMENT_DSS_PIT_MIN_TICKERS_PER_DATE", default=0
        )

        initial_amount = get_float_environment_variable(
            "STOCK_INVESTMENT_DSS_FINRL_ENV_INITIAL_AMOUNT", default=1_000_000.0
        )
        hmax = get_int_environment_variable(
            "STOCK_INVESTMENT_DSS_FINRL_ENV_HMAX", default=10000
        )
        buy_cost_pct = get_float_environment_variable(
            "STOCK_INVESTMENT_DSS_FINRL_ENV_BUY_COST_PCT", default=0.001
        )
        sell_cost_pct = get_float_environment_variable(
            "STOCK_INVESTMENT_DSS_FINRL_ENV_SELL_COST_PCT", default=0.001
        )
        reward_scaling = get_float_environment_variable(
            "STOCK_INVESTMENT_DSS_FINRL_ENV_REWARD_SCALING", default=0.0001
        )

        total_steps = get_int_environment_variable(
            "STOCK_INVESTMENT_DSS_IQN_LEARNING_CURVE_TOTAL_STEPS",
            default=10_000,
        )
        learning_starts = get_int_environment_variable(
            "STOCK_INVESTMENT_DSS_IQN_LEARNING_STARTS", default=1000
        )
        eval_interval = get_int_environment_variable(
            "STOCK_INVESTMENT_DSS_IQN_LEARNING_CURVE_EVAL_INTERVAL",
            default=1000,
        )
        max_eval_steps = get_int_environment_variable(
            "STOCK_INVESTMENT_DSS_IQN_LEARNING_CURVE_MAX_EVAL_STEPS",
            default=10_000,
        )
        num_quantiles = get_int_environment_variable(
            "STOCK_INVESTMENT_DSS_IQN_BACKTEST_NUM_QUANTILES",
            default=128,
        )
        score_mode = (
            get_environment_variable(
                "STOCK_INVESTMENT_DSS_IQN_BACKTEST_SCORE_MODE",
                default="q50_minus_cvar_penalty",
            )
            or "q50_minus_cvar_penalty"
        ).strip()
        risk_lambda = get_float_environment_variable(
            "STOCK_INVESTMENT_DSS_IQN_BACKTEST_RISK_LAMBDA", default=0.75
        )
        disable_change_strategy = get_boolean_environment_variable(
            "STOCK_INVESTMENT_DSS_IQN_DISABLE_CHANGE_STRATEGY",
            default=True,
        )

        risk_profile = create_risk_profile_from_environment()
        iqn_config = create_iqn_config_from_environment(
            total_steps=total_steps,
            learning_starts=learning_starts,
            seed=random_seed,
        )

        run_paths = create_run_paths("d_iqn_dss_iqn_learning_curve_smoke_test")
        run_logger = setup_run_logger(run_paths, log_level=log_level)

        run_logger.info("Created run directory: %s", run_paths.run_directory)
        run_logger.info("Run id: %s", run_paths.run_id)
        run_logger.info("Random seed: %s", random_seed)
        run_logger.info("Dataset id: %s", dataset_id)
        run_logger.info("Universe id: %s", universe_id)
        if explicit_tickers_text:
            run_logger.info("Explicit ticker override: %s", explicit_tickers_text)
        run_logger.info("Point in time: %s", point_in_time)
        run_logger.info("Trade end date: %s", trade_end_date)
        run_logger.info("Total train steps: %s", total_steps)
        run_logger.info("Learning starts: %s", learning_starts)
        run_logger.info("Eval interval: %s", eval_interval)
        run_logger.info("Eval score mode: %s", score_mode)
        run_logger.info("Risk lambda: %s", risk_lambda)
        run_logger.info("Disable CHANGE_STRATEGY: %s", disable_change_strategy)
        run_logger.info("IQN config: %s", config_to_dict(iqn_config))

        wandb_config = create_wandb_runtime_config(
            run_id=run_paths.run_id,
            random_seed=random_seed,
            dataset_id=dataset_id,
            universe_id=universe_id,
            tickers=[],
            start_date=start_date,
            end_date=end_date,
            split_id=split_id,
            point_in_time=point_in_time,
            trade_end_date=trade_end_date,
            initial_amount=initial_amount,
            hmax=hmax,
            buy_cost_pct=buy_cost_pct,
            sell_cost_pct=sell_cost_pct,
            reward_scaling=reward_scaling,
            total_steps=total_steps,
            learning_starts=learning_starts,
            eval_interval=eval_interval,
            max_eval_steps=max_eval_steps,
            score_mode=score_mode,
            risk_lambda=risk_lambda,
            disable_change_strategy=disable_change_strategy,
            iqn_config=iqn_config,
            risk_profile=risk_profile,
        )
        wandb_run = init_optional_wandb_run(
            run_paths=run_paths,
            config=wandb_config,
            logger=run_logger,
        )

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

        tickers = list(daily_data_result.tickers)
        if wandb_run is not None:
            try:
                wandb_run.config.update(
                    {
                        "dataset.tickers": tickers,
                        "point_in_time_split.train_row_count": int(
                            len(split_result.train_data)
                        ),
                        "point_in_time_split.trade_row_count": int(
                            len(split_result.trade_data)
                        ),
                    },
                    allow_val_change=True,
                )
            except Exception:
                pass

        train_env, prepared_train_data, train_env_metadata = create_discrete_finrl_env(
            tickers=tickers,
            market_data=split_result.train_data,
            initial_amount=initial_amount,
            hmax=hmax,
            buy_cost_pct=buy_cost_pct,
            sell_cost_pct=sell_cost_pct,
            reward_scaling=reward_scaling,
            risk_profile=risk_profile,
        )

        reset_result = train_env.reset()
        observation, reset_info = unpack_reset_result(reset_result)
        state = np.asarray(observation, dtype=np.float32).reshape(-1)

        state_dim = int(state.shape[0])
        action_dim = int(train_env.action_space.n)

        if iqn_config.state_norm_scale != 1.0:
            state = state / iqn_config.state_norm_scale

        agent = IQNAgent(state_dim=state_dim, action_dim=action_dim, config=iqn_config)

        training_records: list[dict[str, Any]] = []
        episode_records: list[dict[str, Any]] = []
        eval_records: list[dict[str, Any]] = []
        eval_distribution_rows: list[dict[str, Any]] = []
        eval_step_records: list[dict[str, Any]] = []
        losses: list[float] = []

        episode_index = 1
        episode_reward = 0.0
        episode_steps = 0
        done = False

        # Initial untrained eval baseline.
        (
            initial_eval_summary,
            initial_distribution_rows,
            _initial_asset_memory,
            initial_step_rows,
        ) = evaluate_iqn_agent(
            agent=agent,
            tickers=tickers,
            trade_data=split_result.trade_data,
            step=0,
            initial_amount=initial_amount,
            hmax=hmax,
            buy_cost_pct=buy_cost_pct,
            sell_cost_pct=sell_cost_pct,
            reward_scaling=reward_scaling,
            risk_profile=risk_profile,
            score_mode=score_mode,
            risk_lambda=risk_lambda,
            num_quantiles=num_quantiles,
            max_eval_steps=max_eval_steps,
            disable_change_strategy=disable_change_strategy,
            state_norm_scale=iqn_config.state_norm_scale,
        )
        eval_records.append(initial_eval_summary)
        eval_distribution_rows.extend(initial_distribution_rows)
        eval_step_records.extend(initial_step_rows)
        wandb_log_metrics(
            wandb_run,
            {
                "eval/final_value": initial_eval_summary.get("final_value"),
                "eval/total_return_pct": initial_eval_summary.get("total_return_pct"),
                "eval/annualized_sharpe": initial_eval_summary.get("annualized_sharpe"),
                "eval/max_drawdown_pct": initial_eval_summary.get("max_drawdown_pct"),
                "eval/cvar_pct": initial_eval_summary.get("cvar_pct"),
                "eval/executed_steps": initial_eval_summary.get("executed_eval_steps"),
            },
            step=0,
        )

        for step in range(1, total_steps + 1):
            if done:
                episode_records.append(
                    {
                        "episode": episode_index,
                        "ended_at_step": step - 1,
                        "episode_reward": episode_reward,
                        "episode_steps": episode_steps,
                    }
                )

                reset_result = train_env.reset()
                observation, reset_info = unpack_reset_result(reset_result)
                state = np.asarray(observation, dtype=np.float32).reshape(-1)
                if iqn_config.state_norm_scale != 1.0:
                    state = state / iqn_config.state_norm_scale
                episode_index += 1
                episode_reward = 0.0
                episode_steps = 0
                done = False

            action_mask_info = train_env.get_action_mask()
            action_mask = action_mask_info.get("mask_vector")
            action_mask = maybe_disable_change_strategy(
                action_mask,
                disable_change_strategy=disable_change_strategy,
            )

            action = agent.select_action(
                state=state,
                step=step,
                eval_mode=False,
                action_mask=action_mask,
            )

            next_observation, reward, terminated, truncated, info = train_env.step(
                action
            )
            next_state = np.asarray(next_observation, dtype=np.float32).reshape(-1)
            if iqn_config.state_norm_scale != 1.0:
                next_state = next_state / iqn_config.state_norm_scale
            done = bool(terminated or truncated)

            add_transition_to_replay_buffer(
                agent=agent,
                state=state,
                action=int(action),
                reward=float(reward),
                next_state=next_state,
                done=done,
            )

            buffer_size = get_buffer_length(agent)
            loss_value = None

            if step >= learning_starts and buffer_size >= iqn_config.batch_size:
                loss_value = agent.learn()
                losses.append(loss_value)

            target_update_interval = int(
                getattr(iqn_config, "target_update_interval", 500)
            )
            if step > 0 and step % target_update_interval == 0:
                agent.update_target_network()

            episode_reward += float(reward)
            episode_steps += 1

            decision_record = info.get("decision_record", {})
            state_after = decision_record.get("state_after") or {}

            training_records.append(
                {
                    "step": step,
                    "episode": episode_index,
                    "action": int(action),
                    "action_label": action_to_label(action),
                    "reward": float(reward),
                    "episode_reward_so_far": episode_reward,
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "done": done,
                    "buffer_size": buffer_size,
                    "loss": loss_value,
                    "epsilon": agent.epsilon(step),
                    "effective_action": decision_record.get(
                        "effective_decision_action_label"
                    ),
                    "action_was_masked": decision_record.get("action_was_masked"),
                    "portfolio_value_after": state_after.get("portfolio_value"),
                    "cash_after": state_after.get("cash"),
                    "finrl_cost": decision_record.get("finrl_cost"),
                    "finrl_trades": decision_record.get("finrl_trades"),
                }
            )

            log_interval = int(getattr(iqn_config, "log_interval", 1000) or 1000)
            if step == 1 or step % log_interval == 0 or step == total_steps:
                wandb_log_metrics(
                    wandb_run,
                    {
                        "train/reward": float(reward),
                        "train/episode_reward_so_far": episode_reward,
                        "train/loss": loss_value,
                        "train/epsilon": agent.epsilon(step),
                        "train/buffer_size": buffer_size,
                        "train/portfolio_value_after": state_after.get(
                            "portfolio_value"
                        ),
                        "train/cash_after": state_after.get("cash"),
                        "train/finrl_cost": decision_record.get("finrl_cost"),
                        "train/finrl_trades": decision_record.get("finrl_trades"),
                    },
                    step=step,
                )

            state = next_state

            if step % eval_interval == 0 or step == total_steps:
                eval_summary, distribution_rows, _asset_memory, step_rows = (
                    evaluate_iqn_agent(
                        agent=agent,
                        tickers=tickers,
                        trade_data=split_result.trade_data,
                        step=step,
                        initial_amount=initial_amount,
                        hmax=hmax,
                        buy_cost_pct=buy_cost_pct,
                        sell_cost_pct=sell_cost_pct,
                        reward_scaling=reward_scaling,
                        risk_profile=risk_profile,
                        score_mode=score_mode,
                        risk_lambda=risk_lambda,
                        num_quantiles=num_quantiles,
                        max_eval_steps=max_eval_steps,
                        disable_change_strategy=disable_change_strategy,
                        state_norm_scale=iqn_config.state_norm_scale,
                    )
                )
                eval_records.append(eval_summary)
                eval_distribution_rows.extend(distribution_rows)
                eval_step_records.extend(step_rows)
                run_logger.info(
                    "Eval at step %s: return=%s sharpe=%s drawdown=%s actions=%s",
                    step,
                    eval_summary.get("total_return_pct"),
                    eval_summary.get("annualized_sharpe"),
                    eval_summary.get("max_drawdown_pct"),
                    eval_summary.get("action_counts"),
                )
                wandb_log_metrics(
                    wandb_run,
                    {
                        "eval/final_value": eval_summary.get("final_value"),
                        "eval/total_return_pct": eval_summary.get("total_return_pct"),
                        "eval/annualized_sharpe": eval_summary.get("annualized_sharpe"),
                        "eval/max_drawdown_pct": eval_summary.get("max_drawdown_pct"),
                        "eval/cvar_pct": eval_summary.get("cvar_pct"),
                        "eval/executed_steps": eval_summary.get("executed_eval_steps"),
                    },
                    step=step,
                )

        if episode_steps > 0:
            episode_records.append(
                {
                    "episode": episode_index,
                    "ended_at_step": total_steps,
                    "episode_reward": episode_reward,
                    "episode_steps": episode_steps,
                }
            )

        model_path = run_paths.models_directory / "iqn_learning_curve_model.pt"
        prepared_train_data_path = (
            run_paths.data_directory / "iqn_learning_curve_prepared_train_data.csv"
        )
        training_records_path = (
            run_paths.data_directory / "iqn_learning_curve_training_records.csv"
        )
        episode_records_path = (
            run_paths.data_directory / "iqn_learning_curve_episode_records.csv"
        )
        eval_records_path = (
            run_paths.data_directory / "iqn_learning_curve_eval_records.csv"
        )
        eval_distribution_path = (
            run_paths.data_directory / "iqn_learning_curve_eval_distributions.csv"
        )
        eval_step_records_path = (
            run_paths.data_directory / "iqn_learning_curve_eval_step_records.csv"
        )
        train_asset_memory_path = (
            run_paths.data_directory / "iqn_learning_curve_train_asset_memory.csv"
        )
        train_decision_memory_path = (
            run_paths.data_directory / "iqn_learning_curve_train_decision_memory.json"
        )

        agent.save(str(model_path))
        prepared_train_data.to_csv(prepared_train_data_path)

        training_df = pd.DataFrame(training_records)
        episode_df = pd.DataFrame(episode_records)
        eval_df = pd.DataFrame(eval_records)
        eval_distribution_df = pd.DataFrame(eval_distribution_rows)
        eval_step_df = pd.DataFrame(eval_step_records)

        training_df.to_csv(training_records_path, index=False)
        episode_df.to_csv(episode_records_path, index=False)
        eval_df.to_csv(eval_records_path, index=False)
        eval_distribution_df.to_csv(eval_distribution_path, index=False)
        eval_step_df.to_csv(eval_step_records_path, index=False)
        train_env.save_asset_memory().to_csv(train_asset_memory_path, index=False)
        write_json(
            train_decision_memory_path, {"decisions": train_env.save_decision_memory()}
        )

        plot_paths: dict[str, str] = {}

        if not training_df.empty:
            loss_df = training_df.dropna(subset=["loss"]).copy()
            if not loss_df.empty:
                loss_df["loss_moving_average"] = (
                    loss_df["loss"]
                    .rolling(
                        window=min(100, max(1, len(loss_df))),
                        min_periods=1,
                    )
                    .mean()
                )
                loss_plot_path = (
                    run_paths.summary_directory / "iqn_learning_curve_loss.png"
                )
                save_line_plot(
                    data=loss_df,
                    x_column="step",
                    y_column="loss_moving_average",
                    output_path=loss_plot_path,
                    title="IQN Learning Curve: Quantile Huber Loss",
                    x_label="Training step",
                    y_label="Loss moving average",
                )
                plot_paths["loss_plot_path"] = str(loss_plot_path)

            epsilon_plot_path = (
                run_paths.summary_directory / "iqn_learning_curve_epsilon.png"
            )
            save_line_plot(
                data=training_df,
                x_column="step",
                y_column="epsilon",
                output_path=epsilon_plot_path,
                title="IQN Exploration Schedule",
                x_label="Training step",
                y_label="Epsilon",
            )
            plot_paths["epsilon_plot_path"] = str(epsilon_plot_path)

            if "portfolio_value_after" in training_df.columns:
                portfolio_df = training_df.dropna(
                    subset=["portfolio_value_after"]
                ).copy()
                if not portfolio_df.empty:
                    portfolio_plot_path = (
                        run_paths.summary_directory / "iqn_training_portfolio_value.png"
                    )
                    save_line_plot(
                        data=portfolio_df,
                        x_column="step",
                        y_column="portfolio_value_after",
                        output_path=portfolio_plot_path,
                        title="IQN Training: Portfolio Value Over Time",
                        x_label="Training step",
                        y_label="Portfolio value",
                    )
                    plot_paths["training_portfolio_value_plot_path"] = str(
                        portfolio_plot_path
                    )

        if not episode_df.empty:
            episode_plot_path = (
                run_paths.summary_directory / "iqn_learning_curve_episode_reward.png"
            )
            save_line_plot(
                data=episode_df,
                x_column="episode",
                y_column="episode_reward",
                output_path=episode_plot_path,
                title="IQN Learning Curve: Episode Reward",
                x_label="Episode",
                y_label="Episode reward",
            )
            plot_paths["episode_reward_plot_path"] = str(episode_plot_path)

            smoothed_episode_plot_path = (
                run_paths.summary_directory
                / "iqn_learning_curve_episode_reward_smoothed.png"
            )
            save_iqn_smoothed_episode_reward_curve(
                episode_df=episode_df,
                output_path=smoothed_episode_plot_path,
                window=10,
            )
            plot_paths["smoothed_episode_reward_plot_path"] = str(
                smoothed_episode_plot_path
            )

        if not eval_df.empty:
            eval_plot_specs = [
                (
                    "final_value",
                    "iqn_eval_curve_final_value.png",
                    "IQN Evaluation Curve: Final Portfolio Value",
                    "Final portfolio value",
                ),
                (
                    "total_return_pct",
                    "iqn_eval_curve_total_return.png",
                    "IQN Evaluation Curve: Total Return",
                    "Total return (%)",
                ),
                (
                    "annualized_sharpe",
                    "iqn_eval_curve_sharpe.png",
                    "IQN Evaluation Curve: Sharpe Ratio",
                    "Annualized Sharpe",
                ),
                (
                    "max_drawdown_pct",
                    "iqn_eval_curve_max_drawdown.png",
                    "IQN Evaluation Curve: Maximum Drawdown",
                    "Max drawdown (%)",
                ),
            ]

            for column, filename, title, ylabel in eval_plot_specs:
                if column in eval_df.columns and eval_df[column].notna().any():
                    plot_path = run_paths.summary_directory / filename
                    save_line_plot(
                        data=eval_df.dropna(subset=[column]),
                        x_column="train_step",
                        y_column=column,
                        output_path=plot_path,
                        title=title,
                        x_label="Training step",
                        y_label=ylabel,
                    )
                    plot_paths[f"{column}_plot_path"] = str(plot_path)

            classic_eval_plot_path = (
                run_paths.summary_directory / "iqn_eval_learning_curve_classic.png"
            )
            save_iqn_classic_eval_learning_curve(
                eval_df=eval_df,
                output_path=classic_eval_plot_path,
            )
            plot_paths["classic_eval_learning_curve_plot_path"] = str(
                classic_eval_plot_path
            )

        final_eval = eval_records[-1] if eval_records else {}
        training_action_counts = (
            training_df["action_label"].value_counts().to_dict()
            if not training_df.empty and "action_label" in training_df.columns
            else {}
        )
        effective_action_counts = (
            training_df["effective_action"].fillna("UNKNOWN").value_counts().to_dict()
            if not training_df.empty and "effective_action" in training_df.columns
            else {}
        )

        summary = {
            "status": "ok",
            "project_name": "StockInvestmentDSS",
            "prototype_name": "D-IQN-DSS",
            "run_id": run_paths.run_id,
            "project_root": str(PROJECT_ROOT),
            "run_directory": str(run_paths.run_directory),
            "random_seed": random_seed,
            "dataset_id": dataset_id,
            "universe_id": universe_id,
            "tickers": tickers,
            "point_in_time": point_in_time,
            "trade_end_date": trade_end_date,
            "iqn": {
                "state_dim": state_dim,
                "action_dim": action_dim,
                "config": config_to_dict(iqn_config),
                "total_steps": total_steps,
                "learning_starts": learning_starts,
                "eval_interval": eval_interval,
                "eval_score_mode": score_mode,
                "risk_lambda": risk_lambda,
                "disable_change_strategy": disable_change_strategy,
                "final_buffer_size": get_buffer_length(agent),
                "learn_steps": len(losses),
                "loss_initial": losses[0] if losses else None,
                "loss_final": losses[-1] if losses else None,
                "loss_min": min(losses) if losses else None,
                "loss_max": max(losses) if losses else None,
                "loss_mean": float(np.mean(losses)) if losses else None,
                "training_action_counts": training_action_counts,
                "effective_action_counts": effective_action_counts,
                "final_eval": final_eval,
            },
            "point_in_time_split": {
                "split_id": split_result.split_id,
                "train_row_count": int(len(split_result.train_data)),
                "trade_row_count": int(len(split_result.trade_data)),
            },
            "train_environment": train_env_metadata,
            "outputs": {
                "model_path": str(model_path),
                "prepared_train_data_path": str(prepared_train_data_path),
                "training_records_path": str(training_records_path),
                "episode_records_path": str(episode_records_path),
                "eval_records_path": str(eval_records_path),
                "eval_distribution_path": str(eval_distribution_path),
                "eval_step_records_path": str(eval_step_records_path),
                "train_asset_memory_path": str(train_asset_memory_path),
                "train_decision_memory_path": str(train_decision_memory_path),
                **plot_paths,
            },
            "interpretation": (
                "This run is an IQN learning-curve diagnostic. Training loss is "
                "useful, but the main learning signal for the thesis should be "
                "periodic point-in-time evaluation on portfolio metrics such as "
                "return, Sharpe, max drawdown and CVaR. The classic evaluation "
                "learning curve plots periodic out-of-sample total return over "
                "training timesteps, which is closer to common RL learning curves."
            ),
            "next_step": (
                "Inspect eval curves. If performance remains flat or unstable, "
                "tune reward design, action masks, exploration schedule and train horizon."
            ),
        }

        experiment_context = build_experiment_context_summary(
            run_id=run_paths.run_id,
            run_directory=str(run_paths.run_directory),
            status="ok",
            dataset_id=dataset_id,
            universe_id=universe_id,
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            split_id=split_result.split_id,
            point_in_time=point_in_time,
            trade_end_date=trade_end_date,
            initial_amount=initial_amount,
            total_steps=total_steps,
            learning_starts=learning_starts,
            eval_interval=eval_interval,
            random_seed=random_seed,
            split_result=split_result,
            final_eval=final_eval,
            failure=None,
        )
        data_provenance = build_data_provenance_summary(
            dataset_id=dataset_id,
            universe_id=universe_id,
            tickers=tickers,
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
            daily_data_result=daily_data_result,
            failure=None,
        )
        context_output_paths = write_experiment_context_outputs(
            run_paths=run_paths,
            context=experiment_context,
            provenance=data_provenance,
        )
        log_experiment_context_to_terminal(
            run_logger,
            experiment_context,
            data_provenance,
        )
        summary["experiment_context"] = experiment_context
        summary["data_provenance"] = data_provenance
        summary["outputs"].update(context_output_paths)

        summary_path = run_paths.summary_directory / "iqn_learning_curve_summary.json"
        write_json(summary_path, summary)

        if wandb_run is not None:
            try:
                final_eval_for_wandb = final_eval or {}
                wandb_run.summary.update(
                    {
                        "run_id": run_paths.run_id,
                        "dataset_id": dataset_id,
                        "split_id": split_result.split_id,
                        "random_seed": random_seed,
                        "total_steps": total_steps,
                        "learning_starts": learning_starts,
                        "learn_steps": len(losses),
                        "loss_initial": losses[0] if losses else None,
                        "loss_final": losses[-1] if losses else None,
                        "loss_min": min(losses) if losses else None,
                        "loss_max": max(losses) if losses else None,
                        "loss_mean": float(np.mean(losses)) if losses else None,
                        "final_eval/final_value": final_eval_for_wandb.get(
                            "final_value"
                        ),
                        "final_eval/total_return_pct": final_eval_for_wandb.get(
                            "total_return_pct"
                        ),
                        "final_eval/annualized_sharpe": final_eval_for_wandb.get(
                            "annualized_sharpe"
                        ),
                        "final_eval/max_drawdown_pct": final_eval_for_wandb.get(
                            "max_drawdown_pct"
                        ),
                        "final_eval/cvar_pct": final_eval_for_wandb.get("cvar_pct"),
                        "final_eval/action_counts": final_eval_for_wandb.get(
                            "action_counts"
                        ),
                        "experiment/train_window": f"{experiment_context.get('train_window_start')}_to_{experiment_context.get('train_window_end')}",
                        "experiment/eval_window": f"{experiment_context.get('eval_window_start')}_to_{experiment_context.get('eval_window_end')}",
                        "experiment/market_data_start": experiment_context.get(
                            "market_data_start"
                        ),
                        "experiment/market_data_end": experiment_context.get(
                            "market_data_end"
                        ),
                        "experiment/pit_cutoff": experiment_context.get("pit_cutoff"),
                        "result/profit_loss": experiment_context.get("profit_loss"),
                        "result/total_return_pct": experiment_context.get(
                            "total_return_pct"
                        ),
                        "result/made_money": experiment_context.get("made_money"),
                        "data/final_source_used": data_provenance.get(
                            "final_source_used"
                        ),
                        "data/download_attempted": data_provenance.get(
                            "download_attempted"
                        ),
                        "data/download_success": data_provenance.get(
                            "download_success"
                        ),
                        "data/cache_used": data_provenance.get("cache_used"),
                        "data/import_file_used": data_provenance.get(
                            "import_file_used"
                        ),
                        "data/failed_ticker_count": len(
                            data_provenance.get("failed_tickers") or []
                        ),
                    }
                )
            except Exception:
                run_logger.exception("Failed to update W&B run summary.")

            wandb_log_plot_images(
                wandb_run=wandb_run,
                plot_paths=plot_paths,
                logger=run_logger,
            )
            wandb_log_tables(
                wandb_run=wandb_run,
                training_df=training_df,
                episode_df=episode_df,
                eval_df=eval_df,
                eval_step_df=eval_step_df,
                logger=run_logger,
            )
            wandb_log_artifacts(
                wandb_run=wandb_run,
                run_id=run_paths.run_id,
                output_paths=summary.get("outputs", {}),
                model_path=model_path,
                summary_path=summary_path,
                logger=run_logger,
            )
            wandb_finish(wandb_run)
            wandb_run = None

        run_logger.info("IQN learning curve smoke test completed.")
        run_logger.info("Training steps: %s", total_steps)
        run_logger.info("Learn steps: %s", len(losses))
        run_logger.info("Training action counts: %s", training_action_counts)
        run_logger.info("Final eval: %s", final_eval)
        run_logger.info("Wrote summary: %s", summary_path)

        system_logger.info(
            "StockInvestmentDSS IQN learning curve smoke test completed successfully."
        )

        return 0

    except Exception as exc:
        system_logger.exception(
            "StockInvestmentDSS IQN learning curve smoke test failed."
        )

        if run_paths is not None:
            try:
                run_logger = setup_run_logger(run_paths, log_level=log_level)
                run_logger.exception("Run failed.")
                write_failure_provenance_outputs(
                    run_paths=run_paths,
                    local_values=locals(),
                    exc=exc,
                    logger=run_logger,
                )
                if wandb_run is not None:
                    try:
                        wandb_run.summary.update(
                            {
                                "run_id": run_paths.run_id,
                                "status": "failed_before_training",
                                "failure/type": type(exc).__name__,
                                "failure/message": str(exc),
                            }
                        )
                        import wandb

                        failure_artifact = wandb.Artifact(
                            name=f"{run_paths.run_id}-failure-provenance",
                            type="failure-provenance",
                            description="Experiment context and data provenance written after a failed StockDSS run.",
                        )
                        for path in [
                            run_paths.summary_directory
                            / "experiment_context_summary.md",
                            run_paths.summary_directory
                            / "experiment_context_summary.json",
                            run_paths.summary_directory
                            / "experiment_failure_summary.md",
                            run_paths.summary_directory
                            / "experiment_failure_summary.json",
                            run_paths.data_directory / "data_provenance_summary.json",
                        ]:
                            if path.exists():
                                failure_artifact.add_file(str(path), name=path.name)
                        wandb_run.log_artifact(failure_artifact)
                    except Exception:
                        run_logger.exception(
                            "Failed to log W&B failure provenance artifact."
                        )
            except Exception:
                pass

        wandb_finish(wandb_run)

        return 1


if __name__ == "__main__":
    raise SystemExit(main())
