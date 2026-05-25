# src/stock_investment_dss/runner/run_iqn_vs_baseline_comparison_plot.py

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from stock_investment_dss.utilities.config import get_environment_variable
from stock_investment_dss.utilities.logging import (
    setup_run_logger,
    setup_system_logger,
)
from stock_investment_dss.utilities.paths import PROJECT_ROOT, create_run_paths

COMPARISON_SUMMARY_FILENAME = "iqn_vs_baseline_comparison_summary.csv"
DEFAULT_INITIAL_AMOUNT = 1_000_000.0


FINRL_COLOR_ORDER = {
    "a2c": "tab:blue",
    "ddpg": "tab:orange",
    "ppo": "tab:green",
    "td3": "tab:red",
    "sac": "tab:purple",
    "mvo": "tab:brown",
}


IQN_SEED_COLORS = [
    "tab:cyan",
    "tab:pink",
    "tab:olive",
    "tab:gray",
    "tab:blue",
]


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, default=str)


def get_int_environment_variable(name: str, default: int) -> int:
    value = get_environment_variable(name, default=str(default))
    return int(value or default)


def get_float_environment_variable(name: str, default: float) -> float:
    value = get_environment_variable(name, default=str(default))
    return float(value or default)


def get_bool_environment_variable(name: str, default: bool) -> bool:
    value = get_environment_variable(name, default=str(default).lower())
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def find_latest_comparison_summary_run() -> Path:
    runs_root = PROJECT_ROOT / "outputs" / "runs"

    if not runs_root.exists():
        raise FileNotFoundError(f"Run directory does not exist: {runs_root}")

    candidates = [
        path
        for path in runs_root.iterdir()
        if path.is_dir()
        and path.name.endswith("d_iqn_dss_iqn_vs_baseline_comparison_summary")
        and (path / "summary" / COMPARISON_SUMMARY_FILENAME).exists()
    ]

    if not candidates:
        raise FileNotFoundError(
            "Could not find an IQN vs baseline comparison summary run with "
            f"summary/{COMPARISON_SUMMARY_FILENAME}."
        )

    return sorted(candidates, key=lambda path: path.name)[-1]


def resolve_source_comparison_run_directory() -> Path:
    source_run_id = get_environment_variable(
        "STOCK_INVESTMENT_DSS_COMPARISON_PLOT_SOURCE_RUN_ID",
        default=None,
    )
    source_run_directory = get_environment_variable(
        "STOCK_INVESTMENT_DSS_COMPARISON_PLOT_SOURCE_RUN_DIRECTORY",
        default=None,
    )

    if source_run_directory:
        path = Path(source_run_directory)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path

    if source_run_id:
        return PROJECT_ROOT / "outputs" / "runs" / source_run_id

    return find_latest_comparison_summary_run()


def safe_float(value: Any, default: float = math.nan) -> float:
    try:
        if value is None or value == "":
            return default
        if isinstance(value, float) and math.isnan(value):
            return default
        return float(value)
    except Exception:
        return default


def normalize_label(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def normalize_slug(value: Any) -> str:
    return (
        normalize_label(value)
        .lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("-", "_")
        .replace("__", "_")
    )


def make_curve_label(row: pd.Series) -> str:
    """
    Make labels unique enough for plotting.

    Earlier versions collapsed every IQN risk-aware seed row into the same label
    (for example D-IQN-DSS(q50_minus_cvar_penalty)), so only one seed curve was
    selected. For multiseed rows, the strategy name is already the clearest label.
    """
    strategy = normalize_label(row.get("strategy"))
    source = normalize_label(row.get("source"))
    variant = normalize_label(row.get("variant"))
    score_mode = normalize_label(row.get("score_mode"))

    if source.startswith("D-IQN-DSS") and "multiseed" in variant:
        return strategy or f"D-IQN-DSS IQN ({score_mode})"

    if source.startswith("D-IQN-DSS") and score_mode:
        return f"D-IQN-DSS ({score_mode})"

    return strategy or source or "unknown"


def read_csv_safely(path: Path) -> pd.DataFrame | None:
    try:
        if not path.exists() or path.stat().st_size == 0:
            return None
        return pd.read_csv(path)
    except Exception:
        return None


def find_value_column(
    df: pd.DataFrame, preferred_label: str | None = None
) -> str | None:
    candidates = [
        "account_value",
        "portfolio_value",
        "asset_value",
        "total_asset",
        "end_value",
        "value",
        "Mean Var",
        "mvo",
        "dji",
    ]

    if preferred_label and preferred_label in df.columns:
        return preferred_label

    lower_to_original = {str(col).lower(): col for col in df.columns}

    for candidate in candidates:
        if candidate in df.columns:
            return candidate
        if candidate.lower() in lower_to_original:
            return lower_to_original[candidate.lower()]

    numeric_cols = []
    for col in df.columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().any():
            numeric_cols.append(col)

    if not numeric_cols:
        return None

    for col in numeric_cols:
        if str(col).lower() not in {"index", "unnamed: 0", "step", "step_index"}:
            return col

    return numeric_cols[-1]


def find_time_column(df: pd.DataFrame) -> str | None:
    for candidate in ["date", "time", "timestamp", "day", "step_index", "step"]:
        if candidate in df.columns:
            return candidate

    for col in df.columns:
        col_lower = str(col).lower()
        if col_lower.startswith("unnamed") or col_lower in {"index", ""}:
            return col

    return None


def normalize_curve_dataframe(
    df: pd.DataFrame,
    label: str,
    value_column: str | None = None,
) -> pd.DataFrame | None:
    if df is None or df.empty:
        return None

    df = df.copy()
    selected_value_column = value_column or find_value_column(df, preferred_label=label)

    if selected_value_column is None or selected_value_column not in df.columns:
        return None

    time_column = find_time_column(df)

    if time_column is not None and time_column in df.columns:
        time_values = df[time_column]
    else:
        time_values = pd.Series(range(len(df)), name="step_index")

    values = pd.to_numeric(df[selected_value_column], errors="coerce")

    curve = pd.DataFrame(
        {
            "step_index": range(len(df)),
            "time": time_values.astype(str),
            "portfolio_value": values,
            "curve_label": label,
        }
    )

    curve = curve.dropna(subset=["portfolio_value"])

    if curve.empty:
        return None

    return curve


def get_row_run_directory(row: pd.Series) -> Path:
    """Prefer the concrete source run directory for seed rows when available."""
    for column in [
        "source_run_directory",
        "source_run_dir",
        "member_run_directory",
        "run_directory",
    ]:
        value = normalize_label(row.get(column))
        if value:
            path = Path(value)
            if not path.is_absolute():
                path = PROJECT_ROOT / path
            if path.exists():
                return path

    source_run_id = normalize_label(row.get("source_run_id"))
    if source_run_id:
        path = PROJECT_ROOT / "outputs" / "runs" / source_run_id
        if path.exists():
            return path

    run_id = normalize_label(row.get("run_id"))
    if run_id:
        path = PROJECT_ROOT / "outputs" / "runs" / run_id
        if path.exists():
            return path

    return Path("")


def candidate_curve_paths(
    run_directory: Path, strategy: str, source: str, variant: str
) -> list[Path]:
    strategy_slug = normalize_slug(strategy)

    patterns: list[str] = []

    if source.startswith("D-IQN-DSS"):
        # IQN seed runs from learning-curve experiments have train/eval memories.
        # A portfolio-value comparison should use real portfolio trajectories only.
        patterns.extend(
            [
                "data/iqn_learning_curve_train_asset_memory.csv",
                "data/iqn_backtest_asset_memory.csv",
                "data/discrete_dss_asset_memory.csv",
                "data/iqn_backtest_metrics_timeseries.csv",
                "data/*iqn*asset*memory*.csv",
                "data/*asset*memory*.csv",
                "data/*iqn*metrics*timeseries*.csv",
            ]
        )

    if "FinRL" in source:
        patterns.extend(
            [
                f"data/*{strategy_slug}*asset*memory*.csv",
                f"data/*{strategy_slug}*account*value*.csv",
                f"data/*{strategy_slug}*metrics*timeseries*.csv",
                "data/finrl_baseline_asset_memory.csv",
                "data/finrl_baseline_metrics_timeseries.csv",
                "data/*asset*memory*.csv",
                "data/*account*value*.csv",
                "data/*metrics*timeseries*.csv",
            ]
        )

    if strategy_slug in {
        "mvo",
        "dji",
        "djia",
        "dow",
        "mean_var",
        "mean_var_optimization",
    }:
        patterns.extend(
            [
                "data/*mvo*.csv",
                "data/*dji*.csv",
                "data/*djia*.csv",
                "summary/backtest_result.csv",
                "data/backtest_result.csv",
            ]
        )

    patterns.extend(
        [
            "summary/backtest_result.csv",
            "data/backtest_result.csv",
            "outputs/backtest/backtest_result.csv",
            "**/backtest_result.csv",
            "**/*asset*memory*.csv",
            "**/*account*value*.csv",
            "**/*metrics*timeseries*.csv",
        ]
    )

    paths: list[Path] = []
    seen: set[Path] = set()

    for pattern in patterns:
        for path in run_directory.glob(pattern):
            if path.is_file() and path not in seen:
                paths.append(path)
                seen.add(path)

    return paths


def base_metadata(row: pd.Series) -> dict[str, Any]:
    label = make_curve_label(row)
    run_directory = get_row_run_directory(row)
    return {
        "label": label,
        "strategy": normalize_label(row.get("strategy")),
        "source": normalize_label(row.get("source")),
        "variant": normalize_label(row.get("variant")),
        "model_family": normalize_label(row.get("model_family")),
        "run_id": normalize_label(row.get("run_id")),
        "source_run_id": normalize_label(row.get("source_run_id")),
        "run_directory": str(run_directory),
        "curve_found": False,
        "curve_type": "none",
        "curve_path": None,
        "points": 0,
        "synthetic": False,
        "included_in_plot": False,
        "skip_reason": None,
        "notes": None,
    }


def extract_curve_for_row(row: pd.Series) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    strategy = normalize_label(row.get("strategy"))
    source = normalize_label(row.get("source"))
    variant = normalize_label(row.get("variant"))
    label = make_curve_label(row)
    run_directory = get_row_run_directory(row)

    metadata = base_metadata(row)

    if "multiseed_mean" in variant:
        metadata["skip_reason"] = (
            "aggregate_multiseed_mean_has_no_single_real_portfolio_trajectory"
        )
        metadata["notes"] = (
            "The multiseed mean row is an aggregate metric row. It belongs in the "
            "comparison table and learning-curve mean/std plots, but not as a "
            "single portfolio-value trajectory unless an explicit aggregated "
            "trajectory is constructed and labeled as aggregate."
        )
        return None, metadata

    if not run_directory.exists():
        metadata["skip_reason"] = "run_directory_not_found"
        metadata["notes"] = f"run_directory does not exist: {run_directory}"
        return None, metadata

    paths = candidate_curve_paths(
        run_directory,
        strategy=strategy,
        source=source,
        variant=variant,
    )

    for path in paths:
        df = read_csv_safely(path)
        if df is None or df.empty:
            continue

        # FinRL tutorial-style backtest_result.csv may contain many strategy columns.
        if path.name == "backtest_result.csv" and strategy in df.columns:
            curve = normalize_curve_dataframe(df, label=label, value_column=strategy)
        else:
            curve = normalize_curve_dataframe(df, label=label)

        if curve is not None and not curve.empty:
            curve["strategy"] = strategy
            curve["source"] = source
            curve["variant"] = variant
            curve["model_family"] = row.get("model_family")
            curve["score_mode"] = row.get("score_mode")
            curve["risk_lambda"] = row.get("risk_lambda")
            curve["seed"] = row.get("seed")
            curve["source_run_id"] = row.get("source_run_id") or row.get("run_id")
            curve["source_run_directory"] = str(run_directory)
            curve["source_curve_path"] = str(path)
            curve["is_synthetic"] = False

            metadata["curve_found"] = True
            metadata["curve_type"] = "real_portfolio_trajectory"
            metadata["curve_path"] = str(path)
            metadata["points"] = int(len(curve))
            metadata["synthetic"] = False
            metadata["included_in_plot"] = True
            metadata["start_value"] = float(curve["portfolio_value"].iloc[0])
            metadata["end_value"] = float(curve["portfolio_value"].iloc[-1])

            return curve, metadata

    metadata["skip_reason"] = "no_compatible_curve_file_found"
    metadata["notes"] = (
        "No compatible asset_memory/account_value/metrics time-series CSV was found. "
        "This row can still be valid in the metrics comparison table."
    )
    return None, metadata


def add_synthetic_curve_from_summary(
    row: pd.Series,
) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    """
    Fallback only.

    If a row has summary metrics but no saved time series, make a two-point line.
    This is explicitly marked synthetic and should not be interpreted as a real
    backtest trajectory.
    """
    start_value = safe_float(row.get("start_value"), default=DEFAULT_INITIAL_AMOUNT)
    end_value = safe_float(row.get("end_value"), default=math.nan)
    label = make_curve_label(row)

    metadata = base_metadata(row)

    if math.isnan(start_value) or math.isnan(end_value):
        metadata["skip_reason"] = "missing_start_or_end_value_for_synthetic_curve"
        return None, metadata

    curve = pd.DataFrame(
        {
            "step_index": [0, 1],
            "time": ["start", "end"],
            "portfolio_value": [start_value, end_value],
            "curve_label": [label, label],
            "strategy": [row.get("strategy"), row.get("strategy")],
            "source": [row.get("source"), row.get("source")],
            "variant": [row.get("variant"), row.get("variant")],
            "model_family": [row.get("model_family"), row.get("model_family")],
            "score_mode": [row.get("score_mode"), row.get("score_mode")],
            "risk_lambda": [row.get("risk_lambda"), row.get("risk_lambda")],
            "seed": [row.get("seed"), row.get("seed")],
            "source_run_id": [row.get("source_run_id") or row.get("run_id")] * 2,
            "source_run_directory": [row.get("run_directory")] * 2,
            "source_curve_path": ["synthetic_from_summary"] * 2,
            "is_synthetic": [True, True],
        }
    )

    metadata["curve_found"] = True
    metadata["curve_type"] = "synthetic_two_point_summary_curve"
    metadata["synthetic"] = True
    metadata["included_in_plot"] = True
    metadata["skip_reason"] = None
    metadata["notes"] = "synthetic two-point curve from comparison summary"
    metadata["points"] = int(len(curve))
    metadata["start_value"] = start_value
    metadata["end_value"] = end_value

    return curve, metadata


def select_comparison_rows(comparison_table: pd.DataFrame, top_n: int) -> pd.DataFrame:
    if comparison_table.empty:
        return comparison_table

    table = comparison_table.copy()

    if "rank" in table.columns:
        table["rank_numeric"] = pd.to_numeric(table["rank"], errors="coerce")
        table = table.sort_values("rank_numeric", na_position="last")

    # Do not collapse IQN seed rows. They are distinct real trajectories.
    # Remove only exact duplicate rows for safety.
    dedup_cols = [
        col
        for col in ["strategy", "source", "variant", "run_id", "source_run_id"]
        if col in table.columns
    ]
    if dedup_cols:
        table = table.drop_duplicates(subset=dedup_cols, keep="first")

    return table.head(top_n).reset_index(drop=True)


def build_wide_curve_table(long_curves: pd.DataFrame) -> pd.DataFrame:
    if long_curves.empty:
        return long_curves

    wide = long_curves.pivot_table(
        index="step_index",
        columns="curve_label",
        values="portfolio_value",
        aggfunc="first",
    ).reset_index()

    wide.columns.name = None
    return wide


def build_cumulative_return_curves(long_curves: pd.DataFrame) -> pd.DataFrame:
    if long_curves.empty:
        return long_curves

    frames: list[pd.DataFrame] = []
    for label, group in long_curves.groupby("curve_label"):
        group = group.sort_values("step_index").copy()
        start_value = safe_float(group["portfolio_value"].iloc[0])
        if math.isnan(start_value) or abs(start_value) < 1e-12:
            continue
        group["cumulative_return_pct"] = (
            group["portfolio_value"] / start_value - 1.0
        ) * 100.0
        frames.append(group)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def plot_portfolio_curves(
    long_curves: pd.DataFrame,
    output_path: Path,
    initial_amount: float,
    title: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(16, 7))

    for i, (label, group) in enumerate(long_curves.groupby("curve_label")):
        group = group.sort_values("step_index")
        strategy = (
            normalize_slug(group["strategy"].iloc[0]) if "strategy" in group else ""
        )
        source = normalize_label(group["source"].iloc[0]) if "source" in group else ""
        is_iqn = source.startswith("D-IQN-DSS")
        is_synthetic = bool(group.get("is_synthetic", pd.Series([False])).iloc[0])

        color = None
        linewidth = 1.8
        linestyle = "-"
        alpha = 0.95

        if strategy in FINRL_COLOR_ORDER:
            color = FINRL_COLOR_ORDER[strategy]
        elif is_iqn:
            color = IQN_SEED_COLORS[i % len(IQN_SEED_COLORS)]
            linewidth = 1.35
            alpha = 0.75
        if is_synthetic:
            linestyle = ":"
            linewidth = 1.2
            alpha = 0.6

        plt.plot(
            group["step_index"],
            group["portfolio_value"],
            label=label,
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
            alpha=alpha,
        )

    plt.axhline(initial_amount, linestyle="--", linewidth=1, color="black", alpha=0.5)
    plt.title(title)
    plt.xlabel("Backtest step")
    plt.ylabel("Portfolio value")
    plt.grid(axis="y", alpha=0.25)
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_cumulative_return_curves(
    return_curves: pd.DataFrame,
    output_path: Path,
    title: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(16, 7))

    for i, (label, group) in enumerate(return_curves.groupby("curve_label")):
        group = group.sort_values("step_index")
        strategy = (
            normalize_slug(group["strategy"].iloc[0]) if "strategy" in group else ""
        )
        source = normalize_label(group["source"].iloc[0]) if "source" in group else ""
        is_iqn = source.startswith("D-IQN-DSS")
        is_synthetic = bool(group.get("is_synthetic", pd.Series([False])).iloc[0])

        color = FINRL_COLOR_ORDER.get(strategy)
        linewidth = 1.8
        linestyle = "-"
        alpha = 0.95

        if is_iqn:
            color = IQN_SEED_COLORS[i % len(IQN_SEED_COLORS)]
            linewidth = 1.35
            alpha = 0.75
        if is_synthetic:
            linestyle = ":"
            linewidth = 1.2
            alpha = 0.6

        plt.plot(
            group["step_index"],
            group["cumulative_return_pct"],
            label=label,
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
            alpha=alpha,
        )

    plt.axhline(0.0, linestyle="--", linewidth=1, color="black", alpha=0.5)
    plt.title(title)
    plt.xlabel("Backtest step")
    plt.ylabel("Cumulative return (%)")
    plt.grid(axis="y", alpha=0.25)
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def main() -> int:
    log_level = (
        get_environment_variable(
            "STOCK_INVESTMENT_DSS_LOG_LEVEL",
            default="INFO",
        )
        or "INFO"
    )

    system_logger = setup_system_logger(log_level=log_level)
    system_logger.info("Starting StockInvestmentDSS IQN vs baseline comparison plot.")
    system_logger.info("Project root: %s", PROJECT_ROOT)

    run_paths = None

    try:
        source_comparison_run_directory = resolve_source_comparison_run_directory()
        comparison_path = (
            source_comparison_run_directory / "summary" / COMPARISON_SUMMARY_FILENAME
        )

        if not comparison_path.exists():
            raise FileNotFoundError(f"Missing comparison summary: {comparison_path}")

        top_n = get_int_environment_variable(
            "STOCK_INVESTMENT_DSS_COMPARISON_PLOT_TOP_N",
            default=12,
        )
        initial_amount = get_float_environment_variable(
            "STOCK_INVESTMENT_DSS_COMPARISON_PLOT_INITIAL_AMOUNT",
            default=DEFAULT_INITIAL_AMOUNT,
        )
        allow_synthetic = get_bool_environment_variable(
            "STOCK_INVESTMENT_DSS_COMPARISON_PLOT_ALLOW_SYNTHETIC",
            default=False,
        )

        run_paths = create_run_paths("d_iqn_dss_iqn_vs_baseline_comparison_plot")
        run_logger = setup_run_logger(run_paths, log_level=log_level)

        run_logger.info("Created run directory: %s", run_paths.run_directory)
        run_logger.info("Run id: %s", run_paths.run_id)
        run_logger.info("Source comparison run: %s", source_comparison_run_directory)
        run_logger.info("Comparison summary: %s", comparison_path)
        run_logger.info("Top N rows: %s", top_n)
        run_logger.info("Allow synthetic fallback curves: %s", allow_synthetic)

        comparison_table = pd.read_csv(comparison_path)
        selected_rows = select_comparison_rows(comparison_table, top_n=top_n)

        curve_frames: list[pd.DataFrame] = []
        source_records: list[dict[str, Any]] = []

        for _, row in selected_rows.iterrows():
            curve, metadata = extract_curve_for_row(row)

            if curve is None and allow_synthetic:
                curve, synthetic_metadata = add_synthetic_curve_from_summary(row)
                metadata = {**metadata, **synthetic_metadata}

            if curve is not None:
                curve_frames.append(curve)

            source_records.append(metadata)

        if not curve_frames:
            raise ValueError(
                "No portfolio value curves could be extracted from the selected "
                "comparison rows. Try enabling synthetic fallback or check that "
                "asset_memory/account_value files exist in source runs."
            )

        long_curves = pd.concat(curve_frames, ignore_index=True)
        wide_curves = build_wide_curve_table(long_curves)
        return_curves = build_cumulative_return_curves(long_curves)
        return_wide_curves = (
            build_wide_curve_table(
                return_curves.rename(
                    columns={"cumulative_return_pct": "portfolio_value"}
                )
            )
            if not return_curves.empty
            else pd.DataFrame()
        )
        curve_sources = pd.DataFrame(source_records)

        included_rows = curve_sources[curve_sources["included_in_plot"] == True].copy()
        skipped_rows = curve_sources[curve_sources["included_in_plot"] != True].copy()

        long_curves_path = (
            run_paths.data_directory / "iqn_vs_baseline_portfolio_value_curves_long.csv"
        )
        wide_curves_path = (
            run_paths.data_directory / "iqn_vs_baseline_portfolio_value_curves_wide.csv"
        )
        return_curves_path = (
            run_paths.data_directory
            / "iqn_vs_baseline_cumulative_return_curves_long.csv"
        )
        return_wide_curves_path = (
            run_paths.data_directory
            / "iqn_vs_baseline_cumulative_return_curves_wide.csv"
        )
        curve_sources_path = (
            run_paths.data_directory / "iqn_vs_baseline_curve_sources.csv"
        )
        selected_rows_path = (
            run_paths.data_directory / "iqn_vs_baseline_plot_selected_rows.csv"
        )
        included_rows_path = (
            run_paths.data_directory / "iqn_vs_baseline_plot_included_rows.csv"
        )
        skipped_rows_path = (
            run_paths.data_directory / "iqn_vs_baseline_plot_skipped_rows.csv"
        )
        plot_path = (
            run_paths.summary_directory / "iqn_vs_baseline_portfolio_value_curve.png"
        )
        return_plot_path = (
            run_paths.summary_directory / "iqn_vs_baseline_cumulative_return_curve.png"
        )
        summary_path = (
            run_paths.summary_directory / "iqn_vs_baseline_comparison_plot_summary.json"
        )

        long_curves.to_csv(long_curves_path, index=False)
        wide_curves.to_csv(wide_curves_path, index=False)
        return_curves.to_csv(return_curves_path, index=False)
        return_wide_curves.to_csv(return_wide_curves_path, index=False)
        curve_sources.to_csv(curve_sources_path, index=False)
        selected_rows.to_csv(selected_rows_path, index=False)
        included_rows.to_csv(included_rows_path, index=False)
        skipped_rows.to_csv(skipped_rows_path, index=False)

        plot_portfolio_curves(
            long_curves=long_curves,
            output_path=plot_path,
            initial_amount=initial_amount,
            title="Portfolio Value Over Time: IQN vs FinRL Baselines",
        )
        if not return_curves.empty:
            plot_cumulative_return_curves(
                return_curves=return_curves,
                output_path=return_plot_path,
                title="Cumulative Return Over Time: IQN vs FinRL Baselines",
            )

        found_count = (
            int(curve_sources["curve_found"].sum()) if not curve_sources.empty else 0
        )
        included_count = (
            int(curve_sources["included_in_plot"].sum())
            if not curve_sources.empty
            else 0
        )
        skipped_count = int(len(curve_sources) - included_count)
        synthetic_count = int(
            curve_sources.get("synthetic", pd.Series(dtype=bool)).fillna(False).sum()
        )

        skipped_reason_counts = (
            skipped_rows["skip_reason"].fillna("unknown").value_counts().to_dict()
            if not skipped_rows.empty and "skip_reason" in skipped_rows.columns
            else {}
        )

        summary = {
            "status": "ok",
            "project_name": "StockInvestmentDSS",
            "prototype_name": "D-IQN-DSS",
            "run_id": run_paths.run_id,
            "project_root": str(PROJECT_ROOT),
            "run_directory": str(run_paths.run_directory),
            "source_comparison_run_directory": str(source_comparison_run_directory),
            "source_comparison_summary_path": str(comparison_path),
            "selected_rows": int(len(selected_rows)),
            "curves_found": found_count,
            "included_curves": included_count,
            "skipped_rows": skipped_count,
            "skipped_reason_counts": skipped_reason_counts,
            "synthetic_curves": synthetic_count,
            "allow_synthetic_fallback": allow_synthetic,
            "initial_amount_reference_line": initial_amount,
            "outputs": {
                "long_curves_path": str(long_curves_path),
                "wide_curves_path": str(wide_curves_path),
                "return_curves_path": str(return_curves_path),
                "return_wide_curves_path": str(return_wide_curves_path),
                "curve_sources_path": str(curve_sources_path),
                "selected_rows_path": str(selected_rows_path),
                "included_rows_path": str(included_rows_path),
                "skipped_rows_path": str(skipped_rows_path),
                "plot_path": str(plot_path),
                "return_plot_path": str(return_plot_path),
                "summary_path": str(summary_path),
            },
            "interpretation": (
                "This runner creates the V2 equivalent of FinRL's backtest_result.png "
                "by plotting real portfolio value trajectories for FinRL/SB3 baselines, "
                "MVO, and D-IQN-DSS seed runs when compatible time-series outputs are "
                "available. IQN multiseed mean rows are aggregate metric rows and are "
                "reported as skipped for portfolio-value trajectory plotting unless a "
                "separate aggregate trajectory is explicitly constructed."
            ),
            "next_step": (
                "Use this plot together with the comparison summary metrics table and "
                "the IQN multiseed learning-curve mean/std plots."
            ),
        }

        write_json(summary_path, summary)

        run_logger.info("IQN vs baseline comparison plot completed.")
        run_logger.info("Selected rows: %s", len(selected_rows))
        run_logger.info("Curves found: %s", found_count)
        run_logger.info("Included curves: %s", included_count)
        run_logger.info("Skipped rows: %s", skipped_count)
        run_logger.info("Skipped reason counts: %s", skipped_reason_counts)
        run_logger.info("Synthetic curves: %s", synthetic_count)
        run_logger.info("Wrote long curves: %s", long_curves_path)
        run_logger.info("Wrote wide curves: %s", wide_curves_path)
        run_logger.info("Wrote curve sources: %s", curve_sources_path)
        run_logger.info("Wrote plot: %s", plot_path)
        run_logger.info("Wrote cumulative return plot: %s", return_plot_path)
        run_logger.info("Wrote summary: %s", summary_path)

        system_logger.info(
            "StockInvestmentDSS IQN vs baseline comparison plot completed successfully."
        )

        return 0

    except Exception:
        system_logger.exception(
            "StockInvestmentDSS IQN vs baseline comparison plot failed."
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
