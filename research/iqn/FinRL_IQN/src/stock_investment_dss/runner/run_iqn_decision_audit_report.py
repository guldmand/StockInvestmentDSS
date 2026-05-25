"""
Build a ticker-level decision audit report for D-IQN-DSS IQN runs.

Purpose
-------
This runner is intentionally read-only. It scans an existing IQN run directory,
finds decision/evaluation records, and writes a transparent audit package that
answers questions such as:

- Which tickers were available?
- Which ticker was selected per decision?
- Which action was chosen and which action was effectively executed?
- Did the run trade AAPL, MSFT, both, or none?
- Were actions resolved/masked into HOLD/no-op?
- How did cash, trades, costs, and portfolio value change?

The runner supports optional W&B logging through the existing project helper if
available. W&B must never be required for local reproducibility.

Environment variables
---------------------
STOCK_INVESTMENT_DSS_AUDIT_SOURCE_RUN_DIR
    Explicit source run directory. If omitted, the newest run containing
    "iqn_learning_curve_smoke_test" or "iqn_backtest_smoke_test" is used.

STOCK_INVESTMENT_DSS_AUDIT_SOURCE_RUN_ID
    Source run id under outputs/runs. Used only if SOURCE_RUN_DIR is not set.

STOCK_INVESTMENT_DSS_AUDIT_RECENT_RUN_LIMIT
    Number of recent runs to scan when auto-detecting a source run. Default: 80.

STOCK_INVESTMENT_DSS_WANDB_ENABLED
    If true, attempts to log summary metrics, CSV/JSON/MD/PNG artifacts to W&B.
"""

from __future__ import annotations

import ast
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_NAME = "StockInvestmentDSS"
PROTOTYPE_NAME = "D-IQN-DSS"
RUNNER_NAME = "run_iqn_decision_audit_report"


DECISION_HINT_COLUMNS = {
    "step",
    "date",
    "decision_date",
    "train_step",
    "checkpoint_step",
    "chosen_action",
    "chosen_action_label",
    "effective_action",
    "action",
    "action_label",
    "selected_ticker",
    "ticker",
    "cash_before",
    "cash_after",
    "portfolio_value_before",
    "portfolio_value_after",
    "portfolio_value",
    "cost_delta",
    "trades_delta",
    "cash_delta",
    "portfolio_value_delta",
}

ACTION_COLUMNS = [
    "effective_action",
    "chosen_action_label",
    "action_label",
    "chosen_action",
    "action",
]

TICKER_COLUMNS = [
    "selected_ticker",
    "ticker",
    "asset",
    "symbol",
]

STEP_COLUMNS = [
    "step",
    "decision_step",
    "backtest_step",
    "eval_step",
    "timestep",
    "time_step",
]

DATE_COLUMNS = [
    "date",
    "decision_date",
    "trade_date",
]


@dataclass
class AuditContext:
    project_name: str
    prototype_name: str
    runner: str
    run_id: str
    project_root: str
    run_directory: str
    source_run_id: str
    source_run_directory: str
    generated_at: str


def _project_root() -> Path:
    return Path.cwd().resolve()


def _now_run_id() -> str:
    return (
        datetime.now().strftime("%Y_%m_%d_%H%M%S")
        + "_d_iqn_dss_iqn_decision_audit_report"
    )


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _print(msg: str) -> None:
    print(msg, file=sys.stderr)


def _load_source_experiment_context(source_run_dir: Path) -> dict[str, Any] | None:
    """Load source experiment context summary if the IQN runner produced one."""

    candidates = [
        source_run_dir / "summary" / "experiment_context_summary.json",
        source_run_dir / "summary" / "iqn_learning_curve_summary.json",
    ]

    for path in candidates:
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if path.name == "iqn_learning_curve_summary.json":
                return data.get("experiment_context") or data
            return data
        except Exception:
            continue

    return None


def _append_source_experiment_context_to_markdown(
    path: Path,
    source_context: dict[str, Any] | None,
) -> None:
    if not source_context:
        return

    tickers = ", ".join(source_context.get("tickers") or [])
    data = source_context.get("data_provenance") or {}
    made_money = "Yes" if source_context.get("made_money") else "No"

    appendix = f"""

## Source experiment window and result card

```text
Dataset ID:        {source_context.get('dataset_id')}
Universe:          {source_context.get('universe_id')}
Tickers:           {tickers}

Market data start: {source_context.get('market_data_start')}
Market data end:   {source_context.get('market_data_end')}

PIT cutoff:        {source_context.get('pit_cutoff')}
Train window:      {source_context.get('train_window_start')} -> {source_context.get('train_window_end')}
Eval/trade window: {source_context.get('eval_window_start')} -> {source_context.get('eval_window_end')}

Initial capital:   {source_context.get('initial_capital')}
Final value:       {source_context.get('final_value')}
Profit/Loss:       {source_context.get('profit_loss')}
Total return:      {source_context.get('total_return_pct')}
Made money:        {made_money}
```

## Source data provenance

```text
Final source used:  {data.get('final_source_used')}
Actual data method: {data.get('actual_data_method')}
Download attempted: {data.get('download_attempted')}
Download success:   {data.get('download_success')}
Cache used:         {data.get('cache_used')}
Import file used:   {data.get('import_file_used')}
Failed tickers:     {data.get('failed_tickers') or '-'}
```
"""
    with path.open("a", encoding="utf-8") as file:
        file.write(appendix)


def _create_run_dir(project_root: Path, run_id: str) -> Path:
    run_dir = project_root / "outputs" / "runs" / run_id
    for sub in ["data", "audit", "summary", "plots", "logs"]:
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    return run_dir


def _find_source_run(project_root: Path) -> Path:
    explicit_dir = os.getenv("STOCK_INVESTMENT_DSS_AUDIT_SOURCE_RUN_DIR", "").strip()
    if explicit_dir:
        path = Path(explicit_dir).expanduser()
        if not path.is_absolute():
            path = project_root / path
        if not path.exists():
            raise FileNotFoundError(
                f"Explicit audit source run dir does not exist: {path}"
            )
        return path.resolve()

    explicit_id = os.getenv("STOCK_INVESTMENT_DSS_AUDIT_SOURCE_RUN_ID", "").strip()
    if explicit_id:
        path = project_root / "outputs" / "runs" / explicit_id
        if not path.exists():
            raise FileNotFoundError(
                f"Explicit audit source run id does not exist: {path}"
            )
        return path.resolve()

    runs_root = project_root / "outputs" / "runs"
    if not runs_root.exists():
        raise FileNotFoundError(f"No outputs/runs directory found at: {runs_root}")

    recent_limit = _env_int("STOCK_INVESTMENT_DSS_AUDIT_RECENT_RUN_LIMIT", 80)
    candidates = sorted(
        [p for p in runs_root.iterdir() if p.is_dir()],
        key=lambda p: p.name,
        reverse=True,
    )[:recent_limit]

    preferred_patterns = [
        "iqn_learning_curve_smoke_test",
        "iqn_backtest_smoke_test",
    ]
    for pattern in preferred_patterns:
        for candidate in candidates:
            if pattern in candidate.name:
                return candidate.resolve()

    raise FileNotFoundError(
        "Could not auto-detect an IQN source run. Set "
        "STOCK_INVESTMENT_DSS_AUDIT_SOURCE_RUN_DIR or STOCK_INVESTMENT_DSS_AUDIT_SOURCE_RUN_ID."
    )


def _safe_read_csv(path: Path) -> pd.DataFrame | None:
    try:
        if path.stat().st_size == 0:
            return None
        df = pd.read_csv(path)
        if df.empty:
            return None
        return df
    except Exception:
        return None


def _score_decision_frame(df: pd.DataFrame) -> int:
    cols = set(map(str, df.columns))
    score = len(cols.intersection(DECISION_HINT_COLUMNS))
    if any(col in cols for col in ACTION_COLUMNS):
        score += 5
    if any(col in cols for col in TICKER_COLUMNS):
        score += 5
    if any(
        col in cols
        for col in [
            "portfolio_value_before",
            "portfolio_value_after",
            "cash_before",
            "cash_after",
        ]
    ):
        score += 3
    if any(col in cols for col in ["trades_delta", "cost_delta"]):
        score += 3
    return score


def _find_candidate_csvs(source_run_dir: Path) -> list[tuple[int, Path, pd.DataFrame]]:
    candidates: list[tuple[int, Path, pd.DataFrame]] = []
    for path in source_run_dir.rglob("*.csv"):
        df = _safe_read_csv(path)
        if df is None:
            continue
        score = _score_decision_frame(df)
        lower_name = path.name.lower()
        if lower_name == "iqn_learning_curve_eval_step_records.csv":
            score += 100
        elif "eval_step" in lower_name and "record" in lower_name:
            score += 50
        elif "training_records" in lower_name:
            score -= 5
        if score > 0:
            candidates.append((score, path, df))
    candidates.sort(key=lambda item: (item[0], len(item[2])), reverse=True)
    return candidates


def _first_existing_column(df: pd.DataFrame, names: Iterable[str]) -> str | None:
    for name in names:
        if name in df.columns:
            return name
    return None


def _clean_scalar(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, (int, float, str, bool)):
        return value
    return str(value)


def _parse_object(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none", "null"}:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    try:
        return ast.literal_eval(text)
    except Exception:
        return text


def _to_number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        return float(value)
    except Exception:
        return None


def _derive_universe_from_files(
    source_run_dir: Path, decision_df: pd.DataFrame | None
) -> list[str]:
    tickers: set[str] = set()

    if decision_df is not None:
        ticker_col = _first_existing_column(decision_df, TICKER_COLUMNS)
        if ticker_col:
            for raw in decision_df[ticker_col].dropna().astype(str):
                token = raw.strip()
                if token and token.lower() not in {"none", "nan", "cash"}:
                    tickers.add(token.upper())

        for col in decision_df.columns:
            lower = str(col).lower()
            if "ticker" in lower or "tickers" in lower or "universe" in lower:
                for raw in decision_df[col].dropna().head(20):
                    parsed = _parse_object(raw)
                    if isinstance(parsed, list):
                        for item in parsed:
                            token = str(item).strip()
                            if token:
                                tickers.add(token.upper())
                    elif isinstance(parsed, str):
                        # Catch strings like "AAPL,MSFT" or "['AAPL', 'MSFT']".
                        for token in re.split(
                            r"[,;\s]+",
                            parsed.replace("[", " ")
                            .replace("]", " ")
                            .replace("'", " ")
                            .replace('"', " "),
                        ):
                            token = token.strip()
                            if re.fullmatch(r"[A-Za-z.\-]{1,8}", token):
                                tickers.add(token.upper())

    # Lightweight scan of json metadata files.
    for path in list(source_run_dir.rglob("*.json"))[:200]:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")[:200_000]
        except Exception:
            continue
        for key in ["tickers", "ticker_list", "dataset_tickers", "universe"]:
            if key in text:
                try:
                    obj = json.loads(text)
                except Exception:
                    obj = None
                if isinstance(obj, dict):
                    stack = [obj]
                    while stack:
                        item = stack.pop()
                        if isinstance(item, dict):
                            for k, v in item.items():
                                if str(k).lower() in {
                                    "tickers",
                                    "ticker_list",
                                    "dataset_tickers",
                                    "universe",
                                } and isinstance(v, list):
                                    for ticker in v:
                                        token = str(ticker).strip()
                                        if token:
                                            tickers.add(token.upper())
                                elif isinstance(v, (dict, list)):
                                    stack.append(v)
                        elif isinstance(item, list):
                            stack.extend(item)

    return sorted(tickers)


def _normalize_decision_df(df: pd.DataFrame, source_path: Path) -> pd.DataFrame:
    out = pd.DataFrame()
    out["source_file"] = [str(source_path)] * len(df)

    step_col = _first_existing_column(df, STEP_COLUMNS)
    date_col = _first_existing_column(df, DATE_COLUMNS)
    action_col = _first_existing_column(df, ACTION_COLUMNS)
    ticker_col = _first_existing_column(df, TICKER_COLUMNS)

    out["step"] = df[step_col] if step_col else range(len(df))
    out["date"] = df[date_col] if date_col else None
    out["chosen_action"] = df[action_col] if action_col else None
    out["effective_action"] = (
        df["effective_action"]
        if "effective_action" in df.columns
        else out["chosen_action"]
    )
    out["selected_ticker"] = df[ticker_col] if ticker_col else None

    # Preserve useful financial deltas when present.
    for col in [
        "cash_before",
        "cash_after",
        "cash_delta",
        "portfolio_value_before",
        "portfolio_value_after",
        "portfolio_value_delta",
        "portfolio_value",
        "final_value",
        "cost_delta",
        "trades_delta",
        "shares_before",
        "shares_after",
        "holdings_before",
        "holdings_after",
        "position_before",
        "position_after",
        "action_masked",
        "mask_reason",
        "resolver_reason",
        "score_mode",
        "q10",
        "q25",
        "q50",
        "q75",
        "q90",
        "cvar10",
        "cvar_pct",
        "risk_adjusted_score",
        "train_step",
        "checkpoint_step",
    ]:
        if col in df.columns:
            out[col] = df[col]

    # Add derived flags.
    action_series = (
        out["effective_action"]
        .fillna(out["chosen_action"])
        .fillna("")
        .astype(str)
        .str.upper()
    )
    out["is_buy"] = action_series.str.contains("BUY", regex=False)
    out["is_sell"] = action_series.str.contains("SELL", regex=False)
    out["is_rebalance"] = action_series.str.contains("REBALANCE", regex=False)
    out["is_hold"] = action_series.str.contains("HOLD", regex=False) | (
        action_series.str.strip() == ""
    )

    if "trades_delta" in out.columns:
        out["trades_delta_num"] = out["trades_delta"].map(_to_number)
        out["executed_trade"] = out["trades_delta_num"].fillna(0) > 0
    else:
        # Best-effort fallback: BUY/SELL/REBALANCE are intended trades, but might be masked.
        out["executed_trade"] = out["is_buy"] | out["is_sell"] | out["is_rebalance"]

    if "cost_delta" in out.columns:
        out["cost_delta_num"] = out["cost_delta"].map(_to_number)
    else:
        out["cost_delta_num"] = 0.0

    return out


def _choose_best_decision_frame(
    candidates: list[tuple[int, Path, pd.DataFrame]],
) -> tuple[Path | None, pd.DataFrame | None, list[dict[str, Any]]]:
    manifest: list[dict[str, Any]] = []
    for score, path, df in candidates:
        manifest.append(
            {
                "score": score,
                "path": str(path),
                "rows": int(len(df)),
                "columns": list(map(str, df.columns)),
            }
        )

    # Prefer explicit IQN evaluation step ledger if available.
    for score, path, df in candidates:
        if path.name.lower() == "iqn_learning_curve_eval_step_records.csv":
            return path, df, manifest

    # Prefer files with selected_ticker/effective_action and many rows.
    for score, path, df in candidates:
        cols = set(df.columns)
        if any(c in cols for c in TICKER_COLUMNS) and any(
            c in cols for c in ACTION_COLUMNS
        ):
            return path, df, manifest

    if candidates:
        score, path, df = candidates[0]
        return path, df, manifest

    return None, None, manifest


def _summarize_actions(decision_df: pd.DataFrame) -> pd.DataFrame:
    if decision_df.empty:
        return pd.DataFrame()
    grouped = (
        decision_df.groupby(["effective_action"], dropna=False)
        .agg(
            row_count=("effective_action", "size"),
            executed_trade_count=("executed_trade", "sum"),
            total_cost=("cost_delta_num", "sum"),
        )
        .reset_index()
        .sort_values(["row_count", "effective_action"], ascending=[False, True])
    )
    return grouped


def _summarize_ticker_actions(decision_df: pd.DataFrame) -> pd.DataFrame:
    if decision_df.empty:
        return pd.DataFrame()
    df = decision_df.copy()
    df["selected_ticker"] = df["selected_ticker"].fillna("UNKNOWN").astype(str)
    df["effective_action"] = df["effective_action"].fillna("UNKNOWN").astype(str)
    grouped = (
        df.groupby(["selected_ticker", "effective_action"], dropna=False)
        .agg(
            row_count=("effective_action", "size"),
            executed_trade_count=("executed_trade", "sum"),
            total_cost=("cost_delta_num", "sum"),
        )
        .reset_index()
        .sort_values(["selected_ticker", "row_count"], ascending=[True, False])
    )
    return grouped


def _summarize_holdings_like_columns(decision_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if decision_df.empty:
        return pd.DataFrame()

    candidate_cols = [
        c
        for c in decision_df.columns
        if any(token in str(c).lower() for token in ["holdings", "shares", "position"])
    ]
    for col in candidate_cols:
        counter = Counter()
        nonzero_counter = Counter()
        for raw in decision_df[col].dropna():
            parsed = _parse_object(raw)
            if isinstance(parsed, dict):
                for ticker, value in parsed.items():
                    num = _to_number(value)
                    counter[str(ticker).upper()] += 1
                    if num is not None and abs(num) > 1e-12:
                        nonzero_counter[str(ticker).upper()] += 1
        for ticker in sorted(counter):
            rows.append(
                {
                    "column": col,
                    "ticker": ticker,
                    "observed_rows": counter[ticker],
                    "nonzero_rows": nonzero_counter[ticker],
                }
            )
    return pd.DataFrame(rows)


def _make_ticker_action_heatmap(
    ticker_action_df: pd.DataFrame, plot_path: Path
) -> bool:
    if ticker_action_df.empty:
        return False
    pivot = ticker_action_df.pivot_table(
        index="selected_ticker",
        columns="effective_action",
        values="row_count",
        aggfunc="sum",
        fill_value=0,
    )
    if pivot.empty:
        return False
    fig, ax = plt.subplots(
        figsize=(max(8, 1.1 * len(pivot.columns)), max(4, 0.45 * len(pivot.index)))
    )
    im = ax.imshow(pivot.values, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("IQN decision audit: ticker/action counts")
    ax.set_xlabel("Effective action")
    ax.set_ylabel("Selected ticker")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, str(int(pivot.iloc[i, j])), ha="center", va="center")
    fig.colorbar(im, ax=ax, label="count")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)
    return True


def _write_markdown_report(
    path: Path,
    context: AuditContext,
    universe: list[str],
    selected_source_path: Path | None,
    decision_df: pd.DataFrame,
    action_df: pd.DataFrame,
    ticker_action_df: pd.DataFrame,
    holdings_df: pd.DataFrame,
    manifest: list[dict[str, Any]],
) -> None:
    total_rows = int(len(decision_df))
    executed = (
        int(decision_df["executed_trade"].sum())
        if not decision_df.empty and "executed_trade" in decision_df.columns
        else 0
    )
    tickers_selected = []
    if not decision_df.empty and "selected_ticker" in decision_df.columns:
        tickers_selected = sorted(
            t
            for t in decision_df["selected_ticker"]
            .dropna()
            .astype(str)
            .str.upper()
            .unique()
            .tolist()
            if t and t not in {"NONE", "NAN", "UNKNOWN", "CASH"}
        )

    lines: list[str] = []
    lines.append("# IQN Decision Audit Report")
    lines.append("")
    lines.append("## Context")
    lines.append("")
    lines.append(f"- Audit run id: `{context.run_id}`")
    lines.append(f"- Source run id: `{context.source_run_id}`")
    lines.append(f"- Source run directory: `{context.source_run_directory}`")
    lines.append(f"- Selected decision source file: `{selected_source_path}`")
    lines.append(f"- Generated at: `{context.generated_at}`")
    lines.append("")
    lines.append("## Core audit conclusion")
    lines.append("")
    lines.append(f"- Decision rows analyzed: **{total_rows}**")
    lines.append(f"- Executed trade rows: **{executed}**")
    lines.append(
        f"- Universe inferred: **{', '.join(universe) if universe else 'not inferred'}**"
    )
    lines.append(
        f"- Tickers selected by policy: **{', '.join(tickers_selected) if tickers_selected else 'none / not found'}**"
    )
    if universe and tickers_selected:
        missing = sorted(set(universe) - set(tickers_selected))
        lines.append(
            f"- Universe tickers never selected: **{', '.join(missing) if missing else 'none'}**"
        )
    lines.append("")
    lines.append("## Action counts")
    lines.append("")
    if action_df.empty:
        lines.append("No action summary available.")
    else:
        lines.append(action_df.to_markdown(index=False))
    lines.append("")
    lines.append("## Ticker/action counts")
    lines.append("")
    if ticker_action_df.empty:
        lines.append(
            "No ticker/action summary available. This usually means selected_ticker is missing from the source records."
        )
    else:
        lines.append(ticker_action_df.to_markdown(index=False))
    lines.append("")
    lines.append("## Holdings-like columns")
    lines.append("")
    if holdings_df.empty:
        lines.append("No holdings/share/position dictionary columns could be parsed.")
    else:
        lines.append(holdings_df.to_markdown(index=False))
    lines.append("")
    lines.append("## Bitcoin-inspired audit interpretation")
    lines.append("")
    lines.append(
        "This audit is the StockDSS equivalent of a transparent ledger: it does not claim decentralization, "
        "but it makes each model recommendation traceable to the visible point-in-time data, model/config state, "
        "selected action, selected ticker, and portfolio effect."
    )
    lines.append("")
    lines.append("## Candidate files scanned")
    lines.append("")
    for item in manifest[:20]:
        lines.append(
            f"- score={item['score']} rows={item['rows']} path=`{item['path']}`"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def _wandb_log_outputs(run_dir: Path, summary: dict[str, Any]) -> None:
    if not _env_bool("STOCK_INVESTMENT_DSS_WANDB_ENABLED", False):
        return
    try:
        from stock_investment_dss.experiment_tracking.wandb_tracking import (
            init_wandb_run,
        )
    except Exception as exc:
        _print(f"W&B not available for audit logging: {exc}")
        return

    try:
        run = init_wandb_run(
            run_name=str(
                summary.get("run_id") or summary.get("audit_run_id") or RUNNER_NAME
            ),
            job_type="iqn_decision_audit_report",
            tags=["iqn", "audit", "decision-ledger", "stockdss"],
            config={"audit_summary": summary},
        )
        if run is None:
            return
        import wandb

        # Scalar-friendly summary.
        for key, value in summary.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                wandb.log({f"audit/{key}": value})

        # Artifact with all audit outputs.
        artifact = wandb.Artifact(
            name=f"iqn_decision_audit_{summary.get('source_run_id', 'unknown')}",
            type="decision-audit",
            description="Ticker-level IQN decision audit report and transparent ledger files.",
        )
        for folder in ["audit", "summary", "plots", "data"]:
            folder_path = run_dir / folder
            if folder_path.exists():
                artifact.add_dir(str(folder_path), name=folder)
        run.log_artifact(artifact)

        # Log heatmap as image if present.
        heatmap_path = run_dir / "plots" / "decision_audit_ticker_action_heatmap.png"
        if heatmap_path.exists():
            wandb.log({"audit_ticker_action_heatmap": wandb.Image(str(heatmap_path))})

        run.finish()
    except Exception as exc:
        _print(f"W&B audit logging failed but local audit completed: {exc}")


def main() -> None:
    project_root = _project_root()
    run_id = _now_run_id()
    run_dir = _create_run_dir(project_root, run_id)
    source_run_dir = _find_source_run(project_root)
    source_run_id = source_run_dir.name

    _print(f"Starting {PROJECT_NAME} IQN decision audit report.")
    _print(f"Project root: {project_root}")
    _print(f"Audit run dir: {run_dir}")
    _print(f"Source run dir: {source_run_dir}")

    context = AuditContext(
        project_name=PROJECT_NAME,
        prototype_name=PROTOTYPE_NAME,
        runner=RUNNER_NAME,
        run_id=run_id,
        project_root=str(project_root),
        run_directory=str(run_dir),
        source_run_id=source_run_id,
        source_run_directory=str(source_run_dir),
        generated_at=datetime.now().isoformat(timespec="seconds"),
    )

    source_experiment_context = _load_source_experiment_context(source_run_dir)

    candidates = _find_candidate_csvs(source_run_dir)
    selected_path, raw_df, manifest = _choose_best_decision_frame(candidates)

    manifest_path = run_dir / "data" / "decision_audit_candidate_files.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )

    if raw_df is None or selected_path is None:
        summary = {
            **asdict(context),
            "status": "no_decision_records_found",
            "candidate_file_count": len(manifest),
            "message": "No suitable CSV with action/ticker/decision fields was found in the source run.",
        }
        (run_dir / "summary" / "decision_audit_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        _print("No decision records found. Wrote diagnostic summary.")
        return

    decision_df = _normalize_decision_df(raw_df, selected_path)
    universe = _derive_universe_from_files(source_run_dir, decision_df)

    action_df = _summarize_actions(decision_df)
    ticker_action_df = _summarize_ticker_actions(decision_df)
    trades_only_df = (
        decision_df[decision_df["executed_trade"]].copy()
        if "executed_trade" in decision_df.columns
        else pd.DataFrame()
    )
    holdings_df = _summarize_holdings_like_columns(decision_df)

    # Write outputs.
    decision_path = run_dir / "audit" / "decision_audit_by_step.csv"
    trades_path = run_dir / "audit" / "decision_audit_trades_only.csv"
    actions_path = run_dir / "audit" / "decision_audit_by_action.csv"
    ticker_actions_path = run_dir / "audit" / "decision_audit_by_ticker_action.csv"
    holdings_path = run_dir / "audit" / "decision_audit_holdings_by_ticker.csv"
    heatmap_path = run_dir / "plots" / "decision_audit_ticker_action_heatmap.png"
    md_path = run_dir / "summary" / "decision_audit_summary.md"
    json_path = run_dir / "summary" / "decision_audit_summary.json"

    decision_df.to_csv(decision_path, index=False)
    trades_only_df.to_csv(trades_path, index=False)
    action_df.to_csv(actions_path, index=False)
    ticker_action_df.to_csv(ticker_actions_path, index=False)
    holdings_df.to_csv(holdings_path, index=False)
    heatmap_created = _make_ticker_action_heatmap(ticker_action_df, heatmap_path)

    tickers_selected: list[str] = []
    if "selected_ticker" in decision_df.columns:
        tickers_selected = sorted(
            t
            for t in decision_df["selected_ticker"]
            .dropna()
            .astype(str)
            .str.upper()
            .unique()
            .tolist()
            if t and t not in {"NONE", "NAN", "UNKNOWN", "CASH"}
        )

    summary = {
        **asdict(context),
        "status": "ok",
        "source_decision_file": str(selected_path),
        "candidate_file_count": len(manifest),
        "decision_row_count": int(len(decision_df)),
        "executed_trade_row_count": (
            int(decision_df["executed_trade"].sum())
            if "executed_trade" in decision_df.columns
            else None
        ),
        "universe_inferred": universe,
        "universe_count": len(universe),
        "tickers_selected": tickers_selected,
        "tickers_selected_count": len(tickers_selected),
        "universe_tickers_never_selected": (
            sorted(set(universe) - set(tickers_selected)) if universe else []
        ),
        "heatmap_created": heatmap_created,
        "source_experiment_context": source_experiment_context,
        "outputs": {
            "decision_audit_by_step": str(decision_path),
            "decision_audit_trades_only": str(trades_path),
            "decision_audit_by_action": str(actions_path),
            "decision_audit_by_ticker_action": str(ticker_actions_path),
            "decision_audit_holdings_by_ticker": str(holdings_path),
            "decision_audit_ticker_action_heatmap": (
                str(heatmap_path) if heatmap_created else None
            ),
            "decision_audit_summary_md": str(md_path),
            "decision_audit_summary_json": str(json_path),
            "candidate_files_manifest": str(manifest_path),
        },
    }

    _write_markdown_report(
        md_path,
        context,
        universe,
        selected_path,
        decision_df,
        action_df,
        ticker_action_df,
        holdings_df,
        manifest,
    )
    _append_source_experiment_context_to_markdown(md_path, source_experiment_context)
    json_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    _wandb_log_outputs(run_dir, summary)

    _print("IQN decision audit report completed.")
    _print(f"Decision rows: {summary['decision_row_count']}")
    _print(f"Executed trade rows: {summary['executed_trade_row_count']}")
    _print(f"Universe inferred: {summary['universe_inferred']}")
    _print(f"Tickers selected: {summary['tickers_selected']}")
    _print(f"Wrote summary: {json_path}")
    _print(f"Wrote markdown: {md_path}")


if __name__ == "__main__":
    main()
