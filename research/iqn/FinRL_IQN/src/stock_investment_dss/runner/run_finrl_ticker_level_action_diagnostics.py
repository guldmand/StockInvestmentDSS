"""Ticker-level FinRL action diagnostics.

Purpose
-------
Answer concrete questions such as:
- Does the agent trade at all?
- Does it trade AAPL?
- Does it trade MSFT?
- Is a repeated return pattern caused by the same ticker-level action pattern?

This runner is intentionally self-contained and conservative. It scans recent
FinRL baseline-suite run folders under outputs/runs and reads files named
`*_action_memory.csv` and `*_asset_memory.csv`.
"""

from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_NAME = "StockInvestmentDSS"
PROTOTYPE_NAME = "D-IQN-DSS"
RUN_KIND = "finrl_ticker_level_action_diagnostics"


def _project_root() -> Path:
    path = Path.cwd()
    while path != path.parent:
        if (path / "src").exists() and (path / "outputs").exists():
            return path
        path = path.parent
    return Path.cwd()


def _timestamp() -> str:
    return datetime.now().strftime("%Y_%m_%d_%H%M%S")


def _make_run_dirs(root: Path) -> dict[str, Path]:
    run_id = f"{_timestamp()}_d_iqn_dss_{RUN_KIND}"
    run_dir = root / "outputs" / "runs" / run_id
    paths = {
        "run": run_dir,
        "data": run_dir / "data",
        "summary": run_dir / "summary",
        "logs": run_dir / "logs",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    paths["run_id"] = run_id  # type: ignore[assignment]
    return paths


def _parse_list_env(name: str, default: list[str]) -> list[str]:
    value = os.getenv(name)
    if not value:
        return default
    return [item.strip().lower() for item in value.split(",") if item.strip()]


def _find_candidate_runs(project_root: Path, limit: int) -> list[Path]:
    runs_dir = project_root / "outputs" / "runs"
    if not runs_dir.exists():
        return []
    candidates = [p for p in runs_dir.iterdir() if p.is_dir() and "finrl_baseline_suite" in p.name]
    return sorted(candidates, key=lambda p: p.name, reverse=True)[:limit]


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _numeric_columns(df: pd.DataFrame) -> list[str]:
    numeric_cols: list[str] = []
    for col in df.columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().sum() > 0:
            numeric_cols.append(col)
    return numeric_cols


def _infer_action_ticker_columns(action_df: pd.DataFrame, tickers: list[str]) -> dict[str, str | None]:
    """Map ticker to likely action column.

    FinRL action memories are often unnamed numeric columns or ticker-named
    columns depending on export format. Prefer direct ticker column names, then
    fall back to numeric columns in ticker order.
    """

    result: dict[str, str | None] = {}
    lower_to_col = {str(c).lower(): str(c) for c in action_df.columns}

    for ticker in tickers:
        result[ticker] = lower_to_col.get(ticker.lower())

    missing = [ticker for ticker, col in result.items() if col is None]
    if missing:
        numeric_cols = _numeric_columns(action_df)
        # Drop likely date/index columns from fallback.
        numeric_cols = [c for c in numeric_cols if str(c).lower() not in {"date", "day", "index", "Unnamed: 0".lower()}]
        for ticker, col in zip(tickers, numeric_cols):
            if result[ticker] is None:
                result[ticker] = col

    return result


def _portfolio_return_from_asset(asset_df: pd.DataFrame) -> float | None:
    if asset_df.empty:
        return None
    candidate_cols = [
        "account_value",
        "asset",
        "portfolio_value",
        "total_asset",
        "value",
    ]
    col = None
    lower_to_col = {str(c).lower(): str(c) for c in asset_df.columns}
    for name in candidate_cols:
        if name in lower_to_col:
            col = lower_to_col[name]
            break
    if col is None:
        numeric_cols = _numeric_columns(asset_df)
        if not numeric_cols:
            return None
        col = numeric_cols[-1]

    values = pd.to_numeric(asset_df[col], errors="coerce").dropna()
    if len(values) < 2 or values.iloc[0] == 0:
        return None
    return float((values.iloc[-1] / values.iloc[0] - 1.0) * 100.0)


def _hash_series(values: pd.Series) -> str:
    rounded = pd.to_numeric(values, errors="coerce").fillna(0.0).round(8)
    return str(hash(tuple(rounded.tolist())))


def _hash_asset(asset_df: pd.DataFrame) -> str | None:
    if asset_df.empty:
        return None
    numeric_cols = _numeric_columns(asset_df)
    if not numeric_cols:
        return None
    col = numeric_cols[-1]
    return _hash_series(asset_df[col])


@dataclass
class TickerDiagnostic:
    run_id: str
    agent_name: str
    ticker: str
    action_file: str | None
    asset_file: str | None
    action_column: str | None
    action_rows: int
    nonzero_action_count: int | None
    positive_action_count: int | None
    negative_action_count: int | None
    mean_action: float | None
    max_action: float | None
    min_action: float | None
    traded_ticker_flag: bool | None
    action_pattern_hash: str | None
    asset_trajectory_hash: str | None
    total_return_pct_from_asset: float | None


def _diagnose_agent_run(run_dir: Path, agent: str, tickers: list[str]) -> list[TickerDiagnostic]:
    agent_dir = run_dir / "data" / "finrl_baseline_suite" / agent
    if not agent_dir.exists():
        return []

    action_files = sorted(agent_dir.glob("*action_memory.csv"))
    asset_files = sorted(agent_dir.glob("*asset_memory.csv"))
    action_file = action_files[0] if action_files else None
    asset_file = asset_files[0] if asset_files else None

    action_df = _read_csv(action_file) if action_file else pd.DataFrame()
    asset_df = _read_csv(asset_file) if asset_file else pd.DataFrame()

    action_map = _infer_action_ticker_columns(action_df, tickers) if not action_df.empty else {t: None for t in tickers}
    asset_hash = _hash_asset(asset_df)
    asset_return = _portfolio_return_from_asset(asset_df)

    rows: list[TickerDiagnostic] = []
    for ticker in tickers:
        col = action_map.get(ticker)
        if col is None or action_df.empty:
            rows.append(
                TickerDiagnostic(
                    run_id=run_dir.name,
                    agent_name=agent,
                    ticker=ticker,
                    action_file=str(action_file) if action_file else None,
                    asset_file=str(asset_file) if asset_file else None,
                    action_column=None,
                    action_rows=0,
                    nonzero_action_count=None,
                    positive_action_count=None,
                    negative_action_count=None,
                    mean_action=None,
                    max_action=None,
                    min_action=None,
                    traded_ticker_flag=None,
                    action_pattern_hash=None,
                    asset_trajectory_hash=asset_hash,
                    total_return_pct_from_asset=asset_return,
                )
            )
            continue

        values = pd.to_numeric(action_df[col], errors="coerce").fillna(0.0)
        nonzero = int((values.abs() > 1e-12).sum())
        positive = int((values > 1e-12).sum())
        negative = int((values < -1e-12).sum())
        rows.append(
            TickerDiagnostic(
                run_id=run_dir.name,
                agent_name=agent,
                ticker=ticker,
                action_file=str(action_file) if action_file else None,
                asset_file=str(asset_file) if asset_file else None,
                action_column=str(col),
                action_rows=int(len(values)),
                nonzero_action_count=nonzero,
                positive_action_count=positive,
                negative_action_count=negative,
                mean_action=float(values.mean()),
                max_action=float(values.max()),
                min_action=float(values.min()),
                traded_ticker_flag=bool(nonzero > 0),
                action_pattern_hash=_hash_series(values),
                asset_trajectory_hash=asset_hash,
                total_return_pct_from_asset=asset_return,
            )
        )
    return rows


def _aggregate(rows: list[TickerDiagnostic]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([asdict(r) for r in rows])
    grouped = []
    for (agent, ticker), g in df.groupby(["agent_name", "ticker"], dropna=False):
        hashes = [h for h in g["action_pattern_hash"].dropna().unique().tolist()]
        asset_hashes = [h for h in g["asset_trajectory_hash"].dropna().unique().tolist()]
        grouped.append(
            {
                "agent_name": agent,
                "ticker": ticker,
                "run_count": int(len(g)),
                "traded_ticker_runs": int(g["traded_ticker_flag"].fillna(False).sum()),
                "traded_ticker_rate": float(g["traded_ticker_flag"].fillna(False).mean()),
                "nonzero_action_count_mean": float(pd.to_numeric(g["nonzero_action_count"], errors="coerce").mean()),
                "positive_action_count_mean": float(pd.to_numeric(g["positive_action_count"], errors="coerce").mean()),
                "negative_action_count_mean": float(pd.to_numeric(g["negative_action_count"], errors="coerce").mean()),
                "mean_action_mean": float(pd.to_numeric(g["mean_action"], errors="coerce").mean()),
                "unique_action_patterns": len(hashes),
                "unique_asset_trajectories": len(asset_hashes),
                "action_pattern_status": "identical" if len(hashes) == 1 else "varies" if len(hashes) > 1 else "missing",
                "asset_trajectory_status": "identical" if len(asset_hashes) == 1 else "varies" if len(asset_hashes) > 1 else "missing",
                "return_mean": float(pd.to_numeric(g["total_return_pct_from_asset"], errors="coerce").mean()),
                "return_std": float(pd.to_numeric(g["total_return_pct_from_asset"], errors="coerce").std(ddof=0)),
            }
        )
    return pd.DataFrame(grouped)


def main() -> None:
    root = _project_root()
    paths = _make_run_dirs(root)
    run_id = str(paths["run_id"])

    agents = _parse_list_env("STOCK_INVESTMENT_DSS_FINRL_BASELINE_AGENTS", ["a2c", "ddpg", "td3", "ppo", "sac"])
    include_mvo = os.getenv("STOCK_INVESTMENT_DSS_FINRL_BASELINE_INCLUDE_MVO", "true").lower() in {"1", "true", "yes", "on"}
    if include_mvo and "mvo" not in agents:
        agents.append("mvo")
    tickers = [t.upper() for t in _parse_list_env("STOCK_INVESTMENT_DSS_FINRL_TICKERS", ["AAPL", "MSFT"])]
    limit = int(os.getenv("STOCK_INVESTMENT_DSS_FINRL_DIAGNOSTIC_RUN_LIMIT", "30"))

    candidate_runs = _find_candidate_runs(root, limit=limit)
    rows: list[TickerDiagnostic] = []
    for run_dir in candidate_runs:
        for agent in agents:
            rows.extend(_diagnose_agent_run(run_dir, agent, tickers))

    member_df = pd.DataFrame([asdict(r) for r in rows])
    aggregate_df = _aggregate(rows)

    member_path = paths["data"] / "finrl_ticker_level_action_member_diagnostics.csv"
    aggregate_path = paths["summary"] / "finrl_ticker_level_action_aggregate.csv"
    summary_path = paths["summary"] / "finrl_ticker_level_action_diagnostics_summary.json"
    report_path = paths["summary"] / "finrl_ticker_level_action_diagnostics.md"

    member_df.to_csv(member_path, index=False)
    aggregate_df.to_csv(aggregate_path, index=False)

    summary = {
        "status": "ok",
        "project_name": PROJECT_NAME,
        "prototype_name": PROTOTYPE_NAME,
        "run_id": run_id,
        "project_root": str(root),
        "run_directory": str(paths["run"]),
        "candidate_run_count": len(candidate_runs),
        "member_row_count": len(member_df),
        "aggregate_row_count": len(aggregate_df),
        "agents": agents,
        "tickers": tickers,
        "outputs": {
            "member_diagnostics_path": str(member_path),
            "aggregate_path": str(aggregate_path),
            "report_path": str(report_path),
            "summary_path": str(summary_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    with report_path.open("w", encoding="utf-8") as f:
        f.write("# FinRL ticker-level action diagnostics\n\n")
        f.write(f"Candidate runs scanned: `{len(candidate_runs)}`\n\n")
        if aggregate_df.empty:
            f.write("No ticker-level diagnostics were produced.\n")
        else:
            f.write(aggregate_df.to_markdown(index=False))
            f.write("\n")

    print("FinRL ticker-level action diagnostics completed.")
    print(f"Run id: {run_id}")
    print(f"Wrote member diagnostics: {member_path}")
    print(f"Wrote aggregate diagnostics: {aggregate_path}")
    print(f"Wrote report: {report_path}")
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
