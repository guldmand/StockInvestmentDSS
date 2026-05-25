"""FinRL baseline action-pattern diagnostics.

This runner inspects FinRL baseline-suite outputs and answers questions such as:

- Did each agent actually emit non-zero actions?
- Are action patterns identical across seeds?
- Are portfolio trajectories identical across seeds?
- Do the agents appear to trade each ticker differently?
- Do identical final metrics correspond to identical action/asset patterns?

It is intentionally conservative:
- It does not assume a specific FinRL action-memory column schema.
- It treats positive actions as buy-like and negative actions as sell-like when the
  action-memory columns are numeric.
- If ticker names are not available in the action CSV, it maps numeric action
  columns to tickers from STOCK_INVESTMENT_DSS_FINRL_TICKERS, defaulting to
  AAPL,MSFT.

Recommended after:
    run_finrl_baseline_learning_curve_multiseed_launcher.py
or:
    run_finrl_baseline_multiseed_launcher.py
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_NAME = "StockInvestmentDSS"
PROTOTYPE_NAME = "D-IQN-DSS"
RUN_KIND = "finrl_action_pattern_diagnostics"


def _now_run_id() -> str:
    stamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    return f"{stamp}_d_iqn_dss_{RUN_KIND}"


def _find_project_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "src" / "stock_investment_dss").exists():
            return candidate
        if (candidate / "pyproject.toml").exists() and (candidate / "src").exists():
            return candidate
    return current


def _setup_run(project_root: Path) -> tuple[str, Path, logging.Logger]:
    run_id = _now_run_id()
    run_dir = project_root / "outputs" / "runs" / run_id
    for directory in (run_dir / "data", run_dir / "summary", run_dir / "logs"):
        directory.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("stock_investment_dss.run")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s", "%Y-%m-%d %H:%M:%S"
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(run_dir / "logs" / "run.log", encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    logger.info("Starting %s FinRL action-pattern diagnostics.", PROJECT_NAME)
    logger.info("Project root: %s", project_root)
    logger.info("Created run directory: %s", run_dir)
    logger.info("Run id: %s", run_id)
    return run_id, run_dir, logger


def _parse_str_list(value: str | None, default: list[str] | None = None) -> list[str]:
    if not value:
        return default or []
    return [x.strip().lower() for x in value.split(",") if x.strip()]


def _safe_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except Exception:
        return None


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _hash_dataframe(df: pd.DataFrame, rounding: int = 8) -> str:
    if df.empty:
        return "empty"
    prepared = df.copy()
    for col in prepared.columns:
        if pd.api.types.is_numeric_dtype(prepared[col]):
            prepared[col] = pd.to_numeric(prepared[col], errors="coerce").round(
                rounding
            )
        else:
            prepared[col] = prepared[col].astype(str)
    csv_text = prepared.to_csv(index=False)
    return hashlib.sha256(csv_text.encode("utf-8")).hexdigest()[:16]


def _hash_numeric_frame(df: pd.DataFrame, rounding: int = 8) -> str:
    numeric = df.select_dtypes(include=[np.number]).copy()
    if numeric.empty:
        return "no_numeric"
    return _hash_dataframe(numeric, rounding=rounding)


def _find_latest_run_dir(project_root: Path, pattern: str) -> Path | None:
    runs_dir = project_root / "outputs" / "runs"
    if not runs_dir.exists():
        return None
    matches = [p for p in runs_dir.iterdir() if p.is_dir() and pattern in p.name]
    if not matches:
        return None
    return sorted(matches, key=lambda p: p.name, reverse=True)[0]


def _iter_recent_run_dirs(project_root: Path, limit: int) -> list[Path]:
    runs_dir = project_root / "outputs" / "runs"
    if not runs_dir.exists():
        return []
    dirs = [p for p in runs_dir.iterdir() if p.is_dir()]
    return sorted(dirs, key=lambda p: p.name, reverse=True)[:limit]


@dataclass
class ChildRunReference:
    child_run_id: str
    child_run_dir: str
    seed: int | None = None
    train_step: int | None = None
    launcher_run_id: str | None = None
    source_type: str | None = None


def _extract_child_id_from_log(log_path: Path) -> str | None:
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None
    match = re.search(
        r"child_run_id=([0-9_]+_d_iqn_dss_finrl_baseline_suite_smoke_test)", text
    )
    if match:
        return match.group(1)
    matches = re.findall(
        r"Created run directory:\s+(.+?finrl_baseline_suite_smoke_test)", text
    )
    if matches:
        return Path(matches[-1].strip()).name
    return None


def _child_refs_from_launcher(
    project_root: Path, launcher_dir: Path
) -> list[ChildRunReference]:
    refs: list[ChildRunReference] = []
    launcher_run_id = launcher_dir.name
    summary_dir = launcher_dir / "summary"
    logs_dir = launcher_dir / "logs"

    for summary_path in summary_dir.glob("*launcher_summary.json"):
        summary = _read_json(summary_path)
        launched = (
            summary.get("launched_runs")
            or summary.get("runs")
            or summary.get("child_runs")
            or []
        )
        if not isinstance(launched, list):
            continue
        for item in launched:
            if not isinstance(item, dict):
                continue
            child_run_id = (
                item.get("child_run_id")
                or item.get("run_id")
                or item.get("source_run_id")
            )
            log_path = item.get("log_path")
            if not child_run_id and log_path:
                child_run_id = _extract_child_id_from_log(Path(log_path))
            if not child_run_id:
                continue
            refs.append(
                ChildRunReference(
                    child_run_id=child_run_id,
                    child_run_dir=str(project_root / "outputs" / "runs" / child_run_id),
                    seed=_safe_int(item.get("seed")),
                    train_step=_safe_int(
                        item.get("train_step")
                        or item.get("train_steps")
                        or item.get("total_timesteps")
                    ),
                    launcher_run_id=launcher_run_id,
                    source_type="launcher_summary",
                )
            )

    if logs_dir.exists():
        for log_path in sorted(logs_dir.glob("*.log")):
            child_run_id = _extract_child_id_from_log(log_path)
            if not child_run_id:
                continue
            seed = None
            train_step = None
            m_seed = re.search(r"seed[_= -]*(\d+)", log_path.stem, flags=re.IGNORECASE)
            m_step = re.search(
                r"(?:step|train_step|timesteps)[_= -]*(\d+)",
                log_path.stem,
                flags=re.IGNORECASE,
            )
            if m_seed:
                seed = int(m_seed.group(1))
            if m_step:
                train_step = int(m_step.group(1))
            refs.append(
                ChildRunReference(
                    child_run_id=child_run_id,
                    child_run_dir=str(project_root / "outputs" / "runs" / child_run_id),
                    seed=seed,
                    train_step=train_step,
                    launcher_run_id=launcher_run_id,
                    source_type="launcher_log",
                )
            )

    by_key: dict[tuple[str, int | None, int | None], ChildRunReference] = {}
    for ref in refs:
        by_key[(ref.child_run_id, ref.seed, ref.train_step)] = ref
    return list(by_key.values())


def _load_child_refs(
    project_root: Path, logger: logging.Logger
) -> list[ChildRunReference]:
    explicit_launcher = os.getenv(
        "STOCK_INVESTMENT_DSS_FINRL_ACTION_DIAGNOSTICS_LAUNCHER_RUN_ID"
    )
    recent_limit = int(
        os.getenv("STOCK_INVESTMENT_DSS_ACTION_DIAGNOSTICS_RECENT_RUN_LIMIT", "200")
    )

    launcher_dirs: list[Path] = []
    if explicit_launcher:
        launcher_dirs.append(project_root / "outputs" / "runs" / explicit_launcher)
    else:
        for pattern in [
            "finrl_baseline_learning_curve_multiseed_launcher",
            "finrl_baseline_multiseed_launcher",
        ]:
            latest = _find_latest_run_dir(project_root, pattern)
            if latest:
                launcher_dirs.append(latest)

    refs: list[ChildRunReference] = []
    for launcher_dir in launcher_dirs:
        if launcher_dir.exists():
            logger.info("Reading launcher metadata: %s", launcher_dir)
            refs.extend(_child_refs_from_launcher(project_root, launcher_dir))

    if (
        not refs
        or os.getenv(
            "STOCK_INVESTMENT_DSS_ACTION_DIAGNOSTICS_INCLUDE_RECENT_SCAN", "false"
        ).lower()
        == "true"
    ):
        for run_dir in _iter_recent_run_dirs(project_root, recent_limit):
            if "finrl_baseline_suite_smoke_test" not in run_dir.name:
                continue
            refs.append(
                ChildRunReference(
                    child_run_id=run_dir.name,
                    child_run_dir=str(run_dir),
                    source_type="recent_scan",
                )
            )

    deduped: dict[tuple[str, int | None, int | None], ChildRunReference] = {}
    for ref in refs:
        deduped[(ref.child_run_id, ref.seed, ref.train_step)] = ref
    refs = sorted(
        deduped.values(),
        key=lambda r: (r.train_step or -1, r.seed or -1, r.child_run_id),
    )
    logger.info("Child baseline-suite runs discovered: %s", len(refs))
    return refs


def _read_csv_lenient(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_csv(path, header=None)


def _numeric_action_columns(df: pd.DataFrame) -> list[str]:
    numeric_cols: list[str] = []
    for col in df.columns:
        series = pd.to_numeric(df[col], errors="coerce")
        if series.notna().sum() > 0:
            numeric_cols.append(str(col))
    drop_names = {
        "index",
        "date",
        "day",
        "timestamp",
        "time",
        "Unnamed: 0",
        "Unnamed: 0.1",
    }
    filtered = [
        c
        for c in numeric_cols
        if c not in drop_names and not c.lower().startswith("unnamed")
    ]
    return filtered or numeric_cols


def _asset_value_column(df: pd.DataFrame) -> str | None:
    preferred = [
        "account_value",
        "portfolio_value",
        "asset_value",
        "total_asset",
        "total_assets",
        "value",
    ]
    lower_to_col = {str(c).lower(): str(c) for c in df.columns}
    for name in preferred:
        if name in lower_to_col:
            return lower_to_col[name]
    numeric_cols = [
        str(c)
        for c in df.columns
        if pd.to_numeric(df[c], errors="coerce").notna().sum() > 0
    ]
    if not numeric_cols:
        return None
    non_index = [
        c
        for c in numeric_cols
        if not c.lower().startswith("unnamed") and c.lower() not in {"index", "day"}
    ]
    return non_index[-1] if non_index else numeric_cols[-1]


def _ticker_labels(num_actions: int) -> list[str]:
    configured = _parse_str_list(
        os.getenv("STOCK_INVESTMENT_DSS_FINRL_TICKERS")
        or os.getenv("STOCK_INVESTMENT_DSS_TICKERS")
    )
    if not configured:
        configured = ["aapl", "msft"]
    labels = [x.upper() for x in configured]
    if len(labels) >= num_actions:
        return labels[:num_actions]
    return labels + [f"ASSET_{i + 1}" for i in range(len(labels), num_actions)]


def _find_agent_dirs(run_dir: Path) -> list[Path]:
    suite_dir = run_dir / "data" / "finrl_baseline_suite"
    if not suite_dir.exists():
        return []
    return [p for p in suite_dir.iterdir() if p.is_dir()]


@dataclass
class AgentDiagnostics:
    child_run_id: str
    launcher_run_id: str | None
    source_type: str | None
    seed: int | None
    train_step: int | None
    agent_name: str
    has_action_memory: bool
    has_asset_memory: bool
    action_rows: int
    action_numeric_columns: str
    action_hash: str | None
    action_nonzero_total: int | None
    action_nonzero_share: float | None
    buy_like_count_total: int | None
    sell_like_count_total: int | None
    hold_like_count_total: int | None
    mean_abs_action: float | None
    max_abs_action: float | None
    per_ticker_summary_json: str | None
    asset_rows: int
    asset_value_column: str | None
    asset_hash: str | None
    final_value: float | None
    total_return_pct_from_asset: float | None
    max_drawdown_pct_from_asset: float | None
    action_memory_path: str | None
    asset_memory_path: str | None


def _diagnose_agent(ref: ChildRunReference, agent_dir: Path) -> AgentDiagnostics:
    agent_name = agent_dir.name.lower()
    action_paths = sorted(agent_dir.glob("*action_memory*.csv"))
    asset_paths = sorted(agent_dir.glob("*asset_memory*.csv"))
    action_path = action_paths[0] if action_paths else None
    asset_path = asset_paths[0] if asset_paths else None

    action_rows = 0
    action_numeric_columns = ""
    action_hash = None
    action_nonzero_total = None
    action_nonzero_share = None
    buy_like_count_total = None
    sell_like_count_total = None
    hold_like_count_total = None
    mean_abs_action = None
    max_abs_action = None
    per_ticker_summary_json = None

    if action_path and action_path.exists():
        action_df = _read_csv_lenient(action_path)
        action_rows = int(len(action_df))
        numeric_cols = _numeric_action_columns(action_df)
        action_numeric_columns = ",".join(numeric_cols)
        action_numeric = pd.DataFrame()
        for col in numeric_cols:
            action_numeric[col] = pd.to_numeric(action_df[col], errors="coerce")
        action_numeric = action_numeric.dropna(how="all")
        if not action_numeric.empty:
            action_hash = _hash_dataframe(action_numeric, rounding=6)
            values = action_numeric.to_numpy(dtype=float)
            abs_values = np.abs(values)
            nonzero = np.abs(values) > 1e-9
            buy_like = values > 1e-9
            sell_like = values < -1e-9
            hold_like = ~nonzero
            action_nonzero_total = int(nonzero.sum())
            action_nonzero_share = float(nonzero.mean())
            buy_like_count_total = int(buy_like.sum())
            sell_like_count_total = int(sell_like.sum())
            hold_like_count_total = int(hold_like.sum())
            mean_abs_action = float(np.nanmean(abs_values))
            max_abs_action = float(np.nanmax(abs_values))
            labels = _ticker_labels(len(numeric_cols))
            per_ticker = []
            for index, col in enumerate(numeric_cols):
                series = action_numeric[col].to_numpy(dtype=float)
                abs_series = np.abs(series)
                per_ticker.append(
                    {
                        "ticker": labels[index] if index < len(labels) else str(col),
                        "column": str(col),
                        "rows": int(len(series)),
                        "nonzero_count": int((abs_series > 1e-9).sum()),
                        "buy_like_count": int((series > 1e-9).sum()),
                        "sell_like_count": int((series < -1e-9).sum()),
                        "mean_abs_action": float(np.nanmean(abs_series)),
                        "max_abs_action": float(np.nanmax(abs_series)),
                    }
                )
            per_ticker_summary_json = json.dumps(per_ticker, ensure_ascii=False)

    asset_rows = 0
    asset_value_column = None
    asset_hash = None
    final_value = None
    total_return_pct_from_asset = None
    max_drawdown_pct_from_asset = None
    if asset_path and asset_path.exists():
        asset_df = _read_csv_lenient(asset_path)
        asset_rows = int(len(asset_df))
        asset_value_column = _asset_value_column(asset_df)
        asset_hash = _hash_numeric_frame(asset_df, rounding=6)
        if asset_value_column:
            values = pd.to_numeric(
                asset_df[asset_value_column], errors="coerce"
            ).dropna()
            if not values.empty:
                initial = float(values.iloc[0])
                final = float(values.iloc[-1])
                final_value = final
                if initial != 0:
                    total_return_pct_from_asset = float((final / initial - 1.0) * 100.0)
                running_max = values.cummax()
                drawdown = values / running_max - 1.0
                max_drawdown_pct_from_asset = float(drawdown.min() * 100.0)

    return AgentDiagnostics(
        child_run_id=ref.child_run_id,
        launcher_run_id=ref.launcher_run_id,
        source_type=ref.source_type,
        seed=ref.seed,
        train_step=ref.train_step,
        agent_name=agent_name,
        has_action_memory=action_path is not None,
        has_asset_memory=asset_path is not None,
        action_rows=action_rows,
        action_numeric_columns=action_numeric_columns,
        action_hash=action_hash,
        action_nonzero_total=action_nonzero_total,
        action_nonzero_share=action_nonzero_share,
        buy_like_count_total=buy_like_count_total,
        sell_like_count_total=sell_like_count_total,
        hold_like_count_total=hold_like_count_total,
        mean_abs_action=mean_abs_action,
        max_abs_action=max_abs_action,
        per_ticker_summary_json=per_ticker_summary_json,
        asset_rows=asset_rows,
        asset_value_column=asset_value_column,
        asset_hash=asset_hash,
        final_value=final_value,
        total_return_pct_from_asset=total_return_pct_from_asset,
        max_drawdown_pct_from_asset=max_drawdown_pct_from_asset,
        action_memory_path=str(action_path) if action_path else None,
        asset_memory_path=str(asset_path) if asset_path else None,
    )


def _build_member_diagnostics(
    project_root: Path, refs: list[ChildRunReference], logger: logging.Logger
) -> pd.DataFrame:
    records = []
    agents_filter = _parse_str_list(
        os.getenv("STOCK_INVESTMENT_DSS_FINRL_BASELINE_AGENTS"), default=[]
    )
    include_mvo = (
        os.getenv("STOCK_INVESTMENT_DSS_FINRL_BASELINE_INCLUDE_MVO", "true").lower()
        == "true"
    )
    if include_mvo and "mvo" not in agents_filter and agents_filter:
        agents_filter.append("mvo")

    for ref in refs:
        run_dir = Path(ref.child_run_dir)
        if not run_dir.exists():
            logger.warning("Child run directory missing: %s", run_dir)
            continue
        for agent_dir in _find_agent_dirs(run_dir):
            agent_name = agent_dir.name.lower()
            if agents_filter and agent_name not in agents_filter:
                continue
            records.append(asdict(_diagnose_agent(ref, agent_dir)))
    return pd.DataFrame(records)


def _nunique_non_null(series: pd.Series) -> int:
    return int(series.dropna().nunique())


def _mean(group: pd.DataFrame, col: str) -> float | None:
    if col not in group:
        return None
    values = pd.to_numeric(group[col], errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.mean())


def _std(group: pd.DataFrame, col: str) -> float | None:
    if col not in group:
        return None
    values = pd.to_numeric(group[col], errors="coerce").dropna()
    if len(values) <= 1:
        return 0.0 if len(values) == 1 else None
    return float(values.std(ddof=1))


def _interpret_group(
    agent_name: str,
    action_hash_unique: int,
    asset_hash_unique: int,
    group: pd.DataFrame,
) -> str:
    if agent_name == "mvo":
        return "MVO is deterministic in this setup; identical asset trajectories are expected."
    if action_hash_unique == 1 and asset_hash_unique == 1:
        return "Actions and portfolio trajectory are identical across seeds; inspect seed propagation or policy collapse."
    if action_hash_unique > 1 and asset_hash_unique == 1:
        return "Actions vary but portfolio trajectory is identical; environment/action clipping may collapse behavior."
    if action_hash_unique == 1 and asset_hash_unique > 1:
        return "Actions appear identical but portfolio trajectory varies; inspect data/order or asset-memory extraction."
    if action_hash_unique > 1 and asset_hash_unique > 1:
        return "Actions and portfolio trajectory vary across seeds."
    if group["has_action_memory"].sum() == 0:
        return "No action memory found; cannot diagnose action variation."
    return "Partial diagnostics only; inspect member rows."


def _aggregate_patterns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    rows = []
    for (agent_name, train_step), group in df.groupby(
        ["agent_name", "train_step"], dropna=False
    ):
        seeds = sorted([int(x) for x in group["seed"].dropna().unique()])
        action_hash_unique = _nunique_non_null(group["action_hash"])
        asset_hash_unique = _nunique_non_null(group["asset_hash"])
        rows.append(
            {
                "agent_name": agent_name,
                "train_step": train_step,
                "member_count": int(len(group)),
                "seed_count": int(len(seeds)),
                "seeds": ",".join(str(x) for x in seeds),
                "has_action_memory_count": int(group["has_action_memory"].sum()),
                "has_asset_memory_count": int(group["has_asset_memory"].sum()),
                "action_hash_unique_count": action_hash_unique,
                "asset_hash_unique_count": asset_hash_unique,
                "action_pattern_status": (
                    "identical"
                    if action_hash_unique == 1
                    else ("missing" if action_hash_unique == 0 else "varies")
                ),
                "asset_trajectory_status": (
                    "identical"
                    if asset_hash_unique == 1
                    else ("missing" if asset_hash_unique == 0 else "varies")
                ),
                "action_nonzero_total_mean": _mean(group, "action_nonzero_total"),
                "action_nonzero_total_std": _std(group, "action_nonzero_total"),
                "action_nonzero_share_mean": _mean(group, "action_nonzero_share"),
                "buy_like_count_total_mean": _mean(group, "buy_like_count_total"),
                "sell_like_count_total_mean": _mean(group, "sell_like_count_total"),
                "mean_abs_action_mean": _mean(group, "mean_abs_action"),
                "max_abs_action_mean": _mean(group, "max_abs_action"),
                "final_value_mean": _mean(group, "final_value"),
                "final_value_std": _std(group, "final_value"),
                "total_return_pct_from_asset_mean": _mean(
                    group, "total_return_pct_from_asset"
                ),
                "total_return_pct_from_asset_std": _std(
                    group, "total_return_pct_from_asset"
                ),
                "max_drawdown_pct_from_asset_mean": _mean(
                    group, "max_drawdown_pct_from_asset"
                ),
                "max_drawdown_pct_from_asset_std": _std(
                    group, "max_drawdown_pct_from_asset"
                ),
                "diagnostic_interpretation": _interpret_group(
                    agent_name, action_hash_unique, asset_hash_unique, group
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["agent_name", "train_step"], na_position="last"
    )


def _write_markdown_report(
    path: Path,
    run_id: str,
    source_refs: list[ChildRunReference],
    member_df: pd.DataFrame,
    aggregate_df: pd.DataFrame,
) -> None:
    lines = []
    lines.append("# FinRL baseline action-pattern diagnostics")
    lines.append("")
    lines.append(f"Run id: `{run_id}`")
    lines.append("")
    lines.append(
        "This report inspects `action_memory.csv` and `asset_memory.csv` from FinRL baseline-suite runs."
    )
    lines.append("")
    lines.append(
        "Positive numeric actions are interpreted as buy-like, negative numeric actions as sell-like, and zero as hold-like. This is a diagnostic approximation based on the standard FinRL action convention."
    )
    lines.append("")
    lines.append(f"Child runs inspected: **{len(source_refs)}**")
    lines.append(f"Agent-member rows: **{len(member_df)}**")
    lines.append("")
    if aggregate_df.empty:
        lines.append("No aggregate diagnostics were produced.")
    else:
        cols = [
            "agent_name",
            "train_step",
            "seed_count",
            "action_pattern_status",
            "asset_trajectory_status",
            "action_nonzero_total_mean",
            "total_return_pct_from_asset_mean",
            "diagnostic_interpretation",
        ]
        present = [c for c in cols if c in aggregate_df.columns]
        lines.append("## Aggregate diagnostics by agent and training budget")
        lines.append("")
        lines.append(aggregate_df[present].to_markdown(index=False))
        lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "- If `action_pattern_status = identical` for a stochastic RL agent, the seed may not be propagated deeply enough, or the policy/environment may collapse to the same behavior."
    )
    lines.append(
        "- If `asset_trajectory_status = identical`, final metrics will usually also be identical."
    )
    lines.append(
        "- For MVO, identical trajectories are expected unless the inputs or estimation procedure are randomized."
    )
    lines.append(
        "- If actions vary but asset trajectories are identical, inspect action clipping, transaction thresholds, and whether the environment effectively executes trades."
    )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    project_root = _find_project_root()
    run_id, run_dir, logger = _setup_run(project_root)
    data_dir = run_dir / "data"
    summary_dir = run_dir / "summary"

    try:
        refs = _load_child_refs(project_root, logger)
        member_df = _build_member_diagnostics(project_root, refs, logger)
        aggregate_df = _aggregate_patterns(member_df)

        refs_path = data_dir / "finrl_action_pattern_source_runs.csv"
        member_path = data_dir / "finrl_action_pattern_member_diagnostics.csv"
        aggregate_path = (
            summary_dir / "finrl_action_pattern_aggregate_by_agent_step.csv"
        )
        report_path = summary_dir / "finrl_action_pattern_diagnostics.md"
        summary_path = summary_dir / "finrl_action_pattern_diagnostics_summary.json"

        pd.DataFrame([asdict(ref) for ref in refs]).to_csv(refs_path, index=False)
        member_df.to_csv(member_path, index=False)
        aggregate_df.to_csv(aggregate_path, index=False)
        _write_markdown_report(report_path, run_id, refs, member_df, aggregate_df)

        summary = {
            "status": "ok",
            "project_name": PROJECT_NAME,
            "prototype_name": PROTOTYPE_NAME,
            "run_id": run_id,
            "project_root": str(project_root),
            "run_directory": str(run_dir),
            "source_child_run_count": len(refs),
            "member_diagnostic_row_count": int(len(member_df)),
            "aggregate_row_count": int(len(aggregate_df)),
            "outputs": {
                "source_runs_path": str(refs_path),
                "member_diagnostics_path": str(member_path),
                "aggregate_by_agent_step_path": str(aggregate_path),
                "markdown_report_path": str(report_path),
                "summary_path": str(summary_path),
            },
            "interpretation": (
                "This diagnostic inspects FinRL action_memory and asset_memory files. "
                "Identical action and asset hashes across seeds indicate either deterministic behavior, "
                "insufficient seed propagation, or policy/environment collapse to the same behavior."
            ),
        }
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        logger.info("FinRL action-pattern diagnostics completed.")
        logger.info("Source child runs: %s", len(refs))
        logger.info("Member diagnostic rows: %s", len(member_df))
        logger.info("Aggregate rows: %s", len(aggregate_df))
        logger.info("Wrote member diagnostics: %s", member_path)
        logger.info("Wrote aggregate diagnostics: %s", aggregate_path)
        logger.info("Wrote report: %s", report_path)
        logger.info("Wrote summary: %s", summary_path)
        return 0
    except Exception:
        logger.exception("FinRL action-pattern diagnostics failed.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
