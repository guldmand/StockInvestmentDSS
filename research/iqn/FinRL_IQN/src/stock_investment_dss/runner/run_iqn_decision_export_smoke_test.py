#!/usr/bin/env python
"""
run_iqn_decision_export_smoke_test.py

Export date-indexed IQN decision/distribution outputs from an existing
IQN learning-curve smoke test run artifact.

IQN eval outputs use an integer eval_step index (not calendar dates).
This runner bridges that gap by:
  1. Loading experiment_context_summary.json for the eval window.
  2. Deriving eval_step -> date mapping from the market import file.
  3. Pivoting eval_distributions.csv to wide format (one row per step).
  4. Joining step records + distributions + date map.
  5. Writing a date-indexed export CSV for downstream use.

Env vars (all optional):
  STOCK_INVESTMENT_DSS_IQN_EXPORT_SOURCE_RUN_ID
      Override: specific IQN learning-curve run ID to use.
  STOCK_INVESTMENT_DSS_IQN_EXPORT_TRAIN_STEP
      Override: which training checkpoint to export (default: max/final).
"""

import os
import sys
import json
import logging
import pathlib
import datetime

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def _find_project_root() -> pathlib.Path:
    marker = ".env.example"
    here = pathlib.Path(__file__).resolve()
    for parent in [here] + list(here.parents):
        if (parent / marker).exists():
            return parent
    return pathlib.Path.cwd()


PROJECT_ROOT = _find_project_root()
RUNS_DIR = PROJECT_ROOT / "outputs" / "runs"
MARKET_IMPORT_FILE = (
    PROJECT_ROOT / "data" / "market" / "daily" / "imports" / "market_data_full_500.csv"
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Env config
# ---------------------------------------------------------------------------

SOURCE_RUN_ID_OVERRIDE = os.environ.get(
    "STOCK_INVESTMENT_DSS_IQN_EXPORT_SOURCE_RUN_ID", ""
).strip()
TRAIN_STEP_OVERRIDE = os.environ.get(
    "STOCK_INVESTMENT_DSS_IQN_EXPORT_TRAIN_STEP", ""
).strip()

# ---------------------------------------------------------------------------
# Source run discovery
# ---------------------------------------------------------------------------

_IQN_PATTERN = "iqn_learning_curve_smoke_test"


def _discover_source_run() -> pathlib.Path:
    if SOURCE_RUN_ID_OVERRIDE:
        candidate = RUNS_DIR / SOURCE_RUN_ID_OVERRIDE
        if candidate.is_dir():
            log.info("Using explicit source run: %s", SOURCE_RUN_ID_OVERRIDE)
            return candidate
        raise FileNotFoundError(
            f"Explicit source run not found: {candidate}\n"
            f"Set STOCK_INVESTMENT_DSS_IQN_EXPORT_SOURCE_RUN_ID correctly."
        )

    matches = sorted(
        [d for d in RUNS_DIR.iterdir() if d.is_dir() and _IQN_PATTERN in d.name],
        key=lambda d: d.name,
    )
    if not matches:
        raise FileNotFoundError(
            f"No IQN learning-curve runs found in {RUNS_DIR}.\n"
            f"Run run_iqn_learning_curve_smoke_test first."
        )
    chosen = matches[-1]
    log.info("Auto-discovered source run: %s", chosen.name)
    return chosen


# ---------------------------------------------------------------------------
# Experiment context
# ---------------------------------------------------------------------------


def _load_experiment_context(run_dir: pathlib.Path) -> dict:
    for sub in ("summary", "."):
        ctx_path = run_dir / sub / "experiment_context_summary.json"
        if ctx_path.exists():
            with open(ctx_path, encoding="utf-8") as f:
                return json.load(f)
    # Fallback: read iqn_learning_curve_summary.json
    for sub in ("summary", "."):
        summ_path = run_dir / sub / "iqn_learning_curve_summary.json"
        if summ_path.exists():
            with open(summ_path, encoding="utf-8") as f:
                return json.load(f)
    raise FileNotFoundError(
        f"No experiment_context_summary.json or iqn_learning_curve_summary.json in {run_dir}"
    )


# ---------------------------------------------------------------------------
# Eval date map: eval_step i -> calendar date
# ---------------------------------------------------------------------------


def _build_eval_date_map(context: dict) -> dict:
    """Return {eval_step: date_str} using market import file."""
    eval_start = context.get("eval_window_start") or context.get("pit_cutoff")
    eval_end = context.get("eval_window_end") or context.get("trade_end_date")
    tickers = context.get("tickers", [])

    if not eval_start or not eval_end:
        raise ValueError(
            "Cannot derive eval date map: eval_window_start/eval_window_end "
            "not found in experiment context."
        )

    if not MARKET_IMPORT_FILE.exists():
        raise FileNotFoundError(
            f"Market import file not found: {MARKET_IMPORT_FILE}\n"
            "Required for eval_step -> date mapping."
        )

    log.info("Building eval date map from %s", MARKET_IMPORT_FILE.name)
    mdf = pd.read_csv(MARKET_IMPORT_FILE, usecols=["date", "tic"])

    if tickers:
        mdf = mdf[mdf["tic"].isin(tickers)]

    mdf = mdf[(mdf["date"] >= eval_start) & (mdf["date"] <= eval_end)]
    unique_dates = sorted(mdf["date"].unique())
    log.info(
        "Eval date range: %s to %s (%d unique trading days)",
        unique_dates[0] if unique_dates else "?",
        unique_dates[-1] if unique_dates else "?",
        len(unique_dates),
    )

    # eval_step i -> unique_dates[i]
    # The terminal state date (unique_dates[-1]) may exceed the last eval_step
    date_map = {i: d for i, d in enumerate(unique_dates)}
    return date_map


# ---------------------------------------------------------------------------
# Load eval step records
# ---------------------------------------------------------------------------


def _load_step_records(run_dir: pathlib.Path, train_step: int) -> pd.DataFrame:
    path = run_dir / "data" / "iqn_learning_curve_eval_step_records.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"eval_step_records.csv not found in {run_dir / 'data'}"
        )
    df = pd.read_csv(path)
    df_ts = df[df["train_step"] == train_step].copy()
    if df_ts.empty:
        available = sorted(df["train_step"].unique())
        raise ValueError(
            f"No step records for train_step={train_step}. Available: {available}"
        )
    log.info("Loaded %d step records for train_step=%d", len(df_ts), train_step)
    return df_ts.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Load and pivot distributions
# ---------------------------------------------------------------------------

_ACTION_SLUGS = {
    "HOLD": "hold",
    "BUY": "buy",
    "SELL": "sell",
    "REBALANCE": "rebalance",
    "CHANGE_STRATEGY": "change_strategy",
}

_CORE_ACTIONS = ["HOLD", "BUY", "SELL", "REBALANCE"]


def _load_and_pivot_distributions(
    run_dir: pathlib.Path, train_step: int
) -> pd.DataFrame:
    path = run_dir / "data" / "iqn_learning_curve_eval_distributions.csv"
    if not path.exists():
        log.warning("eval_distributions.csv not found — quantile columns will be NaN")
        return pd.DataFrame()

    df = pd.read_csv(path)
    df_ts = df[df["train_step"] == train_step].copy()
    if df_ts.empty:
        log.warning(
            "No distributions for train_step=%d — quantile columns will be NaN",
            train_step,
        )
        return pd.DataFrame()

    log.info("Loaded %d distribution rows for train_step=%d", len(df_ts), train_step)

    # Build wide pivot: one row per eval_step
    records = []
    for eval_step, grp in df_ts.groupby("eval_step"):
        row = {"eval_step": eval_step}
        for _, drow in grp.iterrows():
            raw_action = str(drow.get("action", "")).upper().strip()
            slug = _ACTION_SLUGS.get(raw_action)
            if slug is None:
                continue
            row[f"action_score_{slug}"] = drow.get("score", np.nan)
            row[f"mean_{slug}"] = drow.get("mean", np.nan)
            row[f"q10_{slug}"] = drow.get("q10", np.nan)
            row[f"q25_{slug}"] = drow.get("q25", np.nan)
            row[f"q50_{slug}"] = drow.get("q50", np.nan)
            row[f"q75_{slug}"] = drow.get("q75", np.nan)
            row[f"q90_{slug}"] = drow.get("q90", np.nan)
            row[f"cvar_{slug}"] = drow.get("cvar10", np.nan)
        records.append(row)

    if not records:
        return pd.DataFrame()

    pivot = pd.DataFrame(records).sort_values("eval_step").reset_index(drop=True)
    log.info(
        "Pivoted distributions: %d eval_steps, %d columns",
        len(pivot),
        len(pivot.columns),
    )
    return pivot


# ---------------------------------------------------------------------------
# Build export frame
# ---------------------------------------------------------------------------


def _build_export(
    step_df: pd.DataFrame,
    dist_pivot_df: pd.DataFrame,
    date_map: dict,
    source_run_id: str,
    train_step: int,
) -> pd.DataFrame:
    """Join step records + distributions + date map into final export frame."""
    df = step_df.copy()

    # Add date
    df["date"] = df["eval_step"].map(date_map)
    missing_date = df["date"].isna().sum()
    if missing_date > 0:
        log.warning(
            "%d eval_steps have no date mapping (eval period ended earlier)",
            missing_date,
        )

    # Join distributions if available
    if not dist_pivot_df.empty:
        df = df.merge(dist_pivot_df, on="eval_step", how="left")

    # Canonical action type column
    df["iqn_selected_action"] = df["chosen_action_label"].fillna(
        df.get("effective_action", "")
    )
    df["selected_action_type"] = df["iqn_selected_action"]

    # Uncertainty proxy: q90 - q10 for chosen action
    def _uncertainty_proxy(row):
        action_label = str(row.get("iqn_selected_action", "")).upper().strip()
        slug = _ACTION_SLUGS.get(action_label)
        if not slug:
            return np.nan
        q90 = row.get(f"q90_{slug}", np.nan)
        q10 = row.get(f"q10_{slug}", np.nan)
        if pd.isna(q90) or pd.isna(q10):
            return np.nan
        return q90 - q10

    df["iqn_uncertainty_proxy"] = df.apply(_uncertainty_proxy, axis=1)

    # Risk score: mean - 0.5 * |cvar| for chosen action
    def _risk_score(row):
        action_label = str(row.get("iqn_selected_action", "")).upper().strip()
        slug = _ACTION_SLUGS.get(action_label)
        if not slug:
            return np.nan
        mean_val = row.get(f"mean_{slug}", np.nan)
        cvar_val = row.get(f"cvar_{slug}", np.nan)
        if pd.isna(mean_val) or pd.isna(cvar_val):
            return np.nan
        return mean_val - 0.5 * abs(cvar_val)

    df["iqn_risk_score"] = df.apply(_risk_score, axis=1)

    # Source metadata
    df["iqn_source_run_id"] = source_run_id
    df["train_step"] = train_step

    # Select and order final columns
    core_cols = [
        "date",
        "eval_step",
        "iqn_source_run_id",
        "train_step",
        "iqn_selected_action",
        "selected_action_type",
    ]
    iqn_dist_cols = []
    for slug in ["hold", "buy", "sell", "rebalance"]:
        for stat in ["action_score", "mean", "q10", "q25", "q50", "q75", "q90", "cvar"]:
            col = f"{stat}_{slug}"
            if col in df.columns:
                iqn_dist_cols.append(col)
    meta_cols = ["iqn_uncertainty_proxy", "iqn_risk_score"]

    available = set(df.columns)
    ordered = [c for c in core_cols + iqn_dist_cols + meta_cols if c in available]
    extra = [c for c in df.columns if c not in ordered]
    df = df[ordered + extra]

    return df


# ---------------------------------------------------------------------------
# Summary stats
# ---------------------------------------------------------------------------


def _build_summary(
    export_df: pd.DataFrame,
    source_run_id: str,
    run_dir: pathlib.Path,
    train_step: int,
    context: dict,
    out_dir: pathlib.Path,
) -> dict:
    has_quantiles = (
        "q10_hold" in export_df.columns and not export_df["q10_hold"].isna().all()
    )
    action_counts = export_df["iqn_selected_action"].value_counts().to_dict()
    date_valid = export_df["date"].notna().sum()

    summary = {
        "status": "ok",
        "run_id": f"{out_dir.name}",
        "source_iqn_run_id": source_run_id,
        "source_iqn_run_directory": str(run_dir),
        "train_step_exported": train_step,
        "eval_window_start": context.get("eval_window_start", ""),
        "eval_window_end": context.get("eval_window_end", ""),
        "iqn_tickers": context.get("tickers", []),
        "export_rows": len(export_df),
        "date_mapped_rows": int(date_valid),
        "date_range": {
            "start": str(export_df["date"].min()),
            "end": str(export_df["date"].max()),
        },
        "iqn_action_counts": action_counts,
        "hold_fraction": round(
            action_counts.get("HOLD", 0) / max(len(export_df), 1), 4
        ),
        "has_quantile_distributions": has_quantiles,
        "uncertainty_proxy_mean": (
            round(float(export_df["iqn_uncertainty_proxy"].mean()), 6)
            if "iqn_uncertainty_proxy" in export_df.columns
            else None
        ),
        "note_on_hold_collapse": (
            "WARNING: Final-trained model chose HOLD in "
            f"{action_counts.get('HOLD', 0)}/{len(export_df)} steps "
            f"({100 * action_counts.get('HOLD', 0) / max(len(export_df), 1):.1f}%). "
            "Use STOCK_INVESTMENT_DSS_IQN_EXPORT_TRAIN_STEP=0 for action-diverse early checkpoint."
            if action_counts.get("HOLD", 0) / max(len(export_df), 1) > 0.9
            else "Action distribution appears healthy."
        ),
        "integration_note": (
            "This export can be used by run_combined_iqn_hdp_audit_smoke_test.py. "
            "The combined runner's date range must overlap with this export's date range. "
            "Default combined window 2024-01-01 to 2024-02-01 overlaps with this export "
            "if eval_window covers Jan 2024."
        ),
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    return summary


def _build_md(summary: dict) -> str:
    lines = [
        "# IQN Decision Export — Smoke Test Summary",
        "",
        f"**Source run:** `{summary['source_iqn_run_id']}`  ",
        f"**Training checkpoint:** step `{summary['train_step_exported']}`  ",
        f"**Eval window:** {summary['eval_window_start']} → {summary['eval_window_end']}  ",
        f"**IQN tickers:** {', '.join(summary['iqn_tickers'])}  ",
        "",
        "## Export Statistics",
        "",
        f"- **Total rows:** {summary['export_rows']}",
        f"- **Date-mapped rows:** {summary['date_mapped_rows']}",
        f"- **Date range:** {summary['date_range']['start']} → {summary['date_range']['end']}",
        f"- **Has quantile distributions:** {summary['has_quantile_distributions']}",
        "",
        "## IQN Action Distribution",
        "",
    ]
    for action, cnt in sorted(summary["iqn_action_counts"].items()):
        pct = 100 * cnt / max(summary["export_rows"], 1)
        lines.append(f"- **{action}:** {cnt} ({pct:.1f}%)")
    lines += [
        "",
        "## Notes",
        "",
        f"> {summary['note_on_hold_collapse']}",
        "",
        f"_{summary['integration_note']}_",
        "",
        f"*Created: {summary['created_at']}*",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    log.info("=== IQN Decision Export Smoke Test ===")
    log.info("Project root: %s", PROJECT_ROOT)

    # --- Discover source run ---
    run_dir = _discover_source_run()
    source_run_id = run_dir.name

    # --- Load experiment context ---
    context = _load_experiment_context(run_dir)
    log.info(
        "Eval window: %s to %s | Tickers: %s",
        context.get("eval_window_start"),
        context.get("eval_window_end"),
        context.get("tickers", []),
    )

    # --- Determine train_step ---
    step_rec_path = run_dir / "data" / "iqn_learning_curve_eval_step_records.csv"
    if not step_rec_path.exists():
        raise FileNotFoundError(f"Step records not found: {step_rec_path}")

    all_steps = pd.read_csv(step_rec_path, usecols=["train_step"])[
        "train_step"
    ].unique()
    all_steps_sorted = sorted(all_steps)
    log.info("Available train_steps in step records: %s", all_steps_sorted)

    if TRAIN_STEP_OVERRIDE:
        try:
            train_step = int(TRAIN_STEP_OVERRIDE)
        except ValueError:
            raise ValueError(
                f"STOCK_INVESTMENT_DSS_IQN_EXPORT_TRAIN_STEP must be integer, got: {TRAIN_STEP_OVERRIDE!r}"
            )
        if train_step not in all_steps_sorted:
            raise ValueError(
                f"train_step={train_step} not in available steps: {all_steps_sorted}"
            )
    else:
        train_step = max(all_steps_sorted)

    log.info("Exporting train_step=%d", train_step)

    # --- Build eval date map ---
    date_map = _build_eval_date_map(context)

    # --- Load data ---
    step_df = _load_step_records(run_dir, train_step)
    dist_pivot_df = _load_and_pivot_distributions(run_dir, train_step)

    # --- Build export frame ---
    export_df = _build_export(
        step_df, dist_pivot_df, date_map, source_run_id, train_step
    )
    log.info("Export frame: %d rows × %d cols", len(export_df), len(export_df.columns))

    # --- Output directory ---
    ts = datetime.datetime.now().strftime("%Y_%m_%d_%H%M%S")
    out_dir = RUNS_DIR / f"{ts}_d_iqn_dss_iqn_decision_export_smoke_test"
    (out_dir / "audit").mkdir(parents=True, exist_ok=True)
    (out_dir / "summary").mkdir(parents=True, exist_ok=True)

    # --- Write export CSV ---
    export_path = out_dir / "audit" / "iqn_decision_export.csv"
    export_df.to_csv(export_path, index=False)
    log.info("Wrote %d rows to %s", len(export_df), export_path)

    # --- Build and write summary ---
    summary = _build_summary(
        export_df, source_run_id, run_dir, train_step, context, out_dir
    )
    summ_json_path = out_dir / "summary" / "iqn_decision_export_summary.json"
    summ_md_path = out_dir / "summary" / "iqn_decision_export_summary.md"

    with open(summ_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    log.info("Wrote summary JSON: %s", summ_json_path)

    summ_md = _build_md(summary)
    with open(summ_md_path, "w", encoding="utf-8") as f:
        f.write(summ_md)
    log.info("Wrote summary MD: %s", summ_md_path)

    # --- Final report ---
    log.info("")
    log.info("=== Export Complete ===")
    log.info("Output directory: %s", out_dir)
    log.info("Export CSV: %s (%d rows)", export_path.name, len(export_df))
    log.info("Train step exported: %d", train_step)
    log.info("Date range: %s to %s", export_df["date"].min(), export_df["date"].max())
    log.info("Action counts: %s", summary["iqn_action_counts"])
    log.info("Has quantile distributions: %s", summary["has_quantile_distributions"])
    log.info("")
    log.info("NOTE: %s", summary["note_on_hold_collapse"])
    log.info("")
    log.info("Next step: Pass this export to the combined audit runner:")
    log.info(
        '  $env:STOCK_INVESTMENT_DSS_COMBINED_IQN_DECISION_CSV="%s"',
        str(export_path),
    )
    log.info("  Then rerun run_combined_iqn_hdp_audit_smoke_test")

    return 0


if __name__ == "__main__":
    sys.exit(main())
