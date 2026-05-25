# src/stock_investment_dss/runner/run_combined_iqn_hierarchical_smoke_test.py
"""
Combined IQN + HierarchicalDecisionPolicy smoke test (D-IQN-DSS v3.3).

Loads the latest IQN learning curve run, enriches each eval step with
HierarchicalDecisionPolicy features, and writes a combined audit dataset
suitable for EDL-C (teacher label) training and end-to-end evidence.

Pipeline
--------
IQN learning curve eval data
  (eval_distributions.csv + eval_step_records.csv + experiment_context.json)
  ↓
CombinedIQNHierarchicalPolicy.build_combined_audit()
  ↓
outputs/runs/<timestamp>_d_iqn_dss_combined_iqn_hierarchical_smoke_test/
  audit/combined_iqn_hierarchical_decision_by_step.csv  ← main EDL input
  audit/combined_ticker_score_table.csv
  audit/combined_size_score_table.csv
  summary/combined_iqn_hierarchical_summary.json
  summary/combined_iqn_hierarchical_summary.md

Environment variables
---------------------
STOCK_INVESTMENT_DSS_COMBINED_IQN_SOURCE_RUN_ID   : run dir name (default: latest)
STOCK_INVESTMENT_DSS_COMBINED_IQN_TRAIN_STEP      : train step to use (default: last)
STOCK_INVESTMENT_DSS_COMBINED_IQN_STRATEGY        : balanced_v1|defensive_v1|aggressive_v1
STOCK_INVESTMENT_DSS_COMBINED_MARKET_DATA_FILE    : path to market_data_full_500.csv
STOCK_INVESTMENT_DSS_USE_HIERARCHICAL_POLICY      : true|false (default: true)
STOCK_INVESTMENT_DSS_USE_EDL                      : false (informational, logged only)
STOCK_INVESTMENT_DSS_EDL_VARIANT                  : none (informational, logged only)
STOCK_INVESTMENT_DSS_EDL_GATE_ENABLED             : false (informational, logged only)

Usage (from repo root)
------
    $env:PYTHONPATH = "src"
    python -m stock_investment_dss.runner.run_combined_iqn_hierarchical_smoke_test

    # To use a specific IQN run:
    $env:STOCK_INVESTMENT_DSS_COMBINED_IQN_SOURCE_RUN_ID = "2026_05_21_004044_d_iqn_dss_iqn_learning_curve_smoke_test"
    python -m stock_investment_dss.runner.run_combined_iqn_hierarchical_smoke_test
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default).strip()


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def _write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Wrote CSV: %s (%d rows)", path, len(df))


def _write_md(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    logger.info("Wrote MD: %s", path)


_DEFAULT_MARKET_DATA = "data/market/daily/imports/market_data_full_500.csv"


def _load_market_data(path_str: str, tickers: list) -> pd.DataFrame:
    p = Path(path_str)
    if not p.exists():
        raise FileNotFoundError(
            f"Market data file not found: {p.resolve()}\n"
            "Run Mode A data download or ensure the frozen import CSV exists:\n"
            "  data/market/daily/imports/market_data_full_500.csv"
        )
    df = pd.read_csv(p, low_memory=False)
    df.columns = df.columns.str.lower()
    df["date"] = pd.to_datetime(df["date"])
    tic_col = "tic" if "tic" in df.columns else "ticker"
    filtered = df[df[tic_col].isin(tickers)].copy()
    if filtered.empty:
        available = sorted(df[tic_col].unique().tolist())[:20]
        raise ValueError(
            f"No rows found for tickers {tickers} in market data.\n"
            f"Available tickers (first 20): {available}"
        )
    missing = [t for t in tickers if t not in filtered[tic_col].unique()]
    if missing:
        logger.warning("Missing tickers in market data: %s", missing)
    logger.info(
        "Loaded market data: %d rows for %s",
        len(filtered),
        sorted(filtered[tic_col].unique().tolist()),
    )
    return filtered


# ---------------------------------------------------------------------------
# Summary builders
# ---------------------------------------------------------------------------


def _build_summary_json(
    run_id: str,
    source_iqn_run_id: str,
    train_step: int,
    combined_df: pd.DataFrame,
    ticker_score_df: pd.DataFrame,
    size_score_df: pd.DataFrame,
    warnings: list,
    use_hierarchical: bool,
    use_edl: bool,
    combined_csv_path: Path,
) -> dict:
    iqn_dist_cols = [
        c
        for c in combined_df.columns
        if c.startswith("iqn_q") or c.startswith("iqn_cvar")
    ]
    n_dist_rows = (
        int((combined_df[iqn_dist_cols[0]].notna()).sum()) if iqn_dist_cols else 0
    )

    hdp_col = "hierarchical_action_type"
    n_hdp_rows = (
        int(combined_df[hdp_col].notna().sum()) if hdp_col in combined_df.columns else 0
    )

    iqn_action_counts = {}
    hdp_action_counts = {}
    selected_tickers = []

    if "selected_iqn_action" in combined_df.columns:
        iqn_action_counts = combined_df["selected_iqn_action"].value_counts().to_dict()
    if hdp_col in combined_df.columns:
        hdp_action_counts = combined_df[hdp_col].value_counts().to_dict()
    if "selected_ticker" in combined_df.columns:
        selected_tickers = sorted(
            combined_df["selected_ticker"].dropna().unique().tolist()
        )

    missing_cols = [
        c
        for c in ["iqn_q10_hold", "iqn_q50_buy", "iqn_score_sell", "iqn_action_margin"]
        if c not in combined_df.columns or combined_df[c].isna().all()
    ]

    return {
        "run_id": run_id,
        "source_iqn_run_id": source_iqn_run_id,
        "train_step_used": train_step,
        "n_decision_rows": len(combined_df),
        "n_rows_with_iqn_distribution": n_dist_rows,
        "n_rows_with_hierarchical_enrichment": n_hdp_rows,
        "tickers_selected": selected_tickers,
        "action_counts_before_hierarchy": iqn_action_counts,
        "action_counts_after_hierarchy": hdp_action_counts,
        "missing_column_warnings": missing_cols,
        "non_fatal_warnings": warnings,
        "use_hierarchical_policy": use_hierarchical,
        "use_edl": use_edl,
        "combined_audit_csv": str(combined_csv_path),
        "ticker_score_rows": len(ticker_score_df),
        "size_score_rows": len(size_score_df),
    }


def _build_summary_md(summary: dict) -> str:
    w_block = ""
    if summary["missing_column_warnings"]:
        w_block = (
            "\n### Missing / All-null IQN columns\n"
            + "\n".join(f"- `{c}`" for c in summary["missing_column_warnings"])
            + "\n"
        )
    if summary["non_fatal_warnings"]:
        w_block += "\n### Non-fatal Warnings\n"
        w_block += "\n".join(f"- {w}" for w in summary["non_fatal_warnings"][:20])
        if len(summary["non_fatal_warnings"]) > 20:
            w_block += f"\n- ... ({len(summary['non_fatal_warnings'])-20} more)"
        w_block += "\n"

    iqn_table = "\n".join(
        f"| {a} | {c} |" for a, c in summary["action_counts_before_hierarchy"].items()
    )
    hdp_table = "\n".join(
        f"| {a} | {c} |" for a, c in summary["action_counts_after_hierarchy"].items()
    )

    return f"""# Combined IQN + HierarchicalDecisionPolicy Smoke Test

Run ID: `{summary['run_id']}`

## Source IQN Run
- Run ID: `{summary['source_iqn_run_id']}`
- Train step used: `{summary['train_step_used']}`

## Coverage
| Metric | Value |
|--------|-------|
| Total decision rows | {summary['n_decision_rows']} |
| Rows with IQN distribution features | {summary['n_rows_with_iqn_distribution']} |
| Rows with HDP enrichment | {summary['n_rows_with_hierarchical_enrichment']} |
| Tickers selected | {', '.join(summary['tickers_selected']) or '—'} |

## IQN Action Counts (before HDP)
| Action | Count |
|--------|-------|
{iqn_table or '| — | 0 |'}

## HDP Action Counts (after hierarchy)
| Action | Count |
|--------|-------|
{hdp_table or '| — | 0 |'}

## Configuration
- `USE_HIERARCHICAL_POLICY`: `{summary['use_hierarchical_policy']}`
- `USE_EDL`: `{summary['use_edl']}` (disabled for this runner)

## Output CSV
`{summary['combined_audit_csv']}`

{w_block}

## EDL Label Status
- `edl_a_hindsight_label`: ⬜ blank (requires future-price pass)
- `edl_b_rule_label`: ⬜ blank (requires rule-labeler pass)
- `edl_c_teacher_label`: ✅ populated = HDP final action type

## Notes
- FundamentalFeatureStore has placeholder data only for AAPL/MSFT/NVDA/AMZN/GOOGL.
  The IQN run tickers (JPM/XOM/UNH/KO/WMT) will have zero/NaN fundamental scores.
  This is expected and documented.
- IQN run used eval distributions from the last (best) train checkpoint.
- Technical features (MA50/MA200/momentum etc.) are computed from market history
  including all data up to each decision date (point-in-time safe).
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    _setup_logging()

    # --- Read toggles ---
    source_run_id = _env("STOCK_INVESTMENT_DSS_COMBINED_IQN_SOURCE_RUN_ID", "")
    train_step_str = _env("STOCK_INVESTMENT_DSS_COMBINED_IQN_TRAIN_STEP", "")
    strategy_id = _env("STOCK_INVESTMENT_DSS_COMBINED_IQN_STRATEGY", "balanced_v1")
    market_data_file = _env(
        "STOCK_INVESTMENT_DSS_COMBINED_MARKET_DATA_FILE", _DEFAULT_MARKET_DATA
    )
    use_hierarchical = _env(
        "STOCK_INVESTMENT_DSS_USE_HIERARCHICAL_POLICY", "true"
    ).lower() not in ("false", "0", "no")

    # Informational toggles (EDL is not implemented in this runner)
    use_edl = _env("STOCK_INVESTMENT_DSS_USE_EDL", "false").lower() in (
        "true",
        "1",
        "yes",
    )
    edl_variant = _env("STOCK_INVESTMENT_DSS_EDL_VARIANT", "none")
    edl_gate_enabled = _env("STOCK_INVESTMENT_DSS_EDL_GATE_ENABLED", "false")

    logger.info("=== Combined IQN + HierarchicalDecisionPolicy Smoke Test (v3.3) ===")
    logger.info("Source IQN run ID: %s", source_run_id or "(auto: latest)")
    logger.info("Train step: %s", train_step_str or "(auto: last)")
    logger.info("Strategy: %s", strategy_id)
    logger.info("USE_HIERARCHICAL_POLICY: %s", use_hierarchical)
    logger.info("USE_EDL: %s (disabled for this runner)", use_edl)
    logger.info("EDL_VARIANT: %s", edl_variant)
    logger.info("EDL_GATE_ENABLED: %s", edl_gate_enabled)

    # --- Deferred imports ---
    try:
        from stock_investment_dss.decision.combined_iqn_hierarchical_policy import (
            CombinedIQNHierarchicalPolicy,
            IQNRunLoader,
        )
        from stock_investment_dss.utilities.paths import (
            create_run_paths,
            RUNS_DIRECTORY,
        )
    except ImportError as e:
        logger.error("Import error — is PYTHONPATH set to 'src'?\n  %s", e)
        return 1

    # --- Create run paths ---
    run_paths = create_run_paths("d_iqn_dss_combined_iqn_hierarchical_smoke_test")
    run_paths.run_directory.mkdir(parents=True, exist_ok=True)
    logger.info("Output run directory: %s", run_paths.run_directory)

    # --- Load IQN run ---
    try:
        loader = IQNRunLoader(runs_dir=RUNS_DIRECTORY)
        iqn_data = loader.load(source_run_id or None)
    except FileNotFoundError as e:
        logger.error("Could not load IQN run:\n%s", e)
        return 1

    logger.info(
        "IQN run loaded: %s | tickers: %s | eval: %s → %s | train_steps: %s",
        iqn_data.run_id,
        iqn_data.tickers,
        iqn_data.eval_window_start,
        iqn_data.eval_window_end,
        iqn_data.available_train_steps,
    )

    # --- Resolve train step ---
    train_step: int
    if train_step_str:
        train_step = int(train_step_str)
        if train_step not in iqn_data.available_train_steps:
            logger.error(
                "Requested train_step %d not in available: %s",
                train_step,
                iqn_data.available_train_steps,
            )
            return 1
    else:
        train_step = iqn_data.last_train_step
    logger.info("Using train_step: %d", train_step)

    # --- Load market data ---
    logger.info("Loading market data from: %s", market_data_file)
    try:
        market_df = _load_market_data(market_data_file, iqn_data.tickers)
    except (FileNotFoundError, ValueError) as e:
        logger.error("%s", e)
        return 1

    # --- Build combined policy ---
    policy = CombinedIQNHierarchicalPolicy(
        market_df=market_df,
        strategy_id=strategy_id,
        market_data_path=market_data_file,
    )

    # --- Run combined audit build ---
    logger.info("Building combined audit dataset ...")
    try:
        combined_df, ticker_score_df, size_score_df, warnings = (
            policy.build_combined_audit(
                iqn_data=iqn_data,
                train_step=train_step,
                use_hierarchical_policy=use_hierarchical,
            )
        )
    except Exception as exc:
        logger.exception("Combined audit build failed: %s", exc)
        return 1

    if combined_df.empty:
        logger.error(
            "Combined audit produced 0 rows. Check IQN run data and market data."
        )
        return 1

    logger.info(
        "Combined audit: %d rows, %d columns",
        len(combined_df),
        len(combined_df.columns),
    )

    # --- Write outputs ---
    combined_csv = (
        run_paths.audit_directory / "combined_iqn_hierarchical_decision_by_step.csv"
    )
    _write_csv(combined_csv, combined_df)

    if not ticker_score_df.empty:
        _write_csv(
            run_paths.audit_directory / "combined_ticker_score_table.csv",
            ticker_score_df,
        )
    else:
        logger.info("No ticker score rows (HOLD-only run or HDP disabled)")
        pd.DataFrame().to_csv(
            run_paths.audit_directory / "combined_ticker_score_table.csv", index=False
        )

    if not size_score_df.empty:
        _write_csv(
            run_paths.audit_directory / "combined_size_score_table.csv",
            size_score_df,
        )
    else:
        pd.DataFrame().to_csv(
            run_paths.audit_directory / "combined_size_score_table.csv", index=False
        )

    # --- Build and write summary ---
    summary = _build_summary_json(
        run_id=run_paths.run_id,
        source_iqn_run_id=iqn_data.run_id,
        train_step=train_step,
        combined_df=combined_df,
        ticker_score_df=ticker_score_df,
        size_score_df=size_score_df,
        warnings=warnings,
        use_hierarchical=use_hierarchical,
        use_edl=use_edl,
        combined_csv_path=combined_csv,
    )
    _write_json(
        run_paths.summary_directory / "combined_iqn_hierarchical_summary.json",
        summary,
    )
    _write_md(
        run_paths.summary_directory / "combined_iqn_hierarchical_summary.md",
        _build_summary_md(summary),
    )

    # --- Print preview ---
    preview_cols = [
        "date",
        "selected_iqn_action",
        "hierarchical_action_type",
        "selected_ticker",
        "selected_size",
        "cash_weight",
        "iqn_action_margin",
        "edl_c_teacher_label",
    ]
    available_preview = [c for c in preview_cols if c in combined_df.columns]
    logger.info("\n=== First 5 rows (key columns) ===")
    print(combined_df[available_preview].head(5).to_string(index=False))
    print()

    # --- Final summary ---
    logger.info("=== DONE ===")
    logger.info("Source IQN run: %s", summary["source_iqn_run_id"])
    logger.info("Decision rows: %d", summary["n_decision_rows"])
    logger.info("Rows with IQN dist: %d", summary["n_rows_with_iqn_distribution"])
    logger.info(
        "Rows with HDP enrichment: %d", summary["n_rows_with_hierarchical_enrichment"]
    )
    logger.info("Warnings: %d", len(warnings))
    logger.info("Output directory: %s", run_paths.run_directory)
    logger.info("Combined CSV: %s", combined_csv)

    for w in warnings[:10]:
        logger.warning("  %s", w)
    if len(warnings) > 10:
        logger.warning("  ... (%d more warnings)", len(warnings) - 10)

    return 0


if __name__ == "__main__":
    sys.exit(main())
