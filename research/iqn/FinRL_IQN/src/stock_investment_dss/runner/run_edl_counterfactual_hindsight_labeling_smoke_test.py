"""
Runner: EDL-A Counterfactual Hindsight Labeling Smoke Test (v3.5)

Reads combined IQN+HDP audit CSV and frozen market data, runs the
counterfactual oracle, and writes labeled outputs.

Usage:
    $env:PYTHONPATH="src"
    python -m stock_investment_dss.runner.run_edl_counterfactual_hindsight_labeling_smoke_test

Env vars:
    STOCK_INVESTMENT_DSS_EDL_COUNTERFACTUAL_SOURCE_COMBINED_RUN_ID   (optional, auto-detected)
    STOCK_INVESTMENT_DSS_EDL_COUNTERFACTUAL_HORIZON_DAYS             (default: 20)
    STOCK_INVESTMENT_DSS_EDL_COUNTERFACTUAL_DRAWDOWN_LAMBDA          (default: 0.5)
    STOCK_INVESTMENT_DSS_EDL_COUNTERFACTUAL_MIN_LABEL_MARGIN         (default: 0.005)
    STOCK_INVESTMENT_DSS_EDL_COUNTERFACTUAL_CLASS_SPACE              (default: HOLD,BUY,SELL)
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUTS_DIR = REPO_ROOT / "outputs" / "runs"
MARKET_DATA_PATH = (
    REPO_ROOT / "data" / "market" / "daily" / "imports" / "market_data_full_500.csv"
)

COMBINED_AUDIT_GLOB = "*combined_iqn_hierarchical*"


def _find_combined_run(run_id: str | None) -> Path:
    if run_id:
        candidate = OUTPUTS_DIR / run_id
        if candidate.is_dir():
            return candidate
        raise FileNotFoundError(f"Combined run dir not found: {candidate}")

    dirs = sorted(OUTPUTS_DIR.glob(COMBINED_AUDIT_GLOB))
    if not dirs:
        raise FileNotFoundError(f"No combined run dir found under {OUTPUTS_DIR}")
    chosen = dirs[-1]
    logger.info("Auto-selected combined run: %s", chosen.name)
    return chosen


def _build_run_dir() -> Path:
    ts = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    run_dir = (
        OUTPUTS_DIR / f"{ts}_d_iqn_dss_edl_counterfactual_hindsight_labeling_smoke_test"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "data").mkdir(exist_ok=True)
    (run_dir / "summary").mkdir(exist_ok=True)
    return run_dir


def main() -> None:
    from stock_investment_dss.uncertainty.edl_counterfactual_hindsight_oracle import (
        CounterfactualConfig,
        TickerPriceIndex,
        build_summary,
        label_combined_audit,
    )

    # --- Config from env ---------------------------------------------------
    source_run_id = os.environ.get(
        "STOCK_INVESTMENT_DSS_EDL_COUNTERFACTUAL_SOURCE_COMBINED_RUN_ID"
    )
    horizon_days = int(
        os.environ.get("STOCK_INVESTMENT_DSS_EDL_COUNTERFACTUAL_HORIZON_DAYS", "20")
    )
    drawdown_lambda = float(
        os.environ.get("STOCK_INVESTMENT_DSS_EDL_COUNTERFACTUAL_DRAWDOWN_LAMBDA", "0.5")
    )
    min_label_margin = float(
        os.environ.get(
            "STOCK_INVESTMENT_DSS_EDL_COUNTERFACTUAL_MIN_LABEL_MARGIN", "0.005"
        )
    )
    class_space_str = os.environ.get(
        "STOCK_INVESTMENT_DSS_EDL_COUNTERFACTUAL_CLASS_SPACE", "HOLD,BUY,SELL"
    )
    class_space = [c.strip().upper() for c in class_space_str.split(",")]

    config = CounterfactualConfig(
        horizon_days=horizon_days,
        drawdown_lambda=drawdown_lambda,
        min_label_margin=min_label_margin,
        class_space=class_space,
    )
    logger.info("Config: %s", config)

    # --- Locate inputs -----------------------------------------------------
    combined_run_dir = _find_combined_run(source_run_id)
    audit_csv = (
        combined_run_dir / "audit" / "combined_iqn_hierarchical_decision_by_step.csv"
    )
    if not audit_csv.is_file():
        raise FileNotFoundError(f"Combined audit CSV not found: {audit_csv}")

    if not MARKET_DATA_PATH.is_file():
        raise FileNotFoundError(f"Market data not found: {MARKET_DATA_PATH}")

    # --- Load data ---------------------------------------------------------
    logger.info("Loading combined audit: %s", audit_csv)
    combined_df = pd.read_csv(audit_csv)
    logger.info(
        "Combined audit: %d rows, %d cols", len(combined_df), len(combined_df.columns)
    )

    logger.info("Loading market data: %s", MARKET_DATA_PATH)
    market_df = pd.read_csv(MARKET_DATA_PATH, parse_dates=False)
    market_df["date"] = market_df["date"].astype(str).str[:10]
    logger.info(
        "Market data: %d rows, %d tickers", len(market_df), market_df["tic"].nunique()
    )

    # --- Build index and label --------------------------------------------
    price_index = TickerPriceIndex(market_df)
    logger.info("Running counterfactual oracle ...")
    out_df = label_combined_audit(combined_df, price_index, config)

    # --- Prepare output run dir -------------------------------------------
    run_dir = _build_run_dir()
    logger.info("Output run dir: %s", run_dir)

    # --- Write labeled CSV ------------------------------------------------
    out_csv = run_dir / "data" / "combined_with_counterfactual_hindsight_labels.csv"
    out_df.to_csv(out_csv, index=False)
    logger.info("Labeled CSV written: %s (%d rows)", out_csv, len(out_df))

    # --- Build and write summary ------------------------------------------
    warnings: list = []
    if (
        "eval" in combined_run_dir.name.lower()
        or "pit_eval" in combined_run_dir.name.lower()
    ):
        warnings.append(
            "WARNING: Source combined run appears to be from the final PIT evaluation period. "
            "Do not use this labeled file for tuning."
        )

    summary = build_summary(
        out_df=out_df,
        config=config,
        source_run_id=str(combined_run_dir.name),
        market_data_file=str(MARKET_DATA_PATH),
        output_csv_path=str(out_csv),
        warnings=warnings,
    )

    summary_json = run_dir / "summary" / "edl_counterfactual_hindsight_summary.json"
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Summary JSON: %s", summary_json)

    summary_md = run_dir / "summary" / "edl_counterfactual_hindsight_summary.md"
    _write_summary_md(summary, summary_md)
    logger.info("Summary MD: %s", summary_md)

    # --- Print report ------------------------------------------------------
    print("\n" + "=" * 60)
    print("EDL-A Counterfactual Hindsight Labeling — Smoke Test")
    print("=" * 60)
    print(f"Source run:       {combined_run_dir.name}")
    print(f"Total rows:       {summary['total_rows']}")
    print(f"Labeled rows:     {summary['labeled_rows']}")
    print(f"Unavailable rows: {summary['unavailable_rows']}")
    print(f"Ambiguous rows:   {summary['ambiguous_rows']}")
    print(f"Label distribution: {summary['label_distribution']}")
    print(f"Output dir:       {run_dir}")
    print()
    print("First 10 labeled rows:")
    preview_cols = [
        "date",
        "selected_ticker",
        "final_recommendation_before_edl",
        "edl_a_cf_future_return_pct",
        "edl_a_cf_future_max_drawdown_pct",
        "edl_a_cf_label",
        "edl_a_cf_margin",
        "edl_a_cf_label_reason",
    ]
    available_preview = [c for c in preview_cols if c in out_df.columns]
    labeled_rows = out_df[out_df["edl_a_cf_label_available"] == True]  # noqa: E712
    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        200,
        "display.float_format",
        "{:.4f}".format,
    ):
        print(labeled_rows[available_preview].head(10).to_string(index=False))
    print()

    if warnings:
        for w in warnings:
            print(f"[WARNING] {w}")
        print()

    print("Smoke test PASSED.")


def _write_summary_md(summary: dict, path: Path) -> None:
    lines = [
        "# EDL-A Counterfactual Hindsight Labeling Summary",
        "",
        f"**Source run:** `{summary['source_combined_run_id']}`  ",
        f"**Market data:** `{summary['market_data_file']}`  ",
        f"**Class space:** {summary['class_space']}  ",
        f"**Horizon:** {summary['horizon_days']} trading days  ",
        f"**Drawdown lambda:** {summary['parameters']['drawdown_lambda']}  ",
        f"**Min label margin:** {summary['parameters']['min_label_margin']}  ",
        "",
        "## Row Counts",
        "",
        f"| Metric | Count |",
        f"|--------|-------|",
        f"| Total rows | {summary['total_rows']} |",
        f"| Labeled rows | {summary['labeled_rows']} |",
        f"| Unavailable rows | {summary['unavailable_rows']} |",
        f"| Ambiguous (→ HOLD) | {summary['ambiguous_rows']} |",
        "",
        "## Label Distribution",
        "",
        "| Label | Count |",
        "|-------|-------|",
    ]
    for lbl, cnt in sorted(summary["label_distribution"].items()):
        lines.append(f"| {lbl} | {cnt} |")

    lines += [
        "",
        "## Mean Future Return by Label",
        "",
        "| Label | Mean Return % | Mean Drawdown % |",
        "|-------|:-------------:|:---------------:|",
    ]
    for lbl in ["HOLD", "BUY", "SELL"]:
        ret = summary["mean_future_return_pct_by_label"].get(lbl, "N/A")
        dd = summary["mean_future_max_drawdown_pct_by_label"].get(lbl, "N/A")
        lines.append(f"| {lbl} | {ret} | {dd} |")

    lines += ["", "## Action Before EDL vs CF Label (Confusion)", ""]
    confusion = summary.get("confusion_action_before_edl_vs_cf_label", {})
    if confusion:
        all_cf_labels = sorted({lbl for row in confusion.values() for lbl in row})
        lines.append("| Action \\ CF Label | " + " | ".join(all_cf_labels) + " |")
        lines.append("|" + "---|" * (len(all_cf_labels) + 1))
        for act, row in sorted(confusion.items()):
            vals = " | ".join(str(row.get(lbl, 0)) for lbl in all_cf_labels)
            lines.append(f"| {act} | {vals} |")

    if summary.get("warnings"):
        lines += ["", "## Warnings", ""]
        for w in summary["warnings"]:
            lines.append(f"- {w}")

    lines += ["", f"> {summary['caveat']}", ""]

    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logger.exception("Smoke test failed: %s", exc)
        sys.exit(1)
