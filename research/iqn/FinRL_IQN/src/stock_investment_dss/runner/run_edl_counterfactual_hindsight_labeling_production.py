"""Phase B.2 — EDL-A Counterfactual Hindsight Oracle Production Runner.

Reads Phase B.1's combined IQN+HDP audit CSV and applies the EDL-A
counterfactual hindsight oracle to assign per-row hindsight labels.

These labels feed Phase B.3 (EDL-A v2 training) and constitute the
thesis-defining evidence for academically-correct EDL methodology
(labels are exogenous to model inputs — computed from future price data only).

Usage::

    # All defaults (auto-discovers latest Phase B.1 run):
    python src/stock_investment_dss/runner/run_edl_counterfactual_hindsight_labeling_production.py

    # Explicit source run:
    python src/stock_investment_dss/runner/run_edl_counterfactual_hindsight_labeling_production.py \\
        --source-combined-run-id 2026_05_26_082024_d_iqn_dss_combined_iqn_hdp_audit_production

Output structure::

    outputs/runs/{timestamp}_d_iqn_dss_edl_counterfactual_oracle_production/
    ├── audit/
    │   └── combined_with_counterfactual_labels.csv   ← KEY (91 cols = 77 + 14 cf cols)
    ├── config/
    │   ├── oracle_config.json
    │   └── source_run_metadata.json
    ├── data/
    │   └── label_distribution_summary.csv
    ├── logs/run.log
    ├── metrics/oracle_summary.json
    ├── plots/
    │   ├── label_distribution.png
    │   ├── future_returns_by_label.png
    │   └── score_distributions.png
    └── summary/
        ├── oracle_summary.md
        └── oracle_summary.json

New columns added by oracle (14 edl_a_cf_* columns):
  edl_a_cf_label, edl_a_cf_label_id, edl_a_cf_label_available,
  edl_a_cf_label_reason, edl_a_cf_horizon_days,
  edl_a_cf_buy_score, edl_a_cf_sell_score, edl_a_cf_hold_score,
  edl_a_cf_best_score, edl_a_cf_second_best_score, edl_a_cf_margin,
  edl_a_cf_future_return_pct, edl_a_cf_future_max_drawdown_pct,
  edl_a_cf_risk_adjusted_future_score
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from stock_investment_dss.utilities.paths import (
    RUNS_DIRECTORY,
    create_run_paths,
)  # noqa: E402
from stock_investment_dss.utilities.logging import setup_run_logger  # noqa: E402
from stock_investment_dss.experiment_tracking.wandb_tracking import (  # noqa: E402
    finish_wandb_run,
    init_wandb_run,
    wandb_log,
)

# ---------------------------------------------------------------------------
# Phase B.1 auto-discovery
# ---------------------------------------------------------------------------


def find_latest_b1_run(runs_dir: Path) -> Path:
    """Find the latest Phase B.1 combined IQN+HDP audit production run.

    Glob: ``*_d_iqn_dss_combined_iqn_hdp_audit_production``  (sorted latest-first).
    Validates that ``audit/combined_iqn_hierarchical_decision_by_step.csv`` exists.

    Raises FileNotFoundError if no valid run is found.
    """
    candidates = sorted(
        runs_dir.glob("*_d_iqn_dss_combined_iqn_hdp_audit_production"),
        reverse=True,
    )
    for run_dir in candidates:
        audit_csv = run_dir / "audit" / "combined_iqn_hierarchical_decision_by_step.csv"
        if audit_csv.exists():
            return run_dir
    raise FileNotFoundError(
        "No Phase B.1 audit run found in outputs/runs/. "
        "Run Phase B.1 first:\n"
        "  python src/stock_investment_dss/runner/"
        "run_combined_iqn_hdp_audit_production.py"
    )


# ---------------------------------------------------------------------------
# Market data resolution
# ---------------------------------------------------------------------------


def _resolve_market_data(b1_run_dir: Path, logger: logging.Logger) -> Path:
    """Read the market data file path from the Phase B.1 run config.

    Reads ``{b1_run_dir}/config/production_config.json`` → ``data_file``.
    Falls back to the demo_10_new market data file if the config is absent.
    """
    config_path = b1_run_dir / "config" / "production_config.json"
    if config_path.exists():
        try:
            cfg = json.loads(config_path.read_text(encoding="utf-8"))
            data_file = cfg.get("data_file")
            if data_file:
                candidate = _PROJECT_ROOT / data_file
                if candidate.exists():
                    logger.info(
                        "Market data resolved from B.1 config: %s", candidate.name
                    )
                    return candidate
                logger.warning(
                    "B.1 config data_file not found at %s; falling back", candidate
                )
        except Exception as exc:
            logger.warning("Could not parse B.1 config: %s", exc)

    fallback = (
        _PROJECT_ROOT / "data/market/daily/imports/market_data_demo10_new_2010_2026.csv"
    )
    logger.warning("Using fallback market data: %s", fallback.name)
    return fallback


# ---------------------------------------------------------------------------
# Plot generation
# ---------------------------------------------------------------------------


def _generate_plots(
    out_df: pd.DataFrame,
    plots_dir: Path,
    logger: logging.Logger,
) -> None:
    """Generate 3 diagnostic plots to plots/."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available — plots skipped")
        return

    label_col = "edl_a_cf_label"

    # 1. Label distribution — pie chart
    try:
        counts = out_df[label_col].value_counts()
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.pie(
            counts.values,
            labels=counts.index.tolist(),
            autopct="%1.1f%%",
            startangle=90,
        )
        ax.set_title("EDL-A Counterfactual Label Distribution")
        plt.tight_layout()
        plt.savefig(plots_dir / "label_distribution.png", dpi=120)
        plt.close(fig)
    except Exception as exc:
        logger.warning("label_distribution plot failed: %s", exc)

    # 2. Future return % by label — boxplot
    try:
        ret_col = "edl_a_cf_future_return_pct"
        labeled = out_df[
            out_df["edl_a_cf_label_available"] == True
        ].copy()  # noqa: E712
        if ret_col in labeled.columns and not labeled.empty:
            label_groups = [
                (lbl, labeled[labeled[label_col] == lbl][ret_col].dropna().tolist())
                for lbl in ["HOLD", "BUY", "SELL"]
            ]
            present = [(lbl, grp) for lbl, grp in label_groups if grp]
            if present:
                names = [lbl for lbl, _ in present]
                data = [grp for _, grp in present]
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.boxplot(data)
                ax.set_xticks(range(1, len(names) + 1))
                ax.set_xticklabels(names)
                ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
                ax.set_title("Future Return % by Counterfactual Label")
                ax.set_xlabel("Label")
                ax.set_ylabel(
                    f"Future return % ({out_df['edl_a_cf_horizon_days'].iloc[0] if 'edl_a_cf_horizon_days' in out_df.columns else '?'}-day horizon)"
                )
                plt.tight_layout()
                plt.savefig(plots_dir / "future_returns_by_label.png", dpi=120)
                plt.close(fig)
    except Exception as exc:
        logger.warning("future_returns_by_label plot failed: %s", exc)

    # 3. Score distributions — overlapping histograms (BUY / SELL scores)
    try:
        fig, ax = plt.subplots(figsize=(9, 5))
        plotted = False
        for col, color, lbl in [
            ("edl_a_cf_buy_score", "steelblue", "BUY score"),
            ("edl_a_cf_sell_score", "tomato", "SELL score"),
        ]:
            if col in out_df.columns:
                vals = out_df[col].dropna()
                if not vals.empty:
                    ax.hist(vals, bins=20, alpha=0.6, color=color, label=lbl)
                    plotted = True
        if plotted:
            ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
            ax.set_title("Counterfactual Score Distributions (BUY / SELL)")
            ax.set_xlabel("Score")
            ax.set_ylabel("Frequency")
            ax.legend()
            plt.tight_layout()
            plt.savefig(plots_dir / "score_distributions.png", dpi=120)
        plt.close(fig)
    except Exception as exc:
        logger.warning("score_distributions plot failed: %s", exc)

    logger.info("Plots written to: %s", plots_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase B.2 — EDL-A Counterfactual Hindsight Oracle Production Runner.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--source-combined-run-id",
        default="auto",
        metavar="RUN_ID|auto",
        help="Phase B.1 audit run directory name, or 'auto' for latest.",
    )
    p.add_argument(
        "--horizon-days",
        type=int,
        default=20,
        metavar="N",
        help="Forward return horizon in trading days.",
    )
    p.add_argument(
        "--drawdown-lambda",
        type=float,
        default=0.5,
        metavar="LAMBDA",
        help="Drawdown penalty weight for BUY score computation.",
    )
    p.add_argument(
        "--min-label-margin",
        type=float,
        default=0.005,
        metavar="MARGIN",
        help="Minimum score margin for non-HOLD label assignment.",
    )
    p.add_argument(
        "--class-space",
        default="HOLD,BUY,SELL",
        metavar="CLASS,...",
        help="Comma-separated class names (K=3 for EDL-A v3.5).",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        metavar="PATH",
        help="Override output directory (optional).",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    args = parse_args()

    run_paths = create_run_paths("d_iqn_dss_edl_counterfactual_oracle_production")
    logger = setup_run_logger(run_paths)
    run_start = datetime.now()

    class_space = [c.strip().upper() for c in args.class_space.split(",")]

    logger.info("=== Phase B.2 EDL-A Counterfactual Hindsight Oracle Production ===")
    logger.info("Source run:      %s", args.source_combined_run_id)
    logger.info("Horizon:         %d days", args.horizon_days)
    logger.info("Drawdown lambda: %.3f", args.drawdown_lambda)
    logger.info("Min margin:      %.4f", args.min_label_margin)
    logger.info("Class space:     %s", class_space)
    logger.info("Run dir:         %s", run_paths.run_directory)

    # -----------------------------------------------------------------------
    # 1. Deferred oracle imports
    # -----------------------------------------------------------------------
    try:
        from stock_investment_dss.uncertainty.edl_counterfactual_hindsight_oracle import (
            CounterfactualConfig,
            TickerPriceIndex,
            build_summary,
            label_combined_audit,
        )
    except ImportError as exc:
        logger.error("Import failed — is PYTHONPATH set to 'src'? %s", exc)
        return 1

    # -----------------------------------------------------------------------
    # 2. Locate Phase B.1 audit run
    # -----------------------------------------------------------------------
    if args.source_combined_run_id == "auto":
        try:
            b1_run_dir = find_latest_b1_run(RUNS_DIRECTORY)
            logger.info("B.1 run (auto): %s", b1_run_dir.name)
        except FileNotFoundError as exc:
            logger.error("%s", exc)
            return 1
    else:
        b1_run_dir = RUNS_DIRECTORY / args.source_combined_run_id
        if not b1_run_dir.is_dir():
            logger.error("B.1 run directory not found: %s", b1_run_dir)
            return 1

    audit_csv = b1_run_dir / "audit" / "combined_iqn_hierarchical_decision_by_step.csv"
    if not audit_csv.exists():
        logger.error("B.1 audit CSV not found: %s", audit_csv)
        return 1

    # -----------------------------------------------------------------------
    # 3. Resolve market data (from B.1 config or fallback)
    # -----------------------------------------------------------------------
    market_path = _resolve_market_data(b1_run_dir, logger)
    if not market_path.exists():
        logger.error("Market data not found: %s", market_path)
        return 1

    # -----------------------------------------------------------------------
    # 4. Load data
    # -----------------------------------------------------------------------
    logger.info("Loading B.1 audit: %s", audit_csv.parent.parent.name)
    combined_df = pd.read_csv(audit_csv)
    logger.info("Audit: %d rows, %d cols", len(combined_df), len(combined_df.columns))

    logger.info("Loading market data: %s", market_path.name)
    market_df = pd.read_csv(market_path, low_memory=False, parse_dates=False)
    market_df["date"] = market_df["date"].astype(str).str[:10]
    logger.info(
        "Market data: %d rows, %d tickers",
        len(market_df),
        market_df["tic"].nunique(),
    )

    # -----------------------------------------------------------------------
    # 5. Write config files
    # -----------------------------------------------------------------------
    oracle_cfg_dict: dict = {
        "source_combined_run_id": b1_run_dir.name,
        "horizon_days": args.horizon_days,
        "drawdown_lambda": args.drawdown_lambda,
        "min_label_margin": args.min_label_margin,
        "class_space": class_space,
        "market_data_file": str(market_path),
        "run_start": run_start.isoformat(),
    }
    (run_paths.config_directory / "oracle_config.json").write_text(
        json.dumps(oracle_cfg_dict, indent=2), encoding="utf-8"
    )
    b1_config: dict = {}
    b1_config_path = b1_run_dir / "config" / "production_config.json"
    if b1_config_path.exists():
        try:
            b1_config = json.loads(b1_config_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    (run_paths.config_directory / "source_run_metadata.json").write_text(
        json.dumps(
            {
                "b1_run_id": b1_run_dir.name,
                "b1_audit_csv": str(audit_csv),
                "b1_input_rows": len(combined_df),
                "b1_input_cols": len(combined_df.columns),
                **{f"b1_{k}": v for k, v in b1_config.items() if k != "run_start"},
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    # -----------------------------------------------------------------------
    # 6. W&B setup
    # -----------------------------------------------------------------------
    init_wandb_run(
        run_name=run_paths.run_id,
        config=oracle_cfg_dict,
        group="phase-b2-oracle",
        job_type="counterfactual_labeling",
        tags=["phase-b", "edl-a", "counterfactual", "oracle"],
        run_directory=str(run_paths.run_directory),
    )

    # -----------------------------------------------------------------------
    # 7. Build price index and run oracle
    # -----------------------------------------------------------------------
    config = CounterfactualConfig(
        horizon_days=args.horizon_days,
        drawdown_lambda=args.drawdown_lambda,
        min_label_margin=args.min_label_margin,
        class_space=class_space,
    )
    logger.info("Building TickerPriceIndex (%d rows) ...", len(market_df))
    price_index = TickerPriceIndex(market_df)

    logger.info("Running label_combined_audit() on %d rows ...", len(combined_df))
    out_df = label_combined_audit(combined_df, price_index, config)
    new_cols = len(out_df.columns) - len(combined_df.columns)
    logger.info(
        "Oracle complete: %d rows, %d cols (+%d edl_a_cf_* cols)",
        len(out_df),
        len(out_df.columns),
        new_cols,
    )

    # -----------------------------------------------------------------------
    # 8. Write key output CSV
    # -----------------------------------------------------------------------
    key_csv = run_paths.audit_directory / "combined_with_counterfactual_labels.csv"
    out_df.to_csv(key_csv, index=False)
    logger.info(
        "Key CSV: %s (%d rows, %d cols)", key_csv.name, len(out_df), len(out_df.columns)
    )

    # -----------------------------------------------------------------------
    # 9. Build summary
    # -----------------------------------------------------------------------
    warnings_list: List[str] = []
    summary = build_summary(
        out_df=out_df,
        config=config,
        source_run_id=b1_run_dir.name,
        market_data_file=str(market_path),
        output_csv_path=str(key_csv),
        warnings=warnings_list,
    )
    total = summary.get("total_rows", len(out_df))
    labeled = summary.get("labeled_rows", 0)
    label_dist = summary.get("label_distribution", {})
    summary["edl_a_training_ready"] = labeled > 0 and total > 0
    summary["label_availability_pct"] = round(100.0 * labeled / max(total, 1), 2)
    summary["phase_b3_input_path"] = str(key_csv)
    summary["run_id"] = run_paths.run_id

    # -----------------------------------------------------------------------
    # 10. Write data/label_distribution_summary.csv
    # -----------------------------------------------------------------------
    unavail_count = summary.get("unavailable_rows", 0)
    full_dist = {**label_dist, "UNAVAILABLE": unavail_count}
    pd.DataFrame(
        [
            {"label": lbl, "count": cnt, "pct": round(100.0 * cnt / max(total, 1), 2)}
            for lbl, cnt in sorted(full_dist.items())
        ]
    ).to_csv(run_paths.data_directory / "label_distribution_summary.csv", index=False)

    # -----------------------------------------------------------------------
    # 11. Write metrics JSON
    # -----------------------------------------------------------------------
    (run_paths.metrics_directory / "oracle_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )

    # -----------------------------------------------------------------------
    # 12. Plots
    # -----------------------------------------------------------------------
    _generate_plots(out_df, run_paths.plots_directory, logger)

    # -----------------------------------------------------------------------
    # 13. Summary
    # -----------------------------------------------------------------------
    dur_sec = (datetime.now() - run_start).total_seconds()

    def _label_table(dist: dict, total_n: int) -> str:
        return "\n".join(
            f"| {lbl} | {cnt} | {100 * cnt / max(total_n, 1):.1f}% |"
            for lbl, cnt in sorted(dist.items())
        )

    mean_ret = summary.get("mean_future_return_pct_by_label", {})
    mean_dd = summary.get("mean_future_max_drawdown_pct_by_label", {})
    ret_rows = "\n".join(
        f"| {lbl} | {mean_ret.get(lbl, 'n/a')} | {mean_dd.get(lbl, 'n/a')} |"
        for lbl in ["HOLD", "BUY", "SELL"]
    )
    balance_note = (
        "balanced"
        if min((label_dist.get(k, 0) for k in ["HOLD", "BUY", "SELL"]), default=0) > 5
        else "imbalanced"
    )
    buy_pct = round(100.0 * label_dist.get("BUY", 0) / max(labeled, 1), 1)
    sell_pct = round(100.0 * label_dist.get("SELL", 0) / max(labeled, 1), 1)
    hold_pct = round(100.0 * label_dist.get("HOLD", 0) / max(labeled, 1), 1)

    md = (
        "# EDL-A Counterfactual Hindsight Oracle Production Run\n\n"
        f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}\n"
        f"Source: `{b1_run_dir.name}`\n\n"
        "## Configuration\n\n"
        f"- Horizon: {args.horizon_days} trading days\n"
        f"- Drawdown lambda: {args.drawdown_lambda}\n"
        f"- Min label margin: {args.min_label_margin}\n"
        f"- Class space: {class_space}\n\n"
        "## Label Distribution\n\n"
        "| Label | Count | % |\n"
        "|-------|-------|---|\n"
        f"{_label_table(full_dist, total)}\n\n"
        "## Statistics\n\n"
        f"- Total rows: {total}\n"
        f"- Labeled rows: {labeled}\n"
        f"- Label availability: {summary['label_availability_pct']}%\n"
        f"- Ambiguous labels (→ HOLD): {summary.get('ambiguous_rows', 0)}\n"
        f"- Duration: {int(dur_sec // 60)}m {int(dur_sec % 60)}s\n\n"
        "## Mean Return by Label\n\n"
        "| Label | Mean Return % | Mean Drawdown % |\n"
        "|-------|---------------|-----------------|\n"
        f"{ret_rows}\n\n"
        "## EDL-A Training Readiness\n\n"
        f"- Class balance: {balance_note}\n"
        f"- BUY / SELL / HOLD ratio: {buy_pct:.1f}% / {sell_pct:.1f}% / {hold_pct:.1f}%\n"
        f"- Output ready for Phase B.3: `{key_csv}`\n\n"
        "## Output Files\n\n"
        f"- Key CSV: `{key_csv}`\n"
        f"- Metrics: `{run_paths.metrics_directory / 'oracle_summary.json'}`\n"
        f"- Run dir: `{run_paths.run_directory}`\n"
    )
    (run_paths.summary_directory / "oracle_summary.md").write_text(md, encoding="utf-8")
    (run_paths.summary_directory / "oracle_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )

    # -----------------------------------------------------------------------
    # 14. W&B metrics
    # -----------------------------------------------------------------------
    n = max(total, 1)
    wandb_log(
        {
            "oracle_total_rows": total,
            "oracle_labeled_rows": labeled,
            "oracle_unavailable_rows": summary.get("unavailable_rows", 0),
            "oracle_label_availability_pct": summary["label_availability_pct"],
            "oracle_ambiguous_rows": summary.get("ambiguous_rows", 0),
            **{
                f"oracle_label_pct_{lbl.lower()}": cnt / n
                for lbl, cnt in label_dist.items()
            },
        }
    )

    logger.info("=== Phase B.2 oracle complete ===")
    logger.info("Run directory:     %s", run_paths.run_directory)
    logger.info(
        "Labeled:           %d / %d (%.1f%%)",
        labeled,
        total,
        summary["label_availability_pct"],
    )
    logger.info("Label dist:        %s", label_dist)
    logger.info("Audit columns:     %d", len(out_df.columns))
    logger.info("EDL-A ready:       %s", summary["edl_a_training_ready"])
    logger.info("Duration:          %.1f s", dur_sec)

    try:
        finish_wandb_run()
    except Exception:  # noqa: BLE001
        pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
