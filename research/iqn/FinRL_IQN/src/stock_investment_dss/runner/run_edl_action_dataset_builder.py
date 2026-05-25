#!/usr/bin/env python3
"""
run_edl_action_dataset_builder.py  (EDL v3.2)

Builds train/eval datasets for EDL action classification from frozen market data.
No model training. No GPU required.

Usage
-----
    python -m stock_investment_dss.runner.run_edl_action_dataset_builder

Environment variables
---------------------
    STOCK_INVESTMENT_DSS_EDL_LABEL_MODE     hindsight | rules | iqn_teacher  (default: rules)
    STOCK_INVESTMENT_DSS_EDL_INCLUDE_CHANGE_STRATEGY  true | false  (default: false)
    STOCK_INVESTMENT_DSS_EDL_HORIZON_DAYS   int  (default: 20)
    EDL_TICKERS                             comma-separated  (default: AAPL,MSFT,NVDA,AMZN,GOOGL)
    EDL_TRAIN_START                         YYYY-MM-DD  (default: 2018-01-01)
    EDL_TRAIN_END                           YYYY-MM-DD  (default: 2022-12-31)
    EDL_EVAL_START                          YYYY-MM-DD  (default: 2023-01-01)
    EDL_EVAL_END                            YYYY-MM-DD  (default: 2024-02-01)
    EDL_MARKET_CSV                          path to market_data_full_500.csv
    EDL_HIERARCHICAL_AUDIT_CSV              optional path to hierarchical decision CSV

Output
------
    outputs/runs/<timestamp>_d_iqn_dss_edl_action_dataset_builder/
        data/edl_action_train_dataset.csv
        data/edl_action_eval_dataset.csv
        summary/edl_action_dataset_summary.json
        summary/edl_action_dataset_summary.md
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure src is on path
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from stock_investment_dss.uncertainty.edl_action_classes import EDLActionConfig
from stock_investment_dss.uncertainty.edl_action_dataset import (
    EDLActionDataset,
    check_label_distribution,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("edl_action_dataset_builder")


def _get_env(key: str, default: str) -> str:
    return os.environ.get(key, default).strip()


def main() -> None:
    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------
    edl_config = EDLActionConfig.from_env()

    market_csv = _get_env(
        "EDL_MARKET_CSV",
        str(
            _REPO_ROOT
            / "data"
            / "market"
            / "daily"
            / "imports"
            / "market_data_full_500.csv"
        ),
    )
    tickers_str = _get_env("EDL_TICKERS", "AAPL,MSFT,NVDA,AMZN,GOOGL")
    tickers = [t.strip().upper() for t in tickers_str.split(",") if t.strip()]

    train_start = _get_env("EDL_TRAIN_START", "2018-01-01")
    train_end = _get_env("EDL_TRAIN_END", "2022-12-31")
    eval_start = _get_env("EDL_EVAL_START", "2023-01-01")
    eval_end = _get_env("EDL_EVAL_END", "2024-02-01")

    hier_audit_csv = _get_env("EDL_HIERARCHICAL_AUDIT_CSV", "") or None

    logger.info("=== EDL Action Dataset Builder v3.2 ===")
    logger.info(
        "label_mode=%s  tickers=%s  horizon_days=%d",
        edl_config.label_mode,
        tickers,
        edl_config.horizon_days,
    )
    logger.info(
        "train: %s → %s   eval: %s → %s", train_start, train_end, eval_start, eval_end
    )
    logger.info("market_csv: %s", market_csv)

    if not Path(market_csv).exists():
        logger.error("Market CSV not found: %s", market_csv)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Output directory
    # ------------------------------------------------------------------
    ts = datetime.now(timezone.utc).strftime("%Y_%m_%d_%H%M%S")
    run_name = f"{ts}_d_iqn_dss_edl_action_dataset_builder"
    out_dir = _REPO_ROOT / "outputs" / "runs" / run_name
    data_dir = out_dir / "data"
    summary_dir = out_dir / "summary"
    data_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Output directory: %s", out_dir)

    # ------------------------------------------------------------------
    # Build datasets
    # ------------------------------------------------------------------
    dataset = EDLActionDataset(config=edl_config)
    train_df, eval_df = dataset.build_from_market_data(
        market_csv_path=market_csv,
        tickers=tickers,
        train_start=train_start,
        train_end=train_end,
        eval_start=eval_start,
        eval_end=eval_end,
        hierarchical_audit_csv=hier_audit_csv,
    )

    # ------------------------------------------------------------------
    # Write CSVs
    # ------------------------------------------------------------------
    train_path = data_dir / "edl_action_train_dataset.csv"
    eval_path = data_dir / "edl_action_eval_dataset.csv"
    train_df.to_csv(train_path, index=False)
    eval_df.to_csv(eval_path, index=False)
    logger.info("Wrote train dataset: %s  (%d rows)", train_path, len(train_df))
    logger.info("Wrote eval  dataset: %s  (%d rows)", eval_path, len(eval_df))

    # ------------------------------------------------------------------
    # Label distribution summary
    # ------------------------------------------------------------------
    def label_dist(df) -> dict:
        if "label_str" not in df.columns or df.empty:
            return {}
        vc = df["label_str"].value_counts()
        total = len(df)
        return {
            str(k): {"count": int(v), "pct": round(100 * v / total, 1)}
            for k, v in vc.items()
        }

    train_label_dist = label_dist(train_df)
    eval_label_dist = label_dist(eval_df)

    # ------------------------------------------------------------------
    # Label sanity checks
    # ------------------------------------------------------------------
    train_sanity = check_label_distribution(train_df, split_name="train")
    eval_sanity = check_label_distribution(eval_df, split_name="eval")

    all_warnings = train_sanity["warnings"] + eval_sanity["warnings"]
    for w in all_warnings:
        if w.startswith("CRITICAL"):
            logger.error("LABEL SANITY: %s", w)
        else:
            logger.warning("LABEL SANITY: %s", w)

    label_quality_ok = (
        train_sanity["label_quality_ok"] and eval_sanity["label_quality_ok"]
    )
    if not label_quality_ok:
        logger.error(
            "LABEL SANITY FAILED — dataset has degenerate labels. "
            "Do NOT train on this dataset. Review label_mode or thresholds."
        )

    feature_names = dataset.feature_names()
    summary: dict = {
        "builder_version": "3.2",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "label_mode": edl_config.label_mode,
        "tickers": tickers,
        "train_start": train_start,
        "train_end": train_end,
        "eval_start": eval_start,
        "eval_end": eval_end,
        "horizon_days": edl_config.horizon_days,
        "action_classes": edl_config.action_classes,
        "num_classes": edl_config.num_classes,
        "num_features": len(feature_names),
        "feature_names": feature_names,
        "train_rows": len(train_df),
        "eval_rows": len(eval_df),
        "train_label_distribution": train_label_dist,
        "eval_label_distribution": eval_label_dist,
        "label_sanity": {
            "train": train_sanity,
            "eval": eval_sanity,
            "all_warnings": all_warnings,
            "label_quality_ok": label_quality_ok,
        },
        "market_csv": market_csv,
        "hierarchical_audit_csv": hier_audit_csv or "",
        "train_csv": str(train_path),
        "eval_csv": str(eval_path),
    }

    # Write JSON
    json_path = summary_dir / "edl_action_dataset_summary.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    logger.info("Wrote summary JSON: %s", json_path)

    # Write Markdown
    md_path = summary_dir / "edl_action_dataset_summary.md"
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(f"# EDL Action Dataset Summary (v3.2)\n\n")
        fh.write(f"**Generated:** {summary['timestamp_utc']}\n\n")
        fh.write(f"**Label mode:** `{edl_config.label_mode}`  ")
        fh.write(f"**Action classes:** `{', '.join(edl_config.action_classes)}`\n\n")
        fh.write(f"**Tickers:** `{', '.join(tickers)}`\n\n")
        fh.write("| Split | Start | End | Rows |\n")
        fh.write("|-------|-------|-----|------|\n")
        fh.write(f"| Train | {train_start} | {train_end} | {len(train_df)} |\n")
        fh.write(f"| Eval  | {eval_start}  | {eval_end}  | {len(eval_df)}  |\n")
        fh.write("\n## Train label distribution\n\n")
        for k, v in train_label_dist.items():
            fh.write(f"- **{k}**: {v['count']} ({v['pct']}%)\n")
        fh.write("\n## Eval label distribution\n\n")
        for k, v in eval_label_dist.items():
            fh.write(f"- **{k}**: {v['count']} ({v['pct']}%)\n")
        if all_warnings:
            fh.write("\n## ⚠️ Label sanity warnings\n\n")
            for w in all_warnings:
                fh.write(f"- {w}\n")
            fh.write(
                f"\n**Label quality OK:** `{'YES' if label_quality_ok else 'NO — do NOT train'}`\n"
            )
        fh.write(
            f"\n**Features ({len(feature_names)}):** `{', '.join(feature_names)}`\n"
        )
    logger.info("Wrote summary MD: %s", md_path)

    logger.info("=== Dataset builder complete. NO training was run. ===")


if __name__ == "__main__":
    main()
