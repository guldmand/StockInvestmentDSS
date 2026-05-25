"""
run_edl_hindsight_labeling_smoke_test.py  (EDL v3.4)

Hindsight labeling smoke test runner.

Reads the combined IQN + HierarchicalDecisionPolicy audit CSV and assigns
EDL-A hindsight outcome labels based on future realized returns and drawdown
from frozen market data.

This is NOT a training runner.  It only produces labeled data for downstream
EDL-A supervised training.

Usage
-----
    $env:PYTHONPATH = "src"
    $env:STOCK_INVESTMENT_DSS_EDL_HINDSIGHT_SOURCE_COMBINED_RUN_ID = "<run_id>"
    python -m stock_investment_dss.runner.run_edl_hindsight_labeling_smoke_test

Environment variables
---------------------
    STOCK_INVESTMENT_DSS_EDL_HINDSIGHT_SOURCE_COMBINED_RUN_ID
        Combined IQN+HDP run ID (partial match OK).
        Default: auto-discover latest valid combined_iqn_hierarchical_smoke_test.

    STOCK_INVESTMENT_DSS_EDL_HINDSIGHT_HORIZON_DAYS         int    (default: 20)
    STOCK_INVESTMENT_DSS_EDL_HINDSIGHT_BUY_THRESHOLD        float  (default: 0.03)
    STOCK_INVESTMENT_DSS_EDL_HINDSIGHT_SELL_THRESHOLD       float  (default: -0.03)
    STOCK_INVESTMENT_DSS_EDL_HINDSIGHT_MAX_DRAWDOWN_THRESHOLD float (default: -0.08)
    STOCK_INVESTMENT_DSS_EDL_HINDSIGHT_DRAWDOWN_LAMBDA      float  (default: 0.5)
    STOCK_INVESTMENT_DSS_EDL_HINDSIGHT_INCLUDE_REBALANCE    true/false (default: false)

Market data
-----------
    data/market/daily/imports/market_data_full_500.csv
    Columns: date, tic, close, ...
    Must cover at least HORIZON_DAYS beyond the audit end date.

Output
------
    outputs/runs/<timestamp>_d_iqn_dss_edl_hindsight_labeling_smoke_test/
        data/combined_with_hindsight_labels.csv
        summary/edl_hindsight_labeling_summary.json
        summary/edl_hindsight_labeling_summary.md
"""

from __future__ import annotations

import csv
import json
import logging
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_COMBINED_AUDIT_FILENAME = "audit/combined_iqn_hierarchical_decision_by_step.csv"
_MARKET_DATA_PATH = Path("data/market/daily/imports/market_data_full_500.csv")
_RUNS_DIR = Path("outputs/runs")


# ---------------------------------------------------------------------------
# Env helpers
# ---------------------------------------------------------------------------


def _bool_env(key: str, default: bool) -> bool:
    v = os.environ.get(key, "").strip().lower()
    if v in ("1", "true", "yes"):
        return True
    if v in ("0", "false", "no"):
        return False
    return default


def _float_env(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, str(default)).strip())
    except (ValueError, TypeError):
        return default


def _int_env(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, str(default)).strip())
    except (ValueError, TypeError):
        return default


def _str_env(key: str, default: str) -> str:
    return os.environ.get(key, default).strip() or default


# ---------------------------------------------------------------------------
# Run discovery
# ---------------------------------------------------------------------------


def _find_latest_combined_run() -> Path:
    candidates = sorted(
        [
            d
            for d in _RUNS_DIR.iterdir()
            if d.is_dir()
            and "combined_iqn_hierarchical_smoke_test" in d.name
            and (d / _COMBINED_AUDIT_FILENAME).exists()
        ],
        key=lambda d: d.name,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No valid combined_iqn_hierarchical_smoke_test run found in {_RUNS_DIR}."
        )
    return candidates[0]


def _find_combined_run(run_id: str) -> Path:
    candidates = [
        d
        for d in _RUNS_DIR.iterdir()
        if d.is_dir() and run_id in d.name and (d / _COMBINED_AUDIT_FILENAME).exists()
    ]
    if not candidates:
        raise FileNotFoundError(
            f"Combined run '{run_id}' not found or has no audit CSV in {_RUNS_DIR}."
        )
    return sorted(candidates, key=lambda d: d.name, reverse=True)[0]


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------


def _mean_by_group(
    records: List[dict], label_col: str, value_col: str
) -> Dict[str, Optional[float]]:
    """Compute mean of value_col grouped by label_col (only available rows)."""
    sums: Dict[str, float] = defaultdict(float)
    counts: Dict[str, int] = defaultdict(int)
    for r in records:
        label = r.get(label_col, "")
        val_raw = r.get(value_col, "")
        if label and val_raw != "":
            try:
                sums[label] += float(val_raw)
                counts[label] += 1
            except (ValueError, TypeError):
                pass
    return {k: round(sums[k] / counts[k], 4) if counts[k] > 0 else None for k in sums}


def _label_dist_by_ticker(records: List[dict]) -> Dict[str, Dict[str, int]]:
    """Return {ticker: {label: count}} for labeled rows."""
    result: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in records:
        if not r.get("edl_a_label_available"):
            continue
        ticker = r.get("selected_ticker", "") or "NO_TICKER"
        label = r.get("edl_a_hindsight_label", "")
        if label:
            result[ticker][label] += 1
    return {t: dict(v) for t, v in result.items()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run() -> None:
    try:
        import pandas as pd  # type: ignore
    except ImportError:
        logger.error("pandas is required. Install with: pip install pandas")
        sys.exit(1)

    from stock_investment_dss.uncertainty.edl_hindsight_labeler import (
        HindsightLabelConfig,
        TickerPriceIndex,
        label_combined_audit,
    )

    # -- Resolve inputs -------------------------------------------------------
    combined_run_id = _str_env(
        "STOCK_INVESTMENT_DSS_EDL_HINDSIGHT_SOURCE_COMBINED_RUN_ID", ""
    )
    if combined_run_id:
        combined_run_dir = _find_combined_run(combined_run_id)
    else:
        combined_run_dir = _find_latest_combined_run()
    logger.info("Combined run: %s", combined_run_dir.name)

    if not _MARKET_DATA_PATH.exists():
        logger.error("Market data not found: %s", _MARKET_DATA_PATH)
        sys.exit(1)

    # -- Build config ---------------------------------------------------------
    cfg = HindsightLabelConfig(
        horizon_days=_int_env("STOCK_INVESTMENT_DSS_EDL_HINDSIGHT_HORIZON_DAYS", 20),
        buy_threshold=_float_env(
            "STOCK_INVESTMENT_DSS_EDL_HINDSIGHT_BUY_THRESHOLD", 0.03
        ),
        sell_threshold=_float_env(
            "STOCK_INVESTMENT_DSS_EDL_HINDSIGHT_SELL_THRESHOLD", -0.03
        ),
        max_drawdown_threshold=_float_env(
            "STOCK_INVESTMENT_DSS_EDL_HINDSIGHT_MAX_DRAWDOWN_THRESHOLD", -0.08
        ),
        drawdown_lambda=_float_env(
            "STOCK_INVESTMENT_DSS_EDL_HINDSIGHT_DRAWDOWN_LAMBDA", 0.5
        ),
        include_rebalance=_bool_env(
            "STOCK_INVESTMENT_DSS_EDL_HINDSIGHT_INCLUDE_REBALANCE", False
        ),
    )
    logger.info(
        "Config: h=%d days, buy>=%.2f%%, sell<=%.2f%%, max_dd>=%.2f%%, lambda=%.2f, include_rebalance=%s",
        cfg.horizon_days,
        cfg.buy_threshold * 100,
        cfg.sell_threshold * 100,
        cfg.max_drawdown_threshold * 100,
        cfg.drawdown_lambda,
        cfg.include_rebalance,
    )

    # -- Load market data -----------------------------------------------------
    logger.info("Loading market data: %s", _MARKET_DATA_PATH)
    mkt_df = pd.read_csv(str(_MARKET_DATA_PATH), usecols=["date", "tic", "close"])
    logger.info(
        "Market data: %d rows, %d tickers", len(mkt_df), mkt_df["tic"].nunique()
    )

    price_index = TickerPriceIndex.from_dataframe(mkt_df)

    # -- Load combined audit CSV ----------------------------------------------
    combined_csv = combined_run_dir / _COMBINED_AUDIT_FILENAME
    logger.info("Loading combined audit: %s", combined_csv)
    with open(combined_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    logger.info("Loaded %d rows", len(rows))

    # -- Apply hindsight labels -----------------------------------------------
    labeled_rows = label_combined_audit(rows, price_index, cfg)

    # -- Compute summary statistics -------------------------------------------
    total = len(labeled_rows)
    labeled = sum(1 for r in labeled_rows if r["edl_a_label_available"])
    unavailable = total - labeled

    label_dist = dict(
        Counter(
            r["edl_a_hindsight_label"]
            for r in labeled_rows
            if r["edl_a_label_available"] and r["edl_a_hindsight_label"]
        )
    )
    reason_dist = dict(Counter(r["edl_a_label_reason"] for r in labeled_rows))
    label_dist_by_ticker = _label_dist_by_ticker(labeled_rows)
    mean_return_by_label = _mean_by_group(
        labeled_rows, "edl_a_hindsight_label", "edl_a_future_return_pct"
    )
    mean_drawdown_by_label = _mean_by_group(
        labeled_rows, "edl_a_hindsight_label", "edl_a_future_max_drawdown_pct"
    )

    # Detect if this is likely a final eval period (warn if so)
    dates = [r.get("date", "") for r in rows if r.get("date")]
    min_date = min(dates) if dates else ""
    max_date = max(dates) if dates else ""

    # -- Create output directory ----------------------------------------------
    ts = datetime.now(timezone.utc).strftime("%Y_%m_%d_%H%M%S")
    run_dir = _RUNS_DIR / f"{ts}_d_iqn_dss_edl_hindsight_labeling_smoke_test"
    (run_dir / "data").mkdir(parents=True, exist_ok=True)
    (run_dir / "summary").mkdir(parents=True, exist_ok=True)

    # -- Write labeled CSV ----------------------------------------------------
    out_csv_path = run_dir / "data/combined_with_hindsight_labels.csv"
    if labeled_rows:
        fieldnames = list(labeled_rows[0].keys())
        with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(labeled_rows)
        logger.info("Wrote labeled CSV: %s (%d rows)", out_csv_path, total)

    # -- Write summary JSON ---------------------------------------------------
    summary = {
        "source_combined_run_id": combined_run_dir.name,
        "market_data_file": str(_MARKET_DATA_PATH),
        "horizon_days": cfg.horizon_days,
        "buy_threshold": cfg.buy_threshold,
        "sell_threshold": cfg.sell_threshold,
        "max_drawdown_threshold": cfg.max_drawdown_threshold,
        "drawdown_lambda": cfg.drawdown_lambda,
        "include_rebalance": cfg.include_rebalance,
        "date_range": {"min": min_date, "max": max_date},
        "total_rows": total,
        "labeled_rows": labeled,
        "unavailable_rows": unavailable,
        "label_distribution": label_dist,
        "label_reason_distribution": reason_dist,
        "label_distribution_by_ticker": label_dist_by_ticker,
        "mean_future_return_pct_by_label": mean_return_by_label,
        "mean_future_max_drawdown_pct_by_label": mean_drawdown_by_label,
        "output_csv": str(out_csv_path),
        "run_id": run_dir.name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "edl_a_caveat": (
            "EDL-A hindsight labels use future returns as supervised targets (y_t). "
            "They must NOT be used as input features. "
            "Do not tune on this dataset if it originates from the final PIT evaluation period."
        ),
    }

    summary_json_path = run_dir / "summary/edl_hindsight_labeling_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Wrote summary JSON: %s", summary_json_path)

    # -- Write summary MD -----------------------------------------------------
    def _dist_table(d: dict, col1: str = "Label") -> str:
        if not d:
            return "*(none)*"
        lines = [f"| {col1} | Count |", "|---|---|"]
        for k, v in sorted(d.items()):
            lines.append(f"| {k} | {v} |")
        return "\n".join(lines)

    ticker_table_lines = [
        "| Ticker | BUY | HOLD | SELL | REBALANCE |",
        "|---|---|---|---|---|",
    ]
    for tic in sorted(label_dist_by_ticker):
        d = label_dist_by_ticker[tic]
        ticker_table_lines.append(
            f"| {tic} | {d.get('BUY',0)} | {d.get('HOLD',0)} | {d.get('SELL',0)} | {d.get('REBALANCE',0)} |"
        )
    ticker_table = "\n".join(ticker_table_lines)

    return_table_lines = [
        "| Label | Mean Future Return (%) | Mean Max Drawdown (%) |",
        "|---|---|---|",
    ]
    for lbl in sorted(
        set(list(mean_return_by_label.keys()) + list(mean_drawdown_by_label.keys()))
    ):
        r = mean_return_by_label.get(lbl)
        d = mean_drawdown_by_label.get(lbl)
        return_table_lines.append(
            f"| {lbl} | {r:.2f} | {d:.2f} |"
            if r is not None and d is not None
            else f"| {lbl} | N/A | N/A |"
        )
    return_table = "\n".join(return_table_lines)

    md = f"""# EDL-A Hindsight Labeling Smoke Test Summary

**Run ID:** `{run_dir.name}`  
**Generated:** {datetime.now(timezone.utc).isoformat()}

---

## Source

| Field | Value |
|---|---|
| Combined IQN+HDP run | `{combined_run_dir.name}` |
| Market data | `{_MARKET_DATA_PATH}` |
| Date range | {min_date} to {max_date} |

---

## Label Configuration

| Parameter | Value |
|---|---|
| Horizon h | {cfg.horizon_days} trading days |
| BUY threshold | ≥ {cfg.buy_threshold*100:.1f}% return AND drawdown ≥ {cfg.max_drawdown_threshold*100:.1f}% |
| SELL threshold | ≤ {cfg.sell_threshold*100:.1f}% return OR drawdown < {cfg.max_drawdown_threshold*100:.1f}% |
| Drawdown λ | {cfg.drawdown_lambda} |
| REBALANCE included | {cfg.include_rebalance} |

---

## Row Counts

| Metric | Count |
|---|---|
| Total rows | {total} |
| Labeled (available) | {labeled} |
| Unavailable | {unavailable} |

---

## Label Distribution

{_dist_table(label_dist)}

---

## Label Reason Distribution

{_dist_table(reason_dist, 'Reason')}

---

## Label Distribution by Ticker

{ticker_table}

---

## Mean Future Outcomes by Label

{return_table}

---

## ⚠️ EDL-A Caveat

> EDL-A hindsight labels use **future returns as supervised targets** (`y_t`).  
> They must NOT be used as input features.  
> Do not tune on this dataset if it originates from the **final PIT evaluation period**.

---

## Output Files

- `data/combined_with_hindsight_labels.csv` — original audit rows + hindsight label columns
- `summary/edl_hindsight_labeling_summary.json` — machine-readable summary
- `summary/edl_hindsight_labeling_summary.md` — this file
"""

    summary_md_path = run_dir / "summary/edl_hindsight_labeling_summary.md"
    with open(summary_md_path, "w", encoding="utf-8") as f:
        f.write(md)
    logger.info("Wrote summary MD: %s", summary_md_path)

    # -- Console report -------------------------------------------------------
    logger.info("=" * 60)
    logger.info("EDL-A hindsight labeling complete")
    logger.info("Output dir  : %s", run_dir)
    logger.info("Total rows  : %d", total)
    logger.info("Labeled     : %d", labeled)
    logger.info("Unavailable : %d", unavailable)
    logger.info("Distribution: %s", label_dist)
    logger.info("By reason   : %s", reason_dist)

    # Print first 10 labeled rows
    logger.info("-" * 60)
    logger.info(
        "%-12s %-6s %-28s %-10s %-10s %-6s",
        "date",
        "ticker",
        "action_before_edl",
        "return%",
        "drawdown%",
        "label",
    )
    for r in labeled_rows[:10]:
        logger.info(
            "%-12s %-6s %-28s %-10s %-10s %-6s",
            r.get("date", ""),
            r.get("selected_ticker", "") or "—",
            r.get("final_recommendation_before_edl", ""),
            r.get("edl_a_future_return_pct", "—"),
            r.get("edl_a_future_max_drawdown_pct", "—"),
            r.get("edl_a_hindsight_label", "—"),
        )
    logger.info("=" * 60)


if __name__ == "__main__":
    run()
