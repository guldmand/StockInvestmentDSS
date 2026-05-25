"""
HDP PIT Information Inspection Runner.

Loads an existing hdp_joined_features.csv from a prior fmp_hdp_feature_smoke_test run
and produces a focused audit of point-in-time correctness for selected tickers
and a date range.

Purpose:
  Verify that FMP-derived fundamentals are genuinely point-in-time — i.e. that
  known_at, fundamental values, valuation features and composite scores vary
  historically, and are NOT just current "now" values repeated back in time.

Environment variables
---------------------
STOCK_INVESTMENT_DSS_HDP_INFO_SOURCE_RUN_ID   default: latest fmp_hdp_feature_smoke_test run
STOCK_INVESTMENT_DSS_HDP_INFO_TICKERS         default: AAPL,MSFT
STOCK_INVESTMENT_DSS_HDP_INFO_START_DATE      default: 2024-01-01
STOCK_INVESTMENT_DSS_HDP_INFO_END_DATE        default: 2024-02-01
STOCK_INVESTMENT_DSS_HDP_INFO_SAMPLE_MODE     default: first_last_monthly
    Supported: all | first_last | first_last_monthly | month_end | quarter_end
STOCK_INVESTMENT_DSS_HDP_INFO_MAX_ROWS        default: 200
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_RUNS_DIR = _REPO_ROOT / "outputs" / "runs"

# Ordered display columns (show only those present in the CSV)
_DISPLAY_COLS = [
    "ticker",
    "date",
    "known_at",
    "known_at_effective_date",
    "fundamental_lag_days",
    "close",
    "ma50",
    "ma200",
    "sma50",
    "sma200",
    "price_vs_ma50",
    "price_vs_ma200",
    "recent_return_20d",
    "volatility_20d",
    "drawdown_from_recent_high",
    "momentum_score",
    "technical_risk_score",
    "revenue",
    "revenue_growth",
    "earnings_growth",
    "gross_margin",
    "operating_margin",
    "profit_margin",
    "roe",
    "current_ratio",
    "debt_ratio",
    "free_cash_flow",
    "fcf_margin",
    "pe_ratio",
    "ps_ratio",
    "ev_ebitda",
    "fcf_yield",
    "annualized_revenue",
    "annualized_net_income",
    "annualized_free_cash_flow",
    "market_cap_estimate",
    "valuation_method",
    "valuation_warning",
    "value_score",
    "quality_score",
    "profitability_score",
    "balance_sheet_strength_score",
    "ticker_score",
    "point_in_time_quality",
    "feature_warning",
]

_SCORE_COLS = [
    "value_score",
    "quality_score",
    "profitability_score",
    "balance_sheet_strength_score",
    "ticker_score",
]

_VALUATION_COLS = ["pe_ratio", "ps_ratio", "fcf_yield", "value_score", "ticker_score"]


def _env_str(key: str, default: str) -> str:
    return os.environ.get(key, default).strip()


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, str(default)).strip())
    except ValueError:
        return default


def _find_latest_source_run() -> Optional[str]:
    """Return the name of the most-recent fmp_hdp_feature_smoke_test run."""
    candidates = sorted(
        [
            d.name
            for d in _RUNS_DIR.iterdir()
            if d.is_dir() and "fmp_hdp_feature_smoke_test" in d.name
        ]
    )
    return candidates[-1] if candidates else None


def _load_hdp_features(source_run: str) -> pd.DataFrame:
    path = _RUNS_DIR / source_run / "data" / "hdp_joined_features.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"hdp_joined_features.csv not found in run: {source_run}\n"
            f"Expected path: {path}\n"
            "Run run_fmp_hdp_feature_smoke_test first (with live FMP data)."
        )
    df = pd.read_csv(path, low_memory=False)
    df["date"] = df["date"].astype(str).str[:10]
    if "known_at" in df.columns:
        df["known_at"] = (
            df["known_at"].astype(str).str[:10].replace({"nan": "", "None": ""})
        )
    if "known_at_effective_date" in df.columns:
        df["known_at_effective_date"] = (
            df["known_at_effective_date"]
            .astype(str)
            .str[:10]
            .replace({"nan": "", "None": ""})
        )
    return df


def _sample(df: pd.DataFrame, mode: str, max_rows: int) -> pd.DataFrame:
    """Apply sampling mode per ticker, then cap total rows."""
    if mode == "all":
        sampled = df
    elif mode == "first_last":
        parts = []
        for ticker, g in df.groupby("ticker"):
            g = g.sort_values("date")
            parts.append(g.iloc[[0, -1]] if len(g) > 1 else g)
        sampled = pd.concat(parts) if parts else df
    elif mode == "first_last_monthly":
        parts = []
        for ticker, g in df.groupby("ticker"):
            g = g.sort_values("date")
            g["_ym"] = g["date"].str[:7]
            for _ym, mg in g.groupby("_ym"):
                parts.append(mg.iloc[[0]])
                if len(mg) > 1:
                    parts.append(mg.iloc[[-1]])
        sampled = (
            pd.concat(parts).drop(columns=["_ym"], errors="ignore") if parts else df
        )
    elif mode == "month_end":
        df2 = df.copy()
        df2["_ym"] = df2["date"].str[:7]
        sampled = (
            df2.sort_values("date").groupby(["ticker", "_ym"]).last().reset_index()
        )
        sampled = sampled.drop(columns=["_ym"], errors="ignore")
    elif mode == "quarter_end":
        df2 = df.copy()
        df2["_dt"] = pd.to_datetime(df2["date"])
        df2["_yq"] = (
            df2["_dt"].dt.year.astype(str) + "-Q" + df2["_dt"].dt.quarter.astype(str)
        )
        sampled = (
            df2.sort_values("date").groupby(["ticker", "_yq"]).last().reset_index()
        )
        sampled = sampled.drop(columns=["_yq", "_dt"], errors="ignore")
    else:
        logger.warning("Unknown sample_mode '%s', using 'all'", mode)
        sampled = df

    sampled = sampled.sort_values(["date", "ticker"]).reset_index(drop=True)
    if len(sampled) > max_rows:
        logger.info("Sampling capped at %d rows (was %d)", max_rows, len(sampled))
        sampled = sampled.head(max_rows)
    return sampled


def _pit_violation_check(df: pd.DataFrame) -> List[dict]:
    """Return rows where PIT date > decision date (genuine PIT violation).

    Prefers known_at_effective_date if present; otherwise falls back to known_at.
    """
    use_col = (
        "known_at_effective_date"
        if "known_at_effective_date" in df.columns
        else "known_at"
    )
    violations = []
    for _, row in df.iterrows():
        ka = str(row.get(use_col, ""))
        dt = str(row.get("date", ""))
        if ka and ka != "" and ka != "nan" and dt and ka > dt:
            violations.append(
                {
                    "ticker": row.get("ticker"),
                    "date": dt,
                    use_col: ka,
                    "pit_check_column": use_col,
                }
            )
    return violations


def _unique_known_at_per_ticker(df: pd.DataFrame) -> Dict[str, int]:
    result = {}
    for ticker, g in df.groupby("ticker"):
        known = g["known_at"].dropna().replace("", pd.NA).dropna()
        result[str(ticker)] = int(known.nunique())
    return result


def _snapshot_repeated_check(
    df_full: pd.DataFrame, tickers: List[str]
) -> Dict[str, bool]:
    """Check if a ticker uses a single known_at across its full date range."""
    warnings: Dict[str, bool] = {}
    for ticker in tickers:
        sub = df_full[df_full["ticker"] == ticker]
        known = sub["known_at"].dropna().replace("", pd.NA).dropna()
        if known.empty:
            continue
        n_unique = known.nunique()
        latest = known.max()
        all_same_latest = (known == latest).all()
        if n_unique == 1 and all_same_latest and len(sub) > 5:
            warnings[ticker] = True
    return warnings


def _missing_fundamentals_stats(df: pd.DataFrame) -> dict:
    total = len(df)
    missing_known_at = int(df["known_at"].replace("", pd.NA).isna().sum())
    no_pit = int(
        (
            df.get("feature_warning", pd.Series(dtype=str))
            == "no_pit_fundamental_available"
        ).sum()
    )
    return {
        "total_rows": total,
        "missing_known_at": missing_known_at,
        "missing_known_at_pct": (
            round(100 * missing_known_at / total, 1) if total else 0
        ),
        "no_pit_fundamental_rows": no_pit,
        "no_pit_fundamental_pct": round(100 * no_pit / total, 1) if total else 0,
    }


def _valuation_stats(df: pd.DataFrame) -> dict:
    stats = {}
    for col in _VALUATION_COLS:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            stats[col] = {"min": None, "median": None, "max": None, "n_valid": 0}
        else:
            stats[col] = {
                "min": round(float(s.min()), 4),
                "median": round(float(s.median()), 4),
                "max": round(float(s.max()), 4),
                "n_valid": int(len(s)),
            }
    return stats


def _score_range_check(df: pd.DataFrame) -> dict:
    """Check all score columns are in [0,1]."""
    out_of_range: dict = {}
    for col in _SCORE_COLS:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        bad = s[(s < 0) | (s > 1)]
        if not bad.empty:
            out_of_range[col] = int(len(bad))
    return out_of_range


def _values_vary(df: pd.DataFrame) -> Dict[str, bool]:
    """For each ticker, check if fundamental values change over time (not constant)."""
    vary: Dict[str, bool] = {}
    check_cols = ["revenue", "gross_margin", "pe_ratio", "known_at"]
    for ticker, g in df.groupby("ticker"):
        changed = False
        for col in check_cols:
            if col not in g.columns:
                continue
            s = g[col].replace("", pd.NA).dropna()
            if s.nunique() > 1:
                changed = True
                break
        vary[str(ticker)] = changed
    return vary


def _print_table(df: pd.DataFrame) -> None:
    """Print a readable subset of the DataFrame to console."""
    cols = [c for c in _DISPLAY_COLS if c in df.columns]
    fmt_cols = {
        "close": "{:.2f}",
        "ma50": "{:.2f}",
        "ma200": "{:.2f}",
        "sma50": "{:.2f}",
        "sma200": "{:.2f}",
        "price_vs_ma50": "{:.3f}",
        "price_vs_ma200": "{:.3f}",
        "recent_return_20d": "{:.3f}",
        "volatility_20d": "{:.3f}",
        "drawdown_from_recent_high": "{:.3f}",
        "momentum_score": "{:.3f}",
        "technical_risk_score": "{:.3f}",
        "gross_margin": "{:.3f}",
        "operating_margin": "{:.3f}",
        "profit_margin": "{:.3f}",
        "roe": "{:.3f}",
        "current_ratio": "{:.2f}",
        "debt_ratio": "{:.3f}",
        "fcf_margin": "{:.3f}",
        "revenue_growth": "{:.3f}",
        "earnings_growth": "{:.3f}",
        "pe_ratio": "{:.1f}",
        "ps_ratio": "{:.2f}",
        "ev_ebitda": "{:.1f}",
        "fcf_yield": "{:.3f}",
        "value_score": "{:.3f}",
        "quality_score": "{:.3f}",
        "profitability_score": "{:.3f}",
        "balance_sheet_strength_score": "{:.3f}",
        "ticker_score": "{:.3f}",
    }
    display = df[cols].copy()
    for col, fmt in fmt_cols.items():
        if col in display.columns:
            display[col] = pd.to_numeric(display[col], errors="coerce").apply(
                lambda v: fmt.format(v) if pd.notna(v) else ""
            )
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 300)
    pd.set_option("display.max_colwidth", 18)
    pd.set_option("display.max_rows", 200)
    print(display.to_string(index=False))
    pd.reset_option("display.max_columns")
    pd.reset_option("display.width")
    pd.reset_option("display.max_colwidth")
    pd.reset_option("display.max_rows")


def main() -> None:
    source_run_id = _env_str("STOCK_INVESTMENT_DSS_HDP_INFO_SOURCE_RUN_ID", "")
    tickers_str = _env_str("STOCK_INVESTMENT_DSS_HDP_INFO_TICKERS", "AAPL,MSFT")
    start_date = _env_str("STOCK_INVESTMENT_DSS_HDP_INFO_START_DATE", "2024-01-01")
    end_date = _env_str("STOCK_INVESTMENT_DSS_HDP_INFO_END_DATE", "2024-02-01")
    sample_mode = _env_str(
        "STOCK_INVESTMENT_DSS_HDP_INFO_SAMPLE_MODE", "first_last_monthly"
    )
    max_rows = _env_int("STOCK_INVESTMENT_DSS_HDP_INFO_MAX_ROWS", 200)

    tickers = [t.strip() for t in tickers_str.split(",") if t.strip()]

    # --- Resolve source run ---
    if not source_run_id:
        source_run_id = _find_latest_source_run()
    if not source_run_id:
        logger.error(
            "No fmp_hdp_feature_smoke_test run found in %s\n"
            "Run run_fmp_hdp_feature_smoke_test first.",
            _RUNS_DIR,
        )
        sys.exit(1)

    logger.info("=== HDP PIT Information Inspection ===")
    logger.info("Source run: %s", source_run_id)
    logger.info("Tickers: %s", tickers)
    logger.info("Date range: %s → %s", start_date, end_date)
    logger.info("Sample mode: %s  max_rows: %d", sample_mode, max_rows)

    # --- Load ---
    try:
        df_full = _load_hdp_features(source_run_id)
    except FileNotFoundError as exc:
        logger.error(str(exc))
        sys.exit(1)

    # Filter to requested tickers
    df_tickers = df_full[df_full["ticker"].isin(tickers)] if tickers else df_full.copy()
    if df_tickers.empty:
        logger.error(
            "None of the requested tickers %s found in the feature table.\n"
            "Available tickers: %s",
            tickers,
            sorted(df_full["ticker"].unique().tolist()),
        )
        sys.exit(1)

    # Filter to date range
    df_range = df_tickers[
        (df_tickers["date"] >= start_date) & (df_tickers["date"] <= end_date)
    ].copy()
    if df_range.empty:
        logger.warning(
            "No rows in date range %s → %s. " "Available range: %s → %s",
            start_date,
            end_date,
            df_tickers["date"].min(),
            df_tickers["date"].max(),
        )

    # Apply sampling
    df_sampled = (
        _sample(df_range, sample_mode, max_rows) if not df_range.empty else df_range
    )

    # --- Validation checks ---
    all_warnings: List[str] = []

    # 1. PIT violations
    violations = _pit_violation_check(df_range)
    if violations:
        all_warnings.append(
            f"PIT_VIOLATIONS: {len(violations)} rows have known_at > date"
        )
        logger.warning("PIT VIOLATIONS FOUND: %d rows", len(violations))
        for v in violations[:5]:
            logger.warning("  %s", v)

    # 2. Unique known_at per ticker
    unique_known_at = _unique_known_at_per_ticker(df_range)
    for ticker, n in unique_known_at.items():
        if n == 0:
            all_warnings.append(
                f"NO_KNOWN_AT: {ticker} has no known_at values in range"
            )
        elif n == 1:
            delta_days = (
                (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
                if start_date and end_date
                else 0
            )
            if delta_days > 45:
                all_warnings.append(
                    f"SINGLE_KNOWN_AT: {ticker} has only 1 unique known_at over {delta_days} days"
                )

    # 3. Snapshot repeated check (over full ticker history)
    snapshot_repeated = _snapshot_repeated_check(df_tickers, tickers)
    for ticker, flagged in snapshot_repeated.items():
        if flagged:
            msg = f"possible_current_snapshot_repeated: {ticker}"
            all_warnings.append(msg)
            logger.warning("WARNING: %s", msg)

    # 4. Missing fundamentals
    miss_stats = _missing_fundamentals_stats(
        df_range if not df_range.empty else df_tickers
    )

    # 5. Valuation sanity
    val_stats = _valuation_stats(df_range if not df_range.empty else df_tickers)

    # 6. Score range check
    score_issues = _score_range_check(df_range if not df_range.empty else df_tickers)
    for col, n in score_issues.items():
        all_warnings.append(f"SCORE_OUT_OF_RANGE: {col} has {n} values outside [0,1]")

    # 7. Values vary check
    values_vary = _values_vary(df_range if not df_range.empty else df_tickers)

    # --- Print table ---
    if not df_sampled.empty:
        print("\n" + "=" * 80)
        print(f"HDP PIT Inspection — {source_run_id}")
        print(
            f"Tickers: {', '.join(tickers)}  |  {start_date} → {end_date}  |  mode={sample_mode}"
        )
        print("=" * 80)
        _print_table(df_sampled)
        print("=" * 80 + "\n")
    else:
        print("\n[No rows in selected date range]\n")

    # --- Output run ---
    ts = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    run_name = f"{ts}_d_iqn_dss_hdp_pit_information_inspection"
    run_dir = _RUNS_DIR / run_name
    audit_out = run_dir / "audit"
    summary_out = run_dir / "summary"
    for d in [audit_out, summary_out]:
        d.mkdir(parents=True, exist_ok=True)

    # Write audit CSV
    if not df_sampled.empty:
        audit_path = audit_out / "hdp_pit_information_rows.csv"
        df_sampled.to_csv(audit_path, index=False)
        logger.info("Wrote audit rows: %s (%d rows)", audit_path.name, len(df_sampled))

    # Write latest-per-ticker
    if not df_range.empty:
        latest_per_ticker = (
            df_range.sort_values("date").groupby("ticker").last().reset_index()
        )
        latest_path = audit_out / "hdp_pit_information_by_ticker_latest.csv"
        latest_per_ticker.to_csv(latest_path, index=False)
        logger.info("Wrote latest per ticker: %s", latest_path.name)

    # --- Summary ---
    pit_check_col = (
        "known_at_effective_date"
        if "known_at_effective_date" in df_range.columns
        else "known_at"
    )

    pit_quality_dist: dict = {}
    if "point_in_time_quality" in df_range.columns:
        pit_quality_dist = (
            df_range["point_in_time_quality"]
            .replace("", pd.NA)
            .dropna()
            .value_counts()
            .to_dict()
        )

    valuation_method_dist: dict = {}
    if "valuation_method" in df_range.columns:
        valuation_method_dist = (
            df_range["valuation_method"]
            .replace("", pd.NA)
            .dropna()
            .value_counts()
            .to_dict()
        )

    valuation_warning_dist: dict = {}
    if "valuation_warning" in df_range.columns:
        valuation_warning_dist = (
            df_range["valuation_warning"]
            .replace("", pd.NA)
            .dropna()
            .value_counts()
            .to_dict()
        )

    fundamental_lag_days_val: Optional[int] = None
    if "fundamental_lag_days" in df_range.columns:
        _lag_vals = pd.to_numeric(
            df_range["fundamental_lag_days"], errors="coerce"
        ).dropna()
        if not _lag_vals.empty:
            fundamental_lag_days_val = int(_lag_vals.mode().iloc[0])

    effective_date_range_by_ticker: dict = {}
    if "known_at_effective_date" in df_range.columns:
        for _t, _g in df_range.groupby("ticker"):
            _eff = (
                _g["known_at_effective_date"]
                .replace({"": pd.NA, "nan": pd.NA})
                .dropna()
            )
            if not _eff.empty:
                effective_date_range_by_ticker[str(_t)] = {
                    "min": str(_eff.min()),
                    "max": str(_eff.max()),
                }

    summary = {
        "run_name": run_name,
        "source_run": source_run_id,
        "tickers_requested": tickers,
        "date_range": {"start": start_date, "end": end_date},
        "sample_mode": sample_mode,
        "total_rows_in_range": int(len(df_range)),
        "sampled_rows": int(len(df_sampled)),
        "pit_check_column_used": pit_check_col,
        "pit_violations": len(violations),
        "pit_violation_details": violations[:10],
        "fundamental_lag_days": fundamental_lag_days_val,
        "known_at_effective_date_range_by_ticker": effective_date_range_by_ticker,
        "unique_known_at_per_ticker": unique_known_at,
        "values_vary_per_ticker": values_vary,
        "snapshot_repeated_warning": snapshot_repeated,
        "missing_fundamentals": miss_stats,
        "pit_quality_distribution": pit_quality_dist,
        "valuation_method_distribution": valuation_method_dist,
        "valuation_warning_distribution": valuation_warning_dist,
        "valuation_stats": val_stats,
        "score_out_of_range": score_issues,
        "warnings": all_warnings,
        "pit_looks_safe": len(violations) == 0,
        "fundamentals_appear_time_varying": any(values_vary.values()),
    }

    json_path = summary_out / "hdp_pit_information_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Wrote summary JSON: %s", json_path.name)

    # Markdown summary
    md_lines = [
        "# HDP PIT Information Inspection",
        "",
        f"**Source run:** {source_run_id}",
        f"**Tickers:** {', '.join(tickers)}",
        f"**Date range:** {start_date} → {end_date}",
        f"**Sample mode:** {sample_mode}",
        "",
        "## Row Counts",
        f"- Rows in range: {len(df_range)}",
        f"- Sampled rows written: {len(df_sampled)}",
        "",
        "## PIT Lag",
        f"- fundamental_lag_days: {fundamental_lag_days_val}",
        f"- PIT check column used: {pit_check_col}",
        "",
        "## PIT Correctness",
        f"- PIT violations ({pit_check_col} > date): **{len(violations)}**",
        f"- PIT looks safe: **{summary['pit_looks_safe']}**",
        "",
        "## Known-at Coverage",
    ]
    for ticker, n in unique_known_at.items():
        md_lines.append(f"- {ticker}: {n} unique known_at values in range")
    md_lines += [
        "",
        "## Time Variation",
    ]
    for ticker, varies in values_vary.items():
        md_lines.append(f"- {ticker}: values vary = {varies}")
    md_lines += [
        "",
        "## Missing Fundamentals",
        f"- Rows with missing known_at: {miss_stats['missing_known_at']} ({miss_stats['missing_known_at_pct']}%)",
        f"- Rows with no_pit_fundamental: {miss_stats['no_pit_fundamental_rows']} ({miss_stats['no_pit_fundamental_pct']}%)",
        "",
        "## Valuation Method",
    ]
    for vm, cnt in valuation_method_dist.items():
        md_lines.append(f"- {vm}: {cnt}")
    md_lines += [
        "",
        "## Valuation Warnings",
    ]
    for vw, cnt in valuation_warning_dist.items():
        md_lines.append(f"- {vw}: {cnt}")
    if not valuation_warning_dist:
        md_lines.append("- None")
    md_lines += [
        "",
        "## Valuation Stats (min / median / max)",
    ]
    for col, vs in val_stats.items():
        md_lines.append(
            f"- {col}: {vs.get('min')} / {vs.get('median')} / {vs.get('max')}  (n={vs.get('n_valid')})"
        )
    if all_warnings:
        md_lines += ["", "## Warnings"]
        for w in all_warnings:
            md_lines.append(f"- ⚠️ {w}")
    else:
        md_lines += ["", "## Warnings", "- None"]

    md_path = summary_out / "hdp_pit_information_summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    logger.info("Wrote summary MD: %s", md_path.name)

    logger.info("=== Inspection complete: %s ===", run_dir)

    # Print key findings to console
    print("\n--- Key Findings ---")
    print(f"  Source run:      {source_run_id}")
    print(f"  Date range:      {start_date} → {end_date}")
    print(f"  Rows in range:   {len(df_range)}")
    print(f"  PIT violations:  {len(violations)}")
    print(f"  unique known_at: {unique_known_at}")
    print(f"  values vary:     {values_vary}")
    print(f"  warnings:        {all_warnings or 'none'}")
    print(f"  Output dir:      {run_dir}")


if __name__ == "__main__":
    main()
