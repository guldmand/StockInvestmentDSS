"""
FMP HDP Feature Smoke Test Runner.

Orchestrates the full FMP ingestion + HDP feature building pipeline:
  1. FMPApiClient  →  FMPRawCache  (fetch raw data if live_enabled)
  2. FMPPITFundamentalsBuilder  →  PIT fundamental feature table
  3. HDPTechnicalFeatureBuilder  →  technical feature table
  4. HDPFeatureStore.join()  →  HDP-ready joined feature table
  5. Write summary and all output CSVs

Environment variables
---------------------
FMP_API_KEY                                 (required if live_enabled=true)
STOCK_INVESTMENT_DSS_FMP_LIVE_ENABLED       default: false
STOCK_INVESTMENT_DSS_FMP_CACHE_ONLY         default: true
STOCK_INVESTMENT_DSS_FMP_TICKERS            default: AAPL,MSFT,NVDA,AMZN,GOOGL
STOCK_INVESTMENT_DSS_FMP_START_DATE         default: 2018-01-01
STOCK_INVESTMENT_DSS_FMP_END_DATE           default: 2024-02-01
STOCK_INVESTMENT_DSS_FMP_PERIOD             default: quarter
STOCK_INVESTMENT_DSS_FMP_LIMIT              default: 80
STOCK_INVESTMENT_DSS_HDP_MARKET_DATA_PATH   default: (project default path)
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "src"))


def _env_bool(key: str, default: bool) -> bool:
    return os.environ.get(key, str(default)).strip().lower() in ("1", "true", "yes")


def _env_str(key: str, default: str) -> str:
    return os.environ.get(key, default).strip()


def main() -> None:
    live_enabled = _env_bool("STOCK_INVESTMENT_DSS_FMP_LIVE_ENABLED", False)
    cache_only = _env_bool("STOCK_INVESTMENT_DSS_FMP_CACHE_ONLY", True)
    tickers_str = _env_str(
        "STOCK_INVESTMENT_DSS_FMP_TICKERS", "AAPL,MSFT,NVDA,AMZN,GOOGL"
    )
    start_date = _env_str("STOCK_INVESTMENT_DSS_FMP_START_DATE", "2018-01-01")
    end_date = _env_str("STOCK_INVESTMENT_DSS_FMP_END_DATE", "2024-02-01")
    period = _env_str("STOCK_INVESTMENT_DSS_FMP_PERIOD", "quarter")
    limit = int(_env_str("STOCK_INVESTMENT_DSS_FMP_LIMIT", "80"))
    market_data_path_str = _env_str("STOCK_INVESTMENT_DSS_HDP_MARKET_DATA_PATH", "")
    try:
        lag_days = int(_env_str("STOCK_INVESTMENT_DSS_FMP_FUNDAMENTAL_LAG_DAYS", "1"))
    except ValueError:
        lag_days = 1

    tickers = [t.strip() for t in tickers_str.split(",") if t.strip()]

    ts = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    run_name = f"{ts}_d_iqn_dss_fmp_hdp_feature_smoke_test"
    run_dir = _REPO_ROOT / "outputs" / "runs" / run_name
    data_out = run_dir / "data"
    summary_out = run_dir / "summary"
    for d in [data_out, summary_out]:
        d.mkdir(parents=True, exist_ok=True)

    logger.info("=== FMP HDP Feature Smoke Test ===")
    logger.info("Run dir: %s", run_dir)
    logger.info("Tickers: %s", tickers)
    logger.info(
        "Period: %s, Limit: %d, Date range: %s → %s",
        period,
        limit,
        start_date,
        end_date,
    )
    logger.info("fundamental_lag_days=%d", lag_days)

    # Startup diagnostics — never print the key value
    fmp_key_present = bool(os.environ.get("FMP_API_KEY", "").strip())
    logger.info(
        "live_enabled=%s  cache_only=%s  fmp_api_key_present=%s",
        live_enabled,
        cache_only,
        fmp_key_present,
    )

    if live_enabled and not fmp_key_present:
        logger.error(
            "live_enabled=true but FMP_API_KEY is not set. "
            "Set $env:FMP_API_KEY before running live ingestion."
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # Imports
    # ------------------------------------------------------------------
    from stock_investment_dss.data.fmp_api_client import (
        FMPApiClient,
        FMPLiveDisabledError,
    )
    from stock_investment_dss.data.fmp_raw_cache import FMPRawCache
    from stock_investment_dss.data.fmp_pit_fundamentals_builder import (
        FMPPITFundamentalsBuilder,
    )
    from stock_investment_dss.data.hdp_technical_feature_builder import (
        HDPTechnicalFeatureBuilder,
    )
    from stock_investment_dss.data.hdp_feature_store import HDPFeatureStore

    cache = FMPRawCache()
    # Pass BOTH live_enabled and cache_only so the client resolves correctly
    client = FMPApiClient(live_enabled=live_enabled, cache_only=cache_only)
    fundamentals_builder = FMPPITFundamentalsBuilder()
    market_data_path = Path(market_data_path_str) if market_data_path_str else None
    tech_builder = HDPTechnicalFeatureBuilder(market_data_path=market_data_path)
    store = HDPFeatureStore()

    # ------------------------------------------------------------------
    # Step 1: FMP ingestion (if live_enabled)
    # ------------------------------------------------------------------
    live_calls_made = 0
    cache_hits = 0
    endpoints_used = []

    STATEMENT_ENDPOINTS = [
        "income-statement",
        "balance-sheet-statement",
        "cash-flow-statement",
    ]
    SNAPSHOT_ENDPOINTS = ["profile"]

    for ticker in tickers:
        for ep in STATEMENT_ENDPOINTS:
            if cache.has(ticker, ep, period=period, limit=limit):
                cache_hits += 1
                if ep not in endpoints_used:
                    endpoints_used.append(ep)
                continue
            if not live_enabled:
                logger.debug(
                    "Cache miss for %s/%s — live disabled, skipping", ticker, ep
                )
                continue
            try:
                method = {
                    "income-statement": client.fetch_income_statement,
                    "balance-sheet-statement": client.fetch_balance_sheet,
                    "cash-flow-statement": client.fetch_cash_flow,
                }[ep]
                data = method(ticker, period=period, limit=limit)
                cache.save(ticker, ep, data, period=period, limit=limit)
                live_calls_made += 1
                if ep not in endpoints_used:
                    endpoints_used.append(ep)
            except FMPLiveDisabledError:
                logger.warning("Live FMP disabled — skipping %s/%s", ticker, ep)
            except Exception as exc:
                logger.warning("FMP error for %s/%s: %s", ticker, ep, exc)

        for ep in SNAPSHOT_ENDPOINTS:
            if cache.has(ticker, ep):
                cache_hits += 1
                if ep not in endpoints_used:
                    endpoints_used.append(ep)
                continue
            if not live_enabled:
                continue
            try:
                data = client.fetch_company_profile(ticker)
                cache.save(ticker, ep, data)
                live_calls_made += 1
                if ep not in endpoints_used:
                    endpoints_used.append(ep)
            except Exception as exc:
                logger.warning("FMP profile error for %s: %s", ticker, exc)

    if cache_hits == 0 and not live_enabled:
        logger.error(
            "\n"
            "══════════════════════════════════════════════════════════════\n"
            "  No FMP cache found and live ingestion is DISABLED.\n"
            "\n"
            "  To run live FMP ingestion, set:\n"
            "    $env:FMP_API_KEY       = '<your-key>'\n"
            "    $env:STOCK_INVESTMENT_DSS_FMP_LIVE_ENABLED = 'true'\n"
            "    $env:STOCK_INVESTMENT_DSS_FMP_CACHE_ONLY   = 'false'\n"
            "  Then re-run this script.\n"
            "══════════════════════════════════════════════════════════════"
        )
        sys.exit(0)

    # ------------------------------------------------------------------
    # Step 2: PIT fundamentals
    # ------------------------------------------------------------------
    logger.info("Building PIT fundamentals...")
    try:
        pit_df = fundamentals_builder.build(
            tickers=tickers,
            period=period,
            limit=limit,
            start_date=start_date,
            end_date=end_date,
        )
    except Exception as exc:
        logger.warning("PIT fundamentals build failed: %s", exc)
        pit_df = None

    # ------------------------------------------------------------------
    # Step 3: Technical features
    # ------------------------------------------------------------------
    logger.info("Building technical features...")
    tech_df = None
    try:
        tech_df = tech_builder.build(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
        )
    except FileNotFoundError as exc:
        logger.warning("Market data not found: %s — technical features skipped", exc)
    except Exception as exc:
        logger.warning("Technical features build failed: %s", exc)

    # ------------------------------------------------------------------
    # Step 4: Join into HDP feature table
    # ------------------------------------------------------------------
    hdp_df = None
    if tech_df is not None:
        logger.info("Joining HDP feature table...")
        try:
            hdp_df = store.join(
                tech_df,
                pit_df,
                tickers=tickers,
                start_date=start_date,
                end_date=end_date,
                lag_days=lag_days,
            )
        except Exception as exc:
            logger.warning("HDP feature join failed: %s", exc)

    # ------------------------------------------------------------------
    # Step 5: Write outputs
    # ------------------------------------------------------------------
    if pit_df is not None and not pit_df.empty:
        pit_path = data_out / "fmp_pit_fundamentals_features.csv"
        pit_df.to_csv(pit_path, index=False)
        logger.info("Wrote PIT fundamentals: %s (%d rows)", pit_path.name, len(pit_df))

    if tech_df is not None and not tech_df.empty:
        tech_path = data_out / "hdp_technical_features.csv"
        tech_df.to_csv(tech_path, index=False)
        logger.info(
            "Wrote technical features: %s (%d rows)", tech_path.name, len(tech_df)
        )

    if hdp_df is not None and not hdp_df.empty:
        hdp_path = data_out / "hdp_joined_features.csv"
        hdp_df.to_csv(hdp_path, index=False)
        logger.info(
            "Wrote HDP joined features: %s (%d rows)", hdp_path.name, len(hdp_df)
        )

    # Cache inventory
    inventory = cache.list_cached()
    if inventory:
        import csv

        inv_path = data_out / "fmp_raw_response_inventory.csv"
        with open(inv_path, "w", encoding="utf-8", newline="") as f:
            if inventory:
                writer = csv.DictWriter(f, fieldnames=list(inventory[0].keys()))
                writer.writeheader()
                writer.writerows(inventory)
        logger.info(
            "Wrote cache inventory: %s (%d entries)", inv_path.name, len(inventory)
        )

    # ------------------------------------------------------------------
    # Step 6: Summary
    # ------------------------------------------------------------------
    fund_tickers = (
        sorted(pit_df["ticker"].unique().tolist())
        if pit_df is not None and not pit_df.empty
        else []
    )
    tech_tickers = (
        sorted(tech_df["ticker"].unique().tolist())
        if tech_df is not None and not tech_df.empty
        else []
    )
    missing_fund = sorted(set(tickers) - set(fund_tickers))
    missing_tech = sorted(set(tickers) - set(tech_tickers))

    def _rows_per(df, col="ticker"):
        if df is None or df.empty:
            return {}
        return df.groupby(col).size().to_dict()

    def _date_range(df, date_col="known_at"):
        if df is None or df.empty:
            return {}
        out = {}
        for t in (
            df[df.columns[0]].unique()
            if "ticker" not in df.columns
            else df["ticker"].unique()
        ):
            sub = df[df["ticker"] == t][date_col].dropna()
            if not sub.empty:
                out[t] = {"min": str(sub.min()), "max": str(sub.max())}
        return out

    hdp_cols = list(hdp_df.columns) if hdp_df is not None else []
    pit_qual_counts = {}
    if (
        pit_df is not None
        and not pit_df.empty
        and "point_in_time_quality" in pit_df.columns
    ):
        pit_qual_counts = pit_df["point_in_time_quality"].value_counts().to_dict()

    summary = {
        "run_name": run_name,
        "tickers_requested": tickers,
        "tickers_with_fundamentals": fund_tickers,
        "tickers_with_technical": tech_tickers,
        "missing_fundamentals": missing_fund,
        "missing_technical": missing_tech,
        "endpoints_used": endpoints_used,
        "live_calls_made": live_calls_made,
        "cache_hits": cache_hits,
        "cache_used": cache_hits > 0,
        "live_ingestion_used": live_calls_made > 0,
        "fundamental_rows_per_ticker": _rows_per(pit_df),
        "technical_rows_per_ticker": _rows_per(
            tech_df,
            "ticker" if tech_df is not None and "ticker" in tech_df.columns else "tic",
        ),
        "hdp_rows_per_ticker": _rows_per(hdp_df),
        "fundamental_known_at_range": _date_range(pit_df),
        "pit_quality_distribution": pit_qual_counts,
        "fundamental_lag_days": lag_days,
        "same_day_filings_excluded_by_default": lag_days > 0,
        "hdp_feature_columns": hdp_cols,
        "technical_features_available": [
            c
            for c in [
                "ma50",
                "ma200",
                "sma50",
                "sma200",
                "price_vs_ma50",
                "price_vs_ma200",
                "momentum_score",
                "volatility_20d",
                "drawdown_from_recent_high",
                "technical_risk_score",
                "recent_return_5d",
                "recent_return_20d",
                "macd",
                "rsi_30",
                "cci_30",
                "dx_30",
            ]
            if c in hdp_cols
        ],
        "fundamental_features_available": [
            c
            for c in [
                "revenue",
                "revenue_growth",
                "earnings_growth",
                "gross_margin",
                "operating_margin",
                "profit_margin",
                "net_income",
                "roe",
                "current_ratio",
                "debt_ratio",
                "free_cash_flow",
                "fcf_margin",
                "rd_intensity",
                "quality_score",
                "profitability_score",
                "balance_sheet_strength_score",
            ]
            if c in hdp_cols
        ],
        "valuation_features_available": [
            c
            for c in [
                "pe_ratio",
                "ps_ratio",
                "ev_ebitda",
                "fcf_yield",
                "valuation_score",
                "value_score",
                "annualized_revenue",
                "annualized_net_income",
                "annualized_free_cash_flow",
                "market_cap_estimate",
                "enterprise_value_estimate",
                "valuation_method",
                "valuation_warning",
            ]
            if c in hdp_cols
        ],
        "valuation_method_counts": (
            hdp_df["valuation_method"].value_counts().to_dict()
            if hdp_df is not None
            and "valuation_method" in (hdp_df.columns if hdp_df is not None else [])
            else {}
        ),
        "valuation_warning_counts": (
            hdp_df["valuation_warning"].value_counts().to_dict()
            if hdp_df is not None
            and "valuation_warning" in (hdp_df.columns if hdp_df is not None else [])
            else {}
        ),
        "known_at_effective_date_range_by_ticker": (
            _date_range(hdp_df, "known_at_effective_date")
            if hdp_df is not None
            and "known_at_effective_date"
            in (hdp_df.columns if hdp_df is not None else [])
            else {}
        ),
        "pit_violations_using_effective_date": (
            int(
                (
                    (
                        hdp_df["known_at_effective_date"].astype(str).str[:10]
                        > hdp_df["date"].astype(str).str[:10]
                    )
                    & (hdp_df["known_at_effective_date"].astype(str).str[:10] != "")
                    & (hdp_df["known_at_effective_date"].astype(str).str[:10] != "nan")
                ).sum()
            )
            if hdp_df is not None
            and "known_at_effective_date"
            in (hdp_df.columns if hdp_df is not None else [])
            else None
        ),
        "hdp_ready": hdp_df is not None and not hdp_df.empty,
        "pit_fundamentals_ready": pit_df is not None and not pit_df.empty,
        "technical_features_ready": tech_df is not None and not tech_df.empty,
        "accepted_date_present": pit_qual_counts.get("accepted_date", 0) > 0,
        "filing_date_fallback_rows": pit_qual_counts.get("filing_date", 0),
        "period_end_warn_rows": pit_qual_counts.get("WARN_period_end_only", 0),
        "external_code_reference": [
            "externals/SDU_DataScienceTool/src/sdu_dst/sources/financialmodelingprep.py",
            "externals/DS808_Visualization/clean_dashboard/analytics/investor_snapshot.py",
        ],
    }

    summary_json_path = summary_out / "fmp_hdp_feature_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Wrote summary JSON: %s", summary_json_path.name)

    # Markdown summary
    md_lines = [
        f"# FMP HDP Feature Smoke Test Summary",
        f"",
        f"**Run:** {run_name}",
        f"",
        f"## Tickers",
        f"- Requested: {', '.join(tickers)}",
        f"- With fundamentals: {', '.join(fund_tickers) or 'none'}",
        f"- With technical: {', '.join(tech_tickers) or 'none'}",
        f"- Missing fundamentals: {', '.join(missing_fund) or 'none'}",
        f"- Missing technical: {', '.join(missing_tech) or 'none'}",
        f"",
        f"## Ingestion",
        f"- Live calls made: {live_calls_made}",
        f"- Cache hits: {cache_hits}",
        f"- Endpoints used: {', '.join(endpoints_used) or 'none'}",
        f"",
        f"## PIT Quality",
    ]
    for q, cnt in pit_qual_counts.items():
        md_lines.append(f"- {q}: {cnt} rows")
    md_lines += [
        f"",
        f"## PIT Lag",
        f"- fundamental_lag_days: {lag_days}",
        f"- same_day_filings_excluded: {lag_days > 0}",
        f"- PIT violations (effective date): {summary.get('pit_violations_using_effective_date', 'N/A')}",
        f"",
        f"## HDP Feature Table",
        f"- Columns: {len(hdp_cols)}",
        f"- Tickers: {len(summary['hdp_rows_per_ticker'])}",
        f"- Ready for HDP: {summary['hdp_ready']}",
        f"",
        f"## Technical Features",
        ", ".join(summary["technical_features_available"]) or "none",
        f"",
        f"## Fundamental Features (PIT-safe)",
        ", ".join(summary["fundamental_features_available"]) or "none",
        f"",
        f"## Valuation Features",
        ", ".join(summary["valuation_features_available"]) or "none",
        f"",
        f"## Valuation Method Counts",
    ]
    for vm, cnt in summary.get("valuation_method_counts", {}).items():
        md_lines.append(f"- {vm}: {cnt}")
    md_lines += [
        f"",
        f"## Valuation Warning Counts",
    ]
    for vw, cnt in summary.get("valuation_warning_counts", {}).items():
        md_lines.append(f"- {vw}: {cnt}")

    md_path = summary_out / "fmp_hdp_feature_summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    logger.info("Wrote summary MD: %s", md_path.name)

    logger.info("=== Smoke test complete ===")
    logger.info("Output dir: %s", run_dir)


if __name__ == "__main__":
    main()
