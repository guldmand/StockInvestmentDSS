"""
FMP PIT Fundamentals Builder.

Loads raw cached FMP statement responses and normalizes them into a
point-in-time safe feature table for HDP enrichment.

Point-in-time rule:
  known_at = acceptedDate  (preferred — actual SEC acceptance timestamp)
           | filingDate     (fallback)
           | date           (period end — last resort, flags point_in_time_quality = WARN)

For decision date D, only rows with known_at <= D may be used.

Metric calculations inspired by:
  externals/DS808_Visualization/clean_dashboard/analytics/investor_snapshot.py

Field names follow the FMP quarterly.json schema verified from:
  externals/DS808_Visualization/clean_dashboard/data/financials/AAPL/quarterly.json
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CACHE_BASE = _REPO_ROOT / "data" / "api_cache" / "fmp" / "raw"


class FMPPITFundamentalsBuilder:
    """
    Build PIT-safe fundamental features from cached FMP statement files.

    Usage
    -----
    builder = FMPPITFundamentalsBuilder()
    pit_df = builder.build(tickers=["AAPL", "MSFT", ...], period="quarter")
    row = builder.get_pit_row(pit_df, ticker="AAPL", decision_date="2023-01-15")
    """

    def __init__(self, cache_base: Optional[Path] = None):
        from stock_investment_dss.data.fmp_raw_cache import FMPRawCache

        self.cache = FMPRawCache(cache_base)

    def build(
        self,
        tickers: List[str],
        period: str = "quarter",
        limit: int = 80,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Build PIT fundamental feature table for given tickers.

        Returns a DataFrame with one row per (ticker, known_at) combination,
        containing all computable fundamental features.
        """
        all_rows: List[dict] = []
        for ticker in tickers:
            rows = self._build_ticker(ticker, period, limit, start_date, end_date)
            all_rows.extend(rows)
        if not all_rows:
            logger.warning("No PIT fundamental rows built (cache may be empty)")
            return pd.DataFrame()
        df = pd.DataFrame(all_rows)
        df = df.sort_values(["ticker", "known_at"]).reset_index(drop=True)
        return df

    def get_pit_row(
        self, pit_df: pd.DataFrame, ticker: str, decision_date: str
    ) -> Optional[pd.Series]:
        """
        Return the most recent fundamental row for ticker known at or before decision_date.

        Returns None if no qualifying row exists.
        """
        if pit_df is None or pit_df.empty:
            return None
        subset = pit_df[
            (pit_df["ticker"] == ticker) & (pit_df["known_at"] <= decision_date)
        ]
        if subset.empty:
            return None
        return subset.sort_values("known_at").iloc[-1]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_ticker(
        self,
        ticker: str,
        period: str,
        limit: int,
        start_date: Optional[str],
        end_date: Optional[str],
    ) -> List[dict]:
        inc_raw = self.cache.load(
            ticker, "income-statement", period=period, limit=limit
        )
        bal_raw = self.cache.load(
            ticker, "balance-sheet-statement", period=period, limit=limit
        )
        cf_raw = self.cache.load(
            ticker, "cash-flow-statement", period=period, limit=limit
        )
        profile_raw = self.cache.load(ticker, "profile")

        if inc_raw is None and bal_raw is None and cf_raw is None:
            logger.debug("No cached data for %s", ticker)
            return []

        inc_rows = inc_raw if isinstance(inc_raw, list) else []
        bal_rows = bal_raw if isinstance(bal_raw, list) else []
        cf_rows = cf_raw if isinstance(cf_raw, list) else []

        # Index by date for joining
        bal_by_date = {r.get("date", ""): r for r in bal_rows}
        cf_by_date = {r.get("date", ""): r for r in cf_rows}

        # Profile for static metadata
        sector = industry = company_name = ""
        if profile_raw:
            profile_list = (
                profile_raw if isinstance(profile_raw, list) else [profile_raw]
            )
            if profile_list:
                p = profile_list[0]
                sector = p.get("sector", "")
                industry = p.get("industry", "")
                company_name = p.get("companyName", "")

        rows: List[dict] = []
        for i, inc in enumerate(inc_rows):
            date_str = inc.get("date", "")
            filing_date = inc.get("filingDate", "")
            accepted_date = inc.get("acceptedDate", "")

            known_at, pit_quality = self._resolve_known_at(
                date_str, filing_date, accepted_date
            )

            if start_date and known_at < start_date:
                continue
            if end_date and known_at > end_date:
                continue

            bal = bal_by_date.get(date_str, {})
            cf_row = cf_by_date.get(date_str, {})

            # --- Income metrics ---
            revenue = _safe_float(inc.get("revenue"))
            gross_profit = _safe_float(inc.get("grossProfit"))
            operating_income = _safe_float(inc.get("operatingIncome"))
            net_income = _safe_float(inc.get("netIncome"))
            eps = _safe_float(inc.get("eps"))
            shares = _safe_float(inc.get("weightedAverageShsOut"))
            rd_expenses = _safe_float(inc.get("researchAndDevelopmentExpenses"))
            ebitda = _safe_float(inc.get("ebitda"))

            # --- Balance sheet metrics ---
            total_assets = _safe_float(bal.get("totalAssets"))
            total_liabilities = _safe_float(bal.get("totalLiabilities"))
            total_debt = _safe_float(bal.get("totalDebt"))
            cash = _safe_float(bal.get("cashAndCashEquivalents"))
            equity = _safe_float(bal.get("totalStockholdersEquity"))
            current_assets = _safe_float(
                bal.get("currentAssets") or bal.get("totalCurrentAssets")
            )
            current_liabilities = _safe_float(
                bal.get("currentLiabilities") or bal.get("totalCurrentLiabilities")
            )

            # --- Cash flow metrics ---
            operating_cf = _safe_float(cf_row.get("operatingCashFlow"))
            capex = _safe_float(cf_row.get("capitalExpenditure"))
            free_cash_flow = _safe_float(cf_row.get("freeCashFlow"))
            if (
                free_cash_flow is None
                and operating_cf is not None
                and capex is not None
            ):
                free_cash_flow = operating_cf + capex  # capex is typically negative

            # --- Computed ratios ---
            gross_margin = _safe_div(gross_profit, revenue)
            operating_margin = _safe_div(operating_income, revenue)
            profit_margin = _safe_div(net_income, revenue)
            fcf_margin = _safe_div(free_cash_flow, revenue)
            debt_ratio = _safe_div(total_debt, total_assets)
            roe = _safe_div(net_income, equity)
            current_ratio = _safe_div(current_assets, current_liabilities)
            rd_intensity = _safe_div(rd_expenses, revenue)

            # --- YoY growth (uses 4-period lag for quarterly) ---
            revenue_growth = None
            earnings_growth = None
            lag = 4 if period == "quarter" else 1
            if i + lag < len(inc_rows):
                prev_inc = inc_rows[i + lag]
                prev_rev = _safe_float(prev_inc.get("revenue"))
                prev_ni = _safe_float(prev_inc.get("netIncome"))
                revenue_growth = _safe_pct_change(revenue, prev_rev)
                earnings_growth = _safe_pct_change(net_income, prev_ni)

            # --- Composite scores ---
            quality_score = _composite_quality(
                profit_margin, roe, fcf_margin, debt_ratio
            )
            profitability_score = _composite_profitability(
                gross_margin, operating_margin, profit_margin
            )
            balance_sheet_strength = _composite_balance(
                debt_ratio, current_ratio, fcf_margin
            )

            rows.append(
                {
                    "ticker": ticker,
                    "period_end_date": date_str,
                    "fiscal_year": inc.get("fiscalYear"),
                    "period": inc.get("period"),
                    "filing_date": filing_date,
                    "accepted_date": str(accepted_date)[:10] if accepted_date else "",
                    "known_at": known_at,
                    "point_in_time_quality": pit_quality,
                    "sector": sector,
                    "industry": industry,
                    "company_name": company_name,
                    # Income
                    "revenue": revenue,
                    "gross_profit": gross_profit,
                    "operating_income": operating_income,
                    "net_income": net_income,
                    "eps": eps,
                    "shares_outstanding": shares,
                    "ebitda": ebitda,
                    "rd_expenses": rd_expenses,
                    # Balance
                    "total_assets": total_assets,
                    "total_liabilities": total_liabilities,
                    "total_debt": total_debt,
                    "cash": cash,
                    "equity": equity,
                    "current_assets": current_assets,
                    "current_liabilities": current_liabilities,
                    # Cash flow
                    "operating_cash_flow": operating_cf,
                    "capex": capex,
                    "free_cash_flow": free_cash_flow,
                    # Margins & ratios
                    "gross_margin": gross_margin,
                    "operating_margin": operating_margin,
                    "profit_margin": profit_margin,
                    "fcf_margin": fcf_margin,
                    "debt_ratio": debt_ratio,
                    "roe": roe,
                    "current_ratio": current_ratio,
                    "rd_intensity": rd_intensity,
                    # Growth
                    "revenue_growth": revenue_growth,
                    "earnings_growth": earnings_growth,
                    # Composite scores
                    "quality_score": quality_score,
                    "profitability_score": profitability_score,
                    "balance_sheet_strength_score": balance_sheet_strength,
                    # Valuation inputs (require PIT price for final valuation ratios)
                    "trailing_eps_q": eps,
                    "trailing_revenue_q": revenue,
                    # Snapshot-only valuation fields (filled later from HDP join)
                    "pe_ratio": None,
                    "ps_ratio": None,
                    "ev_ebitda": None,
                    "fcf_yield": None,
                    "valuation_score": None,
                }
            )

        return rows

    @staticmethod
    def _resolve_known_at(date_str: str, filing_date: str, accepted_date: str):
        """Return (known_at_date_str, pit_quality_label)."""
        # acceptedDate may be "2025-10-31 06:01:26" — take date portion
        if accepted_date and str(accepted_date).strip() not in ("", "None", "null"):
            return str(accepted_date).strip()[:10], "accepted_date"
        if filing_date and str(filing_date).strip() not in ("", "None", "null"):
            return str(filing_date).strip()[:10], "filing_date"
        if date_str and str(date_str).strip() not in ("", "None", "null"):
            return str(date_str).strip()[:10], "WARN_period_end_only"
        return "", "unknown"


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _safe_float(val) -> Optional[float]:
    if val is None:
        return None
    try:
        f = float(val)
        return None if np.isnan(f) or np.isinf(f) else f
    except (TypeError, ValueError):
        return None


def _safe_div(numerator, denominator) -> Optional[float]:
    if numerator is None or denominator is None:
        return None
    if denominator == 0:
        return None
    return _safe_float(numerator / denominator)


def _safe_pct_change(current, previous) -> Optional[float]:
    if current is None or previous is None:
        return None
    if previous == 0:
        return None
    return _safe_float((current - previous) / abs(previous))


def _composite_quality(profit_margin, roe, fcf_margin, debt_ratio) -> Optional[float]:
    components = []
    if profit_margin is not None:
        components.append(np.clip(profit_margin / 0.30, -1, 1))
    if roe is not None:
        components.append(np.clip(roe / 0.25, -1, 1))
    if fcf_margin is not None:
        components.append(np.clip(fcf_margin / 0.20, -1, 1))
    if debt_ratio is not None:
        components.append(np.clip(1.0 - debt_ratio, 0, 1))
    if not components:
        return None
    raw = float(np.mean(components))
    return round(np.clip((raw + 1.0) / 2.0, 0.0, 1.0), 4)  # rescale [-1,1] → [0,1]


def _composite_profitability(
    gross_margin, operating_margin, profit_margin
) -> Optional[float]:
    components = []
    if gross_margin is not None:
        components.append(np.clip(gross_margin / 0.60, 0, 1))
    if operating_margin is not None:
        components.append(np.clip(operating_margin / 0.30, 0, 1))
    if profit_margin is not None:
        components.append(np.clip(profit_margin / 0.25, 0, 1))
    if not components:
        return None
    return round(float(np.mean(components)), 4)


def _composite_balance(debt_ratio, current_ratio, fcf_margin) -> Optional[float]:
    components = []
    if debt_ratio is not None:
        components.append(float(np.clip(1.0 - debt_ratio, 0, 1)))
    if current_ratio is not None:
        components.append(float(np.clip(current_ratio / 3.0, 0, 1)))
    if fcf_margin is not None:
        components.append(float(np.clip(fcf_margin / 0.20, 0, 1)))
    if not components:
        return None
    return round(float(np.mean(components)), 4)
