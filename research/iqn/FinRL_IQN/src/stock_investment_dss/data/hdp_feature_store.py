"""
HDP Feature Store.

Joins technical features and PIT fundamental features into a single
HDP-ready feature table.

Join logic:
  1. For each (ticker, date) in the technical table:
     - Find the most recent fundamental row with known_at_effective_date <= date
       where known_at_effective_date = known_at + fundamental_lag_days
     - Join fundamental columns onto the technical row
  2. Compute valuation features using PIT close price + PIT fundamental values
  3. Add valuation audit/provenance columns
  4. Preserve provenance: technical_source, fundamental_source, known_at,
     known_at_effective_date, fundamental_lag_days, point_in_time_quality,
     feature_warning, valuation_method, valuation_warning

PIT rule: only fundamental rows with known_at_effective_date <= decision date are used.
Conservative default: lag_days=1 prevents same-day filing data from being used.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Fundamental columns to carry into the joined table
_FUNDAMENTAL_COLS = [
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
    "sector",
    "industry",
    "company_name",
    "trailing_eps_q",
    "trailing_revenue_q",
    "shares_outstanding",
    "ebitda",
    "total_debt",
    "cash",
    "filing_date",
    "accepted_date",
    "known_at",
    "known_at_effective_date",
    "fundamental_lag_days",
    "point_in_time_quality",
]

# Technical columns expected from HDPTechnicalFeatureBuilder
_TECHNICAL_COLS = [
    "ticker",
    "date",
    "close",
    "macd",
    "rsi_30",
    "cci_30",
    "dx_30",
    "close_30_sma",
    "close_60_sma",
    "ma50",
    "ma200",
    "sma50",
    "sma200",
    "price_vs_ma50",
    "price_vs_ma200",
    "price_vs_sma50",
    "price_vs_sma200",
    "recent_return_5d",
    "recent_return_20d",
    "volatility_20d",
    "drawdown_from_recent_high",
    "momentum_score",
    "technical_risk_score",
]


class HDPFeatureStore:
    """
    Joins technical and PIT fundamental features into HDP-ready rows.

    Usage
    -----
    store = HDPFeatureStore()
    hdp_df = store.join(tech_df, pit_fundamentals_df)
    row = store.get_row(hdp_df, ticker="AAPL", date="2023-01-15")
    """

    def join(
        self,
        tech_df: pd.DataFrame,
        pit_df: Optional[pd.DataFrame],
        tickers: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        lag_days: int = 1,
    ) -> pd.DataFrame:
        """
        Join technical and PIT fundamental feature tables.

        Parameters
        ----------
        lag_days : int
            Conservative PIT lag (default 1).  known_at_effective_date = known_at + lag_days.
            Use 0 to match previous behaviour (same-day filings allowed).

        Returns a wide HDP feature DataFrame with one row per (ticker, date).
        """
        if tech_df is None or tech_df.empty:
            logger.warning("Technical feature table is empty")
            return pd.DataFrame()

        lag_days = max(0, int(lag_days))

        # Normalize ticker column
        if "ticker" not in tech_df.columns and "tic" in tech_df.columns:
            tech_df = tech_df.rename(columns={"tic": "ticker"})

        tech_df = tech_df.copy()
        tech_df["date"] = tech_df["date"].astype(str).str[:10]

        if tickers:
            tech_df = tech_df[tech_df["ticker"].isin(tickers)]
        if start_date:
            tech_df = tech_df[tech_df["date"] >= start_date]
        if end_date:
            tech_df = tech_df[tech_df["date"] <= end_date]

        if pit_df is None or pit_df.empty:
            logger.warning(
                "No PIT fundamental data available; producing technical-only table"
            )
            tech_df["fundamental_source"] = "none"
            tech_df["technical_source"] = "market_data_csv"
            tech_df["feature_warning"] = "no_fundamentals"
            return self._compute_valuation_features(tech_df)

        # Compute known_at_effective_date = known_at + lag_days
        pit_df = pit_df.copy()
        pit_df["known_at"] = pit_df["known_at"].astype(str).str[:10]
        _known_at_dt = pd.to_datetime(
            pit_df["known_at"].replace({"": pd.NaT, "None": pd.NaT}), errors="coerce"
        )
        if lag_days > 0:
            pit_df["known_at_effective_date"] = (
                (_known_at_dt + pd.Timedelta(days=lag_days))
                .dt.strftime("%Y-%m-%d")
                .fillna("")
            )
        else:
            pit_df["known_at_effective_date"] = pit_df["known_at"]
        pit_df["fundamental_lag_days"] = lag_days

        lookup_col = (
            "known_at_effective_date"
            if "known_at_effective_date" in pit_df.columns
            else "known_at"
        )
        if lag_days > 0:
            logger.debug(
                "PIT join: using known_at_effective_date (lag=%d days)", lag_days
            )

        joined_rows = []
        for _, tech_row in tech_df.iterrows():
            ticker = tech_row.get("ticker", tech_row.get("tic", ""))
            date = str(tech_row["date"])[:10]

            fund_row = self._lookup_fundamental(pit_df, ticker, date, lookup_col)
            merged = dict(tech_row)

            if fund_row is not None:
                for col in _FUNDAMENTAL_COLS:
                    if col in fund_row.index:
                        merged[col] = fund_row[col]
                merged["fundamental_source"] = "fmp_pit_cache"
                merged["feature_warning"] = ""
                pit_qual = fund_row.get("point_in_time_quality", "unknown")
                if pit_qual == "WARN_period_end_only":
                    merged["feature_warning"] = "WARN_period_end_only"
            else:
                for col in _FUNDAMENTAL_COLS:
                    if col not in merged:
                        merged[col] = None
                merged["fundamental_source"] = "none"
                merged["feature_warning"] = "no_pit_fundamental_available"
                merged["known_at_effective_date"] = ""
                merged["fundamental_lag_days"] = lag_days

            merged["technical_source"] = "market_data_csv"
            joined_rows.append(merged)

        result = pd.DataFrame(joined_rows)
        result = self._compute_valuation_features(result)
        result = self._compute_composite_scores(result)
        result = result.sort_values(["date", "ticker"]).reset_index(drop=True)
        return result

    def get_row(
        self, hdp_df: pd.DataFrame, ticker: str, date: str
    ) -> Optional[pd.Series]:
        """Return the feature row for a specific ticker and date."""
        subset = hdp_df[(hdp_df["ticker"] == ticker) & (hdp_df["date"] == date)]
        if subset.empty:
            return None
        return subset.iloc[0]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _lookup_fundamental(
        pit_df: pd.DataFrame,
        ticker: str,
        date: str,
        lookup_col: str = "known_at",
    ) -> Optional[pd.Series]:
        """Find the most recent fundamental row with lookup_col <= date."""
        subset = pit_df[
            (pit_df["ticker"] == ticker)
            & (pit_df[lookup_col] <= date)
            & (pit_df[lookup_col] != "")
        ]
        if subset.empty:
            return None
        return subset.sort_values(lookup_col).iloc[-1]

    @staticmethod
    def _compute_valuation_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute PIT valuation ratios using close price and trailing fundamental data.

        These require both a current close price (from market data) and
        trailing EPS / revenue / shares from PIT fundamentals.

        Also adds valuation audit/provenance columns:
          - annualized_revenue, annualized_net_income, annualized_free_cash_flow
          - market_cap_estimate, enterprise_value_estimate
          - weighted_average_shares_out (alias of shares_outstanding)
          - valuation_method, valuation_warning
        """

        def _col(name: str) -> pd.Series:
            """Return column as numeric Series; all-NaN Series if column absent."""
            if name in df.columns:
                return pd.to_numeric(df[name], errors="coerce")
            return pd.Series(np.nan, index=df.index, dtype=float)

        close = _col("close")
        eps_q = _col("trailing_eps_q")
        rev_q = _col("trailing_revenue_q")
        shares = _col("shares_outstanding")
        total_debt = _col("total_debt")
        cash = _col("cash")
        ebitda = _col("ebitda")
        fcf_margin = _col("fcf_margin")
        net_income_col = _col("net_income")
        free_cash_flow_col = _col("free_cash_flow")

        # P/E: close / (trailing_eps_q × 4)
        trailing_eps_ttm = eps_q * 4
        pe_ratio = close / trailing_eps_ttm
        pe_ratio = pe_ratio.replace([np.inf, -np.inf], np.nan)

        # P/S: (close × shares) / (trailing_revenue_q × 4)
        market_cap = close * shares
        trailing_rev_ttm = rev_q * 4
        ps_ratio = market_cap / trailing_rev_ttm
        ps_ratio = ps_ratio.replace([np.inf, -np.inf], np.nan)

        # EV/EBITDA: (market_cap + total_debt - cash) / (ebitda × 4)
        ev = market_cap + total_debt.fillna(0) - cash.fillna(0)
        trailing_ebitda_ttm = ebitda * 4
        ev_ebitda = ev / trailing_ebitda_ttm
        ev_ebitda = ev_ebitda.replace([np.inf, -np.inf], np.nan)

        # FCF yield: (fcf_margin × revenue × 4) / market_cap
        fcf_yield = (fcf_margin * trailing_rev_ttm) / market_cap
        fcf_yield = fcf_yield.replace([np.inf, -np.inf], np.nan)

        # --- Audit/provenance columns ---
        annualized_revenue = rev_q * 4
        annualized_net_income = net_income_col * 4
        annualized_free_cash_flow = free_cash_flow_col * 4
        market_cap_estimate = close * shares
        enterprise_value_estimate = (
            market_cap_estimate + total_debt.fillna(0) - cash.fillna(0)
        )
        enterprise_value_estimate = enterprise_value_estimate.replace(
            [np.inf, -np.inf], np.nan
        )
        weighted_average_shares_out = shares.copy()

        # Determine valuation_method per row
        has_shares = shares.notna()
        has_close = close.notna()
        has_rev = rev_q.notna()
        has_ebitda = ebitda.notna()

        valuation_method = pd.Series(
            "quarterly_x4_pit_close", index=df.index, dtype=object
        )
        valuation_method[~(has_shares & has_close & has_rev)] = "partial_data"
        valuation_method[~has_close] = "no_close_available"

        # valuation_warning: flag approximations / missing inputs
        valuation_warning = pd.Series("", index=df.index, dtype=object)
        ebitda_annualized_approx = has_ebitda & (ebitda != 0)
        valuation_warning[ebitda_annualized_approx] = "ev_ebitda_ebitda_annualized_x4"
        # More severe warnings overwrite
        valuation_warning[ebitda == 0] = "ebitda_zero__ev_ebitda_unreliable"
        valuation_warning[~has_shares] = "shares_missing__ps_approximate"
        valuation_warning[~has_close] = "close_missing__valuation_unavailable"

        df = df.copy()
        if "pe_ratio" not in df.columns or df["pe_ratio"].isna().all():
            df["pe_ratio"] = pe_ratio
        if "ps_ratio" not in df.columns or df["ps_ratio"].isna().all():
            df["ps_ratio"] = ps_ratio
        if "ev_ebitda" not in df.columns or df["ev_ebitda"].isna().all():
            df["ev_ebitda"] = ev_ebitda
        if "fcf_yield" not in df.columns or df["fcf_yield"].isna().all():
            df["fcf_yield"] = fcf_yield

        # Audit columns — always written (overwrite if present for freshness)
        df["annualized_revenue"] = annualized_revenue
        df["annualized_net_income"] = annualized_net_income
        df["annualized_free_cash_flow"] = annualized_free_cash_flow
        df["market_cap_estimate"] = market_cap_estimate
        df["enterprise_value_estimate"] = enterprise_value_estimate
        df["weighted_average_shares_out"] = weighted_average_shares_out
        df["valuation_method"] = valuation_method
        df["valuation_warning"] = valuation_warning

        return df

    @staticmethod
    def _compute_composite_scores(df: pd.DataFrame) -> pd.DataFrame:
        """Compute value_score, risk_fit_score, ticker_score composites."""
        df = df.copy()

        def _col(name: str) -> pd.Series:
            """Return column as numeric Series; all-NaN Series if column absent."""
            if name in df.columns:
                return pd.to_numeric(df[name], errors="coerce")
            return pd.Series(np.nan, index=df.index, dtype=float)

        pe = _col("pe_ratio")
        ps = _col("ps_ratio")
        fcf_y = _col("fcf_yield")

        value_components = []
        if pe.notna().any():
            value_components.append(
                pd.Series(np.clip(1.0 - pe / 60.0, 0, 1), index=df.index).fillna(0.5)
            )
        if ps.notna().any():
            value_components.append(
                pd.Series(np.clip(1.0 - ps / 20.0, 0, 1), index=df.index).fillna(0.5)
            )
        if fcf_y.notna().any():
            value_components.append(
                pd.Series(np.clip(fcf_y / 0.06, 0, 1), index=df.index).fillna(0)
            )

        if value_components:
            df["value_score"] = pd.concat(value_components, axis=1).mean(axis=1)
        else:
            df["value_score"] = pd.Series(np.nan, index=df.index)

        # risk_fit_score: lower technical_risk + stronger balance sheet
        tech_risk = _col("technical_risk_score").fillna(0.5)
        bs_strength = _col("balance_sheet_strength_score").fillna(0.5)
        df["risk_fit_score"] = ((1.0 - tech_risk) * 0.5 + bs_strength * 0.5).clip(0, 1)

        # ticker_score: overall composite
        qual = _col("quality_score").fillna(0.5)
        mom = _col("momentum_score").fillna(0)
        mom_scaled = (mom + 1.0) / 2.0  # [-1,1] → [0,1]
        val = _col("value_score").fillna(0.5)
        risk_fit = df["risk_fit_score"]

        df["ticker_score"] = (
            0.30 * qual + 0.25 * mom_scaled + 0.25 * val + 0.20 * risk_fit
        ).clip(0, 1)

        return df
