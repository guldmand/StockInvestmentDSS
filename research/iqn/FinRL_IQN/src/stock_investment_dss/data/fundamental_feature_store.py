# src/stock_investment_dss/data/fundamental_feature_store.py
"""
Fundamental feature store for the D-IQN-DSS hierarchical policy PoC.

**IMPORTANT — v3.0 PoC limitation:**
All fundamentals in this module are FROZEN SNAPSHOT PLACEHOLDERS.
They are deterministic, hand-coded approximations for the demo_5 universe
(AAPL, MSFT, NVDA, AMZN, GOOGL) based on publicly known data ranges.

These placeholders must NOT be used in a live or production backtest.
All rows are explicitly marked: source = "frozen_snapshot_placeholder"

Future integration requirements:
  - Live fundamentals must be fetched and cached via FMP or SDU DataScienceTool
    BEFORE the backtest loop starts (not inside each step)
  - Point-in-time discipline: only use rows where available_from <= decision_date
  - FMP earnings releases are typically 2–6 weeks after fiscal quarter end;
    available_from should reflect the actual public filing date, not the fiscal_date
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Placeholder snapshot — demo_5 universe
# Each ticker has one "annual" entry per fiscal year with representative values.
# available_from simulates the approximate SEC/filing public availability date.
# ---------------------------------------------------------------------------
_PLACEHOLDER_FUNDAMENTALS: list[dict] = [
    # ---- AAPL ----
    dict(
        ticker="AAPL",
        fiscal_date="2022-09-24",
        report_date="2022-10-27",
        available_from="2022-11-01",
        source="frozen_snapshot_placeholder",
        ingested_at="2026-01-01",
        sector="Technology",
        industry="Consumer Electronics",
        revenue_growth=0.08,
        earnings_growth=0.09,
        profit_margin=0.25,
        pe_ratio=24.0,
        ps_ratio=6.0,
        free_cash_flow=90_000,
        debt_ratio=0.32,
        roe=0.175,
        roic=0.32,
        ev_to_ebitda=18.0,
        fcf_yield=0.042,
    ),
    dict(
        ticker="AAPL",
        fiscal_date="2023-09-30",
        report_date="2023-11-02",
        available_from="2023-11-07",
        source="frozen_snapshot_placeholder",
        ingested_at="2026-01-01",
        sector="Technology",
        industry="Consumer Electronics",
        revenue_growth=-0.03,
        earnings_growth=0.05,
        profit_margin=0.26,
        pe_ratio=28.0,
        ps_ratio=7.0,
        free_cash_flow=99_000,
        debt_ratio=0.31,
        roe=0.172,
        roic=0.30,
        ev_to_ebitda=20.0,
        fcf_yield=0.038,
    ),
    # ---- MSFT ----
    dict(
        ticker="MSFT",
        fiscal_date="2022-06-30",
        report_date="2022-07-26",
        available_from="2022-08-01",
        source="frozen_snapshot_placeholder",
        ingested_at="2026-01-01",
        sector="Technology",
        industry="Software",
        revenue_growth=0.18,
        earnings_growth=0.19,
        profit_margin=0.37,
        pe_ratio=29.0,
        ps_ratio=10.0,
        free_cash_flow=65_000,
        debt_ratio=0.22,
        roe=0.44,
        roic=0.29,
        ev_to_ebitda=22.0,
        fcf_yield=0.030,
    ),
    dict(
        ticker="MSFT",
        fiscal_date="2023-06-30",
        report_date="2023-07-25",
        available_from="2023-08-01",
        source="frozen_snapshot_placeholder",
        ingested_at="2026-01-01",
        sector="Technology",
        industry="Software",
        revenue_growth=0.07,
        earnings_growth=0.15,
        profit_margin=0.38,
        pe_ratio=33.0,
        ps_ratio=11.0,
        free_cash_flow=72_000,
        debt_ratio=0.20,
        roe=0.47,
        roic=0.30,
        ev_to_ebitda=25.0,
        fcf_yield=0.028,
    ),
    # ---- NVDA ----
    dict(
        ticker="NVDA",
        fiscal_date="2023-01-29",
        report_date="2023-02-22",
        available_from="2023-03-01",
        source="frozen_snapshot_placeholder",
        ingested_at="2026-01-01",
        sector="Technology",
        industry="Semiconductors",
        revenue_growth=-0.16,
        earnings_growth=-0.55,
        profit_margin=0.16,
        pe_ratio=60.0,
        ps_ratio=15.0,
        free_cash_flow=3_800,
        debt_ratio=0.19,
        roe=0.36,
        roic=0.21,
        ev_to_ebitda=55.0,
        fcf_yield=0.007,
    ),
    dict(
        ticker="NVDA",
        fiscal_date="2024-01-28",
        report_date="2024-02-21",
        available_from="2024-03-01",
        source="frozen_snapshot_placeholder",
        ingested_at="2026-01-01",
        sector="Technology",
        industry="Semiconductors",
        revenue_growth=1.22,
        earnings_growth=5.80,
        profit_margin=0.55,
        pe_ratio=65.0,
        ps_ratio=30.0,
        free_cash_flow=27_000,
        debt_ratio=0.15,
        roe=1.24,
        roic=0.80,
        ev_to_ebitda=55.0,
        fcf_yield=0.012,
    ),
    # ---- AMZN ----
    dict(
        ticker="AMZN",
        fiscal_date="2022-12-31",
        report_date="2023-02-02",
        available_from="2023-02-07",
        source="frozen_snapshot_placeholder",
        ingested_at="2026-01-01",
        sector="Consumer Cyclical",
        industry="Internet Retail",
        revenue_growth=0.09,
        earnings_growth=-0.30,
        profit_margin=0.02,
        pe_ratio=90.0,
        ps_ratio=2.0,
        free_cash_flow=-12_000,
        debt_ratio=0.40,
        roe=0.02,
        roic=0.04,
        ev_to_ebitda=18.0,
        fcf_yield=-0.005,
    ),
    dict(
        ticker="AMZN",
        fiscal_date="2023-12-31",
        report_date="2024-02-01",
        available_from="2024-02-06",
        source="frozen_snapshot_placeholder",
        ingested_at="2026-01-01",
        sector="Consumer Cyclical",
        industry="Internet Retail",
        revenue_growth=0.12,
        earnings_growth=2.80,
        profit_margin=0.06,
        pe_ratio=60.0,
        ps_ratio=3.0,
        free_cash_flow=36_000,
        debt_ratio=0.38,
        roe=0.17,
        roic=0.09,
        ev_to_ebitda=22.0,
        fcf_yield=0.018,
    ),
    # ---- GOOGL ----
    dict(
        ticker="GOOGL",
        fiscal_date="2022-12-31",
        report_date="2023-02-02",
        available_from="2023-02-07",
        source="frozen_snapshot_placeholder",
        ingested_at="2026-01-01",
        sector="Communication Services",
        industry="Internet Content",
        revenue_growth=0.10,
        earnings_growth=-0.21,
        profit_margin=0.21,
        pe_ratio=21.0,
        ps_ratio=4.5,
        free_cash_flow=60_000,
        debt_ratio=0.11,
        roe=0.24,
        roic=0.20,
        ev_to_ebitda=15.0,
        fcf_yield=0.042,
    ),
    dict(
        ticker="GOOGL",
        fiscal_date="2023-12-31",
        report_date="2024-01-30",
        available_from="2024-02-04",
        source="frozen_snapshot_placeholder",
        ingested_at="2026-01-01",
        sector="Communication Services",
        industry="Internet Content",
        revenue_growth=0.09,
        earnings_growth=0.23,
        profit_margin=0.24,
        pe_ratio=24.0,
        ps_ratio=5.5,
        free_cash_flow=69_000,
        debt_ratio=0.09,
        roe=0.26,
        roic=0.22,
        ev_to_ebitda=18.0,
        fcf_yield=0.038,
    ),
]

_SCORE_COLS = [
    "value_score",
    "quality_score",
    "profitability_score",
    "valuation_score",
    "risk_fit_score",
    "strategy_fit_score",
]


def _compute_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive normalised [0, 1] scores from raw fundamental columns.
    These are simplified heuristics suitable for the PoC.
    """
    df = df.copy()

    # value_score: high FCF yield + low PE/PS → value
    fcf_norm = (df["fcf_yield"] / 0.05).clip(0, 1)
    pe_inv = (1 / df["pe_ratio"].replace(0, float("nan")) / 0.05).clip(0, 1)
    df["value_score"] = (0.60 * fcf_norm + 0.40 * pe_inv).clip(0, 1)

    # quality_score: high ROE + high ROIC + low debt
    roe_norm = (df["roe"] / 0.5).clip(0, 1)
    roic_norm = (df["roic"] / 0.4).clip(0, 1)
    debt_inv = 1 - df["debt_ratio"].clip(0, 1)
    df["quality_score"] = (0.35 * roe_norm + 0.35 * roic_norm + 0.30 * debt_inv).clip(
        0, 1
    )

    # profitability_score: profit margin + revenue/earnings growth
    margin_norm = (df["profit_margin"] / 0.4).clip(0, 1)
    rev_g = ((df["revenue_growth"] + 0.2) / 0.4).clip(0, 1)
    ear_g = ((df["earnings_growth"] + 0.5) / 1.5).clip(0, 1)
    df["profitability_score"] = (0.40 * margin_norm + 0.30 * rev_g + 0.30 * ear_g).clip(
        0, 1
    )

    # valuation_score: lower EV/EBITDA is better relative value
    ev_inv = 1 - (df["ev_to_ebitda"] / 60.0).clip(0, 1)
    ps_inv = 1 - (df["ps_ratio"] / 20.0).clip(0, 1)
    df["valuation_score"] = (0.60 * ev_inv + 0.40 * ps_inv).clip(0, 1)

    # risk_fit_score: low debt + high FCF → lower financial risk
    df["risk_fit_score"] = (0.50 * debt_inv + 0.50 * fcf_norm).clip(0, 1)

    # strategy_fit_score: balanced composite; can be overridden by strategy config
    df["strategy_fit_score"] = (
        0.25 * df["quality_score"]
        + 0.25 * df["profitability_score"]
        + 0.25 * df["value_score"]
        + 0.25 * df["valuation_score"]
    ).clip(0, 1)

    return df


class FundamentalFeatureStore:
    """
    Provides frozen fundamental feature snapshots for the PoC.

    Point-in-time discipline: only rows where available_from <= decision_date
    are returned, preventing look-ahead bias.

    FUTURE WORK:
        Replace `_PLACEHOLDER_FUNDAMENTALS` with cached FMP/SDU snapshots
        fetched before the backtest loop. Do NOT call live FMP APIs inside
        the backtest step — this violates point-in-time discipline and
        introduces look-ahead bias.
    """

    def __init__(self) -> None:
        raw = pd.DataFrame(_PLACEHOLDER_FUNDAMENTALS)
        raw["available_from"] = pd.to_datetime(raw["available_from"])
        raw["fiscal_date"] = pd.to_datetime(raw["fiscal_date"])
        raw["report_date"] = pd.to_datetime(raw["report_date"])
        self._store: pd.DataFrame = _compute_scores(raw)

    @property
    def all_tickers(self) -> list[str]:
        return sorted(self._store["ticker"].unique().tolist())

    def get_as_of(self, decision_date: str) -> pd.DataFrame:
        """
        Return one row per ticker — the most recent fundamental snapshot
        available as of *decision_date* (available_from <= decision_date).

        Returns an empty DataFrame for tickers with no data yet.
        """
        cutoff = pd.Timestamp(decision_date)
        visible = self._store[self._store["available_from"] <= cutoff]

        if visible.empty:
            logger.warning(
                "FundamentalFeatureStore: no fundamentals visible as of %s",
                decision_date,
            )
            return pd.DataFrame(columns=self._store.columns)

        latest = (
            visible.sort_values("available_from")
            .groupby("ticker")
            .tail(1)
            .reset_index(drop=True)
        )
        return latest

    def get_scores_as_of(
        self, decision_date: str, tickers: Optional[list[str]] = None
    ) -> pd.DataFrame:
        """
        Return a compact score table (ticker + all score columns) as of decision_date.
        Optionally filter to *tickers* list.
        """
        df = self.get_as_of(decision_date)
        if tickers is not None:
            df = df[df["ticker"].isin(tickers)]
        cols = ["ticker", "sector", "industry"] + _SCORE_COLS
        present = [c for c in cols if c in df.columns]
        return df[present].reset_index(drop=True)
