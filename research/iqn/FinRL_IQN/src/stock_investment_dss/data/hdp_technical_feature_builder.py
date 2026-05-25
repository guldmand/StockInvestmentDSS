"""
HDP Technical Feature Builder.

Extends the existing TechnicalFeatureBuilder with additional columns
needed for HDP ticker selection and size decisions:
  - recent_return_5d, recent_return_20d
  - volatility_20d
  - drawdown_from_recent_high
  - momentum_score (from base builder)
  - technical_risk_score (composite: volatility + drawdown)
  - sma50, sma200 (aliases for MA50/MA200)
  - price_vs_sma50, price_vs_sma200

Loads market data from the FinRL market data CSV used in Mode B experiments.

Point-in-time rule: all features computed using only current/past rows
via rolling windows (no look-ahead).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from stock_investment_dss.data.technical_feature_builder import TechnicalFeatureBuilder

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_MARKET_DATA= (
    _REPO_ROOT / "data" / "market" / "daily" / "imports" / "market_data_full_500.csv"
)


class HDPTechnicalFeatureBuilder:
    """
    Build a HDP-ready technical feature table from FinRL market data.

    The table has one row per (ticker, date) with all technical features
    required by HDP for ticker selection and size decisions.

    Usage
    -----
    builder = HDPTechnicalFeatureBuilder()
    tech_df = builder.build(tickers=["AAPL", "MSFT"], start_date="2018-01-01", end_date="2024-02-01")
    row = builder.get_snapshot(tech_df, ticker="AAPL", as_of_date="2023-01-15")
    """

    def __init__(
        self,
        market_data_path: Optional[Path] = None,
        ma50_window: int = 50,
        ma200_window: int = 200,
        vol_window: int = 20,
        high_window: int = 60,
    ):
        self.market_data_path = (
            Path(market_data_path) if market_data_path else _DEFAULT_MARKET_DATA
        )
        self._base_builder = TechnicalFeatureBuilder(
            ma50_window=ma50_window,
            ma200_window=ma200_window,
            recent_return_window=20,
            volatility_window=vol_window,
            high_window=high_window,
        )
        self._vol_window = vol_window

    def load_market_data(
        self,
        tickers: Optional[list] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Load and filter FinRL market data CSV."""
        if not self.market_data_path.is_file():
            raise FileNotFoundError(f"Market data not found: {self.market_data_path}")
        logger.info("Loading market data from %s", self.market_data_path)
        df = pd.read_csv(self.market_data_path, parse_dates=False)
        df["date"] = pd.to_datetime(df["date"]).dt.date.astype(str)
        if tickers:
            df = df[df["tic"].isin(tickers)]
        if start_date:
            df = df[df["date"] >= start_date]
        if end_date:
            df = df[df["date"] <= end_date]
        logger.info(
            "Market data loaded: %d rows, %d tickers", len(df), df["tic"].nunique()
        )
        return df

    def build(
        self,
        tickers: Optional[list] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        market_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Build HDP technical feature table.

        Parameters
        ----------
        tickers : list, optional
            Filter to specific tickers.
        start_date, end_date : str, optional
            Date range filter (YYYY-MM-DD).
        market_df : DataFrame, optional
            Pre-loaded market data (skips file load).
        """
        if market_df is None:
            market_df = self.load_market_data(tickers, start_date, end_date)

        # Base technical features (MA50, MA200, momentum_score, drawdown, etc.)
        enriched = self._base_builder.build(market_df)

        # Add additional HDP features per ticker
        enriched = enriched.groupby("tic", group_keys=False).apply(
            self._add_hdp_features
        )
        enriched = enriched.sort_values(["date", "tic"]).reset_index(drop=True)

        # Rename/alias columns to match HDP feature spec
        enriched = self._normalize_column_names(enriched)

        return enriched

    def get_snapshot(
        self,
        tech_df: pd.DataFrame,
        ticker: str,
        as_of_date: str,
    ) -> Optional[pd.Series]:
        """
        Return the latest technical feature row for ticker on or before as_of_date.
        """
        subset = tech_df[
            (tech_df["ticker"] == ticker) & (tech_df["date"] <= as_of_date)
        ]
        if subset.empty:
            return None
        return subset.sort_values("date").iloc[-1]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _add_hdp_features(self, g: pd.DataFrame) -> pd.DataFrame:
        g = g.copy().sort_values("date")
        close = g["close"]

        # 5-day return
        g["recent_return_5d"] = close.pct_change(5)
        # 20-day return (alias for existing recent_return)
        if "recent_return" in g.columns:
            g["recent_return_20d"] = g["recent_return"]
        else:
            g["recent_return_20d"] = close.pct_change(20)

        # Volatility 20d (annualised)
        log_ret = np.log(close / close.shift(1))
        g["volatility_20d"] = log_ret.rolling(
            self._vol_window, min_periods=5
        ).std() * np.sqrt(252)

        # Technical risk score: composite of volatility + drawdown depth
        vol_score = (g["volatility_20d"] / 0.60).clip(
            0, 1
        )  # 60% annualised vol → score=1
        drawdown_col = (
            "drawdown_from_recent_high"
            if "drawdown_from_recent_high" in g.columns
            else None
        )
        if drawdown_col:
            dd_score = (-g[drawdown_col] / 0.30).clip(0, 1)  # -30% drawdown → score=1
            g["technical_risk_score"] = (0.5 * vol_score + 0.5 * dd_score).clip(0, 1)
        else:
            g["technical_risk_score"] = vol_score

        return g

    @staticmethod
    def _normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
        """Rename columns to match HDP feature spec."""
        rename = {
            "tic": "ticker",
            "MA50": "ma50",
            "MA200": "ma200",
            # price_vs_ma50, price_vs_ma200 already lowercase from base builder
            "close_30_sma": "sma30",
            "close_60_sma": "sma60",
        }
        df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

        # sma50 / sma200 aliases
        if "ma50" in df.columns and "sma50" not in df.columns:
            df["sma50"] = df["ma50"]
        if "ma200" in df.columns and "sma200" not in df.columns:
            df["sma200"] = df["ma200"]

        # price_vs_sma50 / price_vs_sma200 aliases
        if "price_vs_ma50" in df.columns and "price_vs_sma50" not in df.columns:
            df["price_vs_sma50"] = df["price_vs_ma50"]
        if "price_vs_ma200" in df.columns and "price_vs_sma200" not in df.columns:
            df["price_vs_sma200"] = df["price_vs_ma200"]

        return df
