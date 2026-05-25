# src/stock_investment_dss/data/technical_feature_builder.py
"""
Technical feature builder for the D-IQN-DSS hierarchical policy PoC.

Consumes FinRL-style market data and enriches it with trend, momentum,
mean-reversion, and risk indicators that the rule-based tier of the
hierarchical decision policy uses for ticker selection and sizing.

Input DataFrame must contain at minimum:
    date, tic, close

Optional columns (used when present):
    macd, rsi_30, cci_30, dx_30, close_30_sma, close_60_sma

All new columns are computed per ticker in chronological order.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Rolling windows
_MA50_WINDOW = 50
_MA200_WINDOW = 200
_RECENT_RETURN_WINDOW = 20  # ~1 trading month
_VOLATILITY_WINDOW = 20
_HIGH_WINDOW = 60  # recent high lookback


class TechnicalFeatureBuilder:
    """
    Adds technical features to a FinRL-style market DataFrame.

    Usage
    -----
    builder = TechnicalFeatureBuilder()
    enriched_df = builder.build(raw_df)
    """

    def __init__(
        self,
        ma50_window: int = _MA50_WINDOW,
        ma200_window: int = _MA200_WINDOW,
        recent_return_window: int = _RECENT_RETURN_WINDOW,
        volatility_window: int = _VOLATILITY_WINDOW,
        high_window: int = _HIGH_WINDOW,
    ) -> None:
        self.ma50_window = ma50_window
        self.ma200_window = ma200_window
        self.recent_return_window = recent_return_window
        self.volatility_window = volatility_window
        self.high_window = high_window

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return a new DataFrame with all technical features appended.

        The input DataFrame is not modified.
        """
        required = {"date", "tic", "close"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"TechnicalFeatureBuilder: missing required columns: {missing}"
            )

        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(["tic", "date"]).reset_index(drop=True)

        df = df.groupby("tic", group_keys=False).apply(self._enrich_ticker)
        df = df.sort_values(["date", "tic"]).reset_index(drop=True)
        return df

    def build_latest_snapshot(
        self, df: pd.DataFrame, as_of_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Return one row per ticker representing the most-recent state
        on or before *as_of_date* (defaults to the last available date).
        """
        enriched = self.build(df)
        enriched["date"] = pd.to_datetime(enriched["date"])

        if as_of_date is not None:
            cutoff = pd.Timestamp(as_of_date)
            enriched = enriched[enriched["date"] <= cutoff]

        return (
            enriched.sort_values("date").groupby("tic").tail(1).reset_index(drop=True)
        )

    # ------------------------------------------------------------------
    # Per-ticker computation
    # ------------------------------------------------------------------

    def _enrich_ticker(self, g: pd.DataFrame) -> pd.DataFrame:
        g = g.copy().sort_values("date")
        close = g["close"]

        # --- Moving averages ---
        g["MA50"] = close.rolling(self.ma50_window, min_periods=1).mean()
        g["MA200"] = close.rolling(self.ma200_window, min_periods=1).mean()

        # --- Price vs moving averages (fractional deviation) ---
        g["price_vs_ma50"] = (close - g["MA50"]) / g["MA50"].replace(0, np.nan)
        g["price_vs_ma200"] = (close - g["MA200"]) / g["MA200"].replace(0, np.nan)

        # --- Price vs FinRL SMAs (if available) ---
        if "close_30_sma" in g.columns:
            g["price_vs_sma30"] = (close - g["close_30_sma"]) / g[
                "close_30_sma"
            ].replace(0, np.nan)
        else:
            sma30 = close.rolling(30, min_periods=1).mean()
            g["price_vs_sma30"] = (close - sma30) / sma30.replace(0, np.nan)

        if "close_60_sma" in g.columns:
            g["price_vs_sma60"] = (close - g["close_60_sma"]) / g[
                "close_60_sma"
            ].replace(0, np.nan)
        else:
            sma60 = close.rolling(60, min_periods=1).mean()
            g["price_vs_sma60"] = (close - sma60) / sma60.replace(0, np.nan)

        # --- Recent return ---
        g["recent_return"] = close.pct_change(self.recent_return_window)

        # --- Volatility score (normalised rolling std of log-returns) ---
        log_ret = np.log(close / close.shift(1))
        rolling_std = log_ret.rolling(self.volatility_window, min_periods=5).std()
        annualised_vol = rolling_std * np.sqrt(252)
        # Normalise to [0, 1] range via soft clip; 0 = low vol, 1 = very high vol
        g["volatility_score"] = (annualised_vol / 1.0).clip(0, 1)

        # --- Drawdown from recent high ---
        rolling_high = close.rolling(self.high_window, min_periods=1).max()
        g["drawdown_from_recent_high"] = (close - rolling_high) / rolling_high.replace(
            0, np.nan
        )

        # --- Momentum score [-1, 1] ---
        # Combines: recent_return, price_vs_ma50, price_vs_ma200, optional rsi/macd
        g["momentum_score"] = self._compute_momentum_score(g)

        # --- Mean reversion score [-1, 1] ---
        # Positive = oversold (potential buy), negative = overbought (stretched)
        g["mean_reversion_score"] = self._compute_mean_reversion_score(g)

        return g

    def _compute_momentum_score(self, g: pd.DataFrame) -> pd.Series:
        """
        Composite momentum score in [-1, 1].

        Components (each clipped to [-1, 1] before weighting):
          0.40 * recent_return (20-day, normalised)
          0.30 * price_vs_ma50 (sign + magnitude)
          0.30 * price_vs_ma200 (longer trend)
          Optional MACD confirmation added if present.
        """
        w_ret, w_ma50, w_ma200 = 0.40, 0.30, 0.30

        ret_norm = (g["recent_return"] / 0.20).clip(-1, 1)  # ±20% normaliser
        vs_ma50_norm = (g["price_vs_ma50"] / 0.15).clip(-1, 1)  # ±15% normaliser
        vs_ma200_norm = (g["price_vs_ma200"] / 0.20).clip(-1, 1)

        score = w_ret * ret_norm + w_ma50 * vs_ma50_norm + w_ma200 * vs_ma200_norm

        if "macd" in g.columns:
            close_mean = g["close"].rolling(20, min_periods=1).mean().replace(0, np.nan)
            macd_norm = (g["macd"] / close_mean * 10).clip(-1, 1).fillna(0)
            # Blend in MACD with small weight — rescale the existing components
            score = 0.85 * score + 0.15 * macd_norm

        return score.clip(-1, 1)

    def _compute_mean_reversion_score(self, g: pd.DataFrame) -> pd.Series:
        """
        Mean reversion score in [-1, 1].

        Positive = potential buy (oversold relative to trends).
        Negative = stretched upward (overbought relative to trends).

        Primarily derived from price deviation from moving averages and RSI.
        """
        # Negative price_vs_ma50 → potential mean reversion upward → positive score
        base = (-g["price_vs_ma50"] / 0.15).clip(-1, 1)

        if "rsi_30" in g.columns:
            # RSI < 30 → oversold → +1, RSI > 70 → overbought → -1
            rsi_norm = ((50.0 - g["rsi_30"]) / 50.0).clip(-1, 1)
            base = 0.60 * base + 0.40 * rsi_norm

        if "cci_30" in g.columns:
            # CCI < -100 → oversold, CCI > +100 → overbought
            cci_norm = ((-g["cci_30"]) / 200.0).clip(-1, 1)
            base = 0.80 * base + 0.20 * cci_norm

        return base.clip(-1, 1)
