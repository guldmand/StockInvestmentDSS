# src/stock_investment_dss/uncertainty/edl_action_dataset.py
"""
EDL Action Dataset Builder (v3.2)

Builds supervised datasets for training EDL action classifiers from:
1. Market data (technical/fundamental features)
2. HierarchicalDecisionPolicy audit outputs (where available)

Label modes
-----------
A. hindsight   — best realized risk-adjusted action over HORIZON_DAYS
B. rules       — rule/baseline policy labels
C. iqn_teacher — action selected by IQN + hierarchical policy

Point-in-time safety
--------------------
Input features at time t must only use information available at t.
Labels for EDL-A may use outcomes at t + horizon_days (supervised target only).
Labels are NEVER stored in the feature vector.

References
----------
See copilot-diagnostics/design/edl_uncertainty_poc/edl_v3_2_training_label_design.md
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from stock_investment_dss.uncertainty.edl_action_classes import (
    EDLActionConfig,
    action_to_idx,
    get_action_classes,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature column specification
# ---------------------------------------------------------------------------

# Technical features expected in market data CSV
TECHNICAL_FEATURE_COLS = [
    "macd",
    "rsi_30",
    "cci_30",
    "dx_30",
]

# Price trend features (derived or present in market data)
TREND_FEATURE_COLS = [
    "close",
    "MA50",
    "MA200",
]

# Fundamental features (may be present or absent — will use 0.5 default if absent)
FUNDAMENTAL_FEATURE_COLS = [
    "revenue_growth",
    "earnings_growth",
    "profit_margin",
    "pe_ratio",
    "ps_ratio",
    "free_cash_flow_yield",
    "debt_ratio",
]

# Portfolio / risk features (present in hierarchical audit if available)
PORTFOLIO_FEATURE_COLS = [
    "portfolio_cash_weight",
    "risk_adjusted_allocation_fraction",
]

# EDL feature names in order (must match feature vector produced by build_feature_row)
EDL_FEATURE_NAMES = (
    TECHNICAL_FEATURE_COLS
    + [
        "price_vs_ma50",
        "price_vs_ma200",
        "drawdown_from_recent_high",
        "recent_return_5d",
        "recent_return_20d",
        "volatility_20d",
    ]
    + FUNDAMENTAL_FEATURE_COLS
    + PORTFOLIO_FEATURE_COLS
)

EDL_INPUT_DIM = len(EDL_FEATURE_NAMES)


# ---------------------------------------------------------------------------
# Rule-based label policy (EDL-B)
# ---------------------------------------------------------------------------


@dataclass
class RuleBasedLabelConfig:
    """
    Configuration for the rule/baseline policy label generator.

    Uses actual market data columns available in market_data_full_500.csv:
    rsi_30, macd, drawdown_from_recent_high (computed), recent_return_20d (computed),
    volatility_20d (computed).
    """

    # SELL signals
    rsi_sell_min: float = 68.0  # RSI > this → overbought signal
    return_sell_threshold: float = -0.07  # 20d return < this → strong SELL
    drawdown_sell_threshold: float = -0.12  # drawdown < this → forced SELL

    # BUY signals
    rsi_buy_max: float = 38.0  # RSI < this → oversold signal
    return_buy_threshold: float = 0.05  # 20d return > this → momentum BUY
    macd_buy_min: float = 0.0  # macd > 0 confirms BUY

    # REBALANCE signals (high volatility/risk)
    volatility_rebalance_min: float = 0.022  # daily volatility > this → REBALANCE
    drawdown_rebalance_threshold: float = (
        -0.07
    )  # drawdown < this BUT > sell → REBALANCE

    default_label: str = "HOLD"


class RuleBasedLabelPolicy:
    """
    Generates EDL-B training labels using transparent rules on actual market data columns.

    Uses columns available in market_data_full_500.csv:
      rsi_30, macd — present in raw data
      drawdown_from_recent_high, recent_return_20d, volatility_20d — computed by _enrich_ticker_features

    Rules (in priority order):
    1. Very large drawdown → SELL
    2. High volatility + moderate drawdown → REBALANCE
    3. RSI oversold + MACD positive → BUY
    4. Strong positive momentum (20d return) → BUY
    5. RSI overbought + momentum negative → SELL
    6. Otherwise → HOLD
    """

    def __init__(self, cfg: Optional[RuleBasedLabelConfig] = None) -> None:
        self.cfg = cfg or RuleBasedLabelConfig()

    def generate_label(self, features: dict) -> str:
        """
        Generate a label from a market data feature dictionary.

        Parameters
        ----------
        features : dict
            Should contain columns from market_data_full_500.csv plus
            enriched columns computed by _enrich_ticker_features:
            rsi_30, macd, drawdown_from_recent_high, recent_return_20d, volatility_20d.

        Returns
        -------
        str: one of HOLD / BUY / SELL / REBALANCE
        """
        cfg = self.cfg

        def _f(key: str, default: float = 0.0) -> float:
            try:
                v = float(features.get(key, default))
                return v if math.isfinite(v) else default
            except (TypeError, ValueError):
                return default

        rsi = _f("rsi_30", 50.0)
        macd = _f("macd", 0.0)
        drawdown = _f("drawdown_from_recent_high", 0.0)
        ret_20d = _f("recent_return_20d", 0.0)
        vol_20d = _f("volatility_20d", 0.015)

        # Priority 1: strong drawdown → forced SELL
        if drawdown < cfg.drawdown_sell_threshold:
            return "SELL"

        # Priority 2: high volatility + moderate drawdown → REBALANCE
        if (
            vol_20d > cfg.volatility_rebalance_min
            and drawdown < cfg.drawdown_rebalance_threshold
        ):
            return "REBALANCE"

        # Priority 3: oversold RSI + positive MACD → BUY
        if rsi < cfg.rsi_buy_max and macd > cfg.macd_buy_min:
            return "BUY"

        # Priority 4: strong positive momentum → BUY (not overbought)
        if ret_20d > cfg.return_buy_threshold and rsi < 65.0:
            return "BUY"

        # Priority 5: overbought OR strongly negative return → SELL
        if rsi > cfg.rsi_sell_min and ret_20d < 0.0:
            return "SELL"
        if ret_20d < cfg.return_sell_threshold:
            return "SELL"

        return cfg.default_label


# ---------------------------------------------------------------------------
# Hindsight oracle label builder (EDL-A)
# ---------------------------------------------------------------------------


class HindsightOracleLabelBuilder:
    """
    Generates EDL-A oracle labels from market data.

    For each (ticker, date) at time t:
    - Computes forward return over HORIZON_DAYS for the same ticker
    - Label = BUY / SELL / HOLD based on realised forward return

    CRITICAL: labels use FUTURE data (t+h), but input features use ONLY data at t.
    This is safe for supervised training but must never be used at inference time.

    Works per-ticker to avoid cross-ticker index arithmetic bugs.
    """

    def __init__(
        self,
        horizon_days: int = 20,
        buy_threshold: float = 0.03,
        sell_threshold: float = -0.03,
        rebalance_volatility_threshold: float = 0.025,
    ) -> None:
        self.horizon_days = horizon_days
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.rebalance_volatility_threshold = rebalance_volatility_threshold

    def build_labels(
        self,
        market_df: pd.DataFrame,
        tickers: List[str],
        dates: List[str],
    ) -> pd.DataFrame:
        """
        Build per-(ticker, date) hindsight oracle labels.

        Parameters
        ----------
        market_df : pd.DataFrame
            Market data with columns: date, ticker, close, and optionally
            volatility_20d (from enrichment).
        tickers : list of str
        dates : list of ISO date strings (the subset to label)

        Returns
        -------
        pd.DataFrame with columns: date, ticker, label_A
        """
        dates_set = set(dates)
        logger.info(
            "Building hindsight oracle labels per ticker, horizon=%d days, "
            "buy_threshold=%.3f, sell_threshold=%.3f",
            self.horizon_days,
            self.buy_threshold,
            self.sell_threshold,
        )
        rows = []
        for ticker in tickers:
            ticker_df = (
                market_df[market_df["ticker"].str.upper() == ticker.upper()]
                .sort_values("date")
                .reset_index(drop=True)
            )
            if ticker_df.empty or "close" not in ticker_df.columns:
                continue

            for i, row in ticker_df.iterrows():
                date_str = str(row["date"])
                if date_str not in dates_set:
                    continue

                future_i = i + self.horizon_days
                if future_i >= len(ticker_df):
                    label = "HOLD"
                else:
                    current_close = float(row["close"])
                    future_close = float(ticker_df.iloc[future_i]["close"])
                    if current_close <= 0:
                        label = "HOLD"
                    else:
                        fwd_ret = (future_close - current_close) / current_close
                        vol = (
                            float(row.get("volatility_20d", 0.0))
                            if "volatility_20d" in row.index
                            else 0.0
                        )
                        if fwd_ret > self.buy_threshold:
                            label = "BUY"
                        elif fwd_ret < self.sell_threshold:
                            label = "SELL"
                        elif vol > self.rebalance_volatility_threshold:
                            label = "REBALANCE"
                        else:
                            label = "HOLD"

                rows.append(
                    {"date": date_str, "ticker": ticker.upper(), "label_A": label}
                )

        result = pd.DataFrame(rows)
        if not result.empty:
            dist = result["label_A"].value_counts().to_dict()
            logger.info("Hindsight oracle label distribution: %s", dist)
        return result


# ---------------------------------------------------------------------------
# Feature row builder
# ---------------------------------------------------------------------------


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        v = float(value)
        return v if math.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def build_feature_row(
    market_row: "pd.Series",
    portfolio_state: Optional[dict] = None,
    ma_window_50: Optional[float] = None,
    ma_window_200: Optional[float] = None,
) -> np.ndarray:
    """
    Build a normalised feature vector from a market data row.

    Parameters
    ----------
    market_row : pd.Series
        Row from market_data_full_500.csv (one ticker/date).
    portfolio_state : dict or None
        Portfolio snapshot for portfolio-level features.
    ma_window_50 / ma_window_200 : float or None
        Pre-computed MA50/MA200 values.

    Returns
    -------
    np.ndarray of shape (EDL_INPUT_DIM,)
    """
    p = portfolio_state or {}
    close = _safe_float(market_row.get("close", 0.0), 0.0)
    ma50 = ma_window_50 or _safe_float(market_row.get("MA50", close), close)
    ma200 = ma_window_200 or _safe_float(market_row.get("MA200", close), close)

    price_vs_ma50 = (close - ma50) / max(ma50, 1e-8)
    price_vs_ma200 = (close - ma200) / max(ma200, 1e-8)

    features = [
        # Technical
        _safe_float(market_row.get("macd", 0.0)),
        _safe_float(market_row.get("rsi_30", 50.0)) / 100.0,  # normalise to [0,1]
        _safe_float(market_row.get("cci_30", 0.0)) / 200.0,  # roughly [-1,1]
        _safe_float(market_row.get("dx_30", 20.0)) / 100.0,
        # Price vs trend
        price_vs_ma50,
        price_vs_ma200,
        _safe_float(market_row.get("drawdown_from_recent_high", 0.0)),
        _safe_float(market_row.get("recent_return_5d", 0.0)),
        _safe_float(market_row.get("recent_return_20d", 0.0)),
        _safe_float(market_row.get("volatility_20d", 0.02)),
        # Fundamentals (default 0.0 = neutral unknown)
        _safe_float(market_row.get("revenue_growth", 0.0)),
        _safe_float(market_row.get("earnings_growth", 0.0)),
        _safe_float(market_row.get("profit_margin", 0.0)),
        _safe_float(market_row.get("pe_ratio", 0.0)),
        _safe_float(market_row.get("ps_ratio", 0.0)),
        _safe_float(market_row.get("free_cash_flow_yield", 0.0)),
        _safe_float(market_row.get("debt_ratio", 0.0)),
        # Portfolio
        _safe_float(p.get("portfolio_cash_weight", 0.8)),
        _safe_float(p.get("risk_adjusted_allocation_fraction", 0.25)),
    ]
    return np.array(features, dtype=np.float32)


# ---------------------------------------------------------------------------
# Feature enrichment — compute derived features per ticker from close prices
# ---------------------------------------------------------------------------


def _enrich_ticker_features(mdf: pd.DataFrame) -> pd.DataFrame:
    """
    Compute derived technical features per ticker using rolling windows.

    Adds columns:
        MA50, MA200, drawdown_from_recent_high,
        recent_return_5d, recent_return_20d, volatility_20d

    Uses close_30_sma and close_60_sma as MA50/MA200 proxies if available,
    otherwise computes rolling(50) and rolling(200) from close.

    Point-in-time safe: all features use only past data at time t.
    """
    parts = []
    for ticker, tdf in mdf.groupby("ticker", sort=False):
        tdf = tdf.sort_values("date").copy()
        c = tdf["close"]

        # MA50 / MA200 — prefer pre-computed SMA columns, fall back to rolling
        if "close_30_sma" in tdf.columns:
            tdf["MA50"] = tdf["close_30_sma"]
        else:
            tdf["MA50"] = c.rolling(50, min_periods=1).mean()

        if "close_60_sma" in tdf.columns:
            tdf["MA200"] = tdf["close_60_sma"]
        else:
            tdf["MA200"] = c.rolling(200, min_periods=1).mean()

        # Drawdown from recent 60-day high
        rolling_max = c.rolling(60, min_periods=1).max()
        tdf["drawdown_from_recent_high"] = (c / rolling_max - 1.0).fillna(0.0)

        # Recent returns
        tdf["recent_return_5d"] = c.pct_change(5).fillna(0.0)
        tdf["recent_return_20d"] = c.pct_change(20).fillna(0.0)

        # 20-day rolling daily volatility
        tdf["volatility_20d"] = (
            c.pct_change().rolling(20, min_periods=5).std().fillna(0.015)
        )

        parts.append(tdf)

    if not parts:
        return mdf
    return pd.concat(parts).sort_values(["date", "ticker"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Label distribution sanity check
# ---------------------------------------------------------------------------


def check_label_distribution(df: pd.DataFrame, split_name: str) -> dict:
    """
    Compute label distribution statistics and generate warnings.

    Returns
    -------
    dict with keys: split, n_unique_classes, majority_class, majority_pct,
                    distribution, warnings, label_quality_ok
    """
    if df is None or df.empty or "label_str" not in df.columns:
        return {
            "split": split_name,
            "n_unique_classes": 0,
            "majority_class": None,
            "majority_pct": 100.0,
            "distribution": {},
            "warnings": ["CRITICAL: no label_str column or empty DataFrame"],
            "label_quality_ok": False,
        }

    vc = df["label_str"].value_counts()
    total = len(df)
    n_unique = len(vc)
    majority_action = str(vc.index[0]) if n_unique > 0 else "HOLD"
    majority_pct = round(100.0 * vc.iloc[0] / max(total, 1), 1)
    distribution = {str(k): int(v) for k, v in vc.items()}

    warnings = []
    label_quality_ok = True

    if n_unique < 2:
        warnings.append(
            f"CRITICAL: split '{split_name}' has only {n_unique} unique label class(es) "
            f"— dataset not suitable for EDL multi-class training."
        )
        label_quality_ok = False
    elif majority_pct > 90.0:
        warnings.append(
            f"CRITICAL: split '{split_name}' majority class '{majority_action}' = {majority_pct}% "
            f"(>90%) — severe class imbalance, not suitable for training without reweighting."
        )
        label_quality_ok = False
    elif majority_pct > 80.0:
        warnings.append(
            f"WARNING: split '{split_name}' majority class '{majority_action}' = {majority_pct}% "
            f"(>80%) — significant class imbalance. Consider class-weighted loss."
        )

    return {
        "split": split_name,
        "n_unique_classes": n_unique,
        "majority_class": majority_action,
        "majority_pct": majority_pct,
        "distribution": distribution,
        "warnings": warnings,
        "label_quality_ok": label_quality_ok,
    }


class EDLActionDataset:
    """
    Builds train/eval feature+label datasets for EDL action classifiers.

    Parameters
    ----------
    config : EDLActionConfig
    label_policy : RuleBasedLabelPolicy or None (used for label_mode='rules')
    oracle_builder : HindsightOracleLabelBuilder or None (for label_mode='hindsight')
    """

    def __init__(
        self,
        config: EDLActionConfig,
        label_policy: Optional[RuleBasedLabelPolicy] = None,
        oracle_builder: Optional[HindsightOracleLabelBuilder] = None,
    ) -> None:
        self.config = config
        self.label_policy = label_policy or RuleBasedLabelPolicy()
        self.oracle_builder = oracle_builder or HindsightOracleLabelBuilder(
            horizon_days=config.horizon_days
        )

    def build_from_market_data(
        self,
        market_csv_path: str,
        tickers: List[str],
        train_start: str,
        train_end: str,
        eval_start: str,
        eval_end: str,
        hierarchical_audit_csv: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Build train and eval datasets from frozen market data.

        Parameters
        ----------
        market_csv_path : str
            Path to market_data_full_500.csv.
        tickers : list of str
        train_start, train_end : ISO date strings
        eval_start, eval_end : ISO date strings
        hierarchical_audit_csv : str or None
            Path to hierarchical_decision_by_step.csv for EDL-C labels.

        Returns
        -------
        (train_df, eval_df)
        """
        logger.info("Loading market data from: %s", market_csv_path)
        mdf = self._load_market_data(market_csv_path, tickers)
        logger.info(
            "  %d rows loaded, %d unique dates", len(mdf), mdf["date"].nunique()
        )

        train_rows = self._build_rows(
            mdf, tickers, train_start, train_end, hierarchical_audit_csv
        )
        eval_rows = self._build_rows(
            mdf, tickers, eval_start, eval_end, hierarchical_audit_csv
        )

        train_df = pd.DataFrame(train_rows)
        eval_df = pd.DataFrame(eval_rows)

        logger.info("Dataset: train=%d rows, eval=%d rows", len(train_df), len(eval_df))
        return train_df, eval_df

    def _load_market_data(self, path: str, tickers: List[str]) -> pd.DataFrame:
        df = pd.read_csv(path)
        # Normalise column names
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        if "tic" in df.columns and "ticker" not in df.columns:
            df = df.rename(columns={"tic": "ticker"})
        if "date" not in df.columns:
            raise ValueError("Market data CSV must have a 'date' column.")
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        # Filter to requested tickers
        if "ticker" in df.columns:
            tickers_upper = [t.upper() for t in tickers]
            df = df[df["ticker"].str.upper().isin(tickers_upper)]
        df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
        # Compute derived technical features per ticker
        df = _enrich_ticker_features(df)
        return df

    def _build_rows(
        self,
        mdf: pd.DataFrame,
        tickers: List[str],
        date_start: str,
        date_end: str,
        hierarchical_audit_csv: Optional[str],
    ) -> List[dict]:
        mask = (mdf["date"] >= date_start) & (mdf["date"] <= date_end)
        subset = mdf[mask].copy()

        # Guard: iqn_teacher requires an actual teacher CSV
        if self.config.label_mode == "iqn_teacher" and not hierarchical_audit_csv:
            raise ValueError(
                "iqn_teacher label mode requires a hierarchical audit CSV. "
                "Set env var STOCK_INVESTMENT_DSS_EDL_HIERARCHICAL_AUDIT_CSV to a "
                "valid hierarchical policy audit CSV from run_mode_b_repro_demo5_iqn.ps1, "
                "or switch label_mode to 'rules' or 'hindsight'."
            )

        # Load hindsight oracle labels if needed
        oracle_labels: Optional[pd.DataFrame] = None
        if self.config.label_mode == "hindsight":
            dates = sorted(subset["date"].unique().tolist())
            oracle_labels = self.oracle_builder.build_labels(mdf, tickers, dates)

        # Load IQN teacher labels if audit CSV provided
        teacher_labels: Optional[pd.DataFrame] = None
        if self.config.label_mode == "iqn_teacher" and hierarchical_audit_csv:
            try:
                teacher_labels = pd.read_csv(hierarchical_audit_csv, dtype=str)
                if teacher_labels.empty or "date" not in teacher_labels.columns:
                    raise ValueError("Teacher CSV is empty or missing 'date' column.")
                logger.info(
                    "Loaded teacher labels: %d rows from %s",
                    len(teacher_labels),
                    hierarchical_audit_csv,
                )
            except Exception as e:
                raise RuntimeError(
                    f"iqn_teacher labels unavailable: could not read teacher CSV: {e}"
                ) from e

        rows = []
        for _, row in subset.iterrows():
            date_str = str(row.get("date", ""))
            ticker = str(row.get("ticker", ""))

            feat = build_feature_row(row)

            # Determine label — pass ticker for per-ticker matching
            label_str = self._generate_label(
                row, date_str, ticker, oracle_labels, teacher_labels
            )

            try:
                label_idx = action_to_idx(
                    label_str, self.config.include_change_strategy
                )
            except ValueError:
                label_idx = 0  # HOLD fallback

            record: dict = {
                "date": date_str,
                "ticker": ticker,
                "label": label_idx,
                "label_str": label_str,
                "label_mode": self.config.label_mode,
            }
            for i, name in enumerate(EDL_FEATURE_NAMES):
                record[f"feat_{name}"] = float(feat[i])

            rows.append(record)

        return rows

    def _generate_label(
        self,
        row: "pd.Series",
        date_str: str,
        ticker: str,
        oracle_labels: Optional[pd.DataFrame],
        teacher_labels: Optional[pd.DataFrame],
    ) -> str:
        mode = self.config.label_mode

        if mode == "hindsight" and oracle_labels is not None:
            # Match on (date, ticker) — per-ticker oracle
            mask = (oracle_labels["date"] == date_str) & (
                oracle_labels["ticker"].str.upper() == ticker.upper()
            )
            match = oracle_labels[mask]
            if not match.empty:
                return str(match.iloc[0]["label_A"])
            # Fall back to date-only match if ticker not in oracle (shouldn't happen)
            match_date = oracle_labels[oracle_labels["date"] == date_str]
            if not match_date.empty:
                return str(match_date.iloc[0]["label_A"])
            return "HOLD"

        if mode == "iqn_teacher" and teacher_labels is not None:
            match = teacher_labels[teacher_labels["date"] == date_str]
            if not match.empty:
                return str(match.iloc[0].get("selected_action_type", "HOLD"))
            return "HOLD"

        # mode == "rules" (or fallback)
        features_dict = row.to_dict()
        return self.label_policy.generate_label(features_dict)

    def feature_names(self) -> List[str]:
        return [f"feat_{n}" for n in EDL_FEATURE_NAMES]

    @staticmethod
    def feature_columns_from_df(df: pd.DataFrame) -> List[str]:
        """Return feature column names from a built dataset DataFrame."""
        return [c for c in df.columns if c.startswith("feat_")]

    @staticmethod
    def to_numpy(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract (X, y) numpy arrays from a built dataset DataFrame.

        Returns
        -------
        X : np.ndarray of shape (N, d) — float32
        y : np.ndarray of shape (N,)   — int64 (class indices)
        """
        feat_cols = EDLActionDataset.feature_columns_from_df(df)
        X = df[feat_cols].values.astype(np.float32)
        y = df["label"].values.astype(np.int64)
        return X, y
