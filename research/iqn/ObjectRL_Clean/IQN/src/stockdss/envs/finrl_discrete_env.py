"""
Custom FinRL-Gymnasium-style discrete trading environment for IQN.
Idear:
- use FinRL-style csv data
- Trade exactly one ticker at a time
- Use a simple discrete Action space:
    0 = HOLD
    1 = BUY_25   use 25% of available cash
    2 = BUY_50   use 50% of available cash
    3 = BUY_100  use 100% of available cash
    4 = SELL_25  sell 25% of held shares
    5 = SELL_50  sell 50% of held shares
    6 = SELL_100 sell 100% of held shares

Note: This environment simplifed and acts as a bridge between CartPole and IQN for stock trading decision support system
"""

# Imports
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


@dataclass
class FinRLDiscreteEnvConfig:
    """environment configuration"""

    csv_path: str
    ticker: str
    initial_amount: float = 1_000_000.0
    buy_cost_pct: float = 0.01
    sell_cost_pct: float = 0.01

    # Reward scaling keeps rewards numerically smaller for RL.
    reward_scaling: float = 1e-4


class FinRLDiscreteEnv(gym.Env):
    """
    Minimal single-ticker discrete trading environment.

    Action space:
        0 = HOLD
        1 = BUY_25
        2 = BUY_50
        3 = BUY_100
        4 = SELL_25
        5 = SELL_50
        6 = SELL_100

    Observation:  [cash_ratio, shares_ratio, price_ratio, feature_1, ..., feature_n]
    Portfolio:    portfolio_value = cash + shares_held * current_price
    """

    metadata = {"render_modes": []}
    HOLD = 0
    BUY_25 = 1
    BUY_50 = 2
    BUY_100 = 3
    SELL_25 = 4
    SELL_50 = 5
    SELL_100 = 6

    def __init__(self, config: FinRLDiscreteEnvConfig):
        """Initialize the FinRLDiscreteEnv with the given configuration."""
        super().__init__()
        self.config = config
        self.df = self._load_and_prepare_data(
            csv_path=config.csv_path,
            ticker=config.ticker,
        )

        self.feature_columns = self._infer_feature_columns(self.df)

        # Precompute normalized feature matrix for stable neural network input.
        self.feature_matrix = self.df[self.feature_columns].to_numpy(dtype=np.float32)
        self.feature_matrix = np.nan_to_num(
            self.feature_matrix,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        self.feature_means = self.feature_matrix.mean(axis=0)
        self.feature_stds = self.feature_matrix.std(axis=0)

        # Avoid division by zero for constant columns.
        self.feature_stds = np.where(self.feature_stds == 0.0, 1.0, self.feature_stds)

        self.feature_matrix = (
            self.feature_matrix - self.feature_means
        ) / self.feature_stds

        self.initial_amount = float(config.initial_amount)
        self.buy_cost_pct = float(config.buy_cost_pct)
        self.sell_cost_pct = float(config.sell_cost_pct)
        self.reward_scaling = float(config.reward_scaling)

        self.action_space = spaces.Discrete(7)
        observation_dim = 3 + len(self.feature_columns)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(observation_dim,),
            dtype=np.float32,
        )
        self.day = 0
        self.cash = self.initial_amount
        self.shares_held = 0
        self.portfolio_value = self.initial_amount

    def _load_and_prepare_data(self, csv_path: str, ticker: str) -> pd.DataFrame:
        """Load and prepare the data for the specified ticker."""
        df = pd.read_csv(csv_path)
        if "tic" not in df.columns:
            raise ValueError("Expected column 'tic' in FinRL-style CSV data.")
        if "date" not in df.columns:
            raise ValueError("Expected column 'date' in FinRL-style CSV data.")
        if "close" not in df.columns:
            raise ValueError("Expected column 'close' in FinRL-style CSV data.")
        ticker_df = df[df["tic"] == ticker].copy()
        if ticker_df.empty:
            available = sorted(df["tic"].dropna().unique().tolist())
            raise ValueError(
                f"Ticker '{ticker}' not found in data. "
                f"Available examples: {available[:10]}"
            )
        ticker_df["date"] = pd.to_datetime(ticker_df["date"])
        ticker_df = ticker_df.sort_values("date").reset_index(drop=True)

        # Keep only rows with a valid close price.
        ticker_df = ticker_df.dropna(subset=["close"]).reset_index(drop=True)
        if len(ticker_df) < 2:
            raise ValueError(
                f"Ticker '{ticker}' has too few rows after cleaning: {len(ticker_df)}"
            )
        return ticker_df

    def _infer_feature_columns(self, df: pd.DataFrame) -> list[str]:
        """Infer which columns to use as features, excluding non-numeric and known non-feature columns."""
        excluded = {"date", "tic"}
        numeric_cols = (
            df.drop(columns=list(excluded), errors="ignore")
            .select_dtypes(include=[np.number])
            .columns.tolist()
        )

        # Keep close in features too, because it is useful signal.
        # The observation also includes price_ratio separately.
        return numeric_cols

    def _current_price(self) -> float:
        """Get the current price from the dataframe."""
        return float(self.df.loc[self.day, "close"])

    def _portfolio_value(self, price: float) -> float:
        """Calculate the current portfolio value."""
        return float(self.cash + self.shares_held * price)

    def _get_observation(self) -> np.ndarray:
        """Construct the observation vector for the current day."""
        row = self.df.loc[self.day]
        price = float(row["close"])
        cash_ratio = self.cash / self.initial_amount
        position_value = self.shares_held * price
        shares_ratio = position_value / self.initial_amount
        price_ratio = price / float(self.df.loc[0, "close"])
        features = self.feature_matrix[self.day]
        obs = np.concatenate(
            [
                np.array([cash_ratio, shares_ratio, price_ratio], dtype=np.float32),
                features.astype(np.float32),
            ]
        )

        return obs.astype(np.float32)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        """Reset the environment to the initial state."""
        super().reset(seed=seed)
        self.day = 0
        self.cash = self.initial_amount
        self.shares_held = 0
        self.portfolio_value = self.initial_amount
        obs = self._get_observation()
        info = {
            "date": str(self.df.loc[self.day, "date"].date()),
            "ticker": self.config.ticker,
            "cash": self.cash,
            "shares_held": self.shares_held,
            "portfolio_value": self.portfolio_value,
        }

        return obs, info

    def step(self, action: int):
        """Take an action in the environment and return the result."""
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")
        current_price = self._current_price()
        previous_value = self._portfolio_value(current_price)
        action_name = self._action_name(action)

        transaction_cost = 0.0

        # Handle actions: partial BUY, partial SELL, or HOLD.
        if action == self.BUY_25:
            transaction_cost = self._buy_fraction(current_price, fraction=0.25)
        elif action == self.BUY_50:
            transaction_cost = self._buy_fraction(current_price, fraction=0.50)
        elif action == self.BUY_100:
            transaction_cost = self._buy_fraction(current_price, fraction=1.00)
        elif action == self.SELL_25:
            transaction_cost = self._sell_fraction(current_price, fraction=0.25)
        elif action == self.SELL_50:
            transaction_cost = self._sell_fraction(current_price, fraction=0.50)
        elif action == self.SELL_100:
            transaction_cost = self._sell_fraction(current_price, fraction=1.00)
        # else {
        # HOLD = do nothing
        # }

        # Move to next day.
        self.day += 1

        terminated = self.day >= len(self.df) - 1
        truncated = False
        next_price = self._current_price()
        new_value = self._portfolio_value(next_price)
        self.portfolio_value = new_value
        # reward setup 1
        # raw_reward = new_value - previous_value
        # reward = raw_reward * self.reward_scaling

        # reward setup 2: percentage
        raw_reward = (new_value - previous_value) / previous_value
        reward = raw_reward
        obs = self._get_observation()
        info = {
            "date": str(self.df.loc[self.day, "date"].date()),
            "ticker": self.config.ticker,
            "action": int(action),
            "action_name": action_name,
            "price": next_price,
            "cash": self.cash,
            "shares_held": self.shares_held,
            "portfolio_value": self.portfolio_value,
            "raw_reward": raw_reward,
            "reward": reward,
            "transaction_cost": transaction_cost,
        }

        return obs, float(reward), terminated, truncated, info

    def _buy_fraction(self, price: float, fraction: float) -> float:
        """
        Buy shares using a fraction of available cash.

        Returns:
            transaction_cost paid for the trade.
        """
        if self.cash <= 0:
            return 0.0

        fraction = float(np.clip(fraction, 0.0, 1.0))

        cash_to_use = self.cash * fraction
        effective_price = price * (1.0 + self.buy_cost_pct)

        shares_to_buy = int(cash_to_use // effective_price)

        if shares_to_buy <= 0:
            return 0.0

        gross_trade_value = shares_to_buy * price
        transaction_cost = gross_trade_value * self.buy_cost_pct
        total_cost = gross_trade_value + transaction_cost

        if total_cost > self.cash:
            return 0.0

        self.cash -= total_cost
        self.shares_held += shares_to_buy

        return float(transaction_cost)

    def _sell_fraction(self, price: float, fraction: float) -> float:
        """
        Sell a fraction of currently held shares.

        Returns:
            transaction_cost paid for the trade.
        """
        if self.shares_held <= 0:
            return 0.0

        fraction = float(np.clip(fraction, 0.0, 1.0))

        shares_to_sell = int(self.shares_held * fraction)

        if fraction >= 1.0:
            shares_to_sell = self.shares_held

        if shares_to_sell <= 0:
            return 0.0

        gross_trade_value = shares_to_sell * price
        transaction_cost = gross_trade_value * self.sell_cost_pct
        proceeds = gross_trade_value - transaction_cost

        self.cash += proceeds
        self.shares_held -= shares_to_sell

        return float(transaction_cost)

    def _action_name(self, action: int) -> str:
        if action == self.HOLD:
            return "HOLD"
        if action == self.BUY_25:
            return "BUY_25"
        if action == self.BUY_50:
            return "BUY_50"
        if action == self.BUY_100:
            return "BUY_100"
        if action == self.SELL_25:
            return "SELL_25"
        if action == self.SELL_50:
            return "SELL_50"
        if action == self.SELL_100:
            return "SELL_100"
        return "UNKNOWN"
