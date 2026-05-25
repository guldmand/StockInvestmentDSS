# src/stock_investment_dss/environments/finrl_env_factory.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

REQUIRED_FINRL_ENV_COLUMNS = {"date", "tic", "close"}


@dataclass(frozen=True)
class FinRLStockTradingEnvConfig:
    initial_amount: int = 1_000_000
    hmax: int = 100
    buy_cost_pct: float = 0.001
    sell_cost_pct: float = 0.001
    reward_scaling: float = 1e-4
    print_verbosity: int = 10


def load_finrl_stock_trading_env_class():
    try:
        from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
    except ImportError as exc:
        raise ImportError(
            "Could not import FinRL StockTradingEnv. "
            "Make sure FinRL is installed in the active environment."
        ) from exc

    return StockTradingEnv


def prepare_finrl_stock_trading_dataframe(
    data: pd.DataFrame,
    tickers: tuple[str, ...],
) -> pd.DataFrame:
    frame = data.copy()

    missing_columns = REQUIRED_FINRL_ENV_COLUMNS - set(frame.columns)
    if missing_columns:
        raise ValueError(
            "FinRL StockTradingEnv data is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    requested_tickers = [ticker.upper().strip() for ticker in tickers]

    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["tic"] = frame["tic"].astype(str).str.upper().str.strip()
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")

    frame = frame.dropna(subset=["date", "tic", "close"])
    frame = frame[frame["tic"].isin(requested_tickers)].copy()

    if frame.empty:
        raise ValueError("FinRL StockTradingEnv data is empty after filtering.")

    frame["date"] = frame["date"].dt.strftime("%Y-%m-%d")

    ticker_order = {ticker: index for index, ticker in enumerate(requested_tickers)}
    frame["_ticker_order"] = frame["tic"].map(ticker_order)

    frame = frame.sort_values(["date", "_ticker_order"]).reset_index(drop=True)
    frame["_day_index"] = pd.factorize(frame["date"])[0]

    # FinRL StockTradingEnv expects df.loc[day, :] where day is 0, 1, 2, ...
    frame = frame.set_index("_day_index")
    frame.index.name = "day"

    frame = frame.drop(columns=["_ticker_order"])

    return frame


def infer_technical_indicators(
    data: pd.DataFrame,
    core_columns: set[str] | None = None,
) -> list[str]:
    if core_columns is None:
        core_columns = {
            "date",
            "tic",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "day",
        }

    indicators = [
        column
        for column in data.columns
        if column not in core_columns and pd.api.types.is_numeric_dtype(data[column])
    ]

    return indicators


def create_finrl_stock_trading_env(
    market_data: pd.DataFrame,
    tickers: tuple[str, ...],
    config: FinRLStockTradingEnvConfig | None = None,
    technical_indicators: list[str] | None = None,
):
    env_config = config or FinRLStockTradingEnvConfig()
    StockTradingEnv = load_finrl_stock_trading_env_class()

    prepared_data = prepare_finrl_stock_trading_dataframe(
        data=market_data,
        tickers=tickers,
    )

    if technical_indicators is None:
        technical_indicators = infer_technical_indicators(prepared_data)

    stock_dim = len(tickers)
    state_space = 1 + (2 * stock_dim) + (len(technical_indicators) * stock_dim)
    action_space = stock_dim

    env = StockTradingEnv(
        df=prepared_data,
        stock_dim=stock_dim,
        hmax=env_config.hmax,
        initial_amount=env_config.initial_amount,
        num_stock_shares=[0] * stock_dim,
        buy_cost_pct=[env_config.buy_cost_pct] * stock_dim,
        sell_cost_pct=[env_config.sell_cost_pct] * stock_dim,
        reward_scaling=env_config.reward_scaling,
        state_space=state_space,
        action_space=action_space,
        tech_indicator_list=technical_indicators,
        turbulence_threshold=None,
        risk_indicator_col="turbulence",
        make_plots=False,
        print_verbosity=env_config.print_verbosity,
        initial=True,
        model_name="",
        mode="",
        iteration="",
    )

    metadata = {
        "stock_dim": stock_dim,
        "tickers": list(tickers),
        "state_space": state_space,
        "action_space": action_space,
        "technical_indicators": technical_indicators,
        "trading_days": int(prepared_data.index.nunique()),
        "row_count": int(len(prepared_data)),
        "initial_amount": env_config.initial_amount,
        "hmax": env_config.hmax,
        "buy_cost_pct": env_config.buy_cost_pct,
        "sell_cost_pct": env_config.sell_cost_pct,
        "reward_scaling": env_config.reward_scaling,
    }

    return env, prepared_data, metadata


def make_zero_action(stock_dim: int) -> np.ndarray:
    return np.zeros(stock_dim, dtype=float)


def unpack_reset_result(reset_result: Any):
    if isinstance(reset_result, tuple) and len(reset_result) == 2:
        observation, info = reset_result
        return observation, info

    return reset_result, {}


def unpack_step_result(step_result: Any):
    if isinstance(step_result, tuple) and len(step_result) == 5:
        observation, reward, terminated, truncated, info = step_result
        done = bool(terminated or truncated)
        return observation, reward, done, info

    if isinstance(step_result, tuple) and len(step_result) == 4:
        observation, reward, done, info = step_result
        return observation, reward, bool(done), info

    raise ValueError(
        "Unsupported FinRL environment step return format. "
        f"Got: {type(step_result)} with value={step_result}"
    )


def extract_finrl_state_summary(
    state: list[float] | np.ndarray,
    tickers: tuple[str, ...],
) -> dict[str, Any]:
    state_list = list(state)
    stock_dim = len(tickers)

    cash = float(state_list[0])
    prices = state_list[1 : 1 + stock_dim]
    shares = state_list[1 + stock_dim : 1 + (2 * stock_dim)]

    holdings = {ticker: float(shares[index]) for index, ticker in enumerate(tickers)}

    price_map = {ticker: float(prices[index]) for index, ticker in enumerate(tickers)}

    stock_values = {ticker: holdings[ticker] * price_map[ticker] for ticker in tickers}

    portfolio_value = cash + sum(stock_values.values())

    if portfolio_value > 0:
        position_weights = {
            ticker: stock_values[ticker] / portfolio_value for ticker in tickers
        }
    else:
        position_weights = {ticker: 0.0 for ticker in tickers}

    return {
        "cash": cash,
        "prices": price_map,
        "holdings": holdings,
        "stock_values": stock_values,
        "portfolio_value": float(portfolio_value),
        "position_weights": position_weights,
    }
