"""
Smoke test for the FinRLDiscreteEnv.

Run from project root with:

PowerShell:
    $env:PYTHONPATH="src"
    python -m stockdss.rl.experiments.test_finrl_discrete_env
"""

from __future__ import annotations

from stockdss.envs.finrl_discrete_env import (
    FinRLDiscreteEnv,
    FinRLDiscreteEnvConfig,
)


def main() -> None:
    config = FinRLDiscreteEnvConfig(
        csv_path="train_data.csv",
        ticker="AAPL",
    )

    env = FinRLDiscreteEnv(config)

    obs, info = env.reset()

    separator = "=" * 115
    table_separator = "-" * 115

    print(separator)
    print("FinRLDiscreteEnv smoke test")
    print(separator)
    print("Initial observation shape:", obs.shape)
    print("Initial info:", info)
    print("Action space:", env.action_space)
    print("Observation space:", env.observation_space)
    print(separator)

    print(
        f"{'step':>4} | "
        f"{'date':<10} | "
        f"{'action':<8} | "
        f"{'reward':>10} | "
        f"{'raw_reward':>12} | "
        f"{'cost':>10} | "
        f"{'portfolio':>14} | "
        f"{'cash':>14} | "
        f"{'shares':>8}"
    )
    print(table_separator)

    actions = [
        env.BUY_50,
        env.HOLD,
        env.BUY_25,
        env.HOLD,
        env.SELL_50,
        env.HOLD,
        env.SELL_100,
    ]

    for step, action in enumerate(actions):
        obs, reward, terminated, truncated, info = env.step(action)

        print(
            f"{step:>4} | "
            f"{info['date']:<10} | "
            f"{info['action_name']:<8} | "
            f"{reward:>10.6f} | "
            f"{info['raw_reward']:>12.2f} | "
            f"{info['transaction_cost']:>10.2f} | "
            f"{info['portfolio_value']:>14.2f} | "
            f"{info['cash']:>14.2f} | "
            f"{info['shares_held']:>8}"
        )

        if terminated or truncated:
            break

    env.close()
    print(separator)
    print("Smoke test done")
    print(separator)


if __name__ == "__main__":
    main()
