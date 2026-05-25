"""Small W&B setup check for StockInvestmentDSS."""

from __future__ import annotations

from stock_investment_dss.experiment_tracking.wandb_tracking import (
    finish_wandb_run,
    init_wandb_run,
    is_wandb_enabled,
    wandb_log,
)


def main() -> None:
    print(f"W&B enabled: {is_wandb_enabled()}")
    run = init_wandb_run(
        run_name="wandb_setup_check",
        group="setup",
        job_type="diagnostic",
        tags=["setup", "diagnostic"],
        config={"check": "wandb_setup_check"},
    )
    wandb_log({"setup_check/value": 1.0}, step=1)
    finish_wandb_run()
    print("W&B setup check completed.")
    if run is None:
        print("No W&B run was created because STOCK_INVESTMENT_DSS_WANDB_ENABLED is not true.")


if __name__ == "__main__":
    main()
