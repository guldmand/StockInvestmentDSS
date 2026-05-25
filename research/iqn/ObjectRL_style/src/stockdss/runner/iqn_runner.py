"""
StockDSS IQN Runner.

Master runner for the current PIT milestone.

Runs:

A) FinRL baseline pipeline
   1. Train FinRL baselines on PIT train data
   2. Backtest FinRL baselines on PIT trade data
   3. Visualize FinRL baseline backtest

B) D-IQN-DSS pipeline
   4. Train custom IQN on PIT train data for one ticker
   5. Backtest custom IQN on PIT trade data for one ticker
   6. Visualize IQN decision distribution

Everything is written into one central run folder:

outputs/runs/<run_id>/
├─ run_config.json
├─ run_commands.ps1
├─ run_summary.json
├─ baseline_finrl/
│  ├─ models/
│  ├─ logs/
│  ├─ files/
│  │  ├─ train/
│  │  └─ backtest/
│  └─ plots/
└─ iqn_finrl/
   ├─ models/
   ├─ files/
   │  ├─ train/
   │  └─ backtest/
   ├─ plots/
   └─ visualizations/

Example:

PowerShell:
    $env:PYTHONPATH="src"

    python -m stockdss.runner.iqn_runner `
      --train-data data/train_data_pit_500_2026_01_01.csv `
      --trade-data data/trade_data_pit_500_2026_01_01.csv `
      --dataset-tag pit_500_2026_01_01 `
      --ticker AAPL `
      --finrl-timesteps 500 `
      --iqn-steps 5000 `
      --risk-lambda 0.75 `
      --agents a2c,ddpg,ppo,td3,sac `
      --use-mvo
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# -----------------------------------------------------------------------------
# Args
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run full StockDSS PIT baseline + IQN pipeline."
    )

    parser.add_argument(
        "--train-data",
        required=True,
        help="PIT train CSV path.",
    )

    parser.add_argument(
        "--trade-data",
        required=True,
        help="PIT trade CSV path.",
    )

    parser.add_argument(
        "--dataset-tag",
        required=True,
        help="Dataset tag, e.g. pit_500_2026_01_01.",
    )

    parser.add_argument(
        "--ticker",
        default="AAPL",
        help="Single ticker for IQN training/backtest. Default: AAPL.",
    )

    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional full run id. If omitted, one is generated.",
    )

    parser.add_argument(
        "--finrl-timesteps",
        type=int,
        default=500,
        help="Timesteps for FinRL baseline agents. Default: 500.",
    )

    parser.add_argument(
        "--iqn-steps",
        type=int,
        default=5_000,
        help="Training steps for custom IQN. Default: 5000.",
    )

    parser.add_argument(
        "--agents",
        default="a2c,ddpg,ppo,td3,sac",
        help="Comma-separated FinRL agents. Default: a2c,ddpg,ppo,td3,sac.",
    )

    parser.add_argument(
        "--risk-lambda",
        type=float,
        default=0.75,
        help="Risk lambda for IQN decision scoring. Default: 0.75.",
    )

    parser.add_argument(
        "--initial-amount",
        type=float,
        default=1_000_000.0,
        help="Initial portfolio value. Default: 1000000.",
    )

    parser.add_argument(
        "--use-mvo",
        action="store_true",
        help="Include MVO baseline in FinRL backtest.",
    )

    parser.add_argument(
        "--use-dji",
        action="store_true",
        help="Include DJI baseline if supported by backtest script.",
    )

    parser.add_argument(
        "--skip-summary",
        action="store_true",
        help="Skip automatic final result summary.",
    )

    parser.add_argument(
        "--show-summary",
        action="store_true",
        help="Show final summary dashboard window.",
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Show matplotlib windows in child visualization scripts.",
    )

    parser.add_argument(
        "--skip-finrl",
        action="store_true",
        help="Skip FinRL baseline pipeline.",
    )

    parser.add_argument(
        "--skip-iqn",
        action="store_true",
        help="Skip IQN pipeline.",
    )

    parser.add_argument(
        "--skip-visualizations",
        action="store_true",
        help="Skip visualization scripts.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print commands and write run files. Do not execute.",
    )

    return parser.parse_args()


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def make_run_id(args: argparse.Namespace) -> str:
    if args.run_name:
        return args.run_name.strip().replace(" ", "_")

    timestamp = datetime.now().strftime("%Y_%m_%d_%H%M")
    short_dataset = args.dataset_tag.replace("pit_500_2026_01_01", "pit500_20260101")
    ticker = args.ticker.lower()

    return (
        f"{timestamp}_runner_{short_dataset}_{ticker}_"
        f"f{args.finrl_timesteps}_i{args.iqn_steps}"
    )


def save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, default=str)


def run_command(
    command: list[str],
    label: str,
    dry_run: bool,
) -> dict[str, Any]:
    print()
    print("=" * 100)
    print(label)
    print("=" * 100)
    print(" ".join(command))

    if dry_run:
        return {
            "label": label,
            "command": command,
            "returncode": None,
            "status": "dry_run",
        }

    completed = subprocess.run(command, shell=False)

    status = "ok" if completed.returncode == 0 else "failed"

    return {
        "label": label,
        "command": command,
        "returncode": completed.returncode,
        "status": status,
    }


def write_run_commands(
    commands: list[dict[str, Any]],
    path: Path,
) -> None:
    lines = [
        '$env:PYTHONPATH="src"',
        "",
    ]

    for index, item in enumerate(commands, start=1):
        label = item["label"]
        command = item["command"]

        lines.append(f"# {index}. {label}")
        lines.append(" ".join(command))
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def create_run_folders(run_root: Path) -> dict[str, Path]:
    folders = {
        "run_root": run_root,
        "baseline_root": run_root / "baseline_finrl",
        "baseline_models": run_root / "baseline_finrl" / "models",
        "baseline_logs": run_root / "baseline_finrl" / "logs",
        "baseline_files": run_root / "baseline_finrl" / "files",
        "baseline_train_files": run_root / "baseline_finrl" / "files" / "train",
        "baseline_backtest_files": run_root / "baseline_finrl" / "files" / "backtest",
        "baseline_plots": run_root / "baseline_finrl" / "plots",
        "baseline_visualizations": run_root / "baseline_finrl" / "visualizations",
        "iqn_root": run_root / "iqn_finrl",
        "iqn_models": run_root / "iqn_finrl" / "models",
        "iqn_files": run_root / "iqn_finrl" / "files",
        "iqn_train_files": run_root / "iqn_finrl" / "files" / "train",
        "iqn_backtest_files": run_root / "iqn_finrl" / "files" / "backtest",
        "iqn_plots": run_root / "iqn_finrl" / "plots",
        "iqn_visualizations": run_root / "iqn_finrl" / "visualizations",
        "iqn_decision_visualizations": (
            run_root / "iqn_finrl" / "visualizations" / "iqn_decision_distribution"
        ),
    }

    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)

    return folders


def find_price_from_trade_data(
    trade_data_path: Path,
    ticker: str,
) -> float:
    try:
        import pandas as pd

        df = pd.read_csv(trade_data_path)

        if "tic" not in df.columns or "close" not in df.columns:
            return 0.0

        ticker_df = df[df["tic"] == ticker].copy()

        if ticker_df.empty:
            return 0.0

        if "date" in ticker_df.columns:
            ticker_df["date"] = pd.to_datetime(ticker_df["date"])
            ticker_df = ticker_df.sort_values("date")

        return float(ticker_df["close"].iloc[-1])

    except Exception:
        return 0.0


def append_common_money_args(
    command: list[str],
    args: argparse.Namespace,
) -> list[str]:
    command.extend(
        [
            "--initial-amount",
            str(args.initial_amount),
        ]
    )

    return command


def fail_if_needed(
    run_results: list[dict[str, Any]],
    run_root: Path,
    error_message: str,
) -> None:
    if run_results[-1]["status"] == "failed":
        save_json({"run_results": run_results}, run_root / "run_summary.json")
        raise RuntimeError(error_message)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    run_id = make_run_id(args)
    run_root = Path("outputs") / "runs" / run_id
    folders = create_run_folders(run_root)

    dataset_tag = args.dataset_tag.strip()
    ticker = args.ticker.strip().upper()
    agents = args.agents.strip()

    baseline_run_name = f"{run_id}_base"
    iqn_train_run_name = f"{run_id}_iqn_train"
    iqn_backtest_run_name = f"{run_id}_iqn_backtest"

    baseline_model_dir = folders["baseline_models"]
    baseline_train_output_dir = folders["baseline_train_files"]
    baseline_backtest_output_dir = folders["baseline_backtest_files"]
    baseline_plots_dir = folders["baseline_plots"]

    iqn_model_dir = folders["iqn_models"]
    iqn_model_path = iqn_model_dir / "iqn_agent.pt"
    iqn_train_output_dir = folders["iqn_train_files"]
    iqn_backtest_output_dir = folders["iqn_backtest_files"]
    iqn_plots_dir = folders["iqn_plots"]
    iqn_decision_csv = iqn_backtest_output_dir / "iqn_action_estimates_last_day.csv"
    iqn_decision_visualization_dir = folders["iqn_decision_visualizations"]

    final_trade_price = find_price_from_trade_data(
        trade_data_path=Path(args.trade_data),
        ticker=ticker,
    )

    run_config = {
        "run_id": run_id,
        "run_root": str(run_root),
        "dataset_tag": dataset_tag,
        "ticker": ticker,
        "train_data": args.train_data,
        "trade_data": args.trade_data,
        "finrl_timesteps": args.finrl_timesteps,
        "iqn_steps": args.iqn_steps,
        "agents": agents,
        "risk_lambda": args.risk_lambda,
        "initial_amount": args.initial_amount,
        "use_mvo": args.use_mvo,
        "use_dji": args.use_dji,
        "show": args.show,
        "show_summary": args.show_summary,
        "skip_finrl": args.skip_finrl,
        "skip_iqn": args.skip_iqn,
        "skip_visualizations": args.skip_visualizations,
        "skip_summary": args.skip_summary,
        "dry_run": args.dry_run,
        "baseline_run_name": baseline_run_name,
        "iqn_train_run_name": iqn_train_run_name,
        "iqn_backtest_run_name": iqn_backtest_run_name,
        "baseline_train_output_dir": str(baseline_train_output_dir),
        "baseline_backtest_output_dir": str(baseline_backtest_output_dir),
        "baseline_model_dir": str(baseline_model_dir),
        "baseline_plots_dir": str(baseline_plots_dir),
        "iqn_train_output_dir": str(iqn_train_output_dir),
        "iqn_model_dir": str(iqn_model_dir),
        "iqn_model_path": str(iqn_model_path),
        "iqn_backtest_output_dir": str(iqn_backtest_output_dir),
        "iqn_plots_dir": str(iqn_plots_dir),
        "iqn_decision_csv": str(iqn_decision_csv),
        "iqn_decision_visualization_dir": str(iqn_decision_visualization_dir),
        "final_trade_price": final_trade_price,
    }

    save_json(run_config, run_root / "run_config.json")

    commands_to_write: list[dict[str, Any]] = []
    run_results: list[dict[str, Any]] = []

    print("=" * 100)
    print("StockDSS IQN Runner")
    print("=" * 100)
    print(f"Run id:       {run_id}")
    print(f"Run root:     {run_root}")
    print(f"Dataset tag:  {dataset_tag}")
    print(f"Ticker:       {ticker}")
    print("=" * 100)

    # ------------------------------------------------------------------
    # A1. Train FinRL baselines
    # ------------------------------------------------------------------
    if not args.skip_finrl:
        command = [
            sys.executable,
            "-m",
            "stockdss.rl.experiments.train_finrl_baselines_pit",
            "--train-data",
            args.train_data,
            "--dataset-tag",
            dataset_tag,
            "--run-name",
            baseline_run_name,
            "--total-timesteps",
            str(args.finrl_timesteps),
            "--agents",
            agents,
            "--run-root",
            str(run_root),
        ]

        command = append_common_money_args(command, args)

        commands_to_write.append(
            {
                "label": "Train FinRL baselines on PIT data",
                "command": command,
            }
        )

        run_results.append(
            run_command(
                command=command,
                label="A1. Train FinRL baselines on PIT data",
                dry_run=args.dry_run,
            )
        )

        fail_if_needed(
            run_results=run_results,
            run_root=run_root,
            error_message="FinRL baseline training failed.",
        )

        # ------------------------------------------------------------------
        # A2. Backtest FinRL baselines
        # ------------------------------------------------------------------
        command = [
            sys.executable,
            "-m",
            "stockdss.rl.experiments.backtest_finrl_baselines_pit",
            "--train-data",
            args.train_data,
            "--trade-data",
            args.trade_data,
            "--dataset-tag",
            dataset_tag,
            "--run-name",
            baseline_run_name,
            "--agents",
            agents,
            "--run-root",
            str(run_root),
        ]

        command = append_common_money_args(command, args)

        if args.use_mvo:
            command.append("--use-mvo")

        if args.use_dji:
            command.append("--use-dji")

        commands_to_write.append(
            {
                "label": "Backtest FinRL baselines on PIT data",
                "command": command,
            }
        )

        run_results.append(
            run_command(
                command=command,
                label="A2. Backtest FinRL baselines on PIT data",
                dry_run=args.dry_run,
            )
        )

        fail_if_needed(
            run_results=run_results,
            run_root=run_root,
            error_message="FinRL baseline backtest failed.",
        )

        # ------------------------------------------------------------------
        # A3. Visualize FinRL baselines
        # ------------------------------------------------------------------
        if not args.skip_visualizations:
            command = [
                sys.executable,
                "-m",
                "stockdss.rl.experiments.visualize_finrl_backtest_pit",
                "--dataset-tag",
                dataset_tag,
                "--run-name",
                baseline_run_name,
                "--run-root",
                str(run_root),
            ]

            if args.show:
                command.append("--show")

            commands_to_write.append(
                {
                    "label": "Visualize FinRL baseline backtest",
                    "command": command,
                }
            )

            run_results.append(
                run_command(
                    command=command,
                    label="A3. Visualize FinRL baseline backtest",
                    dry_run=args.dry_run,
                )
            )

            fail_if_needed(
                run_results=run_results,
                run_root=run_root,
                error_message="FinRL baseline visualization failed.",
            )

    # ------------------------------------------------------------------
    # B1. Train IQN
    # ------------------------------------------------------------------
    if not args.skip_iqn:
        command = [
            sys.executable,
            "-m",
            "stockdss.rl.experiments.train_iqn_finrl_pit_single_ticker",
            "--train-data",
            args.train_data,
            "--dataset-tag",
            dataset_tag,
            "--run-name",
            iqn_train_run_name,
            "--ticker",
            ticker,
            "--total-steps",
            str(args.iqn_steps),
            "--log-interval",
            "1000",
            "--save-every",
            "10000",
            "--run-root",
            str(run_root),
        ]

        command = append_common_money_args(command, args)

        commands_to_write.append(
            {
                "label": "Train D-IQN-DSS on PIT single ticker",
                "command": command,
            }
        )

        run_results.append(
            run_command(
                command=command,
                label="B1. Train D-IQN-DSS on PIT single ticker",
                dry_run=args.dry_run,
            )
        )

        fail_if_needed(
            run_results=run_results,
            run_root=run_root,
            error_message="IQN training failed.",
        )

        # ------------------------------------------------------------------
        # B2. Backtest IQN
        # ------------------------------------------------------------------
        command = [
            sys.executable,
            "-m",
            "stockdss.rl.experiments.backtest_iqn_pit_single_ticker",
            "--trade-data",
            args.trade_data,
            "--dataset-tag",
            dataset_tag,
            "--run-name",
            iqn_backtest_run_name,
            "--ticker",
            ticker,
            "--model-path",
            str(iqn_model_path),
            "--risk-lambda",
            str(args.risk_lambda),
            "--run-root",
            str(run_root),
        ]

        command = append_common_money_args(command, args)

        if args.show:
            command.append("--show")

        commands_to_write.append(
            {
                "label": "Backtest D-IQN-DSS on PIT single ticker",
                "command": command,
            }
        )

        run_results.append(
            run_command(
                command=command,
                label="B2. Backtest D-IQN-DSS on PIT single ticker",
                dry_run=args.dry_run,
            )
        )

        fail_if_needed(
            run_results=run_results,
            run_root=run_root,
            error_message="IQN backtest failed.",
        )

        # ------------------------------------------------------------------
        # B3. Visualize IQN decision distribution
        # ------------------------------------------------------------------
        if not args.skip_visualizations:
            command = [
                sys.executable,
                "-m",
                "stockdss.rl.experiments.visualize_iqn_decision_distribution",
                "--decision-csv",
                str(iqn_decision_csv),
                "--date",
                "last_trade_day",
                "--ticker",
                ticker,
                "--price",
                str(final_trade_price),
                "--risk-lambda",
                str(args.risk_lambda),
                "--output-dir",
                str(iqn_decision_visualization_dir),
            ]

            if args.show:
                command.append("--show")

            commands_to_write.append(
                {
                    "label": "Visualize D-IQN-DSS decision distribution",
                    "command": command,
                }
            )

            run_results.append(
                run_command(
                    command=command,
                    label="B3. Visualize D-IQN-DSS decision distribution",
                    dry_run=args.dry_run,
                )
            )

            fail_if_needed(
                run_results=run_results,
                run_root=run_root,
                error_message="IQN decision visualization failed.",
            )

        # ------------------------------------------------------------------
        # C1. Summarize full runner result
        # ------------------------------------------------------------------
        if not args.skip_summary:
            command = [
                sys.executable,
                "-m",
                "stockdss.runner.summarize_run_results",
                "--run-root",
                str(run_root),
            ]

            if args.show_summary:
                command.append("--show")

            commands_to_write.append(
                {
                    "label": "Summarize full runner result",
                    "command": command,
                }
            )

            run_results.append(
                run_command(
                    command=command,
                    label="C1. Summarize full runner result",
                    dry_run=args.dry_run,
                )
            )

            fail_if_needed(
                run_results=run_results,
                run_root=run_root,
                error_message="Runner result summary failed.",
            )

    write_run_commands(commands_to_write, run_root / "run_commands.ps1")

    run_summary = {
        "run_id": run_id,
        "run_root": str(run_root),
        "status": "dry_run" if args.dry_run else "finished",
        "run_results": run_results,
        "central_folders": {key: str(value) for key, value in folders.items()},
        "model_paths": {
            "baseline_model_dir": str(baseline_model_dir),
            "iqn_model_path": str(iqn_model_path),
        },
        "output_dirs": {
            "baseline_train_output_dir": str(baseline_train_output_dir),
            "baseline_backtest_output_dir": str(baseline_backtest_output_dir),
            "baseline_plots_dir": str(baseline_plots_dir),
            "iqn_train_output_dir": str(iqn_train_output_dir),
            "iqn_backtest_output_dir": str(iqn_backtest_output_dir),
            "iqn_plots_dir": str(iqn_plots_dir),
            "iqn_decision_visualization_dir": str(iqn_decision_visualization_dir),
            "summary_dir": str(run_root / "summary"),
        },
    }

    save_json(run_summary, run_root / "run_summary.json")

    print()
    print("=" * 100)
    print("IQN Runner finished")
    print("=" * 100)
    print(f"Central run folder: {run_root.resolve()}")
    print()
    print("Key folders:")
    print(f"- Baseline FinRL models:         {folders['baseline_models']}")
    print(f"- Baseline FinRL logs:           {folders['baseline_logs']}")
    print(f"- Baseline FinRL train files:    {folders['baseline_train_files']}")
    print(f"- Baseline FinRL backtest files: {folders['baseline_backtest_files']}")
    print(f"- Baseline FinRL plots:          {folders['baseline_plots']}")
    print(f"- IQN FinRL models:              {folders['iqn_models']}")
    print(f"- IQN FinRL train files:         {folders['iqn_train_files']}")
    print(f"- IQN FinRL backtest files:      {folders['iqn_backtest_files']}")
    print(f"- IQN FinRL plots:               {folders['iqn_plots']}")
    print(f"- IQN FinRL visualizations:      {folders['iqn_visualizations']}")
    print(f"- Summary:                       {run_root / 'summary'}")
    print()
    print("Run files:")
    print(f"- {run_root / 'run_config.json'}")
    print(f"- {run_root / 'run_commands.ps1'}")
    print(f"- {run_root / 'run_summary.json'}")
    print("=" * 100)


if __name__ == "__main__":
    main()
