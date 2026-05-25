# src/stock_investment_dss/runner/run_iqn_experiment_from_config.py
"""
Thin adapter: run the existing IQN multiseed launcher from an experiment manifest.

Usage:
    python -m stock_investment_dss.runner.run_iqn_experiment_from_config \\
        --config configs/experiments/clean_25k_baseline_v1.json

What this does:
1. Loads the manifest JSON.
2. Clears all IQN hyperparameter env vars from os.environ.
3. Sets os.environ from manifest["env_overrides"].
4. Unsets env vars from manifest["env_unset"].
5. Runs verify_experiment_config logic inline (same assertions).
6. If verification passes, calls run_iqn_learning_curve_multiseed_launcher.main().

This script does NOT contain any training logic of its own.
It is a clean environment setup shim around the existing launcher.
The existing launcher (run_iqn_learning_curve_multiseed_launcher) is
the execution backend and its code is not modified.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# All IQN hyperparameter env vars to clear before loading manifest.
_IQN_HYPERPARAMETER_ENV_VARS = [
    "STOCK_INVESTMENT_DSS_IQN_BATCH_SIZE",
    "STOCK_INVESTMENT_DSS_IQN_REPLAY_CAPACITY",
    "STOCK_INVESTMENT_DSS_IQN_TARGET_UPDATE_INTERVAL",
    "STOCK_INVESTMENT_DSS_IQN_NUM_TAU_SAMPLES",
    "STOCK_INVESTMENT_DSS_IQN_NUM_TAU_PRIME_SAMPLES",
    "STOCK_INVESTMENT_DSS_IQN_NUM_ACTION_QUANTILES",
    "STOCK_INVESTMENT_DSS_IQN_EPSILON_START",
    "STOCK_INVESTMENT_DSS_IQN_EPSILON_FINAL",
    "STOCK_INVESTMENT_DSS_IQN_EPSILON_DECAY_STEPS",
    "STOCK_INVESTMENT_DSS_IQN_EPSILON_EVAL",
    "STOCK_INVESTMENT_DSS_IQN_HIDDEN_DIM",
    "STOCK_INVESTMENT_DSS_IQN_COSINE_EMBEDDING_DIM",
    "STOCK_INVESTMENT_DSS_IQN_LEARNING_RATE",
    "STOCK_INVESTMENT_DSS_IQN_LR",
    "STOCK_INVESTMENT_DSS_IQN_GAMMA",
    "STOCK_INVESTMENT_DSS_IQN_KAPPA",
    "STOCK_INVESTMENT_DSS_IQN_GRAD_CLIP_NORM",
    "STOCK_INVESTMENT_DSS_IQN_USE_LAYER_NORM",
    "STOCK_INVESTMENT_DSS_IQN_CONFIG_PRESET",
    "STOCK_INVESTMENT_DSS_IQN_TOTAL_STEPS",
    "STOCK_INVESTMENT_DSS_IQN_LEARNING_STARTS",
    "STOCK_INVESTMENT_DSS_IQN_BACKTEST_SCORE_MODE",
    "STOCK_INVESTMENT_DSS_IQN_BACKTEST_RISK_LAMBDA",
    "STOCK_INVESTMENT_DSS_IQN_BACKTEST_NUM_QUANTILES",
    "STOCK_INVESTMENT_DSS_IQN_EVAL_QUANTILES",
    "STOCK_INVESTMENT_DSS_IQN_BACKTEST_MAX_STEPS",
]


def _setup_clean_env_from_manifest(manifest: dict) -> list[str]:
    """Clear IQN env vars, apply manifest overrides. Returns list of cleared keys."""
    cleared = []
    for key in _IQN_HYPERPARAMETER_ENV_VARS:
        if key in os.environ:
            del os.environ[key]
            cleared.append(key)

    for key, value in manifest.get("env_overrides", {}).items():
        os.environ[key] = str(value)

    for key in manifest.get("env_unset", []):
        os.environ.pop(key, None)

    return cleared


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run IQN multiseed experiment from a manifest JSON."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to experiment manifest JSON",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        default=False,
        help="Skip config verification (not recommended for thesis experiments)",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[ERROR] Manifest not found: {config_path}", file=sys.stderr)
        return 1

    with config_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    experiment_id = manifest.get("experiment_id", config_path.stem)
    print(f"[run_iqn_experiment_from_config] Experiment: {experiment_id}")
    print(f"[run_iqn_experiment_from_config] Manifest:   {config_path}")

    # Step 1: Clean environment — must happen before any IQN config import.
    cleared = _setup_clean_env_from_manifest(manifest)
    if cleared:
        print(
            f"[run_iqn_experiment_from_config] Cleared {len(cleared)} residual IQN env vars."
        )
    print(
        f"[run_iqn_experiment_from_config] Applied {len(manifest.get('env_overrides', {}))} env overrides."
    )

    # Step 2: Verify config unless explicitly skipped.
    if not args.skip_verify:
        print("[run_iqn_experiment_from_config] Running config verification...")
        import subprocess

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "stock_investment_dss.runner.verify_experiment_config",
                "--config",
                str(config_path),
            ],
            env=os.environ.copy(),
        )
        if result.returncode != 0:
            print(
                "[run_iqn_experiment_from_config] Config verification FAILED. Aborting.",
                file=sys.stderr,
            )
            return 1
        print("[run_iqn_experiment_from_config] Config verification PASSED.")
    else:
        print("[run_iqn_experiment_from_config] WARNING: Config verification skipped.")

    # Step 3: Call existing launcher — no training logic here.
    from stock_investment_dss.runner.run_iqn_learning_curve_multiseed_launcher import (
        main as launcher_main,
    )

    print("[run_iqn_experiment_from_config] Delegating to multiseed launcher...")
    return launcher_main()


if __name__ == "__main__":
    raise SystemExit(main())
