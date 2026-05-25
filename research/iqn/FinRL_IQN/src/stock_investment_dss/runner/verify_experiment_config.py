# src/stock_investment_dss/runner/verify_experiment_config.py
"""
Verify that the effective runtime IQN config matches an experiment manifest.

Usage:
    python -m stock_investment_dss.runner.verify_experiment_config \\
        --config configs/experiments/clean_25k_baseline_v1.json

What this script does:
1. Loads the manifest JSON.
2. Removes all known IQN hyperparameter env vars from os.environ so no
   residual .env or shell values can bleed through.
3. Sets os.environ from manifest["env_overrides"].
4. Unsets env vars listed in manifest["env_unset"].
5. Calls build_iqn_config() — the same function the training pipeline uses.
6. Asserts every value in manifest["expected_iqn_config"] matches the result.
7. Prints a clear PASS / FAIL summary.
8. Writes outputs/config_checks/<experiment_id>_config_check.json.
9. Exits 0 on PASS, 1 on any mismatch.

Does NOT train. Does NOT import model, environment, or data-pipeline code.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# All IQN hyperparameter env vars that could contaminate a thesis experiment.
# These are forcibly cleared before loading manifest values.
# ---------------------------------------------------------------------------
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


def _clear_iqn_env_vars() -> list[str]:
    """Remove all IQN hyperparameter env vars. Returns list of cleared names."""
    cleared = []
    for key in _IQN_HYPERPARAMETER_ENV_VARS:
        if key in os.environ:
            del os.environ[key]
            cleared.append(key)
    return cleared


def _load_manifest(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _apply_env_overrides(manifest: dict) -> None:
    for key, value in manifest.get("env_overrides", {}).items():
        os.environ[key] = str(value)


def _apply_env_unset(manifest: dict) -> None:
    for key in manifest.get("env_unset", []):
        os.environ.pop(key, None)


def _check_value(field: str, expected, actual) -> tuple[bool, str]:
    if isinstance(expected, float):
        ok = abs(float(actual) - expected) < 1e-9
    elif isinstance(expected, bool):
        ok = bool(actual) == expected
    elif isinstance(expected, int):
        ok = int(actual) == expected
    else:
        ok = actual == expected

    status = "PASS" if ok else "FAIL"
    msg = f"  [{status}] {field}: expected={expected!r}  actual={actual!r}"
    return ok, msg


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify effective IQN config against an experiment manifest."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to experiment manifest JSON (e.g. configs/experiments/clean_25k_baseline_v1.json)",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/config_checks",
        help="Directory for the verification artifact (default: outputs/config_checks)",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[ERROR] Manifest not found: {config_path}", file=sys.stderr)
        return 1

    print(f"[verify_experiment_config] Loading manifest: {config_path}")
    manifest = _load_manifest(config_path)
    experiment_id = manifest.get("experiment_id", config_path.stem)

    # Step 1: Clear all IQN hyperparameter env vars to prevent contamination.
    cleared = _clear_iqn_env_vars()
    if cleared:
        print(f"[verify] Cleared {len(cleared)} pre-existing IQN env vars: {cleared}")
    else:
        print("[verify] No pre-existing IQN env vars to clear.")

    # Step 2: Apply manifest env_overrides.
    _apply_env_overrides(manifest)
    print(
        f"[verify] Applied {len(manifest.get('env_overrides', {}))} env_overrides from manifest."
    )

    # Step 3: Unset env_unset entries.
    _apply_env_unset(manifest)

    # Step 4: Build the effective IQN config using the same code path as training.
    # Import here (after env setup) so env vars are in place when module is loaded.
    # build_iqn_config() calls IQNConfig.stockdss_long_v1() which reads env vars.
    from stock_investment_dss.rl.config.iqn_config import build_iqn_config

    total_steps = int(
        os.environ.get("STOCK_INVESTMENT_DSS_IQN_LEARNING_CURVE_TOTAL_STEPS", "10000")
    )
    learning_starts = int(
        os.environ.get("STOCK_INVESTMENT_DSS_IQN_LEARNING_STARTS", "1000")
    )

    iqn_config = build_iqn_config()
    # Apply the same candidate-value overrides that create_iqn_config_from_environment() does.
    # We replicate only the values that differ between build_iqn_config() and the full override,
    # specifically: total_steps, learning_starts, and lr (env var name difference).
    iqn_config.total_steps = total_steps
    iqn_config.learning_starts = learning_starts
    # lr may be set via LEARNING_RATE (smoke test) or IQN_LR (config); take LEARNING_RATE if set.
    lr_override = os.environ.get("STOCK_INVESTMENT_DSS_IQN_LEARNING_RATE")
    if lr_override is not None:
        try:
            iqn_config.lr = float(lr_override)
        except ValueError:
            pass

    config_dict = asdict(iqn_config)
    # Convert torch.device to string if present.
    config_dict = {
        k: str(v) if hasattr(v, "type") else v for k, v in config_dict.items()
    }

    # Step 5: Assert each expected value.
    expected = manifest.get("expected_iqn_config", {})
    failures: list[str] = []
    pass_lines: list[str] = []

    print("\n[verify] --- IQN Config Assertion ---")
    for field, expected_val in expected.items():
        actual_val = config_dict.get(field)
        ok, msg = _check_value(field, expected_val, actual_val)
        print(msg)
        if ok:
            pass_lines.append(msg)
        else:
            failures.append(msg)

    # Step 6: Print overall result.
    print()
    if failures:
        print(f"[verify] RESULT: FAIL — {len(failures)} assertion(s) failed:")
        for line in failures:
            print(f"  {line}")
        result_status = "FAIL"
    else:
        print(f"[verify] RESULT: PASS — all {len(expected)} assertions match.")
        result_status = "PASS"

    # Step 7: Write verification artifact.
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / f"{experiment_id}_config_check.json"

    artifact = {
        "experiment_id": experiment_id,
        "manifest_path": str(config_path),
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "result": result_status,
        "failures": failures,
        "cleared_env_vars": cleared,
        "total_assertions": len(expected),
        "passed_assertions": len(expected) - len(failures),
        "effective_iqn_config": config_dict,
        "expected_iqn_config": expected,
        "env_overrides_applied": manifest.get("env_overrides", {}),
    }
    with artifact_path.open("w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2, ensure_ascii=False, default=str)

    print(f"[verify] Artifact written: {artifact_path}")

    return 0 if result_status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
