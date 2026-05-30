"""IQN + HDP + EDL portfolio backtest runner — 4-ablation comparison.

Replays the pre-computed combined_with_counterfactual_labels.csv decisions
against a live FinRL portfolio environment to produce fair, apples-to-apples
portfolio performance metrics for four ablation tiers:

  A1  IQN-only             — raw IQN actions, resolver default sizing
  A2  IQN + HDP            — HDP actions + HDP-chosen size (size_hint)
  A3  IQN + EDL gate       — IQN actions filtered by EDL gate (can force HOLD)
  A4  IQN + HDP + EDL gate — HDP actions + EDL gate + EDL-modulated sizing

All four ablations use the same environment, same initial capital, and the same
PIT eval window.  No IQN inference is run at backtest time — actions are read
directly from the pre-computed CSV.

Bug fixes vs B.6.4 / B.6.5:
  - Correct class order: IDX_TO_ACTION = {0:"HOLD", 1:"BUY", 2:"SELL"}
    (matches training CLASS_TO_ID in run_edl_action_training_v3_production.py)
  - No double-+1: alpha = model(X).cpu().numpy()
    (model.forward() already returns relu+1; B.6.4/B.6.5 added another +1)

Usage::

    python -m stock_investment_dss.runner.run_iqn_hdp_edl_portfolio_backtest \\
        --combined-csv outputs/runs/<b2_run>/audit/combined_with_counterfactual_labels.csv \\
        --merged-dir  outputs/runs/MERGED_..._COMPLETE \\
        --market-data data/market/daily/imports/market_data_full_500.csv \\
        --manifest    configs/experiments/sp500_8020_v1.json \\
        --output-dir  outputs/runs/<this_run> \\
        --ablations   a1,a2,a3,a4 \\
        --smoke
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from stock_investment_dss.data.point_in_time_split import create_point_in_time_split
from stock_investment_dss.decision.action_mask import DSSActionMaskGenerator
from stock_investment_dss.decision.decision_actions import DSSDecisionAction
from stock_investment_dss.decision.investor_risk_profile import InvestorRiskProfile
from stock_investment_dss.decision.risk_aware_action_resolver import (
    RiskAwareActionResolver,
)
from stock_investment_dss.environments.discrete_finrl_decision_env import (
    DiscreteFinRLDecisionEnv,
)
from stock_investment_dss.environments.finrl_env_factory import (
    FinRLStockTradingEnvConfig,
    create_finrl_stock_trading_env,
    extract_finrl_state_summary,
    unpack_reset_result,
    unpack_step_result,
)
from stock_investment_dss.metrics.trading_metrics import calculate_account_metrics
from stock_investment_dss.uncertainty.edl_gate import EDLGate, EDLGateConfig
from stock_investment_dss.utilities.config import get_environment_variable
from stock_investment_dss.utilities.logging import setup_run_logger, setup_system_logger
from stock_investment_dss.utilities.paths import create_run_paths

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (bug-fixed class order — matches training CLASS_TO_ID)
# ---------------------------------------------------------------------------
IDX_TO_ACTION = {0: "HOLD", 1: "BUY", 2: "SELL"}

# Feature exclusion sets (same as B.3 v3 training and B.6.4)
_EXCLUDE_PREFIXES = ("edl_a_", "edl_b_", "edl_c_", "edl_label_")
_EXCLUDE_EXACT = {
    "decision_id",
    "date",
    "visible_data_cutoff",
    "eval_step",
    "source_iqn_run_id",
    "dataset_id",
    "pit_split_id",
    "selected_iqn_action",
    "iqn_chosen_action",
    "hierarchical_action_type",
    "selected_ticker",
    "selected_size",
    "final_recommendation_before_edl",
    "final_recommendation_source",
    "selected_action_type",
}


# ---------------------------------------------------------------------------
# EDL model architecture (exact replica of training — matches B.6.4)
# ---------------------------------------------------------------------------


class EDLActionNetworkV3(nn.Module):
    """Exact replica of EDL-A architecture used in Phase B.3 v3 training.

    Matches run_edl_action_training_v3_production.py exactly:
      - attribute 'body' (not 'net')
      - LayerNorm after each Linear hidden layer
      - F.relu evidence head returning alpha = evidence + 1.0
    """

    _ACTIVATION_MAP = {
        "ReLU": nn.ReLU,
        "GELU": nn.GELU,
        "SiLU": nn.SiLU,
        "Mish": nn.Mish,
    }

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: list,
        activation: str = "SiLU",
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if activation not in self._ACTIVATION_MAP:
            raise ValueError(f"Unknown activation '{activation}'")
        ActFn = self._ACTIVATION_MAP[activation]
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.LayerNorm(h))
            layers.append(ActFn())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, num_classes))
        self.body = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.body(x)
        evidence = F.relu(logits)
        return evidence + 1.0  # returns alpha directly


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_float_env(name: str, default: float) -> float:
    v = get_environment_variable(name, default=str(default))
    return float(v or default)


def _get_int_env(name: str, default: int) -> int:
    v = get_environment_variable(name, default=str(default))
    return int(v or default)


def _create_risk_profile() -> InvestorRiskProfile:
    preset = (
        (
            get_environment_variable(
                "STOCK_INVESTMENT_DSS_INVESTOR_RISK_PROFILE",
                default="balanced",
            )
            or "balanced"
        )
        .strip()
        .lower()
    )
    if preset == "defensive":
        return InvestorRiskProfile.defensive()
    if preset == "aggressive":
        return InvestorRiskProfile.aggressive()
    return InvestorRiskProfile.balanced()


def _infer_feature_columns(df: pd.DataFrame) -> list[str]:
    """Determine feature columns (same exclusion sets as B.3 v3 training)."""
    return [
        col
        for col in df.columns
        if col not in _EXCLUDE_EXACT
        and not any(col.startswith(p) for p in _EXCLUDE_PREFIXES)
        and df[col].dtype != "object"
    ]


def _load_edl_models(
    merged_dir: Path,
    input_dim: int,
    cfg: dict,
    device: torch.device,
) -> list:
    """Load 10-fold EDL ensemble from MERGED folder."""
    hidden_dims = cfg["hidden_dims"]
    activation = cfg.get("activation", "SiLU")
    dropout = float(cfg.get("dropout", 0.0))
    models = []
    for fold_idx in range(10):
        model_path = (
            merged_dir / "models" / f"edl_action_classifier_v3_fold_{fold_idx}.pt"
        )
        if not model_path.exists():
            raise FileNotFoundError(f"Missing fold model: {model_path}")
        model = EDLActionNetworkV3(input_dim, 3, hidden_dims, activation, dropout)
        state = torch.load(model_path, map_location=device, weights_only=False)
        if isinstance(state, dict) and "state_dict" in state:
            model.load_state_dict(state["state_dict"])
        else:
            model.load_state_dict(state)
        model.eval()
        model.to(device)
        models.append(model)
    return models


def _ensemble_predict(models: list, X: np.ndarray, device: torch.device) -> dict:
    """
    Run ensemble inference (bug-fixed: no double +1).

    model.forward() returns relu+1 (alpha) directly.  We do NOT add another
    +1 here (that would be the B.6.4/B.6.5 double-+1 bug).

    Returns:
        ensemble_pred   (N,)  argmax class index
        vacuity         (N,)  epistemic uncertainty K / sum(alpha)
        disagreement    (N,)  fraction of folds disagreeing with ensemble
    """
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    K = 3
    per_fold_alpha = []
    per_fold_preds = []
    with torch.no_grad():
        for model in models:
            alpha = model(X_t).cpu().numpy()  # already relu+1; NO extra +1
            per_fold_alpha.append(alpha)
            per_fold_preds.append(np.argmax(alpha, axis=1))
    ensemble_alpha = np.mean(np.stack(per_fold_alpha, axis=0), axis=0)
    S = ensemble_alpha.sum(axis=1, keepdims=True)
    vacuity = K / S.flatten()
    ensemble_pred = np.argmax(ensemble_alpha, axis=1)
    per_fold_preds_arr = np.stack(per_fold_preds, axis=0)
    disagreement = np.mean(per_fold_preds_arr != ensemble_pred[np.newaxis, :], axis=0)
    return {
        "ensemble_pred": ensemble_pred,
        "vacuity": vacuity,
        "disagreement": disagreement,
    }


def _normalize_for_gate(action_raw: str) -> str | None:
    """Map action string for EDL gate input.

    Returns None if the action should bypass the gate (REBALANCE /
    CHANGE_STRATEGY are not modelled by the 3-class EDL).
    Returns BUY, SELL, or HOLD otherwise.
    """
    a = str(action_raw).upper().strip()
    if a in ("REBALANCE", "CHANGE_STRATEGY"):
        return None
    if a.startswith("BUY"):
        return "BUY"
    if a.startswith("SELL"):
        return "SELL"
    return "HOLD"


def _safe_dss_action(action_str: str) -> DSSDecisionAction:
    """Convert action string to DSSDecisionAction; falls back to HOLD on error."""
    try:
        return DSSDecisionAction[action_str.upper().strip()]
    except (KeyError, AttributeError):
        log.warning("Unrecognised DSS action '%s'; defaulting to HOLD.", action_str)
        return DSSDecisionAction.HOLD


# ---------------------------------------------------------------------------
# Environment factory (mirrors create_backtest_environment from smoke test)
# ---------------------------------------------------------------------------


def _create_env(
    tickers: list[str],
    trade_data: pd.DataFrame,
    initial_amount: float,
    hmax: int,
    transaction_cost_pct: float,
    risk_profile: InvestorRiskProfile,
):
    """Create DiscreteFinRLDecisionEnv + resolver + mask_generator."""
    finrl_env, _, _ = create_finrl_stock_trading_env(
        market_data=trade_data,
        tickers=tickers,
        config=FinRLStockTradingEnvConfig(
            initial_amount=initial_amount,
            hmax=hmax,
            buy_cost_pct=transaction_cost_pct,
            sell_cost_pct=transaction_cost_pct,
            reward_scaling=1.0,
            print_verbosity=10_000,
        ),
        technical_indicators=None,
    )
    resolver = RiskAwareActionResolver(
        tickers=tickers,
        hmax=hmax,
        risk_profile=risk_profile,
    )
    action_mask_generator = DSSActionMaskGenerator(
        tickers=tickers,
        risk_profile=risk_profile,
        allow_change_strategy_without_signal=True,
    )
    env = DiscreteFinRLDecisionEnv(
        finrl_env=finrl_env,
        tickers=tickers,
        resolver=resolver,
        action_mask_generator=action_mask_generator,
        enforce_action_mask=True,
    )
    return env, resolver, action_mask_generator


# ---------------------------------------------------------------------------
# Per-ablation step loop
# ---------------------------------------------------------------------------


def _run_ablation_loop(
    *,
    ablation: str,
    combined_df: pd.DataFrame,
    env: DiscreteFinRLDecisionEnv,
    resolver: RiskAwareActionResolver,
    action_mask_generator: DSSActionMaskGenerator,
    tickers: list[str],
    ensemble_pred: np.ndarray,
    vacuity: np.ndarray,
    disagreement: np.ndarray,
    gate_config: EDLGateConfig,
) -> tuple[list[str], list[float], list[dict]]:
    """Step through combined_df, return (dates, portfolio_values, action_records).

    ``action_records`` holds one dict per trading day with the raw IQN-proposed
    action and the action this ablation actually executed after HDP/EDL governance.
    The per-ablation action distribution (dashboard Panel 4) is derived from the
    ``final_action`` column of these records.
    """
    obs, _ = unpack_reset_result(env.reset())
    finrl_env = env.finrl_env

    dates: list[str] = []
    portfolio_values: list[float] = []
    action_records: list[dict] = []

    use_bypass = ablation in ("a2", "a4")  # bypass DiscreteFinRLDecisionEnv
    use_gate = ablation in ("a3", "a4")

    for row_idx, row in enumerate(combined_df.itertuples()):
        row_date = str(getattr(row, "date", f"step_{row_idx}"))

        # ------------------------------------------------------------------
        # Determine action and size for this step
        # ------------------------------------------------------------------
        if ablation == "a1":
            final_action = str(getattr(row, "iqn_chosen_action", "HOLD")).upper()
            final_size: float | None = None  # resolver uses its default

        elif ablation == "a2":
            final_action = str(getattr(row, "hierarchical_action_type", "HOLD")).upper()
            raw_size = getattr(row, "selected_size_fraction", None)
            final_size = float(raw_size) if raw_size and not pd.isna(raw_size) else 0.0

        elif ablation == "a3":
            base_action_raw = str(getattr(row, "iqn_chosen_action", "HOLD")).upper()
            gate_input = _normalize_for_gate(base_action_raw)
            if gate_input is None:
                # REBALANCE / CHANGE_STRATEGY: bypass gate, execute as-is
                final_action = base_action_raw
                final_size = None
            else:
                edl_pred_action = IDX_TO_ACTION[int(ensemble_pred[row_idx])]
                gate_result = EDLGate(gate_config).apply(
                    selected_action=gate_input,
                    selected_size=str(getattr(row, "selected_size", "")),
                    original_fraction=1.0,
                    vacuity=float(vacuity[row_idx]),
                    edl_agrees=(edl_pred_action == gate_input),
                    edl_predicted_action=edl_pred_action,
                    p_rebalance=0.0,
                    p_change_strategy=0.0,
                    disagreement_score=float(disagreement[row_idx]),
                    uncertainty_penalty=(
                        gate_config.uncertainty_lambda * float(vacuity[row_idx])
                    ),
                )
                final_action = gate_result.final_action_after_edl_gate.upper()
                final_size = None  # A3 does not apply EDL size reduction to env

        elif ablation == "a4":
            base_action_raw = str(
                getattr(row, "hierarchical_action_type", "HOLD")
            ).upper()
            raw_size = getattr(row, "selected_size_fraction", None)
            base_size = float(raw_size) if raw_size and not pd.isna(raw_size) else 0.0

            gate_input = _normalize_for_gate(base_action_raw)
            if gate_input is None:
                # REBALANCE / CHANGE_STRATEGY: bypass gate
                final_action = base_action_raw
                final_size = base_size
            else:
                edl_pred_action = IDX_TO_ACTION[int(ensemble_pred[row_idx])]
                gate_result = EDLGate(gate_config).apply(
                    selected_action=gate_input,
                    selected_size=str(getattr(row, "selected_size", "")),
                    original_fraction=base_size,
                    vacuity=float(vacuity[row_idx]),
                    edl_agrees=(edl_pred_action == gate_input),
                    edl_predicted_action=edl_pred_action,
                    p_rebalance=0.0,
                    p_change_strategy=0.0,
                    disagreement_score=float(disagreement[row_idx]),
                    uncertainty_penalty=(
                        gate_config.uncertainty_lambda * float(vacuity[row_idx])
                    ),
                )
                final_action = gate_result.final_action_after_edl_gate.upper()
                final_size = gate_result.final_fraction_after_edl_gate

        else:
            raise ValueError(f"Unknown ablation: {ablation!r}")

        # ------------------------------------------------------------------
        # Step the environment
        # ------------------------------------------------------------------
        dss_action = _safe_dss_action(final_action)

        if use_bypass:
            # A2 / A4: call resolver + finrl_env directly so size_hint takes effect
            state_summary = extract_finrl_state_summary(obs, tickers)
            mask_result = action_mask_generator.generate(state_summary)
            size_hint_arg = final_size  # may be None (REBALANCE/CHANGE_STRATEGY)

            if mask_result.is_allowed(dss_action):
                resolved = resolver.resolve(
                    dss_action, state_summary, size_hint=size_hint_arg
                )
            else:
                resolved = resolver.resolve_blocked_action(
                    dss_action,
                    state_summary,
                    blocked_reason=mask_result.blocked_reasons.get(
                        dss_action.name, "action masked"
                    ),
                )

            cont_action = np.array(resolved.continuous_action, dtype=float)
            step_result = finrl_env.step(cont_action)
            obs, reward, done, _ = unpack_step_result(step_result)
            pv = extract_finrl_state_summary(obs, tickers)["portfolio_value"]

        else:
            # A1 / A3: use DiscreteFinRLDecisionEnv.step() (resolver picks sizing)
            step_result = env.step(dss_action.value)
            obs, reward, done, info = unpack_step_result(step_result)
            pv = info["decision_record"]["state_after"]["portfolio_value"]

        dates.append(row_date)
        portfolio_values.append(float(pv))
        action_records.append(
            {
                "date": row_date,
                "iqn_action": str(
                    getattr(row, "iqn_chosen_action", "HOLD")
                ).upper(),
                "final_action": final_action,
            }
        )

        if done:
            log.info(
                "Ablation %s: env terminated after %d steps.", ablation, row_idx + 1
            )
            break

    return dates, portfolio_values, action_records


# ---------------------------------------------------------------------------
# Write per-ablation outputs
# ---------------------------------------------------------------------------


def _write_ablation_outputs(
    ablation: str,
    dates: list[str],
    portfolio_values: list[float],
    action_records: list[dict],
    initial_amount: float,
    output_dir: Path,
) -> dict[str, Path]:
    """Write account_values.csv and metrics.csv for one ablation."""
    abl_dir = output_dir / ablation
    abl_dir.mkdir(parents=True, exist_ok=True)

    account_df = pd.DataFrame(
        {
            "date": dates,
            "account_value": portfolio_values,
            "strategy": f"IQN_HDP_EDL_{ablation.upper()}",
        }
    )

    strategy_label = {
        "a1": "IQN_only",
        "a2": "IQN_HDP",
        "a3": "IQN_EDL",
        "a4": "IQN_HDP_EDL",
    }.get(ablation, f"IQN_ABL_{ablation.upper()}")

    metrics = calculate_account_metrics(
        account_df,
        strategy=strategy_label,
        source=ablation,
        initial_amount=initial_amount,
    )

    account_path = abl_dir / "account_values.csv"
    metrics_path = abl_dir / "metrics.csv"
    actions_path = abl_dir / "actions.csv"

    account_df.to_csv(account_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    pd.DataFrame(action_records).to_csv(actions_path, index=False)

    log.info(
        "Ablation %s complete: %d steps, final PV=%.2f, "
        "total_return=%.2f%%, sharpe=%.3f",
        ablation,
        len(portfolio_values),
        portfolio_values[-1] if portfolio_values else float("nan"),
        float(metrics["total_return_pct"].iloc[0]),
        float(metrics["annualized_sharpe"].iloc[0]),
    )
    return {"account": account_path, "metrics": metrics_path, "actions": actions_path}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(
        description="IQN+HDP+EDL 4-ablation portfolio backtest.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--combined-csv", required=True)
    p.add_argument("--merged-dir", required=True)
    p.add_argument("--market-data", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument(
        "--ablations",
        default="a1,a2,a3,a4",
        help="Comma-separated ablation IDs to run.",
    )
    p.add_argument("--initial-amount", type=float, default=1_000_000.0)
    p.add_argument("--transaction-cost-pct", type=float, default=0.001)
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke mode: use only the first 100 rows of combined_df.",
    )
    args = p.parse_args()

    log_level = (
        get_environment_variable("STOCK_INVESTMENT_DSS_LOG_LEVEL", default="INFO")
        or "INFO"
    )
    setup_system_logger(log_level=log_level)

    ablations = [a.strip().lower() for a in args.ablations.split(",") if a.strip()]
    valid_ablations = {"a1", "a2", "a3", "a4"}
    for abl in ablations:
        if abl not in valid_ablations:
            log.error("Unknown ablation '%s'. Must be one of %s.", abl, valid_ablations)
            return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load combined CSV
    # ------------------------------------------------------------------
    log.info("Loading combined CSV: %s", args.combined_csv)
    combined_df = (
        pd.read_csv(args.combined_csv).sort_values("eval_step").reset_index(drop=True)
    )
    if args.smoke:
        log.info("Smoke mode: limiting to first 100 rows.")
        combined_df = combined_df.head(100)
    log.info(
        "Combined CSV: %d rows, %d columns.", len(combined_df), len(combined_df.columns)
    )

    # ------------------------------------------------------------------
    # Load manifest → eval dates and tickers
    # ------------------------------------------------------------------
    with open(args.manifest, encoding="utf-8") as fh:
        manifest: dict[str, Any] = json.load(fh)
    dataset_cfg = manifest.get("dataset", manifest)
    eval_start = str(dataset_cfg.get("eval_start", "2024-01-01"))
    eval_end = str(dataset_cfg.get("eval_end", "2026-05-26"))
    log.info("Manifest eval window: %s → %s", eval_start, eval_end)

    # ------------------------------------------------------------------
    # Load market data and create PIT split
    # ------------------------------------------------------------------
    log.info("Loading market data: %s", args.market_data)
    market_df = pd.read_csv(args.market_data)
    split_result = create_point_in_time_split(
        data=market_df,
        split_id="sp500_pit_backtest",
        point_in_time=eval_start,
        trade_end_date=eval_end,
    )
    trade_data = split_result.trade_data
    tickers = sorted(trade_data["tic"].unique().tolist())
    log.info("Trade data: %d rows, %d tickers.", len(trade_data), len(tickers))

    # ------------------------------------------------------------------
    # Build environment
    # ------------------------------------------------------------------
    risk_profile = _create_risk_profile()
    hmax = _get_int_env("STOCK_INVESTMENT_DSS_FINRL_ENV_HMAX", default=10_000)
    log.info("Creating env: hmax=%d, tc=%.4f%%", hmax, args.transaction_cost_pct * 100)

    env, resolver, action_mask_generator = _create_env(
        tickers=tickers,
        trade_data=trade_data,
        initial_amount=args.initial_amount,
        hmax=hmax,
        transaction_cost_pct=args.transaction_cost_pct,
        risk_profile=risk_profile,
    )

    # ------------------------------------------------------------------
    # Load EDL ensemble and run inference (all rows at once)
    # ------------------------------------------------------------------
    merged_dir = Path(args.merged_dir)
    feature_cols = _infer_feature_columns(combined_df)
    log.info("EDL feature columns: %d", len(feature_cols))
    X = combined_df[feature_cols].fillna(0.0).values.astype(np.float32)

    cfg_path = merged_dir / "hp_search" / "best_config.json"
    with open(cfg_path, encoding="utf-8") as fh:
        edl_cfg = json.load(fh)

    device = torch.device("cpu")
    log.info("Loading EDL ensemble from %s", merged_dir)
    models = _load_edl_models(
        merged_dir=merged_dir,
        input_dim=len(feature_cols),
        cfg=edl_cfg,
        device=device,
    )
    log.info("Running EDL inference on %d rows...", len(combined_df))
    edl_results = _ensemble_predict(models, X, device)
    ensemble_pred = edl_results["ensemble_pred"]
    vacuity = edl_results["vacuity"]
    disagreement = edl_results["disagreement"]
    log.info(
        "EDL inference done: vacuity mean=%.4f, disagreement mean=%.4f",
        float(vacuity.mean()),
        float(disagreement.mean()),
    )

    gate_config = EDLGateConfig()

    # ------------------------------------------------------------------
    # Run ablations
    # ------------------------------------------------------------------
    all_results: dict[str, dict[str, Path]] = {}

    for ablation in ablations:
        log.info("=" * 60)
        log.info("Running ablation: %s", ablation.upper())
        log.info("=" * 60)

        dates, portfolio_values, action_records = _run_ablation_loop(
            ablation=ablation,
            combined_df=combined_df,
            env=env,
            resolver=resolver,
            action_mask_generator=action_mask_generator,
            tickers=tickers,
            ensemble_pred=ensemble_pred,
            vacuity=vacuity,
            disagreement=disagreement,
            gate_config=gate_config,
        )

        all_results[ablation] = _write_ablation_outputs(
            ablation=ablation,
            dates=dates,
            portfolio_values=portfolio_values,
            action_records=action_records,
            initial_amount=args.initial_amount,
            output_dir=output_dir,
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    log.info("=" * 60)
    log.info("All ablation portfolio backtest results:")
    log.info("  %-6s  %-20s  %s", "ABL", "metrics.csv", "account_values.csv")
    for abl, paths in all_results.items():
        log.info(
            "  %-6s  %-20s  %s",
            abl.upper(),
            str(paths.get("metrics", "")),
            str(paths.get("account", "")),
        )
    log.info("=" * 60)

    # Write combined summary CSV
    summary_rows = []
    for abl, paths in all_results.items():
        metrics_path = paths.get("metrics")
        if metrics_path and metrics_path.exists():
            df_m = pd.read_csv(metrics_path)
            if not df_m.empty:
                row = df_m.iloc[0].to_dict()
                row["ablation"] = abl
                summary_rows.append(row)
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_path = output_dir / "ablation_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        log.info("Ablation summary: %s", summary_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
