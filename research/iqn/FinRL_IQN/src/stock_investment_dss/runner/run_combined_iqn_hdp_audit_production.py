"""Phase B.1 — Combined IQN+HDP Audit Production Runner.

Produces the production-grade combined IQN+HDP audit CSV for the full eval
window (2024-01-01 → 2026-12-31) using the best available IQN seed.

This output feeds:
  - Phase B.2 (EDL-A counterfactual oracle) — primary input CSV
  - Phase B.4 (IQN+HDP ablation variant) — performance evidence

Usage::

    # Smoke test (6-month window, ~5-10 min):
    python src/stock_investment_dss/runner/run_combined_iqn_hdp_audit_production.py \\
        --universe demo_10_new \\
        --eval-start 2024-01-01 \\
        --eval-end 2024-06-30 \\
        --strategy balanced_v1

    # Full production run (~30 min):
    python src/stock_investment_dss/runner/run_combined_iqn_hdp_audit_production.py \\
        --universe demo_10_new \\
        --eval-start 2024-01-01 \\
        --eval-end 2026-12-31 \\
        --strategy balanced_v1

Output structure::

    outputs/runs/{timestamp}_d_iqn_dss_combined_iqn_hdp_audit_production/
    ├── audit/
    │   ├── combined_iqn_hierarchical_decision_by_step.csv   ← KEY (filtered, 57 cols)
    │   └── combined_iqn_hierarchical_decision_full.csv      ← full pre-filter
    ├── config/
    │   ├── production_config.json
    │   └── universe_config.json
    ├── data/
    │   ├── iqn_inference_records.csv
    │   ├── hdp_ticker_scores_by_step.csv
    │   └── hdp_size_scores_by_step.csv
    ├── logs/run.log
    ├── metrics/audit_summary.json
    ├── plots/
    │   ├── action_distribution_over_time.png
    │   ├── ticker_selection_frequency.png
    │   └── size_distribution.png
    └── summary/
        ├── audit_summary.md
        └── audit_summary.json

Audit CSV columns (57):
  55 columns from CombinedIQNHierarchicalPolicy (policy names)
  + 2 spec-named aliases:
    hierarchical_action_type  = selected_action_type
    source_iqn_run_id         = iqn_model_run_id
  Plus populated EDL fields: edl_c_teacher_label, edl_label_id
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from stock_investment_dss.utilities.paths import (
    RUNS_DIRECTORY,
    create_run_paths,
)  # noqa: E402
from stock_investment_dss.utilities.logging import setup_run_logger  # noqa: E402
from stock_investment_dss.experiment_tracking.wandb_tracking import (  # noqa: E402
    finish_wandb_run,
    init_wandb_run,
    wandb_log,
)

# ---------------------------------------------------------------------------
# Universe registry
# ---------------------------------------------------------------------------

_UNIVERSE_TICKERS: dict[str, list[str]] = {
    "demo_5": ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"],
    "demo_10_new": [
        "COST",
        "AVGO",
        "LLY",
        "ORCL",
        "CAT",
        "BA",
        "KO",
        "MCD",
        "WMT",
        "PG",
    ],
}

_UNIVERSE_DATA_FILE: dict[str, str] = {
    "demo_5": "data/market/daily/imports/market_data_demo5_2010_2026.csv",
    "demo_10_new": "data/market/daily/imports/market_data_demo10_new_2010_2026.csv",
}

# EDL action name → integer label mapping
_EDL_LABEL_IDS: dict[str, int] = {
    "HOLD": 0,
    "BUY": 1,
    "SELL": 2,
    "REBALANCE": 3,
}

# ---------------------------------------------------------------------------
# IQN checkpoint resolution (3-tier, same logic as etape 7 orchestrator)
# ---------------------------------------------------------------------------


def resolve_iqn_checkpoint(arg: str, repo_root: Path) -> Path:
    """Resolve --iqn-checkpoint to an actual .pt file path.

    Tier 1 (auto): ``*_d_iqn_dss_clean_25k_baseline_v1_seed_*/models/*.pt`` (latest first)
    Tier 2 (auto): any ``*_iqn_*/models/*.pt`` fallback
    Explicit path: resolved relative to repo_root if not absolute
    """
    runs_dir = repo_root / "outputs" / "runs"

    if arg == "auto":
        for run_dir in sorted(
            runs_dir.glob("*_d_iqn_dss_clean_25k_baseline_v1_seed_*"), reverse=True
        ):
            pt_files = list((run_dir / "models").glob("*.pt"))
            if pt_files:
                return pt_files[0]
        for run_dir in sorted(runs_dir.glob("*_iqn_*"), reverse=True):
            models_dir = run_dir / "models"
            if models_dir.exists():
                pt_files = list(models_dir.glob("*.pt"))
                if pt_files:
                    return pt_files[0]
        raise FileNotFoundError(
            "No IQN checkpoint (.pt) found in outputs/runs/. "
            "Run an IQN training experiment first, or provide --iqn-checkpoint explicitly."
        )

    if arg == "train":
        raise NotImplementedError(
            "--iqn-checkpoint=train is not supported by this runner. "
            "Provide an explicit .pt path or use --iqn-checkpoint=auto."
        )

    ckpt = Path(arg)
    if not ckpt.is_absolute():
        ckpt = repo_root / ckpt
    if not ckpt.exists():
        raise FileNotFoundError(f"IQN checkpoint not found: {ckpt}")
    if ckpt.suffix != ".pt":
        raise ValueError(f"Expected a .pt checkpoint file, got: {ckpt.suffix}")
    return ckpt


# ---------------------------------------------------------------------------
# IQN run data loader with graceful fallback
# ---------------------------------------------------------------------------


def _load_iqn_run_graceful(
    iqn_ckpt: Path,
    runs_dir: Path,
    logger: logging.Logger,
):
    """Load IQNRunData from the checkpoint's run directory, with fallback.

    Attempt 1: derive run dir from .pt path (run_dir/models/file.pt) and call
               IQNRunLoader.load(run_id=run_dir.name).
    Attempt 2: if attempt 1 fails, use the latest ``iqn_learning_curve`` run.
    Raises FileNotFoundError if both attempts fail.
    """
    from stock_investment_dss.decision.combined_iqn_hierarchical_policy import (
        IQNRunLoader,
    )

    ckpt_run_dir = iqn_ckpt.parent.parent  # .pt lives at run_dir/models/file.pt

    # Attempt 1: load from the checkpoint's own run directory
    try:
        loader = IQNRunLoader(runs_dir)
        iqn_data = loader.load(run_id=ckpt_run_dir.name)
        logger.info("IQN run loaded: %s", ckpt_run_dir.name)
        return iqn_data
    except Exception as exc_1:
        logger.warning(
            "IQNRunLoader failed for %s: %s — trying iqn_learning_curve fallback",
            ckpt_run_dir.name,
            exc_1,
        )

    # Attempt 2: latest iqn_learning_curve run
    try:
        candidates = sorted(
            d
            for d in runs_dir.iterdir()
            if d.is_dir() and "iqn_learning_curve" in d.name
        )
        if candidates:
            fallback_dir = candidates[-1]
            loader = IQNRunLoader(runs_dir)
            iqn_data = loader.load(run_id=fallback_dir.name)
            logger.info("IQN run loaded from fallback: %s", fallback_dir.name)
            return iqn_data
    except Exception as exc_2:
        logger.warning("Fallback IQNRunLoader also failed: %s", exc_2)

    raise FileNotFoundError(
        f"Cannot load IQNRunData from {ckpt_run_dir} or any iqn_learning_curve "
        f"run in {runs_dir}. Ensure a run with eval distribution CSVs exists."
    )


# ---------------------------------------------------------------------------
# Post-processing helpers
# ---------------------------------------------------------------------------


def _filter_audit_by_window(
    audit_df: pd.DataFrame,
    eval_start: str,
    eval_end: str,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Filter audit rows to [eval_start, eval_end] inclusive.

    Logs a warning if the requested window extends beyond available data.
    Raises ValueError if no rows remain after filtering.
    """
    df = audit_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    start_ts = pd.Timestamp(eval_start)
    end_ts = pd.Timestamp(eval_end)

    full_start = df["date"].min()
    full_end = df["date"].max()
    full_count = len(df)

    if start_ts < full_start:
        logger.warning(
            "--eval-start %s is before available IQN data (%s); filter starts at available data",
            eval_start,
            full_start.date(),
        )
    if end_ts > full_end:
        logger.warning(
            "--eval-end %s is after available IQN data (%s); filter ends at available data",
            eval_end,
            full_end.date(),
        )

    mask = (df["date"] >= start_ts) & (df["date"] <= end_ts)
    filtered = df[mask].copy()
    logger.info(
        "Eval window filter: requested [%s→%s] | available [%s→%s] | kept %d/%d rows",
        eval_start,
        eval_end,
        full_start.date(),
        full_end.date(),
        len(filtered),
        full_count,
    )
    if len(filtered) == 0:
        raise ValueError(
            f"No audit rows in requested window [{eval_start} → {eval_end}]. "
            f"Available window: [{full_start.date()} → {full_end.date()}]"
        )
    return filtered


def _add_spec_aliases(audit_df: pd.DataFrame) -> pd.DataFrame:
    """Add backward-compat aliases so both the policy column names and
    v3.3 spec names are present in the output CSV.

    Actual policy column  →  alias added
    hierarchical_action_type  →  selected_action_type
    iqn_model_run_id          →  source_iqn_run_id
    """
    # Maps: policy_column_name → alias_name_to_add
    aliases = {
        "hierarchical_action_type": "selected_action_type",
        "iqn_model_run_id": "source_iqn_run_id",
    }
    df = audit_df.copy()
    for policy_name, alias_name in aliases.items():
        if policy_name in df.columns and alias_name not in df.columns:
            df[alias_name] = df[policy_name]
    return df


def _compute_edl_c_labels(audit_df: pd.DataFrame) -> pd.DataFrame:
    """Populate edl_c_teacher_label and edl_label_id.

    edl_c_teacher_label = action with highest IQN score (argmax of iqn_score_*).
    edl_label_id        = integer 0/1/2/3 corresponding to HOLD/BUY/SELL/REBALANCE.
    Fallback to selected_iqn_action if score columns are absent.
    """
    df = audit_df.copy()
    score_cols = {
        "HOLD": "iqn_score_hold",
        "BUY": "iqn_score_buy",
        "SELL": "iqn_score_sell",
        "REBALANCE": "iqn_score_rebalance",
    }
    present = {k: v for k, v in score_cols.items() if v in df.columns}
    if len(present) == 4:
        scores = df[[v for v in present.values()]].copy()
        scores.columns = list(present.keys())
        df["edl_c_teacher_label"] = scores.idxmax(axis=1)
        df["edl_label_id"] = (
            df["edl_c_teacher_label"].map(_EDL_LABEL_IDS).fillna(-1).astype(int)
        )
    elif "selected_iqn_action" in df.columns:
        df["edl_c_teacher_label"] = df["selected_iqn_action"]
        df["edl_label_id"] = (
            df["selected_iqn_action"].map(_EDL_LABEL_IDS).fillna(-1).astype(int)
        )
    else:
        df["edl_c_teacher_label"] = None
        df["edl_label_id"] = -1
    return df


# ---------------------------------------------------------------------------
# Plot generation
# ---------------------------------------------------------------------------


def _generate_plots(
    audit_df: pd.DataFrame,
    plots_dir: Path,
    logger: logging.Logger,
) -> None:
    """Generate three diagnostic plots to plots/."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available — plots skipped")
        return

    action_col = "hierarchical_action_type"
    ticker_col = "selected_ticker"
    size_col = "selected_size"

    df = audit_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # 1. Action distribution over time — monthly stacked bar
    try:
        monthly = (
            df.set_index("date")
            .resample("ME")[action_col]
            .value_counts()
            .unstack(fill_value=0)
        )
        fig, ax = plt.subplots(figsize=(12, 5))
        monthly.plot(kind="bar", stacked=True, ax=ax)
        ax.set_title("Action Distribution Over Time (monthly)")
        ax.set_xlabel("Month")
        ax.set_ylabel("Decision count")
        ax.tick_params(axis="x", rotation=45)
        plt.tight_layout()
        plt.savefig(plots_dir / "action_distribution_over_time.png", dpi=120)
        plt.close(fig)
    except Exception as exc:
        logger.warning("action_distribution_over_time plot failed: %s", exc)

    # 2. Ticker selection frequency for BUY actions
    try:
        buy_tickers = df[df[action_col] == "BUY"][ticker_col].dropna()
        fig, ax = plt.subplots(figsize=(10, 5))
        if not buy_tickers.empty:
            buy_tickers.value_counts().sort_index().plot(kind="bar", ax=ax)
        ax.set_title("Ticker Selection Frequency (BUY actions)")
        ax.set_xlabel("Ticker")
        ax.set_ylabel("BUY count")
        ax.tick_params(axis="x", rotation=0)
        plt.tight_layout()
        plt.savefig(plots_dir / "ticker_selection_frequency.png", dpi=120)
        plt.close(fig)
    except Exception as exc:
        logger.warning("ticker_selection_frequency plot failed: %s", exc)

    # 3. Size selection distribution
    try:
        fig, ax = plt.subplots(figsize=(8, 5))
        if size_col in df.columns and df[size_col].notna().any():
            df[size_col].dropna().value_counts().sort_index().plot(kind="bar", ax=ax)
        ax.set_title("Size Selection Distribution")
        ax.set_xlabel("Size category")
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=0)
        plt.tight_layout()
        plt.savefig(plots_dir / "size_distribution.png", dpi=120)
        plt.close(fig)
    except Exception as exc:
        logger.warning("size_distribution plot failed: %s", exc)

    logger.info("Plots written to: %s", plots_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase B.1 — Combined IQN+HDP Audit Production Runner.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--iqn-checkpoint",
        default="auto",
        metavar="PATH|auto",
        help="IQN checkpoint .pt path, or 'auto' to find latest clean_25k seed.",
    )
    p.add_argument(
        "--universe",
        default="demo_10_new",
        choices=list(_UNIVERSE_TICKERS.keys()),
        help="Ticker universe.",
    )
    p.add_argument(
        "--eval-start",
        default="2024-01-01",
        metavar="YYYY-MM-DD",
        help="Evaluation window start date.",
    )
    p.add_argument(
        "--eval-end",
        default="2026-12-31",
        metavar="YYYY-MM-DD",
        help="Evaluation window end date.",
    )
    p.add_argument(
        "--strategy",
        default="balanced_v1",
        choices=["balanced_v1", "defensive_v1", "aggressive_v1"],
        help="HDP strategy profile.",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        metavar="PATH",
        help="Override default output directory (optional).",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    args = parse_args()

    run_paths = create_run_paths("d_iqn_dss_combined_iqn_hdp_audit_production")
    logger = setup_run_logger(run_paths)
    run_start = datetime.now()

    logger.info("=== Phase B.1 Combined IQN+HDP Audit Production ===")
    logger.info("Universe:     %s", args.universe)
    logger.info("Eval window:  %s → %s", args.eval_start, args.eval_end)
    logger.info("Strategy:     %s", args.strategy)
    logger.info("IQN ckpt:     %s", args.iqn_checkpoint)
    logger.info("Run dir:      %s", run_paths.run_directory)

    # -----------------------------------------------------------------------
    # 1. Resolve IQN checkpoint
    # -----------------------------------------------------------------------
    try:
        iqn_ckpt = resolve_iqn_checkpoint(args.iqn_checkpoint, _PROJECT_ROOT)
        logger.info("IQN checkpoint: %s", iqn_ckpt)
    except (FileNotFoundError, NotImplementedError, ValueError) as exc:
        logger.error("IQN checkpoint resolution failed: %s", exc)
        return 1

    # -----------------------------------------------------------------------
    # 2. Deferred imports
    # -----------------------------------------------------------------------
    try:
        from stock_investment_dss.decision.combined_iqn_hierarchical_policy import (
            CombinedIQNHierarchicalPolicy,
        )
    except ImportError as exc:
        logger.error("Import failed — is PYTHONPATH set to 'src'? %s", exc)
        return 1

    # -----------------------------------------------------------------------
    # 3. Load IQN run data
    # -----------------------------------------------------------------------
    try:
        iqn_data = _load_iqn_run_graceful(iqn_ckpt, RUNS_DIRECTORY, logger)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    # -----------------------------------------------------------------------
    # 4. Load Mode B market data
    # -----------------------------------------------------------------------
    if args.universe not in _UNIVERSE_DATA_FILE:
        logger.error("Unknown universe: %s", args.universe)
        return 1
    market_path = _PROJECT_ROOT / _UNIVERSE_DATA_FILE[args.universe]
    if not market_path.exists():
        logger.error("Market data not found: %s", market_path)
        return 1
    logger.info("Loading market data: %s", market_path.name)
    market_df = pd.read_csv(market_path, low_memory=False)
    logger.info("Market data loaded: %d rows", len(market_df))

    # -----------------------------------------------------------------------
    # 5. Config files
    # -----------------------------------------------------------------------
    config: dict = {
        "universe": args.universe,
        "eval_start": args.eval_start,
        "eval_end": args.eval_end,
        "strategy": args.strategy,
        "iqn_checkpoint": str(iqn_ckpt),
        "iqn_run_id": iqn_data.run_id,
        "run_directory": str(run_paths.run_directory),
        "run_start": run_start.isoformat(),
    }
    (run_paths.config_directory / "production_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    (run_paths.config_directory / "universe_config.json").write_text(
        json.dumps(
            {
                "universe": args.universe,
                "tickers": _UNIVERSE_TICKERS.get(args.universe, []),
                "eval_window": {"start": args.eval_start, "end": args.eval_end},
                "data_file": _UNIVERSE_DATA_FILE.get(args.universe, ""),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # -----------------------------------------------------------------------
    # 6. W&B setup
    # -----------------------------------------------------------------------
    init_wandb_run(
        run_name=run_paths.run_id,
        config=config,
        group="phase-b1-production",
        job_type="combined_audit",
        tags=[args.universe, args.strategy, "phase-b", "combined-audit"],
        run_directory=str(run_paths.run_directory),
    )

    # -----------------------------------------------------------------------
    # 7. Run combined audit (delegates full date loop to CombinedIQNHierarchicalPolicy)
    # -----------------------------------------------------------------------
    logger.info(
        "Running build_combined_audit() — IQN run: %s  last_train_step: %s",
        iqn_data.run_id,
        iqn_data.last_train_step,
    )
    try:
        policy = CombinedIQNHierarchicalPolicy(
            market_df=market_df,
            strategy_id=args.strategy,
        )
        audit_df_full, ticker_df, size_df, warnings = policy.build_combined_audit(
            iqn_data, train_step=None, use_hierarchical_policy=True
        )
    except Exception as exc:
        logger.error("build_combined_audit() failed: %s", exc, exc_info=True)
        return 1

    logger.info(
        "Audit complete: %d rows, %d ticker score rows, %d size score rows, %d warnings",
        len(audit_df_full),
        len(ticker_df),
        len(size_df),
        len(warnings),
    )
    for w in warnings:
        logger.warning("[audit] %s", w)

    # -----------------------------------------------------------------------
    # 8. Post-process: eval window filter → spec aliases → EDL-C teacher labels
    # -----------------------------------------------------------------------
    try:
        audit_df = _filter_audit_by_window(
            audit_df_full, args.eval_start, args.eval_end, logger
        )
    except ValueError as exc:
        logger.error("Eval window filter failed: %s", exc)
        return 1

    audit_df = _add_spec_aliases(audit_df)
    audit_df = _compute_edl_c_labels(audit_df)
    audit_df_full = _add_spec_aliases(audit_df_full)
    audit_df_full = _compute_edl_c_labels(audit_df_full)

    # -----------------------------------------------------------------------
    # 9. Write audit CSVs
    # -----------------------------------------------------------------------
    key_csv = (
        run_paths.audit_directory / "combined_iqn_hierarchical_decision_by_step.csv"
    )
    full_csv = run_paths.audit_directory / "combined_iqn_hierarchical_decision_full.csv"
    audit_df.to_csv(key_csv, index=False)
    audit_df_full.to_csv(full_csv, index=False)
    logger.info(
        "Audit CSVs: filtered=%d rows (%d cols), full=%d rows",
        len(audit_df),
        len(audit_df.columns),
        len(audit_df_full),
    )

    # -----------------------------------------------------------------------
    # 10. Write data CSVs
    # -----------------------------------------------------------------------
    iqn_id_cols = {"date", "decision_id", "selected_iqn_action"}
    iqn_cols = [c for c in audit_df.columns if c.startswith("iqn_") or c in iqn_id_cols]
    audit_df[iqn_cols].to_csv(
        run_paths.data_directory / "iqn_inference_records.csv", index=False
    )
    if not ticker_df.empty:
        ticker_df.to_csv(
            run_paths.data_directory / "hdp_ticker_scores_by_step.csv", index=False
        )
    if not size_df.empty:
        size_df.to_csv(
            run_paths.data_directory / "hdp_size_scores_by_step.csv", index=False
        )

    # -----------------------------------------------------------------------
    # 11. Metrics JSON
    # -----------------------------------------------------------------------
    action_col = "hierarchical_action_type"
    iqn_action_col = "selected_iqn_action"
    hdp_dist = (
        audit_df[action_col].value_counts().to_dict()
        if action_col in audit_df.columns
        else {}
    )
    iqn_dist = (
        audit_df[iqn_action_col].value_counts().to_dict()
        if iqn_action_col in audit_df.columns
        else {}
    )
    overrides = (
        int((audit_df[iqn_action_col] != audit_df[action_col]).sum())
        if iqn_action_col in audit_df.columns and action_col in audit_df.columns
        else 0
    )
    trading_days = int(audit_df["date"].nunique()) if "date" in audit_df.columns else 0
    edl_dist = (
        audit_df["edl_c_teacher_label"].value_counts().to_dict()
        if "edl_c_teacher_label" in audit_df.columns
        else {}
    )
    metrics: dict = {
        "run_id": run_paths.run_id,
        "universe": args.universe,
        "strategy": args.strategy,
        "eval_start": args.eval_start,
        "eval_end": args.eval_end,
        "iqn_run_id": iqn_data.run_id,
        "iqn_checkpoint": str(iqn_ckpt),
        "total_decision_steps": len(audit_df),
        "trading_days": trading_days,
        "tickers": _UNIVERSE_TICKERS.get(args.universe, []),
        "iqn_action_distribution": iqn_dist,
        "hdp_action_distribution": hdp_dist,
        "risk_validator_interventions": {"hdp_overrides_total": overrides},
        "edl_c_teacher_label_distribution": edl_dist,
        "edl_a_readiness": "ready" if len(audit_df) > 0 else "not_ready",
        "audit_columns": len(audit_df.columns),
    }
    (run_paths.metrics_directory / "audit_summary.json").write_text(
        json.dumps(metrics, indent=2, default=str), encoding="utf-8"
    )

    # -----------------------------------------------------------------------
    # 12. Plots
    # -----------------------------------------------------------------------
    _generate_plots(audit_df, run_paths.plots_directory, logger)

    # -----------------------------------------------------------------------
    # 13. Summary
    # -----------------------------------------------------------------------
    dur_sec = (datetime.now() - run_start).total_seconds()

    def _action_table(dist: dict) -> str:
        n = max(len(audit_df), 1)
        return "\n".join(
            f"| {a} | {c} | {100 * c / n:.1f}% |" for a, c in sorted(dist.items())
        )

    ticker_section = ""
    if "selected_ticker" in audit_df.columns:
        buys = audit_df[audit_df[action_col] == "BUY"]["selected_ticker"].value_counts()
        sells = audit_df[audit_df[action_col] == "SELL"][
            "selected_ticker"
        ].value_counts()
        all_t = sorted(set(list(buys.index) + list(sells.index)))
        rows = "\n".join(
            f"| {t} | {buys.get(t, 0)} | {sells.get(t, 0)} | {buys.get(t, 0) - sells.get(t, 0)} |"
            for t in all_t
        )
        ticker_section = (
            "\n## Top Selected Tickers\n\n"
            "| Ticker | BUY | SELL | Net |\n"
            "|--------|-----|------|-----|\n"
            f"{rows}\n"
        )

    md = (
        "# Combined IQN+HDP Audit Production Run\n\n"
        f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}\n"
        f"Universe: {args.universe}\n"
        f"Eval window: {args.eval_start} → {args.eval_end}\n"
        f"IQN checkpoint: {iqn_ckpt}\n"
        f"HDP strategy: {args.strategy}\n\n"
        "## Run Statistics\n\n"
        f"- Total decision steps: {len(audit_df)}\n"
        f"- Trading days: {trading_days}\n"
        f"- Tickers: {len(_UNIVERSE_TICKERS.get(args.universe, []))}\n"
        f"- Total duration: {int(dur_sec // 60)}m {int(dur_sec % 60)}s\n"
        f"- Audit columns: {len(audit_df.columns)}\n\n"
        "## Action Distribution (IQN raw)\n\n"
        "| Action | Count | % |\n"
        "|--------|-------|---|\n"
        f"{_action_table(iqn_dist)}\n\n"
        "## HDP Action Distribution (after rule layer)\n\n"
        "| Action | Count | % |\n"
        "|--------|-------|---|\n"
        f"{_action_table(hdp_dist)}\n\n"
        "## Risk Validator Interventions\n\n"
        f"- HDP overrides total: {overrides}\n"
        f"{ticker_section}"
        "\n## Column Naming\n\n"
        "This CSV uses dual naming for backward compatibility:\n\n"
        "| Primary policy column       | Alias added |\n"
        "|-----------------------------|-------------|\n"
        "| `hierarchical_action_type`  | `selected_action_type` |\n"
        "| `iqn_model_run_id`          | `source_iqn_run_id`    |\n\n"
        "## Output Files\n\n"
        f"- Key CSV (filtered): `{key_csv}`\n"
        f"- Full CSV:           `{full_csv}`\n"
        f"- Metrics:            `{run_paths.metrics_directory / 'audit_summary.json'}`\n"
        f"- Run directory:      `{run_paths.run_directory}`\n"
    )
    (run_paths.summary_directory / "audit_summary.md").write_text(md, encoding="utf-8")
    (run_paths.summary_directory / "audit_summary.json").write_text(
        json.dumps(metrics, indent=2, default=str), encoding="utf-8"
    )

    # -----------------------------------------------------------------------
    # 14. W&B metrics
    # -----------------------------------------------------------------------
    n = max(len(audit_df), 1)
    wandb_log(
        {
            "audit_total_steps": len(audit_df),
            "audit_trading_days": trading_days,
            "eval_window_days": trading_days,
            "hdp_intervention_count": overrides,
            **{f"audit_iqn_action_pct_{k.lower()}": v / n for k, v in iqn_dist.items()},
            **{f"audit_hdp_action_pct_{k.lower()}": v / n for k, v in hdp_dist.items()},
        }
    )

    logger.info("=== Phase B.1 audit complete ===")
    logger.info("Run directory:          %s", run_paths.run_directory)
    logger.info("Audit rows (filtered):  %d", len(audit_df))
    logger.info("Audit columns:          %d", len(audit_df.columns))
    logger.info("EDL-A readiness:        %s", metrics["edl_a_readiness"])
    logger.info("Duration:               %.1f s", dur_sec)

    try:
        finish_wandb_run()
    except Exception:  # noqa: BLE001
        pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
