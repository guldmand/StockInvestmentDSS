#!/usr/bin/env python3
"""Assemble a self-contained DEMO500 thesis-summary deliverable folder.

Copies the relevant source artifacts (no .pt model files) from every pipeline
layer into one timestamped folder, and generates:
  - aggregate_metrics.csv   (all 20 strategies; values come from the dashboard
                             loaders, so they match the dashboard exactly)
  - inputs_manifest.json    (the 13 source run dirs + timestamps)
  - README.md               (one-page description)
  - logs/build.log          (full copy log)

No hardcoded performance values. Fails loud if any source file is missing.

EDL source: MERGED_2026_05_28_d_iqn_dss_edl_action_training_v3_COMPLETE
(the merged/complete output that the Layer 4 backtest actually consumes).
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# Reuse the dashboard's loaders + path constants so numbers stay consistent.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import portfolio_summary_dashboard as dash  # noqa: E402

ROOT = dash.ROOT
log = logging.getLogger("build_demo500_summary")

# EDL deliverables (relative path in MERGED dir -> flat name in layer3_edl/)
MERGED_EDL_DIR = "outputs/runs/MERGED_2026_05_28_d_iqn_dss_edl_action_training_v3_COMPLETE"
EDL_FILES = [
    ("audit/edl_v3_test_predictions_ensemble.csv", "edl_v3_test_predictions_ensemble.csv"),
    ("audit/edl_v3_test_confusion_matrix_ensemble_10fold.csv",
     "edl_v3_test_confusion_matrix_ensemble_10fold.csv"),
    ("summary/edl_v3_training_summary.json", "edl_v3_training_summary.json"),
    ("summary/edl_v3_training_summary.md", "edl_v3_training_summary.md"),
    ("plots/final_training_curves.png", "final_training_curves.png"),
    ("plots/test_uncertainty_distribution.png", "test_uncertainty_distribution.png"),
    ("plots/test_confusion_matrix.png", "test_confusion_matrix.png"),
    ("plots/cv_per_fold_metrics.png", "cv_per_fold_metrics.png"),
]

LAYER2_FILES = [
    "finrl_baseline_multiseed_aggregate_by_agent.csv",
    "finrl_baseline_multiseed_aggregate_by_strategy.csv",
    "finrl_baseline_multiseed_total_return_pct_mean_std.png",
    "finrl_baseline_multiseed_annualized_sharpe_mean_std.png",
    "finrl_baseline_multiseed_max_drawdown_pct_mean_std.png",
    "finrl_baseline_multiseed_cvar_pct_mean_std.png",
]
LAYER1A_FILES = ["algorithmic_baselines_summary.csv", "portfolio_strategy_summary.csv"]
LAYER4_ABLATIONS = ["a1", "a2", "a3", "a4"]
LAYER4_PER_ABLATION = ["account_values.csv", "actions.csv", "metrics.csv"]


def _fail(msg: str) -> None:
    raise SystemExit(f"[build_demo500_summary] ERROR: {msg}")


def _rel(p: Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def copy_file(src: Path, dst: Path) -> None:
    if not src.exists():
        _fail(f"missing source file: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    log.info("cp %s  ->  %s", _rel(src), _rel(dst))


def _mtime(p: Path) -> str:
    return datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds")


def build(layer4_dir: Path, ts: str) -> Path:
    out_dir = ROOT / "outputs" / "runs" / f"{ts}_d_iqn_dss_demo500_summary"
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)

    # attach a file handler now that logs/ exists
    fh = logging.FileHandler(out_dir / "logs" / "build.log", mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(fh)

    log.info("=== building DEMO500 summary at %s ===", _rel(out_dir))
    log.info("EDL source: %s", MERGED_EDL_DIR)
    log.info("Layer 4 source: %s", _rel(layer4_dir))

    # ---- derived source dirs (from the dashboard's path constants) ----------
    l1a_summary = ROOT / Path(dash.LAYER1A_PORTFOLIO).parent     # .../summary
    l1a_run = l1a_summary.parent                                 # run dir
    l2_summary = ROOT / Path(dash.LAYER2_AGGREGATE).parent       # .../summary
    l2_run = l2_summary.parent
    edl_dir = ROOT / MERGED_EDL_DIR

    # ---- Layer 1a -----------------------------------------------------------
    for f in LAYER1A_FILES:
        copy_file(l1a_summary / f, out_dir / "layer1a" / f)

    # ---- Layer 1b (clean per-strategy names) --------------------------------
    for rel in dash.LAYER1B_DIRS:
        run_dir = ROOT / rel
        matches = list(
            run_dir.glob("metrics/single_ticker_portfolio/*/portfolio_metrics.csv")
        )
        if len(matches) != 1:
            _fail(f"expected 1 portfolio_metrics.csv under {run_dir}, got {len(matches)}")
        strategy = matches[0].parent.name
        copy_file(matches[0], out_dir / "layer1b" / f"{strategy}_portfolio_metrics.csv")

    # ---- Layer 2 ------------------------------------------------------------
    for f in LAYER2_FILES:
        copy_file(l2_summary / f, out_dir / "layer2" / f)

    # ---- Layer 3 EDL --------------------------------------------------------
    for src_rel, dst_name in EDL_FILES:
        copy_file(edl_dir / src_rel, out_dir / "layer3_edl" / dst_name)

    # ---- Layer 4 scenarioX --------------------------------------------------
    copy_file(layer4_dir / "ablation_summary.csv",
              out_dir / "layer4_scenarioX" / "ablation_summary.csv")
    for abl in LAYER4_ABLATIONS:
        for f in LAYER4_PER_ABLATION:
            copy_file(layer4_dir / abl / f, out_dir / "layer4_scenarioX" / abl / f)

    # ---- main dashboard figure ---------------------------------------------
    copy_file(ROOT / "outputs" / "portfolio_summary_dashboard.png",
              out_dir / "portfolio_summary_dashboard.png")

    # ---- aggregate_metrics.csv (values from dashboard loaders) --------------
    l1b = dash.load_layer1b()
    l1a = dash.load_layer1a_portfolio()
    l2 = dash.load_layer2()
    l4 = dash.load_layer4_metrics(layer4_dir)
    agg = pd.concat([l1b, l1a, l2, l4], ignore_index=True)
    agg = agg.sort_values("return", ascending=False).reset_index(drop=True)
    agg_path = out_dir / "aggregate_metrics.csv"
    agg.to_csv(agg_path, index=False)
    log.info("generated %s (%d strategies)", _rel(agg_path), len(agg))

    # ---- inputs_manifest.json (13 source run dirs) --------------------------
    source_runs = []
    for rel in dash.LAYER1B_DIRS:
        d = ROOT / rel
        source_runs.append({"layer": "1b", "role": "single-ticker rule-based portfolio",
                            "path": rel, "mtime": _mtime(d)})
    source_runs.append({"layer": "1a", "role": "algorithmic baselines + portfolio benchmarks",
                        "path": _rel(l1a_run), "mtime": _mtime(l1a_run)})
    source_runs.append({"layer": "2", "role": "FinRL parametric RL multiseed",
                        "path": _rel(l2_run), "mtime": _mtime(l2_run)})
    source_runs.append({"layer": "3_edl", "role": "EDL action classifier (merged/complete)",
                        "path": MERGED_EDL_DIR, "mtime": _mtime(edl_dir)})
    source_runs.append({"layer": "4_scenarioX", "role": "IQN+HDP+EDL ablation backtest",
                        "path": _rel(layer4_dir), "mtime": _mtime(layer4_dir)})

    manifest = {
        "generated_at": ts,
        "summary_dir": _rel(out_dir),
        "universe": "SP500 (467 tickers)",
        "trade_window": "2024-01-01 to 2026-05-26 (554 trading days)",
        "n_source_runs": len(source_runs),
        "source_runs": source_runs,
    }
    manifest_path = out_dir / "inputs_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    log.info("generated %s (%d source runs)", _rel(manifest_path), len(source_runs))

    # ---- README.md ----------------------------------------------------------
    readme = f"""# DEMO500 Thesis Summary

Self-contained deliverable bundling every artifact behind the D-IQN-DSS
portfolio comparison. Generated {ts}.

**Universe:** SP500 (467 tickers) · **Window:** 2024-01-01 to 2026-05-26
(554 trading days) · **Initial capital:** $1,000,000.

## Layout
- `portfolio_summary_dashboard.png` — main 4-panel figure (return, drawdown,
  Sharpe, per-ablation action distribution).
- `aggregate_metrics.csv` — all 20 strategies (return, Sharpe, max drawdown,
  category). Values are produced by the dashboard loaders, so they match the
  figure exactly. **This is the canonical numbers table.**
- `inputs_manifest.json` — the {len(source_runs)} source run directories used.
- `layer1a/` — algorithmic baselines + classical portfolio benchmarks
  (`scope=portfolio`: equal-weight buy&hold, naive 1/N rebalanced).
- `layer1b/` — 9 single-ticker rule-based portfolio strategies (467 tickers).
- `layer2/` — FinRL parametric RL multiseed (A2C/PPO/DDPG/SAC/TD3) + MVO.
- `layer3_edl/` — EDL action classifier: test predictions (for calibration
  plots), confusion matrix, training summary (.json/.md) and diagnostic plots.
- `layer4_scenarioX/` — IQN+HDP+EDL 4-ablation backtest (A1-A4): per-ablation
  `account_values.csv`, `actions.csv`, `metrics.csv` + `ablation_summary.csv`.

## Notes
- Layer 4 is an offline replay of the IQN decisions (see thesis methodology);
  `actions.csv` records the raw IQN action and the action executed after
  HDP/EDL governance per ablation.
- `.pt` model weights are intentionally excluded (large, not needed for the
  thesis write-up).
"""
    readme_path = out_dir / "README.md"
    readme_path.write_text(readme)
    log.info("generated %s", _rel(readme_path))

    log.info("=== done: %s ===", _rel(out_dir))
    return out_dir


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--layer4-dir", default=None,
                    help="Layer 4 scenarioX dir (default: newest *_scenarioX run).")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])

    layer4_dir = dash.resolve_layer4_dir(args.layer4_dir)
    ts = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    out_dir = build(layer4_dir, ts)
    print(f"\n[build_demo500_summary] SUMMARY DIR: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
