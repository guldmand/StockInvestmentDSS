#!/usr/bin/env python3
"""EDL calibration plots — does the evidential uncertainty (vacuity) mean anything?

Two panels in one figure, read entirely from the EDL test-predictions CSV
(no hardcoded values):

  1. Error-vs-vacuity bucket plot — predictions binned by vacuity; per-bin error
     rate (1 - accuracy). Calibrated == rising curve (more uncertain -> more wrong).

  2. Accuracy-rejection curve — sort by vacuity (most uncertain first); reject the
     top fraction and measure accuracy on what remains. Calibrated == accuracy
     rises as rejection rises (deferring uncertain cases to a human helps).

Data source (auto-detected): newest *_demo500_summary/layer3_edl/
edl_v3_test_predictions_ensemble.csv  (override with --csv).

Output: outputs/edl_calibration_plots.png AND
        <summary_dir>/layer3_edl/edl_calibration_plots.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

VACUITY_CANDIDATES = ["vacuity", "uncertainty", "vacuity_score",
                      "epistemic_uncertainty", "u"]
PRED_CANDIDATES = ["pred_label", "predicted_label", "prediction", "pred", "y_pred"]
TRUE_CANDIDATES = ["true_label", "label", "y_true", "target", "actual"]

EDL_CSV_NAME = "edl_v3_test_predictions_ensemble.csv"


def _fail(msg: str) -> None:
    raise SystemExit(f"[edl_calibration_plots] ERROR: {msg}")


def _pick(df: pd.DataFrame, candidates: list[str], what: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    _fail(f"could not find a {what} column. Tried {candidates}. "
          f"Available columns: {list(df.columns)}")
    return ""  # unreachable


def resolve_csv(cli_csv: str | None) -> Path:
    if cli_csv:
        p = Path(cli_csv) if Path(cli_csv).is_absolute() else ROOT / cli_csv
        return p if p.exists() else _fail(f"--csv not found: {p}")  # type: ignore
    summaries = sorted((ROOT / "outputs" / "runs").glob("*_d_iqn_dss_demo500_summary"))
    if not summaries:
        _fail("no *_demo500_summary dir found; pass --csv explicitly")
    csv = summaries[-1] / "layer3_edl" / EDL_CSV_NAME
    if not csv.exists():
        _fail(f"expected EDL predictions at {csv}")
    return csv


def build(csv_path: Path, n_bins: int) -> None:
    df = pd.read_csv(csv_path)
    vac_col = _pick(df, VACUITY_CANDIDATES, "vacuity")
    pred_col = _pick(df, PRED_CANDIDATES, "predicted-label")
    true_col = _pick(df, TRUE_CANDIDATES, "true-label")

    vac = df[vac_col].astype(float).to_numpy()
    correct = (df[pred_col].astype(str).str.upper()
               == df[true_col].astype(str).str.upper()).to_numpy()
    n = len(df)

    # ---- console: columns + stats --------------------------------------
    print(f"[edl_calibration_plots] CSV: {csv_path}")
    print(f"  predictions (rows): {n}")
    print(f"  columns used  -> vacuity={vac_col!r}  predicted={pred_col!r}  true={true_col!r}")
    if "correct" in df.columns:
        agree = bool((df["correct"].astype(bool).to_numpy() == correct).all())
        print(f"  (CSV also has a 'correct' column; matches pred==true: {agree})")
    print(f"  overall accuracy: {correct.mean():.4f}")
    print(f"  vacuity stats -> min={vac.min():.4f}  mean={vac.mean():.4f}  "
          f"max={vac.max():.4f}  std={vac.std():.4f}")

    # ---- Panel 1: error-vs-vacuity bins (equal width) ------------------
    edges = np.linspace(vac.min(), vac.max(), n_bins + 1)
    bin_idx = np.clip(np.digitize(vac, edges[1:-1]), 0, n_bins - 1)
    centers, err_rates, sizes, accs = [], [], [], []
    print(f"\n  Per-bin (equal-width vacuity bins, n_bins={n_bins}):")
    print(f"    {'bin':>3} {'range':>17} {'n':>4} {'acc':>7} {'err':>7}")
    for b in range(n_bins):
        mask = bin_idx == b
        size = int(mask.sum())
        lo, hi = edges[b], edges[b + 1]
        centers.append((lo + hi) / 2)
        sizes.append(size)
        if size == 0:
            accs.append(np.nan)
            err_rates.append(np.nan)
            print(f"    {b:>3} [{lo:.3f},{hi:.3f}] {size:>4} {'--':>7} {'--':>7}")
        else:
            acc = correct[mask].mean()
            accs.append(acc)
            err_rates.append(1.0 - acc)
            print(f"    {b:>3} [{lo:.3f},{hi:.3f}] {size:>4} {acc:>7.3f} {1-acc:>7.3f}")

    # ---- Panel 2: accuracy-rejection curve -----------------------------
    order = np.argsort(vac)            # ascending: most certain first
    correct_sorted = correct[order]
    vac_sorted = vac[order]
    # keep k most-certain (k = 1..n); rejection = (n-k)/n
    ks = np.arange(1, n + 1)
    acc_kept = np.cumsum(correct_sorted) / ks
    rejection = (n - ks) / n           # 0 .. (n-1)/n

    # optimal rejection: max accuracy with a coverage guard (keep >= 10%)
    min_keep = max(1, int(np.ceil(0.10 * n)))
    valid = ks >= min_keep
    best_local = np.argmax(acc_kept[valid])
    best_k = ks[valid][best_local]
    best_rej = rejection[valid][best_local]
    best_acc = acc_kept[valid][best_local]
    # vacuity threshold = highest vacuity still kept at that point
    vac_thresh = vac_sorted[best_k - 1]
    print(f"\n  Accuracy-rejection (coverage guard: keep >= {min_keep} = 10%):")
    for r in (0.0, 0.1, 0.2, 0.3, 0.5):
        k = max(1, int(round((1 - r) * n)))
        print(f"    reject {r*100:4.0f}%  keep {k:>3}  acc {np.cumsum(correct_sorted)[k-1]/k:.3f}")
    print(f"  OPTIMAL: reject {best_rej*100:.1f}%  (keep {best_k}/{n})  "
          f"acc {best_acc:.3f}  vacuity_threshold {vac_thresh:.4f}")

    # ---- figure ---------------------------------------------------------
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        "EDL Calibration — does evidential vacuity track decision error?\n"
        f"EDL action classifier · {n} test predictions · overall accuracy "
        f"{correct.mean()*100:.1f}%",
        fontsize=13, fontweight="bold",
    )

    # Panel 1
    cx = np.array(centers)
    cy = np.array(err_rates)
    ok = ~np.isnan(cy)
    axL.plot(cx[ok], cy[ok], "o-", color="#C0392B", linewidth=2, markersize=8)
    axL.axhline(1 - correct.mean(), color="gray", ls="--", lw=1,
                label=f"overall error ({1-correct.mean():.2f})")
    for x, y, s in zip(centers, err_rates, sizes):
        if not np.isnan(y):
            axL.annotate(f"n={s}", (x, y), textcoords="offset points",
                         xytext=(0, 8), ha="center", fontsize=8)
    axL.set_xlabel("Vacuity (uncertainty) bin centre — low → high")
    axL.set_ylabel("Error rate (1 − accuracy)")
    axL.set_title("Error vs. vacuity\n(calibrated = rising)", fontweight="bold")
    axL.grid(alpha=0.3)
    axL.legend(fontsize=9)

    # Panel 2
    axR.plot(rejection * 100, acc_kept, "-", color="#1F77B4", linewidth=2)
    axR.axhline(correct.mean(), color="gray", ls="--", lw=1,
                label=f"no rejection ({correct.mean():.2f})")
    axR.scatter([best_rej * 100], [best_acc], color="#E67E22", zorder=5, s=70,
                label=f"optimal: reject {best_rej*100:.0f}% → acc {best_acc:.2f}")
    axR.set_xlabel("Rejection rate (% most-uncertain deferred to human)")
    axR.set_ylabel("Accuracy on remaining predictions")
    axR.set_title("Accuracy-rejection curve\n(calibrated = rising)", fontweight="bold")
    axR.grid(alpha=0.3)
    axR.legend(fontsize=9, loc="lower right")

    fig.tight_layout(rect=(0, 0, 1, 0.93))

    # ---- save to both locations ----------------------------------------
    flat = ROOT / "outputs" / "edl_calibration_plots.png"
    pkg = csv_path.parent / "edl_calibration_plots.png"
    for out in (flat, pkg):
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"  saved: {out}")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default=None, help="EDL predictions CSV (default: auto).")
    ap.add_argument("--bins", type=int, default=5, help="number of vacuity bins.")
    args = ap.parse_args()
    build(resolve_csv(args.csv), args.bins)
    return 0


if __name__ == "__main__":
    sys.exit(main())
