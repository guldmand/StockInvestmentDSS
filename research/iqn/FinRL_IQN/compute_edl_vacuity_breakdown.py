"""
Beregner per-klasse vacuity og correct/incorrect vacuity for alle tre EDL-tilgange.
Kør i samme conda-env (ml) som dine andre kørsler.

Brug:
    cd /home/guldmand/Dropbox/DataScience/Speciale/Devices/dev/StockInvestmentDSS/research/iqn/FinRL_IQN
    python compute_edl_vacuity_breakdown.py > metric_dumps/EDL_vacuity_breakdown.txt
"""

import pandas as pd
from pathlib import Path

RUN = Path("outputs/runs/2026_05_28_150606_d_iqn_dss_edl_action_training_v3_production")
AUDIT = RUN / "audit"

approaches = {
    "ensemble_10fold":            AUDIT / "edl_v3_test_predictions_ensemble.csv",
    "single_best_fold":           AUDIT / "edl_v3_test_predictions_single_best_fold.csv",
    "single_final_full_trainstar": AUDIT / "edl_v3_test_predictions_single_final.csv",
}

for name, path in approaches.items():
    print("=" * 70)
    print(f"  {name}")
    print("=" * 70)
    df = pd.read_csv(path)
    df = df.dropna(subset=["true_label"])  # drop evt. blanke linjer

    n = len(df)
    overall_vac = df["vacuity"].mean()
    correct = df[df["correct"] == True]
    incorrect = df[df["correct"] == False]

    print(f"  N rows:                {n}")
    print(f"  Overall mean vacuity:  {overall_vac:.4f}")
    print()

    print("  --- Per-klasse vacuity (grupperet på SAND klasse) ---")
    per_true = df.groupby("true_label")["vacuity"].agg(["count", "mean", "std"]).round(4)
    print(per_true.to_string())
    print()

    print("  --- Per-klasse vacuity (grupperet på FORUDSAGT klasse) ---")
    per_pred = df.groupby("pred_label")["vacuity"].agg(["count", "mean", "std"]).round(4)
    print(per_pred.to_string())
    print()

    print("  --- VIGTIGT: Vacuity for CORRECT vs INCORRECT ---")
    print(f"  Correct   (n={len(correct):3d}):  mean vacuity = {correct['vacuity'].mean():.4f}, std = {correct['vacuity'].std():.4f}")
    print(f"  Incorrect (n={len(incorrect):3d}):  mean vacuity = {incorrect['vacuity'].mean():.4f}, std = {incorrect['vacuity'].std():.4f}")
    gap = incorrect["vacuity"].mean() - correct["vacuity"].mean()
    print(f"  Gap (incorrect - correct):     {gap:+.4f}   (positiv = modellen ER mere usikker når den tager fejl, GODT)")
    print()

    print("  --- Per-klasse: correct vs incorrect vacuity ---")
    for cls in ["HOLD", "BUY", "SELL"]:
        cls_df = df[df["true_label"] == cls]
        c = cls_df[cls_df["correct"] == True]
        i = cls_df[cls_df["correct"] == False]
        c_vac = c["vacuity"].mean() if len(c) else float("nan")
        i_vac = i["vacuity"].mean() if len(i) else float("nan")
        print(f"    {cls:5s}: correct n={len(c):2d} vac={c_vac:.4f}  |  incorrect n={len(i):2d} vac={i_vac:.4f}")
    print()
