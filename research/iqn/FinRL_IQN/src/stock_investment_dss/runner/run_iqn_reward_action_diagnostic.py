# src/stock_investment_dss/runner/run_iqn_reward_action_diagnostic.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from stock_investment_dss.utilities.config import get_environment_variable
from stock_investment_dss.utilities.logging import setup_run_logger, setup_system_logger
from stock_investment_dss.utilities.paths import PROJECT_ROOT, create_run_paths


RUN_KIND = "d_iqn_dss_iqn_reward_action_diagnostic"


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def n(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in df.columns:
        return pd.Series([default] * len(df), index=df.index)
    return pd.to_numeric(df[column], errors="coerce").fillna(default)


def latest_summary_run() -> Path:
    explicit = get_environment_variable("STOCK_INVESTMENT_DSS_IQN_REWARD_DIAGNOSTIC_SOURCE_SUMMARY_RUN_ID", default="")
    if explicit:
        path = PROJECT_ROOT / "outputs" / "runs" / explicit
        if not path.exists():
            raise FileNotFoundError(path)
        return path.resolve()

    root = PROJECT_ROOT / "outputs" / "runs"
    candidates = sorted(
        [
            p for p in root.iterdir()
            if p.is_dir()
            and p.name.endswith("d_iqn_dss_iqn_learning_curve_multiseed_summary")
            and (p / "summary" / "iqn_learning_curve_multiseed_final_records.csv").exists()
            and (p / "data" / "iqn_learning_curve_multiseed_run_index.csv").exists()
        ],
        key=lambda p: p.name,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError("No IQN multiseed summary run found.")
    return candidates[0].resolve()


def child_run_for_seed(seed: int, run_index: pd.DataFrame) -> Path | None:
    rows = run_index[run_index["seed"].astype("Int64") == int(seed)]
    if rows.empty:
        return None
    row = rows.iloc[-1].to_dict()
    candidates = [row.get("run_directory"), row.get("source_run_directory"), row.get("source_run_dir")]
    run_id = row.get("run_id") or row.get("source_run_id")
    if run_id:
        candidates.append(PROJECT_ROOT / "outputs" / "runs" / str(run_id))
    for c in candidates:
        if c is None or str(c).strip() == "":
            continue
        p = Path(str(c))
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        if p.exists():
            return p.resolve()
    return None


def classify_final(final_records: pd.DataFrame) -> pd.DataFrame:
    df = final_records.copy()
    for c in ["seed", "total_trades", "total_return_pct"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["seed_status"] = "unknown"
    df.loc[(df.get("total_trades", 0).fillna(0) <= 0) & (df.get("total_return_pct", 0).fillna(0).abs() < 1e-9), "seed_status"] = "no_trade"
    df.loc[df.get("total_trades", 0).fillna(0) > 0, "seed_status"] = "active_trading"
    return df


def summarize_training(seed: int, status: str, child_id: str, training: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    if training.empty:
        return pd.DataFrame(), {"seed": seed, "seed_status": status, "child_run_id": child_id, "training_found": False}

    df = training.copy()
    df["reward_numeric"] = n(df, "reward")
    df["portfolio_value_after_numeric"] = n(df, "portfolio_value_after")
    df["cash_after_numeric"] = n(df, "cash_after")
    df["cash_weight_proxy"] = df["cash_after_numeric"] / df["portfolio_value_after_numeric"].replace(0, pd.NA)
    df["cash_weight_proxy"] = pd.to_numeric(df["cash_weight_proxy"], errors="coerce")
    df["cash_only_proxy"] = df["cash_weight_proxy"].fillna(0) >= 0.995
    df["finrl_trades_numeric"] = n(df, "finrl_trades")
    df["finrl_cost_numeric"] = n(df, "finrl_cost")
    df["loss_numeric"] = pd.to_numeric(df.get("loss", pd.Series(index=df.index)), errors="coerce")
    df["epsilon_numeric"] = n(df, "epsilon")

    action_series = df.get("action_label", pd.Series(index=df.index, dtype=str)).astype(str).str.upper()
    hold_cash = df[(action_series == "HOLD") & df["cash_only_proxy"]]
    hold_invested = df[(action_series == "HOLD") & (~df["cash_only_proxy"])]
    buy = df[action_series == "BUY"]

    by_action = (
        df.assign(action_label_upper=action_series)
        .groupby("action_label_upper", dropna=False)
        .agg(
            row_count=("reward_numeric", "count"),
            reward_mean=("reward_numeric", "mean"),
            reward_median=("reward_numeric", "median"),
            reward_min=("reward_numeric", "min"),
            reward_max=("reward_numeric", "max"),
            positive_reward_count=("reward_numeric", lambda s: int((s > 0).sum())),
            negative_reward_count=("reward_numeric", lambda s: int((s < 0).sum())),
            zero_reward_count=("reward_numeric", lambda s: int((s == 0).sum())),
            cash_weight_proxy_mean=("cash_weight_proxy", "mean"),
            cash_only_proxy_count=("cash_only_proxy", "sum"),
            finrl_trades_mean=("finrl_trades_numeric", "mean"),
            finrl_cost_mean=("finrl_cost_numeric", "mean"),
        )
        .reset_index()
        .rename(columns={"action_label_upper": "action"})
    )
    by_action.insert(0, "child_run_id", child_id)
    by_action.insert(0, "seed_status", status)
    by_action.insert(0, "seed", seed)

    summary = {
        "seed": seed,
        "seed_status": status,
        "child_run_id": child_id,
        "training_found": True,
        "training_rows": int(len(df)),
        "training_hold_count": int((action_series == "HOLD").sum()),
        "training_buy_count": int((action_series == "BUY").sum()),
        "training_sell_count": int((action_series == "SELL").sum()),
        "training_rebalance_count": int((action_series == "REBALANCE").sum()),
        "training_reward_mean": float(df["reward_numeric"].mean()),
        "training_reward_sum": float(df["reward_numeric"].sum()),
        "hold_cash_rows": int(len(hold_cash)),
        "hold_cash_reward_mean": float(hold_cash["reward_numeric"].mean()) if not hold_cash.empty else None,
        "hold_invested_rows": int(len(hold_invested)),
        "hold_invested_reward_mean": float(hold_invested["reward_numeric"].mean()) if not hold_invested.empty else None,
        "buy_rows": int(len(buy)),
        "buy_reward_mean": float(buy["reward_numeric"].mean()) if not buy.empty else None,
        "buy_positive_reward_count": int((buy["reward_numeric"] > 0).sum()) if not buy.empty else 0,
        "buy_negative_reward_count": int((buy["reward_numeric"] < 0).sum()) if not buy.empty else 0,
        "cash_only_proxy_share": float(df["cash_only_proxy"].mean()),
        "loss_mean": float(df["loss_numeric"].dropna().mean()) if df["loss_numeric"].notna().any() else None,
        "loss_final": float(df["loss_numeric"].dropna().iloc[-1]) if df["loss_numeric"].notna().any() else None,
        "epsilon_final": float(df["epsilon_numeric"].iloc[-1]) if len(df) else None,
    }
    return by_action, summary


def summarize_final_eval(seed: int, status: str, child_id: str, eval_steps: pd.DataFrame, dist: pd.DataFrame) -> dict[str, Any]:
    out = {"seed": seed, "seed_status": status, "child_run_id": child_id}
    if not eval_steps.empty:
        df = eval_steps.copy()
        if "train_step" in df.columns:
            df["train_step"] = pd.to_numeric(df["train_step"], errors="coerce")
            m = df["train_step"].max()
            df = df[df["train_step"] == m]
            out["final_train_step"] = int(m)
        counts = df.get("chosen_action_label", pd.Series(dtype=str)).fillna("UNKNOWN").value_counts().to_dict()
        out["final_eval_rows"] = int(len(df))
        out["final_eval_action_counts_json"] = json.dumps(counts, sort_keys=True)
        out["final_eval_hold_count"] = int(counts.get("HOLD", 0))
        out["final_eval_buy_count"] = int(counts.get("BUY", 0))
        out["final_eval_trade_rows"] = int((n(df, "trades_delta") > 0).sum()) if "trades_delta" in df.columns else 0
        out["final_eval_masked_count"] = int(df.get("action_was_masked", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()) if "action_was_masked" in df.columns else 0

    if not dist.empty:
        d = dist.copy()
        if "train_step" in d.columns:
            d["train_step"] = pd.to_numeric(d["train_step"], errors="coerce")
            d = d[d["train_step"] == d["train_step"].max()]
        for c in ["score", "q50", "cvar10", "mean"]:
            if c in d.columns:
                d[c] = pd.to_numeric(d[c], errors="coerce")
        def avg(action: str, col: str):
            rows = d[d.get("action", pd.Series(dtype=str)).astype(str).str.upper() == action]
            if rows.empty or col not in rows.columns:
                return None
            v = rows[col].mean()
            return float(v) if pd.notna(v) else None
        out["hold_score_mean"] = avg("HOLD", "score")
        out["buy_score_mean"] = avg("BUY", "score")
        out["hold_q50_mean"] = avg("HOLD", "q50")
        out["buy_q50_mean"] = avg("BUY", "q50")
        out["hold_cvar10_mean"] = avg("HOLD", "cvar10")
        out["buy_cvar10_mean"] = avg("BUY", "cvar10")
        if out.get("hold_score_mean") is not None and out.get("buy_score_mean") is not None:
            out["hold_minus_buy_score"] = out["hold_score_mean"] - out["buy_score_mean"]
    return out


def save_action_plot(seed_summary: pd.DataFrame, path: Path) -> None:
    if seed_summary.empty:
        return
    df = seed_summary.copy()
    for c in ["training_hold_count", "training_buy_count", "training_sell_count", "training_rebalance_count"]:
        if c not in df.columns:
            df[c] = 0
    x = range(len(df))
    width = 0.2
    plt.figure(figsize=(12, 5))
    plt.bar([i - 1.5 * width for i in x], df["training_hold_count"], width=width, label="HOLD")
    plt.bar([i - 0.5 * width for i in x], df["training_buy_count"], width=width, label="BUY")
    plt.bar([i + 0.5 * width for i in x], df["training_sell_count"], width=width, label="SELL")
    plt.bar([i + 1.5 * width for i in x], df["training_rebalance_count"], width=width, label="REBALANCE")
    plt.xticks(list(x), df["seed"].astype(str))
    plt.title("IQN training action counts by seed")
    plt.xlabel("Seed")
    plt.ylabel("Count")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def save_gap_plot(seed_summary: pd.DataFrame, path: Path) -> None:
    if seed_summary.empty or "hold_minus_buy_score" not in seed_summary.columns:
        return
    df = seed_summary.copy()
    df["hold_minus_buy_score"] = pd.to_numeric(df["hold_minus_buy_score"], errors="coerce")
    df = df.dropna(subset=["hold_minus_buy_score"])
    if df.empty:
        return
    plt.figure(figsize=(12, 5))
    plt.bar(df["seed"].astype(str), df["hold_minus_buy_score"])
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.title("Final evaluation: HOLD score minus BUY score")
    plt.xlabel("Seed")
    plt.ylabel("Score gap")
    plt.grid(axis="y", alpha=0.25)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def main() -> int:
    log_level = get_environment_variable("STOCK_INVESTMENT_DSS_LOG_LEVEL", default="INFO") or "INFO"
    system_logger = setup_system_logger(log_level=log_level)
    system_logger.info("Starting StockInvestmentDSS IQN reward/action diagnostic.")
    run_paths = None
    try:
        source = latest_summary_run()
        final_records = classify_final(pd.read_csv(source / "summary" / "iqn_learning_curve_multiseed_final_records.csv"))
        run_index = pd.read_csv(source / "data" / "iqn_learning_curve_multiseed_run_index.csv")
        run_index["seed"] = pd.to_numeric(run_index["seed"], errors="coerce")

        seed_env = get_environment_variable("STOCK_INVESTMENT_DSS_IQN_REWARD_DIAGNOSTIC_SEEDS", default="")
        if seed_env:
            seeds = sorted({int(s.strip()) for s in seed_env.split(",") if s.strip()})
            final_records = final_records[final_records["seed"].astype("Int64").isin(seeds)]
        else:
            seeds = sorted(int(s) for s in final_records["seed"].dropna().unique().tolist())

        run_paths = create_run_paths(RUN_KIND)
        logger = setup_run_logger(run_paths, log_level=log_level)
        logger.info("Source summary run: %s", source)
        logger.info("Seeds: %s", seeds)

        seed_rows, action_frames = [], []
        for _, row in final_records.iterrows():
            seed = int(row["seed"])
            status = str(row.get("seed_status", "unknown"))
            child = child_run_for_seed(seed, run_index)
            if child is None:
                seed_rows.append({"seed": seed, "seed_status": status, "child_run_found": False})
                continue
            training = safe_read_csv(child / "data" / "iqn_learning_curve_training_records.csv")
            eval_steps = safe_read_csv(child / "data" / "iqn_learning_curve_eval_step_records.csv")
            dist = safe_read_csv(child / "data" / "iqn_learning_curve_eval_distributions.csv")
            by_action, train_summary = summarize_training(seed, status, child.name, training)
            eval_summary = summarize_final_eval(seed, status, child.name, eval_steps, dist)
            seed_rows.append({**{f"final_{k}": v for k, v in row.to_dict().items()}, **train_summary, **eval_summary, "child_run_found": True})
            if not by_action.empty:
                action_frames.append(by_action)

        seed_summary = pd.DataFrame(seed_rows).sort_values("seed") if seed_rows else pd.DataFrame()
        by_action = pd.concat(action_frames, ignore_index=True) if action_frames else pd.DataFrame()

        numeric_cols = [
            c for c in [
                "training_reward_mean", "hold_cash_reward_mean", "buy_reward_mean",
                "cash_only_proxy_share", "loss_mean", "loss_final",
                "final_eval_hold_count", "final_eval_buy_count", "final_eval_trade_rows",
                "hold_score_mean", "buy_score_mean", "hold_minus_buy_score",
            ] if c in seed_summary.columns
        ]
        status_comparison = seed_summary.groupby("seed_status")[numeric_cols].agg(["mean", "min", "max"]).reset_index() if numeric_cols else pd.DataFrame()

        seed_path = run_paths.summary_directory / "iqn_reward_action_diagnostic_by_seed.csv"
        action_path = run_paths.summary_directory / "iqn_reward_action_diagnostic_training_by_action.csv"
        status_path = run_paths.summary_directory / "iqn_reward_action_diagnostic_status_comparison.csv"
        json_path = run_paths.summary_directory / "iqn_reward_action_diagnostic_summary.json"
        md_path = run_paths.summary_directory / "iqn_reward_action_diagnostic_summary.md"
        action_plot = run_paths.plots_directory / "iqn_reward_action_training_action_counts.png"
        gap_plot = run_paths.plots_directory / "iqn_reward_action_hold_buy_gap.png"

        seed_summary.to_csv(seed_path, index=False)
        by_action.to_csv(action_path, index=False)
        status_comparison.to_csv(status_path, index=False)
        save_action_plot(seed_summary, action_plot)
        save_gap_plot(seed_summary, gap_plot)

        payload = {
            "status": "ok",
            "run_id": run_paths.run_id,
            "source_summary_run_id": source.name,
            "seeds": seeds,
            "seed_status_counts": seed_summary["seed_status"].value_counts().to_dict() if "seed_status" in seed_summary.columns else {},
            "outputs": {
                "seed_summary": str(seed_path),
                "training_by_action": str(action_path),
                "status_comparison": str(status_path),
                "summary_json": str(json_path),
                "summary_md": str(md_path),
                "action_plot": str(action_plot) if action_plot.exists() else None,
                "gap_plot": str(gap_plot) if gap_plot.exists() else None,
            },
        }
        write_json(json_path, payload)

        lines = [
            "# IQN Reward/Action Diagnostic", "",
            f"- Source summary run: `{source.name}`",
            f"- Seeds inspected: `{seeds}`", "",
            "## Seed-level summary", "",
            seed_summary.to_markdown(index=False) if not seed_summary.empty else "- No seed rows.",
            "", "## Training by action", "",
            by_action.to_markdown(index=False) if not by_action.empty else "- No action rows.",
            "", "## Status comparison", "",
            status_comparison.to_markdown(index=False) if not status_comparison.empty else "- No status rows.",
        ]
        md_path.write_text("\n".join(lines), encoding="utf-8")

        logger.info("IQN reward/action diagnostic completed.")
        system_logger.info("StockInvestmentDSS IQN reward/action diagnostic completed successfully.")
        return 0
    except Exception:
        system_logger.exception("StockInvestmentDSS IQN reward/action diagnostic failed.")
        if run_paths is not None:
            try:
                setup_run_logger(run_paths, log_level=log_level).exception("Run failed.")
            except Exception:
                pass
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
