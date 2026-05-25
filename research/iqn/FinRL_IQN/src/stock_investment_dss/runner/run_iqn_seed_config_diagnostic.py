
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from stock_investment_dss.utilities.config import get_environment_variable
from stock_investment_dss.utilities.logging import setup_run_logger, setup_system_logger
from stock_investment_dss.utilities.paths import PROJECT_ROOT, create_run_paths


RUN_KIND = "d_iqn_dss_iqn_seed_config_diagnostic"


def read_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)


def flatten(obj: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            out.update(flatten(v, key))
    else:
        out[prefix] = obj
    return out


def norm(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    return str(value)


def find_latest_summary_run() -> Path:
    explicit = get_environment_variable("STOCK_INVESTMENT_DSS_IQN_CONFIG_DIAGNOSTIC_SOURCE_SUMMARY_RUN_ID", default="")
    if explicit:
        path = PROJECT_ROOT / "outputs" / "runs" / explicit
        if not path.exists():
            raise FileNotFoundError(path)
        return path.resolve()

    runs = PROJECT_ROOT / "outputs" / "runs"
    candidates = sorted(
        [
            p for p in runs.iterdir()
            if p.is_dir()
            and p.name.endswith("d_iqn_dss_iqn_learning_curve_multiseed_summary")
            and (p / "data" / "iqn_learning_curve_multiseed_run_index.csv").exists()
        ],
        key=lambda p: p.name,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError("No IQN multiseed summary run found.")
    return candidates[0].resolve()


def find_child_run(seed: int, run_index: pd.DataFrame) -> Path | None:
    rows = run_index[run_index["seed"].astype("Int64") == int(seed)]
    if rows.empty:
        return None
    row = rows.iloc[-1].to_dict()
    candidates = [
        row.get("run_directory"),
        row.get("source_run_directory"),
        row.get("source_run_dir"),
    ]
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


def load_default_strategy() -> dict[str, Any]:
    p = PROJECT_ROOT / "src" / "stock_investment_dss" / "strategies" / "predefined" / "balanced.json"
    data = read_json(p)
    if not isinstance(data, dict):
        return {}
    risk = data.get("risk_policy", {})
    constraints = data.get("constraints", {})
    out = {
        "strategy_id": data.get("strategy_id"),
        "strategy_display_name": data.get("display_name"),
        "risk_profile": data.get("risk_profile"),
        "objective": data.get("objective"),
        "fallback_strategy_file": str(p),
    }
    if isinstance(risk, dict):
        for k, v in risk.items():
            out[f"risk_policy.{k}"] = v
            out[k] = v
    if isinstance(constraints, dict):
        for k, v in constraints.items():
            out[f"constraints.{k}"] = v
    return out


def collect_values(child_run: Path) -> tuple[dict[str, Any], list[str]]:
    values: dict[str, Any] = {}
    scanned: list[str] = []

    for p in sorted(child_run.rglob("*.json")):
        if "wandb" in p.parts:
            continue
        data = read_json(p)
        if data is None:
            continue
        scanned.append(str(p))
        flat = flatten(data)

        for key, value in flat.items():
            short = key.split(".")[-1]
            values.setdefault(key, value)
            values.setdefault(short, value)

        # common strategy shortcuts
        for prefix in ["strategy", "strategy_config", "investor_strategy", "selected_strategy"]:
            for field in ["strategy_id", "display_name", "risk_profile", "objective"]:
                value = flat.get(f"{prefix}.{field}")
                if value is not None:
                    name = "strategy_display_name" if field == "display_name" else field
                    values.setdefault(name, value)

        for prefix in ["risk_policy", "strategy.risk_policy", "strategy_config.risk_policy", "investor_strategy.risk_policy"]:
            for field in [
                "lambda_cvar", "lambda_drawdown", "lambda_volatility",
                "lambda_transaction_cost", "lambda_concentration",
                "lambda_strategy_violation", "lambda_epistemic_uncertainty",
                "score_quantile", "downside_metric", "uncertainty_metric",
            ]:
                value = flat.get(f"{prefix}.{field}")
                if value is not None:
                    values.setdefault(field, value)
                    values.setdefault(f"risk_policy.{field}", value)

    return values, scanned


def infer_strategy(row: dict[str, Any]) -> str:
    if row.get("strategy_id"):
        return str(row["strategy_id"])
    if str(row.get("score_quantile", "")).lower() == "q50" and str(row.get("lambda_cvar", "")) in {"0.75", "0.750000"}:
        return "balanced_inferred_q50_lambda_0_75"
    return "unknown"


def main() -> int:
    log_level = get_environment_variable("STOCK_INVESTMENT_DSS_LOG_LEVEL", default="INFO") or "INFO"
    system_logger = setup_system_logger(log_level=log_level)
    system_logger.info("Starting StockInvestmentDSS IQN seed config diagnostic.")

    run_paths = None
    try:
        source = find_latest_summary_run()
        run_index = pd.read_csv(source / "data" / "iqn_learning_curve_multiseed_run_index.csv")
        run_index["seed"] = pd.to_numeric(run_index["seed"], errors="coerce")

        seed_env = get_environment_variable("STOCK_INVESTMENT_DSS_IQN_CONFIG_DIAGNOSTIC_SEEDS", default="")
        if seed_env:
            seeds = sorted({int(s.strip()) for s in seed_env.split(",") if s.strip()})
        else:
            seeds = sorted(int(s) for s in run_index["seed"].dropna().unique().tolist())

        run_paths = create_run_paths(RUN_KIND)
        logger = setup_run_logger(run_paths, log_level=log_level)
        logger.info("Source summary run: %s", source)
        logger.info("Seeds: %s", seeds)

        fallback = load_default_strategy()
        records: list[dict[str, Any]] = []
        scanned_records: list[dict[str, Any]] = []

        key_fields = [
            "strategy_id", "strategy_display_name", "risk_profile", "objective",
            "score_quantile", "downside_metric", "uncertainty_metric",
            "lambda_cvar", "lambda_drawdown", "lambda_volatility",
            "lambda_transaction_cost", "lambda_concentration",
            "lambda_strategy_violation", "lambda_epistemic_uncertainty",
            "risk_lambda", "score_mode", "eval_score_mode", "config_preset",
            "dataset_id", "pit_split_id", "universe_id", "point_in_time", "trade_end_date",
        ]

        for seed in seeds:
            child = find_child_run(seed, run_index)
            rec: dict[str, Any] = {"seed": seed, "source_run_found": child is not None}
            if child is not None:
                rec["source_run_id"] = child.name
                rec["source_run_directory"] = str(child)
                vals, scanned = collect_values(child)
                for k, v in fallback.items():
                    vals.setdefault(k, v)
                for k in key_fields:
                    if k in vals:
                        rec[k] = vals[k]
                rec["strategy_id_inferred"] = infer_strategy(rec)
                rec["scanned_json_file_count"] = len(scanned)
                for sp in scanned:
                    scanned_records.append({"seed": seed, "source_run_id": child.name, "json_file": sp})
            records.append(rec)

        seed_df = pd.DataFrame(records).sort_values("seed")

        equality_rows = []
        for field in [
            "strategy_id", "strategy_id_inferred", "risk_profile", "score_quantile",
            "lambda_cvar", "risk_lambda", "score_mode", "config_preset",
            "dataset_id", "pit_split_id", "universe_id", "point_in_time", "trade_end_date",
        ]:
            if field in seed_df.columns:
                vals = sorted({norm(v) for v in seed_df[field].dropna().tolist()})
                equality_rows.append({
                    "field": field,
                    "equal_across_seeds": len(vals) == 1,
                    "unique_value_count": len(vals),
                    "values": "; ".join(vals),
                })

        equality_df = pd.DataFrame(equality_rows)
        scanned_df = pd.DataFrame(scanned_records)

        if equality_df.empty:
            status = "unknown_no_comparable_fields"
        elif equality_df[equality_df["field"].isin(["strategy_id", "strategy_id_inferred", "risk_profile", "score_quantile", "lambda_cvar"])]["equal_across_seeds"].all():
            status = "same_strategy_risk_config_detected_or_inferred"
        else:
            status = "strategy_risk_config_differs_across_seeds"

        seed_path = run_paths.summary_directory / "iqn_seed_config_diagnostic_by_seed.csv"
        eq_path = run_paths.summary_directory / "iqn_seed_config_diagnostic_equality.csv"
        scanned_path = run_paths.data_directory / "iqn_seed_config_diagnostic_scanned_files.csv"
        json_path = run_paths.summary_directory / "iqn_seed_config_diagnostic_summary.json"
        md_path = run_paths.summary_directory / "iqn_seed_config_diagnostic_summary.md"

        seed_df.to_csv(seed_path, index=False)
        equality_df.to_csv(eq_path, index=False)
        scanned_df.to_csv(scanned_path, index=False)

        summary = {
            "status": "ok",
            "run_id": run_paths.run_id,
            "source_summary_run_id": source.name,
            "seeds": seeds,
            "strategy_config_equal_status": status,
            "outputs": {
                "seed_rows": str(seed_path),
                "equality": str(eq_path),
                "scanned_files": str(scanned_path),
                "summary_json": str(json_path),
                "summary_md": str(md_path),
            },
        }
        write_json(json_path, summary)

        lines = [
            "# IQN Seed Config Diagnostic",
            "",
            f"- Source summary run: {source.name}",
            f"- Seeds: {seeds}",
            f"- Strategy/risk equality status: {status}",
            "",
            "## Seed-level values",
            "",
            seed_df.to_markdown(index=False),
            "",
            "## Equality checks",
            "",
            equality_df.to_markdown(index=False) if not equality_df.empty else "- No equality rows.",
            "",
        ]
        md_path.write_text("\n".join(lines), encoding="utf-8")

        logger.info("IQN seed config diagnostic completed.")
        logger.info("Strategy/risk equality status: %s", status)
        system_logger.info("StockInvestmentDSS IQN seed config diagnostic completed successfully.")
        return 0

    except Exception:
        system_logger.exception("StockInvestmentDSS IQN seed config diagnostic failed.")
        if run_paths is not None:
            try:
                setup_run_logger(run_paths, log_level=log_level).exception("Run failed.")
            except Exception:
                pass
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
