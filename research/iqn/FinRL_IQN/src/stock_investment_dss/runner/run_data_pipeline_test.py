# src/stock_investment_dss/runner/run_data_pipeline_test.py

from __future__ import annotations

import json

from stock_investment_dss.data.finrl_data_pipeline import (
    load_or_create_finrl_daily_dataset,
)
from stock_investment_dss.utilities.config import (
    get_boolean_environment_variable,
    get_environment_variable,
)
from stock_investment_dss.utilities.logging import (
    setup_run_logger,
    setup_system_logger,
)
from stock_investment_dss.utilities.paths import PROJECT_ROOT, create_run_paths
from stock_investment_dss.utilities.seed import set_global_seed


def write_json(path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, default=str)


def get_int_environment_variable(name: str, default: int) -> int:
    value = get_environment_variable(name, default=str(default))
    return int(value or default)


def get_float_environment_variable(name: str, default: float) -> float:
    value = get_environment_variable(name, default=str(default))
    return float(value or default)


def main() -> int:
    log_level = (
        get_environment_variable(
            "STOCK_INVESTMENT_DSS_LOG_LEVEL",
            default="INFO",
        )
        or "INFO"
    )

    system_logger = setup_system_logger(log_level=log_level)
    system_logger.info("Starting StockInvestmentDSS data pipeline test.")
    system_logger.info("Project root: %s", PROJECT_ROOT)

    run_paths = None

    try:
        set_global_seed(42)

        universe_id = (
            get_environment_variable(
                "STOCK_INVESTMENT_DSS_UNIVERSE_ID",
                default=None,
            )
            or get_environment_variable(
                "STOCK_INVESTMENT_DSS_DAILY_DATA_UNIVERSE",
                default="demo_2",
            )
            or "demo_2"
        )

        dataset_id = (
            get_environment_variable(
                "STOCK_INVESTMENT_DSS_DAILY_DATASET_ID",
                default=universe_id,
            )
            or universe_id
        )

        start_date = (
            get_environment_variable(
                "STOCK_INVESTMENT_DSS_DAILY_DATA_START",
                default="2024-01-01",
            )
            or "2024-01-01"
        )

        end_date = (
            get_environment_variable(
                "STOCK_INVESTMENT_DSS_DAILY_DATA_END",
                default="2024-02-01",
            )
            or "2024-02-01"
        )

        use_cache = get_boolean_environment_variable(
            "STOCK_INVESTMENT_DSS_DAILY_DATA_USE_CACHE",
            default=True,
        )

        allow_download = get_boolean_environment_variable(
            "STOCK_INVESTMENT_DSS_DAILY_DATA_ALLOW_DOWNLOAD",
            default=True,
        )

        force_download = get_boolean_environment_variable(
            "STOCK_INVESTMENT_DSS_FORCE_FINRL_DOWNLOAD",
            default=False,
        )

        import_file = get_environment_variable(
            "STOCK_INVESTMENT_DSS_DAILY_DATA_IMPORT_FILE",
            default=None,
        )

        chunk_size = get_int_environment_variable(
            "STOCK_INVESTMENT_DSS_DAILY_DATA_CHUNK_SIZE",
            default=25,
        )

        sleep_seconds = get_float_environment_variable(
            "STOCK_INVESTMENT_DSS_DAILY_DATA_SLEEP_SECONDS",
            default=2.0,
        )

        use_technical_indicators = get_boolean_environment_variable(
            "STOCK_INVESTMENT_DSS_DAILY_DATA_USE_TECHNICAL_INDICATORS",
            default=True,
        )

        use_vix = get_boolean_environment_variable(
            "STOCK_INVESTMENT_DSS_DAILY_DATA_USE_VIX",
            default=False,
        )

        use_turbulence = get_boolean_environment_variable(
            "STOCK_INVESTMENT_DSS_DAILY_DATA_USE_TURBULENCE",
            default=False,
        )

        yfinance_impersonate = (
            get_environment_variable(
                "STOCK_INVESTMENT_DSS_YFINANCE_IMPERSONATE",
                default="firefox135",
            )
            or "firefox135"
        )

        yfinance_timeout_seconds = get_int_environment_variable(
            "STOCK_INVESTMENT_DSS_YFINANCE_TIMEOUT_SECONDS",
            default=30,
        )

        require_all_tickers = get_boolean_environment_variable(
            "STOCK_INVESTMENT_DSS_REQUIRE_ALL_TICKERS",
            default=True,
        )

        run_paths = create_run_paths("d_iqn_dss_data_pipeline_test")
        run_logger = setup_run_logger(run_paths, log_level=log_level)

        run_logger.info("Created run directory: %s", run_paths.run_directory)
        run_logger.info("Run id: %s", run_paths.run_id)
        run_logger.info("Universe id: %s", universe_id)
        run_logger.info("Dataset id: %s", dataset_id)
        run_logger.info("Date range: %s -> %s", start_date, end_date)
        run_logger.info("Use cache: %s", use_cache)
        run_logger.info("Allow download: %s", allow_download)
        run_logger.info("Force download: %s", force_download)
        run_logger.info("Chunk size: %s", chunk_size)
        run_logger.info("Sleep seconds: %s", sleep_seconds)
        run_logger.info("Use technical indicators: %s", use_technical_indicators)
        run_logger.info("Use VIX: %s", use_vix)
        run_logger.info("Use turbulence: %s", use_turbulence)
        run_logger.info("Import file: %s", import_file)
        run_logger.info("yfinance impersonate: %s", yfinance_impersonate)
        run_logger.info("yfinance timeout seconds: %s", yfinance_timeout_seconds)
        run_logger.info("Require all tickers: %s", require_all_tickers)

        result = load_or_create_finrl_daily_dataset(
            universe_id=universe_id,
            dataset_id=dataset_id,
            start_date=start_date,
            end_date=end_date,
            use_cache=use_cache,
            allow_download=allow_download,
            force_download=force_download,
            import_file=import_file,
            chunk_size=chunk_size,
            sleep_seconds=sleep_seconds,
            use_technical_indicators=use_technical_indicators,
            use_vix=use_vix,
            use_turbulence=use_turbulence,
            yfinance_impersonate=yfinance_impersonate,
            yfinance_timeout_seconds=yfinance_timeout_seconds,
            require_all_tickers=require_all_tickers,
        )

        run_data_snapshot_path = (
            run_paths.data_directory / f"market_data_{dataset_id}_1d_finrl.csv"
        )

        run_metadata_snapshot_path = (
            run_paths.data_directory / f"market_data_{dataset_id}_1d_metadata.json"
        )

        result.processed_data.to_csv(run_data_snapshot_path, index=False)
        write_json(run_metadata_snapshot_path, result.metadata)

        summary = {
            "status": "ok",
            "project_name": "StockInvestmentDSS",
            "prototype_name": "D-IQN-DSS",
            "run_id": run_paths.run_id,
            "project_root": str(PROJECT_ROOT),
            "run_directory": str(run_paths.run_directory),
            "data": {
                "dataset_id": result.dataset_id,
                "universe_id": result.universe_id,
                "tickers": list(result.tickers),
                "start_date": result.start_date,
                "end_date": result.end_date,
                "row_count": int(len(result.processed_data)),
                "column_count": int(len(result.processed_data.columns)),
                "columns": list(result.processed_data.columns),
                "actual_data_method": result.metadata.get("actual_data_method"),
                "fallback_used": result.metadata.get("fallback_used"),
                "fallback_reason": result.metadata.get("fallback_reason"),
                "yfinance_fallback_error": result.metadata.get(
                    "yfinance_fallback_error"
                ),
                "yfinance_impersonate": result.metadata.get("yfinance_impersonate"),
                "cache_used": result.metadata.get("cache_used"),
                "download_attempted": result.metadata.get("download_attempted"),
                "downloaded_tickers": result.metadata.get("downloaded_tickers"),
                "failed_tickers": result.metadata.get("failed_tickers"),
                "error_by_ticker": result.metadata.get("error_by_ticker"),
                "download_attempts": result.metadata.get("download_attempts"),
                "download_status": result.metadata.get("download_status"),
                "data_completeness_ok": result.metadata.get("data_completeness_ok"),
                "missing_required_tickers": result.metadata.get("missing_required_tickers"),
                "row_count_by_ticker": result.metadata.get("row_count_by_ticker"),
                "require_all_tickers": result.metadata.get("require_all_tickers"),
                "canonical_processed_file": str(result.paths.processed_file),
                "run_data_snapshot": str(run_data_snapshot_path),
                "run_metadata_snapshot": str(run_metadata_snapshot_path),
            },
            "next_step": "Build point-in-time train/validation/simulation split.",
        }

        summary_path = run_paths.summary_directory / "data_pipeline_test_summary.json"
        write_json(summary_path, summary)

        run_logger.info(
            "Loaded FinRL daily dataset: rows=%s columns=%s method=%s",
            len(result.processed_data),
            len(result.processed_data.columns),
            result.metadata.get("actual_data_method"),
        )
        run_logger.info("Wrote run data snapshot: %s", run_data_snapshot_path)
        run_logger.info("Wrote run metadata snapshot: %s", run_metadata_snapshot_path)
        run_logger.info("Wrote data pipeline summary: %s", summary_path)

        system_logger.info(
            "StockInvestmentDSS data pipeline test completed successfully."
        )

        return 0

    except Exception:
        system_logger.exception("StockInvestmentDSS data pipeline test failed.")

        if run_paths is not None:
            try:
                run_logger = setup_run_logger(run_paths, log_level=log_level)
                run_logger.exception("Run failed.")
            except Exception:
                pass

        return 1


if __name__ == "__main__":
    raise SystemExit(main())
