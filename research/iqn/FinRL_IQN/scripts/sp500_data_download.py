"""
S&P 500 Data Download (16 year historical OHLCV)

Downloads daily historical price data for the S&P 500 universe
(~500 tickers) covering 2010-01-01 to 2026-05-26.

Uses FMP API (premium) with yfinance as fallback.
Saves processed data to:
  data/market/daily/processed/market_data_sp500_long_2010_2026_*_1d_finrl.csv

Configuration:
  - Universe: sp500 (defined in ticker_universes.py)
  - Period: 2010-01-01 to 2026-05-26 (~16 years)
  - REQUIRE_ALL_TICKERS=false → partial data accepted (handles delisting/IPOs)
  - Technical indicators + VIX enabled
  - Cache used to avoid re-downloads

Usage:
  - From repo root:  python scripts/sp500_data_download.py
  - From AMD:        python scripts/sp500_data_download.py

Estimated runtime: 15-90 minutes depending on data source
  (FMP premium: faster, yfinance: slower)
"""

from __future__ import annotations
import os
import sys
import time
import subprocess
from datetime import datetime
from pathlib import Path


def find_repo_root() -> Path:
    """Locate the repository root by looking for src/stock_investment_dss/."""
    current = Path.cwd().resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "src" / "stock_investment_dss").exists():
            return candidate
    raise RuntimeError(
        f"Could not find repo root from cwd={current}. "
        "Run this script from inside the FinRL_IQN repository."
    )


def main() -> int:
    repo_root = find_repo_root()

    # -------------------------------------------------------------------
    # Environment variables for S&P 500 dataset (DATA DOWNLOAD ONLY)
    # Downloads historical OHLCV data via FMP/yfinance for ~500 tickers
    # Period: 2010-01-01 to 2026-05-26 (16 years)
    # -------------------------------------------------------------------
    download_env = {
        # PYTHONPATH for `python -m stock_investment_dss...`
        "PYTHONPATH": "src",
        # Dataset specification (sp500, 500-ticker universe, full history)
        "STOCK_INVESTMENT_DSS_DAILY_DATA_UNIVERSE": "sp500",
        "STOCK_INVESTMENT_DSS_DAILY_DATASET_ID": "sp500_long_2010_2026",
        "STOCK_INVESTMENT_DSS_DAILY_DATA_START": "2010-01-01",
        "STOCK_INVESTMENT_DSS_DAILY_DATA_END": "2026-05-26",
        "STOCK_INVESTMENT_DSS_DAILY_DATA_USE_CACHE": "true",
        "STOCK_INVESTMENT_DSS_DAILY_DATA_ALLOW_DOWNLOAD": "true",
        "STOCK_INVESTMENT_DSS_FORCE_FINRL_DOWNLOAD": "false",
        "STOCK_INVESTMENT_DSS_REQUIRE_ALL_TICKERS": "false",
        # Technical indicators + VIX
        "STOCK_INVESTMENT_DSS_DAILY_DATA_USE_TECHNICAL_INDICATORS": "true",
        "STOCK_INVESTMENT_DSS_DAILY_DATA_USE_VIX": "true",
        # Download settings (chunked + sleep for rate-limit friendliness)
        "STOCK_INVESTMENT_DSS_DAILY_DATA_CHUNK_SIZE": "25",
        "STOCK_INVESTMENT_DSS_DAILY_DATA_SLEEP_SECONDS": "2.0",
        "STOCK_INVESTMENT_DSS_YFINANCE_IMPERSONATE": "firefox135",
        "STOCK_INVESTMENT_DSS_YFINANCE_TIMEOUT_SECONDS": "30",
        # No explicit ticker list — uses sp500 from ticker_universes.py
    }

    # Apply env vars to current process environment
    for key, value in download_env.items():
        os.environ[key] = value

    # CRITICAL: Override .env's import_file to force fresh FMP/yfinance download
    os.environ["STOCK_INVESTMENT_DSS_DAILY_DATA_IMPORT_FILE"] = ""

    # -------------------------------------------------------------------
    # Print configuration
    # -------------------------------------------------------------------
    print()
    print("=" * 70)
    print("S&P 500 Data Download - 16 year historical OHLCV")
    print("=" * 70)
    print()
    print(f"Repo root:    {repo_root}")
    print(f"Python:       {sys.executable}")
    print(f"Start time:   {datetime.now():%Y-%m-%d %H:%M:%S}")
    print()
    print("Configuration:")
    print(
        f"  Universe:        {download_env['STOCK_INVESTMENT_DSS_DAILY_DATA_UNIVERSE']}"
    )
    print(f"  Dataset ID:      {download_env['STOCK_INVESTMENT_DSS_DAILY_DATASET_ID']}")
    print(f"  Start date:      {download_env['STOCK_INVESTMENT_DSS_DAILY_DATA_START']}")
    print(f"  End date:        {download_env['STOCK_INVESTMENT_DSS_DAILY_DATA_END']}")
    print(
        f"  Allow download:  {download_env['STOCK_INVESTMENT_DSS_DAILY_DATA_ALLOW_DOWNLOAD']}"
    )
    print(
        f"  Use cache:       {download_env['STOCK_INVESTMENT_DSS_DAILY_DATA_USE_CACHE']}"
    )
    print(
        f"  Require all:     {download_env['STOCK_INVESTMENT_DSS_REQUIRE_ALL_TICKERS']}"
    )
    print(
        f"  Tech indicators: {download_env['STOCK_INVESTMENT_DSS_DAILY_DATA_USE_TECHNICAL_INDICATORS']}"
    )
    print(
        f"  VIX:             {download_env['STOCK_INVESTMENT_DSS_DAILY_DATA_USE_VIX']}"
    )
    print(
        f"  Chunk size:      {download_env['STOCK_INVESTMENT_DSS_DAILY_DATA_CHUNK_SIZE']}"
    )
    print(
        f"  Sleep seconds:   {download_env['STOCK_INVESTMENT_DSS_DAILY_DATA_SLEEP_SECONDS']}"
    )
    print()
    print("Estimated runtime: 15-90 minutes (depending on FMP/yfinance speed)")
    print("Cancel-safe: cached data is preserved on Ctrl+C")
    print("-" * 70)
    print()

    # -------------------------------------------------------------------
    # Run data pipeline (downloads + processes)
    # -------------------------------------------------------------------
    start_time = time.time()
    result = subprocess.run(
        [
            sys.executable,
            "-u",
            "-m",
            "stock_investment_dss.runner.run_data_pipeline_test",
        ],
        cwd=str(repo_root),
        env=os.environ.copy(),
    )
    duration_seconds = time.time() - start_time
    duration_minutes = duration_seconds / 60.0

    print()
    print("=" * 70)
    print(
        f"S&P 500 data download finished - Duration: {duration_minutes:.1f} min "
        f"({duration_seconds:.0f} sec)"
    )
    print(f"Return code: {result.returncode}")
    print("=" * 70)
    print()

    # -------------------------------------------------------------------
    # Inspect output — show the latest data pipeline run
    # -------------------------------------------------------------------
    runs_dir = repo_root / "outputs" / "runs"

    # Find latest data pipeline run
    pipeline_runs = [
        d for d in runs_dir.iterdir() if d.is_dir() and "data_pipeline_test" in d.name
    ]
    latest_pipeline = (
        max(pipeline_runs, key=lambda p: p.stat().st_mtime) if pipeline_runs else None
    )

    if latest_pipeline:
        print(f"Latest data pipeline run: {latest_pipeline.name}")
        print()

        # Show summary JSON if exists
        summary_json = latest_pipeline / "summary" / "data_pipeline_test_summary.json"
        if summary_json.exists():
            print("--- Pipeline summary (first 50 lines) ---")
            for line in summary_json.read_text(encoding="utf-8").splitlines()[:50]:
                print(line)
            print()

        # Show data files generated
        data_files = (
            list((latest_pipeline / "data").glob("*.csv"))
            if (latest_pipeline / "data").exists()
            else []
        )
        if data_files:
            print("--- Generated data files ---")
            for f in sorted(data_files):
                size_mb = f.stat().st_size / (1024 * 1024)
                print(f"  {f.name} ({size_mb:.1f} MB)")
            print()

        # Show metadata files
        meta_files = (
            list((latest_pipeline / "data").glob("*metadata*.json"))
            if (latest_pipeline / "data").exists()
            else []
        )
        if meta_files:
            print("--- Metadata files ---")
            for f in sorted(meta_files):
                print(f"  {f.name}")
            print()

        # Show directory structure
        print("--- Run directory structure ---")
        for subdir in sorted(latest_pipeline.iterdir()):
            if subdir.is_dir():
                count = sum(1 for _ in subdir.rglob("*") if _.is_file())
                if count > 0:
                    print(f"  {subdir.name}/: {count} files")
    else:
        print("WARNING: No data pipeline run found in outputs/runs/")
        print("The download may have failed before producing output.")

    # Also check the canonical processed data file
    processed_dir = repo_root / "data" / "market" / "daily" / "processed"
    sp500_files = list(processed_dir.glob("market_data_sp500*1d_finrl.csv"))
    if sp500_files:
        print()
        print("--- Canonical processed file(s) ---")
        for f in sorted(sp500_files, key=lambda p: p.stat().st_mtime, reverse=True):
            size_mb = f.stat().st_size / (1024 * 1024)
            mtime = datetime.fromtimestamp(f.stat().st_mtime)
            print(f"  {f.name} ({size_mb:.1f} MB, modified {mtime:%Y-%m-%d %H:%M:%S})")

    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
