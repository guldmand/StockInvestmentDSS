"""
FinRL Stock Trading 2026 - Part 1: Data master + point-in-time split (_explained v2)

WINDOWS SSL FIX: This version monkey-patches yfinance to use curl_cffi browser session
so the script works on Windows. On Linux/Colab the patch has no effect.

Purpose
-------
A safer data script for thesis/PoC experiments.

Instead of treating train_data.csv and trade_data.csv as the source of truth,
this script creates one full processed dataset and then derives train/trade
splits from a configurable point-in-time split.

Typical usage
-------------
Full S&P500 experiment:
  python src/stock_investment_dss/data/FinRL_data_master.py \\
    --universe sp500 \\
    --dataset-tag 500 \\
    --start-date 2010-01-01 \\
    --train-end-date 2025-12-31 \\
    --trade-start-date 2026-01-01 \\
    --trade-end-date 2026-05-26 \\
    --chunk-size 25 \\
    --sleep-seconds 2 \\
    --use-vix false \\
    --use-turbulence false
"""
from __future__ import annotations

# ============================================================
# CRITICAL: Monkey patch yfinance for Windows SSL workaround
# This MUST run BEFORE any "from finrl ..." import
# ============================================================
import os as _os
import sys as _sys

def _patch_yfinance_for_ssl():
    """Make yf.download use curl_cffi session on Windows."""
    try:
        import yfinance as _yf
        from curl_cffi import requests as _curl_requests
    except ImportError:
        print("[WARNING] curl_cffi not installed; skipping SSL patch")
        return
    
    if hasattr(_yf.download, '_curl_cffi_patched'):
        return
    
    impersonate = _os.environ.get(
        "STOCK_INVESTMENT_DSS_YFINANCE_IMPERSONATE",
        "firefox135",
    )
    
    _ssl_session = _curl_requests.Session(impersonate=impersonate)
    _original_download = _yf.download
    
    def _patched_download(*args, **kwargs):
        if 'session' not in kwargs:
            kwargs['session'] = _ssl_session
        return _original_download(*args, **kwargs)
    
    _patched_download._curl_cffi_patched = True
    _yf.download = _patched_download
    
    # CRITICAL: Also patch the yf module reference inside FinRL's yahoodownloader
    # FinRL imports `import yfinance as yf` at module level — we need to update
    # that reference so FinRL's calls go through our patch
    print(f"[INIT] yfinance.download monkey-patched with curl_cffi session (impersonate={impersonate})", flush=True)

_patch_yfinance_for_ssl()

# ALSO patch FinRL's yahoodownloader after it's imported (defense in depth)
def _patch_finrl_yahoodownloader():
    """Patch FinRL's yahoodownloader module to use our patched yf.download."""
    try:
        import yfinance as _yf
        from finrl.meta.preprocessor import yahoodownloader as _ydl_module
        # Replace the yf reference inside FinRL with our patched version
        _ydl_module.yf = _yf
        print(f"[INIT] FinRL yahoodownloader.yf reference updated to patched yfinance", flush=True)
    except ImportError as exc:
        print(f"[WARNING] Could not patch FinRL yahoodownloader: {exc}")

# ============================================================
# Regular imports (after monkey patch)
# ============================================================
import argparse
import itertools
import json
import time
from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from finrl import config_tickers
from finrl.config import INDICATORS
from finrl.meta.preprocessor.preprocessors import FeatureEngineer
from finrl.meta.preprocessor.preprocessors import data_split
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader

# Apply FinRL patch AFTER imports (so FinRL is loaded first)
_patch_finrl_yahoodownloader()


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value = value.lower().strip()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download/process FinRL stock data and create point-in-time train/trade splits.")
    parser.add_argument("--mode", choices=["download-and-split", "split-only"], default="download-and-split")
    parser.add_argument("--universe", choices=["dow30", "sp500", "custom"], default="dow30")
    parser.add_argument("--custom-tickers", type=str, default="", help="Comma-separated tickers when --universe custom is used.")
    parser.add_argument("--max-tickers", type=int, default=None, help="Limit ticker count for quick experiments, e.g. 50.")
    parser.add_argument("--dataset-tag", type=str, default="auto", help="Output postfix, e.g. 50 or 500. Use auto to infer.")
    parser.add_argument("--start-date", type=str, default="2016-01-01")
    parser.add_argument("--train-end-date", type=str, default="2025-12-31")
    parser.add_argument("--trade-start-date", type=str, default="2026-01-01")
    parser.add_argument("--trade-end-date", type=str, default="2026-03-20")
    parser.add_argument("--full-data-path", type=str, default="", help="Existing master CSV for --mode split-only.")
    parser.add_argument("--output-root", type=str, default="outputs")
    parser.add_argument("--chunk-size", type=int, default=25)
    parser.add_argument("--sleep-seconds", type=float, default=2.0)
    parser.add_argument("--use-technical-indicators", type=str2bool, default=True)
    parser.add_argument("--use-vix", type=str2bool, default=True)
    parser.add_argument("--use-turbulence", type=str2bool, default=False)
    parser.add_argument("--write-legacy-names", type=str2bool, default=False, help="Also write train_data.csv and trade_data.csv. Default false.")
    parser.add_argument("--save-raw", type=str2bool, default=True)
    return parser.parse_args()


def normalize_yahoo_ticker(ticker: str) -> str:
    """Yahoo Finance uses BRK-B instead of BRK.B."""
    return ticker.strip().upper().replace(".", "-")


def get_sp500_tickers() -> list[str]:
    """Fetch current S&P 500 constituents from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
        )
    }

    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        tables = pd.read_html(StringIO(response.text))
    except Exception as exc:
        raise RuntimeError(
            "Could not fetch S&P 500 ticker list from Wikipedia. "
            f"Original error: {exc}"
        ) from exc

    if not tables:
        raise RuntimeError("Wikipedia returned no HTML tables for the S&P 500 page.")

    table = tables[0]
    symbol_column = "Symbol" if "Symbol" in table.columns else table.columns[0]
    tickers = table[symbol_column].astype(str).tolist()
    return [normalize_yahoo_ticker(t) for t in tickers]


def get_tickers(args: argparse.Namespace) -> list[str]:
    if args.universe == "dow30":
        tickers = [normalize_yahoo_ticker(t) for t in config_tickers.DOW_30_TICKER]
    elif args.universe == "sp500":
        tickers = get_sp500_tickers()
    else:
        if not args.custom_tickers.strip():
            raise ValueError("--custom-tickers must be provided when --universe custom is used.")
        tickers = [normalize_yahoo_ticker(t) for t in args.custom_tickers.split(",") if t.strip()]

    tickers = list(dict.fromkeys(tickers))

    if args.max_tickers is not None:
        tickers = tickers[: args.max_tickers]

    return tickers


def infer_dataset_tag(args: argparse.Namespace, tickers: list[str] | None = None) -> str:
    if args.dataset_tag != "auto":
        return args.dataset_tag.strip().lstrip("_")
    if args.max_tickers is not None:
        return str(args.max_tickers)
    if args.universe == "sp500":
        return "500"
    if args.universe == "dow30":
        return "30"
    if tickers is not None:
        return str(len(tickers))
    return "run"


def save_dataframe(df: pd.DataFrame, output_dir: Path, filename: str, tag: str, index: bool = False) -> Path:
    path = output_dir / f"{filename}_{tag}.csv"
    df.to_csv(path, index=index)
    print(f"Saved: {path}", flush=True)
    return path


def save_json(data: dict[str, Any], output_dir: Path, filename: str, tag: str) -> Path:
    path = output_dir / f"{filename}_{tag}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"Saved: {path}", flush=True)
    return path


def download_in_chunks(tickers: list[str], start_date: str, end_date: str, chunk_size: int, sleep_seconds: float) -> tuple[pd.DataFrame, list[str], list[str]]:
    frames: list[pd.DataFrame] = []
    downloaded: list[str] = []
    failed: list[str] = []

    for start in range(0, len(tickers), chunk_size):
        chunk = tickers[start : start + chunk_size]
        print("\n" + "-" * 80, flush=True)
        print(f"Downloading chunk {start // chunk_size + 1}: {len(chunk)} tickers", flush=True)
        print(chunk, flush=True)
        print("-" * 80, flush=True)

        try:
            df_chunk = YahooDownloader(
                start_date=start_date,
                end_date=end_date,
                ticker_list=chunk,
            ).fetch_data()
            if df_chunk is not None and len(df_chunk) > 0:
                frames.append(df_chunk)
                downloaded.extend(sorted(df_chunk["tic"].astype(str).unique().tolist()))
                missing = sorted(set(chunk) - set(df_chunk["tic"].astype(str).unique().tolist()))
                failed.extend(missing)
            else:
                failed.extend(chunk)
        except Exception as exc:
            print(f"Chunk failed: {exc}", flush=True)
            failed.extend(chunk)

        if sleep_seconds > 0 and start + chunk_size < len(tickers):
            print(f"Sleeping {sleep_seconds} seconds to reduce rate-limit pressure...", flush=True)
            time.sleep(sleep_seconds)

    if not frames:
        raise RuntimeError("No market data was downloaded. Try smaller chunks, fewer tickers, or run in Colab later.")

    raw = pd.concat(frames, ignore_index=True)
    raw = raw.drop_duplicates(subset=["date", "tic"]).sort_values(["date", "tic"]).reset_index(drop=True)
    downloaded = sorted(set(downloaded))
    failed = sorted(set(failed) - set(downloaded))
    return raw, downloaded, failed


def make_full_grid(processed: pd.DataFrame) -> pd.DataFrame:
    list_ticker = processed["tic"].unique().tolist()
    list_date = list(pd.date_range(processed["date"].min(), processed["date"].max()).astype(str))
    combination = list(itertools.product(list_date, list_ticker))

    processed_full = pd.DataFrame(combination, columns=["date", "tic"]).merge(
        processed, on=["date", "tic"], how="left"
    )
    processed_full = processed_full[processed_full["date"].isin(processed["date"])]
    processed_full = processed_full.sort_values(["date", "tic"])
    processed_full = processed_full.fillna(0)
    return processed_full


def create_splits(full_data: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = data_split(full_data, args.start_date, args.train_end_date)
    trade = data_split(full_data, args.trade_start_date, args.trade_end_date)
    return train, trade


def make_summary(raw: pd.DataFrame | None, full_data: pd.DataFrame, train: pd.DataFrame, trade: pd.DataFrame, args: argparse.Namespace, tag: str) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "dataset_tag": tag,
            "mode": args.mode,
            "universe": args.universe,
            "max_tickers": args.max_tickers,
            "start_date": args.start_date,
            "train_end_date": args.train_end_date,
            "trade_start_date": args.trade_start_date,
            "trade_end_date": args.trade_end_date,
            "raw_rows": len(raw) if raw is not None else None,
            "full_rows": len(full_data),
            "train_rows": len(train),
            "trade_rows": len(trade),
            "full_tickers": full_data["tic"].nunique(),
            "train_tickers": train["tic"].nunique(),
            "trade_tickers": trade["tic"].nunique(),
            "full_start_date": str(full_data["date"].min()),
            "full_end_date": str(full_data["date"].max()),
            "train_start_date_actual": str(train["date"].min()) if len(train) else None,
            "train_end_date_actual": str(train["date"].max()) if len(train) else None,
            "trade_start_date_actual": str(trade["date"].min()) if len(trade) else None,
            "trade_end_date_actual": str(trade["date"].max()) if len(trade) else None,
            "use_technical_indicators": args.use_technical_indicators,
            "use_vix": args.use_vix,
            "use_turbulence": args.use_turbulence,
            "missing_values_full": int(full_data.isna().sum().sum()),
        }
    ])


def write_readme(output_dir: Path, tag: str) -> None:
    text = f"""# Data outputs for dataset `{tag}`

This folder documents how the market data was created and split.

## Core idea

The master file is the source of truth:

```text
market_data_full_{tag}.csv
```

The train/trade files are derived from the master file using the configured point-in-time split:

```text
train_data_{tag}.csv
trade_data_{tag}.csv
```
"""
    path = output_dir / f"README_data_outputs_{tag}.md"
    path.write_text(text, encoding="utf-8")
    print(f"Saved: {path}", flush=True)


def main() -> None:
    args = parse_args()

    tickers: list[str] | None = None
    raw: pd.DataFrame | None = None

    if args.mode == "download-and-split":
        tickers = get_tickers(args)
    tag = infer_dataset_tag(args, tickers)

    output_dir = Path(args.output_root) / f"data_{tag}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80, flush=True)
    print("FinRL Stock Trading 2026 - Part 1 Data master + split (_explained)", flush=True)
    print("=" * 80, flush=True)
    print(f"Mode: {args.mode}", flush=True)
    print(f"Dataset tag: {tag}", flush=True)

    if args.mode == "download-and-split":
        assert tickers is not None
        print(f"Requested ticker count: {len(tickers)}", flush=True)
        save_dataframe(pd.DataFrame({"tic": tickers}), output_dir, "tickers_requested", tag)

        raw, downloaded, failed = download_in_chunks(
            tickers=tickers,
            start_date=args.start_date,
            end_date=args.trade_end_date,
            chunk_size=args.chunk_size,
            sleep_seconds=args.sleep_seconds,
        )
        if args.save_raw:
            raw_path = Path(f"raw_prices_{tag}.csv")
            raw.to_csv(raw_path, index=False)
            print(f"Saved: {raw_path}", flush=True)
            save_dataframe(raw.head(50), output_dir, "raw_prices_head", tag)

        save_dataframe(pd.DataFrame({"tic": downloaded}), output_dir, "tickers_downloaded", tag)
        save_dataframe(pd.DataFrame({"tic": failed}), output_dir, "tickers_failed", tag)

        print("\nFeature engineering", flush=True)
        fe = FeatureEngineer(
            use_technical_indicator=args.use_technical_indicators,
            tech_indicator_list=INDICATORS,
            use_vix=args.use_vix,
            use_turbulence=args.use_turbulence,
            user_defined_feature=False,
        )
        processed = fe.preprocess_data(raw)
        full_data = make_full_grid(processed)
    else:
        if not args.full_data_path:
            raise ValueError("--full-data-path is required when --mode split-only")
        full_path = Path(args.full_data_path)
        if not full_path.exists():
            raise FileNotFoundError(f"Could not find {full_path}")
        print(f"Loading existing master file: {full_path}", flush=True)
        full_data = pd.read_csv(full_path)
        if full_data.columns[0].lower().startswith("unnamed"):
            full_data = full_data.drop(columns=[full_data.columns[0]])

    full_data = full_data.sort_values(["date", "tic"]).reset_index(drop=True)

    full_path = Path(f"market_data_full_{tag}.csv")
    full_data.to_csv(full_path, index=False)
    print(f"Saved: {full_path}", flush=True)
    save_dataframe(full_data.head(50), output_dir, "market_data_full_head", tag)
    save_dataframe(full_data.tail(50), output_dir, "market_data_full_tail", tag)

    train, trade = create_splits(full_data, args)

    train_path = Path(f"train_data_{tag}.csv")
    trade_path = Path(f"trade_data_{tag}.csv")
    train.to_csv(train_path)
    trade.to_csv(trade_path)
    print(f"Saved: {train_path}", flush=True)
    print(f"Saved: {trade_path}", flush=True)

    if args.write_legacy_names:
        train.to_csv("train_data.csv")
        trade.to_csv("trade_data.csv")
        print("Saved legacy compatibility files: train_data.csv, trade_data.csv", flush=True)
    else:
        print("Did NOT overwrite legacy files train_data.csv / trade_data.csv. Use --write-legacy-names true if needed.", flush=True)

    save_dataframe(make_summary(raw, full_data, train, trade, args, tag), output_dir, "data_summary", tag)
    save_dataframe(pd.DataFrame({"feature": full_data.columns.tolist()}), output_dir, "feature_columns", tag)
    save_dataframe(pd.DataFrame({"indicator": INDICATORS}), output_dir, "indicators", tag)

    split_config = {
        "dataset_tag": tag,
        "master_file": str(full_path),
        "train_file": str(train_path),
        "trade_file": str(trade_path),
        "start_date": args.start_date,
        "train_end_date": args.train_end_date,
        "trade_start_date": args.trade_start_date,
        "trade_end_date": args.trade_end_date,
        "note": "train/trade files are derived from the master file using this point-in-time split",
    }
    save_json(split_config, output_dir, "split_config", tag)
    save_json(vars(args), output_dir, "run_config", tag)
    write_readme(output_dir, tag)

    print("\n" + "=" * 80, flush=True)
    print("Data workflow finished", flush=True)
    print("=" * 80, flush=True)
    print(f"Master: {full_path}", flush=True)
    print(f"Train : {train_path}", flush=True)
    print(f"Trade : {trade_path}", flush=True)
    print(f"Explained outputs: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
