# src/stock_investment_dss/data/finrl_data_pipeline.py

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from stock_investment_dss.data.market_data_paths import (
    DailyMarketDataPaths,
    get_daily_market_data_paths,
)
from stock_investment_dss.data.ticker_universes import get_ticker_universe
from stock_investment_dss.utilities.paths import PROJECT_ROOT

FINRL_REQUIRED_COLUMNS = {"date", "tic", "open", "high", "low", "close", "volume"}
FINRL_CORE_COLUMNS = ["date", "tic", "open", "high", "low", "close", "volume"]


@dataclass(frozen=True)
class FinRLDailyDataResult:
    dataset_id: str
    universe_id: str
    tickers: tuple[str, ...]
    start_date: str
    end_date: str
    raw_data: pd.DataFrame
    processed_data: pd.DataFrame
    paths: DailyMarketDataPaths
    metadata: dict[str, Any]


@dataclass(frozen=True)
class DownloadAttemptResult:
    method: str
    raw_data: pd.DataFrame
    downloaded_tickers: list[str]
    failed_tickers: list[str]
    error_by_ticker: dict[str, str]
    attempts: list[dict[str, Any]]
    status: str


def _patch_yfinance_for_windows_ssl() -> None:
    """
    Monkey-patch yfinance.download to use a curl_cffi browser session.

    This enables FinRL's YahooDownloader (which calls yf.download internally
    WITHOUT a session parameter) to work on Windows where the default
    curl_cffi SSL stack fails.

    Effect:
    - On Linux/Colab: No effect (default session works there)
    - On Windows: yf.download uses curl_cffi session with impersonate=firefox135
                  This makes FinRL chunked bulk download work
    """
    import os

    try:
        import yfinance as yf
        from curl_cffi import requests as curl_requests
    except ImportError:
        return  # Skip if not installed

    if hasattr(yf.download, "_curl_cffi_patched"):
        return  # Already patched

    impersonate = os.environ.get(
        "STOCK_INVESTMENT_DSS_YFINANCE_IMPERSONATE",
        "firefox135",
    )

    _ssl_session = curl_requests.Session(impersonate=impersonate)
    _original_download = yf.download

    def _patched_download(*args, **kwargs):
        if "session" not in kwargs:
            kwargs["session"] = _ssl_session
        return _original_download(*args, **kwargs)

    _patched_download._curl_cffi_patched = True
    yf.download = _patched_download
    print(
        f"[INIT] yfinance.download monkey-patched with curl_cffi session (impersonate={impersonate})"
    )


# Apply patch immediately on import
_patch_yfinance_for_windows_ssl()


def _load_finrl_components():
    """
    Import FinRL lazily so the error message is clearer if FinRL is not installed.
    """
    try:
        from finrl import config
        from finrl.meta.preprocessor.preprocessors import FeatureEngineer
        from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
    except ImportError as exc:
        raise ImportError(
            "FinRL is required for the StockInvestmentDSS data pipeline. "
            "Install FinRL in the current environment before running this step."
        ) from exc

    return config, FeatureEngineer, YahooDownloader


def _load_yfinance_components():
    """
    Import yfinance/curl_cffi lazily.

    This keeps the local import fallback usable even if the optional yfinance
    fallback dependencies are not installed.
    """
    try:
        import yfinance as yf
        from curl_cffi import requests as curl_requests
    except ImportError as exc:
        raise ImportError(
            "yfinance and curl_cffi are required for the browser-session "
            "download fallback. Install yfinance==0.2.66 and curl_cffi==0.15.0."
        ) from exc

    return yf, curl_requests


def _resolve_project_path(path_value: str | Path | None) -> Path | None:
    if path_value is None:
        return None

    path = Path(path_value)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def _validate_finrl_dataframe(data: pd.DataFrame, label: str) -> None:
    missing_columns = FINRL_REQUIRED_COLUMNS - set(data.columns)

    if missing_columns:
        raise ValueError(
            f"{label} is missing required FinRL columns: {sorted(missing_columns)}"
        )

    if data.empty:
        raise ValueError(f"{label} is empty.")


def _normalize_date_and_ticker_columns(data: pd.DataFrame) -> pd.DataFrame:
    frame = data.copy()

    if "tic" not in frame.columns and "ticker" in frame.columns:
        frame = frame.rename(columns={"ticker": "tic"})

    frame["tic"] = frame["tic"].astype(str).str.upper().str.strip()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")

    frame = frame.dropna(subset=["date"])
    frame["date"] = frame["date"].dt.strftime("%Y-%m-%d")

    return frame


def _filter_by_tickers_and_date_range(
    data: pd.DataFrame,
    tickers: tuple[str, ...],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    frame = _normalize_date_and_ticker_columns(data)

    requested_tickers = {ticker.upper().strip() for ticker in tickers}
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    frame_dates = pd.to_datetime(frame["date"])

    frame = frame[
        frame["tic"].isin(requested_tickers)
        & (frame_dates >= start)
        & (frame_dates < end)
    ].copy()

    frame = frame.sort_values(["date", "tic"]).reset_index(drop=True)

    return frame


def _save_dataset_artifacts(
    raw_data: pd.DataFrame,
    processed_data: pd.DataFrame,
    metadata: dict[str, Any],
    paths: DailyMarketDataPaths,
) -> None:
    paths.raw_file.parent.mkdir(parents=True, exist_ok=True)
    paths.processed_file.parent.mkdir(parents=True, exist_ok=True)
    paths.metadata_file.parent.mkdir(parents=True, exist_ok=True)

    raw_data.to_csv(paths.raw_file, index=False)
    processed_data.to_csv(paths.processed_file, index=False)

    with paths.metadata_file.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, default=str)


def _load_cached_dataset(
    paths: DailyMarketDataPaths,
    dataset_id: str,
    universe_id: str,
    tickers: tuple[str, ...],
    start_date: str,
    end_date: str,
) -> FinRLDailyDataResult:
    raw_data = pd.read_csv(paths.raw_file)
    processed_data = pd.read_csv(paths.processed_file)

    with paths.metadata_file.open("r", encoding="utf-8") as file:
        cached_metadata = json.load(file)

    cache_origin_data_method = cached_metadata.get("actual_data_method")
    cache_origin_fallback_used = cached_metadata.get("fallback_used")
    cache_origin_download_attempted = cached_metadata.get("download_attempted")
    cache_origin_created_at = cached_metadata.get("created_at")

    metadata = {
        **cached_metadata,
        "cache_used": True,
        "actual_data_method": "canonical_cache",
        "cache_origin_data_method": cache_origin_data_method,
        "cache_origin_fallback_used": cache_origin_fallback_used,
        "cache_origin_download_attempted": cache_origin_download_attempted,
        "cache_origin_created_at": cache_origin_created_at,
        "loaded_at": datetime.now(timezone.utc).isoformat(),
    }

    return FinRLDailyDataResult(
        dataset_id=dataset_id,
        universe_id=universe_id,
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        raw_data=raw_data,
        processed_data=processed_data,
        paths=paths,
        metadata=metadata,
    )


def _download_finrl_raw_data_in_chunks(
    tickers: tuple[str, ...],
    start_date: str,
    end_date: str,
    chunk_size: int,
    sleep_seconds: float,
) -> DownloadAttemptResult:
    """
    Download raw daily data using FinRL YahooDownloader in chunks.

    This mirrors the working Colab strategy: pass a group of tickers to
    FinRL YahooDownloader, then sleep between chunks. FinRL/yfinance may still
    perform per-ticker work internally, so we also record missing tickers and
    chunk-level failure reasons.
    """
    _, _, YahooDownloader = _load_finrl_components()

    frames: list[pd.DataFrame] = []
    downloaded: set[str] = set()
    failed: set[str] = set()
    error_by_ticker: dict[str, str] = {}
    attempts: list[dict[str, Any]] = []

    ticker_list = [ticker.upper().strip() for ticker in tickers if ticker.strip()]
    normalized_chunk_size = max(1, int(chunk_size))

    for chunk_start in range(0, len(ticker_list), normalized_chunk_size):
        chunk = ticker_list[chunk_start : chunk_start + normalized_chunk_size]
        chunk_index = chunk_start // normalized_chunk_size + 1
        attempt: dict[str, Any] = {
            "method": "FinRL YahooDownloader chunked",
            "chunk_index": chunk_index,
            "tickers": list(chunk),
            "downloaded_tickers": [],
            "failed_tickers": [],
            "row_count": 0,
            "error_by_ticker": {},
            "exception": None,
        }

        try:
            chunk_data = YahooDownloader(
                start_date=start_date,
                end_date=end_date,
                ticker_list=chunk,
            ).fetch_data(auto_adjust=False)

            if chunk_data is None or chunk_data.empty:
                reason = (
                    "FinRL YahooDownloader returned an empty dataframe for this chunk."
                )
                failed.update(chunk)
                for ticker in chunk:
                    error_by_ticker[ticker] = reason
                    attempt["error_by_ticker"][ticker] = reason
                attempt["failed_tickers"] = list(chunk)
            else:
                chunk_data = _normalize_date_and_ticker_columns(chunk_data)
                frames.append(chunk_data)

                chunk_downloaded = set(
                    chunk_data["tic"].astype(str).str.upper().unique().tolist()
                )
                missing = sorted(set(chunk) - chunk_downloaded)

                downloaded.update(chunk_downloaded)
                failed.update(missing)

                attempt["downloaded_tickers"] = sorted(chunk_downloaded)
                attempt["failed_tickers"] = missing
                attempt["row_count"] = int(len(chunk_data))

                for ticker in missing:
                    reason = (
                        "Ticker was requested in the FinRL YahooDownloader chunk, "
                        "but no rows for the ticker were returned."
                    )
                    error_by_ticker[ticker] = reason
                    attempt["error_by_ticker"][ticker] = reason

        except Exception as exc:
            reason = repr(exc)
            failed.update(chunk)
            for ticker in chunk:
                error_by_ticker[ticker] = reason
                attempt["error_by_ticker"][ticker] = reason
            attempt["failed_tickers"] = list(chunk)
            attempt["exception"] = reason

        attempts.append(attempt)

        if sleep_seconds > 0 and chunk_start + normalized_chunk_size < len(ticker_list):
            time.sleep(sleep_seconds)

    if not frames:
        raise RuntimeError(
            "FinRL YahooDownloader did not return any market data. "
            "This may be due to Yahoo/yfinance rate limiting or TLS/session issues. "
            f"errors={error_by_ticker}"
        )

    raw_data = pd.concat(frames, ignore_index=True)
    raw_data = raw_data.drop_duplicates(subset=["date", "tic"])
    raw_data = raw_data.sort_values(["date", "tic"]).reset_index(drop=True)

    downloaded_list = sorted(downloaded)
    failed_list = sorted(failed - downloaded)

    _validate_finrl_dataframe(raw_data, "FinRL chunked raw daily data")

    status = "full_success" if not failed_list else "partial_success"

    return DownloadAttemptResult(
        method="FinRL YahooDownloader chunked",
        raw_data=raw_data,
        downloaded_tickers=downloaded_list,
        failed_tickers=failed_list,
        error_by_ticker={
            ticker: error_by_ticker[ticker]
            for ticker in failed_list
            if ticker in error_by_ticker
        },
        attempts=attempts,
        status=status,
    )


def _flatten_yfinance_columns(data: pd.DataFrame) -> pd.DataFrame:
    """
    yfinance may return MultiIndex columns even for a single ticker.

    For per-ticker downloads, we keep the price-field level:
    Adj Close, Close, High, Low, Open, Volume.
    """
    frame = data.copy()

    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)

    return frame


def _normalise_yfinance_single_ticker_frame(
    data: pd.DataFrame,
    ticker: str,
) -> pd.DataFrame:
    frame = _flatten_yfinance_columns(data)

    if frame.empty:
        return frame

    frame = frame.reset_index()

    rename_map = {
        "Date": "date",
        "Datetime": "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
    }

    frame = frame.rename(columns=rename_map)

    if "date" not in frame.columns:
        raise ValueError(f"yfinance data for {ticker} did not contain a date column.")

    frame["tic"] = ticker.upper().strip()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"])
    frame["day"] = frame["date"].dt.dayofweek
    frame["date"] = frame["date"].dt.strftime("%Y-%m-%d")

    keep_columns = [
        column
        for column in [
            "date",
            "tic",
            "open",
            "high",
            "low",
            "close",
            "adj_close",
            "volume",
            "day",
        ]
        if column in frame.columns
    ]

    frame = frame[keep_columns].copy()
    frame = frame.dropna(subset=["open", "high", "low", "close", "volume"])

    if "volume" in frame.columns:
        frame["volume"] = frame["volume"].astype("int64")

    return frame


def _download_yfinance_browser_session_raw_data_in_chunks(
    tickers: tuple[str, ...],
    start_date: str,
    end_date: str,
    chunk_size: int,
    sleep_seconds: float,
    impersonate: str,
    timeout_seconds: int,
) -> DownloadAttemptResult:
    """
    Download raw daily data with yfinance using an explicit curl_cffi browser session.

    This is the local Windows/conda fallback for cases where FinRL's
    YahooDownloader/yfinance default session fails. The output is normalized
    to FinRL's daily stock schema and is then passed to FinRL FeatureEngineer.
    """
    yf, curl_requests = _load_yfinance_components()

    frames: list[pd.DataFrame] = []
    downloaded: set[str] = set()
    failed: set[str] = set()
    error_by_ticker: dict[str, str] = {}
    attempts: list[dict[str, Any]] = []

    ticker_list = [ticker.upper().strip() for ticker in tickers if ticker.strip()]
    normalized_chunk_size = max(1, int(chunk_size))

    for chunk_start in range(0, len(ticker_list), normalized_chunk_size):
        chunk = ticker_list[chunk_start : chunk_start + normalized_chunk_size]
        chunk_index = chunk_start // normalized_chunk_size + 1
        attempt: dict[str, Any] = {
            "method": "yfinance browser-session fallback",
            "chunk_index": chunk_index,
            "tickers": list(chunk),
            "impersonate": impersonate,
            "timeout_seconds": timeout_seconds,
            "downloaded_tickers": [],
            "failed_tickers": [],
            "row_count": 0,
            "error_by_ticker": {},
        }

        for ticker in chunk:
            try:
                session = curl_requests.Session(impersonate=impersonate)

                ticker_data = yf.download(
                    ticker,
                    start=start_date,
                    end=end_date,
                    interval="1d",
                    auto_adjust=False,
                    progress=False,
                    timeout=timeout_seconds,
                    session=session,
                )

                if ticker_data is None or ticker_data.empty:
                    reason = "yfinance returned an empty dataframe."
                    failed.add(ticker)
                    error_by_ticker[ticker] = reason
                    attempt["error_by_ticker"][ticker] = reason
                    continue

                ticker_frame = _normalise_yfinance_single_ticker_frame(
                    data=ticker_data,
                    ticker=ticker,
                )

                if ticker_frame.empty:
                    reason = "normalized yfinance dataframe was empty."
                    failed.add(ticker)
                    error_by_ticker[ticker] = reason
                    attempt["error_by_ticker"][ticker] = reason
                    continue

                _validate_finrl_dataframe(
                    ticker_frame,
                    f"yfinance browser-session data for {ticker}",
                )

                frames.append(ticker_frame)
                downloaded.add(ticker)
                attempt["row_count"] = int(attempt["row_count"]) + int(
                    len(ticker_frame)
                )

            except Exception as exc:
                failed.add(ticker)
                error_by_ticker[ticker] = repr(exc)
                attempt["error_by_ticker"][ticker] = repr(exc)

        attempt["downloaded_tickers"] = sorted(set(chunk) & downloaded)
        attempt["failed_tickers"] = sorted((set(chunk) & failed) - downloaded)
        attempts.append(attempt)

        if sleep_seconds > 0 and chunk_start + normalized_chunk_size < len(ticker_list):
            time.sleep(sleep_seconds)

    if not frames:
        raise RuntimeError(
            "yfinance browser-session fallback did not return any market data. "
            f"impersonate={impersonate}, errors={error_by_ticker}"
        )

    raw_data = pd.concat(frames, ignore_index=True)
    raw_data = raw_data.drop_duplicates(subset=["date", "tic"])
    raw_data = raw_data.sort_values(["date", "tic"]).reset_index(drop=True)

    _validate_finrl_dataframe(raw_data, "yfinance browser-session fallback raw data")

    downloaded_list = sorted(downloaded)
    failed_list = sorted(failed - downloaded)
    status = "full_success" if not failed_list else "partial_success"

    return DownloadAttemptResult(
        method="yfinance browser-session fallback + FinRL FeatureEngineer",
        raw_data=raw_data,
        downloaded_tickers=downloaded_list,
        failed_tickers=failed_list,
        error_by_ticker={
            ticker: error_by_ticker[ticker]
            for ticker in failed_list
            if ticker in error_by_ticker
        },
        attempts=attempts,
        status=status,
    )


def _process_with_finrl_feature_engineer(
    raw_data: pd.DataFrame,
    use_technical_indicators: bool,
    use_vix: bool,
    use_turbulence: bool,
) -> tuple[pd.DataFrame, list[str]]:
    config, FeatureEngineer, _ = _load_finrl_components()

    technical_indicators = list(config.INDICATORS)

    feature_engineer = FeatureEngineer(
        use_technical_indicator=use_technical_indicators,
        tech_indicator_list=technical_indicators,
        use_vix=use_vix,
        use_turbulence=use_turbulence,
        user_defined_feature=False,
    )

    processed_data = feature_engineer.preprocess_data(raw_data.copy())
    processed_data = _normalize_date_and_ticker_columns(processed_data)
    processed_data = processed_data.sort_values(["date", "tic"]).reset_index(drop=True)

    _validate_finrl_dataframe(processed_data, "FinRL processed daily data")

    return processed_data, technical_indicators


def _load_finrl_generated_import(
    import_file: Path,
    tickers: tuple[str, ...],
    start_date: str,
    end_date: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not import_file.exists():
        raise FileNotFoundError(f"Import file does not exist: {import_file}")

    imported = pd.read_csv(import_file)

    _validate_finrl_dataframe(imported, f"FinRL-generated import file: {import_file}")

    processed_data = _filter_by_tickers_and_date_range(
        data=imported,
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
    )

    if processed_data.empty:
        raise ValueError(
            "Import file was loaded, but no rows matched the requested "
            f"tickers/date range. tickers={tickers}, start={start_date}, end={end_date}"
        )

    raw_columns = [
        column for column in FINRL_CORE_COLUMNS if column in processed_data.columns
    ]
    raw_data = processed_data[raw_columns].copy()

    _validate_finrl_dataframe(raw_data, "Raw data derived from imported FinRL master")
    _validate_finrl_dataframe(
        processed_data, "Processed data derived from imported FinRL master"
    )

    return raw_data, processed_data


def _download_completeness_report(
    *,
    requested_tickers: tuple[str, ...],
    downloaded_tickers: list[str],
    failed_tickers: list[str],
    processed_data: pd.DataFrame,
    require_all_tickers: bool,
) -> dict[str, Any]:
    requested = sorted({ticker.upper().strip() for ticker in requested_tickers})
    downloaded = sorted({ticker.upper().strip() for ticker in downloaded_tickers})
    failed = sorted({ticker.upper().strip() for ticker in failed_tickers})
    actual = (
        sorted(processed_data["tic"].astype(str).str.upper().unique().tolist())
        if "tic" in processed_data.columns
        else []
    )
    missing = sorted(set(requested) - set(actual))
    row_count_by_ticker = (
        processed_data.groupby("tic").size().astype(int).to_dict()
        if "tic" in processed_data.columns
        else {}
    )

    download_status = "full_success"
    if missing or failed:
        download_status = "partial_success"
    if not actual:
        download_status = "failed"

    return {
        "require_all_tickers": bool(require_all_tickers),
        "requested_ticker_count": int(len(requested)),
        "downloaded_ticker_count": int(len(actual)),
        "requested_tickers": requested,
        "downloaded_tickers": downloaded,
        "actual_tickers_in_processed_data": actual,
        "failed_tickers": failed,
        "missing_required_tickers": missing,
        "row_count_by_ticker": {str(k): int(v) for k, v in row_count_by_ticker.items()},
        "download_status": download_status,
        "data_completeness_ok": (
            bool(not missing) if require_all_tickers else bool(actual)
        ),
    }


def _raise_if_incomplete(
    *,
    method: str,
    completeness: dict[str, Any],
    error_by_ticker: dict[str, str] | None = None,
) -> None:
    if completeness.get("data_completeness_ok"):
        return

    raise RuntimeError(
        f"{method} produced an incomplete dataset. "
        f"missing_required_tickers={completeness.get('missing_required_tickers')}, "
        f"failed_tickers={completeness.get('failed_tickers')}, "
        f"error_by_ticker={error_by_ticker or {}}"
    )


def _metadata_base(
    *,
    selected_dataset_id: str,
    universe_id: str,
    tickers: tuple[str, ...],
    start_date: str,
    end_date: str,
    use_technical_indicators: bool,
    use_vix: bool,
    use_turbulence: bool,
    chunk_size: int,
    sleep_seconds: float,
    require_all_tickers: bool,
) -> dict[str, Any]:
    return {
        "dataset_id": selected_dataset_id,
        "universe_id": universe_id,
        "tickers": list(tickers),
        "frequency": "1d",
        "start_date": start_date,
        "end_date": end_date,
        "primary_framework": "FinRL",
        "use_technical_indicators": use_technical_indicators,
        "use_vix": use_vix,
        "use_turbulence": use_turbulence,
        "chunk_size": int(chunk_size),
        "sleep_seconds": float(sleep_seconds),
        "require_all_tickers": bool(require_all_tickers),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def load_or_create_finrl_daily_dataset(
    universe_id: str,
    start_date: str,
    end_date: str,
    dataset_id: str | None = None,
    force_download: bool = False,
    use_cache: bool = True,
    allow_download: bool = True,
    import_file: str | Path | None = None,
    chunk_size: int = 25,
    sleep_seconds: float = 2.0,
    use_technical_indicators: bool = True,
    use_vix: bool = False,
    use_turbulence: bool = False,
    yfinance_impersonate: str = "firefox135",
    yfinance_timeout_seconds: int = 30,
    require_all_tickers: bool = True,
) -> FinRLDailyDataResult:
    """
    FinRL-first daily market data pipeline.

    Priority:
    1. Canonical cache.
    2. Chunked FinRL YahooDownloader + FinRL FeatureEngineer.
    3. yfinance browser-session fallback + FinRL FeatureEngineer.
    4. FinRL-generated local import fallback.
    5. Clear failure if all paths fail.

    By default, the pipeline requires all requested tickers to be present.
    Partial downloads are recorded, but they do not count as a clean dataset
    unless require_all_tickers=False.
    """
    selected_dataset_id = dataset_id or universe_id
    paths = get_daily_market_data_paths(selected_dataset_id)
    tickers = get_ticker_universe(universe_id)

    if (
        use_cache
        and not force_download
        and paths.raw_file.exists()
        and paths.processed_file.exists()
        and paths.metadata_file.exists()
    ):
        cached_result = _load_cached_dataset(
            paths=paths,
            dataset_id=selected_dataset_id,
            universe_id=universe_id,
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
        )

        completeness = _download_completeness_report(
            requested_tickers=tickers,
            downloaded_tickers=list(
                cached_result.processed_data["tic"].astype(str).str.upper().unique()
            ),
            failed_tickers=[],
            processed_data=cached_result.processed_data,
            require_all_tickers=require_all_tickers,
        )
        if require_all_tickers and not completeness["data_completeness_ok"]:
            raise RuntimeError(
                "Cached dataset is incomplete for the requested universe. "
                f"completeness={completeness}"
            )
        cached_result.metadata.update(
            {
                "download_status": completeness["download_status"],
                "data_completeness_ok": completeness["data_completeness_ok"],
                "missing_required_tickers": completeness["missing_required_tickers"],
                "row_count_by_ticker": completeness["row_count_by_ticker"],
                "require_all_tickers": bool(require_all_tickers),
            }
        )
        return cached_result

    download_error: str | None = None
    yfinance_fallback_error: str | None = None
    download_attempts: list[dict[str, Any]] = []

    base_metadata = _metadata_base(
        selected_dataset_id=selected_dataset_id,
        universe_id=universe_id,
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        use_technical_indicators=use_technical_indicators,
        use_vix=use_vix,
        use_turbulence=use_turbulence,
        chunk_size=chunk_size,
        sleep_seconds=sleep_seconds,
        require_all_tickers=require_all_tickers,
    )

    if allow_download:
        try:
            finrl_result = _download_finrl_raw_data_in_chunks(
                tickers=tickers,
                start_date=start_date,
                end_date=end_date,
                chunk_size=chunk_size,
                sleep_seconds=sleep_seconds,
            )
            download_attempts.extend(finrl_result.attempts)

            processed_data, technical_indicators = _process_with_finrl_feature_engineer(
                raw_data=finrl_result.raw_data,
                use_technical_indicators=use_technical_indicators,
                use_vix=use_vix,
                use_turbulence=use_turbulence,
            )

            completeness = _download_completeness_report(
                requested_tickers=tickers,
                downloaded_tickers=finrl_result.downloaded_tickers,
                failed_tickers=finrl_result.failed_tickers,
                processed_data=processed_data,
                require_all_tickers=require_all_tickers,
            )
            _raise_if_incomplete(
                method="FinRL YahooDownloader chunked",
                completeness=completeness,
                error_by_ticker=finrl_result.error_by_ticker,
            )

            metadata = {
                **base_metadata,
                "cache_used": False,
                "download_attempted": True,
                "download_method_attempted": "FinRL YahooDownloader chunked",
                "actual_data_method": "FinRL YahooDownloader chunked",
                "fallback_used": False,
                "feature_engineering": "FinRL FeatureEngineer",
                "technical_indicators": technical_indicators,
                "downloaded_tickers": finrl_result.downloaded_tickers,
                "failed_tickers": finrl_result.failed_tickers,
                "error_by_ticker": finrl_result.error_by_ticker,
                "download_attempts": download_attempts,
                "download_status": completeness["download_status"],
                "data_completeness_ok": completeness["data_completeness_ok"],
                "missing_required_tickers": completeness["missing_required_tickers"],
                "row_count_by_ticker": completeness["row_count_by_ticker"],
                "raw_row_count": int(len(finrl_result.raw_data)),
                "processed_row_count": int(len(processed_data)),
                "raw_columns": list(finrl_result.raw_data.columns),
                "processed_columns": list(processed_data.columns),
            }

            _save_dataset_artifacts(
                raw_data=finrl_result.raw_data,
                processed_data=processed_data,
                metadata=metadata,
                paths=paths,
            )

            return FinRLDailyDataResult(
                dataset_id=selected_dataset_id,
                universe_id=universe_id,
                tickers=tickers,
                start_date=start_date,
                end_date=end_date,
                raw_data=finrl_result.raw_data,
                processed_data=processed_data,
                paths=paths,
                metadata=metadata,
            )

        except Exception as exc:
            download_error = repr(exc)

            try:
                yfinance_result = _download_yfinance_browser_session_raw_data_in_chunks(
                    tickers=tickers,
                    start_date=start_date,
                    end_date=end_date,
                    chunk_size=chunk_size,
                    sleep_seconds=sleep_seconds,
                    impersonate=yfinance_impersonate,
                    timeout_seconds=yfinance_timeout_seconds,
                )
                download_attempts.extend(yfinance_result.attempts)

                processed_data, technical_indicators = (
                    _process_with_finrl_feature_engineer(
                        raw_data=yfinance_result.raw_data,
                        use_technical_indicators=use_technical_indicators,
                        use_vix=use_vix,
                        use_turbulence=use_turbulence,
                    )
                )

                completeness = _download_completeness_report(
                    requested_tickers=tickers,
                    downloaded_tickers=yfinance_result.downloaded_tickers,
                    failed_tickers=yfinance_result.failed_tickers,
                    processed_data=processed_data,
                    require_all_tickers=require_all_tickers,
                )
                _raise_if_incomplete(
                    method="yfinance browser-session fallback",
                    completeness=completeness,
                    error_by_ticker=yfinance_result.error_by_ticker,
                )

                metadata = {
                    **base_metadata,
                    "cache_used": False,
                    "download_attempted": True,
                    "download_method_attempted": "FinRL YahooDownloader chunked",
                    "actual_data_method": "yfinance browser-session fallback + FinRL FeatureEngineer",
                    "fallback_used": True,
                    "fallback_reason": download_error,
                    "yfinance_impersonate": yfinance_impersonate,
                    "yfinance_timeout_seconds": yfinance_timeout_seconds,
                    "feature_engineering": "FinRL FeatureEngineer",
                    "technical_indicators": technical_indicators,
                    "downloaded_tickers": yfinance_result.downloaded_tickers,
                    "failed_tickers": yfinance_result.failed_tickers,
                    "error_by_ticker": yfinance_result.error_by_ticker,
                    "download_attempts": download_attempts,
                    "download_status": completeness["download_status"],
                    "data_completeness_ok": completeness["data_completeness_ok"],
                    "missing_required_tickers": completeness[
                        "missing_required_tickers"
                    ],
                    "row_count_by_ticker": completeness["row_count_by_ticker"],
                    "raw_row_count": int(len(yfinance_result.raw_data)),
                    "processed_row_count": int(len(processed_data)),
                    "raw_columns": list(yfinance_result.raw_data.columns),
                    "processed_columns": list(processed_data.columns),
                }

                _save_dataset_artifacts(
                    raw_data=yfinance_result.raw_data,
                    processed_data=processed_data,
                    metadata=metadata,
                    paths=paths,
                )

                return FinRLDailyDataResult(
                    dataset_id=selected_dataset_id,
                    universe_id=universe_id,
                    tickers=tickers,
                    start_date=start_date,
                    end_date=end_date,
                    raw_data=yfinance_result.raw_data,
                    processed_data=processed_data,
                    paths=paths,
                    metadata=metadata,
                )

            except Exception as exc:
                yfinance_fallback_error = repr(exc)

    resolved_import_file = _resolve_project_path(import_file)

    if resolved_import_file is not None:
        raw_data, processed_data = _load_finrl_generated_import(
            import_file=resolved_import_file,
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
        )

        actual_tickers = sorted(
            processed_data["tic"].astype(str).str.upper().unique().tolist()
        )
        completeness = _download_completeness_report(
            requested_tickers=tickers,
            downloaded_tickers=actual_tickers,
            failed_tickers=[],
            processed_data=processed_data,
            require_all_tickers=require_all_tickers,
        )
        _raise_if_incomplete(
            method="FinRL-generated local import",
            completeness=completeness,
            error_by_ticker=None,
        )

        metadata = {
            **base_metadata,
            "cache_used": False,
            "download_attempted": allow_download,
            "download_method_attempted": (
                "FinRL YahooDownloader chunked" if allow_download else None
            ),
            "actual_data_method": "FinRL-generated local import",
            "fallback_used": True,
            "fallback_reason": download_error,
            "yfinance_fallback_error": yfinance_fallback_error,
            "import_file": str(resolved_import_file),
            "feature_engineering": "precomputed FinRL features from imported master file",
            "technical_indicators": [],
            "downloaded_tickers": actual_tickers,
            "failed_tickers": [],
            "error_by_ticker": {},
            "download_attempts": download_attempts,
            "download_status": completeness["download_status"],
            "data_completeness_ok": completeness["data_completeness_ok"],
            "missing_required_tickers": completeness["missing_required_tickers"],
            "row_count_by_ticker": completeness["row_count_by_ticker"],
            "raw_row_count": int(len(raw_data)),
            "processed_row_count": int(len(processed_data)),
            "raw_columns": list(raw_data.columns),
            "processed_columns": list(processed_data.columns),
        }

        _save_dataset_artifacts(
            raw_data=raw_data,
            processed_data=processed_data,
            metadata=metadata,
            paths=paths,
        )

        return FinRLDailyDataResult(
            dataset_id=selected_dataset_id,
            universe_id=universe_id,
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            raw_data=raw_data,
            processed_data=processed_data,
            paths=paths,
            metadata=metadata,
        )

    raise RuntimeError(
        "Could not create FinRL daily dataset. "
        f"Download attempted: {allow_download}. "
        f"FinRL download error: {download_error}. "
        f"yfinance fallback error: {yfinance_fallback_error}. "
        "No valid import_file was provided."
    )
