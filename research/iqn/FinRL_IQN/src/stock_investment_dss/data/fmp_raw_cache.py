"""
FMP Raw Response Cache.

Saves and loads raw FMP API JSON responses as files under:
  data/api_cache/fmp/raw/<ticker>/<endpoint>_<period>.json

Also maintains a metadata inventory CSV:
  data/api_cache/fmp/fmp_cache_inventory.csv

Adapted from the caching pattern in:
  externals/DS808_Visualization/clean_dashboard/analytics/investor_snapshot.py
"""

from __future__ import annotations

import csv
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CACHE_BASE = _REPO_ROOT / "data" / "api_cache" / "fmp" / "raw"
_INVENTORY_PATH= _REPO_ROOT / "data" / "api_cache" / "fmp" / "fmp_cache_inventory.csv"

_INVENTORY_FIELDS = [
    "ticker",
    "endpoint",
    "period",
    "limit",
    "cached_at",
    "requested_at",
    "source",
    "response_row_count",
    "cache_path",
]


class FMPRawCache:
    """
    File-based raw JSON cache for FMP API responses.

    All responses are stored as JSON files. The inventory CSV tracks
    metadata for all cached responses.
    """

    def __init__(self, cache_base: Optional[Path] = None):
        self.cache_base = Path(cache_base) if cache_base else _CACHE_BASE
        self.inventory_path = self.cache_base.parent / "fmp_cache_inventory.csv"
        self.cache_base.mkdir(parents=True, exist_ok=True)

    def cache_key(
        self, ticker: str, endpoint: str, period: str = "", limit: int = 0
    ) -> Path:
        """Return the file path for a given cache key."""
        ticker_dir = self.cache_base / ticker.upper()
        ticker_dir.mkdir(parents=True, exist_ok=True)
        suffix = f"_{period}" if period else ""
        suffix += f"_lim{limit}" if limit else ""
        filename = f"{endpoint.replace('/', '_')}{suffix}.json"
        return ticker_dir / filename

    def has(self, ticker: str, endpoint: str, period: str = "", limit: int = 0) -> bool:
        """Return True if a cached response exists."""
        return self.cache_key(ticker, endpoint, period, limit).is_file()

    def load(
        self, ticker: str, endpoint: str, period: str = "", limit: int = 0
    ) -> Optional[Any]:
        """Load cached response. Returns None if not cached."""
        path = self.cache_key(ticker, endpoint, period, limit)
        if not path.is_file():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            logger.debug("Cache hit: %s", path.name)
            return data
        except Exception as exc:
            logger.warning("Failed to load cache %s: %s", path, exc)
            return None

    def save(
        self,
        ticker: str,
        endpoint: str,
        data: Any,
        period: str = "",
        limit: int = 0,
        query_params: Optional[Dict] = None,
    ) -> Path:
        """Save raw API response and update inventory."""
        path = self.cache_key(ticker, endpoint, period, limit)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

        row_count = (
            len(data)
            if isinstance(data, list)
            else (len(data) if isinstance(data, dict) else 0)
        )
        self._update_inventory(
            ticker=ticker,
            endpoint=endpoint,
            period=period,
            limit=limit,
            cache_path=str(path),
            response_row_count=row_count,
        )
        logger.info(
            "Cached %s/%s → %d rows → %s", ticker, endpoint, row_count, path.name
        )
        return path

    def list_cached(self) -> List[Dict]:
        """Return list of all cache inventory entries."""
        if not self.inventory_path.is_file():
            return []
        with open(self.inventory_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            return list(reader)

    def _update_inventory(
        self,
        ticker: str,
        endpoint: str,
        period: str,
        limit: int,
        cache_path: str,
        response_row_count: int,
    ) -> None:
        """Append or update a row in the inventory CSV."""
        now = datetime.utcnow().isoformat()
        new_row = {
            "ticker": ticker,
            "endpoint": endpoint,
            "period": period,
            "limit": limit,
            "cached_at": now,
            "requested_at": now,
            "source": "financialmodelingprep",
            "response_row_count": response_row_count,
            "cache_path": cache_path,
        }

        existing: List[Dict] = []
        if self.inventory_path.is_file():
            with open(self.inventory_path, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                existing = [
                    r
                    for r in reader
                    if not (
                        r["ticker"] == ticker
                        and r["endpoint"] == endpoint
                        and r.get("period", "") == period
                    )
                ]

        existing.append(new_row)
        with open(self.inventory_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_INVENTORY_FIELDS)
            writer.writeheader()
            writer.writerows(existing)
