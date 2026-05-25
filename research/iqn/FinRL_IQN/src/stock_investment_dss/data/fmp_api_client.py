"""
FMP API Client for StockInvestmentDSS.

Synchronous wrapper around FinancialModelingPrep endpoints.
Adapted from externals/SDU_DataScienceTool/src/sdu_dst/sources/financialmodelingprep.py
(see docs/FMP_HDP_Point_In_Time_Features_v3_6.md for attribution).

Rules:
- API key read from FMP_API_KEY env var only. Never printed or logged.
- live_enabled=False by default; set True only during explicit ingestion phase.
- cache_only=True by default; returns cached data only.
- No live calls inside backtest/training/HDP decision loops.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Literal, Optional

logger = logging.getLogger(__name__)

BASE_URL = "https://financialmodelingprep.com/stable"


class FMPApiClient:
    """
    Synchronous FMP client with live/cache control.

    Parameters
    ----------
    live_enabled : bool
        If False, all fetch_* methods raise FMPLiveDisabledError.
        Set True only during the dedicated ingestion phase.
    cache_only : bool
        Alias for ``not live_enabled``; provided for readability.
    timeout : float
        HTTP request timeout in seconds.
    """

    def __init__(
        self,
        live_enabled: bool = False,
        cache_only: bool = True,
        timeout: float = 30.0,
    ):
        # live_enabled takes precedence; cache_only=True forces live off
        self.live_enabled = bool(live_enabled) and not bool(cache_only)
        self.timeout = timeout
        self._api_key: Optional[str] = None

    def _get_api_key(self) -> str:
        if self._api_key:
            return self._api_key
        key = os.environ.get("FMP_API_KEY", "")
        if not key:
            raise RuntimeError(
                "FMP_API_KEY environment variable is not set. "
                "Set it before running live ingestion."
            )
        self._api_key = key
        return key

    def _get(self, endpoint: str, params: Dict[str, Any]) -> Any:
        """Execute a live GET request. Raises FMPLiveDisabledError if not enabled."""
        if not self.live_enabled:
            raise FMPLiveDisabledError(
                f"Live FMP calls are disabled. "
                f"Set STOCK_INVESTMENT_DSS_FMP_LIVE_ENABLED=true to enable. "
                f"Endpoint requested: {endpoint}"
            )
        try:
            import requests  # local import — stdlib-adjacent
        except ImportError:
            import urllib.request as _ur
            import json as _json
            import urllib.parse as _up

            params["apikey"] = self._get_api_key()
            url = f"{BASE_URL}/{endpoint.lstrip('/')}?{_up.urlencode(params)}"
            logger.debug("FMP GET %s", url.replace(params["apikey"], "***"))
            with _ur.urlopen(url, timeout=self.timeout) as resp:
                return _json.loads(resp.read().decode())
        else:
            params = dict(params)
            params["apikey"] = self._get_api_key()
            url = f"{BASE_URL}/{endpoint.lstrip('/')}"
            logger.debug(
                "FMP GET %s params=%s",
                url,
                {k: ("***" if k == "apikey" else v) for k, v in params.items()},
            )
            resp = requests.get(url, params=params, timeout=self.timeout)
            resp.raise_for_status()
            return resp.json()

    # ------------------------------------------------------------------
    # Public fetch methods — each returns raw parsed JSON (list or dict)
    # ------------------------------------------------------------------

    def fetch_income_statement(
        self,
        symbol: str,
        period: Literal["quarter", "annual"] = "quarter",
        limit: int = 80,
    ) -> List[dict]:
        """Income statement rows with date, filingDate, acceptedDate, revenue, etc."""
        data = self._get(
            "income-statement", {"symbol": symbol, "period": period, "limit": limit}
        )
        return data if isinstance(data, list) else []

    def fetch_balance_sheet(
        self,
        symbol: str,
        period: Literal["quarter", "annual"] = "quarter",
        limit: int = 80,
    ) -> List[dict]:
        """Balance sheet rows."""
        data = self._get(
            "balance-sheet-statement",
            {"symbol": symbol, "period": period, "limit": limit},
        )
        return data if isinstance(data, list) else []

    def fetch_cash_flow(
        self,
        symbol: str,
        period: Literal["quarter", "annual"] = "quarter",
        limit: int = 80,
    ) -> List[dict]:
        """Cash flow statement rows."""
        data = self._get(
            "cash-flow-statement", {"symbol": symbol, "period": period, "limit": limit}
        )
        return data if isinstance(data, list) else []

    def fetch_company_profile(self, symbol: str) -> List[dict]:
        """Company profile (current snapshot): sector, industry, companyName, etc."""
        data = self._get("profile", {"symbol": symbol})
        return (
            data
            if isinstance(data, list)
            else ([data] if isinstance(data, dict) else [])
        )

    def fetch_key_metrics(
        self,
        symbol: str,
        period: Literal["quarter", "annual"] = "quarter",
        limit: int = 80,
    ) -> List[dict]:
        """
        Historical key metrics (FMP Premium). Returns per-period rows with
        revenuePerShareTTM, roe, currentRatio, etc.
        Falls back gracefully if endpoint unavailable.
        """
        try:
            data = self._get(
                "key-metrics", {"symbol": symbol, "period": period, "limit": limit}
            )
            return data if isinstance(data, list) else []
        except Exception as exc:
            logger.warning("fetch_key_metrics failed for %s: %s", symbol, exc)
            return []

    def fetch_financial_ratios(
        self,
        symbol: str,
        period: Literal["quarter", "annual"] = "quarter",
        limit: int = 80,
    ) -> List[dict]:
        """
        Historical financial ratios (FMP Premium). Falls back gracefully.
        """
        try:
            data = self._get(
                "ratios", {"symbol": symbol, "period": period, "limit": limit}
            )
            return data if isinstance(data, list) else []
        except Exception as exc:
            logger.warning("fetch_financial_ratios failed for %s: %s", symbol, exc)
            return []

    def fetch_sec_filings(self, symbol: str, limit: int = 40) -> List[dict]:
        """
        SEC filings metadata (FMP Premium). Falls back gracefully.
        Useful for validating acceptedDate and filingDate alignment.
        """
        try:
            data = self._get("sec-filings", {"symbol": symbol, "limit": limit})
            return data if isinstance(data, list) else []
        except Exception as exc:
            logger.warning("fetch_sec_filings failed for %s: %s", symbol, exc)
            return []


class FMPLiveDisabledError(RuntimeError):
    """Raised when a live FMP call is attempted with live_enabled=False."""
