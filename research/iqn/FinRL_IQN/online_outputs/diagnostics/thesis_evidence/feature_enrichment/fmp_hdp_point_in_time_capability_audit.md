# FMP + HDP Point-in-Time Capability Audit

**Date:** 2026-05-22  
**Session:** b9e47002  
**Phase:** EDL v3.6 — FMP fundamental enrichment for HDP

---

## 1. Existing FMP Code in `externals/SDU_DataScienceTool`

### Class: `FinancialModelingPrepSource`
**File:** `externals/SDU_DataScienceTool/src/sdu_dst/sources/financialmodelingprep.py`

This is a full async FMP client built on top of the project's `ApiClient` (httpx + aiolimiter + tenacity retry + TTLCache/Redis caching).

**Base URL:** `https://financialmodelingprep.com/stable`

**API key handling:** `os.getenv("FMP_API_KEY")` — raises `RuntimeError` if not set. Never logged.

**Available methods:**

| Method | FMP Endpoint | Notes |
|--------|-------------|-------|
| `fetch_stock_news()` | `news/stock` | from/to + paging (limit+page), full history |
| `fetch_general_news()` | `news/general-latest` | Latest only |
| `fetch_press_releases()` | `news/press-releases` | Premium plan only |
| `fetch_company_profile()` | `profile` | Current snapshot |
| `fetch_market_quote()` | `quote` | Current snapshot (price, marketCap, beta, shares, 52w) |
| `fetch_key_metrics()` | `key-metrics-ttm` | TTM only — NOT historical |
| `fetch_financials()` | `income-statement`, `balance-sheet-statement`, `cash-flow-statement`, `earnings` | period=quarter/annual + limit |
| `fetch_ratios()` | `ratios-ttm` | TTM only — NOT historical |
| `fetch_income_statement()` | `income-statement` | period + limit |
| `fetch_market_profile()` | `profile` | Alias for company profile |

**Cache behavior:** In-memory TTL cache via `cachetools.TTLCache` (in-process, not persistent). No file-based cache for raw responses.

**Architecture note:** Uses Python `asyncio` + `httpx.AsyncClient`. Our StockInvestmentDSS scaffold uses synchronous `requests` (cache-first) to avoid async dependency in the backtest/training data preparation phase.

---

## 2. Existing Feature Mapping in `externals/DS808_Visualization/clean_dashboard`

### File: `analytics/investor_snapshot.py`

Builds a current-snapshot investor view from FMP data. All metrics are **TTM or current** — not historical PIT.

**Key metrics extracted:**

| Metric | Source | TTM or Historical? |
|--------|--------|:------------------:|
| price | quote.price | Current |
| marketCap | quote.marketCap | Current |
| peTTM | 1/earningsYieldTTM | TTM |
| forwardPE | ratios.forwardPE | Current estimate |
| priceToBook | ratios.priceToBookRatioTTM | TTM |
| evEbitda | key_metrics.evToEBITDATTM | TTM |
| grossMargin | ratios.grossProfitMarginTTM | TTM |
| operatingMargin | ratios.operatingProfitMarginTTM | TTM |
| netMargin | ratios.netProfitMarginTTM | TTM |
| roe | key_metrics.returnOnEquityTTM | TTM |
| roic | key_metrics.returnOnInvestedCapitalTTM | TTM |
| freeCashFlow | key_metrics.freeCashFlowToFirmTTM | TTM |
| fcfYield | key_metrics.freeCashFlowYieldTTM | TTM |
| currentRatio | key_metrics.currentRatioTTM | TTM |
| rdIntensity | income.R&D / income.revenue | Most recent annual |

### Cached Fundamentals
- `data/fundamentals/{TICKER}_snapshot.json` — current-snapshot investors view
- `data/fundamentals/{TICKER}_key_metrics.json` — raw key-metrics response
- Available tickers: AAPL, MSFT, NVDA, NVO, AMZN, GOOGL, TSLA, META, AMD, INTC, JPM, DIS, LLY, V, etc.

### Cached Financials (quarterly/annual)
- `data/financials/AAPL/quarterly.json`, `data/financials/AAPL/annual.json`
- `data/financials/NVO/quarterly.json`, `data/financials/NVO/annual.json`
- **Structure:** `{"income_statement": [...], "balance_sheet": [...], "cash_flow": [...], "earnings": [...]}`

---

## 3. FMP Method Classification for PIT Use

| Method / Endpoint | Date Range Support | PIT Suitability |
|-------------------|--------------------|:---------------:|
| `income-statement` (period+limit) | period + limit only; rows have `date`, `filingDate`, `acceptedDate` | ✅ **PIT-safe** if filtered by `acceptedDate/filingDate <= D` |
| `balance-sheet-statement` (period+limit) | same | ✅ **PIT-safe** |
| `cash-flow-statement` (period+limit) | same | ✅ **PIT-safe** |
| `earnings` | period + limit | ✅ **PIT-safe** (has `date`, `filingDate`) |
| `key-metrics` (non-TTM) | period + limit — historical per period | ✅ **PIT-safe** if available (FMP Premium) |
| `financial-ratios` (non-TTM) | period + limit — historical | ✅ **PIT-safe** if available |
| `key-metrics-ttm` | Current TTM snapshot only | ❌ **NOT PIT-safe** for historical backtest |
| `ratios-ttm` | Current TTM snapshot only | ❌ **NOT PIT-safe** for historical backtest |
| `profile` | Current snapshot | ❌ **NOT PIT-safe** for sector/industry backfill (use as static metadata) |
| `quote` | Current price/cap snapshot | ❌ **NOT PIT-safe** (must use historical price from FinRL market data) |

---

## 4. Confirmed FMP JSON Fields from Cached Data

Verified from `externals/DS808_Visualization/clean_dashboard/data/financials/AAPL/quarterly.json`:

### Income Statement Row
| Field | Present | Sample |
|-------|:-------:|--------|
| date | ✅ | `2025-09-27` |
| filingDate | ✅ | `2025-10-31` |
| acceptedDate | ✅ | `2025-10-31 06:01:26` |
| fiscalYear | ✅ | `2025` |
| period | ✅ | `Q4` |
| symbol | ✅ | `AAPL` |
| revenue | ✅ | 102,466,000,000 |
| grossProfit | ✅ | 48,341,000,000 |
| operatingIncome | ✅ | 32,427,000,000 |
| netIncome | ✅ | 27,466,000,000 |
| eps | ✅ | 1.85 |
| weightedAverageShsOut | ✅ | 14,948,500,000 |
| researchAndDevelopmentExpenses | ✅ | 8,866,000,000 |
| ebitda | ✅ | 35,931,000,000 |

### Balance Sheet Row
| Field | Present | Sample |
|-------|:-------:|--------|
| filingDate | ✅ | `2025-10-31` |
| acceptedDate | ✅ | `2025-10-31 06:01:26` |
| totalAssets | ✅ | 359,241,000,000 |
| totalLiabilities | ✅ | 285,508,000,000 |
| totalDebt | ✅ | 112,377,000,000 |
| cashAndCashEquivalents | ✅ | 33,539,000,000 |
| currentAssets | ❌ | Not present in this version (may differ by FMP plan/endpoint) |
| currentLiabilities | ❌ | Not present — use `totalCurrentLiabilities` if available |
| totalStockholdersEquity | ✅ | 73,733,000,000 |

### Cash Flow Row
| Field | Present | Sample |
|-------|:-------:|--------|
| operatingCashFlow | ✅ | 29,728,000,000 |
| capitalExpenditure | ✅ | -3,242,000,000 |
| freeCashFlow | ✅ | 26,486,000,000 |

**Available date range in AAPL cache:** `2022-12-31` → `2025-09-27` (12 quarters, limit=12)

> **Note:** For backtest 2018–2024, `limit` must be set to ~28+ quarters or use annual + quarterly merge. FMP Premium should provide the full history; limit parameter controls how many periods are returned.

---

## 5. Point-in-Time Rule

> **Critical: The period end `date` is NOT the date the information became public.**
> 
> For a Q4 report with period end `2022-09-24`, the `acceptedDate` is `2022-10-27` — 33 days later.
> Using `date` instead of `acceptedDate` would introduce ~1 month of look-ahead bias per quarter.

**PIT rule for `known_at`:**
1. Use `acceptedDate` if available and valid (non-null, parseable datetime)
2. Else use `filingDate` if available
3. Else use `date` with a `point_in_time_quality = "warn_period_end_only"` flag

**Decision-date filter:**
```python
# For each ticker, at decision_date D:
pit_rows = fundamentals_df[
    (fundamentals_df["ticker"] == ticker) &
    (fundamentals_df["known_at"] <= D)
].sort_values("known_at")
latest_row = pit_rows.iloc[-1]  # most recent known at or before D
```

---

## 6. PIT Fundamental Features from Historical Statements

| Feature | Formula | PIT-safe? |
|---------|---------|:---------:|
| revenue | `revenue` (direct) | ✅ |
| revenue_growth | `(revenue_t - revenue_t-4) / abs(revenue_t-4)` | ✅ (quarterly YoY) |
| earnings_growth | `(netIncome_t - netIncome_t-4) / abs(netIncome_t-4)` | ✅ |
| gross_margin | `grossProfit / revenue` | ✅ |
| operating_margin | `operatingIncome / revenue` | ✅ |
| profit_margin | `netIncome / revenue` | ✅ |
| roe | `netIncome / totalStockholdersEquity` (TTM approx using latest quarter × 4) | ✅ |
| current_ratio | `currentAssets / currentLiabilities` | ⚠️ fields missing in sample; use `cashAndCashEquivalents` proxy if absent |
| debt_ratio | `totalDebt / totalAssets` | ✅ |
| free_cash_flow | `freeCashFlow` (direct) | ✅ |
| fcf_margin | `freeCashFlow / revenue` | ✅ |
| rd_intensity | `researchAndDevelopmentExpenses / revenue` | ✅ |
| quality_score | Composite: `profit_margin + roe_norm + fcf_margin + (1 - debt_ratio)` | ✅ |
| profitability_score | Composite: `gross_margin + operating_margin + profit_margin` | ✅ |
| balance_sheet_strength_score | Composite: `(1 - debt_ratio) + fcf_margin` | ✅ |

---

## 7. Valuation Features Requiring PIT-Aligned Price

| Feature | Formula | Requires PIT price? |
|---------|---------|:------------------:|
| pe_ratio | `close[t] / (trailing_eps × 4)` | ✅ Use FinRL close + EPS from statements |
| ps_ratio | `(close[t] × shares) / (trailing_revenue × 4)` | ✅ |
| ev_ebitda | `(marketCap + totalDebt - cash) / (ebitda × 4)` | ✅ (uses market data close × shares) |
| fcf_yield | `(freeCashFlow × 4) / marketCap` | ✅ |
| earnings_yield | `eps_ttm / close[t]` | ✅ |
| valuation_score | Composite from above | ✅ |

> **Implementation plan:** Compute valuation features at HDP feature join step by combining `close[t]` from FinRL market data with `eps` / `revenue` / `shares` from PIT fundamentals.

---

## 8. Current-Snapshot-Only Features

| Feature | Source | Use Case |
|---------|--------|---------|
| `peTTM`, `ratiosTTM` | `key-metrics-ttm` / `ratios-ttm` | Current dashboard only; NOT for historical backtest |
| `marketCap` | `quote.marketCap` | Use `close × shares_outstanding` from PIT statements instead |
| `forwardPE`, `pegRatio` | `ratios-ttm` | Forward estimates; analyst-dependent; NOT PIT-safe |
| `sector`, `industry` | `profile` | Static metadata — safe to use historically if company doesn't change sector |

---

## 9. Technical Features for HDP (from Market Data)

All features below are computed from FinRL market data using only current/past rows (rolling window, per-ticker, chronological order).

| Feature | Source | Status |
|---------|--------|--------|
| close | Market CSV direct | ✅ Ready |
| macd | Market CSV column | ✅ Ready (already in CSV) |
| rsi_30 | Market CSV column | ✅ Ready |
| cci_30 | Market CSV column | ✅ Ready |
| dx_30 | Market CSV column | ✅ Ready |
| close_30_sma | Market CSV column | ✅ Ready |
| close_60_sma | Market CSV column | ✅ Ready |
| MA50 / ma50 | Rolling 50-day mean (close) | ✅ `TechnicalFeatureBuilder` |
| MA200 / ma200 | Rolling 200-day mean (close) | ✅ `TechnicalFeatureBuilder` |
| price_vs_ma50 | `(close - MA50) / MA50` | ✅ `TechnicalFeatureBuilder` |
| price_vs_ma200 | `(close - MA200) / MA200` | ✅ `TechnicalFeatureBuilder` |
| price_vs_sma50 | `(close - close_30_sma) / close_30_sma` | ✅ (close_30_sma ≈ sma30, not sma50; computed from CSV) |
| price_vs_sma200 | Needs close_200_sma or MA200 | ✅ Use MA200 |
| recent_return_5d | `close.pct_change(5)` | ✅ NEW in hdp_technical_feature_builder |
| recent_return_20d | `close.pct_change(20)` | ✅ `TechnicalFeatureBuilder.recent_return` |
| volatility_20d | `rolling 20d log-return std × sqrt(252)` | ✅ `TechnicalFeatureBuilder.volatility_score` |
| drawdown_from_recent_high | `(close - rolling_max_60d) / rolling_max_60d` | ✅ `TechnicalFeatureBuilder` |
| momentum_score | Composite: return + price_vs_ma50 + price_vs_ma200 + MACD | ✅ `TechnicalFeatureBuilder` |
| technical_risk_score | Composite: volatility + drawdown | ✅ NEW in hdp_technical_feature_builder |

---

## 10. HDP Feature Usage

| Action | Preferred Feature Signals |
|--------|--------------------------|
| **BUY** ticker selection | Strong valuation (low pe_ratio, high fcf_yield), high quality_score, positive momentum_score, low technical_risk_score, low debt_ratio, high revenue/earnings growth |
| **SELL** ticker selection | Negative momentum, high technical_risk_score, high debt_ratio, deteriorating margins, low quality_score, deep drawdown |
| **HOLD** decision | Ambiguous signals, very high uncertainty, recent rebalance, cash constraint |
| **SIZE** selection | Low volatility_20d → larger size; low risk_score → larger size; high cash_weight constraint → smaller; concentration limits → cap |

---

## 11. FMP Premium Data Coverage for Backtest Periods

| Period | Quarters Required | FMP Limit Needed | Confidence |
|--------|:-----------------:|:---------------:|:----------:|
| demo_5: 2018–2024 | ~24 quarters | limit=28 quarterly | ✅ High (FMP Premium has 10+ years history) |
| demo_10: 2015–2024 | ~36 quarters | limit=40 quarterly | ✅ High |
| Longer tests to 2010 | ~56 quarters | limit=60 quarterly | ✅ High (FMP typically provides ~15 years) |

---

## 12. Recommended Implementation Order

| Priority | Item | Phase |
|----------|------|-------|
| A | Historical statement-based PIT fundamentals (income + balance + cashflow) | First |
| B | Technical indicators from existing market data (MA50/200, price_vs_*, momentum, volatility) | First (already partially done) |
| C | Valuation features using PIT close × PIT shares/EPS | Second (after A+B) |
| D | SEC filing metadata validation (check acceptedDate exists) | Ongoing |
| E | Historical `key-metrics` / `financial-ratios` (non-TTM) endpoints if FMP Premium provides | Second |
| F | News/press release count features | Later |

---

## 13. External Repo Attribution for Implementation

| StockInvestmentDSS file | Inspired by |
|------------------------|-------------|
| `fmp_api_client.py` | `SDU_DataScienceTool/src/sdu_dst/sources/financialmodelingprep.py` (endpoint list, API key pattern) |
| `fmp_raw_cache.py` | `SDU_DataScienceTool/src/sdu_dst/sources/financialmodelingprep.py` + `clean_dashboard/analytics/investor_snapshot.py` (snapshot caching pattern) |
| `fmp_pit_fundamentals_builder.py` | `clean_dashboard/analytics/investor_snapshot.py` (field mapping, metric calculations) |
| `hdp_technical_feature_builder.py` | `src/stock_investment_dss/data/technical_feature_builder.py` (existing, extending) |
| `hdp_feature_store.py` | `src/stock_investment_dss/data/fundamental_feature_store.py` (PIT join pattern) |
