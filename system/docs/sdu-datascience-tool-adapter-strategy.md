# SDU_DataScienceTool Adapter Strategy

Issues: #119 and #171  
Target path: `system/docs/sdu-datascience-tool-adapter-strategy.md`  
Status: Architecture / integration boundary note

## Purpose

This document defines how `SDU_DataScienceTool` should support the StockInvestmentDSS PoC without becoming the application framework.

The intended role is:

```text
StockInvestmentDSS backend
  -> data adapter layer
  -> SDU_DataScienceTool where useful
  -> external market/news APIs
  -> DuckDB / runtime-data storage
  -> backend API
  -> Jinja2 frontend
```

The tool should help with API/data-ingestion workflows. It should not replace the FastAPI backend, the Jinja2 frontend, DuckDB storage, FinRL research layer, or the DSS application structure.

---

## 1. Repository role

`SDU_DataScienceTool` is an external dependency/reference used for reusable API and ingestion logic.

Primary role in StockInvestmentDSS:

- API helper
- market-data adapter support
- news/event adapter support
- async REST/API patterns
- caching/rate-limit/retry inspiration
- optional package dependency later

Not primary role:

- not the frontend framework
- not the backend framework
- not the DSS application
- not the portfolio system
- not the RL training framework
- not a replacement for FinRL

---

## 2. Relevant features for the PoC

Relevant features from `SDU_DataScienceTool`:

| Feature | PoC relevance | Decision |
|---|---|---|
| Async REST client | Useful for controlled backend API ingestion | Adapt/use |
| Retry / backoff | Useful for market/news API robustness | Adapt/use |
| Rate limiting | Useful when calling external APIs | Adapt/use |
| TTL caching | Useful for API cost/performance control | Adapt/use later |
| YahooFinanceSource | Useful for historical OHLCV market data | Use/adapt for #166/#172 |
| GDELTSource | Useful for stock/news/event context | Use/adapt later |
| UTC-first timestamps | Useful for point-in-time correctness | Adapt as principle |
| WebSocket streaming | Interesting, but not first PoC priority | Defer |
| Redis cache | Useful later, but not required for V1.0 | Defer |
| Dash dashboards | Existing visualization inspiration only | Reference, not app framework |
| ML modules | Not current integration target | Defer |

---

## 3. Out of scope

The following must remain out of scope for the first PoC integration:

- replacing StockInvestmentDSS backend with SDU_DataScienceTool
- replacing Jinja2 frontend with Dash
- moving dashboard UI into SDU_DataScienceTool
- copying large parts of the repo into StockInvestmentDSS
- direct frontend calls to SDU_DataScienceTool
- direct frontend calls to external APIs
- Redis dependency in the first version
- WebSocket streaming in the first version
- real-time trading or broker integration
- using SDU_DataScienceTool as the RL framework

---

## 4. Integration boundary

### Allowed direction

```text
frontend
  -> /api/*
  -> StockInvestmentDSS backend
  -> local adapter wrapper
  -> SDU_DataScienceTool source/client
  -> external API
  -> normalized result
  -> DuckDB
  -> backend API response
  -> frontend
```

### Forbidden direction

```text
frontend -> SDU_DataScienceTool
frontend -> external API
frontend -> DuckDB
research notebook -> production app state without agreed persistence contract
```

Reason:

The backend must control:

- source tracking
- API keys
- rate limits
- retries
- caching
- ingestion timestamps
- point-in-time correctness
- DuckDB write rules
- auditability

---

## 5. Recommended implementation shape

Create a thin StockInvestmentDSS adapter layer instead of importing SDU_DataScienceTool everywhere.

Suggested files:

```text
system/backend/app/data_sources/
├── __init__.py
├── sdu_dst_adapter.py
├── market_data_adapter.py
└── news_data_adapter.py
```

Possible later structure:

```text
system/backend/app/market_data_service.py
system/backend/app/market_data_repository.py
system/backend/app/news_data_service.py
system/backend/app/news_data_repository.py
```

The backend service layer should call the adapter. The frontend should only call backend endpoints.

---

## 6. Dependency strategy

Preferred order:

1. Reference current SDU_DataScienceTool API and examples.
2. Add a thin wrapper in StockInvestmentDSS backend.
3. Use package installation only when implementation needs it.
4. Pin dependency or commit when it becomes part of the reproducible PoC.
5. Keep the app runnable without SDU_DataScienceTool until the integration issue is implemented.

Possible dependency options:

### Option A — reference only

Use documentation and code as inspiration. No dependency added.

Use when:

- defining boundary
- designing endpoints
- designing DuckDB schema

### Option B — pip install from Git

Use when the backend actually calls the package.

```text
pip install git+https://github.com/guldmand/SDU_DataScienceTool.git
```

Use only after #166/#172 decides it is needed.

### Option C — local clone under `external/`

Only if a later issue explicitly requires source-level development or pinning.

Default decision:

```text
Do not clone or vendor SDU_DataScienceTool into StockInvestmentDSS for #119/#171.
```

---

## 7. Data flow to DuckDB

Market/news data should be normalized before storage.

### Market data target

External source:

```text
SDU_DataScienceTool YahooFinanceSource / yfinance-compatible source
```

Backend target:

```text
GET  /market/search?q=AAPL
GET  /market/stocks/{ticker}
GET  /market/prices/{ticker}
POST /market/ingest/{ticker}
```

DuckDB target:

```text
market_symbols
market_prices_daily
data_ingestion_log
```

### News/event target

External source:

```text
SDU_DataScienceTool GDELTSource or later news provider adapter
```

Backend target:

```text
GET  /market/news/{ticker}
POST /market/news/ingest/{ticker}
```

DuckDB target:

```text
stock_news
data_ingestion_log
```

---

## 8. Relation to FinRL / yfinance

FinRL remains part of the thesis direction. SDU_DataScienceTool should support data ingestion and API access, not replace FinRL.

Recommended interpretation:

```text
FinRL = research/RL framework and environment direction
yfinance-compatible data = first practical market data source
SDU_DataScienceTool = reusable adapter/API helper for market/news data
DuckDB = shared storage and point-in-time data foundation
StockInvestmentDSS backend = owner of app-facing API contracts
```

The first market-data implementation should remain compatible with later FinRL workflows by:

- storing OHLCV data in normalized tables
- keeping source and ingestion timestamps
- avoiding future leakage
- keeping research notebooks separate from app runtime
- writing reusable exports or tables for later experiments

---

## 9. Relationship to DS808 Visualization

`DS808_Visualization` is relevant as a visualization reference, especially the `clean_dashboard/` work with Plotly/Dash stock-news views.

Decision:

- use as visual and data-flow reference
- do not move Dash into the main PoC frontend now
- adapt stock-news chart ideas later to `/stocks/{ticker}`
- keep the main frontend as Jinja2 + static JS/CSS

Possible future issue:

```text
Map DS808 clean_dashboard stock-news visualization to stock detail view
```

---

## 10. Relationship to AI509 NLP Agent

`AI509_NLP_Agent` is relevant as future work only.

Decision:

- do not implement NLP chatbot in V1.0
- document it as a later insight/explanation layer
- keep thesis focus on RL, DSS, portfolio decisions, risk and audit trail
- later use news/text data as context for explanations or summaries

Possible future role:

```text
portfolio / stock state
  -> stored market/news data
  -> NLP agent
  -> explanation, news summary, question answering
```

---

## 11. Implementation order

Recommended order:

1. #119 / #171: define SDU_DataScienceTool role and boundary
2. #166: create market data foundation with FinRL-yfinance and DuckDB
3. #172: integrate SDU_DataScienceTool for market/news API ingestion
4. #167: create stock lookup/detail view
5. #173: map DS808 clean_dashboard stock-news visualization to stock detail view
6. Later: AI509 NLP Agent as future work layer

---

## 12. Test / verification checklist

For #119 and #171:

- SDU_DataScienceTool URL is recorded in the external manifest.
- Intended role in data/API pipeline is described.
- Integration boundary is described.
- No hidden coupling is introduced.
- PoC can still run without SDU_DataScienceTool.
- Relevant SDU_DataScienceTool features are listed.
- Out-of-scope parts are listed.
- Data flow to backend and DuckDB is clear.
- Frontend-to-external-API coupling is explicitly forbidden.
- Relationship to #166 and #172 is clear.

---

## 13. Decision summary

Decision:

```text
SDU_DataScienceTool should be used as a backend-side adapter/helper for API ingestion where useful.
It should not become the application framework.
It should not be called from frontend code.
It should not be blindly copied into StockInvestmentDSS.
It should remain optional until #172 implements concrete integration.
```

The immediate next implementation dependency remains:

```text
#166 Create market data foundation with FinRL-yfinance and DuckDB
```

---

## 14. Close-comment draft for #119

Defined the SDU_DataScienceTool integration strategy in:

```text
system/docs/sdu-datascience-tool-adapter-strategy.md
```

The document records the intended role of SDU_DataScienceTool as a backend-side API/data-ingestion helper, not an application framework. It defines the integration boundary, data flow, relationship to DuckDB, relationship to FinRL/yfinance, and confirms that the PoC must still run without SDU_DataScienceTool until concrete integration is implemented.

---

## 15. Close-comment draft for #171

Defined the SDU_DataScienceTool adapter strategy in:

```text
system/docs/sdu-datascience-tool-adapter-strategy.md
```

The document lists relevant SDU_DataScienceTool features, out-of-scope parts, the backend adapter boundary, DuckDB storage direction, FinRL/yfinance relationship, and how the decision supports #166 and #172.

No implementation was started in this task.
