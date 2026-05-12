# Zero Sum Public Reference Map

Issue: #164 — Map zero-sum-public features to StockInvestmentDSS PoC  
Target path: `system/docs/zero-sum-public-reference-map.md`

## 1. Reference pages reviewed

The following Zero Sum Public / Zero Sum Times pages are useful as application and UI references for the StockInvestmentDSS PoC:

| Zero Sum route | Reference value for StockInvestmentDSS |
|---|---|
| `/portfolio` | Portfolio holdings, position overview, portfolio metrics, risk dashboard and transaction-oriented user flow. |
| `/stocks/AAPL` | Stock detail page with quote, company information, price history, fundamentals and analytical context. |
| `/watchlist` | User-curated list of stocks to monitor before adding to portfolio or strategy workflows. |
| `/compare?tickers=AAPL,MSFT` | Multi-stock comparison surface for relative performance, fundamentals and risk/return context. |
| `/correlation` | Correlation analysis across selected tickers. Useful for diversification and portfolio risk discussion. |
| `/chart` | Chart terminal reference for stock price exploration and overlays. |
| `/technical` | Technical indicator reference: Bollinger Bands, RSI, MACD, moving averages, support/resistance and setup detection. |
| `/scanner` | Market signal scanner reference for later fast-layer signal overview. |
| `/heatmap` | Market overview reference using sector/ticker heatmap visualization. |
| `/bubble` | Alternative market overview reference using bubble map visualization. |
| `/sectors` | Sector overview reference for market context and diversification. |
| `/news?symbol=AAPL` | Stock-specific news surface. Useful later for context and possible AI-generated summaries. |
| `/earnings` | Earnings calendar reference for event-aware decision support. |
| `/learn` | Educational/reference material surface. Useful later, but not part of the core PoC. |

Zero Sum Public is used as a reference only. This issue does not decide to vendor the repository, migrate to Next.js/React, or clone the full Zero Sum application.

## 2. What to adapt

The PoC should adapt the ideas that directly support decision support, portfolio construction and explainable stock investment workflows.

### Adapt early

| Feature | Why it matters | PoC target |
|---|---|---|
| Stock lookup | Users need to search for a ticker before building a portfolio or requesting a decision. | `/stocks` and `/stocks/{symbol}` |
| Stock detail | A DSS user needs a single-stock overview before making buy/hold/sell decisions. | Quote, metadata, price history, basic indicators, risk notes. |
| Portfolio holdings | Core DSS workflow requires user-owned holdings and position state. | `/portfolio` |
| Transactions | Buy/sell decisions need traceability. | `transactions` table + transaction history view. |
| Watchlist | Users need to monitor stocks before buying. | `/watchlist` or watchlist panel on dashboard. |
| Dashboard cards | Fast entry point for backend status, portfolio state, risk output and recommendation status. | `/dashboard` |
| Compare | Users need simple relative comparison between candidate stocks. | `/compare` |
| Correlation | Portfolio risk needs diversification evidence. | `/correlation` |

### Adapt after data foundation

| Feature | Why it matters | PoC target |
|---|---|---|
| Basic chart display | Visual price history is expected in stock decision support. | `/stocks/{symbol}` chart section. |
| Technical indicators | Useful as engineered features and explainable context. | Later `/technical` or stock detail indicator panel. |
| Market overview placeholders | Useful demo surfaces, but not first implementation priority. | `/market-overview`, `/heatmap`, `/scanner` placeholders. |
| News | Useful context and possible future NLP/AI summarization. | `/news?symbol=AAPL` later via backend ingestion. |

## 3. What to defer

The following are explicitly deferred for the thesis PoC:

| Deferred item | Reason |
|---|---|
| Full TradingView clone | Too large; not required for thesis DSS validation. |
| Full real-time market platform | PoC should use stored/reproducible data first. |
| Full fundamentals engine | Nice to have, but market price + portfolio + risk + RL integration comes first. |
| DCF valuation | Not central to RL/DSS PoC. |
| Congress/insider data | Interesting context, but not needed for first thesis system. |
| Full news aggregation platform | Defer; stock-specific news can be added later as a thin backend API. |
| Complete Zero Sum UI clone | We adapt patterns, not the whole product. |
| Next.js/React migration | Current PoC stack is Jinja2 + FastAPI + DuckDB. |

## 4. Current-stack implementation target

The target implementation remains:

```text
system/
├── backend/          # FastAPI API, data access, auth endpoints
├── frontend/         # Jinja2 frontend shell and static assets
├── runtime-data/     # local DuckDB and runtime files, ignored by Git
└── docs/             # concise PoC documentation
```

Implementation rules:

- Frontend must call backend API endpoints.
- Frontend must not call DuckDB directly.
- Frontend must not call external market APIs directly.
- Backend owns market data access, persistence and transformations.
- DuckDB remains the local runtime data target.
- guldNAS paths remain configurable and must not be hardcoded into frontend code.
- Research notebooks and FinRL experiments stay under `research/`.
- Application views stay under `system/frontend/`.
- Backend service code stays under `system/backend/`.

## 5. Required backend endpoints

The following API targets should be treated as the first useful backend surface. Exact naming can be adjusted during implementation.

### System

```text
GET /health
GET /health/duckdb
GET /config/runtime
```

Already exists or partly exists.

### Market data

```text
GET  /market/stocks/search?q=AAPL
GET  /market/stocks/{symbol}
GET  /market/stocks/{symbol}/prices
GET  /market/stocks/{symbol}/summary
GET  /market/stocks/{symbol}/news
```

### Portfolio and transactions

```text
GET    /portfolio
POST   /portfolio/holdings
DELETE /portfolio/holdings/{holding_id}

GET    /transactions
POST   /transactions
```

### Watchlist

```text
GET    /watchlist
POST   /watchlist
DELETE /watchlist/{symbol}
```

### Analytics

```text
GET /analytics/compare?tickers=AAPL,MSFT
GET /analytics/correlation?tickers=AAPL,MSFT,GOOG
GET /analytics/technical/{symbol}
GET /analytics/risk/portfolio
```

### Audit / evidence

```text
GET  /audit/events
POST /audit/events
```

The audit endpoints should support point-in-time evidence and later decision traceability.

## 6. Required DuckDB tables

Initial tables should support stored market data, user portfolio state and traceability.

### First priority

| Table | Purpose |
|---|---|
| `stock_metadata` | Ticker, company name, sector, industry, exchange, currency and basic metadata. |
| `market_prices` | OHLCV price history from FinRL/yfinance-compatible ingestion. |
| `portfolio_holdings` | Current user holdings and position state. |
| `transactions` | Buy/sell/add/remove transaction history. |
| `watchlist_items` | User watchlist symbols. |
| `audit_events` | Point-in-time events for traceability. |

### Second priority

| Table | Purpose |
|---|---|
| `portfolio_snapshots` | Daily/periodic portfolio value and performance snapshots. |
| `technical_indicators` | Stored indicators such as moving averages, RSI, MACD, Bollinger Bands. |
| `correlation_results` | Cached correlation outputs for selected ticker sets. |
| `stock_news` | Stock-specific news metadata and optional summaries. |
| `recommendation_outputs` | DSS recommendation output with model version, data snapshot and explanation metadata. |
| `model_registry` | Later link between trained models, inference artifacts and evaluation outputs. |

## 7. Required frontend templates/views

The Jinja2 frontend should keep the surface small and avoid many independent full HTML pages.

### Core views

| Route | Purpose |
|---|---|
| `/login` | Login/register page with session guard. |
| `/dashboard` | Main DSS shell after login. |
| `/stocks` | Search and stock lookup entry point. |
| `/stocks/{symbol}` | Stock detail page. |
| `/portfolio` | Holdings, transactions and portfolio overview. |
| `/watchlist` | Watchlist management. |
| `/compare` | Multi-stock comparison. |
| `/correlation` | Correlation analysis. |
| `/risk` | Portfolio risk output. |
| `/audit` | Point-in-time evidence and decision traceability. |

### Later / placeholder views

| Route | Purpose |
|---|---|
| `/market-overview` | Heatmap/bubble/sector placeholder. |
| `/technical` | Technical analysis placeholder. |
| `/scanner` | Technical signal scanner placeholder. |
| `/recommendations` | Recommendation output and DSS decision surface. |
| `/settings` | PoC settings and runtime configuration. |
| `/profile` | User/profile placeholder. |

## 8. FinRL/yfinance/data dependency notes

FinRL remains a requirement for the thesis direction and should be reflected in the data foundation. The PoC should use FinRL-compatible data ingestion or data structures where practical, while keeping the first implementation simple.

Recommended approach:

1. Use yfinance-compatible market data as the first quick ingestion target.
2. Store normalized stock metadata and price history in DuckDB.
3. Keep ingestion/backend code compatible with later FinRL environments.
4. Keep research notebooks separate from the web app.
5. Use DuckDB tables as the bridge between:
   - market ingestion,
   - portfolio/risk views,
   - comparison/correlation analytics,
   - later FinRL/RL experiment outputs.
6. Add point-in-time/audit metadata early so later backtests and recommendations can explain what data was known at decision time.

SDU_DataScienceTool can later be integrated as an adapter layer for API calls and ingestion utilities, but it should not block #164.

## 9. Priority order

The recommended implementation order after this mapping is:

| Priority | Issue / workstream | Reason |
|---:|---|---|
| 1 | #165 Define zero-sum-public adaptation boundary | Prevent scope creep and framework migration. |
| 2 | #119 / #171 SDU_DataScienceTool integration strategy | Clarify adapter role before ingestion grows. |
| 3 | #166 Create market data foundation with FinRL-yfinance and DuckDB | Data must exist before stock, portfolio and analytics views become meaningful. |
| 4 | #167 Create Zero Sum inspired stock lookup and stock detail view | First user-facing market data feature. |
| 5 | #168 Create minimal portfolio watchlist and transaction flow | Core DSS user workflow. |
| 6 | #169 Create comparison and correlation prototype from stored market data | Early analytical evidence for diversification/risk. |
| 7 | #170 Create market overview placeholders for heatmap scanner and technical views | Useful UI direction, but not first dependency. |

## 10. First implementation issue

The first true implementation dependency after the mapping/boundary decisions is:

```text
#166 Create market data foundation with FinRL-yfinance and DuckDB
```

Without #166, stock lookup, portfolio analytics, comparison, correlation, technical analysis and market overview pages would become weak placeholders.

## 11. Decision summary

Zero Sum Public should be used as a visual, UX and feature reference for the StockInvestmentDSS PoC.

The PoC should adapt:

- stock lookup,
- stock detail,
- portfolio holdings,
- transactions,
- watchlist,
- compare,
- correlation,
- basic charting,
- later market overview concepts.

The PoC should not become:

- a full Zero Sum clone,
- a full TradingView clone,
- a Next.js/React migration,
- a real-time finance portal,
- a broad financial news/fundamentals platform.

The StockInvestmentDSS direction remains:

```text
Jinja2 frontend
FastAPI backend
DuckDB runtime data
FinRL/yfinance-compatible market data
SDU_DataScienceTool adapter later
point-in-time evidence
RL/DSS-focused thesis PoC
```

## 12. Close-comment draft for issue #164

Created `system/docs/zero-sum-public-reference-map.md`.

The document maps relevant Zero Sum Public reference routes to StockInvestmentDSS PoC targets, including stock lookup, stock detail, portfolio, watchlist, transactions, comparison, correlation, charting, technical analysis, market overview, news and earnings.

The mapping explicitly states that zero-sum-public is reference only. It does not propose vendoring the repository, migrating to Next.js/React, or cloning the full Zero Sum application.

The first implementation dependency is identified as #166: Create market data foundation with FinRL-yfinance and DuckDB.
