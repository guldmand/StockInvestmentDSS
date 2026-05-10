# StockInvestmentDSS

**Data-Driven Stock Investment Decision Support**  
*A Reinforcement Learning Approach to Risk-Sensitive Sequential Portfolio Decisions*

StockInvestmentDSS is a master thesis proof-of-concept for a data-driven decision support system for stock investors.

The goal is not to build an autonomous trading bot or a system that claims to systematically beat the market. The goal is to design and implement a transparent, auditable and risk-aware decision support system that helps an investor translate a chosen strategy and risk profile into consistent buy/hold/sell recommendations.

The system combines:

- point-in-time market data handling
- portfolio and strategy modeling
- reinforcement learning experiments
- risk-aware decision support
- auditability and reproducibility
- local, k3s and future cloud/GPU deployment paths

---

## README Version

Current documentation version: **v4.1**

This version aligns the README with the current PoC project board after the README-alignment, external dependency, research track, ML/Deep RL, W&B, and slow/fast layer task scripts.

README v4.1 reflects the repository structure after project-management scripts **10–13** and the GitHub Project automation workflow.

## Thesis Context

This repository supports the MSc Data Science master thesis:

**Data-Driven Stock Investment Decision Support**  
**A Reinforcement Learning Approach to Risk-Sensitive Sequential Portfolio Decisions**

Author: **Jannik Busse Guldmand**  
Programme: **MSc Data Science, University of Southern Denmark**  
Supervisor: **Professor Melih Kandemir**

---

## Core Problem

Stock investors continuously face sequential decisions:

- which stocks to buy
- which stocks to hold
- which stocks to sell
- when to rebalance
- when to reduce or increase risk exposure
- when a change in market conditions may justify a change in strategy

In practice, such decisions are often inconsistent, intuition-driven or insufficiently grounded in risk-aware data analysis.

This project investigates how a decision support system can help investors make more consistent, data-driven and risk-aware decisions under uncertainty.

---

## Main Goal

Build a working proof-of-concept DSS that can:

1. ingest and store market data
2. preserve point-in-time correctness
3. let a user define a portfolio and investment strategy
4. generate buy/hold/sell decision alternatives
5. attach risk and uncertainty information to those decisions
6. log enough information to reproduce and audit recommendations
7. produce tables, figures and evidence for the thesis report

---

## What This System Is

StockInvestmentDSS is:

- a **decision support system**
- a **research prototype**
- a **PoC for a master thesis**
- a **risk-aware investment assistant**
- an **auditable data-driven architecture**
- a **platform for FinRL / Gymnasium / RL experiments**

---

## What This System Is Not

StockInvestmentDSS is not:

- an autonomous trading bot
- financial advice
- a production trading platform
- a market-beating claim
- a high-frequency trading system
- a black-box model-only project

The system is designed to support human decision-making, not replace it.

---

## Current V1.0 PoC Scope

The first proof-of-concept focuses on a minimal but complete vertical slice:

```text
Data
→ Point-in-time storage
→ Strategy definition
→ Portfolio definition
→ Decision engine
→ Risk output
→ Audit log
→ Demo UI
→ Thesis figures/results
```

The V1.0 PoC should prove that the architecture works end-to-end before more advanced RL, IQN and uncertainty-aware models are expanded.

---

## Architectural Principle

The system is split into two operational speeds:

```text
Slow layer  = offline model training, backtesting and evaluation
Fast layer  = near real-time decision support using available data, cached features and existing models
```

### Slow Layer

The slow layer handles:

- historical data preparation
- FinRL experiments
- Gymnasium environment experiments
- model training
- backtesting
- baseline comparisons
- evaluation metrics
- model registry updates

This layer may run on:

- local development machine
- GPU box
- Vast.ai / Colab / cloud GPU
- future scheduled infrastructure

### Fast Layer

The fast layer handles:

- user login
- stock lookup
- portfolio creation
- strategy creation
- applying user constraints
- generating decision alternatives
- showing risk indicators
- writing audit logs

The fast layer should respond quickly and must not retrain a deep RL model for every user interaction.

### Slow/Fast Research Mapping

The slow/fast split is reflected in both the runnable system and the research workspace.

| Layer | Research location | System location | Purpose |
|---|---|---|---|
| Slow layer | `research/notebooks/`, `research/experiments/slow_layer/` | `workers/finrl-worker/`, `workers/training-job/`, `packages/dss-finrl-adapter/`, `packages/dss-env/` | Offline training, backtesting, evaluation, metrics and model registry updates |
| Fast layer | `research/notebooks/`, `research/experiments/fast_layer/` | `backend/`, `workers/decision-worker/`, `packages/dss-strategies/`, `packages/dss-risk/` | Near real-time decision support, strategy constraints, risk output and audit logging |
| Shared | `research/configs/`, `research/results/` | `packages/`, `sql/`, `configs/` | Shared schemas, configs, data contracts, outputs and thesis evidence |

For V1.0, the slow layer may use a baseline/proxy model output, while the fast layer must be able to generate useful decision support without retraining a deep RL model live.

---

## High-Level System Flow

```text
External APIs / FinRL / yfinance / future sources
        ↓
Raw file storage
        ↓
Point-in-time ingestion
        ↓
DuckDB canonical store
        ↓
Feature pipeline
        ↓
Strategy + portfolio state
        ↓
Decision engine
        ↓
Risk / uncertainty output
        ↓
Audit log
        ↓
Frontend / thesis demo
```

---

## Core Research Direction

The thesis investigates reinforcement learning as part of a risk-aware decision support architecture.

The contribution is not only the RL model. The contribution is the system design around it:

- point-in-time data constraints
- auditable recommendation generation
- baseline comparison
- risk-sensitive evaluation
- human-in-the-loop decision support
- separation between offline training and online inference

---

## Evaluation Focus

The PoC and thesis evaluation should not only report profit.

Relevant metrics include:

| Metric | Purpose |
|---|---|
| Cumulative return | Basic investment performance |
| Annualized return | Comparable performance |
| Sharpe ratio | Risk-adjusted return |
| Maximum drawdown | Downside loss exposure |
| VaR / CVaR proxy | Downside-risk analysis |
| Turnover | Trading aggressiveness |
| Decision traceability | Auditability |
| Point-in-time correctness | Data leakage prevention |

Baseline comparisons should include:

- buy-and-hold
- equal-weight portfolio
- risk-adjusted decision-support output
- later: FinRL / RL agent output
- later: distributional / uncertainty-aware variant

---

## Technology Stack

### Application

| Area | Technology |
|---|---|
| Backend API | Python / FastAPI |
| Frontend | Blazor WebAssembly or lightweight web demo |
| Database | DuckDB |
| Data files | Parquet / raw API snapshots |
| Container runtime | Docker |
| Local orchestration | Docker Compose |
| Cluster orchestration | k3s on Turing Pi |
| CI/CD | GitHub Actions |

### Data Science / RL

| Area | Technology |
|---|---|
| Financial RL framework | FinRL |
| Environment interface | Gymnasium |
| RL prototyping | ObjectRL |
| Deep learning | PyTorch |
| Market data | yfinance / FinRL data tools |
| Additional ingestion | SDU_DataScienceTool |
| Analysis | Python notebooks |
| Experiment tracking | Weights & Biases |
| Storage analytics | DuckDB |

### Infrastructure

| Area | Target |
|---|---|
| Local development | MacBook Pro |
| CPU services / orchestration | Turing Pi 2 with 4× RK1 nodes |
| Persistent storage | guldNAS / Raspberry Pi NAS |
| GPU training | Local GPU box / cloud GPU |
| Public demo target | guldmand.com/data-science/master-thesis |

---

## Experiment Tracking

ML, Deep RL, FinRL, slow-layer training and evaluation experiments may be logged to **Weights & Biases**.

W&B is used for:

- experiment configuration
- training and evaluation metrics
- run metadata
- model and checkpoint references
- dataset and feature build references
- reproducibility notes

W&B should support reproducibility and thesis evidence. It must not become a blocker for V1.0; if credentials are unavailable, experiments should still run locally and log metrics to files, DuckDB, CSV or Parquet.

---

## Device Architecture

```text
MacBook Pro
  - development
  - notebooks
  - Git
  - local testing
  - documentation

GPU Box
  - heavy RL training
  - FinRL training
  - PyTorch CUDA workloads
  - model checkpoints

Turing Pi 2 / k3s
  - orchestration
  - ingestion workers
  - feature workers
  - strategy/decision services
  - monitoring / lightweight services

guldNAS
  - DuckDB storage
  - Parquet datasets
  - raw API snapshots
  - model checkpoints
  - logs
  - backtest results

Cloud GPU / Vast.ai / Colab
  - optional heavy training
  - larger experiments
  - fallback compute
```

---

## Repository Structure

This repository uses the V4 README structure based on the V3 canonical architecture.

Important correction from the original V1 structure:

```text
StockInvestmentDSS/ = repository root
system/             = runnable DSS system / demo
research/           = academic experiments and thesis work
docs/               = architecture and infrastructure documentation
.github/            = CI/CD automation
```

`system/` is not the whole repository root.  
`system/` is the operational DSS system inside the repository.

---

## Project Structure

```text
StockInvestmentDSS/                         # Repository root
├─ README.md                                # Main project overview and active README
├─ .gitignore                               # Ignored files and folders
├─ .env.example                             # Root-level environment template only
│
├─ docs/                                    # Repository-level documentation
│  ├─ architecture/                         # Architecture documentation
│  │  ├─ device-architecture.md             # Mac, GPU box, Turing Pi, guldNAS and cloud mapping
│  │  ├─ software-architecture.md           # DSS, RL, strategy layer, PIT and auditability
│  │  ├─ deployment-architecture.md         # Local, test, staging, prod, GPU and cloud deployment mapping
│  │  ├─ slow-fast-layer-architecture.md    # Offline training vs online decision support architecture
│  │  └─ decision-audit-architecture.md     # Decision traceability and audit log design
│  │
│  ├─ infrastructure/                       # Infrastructure and platform documentation
│  │  ├─ cluster-inventory.md               # Permanent overview of machines, nodes and services
│  │  ├─ guldnas-storage-layout.md          # Persistent NAS folder layout and mount assumptions
│  │  ├─ turingpi-preflight-2026-05-03.md   # Raw Turing Pi health/status log before k3s
│  │  └─ runbooks/                          # Operational step-by-step guides
│  │     ├─ k3s-bootstrap.md                # k3s installation and node join guide
│  │     ├─ guldnas-mount.md                # Mounting guldNAS on local/dev/test machines
│  │     ├─ local-development.md            # Local development runbook
│  │     └─ deployment-runbook.md           # Deployment and rollback runbook
│  │
│  ├─ project-structure/                    # Documentation of project structure evolution
│  │  ├─ project-structure-v1-original.md   # Original detailed V1 structure
│  │  ├─ project-structure-v2-service-oriented.md # Service-oriented architecture proposal
│  │  ├─ project-structure-v3-canonical.md  # Canonical architecture decision
│  │  └─ project-structure-v4-readme.md     # Current README-oriented structure
│  │
│  └─ thesis/                               # Thesis-specific notes and planning
│     ├─ problem-formulation.md             # Thesis problem formulation
│     ├─ research-questions.md              # Research questions and scope
│     ├─ methodology-notes.md               # Methodology notes and assumptions
│     ├─ evaluation-plan.md                 # Evaluation and metrics plan
│     └─ ai-usage-disclosure.md             # AI usage disclosure for thesis/report
│
├─ project-management/                      # GitHub project board automation and scripts
│  ├─ README.md                             # Project-management script overview
│  └─ scripts/                              # gh CLI / project automation scripts
│     ├─ 1_create_board_labels_milestones.sh # Create labels, milestones and initial board setup
│     ├─ 2_create_issues.sh                 # Create initial PoC issues
│     ├─ 3_set_project_field_values.sh      # Set project fields on existing issues
│     ├─ 6_add_missing_v1_poc_tasks.sh      # Add missing V1.0 PoC tasks
│     ├─ 7_fix_priority_fields.sh           # Move priority labels into real Priority field
│     ├─ 8_fix_categories_and_status.sh     # Fix category/status project fields
│     ├─ 9_add_risk_adjusted_baseline_comparison.sh # Add missing baseline comparison issue
│     ├─ 10_add_missing_readme_alignment_issues.sh # Add README-alignment tasks
│     ├─ 11_add_external_dependencies_and_research_tasks.sh # Add external/research tasks
│     ├─ 12_add_model_training_ml_indicator_tasks.sh # Add ML, W&B and indicator tasks
│     └─ 13_add_slow_fast_layer_alignment_tasks.sh # Add slow/fast layer alignment tasks
│
├─ external/                                # External repositories and pinned dependency references
│  ├─ README.md                             # External dependency policy and usage notes
│  └─ external-repos.lock                   # URLs, roles and pinned commits/tags for external repos
│
├─ system/                                  # Runnable DSS system and deployment target
│  ├─ README.md                             # System-specific run and architecture notes
│  ├─ docker-compose.yml                    # Main local Docker Compose entrypoint
│  ├─ .env                                  # Local environment file; ignored by Git
│  ├─ .env.example                          # System-level environment template
│  │
│  ├─ runtime-data/                         # Local runtime data; ignored by Git
│  │  ├─ market_research.duckdb             # Local DuckDB development database
│  │  ├─ logs/                              # Local application logs
│  │  └─ tmp/                               # Temporary runtime files
│  │
│  ├─ frontend/                             # Web client / DSS demo UI
│  │  ├─ Dockerfile                         # Frontend container build
│  │  ├─ nginx.conf                         # Frontend web server configuration
│  │  └─ src/                               # Frontend source code
│  │     ├─ Frontend.csproj                 # Blazor project file
│  │     ├─ Program.cs                      # Frontend application startup
│  │     ├─ App.razor                       # Root Blazor component
│  │     ├─ _Imports.razor                  # Shared Razor imports
│  │     ├─ wwwroot/                        # Static frontend assets
│  │     │  ├─ index.html                   # Frontend host page
│  │     │  └─ appsettings.json             # Frontend runtime settings
│  │     ├─ Layout/                         # Page layouts
│  │     ├─ Pages/                          # Application pages
│  │     ├─ Components/                     # Reusable UI components
│  │     │  ├─ Login/                       # Login and demo-user UI
│  │     │  ├─ Portfolio/                   # Portfolio builder UI
│  │     │  ├─ Strategy/                    # Strategy builder UI
│  │     │  ├─ DecisionCards/               # Buy/hold/sell decision cards
│  │     │  ├─ Risk/                        # Risk indicator UI
│  │     │  └─ Charts/                      # Charts and thesis/demo visualizations
│  │     ├─ Services/                       # Frontend service clients
│  │     │  ├─ ApiClient.cs                 # REST API client
│  │     │  └─ GraphQLClientService.cs      # GraphQL client service
│  │     └─ Models/                         # Frontend DTOs and view models
│  │
│  ├─ backend/                              # FastAPI backend / DSS API
│  │  ├─ Dockerfile                         # Backend container build
│  │  ├─ requirements.txt                   # Python backend dependencies
│  │  ├─ app/                               # Backend application code
│  │  │  ├─ main.py                         # FastAPI application entrypoint
│  │  │  ├─ config.py                       # Backend configuration
│  │  │  ├─ db.py                           # DuckDB connection handling
│  │  │  ├─ logging_config.py               # Logging configuration
│  │  │  │
│  │  │  ├─ api/                            # REST API routes
│  │  │  │  ├─ routes_health.py             # Health check endpoint
│  │  │  │  ├─ routes_auth.py               # Login/demo user endpoints
│  │  │  │  ├─ routes_prices.py             # Price and market-data endpoints
│  │  │  │  ├─ routes_stocks.py             # Stock lookup endpoints
│  │  │  │  ├─ routes_news.py               # News / external signal endpoints
│  │  │  │  ├─ routes_strategies.py         # Strategy endpoints
│  │  │  │  ├─ routes_portfolio.py          # Portfolio endpoints
│  │  │  │  ├─ routes_decisions.py          # Decision-support endpoints
│  │  │  │  ├─ routes_risk.py               # Risk metric endpoints
│  │  │  │  ├─ routes_audit.py              # Audit log endpoints
│  │  │  │  ├─ routes_backtests.py          # Backtest endpoints
│  │  │  │  ├─ routes_experiments.py        # Experiment metadata endpoints
│  │  │  │  └─ routes_predictions.py        # Prediction/model-output endpoints
│  │  │  │
│  │  │  ├─ graphql/                        # Optional GraphQL layer
│  │  │  │  ├─ schema.py                    # GraphQL schema
│  │  │  │  ├─ queries.py                   # GraphQL queries
│  │  │  │  └─ mutations.py                 # GraphQL mutations
│  │  │  │
│  │  │  ├─ services/                       # Backend business logic
│  │  │  │  ├─ auth_service.py              # Demo login/user logic
│  │  │  │  ├─ market_service.py            # Market-data logic
│  │  │  │  ├─ stock_lookup_service.py      # Stock search/lookup logic
│  │  │  │  ├─ news_service.py              # News/external signal logic
│  │  │  │  ├─ strategy_service.py          # Strategy handling logic
│  │  │  │  ├─ portfolio_service.py         # Portfolio logic
│  │  │  │  ├─ decision_service.py          # Buy/hold/sell decision logic
│  │  │  │  ├─ risk_service.py              # Risk metric calculation logic
│  │  │  │  ├─ audit_service.py             # Decision audit logging logic
│  │  │  │  ├─ backtest_service.py          # Backtest orchestration logic
│  │  │  │  ├─ experiment_service.py        # Experiment metadata logic
│  │  │  │  └─ prediction_service.py        # Model prediction/inference logic
│  │  │  │
│  │  │  ├─ repositories/                   # Database access layer
│  │  │  │  ├─ user_repository.py           # User/demo-user queries
│  │  │  │  ├─ market_repository.py         # Market-data queries
│  │  │  │  ├─ raw_data_repository.py       # Raw API response metadata queries
│  │  │  │  ├─ feature_repository.py        # Feature table queries
│  │  │  │  ├─ strategy_repository.py       # Strategy queries
│  │  │  │  ├─ portfolio_repository.py      # Portfolio queries
│  │  │  │  ├─ decision_repository.py       # Decision output queries
│  │  │  │  ├─ audit_repository.py          # Audit log queries
│  │  │  │  ├─ backtest_repository.py       # Backtest result queries
│  │  │  │  ├─ experiment_repository.py     # Experiment metadata queries
│  │  │  │  └─ model_registry_repository.py # Model registry queries
│  │  │  │
│  │  │  ├─ models/                         # Backend DTO models
│  │  │  │  ├─ dto_user.py                  # User DTO
│  │  │  │  ├─ dto_market.py                # Market-data DTO
│  │  │  │  ├─ dto_stock.py                 # Stock DTO
│  │  │  │  ├─ dto_strategy.py              # Strategy DTO
│  │  │  │  ├─ dto_portfolio.py             # Portfolio DTO
│  │  │  │  ├─ dto_decision.py              # Decision DTO
│  │  │  │  ├─ dto_risk.py                  # Risk metric DTO
│  │  │  │  ├─ dto_audit.py                 # Audit log DTO
│  │  │  │  ├─ dto_backtest.py              # Backtest DTO
│  │  │  │  ├─ dto_experiment.py            # Experiment DTO
│  │  │  │  └─ dto_prediction.py            # Prediction DTO
│  │  │  │
│  │  │  └─ schemas/                        # Validation schemas
│  │  │     ├─ strategy_schema.py           # Strategy JSON schema
│  │  │     ├─ portfolio_schema.py          # Portfolio schema
│  │  │     ├─ decision_schema.py           # Decision schema
│  │  │     └─ risk_schema.py               # Risk output schema
│  │  │
│  │  └─ tests/                             # Backend tests
│  │     ├─ test_health.py                  # Health endpoint tests
│  │     ├─ test_auth.py                    # Login/demo user tests
│  │     ├─ test_prices.py                  # Price endpoint tests
│  │     ├─ test_stock_lookup.py            # Stock lookup tests
│  │     ├─ test_strategies.py              # Strategy tests
│  │     ├─ test_portfolio.py               # Portfolio tests
│  │     ├─ test_decisions.py               # Decision engine tests
│  │     ├─ test_audit.py                   # Audit log tests
│  │     └─ test_backtests.py               # Backtest tests
│  │
│  ├─ workers/                              # Background workers and batch jobs
│  │  ├─ ingestion-worker/                  # API/data ingestion worker
│  │  │  ├─ Dockerfile                      # Ingestion worker container build
│  │  │  ├─ requirements.txt                # Ingestion worker dependencies
│  │  │  ├─ worker.py                       # Ingestion worker entrypoint
│  │  │  ├─ sources/                        # External data source adapters
│  │  │  │  ├─ yfinance_source.py           # yfinance data source
│  │  │  │  ├─ finrl_source.py              # FinRL data source
│  │  │  │  ├─ gdelt_source.py              # GDELT/news placeholder source
│  │  │  │  ├─ macro_source.py              # Macro indicator placeholder source
│  │  │  │  └─ fundamentals_source.py       # Company fundamentals placeholder source
│  │  │  └─ tests/                          # Ingestion worker tests
│  │  │
│  │  ├─ feature-worker/                    # Feature engineering worker
│  │  │  ├─ Dockerfile                      # Feature worker container build
│  │  │  ├─ requirements.txt                # Feature worker dependencies
│  │  │  ├─ worker.py                       # Feature worker entrypoint
│  │  │  ├─ features/                       # Feature generation code
│  │  │  │  ├─ technical_indicators.py      # Technical indicators
│  │  │  │  ├─ returns.py                   # Return calculations
│  │  │  │  ├─ volatility.py                # Volatility features
│  │  │  │  ├─ downside_risk.py             # Downside-risk features
│  │  │  │  └─ feature_store.py             # Feature storage logic
│  │  │  └─ tests/                          # Feature worker tests
│  │  │
│  │  ├─ strategy-worker/                   # Strategy validation and transformation worker
│  │  │  ├─ Dockerfile                      # Strategy worker container build
│  │  │  ├─ requirements.txt                # Strategy worker dependencies
│  │  │  ├─ worker.py                       # Strategy worker entrypoint
│  │  │  ├─ strategies/                     # Strategy logic
│  │  │  │  ├─ predefined.py                # Predefined strategy library
│  │  │  │  ├─ custom_strategy.py           # Custom strategy handling
│  │  │  │  ├─ constraints.py               # Locked assets and user constraints
│  │  │  │  └─ strategy_validation.py       # Strategy validation
│  │  │  └─ tests/                          # Strategy worker tests
│  │  │
│  │  ├─ decision-worker/                   # Decision-support worker
│  │  │  ├─ Dockerfile                      # Decision worker container build
│  │  │  ├─ requirements.txt                # Decision worker dependencies
│  │  │  ├─ worker.py                       # Decision worker entrypoint
│  │  │  ├─ decision/                       # Decision engine code
│  │  │  │  ├─ decision_engine.py           # Main decision engine
│  │  │  │  ├─ alternatives.py              # Decision alternative generation
│  │  │  │  ├─ transaction_costs.py         # Transaction cost penalty logic
│  │  │  │  ├─ stop_loss.py                 # Stop-loss trigger logic
│  │  │  │  ├─ take_profit.py               # Take-profit trigger logic
│  │  │  │  └─ audit_writer.py              # Decision audit writer
│  │  │  └─ tests/                          # Decision worker tests
│  │  │
│  │  ├─ finrl-worker/                      # FinRL / RL training and evaluation worker
│  │  │  ├─ Dockerfile                      # FinRL worker container build
│  │  │  ├─ requirements.txt                # FinRL worker dependencies
│  │  │  ├─ train.py                        # Train FinRL/RL agents
│  │  │  ├─ evaluate.py                     # Evaluate trained agents
│  │  │  ├─ configs/                        # Training and evaluation configs
│  │  │  ├─ agents/                         # Agent implementations or adapters
│  │  │  ├─ baselines/                      # Baseline strategy implementations
│  │  │  │  ├─ buy_and_hold.py              # Buy-and-hold baseline
│  │  │  │  ├─ equal_weight.py              # Equal-weight baseline
│  │  │  │  └─ risk_adjusted.py             # Risk-adjusted baseline/proxy
│  │  │  └─ tests/                          # FinRL worker tests
│  │  │
│  │  └─ training-job/                      # Generic offline training job container
│  │     ├─ Dockerfile                      # Training job container build
│  │     ├─ requirements.txt                # Training job dependencies
│  │     ├─ run_training_job.py             # Generic training job entrypoint
│  │     └─ configs/                        # Training job configuration files
│  │
│  ├─ packages/                             # Shared internal Python packages
│  │  ├─ dss-core/                          # Shared domain models and utilities
│  │  │  ├─ pyproject.toml                  # Package metadata
│  │  │  └─ dss_core/                       # dss-core package source
│  │  │     ├─ domain/                      # Shared domain objects
│  │  │     ├─ types/                       # Shared types
│  │  │     ├─ errors/                      # Shared exceptions
│  │  │     └─ utils/                       # Shared utilities
│  │  │
│  │  ├─ dss-data/                          # Shared data access and PIT utilities
│  │  │  ├─ pyproject.toml                  # Package metadata
│  │  │  └─ dss_data/                       # dss-data package source
│  │  │     ├─ duckdb_client.py             # DuckDB client wrapper
│  │  │     ├─ point_in_time.py             # Point-in-time query helpers
│  │  │     ├─ raw_storage.py               # Raw API response storage
│  │  │     ├─ parquet_layout.py            # Parquet folder layout helpers
│  │  │     └─ snapshot_logging.py          # Dataset build and snapshot logging
│  │  │
│  │  ├─ dss-env/                           # Custom Gymnasium trading environment
│  │  │  ├─ pyproject.toml                  # Package metadata
│  │  │  └─ dss_env/                        # dss-env package source
│  │  │     ├─ stock_investment_env.py      # Long-only StockInvestmentEnv
│  │  │     ├─ state.py                     # State representation
│  │  │     ├─ actions.py                   # Action space definition
│  │  │     ├─ rewards.py                   # Reward function definitions
│  │  │     └─ validation.py                # Environment validation helpers
│  │  │
│  │  ├─ dss-strategies/                    # Strategy schema and validation package
│  │  │  ├─ pyproject.toml                  # Package metadata
│  │  │  └─ dss_strategies/                 # dss-strategies package source
│  │  │     ├─ predefined.py                # Predefined strategy profiles
│  │  │     ├─ schema.py                    # Strategy JSON schema
│  │  │     ├─ constraints.py               # Constraint logic
│  │  │     └─ validation.py                # Strategy validation helpers
│  │  │
│  │  ├─ dss-risk/                          # Risk and uncertainty utility package
│  │  │  ├─ pyproject.toml                  # Package metadata
│  │  │  └─ dss_risk/                       # dss-risk package source
│  │  │     ├─ sharpe.py                    # Sharpe-style metrics
│  │  │     ├─ drawdown.py                  # Drawdown metrics
│  │  │     ├─ var_cvar.py                  # VaR / CVaR helpers
│  │  │     ├─ quantiles.py                 # Quantile-risk helpers
│  │  │     └─ uncertainty.py               # Uncertainty proxy helpers
│  │  │
│  │  └─ dss-finrl-adapter/                 # FinRL integration adapter package
│  │     ├─ pyproject.toml                  # Package metadata
│  │     └─ dss_finrl_adapter/              # dss-finrl-adapter package source
│  │        ├─ data_adapter.py              # FinRL data adapter
│  │        ├─ env_adapter.py               # FinRL/Gymnasium environment adapter
│  │        ├─ agent_adapter.py             # FinRL agent adapter
│  │        └─ metrics_adapter.py           # FinRL metrics adapter
│  │
│  ├─ sql/                                  # DuckDB schema, migrations and data tests
│  │  ├─ schemas/                           # SQL schema files
│  │  │  ├─ 001_raw_market_data.sql         # Raw market data schema
│  │  │  ├─ 002_point_in_time_metadata.sql  # Point-in-time metadata schema
│  │  │  ├─ 003_features.sql                # Feature table schema
│  │  │  ├─ 004_portfolios.sql              # Portfolio table schema
│  │  │  ├─ 005_strategies.sql              # Strategy table schema
│  │  │  ├─ 006_decisions.sql               # Decision output schema
│  │  │  ├─ 007_audit_log.sql               # Audit log schema
│  │  │  ├─ 008_backtests.sql               # Backtest result schema
│  │  │  ├─ 009_experiments.sql             # Experiment metadata schema
│  │  │  └─ 010_model_registry.sql          # Model registry schema
│  │  │
│  │  ├─ migrations/                        # DuckDB migration scripts
│  │  ├─ seeds/                             # Demo seed data
│  │  └─ tests/                             # SQL data quality tests
│  │     ├─ test_no_lookahead.sql           # Test against look-ahead leakage
│  │     ├─ test_primary_keys.sql           # Primary key and uniqueness tests
│  │     └─ test_required_columns.sql       # Required column tests
│  │
│  ├─ configs/                              # Environment-specific configuration
│  │  ├─ local.yaml                         # Local development config
│  │  ├─ test.yaml                          # k3s test environment config
│  │  ├─ staging.yaml                       # Staging config
│  │  ├─ prod.yaml                          # Production/demo config
│  │  └─ gpu.yaml                           # GPU training config
│  │
│  ├─ infra/                                # System infrastructure definitions
│  │  ├─ docker/                            # Docker build assets
│  │  │  ├─ backend.Dockerfile              # Backend Dockerfile variant
│  │  │  ├─ frontend.Dockerfile             # Frontend Dockerfile variant
│  │  │  ├─ worker.Dockerfile               # Generic worker Dockerfile
│  │  │  └─ training.Dockerfile             # Training job Dockerfile
│  │  │
│  │  ├─ compose/                           # Docker Compose files
│  │  │  ├─ docker-compose.local.yml        # Local development compose setup
│  │  │  ├─ docker-compose.test.yml         # Test compose setup
│  │  │  └─ docker-compose.gpu.yml          # GPU training compose setup
│  │  │
│  │  └─ k8s/                               # Kubernetes/k3s manifests
│  │     ├─ namespaces/                     # Namespace definitions
│  │     │  ├─ dss-dev.yaml                 # Development namespace
│  │     │  ├─ dss-test.yaml                # Test namespace
│  │     │  └─ dss-prod.yaml                # Production/demo namespace
│  │     │
│  │     ├─ base/                           # Base Kubernetes manifests
│  │     │  ├─ backend-deployment.yaml      # Backend deployment
│  │     │  ├─ frontend-deployment.yaml     # Frontend deployment
│  │     │  ├─ ingestion-worker.yaml        # Ingestion worker deployment
│  │     │  ├─ feature-worker.yaml          # Feature worker deployment
│  │     │  ├─ decision-worker.yaml         # Decision worker deployment
│  │     │  ├─ services.yaml                # Kubernetes services
│  │     │  └─ ingress.yaml                 # Ingress configuration
│  │     │
│  │     ├─ overlays/                       # Environment-specific k8s overlays
│  │     │  ├─ test/                        # Test overlay for Turing Pi/k3s
│  │     │  ├─ staging/                     # Staging overlay
│  │     │  └─ prod/                        # Production/demo overlay
│  │     │
│  │     └─ jobs/                           # Kubernetes jobs and cronjobs
│  │        ├─ ingestion-cronjob.yaml       # Scheduled ingestion job
│  │        ├─ feature-build-job.yaml       # Feature build job
│  │        ├─ backtest-job.yaml            # Backtest job
│  │        └─ training-job.yaml            # RL training job
│  │
│  └─ scripts/                              # Local scripts and operational commands
│     ├─ dev_start.sh                       # Start local development environment
│     ├─ build_demo_db.py                   # Build local demo DuckDB database
│     ├─ ingest_market_data.py              # Ingest market data
│     ├─ build_features.py                  # Build features
│     ├─ run_backtest.py                    # Run backtest
│     ├─ export_results.py                  # Export thesis-ready results
│     └─ smoke_test.py                      # Run local smoke test
│
├─ research/                                # Academic experiments, evaluation and thesis work
│  ├─ README.md                             # Research workspace overview
│  │
│  ├─ notebooks/                            # Research notebooks
│  │  ├─ README.md                          # Notebook index, purpose, inputs and outputs
│  │  ├─ 00_data_check.ipynb                # Data availability and sanity checks
│  │  ├─ 01_slow_layer_finrl_baseline.ipynb # Slow-layer FinRL baseline experiment
│  │  ├─ 02_slow_layer_gymnasium_env.ipynb  # Slow-layer custom Gymnasium environment
│  │  ├─ 03_slow_layer_baseline_comparison.ipynb # Buy-and-hold/equal-weight/risk baseline comparison
│  │  ├─ 04_fast_layer_decision_case.ipynb  # Fast-layer decision-support case
│  │  ├─ 05_fast_layer_strategy_constraints.ipynb # Strategy constraint effects
│  │  ├─ 06_fast_layer_audit_trace.ipynb    # Decision audit trace and reproducibility
│  │  ├─ 07_iqn_extension.ipynb             # IQN/distributional RL extension
│  │  ├─ 08_uncertainty_proxy.ipynb         # Uncertainty proxy / evidential extension
│  │  └─ 09_thesis_figures.ipynb            # Thesis figures and tables
│  │
│  ├─ experiments/                          # Reproducible research experiments
│  │  ├─ slow_layer/                        # Offline training, backtesting and evaluation
│  │  │  ├─ finrl_baseline/                 # FinRL baseline experiment
│  │  │  ├─ buy_and_hold/                   # Buy-and-hold baseline
│  │  │  ├─ equal_weight/                   # Equal-weight baseline
│  │  │  ├─ risk_adjusted_baseline/         # Risk-adjusted comparison
│  │  │  ├─ hyperparameter_baseline/        # Minimal V1.0 hyperparameter config
│  │  │  └─ model_registry_export/          # Export slow-layer outputs to registry
│  │  │
│  │  ├─ fast_layer/                        # Near real-time decision-support experiments
│  │  │  ├─ decision_case/                  # User-facing decision case
│  │  │  ├─ strategy_constraints/           # Constraint-based decision alternatives
│  │  │  ├─ risk_output/                    # Risk summary output
│  │  │  └─ audit_reproducibility/          # Reconstructable decision trace
│  │  │
│  │  ├─ iqn/                               # IQN/distributional RL extension
│  │  ├─ uncertainty/                       # Uncertainty-aware extension
│  │  └─ robustness/                        # Robustness and sensitivity experiments
│  │
│  ├─ configs/                              # Research experiment configs
│  │  ├─ experiment_001_finrl_baseline.yaml # FinRL baseline config
│  │  ├─ experiment_002_baselines.yaml      # Baseline comparison config
│  │  ├─ experiment_003_risk_adjusted.yaml  # Risk-adjusted baseline config
│  │  ├─ experiment_004_iqn_extension.yaml  # IQN extension config
│  │  └─ experiment_005_baseline_training.yaml # Minimal ML/Deep RL baseline training config
│  │
│  ├─ results/                              # Research outputs
│  │  ├─ raw/                               # Raw experiment outputs
│  │  ├─ processed/                         # Processed experiment outputs
│  │  ├─ tables/                            # Thesis-ready tables
│  │  ├─ figures/                           # Thesis-ready figures
│  │  └─ logs/                              # Experiment logs
│  │
│  ├─ report/                               # Thesis LaTeX report
│  │  ├─ report.tex                         # Main LaTeX report file
│  │  ├─ references.bib                     # BibTeX reference file
│  │  ├─ sections/                          # Report sections
│  │  │  ├─ 01_introduction.tex             # Introduction
│  │  │  ├─ 02_background.tex               # Background and related work
│  │  │  ├─ 03_system_design.tex            # System design
│  │  │  ├─ 04_methodology.tex              # Methodology
│  │  │  ├─ 05_results.tex                  # Results / case demonstration
│  │  │  ├─ 06_discussion.tex               # Discussion and limitations
│  │  │  └─ 07_conclusion.tex               # Conclusion and future work
│  │  ├─ figures/                           # Report figure exports
│  │  ├─ tables/                            # Report table exports
│  │  └─ build.sh                           # Reproducible report build command
│  │
│  └─ papers/                               # Local paper notes only; PDFs ignored by Git
│     └─ README.md                          # Paper/reference notes
│
├─ data/                                    # Data documentation only; large data is external
│  └─ README.md                             # Explains that runtime data lives on guldNAS/runtime-data
│
└─ .github/                                 # GitHub automation
   ├─ workflows/                            # GitHub Actions workflows
   │  ├─ ci.yml                             # Main CI workflow
   │  ├─ python-tests.yml                   # Python test workflow
   │  ├─ docker-build.yml                   # Docker build workflow
   │  ├─ docker-compose-smoke-test.yml      # Docker Compose smoke test workflow
   │  ├─ build-and-push.yml                 # Build and push container images
   │  ├─ deploy-test.yml                    # Deploy to test/k3s environment
   │  ├─ deploy-prod.yml                    # Deploy to production/demo environment
   │  ├─ release.yml                        # Release tagging/versioning workflow
   │  ├─ dependency-scan.yml                # Dependency/security scanning workflow
   │  ├─ project-automation-on-issue-closed.yml # Project field automation for closed/reopened issues
   │
   └─ dependabot.yml                        # Dependabot dependency update configuration
```

---

## External Dependencies Principle

External repositories should not be pulled blindly from `main` or `master` during every Docker build.

External dependencies should be documented and pinned by commit, tag or version when used.

Relevant external repositories include:

| Repository | Role |
|---|---|
| FinRL | Financial RL framework |
| Gymnasium | Environment interface |
| ObjectRL | RL prototyping/reference |
| SDU_DataScienceTool | Existing data/API utility |
| zero-sum-public | Frontend/charting reference |

The intended location for external dependency documentation is:

```text
external/
├─ README.md
└─ external-repos.lock
```

For V1.0, external repositories may be documented and pinned before they are cloned or installed. The PoC should remain runnable even if optional external dependencies are deferred.

---

## Canonical Storage Layout

Runtime data should not live permanently inside the Git repository.

Canonical persistent storage is expected to live on guldNAS:

```text
/mnt/nas/stockinvestmentdss/
├── duckdb/
│   └── market_research.duckdb              # Canonical query database
│
├── parquet/
│   ├── raw/                                # Raw normalized tabular data
│   ├── curated/                            # Cleaned and validated datasets
│   ├── features/                           # Feature-engineering outputs
│   ├── backtests/                          # Backtest-ready analytical datasets
│   └── exports/                            # Parquet exports for reproducible experiments
│
├── csv/
│   ├── raw/                                # CSV mirrors of selected raw tabular data
│   ├── curated/                            # Human-readable curated datasets
│   ├── features/                           # Human-readable feature outputs
│   ├── backtests/                          # Backtest result exports
│   ├── thesis-tables/                      # CSV tables used directly in the thesis
│   └── debug/                              # Temporary CSV exports for inspection/debugging
│
├── raw-api-responses/
│   ├── yfinance/                           # Original yfinance API responses
│   ├── finrl/                              # Original FinRL-related downloaded data
│   ├── gdelt/                              # Original news/event API responses
│   ├── macro/                              # Original macro indicator responses
│   └── fundamentals/                       # Original company fundamentals responses
│
├── model-checkpoints/
│   ├── finrl/                              # FinRL model checkpoints
│   ├── iqn/                                # IQN/distributional RL checkpoints
│   └── baselines/                          # Baseline model/config artifacts
│
├── backtest-results/
│   ├── raw/                                # Raw backtest outputs
│   ├── processed/                          # Processed backtest results
│   └── comparisons/                        # Baseline comparison outputs
│
├── experiment-artifacts/
│   ├── configs/                            # Frozen experiment configs
│   ├── logs/                               # Experiment logs
│   ├── metrics/                            # Experiment metrics
│   └── figures/                            # Experiment-generated figures
│
├── reports/
│   ├── tables/                             # Thesis-ready result tables
│   ├── figures/                            # Thesis-ready figures
│   └── exports/                            # Final report/evidence exports
│
└── logs/
    ├── backend/                            # Backend logs
    ├── workers/                            # Worker logs
    ├── ingestion/                          # Ingestion logs
    ├── training/                           # Training logs
    └── audit/                              # Decision audit logs
```

Local development may use:

```text
system/runtime-data/market_research.duckdb
system/runtime-data/parquet/
system/runtime-data/csv/
system/runtime-data/raw-api-responses/
```

but the long-term canonical path should be:

```text
/mnt/nas/stockinvestmentdss/duckdb/market_research.duckdb
/mnt/nas/stockinvestmentdss/parquet/
/mnt/nas/stockinvestmentdss/csv/
/mnt/nas/stockinvestmentdss/raw-api-responses/
```
### Storage Format Principle

The same logical dataset may exist in multiple physical formats because each format serves a different purpose.

| Format | Purpose |
|---|---|
| Raw API responses | Preserve exactly what the system received from external sources |
| DuckDB | Canonical queryable analytical database |
| Parquet | Efficient columnar format for ML, feature pipelines and reproducible experiments |
| CSV | Human-readable exports for debugging, inspection, thesis tables and external review |

DuckDB should be treated as the primary analytical store.

Parquet should be treated as the reproducible data pipeline format.

CSV should be treated as an export/debug/reporting format, not the primary source of truth.


---

## Point-in-Time Data Principle

The system must distinguish between:

```text
event_time     = when the market/company event happened
published_at   = when the information became publicly available
ingested_at    = when this system received/stored the information
decision_time  = when the DSS generated a recommendation
```

For any historical decision at time `t`, the system may only use data where:

```text
known_at <= t
```

This is required to avoid look-ahead bias.

---

## Decision Audit Principle

Every generated recommendation should be reproducible.

A decision should be linked to:

- user / portfolio
- selected strategy
- decision timestamp
- available data snapshot
- feature build ID
- model ID or rule engine version
- input state
- generated alternatives
- risk metrics
- final recommendation
- explanation / warning
- audit log entry

---

## Strategy Layer

Strategies are represented as structured configuration rather than free text.

Example:

```json
{
  "strategy_name": "Balanced long-only strategy",
  "risk_profile": "balanced",
  "investment_horizon": "medium",
  "allowed_assets": ["AAPL", "MSFT", "NVDA"],
  "locked_assets": ["NVDA"],
  "max_position_size": 0.25,
  "stop_loss": 0.10,
  "take_profit": 0.25,
  "objective": "risk_adjusted_return",
  "allow_strategy_switch": true
}
```

The strategy layer is used by both:

- the online decision engine
- the offline evaluation/backtesting pipeline

---

## Reinforcement Learning Scope

The project uses RL as one component inside a broader DSS.

The intended progression is:

```text
1. Simple baseline decision logic
2. Buy-and-hold and equal-weight baselines
3. Standard FinRL agent
4. Custom Gymnasium environment
5. Risk-aware output
6. IQN-style quantile output
7. Evidential uncertainty proxy
8. Full IQN / uncertainty-aware extension if time permits
```

For V1.0, the most important outcome is a functioning decision-support pipeline.

Advanced IQN and evidential uncertainty can be treated as extensions if needed.

---

## V1.0 Definition of Done

The V1.0 PoC is considered successful if the following works end-to-end:

- local app starts
- front page loads
- login or demo user flow works
- user can search or select a stock
- user can create a portfolio
- user can define or select a strategy
- system can ingest market data
- market data is stored in DuckDB
- point-in-time metadata is represented
- decision engine can generate buy/hold/sell alternatives
- decision output includes risk information
- audit log is written
- results can be exported for thesis figures/tables
- README explains how to run and understand the PoC

---

## Recommended Implementation Order

```text
1. Repository structure
2. README4.md
3. .env.example
4. guldNAS folder structure
5. DuckDB canonical path
6. Dockerfile for backend
7. docker-compose.yml for local PoC
8. backend health endpoint
9. minimal frontend shell
10. local front page + login smoke test
11. DuckDB schema
12. yfinance / FinRL data ingestion
13. raw file storage
14. point-in-time ingestion schema
15. portfolio builder
16. strategy JSON schema
17. strategy builder
18. decision engine v1
19. risk metrics
20. audit log
21. baseline comparison
22. thesis figures
23. final demo script
```

---

## Local Development

From the repository root:

```bash
cd system
cp .env.example .env
docker compose up --build
```

Expected local services:

```text
Frontend: http://localhost:3000
Backend:  http://localhost:8000
API docs: http://localhost:8000/docs
```

Health check:

```bash
curl http://localhost:8000/health
```

---

## Environment Files

`.env` files must not be committed.

Use:

```text
.env.example
```

for documented variables.

Expected environment groups:

```text
APP_ENV=local
DUCKDB_PATH=./runtime-data/market_research.duckdb
RAW_DATA_PATH=./runtime-data/raw
PARQUET_PATH=./runtime-data/parquet
MODEL_REGISTRY_PATH=./runtime-data/models
LOG_LEVEL=INFO
```

For guldNAS:

```text
DUCKDB_PATH=/mnt/nas/stockinvestmentdss/duckdb/market_research.duckdb
RAW_DATA_PATH=/mnt/nas/stockinvestmentdss/raw-api-responses
PARQUET_PATH=/mnt/nas/stockinvestmentdss/parquet
MODEL_REGISTRY_PATH=/mnt/nas/stockinvestmentdss/model-checkpoints
```

---

## Kubernetes / k3s Target

The Turing Pi k3s cluster is intended for the test/demo platform.

Planned node roles:

| Node | Role |
|---|---|
| node1 | control-plane |
| node2 | ingestion / scheduler |
| node3 | feature / strategy |
| node4 | tracking / monitoring / lightweight RL |

Recommended deployment order:

```text
1. Verify all nodes
2. Assign stable hostnames and IPs
3. Install k3s on node1
4. Join node2-node4
5. Deploy test namespace
6. Deploy hello-world API
7. Mount guldNAS
8. Deploy dss-api
9. Deploy ingestion-worker
10. Deploy feature-worker
11. Deploy decision-worker
12. Deploy FinRL/ObjectRL worker later
```

---

## CI/CD Target

GitHub Actions should eventually cover:

- Python tests
- linting
- Docker image builds
- Docker Compose smoke test
- dependency scanning
- build-and-push workflow
- test deployment
- production/demo deployment
- rollback gate
- release tagging

For V1.0, the minimum CI requirement is:

```text
backend tests + Docker build + local smoke test
```

---

## Report Integration

The PoC should produce outputs that can be used directly in the thesis:

```text
research/results/tables/
research/results/figures/
research/report/figures/
research/report/tables/
```

Expected thesis artifacts:

- system architecture diagram
- data pipeline diagram
- slow/fast layer diagram
- point-in-time data model
- decision audit example
- baseline comparison table
- risk metric table
- demo screenshots

---

## Security and Privacy

The repository must not include:

- real `.env` files
- API keys
- GitHub tokens
- private SSH keys
- large raw datasets
- model checkpoints
- private user data
- production secrets

Use `.env.example`, GitHub Actions secrets and local ignored files.

---

## Current Project Board

The GitHub Project board is used as the operational sprint board for V1.0 PoC development.

Main views:

- Backlog
- Status
- Roadmap
- Priority
- Category
- Board
- Native

Main categories:

- Development
- Data
- Architecture
- RL / AI
- Evaluation
- Report

Main status flow:

```text
Todo → In Progress → Code Review → Done
```

Main roadmap flow:

```text
Now → Next → Later
```

### Project Board Automation

The GitHub Project board is partly automated.

When an issue is closed, the workflow should:

```text
Percentage = 100%
Roadmap    = cleared
```

When an issue is reopened, the workflow should:

```text
Status = Todo
```

The automation is implemented in:

```text
.github/workflows/project-automation-on-issue-closed.yml
```

The workflow uses the repository secret:

```text
PROJECT_PAT
```

---

## Repository Rules

```text
docs/      = architecture, infrastructure and project documentation
external/  = external dependency documentation and pinned repo references
system/    = runnable DSS application and deployment assets
research/  = thesis experiments, notebooks, report and evaluation
.github/   = CI/CD automation
data/      = documentation only; large data is external
```

Do not commit large runtime files.

Do not commit secrets.

Do not use `system/` as the repository root.

---

## Current Status

This repository is in active thesis PoC development.

The immediate focus is:

```text
1. create the V1.0 project skeleton
2. define local Docker setup
3. define DuckDB storage
4. define external dependency policy
5. verify FinRL / Gymnasium / research environment
6. build a minimal app shell
7. implement data ingestion
8. implement strategy and portfolio flow
9. generate first decision-support output
10. export thesis-ready evidence
```

---

## License

This project is currently private thesis work unless otherwise specified.

---

## Disclaimer

This project is for academic research and educational purposes only.

It does not provide financial advice.

Any generated buy/hold/sell output is part of a research prototype and must not be interpreted as investment advice.