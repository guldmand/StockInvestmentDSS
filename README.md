# StockInvestmentDSS

## Project Structure

<pre>
system/                                # Hele projektroden
├─ docker-compose.yml                  # Container-orkestrering
├─ .env                                # Miljøvariabler
├─ .gitignore                          # Ignorerede filer
├─ README.md                           # Projektoversigt
│
├─ runtime-data/                       # Lokal runtime-data
│  └─ market_research.duckdb           # Central DuckDB-fil
│
├─ .github/                            # GitHub automation
│  └─ workflows/                       # CI/CD workflows
│     ├─ ci.yml                        # Tests og checks
│     ├─ build-and-push.yml            # Build og push
│     ├─ deploy.yml                    # Deployment workflow
│     └─ build-demo-db.yml             # Demo-db build
│
├─ frontend/                           # Webklient
│  ├─ Dockerfile                       # Frontend container build
│  ├─ nginx.conf                       # Webserver konfiguration
│  └─ src/                             # Blazor kildekode
│     ├─ Frontend.csproj               # Projektfil
│     ├─ Program.cs                    # App startup
│     ├─ App.razor                     # Root component
│     ├─ _Imports.razor                # Razor imports
│     ├─ wwwroot/                      # Statiske filer
│     │  ├─ index.html                 # Hovedside
│     │  └─ appsettings.json           # Frontend settings
│     ├─ Layout/                       # Side-layouts
│     ├─ Pages/                        # Sider
│     ├─ Components/                   # UI-komponenter
│     ├─ Services/                     # Frontend services
│     │  ├─ ApiClient.cs               # REST klient
│     │  └─ GraphQLClientService.cs    # GraphQL klient
│     └─ Models/                       # Frontend modeller
│
├─ backend/                            # API og datalag
│  ├─ Dockerfile                       # Backend container build
│  ├─ requirements.txt                 # Python dependencies
│  ├─ app/                             # Backend applikation
│  │  ├─ main.py                       # Backend entrypoint
│  │  ├─ config.py                     # Backend konfiguration
│  │  ├─ db.py                         # DuckDB adgang
│  │  ├─ logging_config.py             # Logging opsætning
│  │  │
│  │  ├─ api/                          # REST endpoints
│  │  │  ├─ routes_health.py           # Health endpoint
│  │  │  ├─ routes_prices.py           # Pris endpoints
│  │  │  ├─ routes_news.py             # Nyheds endpoints
│  │  │  ├─ routes_strategies.py       # Strategi endpoints
│  │  │  ├─ routes_feature_flags.py    # Feature flag endpoints
│  │  │  ├─ routes_portfolio.py        # Portefølje endpoints
│  │  │  ├─ routes_backtests.py        # Backtest endpoints
│  │  │  ├─ routes_experiments.py      # Eksperiment endpoints
│  │  │  └─ routes_predictions.py      # Prediction endpoints
│  │  │
│  │  ├─ graphql/                      # GraphQL lag
│  │  │  ├─ schema.py                  # GraphQL schema
│  │  │  ├─ queries.py                 # GraphQL queries
│  │  │  └─ mutations.py               # GraphQL mutations
│  │  │
│  │  ├─ services/                     # Forretningslogik
│  │  │  ├─ market_service.py          # Markedslogik
│  │  │  ├─ news_service.py            # Nyhedslogik
│  │  │  ├─ strategy_service.py        # Strategilogik
│  │  │  ├─ feature_flag_service.py    # Flaglogik
│  │  │  ├─ portfolio_service.py       # Porteføljelogik
│  │  │  ├─ backtest_service.py        # Backtestlogik
│  │  │  ├─ experiment_service.py      # Eksperimentlogik
│  │  │  └─ prediction_service.py      # Predictionlogik
│  │  │
│  │  ├─ repositories/                 # Database queries
│  │  │  ├─ market_repository.py       # Markedsqueries
│  │  │  ├─ news_repository.py         # Nyhedsqueries
│  │  │  ├─ strategy_repository.py     # Strategiqueries
│  │  │  ├─ feature_flag_repository.py # Flagqueries
│  │  │  ├─ portfolio_repository.py    # Porteføljequeries
│  │  │  ├─ backtest_repository.py     # Backtestqueries
│  │  │  ├─ experiment_repository.py   # Eksperimentqueries
│  │  │  └─ prediction_repository.py   # Predictionqueries
│  │  │
│  │  └─ models/                       # DTO modeller
│  │     ├─ dto_market.py              # Markeds DTO
│  │     ├─ dto_news.py                # Nyheds DTO
│  │     ├─ dto_strategy.py            # Strategi DTO
│  │     ├─ dto_feature_flag.py        # Flag DTO
│  │     ├─ dto_backtest.py            # Backtest DTO
│  │     ├─ dto_experiment.py          # Eksperiment DTO
│  │     └─ dto_prediction.py          # Prediction DTO
│  │
│  └─ tests/                           # Backend tests
│     ├─ test_health.py                # Health tests
│     ├─ test_prices.py                # Pris tests
│     ├─ test_news.py                  # Nyheds tests
│     ├─ test_feature_flags.py         # Flag tests
│     ├─ test_backtests.py             # Backtest tests
│     └─ test_experiments.py           # Eksperiment tests
│
├─ finrl-worker/                       # RL worker
│  ├─ Dockerfile                       # Worker container build
│  ├─ requirements.txt                 # Worker dependencies
│  ├─ worker/                          # Worker kode
│  │  ├─ main.py                       # Worker entrypoint
│  │  ├─ config.py                     # Worker konfiguration
│  │  ├─ db.py                         # Worker DuckDB adgang
│  │  │
│  │  ├─ jobs/                         # Kørbare jobs
│  │  │  ├─ train_model.py             # Træn model
│  │  │  ├─ run_inference.py           # Kør inference
│  │  │  ├─ run_backtest.py            # Kør backtest
│  │  │  ├─ register_strategy.py       # Registrér strategi
│  │  │  ├─ publish_strategy.py        # Publicér strategi
│  │  │  ├─ compute_features.py        # Beregn features
│  │  │  ├─ refresh_signals.py         # Opdatér signaler
│  │  │  └─ evaluate_model.py          # Evaluer model
│  │  │
│  │  ├─ finrl/                        # FinRL integration
│  │  │  ├─ env_builder.py             # Byg miljø
│  │  │  ├─ feature_builder.py         # Byg state features
│  │  │  ├─ model_loader.py            # Indlæs modeller
│  │  │  ├─ policy_runner.py           # Kør policy
│  │  │  └─ evaluation.py              # Worker evaluering
│  │  │
│  │  ├─ uncertainty/                  # Usikkerhedsmodeller
│  │  │  ├─ iqn_wrapper.py             # IQN wrapper
│  │  │  ├─ evidential_wrapper.py      # Evidential wrapper
│  │  │  └─ risk_objectives.py         # Risikoobjektiver
│  │  │
│  │  └─ services/                     # Worker logik
│  │     ├─ training_service.py        # Træningslogik
│  │     ├─ inference_service.py       # Inferencelogik
│  │     ├─ feature_flag_service.py    # Flaglogik
│  │     ├─ strategy_registry_service.py # Strategiregister logik
│  │     ├─ experiment_tracking_service.py # Eksperimenttracking
│  │     └─ backtest_service.py        # Backtestlogik
│  │
│  └─ tests/                           # Worker tests
│     ├─ test_training.py              # Træning tests
│     ├─ test_registry.py              # Register tests
│     ├─ test_backtest.py              # Backtest tests
│     └─ test_uncertainty.py           # Usikkerhed tests
│
├─ sql/                                # Database blueprint
│  ├─ 001_schemas.sql                  # Opret schemas
│  ├─ 002_reference_tables.sql         # Reference tabeller
│  ├─ 003_market_tables.sql            # Markedstabeller
│  ├─ 004_news_tables.sql              # Nyhedstabeller
│  ├─ 005_analytics_tables.sql         # Analytics tabeller
│  ├─ 006_portfolio_tables.sql         # Porteføljetabeller
│  ├─ 007_strategy_registry.sql        # Strategiregister tabeller
│  ├─ 008_feature_flags.sql            # Feature flag tabeller
│  ├─ 009_backtesting_tables.sql       # Backtest tabeller
│  ├─ 010_experiment_tracking.sql      # Eksperiment tabeller
│  ├─ 011_views.sql                    # SQL views
│  ├─ 012_seed_feature_flags.sql       # Seed flags
│  │
│  └─ tests/                           # SQL datatests
│     ├─ test_no_negative_prices.sql   # Ingen negative priser
│     ├─ test_unique_news_ids.sql      # Unikke nyheds-id'er
│     ├─ test_feature_flags_exist.sql  # Flags findes
│     ├─ test_active_strategy_unique.sql # Aktiv strategi unik
│     ├─ test_experiment_ids_unique.sql # Eksperiment-id unikke
│     └─ test_backtest_results_not_null.sql # Backtests ikke tomme
│
├─ scripts/                            # Drifts- og datajobs
│  ├─ apply_migrations.py              # Kør migrations
│  ├─ init_db.py                       # Initialisér database
│  ├─ seed_demo_data.py                # Seed demo-data
│  ├─ ingest_prices.py                 # Importér priser
│  ├─ ingest_news.py                   # Importér nyheder
│  ├─ ingest_macro.py                  # Importér makrodata
│  ├─ ingest_company_fundamentals.py   # Importér fundamentaler
│  ├─ run_sql_tests.py                 # Kør SQL tests
│  ├─ export_snapshot.py               # Eksportér snapshot
│  ├─ import_snapshot.py               # Importér snapshot
│  ├─ register_strategy.py             # Registrér strategi
│  ├─ activate_strategy.py             # Aktivér strategi
│  ├─ build_point_in_time_dataset.py   # Byg PIT datasæt
│  ├─ run_walk_forward_backtest.py     # Kør walk-forward
│  └─ register_experiment.py           # Registrér eksperiment
│
├─ observability/                      # Logging og monitorering
│  ├─ elasticsearch/                   # Elastic opsætning
│  │  └─ README.md                     # Elastic noter
│  ├─ kibana/                          # Kibana opsætning
│  │  └─ README.md                     # Kibana noter
│  ├─ filebeat/                        # Log shipping
│  │  ├─ filebeat.yml                  # Filebeat config
│  │  └─ README.md                     # Filebeat noter
│  └─ logstash/                        # Log pipeline
│     ├─ pipeline/                     # Logstash pipelines
│     │  └─ logstash.conf              # Pipeline config
│     └─ README.md                     # Logstash noter
│
├─ models/                             # Model artifacts
│  ├─ trained/                         # Trænede modeller
│  │  ├─ .gitkeep                      # Behold mappe
│  │  └─ README.md                     # Model noter
│  ├─ exported/                        # Publicerede modeller
│  │  ├─ .gitkeep                      # Behold mappe
│  │  └─ README.md                     # Export noter
│  └─ metadata/                        # Model metadata
│     └─ .gitkeep                      # Behold mappe
│
├─ logs/                               # Lokale logs
│  ├─ backend/                         # Backend logs
│  │  └─ .gitkeep                      # Behold mappe
│  ├─ finrl-worker/                    # Worker logs
│  │  └─ .gitkeep                      # Behold mappe
│  └─ .gitkeep                         # Behold mappe
│
└─ docs/                               # Projektdokumentation
   ├─ architecture.md                  # Arkitekturguide
   ├─ deployment.md                    # Deploy guide
   ├─ data-model.md                    # Datamodel guide
   ├─ feature-flags.md                 # Flag guide
   ├─ strategy-registry.md             # Strategiguide
   ├─ observability.md                 # Logging guide
   ├─ backtesting.md                   # Backtest guide
   ├─ experiments.md                   # Eksperiment guide
   └─ evaluation.md                    # Evalueringsguide


research/
├─ notebooks/                          # Interaktive analyser
│  ├─ 01_explore_market_data.ipynb     # Udforsk markedsdata
│  ├─ 02_explore_news.ipynb            # Udforsk nyheder
│  ├─ 03_feature_engineering.ipynb     # Prototype features
│  ├─ 04_finrl_training.ipynb          # Prototype træning
│  ├─ 05_backtest_evaluation.ipynb     # Evaluér backtests
│  ├─ 06_publish_strategy.ipynb        # Publicér strategi
│  ├─ 07_distributional_rl_experiment.ipynb # Test distributional RL
│  └─ 08_uncertainty_analysis.ipynb    # Analysér usikkerhed
│
└─ research/                           # Akademisk motor
   ├─ experiments/                     # Eksperimentkode
   │  ├─ experiment_config.yaml        # Eksperiment settings
   │  ├─ experiment_runner.py          # Kør eksperimenter
   │  ├─ experiment_registry.py        # Registrér eksperimenter
   │  └─ compare_experiments.py        # Sammenlign eksperimenter
   │
   ├─ backtesting/                     # Backtesting kode
   │  ├─ walk_forward.py               # Walk-forward logik
   │  ├─ point_in_time_loader.py       # PIT dataloading
   │  ├─ portfolio_simulator.py        # Porteføljesimulator
   │  └─ evaluation_metrics.py         # Backtest metrics
   │
   ├─ feature_engineering/             # Featureudvikling
   │  ├─ price_features.py             # Prisfeatures
   │  ├─ news_features.py              # Nyhedsfeatures
   │  ├─ macro_features.py             # Makrofeatures
   │  ├─ fundamental_features.py       # Fundamentalfeatures
   │  └─ feature_pipeline.py           # Feature pipeline
   │
   ├─ uncertainty_models/              # Usikkerhedsmetoder
   │  ├─ evidential_learning.py        # Evidential læring
   │  ├─ distributional_rl_helpers.py  # Distributional helpers
   │  ├─ uncertainty_scoring.py        # Usikkerhedsscorer
   │  └─ risk_sensitive_decision_rules.py # Risiko beslutningsregler
   │
   └─ evaluation/                      # Evalueringskode
      ├─ sharpe.py                     # Sharpe beregning
      ├─ drawdown.py                   # Drawdown beregning
      ├─ downside_risk.py              # Downside risk
      └─ regime_analysis.py            # Regimeanalyse


</pre>
