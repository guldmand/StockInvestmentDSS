# The Guldmand Roadmap for IQN v4.0

FinRL data pipeline ✅
↓
Point-in-time split ✅
↓
FinRL StockTradingEnv zero/HOLD smoke test ✅
↓
FinRL StockTradingEnv trade-action smoke test ✅
↓
Continuous FinRL-compatible baseline training ✅ / suite smoke test
↓
Continuous FinRL-compatible baseline backtest ✅ / suite smoke test
↓
FinRL baseline suite ✅ / v1
  - A2C ✅
  - DDPG ✅
  - TD3 ✅
  - PPO ✅
  - SAC ✅
  - MVO ✅
↓
Discrete DSS decision layer over FinRL StockTradingEnv ✅
↓
Smart FinRL environment adapter ✅
↓
Risk-aware action resolver ✅ / v1
↓
Current portfolio state ✅ / v1
↓
Investor profile / risk_willingness ✅ / v1
↓
Investor strategy config ✅ / basic
↓
Action generator / action mask ✅ / v1
↓
Transaction / decision / audit ledger ✅ / v1
↓
Portfolio metrics over time ✅ / v1
↓
IQN estimates return distributions ✅ / smoke test
↓
IQN decision distribution table ✅ / smoke test
↓
IQN backtest / evaluation loop ✅ / technical smoke test
↓
IQN risk-aware decision selection ✅ / diagnostic v1
↓
IQN learning diagnostics ✅ / v2
↓
IQN long-period 5-seed learning experiment ✅ / earlier demo_2 long split
↓
FinRL baseline suite on same long split ✅ / earlier demo_2 long split
↓
Fair IQN vs baseline comparison summary ✅ / earlier demo_2 long split
↓
Fair comparison plots ✅ / earlier demo_2 long split
↓
Metrics comparison plots ✅ / earlier demo_2 long split
↓
IQN config + hyperparameter system ✅ / v2.6
↓
W&B experiment tracking ✅
↓
Secure W&B credential handling ✅
↓
Data mode separation ✅ / v2.8
  - Mode A live download verification ✅
  - Mode B reproducible thesis experiment ✅
↓
FinRL/yfinance data provenance ✅ / v2.8
  - FinRL YahooDownloader chunked ✅ / works, but unstable locally
  - yfinance browser-session fallback ✅ / firefox135
  - CSV import fallback ✅ / only when explicitly allowed
  - no hidden import fallback in Mode A ✅
  - frozen import/master source explicit in Mode B ✅
  - data completeness validation ✅
  - row count by ticker ✅
  - download attempts logged ✅
  - missing/failed tickers logged ✅
↓
Demo_5 Mode A download test ✅
↓
Demo_5 Mode B reproducible IQN single-seed ✅
↓
Demo_5 Mode B IQN multiseed ✅ / unstable baseline diagnosed
↓
HOLD-collapse diagnostic ✅ / v3.0
↓
LayerNorm IQN stability fix ✅ / v3.0
↓
LayerNorm robustness tests ✅ / diagnostic
↓
Demo_5 Mode B FinRL/MVO baseline multiseed ✅
↓
Demo_5 LayerNorm q50 thesis evidence package ✅
↓
Hierarchical DSS policy PoC ✅ / v3.0
↓
Combined IQN + HierarchicalDecisionPolicy audit ✅ / v3.3
↓
EDL-C teacher-imitation uncertainty track ✅ / integration evidence only
↓
Corrected EDL methodology ✅ / v3.4 design
↓
EDL-A hindsight labeler ✅ / first smoke test
↓
Counterfactual EDL-A oracle ✅ / v3.5 smoke-test complete
↓
Frozen FMP/HDP point-in-time fundamentals feature layer ✅ / v3.6
↓
Repository cleanup and reproducibility repair ✅ / v3.7
↓
Manifest-driven clean baseline infrastructure ✅ / v3.7
↓
Clean HOLD diagnostic instrumentation ✅ / v3.7
↓
Clean 25k IQN-only baseline ✅ / v3.7
↓
Clean 25k HOLD diagnosis ✅ / v3.7
↓
Clean diagnostic plots ✅ / v3.7
↓
Clean HOLD diagnostic interpretation note ✅ / v3.7
↓
Clean 25k thesis evidence package ✅ / v3.7
↓
Next thesis evidence decision ✅ / v3.7
↓
Clean 25k thesis section draft ✅ / v3.7
↓
Thesis-wide standards (apply to all etaper and TODO items):
  - all generated files use neutral, academic project documentation
  - no references to AI assistants, prompts, conversations, or model names
  - frame all decisions as technical/methodological choices with rationale
  - all artifacts must be ready for thesis submission to supervisor and examiner
↓
V1 → V2 thesis port plan ✅ / etaper 1–4 done, etaper 5–7 pending
  - V2 = canonical thesis experiment pipeline
  - V1 = visualization/design/reference implementation donor
  - one Copilot /plan per etape, no giga-plans
  - backup before each etape
  - academic repository hygiene
↓
V1 etape 1 IQN distributional decision visualizations ✅
  - 15 plot files + 3 CSV files
  - quantile function, return distribution, risk-adjusted score
  - 4-panel decision dashboard
  - 3 snapshots (seed 5 final, seed 5 mid, seed 6 collapse)
  - hybrid highlighting (chosen + risk-adjusted winner)
↓
V1 etape 2 trading_metrics canonical port ✅
  - decision: B — V2 partially covers, wrapper approach
  - new file: src/stock_investment_dss/metrics/trading_metrics.py
  - wrapper calculate_account_metrics → V2 compute_portfolio_metrics
  - V1 helpers ported: load_trade_data_single_ticker, output dirs, save_account_value_plot
  - methodology aligned with V2 canonical
  - backup/tag: etape-2-complete
↓
V1 etape 3 algorithmic baselines port ✅
  - 10 single-ticker baselines + 2 portfolio baselines (static + DeMiguel rebalanced 1/N)
  - 24 single-ticker configs + 2 portfolio = 26 strategy outputs
  - 3e grid runner orchestrator
  - bollinger num_std fix (force_recompute_bands)
  - aggregated summary CSVs
  - 25 PNG plots + 25 metrics CSVs + 26 configs + 26 data files
  - finding: equal_weight_buy_and_hold +4342% / -33.73% drawdown
  - finding: naive_one_over_n_rebalanced_21d +1950.73% / -29.73%
  - finding: best KO single-ticker bollinger_mr_20_2 +275.82%
  - backup/tag: etape-3-complete
  ↓
V1 etape 4½ integration bonus features ✅ / 2026-05-25
  - 3-layer pipeline orchestrator: scripts/all_baselines_and_iqn_demo_5.py ✅
  - runs algorithmic + FinRL + IQN end-to-end in ~3 minutes ✅
  - PIT window filtering correction across all 11 algorithmic baselines ✅
  - uniform 0.1% transaction cost applied to all 242 algorithmic configs ✅
  - 3 new visualization modules (algorithmic, finrl multi-baseline, transaction logs) ✅
  - IQN device=auto bug fix in run_iqn_learning_curve_smoke_test.py ✅
  - Windows long path support + sys.path injection for reproducibility ✅
  - data migration completed (Device A → Device B via robocopy) ✅
  - finding: AVGO buy_and_hold +290.1% (was +31,063% pre-fix) ✅
  - finding: equal_weight portfolio +90.8% over PIT window ✅
↓
V1 etape 5 summary dashboard port ✅ DONE / 2026-05-26 05:18
  - new file: src/stock_investment_dss/visualization/summary_dashboard.py ✅
  - 4-panel layout matching v1 reference (StockDSS Runner Summary) ✅
  - panel 1: total return by strategy (horizontal bar) ✅
  - panel 2: maximum drawdown by strategy (horizontal bar) ✅
  - panel 3: annualized Sharpe by strategy (horizontal bar) ✅
  - panel 4: last IQN decision risk-adjusted action score ✅
  - color coding: FinRL blue, algorithmic green, IQN orange ✅
  - 246 strategies populated:
    - 240 algorithmic baselines (etape 3 grid)
    - 5 FinRL agents (10-seed multiseed: A2C/DDPG/PPO/TD3/SAC)
    - 1 MVO baseline
    - 1 D-IQN-DSS (10-seed clean_25k)
  - output: outputs/runs/2026_05_26_051810_d_iqn_dss_summary_dashboard/ ✅
    - data/strategies_combined.csv (246 rows)
    - summary/summary_dashboard.png
    - config/, plots/, logs/ (rich output)
  - 10-seed multiseed aggregation:
    - Action distribution: HOLD 61.4%, BUY 15.8%, SELL 22.6%, REBALANCE 0.2%
    - D-IQN-DSS mean: return 43.86%, Sharpe 1.48, drawdown -8.16%
  - rich output pattern (config/, plots/, summary/, data/, logs/) ✅
  - W&B logging via wandb_tracking helper ✅
  - HDP/EDL variants deferred to Phase B ablation ✅
  - py_compile passes ✅
  - backup/tag: etape-5-complete ✅
↓
V1 etape 6 compare_algorithmic_results report ✅ DONE / 2026-05-26 06:34
  - new file: src/stock_investment_dss/algorithmic_trading/experiments/compare_algorithmic_results.py ✅
  - new file: scripts/build_comparison_report.py ✅ (CLI auto-discoverer)
  - function: find_metric_files (auto-discovers strategies_combined.csv) ✅
  - function: load_metrics (concat + numeric coerce + dedup) ✅
  - function: add_rankings (rank_return, rank_sharpe, rank_drawdown, combined_rank) ✅
  - function: make_insights (5-section markdown) ✅
  - function: save_markdown_report (4 CSV outputs + markdown) ✅
  - input: 2026_05_26_051810_d_iqn_dss_summary_dashboard/data/strategies_combined.csv
  - output: outputs/runs/2026_05_26_063445_d_iqn_dss_comparison_report/ ✅
    - summary/comparison.md (thesis-citable)
    - summary/comparison.csv
    - summary/algorithmic_only.csv
    - summary/rl_only.csv
    - data/strategies_ranked_full.csv
    - config/comparison_report_config.json
    - logs/run.log
  - KEY FINDINGS:
    - Top 5: CAT_breakout_55 (rank 16.33), WMT_momentum_10, CAT_sma_20_50, ORCL_vol_filter, WMT_momentum_60
    - Best algorithmic: CAT_breakout_55 (+190.22%, Sharpe 1.85)
    - Best FinRL: MVO (rank 49.00, +126.69%)
    - D-IQN-DSS: rank 37.67/246 (top 15%), Sharpe rank 20, drawdown rank 3
    - DeMiguel 1/N rebalanced: rank 36.67 (academic validation)
    - Tier comparison (mean): IQN Sharpe 1.48 > FinRL 1.31 > Algorithmic 0.51
  - W&B logging with 11 metrics including iqn_rank_return, iqn_rank_sharpe, iqn_rank_drawdown ✅
  - rich output pattern (config/, summary/, data/, logs/) ✅
  - py_compile passes ✅
  - backup/tag: etape-6-complete ✅
↓
V1 etape 7 Demo_Baselines_and_IQN orchestrator ✅ DONE / 2026-05-26 07:30
  - new file: src/stock_investment_dss/runner/run_demo_baselines_and_iqn.py ✅
  - single thesis-demo runner orchestrating entire pipeline ✅
  - CLI args: --universe, --pit-decision-date, --iqn-checkpoint, --finrl-timesteps,
    --skip-finrl-training, --output-dir, --continue-on-error ✅
  - steps: load Mode B dataset → algorithmic → FinRL train+backtest → IQN inference
    → decision dashboard (skipped) → summary dashboard → comparison report → master summary ✅
  - output: outputs/runs/2026_05_26_072929_d_iqn_dss_demo_baselines_and_iqn/ ✅
  - showcase script for thesis evidence and professor demo ✅
  - rich output pattern + W&B logging ✅
  - py_compile passes ✅
  - smoke test with --skip-finrl-training=true (58 seconds end-to-end) ✅
  - 3-tier IQN checkpoint resolution + SHA-256 dataset hashing ✅
  - backup/tag: etape-7-complete ✅
↓
B) IQN rich-output patch ⬅️ OPTIONAL POLISH AFTER ETAPE 5
  - decision: whether to apply same RICH pattern to IQN learning curve runner
  - current IQN run state (37 files):
    - summary/ (14 files: 8 plots, 4 JSONs, 2 md)
    - data/ (10 files)
    - models/ (1 .pt)
    - logs/ (2 files)
    - wandb/ (12 files)
    - audit/, config/, metrics/, plots/ EMPTY
  - decision pending: is current output sufficient for thesis or apply RICH patch?
  - if applied: move 8 plots from summary/ → plots/ + add config/metrics
  - IQN already has rich W&B integration (87 references in IQN learning curve runner)
↓
After 7 etapes complete:
  ✅ single source of truth for trading metrics
  ✅ 10 algorithmic + 5 FinRL + 1 MVO + IQN strategies comparable
  ✅ V1 IQN distributional decision dashboard available in V2
  ✅ reproducible thesis demo via single command
  ✅ V2 remains canonical pipeline
↓
Full FinRL multiseed thesis evidence run ⬅️ WHEN READY
  - 5 seeds × 6 agents × 25,000 timesteps × 10 tickers
  - estimated runtime: ~58 minutes
  - W&B group: finrl-baseline-multiseed-demo10-25k-thesis
  - command: python scripts\finrl_multiseed_demo_10_new.py
  - produces thesis-grade FinRL evidence with statistical variance
↓
Full FinRL multiseed thesis evidence run ✅ DONE / 2026-05-26 ~05:00
  - confirmed: outputs/runs/2026_05_26_050316_d_iqn_dss_finrl_baseline_multiseed_summary
  - All 6 agents (A2C, DDPG, PPO, TD3, SAC, MVO) × multiseed ✅
  - Used in etape 7 demo orchestrator successfully ✅
↓
EDL-A dataset builder ⬅️ NEXT (Phase B.2 — after Phase B.1)
  - existing infrastructure: edl_action_dataset_v2.py + run_edl_action_dataset_v2_builder.py
  - build from counterfactual hindsight labels (counterfactual mode, NOT iqn_teacher)
  - exclude future outcome columns from input features
  - train/validation/test split inside PIT training period (80/20 time-ordered)
  - final PIT evaluation period kept untouched
↓
Reference-aligned EDL-A training ⬅️ Phase B.3 — after EDL-A DATASET
  - existing infrastructure: run_edl_action_training_v2_smoke_test.py
  - scale up: 50 epochs (not 10 as smoke)
  - LABEL MODE = "counterfactual" (CRITICAL — not iqn_teacher!)
  - MSE / log / digamma losses
  - KL annealing
  - evidence activation variants
  - best validation checkpoint
  - majority baseline comparison
  - balanced accuracy
  - macro F1
  - vacuity correct vs incorrect
↓
IQN + HDP + EDL-A gate ⬅️ Phase B.4 — after EDL-A VALIDATION
  - existing infrastructure: run_edl_action_gate_end_to_end_smoke_test.py
  - uncertainty-aware decision gating
  - human-review flags
  - pass-through / reduce / hold logic
  - compare against IQN-only and IQN+HDP
↓
Ablation suite ⬅️ AFTER CLEAN DIAGNOSTIC PLOTS
  - IQN only
  - IQN + LayerNorm
  - IQN + HDP
  - IQN + HDP + EDL-C
  - IQN + HDP + EDL-A
  - toggles for HDP on/off
  - toggles for EDL on/off
  - optional clean 50k only after clean 25k interpretation
  - optional epsilon schedule ablation
↓
↓
Ablation suite ⬅️ Phase B.5 — KERNE THESIS QUESTION
  - 4-way ablation matrix:
    - IQN only (Phase B.1 output partial)
    - IQN + HDP (Phase B.1 output) 
    - IQN + EDL-A (subset of B.4)
    - IQN + HDP + EDL-A (full stack, B.4)
  - existing infrastructure: 
    - run_iqn_vs_baseline_comparison_summary.py
    - run_iqn_vs_baseline_comparison_plot.py
  - toggles via env vars: 
    - STOCK_INVESTMENT_DSS_USE_HIERARCHICAL_POLICY=true/false
    - STOCK_INVESTMENT_DSS_USE_EDL=true/false
    - STOCK_INVESTMENT_DSS_EDL_VARIANT=A/B/C
  - DEFERRED to post-thesis (optional):
    - clean 50k baseline 
    - epsilon schedule ablation
↓
Demo_5 / Demo_10 comparison summary ⬅️ REFRESH AFTER CLEAN PIPELINE
Demo_5 / Demo_10 comparison summary ⬅️ Phase C — REFRESH AFTER ABLATION
↓
Demo_5 / Demo_10 comparison plots ⬅️ REFRESH AFTER CLEAN PIPELINE
Demo_5 / Demo_10 comparison plots ⬅️ Phase C — REFRESH AFTER ABLATION
↓
Thesis evidence package ⬅️ REFRESH AFTER CLEAN PIPELINE
Thesis evidence package ⬅️ Phase C — FINAL REFRESH
↓
Demo_30 Mode B experiment ⬅️ OPTIONAL / IF TIME
Demo_30 Mode B experiment ⏳ OPTIONAL / IF TIME
↓
IQN hyperparameter tuning / validation design ⬅️ MODEL-QUALITY STEP
IQN hyperparameter tuning / validation design ⏳ POST-THESIS
↓
CHANGE_STRATEGY later reactivation ⬅️ IMPORTANT LATER
CHANGE_STRATEGY later reactivation ⏳ POST-THESIS
↓
Watchlist / candidate universe ⬅️ LATER
Watchlist / candidate universe ⏳ POST-THESIS
↓
Existing portfolio / new portfolio input ⬅️ LATER
Existing portfolio / new portfolio input ⏳ POST-THESIS
↓
Recommendation engine ⬅️ LATER / WEB POC
Recommendation engine ⏳ POST-THESIS / WEB POC
↓
Audit log ✅ / v1, later expanded
Audit log ✅ / v1, later expanded
↓
Common evaluation suite ✅ / partly, improving toward thesis-grade evaluation
Common evaluation suite ✅ / partly (improving toward thesis-grade evaluation)


--------------------------------------------------

2️⃣ Hvad betyder "Phase B"?
Lad mig konsolidere det helt klart:

Phase       Hvad                            Estimeret tid       Status
Phase A     V1→V2 port (etape 1-7)          ~24 timer✅        COMPLETE
Phase B     EDL-A production + ablation     ~4-5 timer🟡       STARTING B.1
Phase C     Thesis evidence refresh         ~2-3 timer⏳       Pending
Phase D     Demo_30 (optional)              10-20 timer⏳      Optional
Phase E     Thesis writing                  3 dage⏳           Friday-Sunday


Phase B detaljer:

Sub-phase   Hvad                             Tid                Status
B.1 Combined IQN+HDP audit production        ~30 min🟡         NOW
B.2 EDL-A counterfactual oracle production   ~15 min⏳         Next
B.3 EDL-A v2 training (counterfactual labels)~1 hour⏳         Next
B.44-way ablation runs (parallelizable)      ~2 hours⏳        Next
B.5Ablation comparison dashboard             ~30 min⏳         Next

--------------------------------------------------

Den oprindelige turbo-plan var:
04:30 → Start FinRL multiseed
05:00 → Verify smoke tests
06:00 → FinRL done, restart dashboard
07:00 → Test etape 6+7, commit, SLEEP

Actual status (06:35 + nu):


Plan          Status

04:30 FinRL multiseed           ✅ Done (~05:00)
05:00 Smoke tests               ✅ Done (alle 8 smoke tests passerede)
06:00 Summary dashboard         ✅ Done (~05:18)
07:00 Test etape 6              ✅ Done (06:34)
07:30 Test etape 7              ✅ Done (07:30)
07:40 Commit alt                ✅ Done (07:40)
07:40 SLEEP                     ❌ Du sover ikke endnu!








































----------------------


# The Guldmand Detailed Roadmap for IQN v4.0


## Current status headline (2026-05-25)
- Project has moved through 4 of 7 V1→V2 thesis port etaper.
- Etaper 1, 2, 3, 4 are complete with thesis-grade reproducible output.
- Etaper 5, 6, 7 are pending (summary dashboard, comparison report, demo orchestrator).
- All three baseline lag now operational:
  - 🟢 Algorithmic Trading (etape 3): 26 strategy configs, 25 plots, full rich output
  - 🔵 FinRL Baselines (etape 4): 6 agents, multiseed, full rich output, W&B logging
  - 🟠 IQN (already operational): 137 historical W&B runs, decision distributions, learning curves
- Mode A and Mode B data separation operational.
- Mode B reproducible thesis experiments validated.
- Clean 25k IQN-only baseline complete with 10 seeds.
- Clean HOLD diagnostic complete with thesis-safe interpretation.
- LayerNorm IQN stability fix operational.
- Repository cleanup and run_registry → outputs/runs migration complete.
- Manifest-driven clean baseline infrastructure operational.
- 5 V2 bugs found and fixed during etape 4 night-session (2026-05-24/25).
- Pipeline now produces canonical outputs/runs/{timestamp}_d_iqn_dss_{name}/ with
  9-directory structure (audit/, config/, data/, logs/, metrics/, models/, plots/,
  summary/, wandb/).
- Micro-multiseed-first rule established as standard workflow.
- W&B cloud logging functional for all 3 baseline lag plus IQN.
- Next active work: etape 5 summary dashboard combining all 3 lag.


## Architecture and infrastructure ✅ / v4.0
- V2 canonical pipeline at C:\Users\gurug\Dropbox\DataScience\Speciale\D-IQN-DSS\FinRL_IQN ✅
- V1 reference (read-only) at external/ObjectRL_style/src/stockdss/ ✅
- V2 outputs directory: outputs/runs/{YYYY_MM_DD_HHMMSS}_d_iqn_dss_{name}/ ✅
- 9-directory canonical structure per run:
  - audit/ (used by clean_25k_thesis_evidence and iqn_decision_audit_report)
  - config/ (used by algorithmic_baseline_grid and FinRL RICH patch)
  - data/ (used by all training/evaluation runners)
  - logs/ (auto-populated by setup_run_logger)
  - metrics/ (used by algorithmic_baseline_grid and FinRL RICH patch)
  - models/ (used by all training runners)
  - plots/ (used by algorithmic_baseline_grid and FinRL RICH patch)
  - summary/ (used by all runners — primary plot/summary location)
  - wandb/ (auto-generated by wandb.init() with dir=run_directory)
- create_run_paths(run_name) helper in src/stock_investment_dss/utilities/paths.py ✅
- 56 V2 runners total ✅
- Scripts directory: cross-platform Python only (no .ps1/.sh) ✅


## Data pipeline ✅ / v2.8
- FinRL YahooDownloader chunked ✅
- yfinance browser-session fallback (firefox135) ✅
- CSV import fallback (only when explicitly allowed) ✅
- canonical cache ✅
- cache-origin metadata ✅
- data completeness validation ✅
- row count by ticker ✅
- failed ticker reporting ✅
- Mode A / Mode B separation ✅
- Mode B frozen import/master file ✅


## Point-in-time split ✅
- Train/trade split ✅
- No look-ahead / no leakage ✅
- PIT metadata ✅
- Dataset/split ID logged ✅
- Need internal EDL train/validation/test split inside PIT training period ⬅️ pending


## Demo configurations
- Demo_5: AAPL, MSFT, NVDA, AMZN, GOOGL ✅ (Mode A + Mode B operational)
- Demo_10_new: COST, AVGO, LLY, ORCL, CAT, BA, KO, MCD, WMT, PG ✅
- Demo_10_new train: 2010-01-01 → 2023-12-31 ✅
- Demo_10_new eval: 2024-01-01 → 2026-12-31 ✅
- Demo_30 Mode B ⬅️ optional / if time


## FinRL baseline suite ✅ / v3.7 + v4.0 (RICH output)
- 6 agents implemented: A2C, DDPG, PPO, TD3, SAC, MVO ✅
- Trading status, cost, trades logging ✅
- Per-agent rich output structure (after etape 4 RICH patch 2026-05-25):
  - data/finrl_baseline_suite/{agent}/{agent}_action_memory.csv ✅
  - data/finrl_baseline_suite/{agent}/{agent}_asset_memory.csv ✅
  - data/finrl_baseline_suite/{agent}/{agent}_metrics_summary.json ✅
  - data/finrl_baseline_suite/{agent}/{agent}_metrics_timeseries.csv ✅
  - metrics/finrl_baseline_suite/{agent}/{agent}_metrics.json ✅
  - models/finrl_baseline_suite/{agent}.zip ✅ (RL agents only)
  - plots/finrl_baseline_suite/{agent}/{agent}_portfolio_value.png ✅
  - plots/finrl_baseline_suite/{agent}/{agent}_actions_over_time.png ✅
  - plots/finrl_baseline_suite/{agent}/{agent}_action_distribution.png ✅
- config/finrl_baseline_suite_config.json (all env vars + suite params) ✅
- summary/finrl_baseline_suite_comparison_snapshot.csv ✅
- summary/finrl_baseline_suite_smoke_summary.json ✅
- wandb/ auto-generated inside run directory ✅
- Seed injection bug fix: env-var driven (RANDOM_SEED / FINRL_SEED / SB3_SEED) ✅
- W&B logging integrated via experiment_tracking/wandb_tracking.py ✅
- 5 outputs per multiseed batch (launcher + N seeds + aggregator) ✅


## Demo_5 Mode B FinRL/MVO baseline multiseed ✅
- All 6 agents × 5 seeds ✅
- Same demo_5 dataset/split/tickers as IQN ✅
- Mode B frozen data source ✅


## Demo_10_new FinRL baseline multiseed ✅ / micro validated, full pending
- Micro multiseed (3 seeds × 3 agents × 500 steps × 5 tickers): ✅ ~70 sec
- Full multiseed (5 seeds × 6 agents × 25k steps × 10 tickers): ⬅️ pending, ~58 min


## Discrete DSS decision layer ✅ / v1
- High-level discrete IQN/DSS actions (HOLD, BUY, SELL, REBALANCE) ✅
- CHANGE_STRATEGY temporarily disabled ✅
- IQN-compatible discrete action space ✅
- Trading delegated to FinRL StockTradingEnv ✅


## IQN ✅ / operational with 137 W&B runs
- IQN-compatible env smoke test ✅
- IQN agent + training loop ✅
- Replay buffer + tau sampling + quantile Huber loss ✅
- Target network update ✅
- Action-mask-aware IQN selection ✅
- Decision distribution table ✅
- LayerNorm state-encoder stability fix ✅
- 14 IQN runners in V2:
  - run_iqn_learning_curve_smoke_test.py (87 W&B references) ✅
  - run_iqn_learning_curve_multiseed_launcher.py ✅
  - run_iqn_learning_curve_multiseed_summary.py ✅
  - run_iqn_train_smoke_test.py ✅
  - run_iqn_backtest_smoke_test.py ✅
  - run_iqn_decision_audit_report.py (11 W&B references) ✅
  - run_iqn_decision_distribution_smoke_test.py ✅
  - run_iqn_decision_export_smoke_test.py ✅
  - run_iqn_experiment_from_config.py ✅
  - run_iqn_no_trade_diagnostic.py ✅
  - run_iqn_reward_action_diagnostic.py ✅
  - run_iqn_seed_config_diagnostic.py ✅
  - run_iqn_vs_baseline_comparison_plot.py ✅
  - run_iqn_vs_baseline_comparison_summary.py ✅
  - run_iqn_vs_baseline_metrics_plots.py ✅
- IQN run output structure (37 files):
  - summary/ (14 files: 8 plots, 4 JSONs, 2 markdown) ✅
  - data/ (10 files: episode/eval/training records, distributions, decision memory) ✅
  - models/ (1 .pt) ✅
  - logs/ (2 files) ✅
  - wandb/ (12 files) ✅
  - audit/, config/, metrics/, plots/ EMPTY (V2 standard, IQN plots in summary/)
- IQN learning diagnostics ✅


## HOLD-collapse diagnostic ✅ / v3.0
- Original 3/5 demo_5 no-trade root cause: raw FinRL state scale instability ✅
- Transaction-cost / CVaR-only / gradient-clip hypotheses rejected ✅
- LayerNorm selected as fix ✅


## Clean 25k baseline ✅ / v3.7
- configs/experiments/clean_25k_baseline_v1.json ✅
- verify_experiment_config.py ✅ (17/17 assertions passed)
- run_iqn_experiment_from_config.py ✅
- 10/10 seeds completed ✅
- 1/10 full HOLD collapse, 9/10 non-zero trading ✅
- masked_action_rate=0.0 across all seeds ✅
- residual Q-policy attractor identified ✅
- Clean diagnostic plots, interpretation note, evidence package ✅
- Thesis section draft (Markdown + LaTeX) ✅


## Algorithmic baselines ✅ / etape 3
- 10 single-ticker baselines:
  - buy_and_hold, sma_crossover, ema_crossover, macd_signal,
    rsi_mean_reversion, bollinger_mean_reversion, breakout, momentum,
    volatility_filter, plus equal_weight_buy_and_hold portfolio ✅
- 2 portfolio baselines: equal_weight_buy_and_hold (static),
  naive_one_over_n_rebalanced_21d (DeMiguel) ✅
- 26 total strategy outputs in algorithmic_baseline_grid_demo_10_new ✅
- Rich output: 26 configs + 26 data + 25 metrics + 25 plots + 3 aggregate summaries ✅
- Bollinger num_std fix (force_recompute_bands) ✅
- PIT-safe (signals shifted by 1 day) ✅
- Zero transaction costs (default) ✅


## Hierarchical DSS policy PoC ✅ / v3.0
- Action type → ticker selection → size selection ✅
- Technical feature builder ✅
- MA50 / MA200 features ✅
- Frozen fundamental feature store placeholder ✅
- Ticker selector ✅
- Size selector ✅
- BUY/HOLD smoke tests ✅


## Combined IQN + HDP audit ✅ / v3.3
- IQN action distribution features ✅
- HDP ticker/size enrichment ✅
- EDL-C teacher label generation ✅
- Source-run control ✅


## EDL methodology ✅ / v3.4 + v3.5
- EDL-C teacher-imitation track (integration evidence only) ✅
- EDL-A hindsight labeler (first smoke test) ✅
- Counterfactual EDL-A oracle (smoke-test complete) ✅
- A/B/C clarified as label-source variants ✅
- EDL-A identified as main supervised correctness track ✅


## Frozen FMP/HDP point-in-time fundamentals ✅ / v3.6
- FMP live/cache mode separation ✅
- acceptedDate → filingDate → date fallback ✅
- PIT fundamentals builder ✅
- HDP joined feature table ✅
- PIT information inspection runner ✅
- Conservative PIT lag / valuation-audit transparency patch ⏳


## Repository hygiene ✅ / v3.7
- Repo root cleanup ✅
- Historical PowerShell launchers removed ✅
- Legacy W&B artifacts handled ✅
- .env IQN hyperparameter contamination removed ✅
- run_registry → outputs/runs migration complete ✅
- 417 local runs inventoried/classified ✅
- May 22 demo10_new runs marked confounded/debug ✅
- 29 cache files + 20 runs migrated to V2 canonical paths ✅
- 185 artifacts migrated from outputs/run_registry → outputs/runs (Plan 2) ✅
- Path bug fix: parents[4] → parents[3] in 6 files (Plan 1) ✅


## V2 W&B coverage (audited 2026-05-25)
- 6 of 56 V2 runners have W&B integration (all IQN-related):
  - run_iqn_learning_curve_smoke_test.py (87 references)
  - run_clean_25k_thesis_evidence_package.py (17 references)
  - run_wandb_setup_check.py (12 references)
  - run_iqn_decision_audit_report.py (11 references)
  - run_smoke_test.py (4 references)
  - run_iqn_seed_config_diagnostic.py (1 reference)
- 50 of 56 V2 runners have NO W&B integration (by design)
- FinRL runners added W&B via etape 4 RICH patch (2026-05-25)
- Two W&B helpers exist:
  - experiment_tracking/wandb_tracking.py (simple, 120 lines, run_directory param)
  - utilities/experiment_tracking.py (legacy ExperimentTracker dataclass, not used)


## Workflow standards ✅ / v4.0
- HARD IMPLEMENT NOW trigger required before any code changes ✅
- Backup before each major change ✅
- One Copilot /plan per etape (no giga-plans) ✅
- Academic repository hygiene (no AI/prompt/conversation references) ✅
- Micro-multiseed-first rule: always validate pipeline with <5 min run before
  committing to long compute ✅
- Cross-platform Python scripts only (no .ps1/.sh wrappers for new scripts) ✅
- Triangulated review workflow (Copilot/ChatGPT/Claude) ✅


## Lessons learned from etape 4 (2026-05-24/25 night session)
- Multiseed launcher must inject seeds via env vars, not via hardcoded constants
- V2 aggregator filter is sensitive to launcher output; depends on seed injection working
- V2 outputs/runs canonical structure: audit/, config/, data/, logs/, metrics/,
  models/, plots/, summary/, wandb/ (9 directories)
- Not all directories are used by all runners; algorithmic_baseline_grid uses 6/9,
  IQN learning curve uses 5/9, FinRL RICH patch uses 8/9
- wandb.init() defaults to CWD/wandb/ unless dir= is explicitly passed
- Wandb-core daemon process is normal and persistent; it does not need to be killed
  unless cleaning up locked stale directories
- Dropbox does not interfere with wandb file locks under normal operation
- Always check W&B mappe placering efter W&B integration tilføjes
- Subprocess inherits env vars from parent process automatically (no special setup needed)


## Pending work items (v4.0 → v4.1)


### Immediate next (etape 5 — Summary Dashboard)
- 4-panel layout matching V1 "StockDSS Runner Summary" reference
- Aggregate across algorithmic + FinRL + IQN
- Color coding: FinRL blue, algorithmic green, IQN orange
- Last IQN decision risk-adjusted action score panel
- Rich output + W&B logging consistent with etape 4 pattern


### After etape 5
- Etape 6: compare_algorithmic_results report (markdown + CSV)
- Etape 7: Demo_Baselines_and_IQN orchestrator (single thesis-demo runner)
- Full FinRL multiseed thesis evidence run (~58 min, 5 seeds × 6 agents × 25k steps)


### Optional polish
- B) IQN rich-output patch (decide whether to move plots from summary/ to plots/)
- run_finrl_baseline_multiseed_summary.py rich-output patch (currently 2 directories,
  could be extended to match RICH pattern)


### EDL-A track (after baseline thesis evidence locked)
- EDL-A dataset builder from counterfactual hindsight labels
- Reference-aligned EDL-A training (MSE/log/digamma + KL annealing)
- IQN + HDP + EDL-A uncertainty gate
- Ablation suite (IQN-only / +LayerNorm / +HDP / +EDL-C / +EDL-A)


### Optional / time permitting
- Demo_30 Mode B experiment
- IQN hyperparameter tuning
- CHANGE_STRATEGY reactivation
- Watchlist / candidate universe
- Existing portfolio / new portfolio input
- Recommendation engine
- Web PoC


## Common evaluation suite ✅ / partly, improving
- Evaluation/configs/default.json ✅
- Cumulative return, max drawdown, Sharpe, volatility, CVaR, downside risk ✅
- Turnover (estimate) ✅
- Transaction costs ✅
- IQN vs baseline comparison summary ✅ (earlier, refresh after etape 5)
- Continuous-vs-discrete comparison ✅ (initial)
- Strategy violation metrics ⬅️ later
- Decision stability ⬅️ later
- Human-in-the-loop point-in-time simulation mode ⬅️ later
- EDL evaluation metrics ⏳
- Calibration / confidence quality ⏳
- Uncertainty threshold curves ⏳


## Backups created during etape 4 (2026-05-24/25)
- backup_plan_1a_20260524_192717 (Plan 1 path fix)
- backup_plan_2_1_20260524_193818 (Plan 2 migration phase 1)
- backup_plan_2_2_20260524_202820 (Plan 2 migration phase 2)
- backup_plan_2_3_20260524_205317 (Plan 2 migration phase 3)
- backup_seed_fix_20260524_230939 (FinRL suite seed injection fix)
- backup_rich_patch_20260524_235814 (FinRL RICH patch)
- backup_wandb_dir_fix_20260525_001545 (wandb_tracking dir parameter)