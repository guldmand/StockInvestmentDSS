# src/stock_investment_dss/decision/__init__.py
"""
D-IQN-DSS decision package — hierarchical policy PoC (v3.0).

Modules:
  decision_actions       — DSSDecisionAction enum (pre-existing)
  investor_risk_profile  — InvestorRiskProfile dataclass (pre-existing)
  action_mask            — Action masking utilities (pre-existing)
  risk_aware_action_resolver — IQN risk-aware resolution (pre-existing)
  ticker_selector        — Rule-based ticker scoring and selection (v3.0)
  size_selector          — Discrete size selection with risk adjustments (v3.0)
  hierarchical_decision_policy — 5-stage orchestrator (v3.0)
"""
