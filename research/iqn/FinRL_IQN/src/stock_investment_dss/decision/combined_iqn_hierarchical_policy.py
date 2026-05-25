# src/stock_investment_dss/decision/combined_iqn_hierarchical_policy.py
"""
CombinedIQNHierarchicalPolicy — Integration layer for D-IQN-DSS (v3.3)

Connects IQN learning curve run output with HierarchicalDecisionPolicy to
produce a combined audit dataset for EDL-C training and end-to-end testing.

Pipeline
--------
IQN eval data (eval_distributions + eval_step_records)
  → eval_step → trading date mapping (via market data + eval window)
  → HierarchicalDecisionPolicy.decide() enrichment
  → combined audit row (IQN quantile features + HDP features + market features)

Output
------
combined_iqn_hierarchical_decision_by_step.csv
  - One row per (train_step, eval_step)
  - IQN action distribution columns (q10/q50/q90/cvar10/score per action)
  - HierarchicalDecisionPolicy enrichment columns
  - Market / technical feature snapshot at decision date
  - EDL label placeholder columns (edl_a_*, edl_b_*, edl_c_*)

Point-in-time safety
--------------------
All features use data available at decision_date or earlier.
EDL-A labels (hindsight) are NOT populated here — they require future prices
and must be added in a separate pass or via edl_action_dataset_v2.

References
----------
See copilot-diagnostics/design/edl_uncertainty_poc/edl_v3_3_reference_repo_alignment_and_integration_plan.md
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# IQN action names (5 actions: 0=HOLD, 1=BUY, 2=SELL, 3=REBALANCE, 4=CHANGE_STRATEGY)
_IQN_ACTION_NAMES: Dict[int, str] = {
    0: "HOLD",
    1: "BUY",
    2: "SELL",
    3: "REBALANCE",
    4: "CHANGE_STRATEGY",
}

# EDL action space (4 actions — CHANGE_STRATEGY excluded from EDL output space)
_EDL_ACTION_NAMES = ["HOLD", "BUY", "SELL", "REBALANCE"]

# IQN quantile columns to pivot into wide format
_DIST_SCALAR_COLS = ["q10", "q25", "q50", "q75", "q90", "cvar10", "score", "mean"]

# Default market data import path
_DEFAULT_MARKET_DATA_PATH = "data/market/daily/imports/market_data_full_500.csv"

# IQN learning curve run suffix pattern
_IQN_RUN_SUFFIX = "iqn_learning_curve_smoke_test"

# Required IQN run data files
_REQUIRED_IQN_FILES = [
    "data/iqn_learning_curve_eval_distributions.csv",
    "data/iqn_learning_curve_eval_step_records.csv",
    "summary/experiment_context_summary.json",
]


# ---------------------------------------------------------------------------
# IQNRunData dataclass
# ---------------------------------------------------------------------------


@dataclass
class IQNRunData:
    """
    Holds loaded data from an IQN learning curve run.

    Attributes
    ----------
    run_dir : Path
        Absolute path to the IQN run directory.
    run_id : str
        Run identifier (directory name).
    ctx : dict
        Parsed experiment_context_summary.json.
    eval_step_df : pd.DataFrame
        eval_step_records for all train steps.
    dist_df : pd.DataFrame
        eval_distributions for all train steps (5 rows per eval_step).
    tickers : list[str]
        Tickers used in the IQN run.
    eval_window_start : str
        ISO date string for eval window start.
    eval_window_end : str
        ISO date string for eval window end.
    available_train_steps : list[int]
        Sorted list of train steps with eval data.
    """

    run_dir: Path
    run_id: str
    ctx: dict
    eval_step_df: pd.DataFrame
    dist_df: pd.DataFrame
    tickers: List[str] = field(default_factory=list)
    eval_window_start: str = ""
    eval_window_end: str = ""
    available_train_steps: List[int] = field(default_factory=list)

    @property
    def last_train_step(self) -> int:
        return max(self.available_train_steps)

    @property
    def dataset_id(self) -> str:
        return self.ctx.get("dataset_id", "unknown")

    @property
    def split_id(self) -> str:
        return self.ctx.get("split_id", "unknown")


# ---------------------------------------------------------------------------
# IQNRunLoader
# ---------------------------------------------------------------------------


class IQNRunLoader:
    """
    Loads and validates an IQN learning curve run directory.

    Usage
    -----
    loader = IQNRunLoader(runs_dir)
    iqn_data = loader.load(run_id_or_none)
    """

    def __init__(self, runs_dir: Optional[Path] = None) -> None:
        self.runs_dir = runs_dir or _default_runs_dir()

    def find_latest_iqn_run(self) -> Path:
        """
        Find the most recent IQN learning curve smoke test run directory.

        Raises
        ------
        FileNotFoundError
            If no IQN learning curve run directory is found.
        """
        candidates = sorted(
            [
                d
                for d in self.runs_dir.iterdir()
                if d.is_dir() and _IQN_RUN_SUFFIX in d.name
            ],
            key=lambda p: p.name,
        )
        if not candidates:
            raise FileNotFoundError(
                f"No IQN learning curve smoke test run found in: {self.runs_dir}\n"
                f"Expected directory name matching: *{_IQN_RUN_SUFFIX}*\n"
                f"Run the IQN learning curve smoke test first:\n"
                f"  python -m stock_investment_dss.runner.run_iqn_learning_curve_smoke_test"
            )
        chosen = candidates[-1]
        logger.info("Found IQN run: %s", chosen.name)
        return chosen

    def load(self, run_id: Optional[str] = None) -> IQNRunData:
        """
        Load an IQN run by run_id, or find the latest if run_id is None.

        Parameters
        ----------
        run_id : str, optional
            Run directory name. If None, uses the latest.

        Returns
        -------
        IQNRunData
        """
        if run_id:
            run_dir = self.runs_dir / run_id
            if not run_dir.is_dir():
                raise FileNotFoundError(
                    f"IQN run directory not found: {run_dir}\n"
                    f"Available runs in {self.runs_dir}:\n"
                    + "\n".join(
                        f"  {d.name}"
                        for d in sorted(self.runs_dir.iterdir())
                        if d.is_dir() and _IQN_RUN_SUFFIX in d.name
                    )
                )
        else:
            run_dir = self.find_latest_iqn_run()

        return self._load_from_dir(run_dir)

    def _load_from_dir(self, run_dir: Path) -> IQNRunData:
        """Load and validate all required IQN run files."""
        # Check required files
        missing = [
            str(run_dir / f) for f in _REQUIRED_IQN_FILES if not (run_dir / f).exists()
        ]
        if missing:
            found = sorted(str(p) for p in run_dir.rglob("*.csv")) + sorted(
                str(p) for p in run_dir.rglob("*.json")
            )
            raise FileNotFoundError(
                f"IQN run is missing required files:\n"
                + "\n".join(f"  MISSING: {f}" for f in missing)
                + f"\n\nFiles found in {run_dir}:\n"
                + "\n".join(f"  {f}" for f in found[:30])
            )

        # Load context
        with open(run_dir / "summary" / "experiment_context_summary.json") as f:
            ctx = json.load(f)

        # Load eval step records
        eval_step_df = pd.read_csv(
            run_dir / "data" / "iqn_learning_curve_eval_step_records.csv"
        )
        logger.info(
            "Loaded eval_step_records: %d rows, train_steps: %s",
            len(eval_step_df),
            sorted(eval_step_df["train_step"].unique()),
        )

        # Load eval distributions
        dist_df = pd.read_csv(
            run_dir / "data" / "iqn_learning_curve_eval_distributions.csv"
        )
        logger.info("Loaded eval_distributions: %d rows", len(dist_df))

        tickers = ctx.get("tickers", [])
        eval_start = ctx.get("eval_window_start", "")
        eval_end = ctx.get("eval_window_end", "")
        available_train_steps = sorted(eval_step_df["train_step"].unique().tolist())

        logger.info(
            "IQN run %s: tickers=%s eval=%s→%s train_steps=%s",
            run_dir.name,
            tickers,
            eval_start,
            eval_end,
            available_train_steps,
        )

        return IQNRunData(
            run_dir=run_dir,
            run_id=run_dir.name,
            ctx=ctx,
            eval_step_df=eval_step_df,
            dist_df=dist_df,
            tickers=tickers,
            eval_window_start=eval_start,
            eval_window_end=eval_end,
            available_train_steps=available_train_steps,
        )


# ---------------------------------------------------------------------------
# Distribution pivot helpers
# ---------------------------------------------------------------------------


def _pivot_distributions(dist_df: pd.DataFrame, train_step: int) -> pd.DataFrame:
    """
    Pivot IQN distributions for a given train_step to wide format.

    Input: one row per (train_step, eval_step, action) — 5 actions per step.
    Output: one row per eval_step with columns like:
      iqn_q10_hold, iqn_q50_hold, ..., iqn_q10_buy, ..., iqn_score_hold, ...
      iqn_action_margin, iqn_chosen_action, iqn_chosen_action_index

    Parameters
    ----------
    dist_df : pd.DataFrame
        Full distributions dataframe (all train steps).
    train_step : int
        Which train step to pivot.

    Returns
    -------
    pd.DataFrame
        Wide-format distributions, indexed by eval_step.
    """
    ts_df = dist_df[dist_df["train_step"] == train_step].copy()
    if ts_df.empty:
        raise ValueError(
            f"No distributions found for train_step={train_step}. "
            f"Available: {sorted(dist_df['train_step'].unique())}"
        )

    # Normalise action name to lowercase for column names
    ts_df["action_lower"] = (
        ts_df["action"].str.lower().str.replace("_", "", regex=False)
    )

    # Build wide-format: one row per eval_step
    rows = []
    for eval_step, grp in ts_df.groupby("eval_step"):
        row: dict = {"eval_step": int(eval_step)}

        # IQN chosen action (same for all rows in this group)
        chosen_action = grp["chosen_action"].iloc[0]
        row["iqn_chosen_action"] = str(chosen_action)

        # Find action_index for chosen action
        chosen_idx_rows = grp[grp["action"] == chosen_action]["action_index"]
        row["iqn_chosen_action_index"] = (
            int(chosen_idx_rows.iloc[0]) if len(chosen_idx_rows) > 0 else -1
        )

        # Per-action quantile features
        score_map: dict = {}
        for _, arow in grp.iterrows():
            aname = str(arow["action"]).lower().replace("_", "")
            for col in _DIST_SCALAR_COLS:
                if col in arow.index:
                    colname = f"iqn_{col}_{aname}"
                    row[colname] = float(arow[col]) if pd.notna(arow[col]) else None
            # Store score for margin calculation
            if "score" in arow.index and pd.notna(arow["score"]):
                score_map[str(arow["action"])] = float(arow["score"])

        # Action margin: score[top1] - score[top2] (over allowed actions)
        allowed_grp = grp[grp.get("allowed", pd.Series([True] * len(grp))).astype(bool)]
        if "allowed" in grp.columns:
            allowed_scores = [
                float(r["score"])
                for _, r in grp[grp["allowed"] == True].iterrows()  # noqa: E712
                if pd.notna(r.get("score"))
            ]
        else:
            allowed_scores = [v for v in score_map.values()]

        if len(allowed_scores) >= 2:
            sorted_scores = sorted(allowed_scores, reverse=True)
            row["iqn_action_margin"] = sorted_scores[0] - sorted_scores[1]
        elif len(allowed_scores) == 1:
            row["iqn_action_margin"] = 0.0
        else:
            row["iqn_action_margin"] = None

        rows.append(row)

    wide_df = pd.DataFrame(rows)
    logger.info(
        "Pivoted distributions: %d eval_steps, %d columns",
        len(wide_df),
        len(wide_df.columns),
    )
    return wide_df


# ---------------------------------------------------------------------------
# Date sequence helper
# ---------------------------------------------------------------------------


def build_eval_date_sequence(
    market_df: pd.DataFrame,
    tickers: List[str],
    eval_window_start: str,
    n_steps: int,
) -> List[str]:
    """
    Build the ordered list of trading dates for the eval period.

    Takes the first `n_steps` trading days on or after eval_window_start
    for the given tickers from the market data.

    Parameters
    ----------
    market_df : pd.DataFrame
        Full market data (date, tic, close, ...).
    tickers : list[str]
        IQN universe tickers.
    eval_window_start : str
        ISO date string — first date of eval window.
    n_steps : int
        Number of eval steps to map.

    Returns
    -------
    list[str]
        Sorted ISO date strings, len == n_steps (or less if data is insufficient).
    """
    tic_col = "tic" if "tic" in market_df.columns else "ticker"
    df = market_df[market_df[tic_col].isin(tickers)].copy()
    df["date"] = pd.to_datetime(df["date"])
    cutoff = pd.Timestamp(eval_window_start)
    eval_dates = sorted(df[df["date"] >= cutoff]["date"].unique())
    eval_date_strs = [str(d.date()) for d in eval_dates[:n_steps]]
    if len(eval_date_strs) < n_steps:
        logger.warning(
            "Only %d eval dates found (expected %d) — market data may be truncated",
            len(eval_date_strs),
            n_steps,
        )
    logger.info(
        "Eval date sequence: %d dates from %s to %s",
        len(eval_date_strs),
        eval_date_strs[0] if eval_date_strs else "N/A",
        eval_date_strs[-1] if eval_date_strs else "N/A",
    )
    return eval_date_strs


# ---------------------------------------------------------------------------
# Portfolio state reconstruction
# ---------------------------------------------------------------------------


def _reconstruct_portfolio(
    step_row: pd.Series,
    price_snapshot: Optional[pd.DataFrame],
) -> object:
    """
    Reconstruct a PortfolioState from an eval step record row.

    Uses cash_before and portfolio_value_before from the IQN step record.
    If holdings_before_json is available, reconstructs per-ticker holding values
    using current prices from the snapshot.
    """
    from stock_investment_dss.decision.hierarchical_decision_policy import (
        PortfolioState,
    )

    total_value = float(step_row.get("portfolio_value_before", 1_000_000.0))
    cash = float(step_row.get("cash_before", total_value))

    holdings_raw: dict = {}
    holdings_json = step_row.get("holdings_before_json")
    if holdings_json and pd.notna(holdings_json):
        try:
            holdings_raw = json.loads(str(holdings_json))
        except (json.JSONDecodeError, TypeError):
            pass

    # Build holding values from shares * current price
    holding_values: dict = {}
    if price_snapshot is not None and holdings_raw:
        tic_col = "tic" if "tic" in price_snapshot.columns else "ticker"
        for ticker, shares in holdings_raw.items():
            ticker_row = price_snapshot[price_snapshot[tic_col] == ticker]
            if not ticker_row.empty:
                price = float(ticker_row["close"].iloc[0])
                holding_values[ticker] = float(shares) * price
            else:
                holding_values[ticker] = 0.0

    return PortfolioState(
        total_value=total_value,
        cash=cash,
        holdings={t: float(v) for t, v in holdings_raw.items()},
        holding_values=holding_values,
    )


# ---------------------------------------------------------------------------
# CombinedIQNHierarchicalPolicy
# ---------------------------------------------------------------------------


class CombinedIQNHierarchicalPolicy:
    """
    Builds a combined audit dataset from IQN eval data + HierarchicalDecisionPolicy.

    For each eval step at a given train_step:
    1. Map eval_step → trading date
    2. Get IQN action distribution features (pivoted to wide format)
    3. Get market snapshot features at that date (technical + fundamental)
    4. Call HierarchicalDecisionPolicy.decide() with IQN chosen action
    5. Produce a combined audit row

    Parameters
    ----------
    market_df : pd.DataFrame
        Full market data (used for technical feature building).
    strategy_id : str
        HDP strategy profile.
    defensive_strategy : bool
        HDP defensive mode.
    market_data_path : str
        Path string used for metadata/audit.
    """

    def __init__(
        self,
        market_df: pd.DataFrame,
        strategy_id: str = "balanced_v1",
        defensive_strategy: bool = False,
        market_data_path: str = _DEFAULT_MARKET_DATA_PATH,
    ) -> None:
        self.market_df = market_df
        self.strategy_id = strategy_id
        self.defensive_strategy = defensive_strategy
        self.market_data_path = market_data_path

        # Deferred imports — keep at call time to surface ImportErrors cleanly
        self._tech_builder = None
        self._fund_store = None
        self._policy = None
        self._risk_profile = None

        # Cache for built tech features (built once, reused per eval step)
        self._cached_tech_df: Optional[pd.DataFrame] = None

    def _ensure_deps(self) -> None:
        if self._policy is not None:
            return
        from stock_investment_dss.data.fundamental_feature_store import (
            FundamentalFeatureStore,
        )
        from stock_investment_dss.data.technical_feature_builder import (
            TechnicalFeatureBuilder,
        )
        from stock_investment_dss.decision.hierarchical_decision_policy import (
            HierarchicalDecisionPolicy,
        )
        from stock_investment_dss.decision.investor_risk_profile import (
            InvestorRiskProfile,
        )

        self._risk_profile = InvestorRiskProfile.balanced()
        self._tech_builder = TechnicalFeatureBuilder()
        self._fund_store = FundamentalFeatureStore()
        self._policy = HierarchicalDecisionPolicy(
            risk_profile=self._risk_profile,
            strategy_id=self.strategy_id,
            defensive_strategy=self.defensive_strategy,
        )
        logger.info(
            "Initialised HierarchicalDecisionPolicy (strategy=%s)", self.strategy_id
        )

    def build_combined_audit(
        self,
        iqn_data: IQNRunData,
        train_step: Optional[int] = None,
        use_hierarchical_policy: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
        """
        Build the combined audit dataset for a given IQN train_step.

        Parameters
        ----------
        iqn_data : IQNRunData
            Loaded IQN run data.
        train_step : int, optional
            Which checkpoint to use. Defaults to iqn_data.last_train_step.
        use_hierarchical_policy : bool
            If False, skips HDP enrichment (IQN action is passed through directly).

        Returns
        -------
        combined_df : pd.DataFrame
            Main audit table (one row per eval_step).
        ticker_score_df : pd.DataFrame
            Ticker score rows from HDP (one row per ticker per eval_step).
        size_score_df : pd.DataFrame
            Size score rows from HDP.
        warnings : list[str]
            Non-fatal warnings about missing data.
        """
        self._ensure_deps()

        ts = train_step if train_step is not None else iqn_data.last_train_step
        logger.info(
            "Building combined audit: run=%s train_step=%d",
            iqn_data.run_id,
            ts,
        )

        warnings: List[str] = []

        # ------------------------------------------------------------------
        # Step 1: Pivot IQN distributions
        # ------------------------------------------------------------------
        dist_wide = _pivot_distributions(iqn_data.dist_df, train_step=ts)
        n_steps = len(dist_wide)
        logger.info("Pivoted distributions: %d eval_steps", n_steps)

        # ------------------------------------------------------------------
        # Step 2: Build eval date sequence
        # ------------------------------------------------------------------
        eval_dates = build_eval_date_sequence(
            market_df=self.market_df,
            tickers=iqn_data.tickers,
            eval_window_start=iqn_data.eval_window_start,
            n_steps=n_steps,
        )
        if len(eval_dates) < n_steps:
            w = (
                f"Only {len(eval_dates)} eval dates found (expected {n_steps}). "
                "Some eval_steps will have no date."
            )
            warnings.append(w)
            logger.warning(w)

        # Map eval_step → date
        step_to_date = {i: d for i, d in enumerate(eval_dates)}

        # ------------------------------------------------------------------
        # Step 3: IQN eval step records for this train_step
        # ------------------------------------------------------------------
        step_df = iqn_data.eval_step_df[
            iqn_data.eval_step_df["train_step"] == ts
        ].set_index("eval_step")

        # ------------------------------------------------------------------
        # Step 4: Technical features (build once, reuse per step)
        # ------------------------------------------------------------------
        if self._cached_tech_df is None:
            logger.info("Building technical features (full history, one-time) ...")
            self._cached_tech_df = self._tech_builder.build(self.market_df)
            self._cached_tech_df["date"] = pd.to_datetime(self._cached_tech_df["date"])
            logger.info("Technical features built: %d rows", len(self._cached_tech_df))
        all_tech_df = self._cached_tech_df

        # ------------------------------------------------------------------
        # Step 5: Iterate eval_steps and build combined rows
        # ------------------------------------------------------------------
        combined_rows: List[dict] = []
        ticker_score_rows: List[dict] = []
        size_score_rows: List[dict] = []

        n_with_dist = 0
        n_with_hdp = 0

        for _, dist_row in dist_wide.iterrows():
            eval_step = int(dist_row["eval_step"])
            decision_date = step_to_date.get(eval_step)

            if decision_date is None:
                w = f"eval_step={eval_step} has no mapped date — skipping"
                warnings.append(w)
                logger.warning(w)
                continue

            n_with_dist += 1

            # IQN step record for this eval_step
            iqn_step = step_df.loc[eval_step] if eval_step in step_df.index else None

            # IQN action info
            iqn_chosen = str(dist_row.get("iqn_chosen_action", "HOLD"))
            iqn_chosen_idx = int(dist_row.get("iqn_chosen_action_index", 0))

            # Market snapshot at decision_date (PIT-safe: use pre-built df, filter to cutoff)
            cutoff_ts = pd.Timestamp(decision_date)
            snap = (
                all_tech_df[all_tech_df["date"] <= cutoff_ts]
                .sort_values("date")
                .groupby("tic")
                .tail(1)
                .reset_index(drop=True)
            )
            price_snapshot = snap  # for portfolio reconstruction

            # Fund features
            fund_df = self._fund_store.get_scores_as_of(decision_date, iqn_data.tickers)

            # Join tech + fund
            tic_col_snap = "tic" if "tic" in snap.columns else "ticker"
            joined = snap.merge(
                fund_df.rename(columns={"ticker": tic_col_snap}),
                on=tic_col_snap,
                how="left",
                suffixes=("", "_fund"),
            )

            # Portfolio state reconstruction
            if iqn_step is not None:
                portfolio = _reconstruct_portfolio(iqn_step, price_snapshot=snap)
            else:
                from stock_investment_dss.decision.hierarchical_decision_policy import (
                    PortfolioState,
                )

                portfolio = PortfolioState(total_value=1_000_000.0, cash=1_000_000.0)

            # ------------------------------------------------------------------
            # HDP enrichment
            # ------------------------------------------------------------------
            hdp_action = iqn_chosen
            hdp_ticker = None
            hdp_size = None
            hdp_fraction = 0.0
            hdp_ticker_score = None
            hdp_size_score = None
            hdp_risk_ok = True
            hdp_bear_signal = False
            hdp_bear_penalty = 0.0
            hdp_risk_adjusted_fraction = 0.0
            step_ticker_scores: List[dict] = []
            step_size_scores: List[dict] = []

            if use_hierarchical_policy:
                try:
                    from stock_investment_dss.decision.decision_actions import (
                        parse_action_label,
                    )

                    # Build IQN action scores for Stage 1 (from score column per action)
                    stage1_scores = {}
                    for action_name in [
                        "HOLD",
                        "BUY",
                        "SELL",
                        "REBALANCE",
                        "CHANGE_STRATEGY",
                    ]:
                        col = f"iqn_score_{action_name.lower().replace('_','')}"
                        if col in dist_row and pd.notna(dist_row[col]):
                            stage1_scores[action_name] = float(dist_row[col])

                    action_type = parse_action_label(iqn_chosen)
                    decision = self._policy.decide(
                        action_type=action_type,
                        features=joined,
                        portfolio=portfolio,
                        decision_date=decision_date,
                        visible_data_cutoff=decision_date,
                        iqn_model_run_id=iqn_data.run_id,
                        score_mode="iqn_q50",
                        stage_1_action_type_scores=stage1_scores or None,
                    )

                    hdp_action = decision.selected_action_type
                    hdp_ticker = decision.selected_ticker
                    hdp_size = decision.selected_size
                    hdp_fraction = decision.selected_fraction
                    hdp_risk_adjusted_fraction = (
                        decision.risk_adjusted_allocation_fraction
                    )

                    risk_checks = decision.risk_checks
                    hdp_risk_ok = all(
                        v
                        for k, v in risk_checks.items()
                        if isinstance(v, bool) and k not in ("warnings",)
                    )
                    hdp_bear_signal = bool(risk_checks.get("bear_market_signal", False))
                    hdp_bear_penalty = float(
                        risk_checks.get("bear_market_penalty", 0.0)
                    )

                    # Ticker scores
                    for ts_row in decision.stage_2_ticker_scores:
                        r = {
                            "decision_id": str(decision.decision_id),
                            "date": decision_date,
                            "eval_step": eval_step,
                            "iqn_action": iqn_chosen,
                            "hdp_action": hdp_action,
                        }
                        r.update(ts_row)
                        step_ticker_scores.append(r)
                        if ts_row.get("ticker") == hdp_ticker:
                            hdp_ticker_score = ts_row.get("total_score")

                    # Size scores
                    for ss_row in decision.stage_3_size_scores:
                        r = {
                            "decision_id": str(decision.decision_id),
                            "date": decision_date,
                            "eval_step": eval_step,
                            "selected_size": hdp_size,
                        }
                        r.update(ss_row)
                        step_size_scores.append(r)
                        if ss_row.get("size_label") == hdp_size:
                            hdp_size_score = ss_row.get("score")

                    n_with_hdp += 1

                except Exception as exc:
                    w = f"eval_step={eval_step}: HDP failed — {exc}"
                    warnings.append(w)
                    logger.warning(w)
                    hdp_action = iqn_chosen  # fall through: use IQN action

            ticker_score_rows.extend(step_ticker_scores)
            size_score_rows.extend(step_size_scores)

            # ------------------------------------------------------------------
            # Market/technical features for selected ticker (or first ticker)
            # ------------------------------------------------------------------
            primary_ticker = hdp_ticker or (
                str(iqn_step.get("selected_ticker", "")) if iqn_step is not None else ""
            )
            if not primary_ticker and iqn_data.tickers:
                primary_ticker = iqn_data.tickers[0]

            tic_col_j = "tic" if "tic" in joined.columns else "ticker"
            ticker_row = (
                joined[joined[tic_col_j] == primary_ticker]
                if primary_ticker
                else pd.DataFrame()
            )
            tr = ticker_row.iloc[0] if not ticker_row.empty else pd.Series(dtype=float)

            def _f(col: str, default=None):
                v = tr.get(col, default) if not tr.empty else default
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    return default
                return (
                    float(v)
                    if isinstance(v, (int, float, np.integer, np.floating))
                    else v
                )

            # ------------------------------------------------------------------
            # Build combined row
            # ------------------------------------------------------------------
            decision_id = str(uuid.uuid4())[:12]

            row: dict = {
                # Identity
                "decision_id": decision_id,
                "date": decision_date,
                "visible_data_cutoff": decision_date,
                "eval_step": eval_step,
                "source_iqn_run_id": iqn_data.run_id,
                "dataset_id": iqn_data.dataset_id,
                "pit_split_id": iqn_data.split_id,
                # IQN action
                "selected_iqn_action": iqn_chosen,
                "selected_iqn_action_index": iqn_chosen_idx,
            }

            # IQN distribution features (from dist_wide row)
            for col in dist_row.index:
                if col not in ("eval_step",):
                    row[col] = dist_row[col]

            # HDP columns
            row["hierarchical_action_type"] = hdp_action
            row["selected_ticker"] = hdp_ticker or ""
            row["selected_size"] = hdp_size or ""
            row["selected_size_fraction"] = hdp_fraction
            row["ticker_score"] = hdp_ticker_score
            row["size_score"] = hdp_size_score

            # Market / technical features
            row["price"] = _f("close")
            row["ma50"] = _f("MA50")
            row["ma200"] = _f("MA200")
            row["price_vs_ma50"] = _f("price_vs_ma50")
            row["price_vs_ma200"] = _f("price_vs_ma200")
            row["momentum_score"] = _f("momentum_score")
            row["value_score"] = _f("value_score")
            row["quality_score"] = _f("quality_score")
            row["risk_score"] = _f("risk_fit_score")

            # Portfolio features
            row["cash_weight"] = (
                float(portfolio.cash_weight)
                if hasattr(portfolio, "cash_weight")
                else None
            )

            # Final recommendation before EDL
            row["final_recommendation_before_edl"] = hdp_action
            row["final_recommendation_source"] = (
                "iqn_hierarchical" if use_hierarchical_policy else "iqn_only"
            )

            # EDL label placeholders
            row["edl_a_hindsight_label"] = None
            row["edl_a_future_return_horizon"] = None
            row["edl_a_future_return_pct"] = None
            row["edl_b_rule_label"] = None
            row["edl_c_teacher_label"] = hdp_action  # IQN+HDP recommendation

            combined_rows.append(row)

        # ------------------------------------------------------------------
        # Step 6: Assemble DataFrames
        # ------------------------------------------------------------------
        combined_df = pd.DataFrame(combined_rows)
        ticker_score_df = (
            pd.DataFrame(ticker_score_rows) if ticker_score_rows else pd.DataFrame()
        )
        size_score_df = (
            pd.DataFrame(size_score_rows) if size_score_rows else pd.DataFrame()
        )

        # Ensure required columns exist (fill with None if absent)
        combined_df = _ensure_required_columns(combined_df)

        logger.info(
            "Combined audit built: %d rows, %d with IQN dist, %d with HDP enrichment",
            len(combined_df),
            n_with_dist,
            n_with_hdp,
        )
        if warnings:
            logger.warning("Build completed with %d warnings", len(warnings))

        return combined_df, ticker_score_df, size_score_df, warnings


# ---------------------------------------------------------------------------
# Required column enforcement
# ---------------------------------------------------------------------------

_REQUIRED_COMBINED_COLUMNS = [
    "decision_id",
    "date",
    "visible_data_cutoff",
    "eval_step",
    "source_iqn_run_id",
    "dataset_id",
    "pit_split_id",
    "selected_iqn_action",
    "selected_iqn_action_index",
    "iqn_q10_hold",
    "iqn_q50_hold",
    "iqn_q90_hold",
    "iqn_cvar10_hold",
    "iqn_score_hold",
    "iqn_q10_buy",
    "iqn_q50_buy",
    "iqn_q90_buy",
    "iqn_cvar10_buy",
    "iqn_score_buy",
    "iqn_q10_sell",
    "iqn_q50_sell",
    "iqn_q90_sell",
    "iqn_cvar10_sell",
    "iqn_score_sell",
    "iqn_q10_rebalance",
    "iqn_q50_rebalance",
    "iqn_q90_rebalance",
    "iqn_cvar10_rebalance",
    "iqn_score_rebalance",
    "iqn_action_margin",
    "hierarchical_action_type",
    "selected_ticker",
    "selected_size",
    "selected_size_fraction",
    "ticker_score",
    "size_score",
    "price",
    "ma50",
    "ma200",
    "price_vs_ma50",
    "price_vs_ma200",
    "momentum_score",
    "value_score",
    "quality_score",
    "risk_score",
    "cash_weight",
    "final_recommendation_before_edl",
    "final_recommendation_source",
    "edl_a_hindsight_label",
    "edl_a_future_return_horizon",
    "edl_a_future_return_pct",
    "edl_b_rule_label",
    "edl_c_teacher_label",
]


def _ensure_required_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add any missing required columns as None."""
    missing = [c for c in _REQUIRED_COMBINED_COLUMNS if c not in df.columns]
    if missing:
        logger.warning("Adding %d missing required columns: %s", len(missing), missing)
        for col in missing:
            df[col] = None
    return df


# ---------------------------------------------------------------------------
# Run discovery helper
# ---------------------------------------------------------------------------


def _default_runs_dir() -> Path:
    """Return the runs directory from the project root."""
    from stock_investment_dss.utilities.paths import RUNS_DIRECTORY

    return RUNS_DIRECTORY
